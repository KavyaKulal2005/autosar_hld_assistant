"""Retrieval baseline and optional Sentence Transformer semantic retrieval."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ID_PATTERN = re.compile(r"\bRS_[A-Za-z]+_\d+\b", re.I)
ID_BONUS = 0.5


def id_bonus(query, chunk):
    """Exact-ID lookup: a query naming a requirement ID (e.g. RS_Main_00700) strongly favours the chunk
    that DEFINES it (starts with its '[ID]' heading). Queries without such IDs are unaffected."""
    ids = {m.upper() for m in ID_PATTERN.findall(query)}
    head = chunk["text"].lstrip().upper()
    return ID_BONUS if any(head.startswith(f"[{i}]") for i in ids) else 0.0


class Retriever:
    def __init__(self, chunks, prefer_embeddings=True):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.mode = "tfidf"
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts) if self.texts else None
        self.embedder = None
        self.embeddings = None
        if prefer_embeddings and self.texts:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.embeddings = self.embedder.encode(self.texts, normalize_embeddings=True)
                self.mode = "sentence-transformers"
            except Exception:
                # Offline/no model cache: TF-IDF remains fully functional.
                self.embedder = None
                self.embeddings = None

    def search(self, query, top_k=4, document_id=None):
        if not self.chunks:
            return []
        allowed = [i for i, c in enumerate(self.chunks)
                   if document_id is None or c["document_id"] == document_id]
        if not allowed:
            return []
        if self.mode == "sentence-transformers" and self.embedder is not None:
            q = self.embedder.encode([query], normalize_embeddings=True)[0]
            scores = np.dot(self.embeddings[allowed], q)
        else:
            q = self.vectorizer.transform([query])
            scores = cosine_similarity(q, self.matrix[allowed]).ravel()
        scores = [float(sc) + id_bonus(query, self.chunks[i]) for i, sc in zip(allowed, scores)]
        ranked = sorted(zip(allowed, scores), key=lambda x: float(x[1]), reverse=True)
        out = []
        for idx, score in ranked[:max(1, top_k)]:
            row = dict(self.chunks[idx])
            row["score"] = min(1.0, float(score))
            out.append(row)
        return out

class HybridRetriever:
    """Weighted fusion of TF-IDF and embedding cosine scores: score = a*tfidf + (1-a)*embedding.

    Wraps a Retriever that already has embeddings, so the model is loaded once.
    Exact identifiers favour TF-IDF; reworded questions favour embeddings.
    """
    def __init__(self, base, alpha=0.5):
        if base.embedder is None or base.embeddings is None:
            raise ValueError("HybridRetriever needs a Retriever with embeddings loaded.")
        self.base, self.alpha, self.chunks = base, alpha, base.chunks
        self.mode = "hybrid"
        self._cache = {}

    def component_scores(self, query):
        if query not in self._cache:
            tfidf = cosine_similarity(self.base.vectorizer.transform([query]), self.base.matrix).ravel()
            q = self.base.embedder.encode([query], normalize_embeddings=True)[0]
            self._cache[query] = (tfidf, np.dot(self.base.embeddings, q))
        return self._cache[query]

    def search(self, query, top_k=4, document_id=None):
        allowed = [i for i, c in enumerate(self.chunks) if document_id is None or c["document_id"] == document_id]
        if not allowed:
            return []
        tfidf, emb = self.component_scores(query)
        scores = self.alpha * tfidf + (1 - self.alpha) * emb
        ranked = sorted(((i, float(scores[i]) + id_bonus(query, self.chunks[i])) for i in allowed),
                        key=lambda x: x[1], reverse=True)
        out = []
        for idx, score in ranked[:max(1, top_k)]:
            row = dict(self.chunks[idx])
            row["score"] = min(1.0, float(score))
            out.append(row)
        return out


def answer_from_evidence(question, results, min_score=0.01):
    """Conservative extractive-style answer: returns evidence rather than inventing facts."""
    if not results:
        return {"answer": "I could not find relevant evidence in the available documents.", "citations": [], "abstained": True}
    useful = [r for r in results if r.get("score", 0) >= min_score]
    if not useful:
        return {"answer": "The available documents do not provide enough relevant evidence to answer confidently.", "citations": [], "abstained": True}
    citations = []
    for r in useful[:3]:
        citations.append({"document_id": r["document_id"], "source": r["source"], "page": r["page"], "score": round(r["score"], 4)})
    evidence = " ".join(r["text"].replace("\n", " ").strip() for r in useful[:2])
    return {
        "answer": "Based on the retrieved document evidence: " + evidence,
        "citations": citations,
        "abstained": False
    }

# ---- requirement-level answering (add to src/retrieval.py) ----------------------------------
REQ_BLOCK = re.compile(r"\[(RS_[A-Za-z]+_\d+)\](?:\{[A-Z]+\})?[ \t]*([^\n]*)\n?(.*?)(?=\[RS_[A-Za-z]+_\d+\](?:\{[A-Z]+\})|\Z)", re.S)
NOISE = re.compile(r"\d+\s+of\s+\d+|Document ID\s+\d+:\s*\S+|(?:\.\s?){4,}\s*\d*")


def parse_requirements(text):
    """Turn a chunk into clean records: id, title, description, rationale. Drops footers / TOC dots."""
    recs = []
    for m in REQ_BLOCK.finditer(text):
        full = m.group(2) + "\n" + m.group(3)
        cut = re.search(r"\u2308|Description:", full)
        head, rest = (full[:cut.start()], full[cut.start():]) if cut else (full, "")
        body = NOISE.sub(" ", rest)
        body = re.sub(r"[\u2308\u230b\u230a\u2309|]", " ", body)       # AUTOSAR corner marks
        body = re.sub(r"\(RS_[A-Za-z_0-9, ]+\)", " ", body)             # trailing "(RS_PO_00005, ...)" refs
        body = re.sub(r"\s+", " ", body).strip(" ]")
        d = re.search(r"Description:\s*(.*?)(?:\s*Rationale:|\s*Use Case:|$)", body)
        r = re.search(r"Rationale:\s*(.*?)(?:\s*Use Case:|$)", body)
        title = re.sub(r"\s+", " ", head).strip(" [")
        if title and re.search(r"(?:\.\s?){3,}", title):                 # TOC line, not a requirement
            continue
        recs.append({"id": m.group(1), "title": title,
                     "description": (d.group(1) if d else body).strip(),
                     "rationale": r.group(1).strip() if r else ""})
    return recs


def answer_requirements(question, results, top_n=3):
    """Answer 'which requirement ...' questions with requirement IDs, not raw chunk text."""
    q_terms = {w for w in re.findall(r"[a-z0-9\-]{3,}", question.lower())} - {"which", "what", "requirement", "requirements", "does", "covers", "cover", "with"}
    scored = []
    for res in results:
        for rec in parse_requirements(res["text"]):
            hay = f"{rec['title']} {rec['description']} {rec['rationale']}".lower()
            overlap = sum(t in hay for t in q_terms) / max(1, len(q_terms))
            scored.append((overlap + 0.2 * res.get("score", 0), rec, res))
    scored = [s for s in scored if s[0] > 0.3]
    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    seen, lines, cites = set(), [], []
    for _, rec, res in scored:
        if rec["id"] in seen:
            continue
        seen.add(rec["id"])
        lines.append(f"**{rec['id']} - {rec['title']}**: {rec['description']}")
        cites.append({"document_id": res["document_id"], "source": res["source"], "page": res["page"], "requirement": rec["id"]})
        if len(lines) == top_n:
            break
    return {"answer": "\n\n".join(lines), "citations": cites, "abstained": False}
