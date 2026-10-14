"""Retrieval baseline and optional Sentence Transformer semantic retrieval."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        ranked = sorted(zip(allowed, scores), key=lambda x: float(x[1]), reverse=True)
        out = []
        for idx, score in ranked[:max(1, top_k)]:
            row = dict(self.chunks[idx])
            row["score"] = float(score)
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
        ranked = sorted(((i, scores[i]) for i in allowed), key=lambda x: float(x[1]), reverse=True)
        out = []
        for idx, score in ranked[:max(1, top_k)]:
            row = dict(self.chunks[idx])
            row["score"] = float(score)
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
