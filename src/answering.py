"""Grounded answering: abstention rules + structured answers built from extracted relations.

Flow: retrieve evidence -> decide whether to abstain -> if answering, turn the relations found in
the *retrieved documents* into a short cited answer; otherwise show the retrieved passages.
No generative model is used, so every statement traces back to a document line.
"""
import json
import re
from pathlib import Path
from src.relations import extract_relations, words, ID
from src.retrieval import answer_requirements
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FALLBACK_THRESHOLDS = {"tfidf": 0.2428, "hybrid": 0.3028, "sentence-transformers": 0.3797}
CANDIDATE = re.compile(rf"\b{ID}\b")


def tuned_threshold(mode):
    """Abstention threshold tuned on the dev split (reports/retrieval_eval_<mode>.json)."""
    path = REPORTS / f"retrieval_eval_{mode}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["abstention"]["identifier_grounding_plus_threshold"]["threshold"])
    except Exception:
        return FALLBACK_THRESHOLDS.get(mode, 0.3)


def tuned_alpha(default=0.4):
    try:
        data = json.loads((REPORTS / "retrieval_eval_hybrid.json").read_text(encoding="utf-8"))
        return float(data["alpha_tfidf_weight"])
    except Exception:
        return default


def build_relation_index(pages):
    """All relations in the loaded documents, tagged with document and page."""
    rels = []
    for p in pages:
        for r in extract_relations(p["text"]):
            rels.append({**r, "doc": p["document_id"], "page": p["page"], "source": p["source"]})
    return rels


def _entity_names(rels):
    names = set()
    for r in rels:
        for k in ("component", "signal", "interface", "target", "consumer", "port"):
            if r.get(k):
                names.add(r[k])
    return names


def _match_entities(question, names):
    q = question.lower()
    exact = [n for n in names if re.search(rf"\b{re.escape(n)}\b", question)]
    spoken = [n for n in names if n not in exact and re.search(rf"\b{re.escape(words(n))}\b", q)]
    return exact + sorted(spoken, key=lambda n: -len(n))


def _intent(question):
    q = question.lower()
    if re.search(r"depend|rely|relies", q):
        return "depends_on"
    if re.search(r"provide|offer", q):
        return "provides"
    if re.search(r"consum|receiv|use ", q):
        return "consumes"
    if re.search(r"publish|produce|source|originat|send|emit", q):
        return "publishes"
    return None


def _sentence(r):
    t = r["type"]
    if t == "publishes":
        via = f" via the {r['port']} port" if r.get("port") else ""
        return f"{r['component']} publishes {r['signal']}{via}"
    if t == "consumes":
        return f"{r['component']} consumes {r['signal']}"
    if t == "provides":
        to = f" to {r['consumer']}" if r.get("consumer") else ""
        return f"{r['component']} provides the {r['interface']} interface{to}"
    return f"{r['component']} depends on {r['target']} for {r['signal']}"


def structured_answer(question, results, rel_index, max_facts=6):
    """Facts from relations in the retrieved documents that match the question's entities."""
    docs = {r["document_id"] for r in results}
    scoped = [r for r in rel_index if r["doc"] in docs]
    entities = _match_entities(question, _entity_names(scoped))
    if not entities:
        return None
    intent = _intent(question)
    hits = []
    for r in scoped:
        values = {r.get(k) for k in ("component", "signal", "interface", "target", "consumer", "port")}
        if not set(entities) & values:
            continue
        if intent and r["type"] != intent:
            continue
        hits.append(r)
    if not hits:
        return None
    seen, facts = set(), []
    for r in hits:
        s = _sentence(r)
        key = (s, r["doc"])
        if key in seen:
            continue
        seen.add(key)
        facts.append({"fact": s, "doc": r["doc"], "page": r["page"], "source": r["source"], "evidence": r["evidence"]})
    return facts[:max_facts]
def _identifier_evidence(identifier, results, max_chars=900):
    """Find the passage associated with an exact requirement identifier."""
    for r in results[:5]:
        text = re.sub(r"\s+", " ", r.get("text", "")).strip()
        match = re.search(
            rf"(?<!\w){re.escape(identifier)}(?!\w)", text
        )
        if not match:
            continue

        # Prefer the text following the requirement ID.
        remaining = text[match.end():]
        next_id = re.search(rf"\b{ID}\b", remaining)

        snippet = (
            remaining[:next_id.start()]
            if next_id
            else remaining[:max_chars]
        ).strip(" \t\n.;:-")

        # If the ID is near the end of a passage, include nearby context.
        if len(snippet) < 40:
            start = max(0, match.start() - 250)
            end = min(len(text), match.end() + max_chars)
            snippet = text[start:end].strip()

        if snippet:
            return snippet[:max_chars], r

    return None, None

def answer_question(question, results, rel_index, threshold):
    """Return a grounded answer with supporting citations."""

    if not results:
        reason = "No passages were retrieved."
        return {
            "answer": (
                "I cannot answer this confidently from the loaded documents. "
                + reason
            ),
            "facts": [],
            "citations": [],
            "abstained": True,
            "reason": reason,
            "mode": "abstain",
        }

    ids = CANDIDATE.findall(question)
    req_ids = re.findall(r"\bRS_[A-Za-z]+_\d+\b", question, re.I)
    top_text = " ".join(r["text"] for r in results[:5])
    missing = [identifier for identifier in ids if identifier not in top_text]

    # Answer requirement-ID questions directly from evidence.
    if req_ids and not missing:
        snippet, source_result = _identifier_evidence(req_ids[0], results)
        if snippet and source_result:
            citation = {
                "document_id": source_result["document_id"],
                "source": source_result["source"],
                "page": source_result["page"],
                "score": round(source_result["score"], 4),
            }
            return {
                "answer": f"Requirement {req_ids[0]}: {snippet}",
                "facts": [],
                "citations": [citation],
                "abstained": False,
                "reason": "",
                "mode": "requirement",
            }

    # Extract structured facts from the retrieved documents.
    facts = (
        None
        if missing
        else structured_answer(question, results, rel_index)
    )

    # Abstain if the requested identifier is missing or evidence is weak.
    if missing or (
        not facts and results[0]["score"] < threshold
    ):
        if missing:
            reason = (
                f"The identifier {missing[0]} does not appear "
                "in the top retrieved passages."
            )
        else:
            reason = (
                "No matching relation was found and the best retrieval "
                f"score {results[0]['score']:.3f} is below the tuned threshold."
            )

        return {
            "answer": (
                "I cannot answer this confidently from the loaded documents. "
                + reason
            ),
            "facts": [],
            "citations": [],
            "abstained": True,
            "reason": reason,
            "mode": "abstain",
        }

    # Build a readable answer from extracted facts.
    if facts:
        by_fact = {}
        for fact in facts:
            by_fact.setdefault(fact["fact"], []).append(fact["doc"])

        lines = [
            f"{fact} ({', '.join(sorted(set(documents)))})"
            for fact, documents in by_fact.items()
        ]

        citations_by_source = {}
        for fact in facts:
            key = (fact["doc"], fact["page"])
            citations_by_source[key] = {
                "document_id": fact["doc"],
                "source": fact["source"],
                "page": fact["page"],
                "score": round(results[0]["score"], 4),
            }

        return {
            "answer": "; ".join(lines),
            "facts": facts,
            "citations": list(citations_by_source.values()),
            "abstained": False,
            "reason": "",
            "mode": "structured",
        }
        req = answer_requirements(question, results)
        if req:
            return {
            "answer": req["answer"],
            "facts": [],
            "citations": [{**c, "score": round(results[0]["score"], 4)} for c in req["citations"]],
            "abstained": False,
            "reason": "",
            "mode": "requirement",
        }
    # Fallback: show a short excerpt rather than a long raw passage.
    citations = [
        {
            "document_id": result["document_id"],
            "source": result["source"],
            "page": result["page"],
            "score": round(result["score"], 4),
        }
        for result in results[:3]
    ]

    evidence = " ".join(
        " ".join(result["text"].split())
        for result in results[:2]
    )
    if len(evidence) > 700:
        evidence = evidence[:697].rsplit(" ", 1)[0] + "..."

    return {
        "answer": (
            "No exact relation was extracted. "
            "Here is the closest retrieved evidence: " + evidence
        ),
        "facts": [],
        "citations": citations,
        "abstained": False,
        "reason": "",
        "mode": "evidence",
    }