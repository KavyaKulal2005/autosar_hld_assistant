"""Grounded answering: abstention rules + structured answers built from extracted relations.

Flow: retrieve evidence -> decide whether to abstain -> if answering, turn the relations found in
the *retrieved documents* into a short cited answer; otherwise show the retrieved passages.
No generative model is used, so every statement traces back to a document line.
"""
import json
import re
from pathlib import Path
from src.relations import extract_relations, words, ID

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


def answer_question(question, results, rel_index, threshold):
    """Full grounded answer.

    Order: (1) unknown identifier in the question -> abstain; (2) relations extracted from the
    retrieved documents that match the question -> structured cited answer; (3) otherwise abstain
    when the best retrieval score is below the tuned threshold; (4) else show closest evidence.
    """
    if not results:
        reason = "No passages were retrieved."
        return {"answer": "I cannot answer this confidently from the loaded documents. " + reason,
                "facts": [], "citations": [], "abstained": True, "reason": reason, "mode": "abstain"}
    ids = CANDIDATE.findall(question)
    top_text = " ".join(r["text"] for r in results[:3])
    missing = [i for i in ids if i not in top_text]
    facts = None if missing else structured_answer(question, results, rel_index)
    if missing or (not facts and results[0]["score"] < threshold):
        reason = (f"The identifier {missing[0]} does not appear in the top retrieved passages." if missing else
                  f"No matching relation was found and the best retrieval score {results[0]['score']:.3f} is below the tuned threshold {threshold:.3f}.")
        return {"answer": "I cannot answer this confidently from the loaded documents. " + reason,
                "facts": [], "citations": [], "abstained": True, "reason": reason, "mode": "abstain"}
    if facts:
        by_fact = {}
        for f in facts:
            by_fact.setdefault(f["fact"], []).append(f["doc"])
        lines = [f"{k} ({', '.join(sorted(set(v)))})" for k, v in by_fact.items()]
        cites = {(f["doc"], f["page"]): {"document_id": f["doc"], "source": f["source"], "page": f["page"],
                                         "score": round(results[0]["score"], 4)} for f in facts}
        return {"answer": "; ".join(lines) + ".", "facts": facts, "citations": list(cites.values()),
                "abstained": False, "reason": "", "mode": "structured"}
    cites = [{"document_id": r["document_id"], "source": r["source"], "page": r["page"], "score": round(r["score"], 4)}
             for r in results[:3]]
    evidence = " ".join(r["text"].replace("\n", " ").strip() for r in results[:2])
    return {"answer": "No exact relation was extracted, showing the closest retrieved evidence: " + evidence,
            "facts": [], "citations": cites, "abstained": False, "reason": "", "mode": "evidence"}
