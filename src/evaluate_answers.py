"""End-to-end answer evaluation: retrieval + abstention + structured answer, per retriever mode.

Compares the original extractive answer (raw passage text) with the grounded structured answer
on the same synthetic questions. Writes reports/answer_eval_<mode>.json.
Synthetic data only: relative comparison, not real-world accuracy.
"""
import json
from src.document_loader import load_sample_documents, chunk_pages
from src.retrieval import Retriever, HybridRetriever, answer_from_evidence
from src.evaluate_retrieval import build_questions, CORPUS, REPORTS
from src.answering import answer_question, build_relation_index, tuned_threshold, tuned_alpha


def _contains_gold(text, gold):
    return any(term in text for _, term in gold)


def evaluate(retriever, pages, questions, mode):
    rel_index = build_relation_index(pages)
    thr = tuned_threshold(mode)
    valid_pages = {(p["document_id"], p["page"]) for p in pages}
    rows = {"old": [], "new": []}
    abst_ok, abst_n, cite_ok, cite_n = 0, 0, 0, 0
    structured = 0
    for q in questions:
        res = retriever.search(q["question"], top_k=5)
        new = answer_question(q["question"], res, rel_index, thr)
        if q["answerable"]:
            old = answer_from_evidence(q["question"], res[:3], min_score=0.0)
            rows["old"].append(1.0 if _contains_gold(old["answer"], q["gold"]) else 0.0)
            rows["new"].append(1.0 if (not new["abstained"] and _contains_gold(new["answer"], q["gold"])) else 0.0)
            structured += new["mode"] == "structured"
            if not new["abstained"]:
                cite_n += 1
                cite_ok += all((c["document_id"], c["page"]) in valid_pages for c in new["citations"])
        else:
            abst_n += 1
            abst_ok += new["abstained"]
    n = max(1, len(rows["new"]))
    return {"mode": mode, "threshold": thr, "n_answerable": len(rows["new"]), "n_unanswerable": abst_n,
            "old_extractive_answer_contains_gold": round(sum(rows["old"]) / n, 3),
            "grounded_answer_contains_gold": round(sum(rows["new"]) / n, 3),
            "structured_answer_rate": round(structured / n, 3),
            "correct_abstention_rate_on_unanswerable": round(abst_ok / max(1, abst_n), 3),
            "citation_validity_rate": round(cite_ok / max(1, cite_n), 3),
            "note": "Synthetic, template-generated corpus. Relative comparison only."}


def run():
    pages = load_sample_documents(CORPUS)
    chunks = chunk_pages(pages)
    questions = build_questions(json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8")))
    out = {}
    base = Retriever(chunks, prefer_embeddings=True)
    tfidf = Retriever(chunks, prefer_embeddings=False)
    out["tfidf"] = evaluate(tfidf, pages, questions, "tfidf")
    if base.mode == "sentence-transformers":
        out["hybrid"] = evaluate(HybridRetriever(base, tuned_alpha()), pages, questions, "hybrid")
    REPORTS.mkdir(exist_ok=True)
    for mode, r in out.items():
        (REPORTS / f"answer_eval_{mode}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    for mode, r in run().items():
        print(f"\n[{mode}] threshold={r['threshold']}")
        for k, v in r.items():
            if k not in ("mode", "threshold", "note"):
                print(f"  {k:46s}{v}")
