"""Evaluate retrieval using the included small, manually verified synthetic test set."""
from pathlib import Path
import json
import pandas as pd
from src.document_loader import load_sample_documents, chunk_pages
from src.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]

def evaluate(prefer_embeddings=False, top_k=3):
    pages = load_sample_documents(ROOT / "data" / "sample_documents")
    chunks = chunk_pages(pages)
    retriever = Retriever(chunks, prefer_embeddings=prefer_embeddings)
    questions = pd.read_csv(ROOT / "data" / "evaluation_questions.csv")
    rows = []
    hits = 0
    reciprocal_ranks = []
    for _, item in questions.iterrows():
        results = retriever.search(item["question"], top_k=top_k)
        expected_doc = str(item["expected_document_id"])
        rank = next((i + 1 for i, r in enumerate(results) if r["document_id"] == expected_doc), None)
        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1 / rank if rank else 0)
        rows.append({
            "question": item["question"],
            "expected_document_id": expected_doc,
            "retrieved_document_ids": [r["document_id"] for r in results],
            "hit_at_k": hit,
            "reciprocal_rank": 1 / rank if rank else 0,
            "top_score": results[0]["score"] if results else 0
        })
    n = len(questions)
    result = {
        "retriever_mode": retriever.mode,
        "test_questions": n,
        "top_k": top_k,
        "document_hit_rate_at_k": hits / n if n else 0,
        "mrr": sum(reciprocal_ranks) / n if n else 0,
        "per_question": rows,
        "note": "Small synthetic evaluation set. Metrics are not evidence of real-world automotive performance."
    }
    out = ROOT / "reports" / "evaluation_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out}")
    return result

if __name__ == "__main__":
    evaluate(prefer_embeddings=False)
