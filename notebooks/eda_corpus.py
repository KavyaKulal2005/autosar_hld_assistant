"""EDA for the synthetic HLD corpus: size, structure, entities, relations, chunking, vocabulary."""
from pathlib import Path
import json
import re
import sys
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.document_loader import load_sample_documents, chunk_pages

CORPUS = ROOT / "data" / "corpus"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
pages = load_sample_documents(CORPUS)
chunks = chunk_pages(pages)

doc_df = pd.DataFrame([{
    "document_id": p["document_id"],
    "words": len(p["text"].split()),
    "lines": len(p["text"].splitlines()),
    "sections": len(re.findall(r"^SECTION:", p["text"], flags=re.M)),
} for p in pages])
chunk_df = pd.DataFrame(chunks)
chunk_df["characters"] = chunk_df["text"].str.len()

ent_rows = [{"document_id": d, "type": t, "entity": e} for d, m in manifest.items() for e, t in m["labels"].items()]
ent_df = pd.DataFrame(ent_rows)
rel_df = pd.DataFrame([{"document_id": d, **r} for d, m in manifest.items() for r in m["relations"]])
issue_df = pd.DataFrame([{"document_id": d, **i} for d, m in manifest.items() for i in m["issues"]])

vocab = Counter(w.lower() for p in pages for w in re.findall(r"[A-Za-z]+", p["text"]))
shared = ent_df.groupby("entity")["document_id"].nunique()
summary = {
    "documents": len(doc_df),
    "revision_pairs": sum(1 for m in manifest.values() if m["revision_of"]),
    "total_words": int(doc_df["words"].sum()),
    "avg_words_per_document": round(doc_df["words"].mean(), 1),
    "chunks": len(chunk_df),
    "chunk_chars_mean": round(chunk_df["characters"].mean(), 1),
    "chunk_chars_max": int(chunk_df["characters"].max()),
    "unique_entities": int(ent_df["entity"].nunique()),
    "entities_by_type": ent_df.drop_duplicates(["document_id", "entity"])["type"].value_counts().to_dict(),
    "relations_by_type": rel_df["type"].value_counts().to_dict(),
    "injected_issues_by_type": issue_df["type"].value_counts().to_dict() if len(issue_df) else {},
    "entities_appearing_in_multiple_docs": int((shared > 1).sum()),
    "vocabulary_size": len(vocab),
    "missing_values_in_entity_table": int(ent_df.isna().sum().sum()),
}
(REPORTS / "corpus_eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
doc_df.set_index("document_id")["words"].plot(kind="bar", ax=ax[0, 0], title="Words per document")
ent_df.drop_duplicates(["document_id", "entity"])["type"].value_counts().plot(kind="bar", ax=ax[0, 1], title="Entities by type (per document)")
rel_df["type"].value_counts().plot(kind="bar", ax=ax[1, 0], title="Relations by type")
chunk_df["characters"].plot(kind="hist", bins=15, ax=ax[1, 1], title="Chunk length (characters)")
for a in ax.ravel():
    a.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(REPORTS / "corpus_eda.png", dpi=150)
print("Saved reports/corpus_eda.png and reports/corpus_eda_summary.json")
