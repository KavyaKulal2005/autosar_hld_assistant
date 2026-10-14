"""Quick EDA script for synthetic architecture dataset and text chunks."""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.document_loader import load_sample_documents, chunk_pages

entities = pd.read_csv(ROOT / "data" / "entities.csv")
pages = load_sample_documents(ROOT / "data" / "sample_documents")
chunks = chunk_pages(pages)
(ROOT / "reports").mkdir(exist_ok=True)

print("Entity dataset shape:", entities.shape)
print("\nMissing values:\n", entities.isna().sum())
print("\nUnique documents:", entities["document_id"].nunique())
for col in ["component_name", "interface_name", "signal_name"]:
    print(f"\nTop {col} values:\n", entities[col].value_counts().head(10))

chunk_df = pd.DataFrame(chunks)
chunk_df["character_count"] = chunk_df["text"].str.len()
print("\nChunk summary:\n", chunk_df["character_count"].describe())

entities["component_name"].value_counts().plot(kind="bar", title="Component record counts", rot=45)
plt.tight_layout()
plt.savefig(ROOT / "reports" / "component_counts.png", dpi=150)
plt.close()

chunk_df["character_count"].plot(kind="hist", bins=10, title="Chunk character length distribution")
plt.tight_layout()
plt.savefig(ROOT / "reports" / "chunk_lengths.png", dpi=150)
plt.close()
print("\nSaved charts in reports/.")
