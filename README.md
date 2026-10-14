# AUTOSAR HLD Document Analysis Assistant

Student prototype for Case Study 1 (Automotive Engineering AI). It helps engineers search and check
AUTOSAR High-Level Design documents: cited question answering, entity and relation extraction,
consistency checks and revision comparison. **All data is synthetic and template-generated; metrics are
relative comparisons, not real-world accuracy.** It is not safety-certified and does not approve designs.

## Pipeline
1. **Ingest** PDF/TXT/MD (PyMuPDF), clean text, chunk by paragraph with page metadata.
2. **Retrieve** with TF-IDF, Sentence-Transformer embeddings (all-MiniLM-L6-v2) or a **hybrid**
   `alpha*TF-IDF + (1-alpha)*embedding` score (alpha tuned on a dev split).
3. **Decide to answer or abstain**: unknown identifiers in the question, or low score with no matching relation.
4. **Answer from extracted relations** (publishes / consumes / provides / depends on) found in the
   retrieved documents, with document and page citations; otherwise show the closest passages.
5. **Human review**: reviewer decision and every query are appended to `logs/review_log.jsonl`.

Other features: trained entity classifier (component / interface / port / signal), deterministic
consistency checks (undefined component, orphan signal, unconsumed signal), revision comparison
(entity and relation diff), CSV export of entities, relations and findings.

## Results (synthetic corpus: 13 documents, 75 chunks, 28 answerable + 8 unanswerable questions)
| Retriever | overall term@3 | overall MRR | keyword term@3 | paraphrase term@3 |
|---|---|---|---|---|
| TF-IDF | 0.679 | 0.631 | 1.000 | 0.357 |
| Embeddings only | 0.536 | 0.424 | 0.500 | 0.571 |
| Hybrid (alpha 0.4) | 0.750 | 0.636 | 1.000 | 0.500 |

Hybrid keeps exact-identifier strength and recovers part of the paraphrase gap; it loses a little at
rank 1 on keyword questions (MRR 0.929). Test split is small (n=16), so treat differences as indicative.

Entity classifier (held-out documents with unseen names): regex 0.232 accuracy, suffix rules 0.622,
logistic regression 0.988. Four model families, document-grouped CV and chi-square selection
(200 of 753 features keep macro-F1 0.999) are in `reports/feature_analysis.json`. PCA on embeddings needs
22 components for 90% variance but lowers retrieval quality (term@3 0.536 at 384 dims vs 0.357 at 20 dims).

Relation extraction recovers 258/258 manifest relations; consistency checks find 17/17 injected issues with
no false positives. These rules match the generator's phrasing, so real HLDs will need broader patterns.

## Setup (Windows)
```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The first embedding run downloads the model once; without it the app falls back to TF-IDF.

## Reproduce the evaluations
```
python -m src.evaluate_entities      # train + evaluate entity classifier
python -m src.evaluate_retrieval     # TF-IDF vs embeddings vs hybrid, abstention tuning
python -m src.evaluate_answers       # end-to-end answers incl. abstention (reports/answer_eval_*.json)
python -m src.evaluate_relations     # relation extraction + consistency checks vs manifest
python -m src.feature_analysis       # classifiers, chi-square, PCA
python -m unittest discover -s tests -v
```

## Folder structure
- `app.py` Streamlit app (4 tabs)
- `src/document_loader.py` ingestion and chunking
- `src/retrieval.py` TF-IDF, embeddings, hybrid retriever
- `src/entity_model.py` trained entity classifier
- `src/relations.py` relation extraction, consistency checks, revision diff
- `src/answering.py` abstention rules and structured cited answers
- `src/evaluate_*.py`, `src/feature_analysis.py` experiments; `reports/` saved results
- `data/corpus` synthetic HLDs + `manifest.json` gold labels; `tests/` unit tests; `Dockerfile`

## Responsible AI and limits
- Answers are grounded in retrieved text and cite document and page; the app abstains instead of guessing.
- No generative model is used, so nothing is invented; this also means no free-form summarisation.
- OCR for scanned PDFs, diagrams and real AUTOSAR XML (ARXML) are not covered.
- Do not upload confidential documents unless you are authorised to process them locally.
- Outputs need qualified human review before any design, safety or release decision.
