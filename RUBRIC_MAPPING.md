# Rubric mapping (100 marks)

| # | Parameter | Marks | Evidence in this project |
|---|---|---:|---|
| 1 | Problem understanding & automotive context | 5 | README, Case Study 1 scope; HLD review, traceability, impact analysis |
| 2 | AI/ML fundamentals & approach selection | 5 | TF-IDF vs embeddings vs hybrid; why RAG with abstention; rule-based vs trained entity model |
| 3 | Dataset understanding, preprocessing & EDA | 15 | `data/corpus` + `manifest.json`, `src/corpus_builder.py`, `notebooks/eda_corpus.py`, `reports/corpus_eda.png`, chunking in `src/document_loader.py` |
| 4 | Feature engineering, selection & dimensionality reduction | 8 | `src/entity_model.py` features; chi-square selection and PCA in `src/feature_analysis.py`, `reports/feature_analysis.json`, PCA plots |
| 5 | Model development using syllabus techniques | 18 | Logistic regression, naive Bayes, SVM, random forest; TF-IDF/embedding retrieval; hybrid fusion; relation extraction |
| 6 | Model evaluation, validation & optimization | 15 | Grouped CV, held-out documents, dev/test split, alpha and threshold tuning, hit@k/MRR, failure CSVs, `answer_eval_*.json` |
| 7 | Completeness & working demonstration | 15 | Streamlit app: cited Q&A, entities, relations, consistency, revision compare, exports, review log |
| 8 | Python implementation & code quality | 6 | Modular `src/`, 28 unit tests, `requirements.txt` |
| 9 | Deployment & MLOps basics | 5 | Dockerfile, saved model and metrics JSON, audit log, reproducible evaluation commands |
| 10 | Responsible AI & deployment ethics | 3 | Abstention, citations, human review log, synthetic-data and licensing limits |
| 11 | Technical Q&A | 3 | `QA_PREP.md` |
| 12 | Communication & presentation | 2 | Slides and demo script (update with the numbers above) |
| | **Total** | **100** | |
