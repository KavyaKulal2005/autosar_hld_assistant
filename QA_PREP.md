# Technical Q&A preparation

**Why RAG instead of only an LLM?** Answers must be traceable to approved documents. Retrieval plus citations
limits invented relationships; the app abstains when evidence is weak.

**Why hybrid retrieval?** TF-IDF is perfect on exact identifiers (keyword term@3 1.0) but weak on reworded
questions (0.357). Embeddings help paraphrases (0.571) but hurt exact names (0.5). Weighted fusion
(alpha 0.4, tuned on dev) gives 1.0 and 0.5 and the best overall term@3 (0.75).

**Why term@k and MRR?** A hit counts only if the retrieved chunk contains the gold component name, not just
the right document. MRR rewards ranking the correct chunk early.

**Why is cross-validation F1 1.0 for the entity classifier?** The data is template-generated, so patterns
repeat. Held-out documents with unseen names and phrasing are the honest number (logistic regression 0.983).

**What does chi-square selection show?** 200 of 753 features keep macro-F1 at 0.999; the top features are
context words (for example "see", "for", "interface"), so the model labels names from their surroundings.

**Why does PCA not help retrieval?** Only 75 chunks, so compression discards query-relevant directions:
term@3 0.536 at 384 dimensions vs 0.357 at 20. PCA is kept for visualisation and analysis.

**How does abstention work and how was it tuned?** Abstain when an identifier in the question is not in the
top passages, or when no relation matches and the best score is below a threshold tuned on the dev split.
Report on the test split. On unanswerable questions the new flow abstains 100% (TF-IDF run); the old flow 0%.

**Why rule-based relation extraction?** Transparent, testable and exact on the synthetic phrasing (258/258
relations, 17/17 injected issues). Limitation: real HLDs vary, so patterns must be extended or replaced by a
trained or LLM extractor.

**Main limitations?** Synthetic data, small test set (n=16), no OCR, no ARXML, no generative summaries, English
templates only.

**What would you do with a real LLM?** Keep retrieval and the abstention rules, give the LLM only the retrieved
passages, require citations, and validate its output against the relation extractor.

**Data privacy?** Everything runs locally; uploaded files stay in `.uploaded/`; logs hold questions and
decisions only. Use authorised documents.
