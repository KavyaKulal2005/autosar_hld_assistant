"""AUTOSAR HLD Document Analysis Assistant (Streamlit).

Pipeline: ingest -> chunk -> hybrid retrieval (TF-IDF + embeddings) -> abstention rules ->
structured, cited answer from extracted relations -> human review log.
"""
import json
import time
from pathlib import Path
import pandas as pd
import streamlit as st
from src.document_loader import load_sample_documents, load_file, chunk_pages
from src.retrieval import Retriever, HybridRetriever, answer_requirements
from src.analysis import extract_entities
from src.entity_model import load_model, predict_entities
from src.relations import check_consistency, compare_relations
from src.answering import answer_question, build_relation_index, tuned_alpha, tuned_threshold

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
SAMPLE_SETS = {
    "Synthetic corpus (13 documents, evaluated)": ROOT / "data" / "corpus",
    "Small demo HLDs (2 documents)": ROOT / "data" / "sample_documents",
    "None (use uploads only)": None,
}

st.set_page_config(page_title="AUTOSAR HLD Assistant", page_icon="🚘", layout="wide")
st.title("AUTOSAR HLD Document Analysis Assistant")
st.caption("Student prototype • synthetic data • evidence-based architecture exploration • human review required")


def load_json(name):
    path = REPORTS / name
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


def log_event(kind, payload):
    """Append an audit record (question, decision, mode) to logs/review_log.jsonl."""
    try:
        LOGS.mkdir(exist_ok=True)
        with open(LOGS / "review_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **payload}) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Knowledge base")
    sample_name = st.selectbox("Sample documents", list(SAMPLE_SETS))
    use_embeddings = st.checkbox("Use Sentence Transformers embeddings", value=True,
                                 help="If the model is unavailable/offline the app falls back to TF-IDF.")
    use_hybrid = st.checkbox("Hybrid retrieval (TF-IDF + embeddings)", value=True)
    alpha = st.slider("TF-IDF weight (alpha)", 0.0, 1.0, float(tuned_alpha()), 0.1,
                      help="Tuned on the dev split; reports/retrieval_eval_hybrid.json")
    top_k = st.slider("Retrieved passages", 1, 8, 4)
    st.info("Use only documents you are authorized to process. AI findings require human review.")

pages = []
sample_dir = SAMPLE_SETS[sample_name]
if sample_dir is not None:
    try:
        pages.extend(load_sample_documents(sample_dir))
    except Exception as exc:
        st.error(f"Could not load sample documents: {exc}")

uploads = st.file_uploader("Upload HLD documents (PDF, TXT, MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)
if uploads:
    temp_dir = ROOT / ".uploaded"
    temp_dir.mkdir(exist_ok=True)
    for upload in uploads:
        temp_path = temp_dir / upload.name
        temp_path.write_bytes(upload.getvalue())
        try:
            pages.extend(load_file(temp_path))
        except Exception as exc:
            st.warning(f"Skipped {upload.name}: {exc}")

if not pages:
    st.warning("Choose a sample document set or upload a PDF/TXT/MD file to begin.")
    st.stop()

chunks = chunk_pages(pages)
if not chunks:
    st.error("No readable text was extracted. For scanned PDFs, OCR is not included in this prototype.")
    st.stop()


@st.cache_resource(show_spinner="Preparing retrieval index...")
def build_retriever(chunk_signature, prefer_embeddings):
    # chunk_signature invalidates the cache when document text changes.
    return Retriever(st.session_state["_chunks"], prefer_embeddings=prefer_embeddings)


signature = tuple((c["document_id"], c["page"], c["text"]) for c in chunks)
st.session_state["_chunks"] = chunks
try:
    base = build_retriever(signature, use_embeddings)
except Exception as exc:
    st.warning(f"Embedding initialization failed; using TF-IDF. Details: {exc}")
    base = Retriever(chunks, prefer_embeddings=False)

if use_hybrid and base.mode == "sentence-transformers":
    retriever = HybridRetriever(base, alpha)
else:
    retriever = base
rel_index = build_relation_index(pages)

with st.sidebar:
    default_thr = tuned_threshold(retriever.mode)
    threshold = st.slider("Abstention threshold", 0.0, 1.0, float(round(default_thr, 4)), 0.01,
                          help="Tuned on the synthetic dev split. Re-tune for new document types.")
    st.caption(f"Retrieval mode: **{retriever.mode}** • {len(chunks)} chunks • {len({p['document_id'] for p in pages})} documents")

tab1, tab2, tab3, tab4 = st.tabs(["Ask a question", "Entities & relations", "Consistency & revisions", "Dataset & evaluation"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.subheader("Evidence-based question answering")
    question = st.text_input("Ask about components, interfaces, signals or dependencies",
                             placeholder="Which component publishes EngineStatus?")
    if st.button("Search documents", type="primary", disabled=not question.strip()):
        results = retriever.search(question, top_k=top_k)
        response = answer_question(question, results, rel_index, threshold)
        if response["mode"] == "evidence":          # the "no relation matched" case
            req = answer_requirements(question, results)
            if req:
                response = {**response,
                            "mode":"requirement",
                            "answer": req["answer"],
                            "citations": [{**c, "score": round(results[0]["score"], 4)} for c in req["citations"]]}
        st.session_state["last"] = {"question": question, "results": results, "response": response, "mode": retriever.mode}
        log_event("query", {"question": question, "retrieval_mode": retriever.mode, "answer_mode": response["mode"],
                            "abstained": response["abstained"], "top_score": round(results[0]["score"], 4) if results else None,
                            "citations": [f"{c['document_id']}:p{c['page']}" for c in response["citations"]]})
    last = st.session_state.get("last")
    if last:
        response, results = last["response"], last["results"]
        label = {"structured": "Structured answer from extracted relations",
         "requirement": "Matching requirement(s)",
         "evidence": "Closest evidence (no relation matched)",
         "abstain": "Abstained"}.get(response["mode"], "Answer")
        st.markdown(f"### Answer — {label}")
        if response["abstained"]:
            st.warning(response["answer"])
        else:
            st.write(response["answer"])
        st.caption(f"Retrieval mode: {last['mode']} • threshold {threshold:.3f}")
        if response["citations"]:
            st.markdown("### Sources")
            for cite in response["citations"]:
                st.write(f"- **{cite['source']}**, page {cite['page']} (document {cite['document_id']}; retrieval score {cite['score']})")
        if response["facts"]:
            with st.expander("Evidence lines behind each fact"):
                for f in response["facts"]:
                    st.write(f"**{f['doc']}** — {f['evidence']}")
        st.markdown("### Retrieved passages")
        for i, row in enumerate(results, 1):
            with st.expander(f"{i}. {row['source']} — page {row['page']} — score {row['score']:.3f}"):
                st.write(row["text"])
        with st.form("review_form"):
            st.markdown("**Reviewer decision**")
            decision = st.radio("Is this answer acceptable?", ["Accept", "Needs correction", "Reject"], horizontal=True)
            note = st.text_input("Reviewer note (optional)")
            if st.form_submit_button("Save decision"):
                log_event("review", {"question": last["question"], "decision": decision, "note": note,
                                     "answer_mode": response["mode"]})
                st.success("Decision saved to logs/review_log.jsonl")
    st.caption("Answers are built only from relations found in the retrieved documents, or the app abstains. It is not an unrestricted LLM.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Entity extraction")
    all_text = "\n".join(p["text"] for p in pages)
    entity_model = load_model()
    if entity_model is not None:
        st.caption("Trained classifier (logistic regression on identifier shape + sentence context). See reports/entity_metrics.json.")
        entities = predict_entities(all_text, entity_model)
        kinds = ("component", "interface", "port", "signal")
    else:
        st.warning("No trained model found. Run `python -m src.evaluate_entities` to train it. Showing the regex fallback.")
        entities = extract_entities(all_text)
        kinds = ("component", "interface", "signal")
    cols = st.columns(len(kinds))
    for col, kind in zip(cols, kinds):
        with col:
            st.metric(f"Unique {kind}s", len(entities[kind]))
            st.write(entities[kind][:40] or ["None detected"])
            if len(entities[kind]) > 40:
                st.caption(f"+{len(entities[kind]) - 40} more (see CSV export)")
    export_rows = [{"entity": e, "type": k} for k in kinds for e in entities[k]]
    if export_rows:
        st.download_button("Export entities (CSV)", pd.DataFrame(export_rows).to_csv(index=False),
                           file_name="hld_entities.csv", mime="text/csv")
    st.subheader("Relations (publishes / consumes / provides / depends on)")
    if rel_index:
        rel_df = pd.DataFrame(rel_index).drop(columns=["evidence"], errors="ignore")
        rel_df = rel_df[[c for c in ["doc", "page", "type", "component", "signal", "interface", "port", "target", "consumer"] if c in rel_df]]
        kind_filter = st.multiselect("Relation types", sorted(rel_df["type"].unique()), default=sorted(rel_df["type"].unique()))
        shown = rel_df[rel_df["type"].isin(kind_filter)]
        st.dataframe(shown, use_container_width=True)
        st.download_button("Export relations (CSV)", shown.to_csv(index=False), file_name="hld_relations.csv", mime="text/csv")
    else:
        st.info("No relations were extracted from the loaded documents.")
    st.write("Predictions are suggestions and should be checked by a human reviewer.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Consistency findings (per document)")
    docs = {}
    for p in pages:
        docs.setdefault(p["document_id"], []).append(p["text"])
    findings = []
    for doc_id, texts in docs.items():
        for f in check_consistency("\n".join(texts)):
            findings.append({"document": doc_id, **f})
    if findings:
        fdf = pd.DataFrame(findings)
        st.dataframe(fdf, use_container_width=True)
        st.download_button("Export findings (CSV)", fdf.to_csv(index=False), file_name="hld_findings.csv", mime="text/csv")
    else:
        st.success("No findings were flagged by the deterministic rules.")
    st.caption("Rules: component used but not in the Component Catalogue; signal consumed but never published; "
               "signal published but never consumed. Advisory only, not a full AUTOSAR compliance checker.")
    st.markdown("#### Compare two documents (revision check)")
    doc_ids = list(docs)
    if len(doc_ids) >= 2:
        c1, c2 = st.columns(2)
        old_id = c1.selectbox("Earlier document", doc_ids, index=0)
        new_id = c2.selectbox("Later document", doc_ids, index=1)
        if st.button("Compare"):
            old_text, new_text = "\n".join(docs[old_id]), "\n".join(docs[new_id])
            model = load_model()
            if model is not None:
                old_e, new_e = predict_entities(old_text, model), predict_entities(new_text, model)
                rows = []
                for kind in ("component", "interface", "port", "signal"):
                    rows += [{"change": "removed", "entity_type": kind, "entity": e} for e in sorted(set(old_e[kind]) - set(new_e[kind]))]
                    rows += [{"change": "added", "entity_type": kind, "entity": e} for e in sorted(set(new_e[kind]) - set(old_e[kind]))]
                st.markdown("**Entity changes**")
                st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame([{"result": "No entity changes detected"}]), use_container_width=True)
            rel_changes = compare_relations(old_text, new_text)
            st.markdown("**Relation changes**")
            st.dataframe(pd.DataFrame(rel_changes) if rel_changes else pd.DataFrame([{"result": "No relation changes detected"}]), use_container_width=True)
    else:
        st.info("Load at least two documents to compare revisions.")

# ---------------------------------------------------------------- tab 4
with tab4:
    st.subheader("Current knowledge base")
    summary = pd.DataFrame([{"document_id": p["document_id"], "source": p["source"], "page": p["page"], "characters": len(p["text"])} for p in pages])
    st.dataframe(summary, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total chunks", len(chunks))
    m2.metric("Retrieval mode", retriever.mode)
    m3.metric("Relations extracted", len(rel_index))

    st.markdown("#### Retrieval comparison (synthetic evaluation set)")
    rows = []
    for mode in ("tfidf", "sentence-transformers", "hybrid"):
        d = load_json(f"retrieval_eval_{mode}.json")
        if d and d.get("available"):
            bt = d["by_type"]
            rows.append({"retriever": mode, "overall term@3": bt["overall"]["term_hit@3"], "overall MRR": bt["overall"]["term_mrr"],
                         "keyword term@3": bt["keyword"]["term_hit@3"], "paraphrase term@3": bt["paraphrase"]["term_hit@3"],
                         "paraphrase MRR": bt["paraphrase"]["term_mrr"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Run `python -m src.evaluate_retrieval` to generate the comparison.")

    st.markdown("#### End-to-end answers (retrieval + abstention + structured answer)")
    arows = []
    for mode in ("tfidf", "hybrid"):
        d = load_json(f"answer_eval_{mode}.json")
        if d:
            arows.append({"retriever": mode, **{k: v for k, v in d.items() if k not in ("mode", "note")}})
    if arows:
        st.dataframe(pd.DataFrame(arows), use_container_width=True)
    else:
        st.info("Run `python -m src.evaluate_answers`.")

    rel_eval = load_json("relation_eval.json")
    if rel_eval:
        st.markdown("#### Relation extraction and consistency checks vs manifest")
        st.json(rel_eval)

    fa = load_json("feature_analysis.json")
    if fa:
        st.markdown("#### Entity classifier comparison (document-grouped CV and held-out documents)")
        comp = pd.DataFrame(fa["classifier_comparison"]).T[["macro_f1_mean", "macro_f1_std", "heldout_accuracy", "heldout_macro_f1", "seconds"]]
        st.dataframe(comp, use_container_width=True)
        st.markdown("#### Chi-square feature selection (macro-F1 by number of features kept)")
        st.write({k: v["macro_f1_mean"] for k, v in fa["feature_selection"]["by_k"].items()})
        pca = fa.get("pca", {})
        if pca.get("available"):
            st.markdown(f"#### PCA on embeddings (90% variance needs {pca['components_for_90pct_variance']} components)")
            st.dataframe(pd.DataFrame(pca["rows"]), use_container_width=True)
            for img in ("pca_explained_variance.png", "pca_chunks.png"):
                if (REPORTS / img).exists():
                    st.image(str(REPORTS / img))
    st.caption("All numbers come from a small synthetic, template-generated corpus: relative comparisons, not real-world accuracy.")

st.divider()
st.caption("Educational prototype only. Do not use its output as the sole basis for safety, cybersecurity, compliance or vehicle release decisions.")
