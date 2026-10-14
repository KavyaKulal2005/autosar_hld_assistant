"""Feature engineering, selection and dimensionality-reduction experiments.

1. Entity classifier: compare 4 model families with document-grouped cross-validation,
   tune logistic-regression C, and test chi-square feature selection.
2. PCA on chunk embeddings: explained variance, 2-D map of chunks, and how retrieval
   quality changes as embeddings are compressed to fewer dimensions.
Synthetic data only: results are relative comparisons, not real-world accuracy.
"""
import csv
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from src import entity_model as em
from src.synthetic_data import generate, OUT_DIR
from src.document_loader import load_sample_documents, chunk_pages
from src.retrieval import Retriever
from src.evaluate_retrieval import build_questions, _rank_metrics, CORPUS

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PCA_DIMS = [5, 10, 20, 40, 70]


def _mention_table():
    meta = generate()
    X, y, groups, is_test = [], [], [], []
    for gi, (doc_id, info) in enumerate(sorted(meta.items())):
        text = (OUT_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
        xs, ys, _ = em.build_dataset([(text, info["labels"])])
        X += xs
        y += ys
        groups += [gi] * len(xs)
        is_test += [info["split"] == "test"] * len(xs)
    return X, np.array(y), np.array(groups), np.array(is_test)


def _cv(make_pipe, X, y, groups, n_splits=5):
    accs, f1s = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pipe = make_pipe().fit([X[i] for i in tr], y[tr])
        pred = pipe.predict([X[i] for i in te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return {"accuracy_mean": round(float(np.mean(accs)), 3), "macro_f1_mean": round(float(np.mean(f1s)), 3),
            "macro_f1_std": round(float(np.std(f1s)), 3)}


MODELS = {
    "logistic_regression": lambda: make_pipeline(DictVectorizer(sparse=False), LogisticRegression(C=5.0, max_iter=2000)),
    "naive_bayes": lambda: make_pipeline(DictVectorizer(sparse=False), MultinomialNB()),
    "linear_svm": lambda: make_pipeline(DictVectorizer(sparse=False), LinearSVC(C=1.0)),
    "random_forest": lambda: make_pipeline(DictVectorizer(sparse=False), RandomForestClassifier(n_estimators=200, random_state=0)),
}


def compare_classifiers(X, y, groups, is_test):
    out = {}
    tr, te = np.where(~is_test)[0], np.where(is_test)[0]
    for name, make in MODELS.items():
        t0 = time.time()
        out[name] = _cv(make, X, y, groups)
        out[name]["seconds"] = round(time.time() - t0, 2)
        pipe = make().fit([X[i] for i in tr], y[tr])
        pred = pipe.predict([X[i] for i in te])
        out[name]["heldout_accuracy"] = round(accuracy_score(y[te], pred), 3)
        out[name]["heldout_macro_f1"] = round(f1_score(y[te], pred, average="macro", zero_division=0), 3)
    return out


def tune_logreg(X, y, groups):
    rows = {}
    for c in (0.1, 1.0, 5.0, 20.0):
        rows[str(c)] = _cv(lambda c=c: make_pipeline(DictVectorizer(sparse=False), LogisticRegression(C=c, max_iter=2000)), X, y, groups)
    best = max(rows, key=lambda k: rows[k]["macro_f1_mean"])
    return {"by_C": rows, "best_C": float(best)}


def feature_selection(X, y, groups):
    n_total = DictVectorizer(sparse=False).fit_transform(X).shape[1]
    out = {"total_features": n_total, "by_k": {}}
    for k in (25, 50, 100, 200, "all"):
        if k != "all" and k >= n_total:
            continue
        def make(k=k):
            sel = SelectKBest(chi2, k=k if k != "all" else "all")
            return make_pipeline(DictVectorizer(sparse=False), sel, LogisticRegression(C=5.0, max_iter=2000))
        out["by_k"][str(k)] = _cv(make, X, y, groups)
    dv = DictVectorizer(sparse=False)
    M = dv.fit_transform(X)
    scores, _ = chi2(M, y)
    names = np.array(dv.get_feature_names_out())
    top = np.argsort(np.nan_to_num(scores))[::-1][:15]
    out["top_features_by_chi2"] = [{"feature": str(names[i]), "chi2": round(float(scores[i]), 1)} for i in top]
    return out


def pca_experiments(chunks, questions):
    base = Retriever(chunks, prefer_embeddings=True)
    if base.mode != "sentence-transformers":
        return {"available": False, "reason": "Embedding model could not be loaded."}
    E = np.asarray(base.embeddings)
    answerable = [q for q in questions if q["answerable"]]
    Q = np.asarray(base.embedder.encode([q["question"] for q in answerable], normalize_embeddings=True))
    dims = [d for d in PCA_DIMS if d < min(E.shape)]

    def evaluate(Ec, Qc):
        Ec = Ec / (np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-12)
        Qc = Qc / (np.linalg.norm(Qc, axis=1, keepdims=True) + 1e-12)
        ranks = []
        for q, qv in zip(answerable, Qc):
            order = np.argsort(-(Ec @ qv))[:5]
            res = [chunks[i] for i in order]
            ranks.append(_rank_metrics(res, q["gold"])[1])
        return {"term_hit@3": round(sum(1 for r in ranks if r and r <= 3) / len(ranks), 3),
                "term_mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3)}

    rows = [{"dimensions": E.shape[1], "explained_variance": 1.0, **evaluate(E, Q)}]
    for d in dims:
        pca = PCA(n_components=d, random_state=0).fit(E)
        rows.append({"dimensions": d, "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 3),
                     **evaluate(pca.transform(E), pca.transform(Q))})
    rows.sort(key=lambda r: r["dimensions"])
    full = PCA(n_components=min(E.shape) - 1, random_state=0).fit(E)
    cum = np.cumsum(full.explained_variance_ratio_)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cum) + 1), cum)
    plt.xlabel("PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA of chunk embeddings")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "pca_explained_variance.png", dpi=150)
    plt.close()
    P = PCA(n_components=2, random_state=0).fit_transform(E)
    base_ids = sorted({c["document_id"].replace("-B", "") for c in chunks})
    plt.figure(figsize=(7, 5))
    for k, did in enumerate(base_ids):
        idx = [i for i, c in enumerate(chunks) if c["document_id"].replace("-B", "") == did]
        plt.scatter(P[idx, 0], P[idx, 1], s=22, label=did, color=plt.cm.tab10(k % 10))
    plt.legend(fontsize=6, ncol=2)
    plt.title("Chunk embeddings (PCA, 2 components)")
    plt.tight_layout()
    plt.savefig(REPORTS / "pca_chunks.png", dpi=150)
    plt.close()
    with open(REPORTS / "pca_retrieval_dims.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"available": True, "rows": rows, "components_for_90pct_variance": int(np.searchsorted(cum, 0.9) + 1)}


def run():
    REPORTS.mkdir(exist_ok=True)
    X, y, groups, is_test = _mention_table()
    result = {"mentions": len(y), "documents": int(len(set(groups))),
              "classifier_comparison": compare_classifiers(X, y, groups, is_test),
              "logreg_tuning": tune_logreg(X, y, groups),
              "feature_selection": feature_selection(X, y, groups)}
    chunks = chunk_pages(load_sample_documents(CORPUS))
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    result["pca"] = pca_experiments(chunks, build_questions(manifest))
    result["note"] = "Synthetic, template-generated data. Results are relative comparisons only."
    (REPORTS / "feature_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def print_summary(r):
    print(f"\nEntity classifier: {r['mentions']} mentions in {r['documents']} documents (document-grouped 5-fold CV)")
    print("(held-out = train on 12 docs, test on 5 docs with unseen names and unseen phrasing)")
    print(f"{'model':22s}{'cv_F1':>8s}{'cv_std':>8s}{'heldout_acc':>13s}{'heldout_F1':>12s}{'sec':>6s}")
    for name, m in r["classifier_comparison"].items():
        print(f"{name:22s}{m['macro_f1_mean']:>8}{m['macro_f1_std']:>8}{m['heldout_accuracy']:>13}{m['heldout_macro_f1']:>12}{m['seconds']:>6}")
    t = r["logreg_tuning"]
    print("\nLogReg C tuning:", {c: v["macro_f1_mean"] for c, v in t["by_C"].items()}, "-> best C =", t["best_C"])
    fs = r["feature_selection"]
    print(f"\nChi-square selection ({fs['total_features']} features total):",
          {k: v["macro_f1_mean"] for k, v in fs["by_k"].items()})
    print("Top features:", ", ".join(f["feature"] for f in fs["top_features_by_chi2"][:8]))
    p = r["pca"]
    if not p["available"]:
        print(f"\nPCA skipped: {p['reason']}")
    else:
        print(f"\nPCA on chunk embeddings (90% variance needs {p['components_for_90pct_variance']} components):")
        print(f"{'dims':>6s}{'var':>8s}{'term@3':>9s}{'MRR':>8s}")
        for row in p["rows"]:
            print(f"{row['dimensions']:>6d}{row['explained_variance']:>8}{row['term_hit@3']:>9}{row['term_mrr']:>8}")
    print(f"\nSaved: {REPORTS / 'feature_analysis.json'} and PCA plots in reports/")


if __name__ == "__main__":
    print_summary(run())
