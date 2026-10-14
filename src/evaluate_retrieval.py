"""Retrieval evaluation on the synthetic HLD corpus: TF-IDF vs Sentence Transformers.

Questions are generated from data/corpus/manifest.json, so every gold answer is known.
Question types: keyword, paraphrase, interface, dependency, unanswerable.
Reports chunk-level term hit@k and MRR, document hit@k, abstention quality (threshold
tuned on a dev split, reported on a test split), answer quality and citation validity,
and writes per-question failures for error analysis.
Synthetic data only: scores are relative comparisons, not real-world accuracy.
"""
import csv
import json
import random
import re
from pathlib import Path
from src.document_loader import load_sample_documents, chunk_pages
from src.retrieval import Retriever, HybridRetriever, answer_from_evidence
from src.entity_model import CANDIDATE

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
REPORTS = ROOT / "reports"
KS = (1, 3, 5)

FAKE_SIGNALS = ["ThrottleLeakRate", "TyrePressureDelta", "HornDutyCycle", "RadarBlindness"]
OUT_OF_SCOPE = ["What is the CAN FD data phase bit rate?", "Who approved this architecture?",
                "What is the ASIL level of the braking function?", "Which OEM owns this design?"]
PARAPHRASES = ["Which software component is the source of the {s} signal?", "What produces the {s} signal?",
               "Where does the {s} signal originate?"]


def words(name):
    """EngineStatus -> 'engine status' (removes the exact-identifier lexical match)."""
    return " ".join(w.lower() for w in re.findall(r"[A-Z][a-z0-9]*", name))


def build_questions(manifest, seed=11):
    rng = random.Random(seed)
    pub, prov, dep = {}, {}, {}
    for doc, m in manifest.items():
        for r in m["relations"]:
            if r["type"] == "publishes":
                pub.setdefault(r["signal"], []).append((doc, r["component"]))
            elif r["type"] == "provides":
                prov.setdefault(r["interface"], []).append((doc, r["component"]))
            elif r["type"] == "depends_on":
                dep.setdefault((r["component"], r["signal"]), []).append((doc, r["target"]))
    qs = []
    sigs = sorted(pub)
    rng.shuffle(sigs)
    for s in sigs[:8]:
        qs.append({"question": f"Which component publishes {s}?", "type": "keyword", "gold": pub[s]})
    for k, s in enumerate(sigs[8:16]):
        qs.append({"question": PARAPHRASES[k % 3].format(s=words(s)), "type": "paraphrase", "gold": pub[s]})
    ifs = sorted(prov)
    rng.shuffle(ifs)
    for i in ifs[:3]:
        qs.append({"question": f"Which component provides the {i} interface?", "type": "keyword", "gold": prov[i]})
    for i in ifs[3:6]:
        qs.append({"question": f"Who offers the {words(i)} interface to other components?", "type": "paraphrase", "gold": prov[i]})
    deps = sorted(dep)
    rng.shuffle(deps)
    for c, s in deps[:3]:
        qs.append({"question": f"What does {c} depend on for {s}?", "type": "keyword", "gold": dep[(c, s)]})
    for c, s in deps[3:6]:
        qs.append({"question": f"What does the {words(c)} rely on for {words(s)}?", "type": "paraphrase", "gold": dep[(c, s)]})
    for s in FAKE_SIGNALS:
        qs.append({"question": f"Which component publishes {s}?", "type": "unanswerable", "gold": []})
    for text in OUT_OF_SCOPE:
        qs.append({"question": text, "type": "unanswerable", "gold": []})
    rng.shuffle(qs)
    for idx, q in enumerate(qs):
        q["answerable"] = bool(q["gold"])
        q["split"] = "dev" if idx % 2 == 0 else "test"
    return qs


def _terms_by_doc(gold):
    out = {}
    for doc, term in gold:
        out.setdefault(doc, set()).add(term)
    return out


def _rank_metrics(results, gold):
    terms = _terms_by_doc(gold)
    doc_rank = next((i + 1 for i, r in enumerate(results) if r["document_id"] in terms), None)
    term_rank = next((i + 1 for i, r in enumerate(results)
                      if r["document_id"] in terms and any(t in r["text"] for t in terms[r["document_id"]])), None)
    return doc_rank, term_rank


def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def _tune_threshold(rows, key="top_score"):
    dev = [r for r in rows if r["split"] == "dev"]
    cands = sorted({r[key] for r in dev})
    best, best_t = -1, 0.0
    for t in cands:
        ans = [r for r in dev if r["answerable"]]
        una = [r for r in dev if not r["answerable"]]
        tpr = sum(r[key] >= t for r in ans) / max(1, len(ans))
        tnr = sum(r[key] < t for r in una) / max(1, len(una))
        bal = (tpr + tnr) / 2
        if bal > best or (bal == best and t > best_t):
            best, best_t = bal, t
    return best_t, best


def _balanced(rows, t, key="top_score"):
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    tpr = sum(r[key] >= t for r in ans) / max(1, len(ans))
    tnr = sum(r[key] < t for r in una) / max(1, len(una))
    return round((tpr + tnr) / 2, 3), round(tpr, 3), round(tnr, 3)


def evaluate_mode(chunks, pages, questions, prefer_embeddings, retriever=None):
    if retriever is None:
        retriever = Retriever(chunks, prefer_embeddings=prefer_embeddings)
        wanted = "sentence-transformers" if prefer_embeddings else "tfidf"
    else:
        wanted = retriever.mode
    if retriever.mode != wanted:
        return {"mode": wanted, "available": False,
                "reason": "Embedding model could not be loaded; install sentence-transformers and ensure the model is downloaded."}
    valid_pages = {(p["document_id"], p["page"]) for p in pages}
    rows = []
    for q in questions:
        res = retriever.search(q["question"], top_k=max(KS))
        doc_rank, term_rank = _rank_metrics(res, q["gold"]) if q["answerable"] else (None, None)
        ids = CANDIDATE.findall(q["question"])
        top_text = " ".join(x["text"] for x in res[:3])
        ids_missing = bool(ids) and any(i not in top_text for i in ids)
        top = res[0]["score"] if res else 0.0
        rows.append({**q, "top_score": top, "results": res, "doc_rank": doc_rank, "term_rank": term_rank,
                     "ids_missing": ids_missing, "eff_score": -1.0 if ids_missing else top})
    answerable = [r for r in rows if r["answerable"]]
    by_type = {}
    for typ in ("keyword", "paraphrase", "overall"):
        sub = answerable if typ == "overall" else [r for r in answerable if r["type"] == typ]
        entry = {"n": len(sub)}
        for k in KS:
            entry[f"doc_hit@{k}"] = _mean([1.0 if r["doc_rank"] and r["doc_rank"] <= k else 0.0 for r in sub])
            entry[f"term_hit@{k}"] = _mean([1.0 if r["term_rank"] and r["term_rank"] <= k else 0.0 for r in sub])
        entry["term_mrr"] = _mean([1 / r["term_rank"] if r["term_rank"] else 0.0 for r in sub])
        by_type[typ] = entry
    test_ans = [r for r in answerable if r["split"] == "test"]
    test_split = {"n": len(test_ans),
                  "term_hit@3": _mean([1.0 if r["term_rank"] and r["term_rank"] <= 3 else 0.0 for r in test_ans]),
                  "term_mrr": _mean([1 / r["term_rank"] if r["term_rank"] else 0.0 for r in test_ans])}
    tau, dev_bal = _tune_threshold(rows)
    test_rows = [r for r in rows if r["split"] == "test"]
    test_bal, test_tpr, test_tnr = _balanced(test_rows, tau)
    tau_h, dev_bal_h = _tune_threshold(rows, "eff_score")
    h_bal, h_tpr, h_tnr = _balanced(test_rows, tau_h, "eff_score")
    default_t = 0.0 if wanted == "tfidf" else 0.01
    una = [r for r in rows if not r["answerable"]]
    default_abstain = _mean([1.0 if r["top_score"] < default_t or not r["results"] else 0.0 for r in una])
    contains, valid, answered = [], [], 0
    for r in answerable:
        ans = ({"abstained": True} if r["ids_missing"]
               else answer_from_evidence(r["question"], r["results"][:3], min_score=tau_h))
        if ans["abstained"]:
            contains.append(0.0)
            continue
        answered += 1
        terms = {t for _, t in r["gold"]}
        contains.append(1.0 if any(t in ans["answer"] for t in terms) else 0.0)
        valid.append(1.0 if all((c["document_id"], c["page"]) in valid_pages for c in ans["citations"]) else 0.0)
    failures = [{"question": r["question"], "type": r["type"], "gold": json.dumps(r["gold"]),
                 "top_score": round(r["top_score"], 4),
                 "retrieved_docs": " ".join(x["document_id"] for x in r["results"][:3])}
                for r in answerable if not (r["term_rank"] and r["term_rank"] <= 3)]
    return {"mode": retriever.mode, "available": True,
            "n_answerable": len(answerable), "n_unanswerable": len(una),
            "by_type": by_type, "test_split_only": test_split,
            "abstention": {"score_threshold_only": {"threshold": round(tau, 4), "dev_balanced_accuracy": round(dev_bal, 3),
                                                    "test_balanced_accuracy": test_bal, "test_answer_rate": test_tpr,
                                                    "test_abstain_rate": test_tnr},
                           "identifier_grounding_plus_threshold": {"threshold": round(tau_h, 4),
                                                                   "dev_balanced_accuracy": round(dev_bal_h, 3),
                                                                   "test_balanced_accuracy": h_bal,
                                                                   "test_answer_rate": h_tpr, "test_abstain_rate": h_tnr},
                           "app_default_threshold": default_t,
                           "app_default_abstain_rate_on_unanswerable": default_abstain},
            "answer_quality": {"answered_questions": answered,
                               "answer_contains_gold_term_rate": _mean(contains),
                               "citation_validity_rate": _mean(valid)},
            "failures_at_3": failures,
            "note": "Synthetic, template-generated corpus. Scores are relative, not real-world accuracy."}


ALPHAS = [round(a * 0.1, 1) for a in range(11)]


def tune_alpha(base, questions):
    """Pick the fusion weight that maximises dev-split term MRR (answerable questions only)."""
    dev = [q for q in questions if q["answerable"] and q["split"] == "dev"]
    best_a, best_m = 0.5, -1.0
    for a in ALPHAS:
        hr = HybridRetriever(base, a)
        ranks = [_rank_metrics(hr.search(q["question"], top_k=max(KS)), q["gold"])[1] for q in dev]
        m = sum(1 / r for r in ranks if r) / max(1, len(dev))
        if m > best_m + 1e-9:
            best_a, best_m = a, m
    return best_a, round(best_m, 3)


def run():
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    pages = load_sample_documents(CORPUS)
    chunks = chunk_pages(pages)
    questions = build_questions(manifest)
    REPORTS.mkdir(exist_ok=True)
    with open(ROOT / "data" / "eval_questions_corpus.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "type", "answerable", "split", "gold"])
        for q in questions:
            w.writerow([q["question"], q["type"], q["answerable"], q["split"], json.dumps(q["gold"])])
    results = {}
    for emb in (False, True):
        r = evaluate_mode(chunks, pages, questions, emb)
        results[r["mode"]] = r
        (REPORTS / f"retrieval_eval_{r['mode']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        if r["available"] and r["failures_at_3"]:
            with open(REPORTS / f"retrieval_failures_{r['mode']}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(r["failures_at_3"][0]))
                w.writeheader()
                w.writerows(r["failures_at_3"])
    emb = results.get("sentence-transformers")
    if emb and emb["available"]:
        base = Retriever(chunks, prefer_embeddings=True)
        alpha, dev_mrr = tune_alpha(base, questions)
        r = evaluate_mode(chunks, pages, questions, True, retriever=HybridRetriever(base, alpha))
        r["alpha_tfidf_weight"], r["alpha_dev_term_mrr"] = alpha, dev_mrr
        results["hybrid"] = r
        (REPORTS / "retrieval_eval_hybrid.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        if r["failures_at_3"]:
            with open(REPORTS / "retrieval_failures_hybrid.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(r["failures_at_3"][0]))
                w.writeheader()
                w.writerows(r["failures_at_3"])
    return results, len(chunks), len(questions)


def print_summary(results, n_chunks, n_questions):
    print(f"\nCorpus chunks: {n_chunks} | questions: {n_questions}")
    for mode, r in results.items():
        if not r["available"]:
            print(f"\n[{mode}] SKIPPED: {r['reason']}")
            continue
        print(f"\n[{mode}]  (answerable={r['n_answerable']}, unanswerable={r['n_unanswerable']})")
        print(f"{'type':12s}{'n':>3s}{'doc@1':>7s}{'doc@3':>7s}{'term@1':>8s}{'term@3':>8s}{'term@5':>8s}{'MRR':>7s}")
        for typ, e in r["by_type"].items():
            print(f"{typ:12s}{e['n']:>3d}{e['doc_hit@1']:>7}{e['doc_hit@3']:>7}{e['term_hit@1']:>8}{e['term_hit@3']:>8}{e['term_hit@5']:>8}{e['term_mrr']:>7}")
        t = r["test_split_only"]
        extra = f" | TF-IDF weight alpha={r['alpha_tfidf_weight']}" if "alpha_tfidf_weight" in r else ""
        print(f"test split only (n={t['n']}): term@3={t['term_hit@3']} term_MRR={t['term_mrr']}{extra}")
        a = r["abstention"]
        for name in ("score_threshold_only", "identifier_grounding_plus_threshold"):
            x = a[name]
            print(f"abstention [{name}]: threshold={x['threshold']} test balanced acc={x['test_balanced_accuracy']} "
                  f"(answers {x['test_answer_rate']}, abstains {x['test_abstain_rate']})")
        print(f"app default threshold {a['app_default_threshold']} abstains on {a['app_default_abstain_rate_on_unanswerable']} of unanswerable")
        q = r["answer_quality"]
        print(f"answer contains gold term: {q['answer_contains_gold_term_rate']} | citation validity: {q['citation_validity_rate']}")
    print(f"\nSaved per-mode JSON and failure CSVs in {REPORTS}")


if __name__ == "__main__":
    print_summary(*run())
