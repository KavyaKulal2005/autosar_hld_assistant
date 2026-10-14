"""Evaluate rule-based relation extraction and consistency checks against the synthetic manifest.

Gold relations come from manifest['relations']; gold findings from manifest['issues'].
Relations that the manifest does not list but that exist in the text only because an issue was
injected (e.g. a component that consumes an unpublished signal) are counted separately.
Synthetic, template-generated data: results show the rules work on this phrasing, not on real HLDs.
"""
import json
from src.relations import extract_relations, check_consistency, relation_key
from src.evaluate_retrieval import CORPUS, REPORTS


def _gold_key(r):
    if r["type"] == "provides":
        return ("provides", r["component"], r["interface"], r.get("consumer"))
    return relation_key({**r, "port": r.get("port")})


def run():
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    tp = fn = extra = extra_explained = 0
    f_tp = f_fp = f_fn = 0
    for doc, meta in manifest.items():
        text = (CORPUS / f"{doc}.txt").read_text(encoding="utf-8")
        got = {relation_key(r) for r in extract_relations(text)}
        gold = {_gold_key(r) for r in meta["relations"]}
        tp += len(got & gold)
        fn += len(gold - got)
        issue_items = {i["item"] for i in meta["issues"]}
        for k in got - gold:
            extra += 1
            extra_explained += any(x in issue_items for x in k[1:])
        found = {(f["type"], f["item"]) for f in check_consistency(text) if f["type"] in ("undefined_component", "orphan_signal")}
        want = {(i["type"], i["item"]) for i in meta["issues"]}
        f_tp += len(found & want)
        f_fp += len(found - want)
        f_fn += len(want - found)
    out = {"documents": len(manifest), "gold_relations_found": tp, "gold_relations_missed": fn,
           "relation_recall": round(tp / max(1, tp + fn), 3),
           "extra_relations_not_in_manifest": extra, "extra_explained_by_injected_issues": extra_explained,
           "consistency_findings": {"true_positive": f_tp, "false_positive": f_fp, "false_negative": f_fn,
                                    "precision": round(f_tp / max(1, f_tp + f_fp), 3),
                                    "recall": round(f_tp / max(1, f_tp + f_fn), 3)},
           "note": "Synthetic, template-generated data. Rules match the generator's phrasing; real HLDs will need broader patterns."}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "relation_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
