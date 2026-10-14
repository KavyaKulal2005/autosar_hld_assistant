"""Train and evaluate the entity classifier on held-out synthetic documents,
and compare against a hard-coded-regex baseline and a suffix-rule baseline."""
import json
import re
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.synthetic_data import generate, OUT_DIR
from src import entity_model as em
from src.analysis import ENTITY_PATTERNS

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["component", "interface", "port", "signal"]

# Hand-labelled gold for the two original sample documents (second, out-of-distribution test).
SAMPLE_GOLD = {
    "HLD-01": {"SensorManager": "component", "SpeedSensor": "component", "VehicleGateway": "component",
               "BrakeController": "component", "BrakeModule": "component", "BodyControl": "component",
               "VehicleSpeed": "signal", "BrakeStatus": "signal", "DoorStatus": "signal",
               "CabinLightCommand": "signal", "SpeedData": "interface", "GatewayInput": "interface",
               "BrakeStatusPort": "port"},
    "HLD-02": {"VehicleGateway": "component", "BodyControl": "component", "DoorSensor": "component",
               "VehicleSpeed": "signal", "GatewayHealth": "signal", "DoorStatus": "signal",
               "CabinLightCommand": "signal", "SpeedData": "interface",
               "GatewayStatusPort": "port", "DoorStatusPort": "port"},
}


def regex_baseline(token):
    """The original hard-coded regex from src/analysis.py (generously splits *Port into 'port')."""
    for kind in ("component", "signal", "interface"):
        if re.fullmatch(ENTITY_PATTERNS[kind], token):
            return "port" if kind == "interface" and token.endswith("Port") else kind
    return "other"


def suffix_baseline(token):
    t = token.lower()
    if t.endswith(("manager", "controller", "module", "handler", "monitor", "driver", "sensor", "gateway", "control")):
        return "component"
    if t.endswith(("port", "channel", "input", "output")):
        return "port"
    if t.endswith(("interface", "data", "service", "info")):
        return "interface"
    if t.endswith(("speed", "status", "health", "command", "torque", "temperature", "angle", "voltage")):
        return "signal"
    return "other"


def _entity_level(docs, predict_fn):
    """Unique (doc, token) entity-level gold vs predicted lists."""
    gold, pred = [], []
    for text, labels in docs:
        for token in sorted({t for t, _, _ in em.find_mentions(text)} & set(labels)):
            gold.append(labels[token])
            pred.append(predict_fn(text, token))
    return gold, pred


def _report(gold, pred):
    rep = classification_report(gold, pred, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "accuracy": round(accuracy_score(gold, pred), 3),
        "macro_f1": round(f1_score(gold, pred, labels=CLASSES, average="macro", zero_division=0), 3),
        "per_class": {c: {k: round(rep[c][k], 3) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])}
                      for c in CLASSES},
    }


def run():
    meta = generate()
    def load(split):
        docs = []
        for doc_id, info in meta.items():
            if info["split"] == split:
                docs.append(((OUT_DIR / f"{doc_id}.txt").read_text(encoding="utf-8"), info["labels"]))
        return docs
    train_docs, test_docs = load("train"), load("test")
    model = em.train(train_docs)

    def ml_fn_factory(docs):
        cache = {id(text): em.predict_entities(text, model) for text, _ in docs}
        def fn(text, token):
            for kind, toks in cache[id(text)].items():
                if token in toks:
                    return kind
            return "other"
        return fn

    results = {"train_documents": len(train_docs), "test_documents": len(test_docs), "synthetic_held_out": {}, "original_samples": {}}
    for name, fn in (("regex_baseline", lambda t, tok: regex_baseline(tok)),
                     ("suffix_rule_baseline", lambda t, tok: suffix_baseline(tok)),
                     ("trained_logreg", ml_fn_factory(test_docs))):
        g, p = _entity_level(test_docs, fn)
        results["synthetic_held_out"][name] = _report(g, p)

    sample_docs = []
    for doc_id, gold in SAMPLE_GOLD.items():
        text = (ROOT / "data" / "sample_documents" / f"{doc_id}.txt").read_text(encoding="utf-8")
        sample_docs.append((text, gold))
    for name, fn in (("regex_baseline", lambda t, tok: regex_baseline(tok)),
                     ("suffix_rule_baseline", lambda t, tok: suffix_baseline(tok)),
                     ("trained_logreg", ml_fn_factory(sample_docs))):
        g, p = _entity_level(sample_docs, fn)
        results["original_samples"][name] = _report(g, p)
    results["note"] = ("Synthetic, template-generated data. The original-sample gold labels were assigned by hand. "
                       "Scores show relative behaviour on unseen identifiers, not real-world AUTOSAR accuracy.")
    out = ROOT / "reports" / "entity_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def print_summary(r):
    for split in ("synthetic_held_out", "original_samples"):
        print(f"\n== {split} ==")
        print(f"{'method':24s}{'accuracy':>10s}{'macro_F1':>10s}")
        for name, rep in r[split].items():
            print(f"{name:24s}{rep['accuracy']:>10}{rep['macro_f1']:>10}")
    print(f"\nSaved: {ROOT / 'reports' / 'entity_metrics.json'}")


if __name__ == "__main__":
    print_summary(run())
