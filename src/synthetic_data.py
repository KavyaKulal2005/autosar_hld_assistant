"""Generate labelled synthetic AUTOSAR-style HLD documents for entity-recognition experiments.

Every document is plain text plus a labels.json mapping identifier -> entity type.
Train documents and test documents use different name vocabularies so the test
split measures generalisation to unseen identifiers (not memorisation).
Synthetic data only: not real AUTOSAR or vehicle content.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synthetic_hlds"

VOCAB = {
    "train": {
        "comp_prefix": ["Sensor", "Brake", "Door", "Engine", "Gateway", "Body", "Climate", "Steering", "Battery", "Network"],
        "comp_suffix": ["Manager", "Controller", "Module", "Handler", "Monitor", "Service", "Driver"],
        "sig_prefix": ["Vehicle", "Wheel", "Engine", "Brake", "Door", "Battery", "Cabin", "Steering", "Gear"],
        "sig_suffix": ["Speed", "Status", "Health", "Command", "Torque", "Temperature", "Angle", "Voltage"],
        "port_alt": ["Link", "Tap", "Line"],
        "iface_alt": ["Api", "Contract", "Facade"],
    },
    "test": {
        "comp_prefix": ["Wiper", "Seat", "Parking", "Cruise", "Lighting", "Diag", "Charging"],
        "comp_suffix": ["Manager", "Controller", "Monitor", "Coordinator", "Supervisor", "Driver"],
        "sig_prefix": ["Wiper", "Seat", "Parking", "Cruise", "Ambient", "Charge", "Pedal"],
        "sig_suffix": ["Speed", "State", "Request", "Position", "Torque", "Level", "Current"],
        "port_alt": ["Link", "Hook", "Line"],
        "iface_alt": ["Api", "Bridge", "Contract"],
    },
}
PORT_SUFFIX = ["Port", "Channel", "Input", "Output"]
IFACE_SUFFIX = ["Interface", "Data", "Service", "Info"]
DISTRACTORS = ["RevisionHistory", "DesignReview", "SafetyCase", "ReleaseNotes", "SystemOverview",
               "TestPlan", "ChangeRequest", "BuildConfig"]


def _names(rng, vocab, kind, n):
    out = set()
    while len(out) < n:
        if kind == "component":
            out.add(rng.choice(vocab["comp_prefix"]) + rng.choice(vocab["comp_suffix"]))
        elif kind == "signal":
            out.add(rng.choice(vocab["sig_prefix"]) + rng.choice(vocab["sig_suffix"]))
        elif kind == "port":
            base = rng.choice(vocab["sig_prefix"]) + rng.choice(["Status", "Data", "Cmd", "Feed"])
            suffix = rng.choice(vocab["port_alt"]) if rng.random() < 0.35 else rng.choice(PORT_SUFFIX)
            out.add(base + suffix)
        else:
            base = rng.choice(vocab["comp_prefix"]) + rng.choice(["Speed", "Torque", "State", "Mode"])
            suffix = rng.choice(vocab["iface_alt"]) if rng.random() < 0.35 else rng.choice(IFACE_SUFFIX)
            out.add(base + suffix)
    return sorted(out)


def _sentence(rng, comps, sigs, ports, ifaces, split="train"):
    c, c2 = rng.sample(comps, 2)
    s = rng.choice(sigs)
    p = rng.choice(ports)
    i = rng.choice(ifaces)
    templates = [
        f"The {c} publishes the {s} signal via the {p} port.",
        f"The {c} sends {s} via {p}.",
        f"{p} carries {s} from the {c}.",
        f"The {c} publishes {s} through {p}.",
        f"The {c} provides the {i} interface to the {c2} component.",
        f"The {c} consumes {s} through the {i} interface.",
        f"The {c} exposes {i} for {s} exchange.",
        f"The {c} uses {i}.",
        f"The {c} receives {s} from {c2}.",
        f"The {c} component depends on {c2} for {s}.",
        f"The {c} forwards selected signals to the {c2} component.",
        f"The {c} publishes {s}.",
        f"The {c} consumes {s}.",
    ]
    if split == "test" and rng.random() < 0.5:
        # Phrasings never seen in training: tests robustness to unseen wording.
        templates = [
            f"{s} is emitted by the {c} on {p}.",
            f"Via {p}, the {c} distributes {s}.",
            f"The {c} offers {i} to {c2}.",
            f"Access to {s} is granted through {i}.",
            f"{c} and {c2} exchange {s}.",
            f"The {c2} subscribes to {s} from {c}.",
        ]
    return rng.choice(templates)


def make_document(doc_id, split, seed):
    rng = random.Random(seed)
    vocab = VOCAB[split]
    comps = _names(rng, vocab, "component", rng.randint(4, 6))
    sigs = _names(rng, vocab, "signal", rng.randint(4, 6))
    ports = _names(rng, vocab, "port", rng.randint(2, 3))
    ifaces = _names(rng, vocab, "interface", rng.randint(2, 3))
    lines = [f"DOCUMENT: {doc_id}", f"TITLE: Synthetic Architecture {doc_id}", "REVISION: A", "SECTION: Overview"]
    for _ in range(rng.randint(14, 20)):
        lines.append(_sentence(rng, comps, sigs, ports, ifaces, split))
    for d in rng.sample(DISTRACTORS, 2):
        lines.append(f"See the {d} for further details.")
    text = "\n".join(lines)
    labels = {}
    for names, kind in ((comps, "component"), (sigs, "signal"), (ports, "port"), (ifaces, "interface"), (DISTRACTORS, "other")):
        for n in names:
            if n in text:
                labels[n] = kind
    # Names that collide across roles would make labels ambiguous; drop them.
    roles = {}
    for names, kind in ((comps, "component"), (sigs, "signal"), (ports, "port"), (ifaces, "interface")):
        for n in names:
            roles.setdefault(n, set()).add(kind)
    for n, r in roles.items():
        if len(r) > 1:
            labels.pop(n, None)
    return text, labels


def generate(n_train=12, n_test=5, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for split, n, base in (("train", n_train, 100), ("test", n_test, 900)):
        for k in range(n):
            doc_id = f"SYN-{split.upper()}-{k+1:02d}"
            text, labels = make_document(doc_id, split, base + k)
            (out_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            meta[doc_id] = {"split": split, "labels": labels}
    (out_dir / "labels.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    m = generate()
    print(f"Generated {len(m)} documents in {OUT_DIR}")
