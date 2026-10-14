"""Build a larger synthetic HLD corpus (structured sections, catalogue, tables, revisions).

Produces data/corpus/*.txt and data/corpus/manifest.json containing, per document:
entity labels, relations (publish/consume/provide/depend) and injected issues,
so retrieval, extraction and consistency checks can be evaluated against known truth.
Synthetic data only: not real AUTOSAR or vehicle content.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus"

COMP_PREFIX = ["Sensor", "Brake", "Door", "Engine", "Gateway", "Body", "Climate", "Steering", "Battery", "Network",
               "Wiper", "Seat", "Parking", "Cruise", "Lighting", "Diag", "Charging", "Airbag", "Window", "Mirror"]
COMP_SUFFIX = ["Manager", "Controller", "Module", "Handler", "Monitor", "Service", "Driver", "Coordinator"]
SIG_PREFIX = ["Vehicle", "Wheel", "Engine", "Brake", "Door", "Battery", "Cabin", "Steering", "Gear",
              "Wiper", "Seat", "Parking", "Cruise", "Ambient", "Charge", "Pedal", "Window", "Mirror"]
SIG_SUFFIX = ["Speed", "Status", "Health", "Command", "Torque", "Temperature", "Angle", "Voltage",
              "State", "Request", "Position", "Level", "Current"]
PORT_SUFFIX = ["Port", "Channel", "Input", "Output"]
IFACE_SUFFIX = ["Interface", "Data", "Service", "Info"]
ROLES = ["coordinates requests from other software components", "monitors the health of connected sensors",
         "arbitrates access to shared vehicle resources", "converts raw measurements into engineering values",
         "reports diagnostic events to the diagnostic manager", "supervises start-up and shutdown sequences"]
THEMES = ["Braking", "Body Control", "Powertrain", "Gateway", "Comfort", "Charging", "Parking", "Lighting", "Diagnostics", "Chassis"]


def _pick(rng, pre, suf, n, taken):
    out = []
    while len(out) < n:
        name = rng.choice(pre) + rng.choice(suf)
        if name not in taken and name not in out:
            out.append(name)
    taken.update(out)
    return out


def build_doc(doc_id, seed, theme, revision="A"):
    rng = random.Random(seed)
    taken = set()
    comps = _pick(rng, COMP_PREFIX, COMP_SUFFIX, rng.randint(5, 7), taken)
    sigs = _pick(rng, SIG_PREFIX, SIG_SUFFIX, rng.randint(5, 7), taken)
    ports = [rng.choice(SIG_PREFIX) + rng.choice(["Status", "Data", "Feed"]) + rng.choice(PORT_SUFFIX) for _ in range(3)]
    ifaces = [rng.choice(COMP_PREFIX) + rng.choice(["Speed", "Torque", "State", "Mode"]) + rng.choice(IFACE_SUFFIX) for _ in range(3)]
    ports, ifaces = sorted(set(ports) - taken), sorted(set(ifaces) - taken)
    return {"doc_id": doc_id, "theme": theme, "revision": revision,
            "components": comps, "signals": sigs, "ports": ports, "interfaces": ifaces}


def render(spec, rng, inject=True):
    comps, sigs, ports, ifaces = spec["components"], spec["signals"], spec["ports"], spec["interfaces"]
    relations, issues = [], []
    published = {}
    lines = [f"DOCUMENT: {spec['doc_id']}", f"TITLE: Synthetic {spec['theme']} Architecture",
             f"REVISION: {spec['revision']}", "SECTION: Overview",
             f"This synthetic high-level design describes the {spec['theme'].lower()} software architecture.",
             f"It defines {len(comps)} software components, {len(ifaces)} interfaces and {len(sigs)} signals.", "",
             "SECTION: Component Catalogue"]
    for c in comps:
        lines.append(f"- {c}: {rng.choice(ROLES)}.")
    lines += ["", "SECTION: Signals"]
    for s in sigs:
        pub = rng.choice(comps)
        port = rng.choice(ports)
        cons = rng.choice([c for c in comps if c != pub])
        published[s] = pub
        lines.append(f"The {pub} publishes the {s} signal via the {port} port and the {cons} consumes {s}.")
        relations.append({"type": "publishes", "component": pub, "signal": s, "port": port})
        relations.append({"type": "consumes", "component": cons, "signal": s})
    lines += ["", "SECTION: Interfaces"]
    for i in ifaces:
        prov, cons = rng.sample(comps, 2)
        sig = rng.choice(sigs)
        lines.append(f"The {prov} provides the {i} interface to the {cons} component for {sig}.")
        lines.append(f"| {prov} | {i} | {cons} | {sig} |")
        relations.append({"type": "provides", "component": prov, "interface": i, "consumer": cons})
    lines += ["", "SECTION: Dependencies"]
    for _ in range(rng.randint(3, 5)):
        a, b = rng.sample(comps, 2)
        s = rng.choice(sigs)
        lines.append(f"The {a} component depends on {b} for {s}.")
        relations.append({"type": "depends_on", "component": a, "target": b, "signal": s})
    if inject and rng.random() < 0.7:
        ghost = rng.choice(COMP_PREFIX) + "Proxy"
        lines.append(f"The {rng.choice(comps)} component depends on {ghost} for {rng.choice(sigs)}.")
        issues.append({"type": "undefined_component", "item": ghost})
    if inject and rng.random() < 0.7:
        orphan = rng.choice(SIG_PREFIX) + "Diagnostics"
        lines.append(f"The {rng.choice(comps)} consumes {orphan}, which no component is documented to publish.")
        issues.append({"type": "orphan_signal", "item": orphan})
    lines += ["", "SECTION: Test Considerations",
              "Each interface requires one positive and one negative integration test.",
              "", "SECTION: Revision History", f"Revision {spec['revision']}: synthetic release for evaluation."]
    labels = {n: k for names, k in ((comps, "component"), (sigs, "signal"), (ports, "port"), (ifaces, "interface")) for n in names}
    for issue in issues:
        labels.setdefault(issue["item"], "component" if issue["type"] == "undefined_component" else "signal")
    return "\n".join(lines), labels, relations, issues


def revise(spec, rng):
    """Create revision B: remove one component, add one, rename one signal. Returns (new_spec, changes)."""
    new = json.loads(json.dumps(spec))
    new["revision"] = "B"
    new["doc_id"] = spec["doc_id"] + "-B"
    removed = new["components"].pop()
    taken = set(new["components"] + new["signals"])
    added = _pick(rng, COMP_PREFIX, COMP_SUFFIX, 1, taken)[0]
    new["components"].append(added)
    old_sig = new["signals"][0]
    renamed = _pick(rng, SIG_PREFIX, SIG_SUFFIX, 1, taken)[0]
    new["signals"][0] = renamed
    changes = [{"type": "removed", "entity_type": "component", "entity": removed},
               {"type": "added", "entity_type": "component", "entity": added},
               {"type": "removed", "entity_type": "signal", "entity": old_sig},
               {"type": "added", "entity_type": "signal", "entity": renamed}]
    return new, changes


def build(n_docs=10, n_revisions=3, out_dir=OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.txt"):
        old.unlink()
    manifest = {}
    for k in range(n_docs):
        doc_id = f"CORP-{k+1:02d}"
        spec = build_doc(doc_id, 5000 + k, THEMES[k % len(THEMES)])
        rng = random.Random(7000 + k)
        text, labels, relations, issues = render(spec, rng)
        (out_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        manifest[doc_id] = {"labels": labels, "relations": relations, "issues": issues, "revision_of": None, "changes": []}
        if k < n_revisions:
            new_spec, _ = revise(spec, random.Random(9000 + k))
            rtext, rlabels, rrel, rissues = render(new_spec, random.Random(7000 + k), inject=False)
            (out_dir / f"{new_spec['doc_id']}.txt").write_text(rtext, encoding="utf-8")
            changes = []
            for kind in ("component", "signal"):
                old_set = {n for n, t in manifest[doc_id]["labels"].items() if t == kind}
                new_set = {n for n, t in rlabels.items() if t == kind}
                changes += [{"type": "removed", "entity_type": kind, "entity": n} for n in sorted(old_set - new_set)]
                changes += [{"type": "added", "entity_type": kind, "entity": n} for n in sorted(new_set - old_set)]
            manifest[new_spec["doc_id"]] = {"labels": rlabels, "relations": rrel, "issues": rissues,
                                            "revision_of": doc_id, "changes": changes}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build()
    print(f"Built {len(m)} documents in {OUT}")
