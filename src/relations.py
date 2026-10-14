"""Deterministic relation extraction and consistency checks for architecture text.

Relations: publishes / consumes / provides / depends_on.
These rules run on the same text the retriever indexes, so every fact can be traced
back to a document and page. Used for structured answers, consistency checks and
revision comparison. Rule-based by design: transparent and testable.
"""
import re

ID = r"[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+"
GAP = r"(?:\s[^.|\n]{0,90}?)?"

PUB = re.compile(rf"\b({ID}){GAP}\spublishes\s(?:the\s)?({ID})(?:\ssignal)?(?:\s(?:via|through)\sthe\s({ID})\s(?:port|interface))?")
CON = re.compile(rf"\b({ID})(?:\scomponent)?\sconsumes\s(?:the\s)?({ID})\b")
PROV = re.compile(rf"\b({ID})\sprovides\sthe\s({ID})\sinterface(?:\sto\sthe\s({ID})\scomponent)?")
DEP = re.compile(rf"\b({ID})\scomponent\sdepends\son\s({ID})\sfor\s({ID})\b")
CATALOGUE_LINE = re.compile(rf"^-\s({ID}):")


def words(name):
    """EngineStatus -> 'engine status'."""
    return " ".join(w.lower() for w in re.findall(r"[A-Z][a-z0-9]*", name))


def extract_relations(text):
    """Return a list of relation dicts found in the text (duplicates removed)."""
    out, seen = [], set()

    def add(rel, evidence):
        key = tuple(sorted(rel.items()))
        if key not in seen:
            seen.add(key)
            out.append({**rel, "evidence": evidence.strip()})

    for line in text.replace("\r", "").splitlines():
        if line.startswith("|"):
            continue
        for m in PUB.finditer(line):
            add({"type": "publishes", "component": m.group(1), "signal": m.group(2), **({"port": m.group(3)} if m.group(3) else {})}, line)
        for m in CON.finditer(line):
            add({"type": "consumes", "component": m.group(1), "signal": m.group(2)}, line)
        for m in PROV.finditer(line):
            add({"type": "provides", "component": m.group(1), "interface": m.group(2), **({"consumer": m.group(3)} if m.group(3) else {})}, line)
        for m in DEP.finditer(line):
            add({"type": "depends_on", "component": m.group(1), "target": m.group(2), "signal": m.group(3)}, line)
    return out


def extract_catalogue(text):
    """Components declared in a 'Component Catalogue' section (empty set if there is none)."""
    names, active = set(), False
    for line in text.replace("\r", "").splitlines():
        if line.startswith("SECTION:"):
            active = "catalogue" in line.lower()
            continue
        if active:
            m = CATALOGUE_LINE.match(line.strip())
            if m:
                names.add(m.group(1))
    return names


def relation_key(rel):
    """Comparable identity of a relation (ignores evidence text)."""
    t = rel["type"]
    if t == "publishes":
        return (t, rel["component"], rel["signal"], rel.get("port"))
    if t == "consumes":
        return (t, rel["component"], rel["signal"])
    if t == "provides":
        return (t, rel["component"], rel["interface"], rel.get("consumer"))
    return (t, rel["component"], rel["target"], rel["signal"])


def check_consistency(text):
    """Deterministic findings: undefined components, orphan (never published) and unconsumed signals."""
    rels = extract_relations(text)
    catalogue = extract_catalogue(text)
    published = {r["signal"] for r in rels if r["type"] == "publishes"}
    consumed = {r["signal"] for r in rels if r["type"] == "consumes"}
    referenced = set()
    for r in rels:
        referenced.add(r["component"])
        for k in ("target", "consumer"):
            if r.get(k):
                referenced.add(r[k])
    findings = []
    if catalogue:
        for c in sorted(referenced - catalogue):
            findings.append({"type": "undefined_component", "item": c, "severity": "review",
                             "message": f"{c} is referenced in a relation but not listed in the Component Catalogue."})
    for s in sorted(consumed - published):
        findings.append({"type": "orphan_signal", "item": s, "severity": "review",
                         "message": f"{s} is consumed but no component is documented to publish it."})
    for s in sorted(published - consumed):
        findings.append({"type": "unconsumed_signal", "item": s, "severity": "info",
                         "message": f"{s} is published but no consumer is documented."})
    return findings


def compare_relations(old_text, new_text):
    """Relation-level revision diff (added / removed facts)."""
    old = {relation_key(r): r for r in extract_relations(old_text)}
    new = {relation_key(r): r for r in extract_relations(new_text)}
    rows = []
    for k in sorted(set(old) - set(new), key=str):
        rows.append({"change": "removed", "relation": " | ".join(str(x) for x in k if x)})
    for k in sorted(set(new) - set(old), key=str):
        rows.append({"change": "added", "relation": " | ".join(str(x) for x in k if x)})
    return rows
