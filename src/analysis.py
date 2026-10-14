"""Lightweight entity extraction and deterministic architecture checks."""
import re

ENTITY_PATTERNS = {
    "component": r"\b(?:SensorManager|SpeedSensor|BrakeController|BrakeModule|VehicleGateway|BodyControl|DoorSensor)\b",
    "interface": r"\b[A-Z][A-Za-z0-9]*(?:Port|Interface|Data)\b",
    "signal": r"\b(?:VehicleSpeed|BrakeStatus|GatewayHealth|DoorStatus|CabinLightCommand)\b"
}

def extract_entities(text):
    found = {}
    for kind, pattern in ENTITY_PATTERNS.items():
        found[kind] = sorted(set(re.findall(pattern, text)))
    return found

def validate_entity_references(text, known_components=None):
    """Flags component-like tokens not in the supplied allowlist; advisory only."""
    known_components = set(known_components or [
        "SensorManager", "SpeedSensor", "BrakeController", "BrakeModule",
        "VehicleGateway", "BodyControl", "DoorSensor"
    ])
    candidates = set(re.findall(r"\b[A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z0-9]*)+\b", text))
    ignored = {"VehicleSpeed", "BrakeStatus", "GatewayHealth", "DoorStatus",
               "CabinLightCommand", "SpeedData", "BrakeStatusPort",
               "GatewayStatusPort", "DoorStatusPort", "GatewayInput"}
    findings = []
    for item in sorted(candidates - known_components - ignored):
        if item not in {"DOCUMENT", "TITLE", "REVISION", "SECTION", "AUTOSAR"}:
            findings.append({
                "type": "unrecognized_component_like_term",
                "item": item,
                "severity": "review",
                "message": f"'{item}' looks like a technical identifier but is not in the demo component allowlist."
            })
    return findings

def compare_revisions(old_text, new_text):
    old = extract_entities(old_text)
    new = extract_entities(new_text)
    changes = []
    for kind in ("component", "interface", "signal"):
        for item in sorted(set(old[kind]) - set(new[kind])):
            changes.append({"type": "removed", "entity_type": kind, "entity": item})
        for item in sorted(set(new[kind]) - set(old[kind])):
            changes.append({"type": "added", "entity_type": kind, "entity": item})
    return changes
