import unittest
from src.relations import extract_relations, extract_catalogue, check_consistency, compare_relations, words

DOC = """SECTION: Component Catalogue
- SeatDriver: arbitrates access.
- BrakeService: reports events.

SECTION: Signals
The BrakeService publishes the EngineStatus signal via the WheelStatusInput port and the SeatDriver consumes EngineStatus.

SECTION: Interfaces
The BrakeService provides the GatewaySpeedInfo interface to the SeatDriver component for EngineStatus.
| BrakeService | GatewaySpeedInfo | SeatDriver | EngineStatus |

SECTION: Dependencies
The BrakeService component depends on BatteryProxy for EngineStatus.
The SeatDriver consumes BrakeDiagnostics, which no component is documented to publish.
"""


class RelationTests(unittest.TestCase):
    def test_extracts_all_relation_types(self):
        types = {r["type"] for r in extract_relations(DOC)}
        self.assertEqual(types, {"publishes", "consumes", "provides", "depends_on"})

    def test_publish_has_port(self):
        pub = [r for r in extract_relations(DOC) if r["type"] == "publishes"][0]
        self.assertEqual((pub["component"], pub["signal"], pub["port"]), ("BrakeService", "EngineStatus", "WheelStatusInput"))

    def test_table_rows_are_not_double_counted(self):
        self.assertEqual(len([r for r in extract_relations(DOC) if r["type"] == "provides"]), 1)

    def test_catalogue(self):
        self.assertEqual(extract_catalogue(DOC), {"SeatDriver", "BrakeService"})

    def test_consistency_flags_injected_issues(self):
        found = {(f["type"], f["item"]) for f in check_consistency(DOC)}
        self.assertIn(("undefined_component", "BatteryProxy"), found)
        self.assertIn(("orphan_signal", "BrakeDiagnostics"), found)

    def test_no_undefined_check_without_catalogue(self):
        text = "The SensorManager publishes the VehicleSpeed signal.\nThe BrakeController consumes VehicleSpeed."
        self.assertEqual([f for f in check_consistency(text) if f["type"] == "undefined_component"], [])

    def test_compare_relations(self):
        old = "The BrakeService publishes the EngineStatus signal."
        new = "The BrakeService publishes the WheelState signal."
        changes = {(c["change"]) for c in compare_relations(old, new)}
        self.assertEqual(changes, {"added", "removed"})

    def test_words(self):
        self.assertEqual(words("EngineStatus"), "engine status")


if __name__ == "__main__":
    unittest.main()
