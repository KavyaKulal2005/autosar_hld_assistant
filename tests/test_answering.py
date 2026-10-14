import unittest
from src.answering import answer_question, build_relation_index

PAGES = [{"document_id": "D1", "source": "d1.txt", "page": 1,
          "text": "The BrakeService publishes the EngineStatus signal via the WheelStatusInput port and the SeatDriver consumes EngineStatus.\n"
                  "The SeatDriver component depends on ParkingMonitor for WiperTemperature."}]
IDX = build_relation_index(PAGES)


def res(score=0.5, text=None):
    return [{"document_id": "D1", "source": "d1.txt", "page": 1, "score": score, "text": text or PAGES[0]["text"]}]


class AnsweringTests(unittest.TestCase):
    def test_structured_answer_with_citation(self):
        a = answer_question("Which component publishes EngineStatus?", res(), IDX, 0.3)
        self.assertEqual(a["mode"], "structured")
        self.assertIn("BrakeService publishes EngineStatus", a["answer"])
        self.assertEqual(a["citations"][0]["document_id"], "D1")

    def test_paraphrase_resolves_entity_words(self):
        a = answer_question("What produces the engine status signal?", res(), IDX, 0.3)
        self.assertEqual(a["mode"], "structured")
        self.assertIn("BrakeService", a["answer"])

    def test_dependency_intent(self):
        a = answer_question("What does SeatDriver depend on for WiperTemperature?", res(), IDX, 0.3)
        self.assertIn("depends on ParkingMonitor", a["answer"])

    def test_abstains_on_unknown_identifier(self):
        a = answer_question("Which component publishes FooBarBaz?", res(), IDX, 0.3)
        self.assertTrue(a["abstained"])
        self.assertEqual(a["citations"], [])

    def test_abstains_on_low_score_without_relation(self):
        a = answer_question("What is the bit rate?", res(score=0.05), IDX, 0.3)
        self.assertTrue(a["abstained"])

    def test_abstains_without_results(self):
        self.assertTrue(answer_question("anything", [], IDX, 0.3)["abstained"])


if __name__ == "__main__":
    unittest.main()
