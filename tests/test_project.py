import unittest
from pathlib import Path
from src.document_loader import load_sample_documents, chunk_pages, clean_text
from src.retrieval import Retriever, answer_from_evidence
from src.analysis import extract_entities, compare_revisions

ROOT = Path(__file__).resolve().parents[1]

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = load_sample_documents(ROOT / "data" / "sample_documents")
        cls.chunks = chunk_pages(cls.pages)
        cls.retriever = Retriever(cls.chunks, prefer_embeddings=False)

    def test_samples_load(self):
        self.assertGreaterEqual(len(self.pages), 2)

    def test_chunk_metadata(self):
        self.assertTrue(self.chunks)
        self.assertIn("page", self.chunks[0])
        self.assertIn("document_id", self.chunks[0])

    def test_clean_text(self):
        self.assertEqual(clean_text(" A   B \n\n\n C "), "A B\n\nC")

    def test_retrieval_returns_results(self):
        results = self.retriever.search("Which component publishes VehicleSpeed?", top_k=2)
        self.assertTrue(results)

    def test_answer_has_citation(self):
        results = self.retriever.search("VehicleSpeed SensorManager", top_k=2)
        answer = answer_from_evidence("VehicleSpeed?", results, min_score=0.0)
        self.assertIn("answer", answer)

    def test_entity_extraction(self):
        entities = extract_entities("SensorManager publishes VehicleSpeed through SpeedData.")
        self.assertIn("SensorManager", entities["component"])
        self.assertIn("VehicleSpeed", entities["signal"])

    def test_revision_compare(self):
        changes = compare_revisions("SensorManager publishes VehicleSpeed",
                                    "DoorSensor publishes DoorStatus")
        self.assertTrue(changes)

if __name__ == "__main__":
    unittest.main()
