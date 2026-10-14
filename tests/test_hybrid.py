import unittest
import numpy as np
from src.retrieval import Retriever, HybridRetriever


class FakeEmbedder:
    """Deterministic stand-in so the hybrid logic can be tested without downloading a model."""
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.zeros(16)
            for w in t.lower().split():
                v[hash(w) % 16] += 1
            n = np.linalg.norm(v) or 1.0
            out.append(v / n)
        return np.array(out)


class HybridTests(unittest.TestCase):
    def setUp(self):
        chunks = [{"document_id": f"D{i}", "source": "s", "page": 1, "text": t} for i, t in enumerate(
            ["SensorManager publishes VehicleSpeed", "BrakeController consumes BrakeStatus", "DoorSensor publishes DoorStatus"])]
        self.base = Retriever(chunks, prefer_embeddings=False)
        self.base.embedder = FakeEmbedder()
        self.base.embeddings = self.base.embedder.encode(self.base.texts)

    def test_requires_embeddings(self):
        plain = Retriever(self.base.chunks, prefer_embeddings=False)
        with self.assertRaises(ValueError):
            HybridRetriever(plain)

    def test_alpha_one_matches_tfidf_ranking(self):
        hybrid = HybridRetriever(self.base, alpha=1.0)
        q = "Which component publishes VehicleSpeed?"
        self.assertEqual(hybrid.search(q, top_k=3)[0]["document_id"], self.base.search(q, top_k=3)[0]["document_id"])

    def test_mode_and_scores(self):
        hybrid = HybridRetriever(self.base, alpha=0.5)
        res = hybrid.search("DoorStatus", top_k=2)
        self.assertEqual(hybrid.mode, "hybrid")
        self.assertEqual(len(res), 2)
        self.assertGreaterEqual(res[0]["score"], res[1]["score"])


if __name__ == "__main__":
    unittest.main()
