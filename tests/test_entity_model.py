import unittest
from src.entity_model import find_mentions, mention_features, train, predict_entities
from src.synthetic_data import make_document


class EntityModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = [make_document(f"T{i}", "train", 100 + i) for i in range(8)]
        cls.model = train(docs, save=False)

    def test_candidates_are_camelcase_only(self):
        toks = [t for t, _, _ in find_mentions("The SensorManager sends VehicleSpeed. See AUTOSAR.")]
        self.assertEqual(toks, ["SensorManager", "VehicleSpeed"])

    def test_token_identity_not_a_feature(self):
        feats = mention_features("SensorManager", "The SensorManager sends VehicleSpeed.", 4)
        self.assertFalse(any(k.startswith("tok=") for k in feats))

    def test_predicts_unseen_names(self):
        text = "The WiperManager publishes the WiperSpeed signal via the WiperStatusPort port."
        out = predict_entities(text, self.model)
        self.assertIn("WiperManager", out["component"])
        self.assertIn("WiperSpeed", out["signal"])

    def test_no_model_returns_empty(self):
        self.assertEqual(predict_entities("The AbcDef runs.", None)["component"], [])


if __name__ == "__main__":
    unittest.main()
