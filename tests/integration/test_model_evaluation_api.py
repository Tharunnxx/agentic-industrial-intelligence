"""Synthetic integration tests for the measured evaluation endpoint."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from fastapi.testclient import TestClient
    HAVE_API = True
except Exception:
    HAVE_API = False

try:
    import sklearn  # noqa
    HAVE_SK = True
except Exception:
    HAVE_SK = False

from tests.helpers import make_settings, train_synthetic_artifact


@unittest.skipUnless(HAVE_API and HAVE_SK, "FastAPI, httpx and scikit-learn are required")
class ModelEvaluationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agentic_app.api.routes import create_app
        cls.tmp = Path(tempfile.mkdtemp())
        train_synthetic_artifact(cls.tmp / "model.joblib")
        # Synthetic-only fixtures. Never used as real project results.
        ops = []
        for vehicle in range(1, 11):
            for step in (1, 2):
                ops.append({"vehicle_id": vehicle, "time_step": step,
                            "171_0": vehicle * .1, "666_0": vehicle * .2,
                            "309_0": vehicle * .3, "272_1": vehicle * .4})
        pd.DataFrame(ops).to_csv(cls.tmp / "validation_operational_readouts.csv", index=False)
        pd.DataFrame([{"vehicle_id": v, "Spec_0": "Cat0" if v % 2 else "Cat1",
                       "Spec_1": "Cat2"} for v in range(1, 11)]).to_csv(
                           cls.tmp / "validation_specifications.csv", index=False)
        pd.DataFrame([{"vehicle_id": v, "class_label": (v - 1) % 5}
                      for v in range(1, 11)]).to_csv(cls.tmp / "validation_labels.csv", index=False)
        cls.client = TestClient(create_app(make_settings(cls.tmp)))

    def test_evaluation_endpoint_returns_frontend_contract(self):
        response = self.client.get("/model/evaluate?split=validation")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        for key in ("n_vehicles", "actual_class_counts", "raw_argmax_prediction_counts",
                    "class_4_prediction_rate", "raw_argmax", "expected_cost", "policies",
                    "always_class_4_baseline_mean_cost", "always_class_0_baseline_mean_cost"):
            self.assertIn(key, body)
        self.assertEqual(body["n_vehicles"], 10)
        self.assertEqual(len(body["raw_argmax"]["confusion_matrix"]), 5)
        self.assertIn("accuracy", body["raw_argmax"])
        self.assertEqual(body["policies"]["always_4"]["prediction_counts"][4], 10)

    def test_train_split_rejected_as_unlabelled(self):
        response = self.client.get("/model/evaluate?split=train")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "UNSUPPORTED_EVALUATION_SPLIT")

    def test_unknown_split_rejected(self):
        response = self.client.get("/model/evaluate?split=other")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
