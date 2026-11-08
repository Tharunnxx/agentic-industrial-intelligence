"""API tests for calibration, vehicle search, analysis and results routes (need fastapi + httpx; skipped otherwise).
The vehicles and the tiny model here are SYNTHETIC and exist only to test the software."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from fastapi.testclient import TestClient
    HAVE_API = True
except Exception:  # pragma: no cover
    HAVE_API = False

from tests.helpers import make_settings, train_synthetic_artifact


def build_synthetic_index(directory: Path, n: int = 30):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    raw = rng.dirichlet([3, 1, 1, 1, 2], size=n)
    ids = [str(2000 + i) for i in range(n)]
    pd.DataFrame({"vehicle_id": ids, "time_step": 5.0, **{f"p{i}": raw[:, i] for i in range(5)}, "class_label": rng.integers(0, 5, n)}).to_csv(directory / "validation_scored.csv", index=False)
    pd.DataFrame({"vehicle_id": ids, "time_step": 5.0, "171_0": 1.0, "666_0": 2.0, "309_0": 3.0, "272_1": 4.0, "Spec_0": "Cat0", "Spec_1": "Cat2"}).to_csv(directory / "validation_vehicles.csv.gz", index=False, compression="gzip")


@unittest.skipUnless(HAVE_API, "fastapi/httpx not installed")
class CalibratedApiTests(unittest.TestCase):
    def setUp(self):
        from agentic_app.api.routes import create_app
        self.tmp = Path(tempfile.mkdtemp())
        train_synthetic_artifact(self.tmp / "model.joblib")
        self.settings = make_settings(self.tmp, AGENTIC_CALIBRATION="on")
        self.client = TestClient(create_app(self.settings))

    def test_calibration_endpoint_reports_the_saved_numbers(self):
        body = self.client.get("/calibration").json()
        self.assertTrue(body["enabled"])
        self.assertAlmostEqual(body["slope_a"], 2.1492722122485537)
        self.assertEqual(body["priority_bands"]["high_at_calibrated_p_any"], 0.05)

    def test_vehicle_routes_explain_missing_index(self):
        r = self.client.get("/vehicles")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "INDEX_NOT_BUILT")
        self.assertIn("build_vehicle_index", r.json()["error"]["message"])

    def test_search_analysis_and_analytics(self):
        build_synthetic_index(self.tmp / "index")
        s = self.client.get("/vehicles", params={"split": "validation", "page_size": 5}).json()
        self.assertEqual(s["total"], 30)
        self.assertNotIn("class_label", s["vehicles"][0])
        vid = s["vehicles"][0]["vehicle_id"]
        run = self.client.post(f"/analysis/validation/{vid}")
        self.assertEqual(run.status_code, 200)
        body = run.json()
        self.assertEqual(body["status"], "completed")
        self.assertIsNotNone(body["prediction"]["report"])
        self.assertIn("recorded_class_label", body["evaluation_only"])
        self.assertEqual(self.client.post("/analysis/validation/does-not-exist").status_code, 404)
        a = self.client.get("/analytics/validation").json()
        self.assertEqual(a["n_vehicles"], 30)
        self.assertEqual(self.client.get("/analytics/other").status_code, 422)

    def test_model_comparison_route(self):
        body = self.client.get("/results/model-comparison").json()
        self.assertTrue(any(r["key"] == "m2_calibrated" and r["split"] == "test" for r in body["rows"]))

    def test_legacy_routes_still_work(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/samples").status_code, 200)


if __name__ == "__main__":
    unittest.main()
