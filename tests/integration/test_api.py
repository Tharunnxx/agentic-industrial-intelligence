"""API tests (need fastapi + httpx; skipped otherwise)."""
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    HAVE_API = True
except Exception:  # pragma: no cover
    HAVE_API = False

from tests.helpers import make_settings, train_synthetic_artifact, valid_payload

try:
    import sklearn  # noqa: F401
    HAVE_SK = True
except ImportError:  # pragma: no cover
    HAVE_SK = False


@unittest.skipUnless(HAVE_API, "fastapi/httpx not installed")
class ApiWithoutModelTests(unittest.TestCase):
    def setUp(self):
        from agentic_app.api.routes import create_app
        self.client = TestClient(create_app(make_settings(Path(tempfile.mkdtemp()))))

    def test_health_works_without_model(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["model_loaded"])
        self.assertEqual(r.json()["model_error"]["code"], "MODEL_FILE_NOT_FOUND")

    def test_model_info_503(self):
        r = self.client.get("/model/info")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "MODEL_FILE_NOT_FOUND")

    def test_predict_and_workflow_503_without_fabrication(self):
        for path in ("/predict", "/workflow/run"):
            r = self.client.post(path, json=valid_payload())
            self.assertEqual(r.status_code, 503)
            self.assertIsNone(r.json()["prediction"])
            self.assertEqual(r.json()["status"], "failed")

    def test_samples_empty_hint(self):
        r = self.client.get("/samples")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["samples"], [])
        self.assertIn("extract_samples", r.json()["hint"])

    def test_sample_path_traversal_rejected(self):
        self.assertEqual(self.client.get("/samples/..%2Fsecret").status_code, 404)


@unittest.skipUnless(HAVE_API and HAVE_SK, "fastapi/httpx/scikit-learn not installed")
class ApiWithSyntheticModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agentic_app.api.routes import create_app
        tmp = Path(tempfile.mkdtemp())
        train_synthetic_artifact(tmp / "model.joblib")
        cls.client = TestClient(create_app(make_settings(tmp)))

    def test_health_and_info(self):
        self.assertTrue(self.client.get("/health").json()["model_loaded"])
        self.assertEqual(self.client.get("/model/info").status_code, 200)

    def test_predict(self):
        r = self.client.post("/predict", json=valid_payload())
        self.assertEqual(r.status_code, 200)
        self.assertIsNotNone(r.json()["prediction"])
        self.assertIsNone(r.json()["recommendation"])

    def test_workflow(self):
        r = self.client.post("/workflow/run", json=valid_payload())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ("prediction", "investigation", "recommendation", "verification", "trace", "final_status"):
            self.assertIn(key, body)
        self.assertEqual(len(body["trace"]), 5)

    def test_upload_csv_pair_and_load_compatible_vehicle(self):
        import io
        # Upload a small model-compatible operational file and matching specification file.
        ops = "vehicle_id,time_step,171_0,666_0,309_0,272_1\n101,1,0.1,0.2,0.3,0.4\n101,2,0.2,0.3,0.4,0.5\n"
        specs = "vehicle_id,Spec_0,Spec_1\n101,Cat0,Cat2\n"
        r = self.client.post("/uploads", files=[
            ("files", ("validation_operational_readouts.csv", io.BytesIO(ops.encode()), "text/csv")),
            ("files", ("validation_specifications.csv", io.BytesIO(specs.encode()), "text/csv")),
        ])
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["compatible"])
        dataset_id = body["dataset_id"]
        vehicles = self.client.get(f"/uploads/{dataset_id}/vehicles?split=validation&page_size=10")
        self.assertEqual(vehicles.status_code, 200, vehicles.text)
        self.assertEqual(vehicles.json()["total_vehicles"], 1)
        vehicle = self.client.get(f"/uploads/{dataset_id}/vehicles/101?split=validation")
        self.assertEqual(vehicle.status_code, 200, vehicle.text)
        payload = vehicle.json()["payload"]
        self.assertEqual(payload["vehicle_id"], "101")
        self.assertEqual(payload["time_step"], 2.0)
        self.assertEqual(set(payload["features"]), set(["171_0", "666_0", "309_0", "272_1", "Spec_0", "Spec_1"]))

    def test_upload_train_only_pair_defaults_to_train_for_vehicle_browsing(self):
        import io
        r = self.client.post("/uploads", files=[
            ("files", ("train_operational_readouts.csv", io.BytesIO(
                b"vehicle_id,time_step,171_0,666_0,309_0,272_1\n303,1,0.1,0.2,0.3,0.4\n303,2,0.2,0.3,0.4,0.5\n"), "text/csv")),
            ("files", ("train_specifications.csv", io.BytesIO(
                b"vehicle_id,Spec_0,Spec_1\n303,Cat0,Cat2\n"), "text/csv")),
        ])
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["compatible"])
        self.assertEqual(body["recommended_split"], "train")
        vehicles = self.client.get(f"/uploads/{body['dataset_id']}/vehicles?split=validation&page_size=10")
        self.assertEqual(vehicles.status_code, 200, vehicles.text)
        self.assertEqual(vehicles.json()["split"], "train")
        self.assertEqual(vehicles.json()["total_vehicles"], 1)
        vehicle = self.client.get(f"/uploads/{body['dataset_id']}/vehicles/303?split=validation")
        self.assertEqual(vehicle.status_code, 200, vehicle.text)
        self.assertEqual(vehicle.json()["split"], "train")
        self.assertEqual(len(vehicle.json()["payload"]["features"]), 6)

    def test_upload_complete_zip_dataset(self):
        import io
        import zipfile
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("validation_operational_readouts.csv", "vehicle_id,time_step,171_0,666_0,309_0,272_1\n202,4,1,2,3,4\n")
            archive.writestr("validation_specifications.csv", "vehicle_id,Spec_0,Spec_1\n202,Cat0,Cat2\n")
            archive.writestr("validation_labels.csv", "vehicle_id,class_label\n202,1\n")
        r = self.client.post("/uploads", files={"files": ("scania_validation.zip", buffer.getvalue(), "application/zip")})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["compatible"])
        self.assertEqual(body["total_files"], 3)
        vehicle = self.client.get(f"/uploads/{body['dataset_id']}/vehicles/202?split=validation")
        self.assertEqual(vehicle.status_code, 200, vehicle.text)
        self.assertNotIn("class_label", vehicle.json()["payload"]["features"])

    def test_validation_error_422_with_details(self):
        bad = valid_payload()
        bad["features"]["171_0"] = "abc"
        r = self.client.post("/workflow/run", json=bad)
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["error"]["code"], "INPUT_VALIDATION_FAILED")
        self.assertTrue(r.json()["error"]["details"])


if __name__ == "__main__":
    unittest.main()
