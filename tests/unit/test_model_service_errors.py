import tempfile
import unittest
from pathlib import Path

from agentic_app.agents.orchestrator import Orchestrator
from agentic_app.core.config import Settings
from agentic_app.services.model_service import ModelService, ModelUnavailable
from tests.helpers import make_settings, valid_payload


class MissingArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.settings = make_settings(self.tmp)  # model.joblib does not exist

    def test_missing_model_reports_unavailable(self):
        ms = ModelService(self.settings)
        self.assertFalse(ms.ensure_loaded())
        self.assertEqual(ms.error.code, "MODEL_FILE_NOT_FOUND")
        with self.assertRaises(ModelUnavailable):
            ms.info()
        with self.assertRaises(ModelUnavailable):
            ms.predict_proba({})

    def test_workflow_fails_without_fabricating_prediction(self):
        res = Orchestrator(self.settings, ModelService(self.settings)).run_workflow(valid_payload())
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.error.code, "MODEL_FILE_NOT_FOUND")
        self.assertIsNone(res.prediction)
        self.assertIsNone(res.recommendation)
        self.assertEqual(res.trace[0].status, "error")

    def test_corrupt_artifact_is_reported(self):
        (self.tmp / "model.joblib").write_bytes(b"not a joblib file")
        ms = ModelService(self.settings)
        self.assertFalse(ms.ensure_loaded())
        self.assertEqual(ms.error.code, "MODEL_LOAD_FAILED")


class ConfigTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        s = Settings.from_env(environ={"AGENTIC_DATA_DIR": tempfile.mkdtemp()})
        self.assertEqual(s.model_path.name, "cost_sensitive_model.joblib")
        self.assertEqual(s.max_replans, 1)

    def test_invalid_values_rejected(self):
        for env in ({"AGENTIC_MAX_REPLANS": "5"}, {"AGENTIC_MAX_REPLANS": "x"}, {"AGENTIC_HIGH_P_ANY": "2"},
                    {"AGENTIC_MEDIUM_P_ANY": "0.9", "AGENTIC_HIGH_P_ANY": "0.5"}):
            with self.assertRaises(ValueError):
                Settings.from_env(environ=env)


if __name__ == "__main__":
    unittest.main()
