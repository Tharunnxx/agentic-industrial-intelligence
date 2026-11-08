"""Integration tests: real ModelService + a tiny SYNTHETIC sklearn pipeline (random data; software tests only)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    import sklearn  # noqa: F401
    HAVE_SK = True
except ImportError:  # pragma: no cover
    HAVE_SK = False

from agentic_app.agents.orchestrator import Orchestrator
from agentic_app.core.costs import COST_MATRIX
from agentic_app.services.model_service import ModelService
from tests.helpers import NUMERIC, SPECS, make_settings, train_synthetic_artifact, valid_payload


@unittest.skipUnless(HAVE_SK, "scikit-learn/joblib not installed")
class SyntheticModelWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        train_synthetic_artifact(cls.tmp / "model.joblib")
        cls.settings = make_settings(cls.tmp)
        cls.ms = ModelService(cls.settings)

    def test_model_loads_and_reports_info(self):
        self.assertTrue(self.ms.ensure_loaded())
        info = self.ms.info()
        self.assertEqual(info["classes"], [0, 1, 2, 3, 4])
        self.assertEqual(info["n_numeric_features"], len(NUMERIC))
        self.assertEqual(info["n_specification_features"], len(SPECS))
        self.assertTrue(info["specification_categories_known"])
        self.assertEqual(info["artifact_format"], "dict{model, feature_cols}")
        self.assertFalse(info["calibration_wrapper_detected"])
        json.dumps(info)

    def test_service_probabilities_equal_direct_pipeline(self):
        art = joblib.load(self.settings.model_path)
        payload = valid_payload()
        row = pd.DataFrame([payload["features"]])[art["feature_cols"]]
        direct = art["model"].predict_proba(row)[0]
        via_service = self.ms.predict_proba({c: payload["features"][c] for c in art["feature_cols"]})
        np.testing.assert_allclose(via_service, direct)

    def test_end_to_end_workflow(self):
        res = Orchestrator(self.settings, self.ms).run_workflow(valid_payload())
        self.assertEqual(res.status, "completed", res.error)
        self.assertIn(res.final_status, {"verified", "needs_review"})
        probs = np.array([res.prediction.probabilities[str(i)] for i in range(5)])
        self.assertAlmostEqual(probs.sum(), 1.0)
        self.assertEqual(res.recommendation.recommended_class, int((probs @ COST_MATRIX).argmin()))
        self.assertEqual(len(res.trace), 5)
        json.dumps(res.to_dict())

    def test_null_numeric_is_handled_by_model_imputer(self):
        p = valid_payload()
        p["features"][NUMERIC[0]] = None
        res = Orchestrator(self.settings, self.ms).run_workflow(p)
        self.assertEqual(res.status, "completed", res.error)
        self.assertEqual(res.investigation.evidence[0].value["n_null"], 1)

    def test_deterministic_with_real_model(self):
        a = Orchestrator(self.settings, self.ms).run_workflow(valid_payload()).to_dict()
        b = Orchestrator(self.settings, self.ms).run_workflow(valid_payload()).to_dict()
        for key in ("prediction", "recommendation", "verification", "final_status"):
            self.assertEqual(a[key], b[key])

    def test_validation_errors_do_not_reach_model(self):
        p = valid_payload()
        p["features"]["bogus"] = 1
        res = Orchestrator(self.settings, self.ms).run_workflow(p)
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.error.code, "INPUT_VALIDATION_FAILED")
        self.assertIsNone(res.prediction)

    def test_batch_frame_matches_single_rows(self):
        art = joblib.load(self.settings.model_path)
        rows = []
        for i in range(3):
            f = valid_payload()["features"]
            f[NUMERIC[0]] = float(i)
            rows.append(f)
        frame = pd.DataFrame(rows)
        batch = self.ms.predict_proba_frame(frame)
        for i, f in enumerate(rows):
            np.testing.assert_allclose(batch[i], self.ms.predict_proba({c: f[c] for c in art["feature_cols"]}))


@unittest.skipUnless(HAVE_SK, "scikit-learn/joblib not installed")
class ArtifactVariantTests(unittest.TestCase):
    def test_bare_estimator_artifact_without_feature_names_is_rejected(self):
        tmp = Path(tempfile.mkdtemp())
        train_synthetic_artifact(tmp / "model.joblib", fmt="bare")
        ms = ModelService(make_settings(tmp))
        # A Pipeline fitted on a DataFrame exposes feature_names_in_, so it is accepted as a bare estimator.
        self.assertTrue(ms.ensure_loaded())
        self.assertEqual(ms.info()["artifact_format"], "bare estimator")

    def test_artifact_using_target_as_feature_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        train_synthetic_artifact(tmp / "model.joblib")
        art = joblib.load(tmp / "model.joblib")
        art["feature_cols"] = art["feature_cols"] + ["class_label"]
        joblib.dump(art, tmp / "model.joblib")
        ms = ModelService(make_settings(tmp))
        self.assertFalse(ms.ensure_loaded())
        self.assertEqual(ms.error.code, "MODEL_USES_TARGET")

    def test_artifact_without_predict_proba_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        joblib.dump({"model": object(), "feature_cols": ["a"]}, tmp / "model.joblib")
        ms = ModelService(make_settings(tmp))
        self.assertFalse(ms.ensure_loaded())
        self.assertEqual(ms.error.code, "MODEL_ARTIFACT_UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()
