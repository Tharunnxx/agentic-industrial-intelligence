"""Test helpers. Everything here is SYNTHETIC and used only to test software behaviour, never to report model results."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from agentic_app.core.config import Settings
from agentic_app.schemas.types import FeatureSchema

NUMERIC = ["171_0", "666_0", "309_0", "272_1"]
SPECS = ["Spec_0", "Spec_1"]
SPEC_VALUES = {"Spec_0": ["Cat0", "Cat1"], "Spec_1": ["Cat0", "Cat2"]}


def make_settings(tmp: Path, **overrides) -> Settings:
    env = {"AGENTIC_DATA_DIR": str(tmp), "AGENTIC_MODEL_PATH": str(tmp / "model.joblib"),
           "AGENTIC_SAMPLES_DIR": str(tmp / "samples"), "AGENTIC_KNOWN_RESULTS": str(tmp / "known.json"),
           "AGENTIC_CALIBRATION": "off",  # legacy tests exercise the raw-probability path; calibrated tests switch it on explicitly
           "AGENTIC_INDEX_DIR": str(tmp / "index")}
    env.update({k: str(v) for k, v in overrides.items()})
    return Settings.from_env(environ=env)


def make_schema() -> FeatureSchema:
    return FeatureSchema(NUMERIC + SPECS, list(NUMERIC), list(SPECS), {k: list(v) for k, v in SPEC_VALUES.items()})


def valid_payload(**kw) -> Dict:
    features = {c: float(i) + 0.5 for i, c in enumerate(NUMERIC)}
    features.update({"Spec_0": "Cat0", "Spec_1": "Cat2"})
    payload = {"vehicle_id": 101, "time_step": 12.0, "features": features}
    payload.update(kw)
    return payload


def train_synthetic_artifact(path: Path, fmt: str = "dict") -> None:
    """Fit a tiny sklearn pipeline on RANDOM data (same step names as the project's pipelines) and save it."""
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    for c in SPECS:
        frame[c] = rng.choice(SPEC_VALUES[c], size=n)
    y = np.arange(n) % 5
    pipe = Pipeline([
        ("pre", ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMERIC),
                                   ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SPECS)])),
        ("classifier", RandomForestClassifier(n_estimators=15, random_state=0)),
    ])
    pipe.fit(frame[NUMERIC + SPECS], y)
    joblib.dump({"model": pipe, "feature_cols": NUMERIC + SPECS} if fmt == "dict" else pipe, path)


class StubModelService:
    """Test double returning caller-supplied probabilities (unit tests only)."""

    def __init__(self, probs: List[float], schema: FeatureSchema = None, calibrated: bool = False, notes: List[str] = None):
        self._probs = np.array(probs, dtype=float)
        self._schema = schema or make_schema()
        self.model_file = "stub.joblib"
        self._calibrated, self._notes = calibrated, notes or []

    def schema(self):
        return self._schema

    def predict_proba(self, features):
        return self._probs

    def is_calibrated_wrapper(self):
        return self._calibrated

    def reliability_notes(self):
        return list(self._notes)
