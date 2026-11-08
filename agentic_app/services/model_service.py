"""Loads the EXISTING joblib artifact and exposes a safe predict_proba. Never fabricates predictions."""
from __future__ import annotations

import datetime as _dt
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from ..core.config import Settings
from ..core.costs import CLASSES, N_CLASSES, check_probabilities
from ..schemas.types import FeatureSchema

SPEC_PREFIX = "Spec_"  # same convention as eval_calibration_experiment.py


class ModelUnavailable(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class PredictionError(Exception):
    code = "PREDICTION_FAILED"


def _py(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _walk(obj: Any, depth: int = 0) -> Iterator[Any]:
    """Yield obj and nested estimators (pipelines, calibrators, frozen wrappers), best effort."""
    if obj is None or depth > 5:
        return
    yield obj
    for _, step in (getattr(obj, "steps", None) or []):
        yield from _walk(step, depth + 1)
    for attr in ("estimator", "estimator_", "base_estimator"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            yield from _walk(child, depth + 1)
    for cc in (getattr(obj, "calibrated_classifiers_", None) or [])[:1]:
        yield from _walk(getattr(cc, "estimator", None), depth + 1)


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._loaded = False
        self._model: Any = None
        self._schema: Optional[FeatureSchema] = None
        self._classes: List[int] = list(CLASSES)
        self._error: Optional[ModelUnavailable] = None
        self._meta: Dict[str, Any] = {}
        self._known: Dict[str, Any] = {}

    # ------------------------------------------------------------------ loading
    def ensure_loaded(self) -> bool:
        with self._lock:
            if not self._loaded:
                try:
                    self._load()
                    self._error = None
                except ModelUnavailable as exc:
                    self._error = exc
                    self._model = None
                except Exception as exc:  # defensive: never crash the API on load
                    self._error = ModelUnavailable("MODEL_LOAD_FAILED", f"{type(exc).__name__}: {exc}")
                    self._model = None
                self._loaded = True
        return self._model is not None

    def reload(self) -> bool:
        with self._lock:
            self._loaded = False
        return self.ensure_loaded()

    @property
    def error(self) -> Optional[ModelUnavailable]:
        self.ensure_loaded()
        return self._error

    def _require(self) -> None:
        if not self.ensure_loaded():
            raise self._error or ModelUnavailable("MODEL_UNAVAILABLE", "Model is not loaded.")

    def _load(self) -> None:
        path = self.settings.model_path
        if not path.is_file():
            raise ModelUnavailable(
                "MODEL_FILE_NOT_FOUND",
                f"Model artifact not found at {path}. Set AGENTIC_MODEL_PATH (see .env.example). No predictions are made without a model.",
            )
        import joblib  # local import keeps unit tests light

        try:
            artifact = joblib.load(path)  # NOTE: only load artifacts you created yourself (pickle).
        except Exception as exc:
            raise ModelUnavailable(
                "MODEL_LOAD_FAILED",
                f"Could not load {path.name}: {type(exc).__name__}: {exc}. Joblib files need the same scikit-learn/numpy/pandas "
                "versions that created them; run the platform from the Python environment you trained with.",
            ) from exc

        if isinstance(artifact, dict):
            model, cols = artifact.get("model"), artifact.get("feature_cols")
            fmt = "dict{model, feature_cols}"
            keys = sorted(map(str, artifact.keys()))
        else:
            model, cols = artifact, getattr(artifact, "feature_names_in_", None)
            fmt, keys = "bare estimator", []
        if model is None or not hasattr(model, "predict_proba"):
            raise ModelUnavailable("MODEL_ARTIFACT_UNSUPPORTED", "Artifact has no estimator with predict_proba.")
        if cols is None or len(cols) == 0:
            raise ModelUnavailable("MODEL_ARTIFACT_UNSUPPORTED", "Artifact does not declare its feature columns ('feature_cols').")
        cols = [str(c) for c in cols]

        classes = None
        for candidate in (model, *[s for _, s in (getattr(model, "steps", None) or [])][-1:]):
            if hasattr(candidate, "classes_"):
                classes = [int(c) for c in candidate.classes_]
                break
        if classes != list(CLASSES):
            raise ModelUnavailable("MODEL_CLASSES_MISMATCH", f"Model classes are {classes}; expected {list(CLASSES)}.")

        warnings: List[str] = []
        if "class_label" in cols:
            raise ModelUnavailable("MODEL_USES_TARGET", "Artifact lists 'class_label' as an input feature (target leakage); refusing to serve it.")
        for risky in ("vehicle_id", "time_step"):
            if risky in cols:
                warnings.append(f"Artifact uses '{risky}' as a model feature; the project policy is to avoid this unless justified and documented.")

        spec_cols = [c for c in cols if c.startswith(SPEC_PREFIX)]
        numeric_cols = [c for c in cols if c not in spec_cols]
        categories = self._spec_categories(model, set(spec_cols))
        if spec_cols and not categories:
            warnings.append("Could not read known specification categories from the fitted encoder; unseen-category checks are disabled.")

        self._model = model
        self._classes = classes
        self._schema = FeatureSchema(cols, numeric_cols, spec_cols, categories)
        stat = path.stat()
        type_names = [type(o).__name__ for o in _walk(model)]
        self._meta = {
            "model_file": path.name,
            # Expose the artifact name, not a machine-specific absolute path.
            "size_bytes": stat.st_size,
            "modified_utc": _dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.timezone.utc).isoformat(),
            "artifact_format": fmt,
            "artifact_keys": keys,
            "estimator_types": type_names,
            "calibration_wrapper_detected": "CalibratedClassifierCV" in type_names,
            "compatibility_warnings": warnings,
        }
        self._known = self._load_known_results(path.name)

    @staticmethod
    def _spec_categories(model: Any, spec_cols: set) -> Dict[str, List[Any]]:
        cats: Dict[str, List[Any]] = {}
        try:
            for obj in _walk(model):
                for _, transformer, columns in (getattr(obj, "transformers_", None) or []):
                    encoder = transformer.steps[-1][1] if hasattr(transformer, "steps") else transformer
                    categories = getattr(encoder, "categories_", None)
                    if categories is None or isinstance(columns, str) or len(columns) != len(categories):
                        continue
                    for col, cat in zip(columns, categories):
                        if col in spec_cols:
                            cats[col] = [_py(v) for v in cat]
        except Exception:
            return {}
        return cats

    def _load_known_results(self, filename: str) -> Dict[str, Any]:
        path = self.settings.known_results_path
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("entries", {}).get(filename, {})
        except Exception:
            return {}

    # ------------------------------------------------------------------ public API
    def schema(self) -> FeatureSchema:
        self._require()
        assert self._schema is not None
        return self._schema

    @property
    def model_file(self) -> str:
        return self.settings.model_path.name

    def reliability_notes(self) -> List[str]:
        self._require()
        return list(self._known.get("reliability_notes", []))

    def is_calibrated_wrapper(self) -> bool:
        self._require()
        return bool(self._meta.get("calibration_wrapper_detected"))

    def predict_proba_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Batch probabilities (n, 5) ordered by class 0..4. Used by the evaluation script."""
        self._require()
        schema = self.schema()
        missing = [c for c in schema.feature_cols if c not in frame.columns]
        if missing:
            raise PredictionError(f"Frame is missing model features: {missing[:8]}")
        raw = np.asarray(self._model.predict_proba(frame[schema.feature_cols]), dtype=float)
        out = np.zeros((raw.shape[0], N_CLASSES))
        for position, cls in enumerate(self._classes):
            out[:, cls] = raw[:, position]
        check_probabilities(out)
        return out

    def predict_proba(self, features: Dict[str, Any]) -> np.ndarray:
        """Probabilities (5,) for one validated readout (dict ordered like feature_cols)."""
        schema = self.schema()
        frame = pd.DataFrame([[features[c] for c in schema.feature_cols]], columns=schema.feature_cols)
        for col in schema.numeric_cols:
            frame[col] = frame[col].astype(float)
        try:
            return self.predict_proba_frame(frame)[0]
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"{type(exc).__name__}: {exc}") from exc

    def info(self) -> Dict[str, Any]:
        self._require()
        schema = self.schema()
        return {
            **self._meta,
            "classes": self._classes,
            "n_features": len(schema.feature_cols),
            "n_numeric_features": len(schema.numeric_cols),
            "n_specification_features": len(schema.spec_cols),
            "specification_categories_known": bool(schema.spec_categories),
            "feature_columns": schema.feature_cols,
            "known_results": self._known,
            "calibration_note": (
                "A calibration wrapper was detected in the artifact." if self._meta["calibration_wrapper_detected"]
                else "No calibration wrapper detected; probabilities are the estimator's raw outputs and are not verified as calibrated."
            ),
        }
