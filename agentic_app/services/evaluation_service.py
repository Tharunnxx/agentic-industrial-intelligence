"""Read-only, label-isolated evaluation of the saved model on labelled splits."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ..core.costs import CLASSES, N_CLASSES, min_cost_class, realised_cost
from .model_service import ModelService, ModelUnavailable, PredictionError


class EvaluationError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 422):
        self.code, self.status_code = code, status_code
        super().__init__(message)


class EvaluationService:
    """Evaluate only validation/test labelled splits; labels never enter model inputs."""
    def __init__(self, data_dir: Path, model_service: ModelService):
        self.data_dir = Path(data_dir)
        self.model_service = model_service

    def evaluate(self, split: str = "validation") -> dict[str, Any]:
        if split not in ("validation", "test"):
            raise EvaluationError("UNSUPPORTED_EVALUATION_SPLIT", "Evaluation is available for validation and test splits only.")
        paths = {kind: self.data_dir / f"{split}_{kind}.csv"
                 for kind in ("operational_readouts", "specifications", "labels")}
        missing_files = [str(path) for path in paths.values() if not path.is_file()]
        if missing_files:
            raise EvaluationError("EVALUATION_DATA_MISSING", "Required evaluation files are missing: " + ", ".join(missing_files), 404)

        if not self.model_service.ensure_loaded():
            error = self.model_service.error
            raise EvaluationError(error.code if error else "MODEL_UNAVAILABLE", str(error or "Model unavailable."), 503)

        schema = self.model_service.schema()
        try:
            ops_header = pd.read_csv(paths["operational_readouts"], nrows=0).columns.tolist()
            spec_header = pd.read_csv(paths["specifications"], nrows=0).columns.tolist()
            ops_features = [c for c in schema.feature_cols if c in ops_header]
            spec_features = [c for c in schema.feature_cols if c in spec_header and c not in ops_header]
            unresolved = [c for c in schema.feature_cols if c not in ops_features and c not in spec_features]
            if unresolved:
                raise EvaluationError("EVALUATION_FEATURES_MISSING", f"Model features absent from split: {unresolved[:10]}")

            # Keep only model inputs and join keys while reading the large operational CSV.
            ops = pd.read_csv(paths["operational_readouts"],
                              usecols=list(dict.fromkeys(["vehicle_id", "time_step"] + ops_features)),
                              dtype={"vehicle_id": "string"})
            latest = ops.sort_values("time_step", kind="mergesort").drop_duplicates("vehicle_id", keep="last")
            specs = pd.read_csv(paths["specifications"],
                                usecols=list(dict.fromkeys(["vehicle_id"] + spec_features)),
                                dtype={"vehicle_id": "string"})
            labels = pd.read_csv(paths["labels"], usecols=["vehicle_id", "class_label"],
                                 dtype={"vehicle_id": "string"})
            if labels["vehicle_id"].duplicated().any() or specs["vehicle_id"].duplicated().any():
                raise EvaluationError("EVALUATION_DUPLICATE_KEYS", "Specifications and labels must have one row per vehicle.")
            # Ground truth is joined only after all model probabilities are produced.
            features = latest.merge(specs, on="vehicle_id", how="left", validate="one_to_one")
            absent_specs = features[spec_features].isna().all(axis=1) if spec_features else pd.Series(False, index=features.index)
            if absent_specs.any():
                raise EvaluationError("EVALUATION_SPEC_JOIN_FAILED", f"Specifications missing for {int(absent_specs.sum())} vehicles.")
            probs = self.model_service.predict_proba_frame(features)
            scored = pd.DataFrame({"vehicle_id": features["vehicle_id"].astype(str)})
            scored["probabilities"] = list(probs)
            scored = scored.merge(labels, on="vehicle_id", how="inner", validate="one_to_one")
            if scored.empty:
                raise EvaluationError("EVALUATION_NO_LABEL_MATCHES", "No operational vehicles matched labelled records.")
            if scored["class_label"].isna().any():
                raise EvaluationError("EVALUATION_LABELS_MISSING", "Some matched vehicles have missing labels.")
            y = scored["class_label"].astype(int).to_numpy()
            if not np.isin(y, CLASSES).all():
                raise EvaluationError("EVALUATION_INVALID_LABELS", "Labels must be integers from class 0 through 4.")
            # Keep predictions aligned with the same vehicle order as the scored join.
            probability_by_id = {str(v): pr for v, pr in zip(features["vehicle_id"], probs)}
            probs = np.vstack([probability_by_id[v] for v in scored["vehicle_id"].astype(str)])
        except EvaluationError:
            raise
        except (ModelUnavailable, PredictionError) as exc:
            raise EvaluationError(getattr(exc, "code", "EVALUATION_PREDICTION_FAILED"), str(exc), 503) from exc
        except Exception as exc:
            raise EvaluationError("EVALUATION_FAILED", f"{type(exc).__name__}: {exc}", 500) from exc

        pred_raw = probs.argmax(axis=1).astype(int)
        pred_cost = np.asarray(min_cost_class(probs), dtype=int)
        policies = {
            "always_0": self._policy(y, np.zeros(len(y), dtype=int)),
            "always_4": self._policy(y, np.full(len(y), 4, dtype=int)),
            "raw_argmax": self._policy(y, pred_raw),
            "expected_cost": self._policy(y, pred_cost),
        }
        raw = policies["raw_argmax"]
        ec = policies["expected_cost"]
        actual_counts = {str(c): int((y == c).sum()) for c in CLASSES}
        pred_counts = {str(c): int((pred_raw == c).sum()) for c in CLASSES}
        return {
            "evaluation_label": f"{split.title()} labelled split · measured saved-model evaluation",
            "protocol": "Latest operational readout per vehicle + specification row; labels joined after prediction for scoring only.",
            "warning": "Probabilities are raw model outputs and are not verified as calibrated." if not self.model_service.is_calibrated_wrapper() else "",
            "split": split,
            "model_file": self.model_service.model_file,
            "n_vehicles": int(len(y)),
            "actual_class_counts": actual_counts,
            "raw_argmax_prediction_counts": pred_counts,
            "class_4_prediction_rate": float(pred_counts["4"] / len(y)),
            "raw_argmax": raw,
            "expected_cost": ec,
            "policies": policies,
            "always_class_4_baseline_mean_cost": policies["always_4"]["mean_cost"],
            "always_class_0_baseline_mean_cost": policies["always_0"]["mean_cost"],
            "cost_matrix_orientation": "Rows = actual class; columns = predicted class.",
        }

    @staticmethod
    def _policy(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
        labels = list(CLASSES)
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        costs = realised_cost(y, pred)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "total_cost": float(costs.sum()),
            "mean_cost": float(costs.mean()),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            "prediction_counts": np.bincount(pred, minlength=N_CLASSES).tolist(),
            "per_class": {str(c): {"precision": float(precision[i]), "recall": float(recall[i]),
                                   "f1": float(f1[i]), "support": int(support[i])}
                          for i, c in enumerate(labels)},
        }
