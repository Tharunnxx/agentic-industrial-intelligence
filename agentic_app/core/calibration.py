"""Calibration layer: turns the model's inflated raw scores into realistic failure probabilities.

WHY THIS EXISTS (plain English)
  The saved Random Forest was trained with class balancing, so its raw "probabilities" say a typical vehicle has a
  ~60% chance of failure, while only ~2.7% of validation vehicles actually had a failure. The model still RANKS vehicles
  sensibly (risk groups with higher scores really fail more often), but its scale is wrong.

WHAT IT DOES
  1. Take the raw score for "any failure":  s = logit(1 - p0)
  2. Platt scaling with two numbers fitted in experiments/m2_calibrated_cost.py:  c = sigmoid(a * s + b)
  3. Split c over classes 1-4 in the same proportions the model gave them (variant "model"):  q_i = c * p_i / (1 - p0)
  4. q0 = 1 - c.
The two numbers (a, b) were fitted on the official validation set only and checked once on the test set.
The original model file is never modified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

EPS = 1e-6
GROUP_NAMES = {1: "lowest 20%", 2: "20-40%", 3: "40-60%", 4: "60-80%", 5: "highest 20%"}


class CalibrationError(Exception):
    pass


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, slope: float, intercept: float, variant: str = "model", val_failure_mix=None,
                 meta: Optional[Dict[str, Any]] = None, reference: Optional[Dict[str, Any]] = None,
                 evidence: Optional[Dict[str, Any]] = None):
        if variant not in ("model", "valmix"):
            raise CalibrationError(f"Unknown severity variant {variant!r}.")
        self.slope, self.intercept, self.variant = float(slope), float(intercept), variant
        self.val_failure_mix = np.asarray(val_failure_mix if val_failure_mix is not None else [0.25] * 4, dtype=float)
        self.meta = meta or {}
        self.reference = reference
        self.evidence = evidence or {}  # measured facts from the held-out test run (used in agent limitation texts)
        self._sorted = np.asarray(reference["raw_p_any_sorted"], dtype=float) if reference else None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Path, reference_path: Optional[Path] = None, results_dir: Optional[Path] = None) -> "Calibrator":
        path = Path(path)
        if not path.is_file():
            raise CalibrationError(f"Calibration file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("slope_a", "intercept_b", "selected_variant"):
            if key not in data:
                raise CalibrationError(f"Calibration file lacks '{key}'.")
        reference = None
        if reference_path and Path(reference_path).is_file():
            reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        return cls(data["slope_a"], data["intercept_b"], data["selected_variant"], data.get("val_failure_mix_1_to_4"),
                   meta=data, reference=reference, evidence=cls._evidence(results_dir))

    @staticmethod
    def _evidence(results_dir: Optional[Path]) -> Dict[str, Any]:
        """Facts copied from the project's own measured test run (config/results/m2_test.json); never invented."""
        if not results_dir:
            return {}
        f = Path(results_dir) / "m2_test.json"
        if not f.is_file():
            return {}
        auc = None
        try:
            auc = json.loads((Path(results_dir) / "m1_validation.json").read_text(encoding="utf-8")).get("any_failure_auc")
        except Exception:
            pass
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            m2, base = d["m2_test"], d["baselines"]
            cm = np.array(m2["confusion_matrix_rows_actual_cols_predicted"])
            fail = cm[1:]
            return {
                "validation_any_failure_auc": auc,
                "test_vehicles": int(cm.sum()), "test_failure_vehicles": int(fail.sum()),
                "test_mean_cost": m2["mean_cost"], "test_always_4_mean_cost": base["always_4"]["mean_cost"],
                "test_original_cost_rule_mean_cost": base["M0_original_cost_rule"]["mean_cost"],
                "test_flagged_share": 1 - m2["prediction_counts_0_to_4"][0] / int(cm.sum()),
                "test_failure_recall_any": float(fail[:, 1:].sum() / fail.sum()),
                "test_classes_1_to_3_correct": int(cm[1, 1] + cm[2, 2] + cm[3, 3]),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------ maths
    def apply(self, probabilities):
        """Raw class probabilities (5,) or (n, 5) -> calibrated probabilities of the same shape."""
        P = np.asarray(probabilities, dtype=float)
        single = P.ndim == 1
        P = np.atleast_2d(P)
        c = 1.0 / (1.0 + np.exp(-(self.slope * _logit(1.0 - P[:, 0]) + self.intercept)))
        if self.variant == "model":
            r = P[:, 1:] / np.clip(1.0 - P[:, 0], EPS, None)[:, None]
        else:
            r = np.tile(self.val_failure_mix, (len(P), 1))
        Q = np.column_stack([1.0 - c, c[:, None] * r])
        Q = Q / Q.sum(axis=1, keepdims=True)
        return Q[0] if single else Q

    def risk_context(self, raw_p_any: float) -> Optional[Dict[str, Any]]:
        """Where a vehicle's RAW risk score sits among the 5,046 official validation vehicles."""
        if self._sorted is None or not self.reference:
            return None
        n = len(self._sorted)
        pct = float(np.searchsorted(self._sorted, raw_p_any, side="right") / n * 100.0)
        edges = self.reference["group_edges_raw_p_any"]
        k = int(np.searchsorted(edges, raw_p_any, side="right")) + 1
        g = self.reference["groups"][k - 1]
        return {
            "reference_n": n, "percentile": round(pct, 1), "group": k, "group_name": GROUP_NAMES[k],
            "group_vehicles": g["n"], "group_failures": g["observed_failures"],
            "group_observed_failure_rate": g["observed_failure_rate"],
            "group_mean_calibrated_p_any": g["mean_calibrated_p_any"],
            "base_failure_rate": self.reference["base_failure_rate"],
            "reference": f"{n:,} official validation vehicles (latest readout each)",
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "method": "Platt scaling of P(any failure), failure mass split over classes 1-4 as the model split it"
                      if self.variant == "model" else "Platt scaling of P(any failure), fixed validation failure mix",
            "formula": "c = 1 / (1 + exp(-(a * logit(1 - p0) + b)));  q0 = 1 - c;  q_i = c * p_i / (1 - p0)",
            "slope_a": self.slope, "intercept_b": self.intercept, "variant": self.variant,
            "fitted_on": self.meta.get("calibration_set", "official validation"),
            "cross_validation": self.meta.get("cv_summary", {}),
            "test_evidence": self.evidence,
            "reference_groups": (self.reference or {}).get("groups", []),
            "base_failure_rate": (self.reference or {}).get("base_failure_rate"),
        }
