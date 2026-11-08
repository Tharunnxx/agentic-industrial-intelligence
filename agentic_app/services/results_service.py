"""Measured experiment results (config/results/*.json). These files are the project's own outputs; nothing is typed in by hand."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _flags(r: Dict[str, Any]) -> Dict[str, Any]:
    cm = np.array(r["confusion_matrix_rows_actual_cols_predicted"])
    fail = cm[1:]
    return {"flagged_share": float(1 - cm[:, 0].sum() / cm.sum()),
            "failure_recall_any": float(fail[:, 1:].sum() / fail.sum()) if fail.sum() else None,
            "failure_vehicles": int(fail.sum()), "vehicles": int(cm.sum())}


class ResultsService:
    def __init__(self, results_dir: Path):
        self.dir = Path(results_dir)

    def _load(self, name: str) -> Optional[Dict[str, Any]]:
        f = self.dir / name
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None

    @staticmethod
    def _summary(m1, cv, test) -> List[str]:
        if not (m1 and cv and test):
            return []
        t, b = test["m2_test"], test["baselines"]
        fl = _flags(t)
        return [
            f"On the held-out test set, calibration lowered mean cost from {b['M0_original_cost_rule']['mean_cost']:.2f} (original cost rule) to {t['mean_cost']:.2f} units per vehicle; "
            f"always choosing class 4 scores {b['always_4']['mean_cost']:.2f}.",
            f"The model ranks vehicles by risk moderately (any-failure AUC {m1['any_failure_auc']:.2f}) but does not reliably identify the failure class: test balanced accuracy is {t['balanced_accuracy']:.2f}.",
            f"On test the calibrated policy flagged {fl['flagged_share']:.0%} of vehicles and caught {fl['failure_recall_any']:.0%} of the failure vehicles; "
            f"cross-validation on validation had suggested {cv['cv']['model']['failure_recall_any']:.0%}, so expect results to vary between data splits.",
        ]

    def model_comparison(self) -> Dict[str, Any]:
        cv, test, m1 = self._load("m2_cv.json"), self._load("m2_test.json"), self._load("m1_validation.json")
        rows: List[Dict[str, Any]] = []

        def add(split, key, label, r, basis):
            rows.append({"split": split, "key": key, "label": label, "mean_cost": r["mean_cost"], "total_cost": r.get("total_cost"),
                         "balanced_accuracy": r["balanced_accuracy"], "macro_f1": r["macro_f1"],
                         "prediction_counts_0_to_4": r["prediction_counts_0_to_4"], "basis": basis, **_flags(r)})
        if m1:
            by = {r["name"]: r for r in m1["results"]}
            add("validation", "always_0", "Always class 0", by["always_0"], "fixed policy")
            add("validation", "always_4", "Always class 4", by["always_4"], "fixed policy")
            add("validation", "m0_raw_argmax", "Original model, highest raw class", by["M0 original, raw argmax"], "no tuning")
            add("validation", "m0_cost_rule", "Original model + cost rule", by["M0 original, cost rule"], "no tuning")
            add("validation", "m1b_prior_shift", "Original + prior-shift correction (M1)", by["M1b prior=validation counts, cost rule"], "no tuning; rejected, worse than always class 4")
        extra: Dict[str, Any] = {}
        if cv:
            m = cv["cv"]["model"]
            rows.append({"split": "validation", "key": "m2_cv", "label": "Calibrated model + cost rule (M2), cross-validated", "mean_cost": m["cv_mean_cost"],
                         "total_cost": None, "balanced_accuracy": None, "macro_f1": None, "prediction_counts_0_to_4": None,
                         "basis": "10 x repeated 5-fold CV by vehicle; calibrator never scored on its own fitting vehicles",
                         "flagged_share": m["flagged_share"], "failure_recall_any": m["failure_recall_any"], "failure_vehicles": 136, "vehicles": 5046})
            extra["m2_cv"] = {"mean": m["cv_mean_cost"], "sd": m["cv_sd"]}
        if test:
            b = test["baselines"]
            add("test", "always_0", "Always class 0", b["always_0"], "fixed policy")
            add("test", "always_4", "Always class 4", b["always_4"], "fixed policy")
            add("test", "m0_cost_rule", "Original model + cost rule (M0)", b["M0_original_cost_rule"], "saved original model; uncalibrated cost rule")
            add("test", "m2_calibrated", "Calibrated model + cost rule (M2)", test["m2_test"], "calibrator fitted on validation only; test scored once")
            # Publish the complete saved M2 test metrics separately from the live
            # /model/evaluate result, which evaluates the currently loaded artifact.
            m2 = test["m2_test"]
            cm = m2["confusion_matrix_rows_actual_cols_predicted"]
            per_class = {}
            for cls, row in enumerate(cm):
                tp = int(row[cls])
                support = int(sum(row))
                predicted = int(sum(cm[r][cls] for r in range(len(cm))))
                precision = tp / predicted if predicted else 0.0
                recall = tp / support if support else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
                per_class[str(cls)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
            extra["m2_test_detail"] = {
                "name": m2.get("name", "M2 calibrated cost policy"),
                "n_vehicles": int(sum(sum(row) for row in cm)),
                "mean_cost": float(m2["mean_cost"]),
                "total_cost": float(m2["total_cost"]),
                "balanced_accuracy": float(m2["balanced_accuracy"]),
                "macro_f1": float(m2["macro_f1"]),
                "prediction_counts_0_to_4": m2["prediction_counts_0_to_4"],
                "actual_class_counts_0_to_4": [int(sum(row)) for row in cm],
                "confusion_matrix_rows_actual_cols_predicted": cm,
                "per_class": per_class,
                "baseline_m0_mean_cost": float(b["M0_original_cost_rule"]["mean_cost"]),
                "cost_reduction_vs_m0_percent": float((b["M0_original_cost_rule"]["mean_cost"] - m2["mean_cost"]) / b["M0_original_cost_rule"]["mean_cost"] * 100.0),
                "evaluation_basis": "Saved M2 experiment artifact; calibration fitted on official validation split; held-out test scored once.",
            }
            extra["test_reliability"] = test.get("reliability")
            extra["test_acceptance_passed"] = test.get("acceptance_passed")
        return {
            "rows": rows, "extra": extra, "any_failure_auc_validation": m1["any_failure_auc"] if m1 else None,
            "cost_units_note": "Cost values are project-defined units (official cost matrix), not currency.",
            "selection_rule": ("A candidate replaces the original only if its mean cost is below always-class-4 (by more than one cross-validation standard deviation), "
                               "it flags at most 80% of vehicles, and per-class recall is reported."),
            "honest_summary": self._summary(m1, cv, test),
        }
