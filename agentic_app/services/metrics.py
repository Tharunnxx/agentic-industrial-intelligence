"""Evaluation metrics and decision policies (used by scripts/evaluate_saved_model.py)."""
from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from ..core.costs import CLASSES, N_CLASSES, min_cost_class, realised_cost

POLICIES = ("always_0", "always_4", "raw_argmax", "expected_cost")


def policy_predictions(policy: str, probs: np.ndarray) -> np.ndarray:
    n = probs.shape[0]
    if policy == "always_0":
        return np.zeros(n, dtype=int)
    if policy == "always_4":
        return np.full(n, 4, dtype=int)
    if policy == "raw_argmax":
        return probs.argmax(axis=1).astype(int)
    if policy == "expected_cost":
        return np.asarray(min_cost_class(probs), dtype=int)
    raise ValueError(f"Unknown policy {policy!r}")


def evaluate_policy(name: str, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    costs = realised_cost(y_true, y_pred)
    labels = list(CLASSES)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    auc = None  # only computable when both "no failure" and "any failure" vehicles are present
    if len(np.unique(y_true > 0)) == 2:
        auc = float(roc_auc_score((y_true > 0).astype(int), 1.0 - probs[:, 0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns when a degenerate policy predicts one class
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "policy": name,
        "n_vehicles": int(len(y_true)),
        "total_cost": float(costs.sum()),
        "mean_cost": float(costs.mean()),
        "balanced_accuracy": balanced,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "any_failure_auc": auc,
        "confusion_matrix_rows_actual_cols_predicted": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "prediction_counts_0_to_4": np.bincount(y_pred, minlength=N_CLASSES).tolist(),
        "per_class": {str(c): {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(labels)},
    }
