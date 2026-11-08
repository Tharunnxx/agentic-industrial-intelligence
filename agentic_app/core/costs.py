"""Official asymmetric cost matrix and expected-cost decision rule.

Orientation (verified by unit tests): rows = ACTUAL class, columns = PREDICTED class.
    ExpectedCost(j) = sum_i p(i) * COST_MATRIX[i][j]
Ties are broken towards the lower class index (numpy argmin), which matches the existing
``expected_cost_predict`` in ``eval_calibration_experiment.py``.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

CLASSES = (0, 1, 2, 3, 4)
N_CLASSES = len(CLASSES)

COST_MATRIX = np.array(
    [
        [0, 7, 8, 9, 10],
        [200, 0, 7, 8, 9],
        [300, 200, 0, 7, 8],
        [400, 300, 200, 0, 7],
        [500, 400, 300, 200, 0],
    ],
    dtype=float,
)
COST_MATRIX.setflags(write=False)

ArrayLike = Union[Sequence[float], np.ndarray]


def check_probabilities(p: ArrayLike, sum_tol: float = 1e-3) -> np.ndarray:
    """Return ``p`` as a float array after verifying it is a valid 5-class distribution."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim not in (1, 2) or arr.shape[-1] != N_CLASSES:
        raise ValueError(f"Expected probabilities with last dimension {N_CLASSES}, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Probabilities contain NaN or infinite values.")
    if (arr < -1e-9).any() or (arr > 1 + 1e-9).any():
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.allclose(arr.sum(axis=-1), 1.0, atol=sum_tol):
        raise ValueError("Probabilities must sum to 1 (within tolerance).")
    return arr


def expected_costs(p: ArrayLike) -> np.ndarray:
    """Expected cost of predicting each class j: p @ COST_MATRIX (shape (5,) or (n, 5))."""
    return check_probabilities(p) @ COST_MATRIX


def min_cost_class(p: ArrayLike):
    """Class (or array of classes) with minimum expected cost."""
    ec = expected_costs(p)
    result = ec.argmin(axis=-1)
    return int(result) if np.ndim(result) == 0 else result


def realised_cost(actual, predicted):
    """Cost of predicting ``predicted`` when the truth is ``actual`` (row = actual)."""
    return COST_MATRIX[np.asarray(actual, dtype=int), np.asarray(predicted, dtype=int)]
