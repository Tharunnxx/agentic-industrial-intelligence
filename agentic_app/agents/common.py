"""Small helpers shared by the agents."""
from __future__ import annotations

from typing import List

from ..schemas.types import PredictionResult


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * float(x):.{digits}f}%"


def decision_probs(prediction: PredictionResult) -> List[float]:
    """The probabilities the decision is based on: calibrated if calibration was applied, otherwise the raw ones."""
    src = prediction.decision_probabilities or prediction.probabilities
    return [float(src[str(c)]) for c in prediction.classes]


def raw_probs(prediction: PredictionResult) -> List[float]:
    return [float(prediction.probabilities[str(c)]) for c in prediction.classes]


def above(percentile: float) -> str:
    """'above 73% of' / 'above all' wording for a percentile rank."""
    return "above all" if percentile >= 99.95 else f"above {percentile:.0f}% of"
