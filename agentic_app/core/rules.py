"""Documented, configurable prioritisation rule (shared by Decision and Verification agents).

Inputs: class probabilities p (0..4).
    p_any  = 1 - p0            (probability of any failure class 1-4)
    p_high = p3 + p4           (probability of the two closest-to-repair classes)
Rules are evaluated top-down; the first match wins:
    critical : p_high >= critical_p_high
    high     : p_any  >= high_p_any
    medium   : p_any  >= medium_p_any
    low      : otherwise
Priority deliberately does NOT use the minimum-expected-cost class: with the official asymmetric matrix a
missed failure costs 20-50x a false alarm, so that class becomes 4 once P(any failure) is only ~2%, which would
mark almost every vehicle "high". The cost-minimising class is reported separately as the decision label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import Settings
from .costs import check_probabilities, min_cost_class

PRIORITY_LEVELS = ("low", "medium", "high", "critical")


@dataclass(frozen=True)
class PriorityDecision:
    priority: str
    rule_id: str
    rationale: str


def compute_priority(probabilities: Sequence[float], settings: Settings, calibrated: bool = False) -> PriorityDecision:
    p = check_probabilities(probabilities)
    p_any = float(1.0 - p[0])
    p_high = float(p[3] + p[4])
    if calibrated:
        # Calibrated mode: the probabilities are realistic, so the bands are small numbers.
        #   low      : the expected-cost rule itself says "class 0" (no action is cheaper)
        #   medium   : the rule recommends action, calibrated P(any failure) below the high band
        #   high     : calibrated P(any failure) >= cal_high_p_any      (default 5%, ~2x the 2.7% base rate)
        #   critical : calibrated P(any failure) >= cal_critical_p_any  (default 10%, ~4x the base rate)
        # These bands are a documented policy choice for ordering inspections. They were NOT tuned on test labels.
        best = min_cost_class(p)
        if best == 0:
            return PriorityDecision("low", "C-LOW", f"Expected-cost minimum is class 0 (no action); calibrated P(any failure)={p_any:.3f}")
        if p_any >= settings.cal_critical_p_any:
            return PriorityDecision("critical", "C-CRIT", f"Calibrated P(any failure)={p_any:.3f} >= {settings.cal_critical_p_any}")
        if p_any >= settings.cal_high_p_any:
            return PriorityDecision("high", "C-HIGH", f"Calibrated P(any failure)={p_any:.3f} >= {settings.cal_high_p_any}")
        return PriorityDecision("medium", "C-MED", f"Expected-cost rule recommends action (class {best}); calibrated P(any failure)={p_any:.3f} < {settings.cal_high_p_any}")
    if p_high >= settings.critical_p_high:
        return PriorityDecision("critical", "R-CRIT", f"P(class 3 or 4)={p_high:.3f} >= {settings.critical_p_high}")
    if p_any >= settings.high_p_any:
        return PriorityDecision("high", "R-HIGH-P", f"P(any failure)={p_any:.3f} >= {settings.high_p_any}")
    if p_any >= settings.medium_p_any:
        return PriorityDecision("medium", "R-MED-P", f"P(any failure)={p_any:.3f} >= {settings.medium_p_any}")
    return PriorityDecision("low", "R-LOW", f"P(any failure)={p_any:.3f} below {settings.medium_p_any}")
