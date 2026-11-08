"""Prediction Agent: model inference (+ optional calibration). Never invents a prediction."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..core.calibration import Calibrator
from ..core.costs import CLASSES, check_probabilities
from ..schemas.types import AgentReport, Finding, PredictionResult, ValidatedReadout
from ..services.model_service import ModelService
from .base import Agent
from .common import above, pct


def normalised_entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / math.log2(len(p)))


class PredictionAgent(Agent):
    name = "PredictionAgent"
    responsibility = "Run the saved model on a validated readout, calibrate its failure risk, and report class probabilities."

    def __init__(self, model_service: ModelService, calibrator: Optional[Calibrator] = None):
        self.model_service = model_service
        self.calibrator = calibrator

    def run(self, readout: ValidatedReadout) -> PredictionResult:
        raw = check_probabilities(self.model_service.predict_proba(readout.features))
        return self.from_probabilities(raw, readout)

    def from_probabilities(self, raw, readout: Optional[ValidatedReadout] = None) -> PredictionResult:
        """Everything after the model call. Also used by the batch script that re-uses already computed raw probabilities."""
        raw = check_probabilities(raw)
        p = [float(x) for x in raw]
        argmax = int(np.argmax(raw))
        p_any = float(1.0 - p[0])
        p_high = float(p[3] + p[4])
        wrapper = self.model_service.is_calibrated_wrapper()
        cal = self.calibrator
        if cal is not None:
            q = check_probabilities(cal.apply(raw))
            dec = {str(c): float(q[c]) for c in CLASSES}
            q_any = float(1.0 - q[0])
            ctx = cal.risk_context(p_any)
            calibration_note = (f"The raw model output is NOT calibrated (it was trained with class balancing). A two-number calibration "
                                f"(slope {cal.slope:.3f}, intercept {cal.intercept:.3f}), fitted on the official validation set, was applied; "
                                f"decisions use the calibrated probabilities.")
        else:
            dec, q_any, ctx = {}, None, None
            calibration_note = ("A calibration wrapper was detected in the artifact." if wrapper else
                                "No calibration wrapper detected: probabilities are raw model outputs and were not verified as calibrated.")

        if cal is not None:
            summary = (f"Calibrated probability of any failure class (1-4) is {pct(q_any)} (the raw model score said {pct(p_any)}, which is inflated). "
                       f"Highest raw class: {argmax} ({pct(p[argmax])}).")
        else:
            summary = (f"Model probability of any failure class (1-4) is {pct(p_any)}; of the two closest-to-repair classes (3-4) {pct(p_high)}. "
                       f"Highest single-class probability: class {argmax} ({pct(p[argmax])}).")
        ent = normalised_entropy(raw)
        result = PredictionResult(
            model_file=self.model_service.model_file, classes=list(CLASSES),
            probabilities={str(c): p[c] for c in CLASSES}, predicted_class_argmax=argmax, argmax_probability=p[argmax],
            probability_any_failure=p_any, probability_class_3_or_4=p_high, entropy_normalised=ent,
            risk_summary=summary, calibration_note=calibration_note, model_notes=self.model_service.reliability_notes(),
            decision_probabilities=dec, calibration_applied=cal is not None,
            calibrated_probability_any_failure=q_any, risk_context=ctx,
        )
        result.report = self._report(result, readout)
        return result

    # ------------------------------------------------------------------ explanation
    def _report(self, r: PredictionResult, readout: Optional[ValidatedReadout]) -> AgentReport:
        cal, ev = self.calibrator, (self.calibrator.evidence if self.calibrator else {})
        received = []
        if readout is not None:
            received.append(f"Validated readout for vehicle {readout.vehicle_id}: {readout.n_numeric} sensor-style features "
                            f"({len(readout.missing_numeric)} missing) and {readout.n_spec} specification features.")
        received.append(f"Saved model artifact: {r.model_file} (Random Forest pipeline with its own imputer and encoder).")
        performed = ["Passed the 113 inputs through the saved Random Forest and read its five class scores."]
        findings: List[Finding] = [Finding(
            "P1", "Raw model output",
            f"The model's highest raw class is {r.predicted_class_argmax} ({pct(r.argmax_probability)}). Its raw P(any failure) is {pct(r.probability_any_failure)}. "
            f"Raw scores are not real-world probabilities: the forest was trained on balanced classes, so it overstates failure for almost every vehicle.",
            "caution", ["M1", "M2"])]
        if r.calibration_applied:
            performed.append("Applied the validation-fitted calibration to correct the scale of the raw failure score (the model's ranking is kept).")
            base = (self.calibrator.reference or {}).get("base_failure_rate")
            base_txt = f" For comparison, {pct(base)} of validation vehicles had a recorded failure." if base else ""
            findings.append(Finding(
                "P2", "Calibrated failure probability",
                f"After calibration, P(any failure, classes 1-4) = {pct(r.calibrated_probability_any_failure)}.{base_txt} "
                f"This calibrated value, not the raw score, drives the cost decision.", "key", ["C1"]))
            if r.risk_context:
                c = r.risk_context
                performed.append(f"Placed this vehicle's risk score among {c['reference_n']:,} validation vehicles.")
                findings.append(Finding(
                    "P3", "Where this vehicle sits among validation vehicles",
                    f"Its risk score is {above(c['percentile'])} validation vehicles (risk group: {c['group_name']}). "
                    f"In that group, {c['group_failures']} of {c['group_vehicles']} vehicles ({pct(c['group_observed_failure_rate'])}) had a recorded failure.",
                    "key", ["R1"]))
        findings.append(Finding(
            "P4", "How spread out the raw scores are",
            f"Normalised entropy is {r.entropy_normalised:.2f} (0 = all weight on one class, 1 = equal weight on all five). "
            f"High values mean the model does not strongly favour any single class. This describes the raw score spread; it is not a confidence level.",
            "info", ["M1"]))
        limitations = ["Probabilities are model scores. Only the calibrated failure probability has been checked against real outcomes, and only on the validation and test sets."]
        if r.calibration_applied:
            auc = ev.get("validation_any_failure_auc")
            limitations.append("Calibration fixes the scale of the risk score, not the model's ability to rank vehicles"
                               + (f" (any-failure AUC on validation: {auc:.3f}; 0.5 would be a coin flip)." if auc else "."))
            if ev:
                limitations.append(f"Held-out test result of the calibrated policy: it flagged {pct(ev['test_flagged_share'])} of vehicles and caught "
                                   f"{pct(ev['test_failure_recall_any'])} of the failure vehicles; it identified the exact class correctly for "
                                   f"{ev['test_classes_1_to_3_correct']} of the class 1, 2 and 3 vehicles.")
        return AgentReport(self.name, "Runs the saved model and states the failure risk as probabilities.", received, performed, findings,
                           ["Raw class probabilities", "Calibrated class probabilities" if r.calibration_applied else "(no calibration applied)",
                            "Risk placement among validation vehicles" if r.risk_context else "(no population reference available)"],
                           limitations)
