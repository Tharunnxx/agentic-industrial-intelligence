"""Investigation Agent: rule-based evidence review. Separates observed evidence from hypotheses.

It does NOT diagnose faults. Feature names are anonymised, so nothing here can name a component or a root cause.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..core.config import Settings
from ..core.costs import expected_costs, min_cost_class
from ..schemas.types import (AgentReport, EvidenceItem, Finding, Hypothesis, Investigation, PredictionResult, RiskIndicator,
                             UncertaintyAssessment, ValidatedReadout)
from .base import Agent
from .common import above, decision_probs, pct, raw_probs

CENTRAL_RANGE_EXPECTED = 0.02  # a value outside the 1st-99th percentile happens for ~2% of readings by chance


class InvestigationAgent(Agent):
    name = "InvestigationAgent"
    responsibility = "Turn the prediction and the supplied readout into structured evidence, risk indicators, gaps and uncertainty."

    def __init__(self, settings: Settings, feature_reference: Optional[Dict[str, Any]] = None):
        self.settings = settings
        self.feature_reference = feature_reference  # per-feature p01/p50/p99 of validation vehicles (descriptive only)

    # ------------------------------------------------------------------ unusual readings
    def _unusual_readings(self, readout: ValidatedReadout) -> Optional[Dict[str, Any]]:
        ref = (self.feature_reference or {}).get("features")
        if not ref:
            return None
        checked, above, below = 0, [], []
        for name, value in readout.features.items():
            r = ref.get(name)
            if r is None or isinstance(value, str) or value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(v):
                continue
            checked += 1
            span = max(r["p99"] - r["p01"], 1e-9)
            if v > r["p99"]:
                above.append((name, (v - r["p99"]) / span))
            elif v < r["p01"]:
                below.append((name, (r["p01"] - v) / span))
        if checked == 0:
            return None
        outside = len(above) + len(below)
        top = sorted(above + below, key=lambda t: -t[1])[:5]
        return {"checked": checked, "above_p99": len(above), "below_p01": len(below), "outside": outside,
                "expected_by_chance": round(CENTRAL_RANGE_EXPECTED * checked, 1), "most_extreme": [n for n, _ in top],
                "reference": f"{self.feature_reference.get('n_vehicles', '?')} validation vehicles"}

    # ------------------------------------------------------------------ main
    def run(self, readout: ValidatedReadout, prediction: PredictionResult) -> Investigation:
        s = self.settings
        cal = prediction.calibration_applied
        probs = decision_probs(prediction)
        raw = raw_probs(prediction)
        ec = expected_costs(probs)
        c_star = min_cost_class(probs)
        order = sorted(range(5), key=lambda j: ec[j])
        runner_up = order[1]
        margin_ratio = (ec[runner_up] - ec[c_star]) / ec[runner_up] if ec[runner_up] > 0 else 1.0
        p_any = float(1.0 - probs[0])
        p_high = float(probs[3] + probs[4])
        raw_any = prediction.probability_any_failure
        unusual = self._unusual_readings(readout)
        ctx = prediction.risk_context

        # ---------------- evidence (what was observed / output / derived) ----------------
        evidence: List[EvidenceItem] = [
            EvidenceItem("E1", "observed",
                         f"Readout supplied with {readout.n_numeric} numeric features ({len(readout.missing_numeric)} null, "
                         f"{readout.missing_fraction:.1%}) and {readout.n_spec} specification fields.",
                         {"n_numeric": readout.n_numeric, "n_null": len(readout.missing_numeric), "n_spec": readout.n_spec}),
            EvidenceItem("E2", "observed", "Specification field values as supplied.", readout.spec_values),
        ]
        if readout.vehicle_id is not None or readout.time_step is not None:
            evidence.append(EvidenceItem("E3", "observed", "Readout identifiers as supplied (not used as model features).",
                                         {"vehicle_id": readout.vehicle_id, "time_step": readout.time_step}))
        evidence += [
            EvidenceItem("M1", "model_output", f"Raw class probabilities from {prediction.model_file}.", prediction.probabilities),
            EvidenceItem("M2", "model_output", f"Raw P(any failure, classes 1-4) = {raw_any:.4f}.", raw_any),
            EvidenceItem("M3", "model_output", f"Raw P(class 3 or 4) = {prediction.probability_class_3_or_4:.4f}.", prediction.probability_class_3_or_4),
        ]
        if cal:
            evidence.append(EvidenceItem("C1", "derived", f"Calibrated P(any failure) = {p_any:.4f} (raw score was {raw_any:.4f}).",
                                         {"calibrated_probabilities": prediction.decision_probabilities}))
        if ctx:
            evidence.append(EvidenceItem("R1", "derived",
                                         f"Raw risk score is {above(ctx['percentile'])} the {ctx['reference_n']:,} validation vehicles; "
                                         f"in this risk group {ctx['group_failures']}/{ctx['group_vehicles']} vehicles had a recorded failure.", ctx))
        if unusual:
            evidence.append(EvidenceItem("O1", "derived",
                                         f"{unusual['outside']} of {unusual['checked']} readings lie outside the central 98% range of validation vehicles "
                                         f"(about {unusual['expected_by_chance']} expected by chance).", unusual))
        evidence.append(EvidenceItem("D1", "derived",
                                     f"Expected cost per decision class (official matrix, rows=actual, cols=decision) using "
                                     f"{'calibrated' if cal else 'raw'} probabilities: {[round(float(x), 4) for x in ec]}; minimum at class {c_star}; "
                                     f"raw argmax is class {prediction.predicted_class_argmax}.",
                                     {"expected_costs": {str(i): float(v) for i, v in enumerate(ec)}, "min_cost_class": c_star}))

        # ---------------- risk indicators (rules) ----------------
        if cal:
            indicators = [
                RiskIndicator("RI1", "cost rule recommends an action (class > 0)", "Expected-cost minimum is a failure decision", c_star, 0, c_star > 0),
                RiskIndicator("RI2", "calibrated P(any failure) >= high band", "Calibrated failure probability is high", p_any, s.cal_high_p_any, p_any >= s.cal_high_p_any),
                RiskIndicator("RI3", "calibrated P(any failure) >= critical band", "Calibrated failure probability is very high", p_any, s.cal_critical_p_any, p_any >= s.cal_critical_p_any),
            ]
        else:
            indicators = [
                RiskIndicator("RI1", "P(any failure) >= medium threshold", "Moderate model-estimated failure probability", p_any, s.medium_p_any, p_any >= s.medium_p_any),
                RiskIndicator("RI2", "P(any failure) >= high threshold", "High model-estimated failure probability", p_any, s.high_p_any, p_any >= s.high_p_any),
                RiskIndicator("RI3", "P(class 3 or 4) >= critical threshold", "Probability mass on the closest-to-repair classes", p_high, s.critical_p_high, p_high >= s.critical_p_high),
            ]
        indicators += [
            RiskIndicator("RI4", "min-expected-cost class differs from raw argmax", "Cost asymmetry changes the decision", c_star, prediction.predicted_class_argmax, c_star != prediction.predicted_class_argmax),
            RiskIndicator("RI5", "null fraction >= review threshold", "Incomplete readout", readout.missing_fraction, s.review_missing_fraction, readout.missing_fraction >= s.review_missing_fraction),
            RiskIndicator("RI6", "specification value unseen in training", "Input outside the encoder's known categories", len(readout.unseen_spec_values), 0, bool(readout.unseen_spec_values)),
        ]

        # ---------------- hypotheses (always unverified) ----------------
        flag_threshold = s.cal_high_p_any if cal else s.medium_p_any
        hypotheses: List[Hypothesis] = []
        if c_star > 0 or p_any >= flag_threshold:
            hypotheses.append(Hypothesis("H1", "The readout resembles training readouts that preceded a recorded repair; this is a statistical association of the model, "
                                               "not a diagnosed fault. The affected component cannot be identified because feature names are anonymised.", ["M2", "E1"]))
        if (not cal) and p_high >= s.critical_p_high:
            hypotheses.append(Hypothesis("H2", "A repair window close to the readout time is possible (classes 3-4 are the closest-to-repair classes under the dataset's label definition). "
                                               "Unverified until a qualified engineer reviews vehicle records.", ["M3"]))
        if unusual and unusual["outside"] > 2 * max(unusual["expected_by_chance"], 1):
            hypotheses.append(Hypothesis("H3", "Several readings are unusual compared with validation vehicles. This may reflect a different operating profile, "
                                               "a sensor or data issue, or a developing fault; the data cannot tell which.", ["O1"]))
        if not hypotheses:
            hypotheses.append(Hypothesis("H1", "Low estimated risk does not rule out a developing fault; a single readout carries no trend information.", ["M2"]))

        # ---------------- missing information ----------------
        missing_info = [
            "No repair or maintenance history was supplied.",
            "Only a single readout was supplied; no trend or time-series evidence is available.",
            f"The {readout.n_numeric} operational features are anonymised; no sensor-level root cause can be derived from them.",
        ]
        if readout.missing_numeric:
            shown = ", ".join(readout.missing_numeric[:10]) + (" ..." if len(readout.missing_numeric) > 10 else "")
            missing_info.append(f"{len(readout.missing_numeric)} numeric feature(s) are null: {shown}.")
        if readout.vehicle_id is None:
            missing_info.append("No vehicle_id supplied; the result cannot be linked to a vehicle record.")

        # ---------------- uncertainty ----------------
        factors: List[str] = []
        notes = list(prediction.model_notes)
        level = "moderate" if cal else "low"
        if readout.missing_fraction >= s.review_missing_fraction:
            factors.append(f"{readout.missing_fraction:.1%} of numeric features are null."); level = "high"
        elif readout.missing_fraction >= 0.2:
            factors.append(f"{readout.missing_fraction:.1%} of numeric features are null."); level = "moderate" if level != "high" else level
        if readout.unseen_spec_values:
            factors.append("A specification value was not seen during training."); level = "high"
        if cal:
            factors.append("Calibration adjusts the failure-risk score scale; it does not improve the underlying model's ability to rank or identify failure classes. Consult the held-out evaluation for measured ranking performance.")
            if margin_ratio < 0.10:
                factors.append(f"Close call: the recommended decision (class {c_star}) is only {margin_ratio:.1%} cheaper than class {runner_up}; "
                               f"a small change in risk could flip it.")
        else:
            if prediction.argmax_probability < 0.5:
                factors.append(f"Probability is spread out (highest single class {prediction.argmax_probability:.1%}, normalised entropy {prediction.entropy_normalised:.2f}).")
                level = "high" if level == "high" else "moderate"
            if notes:
                factors.append("Documented evaluation of this model artifact shows poor validation behaviour (see reliability notes)."); level = "high"
            if "No calibration" in prediction.calibration_note:
                factors.append("Probabilities are not verified as calibrated."); level = "moderate" if level == "low" else level
        if not factors:
            factors.append("No uncertainty factor exceeded its threshold; this does not make the prediction certain.")

        triggered = sum(i.triggered for i in indicators)
        summary = (f"{triggered} of {len(indicators)} risk rules triggered; {'calibrated ' if cal else ''}P(any failure)={p_any:.1%}, "
                   f"minimum-expected-cost class {c_star}, uncertainty {level}.")
        investigation = Investigation(summary, evidence, indicators, hypotheses, missing_info,
                                      UncertaintyAssessment(level, factors, notes, readout.missing_fraction, prediction.entropy_normalised))
        investigation.report = self._report(readout, prediction, investigation, ctx, unusual, c_star, runner_up, margin_ratio, p_any, raw, ec)
        return investigation

    # ------------------------------------------------------------------ explanation
    def _report(self, readout, prediction, inv, ctx, unusual, c_star, runner_up, margin_ratio, p_any, raw, ec) -> AgentReport:
        cal = prediction.calibration_applied
        findings: List[Finding] = []
        dq = (f"{readout.n_numeric - len(readout.missing_numeric)} of {readout.n_numeric} sensor-style features are present "
              f"({pct(readout.missing_fraction)} missing).")
        if readout.unseen_spec_values:
            dq += f" A specification value was never seen in training: {readout.unseen_spec_values}."
        else:
            dq += " All specification values were seen in training."
        findings.append(Finding("I1", "Data completeness", dq, "caution" if (readout.missing_numeric or readout.unseen_spec_values) else "info", ["E1", "E2"]))
        if ctx:
            findings.append(Finding(
                "I2", "Risk compared with validation vehicles",
                f"The vehicle's risk score is {above(ctx['percentile'])} validation vehicles. Vehicles in its group ({ctx['group_name']}) "
                f"had a recorded failure {pct(ctx['group_observed_failure_rate'])} of the time, against {pct(ctx['base_failure_rate'])} overall.", "key", ["R1"]))
        # how the model splits the risk over failure classes
        fail = [(c, decision_probs(prediction)[c]) for c in range(1, 5)]
        tot = sum(p for _, p in fail) or 1.0
        split = ", ".join(f"class {c} {pct(p / tot, 0)}" for c, p in fail)
        findings.append(Finding(
            "I3", "Which failure class, if any?",
            f"If a failure is coming, the model splits the risk as: {split}. Treat this split with caution: on held-out data the model has not reliably "
            f"identified which failure class a vehicle belongs to.", "caution", ["M1"]))
        if unusual:
            names = ", ".join(unusual["most_extreme"]) if unusual["most_extreme"] else "none"
            findings.append(Finding(
                "I4", "Unusual readings",
                f"{unusual['outside']} of {unusual['checked']} readings are outside the central 98% range of validation vehicles "
                f"(about {unusual['expected_by_chance']} would be expected by chance). Most extreme: {names}. "
                f"Feature names are anonymised, so these cannot be linked to a physical component.",
                "caution" if unusual["outside"] > 2 * max(unusual["expected_by_chance"], 1) else "info", ["O1"]))
        if c_star != prediction.predicted_class_argmax:
            findings.append(Finding(
                "I5", "Prediction and decision differ",
                f"The model's highest raw class is {prediction.predicted_class_argmax}, but the cheapest decision is class {c_star}. "
                f"This is expected: a missed failure costs 200-500 units while a false alarm costs 7-10, so the decision favours acting even at modest risk. "
                f"The prediction says what is most likely; the decision says what costs least on average.", "key", ["D1"]))
        findings.append(Finding(
            "I6", "Decision margin",
            f"Class {c_star} has the lowest expected cost ({ec[c_star]:.2f} units); the next best is class {runner_up} ({ec[runner_up]:.2f}). "
            f"The gap is {margin_ratio:.1%} of the runner-up's cost." + (" This is a close call." if margin_ratio < 0.10 else ""),
            "caution" if margin_ratio < 0.10 else "info", ["D1"]))
        performed = ["Checked data completeness and specification values against what the model saw in training."]
        if ctx:
            performed.append("Compared the vehicle's risk score with the validation population and the failure rate seen there.")
        if unusual:
            performed.append(f"Compared all {unusual['checked']} sensor-style readings with the central 98% range of validation vehicles.")
        performed += ["Computed expected cost for each of the five decisions and found the cheapest.", f"Evaluated {len(inv.risk_indicators)} risk rules."]
        return AgentReport(
            self.name, "Reviews the prediction as evidence: what supports it, what is missing, how uncertain it is. It does not diagnose faults.",
            [f"Validated readout for vehicle {readout.vehicle_id}.", "Prediction Agent output: raw and calibrated probabilities, risk placement."],
            performed, findings,
            [f"{sum(i.triggered for i in inv.risk_indicators)} of {len(inv.risk_indicators)} risk rules triggered", f"Uncertainty level: {inv.uncertainty.level}",
             "Evidence items and unverified hypotheses", "Missing-information list"],
            ["Feature names are anonymised: nothing here identifies a component or a root cause.",
             "Only one readout per vehicle: no trend or history was available.",
             "No maintenance records were available.",
             "Hypotheses are statistical associations and stay unverified until a qualified engineer reviews them."])
