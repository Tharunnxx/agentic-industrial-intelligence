"""Decision/Response Agent: expected-cost decision, documented priority rule, recommended (never executed) actions."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..core.config import Settings
from ..core.costs import COST_MATRIX, expected_costs, min_cost_class
from ..core.rules import compute_priority
from ..schemas.types import (Action, AgentReport, Finding, Investigation, PredictionResult, Recommendation, ValidatedReadout)
from .base import Agent
from .common import decision_probs, pct, raw_probs

ORIENTATION = "COST_MATRIX[actual][predicted]: rows = actual class, columns = predicted class"


def cost_breakdown(probs: List[float]) -> Dict[str, Dict[str, float]]:
    """For each decision j: how much each possible ACTUAL class i contributes to its expected cost, P(i) * Cost(i, j)."""
    return {str(j): {str(i): float(probs[i] * COST_MATRIX[i][j]) for i in range(5)} for j in range(5)}


class DecisionAgent(Agent):
    name = "DecisionAgent"
    responsibility = "Choose the minimum-expected-cost class, assign a priority and propose human-approved next steps."

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, readout: ValidatedReadout, prediction: PredictionResult, investigation: Investigation) -> Recommendation:
        return self._build(readout, prediction, investigation, revision=0, addressed=[])

    def revise(self, previous: Recommendation, issue_codes: List[str], readout: ValidatedReadout,
               prediction: PredictionResult, investigation: Investigation) -> Recommendation:
        """One bounded revision: rebuild from the evidence and record which verification issues were addressed."""
        return self._build(readout, prediction, investigation, revision=previous.revision + 1, addressed=list(issue_codes))

    # ------------------------------------------------------------------
    def _build(self, readout, prediction, investigation, revision: int, addressed: List[str]) -> Recommendation:
        s = self.settings
        cal = prediction.calibration_applied
        probs = decision_probs(prediction)
        raw = raw_probs(prediction)
        ec = expected_costs(probs)
        raw_ec = expected_costs(raw)
        c_star = min_cost_class(probs)
        raw_c = min_cost_class(raw)
        order = sorted(range(5), key=lambda j: ec[j])
        runner_up = order[1]
        margin = float(ec[runner_up] - ec[c_star])
        pr = compute_priority(probs, s, calibrated=cal)
        p_any = float(1.0 - probs[0])
        breakdown = cost_breakdown(probs)
        actions: List[Action] = []

        def add(kind: str, text: str, urgency: str) -> None:
            actions.append(Action(id=f"A{len(actions) + 1}", type=kind, description=text, urgency=urgency))

        if pr.priority == "critical":
            add("escalate_to_engineer", "Recommend escalation to a maintenance engineer for prompt review of this vehicle's records and readouts.", "prompt")
            add("schedule_inspection", "Recommend that the engineer decide whether and when to inspect the vehicle.", "engineer to decide")
        elif pr.priority == "high":
            add("schedule_inspection", "Recommend an inspection request for engineer approval; the engineer decides timing.", "soon")
        elif pr.priority == "medium":
            add("increase_monitoring", "Recommend closer monitoring: review upcoming readouts for this vehicle at a higher frequency, and consider an inspection at the next planned service.", "routine+")
        else:
            add("routine_monitoring", "Recommend continuing routine monitoring; no additional step is indicated by the model output.", "routine")
        if readout.missing_fraction >= s.review_missing_fraction:
            add("collect_data", "Recommend obtaining a more complete readout before relying on this result.", "before decision")
        add("record_outcome", "Recommend recording the engineer's decision and any inspection outcome so the model can later be checked against it.", "after review")

        available = {e.id for e in investigation.evidence}
        wanted = ["M1", "M2", "M3", "D1", "E1"] + (["C1"] if cal else []) + (["R1"] if prediction.risk_context else [])
        refs = [r for r in wanted if r in available] or ["D1"]
        level = investigation.uncertainty.level
        uncertainty_statement = f"Uncertainty is {level}: " + " ".join(investigation.uncertainty.factors)

        if cal:
            explanation = (
                f"Priority {pr.priority} under rule {pr.rule_id} ({pr.rationale}). "
                f"Decision: class {c_star} has the lowest expected cost ({ec[c_star]:.2f} units); the next best is class {runner_up} ({ec[runner_up]:.2f}). "
                f"Basis: calibrated probabilities (P(any failure) = {p_any:.1%}). The raw model's highest-probability class is {prediction.predicted_class_argmax}; "
                f"applying the cost rule to the raw, uncalibrated scores would have chosen class {raw_c}, but raw scores overstate failure, so they are not used for the decision. "
                f"{uncertainty_statement} This is a recommendation for a qualified maintenance engineer; nothing has been executed."
            )
        else:
            explanation = (
                f"Priority {pr.priority} under rule {pr.rule_id} ({pr.rationale}). "
                f"The minimum-expected-cost class is {c_star} (expected cost {ec[c_star]:.2f}); the raw highest-probability class is "
                f"{prediction.predicted_class_argmax}; because a missed failure costs far more than a false alarm, the cost-minimising label is conservative by design and priority is set from probabilities. Evidence: P(any failure)={prediction.probability_any_failure:.1%}, "
                f"P(class 3-4)={prediction.probability_class_3_or_4:.1%}, {readout.missing_fraction:.1%} of numeric features null. "
                f"{uncertainty_statement} This is a recommendation for a qualified maintenance engineer; nothing has been executed."
            )
        rec = Recommendation(
            recommended_class=c_star, raw_argmax_class=prediction.predicted_class_argmax,
            expected_costs={str(i): float(v) for i, v in enumerate(ec)}, cost_matrix_orientation=ORIENTATION,
            priority=pr.priority, priority_rule=pr.rule_id, priority_rationale=pr.rationale, actions=actions,
            explanation=explanation, evidence_refs=refs, uncertainty_statement=uncertainty_statement,
            requires_human_review=True, revision=revision, addressed_issues=addressed,
            decision_basis="calibrated probabilities" if cal else "raw model probabilities",
            raw_expected_costs={str(i): float(v) for i, v in enumerate(raw_ec)}, raw_min_cost_class=raw_c,
            runner_up_class=runner_up, margin_to_runner_up=margin, cost_breakdown=breakdown,
        )
        rec.report = self._report(rec, prediction, probs, ec, raw_ec, c_star, raw_c, runner_up, pr, cal, p_any)
        return rec

    # ------------------------------------------------------------------ explanation
    def _report(self, rec, prediction, probs, ec, raw_ec, c_star, raw_c, runner_up, pr, cal, p_any) -> AgentReport:
        bd = rec.cost_breakdown
        parts = sorted(((int(i), v) for i, v in bd[str(c_star)].items()), key=lambda t: -t[1])[:3]
        arithmetic = " + ".join(f"P(class {i})={probs[i]:.3f} x cost {COST_MATRIX[i][c_star]:.0f} = {v:.2f}" for i, v in parts if v > 0)
        total0_fail = sum(bd["0"][str(i)] for i in range(1, 5))
        total4_false = bd["4"]["0"]
        findings = [
            Finding("D1", "Recommended decision",
                    f"Class {c_star} has the lowest expected cost: {ec[c_star]:.2f} units. The next best is class {runner_up} at {ec[runner_up]:.2f} units "
                    f"(gap {rec.margin_to_runner_up:.2f}). Basis: {rec.decision_basis}.", "key", ["D1"]),
            Finding("D2", "How the expected cost was calculated",
                    f"Expected cost of a decision = sum over actual classes of P(actual) x Cost(actual, decision). For class {c_star} the largest terms are: "
                    f"{arithmetic or 'all terms are zero'}. Cost values are project-defined units, not currency.", "info", ["D1"]),
            Finding("D3", "The two extreme decisions",
                    f"Class 0 (no action) costs {ec[0]:.2f}; {pct(total0_fail / ec[0]) if ec[0] > 0 else '0%'} of that comes from possible missed failures. "
                    f"Class 4 (maximum action) costs {ec[4]:.2f}; {pct(total4_false / ec[4]) if ec[4] > 0 else '0%'} of that comes from false alarms on a healthy vehicle. "
                    f"The cheapest decision balances these two risks.", "info", ["D1"]),
        ]
        if cal:
            findings.append(Finding(
                "D4", "Why calibrated probabilities are used",
                f"With the raw model scores the cost rule would choose class {raw_c} (expected cost {raw_ec[raw_c]:.2f}). Raw scores say almost every vehicle is likely to fail, "
                f"so that rule recommends maximum action for nearly everyone. With calibrated P(any failure) = {pct(p_any)} the cheapest decision is class {c_star}.",
                "key", ["C1"]))
        findings.append(Finding("D5", "Priority", f"{pr.priority.upper()} under rule {pr.rule_id}: {pr.rationale}.", "key", ["D1"]))
        findings.append(Finding("D6", "Recommended next steps (for engineer approval)", " ".join(f"[{a.id}] {a.description}" for a in rec.actions), "info", ["D1"]))
        return AgentReport(
            self.name, "Chooses the cheapest decision from expected cost, sets a priority and proposes next steps for a human to approve.",
            [f"Prediction output ({'calibrated' if cal else 'raw'} probabilities).", "Investigation output: evidence, risk rules, uncertainty.", "The official cost matrix (rows = actual, columns = decision)."],
            ["Calculated the expected cost of all five decisions.", "Picked the minimum and the runner-up.", "Assigned the priority from the documented rule.", "Drafted next steps; none is executed."],
            findings,
            [f"Recommended class {c_star}, priority {pr.priority}", "Five expected costs and the per-class breakdown", f"{len(rec.actions)} recommended actions", "Plain-language explanation"],
            ["Cost units are project-defined, not money.",
             "The cost matrix makes classes 1-3 rarely cheapest, so read the output as 'act or do not act' plus urgency, not as a precise failure class.",
             "Priority bands are a documented policy choice for ordering inspections, not a measured safety threshold.",
             "Nothing is executed. Every action needs approval from a qualified maintenance engineer."])
