"""Verification Agent: independently re-checks the recommendation against the configured rules."""
from __future__ import annotations

import re
from typing import List

import numpy as np

from typing import Any, Dict, Optional

from ..core.calibration import Calibrator
from ..core.config import Settings
from ..core.costs import COST_MATRIX
from ..core.rules import PRIORITY_LEVELS, compute_priority
from ..schemas.types import (AgentReport, Finding, Investigation, PredictionResult, Recommendation, ValidatedReadout,
                             VerificationCheck, VerificationResult)
from .base import Agent
from .common import decision_probs, pct, raw_probs

# What each check looks at and why a reader should care (shown in the interface).
CHECK_INFO = {
    "V01": ("The recommendation has priority, actions, explanation, uncertainty statement, five costs and evidence references.", "An incomplete recommendation cannot be explained or audited."),
    "V02": ("Expected costs recomputed from the probabilities with the official matrix.", "Confirms the displayed costs really follow from the probabilities (rows = actual class)."),
    "V03": ("The recommended class is the minimum of the recomputed expected costs.", "The decision must be the cheapest option, not just a plausible one."),
    "V04": ("The priority recomputed from the documented priority rule.", "Priority must come from a written rule, not from the agent's mood."),
    "V05": ("No action claims to be executed or automated; wording is checked for forbidden phrases.", "The system only recommends; a human approves any maintenance action."),
    "V06": ("Every evidence id cited by the recommendation exists in the investigation.", "Each claim must trace back to evidence."),
    "V07": ("High/critical priority comes with an escalation or inspection recommendation.", "An urgent priority without a next step is not useful."),
    "V08": ("Hypotheses are marked unverified; evidence kinds are valid; risk indicators have values.", "Keeps observed facts apart from guesses."),
    "V09": ("Probabilities are finite and sum to 1.", "Every later number depends on valid probabilities."),
    "V10": ("Human review is flagged as required.", "Human-in-the-loop approval must never be switched off."),
    "V11": ("Calibrated probabilities recomputed independently from the raw scores with the saved calibration numbers.", "Confirms the calibration step was applied correctly and consistently."),
    "V12": ("The explanation states what the raw, uncalibrated scores would have decided.", "Readers must be able to see the effect of calibration."),
    "V13": ("Priority is consistent with where the vehicle's risk sits among validation vehicles.", "A high priority for a low-ranked vehicle (or the reverse) points to a mistake, so a human should look."),
    "V14": ("All three agent reports are present and each lists its limitations.", "Results must always travel with their limitations."),
}


def refresh_assurance(result: VerificationResult, evidence: Optional[Dict[str, Any]] = None) -> None:
    """Four different meanings of 'OK', kept apart on purpose. Called again if the orchestrator changes the final status."""
    n, passed = len(result.checks), sum(c.passed for c in result.checks)
    sw = {"verified": "passed", "needs_review": "needs_review", "replan_required": "failed"}.get(result.status, result.status)
    ev = evidence or {}
    if ev:
        model_text = (f"Not checked per vehicle. Measured on the held-out test set ({ev['test_vehicles']:,} vehicles): the calibrated policy had a mean cost of "
                      f"{ev['test_mean_cost']:.2f} units per vehicle against {ev['test_original_cost_rule_mean_cost']:.2f} for the original cost rule and "
                      f"{ev['test_always_4_mean_cost']:.2f} for always choosing class 4. It caught {pct(ev['test_failure_recall_any'])} of failure vehicles.")
    else:
        model_text = "Not checked per vehicle. See the Model Performance page for measured results on held-out data."
    result.assurance = {
        "workflow_execution": {"status": "completed", "meaning": "All workflow stages ran without a software error."},
        "software_checks": {"status": sw, "meaning": f"{passed} of {n} software verification checks passed. These check the code's arithmetic and rules, not the real world."},
        "model_correctness": {"status": "not_verified_for_this_vehicle", "meaning": model_text},
        "physical_equipment_safety": {"status": "not_assessed", "meaning": "Nothing here assesses the physical condition or safety of any equipment. Only a qualified engineer can."},
    }

FORBIDDEN = re.compile(
    r"\b(shut[- ]?down|stop(ped)? the (machine|vehicle|line)|halt|disable|automatically (stop|shut|dispatch|order)|"
    r"has been (executed|performed|completed|scheduled|dispatched|ordered)|have been (executed|performed|completed|scheduled)|was (executed|performed|dispatched))\b",
    re.IGNORECASE,
)
ALLOWED_EVIDENCE_KINDS = {"observed", "model_output", "derived"}


class VerificationAgent(Agent):
    name = "VerificationAgent"
    responsibility = "Verify the recommendation against the cost matrix, priority rule, safety rules and evidence."

    def __init__(self, settings: Settings, calibrator: Optional[Calibrator] = None):
        self.settings = settings
        self.calibrator = calibrator

    def run(self, readout: ValidatedReadout, prediction: PredictionResult, investigation: Investigation,
            recommendation: Recommendation) -> VerificationResult:
        s = self.settings
        checks: List[VerificationCheck] = []

        def check(cid: str, name: str, ok: bool, severity: str, msg_ok: str, msg_fail: str) -> None:
            what, why = CHECK_INFO.get(cid, ("", ""))
            checks.append(VerificationCheck(cid, name, bool(ok), severity, msg_ok if ok else msg_fail, what, why))

        r = recommendation
        cal = prediction.calibration_applied
        dec = prediction.decision_probabilities or prediction.probabilities
        probs = np.array([dec.get(str(c), np.nan) for c in prediction.classes], dtype=float)
        prob_ok = probs.shape == (5,) and bool(np.all(np.isfinite(probs))) and abs(float(probs.sum()) - 1.0) < 1e-3
        check("V01", "required_fields",
              r.priority in PRIORITY_LEVELS and len(r.actions) > 0 and bool(r.explanation.strip()) and bool(r.uncertainty_statement.strip())
              and len(r.expected_costs) == 5 and len(r.evidence_refs) > 0,
              "fixable", "All required recommendation fields are present.", "Recommendation is missing priority, actions, explanation, uncertainty statement, expected costs or evidence references.")
        check("V09", "probability_sanity", prob_ok, "blocking", "Probabilities are finite and sum to 1.", "Prediction probabilities are invalid.")

        if prob_ok:
            recomputed = probs @ COST_MATRIX
            try:
                reported = np.array([r.expected_costs[str(i)] for i in range(5)], dtype=float)
                costs_ok = bool(np.allclose(recomputed, reported, atol=1e-6))
            except (KeyError, TypeError, ValueError):
                costs_ok = False
            check("V02", "expected_costs_recomputed", costs_ok, "fixable", "Expected costs match an independent recomputation (rows = actual).", "Reported expected costs differ from the recomputation.")
            check("V03", "recommended_class_minimises_expected_cost", int(recomputed.argmin()) == r.recommended_class, "fixable",
                  f"Recommended class {r.recommended_class} minimises expected cost.", f"Recommended class {r.recommended_class} is not the minimum-expected-cost class ({int(recomputed.argmin())}).")
            expected_priority = compute_priority(probs, s, calibrated=cal).priority
            check("V04", "priority_matches_rule", r.priority == expected_priority, "fixable",
                  f"Priority '{r.priority}' matches the configured rule.", f"Priority '{r.priority}' contradicts the configured rule (expected '{expected_priority}').")

        texts = [a.description for a in r.actions]  # the explanation legitimately says "nothing has been executed"
        offending = [t for t in texts if FORBIDDEN.search(t)]
        flagged = [a.id for a in r.actions if a.executed or a.automated or not a.requires_human_approval]
        check("V05", "no_executed_or_automated_actions", not offending and not flagged, "fixable",
              "Every action is a human-approved recommendation; none is claimed as executed.",
              f"Actions claim execution/automation or use forbidden wording (flags: {flagged}, texts: {len(offending)}).")

        ids = {e.id for e in investigation.evidence}
        unknown = [x for x in r.evidence_refs if x not in ids]
        check("V06", "evidence_refs_resolve", bool(r.evidence_refs) and not unknown, "fixable",
              "All cited evidence exists in the investigation.", f"Recommendation cites unknown evidence ids: {unknown}.")
        check("V07", "priority_action_consistency",
              r.priority not in ("high", "critical") or any(a.type in ("escalate_to_engineer", "schedule_inspection") for a in r.actions),
              "fixable", "High/critical priority includes an escalation or inspection recommendation.", "High/critical priority lacks an escalation or inspection recommendation.")
        check("V10", "human_review_flag", r.requires_human_review is True, "fixable", "Human review is required for the recommendation.", "requires_human_review must be true.")
        check("V08", "evidence_vs_hypothesis_separation",
              all(h.is_hypothesis and h.status == "unverified" for h in investigation.hypotheses)
              and all(e.kind in ALLOWED_EVIDENCE_KINDS for e in investigation.evidence)
              and all(i.observed_value is not None for i in investigation.risk_indicators),
              "blocking", "Hypotheses are flagged unverified; evidence kinds and risk indicators are well-formed.", "Investigation mixes hypotheses with evidence or has malformed indicators.")

        # ---- checks added for the calibrated workflow ----
        if cal or self.calibrator is not None:
            if self.calibrator is None:
                check("V11", "calibration_reproduced", False, "blocking", "", "Calibration was applied but the verifier has no calibrator to recompute it.")
            elif not cal:
                check("V11", "calibration_reproduced", False, "blocking", "", "A calibrator is configured but the prediction was not calibrated.")
            else:
                again = self.calibrator.apply(np.array([prediction.probabilities[str(c)] for c in prediction.classes], dtype=float))
                same = bool(np.allclose(again, probs, atol=1e-9))
                check("V11", "calibration_reproduced", same, "blocking", "Calibrated probabilities match an independent recomputation from the raw scores.",
                      "Calibrated probabilities differ from an independent recomputation.")
            if cal:
                check("V12", "raw_vs_calibrated_disclosed", r.raw_min_cost_class is not None and "raw" in r.explanation.lower(), "fixable",
                      "The explanation discloses what the raw scores would have decided.", "The explanation does not disclose the raw-score decision.")
        ctx = prediction.risk_context
        if ctx:
            pct_rank = ctx["percentile"]
            ok13 = not (r.priority in ("high", "critical") and pct_rank < 50) and not (r.priority == "low" and pct_rank >= 80)
            check("V13", "priority_consistent_with_population", ok13, "blocking",
                  f"Priority '{r.priority}' is consistent with the vehicle's risk percentile ({pct_rank:.0f}).",
                  f"Priority '{r.priority}' looks inconsistent with a risk percentile of {pct_rank:.0f}; a human should review.")
        if r.report is not None:
            reports = [prediction.report, investigation.report, r.report]
            check("V14", "limitations_disclosed", all(x is not None and x.limitations for x in reports), "fixable",
                  "Prediction, investigation and decision reports each state their limitations.", "At least one agent report is missing or lists no limitations.")

        checks.sort(key=lambda c: int(c.id[1:]))
        failed_fixable = [c for c in checks if not c.passed and c.severity == "fixable"]
        failed_blocking = [c for c in checks if not c.passed and c.severity == "blocking"]
        review_reasons: List[str] = []
        if readout.missing_fraction >= s.review_missing_fraction:
            review_reasons.append(f"Data completeness: {readout.missing_fraction:.1%} of numeric features are null (review threshold {s.review_missing_fraction:.0%}).")
        if readout.unseen_spec_values:
            review_reasons.append("A specification value was not seen during model training.")

        if failed_fixable:
            result = VerificationResult("replan_required", checks, [f"{c.id} {c.name}: {c.message}" for c in failed_fixable],
                                        [c.id for c in failed_fixable], review_reasons)
        elif failed_blocking or review_reasons:
            reasons = [f"{c.id} {c.name}: {c.message}" for c in failed_blocking] + review_reasons
            result = VerificationResult("needs_review", checks, reasons, [], review_reasons)
        else:
            result = VerificationResult("verified", checks, ["All verification checks passed; human review of the recommendation is still expected."], [], [])
        evidence = self.calibrator.evidence if self.calibrator else None
        refresh_assurance(result, evidence)
        result.report = self._report(result, checks, recommendation, evidence)
        return result

    def _report(self, result: VerificationResult, checks, recommendation, evidence) -> AgentReport:
        n, passed = len(checks), sum(c.passed for c in checks)
        failed = [c for c in checks if not c.passed]
        findings = [Finding("V-A", "Software verification result",
                            f"{passed} of {n} checks passed. Final status: {result.status}. "
                            + ("; ".join(f"{c.id} {c.name}" for c in failed) if failed else "No check failed."),
                            "key" if not failed else "caution")]
        findings.append(Finding("V-B", "What 'verified' does and does not mean",
                                "Verified means the code's own arithmetic, rules and wording passed independent re-checks. It does not mean the model's prediction is correct, "
                                "and it says nothing about the physical safety of any equipment.", "caution"))
        if recommendation.revision:
            findings.append(Finding("V-C", "Replanning", f"The decision was revised {recommendation.revision} time(s) after these issues: {recommendation.addressed_issues}.", "info"))
        else:
            findings.append(Finding("V-C", "Replanning", "No replanning was needed: the first recommendation passed every blocking and fixable check. "
                                    "Replanning exists as a bounded safety net and runs only when a fixable check fails.", "info"))
        return AgentReport(
            self.name, "Independently re-checks the recommendation. It checks the software's consistency, not the real world.",
            ["Validated readout.", "Prediction, investigation and recommendation outputs."],
            [f"Recomputed expected costs and priority from the probabilities.", f"Ran {n} verification checks.", "Separated software checks from model correctness and physical safety."],
            findings, [f"Status: {result.status}", f"{passed} of {n} checks passed", "Assurance statement (four separate meanings of OK)"],
            ["Checks are rule-based: they can only catch inconsistencies the rules anticipate.",
             "Passing every check does not make the prediction correct.",
             "Replanning cannot change an uncertain model; it only fixes inconsistencies in the decision."])
