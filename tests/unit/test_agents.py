import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agentic_app.agents.decision import DecisionAgent
from agentic_app.agents.investigation import InvestigationAgent
from agentic_app.agents.orchestrator import Orchestrator
from agentic_app.agents.prediction import PredictionAgent
from agentic_app.agents.verification import VerificationAgent
from agentic_app.core.costs import COST_MATRIX
from agentic_app.core.rules import compute_priority
from agentic_app.schemas.types import (Action, Investigation, PredictionResult, Recommendation, VerificationResult)
from agentic_app.services.validation import validate_readout
from tests.helpers import StubModelService, make_schema, make_settings, valid_payload

LOW = [0.97, 0.01, 0.01, 0.005, 0.005]
MID = [0.70, 0.15, 0.10, 0.03, 0.02]
HIGH = [0.25, 0.10, 0.10, 0.20, 0.35]


class AgentTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.settings = make_settings(self.tmp)

    def pipeline(self, probs, payload=None, notes=None):
        readout = validate_readout(payload or valid_payload(), make_schema(), self.settings)
        ms = StubModelService(probs, notes=notes)
        pred = PredictionAgent(ms).run(readout)
        inv = InvestigationAgent(self.settings).run(readout, pred)
        rec = DecisionAgent(self.settings).run(readout, pred, inv)
        ver = VerificationAgent(self.settings).run(readout, pred, inv, rec)
        return readout, pred, inv, rec, ver


class AgentSchemaTests(AgentTestBase):
    def test_prediction_output(self):
        _, pred, *_ = self.pipeline(HIGH)
        self.assertIsInstance(pred, PredictionResult)
        self.assertAlmostEqual(sum(pred.probabilities.values()), 1.0)
        self.assertEqual(pred.predicted_class_argmax, 4)
        self.assertAlmostEqual(pred.probability_any_failure, 0.75)
        self.assertAlmostEqual(pred.probability_class_3_or_4, 0.55)
        self.assertTrue(pred.risk_summary)
        json.dumps(pred.to_dict())

    def test_investigation_separates_evidence_and_hypotheses(self):
        _, _, inv, *_ = self.pipeline(HIGH)
        self.assertIsInstance(inv, Investigation)
        self.assertTrue(inv.evidence and inv.risk_indicators and inv.hypotheses and inv.missing_information)
        self.assertTrue(all(h.is_hypothesis and h.status == "unverified" for h in inv.hypotheses))
        self.assertTrue(all(e.kind in {"observed", "model_output", "derived"} for e in inv.evidence))
        self.assertNotIn("hypothes", " ".join(e.statement.lower() for e in inv.evidence))
        json.dumps(inv.to_dict())

    def test_investigation_does_not_invent_history(self):
        _, _, inv, *_ = self.pipeline(MID)
        text = " ".join(inv.missing_information).lower()
        self.assertIn("no repair or maintenance history", text)

    def test_missing_values_reported(self):
        p = valid_payload()
        p["features"]["171_0"] = None
        _, _, inv, *_ = self.pipeline(MID, payload=p)
        self.assertTrue(any("171_0" in m for m in inv.missing_information))

    def test_recommendation_schema_and_no_execution(self):
        _, _, _, rec, _ = self.pipeline(HIGH)
        self.assertIsInstance(rec, Recommendation)
        self.assertTrue(rec.requires_human_review)
        self.assertTrue(rec.actions)
        for a in rec.actions:
            self.assertIsInstance(a, Action)
            self.assertFalse(a.executed)
            self.assertFalse(a.automated)
            self.assertTrue(a.requires_human_approval)
        json.dumps(rec.to_dict())

    def test_priority_levels(self):
        self.assertEqual(compute_priority(HIGH, self.settings).priority, "critical")
        self.assertEqual(compute_priority(MID, self.settings).priority, "medium")
        self.assertEqual(compute_priority(LOW, self.settings).priority, "low")
        self.assertEqual(compute_priority([0.45, 0.2, 0.2, 0.1, 0.05], self.settings).priority, "high")

    def test_priority_is_configurable(self):
        strict = make_settings(self.tmp, AGENTIC_MEDIUM_P_ANY="0.01", AGENTIC_HIGH_P_ANY="0.02")
        self.assertEqual(compute_priority(LOW, strict).priority, "high")

    def test_recommended_class_matches_expected_cost_matrix(self):
        _, pred, _, rec, _ = self.pipeline(MID)
        probs = [pred.probabilities[str(i)] for i in range(5)]
        manual = [sum(probs[i] * COST_MATRIX[i][j] for i in range(5)) for j in range(5)]
        self.assertEqual(rec.recommended_class, manual.index(min(manual)))
        for j in range(5):
            self.assertAlmostEqual(rec.expected_costs[str(j)], manual[j])

    def test_reliability_notes_raise_uncertainty(self):
        _, _, inv, _, _ = self.pipeline(LOW, notes=["documented poor validation"])
        self.assertEqual(inv.uncertainty.level, "high")


class VerificationTests(AgentTestBase):
    def test_clean_recommendation_is_verified(self):
        *_, ver = self.pipeline(LOW)
        self.assertIsInstance(ver, VerificationResult)
        self.assertEqual(ver.status, "verified", ver.reasons)

    def test_high_missing_fraction_needs_review(self):
        s = make_settings(self.tmp, AGENTIC_REVIEW_MISSING_FRACTION="0.25")
        p = valid_payload()
        p["features"]["171_0"] = None
        self.settings = s
        *_, ver = self.pipeline(LOW, payload=p)
        self.assertEqual(ver.status, "needs_review")
        self.assertTrue(ver.review_reasons)

    def test_wrong_costs_detected(self):
        readout, pred, inv, rec, _ = self.pipeline(MID)
        bad = replace(rec, expected_costs={k: v + 1 for k, v in rec.expected_costs.items()})
        ver = VerificationAgent(self.settings).run(readout, pred, inv, bad)
        self.assertEqual(ver.status, "replan_required")
        self.assertIn("V02", ver.replan_issue_codes)

    def test_wrong_class_priority_and_executed_action_detected(self):
        readout, pred, inv, rec, _ = self.pipeline(HIGH)
        bad = replace(rec, recommended_class=0, priority="low",
                      actions=[Action("A1", "routine_monitoring", "The machine has been shut down.", "now", executed=True)])
        ver = VerificationAgent(self.settings).run(readout, pred, inv, bad)
        self.assertEqual(ver.status, "replan_required")
        for code in ("V03", "V04", "V05"):
            self.assertIn(code, ver.replan_issue_codes)

    def test_missing_explanation_and_bad_refs_detected(self):
        readout, pred, inv, rec, _ = self.pipeline(MID)
        bad = replace(rec, explanation="", evidence_refs=["ZZ9"])
        ver = VerificationAgent(self.settings).run(readout, pred, inv, bad)
        self.assertIn("V01", ver.replan_issue_codes)
        self.assertIn("V06", ver.replan_issue_codes)

    def test_hypothesis_mixed_into_evidence_is_blocking(self):
        readout, pred, inv, rec, _ = self.pipeline(MID)
        inv.hypotheses[0].is_hypothesis = False
        ver = VerificationAgent(self.settings).run(readout, pred, inv, rec)
        self.assertEqual(ver.status, "needs_review")


class FaultyDecisionAgent(DecisionAgent):
    """Emits a rule-violating first draft. ``fix_on_revise`` controls whether the revision is correct."""

    def __init__(self, settings, fix_on_revise):
        super().__init__(settings)
        self.fix_on_revise = fix_on_revise
        self.revise_calls = 0

    def run(self, readout, prediction, investigation):
        return replace(super().run(readout, prediction, investigation), priority="low", recommended_class=0)

    def revise(self, previous, issue_codes, readout, prediction, investigation):
        self.revise_calls += 1
        if self.fix_on_revise:
            return super().revise(previous, issue_codes, readout, prediction, investigation)
        return replace(previous, revision=previous.revision + 1)  # still wrong


class ReplanTests(AgentTestBase):
    def orchestrator(self, agent):
        return Orchestrator(self.settings, StubModelService(HIGH), decision_agent=agent)

    def test_replan_fixes_recommendation(self):
        agent = FaultyDecisionAgent(self.settings, fix_on_revise=True)
        res = self.orchestrator(agent).run_workflow(valid_payload())
        self.assertEqual(res.status, "completed")
        self.assertEqual(res.replan_count, 1)
        self.assertEqual(agent.revise_calls, 1)
        self.assertEqual(res.final_status, "verified")
        self.assertEqual(res.recommendation.revision, 1)
        self.assertEqual(res.recommendation.priority, "critical")
        stages = [t.stage for t in res.trace]
        self.assertEqual(stages.count("decision"), 2)
        self.assertEqual(stages.count("verification"), 2)

    def test_replan_is_bounded_then_escalates_to_human(self):
        agent = FaultyDecisionAgent(self.settings, fix_on_revise=False)
        res = self.orchestrator(agent).run_workflow(valid_payload())
        self.assertEqual(agent.revise_calls, 1)          # exactly one bounded revision, no loop
        self.assertEqual(res.replan_count, 1)
        self.assertEqual(res.final_status, "needs_review")
        self.assertTrue(res.escalated_to_human)
        self.assertTrue(any("exhausted" in r for r in res.verification.reasons))

    def test_zero_replans_escalates_immediately(self):
        self.settings = make_settings(self.tmp, AGENTIC_MAX_REPLANS="0")
        agent = FaultyDecisionAgent(self.settings, fix_on_revise=True)
        res = self.orchestrator(agent).run_workflow(valid_payload())
        self.assertEqual(agent.revise_calls, 0)
        self.assertEqual(res.final_status, "needs_review")


class OrchestratorTests(AgentTestBase):
    def test_full_workflow_and_trace(self):
        res = Orchestrator(self.settings, StubModelService(MID)).run_workflow(valid_payload())
        self.assertEqual(res.status, "completed")
        self.assertEqual([t.stage for t in res.trace], ["input_validation", "prediction", "investigation", "decision", "verification"])
        for t in res.trace:
            self.assertEqual(t.status, "ok")
            self.assertTrue(t.started_at and t.ended_at and t.output is not None)
        json.dumps(res.to_dict())

    def test_deterministic_decisions(self):
        a = Orchestrator(self.settings, StubModelService(MID)).run_workflow(valid_payload()).to_dict()
        b = Orchestrator(self.settings, StubModelService(MID)).run_workflow(valid_payload()).to_dict()
        for key in ("prediction", "investigation", "recommendation", "verification", "final_status", "input_fingerprint"):
            self.assertEqual(a[key], b[key], key)

    def test_validation_failure_is_graceful_and_skips_later_stages(self):
        bad = valid_payload()
        del bad["features"]["171_0"]
        res = Orchestrator(self.settings, StubModelService(MID)).run_workflow(bad)
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.error.code, "INPUT_VALIDATION_FAILED")
        self.assertIsNone(res.prediction)
        statuses = {t.stage: t.status for t in res.trace}
        self.assertEqual(statuses["input_validation"], "error")
        self.assertEqual(statuses["prediction"], "skipped")
        self.assertEqual(statuses["verification"], "skipped")

    def test_unexpected_agent_exception_is_captured(self):
        class Boom(PredictionAgent):
            def run(self, readout):
                raise RuntimeError("boom")
        ms = StubModelService(MID)
        res = Orchestrator(self.settings, ms, prediction_agent=Boom(ms)).run_workflow(valid_payload())
        self.assertEqual(res.error.code, "UNEXPECTED_ERROR")
        self.assertEqual(res.error.stage, "prediction")

    def test_predict_only_has_no_downstream_agents(self):
        res = Orchestrator(self.settings, StubModelService(MID)).predict(valid_payload())
        self.assertEqual(res.workflow, "predict_only")
        self.assertIsNotNone(res.prediction)
        self.assertIsNone(res.recommendation)


if __name__ == "__main__":
    unittest.main()
