"""Tests for the calibration layer, the explanatory agent reports, the new verification checks and the vehicle index.
All inputs here are SYNTHETIC or come from the project's own saved result files; nothing is used to report model performance."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_app.agents.decision import DecisionAgent
from agentic_app.agents.investigation import InvestigationAgent
from agentic_app.agents.orchestrator import Orchestrator
from agentic_app.agents.prediction import PredictionAgent
from agentic_app.agents.verification import VerificationAgent
from agentic_app.core.calibration import Calibrator
from agentic_app.core.costs import COST_MATRIX
from agentic_app.core.rules import compute_priority
from agentic_app.services.batch import run_batch
from agentic_app.services.results_service import ResultsService
from agentic_app.services.validation import validate_readout
from agentic_app.services.vehicle_index import VehicleIndex
from tests.helpers import StubModelService, make_schema, make_settings, valid_payload

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "config" / "calibration"
RESULTS = ROOT / "config" / "results"
RAW_5709 = [0.362, 0.1802, 0.1581, 0.121, 0.1786]   # raw output the real model gave for a healthy vehicle (printed on the project owner's PC)
RAW_HIGH = [0.05, 0.05, 0.10, 0.20, 0.60]


def load_cal():
    return Calibrator.load(CAL / "m2_calibrator.json", CAL / "validation_risk_reference.json", RESULTS)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.settings = make_settings(self.tmp)
        self.cal = load_cal()

    def run_agents(self, raw, payload=None, calibrator=True):
        cal = self.cal if calibrator else None
        readout = validate_readout(payload or valid_payload(), make_schema(), self.settings)
        pred = PredictionAgent(StubModelService(raw), cal).run(readout)
        inv = InvestigationAgent(self.settings).run(readout, pred)
        rec = DecisionAgent(self.settings).run(readout, pred, inv)
        ver = VerificationAgent(self.settings, cal).run(readout, pred, inv, rec)
        return readout, pred, inv, rec, ver


class CalibrationMathTests(Base):
    def test_rows_sum_to_one_and_ranking_is_preserved(self):
        rng = np.random.default_rng(1)
        raw = rng.dirichlet(np.ones(5), size=200)
        q = self.cal.apply(raw)
        self.assertTrue(np.allclose(q.sum(axis=1), 1.0))
        order_raw, order_cal = np.argsort(1 - raw[:, 0]), np.argsort(1 - q[:, 0])
        self.assertTrue(np.array_equal(order_raw, order_cal))      # calibration never reorders vehicles
        self.assertTrue(np.allclose(self.cal.apply(raw[0]), q[0]))  # single vs batch agree

    def test_reproduces_the_project_owners_own_m2_run(self):
        """The decision counts printed by experiments/m2_calibrated_cost.py on the owner's PC: [2877, 0, 0, 1, 2168]."""
        f = ROOT / "outputs" / "graphs" / "validation_predictions.csv"
        if not f.is_file():
            self.skipTest("validation_predictions.csv not present")
        P = pd.read_csv(f)[[f"p{i}" for i in range(5)]].to_numpy()
        decisions = (self.cal.apply(P) @ COST_MATRIX).argmin(axis=1)
        self.assertEqual(np.bincount(decisions, minlength=5).tolist(), [2877, 0, 0, 1, 2168])

    def test_risk_context_uses_real_validation_reference(self):
        ctx = self.cal.risk_context(0.80)           # higher than any typical validation score
        self.assertGreater(ctx["percentile"], 95)
        self.assertEqual(ctx["group"], 5)
        self.assertEqual(ctx["group_failures"] + 0, 61)
        self.assertEqual(self.cal.risk_context(0.10)["group"], 1)

    def test_test_evidence_comes_from_the_saved_test_file(self):
        ev = self.cal.evidence
        self.assertEqual(ev["test_vehicles"], 5045)
        self.assertAlmostEqual(ev["test_failure_recall_any"], 69 / 142)
        self.assertEqual(ev["test_classes_1_to_3_correct"], 0)


class CalibratedPriorityTests(Base):
    def test_bands(self):
        s = self.settings
        self.assertEqual(compute_priority([0.99, 0.004, 0.003, 0.002, 0.001], s, calibrated=True).priority, "low")
        self.assertEqual(compute_priority([0.96, 0.008, 0.008, 0.012, 0.012], s, calibrated=True).priority, "medium")
        self.assertEqual(compute_priority([0.93, 0.01, 0.01, 0.02, 0.03], s, calibrated=True).priority, "high")
        self.assertEqual(compute_priority([0.88, 0.02, 0.02, 0.03, 0.05], s, calibrated=True).priority, "critical")

    def test_raw_rule_unchanged_when_not_calibrated(self):
        self.assertEqual(compute_priority([0.25, 0.1, 0.1, 0.2, 0.35], self.settings).priority, "critical")


class AgentReportTests(Base):
    def test_every_agent_explains_itself(self):
        _, pred, inv, rec, ver = self.run_agents(RAW_5709)
        for result in (pred, inv, rec, ver):
            rep = result.report
            self.assertIsNotNone(rep)
            self.assertTrue(rep.received and rep.performed and rep.findings and rep.passed_on and rep.limitations)
            self.assertTrue(all(f.title and f.detail for f in rep.findings))
            self.assertNotIn("Completed", " ".join(f.detail for f in rep.findings))
            json.dumps(result.to_dict())

    def test_prediction_states_raw_vs_calibrated(self):
        _, pred, *_ = self.run_agents(RAW_5709)
        self.assertTrue(pred.calibration_applied)
        self.assertAlmostEqual(pred.probability_any_failure, 0.638, places=3)             # raw
        self.assertLess(pred.calibrated_probability_any_failure, 0.10)                      # realistic
        self.assertAlmostEqual(sum(pred.decision_probabilities.values()), 1.0)
        self.assertIn("not real-world probabilities", pred.report.findings[0].detail)

    def test_decision_breakdown_adds_up_to_expected_cost(self):
        _, _, _, rec, _ = self.run_agents(RAW_HIGH)
        for j in range(5):
            self.assertAlmostEqual(sum(rec.cost_breakdown[str(j)].values()), rec.expected_costs[str(j)], places=9)
        self.assertEqual(rec.recommended_class, min(range(5), key=lambda j: rec.expected_costs[str(j)]))
        self.assertIn("raw", rec.explanation.lower())
        self.assertEqual(rec.decision_basis, "calibrated probabilities")

    def test_raw_cost_rule_would_have_chosen_class_4_but_calibrated_does_not(self):
        _, _, _, rec, _ = self.run_agents(RAW_5709)
        self.assertEqual(rec.raw_min_cost_class, 4)
        self.assertEqual(rec.recommended_class, 0)
        self.assertEqual(rec.priority, "low")

    def test_hypotheses_stay_unverified_and_no_component_is_named(self):
        _, _, inv, *_ = self.run_agents(RAW_HIGH)
        self.assertTrue(all(h.is_hypothesis and h.status == "unverified" for h in inv.hypotheses))
        text = " ".join(f.detail for f in inv.report.findings).lower()
        self.assertNotIn("bearing", text)
        self.assertIn("anonymised", " ".join(inv.report.limitations).lower())
        self.assertNotIn("above 100%", text)


class VerificationTests(Base):
    def test_all_checks_pass_in_normal_operation(self):
        *_, ver = self.run_agents(RAW_HIGH)
        self.assertEqual(ver.status, "verified")
        ids = {c.id for c in ver.checks}
        self.assertTrue({"V11", "V12", "V13", "V14"} <= ids)
        self.assertTrue(all(c.what_checked and c.why_it_matters for c in ver.checks))

    def test_assurance_keeps_four_meanings_apart(self):
        *_, ver = self.run_agents(RAW_HIGH)
        a = ver.assurance
        self.assertEqual(set(a), {"workflow_execution", "software_checks", "model_correctness", "physical_equipment_safety"})
        self.assertEqual(a["software_checks"]["status"], "passed")
        self.assertEqual(a["model_correctness"]["status"], "not_verified_for_this_vehicle")
        self.assertEqual(a["physical_equipment_safety"]["status"], "not_assessed")

    def test_tampered_calibration_is_caught(self):
        readout, pred, inv, rec, _ = self.run_agents(RAW_HIGH)
        pred.decision_probabilities = {k: (0.2 if True else v) for k, v in pred.decision_probabilities.items()}
        ver = VerificationAgent(self.settings, self.cal).run(readout, pred, inv, rec)
        self.assertNotEqual(ver.status, "verified")
        self.assertFalse(next(c for c in ver.checks if c.id == "V11").passed)

    def test_priority_inconsistent_with_population_needs_human_review(self):
        readout, pred, inv, rec, _ = self.run_agents(RAW_HIGH)
        self.assertIn(rec.priority, ("high", "critical"))
        pred.risk_context = dict(pred.risk_context, percentile=5.0)
        ver = VerificationAgent(self.settings, self.cal).run(readout, pred, inv, rec)
        self.assertEqual(ver.status, "needs_review")
        self.assertFalse(next(c for c in ver.checks if c.id == "V13").passed)

    def test_orchestrator_end_to_end_with_calibration(self):
        orch = Orchestrator(self.settings, StubModelService(RAW_HIGH), calibrator=self.cal)
        res = orch.run_workflow(valid_payload())
        self.assertEqual((res.status, res.final_status, res.replan_count), ("completed", "verified", 0))
        self.assertIsNotNone(res.prediction.report)
        self.assertEqual(res.verification.assurance["software_checks"]["status"], "passed")
        json.dumps(res.to_dict())


class IndexAndResultsTests(Base):
    def make_index(self):
        """SYNTHETIC vehicles, used only to test the lookup code."""
        d = self.tmp / "index"
        d.mkdir()
        rng = np.random.default_rng(3)
        raw = rng.dirichlet([3, 1, 1, 1, 2], size=40)
        ids = [str(1000 + i) for i in range(40)]
        scored = pd.DataFrame({"vehicle_id": ids, "time_step": 10.0, **{f"p{i}": raw[:, i] for i in range(5)}, "class_label": rng.integers(0, 5, 40)})
        scored.to_csv(d / "validation_scored.csv", index=False)
        veh = pd.DataFrame({"vehicle_id": ids, "time_step": 10.0, "171_0": 1.0, "666_0": 2.0, "309_0": np.nan, "272_1": 4.0, "Spec_0": "Cat0", "Spec_1": "Cat2"})
        veh.to_csv(d / "validation_vehicles.csv.gz", index=False, compression="gzip")
        return VehicleIndex(d, self.settings, self.cal), raw

    def test_search_hides_labels_and_sorts_by_risk(self):
        idx, _ = self.make_index()
        r = idx.search("validation", page_size=10)
        self.assertEqual(r["total"], 40)
        self.assertNotIn("class_label", r["vehicles"][0])
        risks = [v["calibrated_p_any_failure"] for v in r["vehicles"]]
        self.assertEqual(risks, sorted(risks, reverse=True))
        self.assertEqual(idx.search("validation", q="1003")["total"], 1)

    def test_payload_is_a_valid_workflow_input(self):
        idx, _ = self.make_index()
        payload = idx.payload("validation", "1005")
        self.assertIsNone(payload["features"]["309_0"])             # NaN becomes null, which the validator accepts
        validate_readout(payload, make_schema(), self.settings)
        self.assertIsNone(idx.payload("validation", "nope"))

    def test_analytics_uses_labels_only_for_scoring(self):
        idx, raw = self.make_index()
        a = idx.analytics("validation")
        self.assertTrue(a["labelled"])
        self.assertEqual(sum(a["actual_class_counts_0_to_4"]), 40)
        names = [p["policy"] for p in a["policies"]]
        self.assertEqual(names, ["always_0", "always_4", "m0_raw_argmax", "m0_cost_rule", "m2_calibrated"])
        self.assertEqual(sum(a["priority_counts"].values()), 40)
        self.assertIn("In-sample", a["evaluation_basis"])

    def test_batch_run_agrees_with_dashboard_for_the_same_vehicle(self):
        idx, raw = self.make_index()
        selected_row = idx.scored("validation").sample(n=1, random_state=2026).index[0]
        first = raw[selected_row].tolist()                            # match the deterministic sampled vehicle
        orch = Orchestrator(self.settings, StubModelService(first), calibrator=self.cal)
        summary = run_batch(orch, idx, "validation", limit=1)
        self.assertEqual(summary["vehicles_completed"], 1)
        self.assertEqual(summary["dashboard_workflow_disagreements"], 0)

    def test_limited_batch_uses_reproducible_random_sample(self):
        idx, raw = self.make_index()
        # Use a constant stub so the sampled vehicle identity does not affect this test.
        orch = Orchestrator(self.settings, StubModelService(raw[0].tolist()), calibrator=self.cal)
        first = run_batch(orch, idx, "validation", limit=12, seed=77)
        second = run_batch(orch, idx, "validation", limit=12, seed=77)
        self.assertEqual(first["vehicles_run"], 12)
        self.assertEqual(first["sampling"]["method"], "deterministic_random_without_replacement")
        self.assertEqual(first["sampling"]["seed"], 77)
        self.assertEqual(first["priority_counts"], second["priority_counts"])
        self.assertEqual(first["selected_vehicle_ids"], second["selected_vehicle_ids"])
        expected_ids = idx.scored("validation").sample(n=12, random_state=77)["vehicle_id"].tolist()
        self.assertEqual(first["selected_vehicle_ids"], expected_ids)

    def test_batch_rejects_nonpositive_limit(self):
        idx, raw = self.make_index()
        orch = Orchestrator(self.settings, StubModelService(raw[0].tolist()), calibrator=self.cal)
        with self.assertRaises(ValueError):
            run_batch(orch, idx, "validation", limit=0)

    def test_model_comparison_is_derived_from_saved_results(self):
        r = ResultsService(RESULTS).model_comparison()
        test_rows = {x["key"]: x for x in r["rows"] if x["split"] == "test"}
        self.assertAlmostEqual(test_rows["m2_calibrated"]["flagged_share"], 1 - 3497 / 5045)
        self.assertLess(test_rows["m2_calibrated"]["mean_cost"], test_rows["always_4"]["mean_cost"])
        self.assertAlmostEqual(test_rows["m2_calibrated"]["failure_recall_any"], 69 / 142)
        self.assertTrue(r["honest_summary"])


if __name__ == "__main__":
    unittest.main()
