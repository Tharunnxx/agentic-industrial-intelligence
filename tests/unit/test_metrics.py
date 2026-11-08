import unittest

import numpy as np

from agentic_app.services.metrics import evaluate_policy, policy_predictions


class MetricsTests(unittest.TestCase):
    def test_policy_costs_match_hand_calculation(self):
        y = np.array([0, 0, 4, 2])
        probs = np.eye(5)[[0, 4, 4, 2]]
        m = evaluate_policy("raw_argmax", y, policy_predictions("raw_argmax", probs), probs)
        # costs: (0,0)=0, (0,4)=10, (4,4)=0, (2,2)=0
        self.assertEqual(m["total_cost"], 10.0)
        self.assertAlmostEqual(m["mean_cost"], 2.5)
        self.assertEqual(np.array(m["confusion_matrix_rows_actual_cols_predicted"]).sum(), 4)

    def test_always_policies(self):
        y = np.array([0, 1, 4])
        probs = np.full((3, 5), 0.2)
        a0 = evaluate_policy("always_0", y, policy_predictions("always_0", probs), probs)
        a4 = evaluate_policy("always_4", y, policy_predictions("always_4", probs), probs)
        self.assertEqual(a0["total_cost"], 0 + 200 + 500)
        self.assertEqual(a4["total_cost"], 10 + 9 + 0)

    def test_expected_cost_policy_uses_matrix(self):
        probs = np.array([[0.9, 0, 0, 0, 0.1]])
        self.assertEqual(policy_predictions("expected_cost", probs).tolist(), [4])

    def test_any_failure_auc_and_per_class(self):
        y = np.array([0, 0, 1, 4])
        probs = np.array([[.9, .1, 0, 0, 0], [.8, .1, .1, 0, 0], [.2, .6, .1, .1, 0], [.1, 0, 0, .1, .8]])
        m = evaluate_policy("raw_argmax", y, policy_predictions("raw_argmax", probs), probs)
        self.assertEqual(m["any_failure_auc"], 1.0)
        self.assertEqual(set(m["per_class"]), {"0", "1", "2", "3", "4"})
        self.assertIn("macro_f1", m)
        self.assertIn("balanced_accuracy", m)

    def test_auc_none_when_single_class(self):
        y = np.zeros(3, dtype=int)
        probs = np.tile([1, 0, 0, 0, 0], (3, 1)).astype(float)
        m = evaluate_policy("always_0", y, np.zeros(3, dtype=int), probs)
        self.assertIsNone(m["any_failure_auc"])


if __name__ == "__main__":
    unittest.main()
