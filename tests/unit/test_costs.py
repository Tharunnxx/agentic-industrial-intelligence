import unittest

import numpy as np

from agentic_app.core.costs import COST_MATRIX, check_probabilities, expected_costs, min_cost_class, realised_cost


class CostMatrixTests(unittest.TestCase):
    def test_matrix_matches_official_values(self):
        expected = [[0, 7, 8, 9, 10], [200, 0, 7, 8, 9], [300, 200, 0, 7, 8], [400, 300, 200, 0, 7], [500, 400, 300, 200, 0]]
        self.assertEqual(COST_MATRIX.tolist(), expected)

    def test_orientation_rows_actual_columns_predicted(self):
        # Actual class 4 predicted as 0 is the costliest mistake (500); actual 0 predicted 4 costs only 10.
        self.assertEqual(realised_cost(4, 0), 500)
        self.assertEqual(realised_cost(0, 4), 10)
        self.assertEqual(realised_cost(2, 1), 200)
        self.assertEqual(realised_cost(1, 2), 7)

    def test_expected_cost_is_probability_times_matrix(self):
        # Certain actual class 3 -> expected cost of predicting j equals row 3 of the matrix.
        np.testing.assert_allclose(expected_costs([0, 0, 0, 1, 0]), COST_MATRIX[3])
        # Certain actual class 0 -> column predicted j costs row 0.
        np.testing.assert_allclose(expected_costs([1, 0, 0, 0, 0]), COST_MATRIX[0])

    def test_expected_cost_manual_mixture(self):
        p = [0.9, 0.0, 0.0, 0.0, 0.1]
        ec = expected_costs(p)
        self.assertAlmostEqual(ec[0], 0.1 * 500)           # predict 0
        self.assertAlmostEqual(ec[4], 0.9 * 10)            # predict 4
        self.assertEqual(min_cost_class(p), 4)             # 9 < 50

    def test_min_cost_class_prefers_zero_when_failure_unlikely(self):
        self.assertEqual(min_cost_class([0.995, 0.002, 0.001, 0.001, 0.001]), 0)

    def test_batch_shape(self):
        out = min_cost_class(np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=float))
        self.assertEqual(out.tolist(), [0, 4])  # certain class 0 -> predict 0; certain class 4 -> predict 4
        self.assertEqual(expected_costs(np.eye(5)).shape, (5, 5))

    def test_invalid_probabilities_rejected(self):
        for bad in ([0.5, 0.5, 0, 0], [0.3] * 5, [float("nan"), 0, 0, 0, 1], [-0.1, 1.1, 0, 0, 0]):
            with self.assertRaises(ValueError):
                check_probabilities(bad)

    def test_matrix_is_read_only(self):
        with self.assertRaises(ValueError):
            COST_MATRIX[0, 0] = 1


if __name__ == "__main__":
    unittest.main()
