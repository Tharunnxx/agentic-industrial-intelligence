import tempfile
import unittest
from pathlib import Path

from agentic_app.services.validation import InputValidationError, validate_readout
from tests.helpers import NUMERIC, make_schema, make_settings, valid_payload


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.settings = make_settings(self.tmp)
        self.schema = make_schema()

    def codes(self, payload):
        with self.assertRaises(InputValidationError) as ctx:
            validate_readout(payload, self.schema, self.settings)
        return {e["code"]: e for e in ctx.exception.errors}

    def test_valid_payload(self):
        r = validate_readout(valid_payload(), self.schema, self.settings)
        self.assertEqual(r.vehicle_id, "101")
        self.assertEqual(list(r.features), self.schema.feature_cols)
        self.assertEqual(r.missing_fraction, 0.0)

    def test_missing_feature_rejected(self):
        p = valid_payload()
        del p["features"][NUMERIC[0]]
        self.assertIn(NUMERIC[0], self.codes(p)["MISSING_FIELD"]["fields"])

    def test_extra_feature_rejected_and_target_flagged(self):
        p = valid_payload()
        p["features"]["class_label"] = 3
        err = self.codes(p)["EXTRA_FIELD"]
        self.assertIn("class_label", err["fields"])
        self.assertIn("target label", err["message"])

    def test_extra_top_level_field_rejected(self):
        self.assertIn("EXTRA_FIELD", self.codes(valid_payload(class_label=2)))

    def test_wrong_types_rejected(self):
        p = valid_payload()
        p["features"][NUMERIC[0]] = "12.5"
        p["features"][NUMERIC[1]] = True
        self.assertEqual(set(self.codes(p)["INVALID_TYPE"]["fields"]), {NUMERIC[0], NUMERIC[1]})

    def test_infinite_rejected(self):
        p = valid_payload()
        p["features"][NUMERIC[0]] = float("inf")
        self.assertIn("INVALID_VALUE", self.codes(p))

    def test_null_numeric_allowed_and_counted(self):
        p = valid_payload()
        p["features"][NUMERIC[0]] = None
        r = validate_readout(p, self.schema, self.settings)
        self.assertEqual(r.missing_numeric, [NUMERIC[0]])
        self.assertAlmostEqual(r.missing_fraction, 1 / len(NUMERIC))

    def test_null_spec_rejected(self):
        p = valid_payload()
        p["features"]["Spec_0"] = None
        self.assertIn("MISSING_VALUE", self.codes(p))

    def test_mostly_empty_readout_rejected(self):
        p = valid_payload()
        for c in NUMERIC:
            p["features"][c] = None
        self.assertIn("INSUFFICIENT_DATA", self.codes(p))

    def test_unseen_spec_category_is_warning_not_error(self):
        p = valid_payload()
        p["features"]["Spec_0"] = "Cat99"
        r = validate_readout(p, self.schema, self.settings)
        self.assertEqual(r.unseen_spec_values, {"Spec_0": "Cat99"})
        self.assertTrue(r.warnings)

    def test_non_object_payloads(self):
        for bad in (None, [], "x", 5):
            self.assertIn("INVALID_PAYLOAD", self.codes(bad))
        self.assertIn("INVALID_PAYLOAD", self.codes({"features": [1, 2]}))

    def test_bad_identifiers(self):
        self.assertIn("INVALID_TYPE", self.codes(valid_payload(vehicle_id=[1])))
        self.assertIn("INVALID_TYPE", self.codes(valid_payload(time_step="soon")))


if __name__ == "__main__":
    unittest.main()
