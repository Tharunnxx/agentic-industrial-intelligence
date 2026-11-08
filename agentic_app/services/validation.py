"""Strict input validation: missing, extra and wrongly typed fields are rejected, never guessed."""
from __future__ import annotations

import math
import numbers
from typing import Any, Dict, List, Mapping

from ..core.config import Settings
from ..schemas.types import FeatureSchema, ValidatedReadout

TOP_LEVEL_KEYS = {"vehicle_id", "time_step", "features"}
TARGET_LIKE_KEYS = {"class_label", "label", "target"}


class InputValidationError(Exception):
    code = "INPUT_VALIDATION_FAILED"

    def __init__(self, errors: List[Dict[str, Any]]):
        self.errors = errors
        super().__init__("; ".join(e["message"] for e in errors))


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _coerce_spec(value: Any, known: List[Any]):
    """Match the type of the encoder's known categories (str vs numeric) so values are not silently unseen."""
    if known and all(_is_real(k) for k in known):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        return int(number) if number.is_integer() and all(float(k).is_integer() for k in known) else number
    return str(value)


def validate_readout(payload: Any, schema: FeatureSchema, settings: Settings) -> ValidatedReadout:
    errors: List[Dict[str, Any]] = []
    if not isinstance(payload, Mapping):
        raise InputValidationError([{"code": "INVALID_PAYLOAD", "fields": [], "message": "Request body must be a JSON object."}])

    extra_top = sorted(set(payload) - TOP_LEVEL_KEYS)
    if extra_top:
        errors.append({"code": "EXTRA_FIELD", "fields": extra_top,
                       "message": f"Unexpected top-level fields: {extra_top}. Allowed: {sorted(TOP_LEVEL_KEYS)}."})

    vehicle_id = payload.get("vehicle_id")
    if vehicle_id is not None:
        if isinstance(vehicle_id, bool) or not isinstance(vehicle_id, (str, int)):
            errors.append({"code": "INVALID_TYPE", "fields": ["vehicle_id"], "message": "vehicle_id must be a string or integer."})
        else:
            vehicle_id = str(vehicle_id)
    time_step = payload.get("time_step")
    if time_step is not None and (not _is_real(time_step) or not math.isfinite(float(time_step))):
        errors.append({"code": "INVALID_TYPE", "fields": ["time_step"], "message": "time_step must be a finite number."})
        time_step = None

    features = payload.get("features")
    if not isinstance(features, Mapping):
        errors.append({"code": "INVALID_PAYLOAD", "fields": ["features"], "message": "'features' must be a JSON object of feature name -> value."})
        raise InputValidationError(errors)

    expected = list(schema.feature_cols)
    missing = [c for c in expected if c not in features]
    if missing:
        errors.append({"code": "MISSING_FIELD", "fields": missing,
                       "message": f"{len(missing)} required feature(s) missing, e.g. {missing[:8]}. "
                                  "Send an explicit null for numeric features that were not measured."})
    extra = sorted(set(features) - set(expected))
    if extra:
        hint = " (the target label must never be supplied)" if set(extra) & TARGET_LIKE_KEYS else ""
        errors.append({"code": "EXTRA_FIELD", "fields": extra,
                       "message": f"{len(extra)} unexpected feature(s): {extra[:8]}{hint}."})

    clean: Dict[str, Any] = {}
    missing_numeric: List[str] = []
    spec_values: Dict[str, Any] = {}
    unseen: Dict[str, Any] = {}
    warnings: List[str] = []
    bad_type: List[str] = []
    bad_value: List[str] = []
    missing_spec: List[str] = []

    for col in expected:
        if col not in features:
            continue
        value = features[col]
        if col in schema.spec_cols:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                missing_spec.append(col)
            elif isinstance(value, bool) or not isinstance(value, (str, int, float)):
                bad_type.append(col)
            else:
                known = schema.spec_categories.get(col, [])
                coerced = _coerce_spec(value, known)
                clean[col] = coerced
                spec_values[col] = coerced
                if known and coerced not in known:
                    unseen[col] = coerced
        else:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                clean[col] = float("nan")
                missing_numeric.append(col)
            elif not _is_real(value):
                bad_type.append(col)
            elif not math.isfinite(float(value)):
                bad_value.append(col)
            else:
                clean[col] = float(value)

    if bad_type:
        errors.append({"code": "INVALID_TYPE", "fields": bad_type,
                       "message": f"Wrong type for {bad_type[:8]}: numeric features need numbers or null; specification fields need text/number."})
    if bad_value:
        errors.append({"code": "INVALID_VALUE", "fields": bad_value, "message": f"Non-finite numeric values (inf) in {bad_value[:8]}."})
    if missing_spec:
        errors.append({"code": "MISSING_VALUE", "fields": missing_spec, "message": f"Specification fields must not be null: {missing_spec}."})

    n_numeric = len(schema.numeric_cols)
    fraction = (len(missing_numeric) / n_numeric) if n_numeric else 0.0
    if not errors and fraction > settings.max_missing_fraction:
        errors.append({"code": "INSUFFICIENT_DATA", "fields": [],
                       "message": f"{fraction:.1%} of numeric features are null (limit {settings.max_missing_fraction:.0%}); "
                                  "refusing to predict from an almost empty readout."})
    if errors:
        raise InputValidationError(errors)

    if unseen:
        warnings.append(f"Specification value(s) not seen by the model's encoder during training: {unseen}.")
    ordered = {c: clean[c] for c in expected}
    return ValidatedReadout(
        vehicle_id=vehicle_id, time_step=float(time_step) if time_step is not None else None,
        features=ordered, n_numeric=n_numeric, n_spec=len(schema.spec_cols),
        missing_numeric=missing_numeric, missing_fraction=float(fraction),
        spec_values=spec_values, unseen_spec_values=unseen, warnings=warnings,
    )
