from pathlib import Path
import pandas as pd
import pytest
from agentic_app.services.dataset_service import DatasetService


def test_vehicle_history_and_specs(tmp_path: Path):
    pd.DataFrame([{"vehicle_id": 7, "Spec_0": "A"}]).to_csv(tmp_path / "validation_specifications.csv", index=False)
    pd.DataFrame([{"vehicle_id": 7, "time_step": 1, "reading": 10}, {"vehicle_id": 7, "time_step": 2, "reading": 11}, {"vehicle_id": 8, "time_step": 1, "reading": 12}]).to_csv(tmp_path / "validation_operational_readouts.csv", index=False)
    result = DatasetService(tmp_path, chunksize=2).vehicle("validation", 7)
    assert result["specifications"]["Spec_0"] == "A"
    assert [r["time_step"] for r in result["operational_history"]] == [1, 2]


def test_records_pagination(tmp_path: Path):
    pd.DataFrame([{"vehicle_id": i, "time_step": 1} for i in range(8)]).to_csv(tmp_path / "test_operational_readouts.csv", index=False)
    result = DatasetService(tmp_path, chunksize=3).records("test", page=2, page_size=3)
    assert [r["vehicle_id"] for r in result["items"]] == [3, 4, 5]


def test_rejects_bad_page(tmp_path: Path):
    with pytest.raises(ValueError):
        DatasetService(tmp_path).records("test", page=0)


def test_records_converts_non_finite_values_to_json_null(tmp_path: Path):
    import json
    import math
    import numpy as np

    pd.DataFrame([
        {"vehicle_id": 1, "time_step": 1, "missing": float("nan"), "positive_inf": float("inf"), "negative_inf": float("-inf")},
        {"vehicle_id": 2, "time_step": 2, "missing": 3.0, "positive_inf": 4.0, "negative_inf": 5.0},
    ]).to_csv(tmp_path / "validation_operational_readouts.csv", index=False)
    result = DatasetService(tmp_path).records("validation", page=1, page_size=10)
    encoded = json.dumps(result, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["items"][0]["missing"] is None
    assert decoded["items"][0]["positive_inf"] is None
    assert decoded["items"][0]["negative_inf"] is None


def test_recursive_json_sanitizer_handles_nested_numpy_and_pandas_values():
    import json
    import numpy as np
    from agentic_app.services.dataset_service import to_json_safe

    value = {"nested": [float("nan"), float("inf"), np.float64("-inf"), np.int64(7), pd.NA]}
    safe = to_json_safe(value)
    assert safe == {"nested": [None, None, None, 7, None]}
    json.dumps(safe, allow_nan=False)


def test_vehicle_history_is_json_safe_with_missing_readings(tmp_path: Path):
    import json

    pd.DataFrame([{"vehicle_id": 7, "Spec_0": pd.NA}]).to_csv(tmp_path / "validation_specifications.csv", index=False)
    pd.DataFrame([{"vehicle_id": 7, "time_step": 1, "reading": float("nan")}]).to_csv(tmp_path / "validation_operational_readouts.csv", index=False)
    result = DatasetService(tmp_path).vehicle("validation", 7)
    json.dumps(result, allow_nan=False)
    assert result["specifications"]["Spec_0"] is None
    assert result["operational_history"][0]["reading"] is None
