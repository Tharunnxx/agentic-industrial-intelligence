import json
from pathlib import Path

from agentic_app.services.results_service import ResultsService


def test_saved_m2_test_detail_is_separate_and_consistent():
    root = Path(__file__).resolve().parents[2]
    service = ResultsService(root / "config" / "results")
    result = service.model_comparison()
    detail = result["extra"]["m2_test_detail"]

    assert detail["n_vehicles"] == 5045
    assert detail["mean_cost"] == 8.637264618434093
    assert detail["baseline_m0_mean_cost"] == 9.739940535183349
    assert round(detail["cost_reduction_vs_m0_percent"], 2) == 11.32
    assert detail["per_class"]["1"]["recall"] == 0.0
    assert detail["per_class"]["2"]["recall"] == 0.0
    assert detail["confusion_matrix_rows_actual_cols_predicted"][0][0] == 3424


def test_model_comparison_labels_live_and_saved_policies_distinctly():
    root = Path(__file__).resolve().parents[2]
    rows = ResultsService(root / "config" / "results").model_comparison()["rows"]
    test_rows = {row["key"]: row for row in rows if row["split"] == "test"}
    assert "M0" in test_rows["m0_cost_rule"]["label"]
    assert "M2" in test_rows["m2_calibrated"]["label"]
    assert test_rows["m0_cost_rule"]["mean_cost"] != test_rows["m2_calibrated"]["mean_cost"]
