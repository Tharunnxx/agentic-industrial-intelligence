"""Evaluate the SAVED model and the decision rules on your validation files (read-only for your data).

Protocol identical to your evaluate_official_validation.py: latest readout per vehicle + specification row + label.
Policies compared: always_0, always_4, raw_argmax, expected_cost (official matrix, rows = actual).
This script computes metrics once. It does not tune thresholds, weights or the model; do not iterate on it
against the same labels. Output goes to agentic_platform/outputs/ (never to your existing result folders).

Usage (from agentic_platform/):  python scripts/evaluate_saved_model.py [--model balanced_baseline_model.joblib]
"""
import _bootstrap  # noqa: F401
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_app.core.config import Settings
from agentic_app.services.metrics import POLICIES, evaluate_policy, policy_predictions
from agentic_app.services.model_service import ModelService


def make_plots(out: Path, results: dict, probs_cache: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, [results[n]["mean_cost"] for n in names], color="#3b6ea5")
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel("Mean cost per vehicle (lower is better)")
    ax.set_title("Mean cost by decision policy (validation set, evaluation only)")
    fig.tight_layout()
    fig.savefig(out / "mean_cost_by_policy.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name in zip(axes, ("raw_argmax", "expected_cost")):
        cm = np.array(results[name]["confusion_matrix_rows_actual_cols_predicted"])
        shown = np.log10(cm + 1)
        ax.imshow(shown, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="black" if shown[i, j] < shown.max() * 0.6 else "white")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class"); ax.set_title(f"{name} (colour = log10 count)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.2
    for k, name in enumerate(names):
        ax.bar(np.arange(5) + k * width, results[name]["prediction_counts_0_to_4"], width, label=name)
    ax.set_xticks(np.arange(5) + 1.5 * width); ax.set_xticklabels([str(c) for c in range(5)])
    ax.set_yscale("symlog"); ax.set_xlabel("Predicted class"); ax.set_ylabel("Vehicles (symlog)")
    ax.legend(); ax.set_title("Prediction counts per class")
    fig.tight_layout()
    fig.savefig(out / "prediction_counts.png", dpi=150)
    plt.close(fig)


def main() -> int:
    settings = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model file (default: AGENTIC_MODEL_PATH)")
    ap.add_argument("--data-dir", default=str(settings.data_dir))
    ap.add_argument("--out-root", default=str(settings.platform_root / "outputs"))
    args = ap.parse_args()
    if args.model:
        p = Path(args.model)
        settings = dataclasses.replace(settings, model_path=p if p.is_absolute() else settings.data_dir / p)
    data_dir = Path(args.data_dir)

    ms = ModelService(settings)
    if not ms.ensure_loaded():
        print(f"[FAIL] {ms.error.code}: {ms.error}")
        return 1
    schema = ms.schema()
    paths = {k: data_dir / f"validation_{k}.csv" for k in ("operational_readouts", "specifications", "labels")}
    for p in paths.values():
        if not p.is_file():
            print(f"[FAIL] Missing {p}")
            return 1

    ops_header = pd.read_csv(paths["operational_readouts"], nrows=0).columns.tolist()
    spec_header = pd.read_csv(paths["specifications"], nrows=0).columns.tolist()
    want_ops = [c for c in schema.feature_cols if c in ops_header]
    want_specs = [c for c in schema.feature_cols if c in spec_header and c not in ops_header]
    unresolved = [c for c in schema.feature_cols if c not in want_ops and c not in want_specs]
    if unresolved:
        print(f"[FAIL] Model features not found in the validation CSVs: {unresolved[:10]}")
        return 1

    print("Reading validation data ...")
    ops = pd.read_csv(paths["operational_readouts"], usecols=sorted(set(["vehicle_id", "time_step"] + want_ops)))
    latest = ops.sort_values("time_step").drop_duplicates("vehicle_id", keep="last")
    specs = pd.read_csv(paths["specifications"], usecols=["vehicle_id"] + want_specs)
    labels = pd.read_csv(paths["labels"])
    data = latest.merge(specs, on="vehicle_id", how="left", validate="one_to_one").merge(labels, on="vehicle_id", how="inner", validate="one_to_one")
    if data["class_label"].isna().any():
        print("[FAIL] Some labels are missing.")
        return 1
    y = data["class_label"].astype(int).to_numpy()
    print(f"Vehicles: {len(y)}; class counts: {np.bincount(y, minlength=5).tolist()}")

    probs = ms.predict_proba_frame(data)
    results = {name: evaluate_policy(name, y, policy_predictions(name, probs), probs) for name in POLICIES}

    out = Path(args.out_root) / f"evaluation_{settings.model_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    payload = {
        "label": "OFFICIAL VALIDATION SET evaluation of a saved model; computed once, no tuning. Not an internal simulation and not a test-set score.",
        "model_file": settings.model_path.name,
        "protocol": "latest readout per vehicle + specification row (same as evaluate_official_validation.py)",
        "n_vehicles": int(len(y)),
        "class_counts_0_to_4": np.bincount(y, minlength=5).tolist(),
        "cost_matrix_orientation": "rows = actual, columns = predicted",
        "calibration_wrapper_detected": ms.is_calibrated_wrapper(),
        "policies": results,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k in ("policy", "n_vehicles", "total_cost", "mean_cost", "balanced_accuracy", "macro_f1", "any_failure_auc")}
                  for r in results.values()]).to_csv(out / "policy_summary.csv", index=False)
    for name, r in results.items():
        pd.DataFrame(r["per_class"]).T.rename_axis("class").to_csv(out / f"per_class_{name}.csv")
        pd.DataFrame(r["confusion_matrix_rows_actual_cols_predicted"]).to_csv(out / f"confusion_{name}.csv", index_label="actual", header=[f"pred_{i}" for i in range(5)])
    pred_df = data[["vehicle_id", "time_step", "class_label"]].copy()
    for i in range(5):
        pred_df[f"p{i}"] = probs[:, i]
    for name in POLICIES:
        pred_df[f"pred_{name}"] = policy_predictions(name, probs)
    pred_df.to_csv(out / "predictions.csv", index=False)
    make_plots(out, results, {})

    print("\nPolicy               total_cost   mean_cost  bal_acc  macro_f1   AUC(any failure)")
    for name, r in results.items():
        auc = "n/a" if r["any_failure_auc"] is None else f"{r['any_failure_auc']:.4f}"
        print(f"{name:20s} {r['total_cost']:10.0f} {r['mean_cost']:11.4f} {r['balanced_accuracy']:8.4f} {r['macro_f1']:9.4f}   {auc}")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
