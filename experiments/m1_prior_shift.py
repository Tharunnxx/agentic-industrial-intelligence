"""Experiment M1 - prior-shift correction of the ORIGINAL saved model (no retraining, model file untouched).

WHY
  train_cost_sensitive.py trained with class_weight="balanced_subsample", so the forest behaves as if every class
  had prior 0.20. Real vehicles are ~97% class 0, so the probabilities overstate failure. Bayes' rule fixes a prior
  mismatch without retraining:   q_i  proportional to  p_i * (target_prior_i / 0.20)

PROTOCOL (declared BEFORE looking at any result)
  * Nothing is fitted or tuned on labels. The only inputs are two FIXED prior vectors, written below with their source.
  * Both priors are reported. Do NOT pick a prior by looking at validation cost and then quote validation as proof:
      - run on --split validation to compare,
      - then run ONCE on --split test for the number you quote.
  * Labels are read only after probabilities exist, and only for scoring.

USAGE (from agentic_platform\\):
  python experiments\\m1_prior_shift.py --split validation
  python experiments\\m1_prior_shift.py --split test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_app.core.config import Settings  # noqa: E402
from agentic_app.core.costs import COST_MATRIX  # noqa: E402

CLASSES = [0, 1, 2, 3, 4]
# Source: training_label_report.txt (all 1,122,452 TRAINING readouts). Uses no validation/test information.
PRIOR_TRAIN_READOUTS = np.array([1096712, 12503, 6179, 3200, 3858], dtype=float)
# Source: official validation label COUNTS (5,046 vehicles, one latest readout each). Five numbers only.
PRIOR_VALIDATION_COUNTS = np.array([4910, 16, 14, 30, 76], dtype=float)
EFFECTIVE_TRAINING_PRIOR = 0.20  # balanced_subsample => each class weighted to equal total weight (assumption, stated)


def load_probabilities(split: str, cache: Path, settings: Settings) -> pd.DataFrame:
    """Return vehicle_id + p0..p4 for the latest readout per vehicle (same protocol as EvaluationService)."""
    if cache.is_file():
        print(f"Using cached probabilities: {cache}")
        return pd.read_csv(cache, dtype={"vehicle_id": str})
    from agentic_app.services.model_service import ModelService
    ms = ModelService(settings)
    if not ms.ensure_loaded():
        raise SystemExit(f"Model unavailable: {ms.error}")
    schema = ms.schema()
    d = settings.data_dir
    ops_p, spec_p = d / f"{split}_operational_readouts.csv", d / f"{split}_specifications.csv"
    ops_h = pd.read_csv(ops_p, nrows=0).columns.tolist()
    spec_h = pd.read_csv(spec_p, nrows=0).columns.tolist()
    ops_f = [c for c in schema.feature_cols if c in ops_h]
    spec_f = [c for c in schema.feature_cols if c in spec_h and c not in ops_h]
    print(f"Reading {ops_p.name} ...")
    ops = pd.read_csv(ops_p, usecols=list(dict.fromkeys(["vehicle_id", "time_step"] + ops_f)), dtype={"vehicle_id": "string"})
    latest = ops.sort_values("time_step", kind="mergesort").drop_duplicates("vehicle_id", keep="last")
    specs = pd.read_csv(spec_p, usecols=list(dict.fromkeys(["vehicle_id"] + spec_f)), dtype={"vehicle_id": "string"})
    feats = latest.merge(specs, on="vehicle_id", how="left", validate="one_to_one")
    print(f"Running the saved model on {len(feats)} vehicles ...")
    probs = ms.predict_proba_frame(feats)
    out = pd.DataFrame({"vehicle_id": feats["vehicle_id"].astype(str)})
    for i in CLASSES:
        out[f"p{i}"] = probs[:, i]
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def shift(P: np.ndarray, prior: np.ndarray) -> np.ndarray:
    prior = prior / prior.sum()
    Q = P * (prior / EFFECTIVE_TRAINING_PRIOR)
    return Q / Q.sum(axis=1, keepdims=True)


def score(name: str, y: np.ndarray, pred: np.ndarray, P: np.ndarray) -> dict:
    cost = COST_MATRIX[y, pred]
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    return {
        "name": name, "mean_cost": float(cost.mean()), "total_cost": float(cost.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "prediction_counts_0_to_4": np.bincount(pred, minlength=5).tolist(),
        "max_single_class_share": float(np.bincount(pred, minlength=5).max() / len(pred)),
        "per_class_recall": {str(c): float(rc[i]) for i, c in enumerate(CLASSES)},
        "per_class_precision": {str(c): float(pr[i]) for i, c in enumerate(CLASSES)},
        "confusion_matrix_rows_actual_cols_predicted": confusion_matrix(y, pred, labels=CLASSES).tolist(),
    }


def reliability(y: np.ndarray, P: np.ndarray, bins: int = 5) -> list:
    """Observed failure rate vs mean predicted P(any failure), in equal-size groups of vehicles."""
    pany = 1 - P[:, 0]
    order = np.argsort(pany)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append({"n": int(len(chunk)), "mean_predicted_p_any_failure": float(pany[chunk].mean()),
                     "observed_failure_rate": float((y[chunk] > 0).mean())})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--labels-path", help="override labels CSV (testing only)")
    args = ap.parse_args()
    settings = Settings.from_env()
    cache = ROOT / "experiments" / "cache" / f"original_model_probs_{args.split}.csv"
    probs_df = load_probabilities(args.split, cache, settings)

    labels_path = Path(args.labels_path) if args.labels_path else settings.data_dir / f"{args.split}_labels.csv"
    labels = pd.read_csv(labels_path, usecols=["vehicle_id", "class_label"], dtype={"vehicle_id": str})
    data = probs_df.merge(labels, on="vehicle_id", how="inner", validate="one_to_one")
    if len(data) != len(probs_df):
        raise SystemExit(f"Label join lost vehicles: {len(probs_df)} -> {len(data)}")
    y = data["class_label"].astype(int).to_numpy()
    P = data[[f"p{i}" for i in CLASSES]].to_numpy()
    print(f"Split: {args.split} | vehicles: {len(y)} | class counts: {np.bincount(y, minlength=5).tolist()}")

    variants = {
        "M0 original, cost rule": P,
        "M1a prior=train readouts, cost rule": shift(P, PRIOR_TRAIN_READOUTS),
        "M1b prior=validation counts, cost rule": shift(P, PRIOR_VALIDATION_COUNTS),
    }
    results = [score("always_0", y, np.zeros(len(y), int), P), score("always_4", y, np.full(len(y), 4, int), P),
               score("M0 original, raw argmax", y, P.argmax(1), P)]
    calib = {}
    for name, Q in variants.items():
        results.append(score(name, y, (Q @ COST_MATRIX).argmin(1), Q))
        results.append(score(name.replace("cost rule", "argmax"), y, Q.argmax(1), Q))
        calib[name.split(",")[0]] = reliability(y, Q)
    auc = float(roc_auc_score((y > 0).astype(int), 1 - P[:, 0]))

    print(f"\nAny-failure AUC (unchanged by any prior shift, it is a monotone rescaling): {auc:.4f}")
    print(f"{'policy':44s} {'mean_cost':>9s} {'bal_acc':>8s} {'macroF1':>8s} {'max_share':>9s}  predicted counts 0..4")
    for r in results:
        print(f"{r['name']:44s} {r['mean_cost']:9.4f} {r['balanced_accuracy']:8.4f} {r['macro_f1']:8.4f} "
              f"{r['max_single_class_share']:9.3f}  {r['prediction_counts_0_to_4']}")
    print("\nReliability (equal-size groups of vehicles, lowest to highest predicted risk):")
    for k, rows in calib.items():
        print(f"  {k}")
        for g in rows:
            print(f"    n={g['n']:5d}  predicted P(any failure)={g['mean_predicted_p_any_failure']:.4f}  observed failure rate={g['observed_failure_rate']:.4f}")

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"m1_prior_shift_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "split": args.split, "n_vehicles": int(len(y)), "any_failure_auc": auc,
        "priors": {"train_readouts": (PRIOR_TRAIN_READOUTS / PRIOR_TRAIN_READOUTS.sum()).tolist(),
                   "validation_counts": (PRIOR_VALIDATION_COUNTS / PRIOR_VALIDATION_COUNTS.sum()).tolist(),
                   "effective_training_prior_assumed": EFFECTIVE_TRAINING_PRIOR},
        "results": results, "reliability": calib,
        "notes": ["No parameter was fitted on labels. The original model file is not modified.",
                  "Prior shift cannot change ranking quality (AUC); it only changes calibration and the cost-rule decisions."],
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
