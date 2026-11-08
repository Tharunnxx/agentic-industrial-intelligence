"""Experiment M2 - calibrate P(any failure) of the ORIGINAL model, then apply the expected-cost rule.
No retraining. The model file is not modified. Needs m1_prior_shift.py in the same folder (it reuses its loader).

IDEA (plain English)
  The model ranks risky vehicles above safe ones (failure rate rises from ~0.4% to ~5-6% across risk groups) but its
  probability SCALE is wrong. We learn a 2-number correction (Platt scaling) that maps the model's risk score to a
  realistic P(any failure). The split of that risk over classes 1-4 then comes either from the model (variant "model")
  or from the fixed validation failure mix 16/14/30/76 (variant "valmix"). The expected-cost rule is then applied.

PROTOCOL (declared BEFORE results; amended once, see note)
  1. --mode cv   : official VALIDATION is used as the calibration set. Honest estimate = repeated 5-fold CV BY VEHICLE
                   (calibrator fitted on 4 folds, scored on the 5th; pooled out-of-fold cost; 10 repeats).
                   The variant with the lower CV mean cost is selected. A calibrator fitted on all validation is saved.
  2. --mode test : the saved calibrator is applied to the TEST split. Run this ONCE. This is the number to quote.
  Acceptance rule (all must hold, on CV for validation and again on test):
     (a) mean cost below always-class-4 (CV: by more than one standard deviation across repeats),
     (b) share of vehicles predicted non-zero (flagged) <= 80%,
     (c) per-class recall is reported, not hidden.
  NOTE (amendment, stated openly): my earlier draft said "no class above ~30% of predictions". With 97% healthy
  vehicles and failures costing 20-50x a false alarm, a sensible policy may legitimately flag many vehicles, so the
  guard is now the 80% flagged-share limit in (b). M1 failed on cost, so this amendment did not affect M1's verdict.
  Limitation: only 136 failure vehicles exist in validation (16/14/30/76), so estimates are noisy; CV sd is reported.

USAGE (from agentic_platform\\):
  python experiments\\m2_calibrated_cost.py --mode cv
  python experiments\\m2_calibrated_cost.py --mode test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_prior_shift import CLASSES, ROOT, load_probabilities, reliability, score  # noqa: E402
from agentic_app.core.config import Settings  # noqa: E402
from agentic_app.core.costs import COST_MATRIX  # noqa: E402

VAL_FAIL_MIX = np.array([16, 14, 30, 76], dtype=float)  # official validation counts of classes 1-4
VAL_FAIL_MIX /= VAL_FAIL_MIX.sum()
EPS = 1e-6
VARIANTS = ("model", "valmix")
CAL_PATH = ROOT / "experiments" / "results" / "m2_calibrator.json"


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(P, y):
    s = logit(1 - P[:, 0]).reshape(-1, 1)
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(s, (y > 0).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def failure_mix_from_labels(y):
    """Estimate the conditional class mix P(class 1..4 | any failure) from supplied labels.

    In cross-validation, callers MUST pass only the training-fold labels. For the final
    calibrator, the full official validation labels are used after variant selection.
    """
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y[y > 0], minlength=5)[1:5].astype(float)
    if counts.sum() <= 0 or np.any(counts <= 0):
        raise ValueError("Each failure class 1..4 must be represented to estimate failure mix.")
    return counts / counts.sum()


def calibrated_probs(P, a, b, severity, severity_mix=None):
    s = logit(1 - P[:, 0])
    c = 1.0 / (1.0 + np.exp(-(a * s + b)))
    if severity == "model":
        r = P[:, 1:] / np.clip(1 - P[:, 0], EPS, None)[:, None]
    else:
        mix = VAL_FAIL_MIX if severity_mix is None else np.asarray(severity_mix, dtype=float)
        mix = mix / mix.sum()
        r = np.tile(mix, (len(P), 1))
    return np.column_stack([1 - c, c[:, None] * r])


def decide(Q):
    return (Q @ COST_MATRIX).argmin(axis=1)


def load_xy(split, settings, labels_path=None):
    cache = ROOT / "experiments" / "cache" / f"original_model_probs_{split}.csv"
    probs = load_probabilities(split, cache, settings)
    lp = Path(labels_path) if labels_path else settings.data_dir / f"{split}_labels.csv"
    labels = pd.read_csv(lp, usecols=["vehicle_id", "class_label"], dtype={"vehicle_id": str})
    data = probs.merge(labels, on="vehicle_id", how="inner", validate="one_to_one")
    if len(data) != len(probs):
        raise SystemExit(f"Label join lost vehicles: {len(probs)} -> {len(data)}")
    return data[[f"p{i}" for i in CLASSES]].to_numpy(), data["class_label"].astype(int).to_numpy()


def baselines(y, P):
    return {"always_0": score("always_0", y, np.zeros(len(y), int), P),
            "always_4": score("always_4", y, np.full(len(y), 4, int), P),
            "M0_original_cost_rule": score("M0 original, cost rule", y, decide(P), P)}


def cv_run(P, y, repeats=10, folds=5, seed=2026):
    out = {v: [] for v in VARIANTS}
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        pred = {v: np.zeros(len(y), int) for v in VARIANTS}
        for tr, te in skf.split(P, y):
            # Fit every label-derived quantity using the training fold only.
            a, b = fit_platt(P[tr], y[tr])
            fold_failure_mix = failure_mix_from_labels(y[tr])
            pred["model"][te] = decide(calibrated_probs(P[te], a, b, "model"))
            pred["valmix"][te] = decide(
                calibrated_probs(P[te], a, b, "valmix", severity_mix=fold_failure_mix)
            )
        for v in VARIANTS:
            cost = COST_MATRIX[y, pred[v]]
            out[v].append({"mean_cost": float(cost.mean()), "flagged_share": float((pred[v] > 0).mean()),
                           "failure_recall_any": float((pred[v][y > 0] > 0).mean())})
    return out


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'policy':40s} {'mean_cost':>9s} {'bal_acc':>8s} {'macroF1':>8s} {'flagged':>8s}  predicted 0..4 | recall 0..4")
    for r in rows:
        flagged = 1 - r["prediction_counts_0_to_4"][0] / sum(r["prediction_counts_0_to_4"])
        rec = [round(r["per_class_recall"][str(c)], 2) for c in CLASSES]
        print(f"{r['name']:40s} {r['mean_cost']:9.4f} {r['balanced_accuracy']:8.4f} {r['macro_f1']:8.4f} {flagged:8.3f}  "
              f"{r['prediction_counts_0_to_4']} | {rec}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "test"], required=True)
    ap.add_argument("--labels-path", help="override labels CSV (code testing only)")
    args = ap.parse_args()
    settings = Settings.from_env()
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.mode == "cv":
        P, y = load_xy("validation", settings, args.labels_path)
        print(f"Calibration set = official validation | vehicles {len(y)} | class counts {np.bincount(y, minlength=5).tolist()}")
        base = baselines(y, P)
        cv = cv_run(P, y)
        always4 = base["always_4"]["mean_cost"]
        summary, passed = {}, {}
        print(f"\nRepeated 5-fold CV by vehicle (10 repeats). always_4 mean cost = {always4:.4f}")
        for v in VARIANTS:
            m = np.array([r["mean_cost"] for r in cv[v]])
            fl = np.mean([r["flagged_share"] for r in cv[v]])
            rc = np.mean([r["failure_recall_any"] for r in cv[v]])
            summary[v] = {"cv_mean_cost": float(m.mean()), "cv_sd": float(m.std(ddof=1)), "flagged_share": float(fl),
                          "failure_recall_any": float(rc), "repeats": cv[v]}
            passed[v] = bool(m.mean() + m.std(ddof=1) < always4 and fl <= 0.80)
            print(f"  variant '{v:7s}': CV mean cost {m.mean():.4f} +/- {m.std(ddof=1):.4f} | flagged {fl:.1%} | "
                  f"failure vehicles flagged {rc:.1%} | acceptance rule: {'PASS' if passed[v] else 'FAIL'}")
        best = min(VARIANTS, key=lambda v: summary[v]["cv_mean_cost"])
        a, b = fit_platt(P, y)  # fitted on ALL validation; used for test
        full_validation_failure_mix = failure_mix_from_labels(y)
        in_sample = score(
            f"M2 {best} (IN-SAMPLE, optimistic)", y,
            decide(calibrated_probs(P, a, b, best, severity_mix=full_validation_failure_mix)), P
        )
        print_table("Reference policies and in-sample fit on validation (in-sample is optimistic, do not quote)",
                    list(base.values()) + [in_sample])
        print(f"\nSelected variant: '{best}'. Platt parameters: slope a={a:.4f}, intercept b={b:.4f}")
        print("Reliability after calibration (in-sample):")
        for g in reliability(y, calibrated_probs(P, a, b, best)):
            print(f"    n={g['n']:5d}  predicted={g['mean_predicted_p_any_failure']:.4f}  observed={g['observed_failure_rate']:.4f}")
        CAL_PATH.write_text(json.dumps({"created": stamp, "calibration_set": "official validation", "slope_a": a, "intercept_b": b,
                                        "selected_variant": best, "val_failure_mix_1_to_4": full_validation_failure_mix.tolist(),
                                        "cv_summary": {k: {kk: vv for kk, vv in s.items() if kk != "repeats"} for k, s in summary.items()},
                                        "acceptance_passed_in_cv": passed}, indent=2), encoding="utf-8")
        (out_dir / f"m2_cv_{stamp}.json").write_text(json.dumps({"baselines": base, "cv": summary, "selected": best}, indent=2), encoding="utf-8")
        print(f"\nSaved calibrator: {CAL_PATH}\nNext (only if satisfied): python experiments\\m2_calibrated_cost.py --mode test   (run ONCE)")
        return 0

    if not CAL_PATH.is_file():
        raise SystemExit("Run --mode cv first (it saves the calibrator).")
    cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    P, y = load_xy("test", settings, args.labels_path)
    print(f"TEST split | vehicles {len(y)} | class counts {np.bincount(y, minlength=5).tolist()} | variant '{cal['selected_variant']}'")
    base = baselines(y, P)
    Q = calibrated_probs(P, cal["slope_a"], cal["intercept_b"], cal["selected_variant"])
    res = score(f"M2 {cal['selected_variant']} (test)", y, decide(Q), Q)
    print_table("TEST results (calibrator fitted on validation only; test labels used for scoring only)", list(base.values()) + [res])
    flagged = 1 - res["prediction_counts_0_to_4"][0] / len(y)
    ok = res["mean_cost"] < base["always_4"]["mean_cost"] and flagged <= 0.80
    print(f"\nAcceptance on test: mean cost {res['mean_cost']:.4f} vs always_4 {base['always_4']['mean_cost']:.4f}; flagged {flagged:.1%} -> {'PASS' if ok else 'FAIL'}")
    (out_dir / f"m2_test_{stamp}.json").write_text(json.dumps({"calibrator": cal, "baselines": base, "m2_test": res,
                                                              "reliability": reliability(y, Q), "acceptance_passed": bool(ok)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
