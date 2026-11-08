#!/usr/bin/env python
"""
Reproduce the three capstone graphs from REAL project artifacts.

Place this file in  agentic_platform/scripts/  and run from the project root:

    python scripts/generate_graphs.py

Optional:  --data-dir  --model-path  --out-dir  --vehicle-id  --chunksize

Protocol (mirrors services/evaluation_service.py, read-only):
  * latest operational readout per vehicle (max time_step) + its specification row
  * model features = artifact['feature_cols'] (105 operational + 8 Spec_)
  * probabilities = artifact['model'].predict_proba (raw, not calibrated)
  * validation_labels.csv is read ONLY AFTER all probabilities exist
  * cost matrix = artifact['cost_matrix'], rows = actual class, cols = predicted class
  * expected cost of choosing class j = sum_i P(actual=i) * C[i, j]
Nothing is retrained or modified. Nothing is invented: every plotted number comes
from the model run or from validation_labels.csv.
"""
from __future__ import annotations

import argparse
import textwrap
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

CLASSES = [0, 1, 2, 3, 4]
INK, MUTED, GRID = "#1F2937", "#6B7280", "#E5E7EB"
BLUE, SLATE, AMBER, GREEN, RED = "#2563EB", "#94A3B8", "#F59E0B", "#059669", "#DC2626"
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


# ----------------------------------------------------------------------------- data
def find_model(project_root: Path) -> Path:
    hits = sorted(project_root.rglob("cost_sensitive_model.joblib"))
    if not hits:
        raise SystemExit("cost_sensitive_model.joblib not found; pass --model-path.")
    return hits[0]


def load_artifact(path: Path):
    art = joblib.load(path)  # only load artifacts you created yourself
    model, cols, cost = art["model"], [str(c) for c in art["feature_cols"]], np.asarray(art["cost_matrix"], float)
    classes = [int(c) for c in model.classes_]
    if classes != CLASSES:
        raise SystemExit(f"Unexpected model classes {classes}")
    if cost.shape != (5, 5):
        raise SystemExit(f"Unexpected cost matrix shape {cost.shape}")
    if "class_label" in cols:
        raise SystemExit("Target leakage: class_label is a model feature.")
    return model, cols, cost


def latest_readouts(path: Path, ops_features, chunksize: int) -> pd.DataFrame:
    """Latest readout (max time_step) per vehicle, reading the big CSV in chunks."""
    use = list(dict.fromkeys(["vehicle_id", "time_step"] + ops_features))
    best = None
    for chunk in pd.read_csv(path, usecols=use, dtype={"vehicle_id": "string"}, chunksize=chunksize):
        merged = chunk if best is None else pd.concat([best, chunk], ignore_index=True)
        merged = merged.sort_values("time_step", kind="mergesort")
        best = merged.drop_duplicates("vehicle_id", keep="last")
    return best.reset_index(drop=True)


def predict_validation(data_dir: Path, model, cols, chunksize: int) -> pd.DataFrame:
    ops_p, spec_p = data_dir / "validation_operational_readouts.csv", data_dir / "validation_specifications.csv"
    for p in (ops_p, spec_p):
        if not p.is_file():
            raise SystemExit(f"Missing required file: {p}")
    ops_header = pd.read_csv(ops_p, nrows=0).columns.tolist()
    spec_header = pd.read_csv(spec_p, nrows=0).columns.tolist()
    ops_feats = [c for c in cols if c in ops_header]
    spec_feats = [c for c in cols if c in spec_header and c not in ops_header]
    unresolved = [c for c in cols if c not in ops_feats and c not in spec_feats]
    if unresolved:
        raise SystemExit(f"Model features absent from validation data: {unresolved[:10]}")
    print(f"features: {len(ops_feats)} operational + {len(spec_feats)} specification")

    latest = latest_readouts(ops_p, ops_feats, chunksize)
    specs = pd.read_csv(spec_p, usecols=list(dict.fromkeys(["vehicle_id"] + spec_feats)), dtype={"vehicle_id": "string"})
    if specs["vehicle_id"].duplicated().any():
        raise SystemExit("Specifications must have one row per vehicle.")
    feats = latest.merge(specs, on="vehicle_id", how="left", validate="one_to_one")
    if spec_feats and feats[spec_feats].isna().all(axis=1).any():
        raise SystemExit("Some vehicles have no specification row.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = np.asarray(model.predict_proba(feats[cols]), float)  # columns follow model.classes_ == 0..4
    out = pd.DataFrame({"vehicle_id": feats["vehicle_id"].astype(str), "time_step": feats["time_step"]})
    for k in CLASSES:
        out[f"p{k}"] = probs[:, k]
    return out  # NOTE: no labels have been read yet


def attach_labels(pred: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(data_dir / "validation_labels.csv", usecols=["vehicle_id", "class_label"], dtype={"vehicle_id": "string"})
    labels["vehicle_id"] = labels["vehicle_id"].astype(str)
    if labels["vehicle_id"].duplicated().any():
        raise SystemExit("validation_labels.csv has duplicate vehicle_id.")
    df = pred.merge(labels, on="vehicle_id", how="inner", validate="one_to_one")
    dropped = len(pred) - len(df)
    if dropped:
        print(f"WARNING: {dropped} predicted vehicles had no label and are excluded from evaluation.")
    df["class_label"] = df["class_label"].astype(int)
    return df


# --------------------------------------------------------------------------- helpers
def expected_costs(probs: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """E[cost | choose j] = sum_i p_i * C[i, j]   (rows = actual, cols = predicted)."""
    return probs @ cost


def policy_metrics(y, pred, cost):
    realised = cost[y, pred]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "accuracy": float((y == pred).mean()),
        "total_cost": float(realised.sum()),
        "mean_cost": float(realised.mean()),
        "class4_prediction_rate": float((pred == 4).mean()),
    }


def add_caption(fig, text, bottom):
    fig.subplots_adjust(bottom=bottom)
    fig.text(0.06, 0.02, textwrap.fill(text, 150), ha="left", va="bottom", fontsize=9.5, color=MUTED, linespacing=1.45)


def style_title(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=34)
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED, va="bottom")


def pick_vehicle(df: pd.DataFrame, cost: np.ndarray, wanted: str | None):
    """Deterministic, disclosed rule (post-inference; labels not used to choose)."""
    probs = df[[f"p{k}" for k in CLASSES]].to_numpy()
    argmax, cdec = probs.argmax(1), expected_costs(probs, cost).argmin(1)
    if wanted is not None:
        hit = df.index[df["vehicle_id"] == str(wanted)]
        if len(hit) == 0:
            raise SystemExit(f"vehicle_id {wanted} not in validation set.")
        return int(hit[0]), f"user-specified vehicle_id={wanted}"
    differs = np.where(argmax != cdec)[0]
    if len(differs):
        i = differs[np.argmax(1 - probs[differs, 0])]
        return int(i), ("auto: among vehicles where the highest-probability class differs from the "
                        "minimum-expected-cost class, the one with the largest P(any failure)")
    i = int(np.argmax(1 - probs[:, 0]))
    return i, "auto: no vehicle had differing argmax/cost decision; vehicle with the largest P(any failure)"


# ------------------------------------------------------------------------------ graphs
def graph1(df, i, rule, out_png, out_csv):
    row = df.iloc[i]
    p = np.array([row[f"p{k}"] for k in CLASSES])
    top = int(p.argmax())
    fig, ax = plt.subplots(figsize=(11, 7.6))
    colors = [BLUE if k == top else "#BFDBFE" for k in CLASSES]
    bars = ax.bar([f"Class {k}" for k in CLASSES], p * 100, color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, p * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + max(p * 100) * 0.015, f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted probability (%)")
    ax.set_xlabel("Failure class")
    ax.set_ylim(0, max(p * 100) * 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    style_title(ax, "Vehicle Failure Probability Distribution",
                f"Validation vehicle {row['vehicle_id']} · latest readout (time_step {row['time_step']:g}) · raw model probabilities")
    ax.legend(handles=[matplotlib.patches.Patch(color=BLUE, label=f"Highest-probability class (Class {top})"),
                       matplotlib.patches.Patch(color="#BFDBFE", label="Other classes")], frameon=False, loc="upper right")
    cap = (f"What this shows: the model's predicted probability for each of the five failure classes for this one vehicle; the five values sum to "
           f"{p.sum() * 100:.1f}%. The model returns a distribution rather than a single answer. Probabilities are RAW outputs of a RandomForest "
           f"pipeline with no calibration wrapper; they have not been verified as calibrated and should be read as model scores, not guaranteed "
           f"likelihoods. Vehicle choice: {rule}. Its official label (revealed only after inference, never an input) is Class {int(row['class_label'])}.")
    add_caption(fig, cap, 0.26)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    pd.DataFrame({"vehicle_id": row["vehicle_id"], "time_step": row["time_step"], "class": CLASSES,
                  "probability": p, "probability_percent": p * 100, "is_highest_probability": [k == top for k in CLASSES],
                  "official_label_post_inference": int(row["class_label"]), "selection_rule": rule}).to_csv(out_csv, index=False)


def graph2(df, i, cost, out_png, out_csv):
    row = df.iloc[i]
    p = np.array([row[f"p{k}"] for k in CLASSES])
    ec = expected_costs(p[None, :], cost)[0]
    best, top = int(ec.argmin()), int(p.argmax())
    fig, ax = plt.subplots(figsize=(11, 7.6))
    colors = [GREEN if k == best else SLATE for k in CLASSES]
    bars = ax.bar([f"Class {k}" for k in CLASSES], ec, color=colors, width=0.62, zorder=3)
    for k, (b, v) in enumerate(zip(bars, ec)):
        ax.text(b.get_x() + b.get_width() / 2, v + ec.max() * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        if k == top:
            b.set_edgecolor(BLUE); b.set_linewidth(2.5)
    ax.set_ylabel("Expected cost (project cost units)")
    ax.set_xlabel("Candidate decision class")
    ax.set_ylim(0, ec.max() * 1.15)
    style_title(ax, "Cost-Sensitive Maintenance Decision Analysis",
                f"Validation vehicle {row['vehicle_id']} · E[cost | choose j] = Σᵢ P(actual = i) · C[i, j]")
    handles = [matplotlib.patches.Patch(color=GREEN, label=f"Lowest expected cost → recommended (Class {best})"),
               matplotlib.patches.Patch(color=SLATE, label="Other candidates")]
    handles.append(matplotlib.patches.Patch(facecolor="white", edgecolor=BLUE, linewidth=2.5, label=f"Highest-probability class (Class {top})"))
    ax.legend(handles=handles, frameon=False, loc="upper right")
    if best != top:
        why = (f"Here the highest-probability class is Class {top} ({p[top] * 100:.1f}%), but choosing Class {best} has the lowest expected cost "
               f"({ec[best]:.1f} vs {ec[top]:.1f} for Class {top}), because the cost matrix penalises under-predicting a failure much more heavily than over-predicting it.")
    else:
        why = (f"For this vehicle the highest-probability class and the lowest-expected-cost class coincide (Class {best}); the two can differ "
               f"when failure classes carry a small probability but a large cost if missed.")
    cap = (f"What this shows: the expected cost of each possible decision, computed from the model's probabilities and the cost matrix stored in the "
           f"model artifact (rows = actual class, columns = predicted class). {why} Cost values are the project's relative cost units, not currency.")
    add_caption(fig, cap, 0.26)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    pd.DataFrame({"vehicle_id": row["vehicle_id"], "candidate_class": CLASSES, "probability": p, "expected_cost": ec,
                  "is_min_expected_cost": [k == best for k in CLASSES], "is_highest_probability": [k == top for k in CLASSES]}).to_csv(out_csv, index=False)
    return best, top


def graph3(df, cost, out_png, out_csv, out_metrics_csv):
    probs = df[[f"p{k}" for k in CLASSES]].to_numpy()
    y = df["class_label"].to_numpy()
    pred_raw = probs.argmax(1)
    pred_cost = expected_costs(probs, cost).argmin(1)
    actual = np.bincount(y, minlength=5)
    raw_c, cost_c = np.bincount(pred_raw, minlength=5), np.bincount(pred_cost, minlength=5)
    n = len(y)

    fig, ax = plt.subplots(figsize=(12, 8.2))
    x, w = np.arange(5), 0.27
    series = [("Actual (official labels)", actual, SLATE, -w), ("Predicted: highest probability (argmax)", raw_c, BLUE, 0),
              ("Predicted: minimum expected cost", cost_c, AMBER, w)]
    for name, vals, color, off in series:
        bars = ax.bar(x + off, vals, w, label=name, color=color, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, (v if v > 0 else 0) * 1.12 + 0.3, f"{int(v):,}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, max(actual.max(), raw_c.max(), cost_c.max()) * 6)
    ticks = [t for t in (0, 1, 10, 100, 1000, 10000) if t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:,}" for t in ticks])
    ax.set_xticks(x); ax.set_xticklabels([f"Class {k}" for k in CLASSES])
    ax.set_ylabel("Number of vehicles (log scale)"); ax.set_xlabel("Failure class")
    style_title(ax, "Actual vs Predicted Failure Class Distribution — Validation Dataset",
                f"{n:,} vehicles · official validation split · latest readout per vehicle · labels used only after inference")
    ax.legend(frameon=False, loc="upper right")

    pol = {"raw_argmax": policy_metrics(y, pred_raw, cost), "expected_cost": policy_metrics(y, pred_cost, cost),
           "always_0": policy_metrics(y, np.zeros(n, int), cost), "always_4": policy_metrics(y, np.full(n, 4), cost)}
    r, c = pol["raw_argmax"], pol["expected_cost"]
    bias = (f"Class 4 is predicted for {raw_c[4]:,} of {n:,} vehicles by argmax ({r['class4_prediction_rate'] * 100:.1f}%) and {cost_c[4]:,} "
            f"({c['class4_prediction_rate'] * 100:.1f}%) by the cost rule, versus {actual[4]:,} actual Class 4 vehicles ({actual[4] / n * 100:.1f}%)")
    cap = (f"Class imbalance: {actual[0]:,} of {n:,} vehicles ({actual[0] / n * 100:.1f}%) are Class 0. {bias}. Metrics (same run): argmax — balanced accuracy "
           f"{r['balanced_accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f}, mean cost {r['mean_cost']:.2f}; minimum-expected-cost — balanced accuracy "
           f"{c['balanced_accuracy']:.3f}, macro-F1 {c['macro_f1']:.3f}, mean cost {c['mean_cost']:.2f}; always-Class-4 baseline mean cost "
           f"{pol['always_4']['mean_cost']:.2f}, always-Class-0 baseline {pol['always_0']['mean_cost']:.2f}. Raw RandomForest probabilities, not calibrated. "
           f"The log axis is needed because Class 0 dominates; compare bar heights by their labels.")
    add_caption(fig, cap, 0.27)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

    pd.DataFrame({"class": CLASSES, "actual_count": actual, "predicted_argmax_count": raw_c, "predicted_min_expected_cost_count": cost_c,
                  "actual_share": actual / n, "predicted_argmax_share": raw_c / n, "predicted_min_cost_share": cost_c / n}).to_csv(out_csv, index=False)
    pd.DataFrame(pol).T.rename_axis("policy").reset_index().to_csv(out_metrics_csv, index=False)
    return n, actual, raw_c, cost_c, pol


# ------------------------------------------------------------------------------ README
def write_readme(path, model_path, data_dir, vid, rule, g2, g3, cost, df, i):
    n, actual, raw_c, cost_c, pol = g3
    best, top = g2
    p = df.iloc[i][[f"p{k}" for k in CLASSES]].to_numpy(float)
    ec = expected_costs(p[None, :], cost)[0]
    r, c = pol["raw_argmax"], pol["expected_cost"]
    cost_txt = "\n".join("  ".join(f"{v:5.0f}" for v in row) for row in cost)
    md = f"""# Capstone graphs: generated from real project data

Generated by `scripts/generate_graphs.py`. All numbers come from the saved model run on the official validation split.

## Inputs
- Model: `{model_path}` (RandomForest pipeline, 113 features = 105 operational + 8 `Spec_`, classes 0–4; no calibration wrapper)
- Data: `{data_dir}` — `validation_operational_readouts.csv`, `validation_specifications.csv`; `validation_labels.csv` read only after inference
- Protocol (same as `services/evaluation_service.py`): latest readout per vehicle + specification row, one prediction per vehicle, {n:,} vehicles
- Cost matrix: stored in the model artifact; rows = actual class, columns = predicted class. Expected cost of choosing class j = Σᵢ P(i)·C[i,j]. Units are project cost units, not currency.

```
{cost_txt}
```

## Graph 1 — `01_vehicle_failure_probabilities.png`
Vehicle `{vid}` ({rule}). Probabilities (%): {', '.join(f'C{k}={p[k] * 100:.2f}' for k in CLASSES)}. These are raw RandomForest outputs, **not verified as calibrated**. Meaning: the model gives a distribution over failure classes, not a single label.

## Graph 2 — `02_expected_maintenance_cost.png`
Same vehicle. Expected costs: {', '.join(f'C{k}={ec[k]:.2f}' for k in CLASSES)}. Lowest: Class {best}; highest probability: Class {top}. The Decision Agent can recommend a class other than the most probable one because missing a failure is penalised far more than a precautionary action.

## Graph 3 — `03_actual_vs_predicted_distribution.png`
| Class | Actual | Predicted (argmax) | Predicted (min expected cost) |
|---|---|---|---|
""" + "\n".join(f"| {k} | {actual[k]:,} | {raw_c[k]:,} | {cost_c[k]:,} |" for k in CLASSES) + f"""

| Policy | Balanced acc. | Macro-F1 | Mean cost | Class-4 prediction rate |
|---|---|---|---|---|
""" + "\n".join(f"| {k} | {v['balanced_accuracy']:.3f} | {v['macro_f1']:.3f} | {v['mean_cost']:.2f} | {v['class4_prediction_rate'] * 100:.1f}% |" for k, v in pol.items()) + f"""

Class imbalance is severe (Class 0 = {actual[0] / n * 100:.1f}%). Read the class-4 columns above for prediction bias; any over-prediction of Class 4 is real and should be stated to reviewers. The log axis is used because Class 0 dominates.

## Limitations
- Probabilities are uncalibrated; do not present them as true likelihoods.
- Graphs 1–2 show one vehicle, selected by the disclosed rule; they illustrate the mechanism, not overall performance.
- Rare classes (1–3) have very few validation vehicles, so per-class results are noisy.
- Only the official validation split is used; no train/test or experiment numbers are mixed in.
- The model was pickled with scikit-learn 1.7.2; run in the environment you trained with.

## Reproduce
`python scripts/generate_graphs.py` from the project root.
"""
    path.write_text(md, encoding="utf-8")


def main():
    here = Path(__file__).resolve().parent
    root = here.parent if here.name == "scripts" else here
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root.parent)
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=root / "outputs" / "graphs")
    ap.add_argument("--vehicle-id", default=None)
    ap.add_argument("--chunksize", type=int, default=100_000)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = a.model_path or find_model(root)

    print("loading model:", model_path)
    model, cols, cost = load_artifact(model_path)
    print("cost matrix (rows=actual, cols=predicted):\n", cost.astype(int))
    pred = predict_validation(a.data_dir, model, cols, a.chunksize)
    pred.to_csv(a.out_dir / "validation_predictions.csv", index=False)  # no labels in this file
    df = attach_labels(pred, a.data_dir)
    print(f"{len(df)} vehicles scored")

    i, rule = pick_vehicle(df, cost, a.vehicle_id)
    vid = df.iloc[i]["vehicle_id"]
    print("selected vehicle:", vid, "|", rule)
    graph1(df, i, rule, a.out_dir / "01_vehicle_failure_probabilities.png", a.out_dir / "graph1_vehicle_probabilities.csv")
    g2 = graph2(df, i, cost, a.out_dir / "02_expected_maintenance_cost.png", a.out_dir / "graph2_expected_cost.csv")
    g3 = graph3(df, cost, a.out_dir / "03_actual_vs_predicted_distribution.png",
                a.out_dir / "graph3_class_distribution.csv", a.out_dir / "graph3_policy_metrics.csv")
    write_readme(a.out_dir / "README.md", model_path, a.data_dir, vid, rule, g2, g3, cost, df, i)
    print("done ->", a.out_dir)


if __name__ == "__main__":
    main()
