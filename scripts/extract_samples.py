"""Extract a few REAL rows from your validation CSVs as demo samples for the web UI.

Uses the same protocol as your app.py: the latest readout per vehicle, merged with its specification row.
Rows are picked by seed (no cherry-picking by prediction). The dataset label is stored under meta only, for
post-hoc comparison in the demo; it is never sent to the agents.

Usage (from agentic_platform/):  python scripts/extract_samples.py [--n-per-class 2] [--seed 2026]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_app.core.config import Settings
from agentic_app.services.model_service import ModelService


def py(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
    return value


def main() -> int:
    settings = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--data-dir", default=str(settings.data_dir))
    ap.add_argument("--out-dir", default=str(settings.samples_dir))
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)

    ms = ModelService(settings)
    if not ms.ensure_loaded():
        print(f"[FAIL] {ms.error.code}: {ms.error}\nSamples are shaped by the model's feature list, so the model must load.")
        return 1
    schema = ms.schema()

    paths = {k: data_dir / f"validation_{k}.csv" for k in ("operational_readouts", "specifications", "labels")}
    for name, p in paths.items():
        if not p.is_file():
            print(f"[FAIL] Missing {p}")
            return 1

    spec_header = pd.read_csv(paths["specifications"], nrows=0).columns.tolist()
    ops_header = pd.read_csv(paths["operational_readouts"], nrows=0).columns.tolist()
    want_ops = [c for c in schema.feature_cols if c in ops_header]
    want_specs = [c for c in schema.feature_cols if c in spec_header and c not in ops_header]
    unresolved = [c for c in schema.feature_cols if c not in want_ops and c not in want_specs]
    if unresolved:
        print(f"[FAIL] Model features not found in the validation CSVs: {unresolved[:10]}")
        return 1

    print("Reading validation readouts ...")
    ops = pd.read_csv(paths["operational_readouts"], usecols=sorted(set(["vehicle_id", "time_step"] + want_ops)))
    latest = ops.sort_values("time_step").drop_duplicates("vehicle_id", keep="last")
    specs = pd.read_csv(paths["specifications"], usecols=["vehicle_id"] + want_specs)
    labels = pd.read_csv(paths["labels"])
    data = latest.merge(specs, on="vehicle_id", how="left", validate="one_to_one").merge(labels, on="vehicle_id", how="inner", validate="one_to_one")

    rng = np.random.default_rng(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for cls in range(5):
        pool = data.index[data["class_label"] == cls].to_numpy()
        if len(pool) == 0:
            print(f"  class {cls}: no vehicles in the validation set")
            continue
        for idx in rng.choice(pool, size=min(args.n_per_class, len(pool)), replace=False):
            row = data.loc[idx]
            features = {c: py(row[c]) for c in schema.feature_cols}
            numeric = [features[c] for c in schema.numeric_cols]
            missing_fraction = sum(v is None for v in numeric) / max(len(numeric), 1)
            vid = int(row["vehicle_id"])
            sample = {
                "vehicle_id": vid,
                "time_step": float(row["time_step"]),
                "features": features,
                "meta": {
                    "origin": "REAL row from the SCANIA validation set (latest readout per vehicle); not synthetic",
                    "source": f"{paths['operational_readouts'].name} + {paths['specifications'].name}",
                    "missing_fraction": round(missing_fraction, 4),
                    "evaluation_only_label": int(row["class_label"]),
                    "label_note": "Official validation label, shown after a run for comparison only; never given to the agents.",
                },
            }
            target = out_dir / f"sample_{vid}_class{cls}.json"
            target.write_text(json.dumps(sample, indent=1, allow_nan=False), encoding="utf-8")
            written += 1
            print(f"  wrote {target.name}  (missing numeric: {missing_fraction:.1%})")
    print(f"\n{written} sample(s) written to {out_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
