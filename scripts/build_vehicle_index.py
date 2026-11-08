"""Build the vehicle index used by the web app (run ONCE, from the agentic_platform folder):

    python scripts\\build_vehicle_index.py

Reads YOUR original CSV files (validation + test), keeps the latest readout of every vehicle (the same protocol as the model
evaluation), scores each vehicle with the saved model, and writes small files into data_index\\ :
    {split}_vehicles.csv.gz   model inputs per vehicle
    {split}_scored.csv        raw probabilities + recorded label (label is used only to score results)
    feature_reference.json    1st / 50th / 99th percentile of each numeric feature over validation vehicles (descriptive only)
The model file is only READ. Nothing in your data folder is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_app.core.config import Settings  # noqa: E402
from agentic_app.services.model_service import ModelService  # noqa: E402


def load_features(ms: ModelService, settings: Settings, split: str) -> pd.DataFrame:
    schema = ms.schema()
    d = settings.data_dir
    ops_p, spec_p = d / f"{split}_operational_readouts.csv", d / f"{split}_specifications.csv"
    for p in (ops_p, spec_p):
        if not p.is_file():
            raise SystemExit(f"Missing file: {p}")
    ops_h = pd.read_csv(ops_p, nrows=0).columns.tolist()
    spec_h = pd.read_csv(spec_p, nrows=0).columns.tolist()
    ops_f = [c for c in schema.feature_cols if c in ops_h]
    spec_f = [c for c in schema.feature_cols if c in spec_h and c not in ops_h]
    print(f"[{split}] reading {ops_p.name} ...")
    ops = pd.read_csv(ops_p, usecols=list(dict.fromkeys(["vehicle_id", "time_step"] + ops_f)), dtype={"vehicle_id": "string"})
    latest = ops.sort_values("time_step", kind="mergesort").drop_duplicates("vehicle_id", keep="last")
    specs = pd.read_csv(spec_p, usecols=list(dict.fromkeys(["vehicle_id"] + spec_f)), dtype={"vehicle_id": "string"})
    feats = latest.merge(specs, on="vehicle_id", how="left", validate="one_to_one")
    feats["vehicle_id"] = feats["vehicle_id"].astype(str)
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation", "test"], choices=["validation", "test"])
    ap.add_argument("--recompute", action="store_true", help="ignore experiments\\cache and run the model again")
    args = ap.parse_args()
    settings = Settings.from_env()
    ms = ModelService(settings)
    if not ms.ensure_loaded():
        raise SystemExit(f"Model unavailable: {ms.error}")
    schema = ms.schema()
    out = settings.index_dir or (ROOT / "data_index")
    out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        feats = load_features(ms, settings, split)
        cache = ROOT / "experiments" / "cache" / f"original_model_probs_{split}.csv"
        probs = None
        if cache.is_file() and not args.recompute:
            c = pd.read_csv(cache, dtype={"vehicle_id": str}).set_index("vehicle_id")
            if set(c.index) == set(feats["vehicle_id"]):
                probs = c.loc[feats["vehicle_id"], [f"p{i}" for i in range(5)]].to_numpy(dtype=float)
                print(f"[{split}] reused cached model probabilities ({cache.name})")
        if probs is None:
            print(f"[{split}] running the saved model on {len(feats)} vehicles ...")
            probs = ms.predict_proba_frame(feats)
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise SystemExit("Probabilities do not sum to 1; stop and investigate.")

        feats[["vehicle_id", "time_step"] + schema.feature_cols].to_csv(out / f"{split}_vehicles.csv.gz", index=False, compression="gzip")
        scored = pd.DataFrame({"vehicle_id": feats["vehicle_id"], "time_step": feats["time_step"]})
        for i in range(5):
            scored[f"p{i}"] = probs[:, i]
        lab_p = settings.data_dir / f"{split}_labels.csv"
        if lab_p.is_file():
            labels = pd.read_csv(lab_p, usecols=["vehicle_id", "class_label"], dtype={"vehicle_id": str})
            scored = scored.merge(labels, on="vehicle_id", how="left", validate="one_to_one")
            print(f"[{split}] labels attached for scoring: {int(scored['class_label'].notna().sum())} of {len(scored)} vehicles")
        scored.to_csv(out / f"{split}_scored.csv", index=False)
        print(f"[{split}] wrote {split}_vehicles.csv.gz and {split}_scored.csv ({len(scored)} vehicles)")

        if split == "validation":
            ref = {}
            for c in schema.numeric_cols:
                v = pd.to_numeric(feats[c], errors="coerce").dropna()
                if len(v):
                    ref[c] = {"p01": float(v.quantile(0.01)), "p50": float(v.quantile(0.50)), "p99": float(v.quantile(0.99))}
            (out / "feature_reference.json").write_text(json.dumps({
                "source": "latest readout of each official validation vehicle; descriptive percentiles only, no labels used",
                "n_vehicles": int(len(feats)), "features": ref}), encoding="utf-8")
            print(f"[validation] wrote feature_reference.json ({len(ref)} numeric features)")
    print("\nDone. Start the app and open the Vehicle Analysis page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
