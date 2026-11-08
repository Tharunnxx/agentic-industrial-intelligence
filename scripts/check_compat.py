"""Read-only compatibility check: can the platform reuse your existing joblib model and local files?

Usage (from agentic_platform/):  python scripts/check_compat.py [--model PATH]
Nothing is trained, written or modified.
"""
import _bootstrap  # noqa: F401
import argparse
import dataclasses
import sys

import pandas as pd

from agentic_app.core.config import Settings
from agentic_app.services.model_service import ModelService


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override AGENTIC_MODEL_PATH (e.g. balanced_baseline_model.joblib)")
    args = ap.parse_args()
    settings = Settings.from_env()
    if args.model:
        from pathlib import Path
        p = Path(args.model)
        settings = dataclasses.replace(settings, model_path=p if p.is_absolute() else settings.data_dir / p)

    print(f"Data folder : {settings.data_dir}")
    print(f"Model file  : {settings.model_path}")
    ms = ModelService(settings)
    if not ms.ensure_loaded():
        print(f"\n[FAIL] {ms.error.code}: {ms.error}")
        return 1
    info = ms.info()
    print("\n[OK] Model artifact loaded")
    for key in ("artifact_format", "artifact_keys", "estimator_types", "n_features", "n_numeric_features",
                "n_specification_features", "specification_categories_known", "calibration_wrapper_detected", "classes"):
        print(f"  {key}: {info[key]}")
    for w in info["compatibility_warnings"]:
        print(f"  [WARNING] {w}")
    for n in info["known_results"].get("reliability_notes", []):
        print(f"  [NOTE] {n}")

    schema = ms.schema()
    status = 0
    files = {"operational readouts": settings.data_dir / "validation_operational_readouts.csv",
             "specifications": settings.data_dir / "validation_specifications.csv",
             "labels": settings.data_dir / "validation_labels.csv"}
    print("\nLocal validation files")
    headers = {}
    for name, path in files.items():
        if path.is_file():
            headers[name] = pd.read_csv(path, nrows=0).columns.tolist()
            print(f"  [OK] {name}: {path.name}")
        else:
            print(f"  [--] {name}: {path.name} not found (only needed for extract_samples.py / evaluate_saved_model.py)")
    if "operational readouts" in headers and "specifications" in headers:
        available = set(headers["operational readouts"]) | set(headers["specifications"])
        missing = [c for c in schema.feature_cols if c not in available]
        if missing:
            print(f"  [FAIL] model features missing from the validation CSVs: {missing[:10]}")
            status = 1
        else:
            print("  [OK] every model feature is present in the validation CSVs")
    print("\nResult:", "COMPATIBLE" if status == 0 else "INCOMPATIBLE")
    return status


if __name__ == "__main__":
    sys.exit(main())
