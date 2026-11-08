"""Run the full four-agent workflow on EVERY vehicle of a split and save a summary for the System Verification page.

    python scripts\\run_batch_workflow.py --split validation          (about 5 minutes; use --limit 200 for a quick trial)
    python scripts\\run_batch_workflow.py --split test

Needs data_index\\ (run scripts\\build_vehicle_index.py first). Real model inference, real vehicles, no synthetic data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_app.agents.orchestrator import Orchestrator  # noqa: E402
from agentic_app.core.calibration import Calibrator  # noqa: E402
from agentic_app.core.config import Settings  # noqa: E402
from agentic_app.services.batch import run_batch  # noqa: E402
from agentic_app.services.model_service import ModelService  # noqa: E402
from agentic_app.services.vehicle_index import VehicleIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--limit", type=int, default=None, help="run a reproducible random sample instead of the full split")
    ap.add_argument("--seed", type=int, default=2026, help="random seed for --limit sampling (default: 2026)")
    args = ap.parse_args()
    settings = Settings.from_env()
    ms = ModelService(settings)
    if not ms.ensure_loaded():
        raise SystemExit(f"Model unavailable: {ms.error}")
    cal = Calibrator.load(settings.calibration_path, settings.calibration_reference_path, settings.results_dir) if settings.use_calibration else None
    ref = json.loads(settings.feature_reference_path.read_text(encoding="utf-8")) if settings.feature_reference_path.is_file() else None
    index = VehicleIndex(settings.index_dir, settings, cal)
    orch = Orchestrator(settings, ms, calibrator=cal, feature_reference=ref)
    summary = run_batch(orch, index, args.split, args.limit, progress=lambda k, n: print(f"  {k}/{n} vehicles ..."), seed=args.seed)
    path = settings.index_dir / f"{args.split}_batch_summary.json"
    if args.limit:
        path = settings.index_dir / f"{args.split}_batch_summary_limit{args.limit}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("vehicles_run", "vehicles_completed", "final_status_counts", "replan_count_distribution",
                                              "priority_counts", "failed_check_counts", "dashboard_workflow_disagreements", "workflow_errors")}, indent=2))
    print("Saved:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
