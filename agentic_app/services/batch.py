"""Run the FULL agent workflow on every vehicle of a split and summarise what happened (real inference, real vehicles)."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Dict, Optional


def run_batch(
    orchestrator,
    index,
    split: str,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    seed: int = 2026,
) -> Dict[str, Any]:
    scored = index.scored(split)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer when provided")
    if limit is not None and limit < len(scored):
        # A bounded batch must represent the split rather than whichever vehicles
        # happen to appear first in the source CSV. A fixed seed makes it repeatable.
        selected = scored.sample(n=limit, random_state=seed, replace=False)
    else:
        selected = scored
    ids = selected["vehicle_id"].tolist()
    final_status, replans, priorities, failed_checks, errors = Counter(), Counter(), Counter(), Counter(), Counter()
    disagreements, unusual_stats, ms = [], [], []
    by_id = scored.set_index("vehicle_id")
    t_all = time.perf_counter()
    for k, vid in enumerate(ids, start=1):
        t0 = time.perf_counter()
        result = orchestrator.run_workflow(index.payload(split, vid))
        ms.append((time.perf_counter() - t0) * 1000)
        if result.error:
            errors[result.error.code] += 1
            continue
        final_status[result.final_status] += 1
        replans[result.replan_count] += 1
        rec = result.recommendation
        priorities[rec.priority] += 1
        for c in result.verification.checks:
            if not c.passed:
                failed_checks[f"{c.id} {c.name}"] += 1
        # The dashboard (index) and the live workflow must agree on every vehicle.
        row = by_id.loc[vid]
        if int(row["decision"]) != rec.recommended_class or row["priority"] != rec.priority:
            disagreements.append({"vehicle_id": vid, "index": [int(row["decision"]), row["priority"]], "workflow": [rec.recommended_class, rec.priority]})
        if progress and k % 250 == 0:
            progress(k, len(ids))
    done = sum(final_status.values())
    return {
        "split": split, "vehicles_run": len(ids), "vehicles_completed": done, "workflow_errors": dict(errors),
        "sampling": {"method": "deterministic_random_without_replacement" if limit is not None and limit < len(scored) else "full_split",
                     "seed": seed if limit is not None and limit < len(scored) else None,
                     "requested_limit": limit},
        "selected_vehicle_ids": ids,
        "final_status_counts": dict(final_status), "replan_count_distribution": {str(k): v for k, v in sorted(replans.items())},
        "priority_counts": {p: priorities.get(p, 0) for p in ("low", "medium", "high", "critical")},
        "failed_check_counts": dict(failed_checks),
        "dashboard_workflow_disagreements": len(disagreements), "disagreement_examples": disagreements[:5],
        "mean_workflow_ms_per_vehicle": round(sum(ms) / len(ms), 2) if ms else None,
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "meaning": ("Each vehicle went through input validation, prediction, investigation, decision and verification. "
                    "'verified' = software consistency checks passed; it is not a claim that predictions are correct."),
    }
