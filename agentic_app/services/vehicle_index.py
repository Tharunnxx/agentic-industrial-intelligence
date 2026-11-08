"""Fast vehicle lookup and dashboard analytics over the real validation/test vehicles.

Built once by scripts/build_vehicle_index.py from YOUR original CSV files (latest readout per vehicle, the same protocol
as the model evaluation). Nothing here is synthetic. Class labels are used ONLY to score results, never to predict.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.calibration import Calibrator
from ..core.config import Settings
from ..core.costs import COST_MATRIX
from ..core.rules import compute_priority
from .metrics import evaluate_policy

SPLITS = ("validation", "test")
PRIORITY_ORDER = ["low", "medium", "high", "critical"]
CAL_HIST_EDGES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 1.0001]
RAW_HIST_EDGES = [round(0.1 * i, 1) for i in range(11)]


class IndexUnavailable(Exception):
    pass


def _clean(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


class VehicleIndex:
    HINT = "Run: python scripts\\build_vehicle_index.py   (it reads your original CSV files once)"

    def __init__(self, index_dir: Path, settings: Settings, calibrator: Optional[Calibrator] = None):
        self.dir = Path(index_dir)
        self.settings = settings
        self.calibrator = calibrator
        self._vehicles: Dict[str, pd.DataFrame] = {}
        self._scored: Dict[str, pd.DataFrame] = {}
        self._analytics: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ files
    def status(self) -> Dict[str, Any]:
        return {s: {"vehicles_file": (self.dir / f"{s}_vehicles.csv.gz").is_file(), "scored_file": (self.dir / f"{s}_scored.csv").is_file()} for s in SPLITS}

    @staticmethod
    def _check_split(split: str) -> str:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        return split

    def _vehicle_frame(self, split: str) -> pd.DataFrame:
        if split not in self._vehicles:
            f = self.dir / f"{self._check_split(split)}_vehicles.csv.gz"
            if not f.is_file():
                raise IndexUnavailable(f"{f.name} not found. {self.HINT}")
            df = pd.read_csv(f, dtype={"vehicle_id": str}, low_memory=False)
            self._vehicles[split] = df.set_index("vehicle_id", drop=False)
        return self._vehicles[split]

    def scored(self, split: str) -> pd.DataFrame:
        """One row per vehicle: raw probabilities, calibrated probabilities, decision, priority, percentile (+ label if present)."""
        if split in self._scored:
            return self._scored[split]
        f = self.dir / f"{self._check_split(split)}_scored.csv"
        if not f.is_file():
            raise IndexUnavailable(f"{f.name} not found. {self.HINT}")
        df = pd.read_csv(f, dtype={"vehicle_id": str})
        P = df[[f"p{i}" for i in range(5)]].to_numpy(dtype=float)
        cal = self.calibrator
        Q = cal.apply(P) if cal is not None else P
        for i in range(5):
            df[f"q{i}"] = Q[:, i]
        df["raw_p_any"] = 1.0 - P[:, 0]
        df["p_any"] = 1.0 - Q[:, 0]
        ec = Q @ COST_MATRIX
        df["decision"] = ec.argmin(axis=1).astype(int)
        df["expected_cost"] = ec.min(axis=1)
        df["priority"] = [compute_priority(q, self.settings, calibrated=cal is not None).priority for q in Q]
        if cal is not None and cal.reference:
            srt = np.asarray(cal.reference["raw_p_any_sorted"])
            df["percentile"] = np.searchsorted(srt, df["raw_p_any"].to_numpy(), side="right") / len(srt) * 100.0
        else:
            df["percentile"] = np.nan
        self._scored[split] = df
        return df

    # ------------------------------------------------------------------ search / payload
    def search(self, split: str, q: str = "", sort: str = "risk", priority: Optional[str] = None, page: int = 1, page_size: int = 25) -> Dict[str, Any]:
        df = self.scored(split)
        if q:
            df = df[df["vehicle_id"].str.startswith(q.strip())]
        if priority:
            if priority not in PRIORITY_ORDER:
                raise ValueError(f"priority must be one of {PRIORITY_ORDER}")
            df = df[df["priority"] == priority]
        df = df.sort_values("p_any", ascending=False) if sort == "risk" else df.sort_values("vehicle_id", key=lambda s: s.astype(float))
        page, page_size = max(1, int(page)), min(200, max(1, int(page_size)))
        total = len(df)
        part = df.iloc[(page - 1) * page_size: page * page_size]
        rows = [{"vehicle_id": r.vehicle_id, "time_step": _clean(r.time_step), "raw_p_any_failure": float(r.raw_p_any),
                 "calibrated_p_any_failure": float(r.p_any), "decision_class": int(r.decision), "priority": r.priority,
                 "risk_percentile": _clean(r.percentile)} for r in part.itertuples()]
        return {"split": split, "total": total, "page": page, "page_size": page_size, "sort": sort, "vehicles": rows,
                "note": "Labels are hidden here; they are used only for scoring on the Model Performance page."}

    def payload(self, split: str, vehicle_id: str) -> Optional[Dict[str, Any]]:
        frame = self._vehicle_frame(split)
        vid = str(vehicle_id).strip()
        if vid not in frame.index:
            return None
        row = frame.loc[vid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        meta = {"vehicle_id", "time_step"}
        feats = {k: _clean(v) for k, v in row.items() if k not in meta}
        return {"vehicle_id": int(vid) if vid.isdigit() else vid, "time_step": _clean(row["time_step"]), "features": feats}

    def reference_label(self, split: str, vehicle_id: str) -> Optional[int]:
        df = self.scored(split)
        if "class_label" not in df.columns:
            return None
        hit = df[df["vehicle_id"] == str(vehicle_id).strip()]
        if hit.empty or pd.isna(hit.iloc[0]["class_label"]):
            return None
        return int(hit.iloc[0]["class_label"])

    # ------------------------------------------------------------------ analytics
    def analytics(self, split: str) -> Dict[str, Any]:
        if split in self._analytics:
            return self._analytics[split]
        df = self.scored(split)
        cal = self.calibrator is not None
        n = len(df)
        P = df[[f"p{i}" for i in range(5)]].to_numpy()
        Q = df[[f"q{i}" for i in range(5)]].to_numpy()
        out: Dict[str, Any] = {
            "split": split, "n_vehicles": n, "calibration_applied": cal,
            "priority_counts": {p: int((df["priority"] == p).sum()) for p in PRIORITY_ORDER},
            "decision_counts_0_to_4": np.bincount(df["decision"].to_numpy(), minlength=5).tolist(),
            "mean_expected_cost_of_chosen_decision": float(df["expected_cost"].mean()),
            "calibrated_p_any_histogram": self._hist(df["p_any"].to_numpy(), CAL_HIST_EDGES),
            "raw_p_any_histogram": self._hist(df["raw_p_any"].to_numpy(), RAW_HIST_EDGES),
            "mean_raw_p_any": float(df["raw_p_any"].mean()), "mean_calibrated_p_any": float(df["p_any"].mean()),
        }
        labelled = "class_label" in df.columns and df["class_label"].notna().all()
        out["labelled"] = bool(labelled)
        if labelled:
            y = df["class_label"].to_numpy(dtype=int)
            out["actual_class_counts_0_to_4"] = np.bincount(y, minlength=5).tolist()
            out["actual_failure_rate"] = float((y > 0).mean())
            policies = [
                ("always_0", "Always class 0 (no action)", np.zeros(n, int), P),
                ("always_4", "Always class 4 (maximum action)", np.full(n, 4, int), P),
                ("m0_raw_argmax", "Original model, highest raw class", P.argmax(axis=1), P),
                ("m0_cost_rule", "Original model + cost rule (raw scores)", (P @ COST_MATRIX).argmin(axis=1), P),
            ]
            if cal:
                policies.append(("m2_calibrated", "Calibrated model + cost rule", (Q @ COST_MATRIX).argmin(axis=1), Q))
            out["policies"] = []
            for key, label, pred, probs in policies:
                r = evaluate_policy(key, y, pred, probs)
                cm = np.array(r["confusion_matrix_rows_actual_cols_predicted"])
                fail = cm[1:]
                r["label"] = label
                r["flagged_share"] = float(1 - cm[:, 0].sum() / cm.sum())
                r["failure_recall_any"] = float(fail[:, 1:].sum() / fail.sum()) if fail.sum() else None
                r["healthy_vehicles_flagged_share"] = float(cm[0, 1:].sum() / cm[0].sum()) if cm[0].sum() else None
                out["policies"].append(r)
            out["evaluation_basis"] = (
                "In-sample for the calibrated policy: the calibration numbers were fitted on this validation set. Use the test split for the unbiased result."
                if (cal and split == "validation") else
                "Out-of-sample for the calibrated policy: the calibration numbers were fitted on validation only." if cal else "No calibration applied.")
            out["reliability"] = self._reliability(df, y)
        self._analytics[split] = out
        return out

    @staticmethod
    def _hist(values: np.ndarray, edges: List[float]) -> Dict[str, Any]:
        counts, _ = np.histogram(values, bins=edges)
        labels = [f"{edges[i] * 100:.1f}-{min(edges[i + 1], 1.0) * 100:.1f}%" for i in range(len(edges) - 1)]
        return {"labels": labels, "counts": counts.astype(int).tolist(), "edges": edges}

    @staticmethod
    def _reliability(df: pd.DataFrame, y: np.ndarray, groups: int = 5) -> List[Dict[str, Any]]:
        order = np.argsort(df["p_any"].to_numpy())
        rows = []
        for k, chunk in enumerate(np.array_split(order, groups), start=1):
            rows.append({"group": k, "n": int(len(chunk)), "mean_predicted_p_any_failure": float(df["p_any"].to_numpy()[chunk].mean()),
                         "observed_failure_rate": float((y[chunk] > 0).mean())})
        return rows

    def batch_summary(self, split: str) -> Optional[Dict[str, Any]]:
        f = self.dir / f"{self._check_split(split)}_batch_summary.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
