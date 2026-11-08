"""Memory-conscious access to SCANIA CSV splits. Original CSVs are read-only."""
from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def to_json_safe(value: Any) -> Any:
    """Recursively convert pandas/NumPy values into strict JSON-safe values.

    Missing and non-finite numeric values become None (JSON null); they are
    never replaced with zero because that would alter sensor semantics.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Handles pandas nullable scalars and NaN without treating arrays as bool.
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    return value


class DatasetService:
    SPLITS = ("train", "validation", "test")

    def __init__(self, data_dir: Path, chunksize: int = 100_000):
        self.data_dir = Path(data_dir)
        self.chunksize = chunksize

    def _path(self, split: str, kind: str) -> Path:
        if split not in self.SPLITS or kind not in {"operational_readouts", "specifications", "labels", "tte"}:
            raise ValueError("Unsupported dataset split or file type")
        return self.data_dir / f"{split}_{kind}.csv"

    def availability(self) -> dict[str, Any]:
        files = {}
        for split in self.SPLITS:
            kinds = ["operational_readouts", "specifications"]
            if split == "train": kinds.append("tte")
            else: kinds.append("labels")
            for kind in kinds:
                p = self._path(split, kind)
                files[p.name] = {"available": p.is_file(), "size_bytes": p.stat().st_size if p.is_file() else None}
        return {"data_dir": str(self.data_dir), "files": files}

    def summary(self) -> dict[str, Any]:
        result = self.availability()
        for split in self.SPLITS:
            result[split] = {}
            for kind in (["operational_readouts", "specifications", "tte"] if split == "train" else ["operational_readouts", "specifications", "labels"]):
                p = self._path(split, kind)
                if not p.is_file():
                    result[split][kind] = {"available": False}
                    continue
                try:
                    header = pd.read_csv(p, nrows=0).columns.tolist()
                    size = p.stat().st_size
                    # Avoid a full 1.13 GB scan just to render the dashboard summary.
                    rows = None if size > 100 * 1024 * 1024 else max(0, sum(1 for _ in open(p, "rb")) - 1)
                    result[split][kind] = {"available": True, "rows": rows, "row_count_deferred": rows is None, "size_bytes": size, "columns": len(header), "column_names": header}
                except Exception as exc:
                    result[split][kind] = {"available": True, "error": f"{type(exc).__name__}: {exc}"}
        return result

    def vehicle(self, split: str, vehicle_id: str | int) -> dict[str, Any] | None:
        if split not in self.SPLITS: raise ValueError("Unsupported dataset split")
        key = str(vehicle_id)
        spec_path = self._path(split, "specifications")
        if not spec_path.is_file(): return None
        specs = pd.read_csv(spec_path, dtype={"vehicle_id": "string"})
        match = specs[specs.vehicle_id.astype(str) == key]
        if match.empty: return None
        spec = to_json_safe(match.iloc[0].to_dict())
        operational_path = self._path(split, "operational_readouts")
        history = []
        if operational_path.is_file():
            for chunk in pd.read_csv(operational_path, dtype={"vehicle_id": "string"}, chunksize=self.chunksize):
                rows = chunk[chunk.vehicle_id.astype(str) == key]
                if not rows.empty:
                    history.extend(to_json_safe(rows.to_dict(orient="records")))
        history.sort(key=lambda r: (r.get("time_step") is None, r.get("time_step") or 0))
        return {"vehicle_id": key, "split": split, "specifications": spec, "operational_history": history,
                "operational_records": len(history), "data_available": bool(history)}

    def records(self, split: str, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        if split not in self.SPLITS: raise ValueError("Unsupported dataset split")
        if page < 1 or not 1 <= page_size <= 500: raise ValueError("page must be >=1 and page_size between 1 and 500")
        path = self._path(split, "operational_readouts")
        if not path.is_file(): return {"items": [], "page": page, "page_size": page_size, "available": False}
        start, stop = (page - 1) * page_size, page * page_size
        items, seen = [], 0
        for chunk in pd.read_csv(path, chunksize=self.chunksize):
            if seen + len(chunk) <= start: seen += len(chunk); continue
            left = max(0, start - seen); right = min(len(chunk), stop - seen)
            if right > left: items.extend(to_json_safe(chunk.iloc[left:right].to_dict(orient="records")))
            seen += len(chunk)
            if seen >= stop: break
        return {"items": items, "page": page, "page_size": page_size, "available": True, "note": "Operational readout pagination; total row count is available from dataset summary."}
