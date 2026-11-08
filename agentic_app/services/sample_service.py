"""Serves sample readouts created by scripts/extract_samples.py from REAL dataset rows. Never generates data."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..core.config import Settings

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


class SampleService:
    def __init__(self, settings: Settings):
        self.dir = settings.samples_dir

    def list(self) -> List[Dict[str, Any]]:
        if not self.dir.is_dir():
            return []
        out = []
        for path in sorted(self.dir.glob("sample_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = data.get("meta", {})
            out.append({"id": path.stem, "vehicle_id": data.get("vehicle_id"), "time_step": data.get("time_step"),
                        "source": meta.get("source"), "missing_fraction": meta.get("missing_fraction")})
        return out

    def get(self, sample_id: str) -> Optional[Dict[str, Any]]:
        if not _SAFE_ID.match(sample_id):
            return None
        path = self.dir / f"{sample_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
