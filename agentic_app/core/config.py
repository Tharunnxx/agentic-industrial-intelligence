"""Configuration, read from environment variables (and an optional ``.env`` file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

PLATFORM_ROOT = Path(__file__).resolve().parents[2]  # .../agentic_platform


def _read_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _float(env: Mapping[str, str], key: str, default: float, lo: float, hi: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}.") from exc
    if not lo <= value <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}, got {value}.")
    return value


@dataclass(frozen=True)
class Settings:
    platform_root: Path
    data_dir: Path
    model_path: Path
    samples_dir: Path
    known_results_path: Path
    frontend_dir: Path
    max_missing_fraction: float = 0.90      # reject a readout if more numeric features are null
    review_missing_fraction: float = 0.50   # at/above this -> verification returns needs_review
    max_replans: int = 1                    # bounded replanning (hard cap 2)
    critical_p_high: float = 0.30           # P(class 3 or 4) >= this -> priority "critical"
    high_p_any: float = 0.50                # P(any failure) >= this -> priority "high"
    medium_p_any: float = 0.20              # P(any failure) >= this -> priority "medium"
    # --- calibrated priority bands (used only when the calibration layer is on) ---
    cal_high_p_any: float = 0.05            # calibrated P(any failure) >= this -> "high"  (about 2x the 2.7% validation base rate)
    cal_critical_p_any: float = 0.10        # calibrated P(any failure) >= this -> "critical" (about 4x the base rate)
    use_calibration: bool = True
    calibration_path: Optional[Path] = None
    calibration_reference_path: Optional[Path] = None
    feature_reference_path: Optional[Path] = None
    results_dir: Optional[Path] = None
    index_dir: Optional[Path] = None

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None, dotenv_path: Optional[Path] = None) -> "Settings":
        env: Dict[str, str] = {}
        if environ is None:
            env.update(_read_dotenv(dotenv_path or PLATFORM_ROOT / ".env"))
            env.update(os.environ)
        else:
            env.update(environ)

        configured_data = env.get("SCANIA_DATA_DIR") or env.get("AGENTIC_DATA_DIR")
        expected = ("test_operational_readouts.csv", "validation_operational_readouts.csv",
                    "train_operational_readouts.csv")

        def _has_split_readouts(folder: Path) -> bool:
            return folder.is_dir() and any((folder / name).is_file() for name in expected)

        if configured_data:
            data_dir = Path(configured_data).expanduser().resolve()
            # Resolve older relative settings and common project layouts without
            # copying the large SCANIA CSV files. Search the configured directory,
            # its /data child, and nearby ancestors (including the user's layout:
            # .../data/agentic_platform_updated/agentic_platform).
            candidates = [data_dir, data_dir / "data"]
            for ancestor in (data_dir, *data_dir.parents):
                candidates.extend([ancestor, ancestor / "data"])
            for candidate in candidates:
                if _has_split_readouts(candidate):
                    data_dir = candidate.resolve()
                    break
        else:
            # Search upward from this project for the first directory containing
            # the split readout files. This supports both sibling-data and
            # ancestor-data layouts.
            candidates = []
            for ancestor in (PLATFORM_ROOT, *PLATFORM_ROOT.parents):
                candidates.extend([ancestor / "data", ancestor])
            data_dir = next((p.resolve() for p in candidates if _has_split_readouts(p)), PLATFORM_ROOT.parent.resolve())
        data_dir = data_dir.resolve()

        def _path(key: str, default: Path) -> Path:
            raw = env.get(key)
            if not raw:
                return default
            p = Path(raw).expanduser()
            return p if p.is_absolute() else (data_dir / p)

        max_replans_raw = env.get("AGENTIC_MAX_REPLANS", "1") or "1"
        try:
            max_replans = int(max_replans_raw)
        except ValueError as exc:
            raise ValueError(f"AGENTIC_MAX_REPLANS must be an integer, got {max_replans_raw!r}.") from exc
        if not 0 <= max_replans <= 2:
            raise ValueError("AGENTIC_MAX_REPLANS must be 0, 1 or 2 (replanning is deliberately bounded).")

        settings = cls(
            platform_root=PLATFORM_ROOT,
            data_dir=data_dir,
            model_path=_path("AGENTIC_MODEL_PATH", (PLATFORM_ROOT / "models" / "cost_sensitive_model.joblib") if (PLATFORM_ROOT / "models" / "cost_sensitive_model.joblib").is_file() else data_dir / "cost_sensitive_model.joblib"),
            samples_dir=_path("AGENTIC_SAMPLES_DIR", PLATFORM_ROOT / "data_samples"),
            known_results_path=_path("AGENTIC_KNOWN_RESULTS", PLATFORM_ROOT / "config" / "known_results.json"),
            frontend_dir=PLATFORM_ROOT / "frontend",
            max_missing_fraction=_float(env, "AGENTIC_MAX_MISSING_FRACTION", 0.90, 0.0, 1.0),
            review_missing_fraction=_float(env, "AGENTIC_REVIEW_MISSING_FRACTION", 0.50, 0.0, 1.0),
            max_replans=max_replans,
            critical_p_high=_float(env, "AGENTIC_CRITICAL_P_HIGH", 0.30, 0.0, 1.0),
            high_p_any=_float(env, "AGENTIC_HIGH_P_ANY", 0.50, 0.0, 1.0),
            medium_p_any=_float(env, "AGENTIC_MEDIUM_P_ANY", 0.20, 0.0, 1.0),
            cal_high_p_any=_float(env, "AGENTIC_CAL_HIGH_P_ANY", 0.05, 0.0, 1.0),
            cal_critical_p_any=_float(env, "AGENTIC_CAL_CRITICAL_P_ANY", 0.10, 0.0, 1.0),
            use_calibration=(env.get("AGENTIC_CALIBRATION", "on") or "on").strip().lower() not in ("off", "0", "false", "no"),
            calibration_path=_path("AGENTIC_CALIBRATION_FILE", PLATFORM_ROOT / "config" / "calibration" / "m2_calibrator.json"),
            calibration_reference_path=_path("AGENTIC_CALIBRATION_REFERENCE", PLATFORM_ROOT / "config" / "calibration" / "validation_risk_reference.json"),
            feature_reference_path=_path("AGENTIC_FEATURE_REFERENCE", PLATFORM_ROOT / "data_index" / "feature_reference.json"),
            results_dir=_path("AGENTIC_RESULTS_DIR", PLATFORM_ROOT / "config" / "results"),
            index_dir=_path("AGENTIC_INDEX_DIR", PLATFORM_ROOT / "data_index"),
        )
        if settings.medium_p_any > settings.high_p_any:
            raise ValueError("AGENTIC_MEDIUM_P_ANY must not exceed AGENTIC_HIGH_P_ANY.")
        if settings.cal_high_p_any > settings.cal_critical_p_any:
            raise ValueError("AGENTIC_CAL_HIGH_P_ANY must not exceed AGENTIC_CAL_CRITICAL_P_ANY.")
        return settings

    def rule_config(self) -> Dict[str, float]:
        """Thresholds that influence decisions (echoed in every workflow result)."""
        return {
            "critical_p_high": self.critical_p_high,
            "high_p_any": self.high_p_any,
            "medium_p_any": self.medium_p_any,
            "review_missing_fraction": self.review_missing_fraction,
            "max_missing_fraction": self.max_missing_fraction,
            "max_replans": self.max_replans,
            "cal_high_p_any": self.cal_high_p_any,
            "cal_critical_p_any": self.cal_critical_p_any,
            "calibration_enabled": self.use_calibration,
        }
