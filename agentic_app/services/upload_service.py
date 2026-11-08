"""Persistent, validated user dataset uploads for the Overview and Vehicle Analysis pages."""
from __future__ import annotations

import json
import re
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd

from .dataset_service import to_json_safe


class UploadDatasetError(ValueError):
    pass


class UploadDatasetService:
    MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB per submitted file
    MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
    SPLITS = ("train", "validation", "test")

    def __init__(self, root: Path, model_service):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_service = model_service

    def _schema(self):
        if not self.model_service.ensure_loaded():
            raise UploadDatasetError("The saved prediction model is unavailable. Uploads can be stored, but model compatibility cannot be checked until the model loads.")
        return self.model_service.schema()

    @staticmethod
    def _safe_name(name: str) -> str:
        base = PurePosixPath(str(name).replace('\\', '/')).name
        base = re.sub(r'[^A-Za-z0-9._-]+', '_', base).strip('._')
        if not base or base in {'.', '..'}:
            raise UploadDatasetError("An uploaded file has an invalid filename.")
        return base[:180]

    @staticmethod
    def _role(filename: str, columns: list[str], schema=None) -> tuple[str | None, str | None]:
        name = Path(filename).stem.lower()
        split = next((s for s in UploadDatasetService.SPLITS if name.startswith(s + "_") or name.endswith("_" + s)), None)
        if "operational_readouts" in name or ("readout" in name and "vehicle_id" in columns and "time_step" in columns):
            return split, "operational_readouts"
        if "vehicle_id" in columns and "time_step" in columns and schema is not None:
            overlap = set(columns) & set(schema.numeric_cols)
            spec_overlap = set(columns) & set(schema.spec_cols)
            if overlap and not spec_overlap:
                return split, "operational_readouts"
            if set(schema.feature_cols).issubset(set(columns)):
                return split, "combined_readouts"
        if "specifications" in name or ("vehicle_id" in columns and "time_step" not in columns and any(c.startswith("Spec_") for c in columns)):
            return split, "specifications"
        if "labels" in name or ("vehicle_id" in columns and any(c in columns for c in ("class_label", "label", "target"))):
            return split, "labels"
        if "tte" in name or ("vehicle_id" in columns and any("tte" in c.lower() for c in columns)):
            return split, "tte"
        if "vehicle_id" in columns and "time_step" in columns:
            return split, "combined_readouts"
        return split, None

    def create(self, uploads: Iterable[Any]) -> dict[str, Any]:
        schema = self._schema()
        dataset_id = uuid.uuid4().hex[:12]
        folder = self.root / dataset_id
        folder.mkdir(parents=True, exist_ok=False)
        saved: list[Path] = []
        total_bytes = 0
        try:
            for upload in uploads:
                filename = self._safe_name(upload.filename or "upload.csv")
                data_file = folder / filename
                upload.file.seek(0, 2)
                size = upload.file.tell()
                upload.file.seek(0)
                if size <= 0 or size > self.MAX_UPLOAD_BYTES:
                    raise UploadDatasetError(f"{filename}: file is empty or exceeds the 2 GiB upload limit.")
                total_bytes += size
                if total_bytes > self.MAX_UPLOAD_BYTES:
                    raise UploadDatasetError("Combined upload exceeds the 2 GiB request limit.")
                with data_file.open("wb") as out:
                    shutil.copyfileobj(upload.file, out, length=1024 * 1024)
                if data_file.suffix.lower() == ".zip":
                    with zipfile.ZipFile(data_file) as archive:
                        expanded = 0
                        for info in archive.infolist():
                            if info.is_dir() or not info.filename.lower().endswith(".csv"):
                                continue
                            pure = PurePosixPath(info.filename.replace('\\', '/'))
                            if pure.is_absolute() or ".." in pure.parts:
                                raise UploadDatasetError("ZIP contains an unsafe file path.")
                            expanded += info.file_size
                            if expanded > self.MAX_UNCOMPRESSED_BYTES:
                                raise UploadDatasetError("ZIP expands beyond the 4 GiB safety limit.")
                            target = folder / self._safe_name(pure.name)
                            with archive.open(info) as src, target.open("wb") as dst:
                                shutil.copyfileobj(src, dst, length=1024 * 1024)
                            saved.append(target)
                    data_file.unlink(missing_ok=True)
                elif data_file.suffix.lower() == ".csv":
                    saved.append(data_file)
                else:
                    raise UploadDatasetError(f"Unsupported file type: {filename}. Upload CSV or ZIP files.")

            if not saved:
                raise UploadDatasetError("No CSV files were found in the upload.")

            files: dict[str, str] = {}
            file_meta: list[dict[str, Any]] = []
            for path in saved:
                try:
                    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
                except Exception as exc:
                    raise UploadDatasetError(f"Could not read {path.name}: {type(exc).__name__}: {exc}") from exc
                split, role = self._role(path.name, header, schema)
                if role is None:
                    raise UploadDatasetError(f"Could not identify the purpose of {path.name}. Include vehicle_id/time_step for readouts or use standard SCANIA filenames.")
                key = f"{split}_{role}" if split else role
                if key in files:
                    raise UploadDatasetError(f"More than one file was supplied for {key}. Upload one file per dataset role/split.")
                files[key] = f"{dataset_id}/{path.name}"
                file_meta.append({"name": path.name, "role": role, "split": split or "generic", "columns": len(header), "size_bytes": path.stat().st_size})

            compatible, compatibility = self._compatibility(folder, files, schema)
            metadata = {"dataset_id": dataset_id, "created_at": pd.Timestamp.utcnow().isoformat(), "files": files,
                        "file_meta": file_meta, "compatible": compatible, "compatibility": compatibility}
            (folder / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return self.summary(dataset_id)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def _resolve(self, files: dict[str, str], split: str, role: str) -> Path | None:
        name = files.get(f"{split}_{role}") or files.get(role)
        return self.root / name if name else None

    def _compatibility(self, folder: Path, files: dict[str, str], schema) -> tuple[bool, dict[str, Any]]:
        """Check model features in their correct source files, not just their union.

        Numeric sensor features must come from operational readouts; Spec_* features
        must come from the specifications file (or a complete combined readout).
        """
        details: list[dict[str, Any]] = []
        found_pair = False
        for split in self.SPLITS:
            ops = self._resolve(files, split, "operational_readouts") or self._resolve(files, split, "combined_readouts")
            specs = self._resolve(files, split, "specifications")
            if not ops or not ops.is_file():
                continue

            op_cols = pd.read_csv(ops, nrows=0).columns.astype(str).tolist()
            combined_path = self._resolve(files, split, "combined_readouts")
            is_combined = combined_path is not None and ops == combined_path
            if specs and specs.is_file():
                spec_cols = pd.read_csv(specs, nrows=0).columns.astype(str).tolist()
            else:
                spec_cols = []

            if is_combined:
                missing_numeric = [c for c in schema.numeric_cols if c not in op_cols]
                missing_specs = [c for c in schema.spec_cols if c not in op_cols]
            else:
                missing_numeric = [c for c in schema.numeric_cols if c not in op_cols]
                missing_specs = [c for c in schema.spec_cols if c not in spec_cols]

            missing = missing_numeric + missing_specs
            identifiers_ok = "vehicle_id" in op_cols and "time_step" in op_cols
            spec_identifier_ok = not schema.spec_cols or is_combined or (specs is not None and "vehicle_id" in spec_cols)
            details.append({
                "split": split,
                "missing_model_features": missing[:30],
                "missing_count": len(missing),
                "missing_numeric_features": missing_numeric[:15],
                "missing_specification_features": missing_specs[:15],
                "operational_identifiers_present": identifiers_ok,
                "specification_vehicle_id_present": spec_identifier_ok,
            })
            if not missing and identifiers_ok and spec_identifier_ok:
                found_pair = True

        message = (
            "All required model features are present in the correct readout/specification files."
            if found_pair else
            "The upload does not contain all model-required features in the expected files. "
            "Operational sensor features must be in the readout CSV and Spec_* features in the "
            "matching specifications CSV. The dataset can be explored, but prediction is disabled."
        )
        return found_pair, {
            "model_feature_count": len(schema.feature_cols),
            "numeric_feature_count": len(schema.numeric_cols),
            "specification_feature_count": len(schema.spec_cols),
            "checks": details,
            "message": message,
        }

    def _effective_split(self, files: dict[str, str], requested: str) -> str:
        """Use the requested split when present; otherwise select an available split.

        The UI defaults to validation, but a valid upload may contain only train
        or test files. Falling back here prevents a compatible train-only ZIP from
        appearing to contain zero vehicles.
        """
        roles = ("operational_readouts", "combined_readouts")
        if any(self._resolve(files, requested, role) is not None for role in roles):
            return requested
        for candidate in ("validation", "train", "test"):
            if any(self._resolve(files, candidate, role) is not None for role in roles):
                return candidate
        return requested

    def _metadata(self, dataset_id: str) -> tuple[Path, dict[str, Any]]:
        if not re.fullmatch(r"[a-f0-9]{12}", dataset_id or ""):
            raise UploadDatasetError("Invalid dataset ID.")
        folder = self.root / dataset_id
        path = folder / "dataset.json"
        if not path.is_file():
            raise UploadDatasetError("Uploaded dataset was not found.")
        return folder, json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        items = []
        for p in sorted(self.root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_dir() and (p / "dataset.json").is_file():
                try:
                    m = json.loads((p / "dataset.json").read_text(encoding="utf-8"))
                    items.append({"dataset_id": m["dataset_id"], "created_at": m.get("created_at"), "compatible": m.get("compatible", False), "file_count": len(m.get("files", {}))})
                except Exception:
                    continue
        return items[:30]

    def summary(self, dataset_id: str) -> dict[str, Any]:
        folder, m = self._metadata(dataset_id)
        schema = self._schema()
        splits: dict[str, Any] = {}
        for split in self.SPLITS:
            role_data = {}
            for role in ("operational_readouts", "specifications", "labels", "tte"):
                p = self._resolve(m["files"], split, role) or (self._resolve(m["files"], split, "combined_readouts") if role == "operational_readouts" else None)
                if p and p.is_file():
                    header = pd.read_csv(p, nrows=0).columns.astype(str).tolist()
                    size = p.stat().st_size
                    rows = None if size > 100 * 1024 * 1024 else max(0, sum(1 for _ in p.open("rb")) - 1)
                    role_data[role] = {"available": True, "rows": rows, "row_count_deferred": rows is None, "columns": len(header), "size_bytes": size, "column_names": header}
            if role_data:
                splits[split] = role_data
        recommended_split = next((s for s in ("validation", "train", "test") if
                                   self._resolve(m["files"], s, "operational_readouts") or
                                   self._resolve(m["files"], s, "combined_readouts")), "validation")
        return {"dataset_id": m["dataset_id"], "created_at": m.get("created_at"), "files": m.get("file_meta", []),
                "compatible": m.get("compatible", False), "compatibility": m.get("compatibility", {}),
                "splits": splits, "available_splits": list(splits), "recommended_split": recommended_split,
                "total_files": len(m.get("files", {})), "data_dir": str(folder)}

    def vehicles(self, dataset_id: str, split: str = "validation", page: int = 1, page_size: int = 50) -> dict[str, Any]:
        if split not in self.SPLITS or page < 1 or not 1 <= page_size <= 200:
            raise UploadDatasetError("Invalid split or pagination values.")
        folder, m = self._metadata(dataset_id)
        split = self._effective_split(m["files"], split)
        p = self._resolve(m["files"], split, "operational_readouts") or self._resolve(m["files"], split, "combined_readouts")
        if p is None:
            return {"items": [], "page": page, "page_size": page_size, "available": False}
        ids: list[str] = []
        seen: set[str] = set()
        for chunk in pd.read_csv(p, usecols=["vehicle_id"], dtype={"vehicle_id": "string"}, chunksize=100_000):
            for value in chunk.vehicle_id.dropna().astype(str):
                if value not in seen:
                    seen.add(value); ids.append(value)
        start = (page - 1) * page_size
        selected = ids[start:start + page_size]
        return {"items": [{"vehicle_id": x} for x in selected], "page": page, "page_size": page_size,
                "total_vehicles": len(ids), "has_next": start + page_size < len(ids), "available": True,
                "split": split}

    def vehicle_payload(self, dataset_id: str, vehicle_id: str, split: str = "validation") -> dict[str, Any]:
        folder, m = self._metadata(dataset_id)
        schema = self._schema()
        split = self._effective_split(m["files"], split)
        ops_path = self._resolve(m["files"], split, "operational_readouts") or self._resolve(m["files"], split, "combined_readouts")
        spec_path = self._resolve(m["files"], split, "specifications")
        if ops_path is None:
            raise UploadDatasetError("No operational readout CSV is available for this split.")
        latest = None
        matched_records = 0
        usecols = None
        header = pd.read_csv(ops_path, nrows=0).columns.astype(str).tolist()
        op_features = [c for c in schema.feature_cols if c in header]
        usecols = list(dict.fromkeys([c for c in ["vehicle_id", "time_step"] + op_features if c in header]))
        for chunk in pd.read_csv(ops_path, usecols=usecols, dtype={"vehicle_id": "string"}, chunksize=100_000):
            rows = chunk[chunk.vehicle_id.astype(str) == str(vehicle_id)]
            if not rows.empty:
                matched_records += len(rows)
                if latest is None:
                    latest = rows
                else:
                    latest = pd.concat([latest, rows], ignore_index=True)
        if latest is None or latest.empty:
            raise UploadDatasetError(f"Vehicle {vehicle_id} was not found in uploaded operational readings.")
        latest = to_json_safe(latest.sort_values("time_step", kind="mergesort").iloc[-1].to_dict())
        features = {c: latest.get(c) for c in op_features}
        specs: dict[str, Any] = {}
        spec_cols = [c for c in schema.feature_cols if c not in features]
        if spec_cols and spec_path and spec_path.is_file():
            header = pd.read_csv(spec_path, nrows=0).columns.astype(str).tolist()
            wanted = [c for c in ["vehicle_id"] + spec_cols if c in header]
            for chunk in pd.read_csv(spec_path, usecols=wanted, dtype={"vehicle_id": "string"}, chunksize=100_000):
                rows = chunk[chunk.vehicle_id.astype(str) == str(vehicle_id)]
                if not rows.empty:
                    specs = to_json_safe(rows.iloc[0].to_dict()); break
        features.update({c: specs.get(c) for c in spec_cols})
        missing = [c for c in schema.feature_cols if c not in features]
        if missing:
            raise UploadDatasetError(f"Vehicle is missing {len(missing)} model feature columns: {missing[:8]}")
        payload = {"vehicle_id": str(vehicle_id), "time_step": float(latest.get("time_step")) if latest.get("time_step") is not None else None,
                   "features": features}
        return {"payload": payload, "operational_records_for_vehicle": matched_records,
                "specifications": specs, "compatible": bool(m.get("compatible")), "split": split}
