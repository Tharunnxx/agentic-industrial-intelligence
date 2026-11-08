"""FastAPI application factory. Routes are thin: all logic lives in the orchestrator and services."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import UploadFile

from ..agents.orchestrator import Orchestrator
from ..core.config import Settings
import json

from ..core.calibration import CalibrationError, Calibrator
from ..services.model_service import ModelService, ModelUnavailable
from ..services.results_service import ResultsService
from ..services.vehicle_index import IndexUnavailable, VehicleIndex
from ..services.sample_service import SampleService
from ..services.dataset_service import DatasetService
from ..services.evaluation_service import EvaluationService, EvaluationError
from ..services.upload_service import UploadDatasetService, UploadDatasetError
from .. import __version__

STATUS_BY_CODE = {
    "INPUT_VALIDATION_FAILED": 422, "MODEL_FILE_NOT_FOUND": 503, "MODEL_LOAD_FAILED": 503, "MODEL_ARTIFACT_UNSUPPORTED": 503,
    "MODEL_CLASSES_MISMATCH": 503, "MODEL_USES_TARGET": 503, "MODEL_UNAVAILABLE": 503, "PREDICTION_FAILED": 500, "UNEXPECTED_ERROR": 500,
}


def create_app(settings: Optional[Settings] = None, model_service: Optional[ModelService] = None,
               orchestrator: Optional[Orchestrator] = None):
    from fastapi import Body, FastAPI, File, UploadFile
    from fastapi.responses import FileResponse, JSONResponse

    settings = settings or Settings.from_env()
    model_service = model_service or ModelService(settings)
    calibrator, calibration_error = None, None
    if orchestrator is not None:
        calibrator = getattr(orchestrator, "calibrator", None)
    elif settings.use_calibration and settings.calibration_path and settings.calibration_path.is_file():
        try:
            calibrator = Calibrator.load(settings.calibration_path, settings.calibration_reference_path, settings.results_dir)
        except (CalibrationError, ValueError, KeyError, OSError) as exc:
            calibration_error = str(exc)  # the app still works, with raw probabilities, and says so on /calibration
    feature_reference = None
    if settings.feature_reference_path and settings.feature_reference_path.is_file():
        try:
            feature_reference = json.loads(settings.feature_reference_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            feature_reference = None
    orchestrator = orchestrator or Orchestrator(settings, model_service, calibrator=calibrator, feature_reference=feature_reference)
    index = VehicleIndex(settings.index_dir or (settings.platform_root / "data_index"), settings, calibrator)
    results = ResultsService(settings.results_dir or (settings.platform_root / "config" / "results"))
    samples = SampleService(settings)
    dataset = DatasetService(settings.data_dir)
    evaluation = EvaluationService(settings.data_dir, model_service)
    # Keep uploads in their existing project-level location when the source CSV
    # directory is changed (for example, from project root to sibling /data).
    upload_root = settings.data_dir / ".agentic_uploads"
    uploads = UploadDatasetService(upload_root, model_service)
    app = FastAPI(title="Agentic AI Industrial Operations Intelligence", version=__version__,
                  description="Predictive-maintenance module: prediction, investigation, decision and verification agents.")

    def result_response(result):
        code = STATUS_BY_CODE.get(result.error.code, 500) if result.error else 200
        return JSONResponse(result.to_dict(), status_code=code)

    @app.get("/health")
    def health():
        loaded = model_service.ensure_loaded()
        body = {"status": "ok", "version": __version__, "model_loaded": loaded}
        if not loaded and model_service.error:
            body["model_error"] = {"code": model_service.error.code, "message": str(model_service.error)}
        return body

    @app.get("/model/info")
    def model_info():
        try:
            return model_service.info()
        except ModelUnavailable as exc:
            return JSONResponse({"error": {"code": exc.code, "message": str(exc)}}, status_code=503)

    @app.get("/model/evaluate")
    def model_evaluate(split: str = "validation"):
        try:
            return evaluation.evaluate(split)
        except EvaluationError as exc:
            return JSONResponse(
                {"error": {"code": exc.code, "message": str(exc)}},
                status_code=exc.status_code,
            )

    @app.post("/predict")
    def predict(payload: Any = Body(...)):
        return result_response(orchestrator.predict(payload))

    @app.post("/workflow/run")
    def workflow_run(payload: Any = Body(...)):
        return result_response(orchestrator.run_workflow(payload))

    @app.get("/samples")
    def list_samples():
        items = samples.list()
        return {"samples": items, "hint": None if items else "No samples found. Run: python scripts\\extract_samples.py"}

    @app.get("/samples/{sample_id}")
    def get_sample(sample_id: str):
        data = samples.get(sample_id)
        if data is None:
            return JSONResponse({"error": {"code": "SAMPLE_NOT_FOUND", "message": f"No sample '{sample_id}'."}}, status_code=404)
        return data

    @app.get("/uploads")
    def list_uploads():
        return {"datasets": uploads.list()}

    @app.post("/uploads")
    def upload_datasets(files: list[UploadFile] = File(...)):
        try:
            return uploads.create(files)
        except UploadDatasetError as exc:
            return JSONResponse({"error": {"code": "UPLOAD_VALIDATION_FAILED", "message": str(exc)}}, status_code=422)
        finally:
            for item in files:
                try:
                    item.file.close()
                except Exception:
                    pass

    @app.get("/uploads/{dataset_id}/summary")
    def uploaded_dataset_summary(dataset_id: str):
        try:
            return uploads.summary(dataset_id)
        except UploadDatasetError as exc:
            return JSONResponse({"error": {"code": "UPLOADED_DATASET_ERROR", "message": str(exc)}}, status_code=404)

    @app.get("/uploads/{dataset_id}/vehicles")
    def uploaded_dataset_vehicles(dataset_id: str, split: str = "validation", page: int = 1, page_size: int = 50):
        try:
            return uploads.vehicles(dataset_id, split, page, page_size)
        except UploadDatasetError as exc:
            return JSONResponse({"error": {"code": "UPLOADED_DATASET_ERROR", "message": str(exc)}}, status_code=422)

    @app.get("/uploads/{dataset_id}/vehicles/{vehicle_id}")
    def uploaded_vehicle_payload(dataset_id: str, vehicle_id: str, split: str = "validation"):
        try:
            return uploads.vehicle_payload(dataset_id, vehicle_id, split)
        except UploadDatasetError as exc:
            return JSONResponse({"error": {"code": "UPLOADED_VEHICLE_ERROR", "message": str(exc)}}, status_code=422)

    @app.get("/dataset/summary")
    def dataset_summary():
        return dataset.summary()

    @app.get("/dataset/records")
    def dataset_records(split: str = "validation", page: int = 1, page_size: int = 50):
        try:
            return dataset.records(split, page, page_size)
        except ValueError as exc:
            return JSONResponse({"error": {"code": "INVALID_DATASET_QUERY", "message": str(exc)}}, status_code=422)

    @app.get("/dataset/vehicles/{vehicle_id}")
    def dataset_vehicle(vehicle_id: str, split: str = "validation"):
        try:
            item = dataset.vehicle(split, vehicle_id)
        except ValueError as exc:
            return JSONResponse({"error": {"code": "INVALID_DATASET_QUERY", "message": str(exc)}}, status_code=422)
        if item is None:
            return JSONResponse({"error": {"code": "VEHICLE_NOT_FOUND", "message": f"Vehicle {vehicle_id} not found in {split}."}}, status_code=404)
        return item

    # ------------------------------------------------------------------ calibrated workflow, vehicles, analytics
    def _index_error(exc):
        if isinstance(exc, IndexUnavailable):
            return JSONResponse({"error": {"code": "INDEX_NOT_BUILT", "message": str(exc)}}, status_code=503)
        return JSONResponse({"error": {"code": "INVALID_QUERY", "message": str(exc)}}, status_code=422)

    @app.get("/calibration")
    def calibration_info():
        if calibrator is None:
            why = calibration_error or ("Calibration is switched off (AGENTIC_CALIBRATION=off)." if not settings.use_calibration else "No calibration file was found.")
            return {"enabled": False, "reason": why, "note": "Decisions use the raw model probabilities, which overstate failure risk."}
        return {"enabled": True, **calibrator.describe(), "priority_bands": {
            "high_at_calibrated_p_any": settings.cal_high_p_any, "critical_at_calibrated_p_any": settings.cal_critical_p_any}}

    @app.get("/index/status")
    def index_status():
        return index.status()

    @app.get("/vehicles")
    def vehicles(split: str = "validation", q: str = "", sort: str = "risk", priority: Optional[str] = None, page: int = 1, page_size: int = 25):
        try:
            return index.search(split, q, sort, priority, page, page_size)
        except (IndexUnavailable, ValueError) as exc:
            return _index_error(exc)

    @app.get("/vehicles/{split}/{vehicle_id}")
    def vehicle_payload(split: str, vehicle_id: str):
        try:
            data = index.payload(split, vehicle_id)
        except (IndexUnavailable, ValueError) as exc:
            return _index_error(exc)
        if data is None:
            return JSONResponse({"error": {"code": "VEHICLE_NOT_FOUND", "message": f"Vehicle {vehicle_id} not found in {split}."}}, status_code=404)
        return data

    @app.post("/analysis/{split}/{vehicle_id}")
    def analyse_vehicle(split: str, vehicle_id: str):
        try:
            payload = index.payload(split, vehicle_id)
        except (IndexUnavailable, ValueError) as exc:
            return _index_error(exc)
        if payload is None:
            return JSONResponse({"error": {"code": "VEHICLE_NOT_FOUND", "message": f"Vehicle {vehicle_id} not found in {split}."}}, status_code=404)
        result = orchestrator.run_workflow(payload)
        body = result.to_dict()
        label = None
        try:
            label = index.reference_label(split, vehicle_id)
        except IndexUnavailable:
            pass
        body["evaluation_only"] = {"split": split, "recorded_class_label": label,
                                   "note": "The recorded label is shown for comparison after the run. It was never given to any agent."}
        return JSONResponse(body, status_code=STATUS_BY_CODE.get(result.error.code, 500) if result.error else 200)

    @app.get("/analytics/{split}")
    def analytics(split: str):
        try:
            return index.analytics(split)
        except (IndexUnavailable, ValueError) as exc:
            return _index_error(exc)

    @app.get("/results/model-comparison")
    def model_comparison():
        return results.model_comparison()

    @app.get("/results/batch-verification/{split}")
    def batch_verification(split: str):
        try:
            data = index.batch_summary(split)
        except ValueError as exc:
            return _index_error(exc)
        if data is None:
            return JSONResponse({"error": {"code": "BATCH_SUMMARY_MISSING", "message": f"Run: python scripts\\run_batch_workflow.py --split {split}"}}, status_code=404)
        return data

    @app.get("/", include_in_schema=False)
    def index_page():
        page = settings.frontend_dir / "index.html"
        if not page.is_file():
            return JSONResponse({"error": {"code": "FRONTEND_MISSING", "message": "frontend/index.html not found."}}, status_code=404)
        return FileResponse(page)

    if settings.frontend_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/assets", StaticFiles(directory=str(settings.frontend_dir)), name="assets")
    return app
