# Agentic AI Platform for Industrial Operations Intelligence and Decision Support

Predictive maintenance is the first module. This folder **extends** your existing SCANIA project: it reuses your saved
joblib model read-only and adds validated input handling, four cooperating agents (Prediction, Investigation,
Decision/Response, Verification), an orchestrator with an execution trace, a FastAPI backend and a local web UI.
Nothing in your existing folder is modified, overwritten or retrained.

## Status (read this first)

| Item | Status |
|---|---|
| Agents, orchestrator, validation, cost logic, evaluation metrics | Implemented. 118 automated unit + integration/API tests pass using a tiny **synthetic** test model (88 unit, 30 integration). |
| FastAPI layer and web UI | Implemented. API integration tests run with FastAPI TestClient; frontend JavaScript passes Node syntax validation. |
| Reuse of **your** `cost_sensitive_model.joblib` and CSVs | Model artifact and its 113-feature schema were inspected. The supplied validation CSV headers pass `scripts/check_compat.py` on the user's machine. The builder environment has scikit-learn 1.8.0 while the artifact was created with 1.7.2, so real-model inference was not certified in the builder environment. |
| Sample vehicles for the UI | Ten sample JSON payloads are included in `data_samples/`; regenerate from local validation CSVs only if needed. |
| Model quality | Weak. Per your own reports the artifact predicts class 4 for ~99% of official-validation vehicles (see `docs/review2_results_summary.md`). The platform surfaces this; it cannot fix it. |

## What I found in your attached files

* `app.py` (Streamlit): the artifact is a joblib **dict** `{"model": sklearn Pipeline, "feature_cols": [...]}`; the pipeline has a step named `classifier`; input is a one-row DataFrame holding the latest readout merged with `Spec_*` columns. The platform supports exactly this format (and bare estimators that expose `feature_names_in_`).
* `eval_calibration_experiment.py`: official cost matrix and `expected_cost_predict` (probabilities @ matrix, argmin) are reused with identical orientation; 105 numeric + 8 `Spec_*` columns; `Spec_*` cast to text. **It never saves a model**, so the calibrated model is not an artifact you own. Run only if you decide to export it (see "Not included" below).
* `evaluate_official_validation.py`, reports: evaluation protocol is "latest readout per vehicle"; reused by `scripts/evaluate_saved_model.py` and the sample extractor.
* Model features are taken from the artifact's own `feature_cols`; the platform never guesses them. If a feature is `vehicle_id`/`time_step` it warns, if `class_label` it refuses to serve the model.

## Local files the application expects

The large SCANIA CSV files are intentionally excluded from this ZIP. Set `SCANIA_DATA_DIR` to the directory that contains the split CSVs (for example, your existing outer `data` folder). The application searches nearby ancestor layouts when no explicit path is set, but an explicit setting is recommended for collaborators.

| File (in `...\2024-34-3\data`) | Needed for |
|---|---|
| `cost_sensitive_model.joblib` (or the file named in `AGENTIC_MODEL_PATH`) | everything except tests |
| `train_*`, `validation_*`, `test_*` split CSVs (kept outside this ZIP) | Upload/evaluation workflows; full-data evaluation and reproduction |

## Install (PowerShell)

Use the **same Python environment** in which you trained the models. Joblib files are tied to the scikit-learn/numpy/pandas versions that wrote them; do not upgrade those. If `pip` proposes upgrading scikit-learn or numpy, cancel and ask for help.

```powershell
# 1. Extract the archive into a working directory. It creates agentic_platform_updated\agentic_platform.
# Keep the large SCANIA CSVs in their existing external data directory.
Expand-Archive -Path "$HOME\Downloads\agentic_platform_updated.zip" -DestinationPath .

# 2. Activate your existing environment (adjust to how you work; skip if you use system Python)
#    .\.venv\Scripts\Activate.ps1

# 3. Install the few extra packages
cd .\agentic_platform_updated\agentic_platform
python -m pip install -r requirements-agentic.txt

# 4. Allow scripts for this PowerShell window only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 5. Check that your existing model and CSVs are compatible (read-only)
python scripts/check_compat.py

# 6. Create demo samples from REAL validation rows
python scripts/extract_samples.py

# 7. Run the tests (unit = fast, no data; -Integration also runs API tests with a synthetic model)
.\run_tests.ps1
.\run_tests.ps1 -Integration

# 8. Start the demo (backend + browser)
.\run_demo.ps1
#    or backend only: .\run_backend.ps1   then open http://127.0.0.1:8000/  (API docs: /docs)
```

OneDrive tip: make sure the `.joblib` and CSV files are fully downloaded ("Always keep on this device"); placeholder files make loading fail or stall.

To demonstrate another artifact, e.g. the balanced baseline: `.\run_demo.ps1` after copying `.env.example` to `.env` and setting `AGENTIC_MODEL_PATH=balanced_baseline_model.joblib`. Per your own reports it has the lower official-validation cost (9.0535 derived vs 9.7495); decide knowingly.

Optional evaluation of a saved model (computed once, never tune against it; writes only to `agentic_platform\outputs\`):

```powershell
python scripts/evaluate_saved_model.py                                   # default model
python scripts/evaluate_saved_model.py --model balanced_baseline_model.joblib
```

## Configuration (`.env`, all optional)

See `.env.example`: model path, priority thresholds, null-fraction review threshold, replan bound (default 1, max 2).
The default review threshold (≥50% null numeric features → `needs_review`) may put many real readouts into `needs_review`; `extract_samples.py` prints each sample's null fraction so you can see this before the review.

## Layout

```
agentic_platform/
  agentic_app/            # package (named agentic_app, not app: your Streamlit app.py sits one level up)
    main.py  api/routes.py
    agents/   prediction.py investigation.py decision.py verification.py orchestrator.py
    core/     costs.py rules.py config.py
    schemas/types.py      services/{model_service,validation,sample_service,metrics}.py
  frontend/index.html     config/known_results.json   data_samples/   models/   outputs/
  scripts/  check_compat.py extract_samples.py evaluate_saved_model.py
  tests/unit  tests/integration
  docs/  architecture.md api.md methodology.md review2_results_summary.md demo_walkthrough_and_viva.md
  run_backend.ps1 run_demo.ps1 run_tests.ps1  .env.example  requirements-agentic.txt  MANIFEST.md
```

## Deliberate deviations from the original brief

* Package is `agentic_app` (avoids clashing with your `app.py`).
* Stage objects are plain dataclasses (allowed by the brief) rather than Pydantic models, so the whole workflow is testable without web dependencies; FastAPI still validates JSON bodies.
* No LLM integration: the core is deterministic and needs no API keys. A narrator could be added outside the decision path.
* **Not included:** a training/export script. Your artifact exists, and you asked to avoid retraining. If you later want the calibrated model from `eval_calibration_experiment.py` as a file, that needs a small separate export step; ask for it and do not tune on validation labels.
* `models/` is unused by default.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/health` shows `MODEL_FILE_NOT_FOUND` | wrong `AGENTIC_MODEL_PATH`/folder; run `python scripts/check_compat.py` |
| `MODEL_LOAD_FAILED` mentioning versions or `sklearn` | different Python environment than the one used for training |
| 422 `MISSING_FIELD` for a sample | the sample was built with a different model; re-run `extract_samples.py` |
| Port busy | `.\run_demo.ps1 -Port 8010` |
| `ModuleNotFoundError: fastapi` | step 3 was run in a different environment |

## Dataset explorer additions (2026-10-30)

The API now exposes:

- `GET /dataset/summary` — availability and schema for train/validation/test files. Large files are not fully scanned just to count rows.
- `GET /dataset/records?split=validation&page=1&page_size=50` — paginated operational readouts.
- `GET /dataset/vehicles/{vehicle_id}?split=validation` — specifications and all matching operational time steps.

Set `SCANIA_DATA_DIR` to the directory containing the SCANIA CSVs (for this project layout, the sibling `data` folder). The application auto-detects `<project parent>/data` when the old parent-folder setting is stale; it reads the large CSVs in place and does not copy them. The original CSVs remain read-only and outside the project. The training operational readouts CSV is optional for starting the application and for serving the existing saved model; operations that require training data must report it as unavailable if it is absent.

The supplied `models/cost_sensitive_model.joblib` was created with scikit-learn 1.7.2. Use the same scikit-learn version for predictions. Loading it under 1.8.0 emitted an `InconsistentVersionWarning`, and inference failed inside `SimpleImputer` (`_fill_dtype` missing). The artifact loaded, but successful prediction was **not** verified in that newer environment. Do not upgrade/downgrade the environment blindly; use the original training environment or install the artifact's compatible scikit-learn version in a separate environment.

## Optional user dataset uploads (Overview + Vehicle Analysis)

The Overview page supports uploading one or more CSV files or a ZIP containing CSV files. Uploads are stored separately from the original SCANIA source in `<SCANIA_DATA_DIR>/.agentic_uploads/` (or the configured data directory) and are not used to overwrite source files.

New endpoints:

- `POST /uploads` — multipart upload; send one or more files under the `files` field.
- `GET /uploads` — list saved upload datasets.
- `GET /uploads/{dataset_id}/summary` — file/schema/compatibility summary.
- `GET /uploads/{dataset_id}/vehicles?split=validation&page=1&page_size=50` — list vehicle IDs.
- `GET /uploads/{dataset_id}/vehicles/{vehicle_id}?split=validation` — return the latest model-ready vehicle payload.

For prediction, uploaded files must contain the saved model's exact feature names across operational readouts and specification files. Uploads that do not meet this requirement remain available for source inspection but are explicitly marked incompatible and cannot be loaded into Vehicle Analysis for inference. Labels are never copied into the prediction payload. CSV/ZIP upload requires `python-multipart`, included in `requirements-agentic.txt`.

The Overview and Vehicle Analysis pages support this feature. Vehicle Analysis has a scoped responsive layout refresh; the other six frontend pages and the existing four-stage agent workflow are unchanged. Upload split detection now reports a recommended split and falls back to an available split (for example, `train`) when the UI's default `validation` split is absent. Compatibility checks separately require numeric model features in operational readouts and `Spec_*` features in the specification file (or a complete combined-readout file).


## Capstone handoff notes (2026-10-30)

- The upload API stores user uploads under `<SCANIA_DATA_DIR>/.agentic_uploads/`, outside this project folder. Keep the same `SCANIA_DATA_DIR` value when moving to this release so existing uploads remain visible.
- A train-only upload is valid: the service reports `recommended_split: "train"` and vehicle browsing automatically falls back to that available split when the UI requests validation by default.
- Compatibility means the exact 113 model feature columns are present in the expected operational/specification files. It does not certify that synthetic sensor values are realistic or that the model is accurate.
- The supplied model artifact was created with scikit-learn 1.7.2. Do not install scikit-learn 1.8.x into the environment used to serve this artifact. Use the original training environment; the builder runtime (1.8.0) can load the artifact but its inference path fails due to a scikit-learn internal attribute mismatch.
- Official-validation performance is weak and strongly biased toward class 4, as described in the model's known-results notes and `docs/review2_results_summary.md`. Present this limitation transparently in the capstone.

## Frontend refresh (2 November 2026)

The browser interface is split into `frontend/index.html`, `frontend/app.css`, and `frontend/app.js`. It is served by the existing FastAPI app at `/`, with static assets under `/assets/`. No frontend package install or external chart library is required; charts are drawn with the browser Canvas API so the dashboard works offline.

The refreshed interface uses the calibrated/indexed API routes:

- `/analytics/{split}` and `/results/model-comparison` for population and policy analytics
- `/vehicles` and `/vehicles/{split}/{vehicle_id}` for indexed vehicle search and feature inspection
- `/analysis/{split}/{vehicle_id}` for the complete four-agent workflow
- `/results/batch-verification/{split}` for saved batch consistency results
- `/calibration`, `/health`, and `/index/status` for system status

The Advanced Input page retains real sample loading, JSON input, prediction-only (`/predict`), full workflow (`/workflow/run`), and CSV/ZIP dataset upload (`/uploads`). The evaluation-only recorded label is shown separately after a vehicle workflow and is never included in the agent request.

The browser-session activity list is intentionally labelled session-only; the backend does not persist individual interactive runs. The UI does not claim that software verification proves model correctness or physical equipment safety. Maintenance actions remain recommendations and require human approval.

### Frontend verification

- `node --check frontend/app.js` — syntax check
- `python -m pytest -q tests/unit tests/integration` — backend test suite
- API smoke-tested: `/health`, `/calibration`, `/index/status`, both `/analytics` splits, `/results/model-comparison`, both batch-verification routes, `/vehicles`, and `/samples`.

The frontend was not browser-automated in this build environment. End-to-end POST workflow testing against the saved model must use the project's pinned scikit-learn 1.7.2 environment; this build environment has scikit-learn 1.8.0 and emits an artifact-version warning for the model saved with 1.7.2.
