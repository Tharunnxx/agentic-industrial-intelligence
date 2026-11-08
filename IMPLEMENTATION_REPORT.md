# Implementation report — 2026-10-30

## Completed in this workspace
- Preserved the existing FastAPI application, four deterministic agent stages, existing UI, model service, and original test suite.
- Added memory-conscious SCANIA dataset service for split/file availability, dataset summaries, paginated operational records, and vehicle specifications plus operational history.
- Added API routes: `/dataset/summary`, `/dataset/records`, and `/dataset/vehicles/{vehicle_id}`.
- Added unit tests for vehicle history, pagination, and invalid pagination.
- Included the separately supplied `cost_sensitive_model.joblib` under `models/`.
- Added `SCANIA_DATA_DIR` support while preserving `AGENTIC_DATA_DIR` compatibility.
- Large operational files are not fully scanned merely to produce the dataset summary.

## Verification performed
- Existing and added tests: **75 passed**.
- `scripts/check_compat.py`: model artifact loaded; 113 model features found in validation readouts/specifications; reported compatible at feature/schema level.
- FastAPI smoke tests: `/health`, `/dataset/summary`, `/dataset/records`, `/dataset/vehicles/22112`, and `/` returned HTTP 200.

## Important limitation — model runtime compatibility
The uploaded model was serialized with scikit-learn **1.7.2**. This environment has scikit-learn **1.8.0**. Loading produced `InconsistentVersionWarning`; a real validation inference attempt failed in `SimpleImputer.transform` because `_fill_dtype` is missing. Therefore model load and feature-schema compatibility are verified, but prediction runtime is NOT verified in this environment. Run in the original scikit-learn 1.7.2 environment before presenting prediction outputs. No model retraining or artifact modification was performed.

## Dataset limitation
The 1.13 GB `train_operational_readouts.csv` was not uploaded. The application supports a configurable local dataset directory and reports the missing file. Training-dependent work and any claim of training-data evaluation were not performed.

## Existing model quality warning
The project documentation records the prior validation behavior: 4,991/5,046 predictions were Class 4, including 4,855/4,910 actual Class 0 vehicles. The model should not be represented as reliable without a corrected, independently verified evaluation.

## Run on Windows
1. Extract this project folder.
2. Keep the SCANIA CSVs in a separate folder, including the large training readouts if available.
3. From the project root, set the dataset directory, for example:
   `$env:SCANIA_DATA_DIR = 'D:\path\to\your\SCANIA\data'` (replace with your local data directory)
4. Use a Python environment with the model's original scikit-learn 1.7.2 version and compatible numpy/pandas versions.
5. Install `requirements-agentic.txt` without upgrading the serialized model dependencies.
6. Run `python scripts/check_compat.py`, then `python -m pytest tests -v`.
7. Start with `powershell -ExecutionPolicy Bypass -File .\run_demo.ps1`.
