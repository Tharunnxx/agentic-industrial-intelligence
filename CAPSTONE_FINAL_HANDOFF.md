# Capstone handoff — 30 October 2026

## What changed in this release

- Fixed upload compatibility checks to validate the correct feature groups:
  - 105 numeric model features must be in operational readouts.
  - 8 `Spec_*` model features must be in the matching specifications file, or all 113 must be in a combined readout.
- Added split discovery and fallback. If the UI requests `validation` but the upload only contains `train`, the API now selects `train` and reports the effective split. This fixes the train-only ZIP case where a compatible upload appeared to have no vehicles.
- Overview upload handling now uses the upload's recommended split when loading its vehicle count and vehicle options.
- Refreshed only the Vehicle Analysis page layout/CSS. The other six frontend pages were not redesigned.
- Added a regression test for a train-only operational/specification pair.

## Verification performed

- 84 unit and integration/API tests passed using synthetic test models.
- Python source compilation passed.
- Frontend JavaScript syntax check passed with Node.
- The real saved model loaded and its schema was inspected: 113 features (105 numeric + 8 specification).
- The supplied synthetic ZIP passed the real model's header compatibility check; the API returned three vehicles and assembled a 113-feature payload.
- Real model inference was not certified in the build runtime because it has scikit-learn 1.8.0 while the saved artifact was created with scikit-learn 1.7.2. Use the original model-training Python environment. Do not upgrade scikit-learn in that environment.

## Run on the presentation machine

1. Extract this ZIP to your existing `data` directory, creating `data\agentic_platform`.
2. Keep the original SCANIA CSVs and `cost_sensitive_model.joblib` in their existing parent `data` directory.
3. Open PowerShell in `agentic_platform`.
4. Confirm the active Python environment uses scikit-learn 1.7.2:
   `python -c "import sklearn; print(sklearn.__version__)"`
5. Install only the platform's additional packages if needed:
   `python -m pip install -r requirements-agentic.txt`
   Do not allow pip to change the installed scikit-learn/numpy/pandas versions.
6. Run:
   `python scripts/check_compat.py`
   `python -m pytest tests/unit tests/integration -q`
   `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1`
7. Open `http://127.0.0.1:8000`.

## Upload smoke test

The ZIP `test_fixtures/scania_compatible_upload_smoke_test.zip` contains synthetic placeholders with the exact operational/specification column names and three matching vehicle IDs. It is for upload, split detection, vehicle-list and payload-shape testing only. Its sensor values are not real SCANIA readings and must not be used to claim model accuracy.

## Important capstone limitations

- The platform is a decision-support prototype. Recommendations are not executed and require a qualified human engineer.
- The saved model's official validation behaviour is weak and strongly biased toward class 4. Present the known results transparently; do not claim production readiness or high predictive accuracy.
- Synthetic upload fixture values are not real measurements.
- Uploaded datasets are stored outside this project at `<SCANIA_DATA_DIR>/.agentic_uploads/`. Keep the same `SCANIA_DATA_DIR` to retain access to existing uploads.
