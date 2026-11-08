PHASE 3 BACKEND PATCH (calibrated, explainable agents + vehicle index + analytics API)

HOW TO APPLY (PowerShell, from the agentic_platform folder):
  Copy-Item . ..\agentic_platform_backup_before_phase3 -Recurse -Exclude models   # safety copy (skips the 278 MB model)
  Expand-Archive -Force "$HOME\Downloads\phase3_backend_patch.zip" .

It only ADDS files and REPLACES the 10 files listed below. Your model, data_samples, frontend and docs are untouched.
Replaced : agents/{decision,investigation,orchestrator,prediction,verification}.py, api/routes.py, core/{config,rules}.py, schemas/types.py, tests/helpers.py
Added    : agents/common.py, core/calibration.py, services/{batch,results_service,vehicle_index}.py,
           config/calibration/*, config/results/*, scripts/{build_vehicle_index,run_batch_workflow}.py,
           tests/unit/test_calibrated_workflow.py, tests/integration/test_calibrated_api.py
