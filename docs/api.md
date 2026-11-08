# API reference

Base URL: `http://127.0.0.1:8000` · interactive docs: `/docs` (FastAPI Swagger UI).

## Request body (`/predict`, `/workflow/run`)

```json
{
  "vehicle_id": 12345,
  "time_step": 87.0,
  "features": { "<feature name>": 1.5, "<feature with no measurement>": null, "Spec_0": "Cat1" }
}
```

* `features` must contain **exactly** the model's feature columns (see `GET /model/info` → `feature_columns`).
* Numeric features: number or explicit `null` (not measured). Strings and booleans are rejected.
* `Spec_*` features: text (or number if the model's encoder was fitted on numbers); never null.
* `vehicle_id` / `time_step` are optional metadata and are **not** passed to the model (unless your artifact itself lists them as features; `GET /model/info` warns in that case).
* Target fields (`class_label`) are rejected.

## Endpoints

| Method / path | Purpose | Success | Errors |
|---|---|---|---|
| `GET /health` | liveness + whether the model loaded | 200 (always, even without a model; `model_loaded: false` + `model_error`) | none |
| `GET /model/info` | artifact format, features, classes, calibration wrapper detection, compatibility warnings, known documented results | 200 | 503 `{"error": {"code", "message"}}` |\n| `GET /model/evaluate?split=validation` | measured labelled-split evaluation of the currently loaded, uncalibrated saved model using the original expected-cost policy (M0); supports `validation` and `test`; predicts before joining labels for scoring | 200 with metrics, confusion matrices, per-class scores and policy costs | 404 missing split files; 422 invalid split/data; 503 unavailable model; 500 evaluation failure |
| `GET /results/model-comparison` | saved experiment comparisons; includes a distinct `extra.m2_test_detail` payload with M2 held-out test confusion matrix and per-class metrics | 200 with measured saved experiment artifacts | Missing result artifacts are omitted; do not substitute live M0 metrics for M2 results |
| `POST /predict` | validation + Prediction Agent only | 200 `WorkflowResult` with `prediction`, `trace` | 422 / 503 / 500 with the same shape (`status: "failed"`, `error`, `trace`) |
| `POST /workflow/run` | full workflow | 200 `WorkflowResult` | 422 / 503 / 500 |
| `GET /samples`, `GET /samples/{id}` | real sample rows created by `scripts/extract_samples.py` | 200 | 404 for unknown ids |
| `GET /` | demo UI | HTML | |

## `WorkflowResult` (abridged)

```json
{
  "workflow": "full", "status": "completed", "final_status": "needs_review",
  "escalated_to_human": true, "replan_count": 0, "input_fingerprint": "ab12...",
  "prediction": {"probabilities": {"0": 0.8, "1": 0.1, "2": 0.04, "3": 0.03, "4": 0.03}, "...": "..."},
  "investigation": {"evidence": [], "risk_indicators": [], "hypotheses": [], "missing_information": [], "uncertainty": {}},
  "recommendation": {"recommended_class": 4, "priority": "medium", "actions": [], "explanation": "...", "requires_human_review": true},
  "verification": {"status": "needs_review", "checks": [], "reasons": []},
  "trace": [{"sequence": 1, "stage": "input_validation", "agent": "Orchestrator", "status": "ok", "started_at": "...", "duration_ms": 1.2}],
  "error": null, "rule_config": {}, "disclaimers": []
}
```

(The numbers above only illustrate the shape; they are not results.)

## Error codes

| Code | HTTP | Meaning |
|---|---|---|
| `INPUT_VALIDATION_FAILED` | 422 | `error.details[]` lists `MISSING_FIELD`, `EXTRA_FIELD`, `INVALID_TYPE`, `INVALID_VALUE`, `MISSING_VALUE`, `INSUFFICIENT_DATA`, `INVALID_PAYLOAD` |
| `MODEL_FILE_NOT_FOUND`, `MODEL_LOAD_FAILED`, `MODEL_ARTIFACT_UNSUPPORTED`, `MODEL_CLASSES_MISMATCH`, `MODEL_USES_TARGET` | 503 | no usable model; nothing is predicted |
| `PREDICTION_FAILED`, `UNEXPECTED_ERROR` | 500 | captured in the trace with the failing stage; later stages are `skipped` |
