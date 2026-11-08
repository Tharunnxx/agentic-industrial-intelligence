# Research handoff — SCANIA Operations Intelligence

**Purpose:** factual handoff for conference-paper drafting. This document describes the supplied implementation and saved experiment artifacts; it is not a claim of industrial deployment or physical validation.

## 1. System description

The platform is a local FastAPI decision-support prototype for SCANIA Component X vehicle records. It accepts 105 operational sensor-style features and 8 vehicle specification features (113 model inputs total). A saved RandomForest artifact supplies class scores. Four deterministic Python agents then perform:

1. **Prediction Agent:** obtains raw model class scores, applies the separately configured Platt calibration overlay to the aggregate any-failure score when enabled, and reports risk context.
2. **Investigation Agent:** evaluates data completeness, unusual feature values relative to the validation reference, rule triggers, evidence and uncertainty. Feature names are anonymised; it cannot diagnose a physical component.
3. **Decision Agent:** computes expected costs under the project cost matrix and creates a recommendation with human approval required.
4. **Verification Agent:** checks schema, probability validity, expected-cost arithmetic, decision/priority consistency, evidence references and safety constraints. `verified` means software checks passed only.

The agents are deterministic Python components, not LLM agents. No maintenance action is executed automatically.

## 2. Dataset and evaluation protocol

The saved experiment artifacts describe an official validation population of 5,046 vehicles and a held-out test population of 5,045 vehicles. The test class counts from the saved M2 confusion matrix are `[4903, 26, 15, 41, 60]`; the strong majority-class imbalance must be made explicit in the paper.

The live `/model/evaluate?split=...` endpoint reads the latest operational readout per vehicle, joins specification features, generates model predictions, and only then joins labels for offline scoring. The `class_label` target is not passed into prediction or the agent workflow. The vehicle analysis route adds `evaluation_only.recorded_class_label` after the workflow has completed; it is explicitly post-run comparison metadata.

**Important policy distinction:** the live evaluation endpoint evaluates the currently loaded saved model's uncalibrated output under the expected-cost rule (M0). The saved M2 experiment is a separate calibrated policy, with calibration fitted on official validation data and the held-out test scored once. The UI and API now label and display these results separately. Never describe the live M0 confusion matrix as M2.

## 3. Saved experiment results — held-out test (M2)

Source: `config/results/m2_test.json` (also archived in `experiments/results/m2_test_20261102_010415.json`).

| Metric | Saved M2 test result |
|---|---:|
| Vehicles | 5,045 |
| Mean project cost | 8.6373 units/vehicle |
| Balanced accuracy | 0.2397 (23.97%) |
| Macro F1 | 0.1706 |
| Predictions for classes 0–4 | `[3497, 0, 0, 18, 1530]` |
| Exact class recall, classes 0–4 | `[69.83%, 0%, 0%, 0%, 50.00%]` |
| Failure-vehicle recall (any predicted class 1–4) | 48.59% (69 of 142) |
| Vehicles flagged as non-class-0 | 30.68% |

M2 mean cost was 8.6373 versus 9.7399 for the saved M0 original-model cost rule, a reduction of 11.32% under the project-defined cost matrix. The always-class-4 baseline mean cost was 9.8456. Cost units are project-defined units, not currency.

**Limitations:** M2 predicts only classes 0, 3 and 4 on this test set. It has zero recall for classes 1 and 2 and only 18 predictions of class 3. Exact class identification is weak; do not claim a strong five-class classifier. The 48.59% any-failure recall is lower than the 71.54% mean reported by validation cross-validation, demonstrating generalisation variability.

## 4. Calibration and cross-validation

Source: `config/calibration/m2_calibrator.json` and `config/results/m2_cv.json`.

- Calibration set: official validation.
- Method: Platt scaling on `P(any failure)` using slope `2.1492722122485537` and intercept `-4.960813486239383`.
- Selected severity split: `model` (the model's relative class 1–4 score proportions are retained).
- Repeated vehicle-level CV summary for the selected model variant: mean cost `7.0801`, standard deviation `0.0705`, flagged share `42.91%`, any-failure recall `71.54%`.
- The alternative `valmix` variant had mean cost `7.1898`, standard deviation `0.0468`, flagged share `51.26%`, any-failure recall `79.19%`; it was not selected.
- Test results are the primary independent final estimate. Validation and CV are model-selection/calibration evidence, not a substitute for test performance.

The stored M1 validation AUC is `0.7073` for any-failure ranking. This is a validation result, not test AUC. Calibration changes score scale; it does not improve ranking or exact class discrimination.

## 5. Workflow verification evidence

Saved batch summaries in `data_index/` report:

| Split | Vehicles processed | Verified | Needs review | Workflow errors | Failed checks | Dashboard disagreements |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 5,046 | 5,044 | 2 | 0 | 0 | 0 |
| Test | 5,045 | 5,042 | 3 | 0 | 0 | 0 |

These are workflow/software consistency results only. They do not measure predictive accuracy, physical condition, or equipment safety.

## 6. Cost matrix and decision interpretation

The cost matrix is defined in `agentic_app/core/costs.py`; rows are actual classes and columns are decisions. The decision rule is `argmin_j sum_i p(i) * C[i,j]`. The matrix is asymmetric and assigns substantially higher costs to missed failures than to false alarms. The values are project-defined units, not money. The class selected by minimum expected cost is a decision under that matrix, not a calibrated statement that the selected class is the most likely class.

Do not alter the matrix or decision rule without a separately versioned experiment and rerunning the full evaluation protocol.

## 7. Leakage and assurance statements

- Model loading rejects an artifact whose declared feature list includes `class_label`.
- Input validation rejects target-like input keys (`class_label`, `label`, `target`).
- Offline evaluation predicts before joining labels for scoring.
- Per-vehicle analysis appends the recorded label only after the workflow and labels it `evaluation_only`.
- Verification checks arithmetic and workflow constraints; it cannot establish that a model prediction is correct or a vehicle is safe.

## 8. Reproducibility package

Relevant files:

- `models/cost_sensitive_model.joblib` — original saved model artifact (do not retrain/overwrite for paper drafting).
- `config/calibration/m2_calibrator.json` — calibration parameters and CV summary.
- `config/calibration/validation_risk_reference.json` — validation population reference.
- `config/results/m1_validation.json`, `m2_cv.json`, `m2_test.json` — canonical result files consumed by the platform.
- `experiments/results/` — timestamped experiment source outputs.
- `experiments/m2_calibrated_cost.py` — M2 experiment implementation.
- `scripts/run_batch_workflow.py` — full workflow batch verification.
- `tests/` — unit and integration tests.
- `docs/methodology.md`, `docs/architecture.md`, `docs/api.md` — implementation documentation.

The large original SCANIA CSV files are intentionally not included in this project ZIP. They must remain in the user's external `data` directory. Record exact file provenance, dataset version, hashes, Python version and installed package versions when reproducing results. The current archive alone cannot independently rerun full-data evaluation without those external CSVs.

## 9. Suggested paper framing

Describe this contribution as an explainable, deterministic multi-agent decision-support orchestration layer around an existing cost-sensitive predictive model. The primary contribution is structured investigation, expected-cost decision explanation, bounded verification/replanning and traceability—not a new high-accuracy classifier and not autonomous maintenance execution.

Avoid unsupported claims such as improved fault diagnosis, root-cause identification, real-time industrial deployment, physical safety assurance, or superior five-class classification. Discuss class imbalance, zero recall for classes 1 and 2, anonymised features, single-readout limitations, validation/test variability, project-defined costs and the need for engineer validation.
