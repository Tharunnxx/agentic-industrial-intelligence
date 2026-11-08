# Methodology, decision logic, limitations, reproducibility

## 1. What is reused, what is new

| Reused unchanged | New (this platform) |
|---|---|
| The saved joblib model (default `cost_sensitive_model.joblib`) and its preprocessing, loaded read-only | Strict input validation, four agents, orchestrator, API, UI, tests, evaluation script |
| Feature convention of your scripts: `Spec_*` columns are categorical, everything else numeric | Per-request trace and verification |
| Official cost matrix, same orientation as `eval_calibration_experiment.py` (rows = actual, columns = predicted) | Documented priority rule |

No model is trained, tuned or modified. `eval_calibration_experiment.py` does **not** save a model (it only writes `results.json` / `cutoff_predictions.csv`), so its calibrated model does not exist as a file. The platform reports whether a loaded artifact contains a `CalibratedClassifierCV`; if not, probabilities are treated as uncalibrated raw model outputs and every result says so.

## 2. Cost-sensitive decision rule

For probabilities `p` over classes 0-4, `ExpectedCost(j) = sum_i p(i) * C[i][j]`, with `C` the official matrix (rows actual, columns predicted). The decision label is `argmin_j ExpectedCost(j)`; ties go to the lowest class (numpy `argmin`, as in your experiment script). Unit tests verify orientation (e.g. actual 4 predicted 0 costs 500, actual 0 predicted 4 costs 10) and hand-computed mixtures.

Consequence worth understanding for the viva: a missed failure costs 20-50x a false alarm, so the label moves away from class 0 once failure probability is only a few percent (about 2% if all the failure mass sits on class 4: predicting 4 costs 10·(1-p4), predicting 0 costs 500·p4). A cost-minimising label is therefore conservative by construction and is **not** a calibrated statement that failure is likely.

## 3. Priority rule (configurable, `core/rules.py`)

`p_any = 1 - p0`, `p_high = p3 + p4`. First match wins:

| Priority | Condition (defaults) |
|---|---|
| critical | `p_high >= 0.30` |
| high | `p_any >= 0.50` |
| medium | `p_any >= 0.20` |
| low | otherwise |

Priority deliberately uses probabilities, not the cost-minimising label (see §2; otherwise nearly every vehicle would be "high"). Thresholds are **defaults chosen for transparency, not tuned on any labels**; change them in `.env`. Actions per priority are recommendations only (`executed=false`, `automated=false`, `requires_human_approval=true`).

## 4. Verification checks

V01 required fields · V02 expected costs recomputed independently · V03 recommended class = argmin · V04 priority matches rule · V05 no executed/automated actions or forbidden wording · V06 cited evidence ids exist · V07 high/critical priority includes escalation or inspection · V08 hypotheses flagged unverified and evidence kinds well-formed · V09 probabilities valid · V10 human-review flag.
`replan_required` = a *fixable* check failed (Decision Agent may revise once, default `AGENTIC_MAX_REPLANS=1`, hard cap 2). `needs_review` = a blocking check failed, data completeness is below the review threshold (default: ≥50% numeric nulls), an unseen specification category appeared, or the replan budget was exhausted (then `escalated_to_human=true`). `verified` = all checks passed and no review trigger; a human still reviews the recommendation.

## 5. Validation of inputs and leakage controls

Missing, extra and wrongly typed fields are rejected with field lists; target-like fields are rejected; `vehicle_id` and `time_step` are metadata only. At load time the platform **refuses** an artifact whose features include `class_label` and **warns** if it includes `vehicle_id` or `time_step`. The "maximum failure class per vehicle" is never used by the platform. Split hygiene (vehicle-level splits) is a property of how your artifacts were trained; the platform cannot verify it from a joblib file, and says so.

## 6. Limitations (honest)

* The documented official-validation behaviour of both existing artifacts is poor (see `review2_results_summary.md`); the platform surfaces this as reliability notes and `high` uncertainty. The agents add structure, traceability and safety around the model; they cannot improve its predictions.
* Single readout per request: no trend/time-series reasoning. Feature names are anonymised, so the Investigation Agent cannot name components or root causes and does not try to.
* Hypotheses are templated statements tied to rule triggers, not diagnoses.
* The priority thresholds and the 50% null-fraction review threshold are design defaults, not validated operating points.
* Joblib files are Python pickles: load only files you created, in the same library versions.
* Latency of the first request includes model loading.

## 7. Reproducibility

Deterministic: no randomness, no network, no LLM in the decision path. Same payload + same artifact → identical prediction/investigation/recommendation/verification (tested). `input_fingerprint` identifies the payload. Sample extraction is seeded (`--seed`, default 2026) and independent of predictions. `scripts/evaluate_saved_model.py` computes metrics once on the validation files with the same protocol as your `evaluate_official_validation.py`; do not tune against it. Record the library versions printed by `scripts/check_compat.py` with any reported result.
