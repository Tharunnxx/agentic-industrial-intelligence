# Demo walkthrough and likely viva questions

## Before the review (once)
1. `python scripts/check_compat.py` (expect `COMPATIBLE`).
2. `python scripts/extract_samples.py` (creates real samples; note the printed missing-value fractions).
3. `.\run_tests.ps1 -Integration`.

## Demo script (about 7 minutes)
1. `.\run_demo.ps1` → browser opens. Point out the header chip "model loaded".
2. Pick a sample (one from class 0 and one from a failure class), **Load sample**. Mention it is a real validation row; the dataset label is withheld.
3. **Run full agent workflow.** Walk top to bottom: caution box (documented model reliability), probabilities, expected-cost table (explain rows = actual), priority and rule id, actions marked "executed: false", investigation (observed evidence vs unverified hypotheses vs missing information), verification checks, trace.
4. Reveal the evaluation-only label after the run; say plainly whether the model was right. Do not cherry-pick.
5. Edit the JSON: delete one feature → 422 with `MISSING_FIELD`; add `"class_label": 3` → rejected as target; set a numeric to `"abc"` → `INVALID_TYPE`.
6. Show `GET /model/info` and `/docs`.
7. Rename the model path temporarily (`AGENTIC_MODEL_PATH=missing.joblib`), restart: workflow returns 503 `MODEL_FILE_NOT_FOUND`, no prediction invented.
8. Show a test for bounded replanning (`tests/unit/test_agents.py::ReplanTests`).

If every demo vehicle ends as `needs_review`, that is the configured behaviour when ≥50% of numeric features are null; adjust `AGENTIC_REVIEW_MISSING_FRACTION` only as a documented design choice, not to make the demo look better.

## Likely viva questions

**Why is this "agentic" and not just functions?** Four components with distinct responsibilities and typed hand-offs: the Verification Agent independently recomputes the Decision Agent's costs and priority and can reject its output, triggering a bounded revision; the orchestrator records a trace. No LLM is used, deliberately, so decisions are reproducible.

**Why no LLM?** Consequential industrial recommendations must be deterministic and auditable. An LLM could later narrate results, but it would sit outside the decision and verification path.

**Did you retrain the model?** No. The saved joblib artifact is loaded read-only. The calibrated model from the experiment script was never saved, so it is not used.

**How do you pick the class?** Minimum expected cost under the official matrix (rows actual, columns predicted). Because misses are 20-50x costlier than false alarms, the label moves to a failure class at only a few percent failure probability (about 2% if the mass is on class 4).

**Why is your model's official cost 9.75 when your internal runs show about 5.4?** Different protocols (random cutoff on training-distribution vehicles vs the latest readout on the official validation set) and different artifacts (the calibrated model was never exported). They are not comparable and I do not claim an official improvement.

**Your model predicts class 4 for almost everything. Is the system useful?** The model is weak: its cost is near the always-class-4 baseline. The platform's value is safe decision support around it: validated inputs, explicit uncertainty and reliability notes, human-approved actions, and verification. Improving the model is separate future work.

**How do you avoid leakage?** The platform rejects target fields, treats `vehicle_id`/`time_step` as metadata, refuses artifacts that use `class_label`, and warns if an artifact uses identifiers. Vehicle-level splitting is a property of training, documented for the experiment script; I cannot verify it for a joblib file.

**What does `verified` mean?** The recommendation obeys the configured rules and the data was complete enough. It does not mean the prediction is correct or that a fault exists.

**What stops infinite agent loops?** `AGENTIC_MAX_REPLANS` (default 1, hard cap 2). After that the result is `needs_review` with `escalated_to_human=true`.

**What can the Investigation Agent not do?** Name components or root causes: the 105 features are anonymised and only one readout is supplied. It reports observed facts, model outputs, derived calculations, rule triggers, and clearly labelled unverified hypotheses.

**How would you deploy this for real?** Calibrate on held-out vehicles, set thresholds from operating-cost data, add repair history and time series, authentication, audit storage and engineer sign-off workflow.

**What would you do with more time?** Export and evaluate a calibrated model once, add trend features per vehicle, and compare priority thresholds on a separate validation set (never on the final test).
