# Review 2 results summary (verified figures only)

Every number below is either copied from one of your own report files (source named) or derived by arithmetic from a confusion matrix / class counts in those files plus the official cost matrix. **Nothing was re-measured by this platform**, and no number from the agent workflow is a performance claim. Internal simulations and official validation are kept apart.

## A. Official validation set (5,046 vehicles; class counts 0-4: 4,910 / 16 / 14 / 30 / 76)

| Item | Total cost | Mean cost / vehicle | Basis |
|---|---:|---:|---|
| `cost_sensitive` model (as reported) | 49,196 | 9.7495 | reported in `cost_sensitive_report.txt` (file→model mapping inferred from the title and `app.py`) |
| `balanced_baseline_model` | 45,684 | 9.0535 | **derived** from the confusion matrix in `official_validation_report.txt` |
| Always class 0 | 57,400 | 11.3753 | **derived** from class counts |
| Always class 4 | 49,566 | 9.8228 | **derived** from class counts |

Other official-validation figures reported: balanced_baseline balanced accuracy 0.2159, macro F1 0.0403.

What the confusion matrices show (counts read directly from your reports):
* `cost_sensitive`: predicts class 4 for 4,991 of 5,046 vehicles (98.9%); 4,855 of 4,910 class-0 vehicles are predicted as class 4; all 76 class-4 vehicles are caught.
* `balanced_baseline`: predicts class 4 for 4,576 of 5,046 vehicles (90.7%); class-0 recall 9.2% (454/4,910).
* Observation (derived): on this protocol the cost-sensitive artifact (9.7495) is only about 0.8% better than the trivial always-class-4 policy (9.8228) and is worse than the balanced baseline (9.0535). Compare before choosing which artifact to demonstrate: set `AGENTIC_MODEL_PATH` to switch.

## B. Internal experiments (not official scores)

| Item | Value | Source / status |
|---|---|---|
| Initial baseline, 100,000-row internal split | balanced accuracy 0.2000, macro F1 0.1977, predicts class 0 for every row | `baseline_evaluation.txt` |
| Calibrated expected-cost, random-cutoff, vehicle-held-out; seeds 42 / 123 / 2026 | mean cost 5.5667 / 5.1278 / 5.4132 | **reported by you**; the `results.json` files were not available to me, so unverified here |

The internal random-cutoff protocol differs from the official "latest readout per vehicle" protocol, and the calibrated model from that experiment was never exported as a file. The gap between ~5.4 (internal) and ~9.7 (official) must not be read as an improvement or a regression of the same system; it compares different protocols and different artifacts.

## C. Platform engineering status

* Automated tests: 72 collected; in the build environment (run with Python's unittest runner, not pytest) 63 passed and 9 FastAPI tests were skipped because `fastapi`/`httpx` were not installed there. The API layer was therefore **not executed** at build time; run `.\run_tests.ps1 -Integration` on your machine to execute it.
* The test model is a tiny synthetic scikit-learn pipeline trained on random numbers inside the test run; it exists only to test software behaviour and produces no reported metric.
* The platform has not been run against your real `cost_sensitive_model.joblib` or your real CSVs (they were not available to the builder). `scripts/check_compat.py` exists to confirm that.
* `scripts/evaluate_saved_model.py` (not yet run on your data) will add policy comparison, per-class metrics, any-failure AUC and plots; add its numbers here only after you run it.

## D. One-paragraph honest statement for the review

"The agentic platform wraps the existing predictive-maintenance model in validated, traceable agents and a verification layer. It reuses the saved model without retraining. The underlying model's official validation behaviour is poor (it predicts class 4 for 91-99% of vehicles, and its cost is close to the always-class-4 policy), so the contribution of this work is the decision-support architecture and its safeguards, not improved prediction accuracy."
