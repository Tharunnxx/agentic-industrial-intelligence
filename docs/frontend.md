# Frontend guide

## Pages

- **Overview:** validation/test population sizes, calibration state, priority and decision distributions, saved batch verification summaries, and browser-session activity.
- **Vehicle analysis:** indexed vehicle search/filter, input feature inspection, and a full four-agent run for a selected vehicle.
- **Agent operations:** each agent's received inputs, performed operations, findings, handoff, and limitations.
- **Maintenance decisions:** all five expected decision costs, selected minimum, priority rationale, proposed actions, and human-approval notice.
- **Model analytics:** split-specific priority/decision/risk distributions and policy comparison metrics returned by the API.
- **System verification:** batch verification summary and latest vehicle-level verification checks.
- **Advanced input:** real sample loader, JSON editor, prediction-only/full-workflow calls, and CSV/ZIP upload.

## Data integrity rules

1. The `/analysis/{split}/{vehicle_id}` endpoint fetches the vehicle features and invokes the workflow. The response's `evaluation_only.recorded_class_label` is rendered only in a separate evaluation-only notice after the workflow has completed.
2. Never send a recorded class label to `/predict`, `/workflow/run`, or any agent.
3. Probability values are shown as probabilities; population risk percentile is already expressed on a 0–100 scale.
4. Expected costs are project-defined cost units, not money.
5. `verified` describes passing software checks. It is not evidence of a correct prediction or physically safe equipment.
6. The four agents are deterministic Python components, not LLM agents.

## Local run

From the project folder, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1
```

Open the local URL printed by the script. For a direct backend start, use `run_backend.ps1` and open `http://127.0.0.1:8000/`.
