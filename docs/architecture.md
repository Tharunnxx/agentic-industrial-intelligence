# Architecture, workflow and data flow

## System architecture

```mermaid
flowchart LR
    subgraph Local["Your existing project folder (read-only for this platform)"]
        M[("cost_sensitive_model.joblib<br/>(or another .joblib)")]
        V[("validation_*.csv<br/>(optional: samples + evaluation)")]
    end
    subgraph Platform["agentic_platform/"]
        UI["frontend/index.html<br/>(browser UI)"]
        API["FastAPI<br/>agentic_app/api/routes.py"]
        ORC["Orchestrator"]
        VAL["Input validation<br/>services/validation.py"]
        MS["ModelService<br/>services/model_service.py"]
        PA["Prediction Agent"]
        IA["Investigation Agent"]
        DA["Decision Agent"]
        VA["Verification Agent"]
        COST["core/costs.py + core/rules.py<br/>(official matrix, priority rule)"]
        KR["config/known_results.json<br/>(facts from your own reports)"]
        SS["SampleService<br/>data_samples/*.json"]
    end
    UI -->|HTTP JSON| API --> ORC
    ORC --> VAL --> MS
    ORC --> PA --> MS
    MS -->|joblib.load once| M
    MS --> KR
    ORC --> IA --> DA --> VA
    DA --> COST
    VA --> COST
    API --> SS
    V -.->|scripts/extract_samples.py| SS
    V -.->|scripts/evaluate_saved_model.py| OUT[("outputs/ metrics, CSV, plots")]
```

## Agent workflow

```mermaid
flowchart TD
    A([POST /workflow/run]) --> B[Input validation<br/>missing / extra / wrong type]
    B -- invalid --> X1[[failed: INPUT_VALIDATION_FAILED, HTTP 422]]
    B -- model missing --> X2[[failed: MODEL_*, HTTP 503<br/>no prediction is fabricated]]
    B -- ok --> C[Prediction Agent<br/>probabilities from saved model]
    C --> D[Investigation Agent<br/>evidence, risk rules, hypotheses,<br/>missing info, uncertainty]
    D --> E[Decision Agent<br/>expected cost, priority rule,<br/>recommended actions]
    E --> F{Verification Agent}
    F -- verified --> G([Final result])
    F -- needs_review --> G
    F -- replan_required --> H{replans used < max<br/>default 1}
    H -- yes --> I[Decision Agent.revise] --> F
    H -- no --> J[Escalate: needs_review +<br/>escalated_to_human] --> G
```

## Sequence for one run

```mermaid
sequenceDiagram
    participant U as Browser / client
    participant API as FastAPI
    participant O as Orchestrator
    participant M as ModelService
    U->>API: POST /workflow/run {vehicle_id, time_step, features}
    API->>O: run_workflow(payload)
    O->>M: schema() / ensure_loaded()
    O->>O: validate_readout (trace entry 1)
    O->>M: predict_proba (Prediction Agent, entry 2)
    O->>O: Investigation (3) -> Decision (4) -> Verification (5)
    opt verification == replan_required and budget left
        O->>O: Decision.revise (6) -> Verification (7)
    end
    O-->>API: WorkflowResult (prediction, investigation, recommendation, verification, trace)
    API-->>U: 200 / 422 / 503 JSON
```

## Data flow and objects

| Stage | Input | Output object (`agentic_app/schemas/types.py`) |
|---|---|---|
| Input validation | request JSON | `ValidatedReadout` (features ordered like the model's `feature_cols`, null counts, unseen categories) |
| Prediction | `ValidatedReadout` | `PredictionResult` (5 probabilities, P(any failure), P(class 3-4), entropy, risk summary, model notes) |
| Investigation | readout + prediction | `Investigation` (evidence items `observed` / `model_output` / `derived`, risk indicators, hypotheses flagged unverified, missing information, uncertainty) |
| Decision | readout + prediction + investigation | `Recommendation` (cost-minimising class, expected costs, priority + rule id, actions, explanation, evidence refs, uncertainty statement) |
| Verification | all of the above | `VerificationResult` (`verified` / `needs_review` / `replan_required`, 10 named checks, reasons) |
| Orchestrator | all | `WorkflowResult` incl. `trace` (stage, agent, status, UTC start/end, duration, input summary, output) |

Each agent has one responsibility; none calls an LLM. The workflow is deterministic: the same payload and the same model file give identical prediction, investigation, recommendation and verification (only trace timestamps and durations differ; `input_fingerprint` is a hash of the payload).
