# Manifest

Everything lives under `agentic_platform/`. No file outside this folder is created or changed.

| File | Purpose |
|---|---|
| `.env.example` | Optional configuration template |
| `.gitignore` | Excludes .env, outputs, samples |
| `MANIFEST.md` | This file |
| `README.md` | Setup, status, assumptions, commands |
| `agentic_app/__init__.py` | Package marker / version |
| `agentic_app/agents/__init__.py` | Package marker |
| `agentic_app/agents/base.py` | Agent base class |
| `agentic_app/agents/decision.py` | Decision/Response Agent (incl. bounded revise) |
| `agentic_app/agents/investigation.py` | Investigation Agent |
| `agentic_app/agents/orchestrator.py` | Workflow orchestration, trace, bounded replanning |
| `agentic_app/agents/prediction.py` | Prediction Agent |
| `agentic_app/agents/verification.py` | Verification Agent (10 checks) |
| `agentic_app/api/__init__.py` | Package marker |
| `agentic_app/api/routes.py` | FastAPI app factory and endpoints |
| `agentic_app/core/__init__.py` | Package marker |
| `agentic_app/core/config.py` | Settings from env/.env |
| `agentic_app/core/costs.py` | Official cost matrix + expected-cost rule |
| `agentic_app/core/rules.py` | Configurable priority rule |
| `agentic_app/main.py` | ASGI entry point (agentic_app.main:app) |
| `agentic_app/schemas/__init__.py` | Package marker |
| `agentic_app/schemas/types.py` | Dataclasses passed between stages |
| `agentic_app/services/__init__.py` | Package marker |
| `agentic_app/services/metrics.py` | Policies and evaluation metrics |
| `agentic_app/services/model_service.py` | Loads existing joblib artifact, safe predict_proba, model info |
| `agentic_app/services/sample_service.py` | Serves real extracted samples |
| `agentic_app/services/validation.py` | Strict input validation |
| `config/known_results.json` | Facts copied/derived from your own report files (with sources) |
| `conftest.py` | Puts this folder on sys.path for pytest |
| `data_samples/README.md` | Explains how samples are created |
| `docs/api.md` | API reference |
| `docs/architecture.md` | Mermaid architecture, workflow, data flow |
| `docs/demo_walkthrough_and_viva.md` | Demo script and viva Q&A |
| `docs/methodology.md` | Method, decision logic, limitations, reproducibility |
| `docs/review2_results_summary.md` | Verified-figures-only results summary |
| `frontend/index.html` | Frontend application shell |
| `frontend/app.css` | Responsive offline-first interface styling |
| `frontend/app.js` | Dashboard, indexed vehicle analysis, agent reports, decisions, analytics, verification and advanced input |
| `docs/frontend.md` | Frontend pages, data-integrity rules and run instructions |
| `models/README.md` | Explains models/ folder |
| `outputs/.gitkeep` | Keeps outputs/ folder |
| `pytest.ini` | pytest config |
| `requirements-agentic.txt` | Extra dependencies (fastapi, uvicorn, httpx, pytest, matplotlib) |
| `run_backend.ps1` | Start FastAPI backend |
| `run_demo.ps1` | Start backend and open the UI |
| `run_tests.ps1` | Run unit (and optionally integration) tests |
| `scripts/_bootstrap.py` | sys.path helper |
| `scripts/check_compat.py` | Read-only compatibility check |
| `scripts/evaluate_saved_model.py` | Evaluate saved model + policies; JSON/CSV/PNG output |
| `scripts/extract_samples.py` | Create real-row samples for the UI |
| `tests/__init__.py` | Package marker |
| `tests/helpers.py` | Synthetic fixtures and stub (tests only) |
| `tests/integration/__init__.py` | Package marker |
| `tests/integration/test_api.py` | API tests (need fastapi+httpx) |
| `tests/integration/test_workflow_synthetic_model.py` | End-to-end with synthetic sklearn model |
| `tests/unit/__init__.py` | Package marker |
| `tests/unit/test_agents.py` | Agent schemas, verification, replanning, determinism, workflow |
| `tests/unit/test_costs.py` | Cost matrix orientation and expected cost |
| `tests/unit/test_metrics.py` | Metrics and policies |
| `tests/unit/test_model_service_errors.py` | Missing/corrupt artifact, config validation |
| `tests/unit/test_validation.py` | Input validation |
