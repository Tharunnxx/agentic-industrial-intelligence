"""Orchestrator: validation -> prediction -> investigation -> decision -> verification (bounded replanning)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.config import Settings
from ..schemas.types import TraceEntry, WorkflowError, WorkflowResult
from ..services.model_service import ModelService, ModelUnavailable, PredictionError
from ..services.validation import InputValidationError, validate_readout
from .decision import DecisionAgent
from .investigation import InvestigationAgent
from .prediction import PredictionAgent
from .verification import VerificationAgent, refresh_assurance

DISCLAIMERS = [
    "Decision support prototype: recommendations are not executed and must be reviewed by a qualified maintenance engineer.",
    "Probabilities come from the saved model artifact; see model notes for its documented validation behaviour.",
    "'Verified' means software consistency checks passed. It does not mean the prediction is correct or that any equipment is safe.",
]
STAGES = ["input_validation", "prediction", "investigation", "decision", "verification"]


class _StageFailure(Exception):
    def __init__(self, error: WorkflowError):
        self.error = error


def _fingerprint(payload: Any) -> str:
    try:
        blob = json.dumps(payload, sort_keys=True, default=str, allow_nan=True)
    except Exception:
        blob = repr(payload)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Orchestrator:
    def __init__(self, settings: Settings, model_service: ModelService, *, prediction_agent=None, investigation_agent=None,
                 decision_agent=None, verification_agent=None, clock: Optional[Callable[[], _dt.datetime]] = None,
                 calibrator=None, feature_reference: Optional[Dict[str, Any]] = None):
        self.settings = settings
        self.model_service = model_service
        self.calibrator = calibrator
        self.prediction_agent = prediction_agent or PredictionAgent(model_service, calibrator)
        self.investigation_agent = investigation_agent or InvestigationAgent(settings, feature_reference)
        self.decision_agent = decision_agent or DecisionAgent(settings)
        self.verification_agent = verification_agent or VerificationAgent(settings, calibrator)
        self._clock = clock or (lambda: _dt.datetime.now(_dt.timezone.utc))

    # ------------------------------------------------------------------ helpers
    def _stage(self, trace: List[TraceEntry], stage: str, agent: str, input_summary: Dict[str, Any], fn: Callable[[], Any],
               summarise: Callable[[Any], Dict[str, Any]]) -> Any:
        started, t0 = self._clock(), time.perf_counter()
        try:
            out = fn()
        except Exception as exc:
            error = self._to_error(stage, exc)
            trace.append(TraceEntry(len(trace) + 1, stage, agent, "error", started.isoformat(), self._clock().isoformat(),
                                    round((time.perf_counter() - t0) * 1000, 3), input_summary, None, error.to_dict()))
            raise _StageFailure(error) from exc
        trace.append(TraceEntry(len(trace) + 1, stage, agent, "ok", started.isoformat(), self._clock().isoformat(),
                                round((time.perf_counter() - t0) * 1000, 3), input_summary, summarise(out)))
        return out

    @staticmethod
    def _to_error(stage: str, exc: Exception) -> WorkflowError:
        if isinstance(exc, InputValidationError):
            return WorkflowError(exc.code, "Input validation failed.", stage, exc.errors)
        if isinstance(exc, ModelUnavailable):
            return WorkflowError(exc.code, str(exc), stage, [])
        if isinstance(exc, PredictionError):
            return WorkflowError(exc.code, str(exc), stage, [])
        return WorkflowError("UNEXPECTED_ERROR", f"{type(exc).__name__}: {exc}", stage, [])

    def _skipped(self, trace: List[TraceEntry], from_stage: str, names: Dict[str, str]) -> None:
        for stage in STAGES[STAGES.index(from_stage):]:
            if stage in names and not any(t.stage == stage for t in trace):
                trace.append(TraceEntry(len(trace) + 1, stage, names[stage], "skipped", None, None, None, {}, None))

    def _names(self, full: bool) -> Dict[str, str]:
        names = {"input_validation": "Orchestrator", "prediction": self.prediction_agent.name}
        if full:
            names.update({"investigation": self.investigation_agent.name, "decision": self.decision_agent.name,
                          "verification": self.verification_agent.name})
        return names

    # ------------------------------------------------------------------ workflows
    def predict(self, payload: Any) -> WorkflowResult:
        return self._run(payload, full=False)

    def run_workflow(self, payload: Any) -> WorkflowResult:
        return self._run(payload, full=True)

    def _run(self, payload: Any, full: bool) -> WorkflowResult:
        trace: List[TraceEntry] = []
        names = self._names(full)
        prediction = investigation = recommendation = verification = None
        error: Optional[WorkflowError] = None
        replans, escalated = 0, False
        final_status: Optional[str] = None
        n_keys = len(payload.get("features", {})) if isinstance(payload, dict) and isinstance(payload.get("features"), dict) else 0

        try:
            def validate():
                schema = self.model_service.schema()  # raises ModelUnavailable if no usable model
                return validate_readout(payload, schema, self.settings)

            readout = self._stage(trace, "input_validation", "Orchestrator", {"n_feature_keys": n_keys}, validate, lambda r: r.to_dict())
            prediction = self._stage(trace, "prediction", self.prediction_agent.name, {"model_file": self.model_service.model_file},
                                     lambda: self.prediction_agent.run(readout), lambda r: r.to_dict())
            if full:
                investigation = self._stage(trace, "investigation", self.investigation_agent.name, {"evidence_sources": ["readout", "prediction"]},
                                            lambda: self.investigation_agent.run(readout, prediction), lambda r: r.to_dict())
                recommendation = self._stage(trace, "decision", self.decision_agent.name, {"revision": 0},
                                             lambda: self.decision_agent.run(readout, prediction, investigation), lambda r: r.to_dict())
                verification = self._stage(trace, "verification", self.verification_agent.name, {"revision": recommendation.revision},
                                           lambda: self.verification_agent.run(readout, prediction, investigation, recommendation), lambda r: r.to_dict())
                while verification.status == "replan_required" and replans < self.settings.max_replans:
                    replans += 1
                    issues = list(verification.replan_issue_codes)
                    recommendation = self._stage(
                        trace, "decision", self.decision_agent.name, {"revision": replans, "issues": issues},
                        lambda: self.decision_agent.revise(recommendation, issues, readout, prediction, investigation), lambda r: r.to_dict())
                    verification = self._stage(trace, "verification", self.verification_agent.name, {"revision": recommendation.revision},
                                               lambda: self.verification_agent.run(readout, prediction, investigation, recommendation), lambda r: r.to_dict())
                if verification.status == "replan_required":  # bounded: escalate instead of looping
                    escalated = True
                    verification.review_reasons.append(f"Replanning budget ({self.settings.max_replans}) exhausted; escalated to human review.")
                    verification.reasons.append(f"Replanning budget ({self.settings.max_replans}) exhausted; escalated to human review after the bounded replanning cycle failed verification.")
                    verification.status = "needs_review"
                    refresh_assurance(verification, self.calibrator.evidence if self.calibrator else None)
                elif verification.status == "needs_review":
                    escalated = True
                final_status = verification.status
        except _StageFailure as failure:
            error = failure.error
            self._skipped(trace, error.stage, names)

        status = "failed" if error else "completed"
        return WorkflowResult(
            workflow="full" if full else "predict_only", status=status, final_status=None if error else final_status,
            escalated_to_human=escalated, replan_count=replans, input_fingerprint=_fingerprint(payload),
            prediction=prediction, investigation=investigation, recommendation=recommendation, verification=verification,
            trace=trace, error=error, rule_config=self.settings.rule_config(), disclaimers=list(DISCLAIMERS),
        )
