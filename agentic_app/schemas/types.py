"""Structured objects passed between pipeline stages (plain dataclasses, JSON-serialisable)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]


# ---------------------------------------------------------------- input / schema
@dataclass
class FeatureSchema(Serializable):
    feature_cols: List[str]
    numeric_cols: List[str]
    spec_cols: List[str]
    spec_categories: Dict[str, List[Any]] = field(default_factory=dict)  # known categories (from the fitted encoder)


@dataclass
class ValidatedReadout:
    vehicle_id: Optional[str]
    time_step: Optional[float]
    features: Dict[str, Any]  # model-ready, ordered as the model expects (not serialised: may hold NaN)
    n_numeric: int
    n_spec: int
    missing_numeric: List[str]
    missing_fraction: float
    spec_values: Dict[str, Any]
    unseen_spec_values: Dict[str, Any]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "time_step": self.time_step,
            "n_numeric": self.n_numeric,
            "n_spec": self.n_spec,
            "n_missing_numeric": len(self.missing_numeric),
            "missing_fraction": self.missing_fraction,
            "spec_values": self.spec_values,
            "unseen_spec_values": self.unseen_spec_values,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------- shared: what each agent explains to the reader
@dataclass
class Finding(Serializable):
    id: str
    title: str
    detail: str
    importance: str = "info"  # "key" | "info" | "caution"
    evidence_refs: List[str] = field(default_factory=list)


@dataclass
class AgentReport(Serializable):
    """Plain-language account of one agent's work: what it received, what it did, what it found, what it passed on."""
    agent: str
    role: str
    received: List[str]
    performed: List[str]
    findings: List[Finding]
    passed_on: List[str]
    limitations: List[str]


# ---------------------------------------------------------------- prediction
@dataclass
class PredictionResult(Serializable):
    model_file: str
    classes: List[int]
    probabilities: Dict[str, float]
    predicted_class_argmax: int
    argmax_probability: float
    probability_any_failure: float
    probability_class_3_or_4: float
    entropy_normalised: float
    risk_summary: str
    calibration_note: str
    model_notes: List[str] = field(default_factory=list)
    # --- added for the calibrated workflow (all optional, so older code keeps working) ---
    decision_probabilities: Dict[str, float] = field(default_factory=dict)  # calibrated if applied, else the raw ones
    calibration_applied: bool = False
    calibrated_probability_any_failure: Optional[float] = None
    risk_context: Optional[Dict[str, Any]] = None
    report: Optional[AgentReport] = None


# ---------------------------------------------------------------- investigation
@dataclass
class EvidenceItem(Serializable):
    id: str
    kind: str  # "observed" (from the input) | "model_output" | "derived" (deterministic calculation)
    statement: str
    value: Any = None


@dataclass
class RiskIndicator(Serializable):
    id: str
    rule: str
    description: str
    observed_value: Any
    threshold: Any
    triggered: bool


@dataclass
class Hypothesis(Serializable):
    id: str
    statement: str
    basis: List[str]
    is_hypothesis: bool = True
    status: str = "unverified"


@dataclass
class UncertaintyAssessment(Serializable):
    level: str  # "low" | "moderate" | "high"
    factors: List[str]
    reliability_notes: List[str]
    missing_fraction: float
    entropy_normalised: float


@dataclass
class Investigation(Serializable):
    summary: str
    evidence: List[EvidenceItem]
    risk_indicators: List[RiskIndicator]
    hypotheses: List[Hypothesis]
    missing_information: List[str]
    uncertainty: UncertaintyAssessment
    report: Optional[AgentReport] = None


# ---------------------------------------------------------------- decision
@dataclass
class Action(Serializable):
    id: str
    type: str  # escalate_to_engineer | schedule_inspection | increase_monitoring | routine_monitoring | collect_data | record_outcome
    description: str
    urgency: str
    executed: bool = False
    automated: bool = False
    requires_human_approval: bool = True


@dataclass
class Recommendation(Serializable):
    recommended_class: int
    raw_argmax_class: int
    expected_costs: Dict[str, float]
    cost_matrix_orientation: str
    priority: str
    priority_rule: str
    priority_rationale: str
    actions: List[Action]
    explanation: str
    evidence_refs: List[str]
    uncertainty_statement: str
    requires_human_review: bool = True
    revision: int = 0
    addressed_issues: List[str] = field(default_factory=list)
    safety_notice: str = (
        "Decision support only: these are recommendations for a qualified maintenance engineer. "
        "No action has been executed and none will be executed automatically."
    )
    # --- added for the calibrated workflow ---
    decision_basis: str = "raw model probabilities"
    raw_expected_costs: Dict[str, float] = field(default_factory=dict)
    raw_min_cost_class: Optional[int] = None
    runner_up_class: Optional[int] = None
    margin_to_runner_up: Optional[float] = None
    cost_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)  # decision j -> {actual class i: P(i) * Cost(i, j)}
    report: Optional[AgentReport] = None


# ---------------------------------------------------------------- verification
@dataclass
class VerificationCheck(Serializable):
    id: str
    name: str
    passed: bool
    severity: str  # "fixable" (Decision agent can revise) | "blocking" (needs a human)
    message: str
    what_checked: str = ""
    why_it_matters: str = ""


@dataclass
class VerificationResult(Serializable):
    status: str  # verified | needs_review | replan_required
    checks: List[VerificationCheck]
    reasons: List[str]
    replan_issue_codes: List[str] = field(default_factory=list)
    review_reasons: List[str] = field(default_factory=list)
    assurance: Dict[str, Any] = field(default_factory=dict)  # four different kinds of "OK", kept apart on purpose
    report: Optional[AgentReport] = None


# ---------------------------------------------------------------- workflow
@dataclass
class TraceEntry(Serializable):
    sequence: int
    stage: str
    agent: str
    status: str  # ok | error | skipped
    started_at: Optional[str]
    ended_at: Optional[str]
    duration_ms: Optional[float]
    input_summary: Dict[str, Any]
    output: Optional[Dict[str, Any]]
    error: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowError(Serializable):
    code: str
    message: str
    stage: str
    details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowResult(Serializable):
    workflow: str  # "full" | "predict_only"
    status: str  # completed | failed
    final_status: Optional[str]  # verified | needs_review | replan_required (None if failed)
    escalated_to_human: bool
    replan_count: int
    input_fingerprint: str
    prediction: Optional[PredictionResult]
    investigation: Optional[Investigation]
    recommendation: Optional[Recommendation]
    verification: Optional[VerificationResult]
    trace: List[TraceEntry]
    error: Optional[WorkflowError]
    rule_config: Dict[str, Any]
    disclaimers: List[str]
