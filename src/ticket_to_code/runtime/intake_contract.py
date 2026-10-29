"""
Phase-1 intake contract and run output contract.

This module turns the agreed policy into executable validation primitives.
"""

from __future__ import annotations

from enum import Enum
from fnmatch import fnmatch
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TicketKind(str, Enum):
    VERSION = "version"
    BUG = "bug"
    PERMISSION = "permission"
    REFACTOR = "refactor"
    CONFIG = "config"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RunStatus(str, Enum):
    SOLVED = "solved"
    NO_ACTION_REQUIRED = "no_action_required"
    NEEDS_RETRY = "needs_retry"
    FAILED_CLOSED = "failed_closed"
    NEED_MORE_INFO = "need_more_info"


class ValidationResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"


class GroundTruthTicket(BaseModel):
    ticket_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    type: TicketKind
    expected_changed_files: List[str] = Field(default_factory=list)
    forbidden_files: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    good_solution_diff: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    risk_level: RiskLevel = RiskLevel.MEDIUM

    @field_validator("expected_changed_files", "forbidden_files", mode="before")
    @classmethod
    def _normalize_paths(cls, value: Any) -> List[str]:
        if value is None:
            return []
        out: List[str] = []
        for item in value:
            s = str(item).replace("\\", "/").strip()
            if s:
                out.append(s)
        return out

    @model_validator(mode="after")
    def _must_have_acceptance(self) -> "GroundTruthTicket":
        if not self.acceptance_criteria:
            raise ValueError("acceptance_criteria must contain at least one item")
        return self


class SuccessPolicy(BaseModel):
    solved_requires_code_correctness: bool = True
    solved_requires_clean_diagnostics_on_changed_files: bool = True
    build_is_mandatory_blocker: bool = False
    tests_are_mandatory_blocker: bool = False


class ConstraintPolicy(BaseModel):
    minimal_edits_only: bool = True
    no_unrelated_file_changes: bool = True
    no_hardcoding_when_forbidden: bool = True
    preserve_public_api_unless_explicit: bool = True
    fail_closed_on_hard_gate_violation: bool = True


class StageRetryPolicy(BaseModel):
    discovery: int = 2
    planning: int = 2
    patch_generation: int = 2
    validation_repair_loops: int = 1


class DecisionPriority(BaseModel):
    order: List[str] = Field(
        default_factory=lambda: [
            "ownership_accuracy",
            "consumer_rewiring_completeness",
            "patch_minimality",
            "validation_strictness",
            "speed",
        ]
    )


class DecisionRecord(BaseModel):
    decision: str
    why: str
    alternatives_rejected: List[str] = Field(default_factory=list)


class ChangedFileRecord(BaseModel):
    path: str
    why: str
    symbols: List[str] = Field(default_factory=list)


class ValidationEvidenceRecord(BaseModel):
    check: str
    result: ValidationResult
    evidence: str


class TicketRunReport(BaseModel):
    ticket_id: str
    status: RunStatus
    findings: List[str] = Field(default_factory=list)
    decisions: List[DecisionRecord] = Field(default_factory=list)
    changed_files: List[ChangedFileRecord] = Field(default_factory=list)
    validation_evidence: List[ValidationEvidenceRecord] = Field(default_factory=list)
    residual_risks: List[str] = Field(default_factory=list)
    retry_count: int = 0


class OrchestratorPolicy(BaseModel):
    success: SuccessPolicy = Field(default_factory=SuccessPolicy)
    constraints: ConstraintPolicy = Field(default_factory=ConstraintPolicy)
    retries: StageRetryPolicy = Field(default_factory=StageRetryPolicy)
    decision_priority: DecisionPriority = Field(default_factory=DecisionPriority)


def file_matches_any(path: str, patterns: List[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(fnmatch(normalized, p) for p in patterns)


def enforce_hard_gates(
    ticket: GroundTruthTicket,
    changed_files: List[str],
) -> List[str]:
    """Return gate violations. Non-empty means fail-closed when policy requires it."""
    violations: List[str] = []
    normalized_changed = [p.replace("\\", "/") for p in changed_files]

    for changed in normalized_changed:
        if file_matches_any(changed, ticket.forbidden_files):
            violations.append(f"forbidden_file_touched:{changed}")

    # Scope sanity: if expected files are known and none were touched, flag it.
    if ticket.expected_changed_files:
        touched_expected = any(
            file_matches_any(changed, ticket.expected_changed_files)
            for changed in normalized_changed
        )
        if not touched_expected:
            violations.append("expected_scope_not_touched")

    return violations


def summarize_validation(
    literal_cleanup_pass: bool,
    diagnostics_pass: bool,
    build_signal: Literal["pass", "fail", "not_run"],
) -> List[ValidationEvidenceRecord]:
    return [
        ValidationEvidenceRecord(
            check="literal_cleanup",
            result=ValidationResult.PASS if literal_cleanup_pass else ValidationResult.FAIL,
            evidence="literal scan completed",
        ),
        ValidationEvidenceRecord(
            check="diagnostics_changed_files",
            result=ValidationResult.PASS if diagnostics_pass else ValidationResult.FAIL,
            evidence="changed-file diagnostics computed",
        ),
        ValidationEvidenceRecord(
            check="build_signal",
            result=ValidationResult(build_signal),
            evidence="build used as non-blocking signal in phase 1",
        ),
    ]


def report_to_dict(report: TicketRunReport) -> Dict[str, Any]:
    """Stable serializer for artifacts."""
    return report.model_dump(mode="json")
