"""
Runnable orchestrator skeleton for the intake contract.

This does not replace the existing workflow. It provides a strict, testable
control loop that can be wired to real discovery/planning/patch/validation nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Protocol

from ticket_to_code.runtime.intake_contract import (
    ChangedFileRecord,
    DecisionRecord,
    GroundTruthTicket,
    OrchestratorPolicy,
    RunStatus,
    TicketRunReport,
    enforce_hard_gates,
)
from ticket_to_code.runtime.stage_depth import choose_stage_depth


class DiscoveryFn(Protocol):
    def __call__(self, ticket: GroundTruthTicket) -> List[str]: ...


class PlanningFn(Protocol):
    def __call__(self, ticket: GroundTruthTicket, candidates: List[str]) -> List[str]: ...


class PatchFn(Protocol):
    def __call__(self, ticket: GroundTruthTicket, plan_files: List[str]) -> List[str]: ...


class ValidateFn(Protocol):
    def __call__(self, ticket: GroundTruthTicket, changed_files: List[str]) -> bool: ...


@dataclass
class StageFns:
    discover: DiscoveryFn
    plan: PlanningFn
    patch: PatchFn
    validate: ValidateFn


@dataclass
class StageSignals:
    complexity: float = 0.5
    coupling: float = 0.5
    risk: float = 0.5
    certainty: float = 0.6
    constraint_strictness: float = 0.8


class ContractOrchestrator:
    """Enforces phase-1 policy: hard gates + bounded retries + evidence-first status."""

    def __init__(self, policy: OrchestratorPolicy | None = None) -> None:
        self.policy = policy or OrchestratorPolicy()

    def run_ticket(
        self,
        ticket: GroundTruthTicket,
        fns: StageFns,
        signals: StageSignals | None = None,
    ) -> TicketRunReport:
        findings: List[str] = []
        decisions: List[DecisionRecord] = []
        changed_files: List[str] = []
        retry_count = 0

        s = signals or StageSignals()
        depth = choose_stage_depth(
            complexity=s.complexity,
            coupling=s.coupling,
            risk=s.risk,
            certainty=s.certainty,
            constraint_strictness=s.constraint_strictness,
        )
        findings.append(f"stage_depth_path={depth.path}")
        findings.append(f"stage_depth_score={depth.score:.3f}")
        if s.certainty < 0.35:
            return TicketRunReport(
                ticket_id=ticket.ticket_id,
                status=RunStatus.NEED_MORE_INFO,
                findings=findings + ["certainty_below_0_35_after_discovery"],
                decisions=decisions,
                changed_files=[],
                residual_risks=["low_discovery_certainty"],
                retry_count=retry_count,
            )

        candidates = self._with_retry(
            stage_name="discovery",
            max_attempts=self.policy.retries.discovery,
            fn=lambda: fns.discover(ticket),
            on_retry=lambda: findings.append("discovery_retry"),
        )
        findings.append(f"candidate_files={len(candidates)}")

        plan_files = self._with_retry(
            stage_name="planning",
            max_attempts=self.policy.retries.planning,
            fn=lambda: fns.plan(ticket, candidates),
            on_retry=lambda: findings.append("planning_retry"),
        )
        decisions.append(
            DecisionRecord(
                decision="plan_files_selected",
                why="ownership and ticket scope mapping",
                alternatives_rejected=[],
            )
        )

        changed_files = self._with_retry(
            stage_name="patch_generation",
            max_attempts=self.policy.retries.patch_generation,
            fn=lambda: fns.patch(ticket, plan_files),
            on_retry=lambda: findings.append("patch_retry"),
        )

        violations = enforce_hard_gates(ticket, changed_files)
        if violations and self.policy.constraints.fail_closed_on_hard_gate_violation:
            return TicketRunReport(
                ticket_id=ticket.ticket_id,
                status=RunStatus.FAILED_CLOSED,
                findings=findings + violations,
                decisions=decisions,
                changed_files=[ChangedFileRecord(path=p, why="patched", symbols=[]) for p in changed_files],
                residual_risks=["hard_gate_violation"],
                retry_count=retry_count,
            )

        valid = fns.validate(ticket, changed_files)
        status = RunStatus.SOLVED if valid else RunStatus.NEEDS_RETRY
        if not valid:
            retry_count += 1

        return TicketRunReport(
            ticket_id=ticket.ticket_id,
            status=status,
            findings=findings,
            decisions=decisions,
            changed_files=[ChangedFileRecord(path=p, why="patched", symbols=[]) for p in changed_files],
            residual_risks=[] if valid else ["acceptance_not_fully_proven"],
            retry_count=retry_count,
        )

    @staticmethod
    def _with_retry(
        stage_name: str,
        max_attempts: int,
        fn: Callable[[], List[str]],
        on_retry: Callable[[], None],
    ) -> List[str]:
        attempts = 0
        last_exc: Exception | None = None
        while attempts <= max_attempts:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                attempts += 1
                if attempts > max_attempts:
                    break
                on_retry()
        raise RuntimeError(f"stage_failed:{stage_name}") from last_exc
