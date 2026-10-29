"""
Central Policy Engine for NemoForge.

Unified arbiter governing edit safety, build scoping, failure actionability,
retry justification, and honest verification claims. Enforces hard safety invariants.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ticket_to_code.execution.build_orchestrator import ClassifiedBuildResult, FailureCategory
from ticket_to_code.intelligence.repository_intelligence import RepositoryCapabilityModel
from ticket_to_code.verification.baseline_verifier import DifferentialStatus, DifferentialVerificationResult

logger = logging.getLogger(__name__)


class PolicyAction(enum.Enum):
    PROCEED_TO_EDIT = "proceed_to_edit"
    PROCEED_TO_VERIFY = "proceed_to_verify"
    TRIGGER_CODE_REPAIR = "trigger_code_repair"
    ESCALATE_ENVIRONMENT_ERROR = "escalate_environment_error"
    ABSTAIN_UNSAFE = "abstain_unsafe"
    ABSTAIN_INCONCLUSIVE = "abstain_inconclusive"
    COMMIT_SUCCESS = "commit_success"


@dataclass
class PolicyDecision:
    action: PolicyAction
    reason: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


class PolicyEngine:
    """Central safety and orchestration arbiter."""

    def __init__(self, capability_model: RepositoryCapabilityModel):
        self.capability_model = capability_model

    def evaluate_edit_safety(
        self,
        target_file: str,
        evidence_snippets: List[str],
        proposed_patch: str,
    ) -> PolicyDecision:
        """Determines whether evidence is sufficient and format strategy is safe."""
        ws_root = Path(self.capability_model.workspace_root)
        target_path = ws_root / target_file

        # Invariant 1: File must exist or be an explicit create task
        if not target_path.exists() and not proposed_patch.startswith("CREATE:"):
            return PolicyDecision(
                action=PolicyAction.ABSTAIN_UNSAFE,
                reason=f"Target file '{target_file}' does not exist on disk and is not a planned creation.",
                confidence=0.0,
                limitations=["TARGET_FILE_NOT_FOUND"],
            )

        # Invariant 2: Target must have discovered evidence
        if not evidence_snippets and target_path.exists():
            return PolicyDecision(
                action=PolicyAction.ABSTAIN_UNSAFE,
                reason=f"Zero evidence items collected for target '{target_file}'. Refusing blind modification.",
                confidence=0.0,
                limitations=["INSUFFICIENT_EVIDENCE"],
            )

        # Invariant 3: Determine editing capability
        ext = target_path.suffix.lower()
        strategy = self.capability_model.editing_strategy_map.get(ext, "conservative_source_preserving")

        return PolicyDecision(
            action=PolicyAction.PROCEED_TO_EDIT,
            reason=f"Target validated with strategy '{strategy}' and {len(evidence_snippets)} evidence items.",
            confidence=0.9,
            evidence=[f"Strategy: {strategy}", f"Evidence count: {len(evidence_snippets)}"],
        )

    def evaluate_build_failure(
        self,
        build_result: ClassifiedBuildResult,
        remaining_retries: int,
    ) -> PolicyDecision:
        """Determines whether a build failure warrants code repair vs environmental escalation."""
        if build_result.success:
            return PolicyDecision(
                action=PolicyAction.PROCEED_TO_VERIFY,
                reason="Build succeeded cleanly with exit code 0.",
                confidence=1.0,
            )

        # Category check: only source compilation or test assertion regressions can trigger code repair
        if build_result.category in (
            FailureCategory.SOURCE_COMPILATION_ERROR,
            FailureCategory.TEST_ASSERTION_REGRESSION,
        ):
            if remaining_retries > 0:
                return PolicyDecision(
                    action=PolicyAction.TRIGGER_CODE_REPAIR,
                    reason=f"Source-related failure: {build_result.summary}. Retries remaining: {remaining_retries}.",
                    confidence=0.85,
                    evidence=[build_result.reproducible_evidence[:500]],
                )
            else:
                return PolicyDecision(
                    action=PolicyAction.ABSTAIN_UNSAFE,
                    reason="Exhausted maximum retry budget without resolving compilation/test regression.",
                    confidence=0.0,
                    limitations=["RETRY_BUDGET_EXHAUSTED"],
                )

        # Configuration, toolchain, or infrastructure failures must NEVER trigger code repair!
        return PolicyDecision(
            action=PolicyAction.ESCALATE_ENVIRONMENT_ERROR,
            reason=(
                f"Refusing code repair for non-code failure category: {build_result.category.value}. "
                f"Summary: {build_result.summary}"
            ),
            confidence=0.0,
            evidence=[build_result.reproducible_evidence[:500]],
            limitations=[f"ENVIRONMENT_BLOCKED_{build_result.category.name}"],
        )

    def evaluate_verification_outcome(
        self,
        diff_result: DifferentialVerificationResult,
    ) -> PolicyDecision:
        """Evaluates differential verification results to determine whether commit is safe."""
        if diff_result.status == DifferentialStatus.ALL_TESTS_PASSED:
            return PolicyDecision(
                action=PolicyAction.COMMIT_SUCCESS,
                reason="All test suites and static compiler checks passed with exit code 0.",
                confidence=1.0,
            )

        if diff_result.status == DifferentialStatus.NO_REGRESSION_OBSERVED:
            if diff_result.is_safe_to_commit:
                return PolicyDecision(
                    action=PolicyAction.COMMIT_SUCCESS,
                    reason=(
                        f"No new regression observed. All remaining failures ({len(diff_result.pre_existing_failures)}) "
                        f"were confirmed pre-existing in baseline and explicitly approved."
                    ),
                    confidence=0.95,
                    evidence=[f"Pre-existing count: {len(diff_result.pre_existing_failures)}"],
                )
            else:
                return PolicyDecision(
                    action=PolicyAction.ABSTAIN_UNSAFE,
                    reason=(
                        f"Candidate blocked: {len(diff_result.pre_existing_failures)} baseline failure(s) remain unapproved. "
                        f"Fail-closed policy requires clean verification gates and explicit baseline waiver."
                    ),
                    confidence=0.0,
                    limitations=["UNAPPROVED_BASELINE_FAILURES"],
                )

        if diff_result.status == DifferentialStatus.INCONCLUSIVE_BLOCKED:
            return PolicyDecision(
                action=PolicyAction.ABSTAIN_INCONCLUSIVE,
                reason=f"Verification was inconclusive or blocked by environment: {diff_result.summary}",
                confidence=0.0,
                limitations=["INCONCLUSIVE_TEST_ENVIRONMENT"],
            )

        # New regression or flaky test
        return PolicyDecision(
            action=PolicyAction.ABSTAIN_UNSAFE,
            reason=f"Candidate introduced new failure regressions: {diff_result.summary}",
            confidence=0.0,
            evidence=[f.identifier for f in diff_result.new_failures[:5]],
            limitations=["NEW_SOURCE_REGRESSION"],
        )
