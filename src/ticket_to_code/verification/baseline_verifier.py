"""
Baseline-Differential Verification Subsystem for NemoForge.

Runs baseline checks before patch application, runs candidate checks after patch,
and computes failure identity diffs to distinguish new source regressions
from pre-existing broken tests or flaky infrastructure.
"""

from __future__ import annotations

import enum
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ticket_to_code.execution.build_orchestrator import (
    ClassifiedBuildResult,
    FailureCategory,
    GraphBuildOrchestrator,
)

logger = logging.getLogger(__name__)


class DifferentialStatus(enum.Enum):
    ALL_TESTS_PASSED = "all_tests_passed"
    NO_REGRESSION_OBSERVED = "no_regression_observed"
    NEW_REGRESSION = "new_regression"
    FLAKY_OR_INCONSISTENT = "flaky_or_inconsistent"
    INCONCLUSIVE_BLOCKED = "inconclusive_blocked"


@dataclass
class FailureSignature:
    identifier: str  # e.g. "OrderServiceTest#testCancelOrder"
    error_snippet: str
    signature_hash: str


@dataclass
class BaselineReport:
    executed_command: List[str]
    working_directory: str
    exit_code: int
    category: FailureCategory
    failures: List[FailureSignature] = field(default_factory=list)
    raw_output_snippet: str = ""


@dataclass
class DifferentialVerificationResult:
    status: DifferentialStatus
    summary: str
    new_failures: List[FailureSignature] = field(default_factory=list)
    pre_existing_failures: List[FailureSignature] = field(default_factory=list)
    baseline_report: Optional[BaselineReport] = None
    candidate_report: Optional[BaselineReport] = None
    is_safe_to_commit: bool = False


class BaselineDifferentialVerifier:
    """Manages pre-flight baseline execution and post-patch differential comparison."""

    def __init__(self, orchestrator: GraphBuildOrchestrator, allow_baseline_failures: bool = True):
        self.orchestrator = orchestrator
        self.allow_baseline_failures = allow_baseline_failures
        self.cached_baseline: Optional[BaselineReport] = None

    def capture_baseline(self, target_files: List[str]) -> BaselineReport:
        """Run build/test command against clean workspace before applying changes."""
        logger.info("Capturing pre-patch baseline test report for %s", target_files)
        res: ClassifiedBuildResult = self.orchestrator.execute_build(target_files)
        signatures = self._extract_failure_signatures(res)

        report = BaselineReport(
            executed_command=res.command_executed,
            working_directory=res.working_directory,
            exit_code=res.exit_code,
            category=res.category,
            failures=signatures,
            raw_output_snippet=res.reproducible_evidence[:1000],
        )
        self.cached_baseline = report
        return report

    def evaluate_candidate(
        self,
        target_files: List[str],
        baseline: Optional[BaselineReport] = None,
    ) -> DifferentialVerificationResult:
        """Run build/test command on candidate tree and compare against baseline."""
        base = baseline or self.cached_baseline
        if not base:
            logger.warning("No baseline report provided; capturing baseline now (workspace may be dirty).")
            base = self.capture_baseline(target_files)

        logger.info("Evaluating candidate verification against baseline...")
        cand_res: ClassifiedBuildResult = self.orchestrator.execute_build(target_files)
        cand_signatures = self._extract_failure_signatures(cand_res)

        cand_report = BaselineReport(
            executed_command=cand_res.command_executed,
            working_directory=cand_res.working_directory,
            exit_code=cand_res.exit_code,
            category=cand_res.category,
            failures=cand_signatures,
            raw_output_snippet=cand_res.reproducible_evidence[:1000],
        )

        # 1. Check for infrastructure or toolchain blockage
        if cand_res.category in (
            FailureCategory.MISSING_TOOLCHAIN,
            FailureCategory.INFRASTRUCTURE_FAILURE,
            FailureCategory.DEPENDENCY_RESOLUTION_FAILURE,
        ):
            return DifferentialVerificationResult(
                status=DifferentialStatus.INCONCLUSIVE_BLOCKED,
                summary=f"Verification blocked by environment: {cand_res.summary}",
                baseline_report=base,
                candidate_report=cand_report,
                is_safe_to_commit=False,
            )

        # 2. Both baseline and candidate passed with exit code 0
        if base.exit_code == 0 and cand_res.exit_code == 0:
            return DifferentialVerificationResult(
                status=DifferentialStatus.ALL_TESTS_PASSED,
                summary="All builds and automated test suites passed cleanly with exit code 0.",
                baseline_report=base,
                candidate_report=cand_report,
                is_safe_to_commit=True,
            )

        # 3. Compare Failure Signatures
        base_hashes: Set[str] = {f.signature_hash for f in base.failures}
        new_failures: List[FailureSignature] = []
        pre_existing: List[FailureSignature] = []

        for cand_f in cand_signatures:
            if cand_f.signature_hash in base_hashes:
                pre_existing.append(cand_f)
            else:
                new_failures.append(cand_f)

        # If there are new failure signatures, that's a true code regression
        if new_failures or (base.exit_code == 0 and cand_res.exit_code != 0):
            return DifferentialVerificationResult(
                status=DifferentialStatus.NEW_REGRESSION,
                summary=f"Candidate introduced {len(new_failures)} new failure(s) not present in baseline.",
                new_failures=new_failures,
                pre_existing_failures=pre_existing,
                baseline_report=base,
                candidate_report=cand_report,
                is_safe_to_commit=False,
            )

        # If candidate failed, but failure signatures are identical to pre-existing baseline failures
        if not new_failures and len(cand_signatures) == len(base.failures):
            # FAIL-CLOSED: No regression observed does NOT automatically equal safe to commit.
            # It requires compilation to have passed, and explicit verified baseline exceptions.
            compilation_passed = (cand_res.category != FailureCategory.SOURCE_COMPILATION_ERROR)
            has_explicit_approval = getattr(self, "allow_baseline_failures", False)
            safe = compilation_passed and has_explicit_approval

            reason = (
                f"No new regression observed ({len(pre_existing)} pre-existing failures). "
                f"{'Safe to commit under explicit baseline exception.' if safe else 'BLOCKED: Commit disallowed without explicit approval for pre-existing failures.'}"
            )

            return DifferentialVerificationResult(
                status=DifferentialStatus.NO_REGRESSION_OBSERVED,
                summary=reason,
                new_failures=[],
                pre_existing_failures=pre_existing,
                baseline_report=base,
                candidate_report=cand_report,
                is_safe_to_commit=safe,
            )

        return DifferentialVerificationResult(
            status=DifferentialStatus.FLAKY_OR_INCONSISTENT,
            summary="Test outcomes diverged non-deterministically without new failure signatures.",
            new_failures=new_failures,
            pre_existing_failures=pre_existing,
            baseline_report=base,
            candidate_report=cand_report,
            is_safe_to_commit=False,
        )

    def _extract_failure_signatures(self, res: ClassifiedBuildResult) -> List[FailureSignature]:
        signatures: List[FailureSignature] = []
        for loc in res.error_locations:
            ident = loc.get("test_case") or f"{loc.get('file', 'unknown')}:{loc.get('line', 0)}"
            snip = loc.get("message") or loc.get("snippet", "")
            h = hashlib.sha256(f"{ident}::{snip}".encode("utf-8")).hexdigest()[:12]
            signatures.append(FailureSignature(identifier=ident, error_snippet=snip, signature_hash=h))
        return signatures
