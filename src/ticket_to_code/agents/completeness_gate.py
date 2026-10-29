"""
Completeness Gate — Post-Build Semantic & Invariant Verification.

Verifies:
1. Build Success: compiler passed, 0 remaining diagnostics.
2. Semantic Requirement Coverage:
   Requirement → Evidence → ChangeTarget → Patch → Validation.
   Every requirement must be mapped to an implemented, verified target.
3. Target & Scope Authorization:
   All modified files are in authorized writable scope; all patches target authorized ChangeTargets.
4. Ticket-Dependent Test Requirement:
   If ticket explicitly demands tests → tests must pass.
   If ticket does not demand tests → record 'not_required' (does NOT fail completeness).
5. Cross-Artifact Consistency:
   No dangling references between companion files (e.g. Angular HTML ↔ TS).
6. Persistence:
   Persists completion manifest to `.aviator/tickets/<ticket_id>.json`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Set, Any, Union

from ticket_to_code.models import ChangeTarget

logger = logging.getLogger(__name__)


@dataclass
class RequirementCoverageResult:
    requirement_id: str
    description: str
    target_symbol: Optional[str] = None
    target_file: Optional[str] = None
    is_covered: bool = False
    verification_status: str = "PENDING"  # VERIFIED | MISSING_TARGET | PATCH_FAILED


@dataclass
class CompletenessVerdict:
    is_complete: bool
    status: str  # "COMPLETED" | "INCOMPLETE"
    summary: str
    build_passed: bool
    test_status: str  # "passed" | "failed" | "not_required"
    requirements_coverage: List[RequirementCoverageResult] = field(default_factory=list)
    uncovered_requirements: List[str] = field(default_factory=list)
    unauthorized_files_detected: List[str] = field(default_factory=list)
    unauthorized_targets_detected: List[str] = field(default_factory=list)
    cross_artifact_passed: bool = True
    persisted_path: Optional[str] = None


class CompletenessGate:
    """Verifies that a run satisfies all architectural and semantic invariants before completion."""

    @classmethod
    def evaluate(
        cls,
        ticket_id: str,
        ticket_title: str,
        ticket_description: str,
        requirements: List[Dict[str, Any]],
        change_targets: List[ChangeTarget],
        modified_files: Set[str],
        authorized_files: Set[str],
        build_passed: bool,
        tests_passed: Optional[bool] = None,
        cross_artifact_passed: bool = True,
        remaining_diagnostics: Optional[List[Any]] = None,
        workspace_path: Optional[Union[str, Path]] = None,
    ) -> CompletenessVerdict:
        """Evaluate completeness across all criteria."""
        remaining_diagnostics = remaining_diagnostics or []
        uncovered: List[str] = []
        coverage_results: List[RequirementCoverageResult] = []

        # 1. Build & Diagnostic Check
        actual_build_ok = build_passed and len(remaining_diagnostics) == 0

        # 2. File Scope Authorization Check
        unauthorized_files = []
        norm_auth = {a.replace("\\", "/").lower().strip("/") for a in authorized_files}
        for mf in modified_files:
            norm_m = mf.replace("\\", "/").lower().strip("/")
            is_auth = any(norm_m == a or norm_m.endswith("/" + a) or a.endswith("/" + norm_m) for a in norm_auth)
            if not is_auth:
                unauthorized_files.append(mf)

        # 3. ChangeTarget Authorization Check
        unauthorized_targets = []
        for ct in change_targets:
            if not ct.is_authorized and ct.file_path in modified_files:
                unauthorized_targets.append(ct.symbol or ct.file_path)

        # 4. Ticket-Dependent Test Requirement Check
        # Distinguish tickets that explicitly require tests vs those that don't
        tests_demanded = cls._ticket_requires_tests(ticket_title, ticket_description, requirements)
        if not tests_demanded:
            test_status = "not_required"
            test_ok = True
        else:
            if tests_passed is True:
                test_status = "passed"
                test_ok = True
            else:
                test_status = "failed"
                test_ok = False

        # 5. Semantic Requirement → ChangeTarget → Implementation Coverage
        # Check that every requirement has at least one matching authorized ChangeTarget
        for req in requirements:
            req_id = req.get("id") or req.get("requirement_id") or "REQ-?"
            req_desc = req.get("text") or req.get("description") or ""
            
            # Find matching ChangeTarget
            matching_ct = None
            for ct in change_targets:
                if isinstance(ct, str):
                    norm_str = ct.replace("\\", "/").lower()
                    if any(norm_str.endswith(mf.replace("\\", "/").lower()) or mf.replace("\\", "/").lower().endswith(norm_str) for mf in modified_files):
                        matching_ct = ct
                        break
                    continue
                # Direct evidence link or textual association
                if req_id in (getattr(ct, "evidence_ids", []) or []):
                    matching_ct = ct
                    break
                # Check symbol or intent matching requirement text
                if getattr(ct, "symbol", None) and ct.symbol.lower() in req_desc.lower():
                    matching_ct = ct
                    break
                if getattr(ct, "modification_intent", None) and any(w in ct.modification_intent.lower() for w in req_desc.lower().split() if len(w) > 4):
                    matching_ct = ct
                    break

            ct_path = getattr(matching_ct, "file_path", None) or (matching_ct if isinstance(matching_ct, str) else "")
            ct_is_auth = getattr(matching_ct, "is_authorized", True) if not isinstance(matching_ct, str) else True
            ct_symbol = getattr(matching_ct, "symbol", None)


            if matching_ct and ct_is_auth and ct_path:
                # Target exists and is authorized
                norm_ct_file = ct_path.replace("\\", "/").lower()
                file_was_modified = any(
                    mf.replace("\\", "/").lower().endswith(norm_ct_file) or norm_ct_file.endswith(mf.replace("\\", "/").lower())
                    for mf in modified_files
                )
                if file_was_modified:
                    coverage_results.append(RequirementCoverageResult(
                        requirement_id=req_id,
                        description=req_desc,
                        target_symbol=ct_symbol,
                        target_file=ct_path,
                        is_covered=True,
                        verification_status="VERIFIED",
                    ))
                else:
                    uncovered.append(f"{req_id}: Target file '{ct_path}' was not modified")
                    coverage_results.append(RequirementCoverageResult(
                        requirement_id=req_id,
                        description=req_desc,
                        target_symbol=ct_symbol,
                        target_file=ct_path,
                        is_covered=False,
                        verification_status="PATCH_FAILED",
                    ))
            elif requirements:
                # If no requirements were defined, we don't fail, but if requirements exist, require coverage
                uncovered.append(f"{req_id}: No authorized change target implements this requirement")
                coverage_results.append(RequirementCoverageResult(
                    requirement_id=req_id,
                    description=req_desc,
                    is_covered=False,
                    verification_status="MISSING_TARGET",
                ))

        # Overall Completeness Decision
        has_uncovered = len(uncovered) > 0 and len(requirements) > 0
        is_complete = (
            actual_build_ok
            and test_ok
            and len(unauthorized_files) == 0
            and len(unauthorized_targets) == 0
            and cross_artifact_passed
            and not has_uncovered
        )

        status_str = "COMPLETED" if is_complete else "INCOMPLETE"
        
        # Summary description
        reasons = []
        if not actual_build_ok:
            reasons.append("Build failed or compiler diagnostics remain")
        if not test_ok:
            reasons.append("Required tests failed")
        if unauthorized_files:
            reasons.append(f"Unauthorized files modified: {unauthorized_files}")
        if unauthorized_targets:
            reasons.append(f"Unauthorized targets modified: {unauthorized_targets}")
        if not cross_artifact_passed:
            reasons.append("Cross-artifact consistency violation")
        if has_uncovered:
            reasons.append(f"Uncovered requirements: {len(uncovered)}")

        summary = f"Ticket {status_str}" + (f": {'; '.join(reasons)}" if reasons else ". All invariants satisfied.")

        # 6. Persistence
        persisted_path = None
        if workspace_path:
            persisted_path = cls._persist_manifest(
                workspace_path=Path(workspace_path),
                ticket_id=ticket_id,
                verdict_data={
                    "ticket_id": ticket_id,
                    "title": ticket_title,
                    "status": status_str,
                    "evaluated_at": datetime.now().isoformat(),
                    "build_passed": actual_build_ok,
                    "test_status": test_status,
                    "cross_artifact_passed": cross_artifact_passed,
                    "requirements_coverage": [
                        {
                            "id": r.requirement_id,
                            "description": r.description,
                            "target_symbol": r.target_symbol,
                            "target_file": r.target_file,
                            "status": r.verification_status,
                        }
                        for r in coverage_results
                    ],
                    "uncovered_requirements": uncovered,
                    "unauthorized_files": unauthorized_files,
                    "unauthorized_targets": unauthorized_targets,
                }
            )

        return CompletenessVerdict(
            is_complete=is_complete,
            status=status_str,
            summary=summary,
            build_passed=actual_build_ok,
            test_status=test_status,
            requirements_coverage=coverage_results,
            uncovered_requirements=uncovered,
            unauthorized_files_detected=unauthorized_files,
            unauthorized_targets_detected=unauthorized_targets,
            cross_artifact_passed=cross_artifact_passed,
            persisted_path=persisted_path,
        )

    @classmethod
    def _ticket_requires_tests(
        cls, ticket_title: str, ticket_description: str, requirements: List[Dict[str, Any]]
    ) -> bool:
        """Check whether the ticket explicitly demands unit/integration tests."""
        corpus = f"{ticket_title}\n{ticket_description}".lower()
        test_keywords = [
            "unit test",
            "integration test",
            "tests required",
            "write test",
            "add test",
            "test coverage",
            "must include test",
        ]
        if any(kw in corpus for kw in test_keywords):
            return True

        for r in requirements:
            if r.get("requires_test") or r.get("test_required"):
                return True
            r_text = (r.get("text") or r.get("description") or "").lower()
            if any(kw in r_text for kw in test_keywords):
                return True

        return False

    @classmethod
    def _persist_manifest(cls, workspace_path: Path, ticket_id: str, verdict_data: dict) -> str:
        """Save verification state to .aviator/tickets/<ticket_id>.json."""
        try:
            target_dir = workspace_path / ".aviator" / "tickets"
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in ticket_id)
            manifest_file = target_dir / f"{safe_id}.json"
            manifest_file.write_text(json.dumps(verdict_data, indent=2), encoding="utf-8")
            logger.info(f"CompletenessGate: Saved completion manifest to {manifest_file}")
            return str(manifest_file)
        except Exception as exc:
            logger.warning(f"CompletenessGate: Failed to persist manifest: {exc}")
            return ""
