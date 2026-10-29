"""
Scope Policy Enforcer (Machine-Enforced Blast Radius & Boundary Guard)

Grounded in OpenAI Codex Harness & AI IDE Architecture:
1. Machine-Enforced Policy Object: Scope is a hard policy object enforced at the
   filesystem tool boundary, never trusted to prompt-based model discretion.
2. Root Triad: Explicitly maintains:
   - allowed_write_roots: Directories authorized for code mutation.
   - allowed_read_roots: Directories authorized for reading/context.
   - denied_write_roots: Explicit blacklists (backend microservices during UI tickets).
3. Candidate Ranking Scorer:
   score = lexical + semantic + proximity + ownership + recent_change - cross_project_penalty - lang_mismatch_penalty.
4. Structured Scope Expansion: Requires justification, evidence, proposed root, and risk assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ScopePolicyViolation(Exception):
    """Raised when an operation violates machine-enforced scope boundaries."""
    pass


@dataclass
class ScopeExpansionRequest:
    """Structured request required to expand the allowed write blast radius."""
    reason: str
    evidence: List[str]
    proposed_new_root: str
    risk: str = "medium"
    approved: bool = False


@dataclass
class CandidateScoreBreakdown:
    """Telemetry detailing candidate ranking scoring factors."""
    path: str
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    proximity_score: float = 0.0
    ownership_prior: float = 0.0
    recent_change_prior: float = 0.0
    cross_project_penalty: float = 0.0
    language_mismatch_penalty: float = 0.0

    @property
    def final_score(self) -> float:
        total = (
            self.lexical_score
            + self.semantic_score
            + self.proximity_score
            + self.ownership_prior
            + self.recent_change_prior
            - self.cross_project_penalty
            - self.language_mismatch_penalty
        )
        return round(max(0.0, total), 4)


class ScopePolicyEnforcer:
    """Enforces write boundaries and evaluates candidate relevance with hard policy gates."""

    def __init__(
        self,
        workspace_root: str,
        allowed_write_roots: Optional[List[str]] = None,
        allowed_read_roots: Optional[List[str]] = None,
        denied_write_roots: Optional[List[str]] = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.allowed_write_roots: List[Path] = [
            self._normalize_root(r) for r in (allowed_write_roots or [])
        ]
        self.allowed_read_roots: List[Path] = [
            self._normalize_root(r) for r in (allowed_read_roots or [])
        ]
        self.denied_write_roots: List[Path] = [
            self._normalize_root(r) for r in (denied_write_roots or [])
        ]
        self.expansion_history: List[ScopeExpansionRequest] = []

    def _normalize_root(self, root_str: str) -> Path:
        """Normalizes root paths relative to workspace root."""
        p = Path(root_str)
        if not p.is_absolute():
            p = (self.workspace_root / p).resolve()
        return p

    def _canonicalize(self, target_path: str) -> Path:
        """Resolves target path cleanly."""
        p = Path(target_path)
        if not p.is_absolute():
            p = (self.workspace_root / p).resolve()
        else:
            p = p.resolve()
        return p

    def can_read(self, file_path: str) -> bool:
        """Checks if file is authorized for reading."""
        target = self._canonicalize(file_path)
        if not self.allowed_read_roots:
            return True  # Default open read if none specified
        return any(self._is_subpath(target, r) for r in self.allowed_read_roots)

    def can_write(self, file_path: str) -> bool:
        """Checks if file is authorized for writing."""
        target = self._canonicalize(file_path)

        # 1. Deny write roots take absolute precedence
        for denied in self.denied_write_roots:
            if self._is_subpath(target, denied):
                return False

        # 2. Must be contained within at least one allowed write root
        if not self.allowed_write_roots:
            return False  # Fail closed: must explicitly define allowed roots
        return any(self._is_subpath(target, allowed) for allowed in self.allowed_write_roots)

    def validate_write_or_raise(self, file_path: str) -> None:
        """Validates write authorization or raises a fail-closed ScopePolicyViolation."""
        if not self.can_write(file_path):
            norm_path = str(self._canonicalize(file_path)).replace("\\", "/")
            denied_matching = [
                str(d).replace("\\", "/") for d in self.denied_write_roots if self._is_subpath(self._canonicalize(file_path), d)
            ]
            if denied_matching:
                raise ScopePolicyViolation(
                    f"Write denied to '{norm_path}'. Path is inside denied_write_roots: {denied_matching}"
                )
            allowed_strs = [str(a).replace("\\", "/") for a in self.allowed_write_roots]
            raise ScopePolicyViolation(
                f"Write denied to '{norm_path}'. Path is outside allowed_write_roots: {allowed_strs}"
            )

    def request_scope_expansion(self, request: ScopeExpansionRequest) -> bool:
        """Evaluates a structured scope expansion request."""
        self.expansion_history.append(request)

        # Low risk paths or internal component test folders can be approved
        low_risk_indicators = ["test", "spec", "mock", "fixture"]
        is_low_risk = any(ind in request.proposed_new_root.lower() for ind in low_risk_indicators)

        if request.risk.lower() == "low" or is_low_risk:
            request.approved = True
            new_root = self._normalize_root(request.proposed_new_root)
            self.allowed_write_roots.append(new_root)
            logger.info(f"Scope expansion auto-approved for low-risk path: {new_root}")
            return True

        logger.warning(
            f"Scope expansion requires explicit elevation: {request.proposed_new_root} (Risk: {request.risk})"
        )
        return False

    def rank_candidate(
        self,
        candidate_path: str,
        ticket_language: str,
        is_ui_ticket: bool,
        lexical_match: float = 0.0,
        semantic_match: float = 0.0,
        proximity: float = 0.0,
        ownership: float = 0.0,
        recent_change: float = 0.0,
    ) -> CandidateScoreBreakdown:
        """Location-aware candidate ranking function enforcing cross-service penalties."""
        norm_path = candidate_path.replace("\\", "/").lower()
        cross_project_penalty = 0.0
        lang_mismatch_penalty = 0.0

        # Language mismatch check
        is_java_file = norm_path.endswith(".java")
        is_ui_file = norm_path.endswith((".ts", ".html", ".scss", ".css", ".js"))

        if is_ui_ticket and is_java_file:
            cross_project_penalty = 2.5
            lang_mismatch_penalty = 2.0
        elif not is_ui_ticket and ticket_language.lower() == "java" and is_ui_file:
            cross_project_penalty = 2.5
            lang_mismatch_penalty = 2.0

        # Check against denied write roots
        target = self._canonicalize(candidate_path)
        if any(self._is_subpath(target, d) for d in self.denied_write_roots):
            cross_project_penalty += 5.0

        return CandidateScoreBreakdown(
            path=candidate_path,
            lexical_score=lexical_match,
            semantic_score=semantic_match,
            proximity_score=proximity,
            ownership_prior=ownership,
            recent_change_prior=recent_change,
            cross_project_penalty=cross_project_penalty,
            language_mismatch_penalty=lang_mismatch_penalty,
        )

    @staticmethod
    def _is_subpath(target: Path, parent: Path) -> bool:
        """Determines if target is equal to or located within parent directory."""
        try:
            target.relative_to(parent)
            return True
        except ValueError:
            return False
