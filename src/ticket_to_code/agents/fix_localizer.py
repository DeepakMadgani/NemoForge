"""
Fix Localizer — Candidate-based fix location discovery.

Given structured diagnostics, discovers MULTIPLE ranked candidate fix locations.
The definition tells you where the symbol SHOULD exist, not necessarily where
the defect IS.

Core principle:
    NEVER assume definition = fix location.

    A "createUser() not found" error could require fixing:
    - The Service (add the missing method)         → relationship="definition"
    - The Controller (fix the typo in the call)    → relationship="caller"
    - The Interface (add to the contract first)    → relationship="interface"

    All three are candidates. The AI Reasoner decides.

Uses:
    - SymbolResolver (Step 1) for type→file resolution
    - RelationshipAnalyzer (Step 3) for cross-file relationship context
    - ComponentStructureProvider for file grouping

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.agents.symbol_resolver import SymbolResolver, TypeDefinition
    from ticket_to_code.agents.relationship_analyzer import (
        RelationshipAnalyzer,
        RelationshipContext,
    )
    from ticket_to_code.agents.diagnostic_normalizer import StructuredDiagnostic
    from ticket_to_code.agents.component_structure_provider import (
        ComponentStructureProvider,
    )

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CandidateFixLocation:
    """One candidate location where a diagnostic MIGHT be resolved.

    Multiple candidates are returned for each diagnostic.
    The AI Reasoner decides which one is correct.

    confidence is a ranking signal (0.0-1.0), NOT a probability.
    It reflects how commonly this type of fix resolves this class of error,
    based on structural heuristics — not statistical certainty.
    """
    file: str                     # File that could be modified
    symbol: str                   # Specific symbol involved (if known)
    relationship: str             # "definition", "caller", "template_controller",
                                  # "interface", "barrel_export", "importer",
                                  # "error_source"
    reason: str                   # Human-readable explanation
    evidence: list[str] = field(default_factory=list)  # Supporting evidence
    confidence: float = 0.5       # Ranking signal (not probability)
    existing_members: list[str] = field(default_factory=list)  # What's already declared

    def to_prompt_block(self) -> str:
        """Format for LLM consumption."""
        lines = [
            f"  Candidate: {Path(self.file).name} "
            f"(confidence: {self.confidence:.2f}, relationship: {self.relationship})",
            f"    Reason: {self.reason}",
        ]
        if self.evidence:
            for e in self.evidence[:5]:
                lines.append(f"    Evidence: {e}")
        if self.existing_members:
            members_str = ", ".join(self.existing_members[:10])
            lines.append(f"    Existing members: [{members_str}]")
        return "\n".join(lines)


@dataclass
class LocalizationResult:
    """All candidate fix locations for a single diagnostic."""
    diagnostic_index: int         # Index into the diagnostics list
    diagnostic_summary: str       # Brief description of the diagnostic
    candidates: list[CandidateFixLocation] = field(default_factory=list)

    @property
    def top_candidate(self) -> Optional[CandidateFixLocation]:
        """Highest-confidence candidate."""
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.confidence)


# ── Localizer ─────────────────────────────────────────────────────────────────

class FixLocalizer:
    """Traces diagnostic symbols to definitions and discovers candidate fix locations.

    This is NOT a fix engine. It answers:
    "Given this broken thing, WHERE in the codebase MIGHT the fix go?"

    It outputs CANDIDATES, not fixes. The AI Reasoner decides.

    Usage:
        localizer = FixLocalizer(
            symbol_resolver=resolver,
            relationship_analyzer=analyzer,
        )

        results = localizer.localize(diagnostics, context_file="component.html")

        for result in results:
            print(f"Diagnostic: {result.diagnostic_summary}")
            for candidate in result.candidates:
                print(f"  {candidate.file} ({candidate.confidence:.2f})")
    """

    def __init__(
        self,
        symbol_resolver: Optional["SymbolResolver"] = None,
        relationship_analyzer: Optional["RelationshipAnalyzer"] = None,
        component_provider: Optional["ComponentStructureProvider"] = None,
    ):
        self._resolver = symbol_resolver
        self._rel_analyzer = relationship_analyzer
        self._provider = component_provider

    def localize(
        self,
        diagnostics: list["StructuredDiagnostic"],
        context_file: str = "",
    ) -> list[LocalizationResult]:
        """For each diagnostic, discover candidate fix locations.

        Args:
            diagnostics: Structured diagnostics from the normalizer.
            context_file: Optional file context to help resolve ambiguity.

        Returns:
            One LocalizationResult per diagnostic, each with ranked candidates.
        """
        results: list[LocalizationResult] = []

        for i, diag in enumerate(diagnostics):
            result = self._localize_one(i, diag, context_file)
            results.append(result)

        total_candidates = sum(len(r.candidates) for r in results)
        logger.info(
            f"  [FixLocalizer] {len(diagnostics)} diagnostics → "
            f"{total_candidates} candidates"
        )
        return results

    # ── Per-diagnostic localization ───────────────────────────────────────

    def _localize_one(
        self, index: int, diag: "StructuredDiagnostic", context_file: str
    ) -> LocalizationResult:
        """Discover candidates for a single diagnostic."""
        result = LocalizationResult(
            diagnostic_index=index,
            diagnostic_summary=f"{diag.code}: {diag.message[:120]}",
        )

        # Strategy 1: Always include the error source file as a candidate
        if diag.source_file:
            result.candidates.append(CandidateFixLocation(
                file=diag.source_file,
                symbol="",
                relationship="error_source",
                reason="Error was reported in this file",
                confidence=0.5,  # Baseline — often not the right fix location
            ))

        # Strategy 2: Resolve extracted entities to their definitions
        self._localize_by_symbols(diag, result)

        # Strategy 3: Use relationship analysis to find related files
        self._localize_by_relationships(diag, result)

        # Strategy 4: Error-code-specific heuristics
        self._localize_by_error_code(diag, result)

        # Sort candidates by confidence (highest first)
        result.candidates.sort(key=lambda c: c.confidence, reverse=True)

        # Deduplicate by file (keep highest confidence per file)
        result.candidates = self._deduplicate_candidates(result.candidates)

        return result

    def _localize_by_symbols(
        self, diag: "StructuredDiagnostic", result: LocalizationResult
    ):
        """Resolve extracted entities to their definition files."""
        if not self._resolver or not diag.extracted_entities:
            return

        for entity in diag.extracted_entities:
            # Try to resolve as a type/class name
            defn = self._resolver.resolve_type(entity)
            if defn and defn.file_path:
                is_same = _norm(defn.file_path) == _norm(diag.source_file)

                if not is_same:
                    result.candidates.append(CandidateFixLocation(
                        file=defn.file_path,
                        symbol=entity,
                        relationship="definition",
                        reason=f"'{entity}' is defined here — symbol may need to be added/modified",
                        evidence=[
                            f"Class: {defn.class_name}",
                            f"Source: {defn.source}",
                            f"Properties: {len(defn.property_names)}",
                            f"Methods: {len(defn.method_names)}",
                        ],
                        confidence=0.7,
                        existing_members=defn.property_names[:15] + defn.method_names[:15],
                    ))
                else:
                    # Symbol is defined in the same file as the error
                    # Boost the error_source candidate's confidence
                    for cand in result.candidates:
                        if cand.relationship == "error_source":
                            cand.confidence = max(cand.confidence, 0.75)
                            cand.symbol = entity
                            cand.evidence.append(
                                f"'{entity}' is defined in this same file"
                            )

    def _localize_by_relationships(
        self, diag: "StructuredDiagnostic", result: LocalizationResult
    ):
        """Use relationship analysis to find related files."""
        if not self._rel_analyzer or not diag.source_file:
            return

        try:
            # Focused analysis: what files are related to the error source?
            symbols = diag.extracted_entities or []
            rel_ctx = self._rel_analyzer.analyze_for_diagnostic(
                diag.source_file, symbols
            )
        except Exception as exc:
            logger.debug(f"  Relationship analysis failed for {diag.source_file}: {exc}")
            return

        # Add related files as candidates
        for rel in rel_ctx.relationships:
            target = rel.target_file
            if _norm(target) == _norm(diag.source_file):
                target = rel.source_file  # Use the other end

            # Skip if already a candidate
            if any(_norm(c.file) == _norm(target) for c in result.candidates):
                # But enrich the existing candidate with relationship info
                for cand in result.candidates:
                    if _norm(cand.file) == _norm(target):
                        if rel.relationship_type not in cand.evidence:
                            cand.evidence.append(
                                f"Relationship: {rel.relationship_type} "
                                f"(shared: {', '.join(rel.shared_symbols[:5])})"
                            )
                        # Boost confidence based on relationship type
                        cand.confidence = min(1.0, cand.confidence + 0.1)
                continue

            # Determine confidence based on relationship type
            confidence = _relationship_confidence(
                rel.relationship_type, diag.code
            )

            result.candidates.append(CandidateFixLocation(
                file=target,
                symbol=", ".join(rel.shared_symbols[:3]) if rel.shared_symbols else "",
                relationship=rel.relationship_type,
                reason=_relationship_reason(rel.relationship_type, target),
                evidence=[
                    f"Related via: {rel.relationship_type}",
                    f"Direction: {rel.direction}",
                ] + [f"Shared: {s}" for s in rel.shared_symbols[:5]],
                confidence=confidence,
            ))

    def _localize_by_error_code(
        self, diag: "StructuredDiagnostic", result: LocalizationResult
    ):
        """Apply error-code-specific heuristics to adjust confidence.

        Different error types have different likelihood distributions:
        - TS2339 (property missing): Definition file is most likely fix (0.8)
        - TS2304 (name not found): Could be import or typo (0.6 each)
        - Java "cannot find symbol": Similar to TS2339
        """
        if not diag.code:
            return

        code = diag.code.upper()

        # TS2339: Property 'X' does not exist on type 'Y'
        # → Most likely: add property to Y's definition file
        if code == "TS2339":
            for cand in result.candidates:
                if cand.relationship == "definition":
                    cand.confidence = min(1.0, cand.confidence + 0.14)
                    cand.evidence.append(
                        "TS2339 (missing property) → definition is most likely fix"
                    )
                elif cand.relationship == "error_source":
                    # Could also be a typo in the source file
                    cand.evidence.append(
                        "TS2339 → source file might have a typo or wrong reference"
                    )

        # TS2304: Cannot find name 'X'
        # → Could be missing import, typo, or missing declaration
        elif code == "TS2304":
            for cand in result.candidates:
                if cand.relationship == "error_source":
                    cand.confidence = max(cand.confidence, 0.65)
                    cand.evidence.append(
                        "TS2304 → likely missing import or typo in this file"
                    )

        # TS2322/TS2345: Type assignment errors
        # → Usually the source file needs adjustment
        elif code in ("TS2322", "TS2345"):
            for cand in result.candidates:
                if cand.relationship == "error_source":
                    cand.confidence = max(cand.confidence, 0.7)
                    cand.evidence.append(
                        f"{code} → type mismatch usually fixed in the consuming file"
                    )

        # Angular NG errors
        elif code.startswith("NG"):
            for cand in result.candidates:
                if cand.relationship == "template_controller":
                    cand.confidence = min(1.0, cand.confidence + 0.15)
                    cand.evidence.append(
                        f"{code} (Angular) → controller likely needs update"
                    )

    def _deduplicate_candidates(
        self, candidates: list[CandidateFixLocation]
    ) -> list[CandidateFixLocation]:
        """Keep only the highest-confidence candidate per file."""
        seen: dict[str, CandidateFixLocation] = {}
        for cand in candidates:
            key = _norm(cand.file)
            if key not in seen or cand.confidence > seen[key].confidence:
                # Merge evidence from duplicate
                if key in seen:
                    existing_evidence = seen[key].evidence
                    for ev in existing_evidence:
                        if ev not in cand.evidence:
                            cand.evidence.append(ev)
                seen[key] = cand
        return list(seen.values())


# ── Utility ───────────────────────────────────────────────────────────────────

def _norm(path: str) -> str:
    """Normalize path for comparison."""
    return path.replace("\\", "/").lower()


def _relationship_confidence(rel_type: str, error_code: str) -> float:
    """Base confidence for a relationship type as a fix candidate."""
    base_confidence = {
        "template_controller": 0.75,
        "caller_callee": 0.65,
        "interface_implementation": 0.60,
        "service_consumer": 0.60,
        "model_usage": 0.55,
        "barrel_export": 0.50,
        "module_declaration": 0.45,
        "symbol_reference": 0.60,
        "component_styles": 0.30,
        "test_subject": 0.20,
        "routing_declaration": 0.25,
        "associated": 0.40,
    }
    return base_confidence.get(rel_type, 0.40)


def _relationship_reason(rel_type: str, target: str) -> str:
    """Human-readable reason for a relationship-based candidate."""
    target_name = Path(target).name
    reasons = {
        "template_controller": f"'{target_name}' is the controller for this template",
        "caller_callee": f"'{target_name}' is called by or calls this file",
        "interface_implementation": f"'{target_name}' defines the interface contract",
        "service_consumer": f"'{target_name}' is a consumed/injected service",
        "model_usage": f"'{target_name}' defines the model/entity used here",
        "barrel_export": f"'{target_name}' re-exports symbols (barrel/index)",
        "module_declaration": f"'{target_name}' declares this component in its module",
        "symbol_reference": f"'{target_name}' is referenced by a symbol in this file",
    }
    return reasons.get(rel_type, f"'{target_name}' is related to this file")
