"""
Fix Hypothesis Builder — Evidence layer between localization and AI reasoning.

Composes diagnostics, candidate locations, relationship context, and ticket
intent into structured FixHypothesis objects that the AI Reasoner can act on.

The AI receives:
    DIAGNOSTIC + SYMBOL + OWNER + DEFINITION + REFERENCES + RELATED FILES
    + CANDIDATE FIXES + TICKET INTENT + VIOLATED INVARIANT

Instead of just:
    "ERROR: TS2339..."

This is the correct use of an LLM: reasoning over structured evidence,
not file-guessing from raw error text.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.agents.diagnostic_normalizer import StructuredDiagnostic
    from ticket_to_code.agents.fix_localizer import (
        CandidateFixLocation,
        LocalizationResult,
    )
    from ticket_to_code.agents.relationship_analyzer import (
        Invariant,
        RelationshipContext,
    )

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FixHypothesis:
    """A structured hypothesis about what went wrong and possible corrections.

    This is what the AI Reasoner receives — not raw errors, not just candidates,
    but a STRUCTURED EVIDENCE PACKAGE for reasoning.

    The violated_invariant is structured (an Invariant object), not just a string.
    This allows post-patch verification:
        Before: invariant = BROKEN
        AI proposes fix
        After:  invariant = SATISFIED
    """
    diagnostic: "StructuredDiagnostic"
    violated_invariant: Optional["Invariant"]       # Structured, verifiable
    candidates: list["CandidateFixLocation"]
    evidence: list[str] = field(default_factory=list)
    question_for_ai: str = ""                       # The reasoning question
    ticket_context: str = ""                        # Why this change was intended

    def to_prompt_block(self) -> str:
        """Format as a structured evidence block for LLM consumption.

        Example output:
        ─────────────────────────────────────────────────
        HYPOTHESIS #1
        Diagnostic: TS2339 in add-members.component.html:25
        Message: Property 'selectedUserIsExistingProjectMember' does not exist on type 'AddMembersComponent'

        VIOLATED INVARIANT:
          [✗] template_binding: add-members.component.html → selectedUserIsExistingProjectMember
              (expected in add-members.component.ts)

        EVIDENCE:
          - AddMembersComponent is defined in add-members.component.ts
          - AddMembersComponent declares: [selectedMember, organizations, ...]
          - 'selectedUserIsExistingProjectMember' is NOT in the declared members
          - The ticket introduces project-membership behavior

        CANDIDATE FIX LOCATIONS:
          1. add-members.component.ts (confidence: 0.84, relationship: definition)
             → Add the missing property to AddMembersComponent
          2. add-members.component.html (confidence: 0.50, relationship: error_source)
             → The template binding uses the wrong name

        DETERMINE WHETHER:
          A. AddMembersComponent is missing a property that the ticket requires
          B. The HTML template incorrectly references a non-existent property
        ─────────────────────────────────────────────────
        """
        lines = ["─" * 60]

        # Diagnostic summary
        lines.append(f"Diagnostic: {self.diagnostic.code} "
                     f"in {Path(self.diagnostic.source_file).name}"
                     f":{self.diagnostic.line}")
        lines.append(f"Message: {self.diagnostic.message}")
        lines.append("")

        # Violated invariant
        if self.violated_invariant:
            lines.append("VIOLATED INVARIANT:")
            lines.append(f"  {self.violated_invariant}")
            lines.append("")

        # Evidence
        if self.evidence:
            lines.append("EVIDENCE:")
            for e in self.evidence:
                lines.append(f"  - {e}")
            lines.append("")

        # Ticket context
        if self.ticket_context:
            lines.append(f"TICKET CONTEXT: {self.ticket_context}")
            lines.append("")

        # Candidates
        if self.candidates:
            lines.append("CANDIDATE FIX LOCATIONS:")
            for i, cand in enumerate(self.candidates, 1):
                lines.append(f"  {i}. {cand.to_prompt_block()}")
            lines.append("")

        # Question
        if self.question_for_ai:
            lines.append("DETERMINE WHETHER:")
            lines.append(f"  {self.question_for_ai}")

        lines.append("─" * 60)
        return "\n".join(lines)


# ── Builder ───────────────────────────────────────────────────────────────────

class FixHypothesisBuilder:
    """Composes diagnostics, candidates, and context into structured hypotheses.

    Usage:
        builder = FixHypothesisBuilder()

        hypotheses = builder.build_hypotheses(
            diagnostics=diagnostics,
            localization_results=localization_results,
            relationship_context=rel_ctx,
            ticket_intent="Add project-membership state and organization dropdown",
        )

        # Format for LLM
        evidence_block = builder.format_for_prompt(hypotheses)
    """

    def build_hypotheses(
        self,
        diagnostics: list["StructuredDiagnostic"],
        localization_results: list["LocalizationResult"],
        relationship_context: Optional["RelationshipContext"] = None,
        ticket_intent: str = "",
    ) -> list[FixHypothesis]:
        """Build evidence-based hypotheses for the AI to reason about.

        Args:
            diagnostics: Normalized diagnostics from DiagnosticNormalizer.
            localization_results: Candidate locations from FixLocalizer.
            relationship_context: Cross-file relationships from RelationshipAnalyzer.
            ticket_intent: What the ticket/change is trying to accomplish.

        Returns:
            List of FixHypothesis objects, one per diagnostic.
        """
        hypotheses: list[FixHypothesis] = []

        for diag, loc_result in zip(diagnostics, localization_results):
            hypothesis = self._build_one(
                diag, loc_result, relationship_context, ticket_intent
            )
            hypotheses.append(hypothesis)

        logger.info(
            f"  [FixHypothesisBuilder] Built {len(hypotheses)} hypotheses "
            f"from {len(diagnostics)} diagnostics"
        )
        return hypotheses

    def format_for_prompt(self, hypotheses: list[FixHypothesis]) -> str:
        """Format all hypotheses into a single evidence block for the LLM prompt.

        This replaces the raw error text in the system prompt with
        structured reasoning material.
        """
        if not hypotheses:
            return ""

        sections = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║  DIAGNOSTIC INTELLIGENCE — EVIDENCE FOR ERROR RESOLUTION   ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"Total hypotheses: {len(hypotheses)}",
            "",
        ]

        for i, hyp in enumerate(hypotheses, 1):
            sections.append(f"HYPOTHESIS #{i}")
            sections.append(hyp.to_prompt_block())
            sections.append("")

        return "\n".join(sections)

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_one(
        self,
        diag: "StructuredDiagnostic",
        loc_result: "LocalizationResult",
        rel_ctx: Optional["RelationshipContext"],
        ticket_intent: str,
    ) -> FixHypothesis:
        """Build a single hypothesis from one diagnostic + its localization."""

        # 1. Find the most relevant violated invariant
        violated = self._find_violated_invariant(diag, rel_ctx)

        # 2. Gather evidence
        evidence = self._gather_evidence(diag, loc_result, rel_ctx)

        # 3. Formulate the reasoning question
        question = self._formulate_question(diag, loc_result)

        # 4. Extract ticket context
        context = ""
        if ticket_intent:
            context = _truncate(ticket_intent, 300)

        return FixHypothesis(
            diagnostic=diag,
            violated_invariant=violated,
            candidates=loc_result.candidates,
            evidence=evidence,
            question_for_ai=question,
            ticket_context=context,
        )

    def _find_violated_invariant(
        self,
        diag: "StructuredDiagnostic",
        rel_ctx: Optional["RelationshipContext"],
    ) -> Optional["Invariant"]:
        """Find the invariant that this diagnostic represents a violation of."""
        if not rel_ctx:
            return None

        # Look for invariants involving the diagnostic's source file and entities
        for inv in rel_ctx.invariants:
            # Match by symbol
            if inv.symbol in (diag.extracted_entities or []):
                return inv

            # Match by file
            if (inv.source_file and diag.source_file and
                    _norm(inv.source_file) == _norm(diag.source_file)):
                return inv

        # Return the first broken invariant if any
        broken = rel_ctx.get_broken_invariants()
        if broken:
            return broken[0]

        return None

    def _gather_evidence(
        self,
        diag: "StructuredDiagnostic",
        loc_result: "LocalizationResult",
        rel_ctx: Optional["RelationshipContext"],
    ) -> list[str]:
        """Collect all relevant evidence for the hypothesis."""
        evidence: list[str] = []

        # From the diagnostic itself
        if diag.code:
            evidence.append(f"Error code: {diag.code}")
        if diag.source_file:
            evidence.append(
                f"Reported in: {Path(diag.source_file).name} "
                f"(line {diag.line})"
            )

        # From extracted entities
        for entity in (diag.extracted_entities or []):
            evidence.append(f"Referenced symbol: '{entity}'")

        # From candidates
        for cand in loc_result.candidates:
            if cand.relationship == "definition" and cand.existing_members:
                evidence.append(
                    f"'{cand.symbol}' is defined in {Path(cand.file).name}"
                )
                # Check if the referenced symbol is missing
                for entity in (diag.extracted_entities or []):
                    if entity not in cand.existing_members:
                        evidence.append(
                            f"'{entity}' is NOT in the declared members of "
                            f"{Path(cand.file).name}"
                        )

        # From relationship context
        if rel_ctx:
            for rel in rel_ctx.relationships[:5]:
                evidence.append(
                    f"Relationship: {Path(rel.source_file).name} "
                    f"—[{rel.relationship_type}]→ "
                    f"{Path(rel.target_file).name}"
                )

        return evidence

    def _formulate_question(
        self,
        diag: "StructuredDiagnostic",
        loc_result: "LocalizationResult",
    ) -> str:
        """Create a clear reasoning question for the AI.

        The question frames the diagnostic as a choice between candidates,
        helping the LLM reason rather than guess.
        """
        if len(loc_result.candidates) < 2:
            if loc_result.candidates:
                cand = loc_result.candidates[0]
                return (
                    f"Fix this error in {Path(cand.file).name} — "
                    f"it is the only candidate location."
                )
            return "Determine the correct fix location for this error."

        # Build an A/B/C choice
        options: list[str] = []
        labels = "ABCDEFGH"

        for i, cand in enumerate(loc_result.candidates[:len(labels)]):
            label = labels[i]
            file_name = Path(cand.file).name
            options.append(f"{label}. {file_name} — {cand.reason}")

        return "\n  ".join(options)


# ── Utility ───────────────────────────────────────────────────────────────────

def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
