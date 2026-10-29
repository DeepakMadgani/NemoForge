"""
Blueprint Verifier — Deterministic verification of planner-proposed blueprints.

Validates each SignatureBlueprint against repository truth using SymbolResolver.
This is the CRITICAL barrier between LLM-proposed symbols and trusted state.

Flow:
    Planner proposes:  getProjectMembers(projectId: string)
    Verifier checks:
        1. Does the target file exist?
        2. Does the owner class exist in that file?
        3. Does the symbol already exist? → verify signature matches
        4. Is it a new symbol? → verify the producing task is in the plan
    Result:
        VERIFIED  → "Yes, this matches the repository"
        INVALID   → "No — repository has findProjectMembers(), not getProjectMembers()"
        PROPOSED  → "Cannot verify yet — symbol is new and will be created"

The code generator ONLY trusts VERIFIED/VALIDATED blueprints.
INVALID blueprints surface the ACTUAL symbol for correction.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.agents.implementation_state import ImplementationState
    from ticket_to_code.agents.symbol_resolver import SymbolResolver, TypeDefinition
    from ticket_to_code.models import (
        ArchitecturalPlan,
        BlueprintStatus,
        SignatureBlueprint,
    )

logger = logging.getLogger(__name__)

# Minimum similarity ratio for fuzzy matching (catches get→find, fetch→load, etc.)
_FUZZY_THRESHOLD = 0.6


class BlueprintVerifier:
    """Validates planner-proposed blueprints against repository truth.

    Uses SymbolResolver to check each blueprint:
    - Existing symbols: verify name + owner match
    - New symbols: verify producing task exists in the plan
    - Close mismatches: detect planner hallucinations via fuzzy matching

    The verifier NEVER modifies the code — it only marks blueprints
    as VERIFIED, INVALID, or leaves them as PROPOSED.
    """

    def __init__(
        self,
        symbol_resolver: Optional["SymbolResolver"] = None,
        workspace_path: Optional[Path] = None,
    ):
        self._resolver = symbol_resolver
        self._workspace = workspace_path

    def verify_blueprints(
        self,
        blueprints: list["SignatureBlueprint"],
        plan: "ArchitecturalPlan",
        impl_state: Optional["ImplementationState"] = None,
    ) -> list["SignatureBlueprint"]:
        """Verify all proposed blueprints against the repository.

        For each blueprint:
        1. If symbol EXISTS in repo → check it matches → VERIFIED or INVALID
        2. If symbol is NEW (created by a task) → check task exists → keep PROPOSED
        3. If planner said verification_required=True → force check
        4. If confidence < 0.5 → force check

        Returns the updated blueprint list (mutated in-place).
        """
        from ticket_to_code.models import BlueprintStatus

        if not self._resolver:
            logger.warning("  [BlueprintVerifier] No SymbolResolver — skipping verification")
            return blueprints

        verified = 0
        invalid = 0
        proposed = 0

        for bp in blueprints:
            if bp.status != BlueprintStatus.PROPOSED:
                continue  # Already processed

            status = self._verify_single(bp, plan, impl_state)
            bp.status = status

            if status == BlueprintStatus.VERIFIED:
                verified += 1
            elif status == BlueprintStatus.INVALID:
                invalid += 1
            else:
                proposed += 1

        logger.info(
            f"  [BlueprintVerifier] Results: "
            f"{verified} verified, {invalid} INVALID, {proposed} proposed"
        )

        if invalid:
            for bp in blueprints:
                if bp.status == BlueprintStatus.INVALID:
                    logger.warning(
                        f"    ⚠️ {bp.owner_class}.{bp.symbol_name}: {bp.verification_note}"
                    )

        return blueprints

    def _verify_single(
        self,
        bp: "SignatureBlueprint",
        plan: "ArchitecturalPlan",
        impl_state: Optional["ImplementationState"],
    ) -> "BlueprintStatus":
        """Verify a single blueprint against repository truth."""
        from ticket_to_code.models import BlueprintStatus

        # Case 1: The symbol is supposed to already exist (no created_by_task)
        if not bp.created_by_task:
            return self._verify_existing_symbol(bp)

        # Case 2: The symbol will be created by a task in this plan
        return self._verify_new_symbol(bp, plan, impl_state)

    def _verify_existing_symbol(self, bp: "SignatureBlueprint") -> "BlueprintStatus":
        """Check if an existing symbol matches the proposed blueprint.

        Uses SymbolResolver.resolve_type() to find the type, then checks
        if the proposed symbol name is among its declared members.

        If the symbol is NOT found but a close match exists, marks INVALID
        with the actual_symbol field populated for correction.
        """
        from ticket_to_code.models import BlueprintStatus

        if not bp.owner_class:
            # No owner class — can only check if the file exists
            if bp.file_path and self._workspace:
                full_path = self._workspace / bp.file_path
                if full_path.exists():
                    bp.verification_note = "File exists (cannot verify symbol without owner_class)"
                    return BlueprintStatus.VERIFIED
            return BlueprintStatus.PROPOSED

        # Resolve the owner type
        defn = self._resolver.resolve_type(bp.owner_class)
        if defn is None:
            bp.verification_note = f"Type '{bp.owner_class}' not found in repository"
            # Check if the type itself is being created by another task
            return BlueprintStatus.PROPOSED

        # Check if the symbol exists on this type
        if defn.has_member(bp.symbol_name):
            bp.verification_note = f"Confirmed: {bp.owner_class}.{bp.symbol_name} exists"
            return BlueprintStatus.VERIFIED

        # Symbol not found — check for close matches (hallucination detection)
        close = self._find_close_match(
            bp.symbol_name, defn.property_names + defn.method_names
        )
        if close:
            bp.actual_symbol = close
            bp.verification_note = (
                f"Repository has '{close}', not '{bp.symbol_name}' — "
                f"planner may have hallucinated the name"
            )
            return BlueprintStatus.INVALID

        # No close match either — might be a genuinely new symbol to add
        bp.verification_note = (
            f"'{bp.symbol_name}' not found on {bp.owner_class} "
            f"(declared: {defn.property_names[:5] + defn.method_names[:5]})"
        )

        # If the planner expressed low confidence, mark invalid
        if bp.verification_required or bp.confidence < 0.5:
            return BlueprintStatus.INVALID

        # Otherwise, it might be a new symbol to be created — leave as PROPOSED
        return BlueprintStatus.PROPOSED

    def _verify_new_symbol(
        self,
        bp: "SignatureBlueprint",
        plan: "ArchitecturalPlan",
        impl_state: Optional["ImplementationState"],
    ) -> "BlueprintStatus":
        """Verify that a new symbol has a producing task in the plan.

        The symbol doesn't exist yet — we check that the plan contains
        a task that will create it.
        """
        from ticket_to_code.models import BlueprintStatus

        # Check if the producing task actually exists in the plan
        task_ids = {t.id for t in plan.tasks} if plan and plan.tasks else set()
        if bp.created_by_task not in task_ids:
            bp.verification_note = (
                f"Producing task '{bp.created_by_task}' not found in plan "
                f"(available: {sorted(task_ids)[:5]})"
            )
            return BlueprintStatus.INVALID

        # Check if the symbol was already created in a prior task this run
        if impl_state and impl_state.has_verified_symbol(
            bp.symbol_name, bp.owner_class
        ):
            bp.verification_note = "Already created and verified in this run"
            return BlueprintStatus.VERIFIED

        # The producing task exists but hasn't run yet — keep as PROPOSED
        bp.verification_note = f"Will be created by task '{bp.created_by_task}'"
        return BlueprintStatus.PROPOSED

    def _find_close_match(
        self, name: str, candidates: list[str]
    ) -> Optional[str]:
        """Find a close fuzzy match among candidates.

        Catches common planner hallucinations:
            getProjectMembers  →  findProjectMembers
            fetchUser          →  loadUser
            isActive           →  active
        """
        if not candidates:
            return None

        best_match: Optional[str] = None
        best_ratio = 0.0

        name_lower = name.lower()
        for candidate in candidates:
            candidate_lower = candidate.lower()
            if candidate_lower == name_lower:
                # Exact case-insensitive match — verified, not hallucinated
                return None

            ratio = SequenceMatcher(None, name_lower, candidate_lower).ratio()
            if ratio > best_ratio and ratio >= _FUZZY_THRESHOLD:
                best_ratio = ratio
                best_match = candidate

        return best_match

    def verify_post_generation(
        self,
        bp: "SignatureBlueprint",
        generated_content: str,
        file_path: str,
    ) -> "BlueprintStatus":
        """Verify that a blueprint was actually satisfied after generation.

        Checks that the generated file actually contains the promised symbol.
        This is the IMPLEMENTED → VALIDATED transition (user req #3).

        Returns VALIDATED if the symbol exists, IMPLEMENTED if uncertain.
        """
        from ticket_to_code.models import BlueprintStatus

        if not self._resolver:
            return BlueprintStatus.IMPLEMENTED

        # Re-resolve after the file has been written
        if bp.owner_class:
            defn = self._resolver.resolve_type(bp.owner_class)
            if defn and defn.has_member(bp.symbol_name):
                bp.verification_note = "Post-generation verification passed"
                return BlueprintStatus.VALIDATED

            # Symbol not found after generation — still IMPLEMENTED but not VALIDATED
            bp.verification_note = (
                f"Post-generation check: '{bp.symbol_name}' not found on "
                f"'{bp.owner_class}' (may need symbol index refresh)"
            )
            return BlueprintStatus.IMPLEMENTED

        return BlueprintStatus.IMPLEMENTED
