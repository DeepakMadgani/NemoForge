"""
Dependency Context Builder — Bidirectional dependency expansion for code generation.

Combines three discovery directions:

Direction 1 — Blueprint-driven (explicit):
    Task B says it consumes MemberService.getProjectMembers()
    → Resolve from session files or SymbolResolver
    → Build import hints + type definitions
    VERIFIED blueprints shown as trusted; INVALID with correction; PROPOSED with warning.

Direction 2 — Code-driven (discovered):
    Generated code imports MemberService → resolve definition → discover related symbols.
    Catches cases where the planner missed an explicit `consumes`.

Direction 3 — Session-file discovery:
    Same-module files, description-referenced files.
    Migrated from current _build_session_context components.

The DependencyContextBuilder produces a context block that the CodeGenerator
injects into its prompt, right alongside the blueprint and rolling context
from ImplementationState.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.agents.implementation_state import ImplementationState
    from ticket_to_code.agents.symbol_resolver import SymbolResolver
    from ticket_to_code.models import (
        BlueprintStatus,
        DevelopmentTask,
        GenerationHandoff,
        SignatureBlueprint,
        VerifiedSymbol,
    )

logger = logging.getLogger(__name__)

# Common import-path patterns per language
_IMPORT_PATTERNS = {
    ".ts": re.compile(
        r"""(?:import|from)\s+(?:\{[^}]*\}\s+from\s+)?['"]([^'"]+)['"]""",
        re.MULTILINE,
    ),
    ".js": re.compile(
        r"""(?:import|from|require\s*\()\s*['"]([^'"]+)['"]""",
        re.MULTILINE,
    ),
    ".java": re.compile(r"import\s+([\w.]+);", re.MULTILINE),
    ".py": re.compile(
        r"(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
        re.MULTILINE,
    ),
}


class DependencyContextBuilder:
    """Builds dependency context for a code generation task.

    Combines three discovery directions into a single context block.
    The consumer (CodeGenerator) does not need to know which direction
    found each dependency — it just gets a clean, annotated context.
    """

    def __init__(
        self,
        symbol_resolver: Optional["SymbolResolver"] = None,
        workspace_path: Optional[Path] = None,
    ):
        self._resolver = symbol_resolver
        self._workspace = workspace_path

    def build_context(
        self,
        task: "DevelopmentTask",
        impl_state: "ImplementationState",
    ) -> str:
        """Build the dependency context block for a generation task.

        Returns a formatted string ready for prompt injection.
        """
        sections: list[str] = []

        # Direction 1: Blueprint-driven dependencies
        bp_section = self._blueprint_driven(task, impl_state)
        if bp_section:
            sections.append(bp_section)

        # Direction 2: Code-driven discovery (from already-generated files)
        code_section = self._code_driven(task, impl_state)
        if code_section:
            sections.append(code_section)

        # Direction 3: Session-file discovery (same module, description refs)
        session_section = self._session_file_discovery(task, impl_state)
        if session_section:
            sections.append(session_section)

        # Direction 4: Handoff-driven (verified cross-file knowledge)
        handoff_section = self._handoff_driven(task, impl_state)
        if handoff_section:
            sections.append(handoff_section)

        if not sections:
            return ""

        return (
            "═══ DEPENDENCY CONTEXT ═══\n"
            + "\n\n".join(sections)
            + "\n═══ END DEPENDENCY CONTEXT ═══\n"
        )

    # ── Direction 1: Blueprint-driven ────────────────────────────────────

    def _blueprint_driven(
        self,
        task: "DevelopmentTask",
        impl_state: "ImplementationState",
    ) -> str:
        """Resolve consumed blueprints for this task.

        For each consumed blueprint:
        - VERIFIED: show as trusted with signature + import path
        - INVALID: show the CORRECTED actual symbol
        - PROPOSED: show with a warning
        """
        from ticket_to_code.models import BlueprintStatus

        task_id = getattr(task, "id", "")
        _, consumes = impl_state.get_blueprints_for_task(task_id)

        if not consumes:
            return ""

        lines = ["── Blueprint Dependencies ──"]
        for bp in consumes:
            status_icon = {
                BlueprintStatus.EVIDENCE_VERIFIED: "✅",
                BlueprintStatus.BUILD_VALIDATED: "✅",
                BlueprintStatus.DEPENDENCY_VALIDATED: "✅",
                BlueprintStatus.IMPLEMENTED: "⚡",
                BlueprintStatus.IMPLEMENTED_CANDIDATE: "⚡",
                BlueprintStatus.PLANNED: "⚠️",
                BlueprintStatus.INVALID: "❌",
            }.get(bp.status, "?")

            if bp.status == BlueprintStatus.INVALID and bp.actual_symbol:
                # Show the CORRECTED symbol — this is the critical hallucination fix
                lines.append(
                    f"  {status_icon} CORRECTED: '{bp.actual_symbol}' "
                    f"(planner said '{bp.symbol_name}') — use the CORRECTED name"
                )
                # Try to resolve the actual symbol's definition
                actual_def = self._resolve_symbol_context(
                    bp.actual_symbol, bp.owner_class, impl_state
                )
                if actual_def:
                    lines.append(f"      {actual_def}")
            elif bp.status in (
                BlueprintStatus.EVIDENCE_VERIFIED,
                BlueprintStatus.BUILD_VALIDATED,
                BlueprintStatus.DEPENDENCY_VALIDATED,
            ):
                lines.append(
                    f"  {status_icon} [AUTHORITATIVE] {bp.owner_class}.{bp.symbol_name}"
                    f"({bp.signature})"
                )
                if bp.import_path:
                    lines.append(f"      import from: '{bp.import_path}'")

                # If the symbol was created in this run, show its signature
                sym_def = self._resolve_symbol_context(
                    bp.symbol_name, bp.owner_class, impl_state
                )
                if sym_def:
                    lines.append(f"      {sym_def}")
            elif bp.status == BlueprintStatus.PLANNED:
                lines.append(
                    f"  {status_icon} [PLANNER INTENT — NOT VERIFIED] {bp.owner_class}.{bp.symbol_name}"
                    f" — cannot confirm this symbol exists. Verify before using."
                )
            elif bp.status == BlueprintStatus.IMPLEMENTED_CANDIDATE:
                # Created but not yet validated
                lines.append(
                    f"  {status_icon} [CANDIDATE] {bp.owner_class}.{bp.symbol_name}"
                    f"({bp.signature}) [created this run, pending validation]"
                )
                if bp.import_path:
                    lines.append(f"      import from: '{bp.import_path}'")
            else:
                # IMPLEMENTED — validated creation in this run
                lines.append(
                    f"  {status_icon} [VERIFIED GENERATED] {bp.owner_class}.{bp.symbol_name}"
                    f"({bp.signature}) [created this run]"
                )
                if bp.import_path:
                    lines.append(f"      import from: '{bp.import_path}'")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ── Direction 2: Code-driven ─────────────────────────────────────────

    def _code_driven(
        self,
        task: "DevelopmentTask",
        impl_state: "ImplementationState",
    ) -> str:
        """Discover dependencies from imports in already-generated files.

        If the file being generated imports a module that was already generated
        in this run, we surface that module's API as context.
        """
        task_file = getattr(task, "file_path", "")
        if not task_file:
            return ""

        # Check if we already have generated content for the target file
        existing_content = impl_state.get_generated_content(task_file)

        # Determine file extension for import pattern matching
        ext = Path(task_file).suffix.lower()
        pattern = _IMPORT_PATTERNS.get(ext)
        if not pattern:
            return ""

        # If we don't have the content yet, nothing to analyze
        if not existing_content:
            return ""

        discovered: list[str] = []
        import_matches = pattern.findall(existing_content)
        for match in import_matches:
            # Handle Python's (from X import, import X) tuple
            import_path = match if isinstance(match, str) else next(
                (m for m in match if m), ""
            )
            if not import_path:
                continue

            # Check if this import resolves to a session file
            resolved = impl_state.resolve_import(import_path, task_file)
            if resolved and resolved in impl_state.generated_files:
                surface = impl_state.api_surfaces.get(resolved, "")
                if surface:
                    discovered.append(
                        f"  → {import_path} resolves to {resolved} (generated this run)\n"
                        f"    {surface}"
                    )

        if not discovered:
            return ""

        return (
            "── Code-Driven Dependencies (from imports) ──\n"
            + "\n".join(discovered)
        )

    # ── Direction 3: Session-file discovery ──────────────────────────────

    def _session_file_discovery(
        self,
        task: "DevelopmentTask",
        impl_state: "ImplementationState",
    ) -> str:
        """Discover related files by proximity and description references.

        Surfaces:
        - Same-module files (same directory, related naming)
        - Files referenced in the task description
        """
        task_file = getattr(task, "file_path", "")
        task_desc = getattr(task, "description", "")
        if not task_file:
            return ""

        task_dir = os.path.dirname(task_file.replace("\\", "/"))
        task_basename = Path(task_file).stem.lower()

        related: list[str] = []

        for gen_path, gen_content in impl_state.generated_files.items():
            gen_norm = gen_path.replace("\\", "/")
            if gen_norm.lower() == task_file.replace("\\", "/").lower():
                continue  # Skip self

            gen_dir = os.path.dirname(gen_norm)
            gen_basename = Path(gen_path).stem.lower()

            # Same directory → likely same module
            is_same_module = gen_dir.lower() == task_dir.lower()

            # Shared naming (e.g., member.service.ts + member.component.ts)
            shared_prefix = self._shared_prefix(task_basename, gen_basename)

            # Referenced in description
            is_referenced = gen_basename in task_desc.lower()

            if is_same_module or shared_prefix or is_referenced:
                surface = impl_state.api_surfaces.get(gen_path, "")
                reason = []
                if is_same_module:
                    reason.append("same module")
                if shared_prefix:
                    reason.append(f"shared prefix '{shared_prefix}'")
                if is_referenced:
                    reason.append("referenced in description")

                entry = f"  → {gen_path} ({', '.join(reason)})"
                if surface:
                    entry += f"\n    {surface}"
                related.append(entry)

        if not related:
            return ""

        return (
            "── Related Files (session discovery) ──\n"
            + "\n".join(related)
        )

    # ── Direction 4: Handoff-driven ──────────────────────────────────────

    def _handoff_driven(
        self,
        task: "DevelopmentTask",
        impl_state: "ImplementationState",
    ) -> str:
        """Build context from verified generation handoffs.

        KEY INVARIANT: This only retrieves handoffs for tasks that the current
        task explicitly depends on. For a 30-file ticket, Task 30 should only
        see handoffs from its actual dependencies, not all 29 previous tasks.

        Handoffs are AUTHORITATIVE — they represent verified implementation,
        not planner speculation. Authority labels are critical.
        """
        task_deps = getattr(task, "dependencies", [])
        if not task_deps:
            return ""

        # Get dependency-scoped handoffs
        handoffs = impl_state.get_handoffs_for_dependencies(task_deps)
        if not handoffs:
            return ""

        lines = ["── Verified Generation Handoffs (AUTHORITATIVE) ──"]
        for h in handoffs:
            # Authority label: these are verified facts, not predictions
            val_badge = "✅" if h.validation_status == "clean" else "⚠️"
            lines.append(
                f"  {val_badge} From task {h.task_id} → {Path(h.file_path).name}"
            )

            # Show exported symbols (these are VERIFIED to exist)
            # h.exports is List[VerifiedSymbol], not List[str]
            if h.exports:
                export_names = []
                for sym in h.exports[:15]:
                    name = getattr(sym, "name", str(sym))
                    owner = getattr(sym, "owner", "")
                    sig = getattr(sym, "signature", "")
                    if owner:
                        display = f"{owner}.{name}"
                    else:
                        display = name
                    if sig:
                        display += f"({sig})"
                    export_names.append(display)
                lines.append(f"    [AUTHORITATIVE] exports: {', '.join(export_names)}")

            # Show suggestive import path if available
            if h.how_to_consume:
                lines.append(f"    import suggestion: '{h.how_to_consume}' (verify against repo)")

            # Show what changed (AI explanation — suggestive)
            if h.what_changed:
                lines.append(f"    summary: {h.what_changed}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_symbol_context(
        self,
        symbol_name: str,
        owner_class: str,
        impl_state: "ImplementationState",
    ) -> str:
        """Try to resolve a symbol to its definition for context."""
        # Check verified symbols first (cheapest)
        key = f"{owner_class}.{symbol_name}" if owner_class else symbol_name
        verified = impl_state.verified_symbols.get(key)
        if verified:
            return f"signature: {verified.signature}"

        # Try SymbolResolver
        if self._resolver and owner_class:
            defn = self._resolver.resolve_type(owner_class)
            if defn and defn.has_member(symbol_name):
                return f"confirmed on {owner_class} (source: {defn.source})"

        return ""

    @staticmethod
    def _shared_prefix(a: str, b: str) -> str:
        """Find shared naming prefix between two basenames.

        E.g., "member.service" and "member.component" share "member".
        """
        parts_a = re.split(r"[.\-_]", a)
        parts_b = re.split(r"[.\-_]", b)
        shared = []
        for pa, pb in zip(parts_a, parts_b):
            if pa == pb:
                shared.append(pa)
            else:
                break
        prefix = ".".join(shared)
        # Only return if the shared prefix is meaningful (not just "index" etc.)
        return prefix if len(prefix) > 3 else ""
