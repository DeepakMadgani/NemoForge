"""
Implementation State — Central shared state for multi-file code generation.

Connects the four pillars of the Cross-File Knowledge Consistency architecture:
    Pillar 1 (Blueprint):      signature_blueprints, semantic_contract
    Pillar 2 (Rolling Context): generated_files, verified_symbols, api_surfaces
    Pillar 3 (Dependencies):    dependency_graph, import_resolution
    Pillar 4 (Validation):      diagnostics, validation_results

Design constraints:
    - This is the SINGLE SOURCE OF TRUTH for generated-file state.
    - No subsystem may maintain an independent copy of generated_files.
    - CodeGenerator._session_files, EditLoopAgent._written_files, and
      workflow._run_generated_map should all read/write through this object.
    - Bridge methods (from_workflow_state / sync_to_workflow_state) provide
      temporary compatibility with the existing workflow.py state dict.
      This bridge is transitional infrastructure, not a permanent dual-state.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.models import (
        ArchitecturalPlan,
        BlueprintStatus,
        ContextTier,
        CrossFileContract,
        DevelopmentTask,
        GenerationHandoff,
        SignatureBlueprint,
        VerifiedSymbol as VerifiedSymbolModel,
    )
    from ticket_to_code.agents.diagnostic_normalizer import StructuredDiagnostic
    from ticket_to_code.agents.fix_hypothesis_builder import FixHypothesis
    from ticket_to_code.agents.semantic_contract import SemanticContract

logger = logging.getLogger(__name__)


# ── Supporting data classes ──────────────────────────────────────────────────

@dataclass
class VerifiedSymbol:
    """A symbol confirmed to exist in a generated (or existing) file.

    Created after post-generation verification passes —
    i.e., SymbolResolver confirmed the symbol is actually present.
    """
    symbol_name: str
    owner_class: str
    file_path: str
    signature: str              # Full signature string
    verified_from: str          # "session_file", "lsp", "ast_scan", "symbol_resolver"


@dataclass
class ValidationResult:
    """Result of incremental validation for a single file.

    Tracks what scope was checked and what diagnostics were found.
    The validation_scope determines how much of the codebase was included:
    file → module → package → workspace (from cheapest to most expensive).
    """
    file_path: str
    is_clean: bool
    diagnostics: list = field(default_factory=list)  # list[StructuredDiagnostic]
    validation_scope: str = "file"                   # "file", "module", "package", "workspace"
    timestamp: float = field(default_factory=time.time)


@dataclass
class ContextEscalation:
    """Records a context tier escalation for telemetry.

    When the system starts at SIGNATURE and needs to escalate to
    TYPE_CONTRACT or beyond, this record captures why.
    """
    file_path: str
    from_tier: str          # ContextTier value
    to_tier: str            # ContextTier value
    reason: str             # Why escalation was needed
    timestamp: float = field(default_factory=time.time)


@dataclass
class PatchCheckpoint:
    """Pre-patch state for transaction/rollback semantics.

    Before applying any patch, capture the file state so we can
    roll back if validation fails.
    """
    file_path: str
    content_before: str
    patch_description: str
    timestamp: float = field(default_factory=time.time)


# ── Central state ────────────────────────────────────────────────────────────

@dataclass
class ImplementationState:
    """Central shared state connecting all five pillars.

    This is the SINGLE SOURCE OF TRUTH during a multi-file code generation run.
    Every pillar reads from it and writes back to it.

    Usage:
        state = ImplementationState()

        # Pillar 1: Register blueprints from the plan
        state.signature_blueprints = plan.signature_blueprints

        # Pillar 2: Record generated files
        state.update_generated_file("service.ts", content)

        # Pillar 3: Query dependencies
        deps = state.get_dependencies("component.ts")

        # Pillar 4: Record validation
        state.record_validation("service.ts", result)

        # Pillar 5: Record handoffs
        state.add_handoff(handoff)

        # Cross-pillar: Build context for a generation task
        context = state.to_prompt_context(task)
    """

    # ── Pillar 1: Blueprint ──
    signature_blueprints: list = field(default_factory=list)  # list[SignatureBlueprint]
    semantic_contract: Optional[object] = None                # SemanticContract

    # ── Pillar 2: Rolling Context ──
    generated_files: dict[str, str] = field(default_factory=dict)      # path → content
    component_contracts: dict[str, object] = field(default_factory=dict) # path → ComponentContract
    verified_symbols: dict[str, VerifiedSymbol] = field(default_factory=dict)
    api_surfaces: dict[str, str] = field(default_factory=dict)         # path → extracted surface
    context_escalations: list[ContextEscalation] = field(default_factory=list)

    # ── Pillar 3: Dependency Graph ──
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    import_resolution: dict[str, str] = field(default_factory=dict)    # import_path → resolved_file

    # ── Pillar 4: Validation ──
    diagnostics: list = field(default_factory=list)                    # list[StructuredDiagnostic]
    validation_results: dict[str, ValidationResult] = field(default_factory=dict)
    fix_candidates: list = field(default_factory=list)                 # list[FixHypothesis]

    # ── Transaction/Rollback (user req #7) ──
    patch_checkpoints: list[PatchCheckpoint] = field(default_factory=list)

    # ── Pillar 5: Generation Handoffs (Cross-File Intelligence v4) ──
    generation_handoffs: list = field(default_factory=list)  # list[GenerationHandoff]
    evidence_conflicts: list[dict] = field(default_factory=list)  # Conflict records

    # ── Pillar 6: Cross-File Relationships (Language-Agnostic) ──
    # Aggregation layer — does NOT replace Pillars 1-5.
    # Reads from them; they do not read from the registry.
    relationship_registry: Optional[object] = None  # RelationshipRegistry

    # ── Task tracking ──
    completed_tasks: list[str] = field(default_factory=list)
    current_task_id: str = ""

    # ── Pillar 1 queries ─────────────────────────────────────────────────

    def get_verified_blueprints(self) -> list:
        """Return only blueprints that have been verified against the repo."""
        from ticket_to_code.models import BlueprintStatus
        return [
            bp for bp in self.signature_blueprints
            if bp.status in (
                BlueprintStatus.EVIDENCE_VERIFIED,
                BlueprintStatus.BUILD_VALIDATED,
            )
        ]

    def get_trusted_blueprints(self) -> list:
        """Return blueprints safe for downstream consumption.

        Only blueprints at EVIDENCE_VERIFIED or beyond are trusted.
        PLANNED/PROPOSED blueprints are never authoritative.
        """
        from ticket_to_code.models import BlueprintStatus
        return [
            bp for bp in self.signature_blueprints
            if bp.status in (
                BlueprintStatus.EVIDENCE_VERIFIED,
                BlueprintStatus.IMPLEMENTED,
                BlueprintStatus.DEPENDENCY_VALIDATED,
                BlueprintStatus.BUILD_VALIDATED,
            )
        ]

    def get_blueprints_for_task(self, task_id: str) -> tuple[list, list]:
        """Return (produces, consumes) blueprints for a specific task."""
        produces = [
            bp for bp in self.signature_blueprints
            if bp.created_by_task == task_id
        ]
        consumes = [
            bp for bp in self.signature_blueprints
            if task_id in bp.consumed_by_tasks
        ]
        return produces, consumes

    def transition_blueprint(
        self, symbol_name: str, owner_class: str, new_status, note: str = ""
    ) -> bool:
        """Transition a blueprint to a new status. Returns True if found."""
        for bp in self.signature_blueprints:
            if bp.symbol_name == symbol_name and bp.owner_class == owner_class:
                bp.status = new_status
                if note:
                    bp.verification_note = note
                return True
        return False

    # ── Pillar 2 queries ─────────────────────────────────────────────────

    def update_generated_file(self, path: str, content: str) -> None:
        """Record a generated/modified file. This is the canonical write point."""
        self.generated_files[path] = content

    def record_generated(
        self, task_id: str, file_path: str, content: str,
    ) -> None:
        """Record that a task produced generated content.

        This is the high-level method called by CodeGenerator after successful
        generation.  It:
        1. Stores the file content (canonical write).
        2. Transitions produced blueprints → IMPLEMENTED_CANDIDATE (NOT IMPLEMENTED).
           Validation must later promote to IMPLEMENTED.
        3. Extracts a compact API surface for downstream dependency context.

        Key invariant: Generated code → IMPLEMENTED_CANDIDATE, NOT IMPLEMENTED.
        Only validation can promote to IMPLEMENTED.
        """
        import re as _re

        self.update_generated_file(file_path, content)

        # Transition produced blueprints to IMPLEMENTED_CANDIDATE (not IMPLEMENTED!)
        # Validation must confirm before they become authoritative.
        from ticket_to_code.models import BlueprintStatus
        for bp in self.signature_blueprints:
            if (bp.created_by_task == task_id
                    and bp.status in (
                        BlueprintStatus.PLANNED,
                        BlueprintStatus.EVIDENCE_VERIFIED,
                    )):
                bp.status = BlueprintStatus.IMPLEMENTED_CANDIDATE

        # Extract a compact API surface (exported symbols) for downstream tasks.
        # Uses WorkspaceSymbolScanner for language-agnostic extraction — supports
        # Java, TypeScript, Python, Kotlin, etc. instead of JS-only export regex.
        try:
            from ticket_to_code.agents.workspace_symbol_scanner import WorkspaceSymbolScanner
            _scanner = WorkspaceSymbolScanner(".")
            _file_syms = _scanner.scan_content(content, file_path)

            _surface_parts = []
            if _file_syms.class_names:
                _surface_parts.append(f"classes: {', '.join(_file_syms.class_names[:5])}")

            _methods = [
                sym for sym in _file_syms.symbols
                if sym.kind == "method" and sym.access_level in ("public", None, "")
            ]
            if _methods:
                _method_sigs = []
                for m in _methods[:10]:
                    sig = m.name
                    if m.params is not None:
                        sig += f"({', '.join(m.params[:5])})"
                    if m.return_type:
                        sig += f" -> {m.return_type}"
                    _method_sigs.append(sig)
                _surface_parts.append(f"methods: {'; '.join(_method_sigs)}")

            _props = [
                sym for sym in _file_syms.symbols
                if sym.kind == "property" and sym.access_level in ("public", None, "")
            ]
            if _props:
                _surface_parts.append(
                    f"properties: {', '.join(p.name for p in _props[:10])}"
                )

            if _surface_parts:
                self.api_surfaces[file_path] = " | ".join(_surface_parts)
        except Exception as _scan_exc:
            logger.debug(
                f"  [ImplementationState] scan_content failed for {file_path}: {_scan_exc}"
            )
            # Fallback to original JS-only regex if scanner fails
            import re as _re_fb
            _exports = _re_fb.findall(
                r'export\s+(?:class|interface|enum|type|const|function)\s+(\w+)',
                content,
            )
            if _exports:
                self.api_surfaces[file_path] = (
                    f"exports: {', '.join(_exports[:10])}"
                )

        # Extract ComponentContract if TypeScript component
        if file_path.endswith((".component.ts", ".ts")):
            try:
                from ticket_to_code.intelligence.contracts import ComponentContract
                ccontract = ComponentContract.extract_from_ts(content, file_path=file_path)
                self.component_contracts[file_path] = ccontract
            except Exception as _c_err:
                logger.debug(f"ComponentContract extraction failed for {file_path}: {_c_err}")

    def get_generated_content(self, path: str) -> Optional[str]:
        """Get content of a file generated in this run. None if not generated."""
        return self.generated_files.get(path)

    def get_component_contract_for_template(self, template_path: str) -> Optional[object]:
        """Find the matching ComponentContract for an HTML template file."""
        norm_template = template_path.replace("\\", "/").lower()
        for ts_ext in (".component.ts", ".ts"):
            candidate = norm_template.replace(".component.html", ts_ext).replace(".html", ts_ext)
            for path, contract in self.component_contracts.items():
                if path.replace("\\", "/").lower() == candidate:
                    return contract
        return None

    def register_verified_symbol(self, symbol: VerifiedSymbol) -> None:
        """Register a symbol that has been confirmed to exist."""
        key = f"{symbol.owner_class}.{symbol.symbol_name}" if symbol.owner_class else symbol.symbol_name
        self.verified_symbols[key] = symbol

    def has_verified_symbol(self, symbol_name: str, owner_class: str = "") -> bool:
        """Check if a symbol has been verified in this run."""
        key = f"{owner_class}.{symbol_name}" if owner_class else symbol_name
        return key in self.verified_symbols

    def record_escalation(
        self, file_path: str, from_tier: str, to_tier: str, reason: str
    ) -> None:
        """Record a context tier escalation for telemetry."""
        self.context_escalations.append(ContextEscalation(
            file_path=file_path,
            from_tier=from_tier,
            to_tier=to_tier,
            reason=reason,
        ))

    # ── Pillar 3 queries ─────────────────────────────────────────────────

    def get_dependencies(self, file_path: str) -> list[str]:
        """Get files that a given file depends on."""
        return self.dependency_graph.get(file_path, [])

    def set_dependencies(self, file_path: str, deps: list[str]) -> None:
        """Set the dependency list for a file."""
        self.dependency_graph[file_path] = deps

    def resolve_import(self, import_path: str, from_file: str = "") -> Optional[str]:
        """Resolve an import path to a file. Returns None if unresolved."""
        return self.import_resolution.get(import_path)

    # ── Pillar 4 queries ─────────────────────────────────────────────────

    def get_diagnostics_for_file(self, path: str) -> list:
        """Get current diagnostics for a specific file."""
        norm = path.replace("\\", "/").lower()
        return [
            d for d in self.diagnostics
            if getattr(d, "source_file", "").replace("\\", "/").lower() == norm
        ]

    def record_validation(self, path: str, result: ValidationResult) -> None:
        """Record the result of incremental validation for a file."""
        self.validation_results[path] = result
        if not result.is_clean:
            # Merge diagnostics
            existing_files = {
                getattr(d, "source_file", "").replace("\\", "/").lower()
                for d in self.diagnostics
            }
            norm = path.replace("\\", "/").lower()
            # Remove old diagnostics for this file, add new ones
            self.diagnostics = [
                d for d in self.diagnostics
                if getattr(d, "source_file", "").replace("\\", "/").lower() != norm
            ]
            self.diagnostics.extend(result.diagnostics)

    def is_file_clean(self, path: str) -> bool:
        """Check if a file's last validation was clean."""
        result = self.validation_results.get(path)
        return result.is_clean if result else False

    # ── Transaction/Rollback (user req #7) ───────────────────────────────

    def create_checkpoint(self, file_path: str, description: str = "") -> None:
        """Save a pre-patch checkpoint for potential rollback."""
        content = self.generated_files.get(file_path, "")
        self.patch_checkpoints.append(PatchCheckpoint(
            file_path=file_path,
            content_before=content,
            patch_description=description,
        ))

    def rollback_last(self, file_path: str) -> Optional[str]:
        """Roll back to the most recent checkpoint for a file.

        Returns the restored content, or None if no checkpoint exists.
        """
        for i in range(len(self.patch_checkpoints) - 1, -1, -1):
            cp = self.patch_checkpoints[i]
            if cp.file_path == file_path:
                self.generated_files[file_path] = cp.content_before
                self.patch_checkpoints.pop(i)
                logger.info(
                    f"  [ImplementationState] Rolled back {Path(file_path).name} "
                    f"to checkpoint: {cp.patch_description}"
                )
                return cp.content_before
        return None

    # ── Cross-pillar: Context building ───────────────────────────────────

    def to_prompt_context(self, task) -> str:
        """Build the combined context block for a code generation task.

        Merges:
        - Blueprint context (what symbols should exist / be consumed)
        - Rolling context (API surfaces of already-generated files)
        - Dependency context (resolved imports / related files)
        - Generation handoffs (Pillar 5)

        This is the SINGLE entry point for building generation context.
        """
        from ticket_to_code.models import BlueprintStatus

        blocks: list[str] = []

        # 1. Blueprint context (Pillar 1)
        produces, consumes = self.get_blueprints_for_task(
            getattr(task, "id", self.current_task_id)
        )
        if produces or consumes:
            bp_lines = ["═══ CROSS-FILE BLUEPRINT ═══"]
            if produces:
                bp_lines.append("This task PRODUCES:")
                for bp in produces:
                    bp_lines.append(f"  + {bp.owner_class}.{bp.symbol_name}: {bp.signature}")
            if consumes:
                bp_lines.append("This task CONSUMES:")
                for bp in consumes:
                    status_icon = {
                        BlueprintStatus.VERIFIED: "✅",
                        BlueprintStatus.VALIDATED: "✅",
                        BlueprintStatus.IMPLEMENTED: "⚡",
                        BlueprintStatus.PROPOSED: "⚠️",
                        BlueprintStatus.INVALID: "❌",
                    }.get(bp.status, "?")

                    if bp.status == BlueprintStatus.INVALID and bp.actual_symbol:
                        bp_lines.append(
                            f"  {status_icon} CORRECTED: use '{bp.actual_symbol}' "
                            f"(planner proposed '{bp.symbol_name}')"
                        )
                    else:
                        bp_lines.append(
                            f"  {status_icon} {bp.owner_class}.{bp.symbol_name}: "
                            f"{bp.signature} [{bp.status.value}]"
                        )
                        if bp.import_path:
                            bp_lines.append(f"      import from: '{bp.import_path}'")
            blocks.append("\n".join(bp_lines))

        # 2. Rolling context (Pillar 2) — API surfaces of generated files
        task_file = getattr(task, "file_path", "")
        if self.generated_files:
            session_lines = ["═══ FILES GENERATED EARLIER IN THIS RUN ═══"]
            for fp, content in sorted(self.generated_files.items()):
                if fp.replace("\\", "/").lower() != task_file.replace("\\", "/").lower():
                    surface = self.api_surfaces.get(fp)
                    if surface:
                        session_lines.append(f"  ✅ {fp}")
                        session_lines.append(surface)
                    else:
                        session_lines.append(f"  ✅ {fp} (generated, no surface extracted)")
            if len(session_lines) > 1:
                blocks.append("\n".join(session_lines))

        # 3. Generation Handoffs (Pillar 5)
        task_deps = getattr(task, "dependencies", [])
        handoffs = self.get_handoffs_for_dependencies(task_deps)
        if handoffs:
            h_lines = ["═══ GENERATION HANDOFFS (AUTHORITATIVE) ═══"]
            for h in handoffs:
                val_badge = "✅" if h.validation_status == "clean" else "⚠️"
                h_lines.append(f"  {val_badge} From {h.task_id} ({h.file_path}):")
                # h.exports is List[VerifiedSymbol]
                if h.exports:
                    export_names = [
                        getattr(sym, "name", str(sym)) for sym in h.exports[:10]
                    ]
                    h_lines.append(f"    exports: {', '.join(export_names)}")
                if h.how_to_consume:
                    h_lines.append(f"    import: '{h.how_to_consume}'")
            blocks.append("\n".join(h_lines))

        # 4. Component Controller Contract for Template Generation
        if task_file.endswith((".component.html", ".html")):
            contract = self.get_component_contract_for_template(task_file)
            if contract and hasattr(contract, "render_prompt_block"):
                blocks.append(contract.render_prompt_block())

        if not blocks:
            return ""

        return "\n\n".join(blocks) + "\n"

    # ── Bridge to workflow.py state ──────────────────────────────────────

    @classmethod
    def from_workflow_state(cls, state: dict) -> "ImplementationState":
        """Create an ImplementationState from the existing workflow state dict.

        This is TRANSITIONAL infrastructure — allows incremental adoption
        without rewriting workflow.py.

        Reads from:
          - state["architectural_plan"].signature_blueprints
          - _get_transient(state, "semantic_contract")
          - _get_transient(state, "generated_files") or state.get("generated_code")
        """
        impl = cls()

        # Blueprints from the plan
        plan = state.get("architectural_plan")
        if plan and hasattr(plan, "signature_blueprints"):
            impl.signature_blueprints = list(plan.signature_blueprints)

        # Generated files from transient state (if available)
        gen_files = state.get("_generated_files_map")
        if isinstance(gen_files, dict):
            impl.generated_files = dict(gen_files)

        return impl

    def sync_to_workflow_state(self, state: dict) -> None:
        """Write ImplementationState back into the workflow state dict.

        _run_generated_map is a BRIDGE — it reflects ImplementationState,
        not the other way around. Never write to _run_generated_map independently.

        Transitional: ensures downstream nodes that read workflow state
        still see updated data.
        """
        state["_generated_files_map"] = self.generated_files

        # Bridge: sync to _run_generated_map for backward compat
        bridge = state.setdefault("_run_generated_map", {})
        for path, content in self.generated_files.items():
            key = path.replace("\\", "/").lower()
            bridge[key] = content
            bridge[path] = content
            bridge[Path(path).name.lower()] = content

    # ── Pillar 5: Generation Handoffs ──────────────────────────────────

    def add_handoff(self, handoff: "GenerationHandoff") -> None:
        """Record a verified handoff after successful generation + validation.

        Key invariant: This must only be called AFTER validation passes.
        Invalid generated code should NEVER enter the authoritative handoff chain.
        """
        self.generation_handoffs.append(handoff)
        logger.info(
            f"  [ImplementationState] Handoff recorded for task {handoff.task_id} "
            f"({len(handoff.exports)} exports, status={handoff.validation_status})"
        )

    def get_handoffs_for_dependencies(self, task_deps: list[str]) -> list:
        """Retrieve only the handoffs that this task depends on.

        Hard requirement: Do NOT dump all previous handoffs into every prompt.
        For a 30-file ticket, Task 30 should only see handoffs for its actual
        dependencies (e.g., Task 4 and Task 18), not all 29.
        """
        return [h for h in self.generation_handoffs if h.task_id in task_deps]

    def get_handoffs_for_file_deps(self, file_path: str) -> list:
        """Retrieve handoffs for files that the given file imports from."""
        dep_files = self.get_dependencies(file_path)
        # Normalize for comparison
        dep_norms = {d.replace("\\", "/").lower() for d in dep_files}
        return [
            h for h in self.generation_handoffs
            if h.file_path.replace("\\", "/").lower() in dep_norms
        ]

    def get_all_handoffs(self) -> list:
        """Return all handoffs (for debugging/summary only)."""
        return list(self.generation_handoffs)

    def record_conflict(
        self,
        source_a: str,
        claim_a: str,
        authority_a: int,
        source_b: str,
        claim_b: str,
        authority_b: int,
        resolution: str,
    ) -> None:
        """Record an evidence conflict for debugging.

        Per architectural requirement: Never silently discard evidence conflicts.
        Log them so future debugging can trace what happened.
        """
        conflict = {
            "source_a": source_a,
            "claim_a": claim_a,
            "authority_a": authority_a,
            "source_b": source_b,
            "claim_b": claim_b,
            "authority_b": authority_b,
            "resolution": resolution,
            "timestamp": time.time(),
        }
        self.evidence_conflicts.append(conflict)
        winner = source_a if authority_a <= authority_b else source_b
        logger.warning(
            f"  ⚠️ Evidence conflict: "
            f"{source_a}(rank {authority_a}): '{claim_a}' vs "
            f"{source_b}(rank {authority_b}): '{claim_b}' → {winner} wins"
        )

    # ── Summary / logging ────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of current state."""
        from ticket_to_code.models import BlueprintStatus

        bp_counts = {}
        for bp in self.signature_blueprints:
            status = bp.status.value if hasattr(bp.status, "value") else str(bp.status)
            bp_counts[status] = bp_counts.get(status, 0) + 1

        bp_str = ", ".join(f"{v} {k}" for k, v in sorted(bp_counts.items()))

        clean = sum(1 for v in self.validation_results.values() if v.is_clean)
        dirty = len(self.validation_results) - clean

        return (
            f"ImplementationState: "
            f"{len(self.generated_files)} files generated, "
            f"{len(self.verified_symbols)} symbols verified, "
            f"blueprints=[{bp_str}], "
            f"validation={clean} clean / {dirty} dirty, "
            f"{len(self.completed_tasks)} tasks done"
        )
