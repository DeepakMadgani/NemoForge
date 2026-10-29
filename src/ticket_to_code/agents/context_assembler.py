"""
Context Assembler — 8-stage pipeline for code generation context.

Replaces manual context concatenation with a structured pipeline that:
1. Labels every piece of context with its authority level
2. Resolves conflicts when sources disagree
3. Scopes context to task dependencies (no context dumping)
4. Records all conflicts for debugging

Authority Hierarchy (lower number = higher authority):
    1. Repository file system (actual file content)
    2. LSP / compiler / AST
    3. SymbolResolver evidence
    4. Validated generation (IMPLEMENTED)
    5. Unvalidated generation (IMPLEMENTED_CANDIDATE)
    6. RAG / semantic search
    7. Planner output (PLANNED)

The assembler is the SINGLE entry point for building the code generation
prompt context. All four pillars of ImplementationState contribute here.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.agents.implementation_state import ImplementationState
    from ticket_to_code.agents.dependency_context_builder import DependencyContextBuilder
    from ticket_to_code.agents.symbol_resolver import SymbolResolver
    from ticket_to_code.models import DevelopmentTask

logger = logging.getLogger(__name__)


# ── Authority levels ──────────────────────────────────────────────────────────

AUTHORITY_LEVELS = {
    "filesystem":           1,
    "lsp":                  2,
    "compiler":             2,
    "ast":                  2,
    "symbol_resolver":      3,
    "validated_generation":  4,
    "candidate_generation": 5,
    "rag":                  6,
    "planner":              7,
}


# ── Context block ─────────────────────────────────────────────────────────────

@dataclass
class ContextBlock:
    """A single labeled piece of context for prompt injection.

    Every block knows its source and authority level so conflicts
    can be resolved deterministically.
    """
    label: str               # Human-readable label (e.g. "Blueprint Dependencies")
    content: str             # The actual context text
    source: str              # Authority source key (maps to AUTHORITY_LEVELS)
    authority: int           # Computed authority rank (lower = higher authority)
    stage: str               # Which pipeline stage produced this
    file_path: str = ""      # Related file path (if applicable)

    @property
    def header(self) -> str:
        """Formatted header with authority label."""
        badge = {1: "📁", 2: "🔬", 3: "🔍", 4: "✅", 5: "⚡", 6: "📚", 7: "💭"
                 }.get(self.authority, "?")
        return f"{badge} [{self.label}] (authority: {self.source})"


# ── Context budget ────────────────────────────────────────────────────────────

@dataclass
class ContextBudget:
    """Enforces token and size limits on assembled prompt context.

    Guarantees that high-authority context (filesystem truth, compiler diagnostics,
    verified symbols) is admitted first, while low-authority or speculative text
    is bounded or shed to prevent prompt bloat and middle-loss.
    """
    max_tokens: int = 24000               # Default ceiling (~96k chars)
    max_chars_per_block: int = 16000       # Per-block ceiling for auxiliary blocks
    max_target_file_chars: int = 45000     # Dedicated generous allowance for the file being edited
    reserved_headroom_tokens: int = 4000   # Headroom reserved for model completion


# ── Conflict resolution ──────────────────────────────────────────────────────

@dataclass
class ConflictRecord:
    """Records when two sources disagree about the same thing."""
    topic: str
    source_a: str
    claim_a: str
    authority_a: int
    source_b: str
    claim_b: str
    authority_b: int
    winner: str
    resolution: str


class ConflictResolver:
    """Resolves conflicts between context sources using the authority hierarchy.

    Rule: Lower authority number wins. Conflicts are ALWAYS logged.
    Per architectural requirement: Never silently discard evidence conflicts.
    """

    def __init__(self, impl_state: Optional["ImplementationState"] = None):
        self._impl_state = impl_state
        self._conflicts: list[ConflictRecord] = []

    def resolve(
        self,
        topic: str,
        source_a: str,
        claim_a: str,
        source_b: str,
        claim_b: str,
    ) -> tuple[str, str]:
        """Resolve a conflict. Returns (winning_claim, winning_source)."""
        auth_a = AUTHORITY_LEVELS.get(source_a, 99)
        auth_b = AUTHORITY_LEVELS.get(source_b, 99)

        if auth_a <= auth_b:
            winner, loser = source_a, source_b
            winning_claim = claim_a
        else:
            winner, loser = source_b, source_a
            winning_claim = claim_b

        record = ConflictRecord(
            topic=topic,
            source_a=source_a, claim_a=claim_a, authority_a=auth_a,
            source_b=source_b, claim_b=claim_b, authority_b=auth_b,
            winner=winner,
            resolution=f"{winner} (rank {min(auth_a, auth_b)}) overrides "
                       f"{loser} (rank {max(auth_a, auth_b)})",
        )
        self._conflicts.append(record)

        # Also record in ImplementationState if available
        if self._impl_state:
            self._impl_state.record_conflict(
                source_a=source_a, claim_a=claim_a, authority_a=auth_a,
                source_b=source_b, claim_b=claim_b, authority_b=auth_b,
                resolution=record.resolution,
            )

        logger.info(
            f"  [ConflictResolver] {topic}: {winner} wins over {loser}"
        )
        return winning_claim, winner

    @property
    def conflicts(self) -> list[ConflictRecord]:
        return list(self._conflicts)

    @property
    def has_conflicts(self) -> bool:
        return len(self._conflicts) > 0


# ── Context Assembler ─────────────────────────────────────────────────────────

class ContextAssembler:
    """8-stage pipeline that builds the code generation prompt context.

    Stages:
        1. Task metadata (description, allowed methods, edit anchors)
        2. Blueprint context (Pillar 1 — semantic intent from planner)
        3. Handoff context (Pillar 5 — verified cross-file knowledge)
        4. Rolling context (Pillar 2 — API surfaces of generated files)
        5. Dependency context (Pillar 3 — resolved imports)
        6. Repository evidence (SymbolResolver / LSP)
        7. Validation feedback (Pillar 4 — diagnostics from prior attempts)
        8. Conflict resolution and final assembly

    Usage:
        assembler = ContextAssembler(impl_state=impl_state)
        prompt_context = assembler.assemble(task)
    """

    def __init__(
        self,
        impl_state: Optional["ImplementationState"] = None,
        dep_builder: Optional["DependencyContextBuilder"] = None,
        symbol_resolver: Optional["SymbolResolver"] = None,
        workspace_path: Optional[Path] = None,
    ):
        self._impl_state = impl_state
        self._dep_builder = dep_builder
        self._resolver = symbol_resolver
        self._workspace = workspace_path
        self._conflict_resolver = ConflictResolver(impl_state)

    def assemble(
        self,
        task: "DevelopmentTask",
        budget: Optional[ContextBudget] = None,
    ) -> str:
        """Run the 8-stage pipeline and return assembled context bounded by ContextBudget.

        Returns a formatted string ready for prompt injection.
        """
        budget = budget or ContextBudget()
        blocks: list[ContextBlock] = []

        # Stage 1: Task metadata
        blocks.extend(self._stage_task_metadata(task))

        # Stage 2: Blueprint context (Pillar 1)
        blocks.extend(self._stage_blueprint_context(task))

        # Stage 3: Handoff context (Pillar 5)
        blocks.extend(self._stage_handoff_context(task))

        # Stage 3.5: Cross-file relationship context (Pillar 6)
        blocks.extend(self._stage_relationship_context(task))

        # Stage 4: Rolling context (Pillar 2)
        blocks.extend(self._stage_rolling_context(task))

        # Stage 5: Dependency context (Pillar 3)
        blocks.extend(self._stage_dependency_context(task))

        # Stage 6: Repository evidence
        blocks.extend(self._stage_repo_evidence(task))

        # Stage 7: Validation feedback (Pillar 4)
        blocks.extend(self._stage_validation_feedback(task))

        # Stage 8: Conflict resolution + budget-governed final assembly
        return self._stage_final_assembly(blocks, task, budget)

    # ── Stage implementations ─────────────────────────────────────────────

    def _stage_task_metadata(self, task: "DevelopmentTask") -> list[ContextBlock]:
        """Stage 1: Extract task-level context."""
        blocks = []

        # Cross-file contract (semantic intent)
        contract = getattr(task, "cross_file_contract", None)
        if contract:
            produces = getattr(contract, "produces", [])
            consumes = getattr(contract, "consumes", [])
            if produces or consumes:
                lines = []
                if produces:
                    lines.append("This task PROVIDES:")
                    for p in produces:
                        cap = getattr(p, "capability", str(p))
                        shape = getattr(p, "data_shape", "")
                        lines.append(f"  + {cap}" + (f" ({shape})" if shape else ""))
                if consumes:
                    lines.append("This task NEEDS:")
                    for c in consumes:
                        cap = getattr(c, "capability", str(c))
                        from_task = getattr(c, "from_task", "")
                        lines.append(
                            f"  ← {cap}" + (f" (from {from_task})" if from_task else "")
                        )
                blocks.append(ContextBlock(
                    label="Cross-File Contract",
                    content="\n".join(lines),
                    source="planner",
                    authority=AUTHORITY_LEVELS["planner"],
                    stage="task_metadata",
                ))

        return blocks

    def _stage_blueprint_context(self, task: "DevelopmentTask") -> list[ContextBlock]:
        """Stage 2: Blueprint-driven context from Pillar 1."""
        if not self._impl_state:
            return []

        from ticket_to_code.models import BlueprintStatus

        task_id = getattr(task, "id", "")
        produces, consumes = self._impl_state.get_blueprints_for_task(task_id)

        blocks = []
        if produces:
            lines = ["This task PRODUCES these symbols:"]
            for bp in produces:
                lines.append(
                    f"  + {bp.owner_class}.{bp.symbol_name}: {bp.signature} "
                    f"[{bp.status.value}]"
                )
            blocks.append(ContextBlock(
                label="Blueprint Produces",
                content="\n".join(lines),
                source="planner",
                authority=AUTHORITY_LEVELS["planner"],
                stage="blueprint",
            ))

        if consumes:
            lines = ["This task CONSUMES these symbols:"]
            for bp in consumes:
                # Authority depends on blueprint status
                if bp.status in (
                    BlueprintStatus.EVIDENCE_VERIFIED,
                    BlueprintStatus.IMPLEMENTED,
                    BlueprintStatus.DEPENDENCY_VALIDATED,
                    BlueprintStatus.BUILD_VALIDATED,
                ):
                    source = "symbol_resolver"
                    label_tag = "AUTHORITATIVE"
                elif bp.status == BlueprintStatus.IMPLEMENTED_CANDIDATE:
                    source = "candidate_generation"
                    label_tag = "CANDIDATE"
                elif bp.status == BlueprintStatus.INVALID:
                    source = "symbol_resolver"
                    label_tag = "CORRECTED"
                else:
                    source = "planner"
                    label_tag = "PLANNER INTENT"

                if bp.status == BlueprintStatus.INVALID and bp.actual_symbol:
                    lines.append(
                        f"  ❌ [{label_tag}] Use '{bp.actual_symbol}' "
                        f"(planner said '{bp.symbol_name}')"
                    )
                else:
                    lines.append(
                        f"  [{label_tag}] {bp.owner_class}.{bp.symbol_name}: "
                        f"{bp.signature} [{bp.status.value}]"
                    )
                    if bp.import_path:
                        lines.append(f"      import from: '{bp.import_path}'")

            blocks.append(ContextBlock(
                label="Blueprint Consumes",
                content="\n".join(lines),
                source=source,  # Use the last consumed source as representative
                authority=AUTHORITY_LEVELS.get(source, 7),
                stage="blueprint",
            ))

        return blocks

    def _stage_handoff_context(self, task: "DevelopmentTask") -> list[ContextBlock]:
        """Stage 3: Verified generation handoffs (Pillar 5).

        Only retrieves handoffs for explicit task dependencies.
        """
        if not self._impl_state:
            return []

        task_deps = getattr(task, "dependencies", [])
        if not task_deps:
            return []

        handoffs = self._impl_state.get_handoffs_for_dependencies(task_deps)
        if not handoffs:
            return []

        blocks = []
        for h in handoffs:
            lines = [f"From task {h.task_id} → {Path(h.file_path).name}:"]

            if h.exports:
                export_names = [
                    getattr(sym, "name", str(sym)) for sym in h.exports[:15]
                ]
                lines.append(f"  [AUTHORITATIVE] exports: {', '.join(export_names)}")

            if h.how_to_consume:
                lines.append(f"  import suggestion: '{h.how_to_consume}'")

            if h.what_changed:
                lines.append(f"  summary: {h.what_changed}")

            auth_source = (
                "validated_generation"
                if h.validation_status == "clean"
                else "candidate_generation"
            )
            blocks.append(ContextBlock(
                label=f"Handoff: {h.task_id}",
                content="\n".join(lines),
                source=auth_source,
                authority=AUTHORITY_LEVELS[auth_source],
                stage="handoff",
                file_path=h.file_path,
            ))

        return blocks

    def _stage_relationship_context(self, task: "DevelopmentTask") -> list[ContextBlock]:
        """Stage 3.5: Language-agnostic cross-file relationships (Pillar 6).

        Capability-driven context resolution:
        1. What capabilities does this task consume?
        2. Which relationships provide those capabilities?
        3. For each provider, what is its verification status?
        4. What is the minimum context tier needed?

        VERIFIED facts are presented as authoritative.
        PLANNED/INFERRED are clearly labeled as suggestions.
        """
        if not self._impl_state or not self._impl_state.relationship_registry:
            return []

        registry = self._impl_state.relationship_registry

        try:
            resolved = registry.resolve_capabilities_for_task(task)
        except Exception as exc:
            logger.debug(f"Stage 3.5 relationship resolution error: {exc}")
            return []

        if not resolved:
            return []

        blocks = []
        for cap in resolved:
            rel = cap.relationship

            # Determine authority source based on relationship status
            from ticket_to_code.models import RelationshipStatus
            if rel.status in (RelationshipStatus.VERIFIED, RelationshipStatus.VALIDATED):
                source = "validated_generation"
                label = f"✅ [VERIFIED] {cap.capability}"
            elif rel.status == RelationshipStatus.DISCOVERED:
                source = "candidate_generation"
                label = f"⚡ [DISCOVERED] {cap.capability}"
            else:
                source = "planner"
                label = f"💭 [PLANNED] {cap.capability}"

            # Build context text
            ctx_text = cap.context_text or ""
            if not ctx_text and cap.provider_symbol:
                ctx_text = f"Provider: {cap.provider_symbol}"

            if ctx_text:
                blocks.append(ContextBlock(
                    label=label,
                    content=ctx_text,
                    source=source,
                    authority=AUTHORITY_LEVELS.get(source, 7),
                    stage="relationship",
                    file_path=cap.provider_file,
                ))

        if blocks:
            logger.info(
                f"  Stage 3.5: {len(blocks)} cross-file relationship(s) "
                f"injected for {Path(getattr(task, 'file_path', '')).name}"
            )

        return blocks

    def _stage_rolling_context(self, task: "DevelopmentTask") -> list[ContextBlock]:
        """Stage 4: API surfaces of already-generated files (Pillar 2)."""
        if not self._impl_state or not self._impl_state.generated_files:
            return []

        task_file = getattr(task, "file_path", "")
        blocks = []

        for fp, content in sorted(self._impl_state.generated_files.items()):
            if fp.replace("\\", "/").lower() == task_file.replace("\\", "/").lower():
                continue  # Skip self

            surface = self._impl_state.api_surfaces.get(fp)
            if surface:
                blocks.append(ContextBlock(
                    label=f"Generated: {Path(fp).name}",
                    content=surface,
                    source="candidate_generation",
                    authority=AUTHORITY_LEVELS["candidate_generation"],
                    stage="rolling_context",
                    file_path=fp,
                ))

        return blocks

    def _stage_dependency_context(self, task: "DevelopmentTask") -> list[ContextBlock]:
        """Stage 5: Resolved dependency context (Pillar 3).

        Delegates to DependencyContextBuilder if available, and compresses
        multi-line implementation bodies with SkeletonCompressor (Tiered Fidelity).
        """
        if not self._dep_builder or not self._impl_state:
            return []

        dep_context = self._dep_builder.build_context(task, self._impl_state)
        if not dep_context:
            return []

        # Tiered Fidelity: compress implementation bodies in dependency context
        try:
            from ticket_to_code.intelligence.skeleton_compressor import SkeletonCompressor
            task_path = getattr(task, "file_path", "")
            if len(dep_context) > 1200 and ("{" in dep_context and "}" in dep_context):
                if task_path.endswith((".ts", ".tsx", ".js", ".jsx")):
                    compressed = SkeletonCompressor.compress_typescript(dep_context)
                    if len(compressed) < len(dep_context):
                        dep_context = compressed
                elif task_path.endswith(".java"):
                    compressed = SkeletonCompressor.compress_java(dep_context)
                    if len(compressed) < len(dep_context):
                        dep_context = compressed
        except Exception as _comp_err:
            logger.debug(f"Skeleton compression skipped for dependency context: {_comp_err}")

        return [ContextBlock(
            label="Dependency Context",
            content=dep_context,
            source="symbol_resolver",
            authority=AUTHORITY_LEVELS["symbol_resolver"],
            stage="dependency",
        )]

    def _stage_repo_evidence(self, task: "DevelopmentTask") -> list[ContextBlock]:
        """Stage 6: Direct repository evidence (SymbolResolver/LSP).

        Queries SymbolResolver for symbols that the task needs but
        aren't covered by handoffs.
        """
        if not self._resolver or not self._impl_state:
            return []

        # Check for symbols the task consumes that don't have handoff coverage
        task_id = getattr(task, "id", "")
        _, consumes = self._impl_state.get_blueprints_for_task(task_id)

        blocks = []
        for bp in consumes:
            # Only query resolver for symbols not already covered by handoffs
            if bp.owner_class and not self._impl_state.has_verified_symbol(
                bp.symbol_name, bp.owner_class
            ):
                try:
                    defn = self._resolver.resolve_type(bp.owner_class)
                    if defn and defn.has_member(bp.symbol_name):
                        blocks.append(ContextBlock(
                            label=f"Repo Evidence: {bp.owner_class}.{bp.symbol_name}",
                            content=f"Confirmed in {defn.source}: {bp.owner_class}.{bp.symbol_name}",
                            source="symbol_resolver",
                            authority=AUTHORITY_LEVELS["symbol_resolver"],
                            stage="repo_evidence",
                        ))
                except Exception:
                    pass

        return blocks

    def _stage_validation_feedback(self, task: "DevelopmentTask") -> list[ContextBlock]:
        """Stage 7: Validation diagnostics from prior attempts (Pillar 4)."""
        if not self._impl_state:
            return []

        task_file = getattr(task, "file_path", "")
        diags = self._impl_state.get_diagnostics_for_file(task_file)
        if not diags:
            return []

        lines = ["Prior validation found these issues:"]
        for d in diags[:10]:
            msg = getattr(d, "message", str(d))
            lines.append(f"  ⚠ {msg}")

        return [ContextBlock(
            label="Validation Feedback",
            content="\n".join(lines),
            source="compiler",
            authority=AUTHORITY_LEVELS["compiler"],
            stage="validation",
            file_path=task_file,
        )]

    def _stage_final_assembly(
        self,
        blocks: list[ContextBlock],
        task: "DevelopmentTask",
        budget: Optional[ContextBudget] = None,
    ) -> str:
        """Stage 8: Resolve conflicts and assemble final prompt context with budget admission control.

        - Sorts blocks by authority (lower number = higher authority = admitted first)
        - Enforces ContextBudget ceiling so low-authority blocks are shed when budget is full
        - Deduplicates overlapping context
        - Formats with authority labels and badges
        """
        if not blocks:
            return ""

        budget = budget or ContextBudget()

        # Total-ordered sorting: Authority -> File Path -> Label
        # Guarantees byte-deterministic prefix stability for server-side KV caching across turns.
        blocks.sort(key=lambda b: (
            b.authority,
            (getattr(b, "file_path", "") or "").lower(),
            (getattr(b, "label", "") or "").lower()
        ))

        # Assemble with headers and budget admission control
        sections: list[str] = []
        accumulated_tokens = 0
        seen_fingerprints: set[str] = set()
        task_file = getattr(task, "file_path", "").replace("\\", "/").lower()

        for block in blocks:
            raw_content = (block.content or "").strip()
            if not raw_content:
                continue

            # Deduplication fingerprint
            fp = raw_content[:150]
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)

            # Check if this block is the TARGET file itself being edited
            block_fp = getattr(block, "file_path", "").replace("\\", "/").lower()
            is_target_file = bool(
                (task_file and block_fp == task_file)
                or "target file" in block.label.lower()
                or "current file" in block.label.lower()
            )

            # Cap block if it exceeds ceiling: target files get dedicated large allowance, auxiliary blocks get max_chars_per_block
            block_limit = budget.max_target_file_chars if is_target_file else budget.max_chars_per_block
            if len(raw_content) > block_limit:
                raw_content = raw_content[:block_limit] + "\n... [truncated to per-block budget]"

            block_text = f"{block.header}\n{raw_content}"
            est_tokens = max(1, len(block_text) // 4)

            # Target file is ALWAYS admitted without being shed
            if is_target_file or (accumulated_tokens + est_tokens <= budget.max_tokens):
                sections.append(block_text)
                accumulated_tokens += est_tokens
            else:
                # Remaining headroom in budget for auxiliary blocks
                remaining_tokens = budget.max_tokens - accumulated_tokens
                if remaining_tokens > 250 and block.authority <= 3:
                    # High-authority block (compiler, lsp, symbol resolver): admit a trimmed portion
                    allowed_chars = remaining_tokens * 4 - 80
                    trimmed = raw_content[:allowed_chars] + "\n... [truncated to fit context budget]"
                    sections.append(f"{block.header}\n{trimmed}")
                    accumulated_tokens += remaining_tokens
                    logger.info(
                        f"  [ContextAssembler] Truncated block '{block.label}' "
                        f"(authority: {block.source}) to fit budget headroom ({remaining_tokens} tokens)"
                    )
                else:
                    logger.info(
                        f"  [ContextAssembler] Budget cap ({budget.max_tokens} tokens) reached. "
                        f"Shedding lower-authority block '{block.label}' (authority: {block.source})"
                    )
                # Continue checking subsequent blocks in case smaller ones fit
                continue

        # Log conflict summary if any
        if self._conflict_resolver.has_conflicts:
            n = len(self._conflict_resolver.conflicts)
            logger.info(f"  [ContextAssembler] Resolved {n} evidence conflict(s)")

        assembled = "\n\n".join(sections) + "\n" if sections else ""

        logger.info(
            f"  [ContextAssembler] Assembled {len(sections)}/{len(blocks)} context blocks "
            f"(~{accumulated_tokens} tokens) for task {getattr(task, 'id', '?')}"
        )

        return assembled

    # ── Public queries ────────────────────────────────────────────────────

    @property
    def conflicts(self) -> list[ConflictRecord]:
        """Return all conflicts found during assembly."""
        return self._conflict_resolver.conflicts
