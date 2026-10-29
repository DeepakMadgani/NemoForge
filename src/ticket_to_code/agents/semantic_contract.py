"""
Semantic Contract — Generation-time invariant specification.

A SemanticContract is a first-class artifact shared between code generation
and error resolution. It captures the INTENT of a change (from the ticket/plan)
combined with the TRUTH of the codebase (from LSP/SymbolIndex).

Hybrid construction:
    - LLM handles INTENT: "The ticket requires adding a property 'memberOrg' to AddMembersComponent"
    - Deterministic tooling handles TRUTH: "AddMembersComponent is in add-members.component.ts,
      currently has properties [selectedMember, organizations, ...]"

The contract is verified:
    - BEFORE code generation (as a completeness check on the plan)
    - AFTER code generation (as a pre-build validation)
    - DURING error resolution (as invariant context)

This module is language-agnostic. The contract describes WHAT must be true,
not HOW to implement it.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.agents.symbol_resolver import SymbolResolver
    from ticket_to_code.agents.component_structure_provider import ComponentStructureProvider
    from ticket_to_code.agents.implementation_state import ImplementationState

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ContractItem:
    """A single verifiable requirement in a semantic contract.

    Each item describes one thing that MUST be true after code generation:
    - A symbol that must exist
    - A type relationship that must hold
    - A file that must be created/modified
    - An import that must be present
    """
    item_type: str              # "symbol_must_exist", "type_must_match",
                                # "file_must_exist", "import_must_exist",
                                # "property_must_exist", "method_must_exist",
                                # "template_binding_must_resolve"
    file: str                   # File where this requirement applies
    symbol: str                 # Symbol name involved
    owner_type: str = ""        # Class/interface that should own the symbol
    expected_type: str = ""     # Expected type signature (if applicable)
    source: str = ""            # "intent" (from LLM/plan) or "truth" (from tooling)
                                # or "blueprint" (from ImplementationState)
    rationale: str = ""         # Why this item exists
    is_satisfied: Optional[bool] = None

    def __str__(self) -> str:
        status = "✓" if self.is_satisfied else ("✗" if self.is_satisfied is False else "?")
        return (
            f"[{status}] {self.item_type}: {self.symbol} "
            f"in {Path(self.file).name}"
            + (f" on {self.owner_type}" if self.owner_type else "")
        )


@dataclass
class SemanticContract:
    """A complete semantic contract for a code generation task.

    Shared between:
    - Code Generator (must satisfy all items)
    - Pre-build Validator (checks items before compiling)
    - Error Resolution Agent (uses as invariant context)

    The contract is IMMUTABLE once created. If the plan changes,
    a new contract is generated.
    """
    ticket_id: str = ""
    description: str = ""                          # Brief summary of what's being changed
    items: list[ContractItem] = field(default_factory=list)
    files_in_scope: list[str] = field(default_factory=list)  # All files the change touches

    @property
    def intent_items(self) -> list[ContractItem]:
        """Items derived from ticket/plan intent."""
        return [i for i in self.items if i.source == "intent"]

    @property
    def truth_items(self) -> list[ContractItem]:
        """Items derived from deterministic tooling."""
        return [i for i in self.items if i.source == "truth"]

    @property
    def is_fully_satisfied(self) -> bool:
        """True only if ALL items have been checked and are satisfied."""
        return all(i.is_satisfied for i in self.items) if self.items else False

    @property
    def broken_items(self) -> list[ContractItem]:
        """Items that have been checked and found broken."""
        return [i for i in self.items if i.is_satisfied is False]

    @property
    def unchecked_items(self) -> list[ContractItem]:
        """Items that haven't been verified yet."""
        return [i for i in self.items if i.is_satisfied is None]

    def to_prompt_block(self) -> str:
        """Format for LLM consumption (code generator or error resolver)."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║  SEMANTIC CONTRACT — What must be true after this change    ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
        ]

        if self.description:
            lines.append(f"Change: {self.description}")
            lines.append("")

        # Group by file
        by_file: dict[str, list[ContractItem]] = {}
        for item in self.items:
            by_file.setdefault(item.file, []).append(item)

        for file_path, file_items in by_file.items():
            lines.append(f"── {Path(file_path).name} ──")
            for item in file_items:
                lines.append(f"  {item}")
            lines.append("")

        # Summary
        total = len(self.items)
        satisfied = sum(1 for i in self.items if i.is_satisfied)
        broken = sum(1 for i in self.items if i.is_satisfied is False)
        unchecked = sum(1 for i in self.items if i.is_satisfied is None)
        lines.append(
            f"Contract: {total} items | "
            f"✓ {satisfied} satisfied | ✗ {broken} broken | ? {unchecked} unchecked"
        )

        return "\n".join(lines)


# ── Builder ───────────────────────────────────────────────────────────────────

class SemanticContractBuilder:
    """Builds semantic contracts from plan + codebase truth.

    Hybrid approach:
    1. Intent items are extracted from planned changes (LLM output)
    2. Truth items are discovered from existing codebase (LSP/SymbolIndex)

    Usage:
        builder = SemanticContractBuilder(
            symbol_resolver=resolver,
            component_provider=provider,
        )

        # From planned changes
        contract = builder.build_from_plan(
            ticket_id="TICKET-123",
            planned_changes=[
                {"file": "component.ts", "action": "add_property",
                 "symbol": "memberOrg", "type": "Organization"},
            ],
            description="Add org dropdown to member form",
        )

        # Verify the contract
        builder.verify(contract)
    """

    def __init__(
        self,
        symbol_resolver: Optional["SymbolResolver"] = None,
        component_provider: Optional["ComponentStructureProvider"] = None,
    ):
        self._resolver = symbol_resolver
        self._provider = component_provider

    # ── Blueprint-aware contract building ─────────────────────────────────

    def build_from_blueprints(
        self,
        task_id: str,
        impl_state: "ImplementationState",
        description: str = "",
    ) -> SemanticContract:
        """Build a pre-generation contract from ImplementationState blueprints.

        This is the bridge between the blueprint lifecycle (Pillar 1) and the
        existing SemanticContract verification system. Called BEFORE code
        generation to give the LLM a concrete specification of what it must
        produce and what it can depend on.

        Args:
            task_id: The task about to be generated.
            impl_state: Current ImplementationState (holds all blueprints).
            description: Human-readable description for the contract.

        Returns:
            A SemanticContract with:
            - "blueprint" source items for symbols this task must PRODUCE
            - "blueprint" source items for symbols this task CONSUMES
              (pre-verified against already-generated content)
        """
        from ticket_to_code.models import BlueprintStatus
        import re as _re

        produces, consumes = impl_state.get_blueprints_for_task(task_id)
        if not produces and not consumes:
            return SemanticContract(
                ticket_id=task_id,
                description=description,
            )

        contract = SemanticContract(
            ticket_id=task_id,
            description=description or f"Pre-generation contract for {task_id}",
        )

        # ── Produced blueprints → "must exist after generation" ──────────
        for bp in produces:
            if bp.file_path and bp.file_path not in contract.files_in_scope:
                contract.files_in_scope.append(bp.file_path)

            # Determine item type from signature heuristics
            item_type = "symbol_must_exist"
            if bp.signature and "(" in bp.signature:
                item_type = "method_must_exist"
            elif bp.signature and ":" in bp.signature:
                item_type = "property_must_exist"

            status_label = bp.status.value if bp.status else "proposed"
            contract.items.append(ContractItem(
                item_type=item_type,
                file=bp.file_path,
                symbol=bp.symbol_name,
                owner_type=bp.owner_class,
                expected_type=bp.signature,
                source="blueprint",
                rationale=(
                    f"This task must PRODUCE this symbol "
                    f"[{status_label}]"
                    + (f" — consumed by: {', '.join(bp.consumed_by_tasks[:3])}"
                       if bp.consumed_by_tasks else "")
                ),
            ))

        # ── Consumed blueprints → pre-verify against generated content ───
        for bp in consumes:
            if bp.file_path and bp.file_path not in contract.files_in_scope:
                contract.files_in_scope.append(bp.file_path)

            # Check if the consumed symbol was already generated
            is_satisfied = None
            gen_content = impl_state.get_generated_content(bp.file_path)
            if gen_content is not None:
                # Simple check: does the symbol appear in the generated file?
                is_satisfied = bool(
                    _re.search(r'\b' + _re.escape(bp.symbol_name) + r'\b',
                               gen_content)
                )

            item_type = "symbol_must_exist"
            if bp.signature and "(" in bp.signature:
                item_type = "method_must_exist"
            elif bp.signature and ":" in bp.signature:
                item_type = "property_must_exist"

            status_label = bp.status.value if bp.status else "proposed"
            trust_note = ""
            if bp.status in (BlueprintStatus.VERIFIED, BlueprintStatus.VALIDATED):
                trust_note = " ✓ TRUSTED"
            elif bp.status == BlueprintStatus.PROPOSED:
                trust_note = " ⚠ UNVERIFIED"

            contract.items.append(ContractItem(
                item_type=item_type,
                file=bp.file_path,
                symbol=bp.symbol_name,
                owner_type=bp.owner_class,
                expected_type=bp.signature,
                source="blueprint",
                rationale=(
                    f"This task CONSUMES this symbol [{status_label}]{trust_note}"
                    + (f" — produced by: {bp.created_by_task}"
                       if bp.created_by_task else "")
                ),
                is_satisfied=is_satisfied,
            ))

        # Also enrich with truth items from codebase tooling
        self._enrich_with_truth(contract)

        _n_produced = len(produces)
        _n_consumed = len(consumes)
        _n_pre_verified = sum(
            1 for i in contract.items
            if i.source == "blueprint" and i.is_satisfied is True
        )
        logger.info(
            f"  [SemanticContract] Blueprint contract for {task_id}: "
            f"{_n_produced} produces, {_n_consumed} consumes, "
            f"{_n_pre_verified} pre-verified ✓"
        )
        return contract

    def build_from_plan(
        self,
        ticket_id: str,
        planned_changes: list[dict],
        description: str = "",
    ) -> SemanticContract:
        """Build a contract from structured planned changes.

        Each planned change is a dict with:
        - file: Target file path
        - action: "add_property", "add_method", "add_import", "create_file",
                  "modify_interface", "add_template_binding"
        - symbol: The symbol being added/modified
        - type: Expected type (optional)
        - owner: Class/interface that owns the symbol (optional)
        - rationale: Why this change is needed (optional)
        """
        contract = SemanticContract(
            ticket_id=ticket_id,
            description=description,
        )

        for change in planned_changes:
            file_path = change.get("file", "")
            action = change.get("action", "")
            symbol = change.get("symbol", "")
            type_str = change.get("type", "")
            owner = change.get("owner", "")
            rationale = change.get("rationale", "")

            if not file_path or not symbol:
                continue

            # Add to files in scope
            if file_path not in contract.files_in_scope:
                contract.files_in_scope.append(file_path)

            # Create intent item
            item_type = _action_to_item_type(action)
            contract.items.append(ContractItem(
                item_type=item_type,
                file=file_path,
                symbol=symbol,
                owner_type=owner,
                expected_type=type_str,
                source="intent",
                rationale=rationale or f"Required by {ticket_id}",
            ))

        # Enrich with truth items from codebase
        self._enrich_with_truth(contract)

        logger.info(
            f"  [SemanticContract] Built contract: {len(contract.items)} items "
            f"({len(contract.intent_items)} intent, {len(contract.truth_items)} truth)"
        )
        return contract

    def build_from_file_changes(
        self,
        ticket_id: str,
        changed_files: list[str],
        description: str = "",
    ) -> SemanticContract:
        """Build a contract from a list of changed files.

        Lighter-weight than build_from_plan — discovers what MUST be true
        based on the existing relationships of the changed files.

        Used when planned changes aren't available (e.g., error resolution).
        """
        contract = SemanticContract(
            ticket_id=ticket_id,
            description=description,
            files_in_scope=list(changed_files),
        )

        for fp in changed_files:
            self._discover_file_contracts(fp, contract)

        logger.info(
            f"  [SemanticContract] Built from files: {len(contract.items)} items"
        )
        return contract

    def verify(self, contract: SemanticContract) -> list[ContractItem]:
        """Verify all items in a contract. Returns broken items.

        Uses SymbolResolver to check each item against the current codebase state.
        """
        if not self._resolver:
            return []

        for item in contract.items:
            self._verify_item(item)

        broken = contract.broken_items
        if broken:
            logger.info(
                f"  [SemanticContract] Verification: "
                f"{len(broken)}/{len(contract.items)} items BROKEN"
            )
        else:
            logger.info(
                f"  [SemanticContract] Verification: "
                f"all {len(contract.items)} items satisfied ✓"
            )
        return broken

    # ── Internal ──────────────────────────────────────────────────────────

    def _enrich_with_truth(self, contract: SemanticContract):
        """Add truth items by analyzing existing codebase relationships."""
        if not self._provider:
            return

        for fp in list(contract.files_in_scope):
            try:
                structure = self._provider.get_component_structure(fp)
            except Exception:
                continue

            # For each related file, ensure the relationship is preserved
            for related_path, role in structure.related_files.items():
                if related_path not in contract.files_in_scope:
                    contract.files_in_scope.append(related_path)

            # For event bindings, ensure controller methods exist
            for binding in structure.event_bindings:
                contract.items.append(ContractItem(
                    item_type="method_must_exist",
                    file=fp,
                    symbol=binding.method_name,
                    owner_type=structure.class_name,
                    source="truth",
                    rationale=f"Template binding ({binding.binding_kind}) requires this method",
                ))

            # For data gaps, flag them as contract violations
            for gap in structure.data_gaps:
                contract.items.append(ContractItem(
                    item_type="template_binding_must_resolve",
                    file=fp,
                    symbol=gap,
                    owner_type=structure.class_name,
                    source="truth",
                    rationale="Data flow gap detected by template analysis",
                    is_satisfied=False,  # Already known to be broken
                ))

    def _discover_file_contracts(self, file_path: str, contract: SemanticContract):
        """Discover contract items for a single changed file."""
        if not self._resolver:
            return

        # Get the file's class members
        members = self._resolver.get_members_for_file(file_path)
        if not members:
            return

        # Each existing public member is an implicit contract:
        # "this symbol must continue to exist for consumers"
        for method_name in members.method_names[:20]:
            contract.items.append(ContractItem(
                item_type="method_must_exist",
                file=file_path,
                symbol=method_name,
                owner_type=members.type_name,
                source="truth",
                rationale="Existing public method — consumers depend on it",
            ))

    def _verify_item(self, item: ContractItem):
        """Verify a single contract item against the codebase."""
        if not self._resolver:
            return

        if item.item_type in ("property_must_exist", "method_must_exist"):
            if item.owner_type:
                result = self._resolver.has_member(item.owner_type, item.symbol)
                item.is_satisfied = bool(result)
            else:
                defn = self._resolver.find_definition(item.symbol)
                item.is_satisfied = defn is not None

        elif item.item_type == "symbol_must_exist":
            defn = self._resolver.resolve_type(item.symbol)
            item.is_satisfied = defn is not None

        elif item.item_type == "import_must_exist":
            # Can't fully verify imports without parsing — mark as unchecked
            item.is_satisfied = None

        elif item.item_type == "file_must_exist":
            item.is_satisfied = Path(item.file).exists()

        elif item.item_type == "template_binding_must_resolve":
            if item.owner_type:
                result = self._resolver.has_member(item.owner_type, item.symbol)
                item.is_satisfied = bool(result)


# ── Utility ───────────────────────────────────────────────────────────────────

def _action_to_item_type(action: str) -> str:
    """Map planned change actions to contract item types."""
    mapping = {
        "add_property": "property_must_exist",
        "add_method": "method_must_exist",
        "add_import": "import_must_exist",
        "create_file": "file_must_exist",
        "modify_interface": "symbol_must_exist",
        "add_template_binding": "template_binding_must_resolve",
    }
    return mapping.get(action, "symbol_must_exist")
