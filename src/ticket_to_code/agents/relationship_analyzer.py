"""
Relationship Analyzer — Program-understanding layer for cross-file relationships.

Answers fundamental questions about how files and symbols relate to each other:
  - "What files are related to file X?"
  - "What is the relationship between file A and file B?"
  - "What invariants connect these two files?"
  - "What symbols must exist in file Y because file X references them?"

This is a GENERAL program-understanding primitive, not error-specific.
It composes three existing infrastructure layers:
  - SymbolResolver     (symbol_resolver.py)  → symbol→definition→members
  - ComponentStructureProvider (component_structure_provider.py) → file grouping + event bindings
  - DataFlowTracer     (dataflow_tracer.py)  → template↔controller data flow

Used by:
  - Fix Localization (finding candidate fix locations)
  - Semantic Contract Validation (checking invariants)
  - Code Generation (understanding what must exist before generating)

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
    from ticket_to_code.agents.component_structure_provider import (
        ComponentStructureProvider,
        ComponentStructure,
    )

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FileRelationship:
    """A directed relationship between two files.

    Represents: source_file --[relationship_type]--> target_file
    """
    source_file: str              # File that references/depends on target
    target_file: str              # File being referenced/depended on
    relationship_type: str        # "template_controller", "caller_callee",
                                  # "interface_implementation", "import",
                                  # "service_consumer", "model_usage",
                                  # "barrel_export", "test_subject"
    shared_symbols: list[str] = field(default_factory=list)  # Symbols that cross this edge
    direction: str = "uses"       # "uses", "defines", "implements", "tests"


@dataclass
class Invariant:
    """A program invariant that must hold between related files/symbols.

    This is the structured representation the user requested —
    not just a string, but a verifiable relationship.

    Can be checked BEFORE a patch (invariant = BROKEN) and
    AFTER a patch (invariant = SATISFIED) to verify correctness.
    """
    invariant_type: str           # "symbol_resolution", "type_compatibility",
                                  # "interface_satisfaction", "import_resolution",
                                  # "template_binding", "method_contract"
    source_file: str              # File where the reference originates
    target_file: str              # File where the definition should exist
    symbol: str                   # The specific symbol involved
    requirement: str              # Human-readable: "symbol must resolve",
                                  # "method must exist", "type must match"
    owner_type: str = ""          # The type/class that should own the symbol
    is_satisfied: Optional[bool] = None  # None = unchecked, True/False = checked

    def __str__(self) -> str:
        status = "✓" if self.is_satisfied else ("✗" if self.is_satisfied is False else "?")
        return (
            f"[{status}] {self.invariant_type}: "
            f"{Path(self.source_file).name} → {self.symbol} "
            f"(expected in {Path(self.target_file).name})"
        )


@dataclass
class RelationshipContext:
    """Complete relationship context for a set of files.

    This is the "program understanding" that downstream layers consume:
    - Fix Localizer uses relationships to find candidate fix locations
    - Fix Hypothesis Builder uses invariants to frame evidence
    - Semantic Contract uses this to validate generation output
    """
    files: list[str]                          # Files analyzed
    relationships: list[FileRelationship] = field(default_factory=list)
    invariants: list[Invariant] = field(default_factory=list)
    symbol_owners: dict[str, str] = field(default_factory=dict)  # symbol → defining file

    def get_related_files(self, file_path: str) -> list[FileRelationship]:
        """Get all relationships involving a specific file."""
        norm = file_path.replace("\\", "/").lower()
        return [
            r for r in self.relationships
            if r.source_file.replace("\\", "/").lower() == norm
            or r.target_file.replace("\\", "/").lower() == norm
        ]

    def get_broken_invariants(self) -> list[Invariant]:
        """Get all invariants that have been checked and found broken."""
        return [i for i in self.invariants if i.is_satisfied is False]

    def get_unchecked_invariants(self) -> list[Invariant]:
        """Get invariants that haven't been verified yet."""
        return [i for i in self.invariants if i.is_satisfied is None]

    def files_that_define(self, symbol: str) -> list[str]:
        """Find which files define a given symbol."""
        return [
            r.target_file for r in self.relationships
            if symbol in r.shared_symbols and r.direction == "defines"
        ]


# ── Analyzer ──────────────────────────────────────────────────────────────────

class RelationshipAnalyzer:
    """Discovers and analyzes cross-file relationships in a codebase.

    Composes three existing infrastructure layers:
    - SymbolResolver: symbol → file resolution
    - ComponentStructureProvider: file grouping, event bindings, data gaps
    - DataFlowTracer: template ↔ controller data flow (via CSP)

    This is a program-understanding primitive, not an error handler.

    Usage:
        analyzer = RelationshipAnalyzer(
            symbol_resolver=resolver,
            component_provider=provider,
            workspace_path=Path("/workspace"),
        )

        # Discover relationships for a set of files
        ctx = analyzer.analyze_files(["component.ts", "component.html"])

        # Check which invariants are broken
        broken = ctx.get_broken_invariants()

        # Find all files related to a specific file
        related = ctx.get_related_files("component.ts")
    """

    def __init__(
        self,
        symbol_resolver: Optional["SymbolResolver"] = None,
        component_provider: Optional["ComponentStructureProvider"] = None,
        workspace_path: Optional[Path] = None,
    ):
        self._resolver = symbol_resolver
        self._provider = component_provider
        self._workspace = workspace_path

    # ── Core analysis ─────────────────────────────────────────────────────

    def analyze_files(self, file_paths: list[str]) -> RelationshipContext:
        """Discover relationships and invariants for a set of files.

        This is the main entry point. It:
        1. Uses ComponentStructureProvider to find related files
        2. Uses SymbolResolver to map symbols to definitions
        3. Builds FileRelationship objects for cross-file edges
        4. Derives Invariant objects for verifiable relationships
        """
        ctx = RelationshipContext(files=list(file_paths))

        for fp in file_paths:
            # 1. Get component structure (related files, members, bindings)
            self._analyze_component(fp, ctx)

        # 2. Discover symbol-level relationships across all files
        self._discover_symbol_relationships(ctx)

        # 3. Derive invariants from relationships
        self._derive_invariants(ctx)

        logger.info(
            f"  [RelationshipAnalyzer] {len(ctx.files)} files → "
            f"{len(ctx.relationships)} relationships, "
            f"{len(ctx.invariants)} invariants"
        )
        return ctx

    def analyze_for_diagnostic(
        self, source_file: str, symbols: list[str]
    ) -> RelationshipContext:
        """Focused analysis: given a file and symbols, find relationships.

        Lighter-weight than analyze_files() — starts from specific symbols
        rather than discovering everything about a file.

        Used by FixLocalizer when processing a specific diagnostic.
        """
        ctx = RelationshipContext(files=[source_file])

        # Resolve each symbol to its definition
        if self._resolver:
            for sym in symbols:
                defn = self._resolver.resolve_type(sym)
                if defn and defn.file_path:
                    ctx.symbol_owners[sym] = defn.file_path

                    # Add to files if not already there
                    if defn.file_path not in ctx.files:
                        ctx.files.append(defn.file_path)

                    # Create relationship
                    is_same_file = _norm(defn.file_path) == _norm(source_file)
                    if not is_same_file:
                        ctx.relationships.append(FileRelationship(
                            source_file=source_file,
                            target_file=defn.file_path,
                            relationship_type="symbol_reference",
                            shared_symbols=[sym],
                            direction="uses",
                        ))

        # Also get component structure for the source file
        self._analyze_component(source_file, ctx)

        # Derive invariants
        self._derive_invariants(ctx)

        return ctx

    def check_invariant(self, invariant: Invariant) -> bool:
        """Verify whether a specific invariant is currently satisfied.

        Uses SymbolResolver to check if the required symbol exists
        in the expected location.

        Mutates the invariant's is_satisfied field.
        """
        if not self._resolver:
            return False

        if invariant.invariant_type == "symbol_resolution":
            # Check: does the symbol exist on the owner type?
            if invariant.owner_type:
                result = self._resolver.has_member(
                    invariant.owner_type, invariant.symbol
                )
                invariant.is_satisfied = bool(result)
            else:
                # Check: can we find the symbol's definition?
                defn = self._resolver.find_definition(invariant.symbol)
                invariant.is_satisfied = defn is not None

        elif invariant.invariant_type == "template_binding":
            # Check: does the controller have the bound property/method?
            if invariant.owner_type:
                result = self._resolver.has_member(
                    invariant.owner_type, invariant.symbol
                )
                invariant.is_satisfied = bool(result)
            else:
                invariant.is_satisfied = None  # Can't check without owner

        elif invariant.invariant_type == "import_resolution":
            # Check: does the target file define the imported symbol?
            defn = self._resolver.find_definition(invariant.symbol)
            invariant.is_satisfied = defn is not None

        else:
            # Unknown invariant type — can't check
            invariant.is_satisfied = None

        return invariant.is_satisfied or False

    def check_all_invariants(self, ctx: RelationshipContext) -> list[Invariant]:
        """Check all invariants in a context. Returns the broken ones."""
        for inv in ctx.invariants:
            self.check_invariant(inv)
        return ctx.get_broken_invariants()

    # ── Internal analysis methods ─────────────────────────────────────────

    def _analyze_component(self, file_path: str, ctx: RelationshipContext):
        """Use ComponentStructureProvider to discover relationships for one file."""
        if not self._provider:
            return

        try:
            structure = self._provider.get_component_structure(file_path)
        except Exception as exc:
            logger.debug(f"  ComponentStructure failed for {file_path}: {exc}")
            return

        # Convert related files into FileRelationship objects
        for related_path, role in structure.related_files.items():
            rel_type = _role_to_relationship_type(role)
            ctx.relationships.append(FileRelationship(
                source_file=file_path,
                target_file=related_path,
                relationship_type=rel_type,
                direction=_role_to_direction(role),
            ))
            if related_path not in ctx.files:
                ctx.files.append(related_path)

        # Convert event bindings into shared symbols
        for binding in structure.event_bindings:
            # Find which relationship this binding belongs to
            for rel in ctx.relationships:
                if (rel.relationship_type == "template_controller"
                        and _norm(rel.source_file) == _norm(binding.source_file)):
                    if binding.method_name not in rel.shared_symbols:
                        rel.shared_symbols.append(binding.method_name)

        # Record the class name → file mapping
        if structure.class_name:
            ctx.symbol_owners[structure.class_name] = file_path

    def _discover_symbol_relationships(self, ctx: RelationshipContext):
        """Use SymbolResolver to find cross-file symbol dependencies."""
        if not self._resolver:
            return

        # For each file, get its class members and check where referenced types resolve
        for fp in list(ctx.files):
            defn = self._resolver.get_members_for_file(fp)
            if not defn:
                continue

            # Record class → file mapping
            ctx.symbol_owners[defn.type_name] = fp

    def _derive_invariants(self, ctx: RelationshipContext):
        """Derive verifiable invariants from discovered relationships."""

        for rel in ctx.relationships:
            if rel.relationship_type == "template_controller":
                # For each shared symbol (template binding), the controller
                # must declare that symbol
                for sym in rel.shared_symbols:
                    # Find the controller's class name
                    owner = ""
                    for cls_name, cls_file in ctx.symbol_owners.items():
                        if _norm(cls_file) == _norm(rel.target_file):
                            owner = cls_name
                            break

                    ctx.invariants.append(Invariant(
                        invariant_type="template_binding",
                        source_file=rel.source_file,
                        target_file=rel.target_file,
                        symbol=sym,
                        requirement=f"template references '{sym}' → controller must declare it",
                        owner_type=owner,
                    ))

            elif rel.relationship_type == "caller_callee":
                for sym in rel.shared_symbols:
                    ctx.invariants.append(Invariant(
                        invariant_type="symbol_resolution",
                        source_file=rel.source_file,
                        target_file=rel.target_file,
                        symbol=sym,
                        requirement=f"caller invokes '{sym}' → callee must expose it",
                    ))

            elif rel.relationship_type == "interface_implementation":
                for sym in rel.shared_symbols:
                    ctx.invariants.append(Invariant(
                        invariant_type="interface_satisfaction",
                        source_file=rel.source_file,
                        target_file=rel.target_file,
                        symbol=sym,
                        requirement=f"interface declares '{sym}' → implementation must provide it",
                    ))

            elif rel.relationship_type == "symbol_reference":
                for sym in rel.shared_symbols:
                    ctx.invariants.append(Invariant(
                        invariant_type="symbol_resolution",
                        source_file=rel.source_file,
                        target_file=rel.target_file,
                        symbol=sym,
                        requirement=f"'{sym}' must resolve to a definition",
                    ))


# ── Utility ───────────────────────────────────────────────────────────────────

def _norm(path: str) -> str:
    """Normalize path for comparison."""
    return path.replace("\\", "/").lower()


def _role_to_relationship_type(role: str) -> str:
    """Map ComponentStructureProvider roles to relationship types."""
    mapping = {
        "template": "template_controller",
        "controller": "template_controller",
        "styles": "component_styles",
        "test": "test_subject",
        "module": "module_declaration",
        "service": "service_consumer",
        "repository": "caller_callee",
        "model": "model_usage",
        "entity": "model_usage",
        "serializer": "model_usage",
        "view": "caller_callee",
        "routing": "routing_declaration",
        "form": "model_usage",
        "admin": "model_usage",
    }
    return mapping.get(role, "associated")


def _role_to_direction(role: str) -> str:
    """Map ComponentStructureProvider roles to relationship directions."""
    defines_roles = {"template", "styles", "test", "module", "routing"}
    if role in defines_roles:
        return "defines"
    return "uses"
