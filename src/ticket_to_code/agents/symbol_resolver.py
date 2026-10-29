"""
Symbol Resolver — Program-understanding layer for symbol-to-definition resolution.

Answers fundamental questions about a codebase's symbol relationships:
  - "Where is type X defined?"
  - "What members does type X declare?"
  - "Does type X have member Y?"
  - "What file path corresponds to class Z?"

This is a GENERAL program-understanding primitive, not error-specific.
It is used by:
  - Fix Localization (error resolution)
  - Code Generation (symbol validation)
  - Relationship Analysis (cross-file dependency tracing)
  - Semantic Contract Validation (invariant checking)
  - Refactoring (rename/move safety)

Wraps WorkspaceSymbolIndex from lsp_client.py, which already supports:
  TypeScript / JavaScript, Java, Kotlin, Python, C#

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.agents.lsp_client import (
        WorkspaceSymbolIndex,
        ClassMembers,
        PropertyInfo,
        MethodInfo,
    )

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TypeDefinition:
    """Result of resolving a type/class name to its definition.

    Answers: "Where is this type defined, and what does it declare?"
    """
    type_name: str                              # The resolved type/class name
    file_path: str                              # Relative path to the definition file
    class_name: str                             # Actual class name (may differ from type_name
                                                # if type_name was an alias)
    property_names: list[str] = field(default_factory=list)
    method_names: list[str] = field(default_factory=list)
    source: str = "unknown"                     # How the definition was found
                                                # ("cache", "scan", "declaration", "regex")

    def has_member(self, name: str) -> bool:
        """Check if this type declares a property or method with the given name."""
        return name in self.property_names or name in self.method_names

    def missing_members(self, required: list[str]) -> list[str]:
        """Return which of the required names are NOT declared on this type."""
        all_members = set(self.property_names) | set(self.method_names)
        return [r for r in required if r not in all_members]


@dataclass
class SymbolLocation:
    """A resolved symbol — where it is defined and what kind it is."""
    symbol_name: str
    file_path: str                              # File where the symbol is defined
    kind: str                                   # "property", "method", "class", "unknown"
    owner_type: str = ""                        # The type that owns this symbol (if applicable)
    type_hint: str = ""                         # Type annotation (if available)


# ── Resolver ──────────────────────────────────────────────────────────────────

class SymbolResolver:
    """Resolves type/class names to their definitions and members.

    This is a pure program-understanding primitive:
    - It knows about SYMBOLS, not about ERRORS.
    - It wraps WorkspaceSymbolIndex for multi-language support.
    - It provides clean query methods that any layer can call.

    Usage:
        resolver = SymbolResolver(symbol_index)

        # Find where a type is defined
        defn = resolver.resolve_type("AddMembersComponent")
        # → TypeDefinition(file_path="add-members.component.ts", ...)

        # Check if a type has a specific member
        resolver.has_member("AddMembersComponent", "selectedMember")
        # → True

        # Get all members of a type
        members = resolver.get_members("AddMembersComponent")
        # → TypeDefinition(property_names=[...], method_names=[...])

        # Find which file defines a symbol
        loc = resolver.find_definition("AddMembersComponent")
        # → SymbolLocation(file_path="add-members.component.ts", kind="class")
    """

    def __init__(self, symbol_index: "WorkspaceSymbolIndex"):
        self._index = symbol_index

    # ── Core queries ──────────────────────────────────────────────────────

    def resolve_type(self, type_name: str) -> Optional[TypeDefinition]:
        """Find which file defines a type/class and what members it declares.

        Checks the symbol index cache first, then scans the workspace if needed.
        Returns None if the type cannot be resolved.
        """
        # 1. Check if we already know which file defines this type
        file_path = self._index._class_to_file.get(type_name)
        if not file_path:
            # 2. Try to find it by scanning the workspace
            found = self._index.find_class(type_name)
            if found:
                file_path = self._index._class_to_file.get(type_name)

        if not file_path:
            return None

        # 3. Get the full member list
        members = self._index.get_members_for_file(file_path)
        if not members:
            # File exists but couldn't extract members — still return the location
            return TypeDefinition(
                type_name=type_name,
                file_path=file_path,
                class_name=type_name,
                source="file_only",
            )

        return TypeDefinition(
            type_name=type_name,
            file_path=file_path,
            class_name=members.class_name,
            property_names=members.property_names(),
            method_names=members.method_names(),
            source=members.source,
        )

    def find_definition(self, symbol_name: str) -> Optional[SymbolLocation]:
        """Find the file that defines a symbol (class, type, etc.).

        Currently resolves class-level definitions. Future: method-level,
        variable-level via LSP go-to-definition.
        """
        file_path = self._index._class_to_file.get(symbol_name)
        if not file_path:
            found = self._index.find_class(symbol_name)
            if found:
                file_path = self._index._class_to_file.get(symbol_name)

        if file_path:
            return SymbolLocation(
                symbol_name=symbol_name,
                file_path=file_path,
                kind="class",
                owner_type="",
            )
        return None

    def get_members(self, type_name: str) -> Optional[TypeDefinition]:
        """Get the declared members (properties + methods) of a type.

        Alias for resolve_type() — both return TypeDefinition with members.
        Exists for semantic clarity at call sites.
        """
        return self.resolve_type(type_name)

    def has_member(self, type_name: str, member_name: str) -> Optional[bool]:
        """Check if a type declares a specific member.

        Returns:
            True if the member exists.
            False if the type is found but the member is not declared.
            None if the type itself cannot be resolved.
        """
        defn = self.resolve_type(type_name)
        if defn is None:
            return None
        return defn.has_member(member_name)

    def get_file_for_type(self, type_name: str) -> Optional[str]:
        """Get the file path that defines a type. Returns None if unresolved."""
        file_path = self._index._class_to_file.get(type_name)
        if file_path:
            return file_path
        # Try scanning
        found = self._index.find_class(type_name)
        if found:
            return self._index._class_to_file.get(type_name)
        return None

    def is_defined_in(self, type_name: str, file_path: str) -> bool:
        """Check whether a type is defined in a specific file.

        Normalizes paths for comparison (forward slashes, lowercase).
        """
        defn_file = self.get_file_for_type(type_name)
        if not defn_file:
            return False
        return _norm(defn_file) == _norm(file_path)

    # ── Bulk queries ──────────────────────────────────────────────────────

    def resolve_types(self, type_names: list[str]) -> dict[str, TypeDefinition]:
        """Resolve multiple types at once. Returns {type_name: TypeDefinition}."""
        results: dict[str, TypeDefinition] = {}
        for name in type_names:
            defn = self.resolve_type(name)
            if defn:
                results[name] = defn
        return results

    def find_missing_members(
        self, type_name: str, required_members: list[str]
    ) -> Optional[list[str]]:
        """Find which members from a required list are NOT declared on a type.

        Returns:
            List of missing member names.
            None if the type cannot be resolved.
        """
        defn = self.resolve_type(type_name)
        if defn is None:
            return None
        return defn.missing_members(required_members)

    # ── File-based queries ────────────────────────────────────────────────

    def get_members_for_file(self, file_path: str) -> Optional[TypeDefinition]:
        """Get the primary type/class defined in a specific file.

        This is the file→type direction (vs resolve_type which is type→file).
        """
        members = self._index.get_members_for_file(file_path)
        if not members:
            return None
        return TypeDefinition(
            type_name=members.class_name,
            file_path=file_path,
            class_name=members.class_name,
            property_names=members.property_names(),
            method_names=members.method_names(),
            source=members.source,
        )

    def get_raw_class_members(self, file_path: str) -> Optional["ClassMembers"]:
        """Get the raw ClassMembers object from the underlying index.

        Use this when you need the full PropertyInfo/MethodInfo objects
        (types, visibility, decorators), not just names.
        """
        return self._index.get_members_for_file(file_path)


# ── Utility ───────────────────────────────────────────────────────────────────

def _norm(path: str) -> str:
    """Normalize a file path for comparison."""
    return path.replace("\\", "/").lower()
