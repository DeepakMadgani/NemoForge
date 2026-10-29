"""TypeScript Type Contract.

Models hydrated type hierarchies (direct parameter types and transitive dependencies)
with explicit source provenance and array/scalar semantics to prevent hallucinated
object shapes during code generation.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TypeDefinition:
    """Individual extracted TypeScript interface, type alias, or enum."""
    name: str
    kind: str  # "interface", "type", "enum"
    source_file: str  # Normalized relative path
    raw_declaration: str  # Verbatim declaration text
    is_direct: bool = True  # True if directly in signature, False if transitive
    properties: Dict[str, str] = field(default_factory=dict)  # property_name -> type_string


@dataclass
class TypeDependencyEdge:
    """Records a parent-to-child field type reference."""
    parent_type: str
    field_name: str
    target_type: str
    is_array: bool = False

    def render(self) -> str:
        array_suffix = "[]" if self.is_array else ""
        array_note = " (ARRAY: must pass an array of filter objects)" if self.is_array else ""
        return f"{self.field_name}{array_suffix} → {self.target_type}{array_note}"


@dataclass
class MethodTypeContract:
    """Complete type contract for a called service method."""
    service_class: str
    service_file: str
    method_name: str
    method_signature: str
    parameter_name: str
    parameter_type: str
    return_type: str = ""
    direct_types: Dict[str, TypeDefinition] = field(default_factory=dict)
    transitive_types: Dict[str, TypeDefinition] = field(default_factory=dict)
    return_types: Dict[str, TypeDefinition] = field(default_factory=dict)
    dependency_edges: List[TypeDependencyEdge] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.direct_types and not self.transitive_types and not self.return_types

    def validate_code_snippet(self, code: str) -> List[str]:
        """Validates TypeScript code snippets against the hydrated type contract.

        Returns a list of violation messages if any anti-patterns or contract violations
        are detected (e.g. passing a scalar object literal where an array is required).
        """
        violations: List[str] = []
        for edge in self.dependency_edges:
            if edge.is_array:
                field = edge.field_name
                # Regex matching: field: { (a scalar object instead of array [)
                # e.g., email: { eq: searchDataElement }
                pattern = re.compile(rf"""\b{re.escape(field)}\s*:\s*\{{""", re.MULTILINE)
                if pattern.search(code):
                    violations.append(
                        f"Type Contract Violation on '{self.service_class}.{self.method_name}': "
                        f"Field '{field}' is defined as '{edge.target_type}[]' (ARRAY). "
                        f"Passing a single object '{{ ... }}' causes TS2322. "
                        f"Must be wrapped in an array: '[{{ ... }}]'."
                    )
        return violations

    def render_prompt_block(self) -> str:
        """Render authoritative, deterministic type contract for LLM prompt injection."""
        if self.is_empty():
            return ""

        lines = [
            "════════════════════════════════════════════════════════════════",
            f"AUTHORITATIVE REPOSITORY TYPE CONTRACT: {self.service_class}.{self.method_name}",
            "════════════════════════════════════════════════════════════════",
            f"Method Signature: {self.method_signature}",
            f"Service File:     {self.service_file}",
        ]

        # 1. Structural Parameter Hierarchy (Direct vs Transitive)
        if self.dependency_edges:
            lines.append("\nPARAMETER STRUCTURAL HIERARCHY:")
            lines.append(f"  {self.parameter_name}: {self.parameter_type}")
            for edge in self.dependency_edges:
                lines.append(f"    ├── {edge.render()}")

        # 2. Authoritative Type Declarations
        lines.append("\nAUTHORITATIVE TYPE DEFINITIONS:")
        # Render direct types first
        for name, defn in self.direct_types.items():
            lines.append(f"\n  [DIRECT TYPE] {name} (from {defn.source_file}):")
            for decl_line in defn.raw_declaration.strip().splitlines():
                lines.append(f"    {decl_line}")

        # Render transitive types
        for name, defn in self.transitive_types.items():
            lines.append(f"\n  [TRANSITIVE TYPE] {name} (from {defn.source_file}):")
            for decl_line in defn.raw_declaration.strip().splitlines():
                lines.append(f"    {decl_line}")

        # Render return types
        if self.return_types:
            lines.append("\nRETURN TYPES & PROPERTIES (FROM CALLED SERVICE):")
            for name, defn in self.return_types.items():
                lines.append(f"\n  [RETURN TYPE] {name} (from {defn.source_file}):")
                for decl_line in defn.raw_declaration.strip().splitlines():
                    lines.append(f"    {decl_line}")

        lines.append("\nCONTRACT ENFORCEMENT RULES:")
        lines.append("  1. Comply strictly with the field names and types in the declarations above.")
        lines.append("  2. If a parameter field is an Array (e.g. email?: FilterStringInput[]), you MUST pass an array: [ { eq: value } ].")
        lines.append("  3. If a parameter field is a Scalar Object (e.g. projectId: FilterIdInput), you MUST pass an object: { eq: value } (NOT an array).")
        lines.append("  4. For objects returned by this method (e.g. results from service call subscribe), ONLY access properties declared on the [RETURN TYPE] above (e.g., use 'companyName', NOT 'organizationName').")
        lines.append("  5. Do NOT invent new properties or guess property names.")
        lines.append("════════════════════════════════════════════════════════════════\n")

        return "\n".join(lines)

