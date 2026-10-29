"""
ComponentStructureProvider — Language-agnostic component relationship resolver.

Given a file path, returns:
- Related files (template, styles, model, service, etc.)
- Method→event mappings (which methods are called from templates/views)
- Property→type mappings (which properties are available)
- Data flow gaps (what's referenced but never set)

Wraps existing infrastructure:
- DataFlowTracer (Angular template→controller tracing)
- LSP Client / WorkspaceSymbolIndex (type/member extraction for all languages)
- ComponentGroups (file grouping from evidence ranking)
- RelationshipProvider (cross-file edges from SQLite/Neo4j)

Supported languages: TypeScript/Angular, Java/Spring, Python/Django,
React/Vue (JSX/TSX), C#/.NET, Kotlin — anything WorkspaceSymbolIndex supports.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EventBinding:
    """A method call from a template/view to a controller/component."""
    method_name: str        # e.g. "onUserSelect"
    source_file: str        # e.g. "add-members.component.html"
    binding_kind: str       # e.g. "(click)", "@click", "onClick"
    line: int = 0


@dataclass
class ComponentStructure:
    """
    Unified representation of a component's structure across any language.

    Provides the ErrorResolutionAgent with:
    - What files form this component
    - What methods/properties exist
    - Which template events call which methods
    - Where data flow gaps exist
    """
    primary_file: str
    language: str                                  # "typescript", "java", "python", etc.
    related_files: Dict[str, str] = field(default_factory=dict)  # {path: role}
    class_name: str = ""
    properties: List[str] = field(default_factory=list)   # simplified property names
    methods: List[str] = field(default_factory=list)       # simplified method signatures
    event_bindings: List[EventBinding] = field(default_factory=list)
    data_gaps: List[str] = field(default_factory=list)     # human-readable gap descriptions
    imports: List[str] = field(default_factory=list)        # key imports

    def to_prompt_block(self) -> str:
        """Format for LLM consumption in error resolution prompts."""
        lines = [f"COMPONENT: {Path(self.primary_file).name} ({self.language})"]

        if self.class_name:
            lines.append(f"  Class: {self.class_name}")

        if self.related_files:
            lines.append("  Related files:")
            for path, role in self.related_files.items():
                lines.append(f"    {role}: {Path(path).name}")

        if self.properties:
            lines.append("  Properties:")
            for p in self.properties[:20]:
                lines.append(f"    - {p}")

        if self.methods:
            lines.append("  Methods:")
            for m in self.methods[:20]:
                lines.append(f"    - {m}")

        if self.event_bindings:
            lines.append("  Template event bindings:")
            for eb in self.event_bindings[:15]:
                lines.append(f"    {eb.binding_kind} → {eb.method_name}()")

        if self.data_gaps:
            lines.append("  ⚠ DATA FLOW GAPS (template references without backing data):")
            for gap in self.data_gaps[:10]:
                lines.append(f"    - {gap}")

        return "\n".join(lines)


# ── Provider ──────────────────────────────────────────────────────────────────

class ComponentStructureProvider:
    """
    Language-agnostic component relationship resolver.

    Given a file path, returns a ComponentStructure describing that file's
    relationships, members, event bindings, and data flow gaps.

    Reuses existing infrastructure:
    - WorkspaceSymbolIndex from lsp_client.py (member extraction)
    - DataFlowTracer from dataflow_tracer.py (Angular template analysis)
    - Component groups from evidence ranking (file grouping)
    """

    def __init__(self, workspace_path: str, symbol_index=None, component_groups: list = None):
        """
        Args:
            workspace_path: Absolute path to the workspace root
            symbol_index: WorkspaceSymbolIndex instance (from lsp_client.py)
            component_groups: Component groups from state["component_groups"]
        """
        self.workspace_path = Path(workspace_path)
        self.symbol_index = symbol_index
        self.component_groups = component_groups or []
        self._group_cache: Dict[str, Any] = {}
        self._build_group_cache()

    def _build_group_cache(self):
        """Index component groups by file path for fast lookup."""
        for group in self.component_groups:
            primary = getattr(group, "primary_file", "") or ""
            if primary:
                self._group_cache[primary.replace("\\", "/").lower()] = group
            for related in (getattr(group, "related_files", []) or []):
                rel_path = str(related) if isinstance(related, str) else getattr(related, "file_path", "")
                if rel_path:
                    self._group_cache[rel_path.replace("\\", "/").lower()] = group

    def get_component_structure(self, file_path: str) -> ComponentStructure:
        """
        Get full component structure for any supported language.

        Auto-detects language from file extension and routes to the
        appropriate analysis strategy.
        """
        ext = Path(file_path).suffix.lower()

        if ext in (".ts", ".tsx", ".html", ".scss", ".css"):
            return self._angular_structure(file_path)
        elif ext in (".java", ".kt"):
            return self._java_structure(file_path)
        elif ext in (".py",):
            return self._python_structure(file_path)
        elif ext in (".jsx",):
            return self._react_structure(file_path)
        elif ext in (".vue",):
            return self._vue_structure(file_path)
        elif ext in (".cs",):
            return self._dotnet_structure(file_path)
        else:
            return self._generic_structure(file_path)

    # ── Angular / TypeScript ──────────────────────────────────────────────────

    def _angular_structure(self, file_path: str) -> ComponentStructure:
        """
        Build structure for Angular components.

        Uses:
        - DataFlowTracer.extract_angular_bindings() for template→controller links
        - WorkspaceSymbolIndex for class members
        - Component groups for related file discovery
        """
        structure = ComponentStructure(primary_file=file_path, language="typescript/angular")

        # Find related files from component group or naming convention
        related = self._find_related_angular_files(file_path)
        structure.related_files = related

        # Get class members via LSP client
        ts_file = file_path
        if file_path.endswith((".html", ".scss", ".css")):
            # Find the sibling .ts file
            stem = re.sub(r"\.(component\.html|component\.scss|component\.css|html|scss|css)$", "", file_path)
            candidates = [f"{stem}.component.ts", f"{stem}.ts"]
            for cand in candidates:
                if (self.workspace_path / cand).exists():
                    ts_file = cand
                    break

        if self.symbol_index:
            members = self.symbol_index.get_members_for_file(ts_file)
            if members:
                structure.class_name = members.class_name
                structure.properties = [
                    f"{p.name}{'?' if p.optional else ''}: {p.type_str}"
                    + (" @Input" if p.is_input else "")
                    + (" @Output" if p.is_output else "")
                    for p in members.properties
                ]
                structure.methods = [
                    f"{m.name}({', '.join(m.params)}): {m.return_type}"
                    for m in members.methods
                ]

        # Extract template bindings using DataFlowTracer
        html_file = related.get("template") or None
        if not html_file:
            # Convention: same name with .html extension
            for role, path in related.items():
                if path.endswith(".html"):
                    html_file = path
                    break

        if html_file:
            self._extract_angular_bindings(structure, html_file, ts_file)

        return structure

    def _find_related_angular_files(self, file_path: str) -> Dict[str, str]:
        """Find related Angular files by convention or component group."""
        related: Dict[str, str] = {}

        # Check component group cache first
        norm = file_path.replace("\\", "/").lower()
        group = self._group_cache.get(norm)
        if group:
            for f in (getattr(group, "related_files", []) or []):
                f_path = str(f) if isinstance(f, str) else getattr(f, "file_path", "")
                if f_path:
                    role = self._infer_role(f_path)
                    related[f_path] = role
            return related

        # Fallback: naming convention
        stem = re.sub(r"\.(component\.(ts|html|scss|css|spec\.ts))$", "", file_path)
        if stem == file_path:
            stem = re.sub(r"\.(ts|html|scss|css)$", "", file_path)

        conventions = {
            "controller": f"{stem}.component.ts",
            "template": f"{stem}.component.html",
            "styles": f"{stem}.component.scss",
            "test": f"{stem}.component.spec.ts",
            "module": f"{stem}.module.ts",
        }

        for role, path in conventions.items():
            if (self.workspace_path / path).exists() and path != file_path:
                related[path] = role

        return related

    def _extract_angular_bindings(self, structure: ComponentStructure, html_file: str, ts_file: str):
        """Extract Angular template bindings and data flow gaps."""
        try:
            from ticket_to_code.agents.dataflow_tracer import (
                extract_angular_bindings,
                analyze_data_flow,
            )

            html_path = self.workspace_path / html_file
            if not html_path.exists():
                return

            html_content = html_path.read_text(encoding="utf-8", errors="ignore")
            bindings = extract_angular_bindings(html_content)

            # Convert bindings to EventBinding objects
            # Look for method calls in event bindings: (click)="methodName()"
            method_call_re = re.compile(r"\((\w+)\)\s*=\s*[\"'](\w+)\(")
            for line_no, line in enumerate(html_content.splitlines(), 1):
                for m in method_call_re.finditer(line):
                    structure.event_bindings.append(EventBinding(
                        method_name=m.group(2),
                        source_file=html_file,
                        binding_kind=f"({m.group(1)})",
                        line=line_no,
                    ))

            # Run data flow gap analysis
            ts_path = self.workspace_path / ts_file
            ts_content = ts_path.read_text(encoding="utf-8", errors="ignore") if ts_path.exists() else None

            report = analyze_data_flow(
                html_content=html_content,
                html_file=html_file,
                controller_content=ts_content,
                controller_file=ts_file,
                model_contents={},
            )
            structure.data_gaps = [gap.reason for gap in report.gaps]

        except Exception as exc:
            logger.debug("Angular binding extraction failed (non-fatal): %s", exc)

    # ── Java / Spring ─────────────────────────────────────────────────────────

    def _java_structure(self, file_path: str) -> ComponentStructure:
        """Build structure for Java/Spring components."""
        structure = ComponentStructure(primary_file=file_path, language="java")

        # Get class members via LSP client
        if self.symbol_index:
            members = self.symbol_index.get_members_for_file(file_path)
            if members:
                structure.class_name = members.class_name
                structure.properties = [f"{p.name}: {p.type_str}" for p in members.properties]
                structure.methods = [
                    f"{m.visibility} {m.name}({', '.join(m.params)}): {m.return_type}"
                    for m in members.methods
                ]

        # Find related Java files by convention
        related = self._find_related_java_files(file_path)
        structure.related_files = related

        return structure

    def _find_related_java_files(self, file_path: str) -> Dict[str, str]:
        """Find related Java files by naming conventions."""
        related: Dict[str, str] = {}
        stem = Path(file_path).stem

        # Controller → Service, DTO, Repository, Entity patterns
        java_patterns = {
            "Controller": [("Service", "service"), ("DTO", "model"), ("Repository", "repository"), ("Entity", "entity")],
            "Service": [("Controller", "controller"), ("Repository", "repository"), ("DTO", "model")],
            "Repository": [("Service", "service"), ("Entity", "entity")],
            "DTO": [("Controller", "controller"), ("Service", "service")],
        }

        for suffix, companions in java_patterns.items():
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                for comp_suffix, role in companions:
                    comp_name = f"{base}{comp_suffix}.java"
                    # Search in same package and nearby packages
                    parent = Path(file_path).parent
                    candidate = parent / comp_name
                    if (self.workspace_path / str(candidate)).exists():
                        related[str(candidate)] = role

        return related

    # ── Python / Django / Flask ───────────────────────────────────────────────

    def _python_structure(self, file_path: str) -> ComponentStructure:
        """Build structure for Python components."""
        structure = ComponentStructure(primary_file=file_path, language="python")

        if self.symbol_index:
            members = self.symbol_index.get_members_for_file(file_path)
            if members:
                structure.class_name = members.class_name
                structure.properties = [f"{p.name}: {p.type_str}" for p in members.properties]
                structure.methods = [
                    f"{m.name}({', '.join(m.params)}): {m.return_type}"
                    for m in members.methods
                ]

        # Python conventions: views.py ↔ models.py ↔ serializers.py ↔ urls.py
        related = self._find_related_python_files(file_path)
        structure.related_files = related

        return structure

    def _find_related_python_files(self, file_path: str) -> Dict[str, str]:
        """Find related Python files by Django/Flask conventions."""
        related: Dict[str, str] = {}
        name = Path(file_path).name
        parent = Path(file_path).parent

        django_companions = {
            "views.py": [("models.py", "model"), ("serializers.py", "serializer"),
                         ("urls.py", "routing"), ("forms.py", "form"), ("tests.py", "test")],
            "models.py": [("views.py", "view"), ("serializers.py", "serializer"),
                          ("admin.py", "admin")],
            "serializers.py": [("models.py", "model"), ("views.py", "view")],
        }

        companions = django_companions.get(name, [])
        for comp_name, role in companions:
            candidate = parent / comp_name
            if (self.workspace_path / str(candidate)).exists():
                related[str(candidate)] = role

        return related

    # ── React (JSX/TSX) ───────────────────────────────────────────────────────

    def _react_structure(self, file_path: str) -> ComponentStructure:
        """Build structure for React components."""
        structure = ComponentStructure(primary_file=file_path, language="react")

        if self.symbol_index:
            members = self.symbol_index.get_members_for_file(file_path)
            if members:
                structure.class_name = members.class_name
                structure.properties = [f"{p.name}: {p.type_str}" for p in members.properties]
                structure.methods = [f"{m.name}({', '.join(m.params)})" for m in members.methods]

        return structure

    # ── Vue ───────────────────────────────────────────────────────────────────

    def _vue_structure(self, file_path: str) -> ComponentStructure:
        """Build structure for Vue components."""
        structure = ComponentStructure(primary_file=file_path, language="vue")
        # Vue SFCs are self-contained; minimal related-file discovery needed
        return structure

    # ── C# / .NET ─────────────────────────────────────────────────────────────

    def _dotnet_structure(self, file_path: str) -> ComponentStructure:
        """Build structure for .NET/C# components."""
        structure = ComponentStructure(primary_file=file_path, language="csharp")

        if self.symbol_index:
            members = self.symbol_index.get_members_for_file(file_path)
            if members:
                structure.class_name = members.class_name
                structure.properties = [f"{p.name}: {p.type_str}" for p in members.properties]
                structure.methods = [
                    f"{m.visibility} {m.name}({', '.join(m.params)}): {m.return_type}"
                    for m in members.methods
                ]

        # C# conventions: Controller → Service, Repository, Model, ViewModel
        related = self._find_related_dotnet_files(file_path)
        structure.related_files = related

        return structure

    def _find_related_dotnet_files(self, file_path: str) -> Dict[str, str]:
        """Find related C# files by naming conventions."""
        related: Dict[str, str] = {}
        stem = Path(file_path).stem
        parent = Path(file_path).parent

        dotnet_patterns = {
            "Controller": [("Service", "service"), ("ViewModel", "viewmodel"), ("Repository", "repository")],
            "Service": [("Controller", "controller"), ("Repository", "repository"), ("Model", "model")],
            "Repository": [("Service", "service"), ("Entity", "entity")],
        }

        for suffix, companions in dotnet_patterns.items():
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                for comp_suffix, role in companions:
                    comp_name = f"{base}{comp_suffix}.cs"
                    candidate = parent / comp_name
                    if (self.workspace_path / str(candidate)).exists():
                        related[str(candidate)] = role

        return related

    # ── Generic fallback ──────────────────────────────────────────────────────

    def _generic_structure(self, file_path: str) -> ComponentStructure:
        """Fallback for unsupported languages."""
        structure = ComponentStructure(primary_file=file_path, language="generic")

        if self.symbol_index:
            members = self.symbol_index.get_members_for_file(file_path)
            if members:
                structure.class_name = members.class_name
                structure.properties = [p.name for p in members.properties]
                structure.methods = [m.name for m in members.methods]

        return structure

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_role(file_path: str) -> str:
        """Infer the role of a file from its extension/name."""
        name = Path(file_path).name.lower()
        if name.endswith(".html") or name.endswith(".htm"):
            return "template"
        if name.endswith((".scss", ".css", ".less", ".sass")):
            return "styles"
        if name.endswith(".spec.ts") or name.endswith("_test.py") or name.endswith("Test.java"):
            return "test"
        if "service" in name:
            return "service"
        if "model" in name or "dto" in name or "interface" in name:
            return "model"
        if "module" in name:
            return "module"
        if "controller" in name:
            return "controller"
        if "repository" in name or "repo" in name:
            return "repository"
        return "related"
