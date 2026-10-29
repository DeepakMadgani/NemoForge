"""
DataFlow Tracer — mirrors the "read → think → write → verify" loop used by Claude Code / Devin.

Instead of relying on keyword-similarity alone, this module traces the FULL data path
from the backend through the Angular controller to the HTML template, identifying gaps
where the plan writes a template binding with no corresponding code to supply the data.

Key concepts
------------
* TemplateBinding   — one Angular expression found in an .html file
* DataFlowGap       — a binding whose property is never SET in the controller
* DataFlowReport    — full result of tracing one (html, ts) file pair

The node `dataflow_verification_node` runs AFTER `ownership_completeness` but BEFORE
`rag_code` so that any missing TypeScript tasks are injected into the plan in time for
the code generator to process them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TemplateBinding:
    """One Angular expression extracted from an HTML template."""
    expression: str          # raw expression, e.g. "dmember.isExistingMemberInProject"
    variable: str            # left-hand side, e.g. "dmember"
    property: str            # right-hand side, e.g. "isExistingMemberInProject"
    kind: str                # "*ngIf" | "interpolation" | "[attr]" | "structural"
    line: int = 0


@dataclass
class DataFlowGap:
    """A template binding whose data is not supplied by the controller."""
    binding: TemplateBinding
    reason: str              # human-readable explanation for the LLM
    controller_file: str = ""
    model_file: str = ""


@dataclass
class DataFlowReport:
    """Result of tracing one (html_file, ts_controller_file) pair."""
    html_file: str
    controller_file: str
    bindings: list[TemplateBinding] = field(default_factory=list)
    gaps: list[DataFlowGap] = field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        return bool(self.gaps)


# ── Regex patterns ────────────────────────────────────────────────────────────

# Matches {{ expr }} interpolations
_INTERP_RE = re.compile(r"\{\{\s*([^}|]+?)(?:\s*\|[^}]*)?\s*\}\}")

# Matches *ngIf="expr", [disabled]="expr", (click)="expr", [ngClass]="expr", etc.
_BINDING_RE = re.compile(r"""[\[*(][\w.@#$?!-]*[\])*"']?\s*=\s*["']([^"']+)["']""")

# Matches structural directives: *ngFor="let x of items", *ngIf="x.prop"
_STRUCTURAL_RE = re.compile(r'\*ng(?:If|For|Switch|Unless)\s*=\s*"([^"]+)"')

# Property access: variable.property  (filters out plain identifiers and deep chains)
_PROP_ACCESS_RE = re.compile(r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\.(is[A-Z]\w*|existing\w*|has[A-Z]\w*|[\w]+Name|[\w]+Id|[\w]+Flag|[\w]+Org\w*)\b')

# Setter patterns in TypeScript: variable.prop = ...,  this.xxx.prop = ...,
# Object literal: { prop: ... }
_SETTER_RE = re.compile(
    r'(?:'
    r'\b(\w+)\.([\w]+)\s*='   # direct assignment: obj.prop =
    r'|'
    r"['\"](\w+)['\"]\s*:"   # object literal key: "prop":
    r'|'
    r'\b(\w+)\s*:\s*\w'      # shorthand or named: prop: value
    r')'
)

# TypeScript interface property declaration: propName?: type  or  propName: type
_INTERFACE_PROP_RE = re.compile(r'^\s+(\w+)\??\s*:', re.MULTILINE)

# Method body start: public/private/async methodName(
_METHOD_START_RE = re.compile(r'(?:public|private|protected|async)?\s*(\w+)\s*\(')


# ── Core extraction functions ─────────────────────────────────────────────────

def extract_angular_bindings(html_content: str) -> list[TemplateBinding]:
    """
    Extract all Angular template bindings that access object properties.
    Only returns bindings of the form `variable.property` since those are
    the ones that require data to be populated by the controller.
    """
    bindings: list[TemplateBinding] = []
    seen: set[str] = set()
    lines = html_content.splitlines()

    def _add_binding(expr: str, kind: str, line_no: int) -> None:
        for m in _PROP_ACCESS_RE.finditer(expr):
            variable = m.group(1)
            prop = m.group(2)
            key = f"{variable}.{prop}"
            # Skip common Angular directives and built-ins
            if variable.lower() in ("this", "event", "item", "index", "ng", "true", "false", "null"):
                continue
            if key not in seen:
                seen.add(key)
                bindings.append(TemplateBinding(
                    expression=expr.strip(),
                    variable=variable,
                    property=prop,
                    kind=kind,
                    line=line_no,
                ))

    for line_no, line in enumerate(lines, 1):
        for m in _INTERP_RE.finditer(line):
            _add_binding(m.group(1), "interpolation", line_no)
        for m in _STRUCTURAL_RE.finditer(line):
            _add_binding(m.group(1), "*ngIf/*ngFor", line_no)
        for m in _BINDING_RE.finditer(line):
            _add_binding(m.group(1), "[binding]", line_no)

    return bindings


def extract_property_setters(ts_content: str) -> set[str]:
    """
    Extract all property names that are SET anywhere in the TypeScript file.
    Includes: direct assignments (obj.prop = x), object literal keys ({ prop: x }),
    and interface/class property declarations.
    Returns a set of lowercase property names.
    """
    setters: set[str] = set()
    for m in _SETTER_RE.finditer(ts_content):
        name = m.group(2) or m.group(3) or m.group(4)
        if name:
            setters.add(name.lower())
    # Also collect interface property declarations
    for m in _INTERFACE_PROP_RE.finditer(ts_content):
        setters.add(m.group(1).lower())
    return setters


def extract_interface_properties(ts_content: str, interface_name: str) -> set[str]:
    """
    Extract declared properties of a named TypeScript interface/class.
    Returns lowercase property names.
    """
    props: set[str] = set()
    # Find the interface block
    pattern = re.compile(
        rf'(?:interface|class)\s+{re.escape(interface_name)}\s*(?:extends\s+\w+\s*)?{{([^}}]+?)}}',
        re.DOTALL,
    )
    m = pattern.search(ts_content)
    if not m:
        return props
    body = m.group(1)
    for pm in _INTERFACE_PROP_RE.finditer(body):
        props.add(pm.group(1).lower())
    return props


def infer_variable_type(ts_content: str, variable_name: str) -> Optional[str]:
    """
    Infer the TypeScript type of a variable used in templates.
    Looks for: `variable: TypeName[]`, `variable: TypeName`, `let variable: TypeName`.
    Returns the type name string, or None if not found.
    """
    patterns = [
        rf'\b{re.escape(variable_name)}\s*:\s*([A-Z][A-Za-z0-9]+)',
        rf'\b{re.escape(variable_name)}s?\s*:\s*([A-Z][A-Za-z0-9]+)\[\]',
    ]
    for pat in patterns:
        m = re.search(pat, ts_content)
        if m:
            return m.group(1)
    return None


# ── Gap analysis ──────────────────────────────────────────────────────────────

def analyze_data_flow(
    html_content: str,
    html_file: str,
    controller_content: Optional[str],
    controller_file: str,
    model_contents: dict[str, str],  # {file_path: content}
    planned_new_properties: set[str] = frozenset(),  # props being added in THIS run
) -> DataFlowReport:
    """
    The core data-flow analysis.

    For every Angular binding in the HTML that reads `variable.property`:
    1. Check if `property` is SET anywhere in the controller (direct assignment or object literal).
    2. Check if `property` is declared in any known model/interface file.
    3. If neither → DataFlowGap.

    `planned_new_properties` contains lowercase property names that other tasks in this
    run plan to ADD, so we don't flag them as gaps even if not yet on disk.
    """
    report = DataFlowReport(html_file=html_file, controller_file=controller_file)

    bindings = extract_angular_bindings(html_content)
    report.bindings = bindings

    if not bindings:
        return report

    # Setters from controller
    controller_setters: set[str] = set()
    if controller_content:
        controller_setters = extract_property_setters(controller_content)

    # Properties declared across all model files
    model_props: set[str] = set()
    for content in model_contents.values():
        model_props.update(extract_property_setters(content))

    for binding in bindings:
        prop_lower = binding.property.lower()

        in_controller = prop_lower in controller_setters
        in_model = prop_lower in model_props
        in_plan = prop_lower in planned_new_properties

        if in_controller or in_model or in_plan:
            continue  # data flow is intact

        gap = DataFlowGap(
            binding=binding,
            reason=(
                f"Template binding `{binding.variable}.{binding.property}` "
                f"(line {binding.line}, kind={binding.kind}) is never SET in the "
                f"controller `{controller_file}` and is not declared in any model. "
                f"The feature will silently show undefined/null at runtime."
            ),
            controller_file=controller_file,
        )
        report.gaps.append(gap)
        logger.warning(f"  [DataFlow GAP] {gap.reason}")

    return report


# ── Plan-level check ──────────────────────────────────────────────────────────

def trace_plan_data_flow(
    plan_tasks: list,
    workspace_path: str,
    session_map: dict[str, str],
) -> list[DataFlowReport]:
    """
    Run data-flow analysis across ALL HTML tasks in the plan.

    For each HTML task:
    - Read the current HTML file from disk (or session map if already generated)
    - Find the sibling .ts controller (from disk or session map)
    - Find all model files referenced by the controller
    - Run `analyze_data_flow` and collect gaps

    Returns a list of DataFlowReport objects (one per HTML task with bindings).
    """
    ws = Path(workspace_path)
    reports: list[DataFlowReport] = []

    # Build index of all planned tasks by normalized file path
    tasks_by_path: dict[str, object] = {}
    for t in plan_tasks:
        tasks_by_path[t.file_path.replace("\\", "/").lower()] = t

    # Collect properties that WILL be added in this run across all model tasks
    planned_new_properties: set[str] = set()
    for t in plan_tasks:
        fp_lower = t.file_path.replace("\\", "/").lower()
        is_model = any(kw in fp_lower for kw in ("model", "interface", "dto", "type", "displayed", "contract", "member"))
        if is_model:
            content = session_map.get(fp_lower) or _read_disk(ws / t.file_path)
            if content:
                planned_new_properties.update(extract_property_setters(content))
        # Also collect from task descriptions
        desc = getattr(t, "description", "") or ""
        for m in re.finditer(r'\b(is[A-Z]\w+|existing\w+|has[A-Z]\w+|\w+Name|\w+Id|\w+Flag)', desc):
            planned_new_properties.add(m.group(1).lower())

    for task in plan_tasks:
        fp = task.file_path.replace("\\", "/")
        if not fp.endswith((".html", ".htm")):
            continue

        fp_lower = fp.lower()
        html_content = session_map.get(fp_lower) or _read_disk(ws / fp)
        if not html_content:
            continue

        # Find companion controller
        stem = fp.rsplit(".", 1)[0]  # strip .html
        controller_candidates = [f"{stem}.ts", f"{stem}.tsx"]
        controller_file = ""
        controller_content: Optional[str] = None
        for ctrl in controller_candidates:
            ctrl_lower = ctrl.lower()
            ctrl_content = session_map.get(ctrl_lower) or _read_disk(ws / ctrl)
            if ctrl_content:
                controller_file = ctrl
                controller_content = ctrl_content
                break

        if not controller_file:
            logger.info(f"  [DataFlow] No controller found for {fp} — skipping trace")
            continue

        # Collect model files imported by the controller
        model_contents: dict[str, str] = {}
        if controller_content:
            import_re = re.compile(r"import\s*\{[^}]+\}\s*from\s*['\"](\.[^'\"]+)['\"]")
            ctrl_dir = Path(controller_file).parent
            for m in import_re.finditer(controller_content):
                rel = m.group(1)
                for ext in (".ts", ".tsx"):
                    candidate = str(ctrl_dir / (rel + ext)).replace("\\", "/")
                    c_lower = candidate.lstrip("/").lower()
                    content = session_map.get(c_lower) or _read_disk(ws / candidate.lstrip("/"))
                    if content:
                        model_contents[candidate] = content
                        break

        report = analyze_data_flow(
            html_content=html_content,
            html_file=fp,
            controller_content=controller_content,
            controller_file=controller_file,
            model_contents=model_contents,
            planned_new_properties=planned_new_properties,
        )

        if report.bindings:
            reports.append(report)

    return reports


# ── Helper ────────────────────────────────────────────────────────────────────

def _read_disk(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else None
    except Exception:
        return None


def gap_summary(reports: list[DataFlowReport]) -> str:
    """Return a concise human-readable summary of all data-flow gaps."""
    lines = []
    for r in reports:
        if r.has_gaps:
            lines.append(f"HTML: {r.html_file} | Controller: {r.controller_file}")
            for g in r.gaps:
                lines.append(f"  GAP: {g.binding.variable}.{g.binding.property} "
                             f"(line {g.binding.line}) — never set in controller")
    return "\n".join(lines) if lines else "No data-flow gaps found."
