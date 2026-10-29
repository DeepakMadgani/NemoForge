"""Angular Component Controller to Template Contract.

Extracts template-facing public properties and methods from generated TypeScript
component controllers, injects explicit contracts into template generation prompts,
and deterministically validates that HTML bindings do not hallucinate phantom fields.
Follows the pattern: Validate -> Targeted Model Repair -> Revalidate.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple


# Regex to match class definition in TypeScript
_CLASS_RE = re.compile(
    r"""(?:export\s+)?class\s+([A-Za-z0-9_$]+)""",
    re.MULTILINE,
)

# Regex to match class field/property declarations:
# e.g.: isExistingMember: boolean = false;
#       existingOrganizationName: string = '';
#       userExistError: boolean;
_FIELD_RE = re.compile(
    r"""^\s*(?:public\s+)?(?:readonly\s+)?([A-Za-z0-9_$]+)\s*(?::\s*([^=;,\n]+))?(?:\s*=\s*([^;,\n]+))?;""",
    re.MULTILINE,
)

# Regex to match component methods:
# e.g.: onUserSelect(searchData) { ... }
_METHOD_RE = re.compile(
    r"""^\s*(?:public\s+)?([A-Za-z0-9_$]+)\s*\(([^)]*)\)\s*(?::\s*([^{]+))?\s*\{""",
    re.MULTILINE,
)

# Regex for HTML template bindings:
_NG_IF_RE = re.compile(r"""\*ngIf\s*=\s*["']([^"']+)["']""")
_INTERPOLATION_RE = re.compile(r"""\{\{\s*([^}]+)\s*\}\}""")
_BINDING_RE = re.compile(r"""\[(?:[a-zA-Z0-9_.-]+)\]\s*=\s*["']([^"']+)["']""")
_EVENT_RE = re.compile(r"""\((?:[a-zA-Z0-9_.-]+)\)\s*=\s*["']([^"']+)["']""")
_TWO_WAY_RE = re.compile(r"""\[\((?:[a-zA-Z0-9_.-]+)\)\]\s*=\s*["']([^"']+)["']""")


@dataclass
class ComponentContract:
    """Public template-facing contract of an Angular component controller."""
    component_file: str
    class_name: str
    properties: Dict[str, str] = field(default_factory=dict)  # name -> type
    methods: Dict[str, str] = field(default_factory=dict)      # name -> signature

    @classmethod
    def extract_from_ts(cls, ts_content: str, file_path: str = "") -> "ComponentContract":
        """Extract public template-facing properties and methods from a component's TS code."""
        cm = _CLASS_RE.search(ts_content)
        class_name = cm.group(1) if cm else Path(file_path).stem

        properties: Dict[str, str] = {}
        # Parse standard single-line fields
        for fm in _FIELD_RE.finditer(ts_content):
            fname = fm.group(1)
            if fname.startswith("_") or fname in ("constructor", "ngOnInit", "ngOnDestroy", "ngAfterViewInit"):
                continue
            ftype = (fm.group(2) or "any").strip()
            properties[fname] = ftype

        # Parse multiline or initialized properties (e.g. newUser: ItemSelectItem = { ... })
        prop_line_pattern = re.compile(
            r'^\s{2}(?:public\s+|private\s+|protected\s+)?(?:readonly\s+)?([A-Za-z0-9_$]+)\s*(?:[?!]?\s*:\s*([A-Za-z0-9_$<>\[\]|]+))?\s*(?:=|;)',
            re.MULTILINE,
        )
        for pm in prop_line_pattern.finditer(ts_content):
            pname = pm.group(1)
            if pname.startswith("_") or pname in ("constructor", "ngOnInit", "ngOnDestroy", "ngAfterViewInit", "if", "for", "while", "return"):
                continue
            if pname not in properties:
                ptype = (pm.group(2) or "any").strip()
                properties[pname] = ptype

        # Parse methods
        methods: Dict[str, str] = {}
        for mm in _METHOD_RE.finditer(ts_content):
            mname = mm.group(1)
            if mname.startswith("_") or mname in ("constructor", "ngOnInit", "ngOnDestroy", "ngAfterViewInit"):
                continue
            mparams = mm.group(2).strip()
            mret = (mm.group(3) or "void").strip()
            methods[mname] = f"({mparams}): {mret}"

        return cls(
            component_file=file_path,
            class_name=class_name,
            properties=properties,
            methods=methods,
        )

    def render_prompt_block(self) -> str:
        """Render a strict contract block to inject into the template generator prompt."""
        props_lines = [f"  • {name} ({typ})" for name, typ in sorted(self.properties.items())]
        methods_lines = [f"  • {name}{sig}" for name, sig in sorted(self.methods.items())]

        return (
            "════════════════════════════════════════════════════════════════\n"
            f"CONTROLLER CONTRACT FOR TEMPLATE: {self.class_name}\n"
            f"Source: {self.component_file}\n"
            "════════════════════════════════════════════════════════════════\n"
            "The component controller exposes ONLY the following fields for template binding:\n\n"
            "AVAILABLE CONTROLLER PROPERTIES:\n"
            + ("\n".join(props_lines) if props_lines else "  (none declared)")
            + "\n\nAVAILABLE CONTROLLER METHODS:\n"
            + ("\n".join(methods_lines) if methods_lines else "  (none declared)")
            + "\n\nTEMPLATE BINDING INVARIANTS:\n"
            "  1. Bind ONLY to the controller properties and methods listed above.\n"
            "  2. Do NOT invent phantom sub-properties or wrapper flags (e.g. do NOT nest controller\n"
            "     properties under arbitrary helper objects).\n"
            "  3. Bind directly and exclusively to the exact property and method names declared on the controller above.\n"
            "════════════════════════════════════════════════════════════════\n"
        )


class TemplateContractValidator:
    """Validates an Angular HTML template against a ComponentContract."""

    ANGULAR_BUILTINS = {
        "true", "false", "null", "undefined", "let", "of", "as", "in", "index",
        "first", "last", "even", "odd", "this", "else", "then", "trackBy",
        "$event", "event",
        "Math", "Number", "String", "Boolean", "Date", "Array", "Object", "window",
        "$index", "$first", "$last", "$even", "$odd", "$count",
    }

    @classmethod
    def extract_template_locals(cls, html_content: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """Extract template reference variables, loop variables, 'as' aliases, and pipes."""
        template_refs = set(re.findall(r'#([A-Za-z_$][A-Za-z0-9_$]*)', html_content))
        # Loop variables & local declarations: *ngFor="let item of items; let i = index", @let x = 10
        local_vars = set(re.findall(r'\blet\s+([A-Za-z_$][A-Za-z0-9_$]*)', html_content))
        # 'as' aliases: *ngIf="user$ | async as user" or @if (cond; as user)
        as_aliases = set(re.findall(r'\bas\s+([A-Za-z_$][A-Za-z0-9_$]*)', html_content))
        # Angular 17+ control flow @for: @for (item of items; track item.id)
        control_flow_vars = set(re.findall(r'@for\s*\(\s*([A-Za-z_$][A-Za-z0-9_$]*)\s+of\b', html_content))

        pipe_names = set(re.findall(r'\|\s*([A-Za-z_$][A-Za-z0-9_$]*)', html_content))
        all_locals = local_vars | as_aliases | control_flow_vars
        return template_refs, all_locals, pipe_names

    @classmethod
    def extract_bound_identifiers(cls, html_content: str) -> Set[str]:
        """Extract all identifier tokens referenced in template expressions."""
        identifiers: Set[str] = set()

        expressions = []
        for m in _NG_IF_RE.finditer(html_content):
            expressions.append(m.group(1))
        for m in _INTERPOLATION_RE.finditer(html_content):
            expressions.append(m.group(1))
        for m in _BINDING_RE.finditer(html_content):
            expressions.append(m.group(1))
        for m in _TWO_WAY_RE.finditer(html_content):
            expressions.append(m.group(1))
        for m in _EVENT_RE.finditer(html_content):
            expressions.append(m.group(1))

        template_refs, loop_vars, pipe_names = cls.extract_template_locals(html_content)
        all_locals = template_refs | loop_vars | pipe_names | cls.ANGULAR_BUILTINS

        for expr in expressions:
            expr_no_str = re.sub(r"'[^']*'|\"[^\"]*\"", "", expr)
            expr_no_pipes = re.sub(r'\|.*$', '', expr_no_str)
            for mm in re.finditer(r'\b([A-Za-z_$][A-Za-z0-9_$]*)((?:\.[A-Za-z_$][A-Za-z0-9_$]*)*)\b', expr_no_pipes):
                top_id = mm.group(1)
                chain = mm.group(2)
                if top_id in all_locals:
                    continue
                identifiers.add(top_id)
                if chain:
                    first_sub = chain.lstrip(".").split(".", 1)[0]
                    identifiers.add(f"{top_id}.{first_sub}")

        return identifiers

    COMMON_FIELD_BUILTINS: Set[str] = {
        "length", "options", "placeholder", "value", "disabled", "valid", "invalid",
        "dirty", "touched", "pristine", "errors", "controls", "name", "id", "label",
        "items", "size", "roles", "status", "data", "title", "type", "key", "text",
        "description", "selected", "checked", "count", "width", "height", "url",
    }

    @classmethod
    def validate(
        cls,
        html_content: str,
        contract: ComponentContract,
        original_content: Optional[str] = None,
    ) -> List[str]:
        """Validate template expressions against the controller contract.

        Args:
            html_content: Generated or modified HTML content.
            contract: ComponentContract from the companion TypeScript controller.
            original_content: Optional baseline HTML content before edits. If provided,
                pre-existing bindings in the repository are granted baseline immunity
                and will never be flagged as violations.

        Returns:
            List of violation error messages (empty if completely valid).
        """
        import difflib
        violations: List[str] = []
        bound_ids = cls.extract_bound_identifiers(html_content)

        # Baseline immunity: identifiers that already existed in the original file are preserved
        if original_content:
            original_ids = cls.extract_bound_identifiers(original_content)
            bound_ids = bound_ids - original_ids

        known_props = set(contract.properties.keys())
        known_methods = set(contract.methods.keys())
        all_known = known_props | known_methods

        template_refs, loop_vars, pipe_names = cls.extract_template_locals(html_content)
        all_locals = template_refs | loop_vars | pipe_names | cls.ANGULAR_BUILTINS

        checked_bare: Set[str] = set()
        for bid in bound_ids:
            if bid in all_locals or bid in all_known:
                continue

            if "." in bid:
                root_obj, field = bid.split(".", 1)
                # If root object is a template local (e.g. loop var 'user' in 'let user of users'), it's valid
                if root_obj in all_locals:
                    continue

                # If root object is not known on the controller, flag unresolved root
                if root_obj not in all_known:
                    if root_obj.lower() in cls.COMMON_FIELD_BUILTINS:
                        continue
                    candidates = difflib.get_close_matches(root_obj, list(known_props), n=1, cutoff=0.35)
                    cand_hint = f" (did you mean '{candidates[0]}' on the controller?)" if candidates else ""
                    violations.append(
                        f"Binding '{bid}' references unknown object '{root_obj}' which does not exist on controller{cand_hint}"
                    )
                else:
                    # Root object is known on controller. Check if the accessed field is actually declared directly on the controller
                    if field in known_props:
                        violations.append(
                            f"Binding '{bid}' incorrectly accesses '{field}' on '{root_obj}'; '{field}' is declared directly on the controller (use '{field}')"
                        )
                    elif field.lower() not in cls.COMMON_FIELD_BUILTINS:
                        # Check if field was hallucinated when a similar controller property exists
                        candidates = difflib.get_close_matches(field, list(known_props), n=1, cutoff=0.45)
                        if candidates and candidates[0] != field:
                            violations.append(
                                f"Binding '{bid}' references property '{field}' not found on '{root_obj}' (did you mean '{candidates[0]}' on the controller?)"
                            )
            else:
                # Bare identifier binding: e.g. *ngIf="selectedUserIsExistingProjectMember"
                if bid in checked_bare or bid.lower() in cls.COMMON_FIELD_BUILTINS:
                    continue
                checked_bare.add(bid)
                # Find closest candidate in known_props
                candidates = difflib.get_close_matches(bid, list(known_props), n=1, cutoff=0.3)
                cand_hint = f" (did you mean '{candidates[0]}' on the controller?)" if candidates else ""
                violations.append(
                    f"Template expression references '{bid}' which does not exist on controller{cand_hint}"
                )

        return violations
