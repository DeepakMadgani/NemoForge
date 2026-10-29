"""
Generated Reference Validator — prevents hallucinated outbound calls.

After a file is generated, this extracts every outbound member reference
(``receiver.member(...)``) and verifies the callee actually exists, using the
existing SymbolResolver (repository truth) and ImplementationState (symbols
generated+verified this run).

Architecture:
    SYNTAX-SPECIFIC EXTRACTION → GENERIC RESOLUTION → CONTRACT VERIFICATION

    Extractors (syntax-specific):
        - Imperative: ``receiver.member(`` call syntax (Java / TS / C# / Python)
        - Angular Template: ``{{ expr }}``, ``[attr]="expr"``, ``(event)="fn()"``
        - (Future: React JSX, C# Razor, etc.)

    Resolver (generic):
        1. SymbolResolver (repository truth — types on disk)
        2. ImplementationState (symbols generated+verified this run)
        3. written_files (freshly generated content, not yet indexed)

    Each extractor produces ``CrossFileReference`` objects.
    All resolution goes through ONE generic path.

It NEVER promotes a reference to "verified" because the planner or generator
asked for it — only SymbolResolver or verified generated symbols can do that.

A receiver whose type cannot be inferred is reported as UNKNOWN_RECEIVER
(advisory only) and never blocks acceptance.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from ticket_to_code.agents.template_expression_parser import TemplateExpressionParser

logger = logging.getLogger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────────

# receiver.member(  — receiver must start lowercase/underscore/$ so we skip
# static calls on Types (e.g. Optional.ofNullable) and this/self chains.
_CALL_RE = re.compile(r'(?<![\w.])([a-z_$][\w$]*)\.([a-zA-Z_$][\w$]*)\s*\(')

# Java/C#/Kotlin field or param:  private final FooRepository memberRepository;
_TYPE_THEN_NAME_RE = re.compile(r'\b([A-Z][A-Za-z0-9_]*)(?:<[^>]*>)?\s+([a-z_$][\w$]*)\s*[;=,)]')

# TS/Python annotation:  private memberRepo: FooRepository   |   repo: FooRepo
_NAME_THEN_TYPE_RE = re.compile(r'\b([a-z_$][\w$]*)\s*:\s*([A-Z][A-Za-z0-9_]*)')

_SKIP_RECEIVERS = {"this", "self", "super", "cls"}
_FUZZY_THRESHOLD = 0.72

# ── Angular template extraction patterns ───────────────────────────────────────

# {{ expr }} interpolation — captures content between {{ and }}
_INTERPOLATION_RE = re.compile(r'\{\{\s*(.*?)\s*\}\}', re.DOTALL)

# [attr]="expr" property binding — captures the expression
_PROP_BINDING_RE = re.compile(r'\[(?:[\w.]+)\]\s*=\s*"([^"]*)"')

# (event)="expr" event binding — captures the expression
_EVENT_BINDING_RE = re.compile(r'\((?:[\w.]+)\)\s*=\s*"([^"]*)"')

# *ngIf="expr" structural directive — captures the expression
_NG_IF_RE = re.compile(r'\*ngIf\s*=\s*"([^"]*)"')

# *ngFor="let x of expr" — captures the iterator variable and the iterable
_NG_FOR_RE = re.compile(r'\*ngFor\s*=\s*"let\s+(\w+)\s+of\s+([\w.]+)(?:\s*;[^"]*)?"')

# Property access chain: receiver.prop1.prop2 (no trailing parens)
_PROP_ACCESS_RE = re.compile(r'\b([a-z_$][\w$]*)\.([a-zA-Z_$][\w$]*)')

# Method call in expression: receiver.method() or method()
_METHOD_CALL_IN_EXPR_RE = re.compile(r'\b([a-z_$][\w$]*)\.([a-zA-Z_$][\w$]*)\s*\(')

# TS property declaration patterns (for extracting from written_files content)
_TS_PROPERTY_RE = re.compile(
    r'(?:public|private|protected|readonly)?\s*(\w+)\s*[?!]?\s*[:=]',
    re.MULTILINE,
)
_TS_METHOD_RE = re.compile(
    r'(?:public|private|protected|async)?\s*(\w+)\s*\(',
    re.MULTILINE,
)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class CrossFileReference:
    """A single cross-file reference extracted from generated content.

    Language-neutral — any extractor produces these, all resolution goes
    through the same generic path.
    """
    receiver: str              # e.g. "member", "this"
    member: str                # e.g. "existingOrganizationName", "onMemberAdd"
    source_file: str           # file containing the reference
    source_syntax: str         # "angular_binding", "angular_event", "ts_call", etc.
    line_hint: str = ""        # original expression for diagnostics
    receiver_type_hint: str = ""  # inferred type if known (e.g. "Member")


@dataclass
class ReferenceCheck:
    receiver: str
    member: str
    resolved_type: Optional[str]
    status: str                      # ReferenceStatus value
    close_match: Optional[str] = None


@dataclass
class ReferenceValidationResult:
    checks: list = field(default_factory=list)

    @property
    def unresolved(self) -> list:
        return [c for c in self.checks if c.status == "unresolved"]

    @property
    def is_clean(self) -> bool:
        return not self.unresolved


# ── Angular Template Extractor ─────────────────────────────────────────────────

class AngularTemplateExtractor:
    """Extracts cross-file references from Angular HTML templates.

    This is ONLY a reference extractor — it does not resolve anything.
    Resolution is handled by GeneratedReferenceValidator's generic resolver.

    Parses:
        {{ member.existingOrganizationName }}     → binding
        [placeholder]="member.organizationName"   → binding
        (click)="onMemberAdd()"                   → event
        *ngIf="member.isActive"                   → binding
        *ngFor="let m of projectMembers"          → binding
    """

    @staticmethod
    def extract(content: str, file_path: str) -> list[CrossFileReference]:
        """Extract all cross-file references from an Angular template."""
        if not content:
            return []

        refs: list[CrossFileReference] = []
        seen: set[tuple[str, str]] = set()

        # Build ngFor variable map: {iterator_var: iterable_expression}
        ngfor_vars: dict[str, str] = {}
        for m in _NG_FOR_RE.finditer(content):
            ngfor_vars[m.group(1)] = m.group(2)

        # Collect all expressions from Angular-specific syntax
        expressions: list[tuple[str, str]] = []  # (expression, syntax_type)

        for m in _INTERPOLATION_RE.finditer(content):
            expressions.append((m.group(1), "angular_interpolation"))

        for m in _PROP_BINDING_RE.finditer(content):
            expressions.append((m.group(1), "angular_property_binding"))

        for m in _EVENT_BINDING_RE.finditer(content):
            expressions.append((m.group(1), "angular_event_binding"))

        for m in _NG_IF_RE.finditer(content):
            # Strip trailing ; else ... and logical operators for simpler parsing
            expr = m.group(1).split(";")[0].strip()
            expressions.append((expr, "angular_structural_directive"))

        # For *ngFor, the iterable itself is a reference
        for m in _NG_FOR_RE.finditer(content):
            iterable = m.group(2)
            expressions.append((iterable, "angular_structural_directive"))

        # Collect all template-local variables dynamically directly from template AST/syntax
        local_template_vars: set[str] = set(ngfor_vars.keys())
        # 1. Template reference variables: <input #phone> or #var="ngModel"
        for m in re.finditer(r'#([a-zA-Z_$][a-zA-Z0-9_$]*)', content):
            local_template_vars.add(m.group(1))
        # 2. Structural directive let-bindings: let-item, let-i="index"
        for m in re.finditer(r'\blet-([a-zA-Z_$][a-zA-Z0-9_$]*)', content):
            local_template_vars.add(m.group(1))
        for m in re.finditer(r'\blet\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content):
            local_template_vars.add(m.group(1))
        # 3. Structural directive as-bindings: *ngIf="user$ | async as user"
        for m in re.finditer(r'\bas\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content):
            local_template_vars.add(m.group(1))
        # 4. Modern Angular @let and @for syntax
        for m in re.finditer(r'@let\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content):
            local_template_vars.add(m.group(1))
        for m in re.finditer(r'@for\s*\(\s*(?:let\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s+of', content):
            local_template_vars.add(m.group(1))

        # Process each expression using structural TemplateExpressionParser
        # Invariant: Only syntactically identified identifiers/member accesses become references.
        # Literals (true, false, null, undefined, strings, numbers) and template-local vars are never extracted.
        for expr, syntax_type in expressions:
            parsed_refs = TemplateExpressionParser.extract_references(
                expr, syntax_type=syntax_type, local_vars=local_template_vars
            )
            for t_ref in parsed_refs:
                key = (t_ref.receiver, t_ref.member)
                if key not in seen:
                    seen.add(key)
                    refs.append(CrossFileReference(
                        receiver=t_ref.receiver,
                        member=t_ref.member,
                        source_file=file_path,
                        source_syntax=t_ref.source_syntax,
                        line_hint=t_ref.line_hint,
                    ))

        return refs


# ── Main Validator ─────────────────────────────────────────────────────────────

class GeneratedReferenceValidator:
    """Validates outbound references in generated source against repository truth.

    Supports two validation modes:
    1. validate(content) — imperative code references (receiver.member() calls)
    2. validate_template(content, file_path, written_files) — Angular template
       bindings resolved against sibling controller + repository/generated truth

    Both modes use the SAME generic resolution engine.
    """

    def __init__(self, symbol_resolver=None, impl_state=None):
        self._resolver = symbol_resolver
        self._impl = impl_state

    # ── Mode 1: Imperative code references (existing) ────────────────────

    def validate(self, content: str) -> ReferenceValidationResult:
        result = ReferenceValidationResult()
        if not content or not self._resolver:
            return result

        receiver_types = self._infer_receiver_types(content)
        seen: set = set()

        for m in _CALL_RE.finditer(content):
            receiver, member = m.group(1), m.group(2)
            if receiver in _SKIP_RECEIVERS:
                continue
            if (receiver, member) in seen:
                continue
            seen.add((receiver, member))

            owner = receiver_types.get(receiver)
            if not owner:
                result.checks.append(
                    ReferenceCheck(receiver, member, None, "unknown_receiver")
                )
                continue

            status, close = self._resolve_member(owner, member)
            result.checks.append(
                ReferenceCheck(receiver, member, owner, status, close)
            )

        return result

    # ── Mode 2: Template binding references ──────────────────────────────

    def validate_template(
        self,
        html_content: str,
        html_file_path: str,
        written_files: Optional[dict[str, str]] = None,
    ) -> ReferenceValidationResult:
        """Validate Angular template bindings against the generated workspace.

        Architecture:
            1. AngularTemplateExtractor extracts CrossFileReference objects
            2. Sibling TS controller content provides receiver type mapping
            3. Generic resolver checks against:
               - SymbolResolver (repository truth)
               - ImplementationState (generated symbols)
               - written_files (freshly generated content)

        Args:
            html_content: The generated HTML template content.
            html_file_path: Path to the HTML file.
            written_files: Dict of {normalized_path: content} for files
                           written in the current session.

        Returns:
            ReferenceValidationResult with checks for each binding.
        """
        result = ReferenceValidationResult()
        if not html_content:
            return result

        written_files = written_files or {}

        # Step 1: Extract references using Angular-specific extractor
        refs = AngularTemplateExtractor.extract(html_content, html_file_path)
        if not refs:
            return result

        # Step 2: Get sibling TS controller content for type inference
        controller_content = self._get_sibling_controller_content(
            html_file_path, written_files
        )
        controller_members = self._extract_members_from_content(controller_content)
        controller_types = self._infer_receiver_types(controller_content) if controller_content else {}

        # Step 3: Build ngFor type map from controller declarations
        ngfor_type_map = self._build_ngfor_type_map(html_content, controller_content)

        # Step 4: Resolve each reference through the generic engine
        for ref in refs:
            check = self._resolve_template_reference(
                ref, controller_content, controller_members,
                controller_types, ngfor_type_map, written_files
            )
            if check:
                result.checks.append(check)

        return result

    def _resolve_template_reference(
        self,
        ref: CrossFileReference,
        controller_content: str,
        controller_members: set[str],
        controller_types: dict[str, str],
        ngfor_type_map: dict[str, str],
        written_files: dict[str, str],
    ) -> Optional[ReferenceCheck]:
        """Resolve a single template reference through the generic engine.

        Resolution order:
        1. "this" references → check controller members
        2. ngFor variable references → check element type members
        3. Direct variable references → check controller member types
        """
        receiver = ref.receiver
        member = ref.member

        # ── "this" references (controller methods/properties) ────────────
        if receiver == "this":
            if member in controller_members:
                return ReferenceCheck(receiver, member, "controller", "existing_verified")

            # Check via SymbolResolver if controller type is indexed
            close = self._closest(member, list(controller_members))
            return ReferenceCheck(receiver, member, "controller", "unresolved", close)

        # ── ngFor/local variable references ──────────────────────────────
        # e.g. *ngFor="let member of projectMembers" → member.organizationName
        # Need to resolve: what is the element type of projectMembers?
        element_type = ngfor_type_map.get(receiver)

        if element_type:
            # Try resolving the member on the element type
            status, close = self._resolve_member(element_type, member)
            if status not in ("unknown_receiver",):
                return ReferenceCheck(receiver, member, element_type, status, close)

        # ── Fallback: check against ALL known members in controller ──────
        # If type resolution fails (model not generated yet, etc.), check if
        # the member name exists literally in the sibling controller content.
        # This catches the exact existingOrganizationName mismatch.
        if controller_content:
            if re.search(r'\b' + re.escape(member) + r'\b', controller_content):
                return ReferenceCheck(
                    receiver, member, element_type or "unknown",
                    "existing_verified"
                )
            # Check for close matches in controller body
            # Extract all identifiers from controller
            all_identifiers = set(re.findall(r'\b([a-zA-Z_]\w{3,})\b', controller_content))
            close = self._closest(member, list(all_identifiers))
            if close:
                return ReferenceCheck(
                    receiver, member, element_type or "unknown",
                    "unresolved", close
                )

        # ── Check in written_files ───────────────────────────────────────
        # Scan all written files for the member name
        for wf_path, wf_content in written_files.items():
            if wf_content and re.search(r'\b' + re.escape(member) + r'\b', wf_content):
                return ReferenceCheck(
                    receiver, member, element_type or "unknown",
                    "existing_verified"
                )

        # If we have a type but no resolution, report as unresolved
        if element_type:
            return ReferenceCheck(receiver, member, element_type, "unresolved")

        # Can't determine type at all — advisory only
        return ReferenceCheck(receiver, member, None, "unknown_receiver")

    def _get_sibling_controller_content(
        self, html_file_path: str, written_files: dict[str, str]
    ) -> str:
        """Find and return the sibling TS controller content.

        Checks written_files first (most up-to-date), then disk.
        """
        stem = html_file_path.rsplit(".", 1)[0]
        for ext in (".ts", ".tsx"):
            ctrl_path = stem + ext
            ctrl_key = ctrl_path.replace("\\", "/").lower()
            if ctrl_key in written_files:
                return written_files[ctrl_key]
        return ""

    def _build_ngfor_type_map(
        self, html_content: str, controller_content: str
    ) -> dict[str, str]:
        """Build a map of ngFor iterator variables to their element types.

        e.g. *ngFor="let member of projectMembers"
             controller has: projectMembers: ProjectMember[]
             → {"member": "ProjectMember"}
        """
        type_map: dict[str, str] = {}
        if not controller_content:
            return type_map

        for m in _NG_FOR_RE.finditer(html_content):
            iter_var = m.group(1)
            iterable = m.group(2)

            # Look up the type of the iterable in the controller
            # Pattern: iterable_name: Type[] or iterable_name: Array<Type>
            array_type = re.search(
                r'\b' + re.escape(iterable) + r'\s*[?!]?\s*:\s*([A-Z][\w]*)\s*\[\]',
                controller_content
            )
            if array_type:
                type_map[iter_var] = array_type.group(1)
                continue

            generic_type = re.search(
                r'\b' + re.escape(iterable) + r'\s*[?!]?\s*:\s*(?:Array|Observable)\s*<\s*([A-Z][\w]*)',
                controller_content
            )
            if generic_type:
                type_map[iter_var] = generic_type.group(1)

        return type_map

    @staticmethod
    def _extract_members_from_content(content: str) -> set[str]:
        """Extract declared member names (properties + methods) from TS content.

        This is an extraction helper — NOT the source of truth.
        Resolution still goes through SymbolResolver.
        """
        if not content:
            return set()

        members: set[str] = set()
        # TS keywords to skip
        _SKIP = {
            "if", "else", "for", "while", "do", "switch", "case", "break",
            "continue", "return", "new", "delete", "typeof", "instanceof",
            "void", "null", "undefined", "true", "false", "class", "interface",
            "enum", "type", "const", "let", "var", "function", "import",
            "export", "from", "async", "await", "try", "catch", "finally",
            "throw", "extends", "implements", "constructor", "super", "this",
            "static", "readonly", "public", "private", "protected", "abstract",
            "get", "set", "any", "string", "number", "boolean", "object",
            "never", "unknown", "Symbol", "Error", "Promise", "Array",
            "Observable", "Subscription", "Component", "Injectable",
            "OnInit", "OnDestroy", "Input", "Output", "EventEmitter",
            "ViewChild", "NgModule", "NgForm",
        }

        for m in _TS_PROPERTY_RE.finditer(content):
            name = m.group(1)
            if name not in _SKIP and len(name) > 1:
                members.add(name)

        for m in _TS_METHOD_RE.finditer(content):
            name = m.group(1)
            if name not in _SKIP and len(name) > 1:
                members.add(name)

        return members

    # ── Generic resolution internals ─────────────────────────────────────

    def _resolve_member(self, owner: str, member: str):
        """Generic member resolution — used by both imperative and template modes.

        Resolution order:
        1. ImplementationState (symbols generated+verified this run)
        2. SymbolResolver (repository truth)
        3. Fuzzy match for close alternatives
        """
        # 1. Verified generated symbol this run (producer already generated it)
        if self._impl is not None:
            try:
                if self._impl.has_verified_symbol(member, owner):
                    return "generated_verified", None
            except Exception:
                pass

        # 2. Repository truth via SymbolResolver
        if not self._resolver:
            return "unknown_receiver", None

        try:
            defn = self._resolver.resolve_type(owner)
        except Exception:
            defn = None

        if defn is None:
            # Receiver's type isn't resolvable → advisory, never block.
            return "unknown_receiver", None

        try:
            if defn.has_member(member):
                return "existing_verified", None
        except Exception:
            return "unknown_receiver", None

        candidates = list(getattr(defn, "method_names", []) or []) + \
            list(getattr(defn, "property_names", []) or [])
        return "unresolved", self._closest(member, candidates)

    def _infer_receiver_types(self, content: str) -> dict:
        """Map local receiver name → declared type. First declaration wins."""
        types: dict = {}
        for m in _TYPE_THEN_NAME_RE.finditer(content):
            types.setdefault(m.group(2), m.group(1))
        for m in _NAME_THEN_TYPE_RE.finditer(content):
            types.setdefault(m.group(1), m.group(2))
        return types

    @staticmethod
    def _closest(name: str, candidates: list) -> Optional[str]:
        best, ratio = None, 0.0
        nl = name.lower()
        for c in candidates:
            r = SequenceMatcher(None, nl, c.lower()).ratio()
            if r > ratio and r >= _FUZZY_THRESHOLD:
                best, ratio = c, r
        return best
