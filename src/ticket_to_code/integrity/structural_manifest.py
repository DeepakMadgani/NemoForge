"""
Structural Change Manifest and Deletion Authorization Validator.

Extracts AST and template-level structural symbols (classes, methods, fields,
constructors, template tags, imports) to detect and block accidental truncation,
silent omissions, and unauthorized deletions before patches are applied.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class UnauthorizedDeletionError(Exception):
    """Raised when an unauthorized deletion or accidental omission is detected."""

    def __init__(self, file_path: str, symbol: str, reason: str, details: Optional[Dict] = None):
        super().__init__(f"Unauthorized deletion in {file_path}: symbol '{symbol}' — {reason}")
        self.file_path = file_path
        self.symbol = symbol
        self.reason = reason
        self.details = details or {}


@dataclass
class DeleteAuthorization:
    """Explicit, structured authorization for a code deletion."""
    operation: str  # "DELETE_SYMBOL" | "DELETE_REGION" | "DELETE_FILE"
    file: str
    symbol: Optional[str] = None
    reason: str = ""
    requirement_id: str = ""
    evidence: List[str] = field(default_factory=list)

    def is_authorized(self) -> bool:
        return bool(self.requirement_id and self.reason)


@dataclass
class StructuralSymbols:
    """Extracted structural symbols for a file."""
    classes: Set[str] = field(default_factory=set)
    methods: Set[str] = field(default_factory=set)
    properties: Set[str] = field(default_factory=set)
    constructor_params: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    template_directives: Set[str] = field(default_factory=set)


@dataclass
class StructuralDelta:
    """Differences between original and generated structural elements."""
    file_path: str
    added_symbols: List[str] = field(default_factory=list)
    modified_symbols: List[str] = field(default_factory=list)
    removed_symbols: List[str] = field(default_factory=list)
    dropped_params: List[str] = field(default_factory=list)
    is_clean: bool = True
    warnings: List[str] = field(default_factory=list)


class StructuralChangeManifest:
    """Extracts and compares structural elements across versions of code."""

    @staticmethod
    def extract_symbols(content: str, file_path: str) -> StructuralSymbols:
        """Extract structural symbols using AST where available or resilient regex."""
        symbols = StructuralSymbols()
        if not content:
            return symbols

        ext = Path(file_path).suffix.lower()

        # Try tree-sitter first for JS/TS/Java/Python
        try:
            from ticket_to_code.agents.smart_extract import parse_with_treesitter
            tree, lang = parse_with_treesitter(content, file_path)
            if tree is not None:
                StructuralChangeManifest._extract_treesitter_symbols(tree, lang, symbols)
                return symbols
        except Exception as e:
            logger.debug(f"Tree-sitter extraction fallback to regex for {file_path}: {e}")

        # Resilient regex extraction
        if ext in (".ts", ".tsx", ".js", ".jsx"):
            StructuralChangeManifest._extract_ts_js(content, symbols)
        elif ext == ".java":
            StructuralChangeManifest._extract_java(content, symbols)
        elif ext == ".py":
            StructuralChangeManifest._extract_python(content, symbols)
        elif ext in (".html", ".component.html"):
            StructuralChangeManifest._extract_template(content, symbols)

        return symbols

    @staticmethod
    def _extract_treesitter_symbols(tree, lang: str, symbols: StructuralSymbols):
        def _walk(node):
            nt = node.type
            if lang in ("typescript", "javascript"):
                if nt in ("class_declaration", "interface_declaration"):
                    name_node = node.child_by_field_name("name")
                    if name_node and name_node.text:
                        symbols.classes.add(name_node.text.decode("utf-8", errors="ignore"))
                elif nt in ("method_definition", "function_declaration", "generator_function_declaration"):
                    name_node = node.child_by_field_name("name")
                    if name_node and name_node.text:
                        symbols.methods.add(name_node.text.decode("utf-8", errors="ignore"))
                elif nt in ("public_field_definition", "property_definition", "field_definition"):
                    name_node = node.child_by_field_name("property") or node.child_by_field_name("name")
                    if name_node and name_node.text:
                        symbols.properties.add(name_node.text.decode("utf-8", errors="ignore"))
                elif nt == "import_statement":
                    symbols.imports.add(node.text.decode("utf-8", errors="ignore")[:80])
            elif lang == "java":
                if nt in ("class_declaration", "interface_declaration", "enum_declaration"):
                    name_node = node.child_by_field_name("name")
                    if name_node and name_node.text:
                        symbols.classes.add(name_node.text.decode("utf-8", errors="ignore"))
                elif nt in ("method_declaration", "constructor_declaration"):
                    name_node = node.child_by_field_name("name")
                    if name_node and name_node.text:
                        symbols.methods.add(name_node.text.decode("utf-8", errors="ignore"))
                elif nt == "field_declaration":
                    symbols.properties.add(node.text.decode("utf-8", errors="ignore")[:60])
                elif nt == "import_declaration":
                    symbols.imports.add(node.text.decode("utf-8", errors="ignore"))
            elif lang == "python":
                if nt == "class_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node and name_node.text:
                        symbols.classes.add(name_node.text.decode("utf-8", errors="ignore"))
                elif nt == "function_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node and name_node.text:
                        symbols.methods.add(name_node.text.decode("utf-8", errors="ignore"))
                elif nt in ("import_statement", "import_from_statement"):
                    symbols.imports.add(node.text.decode("utf-8", errors="ignore"))

            for child in node.children:
                _walk(child)

        _walk(tree.root_node)

    @staticmethod
    def _extract_ts_js(content: str, symbols: StructuralSymbols):
        for m in re.finditer(r'(?:export\s+)?(?:class|interface)\s+(\w+)', content):
            symbols.classes.add(m.group(1))
        for m in re.finditer(r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*[:\{]', content):
            name = m.group(1)
            if name not in ("if", "for", "while", "switch", "catch", "constructor"):
                symbols.methods.add(name)
        ctor_match = re.search(r'constructor\s*\(([^)]*)\)', content)
        if ctor_match:
            symbols.methods.add("constructor")
            params = re.findall(r'(?:private|public|protected|readonly)?\s*(\w+)\s*[:\,)]', ctor_match.group(1))
            symbols.constructor_params.update(params)
        for m in re.finditer(r'^\s*(?:public|private|protected|readonly)?\s*(\w+)\s*[:=]', content, re.MULTILINE):
            name = m.group(1)
            if name not in ("constructor", "export", "import", "class", "interface", "const", "let", "var"):
                symbols.properties.add(name)

    @staticmethod
    def _extract_java(content: str, symbols: StructuralSymbols):
        for m in re.finditer(r'(?:public|protected|private)?\s*(?:static\s+)?(?:final\s+)?class\s+(\w+)', content):
            symbols.classes.add(m.group(1))
        for m in re.finditer(r'(?:public|protected|private)?\s*(?:static\s+)?[\w\<\>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{', content):
            name = m.group(1)
            if name not in ("if", "for", "while", "switch", "catch"):
                symbols.methods.add(name)

    @staticmethod
    def _extract_python(content: str, symbols: StructuralSymbols):
        for m in re.finditer(r'^\s*class\s+(\w+)', content, re.MULTILINE):
            symbols.classes.add(m.group(1))
        for m in re.finditer(r'^\s*(?:async\s+)?def\s+(\w+)', content, re.MULTILINE):
            symbols.methods.add(m.group(1))

    @staticmethod
    def _extract_template(content: str, symbols: StructuralSymbols):
        for m in re.finditer(r'\*ng(?:If|For|SwitchCase)\s*=\s*"([^"]+)"', content):
            symbols.template_directives.add(m.group(0).strip())
        for m in re.finditer(r'\(([a-zA-Z0-9_\.]+)\)\s*=\s*"([^"]+)"', content):
            symbols.template_directives.add(f"event:{m.group(1)}={m.group(2)}")
        for m in re.finditer(r'\[([a-zA-Z0-9_\.]+)\]\s*=\s*"([^"]+)"', content):
            symbols.template_directives.add(f"binding:{m.group(1)}={m.group(2)}")

    @classmethod
    def compute_delta(cls, original: str, generated: str, file_path: str) -> StructuralDelta:
        """Compute symbol-level structural delta between original and generated code."""
        orig_syms = cls.extract_symbols(original, file_path)
        gen_syms = cls.extract_symbols(generated, file_path)

        added = []
        removed = []
        modified = []

        # Check methods
        for m in gen_syms.methods - orig_syms.methods:
            added.append(f"method:{m}")
        for m in orig_syms.methods - gen_syms.methods:
            removed.append(f"method:{m}")
        for m in orig_syms.methods & gen_syms.methods:
            modified.append(f"method:{m}")

        # Check classes
        for c in gen_syms.classes - orig_syms.classes:
            added.append(f"class:{c}")
        for c in orig_syms.classes - gen_syms.classes:
            removed.append(f"class:{c}")

        # Check dropped constructor parameters
        dropped_params = list(orig_syms.constructor_params - gen_syms.constructor_params)
        for p in dropped_params:
            removed.append(f"constructor_param:{p}")

        # Check dropped template bindings/directives if HTML
        if Path(file_path).suffix.lower() in (".html", ".component.html"):
            gen_keys = {
                d.split("=")[0] for d in gen_syms.template_directives
                if d.startswith(("event:", "binding:"))
            }
            for d in orig_syms.template_directives - gen_syms.template_directives:
                # If the same event or binding target exists (e.g. event:click), it is modified, not removed
                if d.startswith(("event:", "binding:")) and d.split("=")[0] in gen_keys:
                    modified.append(f"template:{d}")
                else:
                    removed.append(f"template:{d}")

        is_clean = len(removed) == 0
        warnings = []
        if removed:
            warnings.append(f"Detected {len(removed)} removed structural symbol(s): {removed}")

        return StructuralDelta(
            file_path=file_path,
            added_symbols=added,
            modified_symbols=modified,
            removed_symbols=removed,
            dropped_params=dropped_params,
            is_clean=is_clean,
            warnings=warnings,
        )


class DeleteAuthorizationValidator:
    """Validates whether any removed symbol has explicit ticket authorization."""

    @staticmethod
    def validate_deletions(
        delta: StructuralDelta,
        authorizations: Optional[List[DeleteAuthorization]] = None,
        protected_regions: Optional[List] = None,
    ) -> List[str]:
        """
        Verify every removed symbol has structured authorization.
        Returns list of violation messages. Empty list = all removals authorized or none.
        """
        if not delta.removed_symbols:
            return []

        auths = authorizations or []
        violations = []

        for item in delta.removed_symbols:
            # item format: "kind:name", e.g. "method:onMemberAdd"
            parts = item.split(":", 1)
            kind = parts[0]
            name = parts[1] if len(parts) > 1 else item

            # Check if any authorization covers this symbol
            matching_auth = None
            for a in auths:
                if a.is_authorized():
                    # check file match
                    norm_a_file = a.file.replace("\\", "/").lower()
                    norm_d_file = delta.file_path.replace("\\", "/").lower()
                    if norm_a_file == norm_d_file or norm_d_file.endswith(norm_a_file):
                        if not a.symbol or a.symbol == name or a.symbol == item:
                            matching_auth = a
                            break

            if not matching_auth:
                violations.append(
                    f"UNAUTHORIZED_DELETION: Symbol '{item}' was removed from {delta.file_path} "
                    f"without explicit ticket deletion authorization (missing requirement_id/reason). "
                    f"Accidental truncation or omission suspected."
                )
            else:
                logger.info(
                    f"Authorized deletion of '{item}' in {delta.file_path} per "
                    f"requirement {matching_auth.requirement_id}: {matching_auth.reason}"
                )

        return violations
