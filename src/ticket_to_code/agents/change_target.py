"""
Change Target Abstraction and Enforcement.

Core Invariants:
1. DISCOVERY != READ_ONLY_REFERENCE != CHANGE_TARGET != AUTHORIZED_CHANGE_TARGET.
   - Discovery / import / RAG never grants write permission.
   - Discovered dependencies default to READ_ONLY_REFERENCE.
   - Only explicit implementation evidence can promote a reference to a ChangeTarget.
   - Only verified authorization promotes a ChangeTarget to an AUTHORIZED_CHANGE_TARGET.
2. A writable file does not grant whole-file write permission.
   - Modifications must target an approved ChangeTarget (symbol, block, or range).
3. Target-First Generation:
   - The generator injects the target block + structural skeleton + read-only references.
   - Whole-file injection is strictly an exceptional fallback.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Set

from ticket_to_code.models import ChangeTarget, DevelopmentTask, TaskType

logger = logging.getLogger(__name__)


class ChangeTargetResolver:
    """Resolves granular change targets from source code using AST/regex analysis."""

    # ── Language Detection ───────────────────────────────────────────────
    _EXT_MAP = {
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".java": "java",
        ".py": "python",
        ".html": "html",
        ".htm": "html",
        ".scss": "scss",
        ".css": "css",
        ".sass": "scss",
        ".less": "scss",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }

    @classmethod
    def detect_language(cls, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return cls._EXT_MAP.get(ext, "unknown")

    # ── Symbol Extraction ────────────────────────────────────────────────
    @classmethod
    def resolve_target(
        cls,
        file_path: str,
        content: str,
        symbol: Optional[str] = None,
        symbol_type: Optional[str] = None,
        reason: str = "",
        modification_intent: str = "",
        authorization_source: str = "",
        evidence_ids: Optional[List[str]] = None,
        readonly_dependencies: Optional[List[str]] = None,
    ) -> ChangeTarget:
        """Resolve a ChangeTarget with line boundaries and surrounding context."""
        evidence_ids = evidence_ids or []
        readonly_dependencies = readonly_dependencies or []

        if not content:
            return ChangeTarget(
                file_path=file_path,
                symbol=symbol,
                symbol_type=symbol_type or "file",
                reason=reason,
                modification_intent=modification_intent,
                authorization_source=authorization_source,
                evidence_ids=evidence_ids,
                readonly_dependencies=readonly_dependencies,
            )

        lang = cls.detect_language(file_path)
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        start_line = None
        end_line = None
        detected_type = symbol_type or "file"
        surrounding_context = None

        if symbol:
            boundary = cls._find_symbol_boundary(lines, symbol, lang)
            if boundary:
                start_line, end_line, detected_type = boundary
                # Extract surrounding context: 3 lines before, 3 lines after
                ctx_start = max(0, start_line - 4)
                ctx_end = min(total_lines, end_line + 3)
                surrounding_context = "".join(lines[ctx_start:ctx_end])

        return ChangeTarget(
            file_path=file_path,
            symbol=symbol,
            symbol_type=detected_type,
            start_line=start_line,
            end_line=end_line,
            surrounding_context=surrounding_context,
            reason=reason,
            modification_intent=modification_intent,
            authorization_source=authorization_source,
            evidence_ids=evidence_ids,
            readonly_dependencies=readonly_dependencies,
            is_authorized=False,
        )

    @classmethod
    def _find_symbol_boundary(
        cls, lines: List[str], symbol: str, lang: str
    ) -> Optional[Tuple[int, int, str]]:
        """Find start_line, end_line, and symbol_type for a given symbol."""
        full_text = "".join(lines)
        sym_pattern = re.escape(symbol)

        # 1. TypeScript / JavaScript / Java method or function
        if lang in ("typescript", "javascript", "java", "python"):
            # Method/Function declaration: e.g. onUserSelected(...) { or public void foo(...) {
            method_re = re.compile(
                rf"(?:(?:public|private|protected|async|static|export)\s+)*"
                rf"(?:function\s+|class\s+|interface\s+)?{sym_pattern}\s*[\(<{{]",
                re.MULTILINE,
            )
            for m in method_re.finditer(full_text):
                start_pos = m.start()
                start_line = full_text[:start_pos].count("\n") + 1
                
                # Determine type
                match_str = m.group(0)
                if "interface" in match_str:
                    sym_type = "interface"
                elif "class" in match_str:
                    sym_type = "class"
                else:
                    sym_type = "method"

                # Find closing brace if balanced
                end_line = cls._find_closing_brace_line(lines, start_line - 1)
                return (start_line, end_line, sym_type)

        # 2. HTML template element (e.g. <ot-item-select ...> or class="...symbol...")
        if lang == "html":
            tag_re = re.compile(rf"<([a-zA-Z0-9_\-]+)[^>]*{sym_pattern}[^>]*>", re.MULTILINE)
            for m in tag_re.finditer(full_text):
                start_pos = m.start()
                start_line = full_text[:start_pos].count("\n") + 1
                tag_name = m.group(1)
                end_line = cls._find_closing_html_tag_line(lines, start_line - 1, tag_name)
                return (start_line, end_line, "template_block")

        # 3. SCSS / CSS selector
        if lang in ("scss", "css"):
            rule_re = re.compile(rf"[^}}\n]*{sym_pattern}[^{{\n]*\{{", re.MULTILINE)
            for m in rule_re.finditer(full_text):
                start_pos = m.start()
                start_line = full_text[:start_pos].count("\n") + 1
                end_line = cls._find_closing_brace_line(lines, start_line - 1)
                return (start_line, end_line, "style_rule")

        # Fallback: line containing the symbol
        for i, line in enumerate(lines):
            if symbol in line:
                return (i + 1, min(len(lines), i + 10), "block")

        return None

    @classmethod
    def _find_closing_brace_line(cls, lines: List[str], start_idx: int) -> int:
        """Find line index containing the matching closing brace."""
        depth = 0
        started = False
        for i in range(start_idx, len(lines)):
            line = lines[i]
            for ch in line:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}":
                    depth -= 1
                    if started and depth == 0:
                        return i + 1
        return min(len(lines), start_idx + 30)

    @classmethod
    def _find_closing_html_tag_line(cls, lines: List[str], start_idx: int, tag_name: str) -> int:
        """Find the matching closing tag line for an HTML element."""
        close_tag = f"</{tag_name}>"
        for i in range(start_idx, len(lines)):
            if close_tag in lines[i]:
                return i + 1
        return min(len(lines), start_idx + 15)


def authorize_change_target(
    target: ChangeTarget,
    authorized_files: Set[str],
    evidence_ids: Set[str],
    is_breaking_type_change: bool = False,
    has_migration_evidence: bool = False,
) -> Tuple[bool, str]:
    """
    Enforces authorization on a proposed ChangeTarget.
    
    Invariants:
    1. File must be in the authorized writable scope.
    2. Must be backed by verifiable evidence_ids or authorization_source.
    3. Unjustified breaking type changes are REJECTED.
    """
    norm_path = target.file_path.replace("\\", "/").lower()
    norm_auth = {a.replace("\\", "/").lower() for a in authorized_files}

    # 1. File authorization check
    is_file_authorized = any(
        norm_path == a or norm_path.endswith("/" + a) or a.endswith("/" + norm_path)
        for a in norm_auth
    )
    if not is_file_authorized:
        return False, f"File '{target.file_path}' is not in authorized writable scope"

    # 2. Evidence validation check
    has_evidence = bool(target.evidence_ids and any(e in evidence_ids for e in target.evidence_ids))
    has_source = bool(target.authorization_source)
    if not (has_evidence or has_source):
        return False, f"Target '{target.symbol or target.file_path}' lacks explicit evidence or authorization source"

    # 3. Breaking type change policy: Unjustified breaking changes are rejected
    if is_breaking_type_change and not has_migration_evidence:
        return False, f"Breaking type change on '{target.symbol}' is unjustified (no migration evidence)"

    target.is_authorized = True
    return True, "Authorized"
