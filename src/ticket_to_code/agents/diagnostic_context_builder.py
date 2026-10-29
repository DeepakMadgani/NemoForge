"""
Diagnostic Context Builder — AST/symbol-first resolution of compiler errors.

Replaces whole-file preloading and arbitrary line-window heuristics with:
1. Compiler diagnostic -> file -> line/column -> AST/symbol resolution -> containing symbol.
2. Target Source: entire containing method/symbol body.
3. Structural Context: class header + relevant property/method signatures.
4. Related Symbols: only symbols actually referenced by target.
5. Parser Fallback: bounded line window only when AST symbol resolution cannot find a boundary.
6. Staleness Tracking: sha256 content hashes to reuse validated evidence without re-querying.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set

from ticket_to_code.agents.smart_extract import (
    _get_reliable_boundaries,
    MethodBoundary,
    detect_language,
)

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticContext:
    """Target-first context for a single compiler diagnostic."""
    file_path: str
    line: int
    column: int
    diagnostic_message: str
    target_symbol: Optional[str] = None
    symbol_kind: str = "method"
    target_source: str = ""
    structural_context: str = ""
    related_symbols: list[str] = field(default_factory=list)
    is_parser_fallback: bool = False
    content_hash: str = ""

    def to_prompt_block(self) -> str:
        """Render target-first context for LLM prompt."""
        header = f"DIAGNOSTIC: {self.file_path}:{self.line}:{self.column} — {self.diagnostic_message}"
        target_hdr = f"TARGET: {self.target_symbol or '(enclosing block)'} ({self.symbol_kind})"
        
        blocks = [header, target_hdr]
        if self.target_source:
            blocks.append(f"TARGET SOURCE:\n```\n{self.target_source}\n```")
        if self.structural_context:
            blocks.append(f"STRUCTURAL CONTEXT:\n{self.structural_context}")
        if self.related_symbols:
            blocks.append(f"RELATED SYMBOLS: {', '.join(self.related_symbols[:10])}")
        if self.is_parser_fallback:
            blocks.append("(Note: AST resolution fallback used — bounded line window)")
        return "\n".join(blocks)


class DiagnosticContextBuilder:
    """Builds focused, symbol-first diagnostic contexts with hash-based caching."""

    def __init__(self):
        self._file_hashes: dict[str, str] = {}
        self._cached_contexts: dict[str, list[DiagnosticContext]] = {}

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute sha256 hash of content."""
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    def is_file_fresh(self, file_path: str, content: str) -> bool:
        """Check if file content matches current cached hash."""
        norm_key = file_path.replace("\\", "/").lower()
        current_hash = self.compute_hash(content)
        return self._file_hashes.get(norm_key) == current_hash

    def update_file_hash(self, file_path: str, content: str) -> None:
        """Record the content hash for a file."""
        norm_key = file_path.replace("\\", "/").lower()
        self._file_hashes[norm_key] = self.compute_hash(content)

    def build_context(
        self,
        file_path: str,
        content: str,
        line: int,
        column: int = 0,
        diagnostic_message: str = "",
    ) -> DiagnosticContext:
        """
        Resolve a compiler diagnostic to its containing AST symbol/method.
        
        Order of resolution:
        1. Parse AST method/symbol boundaries.
        2. Match line to containing method/function boundary.
        3. Extract full method body.
        4. Extract class outline / imports as structural context.
        5. Extract identifiers referenced within target source as related symbols.
        6. Fallback to bounded line window ONLY if AST boundary resolution fails.
        """
        if not content:
            return DiagnosticContext(
                file_path=file_path,
                line=line,
                column=column,
                diagnostic_message=diagnostic_message,
                target_source="(empty file)",
                is_parser_fallback=True,
            )

        content_hash = self.compute_hash(content)
        lines = content.splitlines()
        total_lines = len(lines)
        target_idx = max(0, line - 1)

        # 1. Attempt AST / structural boundary resolution
        boundaries: List[MethodBoundary] = _get_reliable_boundaries(content, file_path, lines)
        matched_mb: Optional[MethodBoundary] = None

        if boundaries and target_idx < total_lines:
            for mb in boundaries:
                if mb.kind != "class" and mb.start_line <= target_idx <= mb.end_line:
                    matched_mb = mb
                    break

        if matched_mb:
            # Full target method body
            start_l = max(0, matched_mb.start_line)
            end_l = min(total_lines, matched_mb.end_line + 1)
            target_source = "\n".join(lines[start_l:end_l])

            # Structural context: class declaration + other signatures
            structural_context = self._extract_structural_context(lines, boundaries, matched_mb)

            # Related symbols referenced within the target method
            related_symbols = self._extract_referenced_symbols(target_source)

            return DiagnosticContext(
                file_path=file_path,
                line=line,
                column=column,
                diagnostic_message=diagnostic_message,
                target_symbol=matched_mb.name,
                symbol_kind=matched_mb.kind or "method",
                target_source=target_source,
                structural_context=structural_context,
                related_symbols=related_symbols,
                is_parser_fallback=False,
                content_hash=content_hash,
            )

        # 2. Fallback: bounded line window (only when AST boundary resolution fails)
        logger.debug(
            f"AST boundary resolution failed for {file_path}:{line}. "
            f"Using bounded line window fallback."
        )
        window_start = max(0, target_idx - 15)
        window_end = min(total_lines, target_idx + 16)
        fallback_source = "\n".join(
            f"{i+1:4d} | {lines[i]}" for i in range(window_start, window_end)
        )

        return DiagnosticContext(
            file_path=file_path,
            line=line,
            column=column,
            diagnostic_message=diagnostic_message,
            target_symbol=None,
            symbol_kind="fallback_window",
            target_source=fallback_source,
            structural_context="",
            related_symbols=[],
            is_parser_fallback=True,
            content_hash=content_hash,
        )

    def _extract_structural_context(
        self,
        lines: List[str],
        boundaries: List[MethodBoundary],
        target_mb: MethodBoundary,
    ) -> str:
        """Extract enclosing class header and outline of sibling methods."""
        parts: List[str] = []

        # Find enclosing class declaration if present
        for b in reversed(boundaries):
            if b.kind == "class" and b.start_line <= target_mb.start_line:
                class_line = lines[b.start_line].strip()
                parts.append(f"CLASS: {class_line}")
                break

        # Sibling signatures outline
        sibling_sigs = [
            f"  - {b.name}(...) [lines {b.start_line+1}-{b.end_line+1}]"
            for b in boundaries
            if b.kind != "class" and b.name != target_mb.name
        ]
        if sibling_sigs:
            parts.append("SIBLING METHODS OUTLINE:")
            parts.extend(sibling_sigs[:15])

        return "\n".join(parts)

    def _extract_referenced_symbols(self, target_source: str) -> list[str]:
        """Extract identifiers invoked or accessed in the target method body."""
        member_calls = re.findall(r'\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(', target_source)
        prop_accesses = re.findall(r'\.([a-zA-Z_$][a-zA-Z0-9_$]*)', target_source)
        
        seen = set()
        result = []
        for sym in member_calls + prop_accesses:
            if sym not in seen and len(sym) > 2 and sym not in ("get", "set", "then", "catch", "subscribe", "map"):
                seen.add(sym)
                result.append(sym)
        return result
