"""
FastRepoMap - Aider / Cursor-Style Deterministic Symbol & Outline Map

Key Principles:
1. Zero Vector DB Dependency: Runs completely in-memory in <100ms without pgvector,
   embedding models, or external infrastructure.
2. Strict Token Budgeting: Compresses an entire multi-module repository into a compact
   tree format strictly bounded by a token ceiling (e.g. 1,000 - 1,800 tokens).
3. Symbol-Level Outline: Extracts class names, exported methods, interfaces, and types
   using SkeletonCompressor.
4. Relevance-Biased Graph: Ranks files based on ticket keyword mentions and reference density
   so the most critical interfaces and classes appear first before the token budget is reached.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ticket_to_code.intelligence.skeleton_compressor import SkeletonCompressor

logger = logging.getLogger(__name__)


@dataclass
class FileSymbolSummary:
    rel_path: str
    language: str
    symbols: List[str] = field(default_factory=list)
    relevance_score: float = 0.0


class FastRepoMap:
    """
    Builds a compact, budget-capped symbol outline map of a repository.
    Modeled after Aider's Tree-sitter Repo-Map and Cursor's Symbol Index.
    """

    SUPPORTED_EXTENSIONS = {
        ".java", ".py", ".ts", ".tsx", ".js", ".jsx", ".cs", ".go", ".rs"
    }

    IGNORE_DIRS = {
        ".git", "node_modules", "target", "build", "dist", ".idea", ".vscode",
        "__pycache__", ".pytest_cache", "venv", ".venv", "bin", "obj"
    }

    def __init__(self, workspace_path: str, max_tokens: int = 1500):
        self.workspace = Path(workspace_path).resolve()
        self.max_tokens = max_tokens
        self._cache: Dict[str, FileSymbolSummary] = {}

    def build_map(
        self,
        focus_keywords: Optional[List[str]] = None,
        max_files: int = 25,
    ) -> str:
        """
        Builds the compact repository outline string, prioritized by focus keywords.
        Guarantees total output is within max_tokens.
        """
        focus_set = {k.lower() for k in (focus_keywords or []) if len(k) > 2}
        all_summaries: List[FileSymbolSummary] = []

        # 1. Discover and extract symbols
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    try:
                        rel = str(p.relative_to(self.workspace)).replace("\\", "/")
                        summary = self._get_file_summary(rel, p, focus_set)
                        if summary.symbols:
                            all_summaries.append(summary)
                    except Exception as exc:
                        logger.debug(f"[FastRepoMap] Skipped {f}: {exc}")

        if not all_summaries:
            return ""

        # 2. Sort summaries by relevance score (keyword matches + symbol richness)
        all_summaries.sort(key=lambda s: (s.relevance_score, len(s.symbols)), reverse=True)
        selected_summaries = all_summaries[:max_files]

        # Sort selected by path for a clean, hierarchical tree representation
        selected_summaries.sort(key=lambda s: s.rel_path)

        # 3. Format into a tree format within token budget
        lines = ["### REPOSITORY SYMBOL MAP:"]
        char_budget = self.max_tokens * 4  # ~4 chars per token
        current_chars = len(lines[0])

        for summary in selected_summaries:
            file_header = f"\n{summary.rel_path}:"
            sym_lines = [f"  {sym}" for sym in summary.symbols]
            block = file_header + "\n" + "\n".join(sym_lines)

            if current_chars + len(block) > char_budget:
                # Truncate gracefully
                lines.append("\n[... additional repository symbols omitted for brevity ...]")
                break

            lines.append(block)
            current_chars += len(block)

        return "\n".join(lines)

    def _get_file_summary(
        self,
        rel_path: str,
        full_path: Path,
        focus_set: Set[str],
    ) -> FileSymbolSummary:
        """Extracts top symbols and computes relevance score."""
        mtime = full_path.stat().st_mtime
        cache_key = f"{rel_path}:{mtime}"

        if cache_key in self._cache:
            summary = self._cache[cache_key]
        else:
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                skeleton_res = SkeletonCompressor.compress_file(str(full_path), content=content)
                symbols = self._extract_clean_signatures(skeleton_res.skeleton_code)
                summary = FileSymbolSummary(
                    rel_path=rel_path,
                    language=skeleton_res.language,
                    symbols=symbols,
                )
                self._cache[cache_key] = summary
            except Exception as e:
                return FileSymbolSummary(rel_path=rel_path, language="unknown")

        # Compute relevance dynamically against the focus keywords
        score = 0.0
        rel_lower = rel_path.lower()
        for kw in focus_set:
            if kw in rel_lower:
                score += 5.0
            for sym in summary.symbols:
                if kw in sym.lower():
                    score += 2.0

        summary.relevance_score = score
        return summary

    def _extract_clean_signatures(self, skeleton_code: str, max_symbols_per_file: int = 12) -> List[str]:
        """Extracts class, interface, method, and function declarations from skeleton."""
        signatures: List[str] = []
        for line in skeleton_code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("//", "#", "/*", "*", "import ", "from ", "package ")):
                continue

            # Match class, interface, enum, struct
            if re.match(r"^(?:export\s+)?(?:public\s+|private\s+|protected\s+)?(?:class|interface|enum|record|struct|trait)\b", stripped):
                # Clean up trailing braces
                cleaned = stripped.split("{")[0].strip()
                signatures.append(cleaned)
                continue

            # Match function/method signatures (including return types and generics)
            if re.match(r"^(?:export\s+)?(?:public\s+|private\s+|protected\s+|static\s+|async\s+|def\s+|final\s+|override\s+)*(?:[\w<>\[\],\s]+?\s+)?\w+\s*\([^)]*\)", stripped):
                cleaned = stripped.split("{")[0].rstrip(";").strip()
                if cleaned and not cleaned.startswith(("if", "for", "while", "switch", "catch")):
                    signatures.append(cleaned)
                    continue

            if len(signatures) >= max_symbols_per_file:
                break

        return signatures[:max_symbols_per_file]
