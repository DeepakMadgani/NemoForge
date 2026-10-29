"""
Standalone Localizer for Ticket-to-Code Chatbot.

Replaces aviator_core.localizer.HybridLocalizer with a lightweight
file-search + SQLite symbol-query implementation.

Zero dependency on aviator_core.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FileCandidate:
    """A file identified as relevant to a task."""
    path: str = ""
    relevance_score: float = 0.0
    method_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    reason: str = ""
    selected: bool = False


@dataclass
class SpringContext:
    """Spring Boot application context metadata."""
    base_package: str = ""
    controllers: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    repositories: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)


class SpringFilter:
    """
    Standalone Spring Boot architecture filter.
    
    Identifies Spring stereotypes (@Controller, @Service, @Repository, etc.)
    in Java/Kotlin files for smarter localization.
    """

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)
        self._context: Optional[SpringContext] = None

    def get_context(self) -> SpringContext:
        """Scan repository and build Spring context."""
        if self._context:
            return self._context

        ctx = SpringContext()
        stereotype_patterns = {
            "controllers": re.compile(r'@(?:Rest)?Controller'),
            "services": re.compile(r'@Service'),
            "repositories": re.compile(r'@Repository'),
            "entities": re.compile(r'@Entity'),
        }

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in {
                "node_modules", ".git", "build", "target", "bin", "obj",
            }]
            for fname in files:
                if fname.endswith((".java", ".kt")):
                    fpath = Path(root) / fname
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                        for key, pattern in stereotype_patterns.items():
                            if pattern.search(content):
                                getattr(ctx, key).append(str(fpath.relative_to(self.repo_path)))
                    except Exception:
                        pass

        self._context = ctx
        return ctx

    def filter_candidates(
        self, candidates: List[FileCandidate], task: str
    ) -> List[FileCandidate]:
        """Boost or demote candidates based on Spring stereotypes."""
        ctx = self.get_context()
        task_lower = task.lower()

        for c in candidates:
            path_str = c.path
            if "controller" in task_lower and path_str in ctx.controllers:
                c.relevance_score *= 1.3
            elif "service" in task_lower and path_str in ctx.services:
                c.relevance_score *= 1.3
            elif "repository" in task_lower and path_str in ctx.repositories:
                c.relevance_score *= 1.3

        return sorted(candidates, key=lambda x: x.relevance_score, reverse=True)


class HybridLocalizer:
    """
    Standalone Hybrid Localizer — combines text search and SQLite symbol
    lookup to identify files relevant to a task description.

    Drop-in replacement for aviator_core.localizer.HybridLocalizer.
    """

    def __init__(self, repo_path: str | Path, sqlite_store=None, **kwargs):
        self.repo_path = Path(repo_path)
        self._owned_store = False
        if isinstance(sqlite_store, (str, Path)):
            from ticket_to_code.runtime.storage import SqliteStore
            self.sqlite_store = SqliteStore(sqlite_store)
            self._owned_store = True
        else:
            self.sqlite_store = sqlite_store

    def close(self):
        """Close owned resources."""
        if getattr(self, "_owned_store", False) and hasattr(self.sqlite_store, "close"):
            try:
                self.sqlite_store.close()
            except Exception:
                pass

    def localize(
        self,
        task_description: str,
        top_k: int = 15,
        **kwargs,
    ) -> List[FileCandidate]:
        """
        Find files most relevant to the given task description.

        Uses keyword extraction + file name matching + optional symbol search.
        """
        keywords = self._extract_keywords(task_description)
        candidates: Dict[str, FileCandidate] = {}

        # 1. Symbol search via SQLite (if available)
        if self.sqlite_store:
            try:
                for kw in keywords[:10]:
                    symbols = self.sqlite_store.search_symbols(kw, limit=10)
                    for sym in symbols:
                        fp = sym["path"] if hasattr(sym, "__getitem__") else getattr(sym, "path", getattr(sym, "file_path", ""))
                        sym_name = sym["name"] if hasattr(sym, "__getitem__") else getattr(sym, "name", "")
                        sym_kind = sym["kind"] if hasattr(sym, "__getitem__") else getattr(sym, "kind", "")
                        start_line = sym["start_line"] if hasattr(sym, "__getitem__") else getattr(sym, "start_line", 1)
                        end_line = sym["end_line"] if hasattr(sym, "__getitem__") else getattr(sym, "end_line", 1)
                        if fp and fp not in candidates:
                            candidates[fp] = FileCandidate(
                                path=fp,
                                relevance_score=0.0,
                                method_name=sym_name if sym_kind in ("method", "function") else None,
                                start_line=start_line,
                                end_line=end_line,
                            )
                        if fp:
                            candidates[fp].relevance_score += 1.0
            except Exception as exc:
                logger.debug("Symbol search failed: %s", exc)

        # 2. File-name keyword matching
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in {
                "node_modules", ".git", "build", "target", "bin", "obj",
                "__pycache__", ".aviator", ".vs", ".idea",
            }]
            for fname in files:
                fpath = Path(root) / fname
                name_lower = fname.lower()
                score = 0.0
                for kw in keywords:
                    if kw.lower() in name_lower:
                        score += 2.0
                if score > 0:
                    fp_str = str(fpath)
                    if fp_str not in candidates:
                        candidates[fp_str] = FileCandidate(path=fp_str, relevance_score=0.0)
                    candidates[fp_str].relevance_score += score

        # 3. Grep-style content search for top keywords
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in {
                "node_modules", ".git", "build", "target", "bin", "obj",
                "__pycache__", ".aviator",
            }]
            for fname in files:
                if not fname.endswith((".java", ".py", ".ts", ".js", ".cs", ".kt", ".go", ".tsx", ".jsx")):
                    continue
                fpath = Path(root) / fname
                if fpath.stat().st_size > 200_000:
                    continue
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    content_lower = content.lower()
                    score = 0.0
                    for kw in keywords[:5]:
                        count = content_lower.count(kw.lower())
                        score += min(count, 5) * 0.5
                    if score > 0:
                        fp_str = str(fpath)
                        if fp_str not in candidates:
                            candidates[fp_str] = FileCandidate(path=fp_str, relevance_score=0.0)
                        candidates[fp_str].relevance_score += score
                except Exception:
                    pass

        # Sort by score and return top_k
        result = sorted(candidates.values(), key=lambda c: c.relevance_score, reverse=True)
        return result[:top_k]

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract meaningful keywords from a task description."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "and", "but", "or", "not", "no", "nor", "so", "yet",
            "this", "that", "these", "those", "it", "its", "i", "we", "you",
            "he", "she", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "our", "their", "what", "which", "who", "when",
            "where", "why", "how", "all", "each", "every", "both", "few",
            "more", "most", "other", "some", "such", "than", "too", "very",
            "just", "also", "add", "update", "fix", "change", "need",
            "create", "implement", "make", "file", "code",
        }
        # Split on non-word chars, filter short/stop words
        words = re.findall(r'\b\w+\b', text)
        keywords = [w for w in words if len(w) > 2 and w.lower() not in stop_words]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for w in keywords:
            low = w.lower()
            if low not in seen:
                seen.add(low)
                unique.append(w)
        return unique
