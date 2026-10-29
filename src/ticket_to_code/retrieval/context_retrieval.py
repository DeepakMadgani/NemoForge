"""
ContextRetrievalService — cascading multi-source context assembly for generation (B2).

Architecture RAG is a single point of failure when pgvector is empty or misconfigured.
This service guarantees non-empty context by cascading through five strategies:

  1. RAG semantic search     (pgvector)
  2. Neo4j graph search      (dependency neighbours of writable files)
  3. SQLite symbol search    (FTS / LIKE on writable file paths + plan keywords)
  4. Filesystem grep         (pattern search on writable files)
  5. DirectRead              (read writable files directly, byte-capped)

Each strategy fires if the previous one returned fewer than MIN_CHUNKS results.
Strategies are composited (not exclusive): all results found before the
threshold is met are returned together.

Author: Deepak Madgani
Date: July 2026
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from ticket_to_code.retrieval.rag_engine import CodebaseRAGEngine
    from ticket_to_code.models import ArchitecturalPlan, CodeChunk, StructuredRequirements
    from ticket_to_code.runtime.run_context import RunContext

logger = logging.getLogger(__name__)

# Minimum number of chunks before we try the next strategy
_MIN_CHUNKS = 5
# Maximum bytes to read from a single file in DirectRead
_MAX_DIRECTREAD_BYTES = 16_000
# Maximum total bytes to read across all DirectRead files
_MAX_DIRECTREAD_TOTAL_BYTES = 64_000




class ContextRetrievalService:
    """
    Cascading context retrieval for code generation (B2).

    Usage::

        svc = ContextRetrievalService(
            rag_engine=agents.rag_engine,
            neo4j_store=agents.localizer.neo4j_store,
            sqlite_store=agents.localizer.sqlite_store,
            workspace_path=workspace_path,
        )
        chunks, sources = svc.retrieve(
            requirements=state["requirements"],
            plan=plan,
            run_ctx=run_ctx,
        )
    """

    def __init__(
        self,
        rag_engine: Optional["CodebaseRAGEngine"] = None,
        neo4j_store=None,
        sqlite_store=None,
        workspace_path: str = ".",
    ) -> None:
        self.rag_engine    = rag_engine
        self.neo4j_store   = neo4j_store
        self.sqlite_store  = sqlite_store
        self.workspace_path = Path(workspace_path).resolve()

    # ── Public entry point ─────────────────────────────────────────────────────

    def retrieve(
        self,
        requirements: Optional["StructuredRequirements"] = None,
        plan: Optional["ArchitecturalPlan"] = None,
        run_ctx: Optional["RunContext"] = None,
        extra_query: str = "",
    ) -> tuple[list, list[str]]:
        """
        Assemble architecture context for code generation.

        Returns:
            (chunks, sources_used) where chunks is a list of raw context objects
            (either CodeChunk models or plain dicts) and sources_used is a list
            of source names in the order they fired.

        Never raises; logs degradation via run_ctx when provided.
        """
        writable_files: List[str] = []
        if plan:
            writable_files = [
                t.file_path for t in (plan.tasks or [])
                if getattr(t, "task_type", None) and t.task_type.value != "read_only"
            ]

        # Build search query from requirements + plan
        query = self._build_query(requirements, plan, extra_query)

        all_chunks: list = []
        sources_used: List[str] = []

        # ── Strategy 1: DirectRead of Writable Files (MANDATORY GROUND TRUTH) ────
        if writable_files:
            try:
                direct_chunks = self._strategy_direct_read(writable_files)
                if direct_chunks:
                    all_chunks.extend(direct_chunks)
                    sources_used.append("DirectRead")
                    logger.info(
                        "ContextRetrieval: DirectRead loaded %d primary target files",
                        len(direct_chunks),
                    )
            except Exception as exc:
                logger.warning("ContextRetrieval: DirectRead failed (%s)", exc)

        # ── Strategy 2: Target-File Tree-Sitter AST Slicing (ON-DEMAND AST) ──────
        if writable_files:
            try:
                ast_chunks = self._strategy_treesitter_slicing(writable_files)
                if ast_chunks:
                    all_chunks.extend(ast_chunks)
                    sources_used.append("TreeSitterAST")
                    logger.info(
                        "ContextRetrieval: TreeSitter AST extracted structure for %d files",
                        len(ast_chunks),
                    )
            except Exception as exc:
                logger.warning("ContextRetrieval: TreeSitter AST slicing failed (%s)", exc)

        # ── Strategy 3: Deterministic Dependency / Import Chasing (NO NEO4J) ─────
        if writable_files:
            try:
                dep_chunks = self._strategy_imported_dependencies(writable_files)
                if dep_chunks:
                    all_chunks.extend(dep_chunks)
                    sources_used.append("ImportedDependencies")
                    logger.info(
                        "ContextRetrieval: Resolved %d imported dependency interfaces",
                        len(dep_chunks),
                    )
            except Exception as exc:
                logger.warning("ContextRetrieval: Dependency resolution failed (%s)", exc)

        # ── Strategy 4: SQLite symbol & dependency search ─────────────────────
        if self.sqlite_store:
            try:
                sqlite_chunks = self._strategy_sqlite(query, writable_files)
                if sqlite_chunks:
                    all_chunks.extend(sqlite_chunks)
                    sources_used.append("SQLite")
                    logger.info(
                        "ContextRetrieval: SQLite returned %d context items",
                        len(sqlite_chunks),
                    )
            except Exception as exc:
                logger.warning("ContextRetrieval: SQLite failed (%s)", exc)

        if len(all_chunks) >= _MIN_CHUNKS:
            return self._finalise(all_chunks, sources_used, run_ctx)

        # ── Strategy 5: Targeted Grep ─────────────────────────────────────────
        if writable_files:
            try:
                grep_chunks = self._strategy_grep(query, writable_files)
                if grep_chunks:
                    all_chunks.extend(grep_chunks)
                    sources_used.append("Grep")
                    logger.info(
                        "ContextRetrieval: Grep returned %d context items",
                        len(grep_chunks),
                    )
            except Exception as exc:
                logger.warning("ContextRetrieval: Grep failed (%s)", exc)

        if not all_chunks:
            logger.warning(
                "ContextRetrieval: Primary strategies returned 0 context items. "
                "Generator will operate with target file direct context."
            )

        return self._finalise(all_chunks, sources_used, run_ctx)

    # ── Private strategy implementations ──────────────────────────────────────

    def _strategy_treesitter_slicing(self, writable_files: List[str]) -> List[dict]:
        """
        On-demand Tree-sitter AST slicing on the target writable files.
        Extracts method boundaries, class signatures, and imports in milliseconds
        without touching an external graph database.
        """
        results: List[dict] = []
        for rel_path in writable_files:
            full = self.workspace_path / rel_path
            if not full.is_file():
                continue
            ext = full.suffix.lower()
            if ext not in (".java", ".ts", ".js", ".py"):
                continue
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
                boundaries_summary = []

                if ext == ".java":
                    try:
                        from ticket_to_code.agents.smart_extract import extract_method_boundaries_java_tree_sitter
                        boundaries = extract_method_boundaries_java_tree_sitter(content)
                        for b in boundaries:
                            boundaries_summary.append(f"- Line {b.start_line + 1}-{b.end_line + 1}: {b.kind} {b.name} -> {b.signature}")
                    except Exception as e:
                        logger.debug("Tree-sitter Java extraction fallback: %s", e)

                elif ext in (".ts", ".js"):
                    try:
                        from ticket_to_code.agents.smart_extract import extract_method_boundaries_typescript_tree_sitter
                        boundaries = extract_method_boundaries_typescript_tree_sitter(content)
                        for b in boundaries:
                            boundaries_summary.append(f"- Line {b.start_line + 1}-{b.end_line + 1}: {b.kind} {b.name} -> {b.signature}")
                    except Exception as e:
                        logger.debug("Tree-sitter TS extraction fallback: %s", e)

                if boundaries_summary:
                    results.append({
                        "file_path": rel_path,
                        "content": f"=== AST Skeleton for {rel_path} ===\n" + "\n".join(boundaries_summary[:60]),
                        "source": "TreeSitterAST",
                        "type": "ast_skeleton",
                    })
            except Exception as exc:
                logger.debug("TreeSitter AST slicing error on %s: %s", rel_path, exc)
        return results

    def _strategy_imported_dependencies(self, writable_files: List[str]) -> List[dict]:
        """
        Deterministic import & dependency resolution:
        Inspects import statements of writable files and retrieves the API surfaces
        of imported interfaces, repositories, and models. Zero Neo4j required.
        """
        results: List[dict] = []
        seen_deps: Set[str] = set()

        for rel_path in writable_files:
            full = self.workspace_path / rel_path
            if not full.is_file():
                continue
            ext = full.suffix.lower()
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
                imported_names: List[str] = []

                if ext == ".java":
                    matches = re.findall(r"^import\s+(?:static\s+)?([a-zA-Z0-9_.]+);", content, re.MULTILINE)
                    for m in matches:
                        class_name = m.split(".")[-1]
                        if class_name and class_name not in seen_deps and not class_name.startswith("java."):
                            imported_names.append(class_name)
                            seen_deps.add(class_name)

                elif ext in (".ts", ".js"):
                    matches = re.findall(r"import\s+\{([^}]+)\}\s+from", content)
                    for m in matches:
                        for item in m.split(","):
                            clean_item = item.strip().split(" as ")[0].strip()
                            if clean_item and clean_item not in seen_deps:
                                imported_names.append(clean_item)
                                seen_deps.add(clean_item)

                for name in imported_names[:12]:
                    dep_chunk = self._resolve_symbol_skeleton(name)
                    if dep_chunk:
                        results.append(dep_chunk)
                        if len(results) >= 8:
                            break
            except Exception as exc:
                logger.debug("Dependency resolution error for %s: %s", rel_path, exc)

        return results

    def _resolve_symbol_skeleton(self, symbol_name: str) -> Optional[dict]:
        """Find the definition file of an imported symbol in SQLite and extract its API skeleton."""
        if not self.sqlite_store:
            return None
        conn = getattr(self.sqlite_store, "_conn", None)
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT DISTINCT path FROM symbols WHERE name = ? LIMIT 1",
                (symbol_name,)
            ).fetchone()
            if row:
                fpath = row["path"] if isinstance(row, dict) or hasattr(row, "__getitem__") else row[0]
                target = self.workspace_path / fpath
                if target.is_file():
                    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                    snippet = "\n".join(lines[:45])
                    return {
                        "file_path": fpath,
                        "content": f"=== Interface/Dependency: {symbol_name} ===\n{snippet}",
                        "source": "ImportedDependencies",
                        "type": "dependency_interface",
                    }
        except Exception:
            pass
        return None

    def _strategy_sqlite(self, query: str, writable_files: List[str]) -> List[dict]:
        """Search SQLite symbol index for query terms, scoped to writable file paths."""
        if not self.sqlite_store:
            return []
        keywords = self._query_to_keywords(query)
        if not keywords:
            return []
        conn = getattr(self.sqlite_store, "_conn", None)
        if conn is None:
            return []
        results: List[dict] = []
        seen: Set[str] = set()
        try:
            for kw in keywords[:6]:
                rows = conn.execute(
                    "SELECT DISTINCT path, qualified_name FROM symbols "
                    "WHERE (LOWER(name) LIKE ? OR LOWER(qualified_name) LIKE ?) LIMIT 20",
                    (f"%{kw.lower()}%", f"%{kw.lower()}%"),
                ).fetchall()
                for path, qname in rows:
                    if path and path not in seen:
                        results.append({
                            "file_path": path,
                            "content": f"Symbol: {qname or path}",
                            "source": "SQLite",
                            "type": "symbol_match",
                        })
                        seen.add(path)
        except Exception as exc:
            logger.debug("_strategy_sqlite error: %s", exc)
        return results

    def _strategy_grep(self, query: str, writable_files: List[str]) -> List[dict]:
        """Grep writable files for query keywords."""
        keywords = self._query_to_keywords(query)[:4]
        if not keywords:
            return []
        results: List[dict] = []
        patterns = [re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords]
        for rel_path in writable_files[:10]:
            full = self.workspace_path / rel_path
            if not full.is_file():
                continue
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
                matching_lines = [
                    ln for ln in content.splitlines()
                    if any(p.search(ln) for p in patterns)
                ]
                if matching_lines:
                    results.append({
                        "file_path": rel_path,
                        "content": "\n".join(matching_lines[:30]),
                        "source": "Grep",
                        "type": "keyword_match",
                    })
            except OSError:
                pass
        return results

    def _strategy_direct_read(self, writable_files: List[str]) -> List[dict]:
        """Read writable files directly, byte-capped per file and total."""
        results: List[dict] = []
        total_bytes = 0
        for rel_path in writable_files:
            if total_bytes >= _MAX_DIRECTREAD_TOTAL_BYTES:
                logger.debug(
                    "DirectRead: total byte cap reached (%d bytes)", total_bytes
                )
                break
            full = self.workspace_path / rel_path
            if not full.is_file():
                continue
            try:
                raw = full.read_bytes()
                text = raw[:_MAX_DIRECTREAD_BYTES].decode("utf-8", errors="replace")
                results.append({
                    "file_path": rel_path,
                    "content": text,
                    "source": "DirectRead",
                    "type": "file_content",
                })
                total_bytes += len(raw[:_MAX_DIRECTREAD_BYTES])
            except OSError as exc:
                logger.debug("DirectRead: cannot read %s: %s", rel_path, exc)
        return results

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_query(
        self,
        requirements: Optional["StructuredRequirements"],
        plan: Optional["ArchitecturalPlan"],
        extra: str,
    ) -> str:
        parts: List[str] = []
        if requirements:
            parts.append(getattr(requirements, "raw_text", "") or "")
            parts.extend(getattr(requirements, "implementation_hints", []) or [])
        if plan and hasattr(plan, "tasks"):
            for t in (plan.tasks or [])[:5]:
                desc = getattr(t, "description", "") or ""
                if desc:
                    parts.append(desc)
        if extra:
            parts.append(extra)
        return " ".join(p for p in parts if p)[:500]

    @staticmethod
    def _query_to_keywords(query: str) -> List[str]:
        """Extract meaningful keywords from a query string."""
        stopwords = {
            "the", "a", "an", "is", "in", "on", "at", "to", "for",
            "and", "or", "of", "with", "by", "from", "should", "must",
            "that", "this", "when", "where", "how", "what",
        }
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", query)
        seen: Set[str] = set()
        result: List[str] = []
        for t in tokens:
            low = t.lower()
            if len(low) >= 4 and low not in stopwords and low not in seen:
                result.append(t)
                seen.add(low)
            if len(result) >= 12:
                break
        return result

    @staticmethod
    def _finalise(
        chunks: list,
        sources_used: List[str],
        run_ctx: Optional["RunContext"],
    ) -> tuple[list, list[str]]:
        if run_ctx is not None:
            run_ctx.architecture_chunks  = len(chunks)
            run_ctx.context_sources_used = list(sources_used)
        return chunks, sources_used
