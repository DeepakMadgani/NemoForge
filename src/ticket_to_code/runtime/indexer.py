"""
Standalone Code Indexer for Ticket-to-Code.

Replaces aviator_core.indexer (index_file, index_repository) with
tree-sitter-based implementations that have zero dependency on aviator_core.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Language extensions → tree-sitter language name mapping
_LANG_MAP = {
    ".py": "python", ".java": "java", ".kt": "kotlin",
    ".ts": "typescript", ".tsx": "typescript", ".js": "javascript",
    ".jsx": "javascript", ".cs": "c_sharp", ".go": "go",
    ".rb": "ruby", ".php": "php", ".rs": "rust",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
    ".swift": "swift", ".scala": "scala",
    ".vue": "vue", ".svelte": "svelte",
}

# Directories to skip during indexing
_SKIP_DIRS = {
    "node_modules", ".git", ".svn", "bin", "obj", "build", "dist",
    "target", "__pycache__", ".idea", ".vs", ".vscode", ".aviator",
    "packages", "vendor", ".gradle", ".mvn",
}

# Max file size to index (500KB)
_MAX_FILE_BYTES = 512_000


def _detect_language(file_path: Path) -> str:
    """Map file extension to language name."""
    return _LANG_MAP.get(file_path.suffix.lower(), "")


def _file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_symbols_basic(file_path: Path, content: str, language: str) -> List[Dict[str, Any]]:
    """
    Basic symbol extraction using line-by-line heuristics.
    
    Works for Java, C#, Python, TypeScript, JavaScript etc. without
    requiring tree-sitter grammar installation for every language.
    """
    symbols: List[Dict[str, Any]] = []
    lines = content.splitlines()
    file_str = str(file_path)
    
    import re
    
    # Java/C#/TypeScript class pattern
    class_pat = re.compile(
        r'^\s*(?:public|private|protected|internal|abstract|static|final|sealed|export|declare)?\s*'
        r'(?:class|interface|enum|record|struct|abstract\s+class)\s+(\w+)',
        re.IGNORECASE,
    )
    # Method/function patterns
    method_pat_java = re.compile(
        r'^\s*(?:public|private|protected|static|final|override|virtual|async|abstract)?\s*'
        r'(?:[\w<>\[\],\s]+)\s+(\w+)\s*\(',
    )
    method_pat_py = re.compile(r'^\s*(?:async\s+)?def\s+(\w+)\s*\(')
    method_pat_ts = re.compile(
        r'^\s*(?:export\s+)?(?:async\s+)?(?:function\s+)?(\w+)\s*\('
    )
    
    current_class = None
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue
        
        # Check for class/interface/enum
        m = class_pat.search(stripped)
        if m:
            name = m.group(1)
            current_class = name
            qn = f"{file_str}::{name}"
            symbols.append({
                "id": qn,
                "name": name,
                "kind": "class",
                "qualified_name": qn,
                "file_path": file_str,
                "start_line": i,
                "end_line": i,
                "signature": stripped[:200],
            })
            continue
        
        # Check for methods/functions
        m = None
        if language == "python":
            m = method_pat_py.search(stripped)
        elif language in ("java", "c_sharp", "kotlin"):
            m = method_pat_java.search(stripped)
        else:
            m = method_pat_ts.search(stripped)
        
        if m:
            name = m.group(1)
            # Skip common false positives
            if name in ("if", "for", "while", "switch", "catch", "return", "new", "import", "from", "else"):
                continue
            parent = current_class or ""
            qn = f"{file_str}::{parent}.{name}" if parent else f"{file_str}::{name}"
            symbols.append({
                "id": qn,
                "name": name,
                "kind": "method" if current_class else "function",
                "qualified_name": qn,
                "file_path": file_str,
                "start_line": i,
                "end_line": i,
                "signature": stripped[:200],
                "parent": parent,
            })
    
    return symbols


def index_file(
    file_path: Path | str,
    sqlite_store,
    neo4j_store=None,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Index a single source file — extract symbols and store them.
    
    Drop-in replacement for aviator_core.indexer.index_file.
    
    Args:
        file_path: Path to the source file.
        sqlite_store: SqliteStore instance for persisting symbols.
        neo4j_store: Optional Neo4jStore for graph relationships.
        
    Returns:
        Dict with indexing stats, or None if the file was skipped.
    """
    file_path = Path(file_path)
    if isinstance(sqlite_store, (str, Path)):
        from ticket_to_code.runtime.storage import SqliteStore
        sqlite_store = SqliteStore(sqlite_store)
    
    if not file_path.exists() or not file_path.is_file():
        return None
    
    if file_path.stat().st_size > _MAX_FILE_BYTES:
        logger.debug("Skipping large file: %s", file_path)
        return None
    
    language = _detect_language(file_path)
    if not language:
        return None
    
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return None
    
    sha = _file_sha256(file_path)
    
    # Upsert file record
    sqlite_store.upsert_file({
        "path": str(file_path),
        "language": language,
        "package": "",
        "sha256": sha,
        "size_bytes": file_path.stat().st_size,
    })
    
    # Extract and store symbols
    symbols = _extract_symbols_basic(file_path, content, language)
    for sym in symbols:
        sqlite_store.upsert_symbol(sym)
    
    # Optionally push to Neo4j
    if neo4j_store:
        try:
            neo4j_store.upsert_file(type("FR", (), {
                "path": str(file_path), "language": language,
                "package": "", "sha256": sha,
                "size_bytes": file_path.stat().st_size,
            })())
            for sym in symbols:
                neo4j_store.upsert_symbol(sym)
        except Exception as exc:
            logger.debug("Neo4j indexing skipped for %s: %s", file_path, exc)
    
    return {"symbols": len(symbols), "file": str(file_path)}


def index_repository(
    repo_path: str | Path,
    sqlite_store,
    neo4j_store=None,
    progress_callback: Optional[Callable] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Index an entire repository — walk all code files and extract symbols.
    
    Drop-in replacement for aviator_core.indexer.index_repository.
    
    Args:
        repo_path: Root path of the repository.
        sqlite_store: SqliteStore for persisting data.
        neo4j_store: Optional Neo4jStore.
        progress_callback: Optional callback(current_file, done, total).
        
    Returns:
        Dict with aggregate indexing statistics.
    """
    repo_path = Path(repo_path)
    owned_store = False
    if isinstance(sqlite_store, (str, Path)):
        from ticket_to_code.runtime.storage import SqliteStore
        sqlite_store = SqliteStore(sqlite_store)
        owned_store = True
    
    # Collect indexable files
    code_files: List[Path] = []
    for root, dirs, files in os.walk(repo_path):
        # Prune skip directories
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            fpath = Path(root) / fname
            if _detect_language(fpath):
                code_files.append(fpath)
    
    total = len(code_files)
    stats = {"files_indexed": 0, "symbols_found": 0, "errors": [], "total_files": total}
    
    for i, fpath in enumerate(code_files, 1):
        try:
            result = index_file(fpath, sqlite_store, neo4j_store)
            if result:
                stats["files_indexed"] += 1
                stats["symbols_found"] += result.get("symbols", 0)
            
            if progress_callback:
                progress_callback(fpath, i, total)
        except Exception as exc:
            stats["errors"].append(f"{fpath}: {exc}")
            logger.warning("Index error on %s: %s", fpath, exc)
    
    logger.info(
        "Repository indexing complete: %d/%d files, %d symbols",
        stats["files_indexed"], total, stats["symbols_found"],
    )
    if owned_store and hasattr(sqlite_store, "close"):
        try:
            sqlite_store.close()
        except Exception:
            pass
    return stats
