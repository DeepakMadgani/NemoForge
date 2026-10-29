"""
Centralized file and directory filtering for workspace indexing and discovery.

Eliminates duplicate _SKIP_DIRS definitions across repository search,
evidence collection, companion resolver, and brain generators.
Hard-excludes build artifacts, minified webpack/angular chunks, and gitignored files.
"""

from __future__ import annotations

import os
import re
import fnmatch
from pathlib import Path
from typing import Optional, Set, Iterable

# Standard directories that should NEVER be indexed, walked, or edited
STANDARD_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", ".idea", ".vscode",
    "target", "build", "out", "bin", "dist", "dist-server",
    "node_modules", ".gradle", ".mvn",
    "generated", "generated-sources", "generated-test-sources",
    "__pycache__", ".angular", ".aviator", ".ticket_to_code",
    "coverage", "logs", "log", ".nyc_output",
    "postman", "traces", "trace", "benchmark_reports_wrapper",
    "public", "wwwroot", ".next", ".nuxt", "bundle",
    "xchange-ui-fixture",
})

# Patterns that mark a file as a compiled or minified build artifact
_MINIFIED_OR_CHUNK_RE = re.compile(
    r"(?:\.[a-f0-9]{8,}\.(?:js|css)$"          # Webpack/Angular chunks e.g. 105.88261251d83cecc2.js
    r"|\.(?:min|bundle|chunk)\.(?:js|css)$"     # .min.js, .bundle.js, .chunk.js
    r"|\.map$"                                  # Source maps
    r"|^(?:runtime|polyfills|main|vendor)\.[a-f0-9]{8,}\.js$)",  # Angular bundle roots
    re.IGNORECASE
)

# Relative directory prefixes that contain compiled production web assets
_COMPILED_PREFIXES = (
    "server/public/",
    "server/dist/",
    "client/dist/",
    "client/build/",
    "public/bundles/",
    "static/bundles/",
    "wwwroot/",
)

# Cache for loaded .gitignore rules per workspace: workspace_str -> list[patterns]
_GITIGNORE_CACHE: dict[str, list[str]] = {}


def _get_gitignore_patterns(workspace_path: Optional[Path | str]) -> list[str]:
    """Read and cache .gitignore patterns from workspace root."""
    if not workspace_path:
        return []
    ws_str = str(Path(workspace_path).resolve())
    if ws_str in _GITIGNORE_CACHE:
        return _GITIGNORE_CACHE[ws_str]

    patterns: list[str] = []
    gi_path = Path(ws_str) / ".gitignore"
    if gi_path.exists():
        try:
            for line in gi_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        except Exception:
            pass
    _GITIGNORE_CACHE[ws_str] = patterns
    return patterns


def is_ignored_path(
    path: Path | str,
    workspace_path: Optional[Path | str] = None,
    check_gitignore: bool = True
) -> bool:
    """
    Check if a file or directory path should be ignored.

    Returns True if the path:
    1. Belongs to a skip directory (e.g. node_modules, public, dist).
    2. Matches a compiled chunk or minified file pattern (e.g. 105.88261251d83cecc2.js).
    3. Resides under known compiled asset paths (e.g. server/public/...).
    4. Matches a pattern in the workspace's .gitignore (if check_gitignore=True).
    """
    path_str = str(path).replace("\\", "/")
    
    # Check relative path if workspace_path is given
    rel_str = path_str
    if workspace_path:
        try:
            rel_str = str(Path(path).relative_to(workspace_path)).replace("\\", "/")
        except Exception:
            pass

    rel_lower = rel_str.lower()
    parts = [p.lower() for p in Path(rel_str).parts]

    # 1. Any directory part in STANDARD_SKIP_DIRS or ending with -fixture
    if any(p in STANDARD_SKIP_DIRS or p.endswith("-fixture") or p.endswith("-fixtures") for p in parts[:-1]):
        return True

    # If the item itself is a directory in STANDARD_SKIP_DIRS or ending with -fixture
    if parts and (parts[-1] in STANDARD_SKIP_DIRS or parts[-1].endswith("-fixture") or parts[-1].endswith("-fixtures")):
        return True

    # 2. Known compiled output prefixes
    if any(rel_lower.startswith(prefix) or f"/{prefix}" in rel_lower for prefix in _COMPILED_PREFIXES):
        return True

    # 3. Filename check for lockfiles, hidden config, and minified/chunk patterns
    fname = parts[-1] if parts else ""
    fname_lower = fname.lower()
    LOCKFILES = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json",
        "pipfile.lock", "poetry.lock", "composer.lock", "cargo.lock", "packages.lock.json",
    }
    if fname_lower in LOCKFILES or fname.startswith("."):
        return True
    if fname and _MINIFIED_OR_CHUNK_RE.search(fname):
        return True

    # 4. Check .gitignore patterns
    if check_gitignore and workspace_path:
        gi_patterns = _get_gitignore_patterns(workspace_path)
        for pattern in gi_patterns:
            clean_pat = pattern.rstrip("/")
            if fnmatch.fnmatch(rel_str, clean_pat) or fnmatch.fnmatch(fname, clean_pat):
                return True
            if pattern.endswith("/") and any(p == clean_pat for p in parts[:-1]):
                return True

    return False


def prune_walk_dirs(dirs: list[str], current_root: Optional[Path | str] = None, workspace_path: Optional[Path | str] = None) -> None:
    """
    In-place prune of directory list for os.walk.
    Example:
        for root, dirs, files in os.walk(workspace):
            prune_walk_dirs(dirs, root, workspace)
    """
    dirs[:] = [
        d for d in dirs
        if d not in STANDARD_SKIP_DIRS
        and not d.endswith("-fixture")
        and not d.endswith("-fixtures")
        and not d.startswith(".")
        and not (current_root and is_ignored_path(Path(current_root) / d, workspace_path=workspace_path, check_gitignore=False))
    ]
