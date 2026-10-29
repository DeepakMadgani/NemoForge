"""
Canonical Path Identity & Component Family Normalization.

Provides workspace-rooted repository-relative identity for all path, ownership,
companion, and change-authorization operations.

Invariants:
1. Workspace-rooted: paths resolve against the actual workspace/repository root.
2. Containment verification: paths outside the workspace root are marked ungrounded
   and not guessed.
3. No hardcoded prefixes: never hardcode 'C:/CC4E', 'CC4E', 'xchange-ui', or any project names.
4. Slashes: POSIX forward slashes only ('/'), no duplicate separators, no '.' or '..'.
5. Component companions: strictly structural component extensions only
   (.component.ts, .component.html, .component.scss, .component.css, .component.spec.ts).
   Services, models, sagas, etc. are NOT companions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

# Strict structural component suffixes only
COMPONENT_EXTENSIONS: tuple[str, ...] = (
    ".component.ts",
    ".component.html",
    ".component.scss",
    ".component.css",
    ".component.spec.ts",
)


def canonical_repo_path(
    path: Union[str, Path],
    workspace_root: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Resolve a file path to its canonical repository-relative form.

    Args:
        path: An absolute or relative path string or Path object.
        workspace_root: The actual workspace/repository root. If None or empty,
            relies on safe relative path normalization without guessing.

    Returns:
        Canonical repository-relative path with forward slashes (e.g. 'src/app/foo.ts').
        Returns None if:
          - path is empty/invalid
          - path is an absolute path that does not reside within workspace_root
          - workspace_root cannot contain the path.
    """
    if not path:
        return None

    raw_str = str(path).strip().replace("\\", "/")
    if not raw_str:
        return None

    p = Path(raw_str)

    if workspace_root:
        try:
            ws_resolved = Path(workspace_root).resolve()
        except Exception:
            return None

        # Try resolving path against workspace_root
        try:
            if p.is_absolute():
                target_resolved = p.resolve()
            else:
                target_resolved = (ws_resolved / p).resolve()

            # Verify containment
            # On Windows, path comparison should be case-insensitive for filesystem containment
            ws_str_norm = str(ws_resolved).replace("\\", "/").rstrip("/")
            tgt_str_norm = str(target_resolved).replace("\\", "/").rstrip("/")

            is_windows = sys.platform == "win32" or os.name == "nt"
            ws_cmp = ws_str_norm.lower() if is_windows else ws_str_norm
            tgt_cmp = tgt_str_norm.lower() if is_windows else tgt_str_norm

            if tgt_cmp == ws_cmp:
                return ""

            if tgt_cmp.startswith(ws_cmp + "/"):
                # Path is inside workspace
                rel = target_resolved.relative_to(ws_resolved)
                rel_str = str(rel).replace("\\", "/")
                # Normalize case consistently on Windows
                return rel_str.lower() if is_windows else rel_str

            # Not contained in workspace_root -> ungrounded
            return None
        except Exception:
            return None

    # No workspace root provided: cannot verify repository containment of absolute paths
    if p.is_absolute():
        return None

    # Clean relative path
    norm = os.path.normpath(raw_str).replace("\\", "/")
    if norm.startswith("../") or norm == "..":
        return None
    norm = norm.lstrip("./")
    is_windows = sys.platform == "win32" or os.name == "nt"
    return norm.lower() if is_windows else norm


def canonical_component_base(
    path: Union[str, Path],
    workspace_root: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Derive canonical component family base (directory + component stem).

    Returns:
        e.g. 'xchange-ui/src/app/modules/members/add-members/add-members'
        or None if path is not a recognized component file.
    """
    canon = canonical_repo_path(path, workspace_root)
    if not canon:
        return None

    low = canon.lower()
    matched_ext: Optional[str] = None
    for ext in COMPONENT_EXTENSIONS:
        if low.endswith(ext):
            matched_ext = ext
            break

    if not matched_ext:
        return None

    # Strip the matched component extension
    base = canon[: len(canon) - len(matched_ext)]
    return base


def is_component_family_member(
    path_a: Union[str, Path],
    path_b: Union[str, Path],
    workspace_root: Optional[Union[str, Path]] = None,
) -> bool:
    """True if path_a and path_b belong to the exact same structural component family."""
    base_a = canonical_component_base(path_a, workspace_root)
    base_b = canonical_component_base(path_b, workspace_root)
    if not base_a or not base_b:
        return False
    return base_a == base_b
