"""Language & Service Resolution Scope.

Provides centralized language, extension, and project-root scoping for all
retrieval, capability discovery, error localization, and repair operations.
Guarantees at query time that symbols in one language/service (e.g. Java in
issues-service) never contaminate searches for another (e.g. TypeScript in
xchange-ui).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union


# Standard source extensions by language
LANGUAGE_EXTENSIONS: dict[str, Tuple[str, ...]] = {
    "typescript": (".ts", ".tsx", ".d.ts"),
    "javascript": (".js", ".jsx", ".mjs"),
    "java": (".java",),
    "python": (".py",),
    "csharp": (".cs",),
    "go": (".go",),
    "html": (".html", ".htm"),
    "css": (".css", ".scss", ".sass", ".less"),
}

# Reverse lookup from extension to canonical language
EXTENSION_TO_LANGUAGE: dict[str, str] = {}
for lang, exts in LANGUAGE_EXTENSIONS.items():
    for ext in exts:
        EXTENSION_TO_LANGUAGE[ext.lower()] = lang


@dataclass(frozen=True)
class ResolutionScope:
    """Explicit boundary contract for symbol and file resolution."""
    language: str  # "typescript", "java", "python", "csharp", "unknown"
    project_root: str  # Top-level service/repo name, e.g. "xchange-ui"
    source_extensions: Tuple[str, ...]  # Allowed extensions, e.g. (".ts", ".tsx", ".d.ts")

    def matches_path(self, path: Union[str, Path]) -> bool:
        """Return True if path satisfies both extension and project_root constraints."""
        p_str = str(path).replace("\\", "/").lower().strip()
        if not p_str:
            return False

        # Extension check
        if self.source_extensions:
            if not any(p_str.endswith(ext.lower()) for ext in self.source_extensions):
                return False

        # Project root boundary check
        if self.project_root:
            pr = self.project_root.replace("\\", "/").lower().strip("/")
            # Valid if path starts with root (e.g. "xchange-ui/...") or contains "/xchange-ui/"
            in_root = (
                p_str.startswith(f"{pr}/")
                or f"/{pr}/" in f"/{p_str}"
                or p_str == pr
            )
            if not in_root:
                return False

        return True

    def build_sql_filter(self, path_column: str = "path") -> Tuple[str, list]:
        """Build SQL WHERE clause fragment and params to enforce scope at query time.

        Returns:
            (sql_fragment, params_list)
            Example:
                (" AND (s.path LIKE ? OR ...) AND (s.path LIKE ? ...)", ['%.ts', 'xchange-ui/%'])
        """
        clauses = []
        params = []

        # 1. Extension filter at query time
        if self.source_extensions:
            ext_clauses = []
            for ext in self.source_extensions:
                ext_clauses.append(f"{path_column} LIKE ?")
                params.append(f"%{ext}")
            clauses.append(f"({' OR '.join(ext_clauses)})")

        # 2. Project root filter at query time
        if self.project_root:
            pr = self.project_root.replace("\\", "/").strip("/")
            root_clauses = [
                f"{path_column} LIKE ?",
                f"{path_column} LIKE ?",
                f"{path_column} LIKE ?",
                f"{path_column} LIKE ?",
            ]
            params.extend([
                f"{pr}/%",
                f"%/{pr}/%",
                f"{pr}\\%",
                f"%\\{pr}\\%",
            ])
            clauses.append(f"({' OR '.join(root_clauses)})")

        if clauses:
            return " AND " + " AND ".join(clauses), params
        return "", []


def infer_resolution_scope(
    file_path: Union[str, Path],
    workspace_root: Optional[Union[str, Path]] = None,
) -> ResolutionScope:
    """Infer the appropriate ResolutionScope from a target file path.

    Examines file extension to determine language and allowed source extensions,
    and inspects directory hierarchy to identify poly-repo service root.
    """
    p = Path(str(file_path).replace("\\", "/"))
    ext = p.suffix.lower()

    # Handle multi-part suffixes like .d.ts
    name_lower = p.name.lower()
    if name_lower.endswith(".d.ts"):
        ext = ".d.ts"

    language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")
    extensions = LANGUAGE_EXTENSIONS.get(language, (ext,) if ext else ())

    # Determine project root
    project_root = ""
    # Normalize relative to workspace_root if provided and p is absolute
    if workspace_root:
        ws = Path(str(workspace_root).replace("\\", "/"))
        try:
            if p.is_absolute() and ws.is_absolute():
                rel = p.relative_to(ws)
                parts = rel.parts
                if len(parts) > 1:
                    project_root = parts[0]
        except Exception:
            pass

    if not project_root:
        parts = p.parts
        # If path looks like "xchange-ui/src/app/...", parts[0] is the service root
        if len(parts) > 1 and not p.is_absolute():
            project_root = parts[0]
        elif len(parts) > 2 and p.is_absolute():
            # For Windows absolute paths C:/CC4E/xchange-ui/...
            # Search for recognizable project roots or use the first directory under root
            for idx, part in enumerate(parts):
                if part.lower() in ("src", "app", "modules", "main"):
                    if idx > 0 and parts[idx - 1].lower() not in ("src", "main"):
                        project_root = parts[idx - 1]
                        break
                    elif idx > 1:
                        project_root = parts[idx - 2]
                        break

    return ResolutionScope(
        language=language,
        project_root=project_root,
        source_extensions=extensions,
    )
