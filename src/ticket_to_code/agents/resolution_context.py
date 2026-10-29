"""
Resolution Context — Build-system-aware partitioning for mixed-technology workspaces.

Groups StructuredDiagnostics into ResolutionContexts based on the
(language, build_tool, compile_root) triple, detected by probing the
filesystem rather than guessing from file extensions.

Why this matters:
  - .ts could be compiled by tsc, Angular CLI, Vite, Webpack, etc.
  - .java could be built by Gradle, Maven, Bazel, Ant, etc.
  - The partitioning key is the BUILD/RESOLUTION ENVIRONMENT, not the language.

Pipeline position:
  Raw Build Output → DiagnosticNormalizer → StructuredDiagnostic[]
                                              ↓
                                    resolve_contexts() (this module)
                                              ↓
                                    ResolutionContext[]
                                              ↓
                                    Sequential resolution passes

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ticket_to_code.agents.diagnostic_normalizer import StructuredDiagnostic

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ResolutionContext:
    """Build-system-aware context for a group of related diagnostics.

    The partitioning key is NOT the language alone — it's the
    (language, build_tool, compile_root) triple. This matters because:
      - .ts could be tsc, Angular CLI, Vite, Webpack, etc.
      - .java could be Gradle, Maven, Bazel, etc.

    Each ResolutionContext gets its own ErrorResolutionAgent instance
    with the correct compile_tool and compile_root, ensuring:
      - TS errors are verified with tsc/ng build
      - Java errors are verified with gradle/maven
      - Neither technology pollutes the other's verification
    """
    language: str           # "typescript", "angular", "java", "python", "csharp"
    build_tool: str         # "tsc", "angular-cli", "gradle", "maven", "pytest", "dotnet"
    compile_root: Path      # Project root where the build command runs
    diagnostics: list[StructuredDiagnostic] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)  # Unique files from diagnostics


# ── Resolution order ──────────────────────────────────────────────────────────
# Frontend-first: TS/Angular errors are typically faster to fix and their
# fixes don't usually affect backend code. Backend errors may depend on
# interface changes made during the frontend pass.

_RESOLUTION_ORDER = {
    "typescript": 0,
    "angular": 0,      # Same priority as TS (merged when sharing compile_root)
    "python": 1,
    "java": 2,
    "csharp": 3,
    "unknown": 99,
}


# ── Build-tool detection ─────────────────────────────────────────────────────

def _detect_frontend_build_tool(directory: Path) -> Optional[tuple[str, Path]]:
    """Detect Angular/TypeScript build tool by walking up from directory.

    Returns (build_tool, compile_root) or None.
    Checks for:
      - angular.json → "angular-cli"
      - tsconfig.json without angular.json → "tsc"
    """
    current = directory.resolve()
    # Don't walk above 5 levels to avoid scanning the entire filesystem
    for _ in range(10):
        if (current / "angular.json").exists():
            return ("angular-cli", current)
        if (current / "tsconfig.json").exists():
            # Check if angular.json is at the same level or above
            # (angular projects always have tsconfig.json too)
            parent = current.parent
            for __ in range(3):
                if (parent / "angular.json").exists():
                    return ("angular-cli", parent)
                if parent == parent.parent:
                    break
                parent = parent.parent
            return ("tsc", current)
        if current == current.parent:
            break
        current = current.parent
    return None


def _detect_jvm_build_tool(file_path: str, workspace_path: Path) -> Optional[tuple[str, Path]]:
    """Detect Java/Kotlin/Scala build tool by walking up from file_path.

    Returns (build_tool, module_root) or None.
    Reuses the same logic as _find_java_module_root in workflow.py.
    """
    ws = workspace_path.resolve()
    try:
        clean_fp = str(file_path).strip().replace("\\", "/")
        if clean_fp.startswith("/") and len(clean_fp) > 2 and clean_fp[1].isalpha() and clean_fp[2] == ":":
            clean_fp = clean_fp[1:]
        p = Path(clean_fp)
        if p.is_absolute():
            current = p.resolve().parent
        else:
            current = (ws / clean_fp).resolve().parent
    except (ValueError, OSError):
        return None

    while True:
        if (current / "pom.xml").exists():
            return ("maven", current)
        if (current / "build.gradle").exists() or (current / "build.gradle.kts").exists():
            return ("gradle", current)
        if current == ws or current == current.parent:
            break
        current = current.parent
    return None


def _detect_python_build_tool(file_path: str, workspace_path: Path) -> Optional[tuple[str, Path]]:
    """Detect Python build/test tool by walking up from file_path.

    Returns (build_tool, project_root) or None.
    """
    ws = workspace_path.resolve()
    try:
        clean_fp = str(file_path).strip().replace("\\", "/")
        if clean_fp.startswith("/") and len(clean_fp) > 2 and clean_fp[1].isalpha() and clean_fp[2] == ":":
            clean_fp = clean_fp[1:]
        p = Path(clean_fp)
        if p.is_absolute():
            current = p.resolve().parent
        else:
            current = (ws / clean_fp).resolve().parent
    except (ValueError, OSError):
        return None

    while True:
        if (current / "pyproject.toml").exists():
            return ("pytest", current)
        if (current / "setup.py").exists() or (current / "setup.cfg").exists():
            return ("pytest", current)
        if current == ws or current == current.parent:
            break
        current = current.parent

    # Fallback: use workspace root
    return ("pytest", ws)


def _detect_dotnet_build_tool(file_path: str, workspace_path: Path) -> Optional[tuple[str, Path]]:
    """Detect .NET build tool by walking up from file_path.

    Returns (build_tool, project_root) or None.
    """
    ws = workspace_path.resolve()
    try:
        clean_fp = str(file_path).strip().replace("\\", "/")
        if clean_fp.startswith("/") and len(clean_fp) > 2 and clean_fp[1].isalpha() and clean_fp[2] == ":":
            clean_fp = clean_fp[1:]
        p = Path(clean_fp)
        if p.is_absolute():
            current = p.resolve().parent
        else:
            current = (ws / clean_fp).resolve().parent
    except (ValueError, OSError):
        return None

    while True:
        for csproj in current.glob("*.csproj"):
            return ("dotnet", current)
        if (current / "*.sln").exists():
            return ("dotnet", current)
        if current == ws or current == current.parent:
            break
        current = current.parent
    return None


# ── Group by language ─────────────────────────────────────────────────────────

def group_by_language(
    diagnostics: list[StructuredDiagnostic],
) -> dict[str, list[StructuredDiagnostic]]:
    """Group diagnostics by their language field.

    Merges 'angular' and 'typescript' into a single 'typescript' group
    because they share the same compile root and build tool.
    """
    groups: dict[str, list[StructuredDiagnostic]] = {}
    for d in diagnostics:
        # Merge angular into typescript — they're compiled together
        lang = "typescript" if d.language == "angular" else (d.language or "unknown")
        groups.setdefault(lang, []).append(d)
    return groups


# ── Public API ────────────────────────────────────────────────────────────────

def resolve_contexts(
    diagnostics: list[StructuredDiagnostic],
    workspace_path: Path,
) -> list[ResolutionContext]:
    """Partition structured diagnostics into ResolutionContexts.

    Pipeline:
      1. Group diagnostics by language (merging angular+typescript)
      2. For each group, detect build_tool + compile_root from filesystem
      3. Create ResolutionContext with the detected triple
      4. Sort by resolution order (frontend first, backend second)

    Build tool detection walks UP from each source file to find:
      - angular.json → "angular-cli"
      - tsconfig.json → "tsc"
      - build.gradle / build.gradle.kts → "gradle"
      - pom.xml → "maven"
      - pyproject.toml / setup.py → "pytest"
      - *.csproj → "dotnet"

    Falls back to language defaults when no build file is found.

    Args:
        diagnostics: Normalized diagnostics from DiagnosticNormalizer.
        workspace_path: Absolute path to the workspace root.

    Returns:
        Sorted list of ResolutionContexts, one per unique build environment.
    """
    if not diagnostics:
        return []

    lang_groups = group_by_language(diagnostics)
    contexts: list[ResolutionContext] = []

    for lang, diags in lang_groups.items():
        # Collect unique source files from this language's diagnostics
        source_files = sorted(set(d.source_file for d in diags if d.source_file))

        # Detect build tool from the first available source file
        build_tool = _default_build_tool(lang)
        compile_root = workspace_path

        # Use the first source file to probe the filesystem
        probe_file = source_files[0] if source_files else ""

        if lang == "typescript":
            detected = _detect_frontend_build_tool(workspace_path)
            if not detected and probe_file:
                # Try probing from the source file's directory
                try:
                    probe_path = (workspace_path / probe_file).resolve()
                    if probe_path.exists():
                        detected = _detect_frontend_build_tool(probe_path.parent)
                    else:
                        # probe_file might be relative to a child directory (e.g. xchange-ui)
                        for child in workspace_path.iterdir():
                            if child.is_dir() and (child / probe_file).exists():
                                detected = _detect_frontend_build_tool((child / probe_file).parent)
                                break
                except (ValueError, OSError):
                    pass

            # If still not detected, search top-level child directories of workspace_path
            if not detected:
                try:
                    for child in workspace_path.iterdir():
                        if child.is_dir() and not child.name.startswith("."):
                            if (child / "angular.json").exists():
                                detected = ("angular-cli", child)
                                break
                            if (child / "tsconfig.json").exists():
                                detected = ("tsc", child)
                                break
                except (ValueError, OSError):
                    pass

            if detected:
                build_tool, compile_root = detected

        elif lang == "java":
            if probe_file:
                detected = _detect_jvm_build_tool(probe_file, workspace_path)
                if detected:
                    build_tool, compile_root = detected

        elif lang == "python":
            if probe_file:
                detected = _detect_python_build_tool(probe_file, workspace_path)
                if detected:
                    build_tool, compile_root = detected

        elif lang == "csharp":
            if probe_file:
                detected = _detect_dotnet_build_tool(probe_file, workspace_path)
                if detected:
                    build_tool, compile_root = detected

        ctx = ResolutionContext(
            language=lang,
            build_tool=build_tool,
            compile_root=compile_root,
            diagnostics=diags,
            source_files=source_files,
        )
        contexts.append(ctx)
        logger.info(
            f"  [ResolutionContext] {lang}/{build_tool} @ {compile_root} "
            f"({len(diags)} diagnostics, {len(source_files)} files)"
        )

    # Sort by resolution order (frontend first)
    contexts.sort(key=lambda c: _RESOLUTION_ORDER.get(c.language, 99))

    logger.info(
        f"  [ResolutionContext] Partitioned {len(diagnostics)} diagnostics "
        f"into {len(contexts)} context(s)"
    )
    return contexts


def _default_build_tool(language: str) -> str:
    """Fallback build tool when filesystem detection finds nothing."""
    defaults = {
        "typescript": "tsc",
        "java": "gradle",
        "python": "pytest",
        "csharp": "dotnet",
    }
    return defaults.get(language, "unknown")
