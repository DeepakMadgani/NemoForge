"""
Multi-Project Workspace Registry — Enhancement 6

Manages a registry of multiple projects within a workspace,
enabling cross-project indexing and evidence collection.

Safety: READ-ONLY for discovery. Only writes registry metadata.

Author: Deepak Madgani
Date: July 2026
"""

import logging
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Iterable
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class ProjectInfo(BaseModel):
    """Information about a single project in the workspace."""
    name: str = Field(...)
    path: str = Field(...)
    technology: str = Field("unknown", description="python | java | nodejs | dotnet | go")
    has_index: bool = Field(False, description="Whether SQLite index exists")
    file_count: int = Field(0)
    last_indexed: Optional[str] = Field(None, description="ISO timestamp")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Other projects this project depends on"
    )


class WorkspaceRegistry(BaseModel):
    """Registry of all projects in the workspace."""
    workspace_root: str = Field("")
    projects: List[ProjectInfo] = Field(default_factory=list)
    total_files: int = Field(0)
    last_scan: Optional[str] = Field(None)


# ============================================================================
# PROJECT DETECTION
# ============================================================================

_PROJECT_MARKERS = {
    "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
    "nodejs": ["package.json"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "dotnet": ["*.csproj", "*.sln"],
    "go": ["go.mod"],
}

_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "dist",
    "build", "target", ".angular", "vendor",
    "bin", "obj", ".idea", ".vscode",
}


class WorkspaceRegistryManager:
    """
    Discovers and manages multiple projects within a workspace.
    Enables cross-project indexing and evidence collection.
    """

    def __init__(self, workspace_root: str):
        self.root = Path(workspace_root)
        self._registry: Optional[WorkspaceRegistry] = None
        self._registry_path = self.root / ".aviator" / "workspace_registry.json"

    # ------------------------------------------------------------------
    # 1. Discover projects
    # ------------------------------------------------------------------

    def discover_projects(self, max_depth: int = 3) -> List[ProjectInfo]:
        """
        Discover all projects under the workspace root.

        Args:
            max_depth: Maximum directory depth to search.

        Returns:
            List of ProjectInfo objects.
        """
        projects: List[ProjectInfo] = []
        seen_paths: set = set()

        for tech, markers in _PROJECT_MARKERS.items():
            for marker in markers:
                for match in self.root.rglob(marker):
                    if any(skip in match.parts for skip in _SKIP_DIRS):
                        continue

                    # Check depth
                    try:
                        depth = len(match.relative_to(self.root).parts)
                    except ValueError:
                        continue
                    if depth > max_depth:
                        continue

                    project_dir = match.parent
                    dir_str = str(project_dir)

                    if dir_str in seen_paths:
                        continue
                    seen_paths.add(dir_str)

                    # Count files
                    file_count = sum(
                        1 for f in project_dir.rglob("*")
                        if f.is_file() and not any(s in f.parts for s in _SKIP_DIRS)
                    )

                    # Check for existing index
                    has_index = (project_dir / ".aviator" / "index.sqlite").exists()

                    projects.append(ProjectInfo(
                        name=project_dir.name,
                        path=str(project_dir),
                        technology=tech,
                        has_index=has_index,
                        file_count=file_count,
                    ))

        logger.info(
            f"WorkspaceRegistry: discovered {len(projects)} projects "
            f"under {self.root}"
        )
        return projects

    # ------------------------------------------------------------------
    # 2. Repository root discovery (Git boundary detection)
    # ------------------------------------------------------------------

    def discover_repo_roots(self) -> List[str]:
        """Discover all Git repository roots under the workspace.

        Walks the entire workspace tree looking for `.git` directories.
        Returns a list of workspace-relative POSIX paths — one per
        discovered repository.

        No depth limit is assumed; nested repository layouts are
        supported.  The walk itself is fast because it only inspects
        directory names and skips heavy subtrees (node_modules, etc.).

        Returns:
            Sorted list of repo root prefixes relative to workspace root.
            Example: ["area-service", "project-service", "xchange-ui"]
        """
        # Skip dirs that can never contain a .git child of interest
        _discovery_skip = {
            "node_modules", "__pycache__", "dist", "build", "target",
            ".angular", "vendor", "bin", "obj", ".idea", ".vscode",
        }

        roots: List[str] = []
        for dirpath, dirnames, _ in os.walk(str(self.root)):
            # Prune heavy subtrees in-place and sort for deterministic discovery
            dirnames[:] = sorted([
                d for d in dirnames
                if d not in _discovery_skip
            ])

            # Check for .git in current directory's children
            if ".git" in dirnames:
                repo_dir = Path(dirpath)
                try:
                    rel = str(repo_dir.relative_to(self.root)).replace("\\", "/")
                except ValueError:
                    continue
                # The workspace root itself may be a git repo — represent as ""
                roots.append(rel if rel != "." else "")
                # Don't recurse into this repo's .git dir
                dirnames[:] = [d for d in dirnames if d != ".git"]

        roots.sort()
        logger.info(
            f"WorkspaceRegistry: discovered {len(roots)} repository roots "
            f"under {self.root}: {roots}"
        )
        return roots

    def resolve_repo_root(self, rel_path: str, repo_roots: List[str] = None) -> Optional[str]:
        """Map a workspace-relative file path to its containing repository root.

        Args:
            rel_path:    Workspace-relative POSIX path (e.g. "xchange-ui/src/foo.ts").
            repo_roots:  Pre-computed list from discover_repo_roots().
                         If None, discovery is run on the fly.

        Returns:
            The repository root prefix (e.g. "xchange-ui"), or None if the
            file is not inside any discovered repository (orphan).
        """
        if repo_roots is None:
            repo_roots = self.discover_repo_roots()

        path_norm = rel_path.replace("\\", "/")
        best: Optional[str] = None
        best_len = -1

        for root in repo_roots:
            if not root:
                # Workspace root is itself a repo — everything matches,
                # but prefer a more specific match if available.
                if best is None:
                    best = root
                    best_len = 0
                continue
            prefix = root + "/"
            if path_norm.startswith(prefix) and len(root) > best_len:
                best = root
                best_len = len(root)

        return best

    def build_repo_root_map(
        self,
        candidate_paths: "Iterable[str]",
        repo_roots: List[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Batch-resolve repository roots for a set of candidate paths.

        Returns a dict mapping each candidate path to its containing
        repository root (or None for orphans).

        This is the intended entry point for the evidence ranking engine:
        resolve all candidates once, then pass the mapping to the ranker.
        """
        if repo_roots is None:
            repo_roots = self.discover_repo_roots()
        return {
            fp: self.resolve_repo_root(fp, repo_roots)
            for fp in candidate_paths
        }

    # ------------------------------------------------------------------
    # 3. Build/load registry
    # ------------------------------------------------------------------

    def build_registry(self) -> WorkspaceRegistry:
        """Discover projects and build the registry."""
        projects = self.discover_projects()
        total = sum(p.file_count for p in projects)

        registry = WorkspaceRegistry(
            workspace_root=str(self.root),
            projects=projects,
            total_files=total,
            last_scan=datetime.now().isoformat(),
        )

        self._registry = registry
        self._save_registry(registry)
        return registry

    def load_registry(self) -> Optional[WorkspaceRegistry]:
        """Load registry from disk cache."""
        if self._registry:
            return self._registry

        if not self._registry_path.exists():
            return None

        try:
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
            self._registry = WorkspaceRegistry(**data)
            return self._registry
        except Exception as exc:
            logger.debug(f"Failed to load workspace registry: {exc}")
            return None

    def _save_registry(self, registry: WorkspaceRegistry) -> None:
        """Save registry to disk."""
        try:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            self._registry_path.write_text(
                registry.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug(f"Failed to save workspace registry: {exc}")

    # ------------------------------------------------------------------
    # 3. Cross-project queries
    # ------------------------------------------------------------------

    def find_project_for_file(self, file_path: str) -> Optional[ProjectInfo]:
        """Find which project a file belongs to."""
        registry = self.load_registry() or self.build_registry()

        fp = Path(file_path)
        for project in registry.projects:
            pp = Path(project.path)
            try:
                fp.relative_to(pp)
                return project
            except ValueError:
                continue

        return None

    def get_projects_by_technology(self, tech: str) -> List[ProjectInfo]:
        """Get all projects of a specific technology."""
        registry = self.load_registry() or self.build_registry()
        return [p for p in registry.projects if p.technology == tech]

    def get_dependent_projects(self, project_name: str) -> List[ProjectInfo]:
        """Get projects that depend on the given project."""
        registry = self.load_registry() or self.build_registry()
        return [
            p for p in registry.projects
            if project_name in p.dependencies
        ]
