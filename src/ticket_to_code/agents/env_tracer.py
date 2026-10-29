"""
Environment Variable Propagation Tracer — Enhancement 11

Traces environment variable usage across the codebase:
- Which files SET vs READ each variable
- Detects mismatches (same var, different values)
- Builds an env var dependency graph

Safety: READ-ONLY. Only reads source files.

Author: Deepak Madgani
Date: July 2026
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from collections import defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class EnvVarUsage(BaseModel):
    """A single environment variable usage in a file."""
    var_name: str = Field(..., description="Environment variable name")
    file_path: str = Field(..., description="File where it's used")
    line_number: int = Field(0, description="Line number")
    usage_type: str = Field("read", description="'read' or 'set'")
    value: Optional[str] = Field(None, description="Value if it's a SET operation")
    pattern: str = Field("", description="The matched pattern (e.g. os.environ, process.env)")


class EnvVarConflict(BaseModel):
    """A detected conflict in environment variable values."""
    var_name: str = Field(...)
    set_locations: List[str] = Field(default_factory=list, description="Files that SET this var")
    values: List[str] = Field(default_factory=list, description="Different values found")
    read_locations: List[str] = Field(default_factory=list, description="Files that READ this var")
    severity: str = Field("warning", description="'warning' or 'error'")


class EnvTraceResult(BaseModel):
    """Complete result of environment variable tracing."""
    has_env_data: bool = Field(False)
    usages: List[EnvVarUsage] = Field(default_factory=list)
    conflicts: List[EnvVarConflict] = Field(default_factory=list)
    env_graph: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=dict,
        description="var_name -> {'readers': [files], 'setters': [files]}"
    )
    total_vars: int = Field(0)
    total_conflicts: int = Field(0)


# ============================================================================
# PATTERNS
# ============================================================================

# Python: os.environ["VAR"], os.environ.get("VAR"), os.getenv("VAR")
_PY_READ = re.compile(
    r'os\.(?:environ\.get|getenv|environ\[)\s*\(\s*["\'](\w+)["\']'
)
_PY_SET = re.compile(
    r'os\.environ\[["\'](\w+)["\']\]\s*=\s*["\']?([^"\'\n]+)["\']?'
)

# Node.js: process.env.VAR, process.env["VAR"]
_NODE_READ = re.compile(
    r'process\.env\.(\w+)|process\.env\[["\'](\w+)["\']\]'
)

# PowerShell: $env:VAR
_PS_READ = re.compile(r'\$env:(\w+)', re.IGNORECASE)
_PS_SET = re.compile(r'\$env:(\w+)\s*=\s*["\']?([^"\'\n]+)["\']?', re.IGNORECASE)

# .env file: VAR=value
_DOTENV = re.compile(r'^(\w+)\s*=\s*(.+)$', re.MULTILINE)

# Shell: export VAR=value
_SHELL_SET = re.compile(r'export\s+(\w+)\s*=\s*["\']?([^"\'\n]+)["\']?')

# Docker/YAML: - VAR=value or VAR: value in environment section
_DOCKER_ENV = re.compile(r'^\s*-?\s*(\w+)\s*[:=]\s*(.+)$', re.MULTILINE)

# Generic ${VAR} or $VAR
_GENERIC_REF = re.compile(r'\$\{(\w+)\}|\$(\w+)')

# Skip directories
_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".idea",
    ".vscode", "bin", "obj", "dist", ".angular",
    "target", "build", "vendor",
}

# File extensions to scan
_SCAN_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".ps1", ".sh", ".bash", ".bat", ".cmd",
    ".env", ".yml", ".yaml",
    ".json", ".toml", ".cfg", ".ini",
    ".cs", ".java", ".go",
    ".dockerfile", ".docker-compose.yml",
}


class EnvTracer:
    """
    Traces environment variable usage across the workspace.
    Builds a graph of which files SET vs READ each variable.
    """

    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)

    def _should_scan(self, path: Path) -> bool:
        """Check if a file should be scanned."""
        if any(skip in path.parts for skip in _SKIP_DIRS):
            return False
        if path.suffix.lower() not in _SCAN_EXTENSIONS:
            # Also scan Dockerfiles and .env files
            if path.name.lower() not in (".env", "dockerfile", ".env.example"):
                return False
        return True

    # ------------------------------------------------------------------
    # 1. Scan a single file
    # ------------------------------------------------------------------

    def _scan_file(self, file_path: Path) -> List[EnvVarUsage]:
        """Scan a single file for env var usages."""
        usages: List[EnvVarUsage] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return usages

        rel_path = str(file_path.relative_to(self.workspace))
        ext = file_path.suffix.lower()
        name = file_path.name.lower()

        # .env files — all lines are SET operations
        if name in (".env", ".env.example", ".env.local", ".env.development"):
            for m in _DOTENV.finditer(content):
                if not m.group(1).startswith("#"):
                    usages.append(EnvVarUsage(
                        var_name=m.group(1),
                        file_path=rel_path,
                        line_number=content[:m.start()].count("\n") + 1,
                        usage_type="set",
                        value=m.group(2).strip(),
                        pattern=".env",
                    ))
            return usages

        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            # Python patterns
            if ext == ".py":
                for m in _PY_SET.finditer(line):
                    usages.append(EnvVarUsage(
                        var_name=m.group(1), file_path=rel_path,
                        line_number=line_num, usage_type="set",
                        value=m.group(2).strip(), pattern="os.environ",
                    ))
                for m in _PY_READ.finditer(line):
                    usages.append(EnvVarUsage(
                        var_name=m.group(1), file_path=rel_path,
                        line_number=line_num, usage_type="read",
                        pattern="os.environ/getenv",
                    ))

            # Node.js patterns
            elif ext in (".js", ".ts", ".jsx", ".tsx"):
                for m in _NODE_READ.finditer(line):
                    var = m.group(1) or m.group(2)
                    usages.append(EnvVarUsage(
                        var_name=var, file_path=rel_path,
                        line_number=line_num, usage_type="read",
                        pattern="process.env",
                    ))

            # PowerShell patterns
            elif ext == ".ps1":
                for m in _PS_SET.finditer(line):
                    usages.append(EnvVarUsage(
                        var_name=m.group(1), file_path=rel_path,
                        line_number=line_num, usage_type="set",
                        value=m.group(2).strip(), pattern="$env:",
                    ))
                for m in _PS_READ.finditer(line):
                    # Only count as read if not already counted as set
                    if not _PS_SET.search(line):
                        usages.append(EnvVarUsage(
                            var_name=m.group(1), file_path=rel_path,
                            line_number=line_num, usage_type="read",
                            pattern="$env:",
                        ))

            # Shell patterns
            elif ext in (".sh", ".bash"):
                for m in _SHELL_SET.finditer(line):
                    usages.append(EnvVarUsage(
                        var_name=m.group(1), file_path=rel_path,
                        line_number=line_num, usage_type="set",
                        value=m.group(2).strip(), pattern="export",
                    ))

            # YAML/Docker patterns
            elif ext in (".yml", ".yaml"):
                for m in _GENERIC_REF.finditer(line):
                    var = m.group(1) or m.group(2)
                    usages.append(EnvVarUsage(
                        var_name=var, file_path=rel_path,
                        line_number=line_num, usage_type="read",
                        pattern="${VAR}",
                    ))

        return usages

    # ------------------------------------------------------------------
    # 2. Build env graph and detect conflicts
    # ------------------------------------------------------------------

    def _build_graph(
        self, usages: List[EnvVarUsage]
    ) -> Tuple[Dict[str, Dict[str, List[str]]], List[EnvVarConflict]]:
        """Build env var graph and detect conflicts."""
        graph: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: {"readers": [], "setters": []}
        )
        values_map: Dict[str, Dict[str, str]] = defaultdict(dict)

        for u in usages:
            key = "setters" if u.usage_type == "set" else "readers"
            if u.file_path not in graph[u.var_name][key]:
                graph[u.var_name][key].append(u.file_path)
            if u.usage_type == "set" and u.value:
                values_map[u.var_name][u.file_path] = u.value

        # Detect conflicts: same var set to different values
        conflicts: List[EnvVarConflict] = []
        for var_name, file_values in values_map.items():
            unique_values = set(file_values.values())
            if len(unique_values) > 1:
                conflicts.append(EnvVarConflict(
                    var_name=var_name,
                    set_locations=list(file_values.keys()),
                    values=list(unique_values),
                    read_locations=graph[var_name]["readers"],
                    severity="error" if len(unique_values) > 2 else "warning",
                ))

        return dict(graph), conflicts

    # ------------------------------------------------------------------
    # 3. Main entry point
    # ------------------------------------------------------------------

    def trace(self, target_vars: Optional[List[str]] = None) -> EnvTraceResult:
        """
        Scan workspace for environment variable usage.

        Args:
            target_vars: If provided, only trace these specific variables.

        Returns:
            EnvTraceResult with complete env var graph.
        """
        if not self.workspace.exists():
            return EnvTraceResult(has_env_data=False)

        all_usages: List[EnvVarUsage] = []
        scanned = 0

        for file_path in self.workspace.rglob("*"):
            if not file_path.is_file():
                continue
            if not self._should_scan(file_path):
                continue
            usages = self._scan_file(file_path)
            all_usages.extend(usages)
            scanned += 1

        # Filter by target vars if specified
        if target_vars:
            target_set = set(v.upper() for v in target_vars)
            all_usages = [u for u in all_usages if u.var_name.upper() in target_set]

        if not all_usages:
            logger.info(f"EnvTracer: scanned {scanned} files, no env vars found")
            return EnvTraceResult(has_env_data=False)

        # Build graph and detect conflicts
        graph, conflicts = self._build_graph(all_usages)

        result = EnvTraceResult(
            has_env_data=True,
            usages=all_usages,
            conflicts=conflicts,
            env_graph=graph,
            total_vars=len(graph),
            total_conflicts=len(conflicts),
        )

        logger.info(
            f"EnvTracer: {len(all_usages)} usages of {len(graph)} vars "
            f"across {scanned} files, {len(conflicts)} conflicts"
        )
        return result
