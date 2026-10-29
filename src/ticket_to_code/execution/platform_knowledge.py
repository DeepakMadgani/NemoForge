"""
Platform-Aware Execution — Enhancement 5

Platform knowledge base that provides Windows/Linux-specific
guidance for builds, encoding issues, path separators, and
port conflicts.

Safety: READ-ONLY. Only provides warnings and suggestions.

Author: Deepak Madgani
Date: July 2026
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class PlatformWarning(BaseModel):
    """A platform-specific warning or suggestion."""
    category: str = Field(..., description="encoding | path | port | permission | line_ending")
    severity: str = Field("warning", description="info | warning | error")
    message: str = Field(...)
    fix_suggestion: Optional[str] = Field(None)
    affected_file: Optional[str] = Field(None)


class PlatformContext(BaseModel):
    """Platform context for the current system."""
    os_name: str = Field("")
    is_windows: bool = Field(False)
    python_version: str = Field("")
    encoding: str = Field("utf-8")
    path_separator: str = Field("/")
    line_ending: str = Field("\\n")
    warnings: List[PlatformWarning] = Field(default_factory=list)


# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

_WINDOWS_ENCODING_FIXES = {
    "UnicodeEncodeError": (
        "Set PYTHONIOENCODING=utf-8 or add "
        "sys.stdout.reconfigure(encoding='utf-8') at module start"
    ),
    "charmap": (
        "Windows console uses cp1252 by default. "
        "Open files with encoding='utf-8' explicitly"
    ),
    "cp1252": (
        "Replace emoji/unicode characters with ASCII equivalents "
        "or set PYTHONUTF8=1 environment variable"
    ),
}

_WINDOWS_PATH_ISSUES = [
    "MAX_PATH (260 char) limit — enable long paths in registry or use \\\\?\\",
    "Use Path() from pathlib instead of string concatenation",
    "Forward slashes work in Python even on Windows",
    "Don't hardcode \\\\ — use os.sep or pathlib",
]

_PORT_CONFLICTS: Dict[int, str] = {
    80: "IIS or Apache may be using port 80",
    443: "IIS may be using port 443",
    3000: "Node.js dev server default",
    5000: "Flask default (also used by Windows ControlPanel)",
    8000: "Django default",
    8080: "Tomcat/Java default",
    5432: "PostgreSQL default",
    7474: "Neo4j browser default",
    7687: "Neo4j bolt protocol default",
}


class PlatformKnowledge:
    """
    Provides platform-aware guidance for builds and code generation.

    Consulted before/after builds and during code generation to
    prevent platform-specific issues.
    """

    def __init__(self):
        self._context: Optional[PlatformContext] = None

    def get_context(self) -> PlatformContext:
        """Get current platform context with detected warnings."""
        if self._context:
            return self._context

        is_win = sys.platform.startswith("win")
        ctx = PlatformContext(
            os_name=sys.platform,
            is_windows=is_win,
            python_version=sys.version.split()[0],
            encoding=sys.getdefaultencoding(),
            path_separator=os.sep,
            line_ending="\\r\\n" if is_win else "\\n",
        )

        # Check for known platform issues
        if is_win:
            # Console encoding
            console_enc = sys.stdout.encoding if sys.stdout else "unknown"
            if console_enc and console_enc.lower() in ("cp1252", "charmap"):
                ctx.warnings.append(PlatformWarning(
                    category="encoding",
                    severity="warning",
                    message=f"Console encoding is {console_enc} — unicode may fail",
                    fix_suggestion="Set $env:PYTHONUTF8=1 or $env:PYTHONIOENCODING='utf-8'",
                ))

            # PYTHONIOENCODING check
            if not os.environ.get("PYTHONIOENCODING"):
                ctx.warnings.append(PlatformWarning(
                    category="encoding",
                    severity="info",
                    message="PYTHONIOENCODING not set — emoji in logs may crash",
                    fix_suggestion="$env:PYTHONIOENCODING='utf-8'",
                ))

        self._context = ctx
        return ctx

    # ------------------------------------------------------------------
    # Pre-build checks
    # ------------------------------------------------------------------

    def pre_build_check(
        self, workspace_path: str, technology: str = "auto"
    ) -> List[PlatformWarning]:
        """
        Check for platform-specific issues before building.

        Returns list of warnings/suggestions.
        """
        warnings: List[PlatformWarning] = []
        ctx = self.get_context()
        ws = Path(workspace_path)

        if ctx.is_windows:
            # Check path lengths
            for p in ws.rglob("*"):
                if len(str(p)) > 250:
                    warnings.append(PlatformWarning(
                        category="path",
                        severity="warning",
                        message=f"Path near MAX_PATH limit ({len(str(p))} chars): {p.name}",
                        affected_file=str(p),
                    ))
                    break  # Only report once

            # Check for port conflicts
            for port, desc in _PORT_CONFLICTS.items():
                if self._is_port_in_use(port):
                    warnings.append(PlatformWarning(
                        category="port",
                        severity="info",
                        message=f"Port {port} is in use ({desc})",
                        fix_suggestion=f"Use a different port or kill the process on {port}",
                    ))

        return warnings

    # ------------------------------------------------------------------
    # Post-build analysis
    # ------------------------------------------------------------------

    def analyze_build_errors(
        self, errors: List[str]
    ) -> List[PlatformWarning]:
        """
        Check if build errors are platform-related.
        """
        warnings: List[PlatformWarning] = []
        error_text = "\n".join(errors)

        for pattern, fix in _WINDOWS_ENCODING_FIXES.items():
            if pattern.lower() in error_text.lower():
                warnings.append(PlatformWarning(
                    category="encoding",
                    severity="error",
                    message=f"Platform encoding error: {pattern}",
                    fix_suggestion=fix,
                ))

        if "Permission denied" in error_text:
            warnings.append(PlatformWarning(
                category="permission",
                severity="error",
                message="Permission denied — file may be locked by another process",
                fix_suggestion="Close IDE, kill background processes, or run as admin",
            ))

        if "CRLF" in error_text or "line ending" in error_text.lower():
            warnings.append(PlatformWarning(
                category="line_ending",
                severity="warning",
                message="Line ending mismatch (CRLF vs LF)",
                fix_suggestion="Set git config core.autocrlf true",
            ))

        return warnings

    # ------------------------------------------------------------------
    # Code generation guidance
    # ------------------------------------------------------------------

    def get_generation_hints(self) -> List[str]:
        """
        Return platform-aware hints for code generation.
        """
        ctx = self.get_context()
        hints = []

        if ctx.is_windows:
            hints.extend([
                "Use pathlib.Path instead of string concatenation for file paths",
                "Open files with encoding='utf-8' explicitly",
                "Avoid emoji in log messages (Windows console may crash)",
                "Use os.path.join() or Path / operator for cross-platform paths",
                "Check for port conflicts before starting servers",
            ])

        return hints

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """Check if a port is currently in use."""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                return s.connect_ex(("localhost", port)) == 0
        except Exception:
            return False
