"""
Runtime Verifier — Enhancement 7

After a build passes, optionally starts the application and checks
for runtime errors in stdout/stderr. If a runtime error is detected,
feeds it back into the pipeline for re-diagnosis.

Safety:
  - Uses subprocess with timeout (max 30s)
  - Only reads stdout/stderr
  - Kills the process after checking

Author: Deepak Madgani
Date: July 2026
"""

import logging
import subprocess
import time
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class RuntimeCheckResult(BaseModel):
    """Result of a runtime verification check."""
    ran_check: bool = Field(False, description="Whether the check was performed")
    started_successfully: bool = Field(False, description="Whether the app started")
    runtime_error_detected: bool = Field(False, description="Whether a runtime error was found")
    error_output: Optional[str] = Field(None, description="Error output captured")
    ready_signal_found: bool = Field(False, description="Whether the ready signal was detected")
    startup_time_ms: int = Field(0, description="Time to start in milliseconds")
    process_exit_code: Optional[int] = Field(None)


# ============================================================================
# STARTUP COMMANDS AND READY SIGNALS
# ============================================================================

_STARTUP_CONFIGS: Dict[str, Dict] = {
    "python": {
        "commands": [
            ["python", "manage.py", "runserver", "--noreload"],
            ["python", "-m", "flask", "run"],
            ["python", "-m", "uvicorn", "main:app"],
            ["python", "app.py"],
        ],
        "ready_signals": [
            "Starting development server",
            "Running on http",
            "Uvicorn running on",
            "Application startup complete",
        ],
        "error_patterns": [
            r"Traceback \(most recent call last\)",
            r"\w+Error:",
            r"\w+Exception:",
        ],
    },
    "nodejs": {
        "commands": [
            ["npm", "start"],
            ["node", "server.js"],
            ["node", "app.js"],
            ["node", "index.js"],
        ],
        "ready_signals": [
            "listening on port",
            "Server started",
            "ready on http",
        ],
        "error_patterns": [
            r"Error:",
            r"TypeError:",
            r"Cannot find module",
            r"SyntaxError:",
        ],
    },
    "java": {
        "commands": [
            ["mvn", "spring-boot:run"],
            ["gradle", "bootRun"],
        ],
        "ready_signals": [
            "Started .* in .* seconds",
            "Tomcat started on port",
            "Application is running",
        ],
        "error_patterns": [
            r"Exception in thread",
            r"APPLICATION FAILED TO START",
            r"Error creating bean",
        ],
    },
    "dotnet": {
        "commands": [
            ["dotnet", "run"],
        ],
        "ready_signals": [
            "Now listening on",
            "Application started",
            "Content root path",
        ],
        "error_patterns": [
            r"Unhandled exception",
            r"System\.\w+Exception",
            r"FATAL",
        ],
    },
}


class RuntimeVerifier:
    """
    After build passes, starts the application and checks for runtime errors.

    The check is non-blocking with a hard timeout. If errors are found,
    they're fed back to the RuntimeLogAnalyzer (Enhancement 1) for diagnosis.
    """

    MAX_STARTUP_WAIT = 30  # Maximum seconds to wait for startup
    OUTPUT_BUFFER_SIZE = 50_000  # Max bytes of output to capture

    def __init__(self, workspace_path: str, technology: str = "auto"):
        self.workspace = Path(workspace_path)
        self.technology = technology

    def _detect_technology(self) -> str:
        """Auto-detect project technology from workspace files."""
        if (self.workspace / "requirements.txt").exists() or \
           (self.workspace / "pyproject.toml").exists():
            return "python"
        if (self.workspace / "package.json").exists():
            return "nodejs"
        if (self.workspace / "pom.xml").exists() or \
           (self.workspace / "build.gradle").exists():
            return "java"
        if list(self.workspace.glob("*.csproj")):
            return "dotnet"
        return "python"  # Default

    def _find_startup_command(self, tech: str) -> Optional[List[str]]:
        """Find a working startup command for the technology."""
        config = _STARTUP_CONFIGS.get(tech)
        if not config:
            return None

        for cmd in config["commands"]:
            executable = cmd[0]
            try:
                result = subprocess.run(
                    [executable, "--version"] if executable != "node" else [executable, "-v"],
                    capture_output=True, timeout=5,
                    cwd=str(self.workspace),
                )
                if result.returncode == 0:
                    return cmd
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        return None

    # ------------------------------------------------------------------
    # Main verification
    # ------------------------------------------------------------------

    def verify(self) -> RuntimeCheckResult:
        """
        Start the application and check for runtime errors.

        Returns:
            RuntimeCheckResult with findings.
        """
        tech = self.technology if self.technology != "auto" else self._detect_technology()
        config = _STARTUP_CONFIGS.get(tech)

        if not config:
            logger.info(f"RuntimeVerifier: no config for technology '{tech}'")
            return RuntimeCheckResult(ran_check=False)

        cmd = self._find_startup_command(tech)
        if not cmd:
            logger.info(f"RuntimeVerifier: no working startup command for {tech}")
            return RuntimeCheckResult(ran_check=False)

        logger.info(f"RuntimeVerifier: starting app with {' '.join(cmd)}")
        start_time = time.time()

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            logger.warning(f"RuntimeVerifier: failed to start process: {exc}")
            return RuntimeCheckResult(ran_check=True, started_successfully=False)

        # Read output until ready signal, error, or timeout
        output_lines: List[str] = []
        ready_found = False
        error_found = False
        error_output = ""

        ready_patterns = [re.compile(p, re.IGNORECASE) for p in config["ready_signals"]]
        error_patterns = [re.compile(p) for p in config["error_patterns"]]

        try:
            deadline = time.time() + self.MAX_STARTUP_WAIT
            total_bytes = 0

            while time.time() < deadline:
                if proc.poll() is not None:
                    # Process exited
                    remaining = proc.stdout.read(self.OUTPUT_BUFFER_SIZE)
                    if remaining:
                        output_lines.extend(remaining.splitlines())
                    break

                line = ""
                try:
                    line = proc.stdout.readline()
                except Exception:
                    break

                if not line:
                    time.sleep(0.1)
                    continue

                output_lines.append(line.rstrip())
                total_bytes += len(line)

                if total_bytes > self.OUTPUT_BUFFER_SIZE:
                    break

                # Check for ready signal
                for rp in ready_patterns:
                    if rp.search(line):
                        ready_found = True
                        break

                # Check for error
                for ep in error_patterns:
                    if ep.search(line):
                        error_found = True
                        error_output = line.strip()
                        # Capture a few more lines of context
                        for _ in range(5):
                            try:
                                extra = proc.stdout.readline()
                                if extra:
                                    error_output += "\n" + extra.rstrip()
                            except Exception:
                                break
                        break

                if ready_found or error_found:
                    break

        finally:
            # Always kill the process
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        elapsed_ms = int((time.time() - start_time) * 1000)

        result = RuntimeCheckResult(
            ran_check=True,
            started_successfully=ready_found and not error_found,
            runtime_error_detected=error_found,
            error_output=error_output if error_found else None,
            ready_signal_found=ready_found,
            startup_time_ms=elapsed_ms,
            process_exit_code=proc.returncode,
        )

        if error_found:
            logger.warning(f"RuntimeVerifier: runtime error detected: {error_output[:200]}")
        elif ready_found:
            logger.info(f"RuntimeVerifier: app started successfully in {elapsed_ms}ms")
        else:
            logger.info(f"RuntimeVerifier: timeout after {elapsed_ms}ms, no clear signal")

        return result
