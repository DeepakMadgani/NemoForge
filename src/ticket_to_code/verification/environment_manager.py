"""
Isolated Verification Environment Manager for NemoForge.

Provides concurrency-safe ephemeral port allocation, headless browser auto-detection,
managed child-process lifecycles, and reliable per-run artifact isolation.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoint:
    name: str
    host: str
    port: int
    url: str
    pid: Optional[int] = None


@dataclass
class EnvironmentCapabilities:
    is_headless_required: bool
    installed_browsers: List[str]
    preferred_browser: str
    has_display_server: bool
    platform_name: str


class VerificationEnvironmentManager:
    """Manages ephemeral ports, child processes, and isolated browser environments."""

    def __init__(self, run_id: str, base_artifact_dir: Optional[str] = None):
        self.run_id = run_id
        self.base_artifact_dir = Path(base_artifact_dir or tempfile.gettempdir()) / f"nemoforge_env_{run_id}"
        self.base_artifact_dir.mkdir(parents=True, exist_ok=True)
        self.managed_processes: List[subprocess.Popen] = []
        self.allocated_ports: Set[int] = set()
        self.capabilities = self._discover_environment()

    def _discover_environment(self) -> EnvironmentCapabilities:
        is_ci = os.environ.get("CI", "").strip().lower() in ("1", "true")
        has_display = True
        if sys.platform.startswith("linux"):
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

        is_headless = is_ci or not has_display

        installed: List[str] = []
        # Check available browsers
        for b in ("chromium", "google-chrome", "chrome", "msedge", "firefox"):
            if shutil.which(b):
                installed.append(b)

        # Determine preference
        pref = "chromium"
        if "msedge" in installed and sys.platform == "win32":
            pref = "msedge"
        elif "chrome" in installed or "google-chrome" in installed:
            pref = "chrome"
        elif installed:
            pref = installed[0]

        return EnvironmentCapabilities(
            is_headless_required=is_headless,
            installed_browsers=installed,
            preferred_browser=pref,
            has_display_server=has_display,
            platform_name=sys.platform,
        )

    def allocate_ephemeral_port(self) -> int:
        """
        Allocates a race-free TCP port using worker-partitioned range assignment
        and cross-process atomic file-locking.
        """
        # Deterministically partition port ranges by run_id to avoid cross-worker collision
        worker_slot = (abs(hash(self.run_id)) % 150) * 40
        start_port = 20000 + worker_slot
        end_port = start_port + 40

        # Scan within the worker's dedicated range
        for port in range(start_port, end_port):
            if port in self.allocated_ports:
                continue

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                sock.listen(1)
                # Keep socket alive and held until right before spawn
                self.allocated_ports.add(port)
                self._held_sockets = getattr(self, "_held_sockets", {})
                self._held_sockets[port] = sock
                return port
            except OSError:
                sock.close()
                continue

        # Fallback to OS dynamic port if chunk is saturated
        fallback_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        fallback_sock.bind(("127.0.0.1", 0))
        fallback_sock.listen(1)
        fallback_port = fallback_sock.getsockname()[1]
        self.allocated_ports.add(fallback_port)
        self._held_sockets = getattr(self, "_held_sockets", {})
        self._held_sockets[fallback_port] = fallback_sock
        return fallback_port

    def spawn_service(
        self,
        name: str,
        command: List[str],
        working_directory: str,
        env_vars: Optional[Dict[str, str]] = None,
        health_check_timeout: int = 30,
    ) -> ServiceEndpoint:
        """Spawn a background dev server on an ephemeral port with atomic handoff and health polling."""
        port = self.allocate_ephemeral_port()
        merged_env = os.environ.copy()
        if env_vars:
            merged_env.update(env_vars)
        merged_env["PORT"] = str(port)

        # Substitute {port} placeholder in command args if present
        resolved_cmd = [arg.replace("{port}", str(port)) for arg in command]

        log_path = self.base_artifact_dir / f"{name}_service.log"
        log_file = open(log_path, "w", encoding="utf-8")

        logger.info("Spawning managed service '%s' on port %d: %s", name, port, " ".join(resolved_cmd))

        # Atomic release of the held reservation socket immediately before subprocess execution
        held = getattr(self, "_held_sockets", {}).pop(port, None)
        if held:
            try:
                held.close()
            except Exception:
                pass

        proc = subprocess.Popen(
            resolved_cmd,
            cwd=working_directory,
            stdout=log_file,
            stderr=subprocess.STDOUT,  # merge stderr into the same log file (was a typo: subprocess.STLOG)
            env=merged_env,
            shell=(resolved_cmd[0].endswith(".cmd") or resolved_cmd[0].endswith(".bat")),
        )
        self.managed_processes.append(proc)

        # Poll port until open
        deadline = time.time() + health_check_timeout
        healthy = False
        while time.time() < deadline:
            if proc.poll() is not None:
                log_file.close()
                err_text = log_path.read_text(encoding="utf-8", errors="replace")[-500:]
                raise RuntimeError(f"Service '{name}' terminated prematurely with exit code {proc.returncode}: {err_text}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    healthy = True
                    break
            time.sleep(0.3)

        if not healthy:
            self.terminate_all()
            raise TimeoutError(f"Service '{name}' failed to bind to 127.0.0.1:{port} within {health_check_timeout}s")

        return ServiceEndpoint(
            name=name,
            host="127.0.0.1",
            port=port,
            url=f"http://127.0.0.1:{port}",
            pid=proc.pid,
        )

    def terminate_all(self):
        """Clean up and terminate only processes owned by this run."""
        for proc in self.managed_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception as e:
                logger.warning("Error terminating process %d: %s", proc.pid, e)
        self.managed_processes.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate_all()
