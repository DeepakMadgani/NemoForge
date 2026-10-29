"""
Go and Rust Execution Engines for Ticket-to-Code.

Supports:
- Go projects using go.mod (go build, go test)
- Rust projects using Cargo.toml (cargo check, cargo test)

Author: Deepak Madgani
Date: September 2026
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from ticket_to_code.models import (
    BuildResult,
    BuildStatus,
    TestResult,
    TestStatus,
)
from ticket_to_code.execution.base_executor import (
    ExecutionEngineBase,
    ExecutionEngineFactory,
)

logger = logging.getLogger(__name__)


class GoExecutionEngine(ExecutionEngineBase):
    """Execution engine for Go projects (go build, go test)."""

    def get_technology_name(self) -> str:
        return "Go"

    def _get_project_file_patterns(self) -> List[str]:
        return ["go.mod"]

    def execute_build(
        self,
        project_file: Optional[str] = None,
        timeout: int = 120,
    ) -> BuildResult:
        target_dir = self.workspace
        if project_file:
            cand = self.workspace / project_file
            target_dir = cand.parent if cand.is_file() else cand

        go_bin = shutil.which("go")
        if not go_bin:
            err = "go executable not found on PATH. Install Go or configure environment."
            logger.error(f"[GoExecutionEngine] {err}")
            return BuildResult(
                status=BuildStatus.FAILURE,
                stdout="",
                stderr=err,
                exit_code=127,
                errors=[err],
                warnings=[],
                duration_seconds=0.1,
            )

        cmd = [go_bin, "build", "./..."]
        start_t = time.time()
        try:
            logger.info(f"Executing Go build: '{' '.join(cmd)}' in {target_dir}")
            res = subprocess.run(
                cmd,
                cwd=str(target_dir),
                shell=(sys.platform == "win32"),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start_t
            raw_err = (res.stderr or "").strip()
            errors = [line.strip() for line in raw_err.splitlines() if line.strip()]

            status = BuildStatus.SUCCESS if res.returncode == 0 else BuildStatus.FAILURE
            return BuildResult(
                status=status,
                stdout=res.stdout or "",
                stderr=res.stderr or "",
                exit_code=res.returncode,
                errors=errors,
                warnings=[],
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            return BuildResult(
                status=BuildStatus.FAILURE,
                stdout="",
                stderr=f"Go build timed out after {timeout}s",
                exit_code=-1,
                errors=[f"Go build timed out after {timeout}s"],
                warnings=[],
                duration_seconds=float(timeout),
            )
        except Exception as exc:
            return BuildResult(
                status=BuildStatus.FAILURE,
                stdout="",
                stderr=str(exc),
                exit_code=-1,
                errors=[str(exc)],
                warnings=[],
                duration_seconds=time.time() - start_t,
            )

    def execute_tests(
        self,
        project_file: Optional[str] = None,
        timeout: int = 180,
    ) -> TestResult:
        target_dir = self.workspace
        if project_file:
            cand = self.workspace / project_file
            target_dir = cand.parent if cand.is_file() else cand

        go_bin = shutil.which("go")
        if not go_bin:
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=["go executable not found on PATH"],
                test_output="go executable not found on PATH",
                duration_seconds=0.1,
            )

        cmd = [go_bin, "test", "./..."]
        start_t = time.time()
        try:
            logger.info(f"Executing Go tests: '{' '.join(cmd)}' in {target_dir}")
            res = subprocess.run(
                cmd,
                cwd=str(target_dir),
                shell=(sys.platform == "win32"),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start_t
            raw_out = (res.stdout or "") + "\n" + (res.stderr or "")

            status = TestStatus.ALL_PASSED if res.returncode == 0 else TestStatus.SOME_FAILED
            return TestResult(
                status=status,
                total_tests=1,
                passed=1 if res.returncode == 0 else 0,
                failed=0 if res.returncode == 0 else 1,
                skipped=0,
                errors=[] if res.returncode == 0 else [raw_out[:500]],
                test_output=raw_out[:2000],
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                status=TestStatus.TIMEOUT,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=[f"Go tests timed out after {timeout}s"],
                test_output=f"Go tests timed out after {timeout}s",
                duration_seconds=float(timeout),
            )
        except Exception as exc:
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=[str(exc)],
                test_output=str(exc),
                duration_seconds=time.time() - start_t,
            )


class RustExecutionEngine(ExecutionEngineBase):
    """Execution engine for Rust projects (cargo check, cargo test)."""

    def get_technology_name(self) -> str:
        return "Rust"

    def _get_project_file_patterns(self) -> List[str]:
        return ["Cargo.toml"]

    def execute_build(
        self,
        project_file: Optional[str] = None,
        timeout: int = 120,
    ) -> BuildResult:
        target_dir = self.workspace
        if project_file:
            cand = self.workspace / project_file
            target_dir = cand.parent if cand.is_file() else cand

        cargo_bin = shutil.which("cargo")
        if not cargo_bin:
            err = "cargo executable not found on PATH. Install Rust/Cargo or configure environment."
            logger.error(f"[RustExecutionEngine] {err}")
            return BuildResult(
                status=BuildStatus.FAILURE,
                stdout="",
                stderr=err,
                exit_code=127,
                errors=[err],
                warnings=[],
                duration_seconds=0.1,
            )

        cmd = [cargo_bin, "check", "--message-format=short"]
        start_t = time.time()
        try:
            logger.info(f"Executing Cargo check: '{' '.join(cmd)}' in {target_dir}")
            res = subprocess.run(
                cmd,
                cwd=str(target_dir),
                shell=(sys.platform == "win32"),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start_t
            raw_err = (res.stderr or "").strip()
            errors = [line.strip() for line in raw_err.splitlines() if "error:" in line.lower() or "error[" in line.lower()]

            status = BuildStatus.SUCCESS if res.returncode == 0 else BuildStatus.FAILURE
            return BuildResult(
                status=status,
                stdout=res.stdout or "",
                stderr=res.stderr or "",
                exit_code=res.returncode,
                errors=errors,
                warnings=[],
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            return BuildResult(
                status=BuildStatus.FAILURE,
                stdout="",
                stderr=f"Cargo check timed out after {timeout}s",
                exit_code=-1,
                errors=[f"Cargo check timed out after {timeout}s"],
                warnings=[],
                duration_seconds=float(timeout),
            )
        except Exception as exc:
            return BuildResult(
                status=BuildStatus.FAILURE,
                stdout="",
                stderr=str(exc),
                exit_code=-1,
                errors=[str(exc)],
                warnings=[],
                duration_seconds=time.time() - start_t,
            )

    def execute_tests(
        self,
        project_file: Optional[str] = None,
        timeout: int = 180,
    ) -> TestResult:
        target_dir = self.workspace
        if project_file:
            cand = self.workspace / project_file
            target_dir = cand.parent if cand.is_file() else cand

        cargo_bin = shutil.which("cargo")
        if not cargo_bin:
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=["cargo executable not found on PATH"],
                test_output="cargo executable not found on PATH",
                duration_seconds=0.1,
            )

        cmd = [cargo_bin, "test"]
        start_t = time.time()
        try:
            logger.info(f"Executing Cargo tests: '{' '.join(cmd)}' in {target_dir}")
            res = subprocess.run(
                cmd,
                cwd=str(target_dir),
                shell=(sys.platform == "win32"),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start_t
            raw_out = (res.stdout or "") + "\n" + (res.stderr or "")

            status = TestStatus.ALL_PASSED if res.returncode == 0 else TestStatus.SOME_FAILED
            return TestResult(
                status=status,
                total_tests=1,
                passed=1 if res.returncode == 0 else 0,
                failed=0 if res.returncode == 0 else 1,
                skipped=0,
                errors=[] if res.returncode == 0 else [raw_out[:500]],
                test_output=raw_out[:2000],
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                status=TestStatus.TIMEOUT,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=[f"Cargo tests timed out after {timeout}s"],
                test_output=f"Cargo tests timed out after {timeout}s",
                duration_seconds=float(timeout),
            )
        except Exception as exc:
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=[str(exc)],
                test_output=str(exc),
                duration_seconds=time.time() - start_t,
            )


# Register Go & Rust with the factory
ExecutionEngineFactory.register("go", GoExecutionEngine)
ExecutionEngineFactory.register("golang", GoExecutionEngine)
ExecutionEngineFactory.register("rust", RustExecutionEngine)
ExecutionEngineFactory.register("cargo", RustExecutionEngine)
