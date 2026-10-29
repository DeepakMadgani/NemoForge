"""
Node.js & TypeScript Execution Engine.

Executes builds and tests using npm, npx, tsc, or Angular CLI.
Replaces legacy sourceless bytecode with production-grade execution logic.

Author: Deepak Madgani
Date: September 2026
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
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


class NodejsExecutionEngine(ExecutionEngineBase):
    """Execution engine for Node.js, TypeScript, and Angular projects."""

    def get_technology_name(self) -> str:
        return "Node.js"

    def _get_project_file_patterns(self) -> List[str]:
        return ["package.json"]

    def execute_build(
        self,
        project_file: Optional[str] = None,
        timeout: int = 120,
    ) -> BuildResult:
        """
        Execute build using npm run build or tsc --noEmit.
        """
        target_dir = self.workspace
        if project_file:
            cand = self.workspace / project_file
            target_dir = cand.parent if cand.is_file() else cand

        use_shell = sys.platform == "win32"
        pkg_json = target_dir / "package.json"
        has_pkg = pkg_json.exists()
        has_tsconfig = (target_dir / "tsconfig.json").exists()

        build_cmd: Optional[str] = None
        if has_pkg:
            try:
                pkg_data = json.loads(pkg_json.read_text(encoding="utf-8"))
                scripts = pkg_data.get("scripts", {})
                if "build" in scripts:
                    build_cmd = "npm run build"
            except Exception as e:
                logger.debug(f"Failed to read package.json in {target_dir}: {e}")

        if not build_cmd and has_tsconfig:
            build_cmd = "npx --no-install tsc --noEmit --skipLibCheck"

        if not build_cmd:
            logger.info(f"No npm build script or tsconfig.json found in {target_dir}; clean.")
            return BuildResult(
                status=BuildStatus.SUCCESS,
                stdout="No build script specified; assumed clean",
                stderr="",
                exit_code=0,
                errors=[],
                warnings=[],
                duration_seconds=0.1,
            )

        try:
            logger.info(f"Executing Node build: '{build_cmd}' in {target_dir}")
            res = subprocess.run(
                build_cmd,
                cwd=str(target_dir),
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            raw_out = (res.stdout or "") + "\n" + (res.stderr or "")
            errors: List[str] = []
            for line in raw_out.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if ": error TS" in stripped or "ERROR in" in stripped or "error:" in stripped.lower():
                    errors.append(stripped)

            status = BuildStatus.SUCCESS if res.returncode == 0 else BuildStatus.FAILED
            return BuildResult(
                status=status,
                stdout=res.stdout or "",
                stderr=res.stderr or "",
                exit_code=res.returncode,
                errors=errors,
                warnings=[],
                duration_seconds=1.0,
            )
        except subprocess.TimeoutExpired:
            return BuildResult(
                status=BuildStatus.FAILED,
                stdout="",
                stderr=f"Build timed out after {timeout}s",
                exit_code=-1,
                errors=[f"Build timed out after {timeout}s"],
                warnings=[],
                duration_seconds=float(timeout),
            )
        except Exception as exc:
            return BuildResult(
                status=BuildStatus.FAILED,
                stdout="",
                stderr=str(exc),
                exit_code=-1,
                errors=[str(exc)],
                warnings=[],
                duration_seconds=0.1,
            )

    def execute_tests(
        self,
        project_file: Optional[str] = None,
        test_target: Optional[str] = None,
        timeout: int = 180,
    ) -> TestResult:
        """
        Execute tests using npm test or jest.
        """
        target_dir = self.workspace
        if project_file:
            cand = self.workspace / project_file
            target_dir = cand.parent if cand.is_file() else cand

        use_shell = sys.platform == "win32"
        pkg_json = target_dir / "package.json"
        test_cmd: Optional[str] = None

        if pkg_json.exists():
            try:
                pkg_data = json.loads(pkg_json.read_text(encoding="utf-8"))
                scripts = pkg_data.get("scripts", {})
                if "test" in scripts:
                    is_angular = (target_dir / "angular.json").exists()
                    if test_target and is_angular:
                        test_cmd = f"npm test -- --include {test_target} --watch=false --browsers=ChromeHeadless"
                    elif test_target:
                        test_cmd = f"npm test -- {test_target}"
                    else:
                        test_cmd = "npm test -- --watch=false --browsers=ChromeHeadless"
            except Exception as e:
                logger.debug(f"Failed to read package.json in {target_dir}: {e}")

        if not test_cmd:
            return TestResult(
                status=TestStatus.NO_TESTS,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=[],
                test_output="No test script found in package.json",
                duration_seconds=0.1,
            )

        try:
            logger.info(f"Executing Node tests: '{test_cmd}' in {target_dir}")
            res = subprocess.run(
                test_cmd,
                cwd=str(target_dir),
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            raw = (res.stdout or "") + "\n" + (res.stderr or "")
            status = TestStatus.ALL_PASSED if res.returncode == 0 else TestStatus.SOME_FAILED
            return TestResult(
                status=status,
                total_tests=1,
                passed=1 if res.returncode == 0 else 0,
                failed=0 if res.returncode == 0 else 1,
                skipped=0,
                errors=[] if res.returncode == 0 else [raw[:500]],
                test_output=raw[:2000],
                duration_seconds=1.0,
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                status=TestStatus.TIMEOUT,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=[f"Node tests timed out after {timeout}s"],
                test_output=f"Node tests timed out after {timeout}s",
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
                duration_seconds=0.1,
            )


# Register engine with factory
ExecutionEngineFactory.register("nodejs", NodejsExecutionEngine)
ExecutionEngineFactory.register("angular", NodejsExecutionEngine)
ExecutionEngineFactory.register("typescript", NodejsExecutionEngine)
ExecutionEngineFactory.register("javascript", NodejsExecutionEngine)
