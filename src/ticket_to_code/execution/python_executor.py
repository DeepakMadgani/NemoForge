"""
Python Execution Engine for pip/pytest/setuptools/poetry projects

Supports:
- pip install (requirements.txt, setup.py, pyproject.toml)
- pytest test execution and parsing
- Python syntax checking (py_compile)
- Virtual environment awareness

Author: Deepak Madgani
Date: July 2026
"""

import logging
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from ticket_to_code.models import BuildResult, TestResult, BuildStatus, TestStatus
from ticket_to_code.execution.base_executor import ExecutionEngineBase

logger = logging.getLogger(__name__)


class PythonExecutionEngine(ExecutionEngineBase):
    """
    Execution engine for Python projects.
    
    Build strategies (in order of preference):
    1. pyproject.toml → pip install -e .
    2. setup.py → pip install -e .
    3. requirements.txt → pip install -r requirements.txt
    4. Fallback → py_compile syntax check on all .py files
    
    Test strategies:
    1. pytest (preferred)
    2. python -m unittest discover
    """
    
    def get_technology_name(self) -> str:
        return "Python"
    
    def _get_project_file_patterns(self) -> list[str]:
        return ["pyproject.toml", "setup.py", "requirements.txt"]
    
    def _resolve_work_dir(self, project_path: str) -> Path:
        """Resolve the working directory from a project_path argument."""
        p = Path(project_path)
        # If an absolute path was given, use it directly
        if p.is_absolute():
            return p if p.is_dir() else p.parent
        # Otherwise treat it relative to workspace
        full = self.workspace_path / project_path
        if full.is_file():
            return full.parent
        return full
    
    def _find_python(self) -> str:
        """Return the best Python executable to use."""
        return sys.executable  # use the same Python that's running the agent
    
    def execute_build(
        self,
        project_path: str,
        timeout: int = 300
    ) -> BuildResult:
        """
        Execute Python build / dependency install.
        
        Strategy:
        1. If requirements.txt exists → pip install -r requirements.txt --dry-run
        2. If pyproject.toml / setup.py → pip install -e . --dry-run
        3. Always run py_compile on every .py file as a syntax check
        """
        logger.info(f"Executing Python build check: {project_path}")
        start_time = datetime.now()
        
        try:
            work_dir = self._resolve_work_dir(project_path)
            python = self._find_python()
            all_stdout: list[str] = []
            all_stderr: list[str] = []
            all_errors: list[str] = []
            all_warnings: list[str] = []
            overall_ok = True
            
            # ── Phase 1: Dependency check ──────────────────────────────
            req_file = work_dir / "requirements.txt"
            pyproject = work_dir / "pyproject.toml"
            setup_py = work_dir / "setup.py"
            
            if req_file.exists():
                logger.info("Found requirements.txt — checking dependencies")
                result = subprocess.run(
                    [python, "-m", "pip", "install", "-r", "requirements.txt", "--dry-run", "--quiet"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                all_stdout.append(result.stdout)
                all_stderr.append(result.stderr)
                if result.returncode != 0:
                    overall_ok = False
                    dep_errors = self._extract_pip_errors(result.stderr)
                    all_errors.extend(dep_errors)
            elif pyproject.exists() or setup_py.exists():
                logger.info("Found pyproject.toml/setup.py — checking package install")
                result = subprocess.run(
                    [python, "-m", "pip", "install", "-e", ".", "--dry-run", "--quiet"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                all_stdout.append(result.stdout)
                all_stderr.append(result.stderr)
                if result.returncode != 0:
                    overall_ok = False
                    dep_errors = self._extract_pip_errors(result.stderr)
                    all_errors.extend(dep_errors)
            
            # ── Phase 2: Syntax check on all .py files ──────────────
            py_files = list(work_dir.rglob("*.py"))
            # Exclude venv / __pycache__ / .git
            ignore_dirs = {".venv", "venv", "env", ".env", "__pycache__", ".git", "node_modules", ".tox"}
            py_files = [
                f for f in py_files
                if not any(part in ignore_dirs for part in f.relative_to(work_dir).parts)
            ]
            
            syntax_failures: list[str] = []
            for py_file in py_files:
                result = subprocess.run(
                    [python, "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    syntax_failures.append(f"Syntax error in {py_file.relative_to(work_dir)}: {error_msg}")
            
            if syntax_failures:
                overall_ok = False
                all_errors.extend(syntax_failures)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            status = BuildStatus.SUCCESS if overall_ok else BuildStatus.FAILURE
            
            build_output = "\n".join(all_stdout)
            build_stderr = "\n".join(all_stderr)
            
            if overall_ok:
                build_output += f"\n\nPython build check PASSED — {len(py_files)} files checked, 0 syntax errors"
            else:
                build_output += f"\n\nPython build check FAILED — {len(all_errors)} error(s)"
            
            build_result = BuildResult(
                status=status,
                stdout=build_output,
                stderr=build_stderr,
                exit_code=0 if overall_ok else 1,
                errors=all_errors,
                warnings=all_warnings,
                duration_seconds=duration
            )
            
            logger.info(
                f"Python build {status.value}: {duration:.1f}s, "
                f"{len(all_errors)} errors, {len(all_warnings)} warnings, "
                f"{len(py_files)} files syntax-checked"
            )
            return build_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Python build timeout after {timeout}s")
            return BuildResult(
                status=BuildStatus.TIMEOUT,
                stdout="",
                stderr=f"Build timeout after {timeout} seconds",
                exit_code=-1,
                errors=[f"Build timeout after {timeout} seconds"],
                duration_seconds=timeout
            )
        except Exception as e:
            logger.error(f"Python build error: {e}")
            return BuildResult(
                status=BuildStatus.ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                errors=[str(e)],
                duration_seconds=0
            )
    
    def execute_tests(
        self,
        project_path: str,
        timeout: int = 600
    ) -> TestResult:
        """
        Execute Python tests using pytest (preferred) or unittest.
        """
        logger.info(f"Executing Python tests: {project_path}")
        start_time = datetime.now()
        
        try:
            work_dir = self._resolve_work_dir(project_path)
            python = self._find_python()
            
            # Try pytest first
            result = subprocess.run(
                [python, "-m", "pytest", "-v", "--tb=short", "--no-header"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Parse pytest output
            counts = self._parse_pytest_output(result.stdout)
            total = counts["total"]
            passed = counts["passed"]
            failed = counts["failed"]
            skipped = counts["skipped"]
            
            # If pytest found 0 tests, try unittest as fallback
            if total == 0:
                result = subprocess.run(
                    [python, "-m", "unittest", "discover", "-v"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                duration = (datetime.now() - start_time).total_seconds()
                counts = self._parse_unittest_output(result.stderr)
                total = counts["total"]
                passed = counts["passed"]
                failed = counts["failed"]
                skipped = counts["skipped"]
            
            # Determine status
            if result.returncode == 0 and failed == 0:
                status = TestStatus.ALL_PASSED
            elif failed > 0 and passed > 0:
                status = TestStatus.SOME_FAILED
            elif passed == 0 and failed > 0:
                status = TestStatus.ALL_FAILED
            else:
                status = TestStatus.ALL_PASSED  # 0 tests is OK
            
            test_result = TestResult(
                status=status,
                total_tests=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=[],
                test_output=result.stdout + result.stderr,
                duration_seconds=duration,
            )
            
            logger.info(
                f"Python tests {status.value}: {total} total, "
                f"{passed} passed, {failed} failed, {skipped} skipped"
            )
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Python test timeout after {timeout}s")
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=[f"Test timeout after {timeout} seconds"],
                test_output="",
                duration_seconds=timeout,
            )
        except Exception as e:
            logger.error(f"Python test error: {e}")
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=[str(e)],
                test_output="",
                duration_seconds=0,
            )
    
    # ─── Private helpers ─────────────────────────────────────────
    
    @staticmethod
    def _extract_pip_errors(stderr: str) -> list[str]:
        """Extract meaningful error messages from pip output."""
        errors: list[str] = []
        for line in stderr.splitlines():
            line = line.strip()
            if not line:
                continue
            # pip ERROR lines
            if line.startswith("ERROR:") or "No matching distribution" in line:
                errors.append(line)
            # ModuleNotFoundError during install
            elif "ModuleNotFoundError" in line or "ImportError" in line:
                errors.append(line)
            elif "Could not find a version" in line:
                errors.append(line)
        if not errors and stderr.strip():
            # Fallback: take first 5 lines
            errors = [l.strip() for l in stderr.splitlines()[:5] if l.strip()]
        return errors
    
    @staticmethod
    def _parse_pytest_output(stdout: str) -> dict:
        """Parse pytest -v output for test counts."""
        counts = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        
        # pytest summary line: "= 5 passed, 2 failed, 1 skipped in 1.23s ="
        # or "= 16 passed in 0.05s ="
        summary_match = re.search(
            r"=+\s*(.*?)\s+in\s+[\d.]+s\s*=+",
            stdout
        )
        if summary_match:
            summary = summary_match.group(1)
            for part in summary.split(","):
                part = part.strip()
                m = re.match(r"(\d+)\s+(\w+)", part)
                if m:
                    count = int(m.group(1))
                    label = m.group(2).lower()
                    if label == "passed":
                        counts["passed"] = count
                    elif label == "failed":
                        counts["failed"] = count
                    elif label in ("skipped", "deselected"):
                        counts["skipped"] += count
                    elif label == "error" or label == "errors":
                        counts["failed"] += count
            counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"]
        
        return counts
    
    @staticmethod
    def _parse_unittest_output(stderr: str) -> dict:
        """Parse unittest output for test counts."""
        counts = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        
        # unittest summary: "Ran 5 tests in 0.001s"
        ran_match = re.search(r"Ran (\d+) tests? in", stderr)
        if ran_match:
            counts["total"] = int(ran_match.group(1))
        
        # "OK" or "FAILED (failures=2, errors=1)"
        if "OK" in stderr and "FAILED" not in stderr:
            counts["passed"] = counts["total"]
        else:
            fail_match = re.search(r"failures=(\d+)", stderr)
            err_match = re.search(r"errors=(\d+)", stderr)
            skip_match = re.search(r"skipped=(\d+)", stderr)
            if fail_match:
                counts["failed"] += int(fail_match.group(1))
            if err_match:
                counts["failed"] += int(err_match.group(1))
            if skip_match:
                counts["skipped"] = int(skip_match.group(1))
            counts["passed"] = counts["total"] - counts["failed"] - counts["skipped"]
        
        return counts


# ── Register with the factory ────────────────────────────────────
from ticket_to_code.execution.base_executor import ExecutionEngineFactory

ExecutionEngineFactory.register("python", PythonExecutionEngine)
ExecutionEngineFactory.register("python-pip", PythonExecutionEngine)
ExecutionEngineFactory.register("python-poetry", PythonExecutionEngine)
