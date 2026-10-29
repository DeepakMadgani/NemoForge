"""
Graph-Driven Build Orchestration & Failure Classification Subsystem for NemoForge.

Separates source-code regressions from environmental, infrastructure,
dependency, and configuration errors before passing execution to repair agents.
"""

from __future__ import annotations

import enum
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ticket_to_code.intelligence.repository_intelligence import RepositoryCapabilityModel
from ticket_to_code.runtime.security import SecurityViolation, run_checked_subprocess

logger = logging.getLogger(__name__)


class FailureCategory(enum.Enum):
    SOURCE_COMPILATION_ERROR = "source_compilation_error"
    DEPENDENCY_RESOLUTION_FAILURE = "dependency_resolution_failure"
    BUILD_CONFIG_FAILURE = "build_config_failure"
    MISSING_TOOLCHAIN = "missing_toolchain"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    TEST_ASSERTION_REGRESSION = "test_assertion_regression"
    TIMEOUT_RESOURCE_EXHAUSTED = "timeout_resource_exhausted"
    UNKNOWN_FAILURE = "unknown_failure"


@dataclass
class ClassifiedBuildResult:
    success: bool
    exit_code: int
    category: FailureCategory
    summary: str
    reproducible_evidence: str
    actionable_for_code_repair: bool
    command_executed: List[str]
    working_directory: str
    duration_ms: int = 0
    stdout: str = ""
    stderr: str = ""
    error_locations: List[Dict[str, Any]] = field(default_factory=list)


class BuildFailureClassifier:
    """Classifies compiler and test runner output into distinct failure domains."""

    @classmethod
    def classify(
        cls,
        exit_code: int,
        stdout: str,
        stderr: str,
        command: List[str],
    ) -> Tuple[FailureCategory, str, bool, List[Dict[str, Any]]]:
        if exit_code == 0:
            return FailureCategory.SOURCE_COMPILATION_ERROR, "Build and checks succeeded with exit code 0", False, []

        combined = f"{stdout}\n{stderr}"
        locations: List[Dict[str, Any]] = []

        # 1. Check Missing Toolchain
        if any(term in combined for term in [
            "command not found", "is not recognized as an internal or external command",
            "No such file or directory: 'mvn'", "No such file or directory: 'npm'",
            "javac: command not found", "Python was not found"
        ]):
            return (
                FailureCategory.MISSING_TOOLCHAIN,
                "Required build toolchain executable is not installed or not in PATH",
                False,
                [],
            )

        # 2. Check Timeout or Resource Exhaustion
        if any(term in combined for term in [
            "timed out", "TimeoutExpired", "java.lang.OutOfMemoryError",
            "GC overhead limit exceeded", "JavaScript heap out of memory"
        ]):
            return (
                FailureCategory.TIMEOUT_RESOURCE_EXHAUSTED,
                "Build process exceeded execution timeout or ran out of system memory",
                False,
                [],
            )

        # 3. Check Infrastructure / External Service / Testcontainers Failure
        if any(term in combined for term in [
            "Connection refused", "connect ECONNREFUSED", "Docker client is not available",
            "Could not connect to Docker", "ContainerFetchException", "RedisConnectionException",
            "Communications link failure", "Failed to connect to localhost",
        ]):
            return (
                FailureCategory.INFRASTRUCTURE_FAILURE,
                "External service or Docker container infrastructure unavailable during test execution",
                False,
                [],
            )

        # 4. Check Dependency Resolution Failure
        if any(term in combined for term in [
            "Could not resolve dependencies", "Failure to find", "npm ERR! ERESOLVE",
            "npm ERR! 404 Not Found", "ResolutionException", "Could not find artifact",
            "ModuleNotFoundError: No module named", "pip._vendor.urllib3.exceptions",
        ]):
            return (
                FailureCategory.DEPENDENCY_RESOLUTION_FAILURE,
                "Package manager could not resolve or download declared dependencies",
                False,
                [],
            )

        # 5. Check Build Configuration Failure
        if any(term in combined for term in [
            "Non-parseable POM", "Project build error", "tsconfig.json does not exist",
            "angular.json: Schema validation failed", "Invalid packaging for parent POM",
            "Malformed pom.xml",
        ]):
            return (
                FailureCategory.BUILD_CONFIG_FAILURE,
                "Build descriptor file (pom.xml, build.gradle, tsconfig.json) contains invalid XML/JSON syntax or schema errors",
                False,
                [],
            )

        # 6. Check Test Assertion Failure (Regression)
        if any(term in combined for term in [
            "AssertionError", "org.opentest4j.AssertionFailedError", "expected: <",
            "junit.framework.AssertionFailedError", "FAIL:", "FAILED (failures=",
            "Expected:", "Received:", "assert ",
        ]):
            # Extract test method locations
            for line in combined.splitlines():
                m = re.search(r"(\w+Test(?:\.\w+)?):.*?(expected|AssertionError)", line)
                if m:
                    locations.append({"test_case": m.group(1), "snippet": line.strip()})
            return (
                FailureCategory.TEST_ASSERTION_REGRESSION,
                "One or more automated test assertions failed",
                True,
                locations,
            )

        # 7. Check Source Compilation / Type Error
        if any(term in combined for term in [
            "COMPILATION ERROR", "[ERROR] /", "cannot find symbol", "symbol:",
            "error TS", "TypeScript error", "SyntaxError:", "TypeError:",
            "incompatible types", "method does not override", "not found: value",
            "JAVA_TYPE_NAME_MISMATCH", "does not match file name", "should be declared in a file named",
            "class, interface, enum, or record expected", "reached end of file while parsing",
        ]):
            for line in combined.splitlines():
                # Matches Java/TypeScript compiler error format: path/File.ext:[line,col] error: ...
                m = re.search(r"([^:\n]+\.(?:java|ts|tsx|js|jsx|py|go|cs)):\[?(\d+)[,:]?(\d+)?\]?:?\s+(.*)", line)
                if m:
                    locations.append({
                        "file": m.group(1).strip(),
                        "line": int(m.group(2)),
                        "column": int(m.group(3)) if m.group(3) else 0,
                        "message": m.group(4).strip(),
                    })
            return (
                FailureCategory.SOURCE_COMPILATION_ERROR,
                "Source code failed static compilation or type checking",
                True,
                locations,
            )

        # 8. Default fallback: Unknown
        return FailureCategory.UNKNOWN_FAILURE, "Unclassified build failure", False, []


class GraphBuildOrchestrator:
    """Executes workspace-aware builds and parses results through the failure classifier."""

    def __init__(self, capability_model: RepositoryCapabilityModel):
        self.capability_model = capability_model

    def execute_build(
        self,
        modified_files: List[str],
        timeout_seconds: int = 180,
    ) -> ClassifiedBuildResult:
        cmd, cwd = self.capability_model.get_reactor_build_scope(modified_files)
        if not cmd:
            return ClassifiedBuildResult(
                success=True,
                exit_code=0,
                category=FailureCategory.SOURCE_COMPILATION_ERROR,
                summary="No build toolchain or test commands required for these files",
                reproducible_evidence="No build command discovered by repository intelligence",
                actionable_for_code_repair=False,
                command_executed=[],
                working_directory=cwd,
            )

        logger.info("Executing graph-aware build command: %s in %s", " ".join(cmd), cwd)
        import time
        start_t = time.time()
        workspace_root = getattr(self.capability_model, "workspace_root", None) or cwd
        try:
            rc, out, err = run_checked_subprocess(
                cmd, cwd=cwd, timeout=timeout_seconds, workspace=workspace_root
            )
            duration_ms = int((time.time() - start_t) * 1000)
            category, summary, actionable, locations = BuildFailureClassifier.classify(rc, out, err, cmd)
            return ClassifiedBuildResult(
                success=(rc == 0),
                exit_code=rc,
                category=category,
                summary=summary,
                reproducible_evidence=(err or out)[-1500:],
                actionable_for_code_repair=actionable,
                command_executed=cmd,
                working_directory=cwd,
                duration_ms=duration_ms,
                stdout=out,
                stderr=err,
                error_locations=locations,
            )
        except SecurityViolation as sv:
            return ClassifiedBuildResult(
                success=False,
                exit_code=126,
                category=FailureCategory.INFRASTRUCTURE_FAILURE,
                summary=f"Build command blocked by security policy: {sv}",
                reproducible_evidence=str(sv),
                actionable_for_code_repair=False,
                command_executed=cmd,
                working_directory=cwd,
            )
        except subprocess.TimeoutExpired as te:
            return ClassifiedBuildResult(
                success=False,
                exit_code=124,
                category=FailureCategory.TIMEOUT_RESOURCE_EXHAUSTED,
                summary=f"Build timed out after {timeout_seconds} seconds",
                reproducible_evidence=str(te),
                actionable_for_code_repair=False,
                command_executed=cmd,
                working_directory=cwd,
            )
        except Exception as e:
            return ClassifiedBuildResult(
                success=False,
                exit_code=1,
                category=FailureCategory.INFRASTRUCTURE_FAILURE,
                summary=f"Failed to spawn build process: {str(e)}",
                reproducible_evidence=str(e),
                actionable_for_code_repair=False,
                command_executed=cmd,
                working_directory=cwd,
            )
