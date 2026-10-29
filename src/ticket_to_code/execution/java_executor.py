"""
Java Execution Engine for Maven and Gradle

Supports:
- Maven (mvn clean install, mvn test)
- Gradle (gradle build, gradle test)
- JUnit test parsing
- Spring Boot microservices
- Docker-based validation

Author: Deepak Madgani
Date: April 2026
"""

import os
import logging
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from ticket_to_code.models import BuildResult, TestResult, BuildStatus, TestStatus
from ticket_to_code.execution.base_executor import ExecutionEngineBase

logger = logging.getLogger(__name__)


class JavaMavenExecutor(ExecutionEngineBase):
    """
    Execution engine for Java projects using Maven.
    
    Commands:
    - Build: mvn clean install
    - Test: mvn test
    - Package: mvn package
    """
    
    def get_technology_name(self) -> str:
        return "Java (Maven)"
    
    def _get_project_file_patterns(self) -> list[str]:
        return ["pom.xml"]
    
    def _get_work_dir(self, project_path: str) -> Path:
        """Resolve working directory from project path, handling root and module pom.xml safely."""
        proj_p = Path(project_path)
        if proj_p.name == "pom.xml":
            return (self.workspace_path / proj_p.parent).resolve()
        return (self.workspace_path / proj_p).resolve()

    def _get_mvn_exec(self, work_dir: Path) -> str:
        """Locate Maven executable, checking module and workspace root for wrappers first."""
        if sys.platform == "win32":
            if (work_dir / "mvnw.cmd").exists():
                return str(work_dir / "mvnw.cmd")
            if (self.workspace_path / "mvnw.cmd").exists():
                return str(self.workspace_path / "mvnw.cmd")
        else:
            if (work_dir / "mvnw").exists():
                return str(work_dir / "mvnw")
            if (self.workspace_path / "mvnw").exists():
                return str(self.workspace_path / "mvnw")
        return "mvn"

    def _get_safe_env(self) -> dict:
        from ticket_to_code.runtime.security import CommandSafetyValidator
        return CommandSafetyValidator.sanitize_environment()

    def execute_build(
        self, 
        project_path: str,
        timeout: int = 300
    ) -> BuildResult:
        """
        Execute Maven build.
        
        Args:
            project_path: Path to pom.xml or module directory
            timeout: Build timeout in seconds
            
        Returns:
            BuildResult with status and output
        """
        logger.info(f"Executing Maven build: {project_path}")
        
        start_time = datetime.now()
        
        try:
            # Determine working directory and executable
            work_dir = self._get_work_dir(project_path)
            mvn_exec = self._get_mvn_exec(work_dir)
            
            # Run Maven build
            result = subprocess.run(
                [mvn_exec, "clean", "install", "-DskipTests"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=(sys.platform == "win32"),
                env=self._get_safe_env()
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Determine status
            if result.returncode == 0 and "BUILD SUCCESS" in result.stdout:
                status = BuildStatus.SUCCESS
            else:
                status = BuildStatus.FAILURE
            
            # Extract errors and warnings
            errors = self._extract_maven_errors(result.stdout + result.stderr)
            warnings = self._extract_maven_warnings(result.stdout)
            
            build_result = BuildResult(
                status=status,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                errors=errors,
                warnings=warnings,
                duration_seconds=duration
            )
            
            logger.info(
                f"Maven build {status.value}: {duration:.1f}s, "
                f"{len(errors)} errors, {len(warnings)} warnings"
            )
            return build_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Maven build timeout after {timeout}s")
            return BuildResult(
                status=BuildStatus.TIMEOUT,
                stdout="",
                stderr=f"Build timeout after {timeout} seconds",
                exit_code=-1,
                errors=[f"Build timeout after {timeout} seconds"],
                duration_seconds=timeout
            )
        except FileNotFoundError:
            logger.warning("Maven (mvn) not found on PATH — build validation skipped")
            return BuildResult(
                status=BuildStatus.ERROR,
                stdout="",
                stderr="Maven (mvn) executable was not found on PATH",
                exit_code=-1,
                errors=["Maven (mvn) not installed; build validation cannot run"],
                warnings=["Build validation skipped because Maven is missing"],
                duration_seconds=0
            )
        except Exception as e:
            logger.error(f"Maven build error: {e}")
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
        timeout: int = 600,
        test_target: Optional[str] = None
    ) -> TestResult:
        """
        Execute Maven tests.
        
        Args:
            project_path: Path to pom.xml or module directory
            timeout: Test timeout in seconds
            test_target: Optional test class or pattern (e.g. 'ParticipatingMemberServiceTest')
            
        Returns:
            TestResult with pass/fail counts
        """
        target_info = f" (target: {test_target})" if test_target else ""
        logger.info(f"Executing Maven tests: {project_path}{target_info}")
        
        start_time = datetime.now()
        
        try:
            # Determine working directory and executable
            work_dir = self._get_work_dir(project_path)
            mvn_exec = self._get_mvn_exec(work_dir)
            
            # Run Maven test
            cmd = [mvn_exec, "test"]
            if test_target:
                cmd.append(f"-Dtest={test_target}")
                
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=(sys.platform == "win32"),
                env=self._get_safe_env()
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Parse test results
            test_counts = self._parse_maven_test_output(result.stdout)
            
            total = test_counts['total']
            passed = test_counts['passed']
            failed = test_counts['failed']
            skipped = test_counts['skipped']
            
            # Determine status
            if result.returncode == 0 and failed == 0:
                status = TestStatus.ALL_PASSED
            elif failed > 0 and passed > 0:
                status = TestStatus.SOME_FAILED
            elif passed == 0 and failed > 0:
                status = TestStatus.ALL_FAILED
            else:
                status = TestStatus.ERROR
            
            # Extract error messages
            errors = self._extract_test_failures(result.stdout)
            
            test_result = TestResult(
                status=status,
                total_tests=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                test_output=result.stdout,
                duration_seconds=duration
            )
            
            logger.info(
                f"Maven tests {status.value}: {passed}/{total} passed, "
                f"{failed} failed, {skipped} skipped"
            )
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Maven test timeout after {timeout}s")
            return TestResult(
                status=TestStatus.TIMEOUT,
                total_tests=0,
                passed=0,
                failed=0,
                errors=[f"Test timeout after {timeout} seconds"],
                test_output="",
                duration_seconds=timeout
            )
        except FileNotFoundError:
            logger.warning("Maven (mvn) not found on PATH — test execution skipped")
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                errors=["Maven (mvn) not installed; test execution cannot run"],
                test_output="Maven (mvn) executable was not found on PATH",
                duration_seconds=0
            )
        except Exception as e:
            logger.error(f"Maven test error: {e}")
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                errors=[str(e)],
                test_output="",
                duration_seconds=0
            )
    
    def _extract_maven_errors(self, output: str) -> list[str]:
        """Extract compilation errors from Maven output"""
        errors = []
        for line in output.split('\n'):
            line_str = line.strip()
            if not line_str or '[WARNING]' in line_str:
                continue
            if '[ERROR]' in line_str:
                # Match compilation failure header
                if 'compilation failure' in line_str.lower():
                    errors.append(line_str)
                # Match file location lines: e.g. [ERROR] /path/File.java:[123,45] ... or [ERROR] ...File.java:123: ...
                elif re.search(r'\.(?:java|kt|scala)(?::\[\d+,\d+\]|:\d+:|:\d+)', line_str, re.IGNORECASE):
                    errors.append(line_str)
                # Match compiler symbol / context lines: [ERROR] symbol: ... or [ERROR] location: ...
                elif re.search(r'\[ERROR\]\s+(?:symbol|location|cannot find|incompatible|package|class|method|expected)\b', line_str, re.IGNORECASE):
                    errors.append(line_str)
                elif re.search(r'\[\d+,\d+\]', line_str):
                    errors.append(line_str)
        # If no specific error lines matched, capture any [ERROR] line that is not a generic summary
        if not errors:
            for line in output.split('\n'):
                line_str = line.strip()
                if '[ERROR]' in line_str and not line_str.startswith('[ERROR] -> [Help 1]'):
                    errors.append(line_str)
        return errors[:30]  # Limit to first 30
    
    def _extract_maven_warnings(self, output: str) -> list[str]:
        """Extract warnings from Maven output"""
        warnings = []
        for line in output.split('\n'):
            if '[WARNING]' in line:
                warnings.append(line.strip())
        return warnings[:20]  # Limit to first 20
    
    def _parse_maven_test_output(self, output: str) -> dict:
        """
        Parse Maven test output for counts.
        
        Example:
        Tests run: 5, Failures: 0, Errors: 0, Skipped: 0
        """
        counts = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
        
        # Look for summary line
        pattern = r'Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)'
        matches = re.findall(pattern, output)
        
        if matches:
            # Take the last match (final summary)
            total, failures, errors, skipped = map(int, matches[-1])
            
            counts['total'] = total
            counts['failed'] = failures + errors  # Combine failures and errors
            counts['skipped'] = skipped
            counts['passed'] = total - counts['failed'] - counts['skipped']
        
        return counts
    
    def _extract_test_failures(self, output: str) -> list[str]:
        """Extract test failure messages"""
        errors = []
        in_failure = False
        
        for line in output.split('\n'):
            if 'Failed tests:' in line or 'Tests in error:' in line:
                in_failure = True
            elif in_failure:
                if line.strip() and not line.startswith('['):
                    errors.append(line.strip())
                if line.startswith('[INFO]') or line.startswith('[ERROR]'):
                    in_failure = False
        
        return errors[:10]  # Limit to first 10


class JavaGradleExecutor(ExecutionEngineBase):
    """
    Execution engine for Java projects using Gradle.
    
    Commands:
    - Build: gradle build
    - Test: gradle test
    - Clean: gradle clean
    """
    
    def get_technology_name(self) -> str:
        return "Java (Gradle)"
    
    def _get_project_file_patterns(self) -> list[str]:
        return ["build.gradle", "build.gradle.kts"]
    
    def _get_work_dir(self, project_path: str) -> Path:
        """Resolve working directory from project path, handling root and module build.gradle safely."""
        proj_p = Path(project_path)
        if proj_p.name in ("build.gradle", "build.gradle.kts"):
            return (self.workspace_path / proj_p.parent).resolve()
        return (self.workspace_path / proj_p).resolve()

    def _get_gradle_exec(self, work_dir: Path) -> str:
        """Locate Gradle executable, checking module and workspace root for gradlew first."""
        if sys.platform == "win32":
            if (work_dir / "gradlew.bat").exists():
                return str(work_dir / "gradlew.bat")
            if (self.workspace_path / "gradlew.bat").exists():
                return str(self.workspace_path / "gradlew.bat")
        else:
            if (work_dir / "gradlew").exists():
                return str(work_dir / "gradlew")
            if (self.workspace_path / "gradlew").exists():
                return str(self.workspace_path / "gradlew")
        return "gradle"

    def execute_build(
        self, 
        project_path: str,
        timeout: int = 300
    ) -> BuildResult:
        """Execute Gradle build"""
        logger.info(f"Executing Gradle build: {project_path}")
        
        start_time = datetime.now()
        
        try:
            # Determine working directory and executable
            work_dir = self._get_work_dir(project_path)
            gradle_exec = self._get_gradle_exec(work_dir)
            
            # Run Gradle build (compile source and test classes, exclude test execution)
            result = subprocess.run(
                [gradle_exec, "build", "-x", "test", "testClasses"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=(sys.platform == "win32"),
                env=self._get_safe_env()
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Determine status
            if result.returncode == 0 and "BUILD SUCCESSFUL" in result.stdout:
                status = BuildStatus.SUCCESS
                errors = []
            else:
                status = BuildStatus.FAILURE
                errors = self._extract_gradle_errors(result.stdout + result.stderr)
            
            build_result = BuildResult(
                status=status,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                errors=errors,
                duration_seconds=duration
            )
            
            logger.info(f"Gradle build {status.value}: {duration:.1f}s")
            return build_result
            
        except subprocess.TimeoutExpired:
            return BuildResult(
                status=BuildStatus.TIMEOUT,
                stdout="",
                stderr=f"Build timeout after {timeout} seconds",
                exit_code=-1,
                errors=[f"Build timeout after {timeout} seconds"],
                duration_seconds=timeout
            )
        except Exception as e:
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
        timeout: int = 600,
        test_target: Optional[str] = None
    ) -> TestResult:
        """Execute Gradle tests"""
        target_info = f" (target: {test_target})" if test_target else ""
        logger.info(f"Executing Gradle tests: {project_path}{target_info}")
        
        start_time = datetime.now()
        
        try:
            work_dir = self._get_work_dir(project_path)
            gradle_exec = self._get_gradle_exec(work_dir)
            
            # Run Gradle test
            cmd = [gradle_exec, "test"]
            if test_target:
                cmd.extend(["--tests", test_target])
                
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=(sys.platform == "win32"),
                env=self._get_safe_env()
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            combined_output = result.stdout + "\n" + result.stderr
            
            # Parse test results
            test_counts = self._parse_gradle_test_output(combined_output, work_dir)
            
            total = test_counts['total']
            passed = test_counts['passed']
            failed = test_counts['failed']
            skipped = test_counts['skipped']
            failed_tests = test_counts.get('failed_tests', [])
            
            # Determine status
            if result.returncode == 0 and failed == 0:
                status = TestStatus.ALL_PASSED
                if total == 0:
                    total = max(passed, 1)
                    passed = total
            else:
                if failed == 0:
                    failed = max(len(failed_tests), 1)
                    total = max(total, failed + passed)
                if passed > 0:
                    status = TestStatus.SOME_FAILED
                else:
                    status = TestStatus.ALL_FAILED
            
            test_errors = [f"Test failed: {t}" for t in failed_tests] if failed_tests else []
            if not test_errors and status != TestStatus.ALL_PASSED:
                test_errors = [f"Gradle test task exited with code {result.returncode}"]
            
            test_result = TestResult(
                status=status,
                total_tests=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=test_errors,
                test_output=combined_output,
                duration_seconds=duration
            )
            
            logger.info(f"Gradle tests {status.value}: {passed}/{total} passed")
            return test_result
            
        except subprocess.TimeoutExpired:
            return TestResult(
                status=TestStatus.TIMEOUT,
                total_tests=0,
                passed=0,
                failed=0,
                errors=[f"Test timeout after {timeout} seconds"],
                test_output="",
                duration_seconds=timeout
            )
        except Exception as e:
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                errors=[str(e)],
                test_output="",
                duration_seconds=0
            )
    
    def _extract_gradle_errors(self, output: str) -> list[str]:
        """Extract errors from Gradle output"""
        errors = []
        for line in output.split('\n'):
            line_str = line.strip()
            if not line_str:
                continue
            if 'warning:' in line_str.lower() and 'error:' not in line_str.lower():
                continue
            if 'note:' in line_str.lower() and 'error:' not in line_str.lower():
                continue
            if 'error:' in line_str.lower() or 'FAILED' in line_str:
                errors.append(line_str)
            elif re.search(r'\.(?:java|kt|scala)(?::\[\d+,\d+\]|:\d+:|:\d+)', line_str, re.IGNORECASE):
                errors.append(line_str)
            elif re.search(r'^\s*(?:symbol|location|cannot find|incompatible|package|class|method|expected)\b', line_str, re.IGNORECASE):
                errors.append(line_str)
        return errors[:30]
    
    def _parse_gradle_test_output(self, output: str, work_dir: Optional[str] = None) -> dict:
        """Parse Gradle test output"""
        counts = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'failed_tests': []}
        
        # Pattern: "5 tests completed, 0 failed, 0 skipped"
        pattern = r'(\d+) tests? completed, (\d+) failed, (\d+) skipped'
        match = re.search(pattern, output)
        
        if match:
            total = int(match.group(1))
            failed = int(match.group(2))
            skipped = int(match.group(3))
            
            counts['total'] = total
            counts['failed'] = failed
            counts['skipped'] = skipped
            counts['passed'] = total - failed - skipped
        
        # Also parse individual test failures: "ClassName > methodName() FAILED"
        failures = re.findall(r'(\S+)\s*>\s*([^\s\(\)]+)(?:\([^\)]*\))?\s*FAILED', output)
        for cls, mth in failures:
            counts['failed_tests'].append(f"{cls}.{mth}")
            
        # Parse XML test results from build/test-results/test if available
        if work_dir:
            test_results_dir = os.path.join(work_dir, "build", "test-results", "test")
            if os.path.isdir(test_results_dir):
                import xml.etree.ElementTree as ET
                xml_total, xml_failed, xml_skipped = 0, 0, 0
                for xml_file in os.listdir(test_results_dir):
                    if xml_file.endswith(".xml"):
                        try:
                            tree = ET.parse(os.path.join(test_results_dir, xml_file))
                            root = tree.getroot()
                            xml_total += int(root.attrib.get('tests', 0))
                            xml_failed += int(root.attrib.get('failures', 0)) + int(root.attrib.get('errors', 0))
                            xml_skipped += int(root.attrib.get('skipped', 0))
                        except Exception:
                            pass
                if xml_total > 0:
                    counts['total'] = max(counts['total'], xml_total)
                    counts['failed'] = max(counts['failed'], xml_failed)
                    counts['skipped'] = max(counts['skipped'], xml_skipped)
                    counts['passed'] = max(counts['passed'], xml_total - xml_failed - xml_skipped)
        
        if counts['failed_tests']:
            counts['failed'] = max(counts['failed'], len(counts['failed_tests']))
            counts['total'] = max(counts['total'], counts['failed'] + counts['passed'])
        
        return counts


# Register Java executors with factory
from ticket_to_code.execution.base_executor import ExecutionEngineFactory

ExecutionEngineFactory.register("java", JavaMavenExecutor)
ExecutionEngineFactory.register("java-maven", JavaMavenExecutor)
ExecutionEngineFactory.register("java-gradle", JavaGradleExecutor)
