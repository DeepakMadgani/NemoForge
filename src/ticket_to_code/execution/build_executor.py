"""
.NET Execution Engine - Smart Build Tool Selection

Automatically detects .NET Framework vs .NET Core and uses appropriate build tool:
- .NET Framework 4.x → msbuild
- .NET Core/.NET 5+ → dotnet build

Supports:
- Enterprise (.NET Framework 4.7.2, 78 projects)
- DataSlave 2.sln with mixed C# and VB.NET
- Any .NET project with automatic detection

Author: Deepak Madgani
Date: May 2026
"""

import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from ticket_to_code.models import BuildResult, TestResult, BuildStatus, TestStatus, ExecutionError
from ticket_to_code.execution.base_executor import ExecutionEngineBase

logger = logging.getLogger(__name__)


class DotNetExecutionEngine(ExecutionEngineBase):
    """
    Smart .NET Execution Engine with Auto-Detection
    
    Automatically detects and uses correct build tool:
    - .NET Framework 4.x → Uses msbuild
    - .NET Core/.NET 5+ → Uses dotnet build
    
    Real-world examples handled:
    - Enterprise: .NET Framework 4.7.2, DataSlave 2.sln, 78 projects
    - Modern projects: .NET 6/7/8 with dotnet CLI
    - Mixed C# + VB.NET solutions
    
    Returns structured results for AI analysis.
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize execution engine.
        
        Args:
            workspace_path: Path to code workspace
        """
        super().__init__(workspace_path)
        self.workspace_path = Path(workspace_path)
        if not self.workspace_path.exists():
            raise ValueError(f"Workspace path does not exist: {workspace_path}")
        
        logger.info(f".NET Execution Engine initialized: {workspace_path}")
    
    def _detect_dotnet_type(self, project_path: str) -> Tuple[str, str]:
        """
        Detect if project is .NET Framework or .NET Core.
        
        Args:
            project_path: Path to .sln or .csproj file
            
        Returns:
            Tuple of (framework_type, version)
            - ("framework", "4.7.2") for .NET Framework
            - ("core", "8.0") for .NET Core/5+
        """
        project_file = Path(project_path)
        
        # If .sln, find first .csproj to analyze
        if project_file.suffix == ".sln":
            solution_dir = project_file.parent if project_file.is_absolute() else self.workspace_path / project_file.parent
            csproj_files = list(solution_dir.glob("**/*.csproj"))
            if csproj_files:
                project_file = csproj_files[0]
            else:
                logger.warning("No .csproj found, defaulting to dotnet build")
                return ("core", "unknown")
        
        # Parse .csproj XML
        try:
            tree = ET.parse(project_file if project_file.is_absolute() else self.workspace_path / project_file)
            root = tree.getroot()
            
            # Check for TargetFramework (new SDK-style)
            target_framework = root.find(".//TargetFramework")
            if target_framework is not None:
                fw = target_framework.text
                if fw.startswith("net") and not fw.startswith("netstandard") and not fw.startswith("netcoreapp"):
                    # net6.0, net7.0, net8.0 → .NET Core/5+
                    version = fw.replace("net", "")
                    return ("core", version)
                elif fw.startswith("netcoreapp"):
                    # netcoreapp3.1 → .NET Core
                    version = fw.replace("netcoreapp", "")
                    return ("core", version)
            
            # Check for TargetFrameworkVersion (old .csproj style)
            target_framework_version = root.find(".//{http://schemas.microsoft.com/developer/msbuild/2003}TargetFrameworkVersion")
            if target_framework_version is not None:
                # v4.7.2, v4.8 → .NET Framework
                version = target_framework_version.text.replace("v", "")
                logger.info(f"Detected .NET Framework {version} - will use msbuild")
                return ("framework", version)
            
            # Default: assume modern .NET
            return ("core", "unknown")
            
        except Exception as e:
            logger.warning(f"Could not parse project file: {e}, defaulting to dotnet build")
            return ("core", "unknown")
    
    def _get_build_command(self, project_path: str, framework_type: str) -> list:
        """
        Get appropriate build command based on framework type.
        
        Args:
            project_path: Path to project/solution
            framework_type: "framework" or "core"
            
        Returns:
            List of command arguments
        """
        if framework_type == "framework":
            # .NET Framework → Use MSBuild
            return [
                "msbuild",
                project_path,
                "/p:Configuration=Release",
                "/p:Platform=Any CPU",
                "/verbosity:normal"
            ]
        else:
            # .NET Core/5+ → Use dotnet CLI
            return [
                "dotnet", "build",
                project_path,
                "--configuration", "Release",
                "--verbosity", "normal"
            ]
    
    def execute_build(
        self, 
        project_path: str,
        timeout: int = 300
    ) -> BuildResult:
        """
        Execute build with AUTOMATIC TOOL DETECTION.
        
        Automatically detects:
        - .NET Framework → Uses msbuild
        - .NET Core/.NET 5+ → Uses dotnet build
        
        Works for ANY .NET project with zero configuration!
        
        Args:
            project_path: Path to .sln or .csproj file
            timeout: Build timeout in seconds
            
        Returns:
            BuildResult with status and output
        """
        logger.info(f"🔧 Executing build: {project_path}")
        
        # STEP 1: Auto-detect framework type
        framework_type, version = self._detect_dotnet_type(project_path)
        logger.info(f"✅ Detected: .NET {framework_type.upper()} {version}")
        
        # STEP 2: Get appropriate build command for this project
        build_command = self._get_build_command(project_path, framework_type)
        logger.info(f"📝 Build command: {' '.join(build_command)}")
        
        start_time = datetime.now()
        
        try:
            # STEP 3: Execute build with correct tool
            result = subprocess.run(
                build_command,
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Determine status
            if result.returncode == 0:
                status = BuildStatus.SUCCESS
            else:
                status = BuildStatus.FAILURE
            
            # Extract errors and warnings
            errors = self._extract_errors(result.stderr + result.stdout)
            warnings = self._extract_warnings(result.stdout)
            
            build_result = BuildResult(
                status=status,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                errors=errors,
                warnings=warnings,
                duration_seconds=duration
            )
            
            logger.info(f"Build {status.value}: {duration:.1f}s, {len(errors)} errors")
            return build_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Build timeout after {timeout}s")
            return BuildResult(
                status=BuildStatus.TIMEOUT,
                stdout="",
                stderr=f"Build timeout after {timeout} seconds",
                exit_code=-1,
                errors=[f"Build timeout after {timeout} seconds"],
                duration_seconds=timeout
            )
        except Exception as e:
            logger.error(f"Build execution error: {e}")
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
        Execute tests with auto-detected test tool.
        
        Args:
            project_path: Path to .sln or .csproj with tests
            timeout: Test timeout in seconds
            
        Returns:
            TestResult with pass/fail counts
        """
        logger.info(f"Executing tests: {project_path}")
        
        # Detect framework type
        framework_type, version = self._detect_dotnet_type(project_path)
        
        start_time = datetime.now()
        
        try:
            # Build test command based on framework type
            if framework_type == "framework":
                # .NET Framework → Use MSTest or vstest.console
                test_command = [
                    "msbuild",
                    project_path,
                    "/t:Test",
                    "/p:Configuration=Release"
                ]
            else:
                # .NET Core/5+ → Use dotnet test
                test_command = [
                    "dotnet", "test",
                    project_path,
                    "--configuration", "Release",
                    "--verbosity", "normal",
                    "--logger", "console;verbosity=detailed"
                ]
            
            logger.info(f"Test command: {' '.join(test_command)}")
            
            # Run tests
            result = subprocess.run(
                test_command,
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Parse test results
            test_counts = self._parse_test_output(result.stdout)
            
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
            errors = self._extract_test_errors(result.stdout)
            
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
                f"Tests {status.value}: {passed}/{total} passed, {failed} failed"
            )
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Test timeout after {timeout}s")
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
            logger.error(f"Test execution error: {e}")
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                errors=[str(e)],
                test_output="",
                duration_seconds=0
            )
    
    def _extract_errors(self, output: str) -> list[str]:
        """Extract error messages from build output"""
        errors = []
        for line in output.split('\n'):
            if 'error' in line.lower() and 'CS' in line:
                errors.append(line.strip())
        return errors
    
    def _extract_warnings(self, output: str) -> list[str]:
        """Extract warning messages from build output"""
        warnings = []
        for line in output.split('\n'):
            if 'warning' in line.lower() and 'CS' in line:
                warnings.append(line.strip())
        return warnings
    
    def _parse_test_output(self, output: str) -> dict:
        """
        Parse dotnet test output for counts.
        
        Example output:
        "Passed! - Failed: 0, Passed: 5, Skipped: 0, Total: 5"
        """
        counts = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
        
        # Look for summary line
        for line in output.split('\n'):
            line = line.strip()
            
            if 'Passed!' in line or 'Failed!' in line:
                # Parse counts
                if 'Failed:' in line:
                    try:
                        failed_str = line.split('Failed:')[1].split(',')[0].strip()
                        counts['failed'] = int(failed_str)
                    except:
                        pass
                
                if 'Passed:' in line:
                    try:
                        passed_str = line.split('Passed:')[1].split(',')[0].strip()
                        counts['passed'] = int(passed_str)
                    except:
                        pass
                
                if 'Skipped:' in line:
                    try:
                        skipped_str = line.split('Skipped:')[1].split(',')[0].strip()
                        counts['skipped'] = int(skipped_str)
                    except:
                        pass
                
                if 'Total:' in line:
                    try:
                        total_str = line.split('Total:')[1].split()[0].strip()
                        counts['total'] = int(total_str)
                    except:
                        pass
        
        return counts
    
    def _extract_test_errors(self, output: str) -> list[str]:
        """Extract test failure messages"""
        errors = []
        in_error_section = False
        
        for line in output.split('\n'):
            line = line.strip()
            
            if 'Failed' in line and 'Test' in line:
                in_error_section = True
                errors.append(line)
            elif in_error_section:
                if line and not line.startswith('---'):
                    errors.append(line)
                if 'Stack Trace:' in line:
                    in_error_section = False
        
        return errors
    
    def get_technology_name(self) -> str:
        """Get the name of the technology stack"""
        return ".NET (Framework + Core)"
    
    def _get_project_file_patterns(self) -> list[str]:
        """
        Get glob patterns for .NET project files.
        
        Returns:
            List of patterns for both .NET Framework and .NET Core
        """
        return ["*.sln", "*.csproj", "*.vbproj", "*.fsproj"]
    
    def detect_project_files(self) -> list[str]:
        """
        Auto-detect .NET project files with .sln priority.
        
        PRIORITY:
        1. Solution files (.sln) - handle multi-project builds
        2. Individual project files (.csproj, .vbproj, .fsproj)
        
        Returns:
            List of project file paths (prioritized)
        """
        project_files = []
        
        # PRIORITY 1: Look for solution files first
        sln_files = list(self.workspace_path.rglob("*.sln"))
        if sln_files:
            logger.info(f"Found {len(sln_files)} solution file(s)")
            return [str(f.relative_to(self.workspace_path)) for f in sln_files]
        
        # PRIORITY 2: Look for individual project files
        for pattern in ["*.csproj", "*.vbproj", "*.fsproj"]:
            found = list(self.workspace_path.rglob(pattern))
            if found:
                logger.info(f"Found {len(found)} {pattern} file(s)")
                project_files.extend([str(f.relative_to(self.workspace_path)) for f in found])
        
        return project_files


# ============================================================================
# CELERY TASKS
# ============================================================================

def create_execution_tasks():
    """Create Celery tasks for async execution"""
    from celery import Celery
    
    app = Celery('autonomous-ticket')
    
    @app.task(name='autonomous_ticket.execute_build')
    def execute_build_task(workspace_path: str, project_path: str) -> dict:
        """
        Celery task for build execution.
        
        Args:
            workspace_path: Workspace directory
            project_path: Project file path
            
        Returns:
            BuildResult as dict
        """
        try:
            engine = ExecutionEngine(workspace_path)
            result = engine.execute_build(project_path)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Build task failed: {e}")
            return BuildResult(
                status=BuildStatus.ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                errors=[str(e)]
            ).model_dump()
    
    @app.task(name='autonomous_ticket.execute_tests')
    def execute_tests_task(workspace_path: str, project_path: str) -> dict:
        """
        Celery task for test execution.
        
        Args:
            workspace_path: Workspace directory
            project_path: Test project file path
            
        Returns:
            TestResult as dict
        """
        try:
            engine = ExecutionEngine(workspace_path)
            result = engine.execute_tests(project_path)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Test task failed: {e}")
            return TestResult(
                status=TestStatus.ERROR,
                total_tests=0,
                passed=0,
                failed=0,
                errors=[str(e)]
            ).model_dump()
    
    return execute_build_task, execute_tests_task


# ============================================================================
# REGISTER EXECUTION ENGINE
# ============================================================================

from ticket_to_code.execution.base_executor import ExecutionEngineFactory

# Register .NET execution engine
ExecutionEngineFactory.register("dotnet", DotNetExecutionEngine)
ExecutionEngineFactory.register(".net", DotNetExecutionEngine)
ExecutionEngineFactory.register("csharp", DotNetExecutionEngine)
ExecutionEngineFactory.register("c#", DotNetExecutionEngine)
