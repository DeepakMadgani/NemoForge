"""
Abstract Execution Engine Base

Technology-agnostic execution layer that can be extended for:
- Java (Maven, Gradle)
- .NET (dotnet CLI)
- Node.js (npm, jest)
- Python (pytest)
- Go (go build, go test)

Author: Deepak Madgani
Date: April 2026
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ticket_to_code.models import BuildResult, TestResult

logger = logging.getLogger(__name__)


class ExecutionEngineBase(ABC):
    """
    Abstract base class for execution engines.
    
    Subclasses implement technology-specific build and test logic.
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize execution engine.
        
        Args:
            workspace_path: Path to code workspace
        """
        self.workspace_path = Path(workspace_path)
        if not self.workspace_path.exists():
            raise ValueError(f"Workspace path does not exist: {workspace_path}")
        
        logger.info(
            f"{self.__class__.__name__} initialized: {workspace_path}"
        )
    
    @property
    def workspace(self) -> Path:
        """Alias for workspace_path as a Path instance."""
        return self.workspace_path
    
    @abstractmethod
    def execute_build(
        self, 
        project_path: str,
        timeout: int = 300
    ) -> BuildResult:
        """
        Execute build command for the technology stack.
        
        Args:
            project_path: Relative path to project/module
            timeout: Build timeout in seconds
            
        Returns:
            BuildResult with status and output
        """
        pass
    
    @abstractmethod
    def execute_tests(
        self,
        project_path: str,
        timeout: int = 600,
        test_target: Optional[str] = None
    ) -> TestResult:
        """
        Execute test command for the technology stack.
        
        Args:
            project_path: Relative path to project/module
            timeout: Test timeout in seconds
            test_target: Optional specific test class or pattern (e.g. '*ParticipatingMemberServiceTest')
            
        Returns:
            TestResult with pass/fail counts
        """
        pass
    
    @abstractmethod
    def get_technology_name(self) -> str:
        """Get the name of the technology stack (e.g., 'Java', '.NET')"""
        pass
    
    def detect_project_files(self) -> list[str]:
        """
        Auto-detect project files in workspace.
        
        Returns:
            List of project file paths
        """
        project_files = []
        
        # Check for technology-specific files
        patterns = self._get_project_file_patterns()
        
        for pattern in patterns:
            found = list(self.workspace_path.rglob(pattern))
            project_files.extend([str(f.relative_to(self.workspace_path)) for f in found])
        
        return project_files
    
    @abstractmethod
    def _get_project_file_patterns(self) -> list[str]:
        """
        Get glob patterns for project files.
        
        Returns:
            List of patterns (e.g., ['pom.xml', '*.csproj'])
        """
        pass


class ExecutionEngineFactory:
    """
    Factory for creating appropriate execution engine based on project type.
    """
    
    _engines = {}
    
    @classmethod
    def register(cls, technology: str, engine_class: type):
        """
        Register an execution engine for a technology.
        
        Args:
            technology: Technology name (e.g., 'java', 'dotnet')
            engine_class: ExecutionEngine subclass
        """
        cls._engines[technology.lower()] = engine_class
        logger.info(f"Registered execution engine: {technology}")
    
    @classmethod
    def create(
        cls, 
        workspace_path: str,
        technology: Optional[str] = None
    ) -> ExecutionEngineBase:
        """
        Create execution engine for workspace.
        
        Args:
            workspace_path: Path to workspace
            technology: Technology name (auto-detected if None)
            
        Returns:
            ExecutionEngine instance
        """
        workspace = Path(workspace_path)
        
        # Auto-detect if not specified
        if technology is None:
            technology = cls._detect_technology(workspace)
        
        # Get engine class
        engine_class = cls._engines.get(technology.lower())
        
        if engine_class is None:
            raise ValueError(
                f"Unsupported technology: {technology}. "
                f"Supported: {list(cls._engines.keys())}"
            )
        
        return engine_class(workspace_path)
    
    @classmethod
    def _detect_technology(cls, workspace: Path) -> str:
        """
        Auto-detect technology from project files.
        
        Args:
            workspace: Workspace path
            
        Returns:
            Technology name
        """
        # 1. Root level markers take absolute priority
        if (workspace / "angular.json").exists() or (workspace / "package.json").exists():
            return "nodejs"
        if (workspace / "pom.xml").exists():
            return "java-maven"
        if (workspace / "build.gradle").exists() or (workspace / "build.gradle.kts").exists():
            return "java-gradle"
        if list(workspace.glob("*.csproj")) or list(workspace.glob("*.sln")):
            return "dotnet"
        if (workspace / "pyproject.toml").exists() or (workspace / "setup.py").exists() or (workspace / "requirements.txt").exists():
            return "python"
        if (workspace / "go.mod").exists():
            return "go"
        if (workspace / "Cargo.toml").exists():
            return "rust"

        # 2. Subdirectory search with strict exclusions (never inspect dependencies or build caches)
        ignored_dirs = {"node_modules", ".venv", "venv", ".git", "bin", "obj", "dist", "target", "build", ".aviator", "brain"}
        def _safe_find(pattern: str) -> bool:
            for p in workspace.rglob(pattern):
                if not any(part in ignored_dirs for part in p.parts):
                    return True
            return False

        if _safe_find("angular.json") or _safe_find("package.json"):
            return "nodejs"
        if _safe_find("build.gradle") or _safe_find("build.gradle.kts"):
            return "java-gradle"
        if _safe_find("pom.xml"):
            return "java-maven"
        if _safe_find("*.csproj") or _safe_find("*.sln"):
            return "dotnet"
        if _safe_find("setup.py") or _safe_find("pyproject.toml") or _safe_find("requirements.txt"):
            return "python"
        if _safe_find("go.mod"):
            return "go"
        if _safe_find("Cargo.toml"):
            return "rust"
        
        raise ValueError(
            f"Could not detect technology in workspace: {workspace}. "
            f"Supported: {list(cls._engines.keys())}"
        )
    
    @classmethod
    def get_supported_technologies(cls) -> list[str]:
        """Get list of supported technologies"""
        return list(cls._engines.keys())
