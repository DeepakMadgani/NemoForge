"""
Execution Configuration - Enterprise-Level Build Strategy Detection

Goes beyond simple language detection to understand:
- Build system (dotnet, msbuild, maven, gradle, npm, docker)
- Test framework (xunit, nunit, junit, pytest, jest)
- Entry point (specific project/solution file)
- Container strategy (Docker, Kubernetes)

This allows for sophisticated multi-environment support.

Author: Deepak Madgani
Date: May 2026
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """
    Comprehensive execution configuration.

    Describes HOW to build/test a project, not just WHAT language it uses.
    """
    language: Literal['csharp', 'java', 'python', 'javascript', 'typescript', 'go']
    build_system: str
    test_framework: Optional[str] = None
    entry_project: Optional[str] = None
    workspace_structure: Optional[str] = None
    container_build: bool = False
    dockerfile_path: Optional[str] = None
    custom_build_script: Optional[str] = None
    framework_version: Optional[str] = None
    requires_sdk: bool = True
    multi_stage_build: bool = False
    build_dependencies: Optional[List[str]] = None

    def __post_init__(self):
        if self.build_dependencies is None:
            self.build_dependencies = []


class ExecutionConfigDetector:
    """
    Intelligent detection of execution configuration.

    Goes beyond simple file pattern matching to understand:
    - Project structure complexity
    - Build tool versions
    - Test framework setup
    - Container strategy
    """

    def __init__(self, workspace_path: Path):
        self.workspace = Path(workspace_path)

    def detect(self, user_override: Optional[ExecutionConfig] = None) -> ExecutionConfig:
        if user_override:
            logger.info(f"Using user-provided execution config: {user_override.language}/{user_override.build_system}")
            return user_override

        language = self._detect_language()
        if language == 'csharp':
            return self._detect_dotnet_config()
        elif language == 'java':
            return self._detect_java_config()
        elif language == 'python':
            return self._detect_python_config()
        elif language in ('javascript', 'typescript'):
            return self._detect_nodejs_config()
        elif language == 'go':
            return self._detect_go_config()
        else:
            raise ValueError(f"Unsupported language detected: {language}")

    def _detect_language(self) -> str:
        if list(self.workspace.rglob('*.sln')) or list(self.workspace.rglob('*.csproj')):
            return 'csharp'
        if list(self.workspace.rglob('pom.xml')) or list(self.workspace.rglob('build.gradle')) or list(self.workspace.rglob('build.gradle.kts')):
            return 'java'
        if list(self.workspace.rglob('pyproject.toml')) or list(self.workspace.rglob('setup.py')) or list(self.workspace.rglob('requirements.txt')):
            return 'python'
        if list(self.workspace.rglob('package.json')) or list(self.workspace.rglob('tsconfig.json')):
            if list(self.workspace.rglob('tsconfig.json')):
                return 'typescript'
            return 'javascript'
        if list(self.workspace.rglob('go.mod')):
            return 'go'
        raise ValueError(f"Could not detect project language in {self.workspace}")

    def _detect_dotnet_config(self) -> ExecutionConfig:
        sln_files = list(self.workspace.rglob('*.sln'))
        csproj_files = list(self.workspace.rglob('*.csproj'))

        entry_project = None
        structure = 'single_project'
        if sln_files:
            entry_project = sln_files[0].name
            structure = 'solution'
        elif csproj_files:
            entry_project = csproj_files[0].name
            structure = 'single_project' if len(csproj_files) == 1 else 'multi_project'

        framework_version = None
        build_system = 'dotnet'
        test_framework = None

        if csproj_files:
            try:
                tree = ET.parse(csproj_files[0])
                root = tree.getroot()
                tf = root.find('.//TargetFramework') or root.find('.//TargetFrameworkVersion')
                if tf is not None and tf.text:
                    framework_version = tf.text
                    if framework_version.startswith('net4') or framework_version.startswith('v4'):
                        build_system = 'msbuild'
                if root.find('.//PackageReference[@Include="xunit"]') is not None:
                    test_framework = 'xunit'
                elif root.find('.//PackageReference[@Include="NUnit"]') is not None:
                    test_framework = 'nunit'
            except Exception as e:
                logger.debug(f"Failed parsing csproj: {e}")

        dockerfile = list(self.workspace.rglob('Dockerfile'))
        container_build = bool(dockerfile)

        return ExecutionConfig(
            language='csharp',
            build_system=build_system,
            test_framework=test_framework or 'xunit',
            entry_project=entry_project,
            workspace_structure=structure,
            framework_version=framework_version,
            container_build=container_build,
            dockerfile_path=str(dockerfile[0]) if dockerfile else None,
        )

    def _detect_java_config(self) -> ExecutionConfig:
        if list(self.workspace.rglob('pom.xml')):
            return ExecutionConfig(
                language='java',
                build_system='maven',
                test_framework='junit',
                entry_project='pom.xml',
            )
        if list(self.workspace.rglob('build.gradle')) or list(self.workspace.rglob('build.gradle.kts')):
            return ExecutionConfig(
                language='java',
                build_system='gradle',
                test_framework='junit',
                entry_project='build.gradle',
            )
        raise ValueError("No Java build files found")

    def _detect_python_config(self) -> ExecutionConfig:
        entry = None
        if list(self.workspace.rglob('pyproject.toml')):
            entry = 'pyproject.toml'
        elif list(self.workspace.rglob('setup.py')):
            entry = 'setup.py'

        return ExecutionConfig(
            language='python',
            build_system='pip',
            test_framework='pytest',
            entry_project=entry,
        )

    def _detect_nodejs_config(self) -> ExecutionConfig:
        is_typescript = bool(list(self.workspace.rglob('tsconfig.json')))
        return ExecutionConfig(
            language='typescript' if is_typescript else 'javascript',
            build_system='npm',
            test_framework='jest',
            entry_project='package.json',
        )

    def _detect_go_config(self) -> ExecutionConfig:
        return ExecutionConfig(
            language='go',
            build_system='go',
            test_framework='go-test',
            entry_project='go.mod',
        )


class AIExecutionConfigDetector(ExecutionConfigDetector):
    def __init__(self, workspace_path: Path, llm_client=None):
        super().__init__(workspace_path)
        self.llm_client = llm_client
