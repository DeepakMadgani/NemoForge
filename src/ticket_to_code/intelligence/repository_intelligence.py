"""
Universal Repository Intelligence Subsystem for NemoForge.

Dynamically discovers repository architecture, module dependency graphs,
build systems, available toolchains, and capability boundaries.
Replaces hardcoded workspace assumptions with empirical capability discovery.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_EXCLUDED_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "target", "build", "dist",
    ".idea", ".vscode", ".venv", "venv", "__pycache__", ".pytest_cache"
}

_EXT_TO_LANG = {
    ".java": "java",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
    ".go": "go",
    ".cs": "csharp",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".md": "markdown",
}


@dataclass
class ModuleDescriptor:
    name: str
    path: str  # Repo-relative path
    build_tool: str  # "maven", "gradle", "npm", "cargo", "poetry", "unknown"
    root_file: str  # e.g. "order-service/pom.xml"
    dependencies: List[str] = field(default_factory=list)  # names of modules this depends on
    dependents: List[str] = field(default_factory=list)  # names of modules that depend on this
    package_name: Optional[str] = None
    is_root: bool = False


@dataclass
class BuildSystemDescriptor:
    build_type: str  # "maven", "gradle", "npm", "cargo", "python", "make"
    wrapper_path: Optional[str] = None  # e.g. "./mvnw", "./gradlew"
    executable: str = ""  # e.g. "mvn", "gradle", "npm"
    root_descriptor: str = ""  # e.g. "pom.xml"
    is_multi_module: bool = False
    modules: List[str] = field(default_factory=list)


@dataclass
class TestCommandDescriptor:
    module_name: str
    command_args: List[str]
    working_directory: str
    description: str
    is_wrapper: bool = False


@dataclass
class RepositoryCapabilityModel:
    workspace_root: str
    languages: Dict[str, List[str]] = field(default_factory=dict)
    modules: Dict[str, ModuleDescriptor] = field(default_factory=dict)
    build_systems: List[BuildSystemDescriptor] = field(default_factory=list)
    available_tools: Dict[str, bool] = field(default_factory=dict)
    test_commands: List[TestCommandDescriptor] = field(default_factory=list)
    capability_gaps: List[str] = field(default_factory=list)
    editing_strategy_map: Dict[str, str] = field(default_factory=dict)
    confidence_score: float = 1.0

    def get_owning_module(self, file_path: str) -> Optional[ModuleDescriptor]:
        """Find the most specific owning module for a repo-relative file path."""
        norm_path = file_path.replace("\\", "/").lstrip("/")
        best_match: Optional[ModuleDescriptor] = None
        best_len = -1
        for mod in self.modules.values():
            mod_path = mod.path.replace("\\", "/").strip("/")
            if not mod_path:
                if best_match is None:
                    best_match = mod
                    best_len = 0
                continue
            if norm_path.startswith(mod_path + "/"):
                if len(mod_path) > best_len:
                    best_len = len(mod_path)
                    best_match = mod
        return best_match

    def get_reactor_build_scope(self, modified_files: List[str]) -> Tuple[List[str], str]:
        """
        Determines the build command and execution directory that respects module dependencies.
        Returns (command_args, working_directory).
        """
        affected_modules: Set[str] = set()
        for f in modified_files:
            mod = self.get_owning_module(f)
            if mod:
                affected_modules.add(mod.name)

        # Check for Maven reactor
        for bs in self.build_systems:
            if bs.build_type == "maven":
                exe = bs.wrapper_path or bs.executable
                if not exe:
                    continue
                if bs.is_multi_module and affected_modules:
                    # In a multi-module Maven repo, use -pl <modules> -am (also-make) from root
                    mod_list = ",".join(sorted(affected_modules))
                    cmd = [exe, "test-compile", "-pl", mod_list, "-am"]
                    return cmd, self.workspace_root
                else:
                    return [exe, "test-compile"], self.workspace_root

            elif bs.build_type == "npm":
                # For npm workspaces or single package
                for mod_name in affected_modules:
                    mod = self.modules.get(mod_name)
                    if mod and mod.path:
                        mod_dir = str(Path(self.workspace_root) / mod.path)
                        return [bs.executable, "test"], mod_dir

        # Fallback to direct discovered test commands
        if self.test_commands:
            t = self.test_commands[0]
            return t.command_args, t.working_directory

        return [], self.workspace_root


class RepositoryIntelligence:
    """Discovers repository architecture, dependencies, and capability boundaries."""

    def __init__(self, workspace_root: str):
        self.workspace_root = str(Path(workspace_root).resolve())

    def analyze(self) -> RepositoryCapabilityModel:
        """Run full empirical discovery on the workspace."""
        root = Path(self.workspace_root)
        languages: Dict[str, List[str]] = {}
        modules: Dict[str, ModuleDescriptor] = {}
        build_systems: List[BuildSystemDescriptor] = []
        available_tools: Dict[str, bool] = {}
        test_commands: List[TestCommandDescriptor] = []
        capability_gaps: List[str] = []
        editing_strategies: Dict[str, str] = {}

        # 1. Discover installed tools
        for tool in ("mvn", "gradle", "npm", "pnpm", "yarn", "pytest", "python", "node", "git", "tsc", "cargo", "docker"):
            available_tools[tool] = shutil.which(tool) is not None

        # Check wrappers in root
        mvnw = root / ("mvnw.cmd" if os.name == "nt" else "mvnw")
        has_mvnw = mvnw.is_file() and os.access(str(mvnw), os.X_OK if os.name != "nt" else os.R_OK)
        available_tools["mvnw"] = has_mvnw

        gradlew = root / ("gradlew.bat" if os.name == "nt" else "gradlew")
        has_gradlew = gradlew.is_file() and os.access(str(gradlew), os.X_OK if os.name != "nt" else os.R_OK)
        available_tools["gradlew"] = has_gradlew

        # 2. Walk workspace and index files & extensions
        file_count = 0
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS]
            rel_dir = os.path.relpath(current_root, root).replace("\\", "/")
            if rel_dir == ".":
                rel_dir = ""

            for file in files:
                file_count += 1
                rel_file = f"{rel_dir}/{file}" if rel_dir else file
                ext = Path(file).suffix.lower()
                lang = _EXT_TO_LANG.get(ext)
                if lang:
                    languages.setdefault(lang, []).append(rel_file)

        # 3. Discover Build Systems & Modules
        self._discover_maven(root, modules, build_systems, test_commands, available_tools)
        self._discover_npm(root, modules, build_systems, test_commands, available_tools)
        self._discover_gradle(root, modules, build_systems, test_commands, available_tools)
        self._discover_python(root, modules, build_systems, test_commands, available_tools)

        # 4. Map Editing Strategies based on syntax capabilities
        for ext, lang in _EXT_TO_LANG.items():
            if lang in ("java", "typescript", "javascript", "python", "go", "csharp"):
                editing_strategies[ext] = "ast_cst"
            elif lang == "yaml":
                editing_strategies[ext] = "yaml_structural"
            elif lang == "xml":
                editing_strategies[ext] = "xml_structural"
            elif lang == "sql":
                editing_strategies[ext] = "token_preserving_sql"
            else:
                editing_strategies[ext] = "conservative_source_preserving"

        # 5. Formulate Capability Gaps & Limitations
        if not build_systems:
            capability_gaps.append("NO_BUILD_SYSTEM_DISCOVERED: Repository lacks recognized build manifests (pom.xml, package.json, build.gradle, pyproject.toml)")
        if not test_commands:
            capability_gaps.append("NO_TEST_RUNNER_DISCOVERED: Repository has no automated verification test suite configured")

        for bs in build_systems:
            if bs.build_type == "maven" and not bs.wrapper_path and not available_tools.get("mvn"):
                capability_gaps.append("MISSING_TOOL: Repository requires Maven but neither ./mvnw nor 'mvn' is installed in PATH")
            elif bs.build_type == "gradle" and not bs.wrapper_path and not available_tools.get("gradle"):
                capability_gaps.append("MISSING_TOOL: Repository requires Gradle but neither ./gradlew nor 'gradle' is installed in PATH")
            elif bs.build_type == "npm" and not available_tools.get("npm"):
                capability_gaps.append("MISSING_TOOL: Repository requires Node/npm but 'npm' is not installed in PATH")

        confidence = 1.0
        if capability_gaps:
            confidence = max(0.2, 1.0 - (len(capability_gaps) * 0.25))

        return RepositoryCapabilityModel(
            workspace_root=self.workspace_root,
            languages=languages,
            modules=modules,
            build_systems=build_systems,
            available_tools=available_tools,
            test_commands=test_commands,
            capability_gaps=capability_gaps,
            editing_strategy_map=editing_strategies,
            confidence_score=confidence,
        )

    def _discover_maven(
        self,
        root: Path,
        modules: Dict[str, ModuleDescriptor],
        build_systems: List[BuildSystemDescriptor],
        test_commands: List[TestCommandDescriptor],
        tools: Dict[str, bool],
    ):
        root_pom = root / "pom.xml"
        exe = str(root / ("mvnw.cmd" if os.name == "nt" else "mvnw")) if tools.get("mvnw") else "mvn"
        is_wrapper = tools.get("mvnw", False)

        if root_pom.is_file():
            bs = BuildSystemDescriptor(
                build_type="maven",
                wrapper_path=exe if is_wrapper else None,
                executable="mvn" if not is_wrapper else exe,
                root_descriptor="pom.xml",
                is_multi_module=False,
            )
            try:
                tree = ET.parse(str(root_pom))
                pom_root = tree.getroot()
                ns = {"m": pom_root.tag.split("}")[0].strip("{")} if "}" in pom_root.tag else {}
                prefix = "m:" if ns else ""

                # Check modules tag
                modules_elem = pom_root.find(f"{prefix}modules", ns)
                module_names = []
                if modules_elem is not None:
                    bs.is_multi_module = True
                    for m_elem in modules_elem.findall(f"{prefix}module", ns):
                        if m_elem.text:
                            module_names.append(m_elem.text.strip())

                bs.modules = module_names
                build_systems.append(bs)

                # Register root module
                root_art = pom_root.find(f"{prefix}artifactId", ns)
                root_name = root_art.text.strip() if root_art is not None and root_art.text else root.name
                root_mod = ModuleDescriptor(
                    name=root_name,
                    path="",
                    build_tool="maven",
                    root_file="pom.xml",
                    is_root=True,
                )
                modules[root_name] = root_mod

                # Register submodules
                for m_path in module_names:
                    sub_pom = root / m_path / "pom.xml"
                    sub_name = m_path
                    deps: List[str] = []
                    if sub_pom.is_file():
                        try:
                            s_tree = ET.parse(str(sub_pom))
                            s_root = s_tree.getroot()
                            s_art = s_root.find(f"{prefix}artifactId", ns)
                            if s_art is not None and s_art.text:
                                sub_name = s_art.text.strip()
                            s_deps = s_root.find(f"{prefix}dependencies", ns)
                            if s_deps is not None:
                                for d in s_deps.findall(f"{prefix}dependency", ns):
                                    d_art = d.find(f"{prefix}artifactId", ns)
                                    if d_art is not None and d_art.text:
                                        deps.append(d_art.text.strip())
                        except Exception:
                            pass

                    mod_desc = ModuleDescriptor(
                        name=sub_name,
                        path=m_path.replace("\\", "/"),
                        build_tool="maven",
                        root_file=f"{m_path}/pom.xml",
                        dependencies=deps,
                    )
                    modules[sub_name] = mod_desc
                    test_commands.append(
                        TestCommandDescriptor(
                            module_name=sub_name,
                            command_args=[exe, "-pl", m_path, "-am", "test"] if bs.is_multi_module else [exe, "test"],
                            working_directory=str(root),
                            description=f"Maven test for module {sub_name} with upstream dependencies",
                            is_wrapper=is_wrapper,
                        )
                    )
            except Exception as e:
                logger.warning("Failed to parse root pom.xml: %s", e)
        else:
            # Check for pom.xml in immediate subdirectories (composite/multi-service repos without parent POM)
            for sub_pom in root.glob("*/pom.xml"):
                if any(part in _EXCLUDED_DIRS for part in sub_pom.parts):
                    continue
                rel_dir = os.path.relpath(sub_pom.parent, root).replace("\\", "/")
                sub_name = rel_dir
                try:
                    tree = ET.parse(str(sub_pom))
                    pom_root = tree.getroot()
                    ns = {"m": pom_root.tag.split("}")[0].strip("{")} if "}" in pom_root.tag else {}
                    prefix = "m:" if ns else ""
                    art = pom_root.find(f"{prefix}artifactId", ns)
                    if art is not None and art.text:
                        sub_name = art.text.strip()
                except Exception:
                    pass

                mod_desc = ModuleDescriptor(
                    name=sub_name,
                    path=rel_dir,
                    build_tool="maven",
                    root_file=f"{rel_dir}/pom.xml",
                )
                modules[sub_name] = mod_desc
                test_commands.append(
                    TestCommandDescriptor(
                        module_name=sub_name,
                        command_args=[exe, "test"],
                        working_directory=str(sub_pom.parent),
                        description=f"Maven test for {rel_dir}",
                        is_wrapper=is_wrapper,
                    )
                )
                build_systems.append(
                    BuildSystemDescriptor(
                        build_type="maven",
                        wrapper_path=exe if is_wrapper else None,
                        executable="mvn" if not is_wrapper else exe,
                        root_descriptor=f"{rel_dir}/pom.xml",
                        modules=[rel_dir],
                    )
                )


    def _discover_npm(
        self,
        root: Path,
        modules: Dict[str, ModuleDescriptor],
        build_systems: List[BuildSystemDescriptor],
        test_commands: List[TestCommandDescriptor],
        tools: Dict[str, bool],
    ):
        for pkg_file in root.glob("**/package.json"):
            if any(part in _EXCLUDED_DIRS for part in pkg_file.parts):
                continue
            rel_dir = os.path.relpath(pkg_file.parent, root).replace("\\", "/")
            if rel_dir == ".":
                rel_dir = ""
            try:
                data = json.loads(pkg_file.read_text(encoding="utf-8"))
                name = data.get("name") or (Path(rel_dir).name if rel_dir else root.name)
                scripts = data.get("scripts", {})
                mod = ModuleDescriptor(
                    name=name,
                    path=rel_dir,
                    build_tool="npm",
                    root_file=f"{rel_dir}/package.json" if rel_dir else "package.json",
                    is_root=(rel_dir == ""),
                )
                modules[name] = mod

                if "test" in scripts:
                    test_commands.append(
                        TestCommandDescriptor(
                            module_name=name,
                            command_args=["npm", "test"],
                            working_directory=str(pkg_file.parent),
                            description=f"npm test in {rel_dir or 'root'}",
                        )
                    )

                if rel_dir == "":
                    build_systems.append(
                        BuildSystemDescriptor(
                            build_type="npm",
                            executable="npm",
                            root_descriptor="package.json",
                            is_multi_module=bool(data.get("workspaces")),
                        )
                    )
            except Exception:
                pass

    def _discover_gradle(
        self,
        root: Path,
        modules: Dict[str, ModuleDescriptor],
        build_systems: List[BuildSystemDescriptor],
        test_commands: List[TestCommandDescriptor],
        tools: Dict[str, bool],
    ):
        gradle_file = root / "build.gradle"
        gradle_kts = root / "build.gradle.kts"
        if not (gradle_file.is_file() or gradle_kts.is_file()):
            return

        exe = str(root / ("gradlew.bat" if os.name == "nt" else "gradlew")) if tools.get("gradlew") else "gradle"
        is_wrapper = tools.get("gradlew", False)

        bs = BuildSystemDescriptor(
            build_type="gradle",
            wrapper_path=exe if is_wrapper else None,
            executable="gradle" if not is_wrapper else exe,
            root_descriptor="build.gradle" if gradle_file.is_file() else "build.gradle.kts",
        )
        build_systems.append(bs)
        test_commands.append(
            TestCommandDescriptor(
                module_name=root.name,
                command_args=[exe, "test"],
                working_directory=str(root),
                description="Gradle test check",
                is_wrapper=is_wrapper,
            )
        )

    def _discover_python(
        self,
        root: Path,
        modules: Dict[str, ModuleDescriptor],
        build_systems: List[BuildSystemDescriptor],
        test_commands: List[TestCommandDescriptor],
        tools: Dict[str, bool],
    ):
        pyproject = root / "pyproject.toml"
        setup_py = root / "setup.py"
        pytest_ini = root / "pytest.ini"
        if pyproject.is_file() or setup_py.is_file() or pytest_ini.is_file():
            bs = BuildSystemDescriptor(
                build_type="python",
                executable="pytest" if tools.get("pytest") else "python",
                root_descriptor="pyproject.toml" if pyproject.is_file() else "setup.py",
            )
            build_systems.append(bs)
            test_commands.append(
                TestCommandDescriptor(
                    module_name=root.name,
                    command_args=["pytest"] if tools.get("pytest") else ["python", "-m", "unittest"],
                    working_directory=str(root),
                    description="Python test execution",
                )
            )
