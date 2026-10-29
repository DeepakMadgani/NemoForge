"""
Dependency Resolution Agent — Enhancement 12

Parses dependency-related build errors (ModuleNotFoundError, ImportError,
ClassNotFoundException, etc.), maps module names to package names, and
adds them to the project manifest.

Safety:
  - Only modifies manifest files (requirements.txt, pom.xml, package.json)
  - Uses a built-in mapping for known packages before LLM fallback
  - Never installs packages directly — only updates manifests

Author: Deepak Madgani
Date: July 2026
"""

import re
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class MissingDependency(BaseModel):
    """A single missing dependency detected from build output."""
    module_name: str = Field(..., description="Import/module name from error")
    package_name: Optional[str] = Field(None, description="Resolved package name")
    error_line: str = Field("", description="Original error line")
    language: str = Field("python", description="python | java | nodejs | dotnet")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class DependencyResolution(BaseModel):
    """Result of dependency resolution analysis."""
    has_missing_deps: bool = Field(False, description="Whether missing deps were found")
    missing_deps: List[MissingDependency] = Field(default_factory=list)
    resolved_count: int = Field(0, description="How many were auto-resolved")
    manifest_updated: bool = Field(False, description="Whether manifest was modified")
    manifest_path: Optional[str] = Field(None, description="Path to updated manifest")
    install_command: Optional[str] = Field(None, description="Command to install deps")


# ============================================================================
# ERROR PATTERNS
# ============================================================================

# Python: ModuleNotFoundError: No module named 'neo4j'
_PY_IMPORT = re.compile(
    r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]?(\w[\w.]*)['\"]?",
    re.IGNORECASE,
)

# Java: ClassNotFoundException, NoClassDefFoundError
_JAVA_IMPORT = re.compile(
    r"(?:ClassNotFoundException|NoClassDefFoundError):\s*([\w.]+)",
    re.IGNORECASE,
)

# Node.js: Cannot find module 'express'
_NODE_IMPORT = re.compile(
    r"Cannot find module ['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

# .NET: The type or namespace name 'X' could not be found
_DOTNET_IMPORT = re.compile(
    r"error CS\d+.*?type or namespace name ['\"]?(\w+)['\"]?.*?could not be found",
    re.IGNORECASE,
)

# ============================================================================
# KNOWN PACKAGE MAPPINGS
# ============================================================================

# module_name → pip package name
_PY_MODULE_TO_PACKAGE: Dict[str, str] = {
    "neo4j": "neo4j",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "jose": "python-jose",
    "jwt": "PyJWT",
    "dateutil": "python-dateutil",
    "attr": "attrs",
    "pydantic": "pydantic",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "requests": "requests",
    "flask": "flask",
    "sqlalchemy": "sqlalchemy",
    "alembic": "alembic",
    "celery": "celery",
    "redis": "redis",
    "psycopg2": "psycopg2-binary",
    "pymongo": "pymongo",
    "boto3": "boto3",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "torch": "torch",
    "transformers": "transformers",
    "langchain": "langchain",
    "langchain_core": "langchain-core",
    "langchain_community": "langchain-community",
    "langgraph": "langgraph",
    "openai": "openai",
    "anthropic": "anthropic",
    "tiktoken": "tiktoken",
    "chromadb": "chromadb",
    "pgvector": "pgvector",
}


class DependencyResolver:
    """
    Parses build errors for missing dependency signals, resolves
    module names to package names, and updates manifests.

    Used inside fix_build_errors_node: if a build error is
    dependency-related, resolve it BEFORE retrying code fixes.
    """

    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)

    # ------------------------------------------------------------------
    # 1. Detect missing dependencies from build output
    # ------------------------------------------------------------------

    def detect_missing_deps(
        self, build_errors: List[str], technology: str = "auto"
    ) -> List[MissingDependency]:
        """
        Parse build error messages for missing dependency patterns.

        Args:
            build_errors: List of error strings from build output.
            technology: "python", "java", "nodejs", "dotnet", or "auto".

        Returns:
            List of MissingDependency objects.
        """
        deps: List[MissingDependency] = []
        seen: set = set()
        error_text = "\n".join(build_errors)

        # Python
        if technology in ("auto", "python"):
            for m in _PY_IMPORT.finditer(error_text):
                module = m.group(1).split(".")[0]  # top-level module
                if module not in seen:
                    seen.add(module)
                    pkg = _PY_MODULE_TO_PACKAGE.get(module, module)
                    deps.append(MissingDependency(
                        module_name=module,
                        package_name=pkg,
                        error_line=m.group(0),
                        language="python",
                        confidence=0.95 if module in _PY_MODULE_TO_PACKAGE else 0.7,
                    ))

        # Java
        if technology in ("auto", "java"):
            for m in _JAVA_IMPORT.finditer(error_text):
                cls = m.group(1)
                if cls not in seen:
                    seen.add(cls)
                    deps.append(MissingDependency(
                        module_name=cls,
                        package_name=None,  # Java needs LLM/Maven Central lookup
                        error_line=m.group(0),
                        language="java",
                        confidence=0.6,
                    ))

        # Node.js
        if technology in ("auto", "nodejs"):
            for m in _NODE_IMPORT.finditer(error_text):
                mod = m.group(1)
                if mod.startswith(".") or mod.startswith("/"):
                    continue  # local module, not a package
                if mod not in seen:
                    seen.add(mod)
                    deps.append(MissingDependency(
                        module_name=mod,
                        package_name=mod,  # npm: module name == package name
                        error_line=m.group(0),
                        language="nodejs",
                        confidence=0.9,
                    ))

        # .NET
        if technology in ("auto", "dotnet"):
            for m in _DOTNET_IMPORT.finditer(error_text):
                ns = m.group(1)
                if ns not in seen:
                    seen.add(ns)
                    deps.append(MissingDependency(
                        module_name=ns,
                        package_name=None,  # Needs NuGet lookup
                        error_line=m.group(0),
                        language="dotnet",
                        confidence=0.5,
                    ))

        logger.info(
            f"DependencyResolver: detected {len(deps)} missing dependencies "
            f"from {len(build_errors)} error lines"
        )
        return deps

    # ------------------------------------------------------------------
    # 2. Resolve unknown packages via LLM
    # ------------------------------------------------------------------

    def resolve_with_llm(self, deps: List[MissingDependency]) -> List[MissingDependency]:
        """
        For dependencies without a known package mapping, ask the LLM.
        """
        unresolved = [d for d in deps if d.package_name is None]
        if not unresolved:
            return deps

        try:
            from ticket_to_code.llm_utils import llm_invoke
            from langchain_core.messages import SystemMessage, HumanMessage

            module_list = "\n".join(
                f"- {d.module_name} ({d.language})" for d in unresolved
            )

            result = llm_invoke(
                messages=[
                    SystemMessage(content=(
                        "You are a package resolution expert. For each module name "
                        "and language, respond with the exact package name to install. "
                        "Respond as JSON: {\"module_name\": \"package_name\", ...}"
                    )),
                    HumanMessage(content=f"Resolve these modules:\n{module_list}"),
                ],
                label="dependency_resolution",
            )

            if result and result.content:
                import json
                try:
                    raw = result.content.strip()
                    if "```json" in raw:
                        raw = raw.split("```json")[1].split("```")[0].strip()
                    elif "```" in raw:
                        raw = raw.split("```")[1].split("```")[0].strip()
                    mapping = json.loads(raw)

                    for d in unresolved:
                        if d.module_name in mapping:
                            d.package_name = mapping[d.module_name]
                            d.confidence = 0.75
                except (json.JSONDecodeError, KeyError):
                    logger.debug("LLM dependency resolution parse failed")

        except Exception as exc:
            logger.debug(f"LLM dependency resolution failed: {exc}")

        return deps

    # ------------------------------------------------------------------
    # 3. Find and update manifest
    # ------------------------------------------------------------------

    def _find_manifest(self, language: str) -> Optional[Path]:
        """Find the project manifest file."""
        patterns = {
            "python": ["requirements.txt", "setup.py", "pyproject.toml"],
            "nodejs": ["package.json"],
            "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
            "dotnet": ["*.csproj"],
        }
        for pattern in patterns.get(language, []):
            matches = list(self.workspace.glob(f"**/{pattern}"))
            if matches:
                return matches[0]
        return None

    def update_manifest(
        self, deps: List[MissingDependency]
    ) -> Tuple[bool, Optional[str]]:
        """
        Add resolved dependencies to the appropriate manifest file.

        Returns:
            (success, manifest_path)
        """
        if not deps:
            return False, None

        # Group by language
        by_lang: Dict[str, List[MissingDependency]] = {}
        for d in deps:
            if d.package_name:
                by_lang.setdefault(d.language, []).append(d)

        updated = False
        manifest_path = None

        for lang, lang_deps in by_lang.items():
            manifest = self._find_manifest(lang)
            if not manifest:
                logger.warning(f"No manifest found for {lang} in {self.workspace}")
                continue

            try:
                if lang == "python" and manifest.name == "requirements.txt":
                    existing = manifest.read_text(encoding="utf-8")
                    existing_pkgs = {
                        line.split("==")[0].split(">=")[0].split("[")[0].strip().lower()
                        for line in existing.splitlines()
                        if line.strip() and not line.startswith("#")
                    }
                    new_lines = []
                    for d in lang_deps:
                        if d.package_name.lower() not in existing_pkgs:
                            new_lines.append(d.package_name)
                            logger.info(f"  Adding to requirements.txt: {d.package_name}")

                    if new_lines:
                        with open(manifest, "a", encoding="utf-8") as f:
                            f.write("\n" + "\n".join(new_lines) + "\n")
                        updated = True
                        manifest_path = str(manifest)

                elif lang == "nodejs" and manifest.name == "package.json":
                    import json
                    pkg_data = json.loads(manifest.read_text(encoding="utf-8"))
                    all_deps = {
                        **pkg_data.get("dependencies", {}),
                        **pkg_data.get("devDependencies", {}),
                    }
                    new_deps = [
                        d for d in lang_deps
                        if d.package_name not in all_deps
                    ]
                    if new_deps:
                        if "dependencies" not in pkg_data:
                            pkg_data["dependencies"] = {}
                        for d in new_deps:
                            pkg_data["dependencies"][d.package_name] = "*"
                            logger.info(f"  Adding to package.json: {d.package_name}")
                        manifest.write_text(
                            json.dumps(pkg_data, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        updated = True
                        manifest_path = str(manifest)

                # Java/dotnet: log but don't auto-edit XML (too risky)
                else:
                    for d in lang_deps:
                        logger.info(
                            f"  Manual action needed: add {d.package_name} to {manifest.name}"
                        )

            except Exception as exc:
                logger.warning(f"Failed to update {manifest}: {exc}")

        return updated, manifest_path

    # ------------------------------------------------------------------
    # 4. Main entry point
    # ------------------------------------------------------------------

    def resolve(
        self, build_errors: List[str], technology: str = "auto"
    ) -> DependencyResolution:
        """
        Main entry point. Detect, resolve, and update manifest.

        Args:
            build_errors: Error messages from build output.
            technology: Project technology.

        Returns:
            DependencyResolution with results.
        """
        # 1. Detect
        deps = self.detect_missing_deps(build_errors, technology)
        if not deps:
            return DependencyResolution(has_missing_deps=False)

        # 2. Resolve unknowns via LLM
        deps = self.resolve_with_llm(deps)

        # 3. Update manifest
        updated, manifest_path = self.update_manifest(deps)
        resolved_count = sum(1 for d in deps if d.package_name is not None)

        # 4. Build install command
        install_cmd = None
        py_deps = [d for d in deps if d.language == "python" and d.package_name]
        node_deps = [d for d in deps if d.language == "nodejs" and d.package_name]
        if py_deps:
            install_cmd = f"pip install {' '.join(d.package_name for d in py_deps)}"
        elif node_deps:
            install_cmd = f"npm install {' '.join(d.package_name for d in node_deps)}"

        result = DependencyResolution(
            has_missing_deps=True,
            missing_deps=deps,
            resolved_count=resolved_count,
            manifest_updated=updated,
            manifest_path=manifest_path,
            install_command=install_cmd,
        )

        logger.info(
            f"DependencyResolver: {len(deps)} deps found, "
            f"{resolved_count} resolved, manifest_updated={updated}"
        )
        return result
