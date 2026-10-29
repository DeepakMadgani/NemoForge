"""
Infrastructure Config Validator — Enhancement 13

Validates YAML, JSON, .env, and .ps1 configuration files.
Provides config-aware code generation mode and plan gating bypass
for non-code config files.

Safety: Validates syntax only. Never auto-corrects config files.

Author: Deepak Madgani
Date: July 2026
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class ConfigValidation(BaseModel):
    """Validation result for a single config file."""
    file_path: str = Field(...)
    file_type: str = Field(..., description="yaml | json | env | ps1 | toml | ini")
    is_valid: bool = Field(True)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    is_config_file: bool = Field(True, description="Whether this is a config vs code file")


class ConfigContext(BaseModel):
    """Config-aware context for code generation."""
    has_config_files: bool = Field(False)
    config_files: List[ConfigValidation] = Field(default_factory=list)
    skip_method_localization: List[str] = Field(
        default_factory=list,
        description="Files that should skip method-level localization"
    )
    skip_structural_checks: List[str] = Field(
        default_factory=list,
        description="Files that should skip structural marker checks in plan gating"
    )


# ============================================================================
# CONFIG FILE DETECTION
# ============================================================================

_CONFIG_EXTENSIONS = {
    ".yml", ".yaml", ".json", ".env", ".toml",
    ".ini", ".cfg", ".conf", ".properties",
    ".ps1", ".sh", ".bash", ".bat", ".cmd",
    ".xml",  # pom.xml, web.config, etc.
}

_CONFIG_NAMES = {
    ".env", ".env.example", ".env.local", ".env.development", ".env.production",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "makefile", "rakefile", "gemfile",
    ".gitignore", ".dockerignore", ".eslintrc",
    "tsconfig.json", "package.json", "angular.json",
    "web.config", "app.config", "appsettings.json",
}


class ConfigValidator:
    """
    Validates configuration files and provides config-aware context
    for code generation and plan gating.
    """

    def __init__(self, workspace_path: str = ""):
        self.workspace = Path(workspace_path) if workspace_path else None

    def is_config_file(self, file_path: str) -> bool:
        """Check if a file is a configuration file (not source code)."""
        p = Path(file_path)
        if p.suffix.lower() in _CONFIG_EXTENSIONS:
            return True
        if p.name.lower() in _CONFIG_NAMES:
            return True
        return False

    # ------------------------------------------------------------------
    # 1. Validate individual config files
    # ------------------------------------------------------------------

    def validate_file(self, file_path: str, content: Optional[str] = None) -> ConfigValidation:
        """
        Validate a configuration file's syntax.

        Args:
            file_path: Path to the file.
            content: File content (reads from disk if not provided).

        Returns:
            ConfigValidation with syntax check results.
        """
        p = Path(file_path)
        ext = p.suffix.lower()
        name = p.name.lower()

        if content is None:
            try:
                full_path = (self.workspace / file_path) if self.workspace else p
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                return ConfigValidation(
                    file_path=file_path,
                    file_type=ext.lstrip(".") or "unknown",
                    is_valid=False,
                    errors=[f"Could not read file: {exc}"],
                )

        file_type = ext.lstrip(".") or "unknown"

        # JSON validation
        if ext == ".json" or name.endswith(".json"):
            return self._validate_json(file_path, content)

        # YAML validation
        if ext in (".yml", ".yaml"):
            return self._validate_yaml(file_path, content)

        # .env validation
        if name.startswith(".env") or ext == ".env":
            return self._validate_env(file_path, content)

        # PowerShell validation
        if ext == ".ps1":
            return self._validate_ps1(file_path, content)

        # Default: treat as valid config
        return ConfigValidation(
            file_path=file_path,
            file_type=file_type,
            is_valid=True,
            is_config_file=True,
        )

    def _validate_json(self, file_path: str, content: str) -> ConfigValidation:
        """Validate JSON syntax."""
        errors = []
        warnings = []

        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(f"JSON syntax error at line {e.lineno}, col {e.colno}: {e.msg}")

        # Check for common issues
        if content.strip() and not content.strip().startswith(("{", "[")):
            warnings.append("JSON content doesn't start with { or [ — may not be valid JSON")

        # Check for trailing commas (common mistake)
        if re.search(r',\s*[}\]]', content):
            warnings.append("Trailing comma detected — may cause parse errors in strict JSON parsers")

        return ConfigValidation(
            file_path=file_path,
            file_type="json",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_yaml(self, file_path: str, content: str) -> ConfigValidation:
        """Validate YAML syntax (basic checks without PyYAML dependency)."""
        errors = []
        warnings = []

        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            # Check for tabs (YAML doesn't allow tabs for indentation)
            if line.startswith("\t"):
                errors.append(f"Line {i}: tabs used for indentation (YAML requires spaces)")

            # Check for inconsistent indentation
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#"):
                indent = len(line) - len(stripped)
                if indent % 2 != 0 and indent > 0:
                    warnings.append(f"Line {i}: odd indentation ({indent} spaces) — may cause issues")

        # Try to parse if PyYAML is available
        try:
            import yaml
            yaml.safe_load(content)
        except ImportError:
            pass  # PyYAML not available, basic checks only
        except yaml.YAMLError as e:
            errors.append(f"YAML parse error: {e}")

        return ConfigValidation(
            file_path=file_path,
            file_type="yaml",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_env(self, file_path: str, content: str) -> ConfigValidation:
        """Validate .env file format."""
        errors = []
        warnings = []

        for i, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                errors.append(f"Line {i}: missing '=' separator: {line[:50]}")
                continue

            key, _, value = line.partition("=")
            key = key.strip()

            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
                warnings.append(f"Line {i}: unusual key format: {key}")

            if not value.strip():
                warnings.append(f"Line {i}: empty value for {key}")

        return ConfigValidation(
            file_path=file_path,
            file_type="env",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_ps1(self, file_path: str, content: str) -> ConfigValidation:
        """Validate PowerShell script (basic syntax checks)."""
        errors = []
        warnings = []

        # Check for mismatched braces
        open_braces = content.count("{")
        close_braces = content.count("}")
        if open_braces != close_braces:
            errors.append(
                f"Mismatched braces: {open_braces} opening, {close_braces} closing"
            )

        # Check for common PowerShell issues
        if "$env:" in content:
            env_vars = re.findall(r'\$env:(\w+)', content)
            if env_vars:
                warnings.append(f"Sets {len(set(env_vars))} environment variables: {', '.join(set(env_vars)[:5])}")

        return ConfigValidation(
            file_path=file_path,
            file_type="ps1",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 2. Build config context for code generation
    # ------------------------------------------------------------------

    def build_config_context(
        self, task_files: List[str]
    ) -> ConfigContext:
        """
        Analyze task files and build config-aware context.

        Config files should skip:
        - Method-level localization (no classes/methods to target)
        - Structural marker checks in plan gating (no code structure)

        Args:
            task_files: List of files from the architectural plan.

        Returns:
            ConfigContext with lists of files that need special handling.
        """
        config_files: List[ConfigValidation] = []
        skip_localization: List[str] = []
        skip_structural: List[str] = []

        for fp in task_files:
            if self.is_config_file(fp):
                validation = self.validate_file(fp)
                config_files.append(validation)
                skip_localization.append(fp)
                skip_structural.append(fp)

        return ConfigContext(
            has_config_files=bool(config_files),
            config_files=config_files,
            skip_method_localization=skip_localization,
            skip_structural_checks=skip_structural,
        )
