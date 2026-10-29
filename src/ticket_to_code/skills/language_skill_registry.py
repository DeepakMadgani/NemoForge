"""
Declarative Language Skill Registry.

Implements the "Agent Skills" and "Language Declarations" pattern from:
- Sean's AI Stories: "Agent Skills In 15 Min | Claude Code Full Guide"
- "Pi Minimal Coding Agent Harness | Skills, Extensions, Packages, Bash"

Collapses scattered language knowledge (build commands, test runners, companion rules,
compilation verifiers) into single declarative skill definitions loaded on demand.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LanguageSkill:
    id: str
    name: str
    extensions: List[str]
    manifest: str
    build_command: str
    test_command: str
    test_file_pattern: str
    companion_rules: List[Dict[str, Any]] = field(default_factory=list)
    verifier: str = ""
    comment_syntax: str = "//"


class LanguageSkillRegistry:
    """Singleton-style registry for declarative language skills."""

    _instance: Optional[LanguageSkillRegistry] = None

    def __init__(self, skills_dir: Optional[Path] = None):
        if skills_dir is None:
            skills_dir = Path(__file__).resolve().parent / "languages"
        self.skills_dir = skills_dir
        self.skills: Dict[str, LanguageSkill] = {}
        self.ext_map: Dict[str, LanguageSkill] = {}
        self.load_skills()

    @classmethod
    def get_instance(cls) -> LanguageSkillRegistry:
        if cls._instance is None:
            cls._instance = LanguageSkillRegistry()
        return cls._instance

    def load_skills(self) -> None:
        self.skills.clear()
        self.ext_map.clear()
        if not self.skills_dir.is_dir():
            logger.warning(f"Language skills directory does not exist: {self.skills_dir}")
            return

        for p in self.skills_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                skill = LanguageSkill(
                    id=data["id"],
                    name=data.get("name", data["id"]),
                    extensions=data.get("extensions", []),
                    manifest=data.get("manifest", ""),
                    build_command=data.get("build_command", ""),
                    test_command=data.get("test_command", ""),
                    test_file_pattern=data.get("test_file_pattern", ""),
                    companion_rules=data.get("companion_rules", []),
                    verifier=data.get("verifier", ""),
                    comment_syntax=data.get("comment_syntax", "//"),
                )
                self.skills[skill.id] = skill
                for ext in skill.extensions:
                    self.ext_map[ext.lower()] = skill
                logger.info(f"Loaded declarative language skill: {skill.id} ({len(skill.extensions)} exts)")
            except Exception as e:
                logger.error(f"Failed to load language skill from {p.name}: {e}")

    def get_skill_for_path(self, file_path: str) -> Optional[LanguageSkill]:
        ext = Path(file_path).suffix.lower()
        return self.ext_map.get(ext)

    def resolve_companions(self, file_path: str, workspace_path: str) -> List[str]:
        """Resolves sibling companions declaratively based on language rules."""
        skill = self.get_skill_for_path(file_path)
        if not skill or not skill.companion_rules:
            return [file_path]

        norm_p = file_path.replace("\\", "/")
        p_obj = Path(norm_p)
        ws = Path(workspace_path)
        companions = [norm_p]

        for rule in skill.companion_rules:
            # 1. Trigger suffix rule (e.g. .component.ts -> .component.html, .component.scss)
            trigger = rule.get("trigger_suffix")
            if trigger and norm_p.endswith(trigger):
                stem = norm_p[:-len(trigger)]
                for comp_suffix in rule.get("companions", []):
                    comp_cand = f"{stem}{comp_suffix}"
                    if (ws / comp_cand).is_file() and comp_cand not in companions:
                        companions.append(comp_cand)

            # 2. Java src/main -> src/test rule
            src_prefix = rule.get("src_prefix")
            test_prefix = rule.get("test_prefix")
            test_suffix = rule.get("test_suffix")
            if src_prefix and test_prefix and src_prefix in norm_p and norm_p.endswith(".java"):
                stem = norm_p[:-5]  # remove .java
                test_cand = norm_p.replace(src_prefix, test_prefix)
                test_cand_stem = test_cand[:-5]
                full_test_cand = f"{test_cand_stem}Test.java"
                if (ws / full_test_cand).is_file() and full_test_cand not in companions:
                    companions.append(full_test_cand)

        return companions
