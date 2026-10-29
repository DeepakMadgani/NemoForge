from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


# Repository-specific invariants  live in a packaged skill JSON and are only
# injected when the workspace looks like that repository (any marker dir present) or when the
# workspace ships an explicit override at .aviator/skills/cloudforge_invariants.json.
REPO_INVARIANT_MARKERS = ("xchange-ui", "issues-service", "project-service", "bim-gateway")
PACKAGED_REPO_INVARIANTS = Path(__file__).parent / "project-architecture" / "cloudforge_invariants.json"


DEFAULT_SKILLS: Dict[str, dict] = {
    "grounded-planning": {
        "id": "grounded-planning",
        "title": "Grounded Planning",
        "rules": [
            "Select files from discovered candidates only",
            "Prefer owner files before consumer files",
            "Require evidence-backed reason for each writable task",
        ],
    },
    "minimal-diff": {
        "id": "minimal-diff",
        "title": "Minimal Diff",
        "rules": [
            "Change the fewest files required",
            "Avoid broad refactors unless requested",
            "Preserve existing public APIs unless ticket requires change",
        ],
    },
    "api-contract": {
        "id": "api-contract",
        "title": "API Contract Safety",
        "rules": [
            "Treat request/response models as compatibility boundaries",
            "List impacted consumers in planning notes",
            "Do not break required fields without migration path",
        ],
    },
    "backward-compat": {
        "id": "backward-compat",
        "title": "Backward Compatibility",
        "rules": [
            "Prefer additive changes over breaking changes",
            "Guard behavior behind defaults when possible",
            "Retain old behavior unless acceptance criteria says otherwise",
        ],
    },
    "consumer-impact": {
        "id": "consumer-impact",
        "title": "Consumer Impact Check",
        "rules": [
            "Identify downstream consumers before edits",
            "Mark verification-only files explicitly",
            "Avoid writing consumer files unless owner change is insufficient",
        ],
    },
    "config-migration": {
        "id": "config-migration",
        "title": "Config Migration",
        "rules": [
            "Update source-of-truth config first",
            "Track all config aliases/variants",
            "Validate runtime compatibility after change",
        ],
    },
    "version-propagation": {
        "id": "version-propagation",
        "title": "Version Propagation",
        "rules": [
            "Locate canonical version owner",
            "Separate owner updates from display/consumer verification",
            "Avoid generated or lock files unless explicitly requested",
        ],
    },
    "runtime-compat": {
        "id": "runtime-compat",
        "title": "Runtime Compatibility",
        "rules": [
            "Respect runtime/tooling constraints in workspace",
            "Fail closed when mandatory runtime checks cannot run",
            "Prefer deterministic checks over assumptions",
        ],
    },
    "rename-propagation": {
        "id": "rename-propagation",
        "title": "Rename Propagation",
        "rules": [
            "Update declarations and all references",
            "Include imports/usages in task sequence",
            "Ensure no stale symbol remains after patch",
        ],
    },
    "public-api-safety": {
        "id": "public-api-safety",
        "title": "Public API Safety",
        "rules": [
            "Treat exported interfaces as stable",
            "Document any signature changes",
            "Add compatibility handling where needed",
        ],
    },
    "bug-isolation": {
        "id": "bug-isolation",
        "title": "Bug Isolation",
        "rules": [
            "Constrain edits to root-cause path",
            "Require symptom->cause->fix chain",
            "Avoid opportunistic unrelated cleanup",
        ],
    },
    "regression-guard": {
        "id": "regression-guard",
        "title": "Regression Guard",
        "rules": [
            "Retain behavior outside ticket scope",
            "Prefer explicit validation points",
            "Flag risk areas for verification",
        ],
    },
    "domain-ownership": {
        "id": "domain-ownership",
        "title": "Domain Ownership Guard",
        "rules": [
            "Each file belongs to exactly one bounded context (service)",
            "Files marked REFERENCE_ONLY must not be modified — read for context only",
            "DO_NOT_MODIFY files are hard-rejected at every pipeline layer",
            "UNKNOWN service ownership → REQUIRE_VERIFICATION, not silent ALLOW",
            "Relevant ≠ Authorized — a file can be relevant but architecturally wrong",
            "Check architecture_model.yaml for service ownership definitions",
        ],
    },
}


class SkillLoader:
    """Load strategy skills from defaults + optional repo-local overrides."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path)

    def load_for_tags(self, tags: List[str], max_skills: int = 6) -> Tuple[List[dict], str]:
        skills: List[dict] = []
        local_index = self._load_local_skill_index()

        for tag in tags:
            if len(skills) >= max_skills:
                break
            skill = local_index.get(tag) or DEFAULT_SKILLS.get(tag)
            if skill:
                skills.append(skill)

        guidance = self._build_guidance(skills)
        return skills, guidance

    def load_repo_invariants(self) -> List[str]:
        """Return repository-specific invariant rules for this workspace, or [] when none apply.

        Resolution order: workspace override (.aviator/skills/cloudforge_invariants.json) wins;
        otherwise the packaged skill is used only when a marker directory exists in the workspace.
        """
        override = self.workspace_path / ".aviator" / "skills" / "cloudforge_invariants.json"
        if not override.is_file():
            override = self.workspace_path / ".aviator" / "skills" / "contentbridge_invariants.json"
        if override.is_file():
            src = override
        elif any((self.workspace_path / m).is_dir() for m in REPO_INVARIANT_MARKERS):
            src = PACKAGED_REPO_INVARIANTS
        else:
            return []
        if not src.is_file():
            return []
        try:
            raw = json.loads(src.read_text(encoding="utf-8"))
            rules = raw.get("rules") if isinstance(raw, dict) else raw
            return [str(r) for r in (rules or [])]
        except Exception:
            return []

    @staticmethod
    def format_repo_invariants(rules: List[str]) -> str:
        """Render rules as a numbered prompt block, or "" when there are none."""
        clean = [str(r).strip() for r in (rules or []) if str(r).strip()]
        if not clean:
            return ""
        lines = ["### REPOSITORY-SPECIFIC INVARIANTS (loaded from skills):", ""]
        for idx, rule in enumerate(clean, 1):
            lines.append(f"{idx}. {rule}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _load_local_skill_index(self) -> Dict[str, dict]:
        index: Dict[str, dict] = {}
        candidates = [
            self.workspace_path / ".aviator" / "skills",
            self.workspace_path / "skills",
        ]
        for skill_dir in candidates:
            if not skill_dir.exists() or not skill_dir.is_dir():
                continue
            for fp in skill_dir.glob("*.json"):
                try:
                    raw = json.loads(fp.read_text(encoding="utf-8"))
                    sid = raw.get("id")
                    if sid:
                        index[str(sid)] = raw
                except Exception:
                    continue
        return index

    def _build_guidance(self, skills: List[dict]) -> str:
        if not skills:
            return "No skills loaded. Use standard grounded planning discipline."

        lines: List[str] = ["=== ACTIVE SKILLS ==="]
        for skill in skills:
            lines.append(f"- {skill.get('id', 'unknown')}: {skill.get('title', 'Untitled')}")
            for rule in skill.get("rules", [])[:4]:
                lines.append(f"  * {rule}")
        lines.append("=== END ACTIVE SKILLS ===")
        return "\n".join(lines)
