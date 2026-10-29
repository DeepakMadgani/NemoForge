from __future__ import annotations

from typing import List, Optional

from ticket_to_code.models import StrategyDecision


class StrategyEngine:
    """Simple deterministic strategy selector for pre-planning policy."""

    def decide(self, ticket, investigation_result=None, requirements=None, runtime_diagnosis=None) -> StrategyDecision:
        title = (getattr(ticket, "title", "") or "").lower()
        description = (getattr(ticket, "description", "") or "").lower()
        text = f"{title} {description}"

        category = "general"
        profile = "balanced"
        skill_tags: List[str] = ["grounded-planning", "minimal-diff"]
        rationale: List[str] = ["Default balanced strategy selected"]

        ticket_type = getattr(investigation_result, "ticket_type", None)
        ticket_type_value = getattr(ticket_type, "value", str(ticket_type or "")).lower()

        # Enhancement 1: Runtime Log Diagnosis — overrides strategy when runtime
        # error data is available, steering the pipeline to prioritise log evidence.
        if runtime_diagnosis and getattr(runtime_diagnosis, "has_runtime_data", False):
            category = "runtime-debug"
            profile = "log-first"
            skill_tags = ["log-analysis", "stack-trace-guided", "minimal-diff"]
            rationale = [
                "Runtime log data detected — using log-first strategy",
                f"Error type: {getattr(runtime_diagnosis, 'error_type', 'unknown')}",
                f"Root file: {getattr(runtime_diagnosis, 'root_file', 'unknown')}",
            ]
        elif any(k in text for k in ["rename", "refactor", "rename symbol"]):
            category = "refactor"
            profile = "safe-refactor"
            skill_tags = ["rename-propagation", "minimal-diff", "public-api-safety"]
            rationale = ["Rename/refactor keywords detected"]
        elif any(k in text for k in ["api", "contract", "endpoint", "schema"]):
            category = "api-change"
            profile = "contract-first"
            skill_tags = ["api-contract", "backward-compat", "consumer-impact"]
            rationale = ["API/contract keywords detected"]
        elif any(k in text for k in ["config", "yaml", "json", "env", "setting", "version"]) or "version" in ticket_type_value:
            category = "config-migration"
            profile = "config-safe"
            skill_tags = ["config-migration", "version-propagation", "runtime-compat"]
            rationale = ["Configuration/version indicators detected"]
        elif any(k in text for k in ["test", "failing", "regression"]):
            category = "bug-fix"
            profile = "evidence-first"
            skill_tags = ["bug-isolation", "minimal-diff", "regression-guard"]
            rationale = ["Bug/regression indicators detected"]

        if requirements and getattr(requirements, "technical_requirements", None):
            rationale.append("Technical requirements present and used as grounding")

        return StrategyDecision(
            category=category,
            strategy_profile=profile,
            required_evidence=[
                "at least one owner file evidence snippet",
                "candidate->requirement mapping"
            ],
            required_validations=[
                "semantic_validation",
                "patch_gate",
                "build_execution"
            ],
            stop_conditions=[
                "no grounded candidates",
                "missing required evidence",
                "build skipped"
            ],
            skill_tags=skill_tags,
            rationale=rationale,
            confidence=0.8,
            explanation=(
                f"Selected strategy profile '{profile}' for category '{category}' "
                f"based on ticket language and deterministic keyword signals."
            ),
        )
