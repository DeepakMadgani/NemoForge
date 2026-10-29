"""
Adaptive Strategy Replanner — Enhancement 4

Analyzes WHY a build/candidate fix failed and selects a fundamentally
different strategy instead of blindly retrying the same approach.

Records FAILED_STRATEGY and STRATEGY_SWITCH memory types so the system
never repeats a strategy that already failed for the same ticket.

Safety:
  - Never writes code or modifies files directly
  - Only produces a new StrategyDecision for the pipeline to follow
  - Tracks failed strategies to prevent loops
  - Max 3 strategy switches per run (hard limit)

Author: Deepak Madgani
Date: July 2026
"""

import logging
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class FailureCategory(str, Enum):
    """Categories of build/validation failure."""
    WRONG_FILE = "wrong_file"           # Localization targeted wrong file
    CODE_ERROR = "code_error"           # Code logic was wrong
    MISSING_DEPENDENCY = "missing_dep"  # Package not installed
    CONFIG_ERROR = "config_error"       # Config file issue
    TYPE_ERROR = "type_error"           # Type mismatch / interface violation
    SCOPE_VIOLATION = "scope_violation" # Patch gate rejected
    UNKNOWN = "unknown"


class StrategySwitch(BaseModel):
    """Record of a strategy switch during a run."""
    from_strategy: str = Field(..., description="Strategy that failed")
    to_strategy: str = Field(..., description="New strategy selected")
    failure_category: FailureCategory = Field(...)
    reason: str = Field(..., description="Why the switch was made")
    attempt_number: int = Field(0, description="Which retry attempt triggered this")


class ReplanDecision(BaseModel):
    """Decision on whether and how to replan."""
    should_switch: bool = Field(False, description="Whether strategy should change")
    failure_category: FailureCategory = Field(FailureCategory.UNKNOWN)
    new_category: Optional[str] = Field(None, description="New strategy category")
    new_profile: Optional[str] = Field(None, description="New strategy profile")
    new_skill_tags: List[str] = Field(default_factory=list)
    reason: str = Field("", description="Human-readable explanation")
    files_to_blacklist: List[str] = Field(
        default_factory=list,
        description="Files that should be blacklisted for next attempt"
    )


# ============================================================================
# FAILURE CLASSIFIER
# ============================================================================

# Error patterns → failure categories
_FAILURE_PATTERNS: Dict[FailureCategory, List[str]] = {
    FailureCategory.MISSING_DEPENDENCY: [
        "modulenotfounderror", "importerror", "no module named",
        "cannot find module", "classnotfoundexception",
        "noclassdeffounderror", "package not found",
    ],
    FailureCategory.TYPE_ERROR: [
        "ts2339", "ts2345", "ts2304", "ts2551",
        "type error", "type mismatch", "incompatible types",
        "cannot convert", "argument type",
    ],
    FailureCategory.WRONG_FILE: [
        "property does not exist on type",
        "cannot find name", "is not a member of",
        "no such file or directory", "file not found",
    ],
    FailureCategory.CONFIG_ERROR: [
        "yaml", "json.decoder", "invalid config",
        "environment variable", "connection refused",
        "permission denied",
    ],
    FailureCategory.SCOPE_VIOLATION: [
        "scope violation", "read-only", "patchvalidator rejected",
        "ownership read-only", "build-fix scope",
    ],
}


# ============================================================================
# STRATEGY ALTERNATIVES
# ============================================================================

_ALTERNATIVE_STRATEGIES: Dict[FailureCategory, Dict[str, Any]] = {
    FailureCategory.WRONG_FILE: {
        "category": "rediscovery",
        "profile": "broader-search",
        "skill_tags": ["expanded-discovery", "cross-module", "minimal-diff"],
        "reason": "Previous file selection was wrong — broadening search scope",
    },
    FailureCategory.CODE_ERROR: {
        "category": "conservative-fix",
        "profile": "minimal-change",
        "skill_tags": ["minimal-diff", "single-method", "regression-guard"],
        "reason": "Code logic error — switching to minimal-change strategy",
    },
    FailureCategory.MISSING_DEPENDENCY: {
        "category": "dependency-first",
        "profile": "install-then-build",
        "skill_tags": ["dependency-resolution", "manifest-update"],
        "reason": "Missing dependency — resolve package before retrying code",
    },
    FailureCategory.CONFIG_ERROR: {
        "category": "config-migration",
        "profile": "config-safe",
        "skill_tags": ["config-migration", "env-tracing", "runtime-compat"],
        "reason": "Configuration error — switching to config-safe strategy",
    },
    FailureCategory.TYPE_ERROR: {
        "category": "type-safe-fix",
        "profile": "interface-first",
        "skill_tags": ["type-checking", "interface-compat", "minimal-diff"],
        "reason": "Type error — switching to interface-first strategy",
    },
    FailureCategory.SCOPE_VIOLATION: {
        "category": "scope-aware",
        "profile": "strict-scope",
        "skill_tags": ["scope-enforcement", "minimal-diff", "planner-aligned"],
        "reason": "Scope violation — switching to strict-scope strategy",
    },
}


class AdaptiveReplanner:
    """
    Analyzes build/validation failures and decides whether to switch
    strategy rather than blindly retrying the same approach.

    Usage:
        replanner = AdaptiveReplanner()
        decision = replanner.analyze_failure(build_errors, attempt_num, failed_strategies)
        if decision.should_switch:
            # Apply new strategy
    """

    MAX_SWITCHES = 3  # Hard limit on strategy switches per run

    def __init__(self):
        self._switches: List[StrategySwitch] = []

    # ------------------------------------------------------------------
    # 1. Classify failure
    # ------------------------------------------------------------------

    def classify_failure(self, error_messages: List[str]) -> FailureCategory:
        """
        Determine what category of failure occurred from error messages.
        """
        error_text = "\n".join(error_messages).lower()

        # Score each category by pattern matches
        scores: Dict[FailureCategory, int] = {}
        for category, patterns in _FAILURE_PATTERNS.items():
            score = sum(1 for p in patterns if p in error_text)
            if score > 0:
                scores[category] = score

        if not scores:
            return FailureCategory.UNKNOWN

        # Return highest scoring category
        best = max(scores, key=scores.get)
        logger.info(f"AdaptiveReplanner: classified failure as {best.value} (score={scores[best]})")
        return best

    # ------------------------------------------------------------------
    # 2. Analyze and decide
    # ------------------------------------------------------------------

    def analyze_failure(
        self,
        error_messages: List[str],
        attempt_number: int,
        current_strategy: str = "general",
        failed_strategies: Optional[List[str]] = None,
    ) -> ReplanDecision:
        """
        Analyze a build/validation failure and decide whether to switch strategy.

        Args:
            error_messages: Build error messages.
            attempt_number: Current retry attempt number.
            current_strategy: Current strategy category.
            failed_strategies: List of previously failed strategy categories.

        Returns:
            ReplanDecision with switch recommendation.
        """
        failed_strategies = failed_strategies or []

        # Hard limit check
        if len(self._switches) >= self.MAX_SWITCHES:
            logger.warning(
                f"AdaptiveReplanner: max switches ({self.MAX_SWITCHES}) reached. "
                f"No more strategy changes."
            )
            return ReplanDecision(
                should_switch=False,
                reason=f"Max strategy switches ({self.MAX_SWITCHES}) exhausted",
            )

        # Only consider switching after first attempt fails
        if attempt_number < 1:
            return ReplanDecision(
                should_switch=False,
                reason="First attempt — not switching yet",
            )

        # Classify the failure
        category = self.classify_failure(error_messages)
        if category == FailureCategory.UNKNOWN:
            return ReplanDecision(
                should_switch=False,
                failure_category=category,
                reason="Could not classify failure — retrying with same strategy",
            )

        # Get alternative strategy
        alt = _ALTERNATIVE_STRATEGIES.get(category)
        if not alt:
            return ReplanDecision(
                should_switch=False,
                failure_category=category,
                reason=f"No alternative strategy for {category.value}",
            )

        # Don't switch to a strategy that already failed
        if alt["category"] in failed_strategies:
            logger.info(
                f"AdaptiveReplanner: {alt['category']} already failed — skipping"
            )
            return ReplanDecision(
                should_switch=False,
                failure_category=category,
                reason=f"Alternative '{alt['category']}' already failed — no switch",
            )

        # Don't switch to the same strategy
        if alt["category"] == current_strategy:
            return ReplanDecision(
                should_switch=False,
                failure_category=category,
                reason=f"Alternative is same as current ({current_strategy}) — no switch",
            )

        # Build blacklist for wrong_file failures
        files_to_blacklist = []
        if category == FailureCategory.WRONG_FILE:
            # Extract file paths from error messages
            import re
            for err in error_messages:
                # Match file paths in error messages
                paths = re.findall(r'[A-Za-z]:\\[^\s:]+|/[^\s:]+', err)
                files_to_blacklist.extend(paths)

        # Record the switch
        switch = StrategySwitch(
            from_strategy=current_strategy,
            to_strategy=alt["category"],
            failure_category=category,
            reason=alt["reason"],
            attempt_number=attempt_number,
        )
        self._switches.append(switch)

        logger.info(
            f"AdaptiveReplanner: SWITCHING strategy "
            f"{current_strategy} -> {alt['category']} "
            f"(reason: {alt['reason']})"
        )

        return ReplanDecision(
            should_switch=True,
            failure_category=category,
            new_category=alt["category"],
            new_profile=alt["profile"],
            new_skill_tags=alt["skill_tags"],
            reason=alt["reason"],
            files_to_blacklist=files_to_blacklist,
        )

    # ------------------------------------------------------------------
    # 3. Memory recording helpers
    # ------------------------------------------------------------------

    def get_switches(self) -> List[StrategySwitch]:
        """Return all strategy switches made during this run."""
        return list(self._switches)

    def get_failed_strategies(self) -> List[str]:
        """Return list of strategy categories that have failed."""
        return [s.from_strategy for s in self._switches]
