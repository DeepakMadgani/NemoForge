"""Stage-depth scoring and path selection.

Implements the agreed scoring formula and gate overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class StageDepthInput:
    complexity: float
    coupling: float
    risk: float
    certainty: float
    constraint_strictness: float


@dataclass
class StageDepthDecision:
    path: str
    stages: int
    score: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def choose_stage_depth(
    complexity: float,
    coupling: float,
    risk: float,
    certainty: float,
    constraint_strictness: float,
) -> StageDepthDecision:
    complexity = _clamp01(complexity)
    coupling = _clamp01(coupling)
    risk = _clamp01(risk)
    certainty = _clamp01(certainty)
    constraint_strictness = _clamp01(constraint_strictness)

    depth_score = (
        0.30 * complexity
        + 0.25 * coupling
        + 0.25 * risk
        + 0.15 * constraint_strictness
        + 0.05 * (1 - certainty)
    )

    if depth_score <= 0.30:
        return StageDepthDecision(path="fast", stages=4, score=depth_score)
    if depth_score <= 0.55:
        return StageDepthDecision(path="standard", stages=6, score=depth_score)
    if depth_score <= 0.75:
        return StageDepthDecision(path="strict", stages=8, score=depth_score)
    return StageDepthDecision(path="high_risk", stages=10, score=depth_score)


def apply_gate_overrides(
    base: StageDepthDecision,
    *,
    has_no_hardcoding_constraint: bool,
    introduced_literal_duplicates: bool,
    forbidden_file_touched: bool,
    certainty_after_second_discovery: float,
) -> Dict[str, str]:
    """Return override actions for orchestrator policy handling."""
    overrides: Dict[str, str] = {}

    if has_no_hardcoding_constraint and introduced_literal_duplicates:
        overrides["path"] = "strict"
        overrides["reason"] = "no_hardcoding constraint violated by duplicate literals"

    if forbidden_file_touched:
        overrides["terminal"] = "failed_closed"
        overrides["reason"] = "forbidden file touched"

    if _clamp01(certainty_after_second_discovery) < 0.35:
        overrides["terminal"] = "need_more_info"
        overrides["reason"] = "certainty below 0.35 after second discovery pass"

    if "path" not in overrides:
        overrides["path"] = base.path

    return overrides
