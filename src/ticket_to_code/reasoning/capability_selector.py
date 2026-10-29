"""
Capability selector for the CC4E reasoning kernel.

Purpose:
- Pick the cheapest useful capability for the current missing need.
- Keep orchestration deterministic and state-aware (not prompt-only).

This module does NOT execute tools. It only recommends/guards action choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ticket_to_code.reasoning.belief_state import objective_coverage
from ticket_to_code.reasoning.tools import Tool, ToolContext


@dataclass
class SelectorDecision:
    need: str
    recommended_action: Optional[str]
    reason: str
    alternatives: List[str]


def _state(ctx: ToolContext) -> Dict[str, float]:
    has_owner = bool(ctx.candidate_files)
    has_file_context = bool(ctx.files_read)
    has_patch = bool(ctx.written_files)
    build_success = ctx.last_build_status in ("success", "skipped")
    build_ran = ctx.build_attempts > 0
    verification = bool(ctx.last_verify_passed)
    contradictions_open = sum(1 for c in (ctx.contradictions or []) if getattr(c, "status", "open") == "open")
    coverage = objective_coverage(ctx.objectives or [])
    return {
        "owner_candidates": 1.0 if has_owner else 0.0,
        "file_context": 1.0 if has_file_context else 0.0,
        "patch": 1.0 if has_patch else 0.0,
        "build_success": 1.0 if build_success else 0.0,
        "build_status": 1.0 if build_ran else 0.0,
        "verification": 1.0 if verification else 0.0,
        "feature_context": 1.0 if any(k.startswith("feature") for k in (ctx.files_read.keys() if ctx.files_read else [])) else 0.0,
        "objective_coverage": coverage,
        "contradictions_open": float(contradictions_open),
        "evidence_count": float(len(ctx.evidence_items or [])),
    }


def _missing_need(s: Dict[str, float]) -> str:
    if s.get("contradictions_open", 0.0) > 0:
        return "resolve_contradictions"
    if s["owner_candidates"] < 1.0:
        return "owner_candidates"
    if s["file_context"] < 1.0:
        return "file_context"
    if s["patch"] < 1.0:
        return "patch"
    if s["build_success"] < 1.0:
        return "build_success"
    if s["verification"] < 1.0:
        return "verification"
    if s.get("objective_coverage", 0.0) < 0.999:
        return "objective_coverage"
    return "none"


def _provides_need(tool: Tool, need: str) -> bool:
    return need in (tool.metadata.provides or [])


def _prereqs_met(tool: Tool, s: Dict[str, float]) -> bool:
    for pre in tool.metadata.prerequisites or []:
        if s.get(pre, 0.0) < 1.0:
            return False
    return True


def _expected_evidence_gain(tool: Tool, s: Dict[str, float], need: str) -> float:
    """State-conditional evidence gain (not static confidence)."""
    gain = float(tool.metadata.confidence_gain)
    provides = set(tool.metadata.provides or [])

    if need in provides:
        gain += 0.20

    # Diminishing returns: repeating an already-satisfied need has low gain.
    if need in {"owner_candidates", "file_context", "patch", "build_success", "verification"}:
        if s.get(need, 0.0) >= 1.0 and need in provides:
            gain *= 0.25

    # Contradictions should prioritize high-reliability checks.
    if need == "resolve_contradictions":
        if tool.name in {"read_file", "run_build", "verify_outcome", "find_owner"}:
            gain += 0.35
        else:
            gain *= 0.70

    # Early stage: discovery tools are better than patch/build actions.
    if s.get("owner_candidates", 0.0) < 1.0 and tool.name in {"write_patch", "run_build", "verify_outcome"}:
        gain *= 0.35

    return max(0.0, gain)


def _expected_objective_delta(tool: Tool, s: Dict[str, float], need: str) -> float:
    provides = set(tool.metadata.provides or [])
    delta = 0.0
    if "patch" in provides and s.get("file_context", 0.0) >= 1.0:
        delta += 0.25
    if "build_success" in provides and s.get("patch", 0.0) >= 1.0:
        delta += 0.30
    if "verification" in provides and s.get("build_success", 0.0) >= 1.0:
        delta += 0.35
    if need == "objective_coverage" and tool.name in {"verify_outcome", "run_build", "write_patch", "read_file"}:
        delta += 0.10
    return delta


def _loop_penalty(ctx: ToolContext, action: str) -> float:
    history = ctx.capability_history or []
    if not history:
        return 0.0
    last = history[-1].get("capability")
    if last != action:
        return 0.0
    # Repeating same action is okay once, then increasingly penalized.
    streak = 0
    for h in reversed(history):
        if h.get("capability") == action:
            streak += 1
        else:
            break
    return min(0.45, 0.08 * float(streak))


def select_capability(
    ctx: ToolContext,
    registry: Dict[str, Tool],
    preferred_capabilities: Optional[List[str]] = None,
) -> SelectorDecision:
    """Return deterministic next-action recommendation from metadata + current state."""
    preferred = set(preferred_capabilities or [])
    s = _state(ctx)
    need = _missing_need(s)

    if need == "none":
        return SelectorDecision(
            need=need,
            recommended_action=None,
            reason="All core needs already satisfied.",
            alternatives=[],
        )

    scored = []
    for name, tool in registry.items():
        if not _prereqs_met(tool, s):
            continue
        provides = set(tool.metadata.provides or [])
        # Candidate actions can be directly need-covering or contribute to objective coverage.
        if need not in provides and tool.name not in {"read_file", "find_owner", "run_build", "verify_outcome", "write_patch"}:
            continue

        evidence_gain = _expected_evidence_gain(tool, s, need)
        objective_delta = _expected_objective_delta(tool, s, need)
        preferred_bonus = 0.10 if name in preferred else 0.0
        experience_prior = 0.0
        store = getattr(ctx, "experience_store", None)
        if store is not None:
            try:
                experience_prior = float(store.prior_for_action(ctx.ticket_type, need, name, s))
            except Exception:
                experience_prior = 0.0
        loop_pen = _loop_penalty(ctx, name)
        cost_pen = 0.10 * float(tool.metadata.cost)
        risk_pen = 0.40 * float(getattr(tool.metadata, "risk", 0.2))

        # Expected next-state value.
        score = (
            (1.35 * evidence_gain)
            + (1.20 * objective_delta)
            + preferred_bonus
            + (0.90 * experience_prior)
            - cost_pen
            - risk_pen
            - loop_pen
        )
        scored.append((score, name, tool))

    if not scored:
        # Relaxation fallback: any prereq-met capability sorted by cost then gain.
        relaxed = []
        for name, tool in registry.items():
            if _prereqs_met(tool, s):
                relaxed.append((tool.metadata.cost, -tool.metadata.confidence_gain, name, tool))
        if not relaxed:
            return SelectorDecision(
                need=need,
                recommended_action=None,
                reason="No capability satisfies prerequisites.",
                alternatives=[],
            )
        relaxed.sort()
        _, _, best_name, _ = relaxed[0]
        alts = [r[2] for r in relaxed[1:4]]
        return SelectorDecision(
            need=need,
            recommended_action=best_name,
            reason="No direct provider for missing need; selected cheapest prerequisite-safe capability.",
            alternatives=alts,
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    alternatives = [x[1] for x in scored[1:4]]
    return SelectorDecision(
        need=need,
        recommended_action=best[1],
        reason=(
            f"Need '{need}'. Selected '{best[1]}' by best expected next-state value "
            f"(cost={best[2].metadata.cost}, risk={getattr(best[2].metadata, 'risk', 0.2)})."
        ),
        alternatives=alternatives,
    )


def should_override_action(
    proposed_action: str,
    decision: SelectorDecision,
    registry: Dict[str, Tool],
) -> bool:
    """Guardrail: override expensive/irrelevant actions when a clear need exists."""
    if decision.recommended_action is None:
        return False
    if proposed_action == decision.recommended_action:
        return False

    proposed = registry.get(proposed_action)
    recommended = registry.get(decision.recommended_action)
    if proposed is None or recommended is None:
        return False

    # If proposed action does not satisfy current need and is significantly costlier/riskier, override.
    proposed_covers_need = decision.need in (proposed.metadata.provides or [])
    costly_detour = proposed.metadata.cost >= (recommended.metadata.cost + 3)
    risky_detour = getattr(proposed.metadata, "risk", 0.2) > (getattr(recommended.metadata, "risk", 0.2) + 0.2)
    return (not proposed_covers_need) and (costly_detour or risky_detour)
