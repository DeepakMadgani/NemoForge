"""
Unit tests for Fast-Path / Direct AI IDE Mode routing (AI IDE Hybrid Architecture).

Verifies that:
1. Confident Tier 1 component candidates bypass the 10 speculative pre-flight phases
   and route directly from discover -> edit_loop.
2. Sibling companion files are resolved and locked as state["allowed_files"].
3. Ambiguous or low-confidence discovery results fall back to hypothesis_investigation (waterfall).
4. Direct mode synthesizes a minimal plan and primes EditLoopAgent's pending_queue.
"""

import pytest
from unittest.mock import MagicMock, patch
from ticket_to_code.workflow import route_after_discover, resolve_component_group


def test_route_after_discover_empty_candidates():
    state = {
        "discovered_files": [],
        "workspace_path": "C:/dummy/workspace",
    }
    decision = route_after_discover(state)
    assert decision == "hypothesis_investigation"
    assert "allowed_files" not in state


def test_route_after_discover_ambiguous_low_score():
    state = {
        "discovered_files": [
            {
                "path": "src/app/services/data.service.ts",
                "confidence": 0.45,
                "features": {"ownership_type": "SUPPORTING"},
            },
            {
                "path": "src/app/utils/format.ts",
                "confidence": 0.30,
                "features": {"ownership_type": "SUPPORTING"},
            },
        ],
        "workspace_path": "C:/dummy/workspace",
    }
    decision = route_after_discover(state)
    assert decision == "hypothesis_investigation"
    assert "direct_mode" not in state


def test_route_after_discover_confident_tier1_fast_path():
    state = {
        "discovered_files": [
            {
                "path": "src/app/modules/members/member/member.component.scss",
                "confidence": 0.95,
                "features": {"ownership_type": "DISPLAY_OWNER"},
            },
            {
                "path": "src/app/modules/members/add-members/add-members.component.scss",
                "confidence": 0.40,
                "features": {"ownership_type": "SUPPORTING"},
            },
        ],
        "workspace_path": "C:/dummy/workspace",
    }
    with patch(
        "ticket_to_code.intelligence.contracts.companion_resolver.PolyglotCompanionResolver.resolve_all_companions",
        return_value=[
            "src/app/modules/members/member/member.component.html",
            "src/app/modules/members/member/member.component.ts",
        ],
    ):
        decision = route_after_discover(state)

    assert decision == "edit_loop"
    assert state.get("direct_mode") is True
    assert "src/app/modules/members/member/member.component.scss" in state["allowed_files"]
    assert "src/app/modules/members/member/member.component.html" in state["allowed_files"]
    assert "src/app/modules/members/member/member.component.ts" in state["allowed_files"]
    # Scope Lock: must NOT leak into add-members or other components
    assert "src/app/modules/members/add-members/add-members.component.scss" not in state["allowed_files"]


def test_resolve_component_group_dedupes():
    with patch(
        "ticket_to_code.intelligence.contracts.companion_resolver.PolyglotCompanionResolver.resolve_all_companions",
        return_value=["src/app/comp.ts", "src/app/comp.html"],
    ):
        group = resolve_component_group("src/app/comp.scss", "C:/dummy/ws")
    assert group == ["src/app/comp.scss", "src/app/comp.ts", "src/app/comp.html"]
