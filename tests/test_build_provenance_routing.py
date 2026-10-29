"""Tests for build-error provenance routing (check_build_status).

Validates that state produced by pre_fix_build_node survives the LangGraph
node boundary and that check_build_status routes correctly for every
provenance/decision combination.

Root cause under test: the 6 state keys written by pre_fix_build_node were
missing from TicketToCodeState TypedDict, so LangGraph silently dropped them.
"""

import types
from unittest.mock import Mock

import pytest


# ── Import the router under test ──────────────────────────────────────────────
from ticket_to_code.workflow import check_build_status, TicketToCodeState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_result(status_value: str, errors: list[str] | None = None):
    """Create a minimal BuildResult-like object."""
    br = Mock()
    br.status = Mock()
    br.status.value = status_value
    br.errors = errors or []
    return br


def _make_state(
    *,
    build_status: str = "failure",
    build_errors: list[str] | None = None,
    differential_accept: bool | None = None,
    infrastructure_only: bool | None = None,
    infrastructure_blocked: bool | None = None,
    diagnostic_summary: str | None = None,
    pre_existing_decision: str | None = None,
    authorized_pre_existing_files: list[str] | None = None,
    retry_attempt: int = 0,
    max_retry_attempts: int = 3,
    consecutive_identical_build_errors: int = 0,
) -> dict:
    """Build a state dict that mirrors what LangGraph provides to the router."""
    state: dict = {
        "build_result": _build_result(build_status, build_errors),
        "retry_attempt": retry_attempt,
        "max_retry_attempts": max_retry_attempts,
        "consecutive_identical_build_errors": consecutive_identical_build_errors,
    }
    # Only set keys that pre_fix_build_node would have set — this lets us
    # test both the "key present" and "key absent" (pre-fix) scenarios.
    if differential_accept is not None:
        state["build_differential_accept"] = differential_accept
    if infrastructure_only is not None:
        state["build_infrastructure_only"] = infrastructure_only
    if infrastructure_blocked is not None:
        state["build_infrastructure_blocked"] = infrastructure_blocked
    if diagnostic_summary is not None:
        state["build_diagnostic_summary"] = diagnostic_summary
    if pre_existing_decision is not None:
        state["pre_existing_decision"] = pre_existing_decision
    if authorized_pre_existing_files is not None:
        state["authorized_pre_existing_files"] = authorized_pre_existing_files
    return state


# ============================================================================
# Test A: Pre-existing errors + leave → test (no fix_build)
# ============================================================================

class TestPreExistingLeave:
    """When all errors are pre-existing and user chose 'leave', skip repair and route to test."""

    def test_routes_to_test(self):
        state = _make_state(
            build_status="failure",
            build_errors=["TS2339: Property 'isProjectMember' does not exist"],
            differential_accept=True,
            pre_existing_decision="leave",
        )
        assert check_build_status(state) == "test"

    def test_does_not_route_to_fix_build(self):
        state = _make_state(
            build_status="failure",
            differential_accept=True,
            pre_existing_decision="leave",
        )
        result = check_build_status(state)
        assert result != "fix_build", (
            "Pre-existing errors with decision='leave' must NOT route to fix_build"
        )

    def test_default_decision_is_leave(self):
        """When pre_existing_decision is not set, default is 'leave'."""
        state = _make_state(
            build_status="failure",
            differential_accept=True,
            # pre_existing_decision NOT set → default "leave"
        )
        assert check_build_status(state) == "test"


# ============================================================================
# Test B: Pre-existing errors + fix → fix_build (repair authorized scope)
# ============================================================================

class TestPreExistingFix:
    """When user chose 'fix', route to fix_build for authorized scope."""

    def test_routes_to_fix_build(self):
        state = _make_state(
            build_status="failure",
            differential_accept=True,
            pre_existing_decision="fix",
            authorized_pre_existing_files=["src/app/some.component.ts"],
        )
        assert check_build_status(state) == "fix_build"

    def test_does_not_route_to_test(self):
        state = _make_state(
            build_status="failure",
            differential_accept=True,
            pre_existing_decision="fix",
        )
        result = check_build_status(state)
        assert result != "test", (
            "Pre-existing errors with decision='fix' must route to fix_build, not test"
        )


# ============================================================================
# Test C: Ticket-introduced errors → fix_build
# ============================================================================

class TestTicketIntroducedErrors:
    """When the ticket introduced new errors, always repair."""

    def test_routes_to_fix_build(self):
        state = _make_state(
            build_status="failure",
            build_errors=["TS2345: Argument of type 'string' is not assignable"],
            differential_accept=False,  # ticket errors present
            pre_existing_decision="leave",
        )
        assert check_build_status(state) == "fix_build"

    def test_routes_to_fix_build_no_classification(self):
        """When classification keys are absent (should not happen after fix,
        but testing the fallback), ticket errors still go to fix_build."""
        state = _make_state(
            build_status="failure",
            # No differential_accept / pre_existing_decision set
        )
        assert check_build_status(state) == "fix_build"


# ============================================================================
# Test D: Mixed errors → fix_build (ticket errors need repair)
# ============================================================================

class TestMixedErrors:
    """Mixed = ticket-introduced + pre-existing. Ticket errors dominate."""

    def test_routes_to_fix_build_with_leave(self):
        """differential_accept is False when ticket errors exist, even if
        pre-existing errors are also present."""
        state = _make_state(
            build_status="failure",
            differential_accept=False,  # ticket errors present → not differential accept
            pre_existing_decision="leave",
        )
        assert check_build_status(state) == "fix_build"

    def test_routes_to_fix_build_with_fix(self):
        state = _make_state(
            build_status="failure",
            differential_accept=False,
            pre_existing_decision="fix",
        )
        assert check_build_status(state) == "fix_build"


# ============================================================================
# Test E: Zero errors / success → test
# ============================================================================

class TestZeroErrors:
    """Build success takes the test path."""

    def test_success_routes_to_test(self):
        state = _make_state(build_status="success")
        assert check_build_status(state) == "test"

    def test_no_build_result_routes_to_memory_update(self):
        state: dict = {"build_result": None}
        assert check_build_status(state) == "memory_update"


# ============================================================================
# Test: Pre-existing + stop → memory_update (safe termination)
# ============================================================================

class TestPreExistingStop:
    """When user chose 'stop', terminate safely."""

    def test_routes_to_memory_update(self):
        state = _make_state(
            build_status="failure",
            differential_accept=True,
            pre_existing_decision="stop",
        )
        assert check_build_status(state) == "memory_update"


# ============================================================================
# Test: Infrastructure-only → memory_update (no source repair)
# ============================================================================

class TestInfrastructureOnly:
    """Infrastructure failures (deps/network) skip source repair."""

    def test_routes_to_memory_update(self):
        state = _make_state(
            build_status="failure",
            infrastructure_only=True,
            diagnostic_summary="infrastructure=3",
        )
        assert check_build_status(state) == "memory_update"


# ============================================================================
# Test: Stuck detection still works alongside provenance routing
# ============================================================================

class TestStuckDetection:
    """Existing stuck detection (consecutive identical errors) must still work."""

    def test_stuck_routes_to_memory_update(self):
        state = _make_state(
            build_status="failure",
            differential_accept=False,
            consecutive_identical_build_errors=2,
            retry_attempt=2,
        )
        assert check_build_status(state) == "memory_update"

    def test_max_retries_routes_to_memory_update(self):
        state = _make_state(
            build_status="failure",
            differential_accept=False,
            retry_attempt=3,
            max_retry_attempts=3,
        )
        assert check_build_status(state) == "memory_update"


# ============================================================================
# Test F: State keys are declared in TicketToCodeState TypedDict
# ============================================================================

class TestStateKeysInTypedDict:
    """The 6 keys written by pre_fix_build_node must be declared in the
    TypedDict so LangGraph preserves them across node boundaries."""

    REQUIRED_KEYS = [
        "build_differential_accept",
        "build_infrastructure_only",
        "build_infrastructure_blocked",
        "build_diagnostic_summary",
        "pre_existing_decision",
        "authorized_pre_existing_files",
    ]

    def test_all_keys_declared_in_typed_dict(self):
        """Each key must be present in TicketToCodeState.__annotations__."""
        annotations = TicketToCodeState.__annotations__
        missing = [k for k in self.REQUIRED_KEYS if k not in annotations]
        assert not missing, (
            f"Keys missing from TicketToCodeState TypedDict: {missing}. "
            f"LangGraph silently drops undeclared keys during state merge, "
            f"so pre_fix_build_node's output would not reach check_build_status."
        )

    def test_key_types_are_optional(self):
        """These keys are Optional because they are only set by pre_fix_build_node
        (not at workflow initialization). Non-Optional would crash on first access."""
        annotations = TicketToCodeState.__annotations__
        for key in self.REQUIRED_KEYS:
            ann = annotations.get(key)
            ann_str = str(ann)
            # Accept Optional[X], X | None, or typing.Optional[X]
            assert (
                "Optional" in ann_str
                or "None" in ann_str
            ), (
                f"TicketToCodeState.{key} should be Optional, got {ann_str}. "
                f"These keys are only populated mid-workflow by pre_fix_build_node."
            )


# ============================================================================
# Test: Routing decision matrix (exhaustive truth table)
# ============================================================================

class TestRoutingDecisionMatrix:
    """Exhaustive truth table for check_build_status routing decisions."""

    @pytest.mark.parametrize(
        "label, build_status, diff_accept, decision, infra_only, expected_route",
        [
            # Success
            ("success", "success", None, None, None, "test"),
            # Pre-existing only
            ("pre-existing+leave", "failure", True, "leave", False, "test"),
            ("pre-existing+default", "failure", True, None, False, "test"),
            ("pre-existing+fix", "failure", True, "fix", False, "fix_build"),
            ("pre-existing+stop", "failure", True, "stop", False, "memory_update"),
            # Infrastructure
            ("infra-only", "failure", False, "leave", True, "memory_update"),
            # Ticket-introduced
            ("ticket-errors+leave", "failure", False, "leave", False, "fix_build"),
            ("ticket-errors+fix", "failure", False, "fix", False, "fix_build"),
            # No classification (legacy/fallback)
            ("no-classification", "failure", None, None, None, "fix_build"),
        ],
    )
    def test_routing(
        self, label, build_status, diff_accept, decision, infra_only, expected_route
    ):
        state = _make_state(
            build_status=build_status,
            differential_accept=diff_accept,
            pre_existing_decision=decision,
            infrastructure_only=infra_only,
        )
        result = check_build_status(state)
        assert result == expected_route, (
            f"[{label}] Expected '{expected_route}' but got '{result}' "
            f"(diff_accept={diff_accept}, decision={decision}, infra={infra_only})"
        )
