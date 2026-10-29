"""Tests for PART 1 (Token Phase Attribution) and PART 2 (Build Diagnostic Reporting).

PART 1 tests verify that _phase_tracked_node() correctly wraps graph-node
functions with phase lifecycle tracking without altering behaviour.

PART 2 tests verify that build diagnostic fields from TicketToCodeState
are correctly extracted, surfaced in the workflow explanation, and
propagated through the terminal SSE broadcast data payload.

Strategy:
  - PART 1: Import _phase_tracked_node from workflow.py and exercise it
    with mock RunContext objects wired through the global _transient_store.
  - PART 2: Since main.py has too many heavy imports (FastAPI, git, etc.)
    to import in the test environment, we extract the two pure functions
    (_extract_workflow_outputs, _build_workflow_explanation) by reading
    the source file and exec'ing just the target functions.  This isolates
    the test from import-chain issues while verifying the REAL production code.
"""

import ast
import importlib
import sys
import textwrap
import types
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ============================================================================
# PART 2 Helpers: Extract pure functions from main.py without full import
# ============================================================================

def _load_functions_from_main():
    """Parse main.py and extract the two target functions via exec().

    This avoids importing the full main.py module (which requires FastAPI,
    git, history_store, dotenv, etc.).
    """
    main_path = Path(__file__).resolve().parents[1] / "chatbot" / "backend" / "main.py"
    source = main_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find the function definitions we need
    target_fns = {"_extract_workflow_outputs", "_build_workflow_explanation"}
    fn_sources: Dict[str, str] = {}
    lines = source.splitlines(keepends=True)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_fns:
            # Extract from source lines (1-indexed to 0-indexed)
            start = node.lineno - 1
            end = node.end_lineno  # end_lineno is inclusive in ast
            fn_source = "".join(lines[start:end])
            fn_sources[node.name] = fn_source

    assert len(fn_sources) == 2, (
        f"Expected 2 functions, found {list(fn_sources.keys())}"
    )

    # Build a minimal namespace with required imports
    namespace: Dict[str, Any] = {
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Any": Any,
    }

    for fn_name, fn_src in fn_sources.items():
        exec(compile(fn_src, f"<main.py::{fn_name}>", "exec"), namespace)

    return namespace["_extract_workflow_outputs"], namespace["_build_workflow_explanation"]


# Load once at module level
_extract_workflow_outputs, _build_workflow_explanation = _load_functions_from_main()


# ============================================================================
# PART 1 Tests: _phase_tracked_node wrapper
# ============================================================================


class TestPhaseTrackedNode:
    """Tests for the _phase_tracked_node wrapper in workflow.py."""

    def _make_wrapper(self):
        """Import _phase_tracked_node from workflow.py."""
        from ticket_to_code.workflow import _phase_tracked_node
        return _phase_tracked_node

    def _make_state_with_run_ctx(self, run_ctx):
        """Create a mock state dict and wire run_ctx into the transient store."""
        from ticket_to_code.workflow import _transient_store, _transient_lock
        ticket = MagicMock()
        ticket.ticket_id = "test-tid-phase"
        state = {"ticket": ticket}
        with _transient_lock:
            _transient_store["test-tid-phase"] = {"run_ctx": run_ctx}
        return state

    def _cleanup_transient(self, tid="test-tid-phase"):
        from ticket_to_code.workflow import _transient_store, _transient_lock
        with _transient_lock:
            _transient_store.pop(tid, None)

    # Case 1: Normal node — wrapper starts/ends phase correctly
    def test_normal_node_phase_lifecycle(self):
        """Wrapper starts phase before call and ends phase after call."""
        _phase_tracked_node = self._make_wrapper()
        run_ctx = MagicMock()
        state = self._make_state_with_run_ctx(run_ctx)

        call_log = []

        def fake_node(s):
            call_log.append("node_called")
            return {"result": "ok"}

        # Signature: _phase_tracked_node(node_name, node_fn)
        wrapped = _phase_tracked_node("test_phase", fake_node)
        result = wrapped(state)

        assert result == {"result": "ok"}
        assert "node_called" in call_log
        run_ctx.start_phase.assert_called_once_with("test_phase")
        run_ctx.end_phase.assert_called_once_with("test_phase")
        self._cleanup_transient()

    # Case 2: Node raises exception — end_phase still fires (finally block)
    def test_exception_still_ends_phase(self):
        """Phase is ended even when the wrapped node raises an exception."""
        _phase_tracked_node = self._make_wrapper()
        run_ctx = MagicMock()
        state = self._make_state_with_run_ctx(run_ctx)

        def exploding_node(s):
            raise ValueError("BOOM")

        wrapped = _phase_tracked_node("explode_phase", exploding_node)

        with pytest.raises(ValueError, match="BOOM"):
            wrapped(state)

        run_ctx.start_phase.assert_called_once_with("explode_phase")
        run_ctx.end_phase.assert_called_once_with("explode_phase")
        self._cleanup_transient()

    # Case 3: No RunContext available — node executes without error
    def test_no_run_context_graceful(self):
        """Wrapper is safe when no run_ctx is in the transient store."""
        _phase_tracked_node = self._make_wrapper()

        def simple_node(s):
            return {"status": "done"}

        # No ticket → _get_transient returns None
        wrapped = _phase_tracked_node("no_ctx_phase", simple_node)
        state = {}

        result = wrapped(state)
        assert result == {"status": "done"}

    # Case 4: Ticket present but no run_ctx in store
    def test_ticket_but_no_run_ctx(self):
        """Wrapper is safe when ticket exists but run_ctx is missing."""
        _phase_tracked_node = self._make_wrapper()
        from ticket_to_code.workflow import _transient_store, _transient_lock

        ticket = MagicMock()
        ticket.ticket_id = "tid-no-ctx"
        with _transient_lock:
            _transient_store["tid-no-ctx"] = {}  # no run_ctx key
        state = {"ticket": ticket}

        def simple_node(s):
            return {"val": 42}

        wrapped = _phase_tracked_node("empty_phase", simple_node)
        result = wrapped(state)
        assert result == {"val": 42}
        self._cleanup_transient("tid-no-ctx")

    # Case 5: investigate_node — mid-node RunContext creation
    def test_investigate_creates_run_context_midway(self):
        """When a node creates RunContext mid-execution, wrapper re-fetches
        the context and ends the phase on the new context."""
        _phase_tracked_node = self._make_wrapper()
        from ticket_to_code.workflow import _transient_store, _transient_lock

        ticket = MagicMock()
        ticket.ticket_id = "tid-investigate"
        # Start with NO run_ctx
        with _transient_lock:
            _transient_store["tid-investigate"] = {}
        state = {"ticket": ticket}

        new_run_ctx = MagicMock()

        def investigate_like_node(s):
            # Simulates investigate_node creating RunContext during execution
            with _transient_lock:
                _transient_store["tid-investigate"]["run_ctx"] = new_run_ctx
            return {"investigated": True}

        wrapped = _phase_tracked_node("investigate", investigate_like_node)
        result = wrapped(state)

        assert result == {"investigated": True}
        # The wrapper should end_phase on the newly-created context
        new_run_ctx.end_phase.assert_called_once_with("investigate")
        self._cleanup_transient("tid-investigate")

    # Case 6: Return value is preserved exactly
    def test_return_value_preserved(self):
        """Wrapped function returns identical result to unwrapped."""
        _phase_tracked_node = self._make_wrapper()
        run_ctx = MagicMock()
        state = self._make_state_with_run_ctx(run_ctx)

        complex_result = {"a": [1, 2, 3], "b": {"nested": True}, "c": None}

        def complex_node(s):
            return complex_result

        wrapped = _phase_tracked_node("complex", complex_node)
        result = wrapped(state)
        assert result is complex_result  # identity check, not just equality
        self._cleanup_transient()


# ============================================================================
# PART 2 Tests: Build Diagnostic Reporting
# ============================================================================


class TestExtractWorkflowOutputsDiagnostics:
    """Tests for build_diagnostics extraction in _extract_workflow_outputs."""

    # Case A: Infrastructure-only failure populates build_diagnostics
    def test_infrastructure_only_diagnostics(self):
        """When build_infrastructure_only is True, diagnostics reflect it."""
        state = {
            "build_infrastructure_only": True,
            "build_differential_accept": False,
            "build_infrastructure_blocked": True,
            "build_diagnostic_summary": "infrastructure=3",
            "pre_existing_decision": "leave",
            "authorized_pre_existing_files": [],
        }
        result = _extract_workflow_outputs(state)
        diag = result.get("build_diagnostics")
        assert diag is not None, "build_diagnostics should be present"
        assert diag["infrastructure_only"] is True
        assert diag["infrastructure_blocked"] is True
        assert diag["differential_accept"] is False
        assert diag["summary"] == "infrastructure=3"

    # Case B: Differential-accept (pre-existing only)
    def test_differential_accept_diagnostics(self):
        """When all errors are pre-existing, differential_accept is True."""
        state = {
            "build_infrastructure_only": False,
            "build_differential_accept": True,
            "build_infrastructure_blocked": False,
            "build_diagnostic_summary": "pre_existing=5",
            "pre_existing_decision": "fix",
            "authorized_pre_existing_files": ["src/foo.java", "src/bar.java"],
        }
        result = _extract_workflow_outputs(state)
        diag = result.get("build_diagnostics")
        assert diag is not None
        assert diag["differential_accept"] is True
        assert diag["infrastructure_only"] is False
        assert diag["pre_existing_decision"] == "fix"
        assert "src/foo.java" in diag["authorized_pre_existing_files"]

    # Case C: No diagnostic fields → build_diagnostics is None
    def test_no_diagnostics_when_absent(self):
        """When no diagnostic fields are present, build_diagnostics is None."""
        state = {
            "generated_code": [],
            "status": "completed",
        }
        result = _extract_workflow_outputs(state)
        assert result.get("build_diagnostics") is None

    # State contract: all 5 diagnostic fields are extracted correctly
    def test_state_contract_all_fields(self):
        """All diagnostic state fields map to the output dict correctly."""
        state = {
            "build_differential_accept": None,
            "build_infrastructure_only": True,
            "build_infrastructure_blocked": False,
            "build_diagnostic_summary": "ticket_introduced=2, infrastructure=1",
            "pre_existing_decision": "stop",
            "authorized_pre_existing_files": [],
        }
        result = _extract_workflow_outputs(state)
        diag = result.get("build_diagnostics")
        # build_infrastructure_only is not None, so diagnostics should exist
        assert diag is not None
        assert diag["infrastructure_only"] is True
        assert diag["infrastructure_blocked"] is False
        assert diag["differential_accept"] is None  # was None in state
        assert "ticket_introduced=2" in diag["summary"]
        assert diag["pre_existing_decision"] == "stop"


class TestBuildWorkflowExplanationDiagnostics:
    """Tests for diagnostic surfacing in _build_workflow_explanation."""

    def test_infra_failure_explanation(self):
        """Failed status with infra diagnostics produces infrastructure-specific explanation."""
        result = _build_workflow_explanation(
            state={"status": "failed", "current_phase": "build"},
            workflow_output={
                "build_diagnostics": {
                    "infrastructure_only": True,
                    "infrastructure_blocked": True,
                    "differential_accept": False,
                    "summary": "infrastructure=3",
                    "pre_existing_decision": "leave",
                    "authorized_pre_existing_files": [],
                },
            },
        )
        assert "infrastructure" in result["summary"].lower()
        assert "source code was not the issue" in result["summary"].lower()
        assert result["next_action"] == "Resolve the environment/dependency issue, then re-run."
        assert result.get("build_diagnostics") is not None

    def test_differential_accept_explanation(self):
        """Failed status with differential-accept produces pre-existing-specific explanation."""
        result = _build_workflow_explanation(
            state={"status": "failed", "current_phase": "build"},
            workflow_output={
                "build_diagnostics": {
                    "infrastructure_only": False,
                    "infrastructure_blocked": False,
                    "differential_accept": True,
                    "summary": "pre_existing=5",
                    "pre_existing_decision": "stop",
                    "authorized_pre_existing_files": [],
                },
            },
        )
        assert "pre-existing" in result["summary"].lower()
        assert "stop" in result["next_action"].lower()

    def test_differential_accept_fix_explanation(self):
        """Differential-accept with fix decision has specific next_action."""
        result = _build_workflow_explanation(
            state={"status": "failed", "current_phase": "build"},
            workflow_output={
                "build_diagnostics": {
                    "infrastructure_only": False,
                    "infrastructure_blocked": False,
                    "differential_accept": True,
                    "summary": "pre_existing=3",
                    "pre_existing_decision": "fix",
                    "authorized_pre_existing_files": ["src/legacy.java"],
                },
            },
        )
        assert "authorized for repair" in result["next_action"].lower()

    def test_generic_failure_explanation(self):
        """Failed status without diagnostics produces generic failure explanation."""
        result = _build_workflow_explanation(
            state={"status": "failed", "current_phase": "build"},
            workflow_output={},
        )
        assert "converge" in result["summary"].lower()
        assert result.get("build_diagnostics") is None

    def test_build_diagnostics_in_return_dict(self):
        """build_diagnostics key is present in explanation return dict."""
        result = _build_workflow_explanation(
            state={"status": "completed", "current_phase": "completed"},
            workflow_output={"build_diagnostics": {"summary": "none"}},
        )
        assert "build_diagnostics" in result
        assert result["build_diagnostics"]["summary"] == "none"
