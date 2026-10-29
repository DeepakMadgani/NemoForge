"""Verification gates must BLOCK, not fail open.

Covers the hardening pass:
  * patch_gate has a real conditional edge (blocked -> verification_failed)
  * outcome_check "error" is not success
  * build SKIPPED (no compiler ran) is routed forward but recorded as unverified
  * test runner error/timeout is never differential-accepted
  * baseline-backed differential acceptance
  * memory_update does not learn from test-failed / unverified runs
  * discovery hard stop is honored by the router
  * preflight fallbacks are NOT_EVALUATED with confidence 0
  * finalize_fast_path_state applies the outcome check to fast paths
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ticket_to_code.models import (
    BuildResult,
    BuildStatus,
    GeneratedCode,
    TestResult,
    TestStatus,
    ValueEdgeTicket,
)
import ticket_to_code.workflow as wf


def _ticket(tid="T-1"):
    return ValueEdgeTicket(ticket_id=tid, title="Add refund status", description="Add REFUNDED to OrderStatus")


# ---------------------------------------------------------------------------
# patch gate
# ---------------------------------------------------------------------------
def test_route_after_patch_gate_blocks_on_hard_failure():
    assert wf.route_after_patch_gate({"patch_gate_blocked": True, "patch_gate_failures": ["secret_scan:x"]}) == "verification_failed"
    assert wf.route_after_patch_gate({"patch_gate_blocked": False}) == "import_validation"
    assert wf.route_after_patch_gate({}) == "import_validation"


def test_graph_wires_patch_gate_conditionally():
    import inspect
    src = inspect.getsource(wf.create_ticket_to_code_graph)
    assert 'add_edge("patch_gate", "import_validation")' not in src
    assert "route_after_patch_gate" in src


# ---------------------------------------------------------------------------
# outcome check routing
# ---------------------------------------------------------------------------
def test_outcome_error_is_not_success_and_exhausts_to_failure():
    state = {"outcome_check_result": {"status": "error", "reason": "boom"}, "outcome_fix_attempt": 0, "max_outcome_fix_attempts": 3}
    assert wf.route_after_outcome_check(state) == "remediation_plan"
    state["outcome_fix_attempt"] = 3
    assert wf.route_after_outcome_check(state) == "verification_failed"


def test_outcome_attempt_budget_is_strict():
    state = {"outcome_check_result": {"status": "PARTIAL"}, "outcome_fix_attempt": 3, "max_outcome_fix_attempts": 3}
    assert wf.route_after_outcome_check(state) == "verification_failed"
    state["outcome_fix_attempt"] = 2
    assert wf.route_after_outcome_check(state) == "remediation_plan"


def test_outcome_correct_still_routes_to_memory_update():
    assert wf.route_after_outcome_check({"outcome_check_result": {"status": "CORRECT"}}) == "memory_update"


# ---------------------------------------------------------------------------
# build skipped
# ---------------------------------------------------------------------------
def test_build_skipped_routes_to_test_but_is_not_success():
    assert BuildStatus.SKIPPED.value == "skipped"
    state = {"build_result": BuildResult(status=BuildStatus.SKIPPED, exit_code=0)}
    assert wf.check_build_status(state) == "test"


# ---------------------------------------------------------------------------
# differential test acceptance
# ---------------------------------------------------------------------------
class _Exec:
    def __init__(self, baseline_result):
        self.baseline_result = baseline_result
        self.calls = 0

    def execute_tests(self, project_file, test_target=None):
        self.calls += 1
        return self.baseline_result


def _failing(lines):
    return TestResult(status=TestStatus.SOME_FAILED, total_tests=3, passed=2, failed=1, errors=lines, test_output="\n".join(lines))


def test_differential_acceptance_requires_baseline_match(tmp_path):
    (tmp_path / "A.java").write_text("new", encoding="utf-8")
    state = {"workspace_path": str(tmp_path), "original_file_contents": {"A.java": "old"}, "generated_code": []}
    gen = [GeneratedCode(file_path="A.java", content="new", language="java", change_type="modify")]
    current = _failing(["FAILED LegacyTest.testOld: assertion error"])

    # same failure on baseline -> accepted, baseline-backed
    affects, compared = wf._differential_test_acceptance(state, _Exec(current), "pom.xml", None, current, gen)
    assert (affects, compared) == (False, True)
    # workspace restored after baseline run
    assert (tmp_path / "A.java").read_text(encoding="utf-8") == "new"

    # baseline green -> failures are new
    green = TestResult(status=TestStatus.ALL_PASSED, total_tests=3, passed=3)
    affects, compared = wf._differential_test_acceptance(state, _Exec(green), "pom.xml", None, current, gen)
    assert (affects, compared) == (True, True)

    # new signature not on baseline -> attributable
    base = _failing(["FAILED OtherTest.testX: boom"])
    affects, compared = wf._differential_test_acceptance(state, _Exec(base), "pom.xml", None, current, gen)
    assert (affects, compared) == (True, True)


def test_differential_acceptance_without_baseline_is_heuristic_only(tmp_path):
    state = {"workspace_path": str(tmp_path), "original_file_contents": {}, "generated_code": []}
    gen = [GeneratedCode(file_path="A.java", content="new", language="java", change_type="modify")]
    current = _failing(["FAILED LegacyTest.testOld: assertion error"])
    affects, compared = wf._differential_test_acceptance(state, _Exec(current), "pom.xml", None, current, gen)
    assert compared is False
    assert affects is False  # heuristic says unrelated, but caller records UNVERIFIED


# ---------------------------------------------------------------------------
# memory_update honesty
# ---------------------------------------------------------------------------
def _agents():
    a = MagicMock()
    a.localizer.repository_brain.record_successful_ticket = MagicMock()
    a.localizer.workspace_path = "."
    return a


def _mem_state(tmp_path, build, test, **extra):
    st = {
        "ticket": _ticket(),
        "workspace_path": str(tmp_path),
        "build_result": build,
        "test_result": test,
        "generated_code": [GeneratedCode(file_path="A.java", content="x", language="java", change_type="modify")],
        "unresolved_checklist_items": [],
    }
    st.update(extra)
    return st


@pytest.fixture(autouse=True)
def _quiet_side_effects():
    with patch.object(wf, "write_trace_artifact", lambda *a, **k: None), \
         patch.object(wf, "log_phase", lambda *a, **k: None), \
         patch.object(wf, "_perform_workspace_rollback", lambda *a, **k: {"rollback_performed": True, "restored_files": [], "removed_files": [], "patch_file": None}):
        yield


def test_memory_update_does_not_learn_from_failed_tests(tmp_path):
    agents = _agents()
    st = _mem_state(tmp_path, BuildResult(status=BuildStatus.SUCCESS, exit_code=0),
                    TestResult(status=TestStatus.SOME_FAILED, total_tests=1, failed=1))
    res = wf.memory_update_node(st, agents)
    assert res["status"] == "verification_failed"
    agents.localizer.repository_brain.record_successful_ticket.assert_not_called()


def test_memory_update_does_not_learn_from_skipped_build(tmp_path):
    agents = _agents()
    st = _mem_state(tmp_path, BuildResult(status=BuildStatus.SKIPPED, exit_code=0),
                    TestResult(status=TestStatus.NO_TESTS))
    res = wf.memory_update_node(st, agents)
    assert res["status"] == "unverified"
    assert res["verification_level"] == "unverified"
    agents.localizer.repository_brain.record_successful_ticket.assert_not_called()


def test_memory_update_heuristic_acceptance_is_unverified(tmp_path):
    agents = _agents()
    st = _mem_state(tmp_path, BuildResult(status=BuildStatus.SUCCESS, exit_code=0),
                    TestResult(status=TestStatus.SOME_FAILED, total_tests=1, failed=1),
                    test_differential_accept=True, test_baseline_compared=False)
    res = wf.memory_update_node(st, agents)
    assert res["status"] == "unverified"
    agents.localizer.repository_brain.record_successful_ticket.assert_not_called()


def test_memory_update_learns_only_from_verified_runs(tmp_path):
    agents = _agents()
    st = _mem_state(tmp_path, BuildResult(status=BuildStatus.SUCCESS, exit_code=0),
                    TestResult(status=TestStatus.ALL_PASSED, total_tests=2, passed=2))
    res = wf.memory_update_node(st, agents)
    assert res["status"] == "memory_updated"
    assert res["verification_level"] == "test_verified"
    agents.localizer.repository_brain.record_successful_ticket.assert_called()


# ---------------------------------------------------------------------------
# discovery hard stop + preflight
# ---------------------------------------------------------------------------
def test_route_after_discover_honors_hard_stop():
    assert wf.route_after_discover({"status": "failed", "discovered_files": [], "error": "hard stop"}) == "verification_failed"


def test_preflight_fallback_is_not_evaluated():
    import inspect
    src = inspect.getsource(wf.preflight_check_node)
    assert '"confidence": 0.95' not in src and '"confidence": 0.90' not in src
    assert src.count('"verdict": "NOT_EVALUATED"') >= 2


# ---------------------------------------------------------------------------
# fast path finalization
# ---------------------------------------------------------------------------
def test_finalize_fast_path_fails_when_outcome_not_correct():
    st = {"ticket": _ticket(), "workspace_path": ".", "generated_code": [],
          "build_result": BuildResult(status=BuildStatus.SUCCESS, exit_code=0),
          "test_result": TestResult(status=TestStatus.ALL_PASSED)}
    with patch.object(wf, "outcome_check_node", return_value={"outcome_check_result": {"status": "FAILED", "reason": "missing REQ-2"}}):
        out = wf.finalize_fast_path_state(st, MagicMock())
    assert out["status"] == "verification_failed"
    assert "missing REQ-2" in out["error"]


def test_finalize_fast_path_completes_only_when_everything_agrees():
    st = {"ticket": _ticket(), "workspace_path": ".", "generated_code": [],
          "build_result": BuildResult(status=BuildStatus.SUCCESS, exit_code=0),
          "test_result": TestResult(status=TestStatus.ALL_PASSED)}
    with patch.object(wf, "outcome_check_node", return_value={"outcome_check_result": {"status": "CORRECT"}}):
        out = wf.finalize_fast_path_state(st, MagicMock())
    assert out["status"] == "completed"
    assert out["verification_level"] == "test_verified"

    st["build_result"] = BuildResult(status=BuildStatus.FAILURE, exit_code=1)
    with patch.object(wf, "outcome_check_node", return_value={"outcome_check_result": {"status": "CORRECT"}}):
        out = wf.finalize_fast_path_state(st, MagicMock())
    assert out["status"] == "verification_failed"


def test_fast_results_reflect_unknowns():
    r = SimpleNamespace(build_passed=None, test_passed=None)
    assert wf._fast_build_result(r).status == BuildStatus.SKIPPED
    assert wf._fast_test_result(r).status == TestStatus.NO_TESTS
    r = SimpleNamespace(build_passed=True, test_passed=False)
    assert wf._fast_build_result(r).status == BuildStatus.SUCCESS
    assert wf._fast_test_result(r).status == TestStatus.SOME_FAILED


def test_run_autonomous_workflow_accepts_api_kwargs():
    import inspect
    sig = inspect.signature(wf.run_autonomous_workflow)
    for name in ("execution_mode", "contract_ticket", "contract_policy", "progress_callback"):
        assert name in sig.parameters
    assert any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
