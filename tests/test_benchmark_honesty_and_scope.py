"""Harness hardening: benchmark honesty, scope matching, stage routing hook, evidence degradation."""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import ticket_to_code.workflow as wf
from ticket_to_code.benchmarks import production_readiness_benchmark as pb
from ticket_to_code.benchmarks.swe_bench_runner import SWEBenchObjectiveRunner, BenchmarkTask


# ---------------------------------------------------------------------------
# swe_bench_runner never fabricates a pass
# ---------------------------------------------------------------------------
def test_swe_bench_ineligible_task_is_not_evaluated_not_passed(tmp_path):
    runner = SWEBenchObjectiveRunner(str(tmp_path), results_dir=str(tmp_path / "res"))
    task = BenchmarkTask(instance_id="x-1", repo="r", problem_statement="p", base_commit="c")
    fake_mod = SimpleNamespace(
        FastPathEligibilityAssessor=SimpleNamespace(is_eligible=staticmethod(lambda *_a, **_k: (False, "nope"))),
        FastPathReActEngine=None,
    )
    with patch.dict("sys.modules", {"ticket_to_code.runtime.fast_path_engine": fake_mod}):
        res = runner.run_single_task(task)
    assert res.passed is False
    assert "not_evaluated" in (res.error_message or "")
    assert res.tokens_in == 0 and res.cost_usd == 0.0


def test_swe_bench_source_has_no_hardcoded_pass():
    src = inspect.getsource(SWEBenchObjectiveRunner.run_single_task)
    assert "passed=True" not in src
    assert "0.000003" not in src


# ---------------------------------------------------------------------------
# production benchmark scores real outcomes
# ---------------------------------------------------------------------------
def test_production_benchmark_success_requires_verification_level():
    assert pb._is_verified_success("completed", {"verification_level": "test_verified"}) is True
    assert pb._is_verified_success("memory_updated", {"verification_level": "compile_only"}) is True
    assert pb._is_verified_success("completed", {"verification_level": "unverified"}) is False
    assert pb._is_verified_success("completed", {}) is False
    assert pb._is_verified_success("verification_failed", {"verification_level": "test_verified"}) is False
    assert pb._terminal_bucket("completed", 2, {"verification_level": "test_verified"}) == pb.STATUS_SOLVED_WITH_PATCH
    assert pb._terminal_bucket("completed", 2, {"verification_level": "unverified"}) == pb.STATUS_OTHER


def test_production_benchmark_restores_workspace(tmp_path):
    (tmp_path / "A.java").write_text("edited", encoding="utf-8")
    (tmp_path / "New.java").write_text("new", encoding="utf-8")
    state = {
        "original_file_contents": {"A.java": "original"},
        "generated_code": [SimpleNamespace(file_path="New.java"), SimpleNamespace(file_path="A.java")],
    }
    pb._restore_workspace_after_ticket(state, str(tmp_path))
    assert (tmp_path / "A.java").read_text(encoding="utf-8") == "original"
    assert not (tmp_path / "New.java").exists()


def test_production_benchmark_reads_run_record_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("TICKET_TO_CODE_TRACE_DIR", str(tmp_path))
    (tmp_path / "T-9").mkdir()
    (tmp_path / "T-9" / "run_record.json").write_text('{"tokens_in": 120, "tokens_out": 30}', encoding="utf-8")
    assert pb._collect_token_metrics(str(tmp_path), "T-9") == (120, 30, 150)


# ---------------------------------------------------------------------------
# scope matching: no bare-basename authorization
# ---------------------------------------------------------------------------
def test_scope_path_match_rejects_bare_basename_suffix():
    assert wf._scope_path_match("order-service/src/OrderService.java", "order-service/src/OrderService.java")
    assert wf._scope_path_match("repo/order-service/src/OrderService.java", "order-service/src/OrderService.java")
    assert wf._scope_path_match("src\\OrderService.java", "src/orderservice.java")
    assert not wf._scope_path_match("other-service/src/OrderService.java", "OrderService.java")
    assert not wf._scope_path_match("OrderService.java", "other-service/src/OrderService.java")
    assert not wf._scope_path_match("a/b.java", "")


# ---------------------------------------------------------------------------
# stage model resolver hook
# ---------------------------------------------------------------------------
def test_stage_model_resolver_swaps_llm_for_node_only():
    agents = SimpleNamespace(llm="default-llm")
    state = {"ticket": SimpleNamespace(ticket_id="T-stage")}
    wf._set_transient("T-stage", "agents", agents)
    seen = {}

    def node(st):
        seen["llm_during"] = agents.llm
        return {}

    try:
        wf.set_stage_model_resolver(lambda name: "fast-model" if name == "fix_build" else None)
        with patch.object(wf, "_stage_llm_for", lambda mid: f"llm:{mid}"), \
             patch.object(wf, "_record_step", lambda *a, **k: None):
            wf._phase_tracked_node("fix_build", node)(state)
            assert seen["llm_during"] == "llm:fast-model"
            assert agents.llm == "default-llm"
            wf._phase_tracked_node("plan", node)(state)
            assert seen["llm_during"] == "default-llm"
    finally:
        wf.set_stage_model_resolver(None)
        wf._clear_transient("T-stage")


# ---------------------------------------------------------------------------
# evidence degradation flows into the outcome verdict
# ---------------------------------------------------------------------------
def test_evidence_node_exposes_degraded_flag_key():
    assert '"evidence_decision_degraded"' in inspect.getsource(wf.evidence_collection_loop_node)


def test_outcome_check_downgrades_correct_when_evidence_degraded():
    src = inspect.getsource(wf.outcome_check_node)
    assert 'state.get("evidence_decision_degraded")' in src


# ---------------------------------------------------------------------------
# write_trace_artifact uses the shared trace dir
# ---------------------------------------------------------------------------
def test_write_trace_artifact_uses_trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TICKET_TO_CODE_TRACE_DIR", str(tmp_path))
    wf.write_trace_artifact(str(tmp_path), "T-42", "x.json", {"a": 1})
    assert (tmp_path / "T-42" / "x.json").is_file()


# ---------------------------------------------------------------------------
# containment in the repair writer
# ---------------------------------------------------------------------------
def test_safe_write_refuses_path_outside_workspace(tmp_path):
    import pytest
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(PermissionError):
        wf._safe_write_or_patch_file(ws, "../escape.txt", "x", {})
    assert not (tmp_path / "escape.txt").exists()
