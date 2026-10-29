"""Step tracing: every workflow node and routing decision lands in steps.jsonl."""

import json
from types import SimpleNamespace

from ticket_to_code import workflow as wf
from ticket_to_code.tracing.step_report import format_steps, load_steps


def _state():
    return {"ticket": SimpleNamespace(ticket_id="T-TRACE"), "retry_attempt": 1, "status": "running"}


def test_node_and_router_steps_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("TICKET_TO_CODE_TRACE_DIR", str(tmp_path))
    wf.reset_step_trace("T-TRACE")

    node = wf._phase_tracked_node("demo", lambda s: {"status": "done", "files": ["a", "b"], "obj": SimpleNamespace(status="ok")})
    assert node(_state())["status"] == "done"

    def boom(s):
        raise ValueError("nope")

    try:
        wf._step_traced("explode", boom)(_state())
    except ValueError:
        pass

    steps = load_steps("T-TRACE", str(tmp_path))
    assert [s["node"] for s in steps] == ["demo", "explode"]
    assert steps[0]["ok"] and steps[0]["produced"]["files"] == "[2 items]"
    assert steps[0]["context"]["retry_attempt"] == 1
    assert not steps[1]["ok"] and "ValueError: nope" in steps[1]["error"]
    assert "ERROR ValueError" in format_steps(steps)


def test_graph_routers_are_traced_and_graph_still_compiles(tmp_path, monkeypatch):
    monkeypatch.setenv("TICKET_TO_CODE_TRACE_DIR", str(tmp_path))
    graph = wf.create_ticket_to_code_graph(str(tmp_path), None)
    assert "ui_verify" in set(graph.compile().get_graph().nodes)

    wf.reset_step_trace("T-TRACE")
    wf._record_step(_state(), "route:check_test_status", 0.0, {"from": "test", "decision": "ui_verify"}, None)
    row = json.loads((tmp_path / "T-TRACE" / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["produced"]["decision"] == "ui_verify"


def test_hung_node_fails_fast_with_timeout_and_is_traced(tmp_path, monkeypatch):
    import time
    import pytest

    monkeypatch.setenv("TICKET_TO_CODE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("TICKET_TO_CODE_NODE_TIMEOUT_S", "0.3")
    wf.reset_step_trace("T-TRACE")

    def hang(s):
        time.sleep(5)
        return {"status": "never"}

    started = time.time()
    with pytest.raises(wf.NodeTimeoutError):
        wf._phase_tracked_node("edit_loop", hang)(_state())
    assert time.time() - started < 2

    steps = load_steps("T-TRACE", str(tmp_path))
    assert not steps[0]["ok"] and "time budget" in steps[0]["error"]


def test_node_exceptions_and_results_pass_through_the_watchdog(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("TICKET_TO_CODE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("TICKET_TO_CODE_NODE_TIMEOUT_S", "5")
    assert wf._step_traced("ok", lambda s: {"status": "fine"})(_state()) == {"status": "fine"}
    with pytest.raises(ValueError):
        wf._step_traced("bad", lambda s: (_ for _ in ()).throw(ValueError("x")))(_state())


def test_reset_keeps_previous_run(tmp_path, monkeypatch):
    monkeypatch.setenv("TICKET_TO_CODE_TRACE_DIR", str(tmp_path))
    wf.reset_step_trace("T-TRACE")
    wf._record_step(_state(), "a", 0.0, {}, None)
    wf.reset_step_trace("T-TRACE")
    assert load_steps("T-TRACE", str(tmp_path)) == []
    assert (tmp_path / "T-TRACE" / "steps.jsonl.prev").is_file()
