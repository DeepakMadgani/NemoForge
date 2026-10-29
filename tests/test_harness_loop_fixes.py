"""Harness loop fixes: items 1 (ReAct engine), 2 (error-resolution guard/truncation),
3 (edit-loop oscillation stop), 4 (edit-loop output budgets), 5 (evidence degradation).
"""

import itertools
import json
import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from ticket_to_code.models import ValueEdgeTicket
from ticket_to_code.runtime.limits import (
    BUILD_OUTPUT_MAX_CHARS,
    EDIT_MAX_TOKENS,
    LATEST_READ_MAX_CHARS,
    OBSERVATION_MAX_CHARS,
    SNIPPET_MAX_CHARS,
    WHOLE_FILE_MAX_TOKENS,
)


class FakeLLM:
    """Mock-free LLM double accepting **kwargs (records them) and returning AIMessage."""

    def __init__(self, responses=None, exc=None, finish_reason=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.finish_reason = finish_reason
        self.call_count = 0
        self.kwargs = []
        self.call_args_list = []

    def invoke(self, messages, **kwargs):
        self.call_count += 1
        self.kwargs.append(dict(kwargs))
        self.call_args_list.append(messages)
        if self.exc is not None:
            raise self.exc
        content = self.responses.pop(0) if self.responses else ""
        meta = {"finish_reason": self.finish_reason} if self.finish_reason else {}
        return AIMessage(content=content, response_metadata=meta)


# ═══════════════════════════════════════════════════════════════════════════
# Item 1 — AutonomousReActEngine
# ═══════════════════════════════════════════════════════════════════════════

def _engine(tmp_path, llm_client=lambda p: ""):
    from ticket_to_code.runtime.react_agent_engine import AutonomousReActEngine
    ticket = ValueEdgeTicket(ticket_id="T-REACT", title="t", description="d")
    return AutonomousReActEngine(workspace_path=str(tmp_path), ticket=ticket, llm_client=llm_client, max_steps=4)


def test_react_1a_latest_read_kept_verbatim_older_observations_compacted(tmp_path):
    big = OBSERVATION_MAX_CHARS + 1000
    if big + 50 >= LATEST_READ_MAX_CHARS:
        pytest.skip("limits overridden so the latest-read budget cannot be distinguished")
    engine = _engine(tmp_path)
    old_content = "A" * big
    new_content = "B" * big
    engine.step_history = [
        {"role": "tool_call", "name": "view_file", "args": {"file_path": "old.ts"}, "result": {"content": old_content}},
        {"role": "tool_call", "name": "grep_search", "args": {"pattern": "x"}, "result": {"matches": []}},
        {"role": "tool_call", "name": "view_file", "args": {"file_path": "new.ts"}, "result": {"content": new_content}},
    ]
    prompt = engine._build_step_prompt()
    assert new_content in prompt, "most recent read must be kept verbatim"
    assert old_content not in prompt, "older read must be compacted"
    assert "[truncated" in prompt


def test_react_1b_write_epoch_allows_reread_after_edit_but_not_duplicate_writes(tmp_path, monkeypatch):
    from ticket_to_code.runtime.react_agent_engine import ToolResult
    engine = _engine(tmp_path)
    f = tmp_path / "a.ts"
    f.write_text("const a = 1;\n", encoding="utf-8")

    view_args = {"file_path": "a.ts", "start_line": 1, "end_line": 10}
    assert engine._execute_tool_call("view_file", view_args).ok is True
    dup = engine._execute_tool_call("view_file", view_args)
    assert dup.ok is False and dup.error.code == "DUPLICATE_ACTION"

    edit_args = {"file_path": "a.ts", "target_content": "const a = 1;", "replacement_content": "const a = 2;"}
    res = engine._execute_tool_call("replace_file_content", edit_args)
    assert res.ok is True, res.error
    assert engine.write_epoch == 1

    # Same read after a write is legitimate again
    again = engine._execute_tool_call("view_file", view_args)
    assert again.ok is True
    assert "const a = 2;" in again.data["content"]

    # Writes stay strict: identical replace is still a duplicate
    dup_edit = engine._execute_tool_call("replace_file_content", edit_args)
    assert dup_edit.ok is False and dup_edit.error.code == "DUPLICATE_ACTION"

    # run_command bumps the epoch too (even when it does nothing)
    monkeypatch.setattr(engine.toolbox, "run_command", lambda **kw: ToolResult(ok=True, data={"exit_code": 0}))
    engine._execute_tool_call("run_command", {"command": "echo hi"})
    assert engine.write_epoch == 2
    assert engine._execute_tool_call("view_file", view_args).ok is True


def test_react_1c_registry_fallback_discards_mock_and_accepts_real_llm(tmp_path, monkeypatch):
    from ticket_to_code.runtime.llm_factory import LLMRegistry
    # conftest patches LLMFactory.get_llm to return a unittest.mock.Mock -> must be discarded
    engine = _engine(tmp_path, llm_client=None)
    assert engine.llm is None
    assert engine.run().status == "no_llm_client"

    fake = FakeLLM()
    monkeypatch.setattr(LLMRegistry, "get_llm", lambda *a, **k: fake)
    engine2 = _engine(tmp_path, llm_client=None)
    assert engine2.llm is fake


def test_react_24_invoke_branch_goes_through_llm_invoke(tmp_path, monkeypatch):
    from ticket_to_code.runtime import react_agent_engine as rae
    seen = {}

    def _fake_llm_invoke(llm, prompt, **kw):
        seen["llm"] = llm
        seen["prompt"] = prompt
        return AIMessage(content='{"tool": "COMPLETE_TASK", "args": {}}')

    monkeypatch.setattr(rae, "llm_invoke", _fake_llm_invoke)
    fake = FakeLLM()
    engine = _engine(tmp_path, llm_client=fake)
    result = engine.run()
    assert seen["llm"] is fake and isinstance(seen["prompt"], str)
    assert fake.call_count == 0  # routed through llm_invoke, not llm.invoke directly
    assert result.status == "no_modifications"


# ═══════════════════════════════════════════════════════════════════════════
# Item 2 — ErrorResolutionAgent consumer guard + truncation
# ═══════════════════════════════════════════════════════════════════════════

def test_consumer_guard_helper():
    from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent as A
    assert A._consumer_guard_blocks(None, "anything.ts") is False
    assert A._consumer_guard_blocks("", "anything.ts") is False
    assert A._consumer_guard_blocks("src/app/consumer.ts", "src/app/consumer.ts") is False
    assert A._consumer_guard_blocks("src/app/consumer.ts", "SRC\\APP\\Consumer.ts") is False
    assert A._consumer_guard_blocks("consumer.ts", "ui/src/app/consumer.ts") is False
    assert A._consumer_guard_blocks("ui/src/app/consumer.ts", "consumer.ts") is False
    assert A._consumer_guard_blocks("src/app/consumer.ts", "src/app/member.service.ts") is True


def _neutralize_agent(agent, monkeypatch, compile_result="✅ BUILD SUCCESS — 0 errors"):
    monkeypatch.setattr(agent, "_build_diagnostic_evidence", lambda errors: "")
    monkeypatch.setattr(agent, "_validate_patch_target", lambda fp, target: "")
    monkeypatch.setattr(agent, "_verify_post_patch", lambda fp: "")
    monkeypatch.setattr(agent, "_tool_compile", lambda: compile_result)
    monkeypatch.setattr(agent, "_parse_error_locations", lambda text: [])


def _force_verified_consumer(monkeypatch, consumer_file):
    """Make the pre-flight attribute every diagnostic to a protected provider with a
    VERIFIED_ALTERNATIVE so resolve_errors() sets target_consumer_file."""
    import ticket_to_code.agents.diagnostic_localizer as dl
    import ticket_to_code.agents.diagnostic_normalizer as dn
    from ticket_to_code.agents.diagnostic_localizer import FailureOwner
    from ticket_to_code.agents.provider_capability_resolver import CapabilityResolutionStatus

    alt = SimpleNamespace(
        method_name="members", signature="members(): Observable<any[]>", parameters=[],
        return_type="Observable<any[]>", repository_evidence=[], usage_example="", compatibility_proof="",
    )

    class _FakeLocalizer:
        def __init__(self, *a, **k):
            pass

        def attribute_failure(self, diag, **kw):
            return SimpleNamespace(
                is_protected_provider=True,
                owner=FailureOwner.CROSS_FILE_CONTRACT,
                consumer_file=consumer_file,
                provider_file="member.service.ts",
                provider_type="service",
                diagnostic=diag,
                capability_resolution=SimpleNamespace(
                    status=CapabilityResolutionStatus.VERIFIED_ALTERNATIVE,
                    provider_name="MemberService",
                    provider_file="member.service.ts",
                    missing_symbol="getProjectMembers",
                    verified_alternatives=[alt],
                ),
            )

    monkeypatch.setattr(dl, "DiagnosticLocalizer", _FakeLocalizer)
    monkeypatch.setattr(dn, "normalize_diagnostics", lambda errs: [SimpleNamespace(raw=e) for e in errs])


_ERR = "consumer.ts(5,3): error TS2339: Property 'getProjectMembers' does not exist on type 'MemberService'."


def _human_feedback(fake, call_index):
    """Content of the HumanMessage the agent sent on call N.

    The agent appends its own AIMessage to the same list object after the call,
    so the last element is the reply — the compacted tool feedback is element 1.
    """
    from langchain_core.messages import HumanMessage
    msgs = fake.call_args_list[call_index]
    humans = [m for m in msgs if isinstance(m, HumanMessage)]
    return humans[-1].content


def test_error_resolution_authorized_consumer_edit_reaches_handler(tmp_path, monkeypatch):
    from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent
    (tmp_path / "consumer.ts").write_text("export class C { a() { return 1; } }\n", encoding="utf-8")
    _force_verified_consumer(monkeypatch, "consumer.ts")

    fake = FakeLLM(responses=[
        json.dumps({"action": "REPLACE_CONTENT", "args": {"file_path": "consumer.ts", "old_content": "return 1;", "new_content": "return 2;"}}),
        json.dumps({"action": "COMPILE", "args": {}}),
    ])
    agent = ErrorResolutionAgent(fake, tmp_path)
    _neutralize_agent(agent, monkeypatch)
    calls = []
    monkeypatch.setattr(agent, "_tool_replace_content", lambda fp, old, new: calls.append((fp, old, new)) or "Replaced 1 occurrence in consumer.ts")

    agent.resolve_errors([_ERR])

    assert calls == [("consumer.ts", "return 1;", "return 2;")], "authorized consumer edit must reach REPLACE_CONTENT"
    assert fake.call_count == 2
    feedback = _human_feedback(fake, 1)
    assert "BLOCKED" not in feedback
    assert "Result for REPLACE_CONTENT('consumer.ts')" in feedback


def test_error_resolution_non_consumer_edit_is_blocked_and_not_applied(tmp_path, monkeypatch):
    from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent
    (tmp_path / "consumer.ts").write_text("export class C {}\n", encoding="utf-8")
    (tmp_path / "member.service.ts").write_text("export class MemberService {}\n", encoding="utf-8")
    _force_verified_consumer(monkeypatch, "consumer.ts")

    fake = FakeLLM(responses=[
        json.dumps({"action": "REPLACE_CONTENT", "args": {"file_path": "member.service.ts", "old_content": "{}", "new_content": "{ x }"}}),
        json.dumps({"action": "COMPILE", "args": {}}),
    ])
    agent = ErrorResolutionAgent(fake, tmp_path)
    _neutralize_agent(agent, monkeypatch)
    calls = []
    monkeypatch.setattr(agent, "_tool_replace_content", lambda fp, old, new: calls.append(fp) or "ok")

    result = agent.resolve_errors([_ERR])

    assert calls == []
    assert result == {}
    assert (tmp_path / "member.service.ts").read_text(encoding="utf-8") == "export class MemberService {}\n"
    feedback = _human_feedback(fake, 1)
    assert "BLOCKED" in feedback and "member.service.ts" in feedback


def test_error_resolution_compile_output_truncated_to_build_budget(tmp_path, monkeypatch):
    import ticket_to_code.agents.diagnostic_normalizer as dn
    from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent
    monkeypatch.setattr(dn, "normalize_diagnostics", lambda errs: [])
    (tmp_path / "consumer.ts").write_text("export class C {}\n", encoding="utf-8")

    noise = "x" * (BUILD_OUTPUT_MAX_CHARS + 500)
    fake = FakeLLM(responses=[
        json.dumps({"action": "COMPILE", "args": {}}),
        json.dumps({"action": "FINISHED", "args": {}}),
    ])
    agent = ErrorResolutionAgent(fake, tmp_path)
    _neutralize_agent(agent, monkeypatch, compile_result="✅ BUILD SUCCESS — 0 errors\n" + noise)

    agent.resolve_errors([_ERR])

    assert fake.call_count == 2
    feedback = _human_feedback(fake, 1)
    assert "Result for COMPILE" in feedback
    assert "[truncated" in feedback
    assert noise not in feedback
    # the retained compile output is bounded by the build budget (+ marker)
    kept = feedback.split("Result for COMPILE:", 1)[1].split("⚡ DIRECTIVE", 1)[0]
    assert len(kept) < BUILD_OUTPUT_MAX_CHARS + 200


# ═══════════════════════════════════════════════════════════════════════════
# Items 3 & 4 — EditLoopAgent
# ═══════════════════════════════════════════════════════════════════════════

def _edit_agent(tmp_path, fake_llm):
    from unittest.mock import Mock
    from ticket_to_code.agents.edit_loop_agent import EditLoopAgent
    symbol_index = Mock()
    symbol_index.get_ts_diagnostics.return_value = []
    symbol_index.get_members_for_file.return_value = None
    return EditLoopAgent(workspace_path=str(tmp_path), llm=fake_llm, symbol_index=symbol_index)


def test_edit_loop_3_oscillation_stops_loop_with_stop_reason(tmp_path, monkeypatch):
    from ticket_to_code.agents.edit_loop_agent import EditDecision, MAX_ITER
    agent = _edit_agent(tmp_path, FakeLLM())
    picks = itertools.cycle(["a.ts", "b.ts"])
    monkeypatch.setattr(agent, "_decide_next_edit", lambda *a, **k: EditDecision(file_path=next(picks), reason="r", edit_description="d"))
    monkeypatch.setattr(agent, "_generate_edit", lambda **k: None)
    monkeypatch.setattr(agent, "_read_file", lambda fp: "content")
    monkeypatch.setattr(agent, "_get_lsp_context", lambda *a, **k: "")

    ticket = SimpleNamespace(ticket_id="T-OSC", title="t", description="d")
    result = agent.run(ticket, requirements=None)

    assert result["stop_reason"] == "oscillation"
    assert result["iterations"] < MAX_ITER
    assert any(r.reason == "per-file edit ceiling reached" for r in result["edit_results"])
    assert result["solved"] is False


def test_edit_loop_4_output_budget_whole_file_vs_edit_and_truncation_warning(tmp_path, caplog):
    from ticket_to_code.agents.edit_loop_agent import EditDecision
    caplog.set_level(logging.WARNING, logger="ticket_to_code.agents.edit_loop_agent")
    ticket = SimpleNamespace(title="t", description="d", ticket_id="T-4")

    fake_create = FakeLLM(responses=["```ts\nexport const a = 1;\n```"], finish_reason="length")
    agent = _edit_agent(tmp_path, fake_create)
    agent._generate_edit(
        decision=EditDecision(file_path="new.ts", reason="r", edit_description="d", is_new_file=True),
        current_content="", ticket=ticket, requirements=None, code_rag_context="", lsp_context="",
    )
    assert fake_create.kwargs[0].get("max_tokens") == WHOLE_FILE_MAX_TOKENS
    assert any("finish_reason=length" in r.getMessage() for r in caplog.records)

    fake_edit = FakeLLM(responses=["<<<<<<< SEARCH\nconst a = 1;\n=======\nconst a = 2;\n>>>>>>> REPLACE"])
    agent2 = _edit_agent(tmp_path, fake_edit)
    (tmp_path / "ex.ts").write_text("const a = 1;\n", encoding="utf-8")
    out = agent2._generate_edit(
        decision=EditDecision(file_path="ex.ts", reason="r", edit_description="d"),
        current_content="const a = 1;\n", ticket=ticket, requirements=None, code_rag_context="", lsp_context="",
    )
    assert fake_edit.kwargs[0].get("max_tokens") == EDIT_MAX_TOKENS
    assert out is not None and "const a = 2;" in out


def test_edit_loop_13_ts_diagnostics_none_is_unverified_not_clean(tmp_path):
    agent = _edit_agent(tmp_path, FakeLLM())
    agent.symbol_index.get_ts_diagnostics.return_value = None
    (tmp_path / "x.ts").write_text(
        "import { Injectable } from '@angular/core';\n\n"
        "@Injectable({ providedIn: 'root' })\n"
        "export class MemberService {\n"
        "  private readonly members: string[] = [];\n\n"
        "  getMembers(): string[] {\n"
        "    return this.members;\n"
        "  }\n\n"
        "  addMember(name: string): void {\n"
        "    this.members.push(name);\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    assert agent._verify_file("x.ts") == []
    assert agent._last_verify_unknown is True
    agent.symbol_index.get_ts_diagnostics.return_value = ["x.ts(1,1): error TS1"]
    assert agent._verify_file("x.ts") == ["x.ts(1,1): error TS1"]
    assert agent._last_verify_unknown is False


# ═══════════════════════════════════════════════════════════════════════════
# Item 5 — EvidenceCollectionLoop degraded decisions
# ═══════════════════════════════════════════════════════════════════════════

def _evidence_loop(tmp_path, monkeypatch):
    from ticket_to_code.agents.evidence_collection_loop import EvidenceCollectionLoop
    monkeypatch.setenv("TICKET_TO_CODE_TRACE_DIR", str(tmp_path / "traces"))
    return EvidenceCollectionLoop(localizer=SimpleNamespace(workspace_path=str(tmp_path)), rag_engine=None)


def _decide(loop):
    ticket = SimpleNamespace(ticket_id="T-EV", title="Fix modal", description="Ghost members in modal")
    return loop._decide_next_action(
        ticket=ticket, hypotheses=None, evidence={}, tried=[], all_items=[], iteration=0, max_iterations=3,
    )


def test_evidence_5_unparseable_llm_output_is_degraded_proceed(tmp_path, monkeypatch):
    loop = _evidence_loop(tmp_path, monkeypatch)
    loop._agentic_llm = FakeLLM(responses=["this is not json at all"])
    action = _decide(loop)
    assert action["type"] == "proceed"
    assert action["degraded"] is True
    assert action["reason"] == "unparseable_action"


def test_evidence_5_llm_error_is_degraded_proceed(tmp_path, monkeypatch):
    loop = _evidence_loop(tmp_path, monkeypatch)
    loop._agentic_llm = FakeLLM(exc=RuntimeError("provider down"))
    action = _decide(loop)
    assert action["type"] == "proceed" and action["degraded"] is True
    assert action["reason"].startswith("llm_error")
    assert "provider down" in action["reason"]


def test_evidence_5_no_llm_is_degraded_proceed(tmp_path, monkeypatch):
    loop = _evidence_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "_get_agentic_llm", lambda: None)
    action = _decide(loop)
    assert action == {
        "type": "proceed", "degraded": True, "reason": "llm_unavailable",
        "reasoning": "LLM unavailable, proceeding with current evidence",
    }


def test_evidence_5_valid_action_is_not_degraded_and_uses_llm_invoke(tmp_path, monkeypatch):
    loop = _evidence_loop(tmp_path, monkeypatch)
    loop._agentic_llm = FakeLLM(responses=['{"type": "proceed", "reasoning": "enough"}'])
    action = _decide(loop)
    assert action["type"] == "proceed" and not action.get("degraded")
    loop._note_decision(action)
    assert loop.decision_degraded is False


def test_evidence_5_note_decision_sets_instance_flags(tmp_path, monkeypatch):
    loop = _evidence_loop(tmp_path, monkeypatch)
    assert loop.decision_degraded is False
    loop._note_decision({"type": "proceed", "reasoning": "fine"})
    assert loop.decision_degraded is False
    loop._note_decision({"type": "proceed", "degraded": True, "reason": "llm_error: boom"})
    assert loop.decision_degraded is True
    assert loop.decision_degraded_reasons == ["llm_error: boom"]


def test_evidence_5_snippet_cap_uses_shared_limit():
    import inspect
    from ticket_to_code.agents import evidence_collection_loop as ecl
    src = inspect.getsource(ecl)
    assert "from ticket_to_code.runtime.limits import SNIPPET_MAX_CHARS" in src
    assert 'line {r.line_number}: {r.matched_text}"[:SNIPPET_MAX_CHARS]' in src
    assert SNIPPET_MAX_CHARS >= 400
