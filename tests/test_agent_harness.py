"""Regression tests for the Chat Agent Harness (Option A: Chat front door → Flow engine).

Covers:
  1. Intent classification — plain defect titles are solvable tasks, questions stay read-only.
  2. The Chat → Flow delegation hand-off (`delegate_ticket_to_flow`) routes to `run_lean_path`.
  3. Polyglot build verification — `_detect_build_command` resolves npm / gradle / maven.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ticket_to_code.runtime.agent_harness import (
    AgentIntent,
    classify_agent_intent,
    classify_agent_intent_hybrid,
    _classify_deterministic,
    HostEnvironmentTools,
    delegate_ticket_to_flow,
)


# ─── 1. Intent classification ────────────────────────────────────────────────

@pytest.mark.parametrize("message", [
    "Removing a coordinator does not revoke contract access",
    "NullPointerException when clicking the save button",
    "Access not revoked after project deletion",
    "Contract members not removed when the project is deleted",
    "The save button does nothing when clicked",
    "Grid shows wrong column order after refresh",
    "Users report an internal server error on checkout",
])
def test_plain_defect_titles_are_solvable(message):
    """Declarative defect / problem statements must be AUTONOMOUS_MODIFICATION, not read-only."""
    assert classify_agent_intent(message) == AgentIntent.AUTONOMOUS_MODIFICATION


@pytest.mark.parametrize("message", [
    "fix the coordinator revoke cascade",
    "implement the contract member removal",
    "refactor the project saga handler",
    "solve this ticket",
])
def test_explicit_modification_verbs(message):
    assert classify_agent_intent(message) == AgentIntent.AUTONOMOUS_MODIFICATION


@pytest.mark.parametrize("message", [
    "What does this button do?",
    "Where is updateProjectUser defined?",
    "How does the revoke flow work?",
    "Can you explain the coordinator removal path?",
    "What is the credit limit calculation?",  # 'credit' must NOT trigger the 'edit' verb
    "[EXPLAIN ONLY] fix the bug",             # explicit override wins over the verb
    "Why did the fix in CC4E-8421 fail? Explain the root cause",
    "Explain how to fix this",
    "Analyze why the fix didn't work",
    "Investigate why the coordinator deletion failed",
    "Diagnose the 500 error in auth service",
    "Why did endpoint X return 401? Explain only, do not modify code",
    "Could you explain why the test failed after the patch was merged?",
])
def test_informational_questions_stay_read_only(message):
    assert classify_agent_intent(message) == AgentIntent.EXPLANATION_REVIEW


@pytest.mark.parametrize("message", [
    "/plan the migration to the new schema",
    "/grill-me on the saga design",
    "Give me an architectural plan for the revoke cascade",
])
def test_planning_requests(message):
    assert classify_agent_intent(message) == AgentIntent.PLANNING_ARCHITECTURE


def test_question_asking_to_also_apply_is_modification():
    """A question that also asks to apply the fix should be treated as a task."""
    msg = "Why does removing a coordinator not revoke access, and fix it?"
    assert classify_agent_intent(msg) == AgentIntent.AUTONOMOUS_MODIFICATION


def test_empty_message_defaults_to_read_only():
    assert classify_agent_intent("") == AgentIntent.EXPLANATION_REVIEW


# ─── 1b. Semantic-regex expansion: state-inconsistency / leakage / colloquial ─

@pytest.mark.parametrize("message", [
    "The project coordinator continues to retain contract permissions upon departure.",
    "Removed users still have access to the project workspace.",
    "Coordinator still active after being removed from the project.",
    "Contract members remain assigned once the project is gone.",
    "There is a memory leak in the saga dispatcher.",
    "Orphaned contract rows left behind after deletion.",
])
def test_state_inconsistency_titles_are_solvable(message):
    """Affirmative 'unwanted persistence' defects (no negation) must still be solvable tasks."""
    assert classify_agent_intent(message) == AgentIntent.AUTONOMOUS_MODIFICATION


@pytest.mark.parametrize("message", [
    "contract member delete endpoint dosent work",   # typo for doesn't
    "the save button is borked",                      # slang for broken
    "revoke flow didnt trigger the saga",             # missing apostrophe
])
def test_colloquial_and_typo_defects_are_solvable(message):
    assert classify_agent_intent(message) == AgentIntent.AUTONOMOUS_MODIFICATION


@pytest.mark.parametrize("message", [
    "Why is contract access still active for removed coordinators?",
    "How come the save button doesn't trigger the saga?",
    "Why does the coordinator still have access after removal?",
])
def test_question_phrased_defects_are_solvable(message):
    """A question that describes a concrete defect is a troubleshoot-and-fix request."""
    assert classify_agent_intent(message) == AgentIntent.AUTONOMOUS_MODIFICATION


def test_explicit_read_only_tag_beats_defect_question():
    """An explicit read-only tag must still win even when a defect is described."""
    msg = "[READ ONLY] Why is access still active for removed coordinators?"
    assert classify_agent_intent(msg) == AgentIntent.EXPLANATION_REVIEW


@pytest.mark.parametrize("message", [
    "Don't just explain only, actually fix the bug",
    "Instead of explaining, implement the contract member removal",
    "Stop explaining and fix CC4E-01",
    "No need to explain, apply the fix",
])
def test_negated_explain_with_modification(message):
    """Negated explanation cues combined with action verbs or defects are tasks."""
    assert classify_agent_intent(message) == AgentIntent.AUTONOMOUS_MODIFICATION


@pytest.mark.parametrize("message", [
    "EXPLAIN ONLY] Do not trigger the autonomous code modification workflow. Analyze CC4E-FS-COMPLEX-03",
    "explain only: what does IssuesController do?",
    "Can you explain why the NullPointerException happens?",
    "Walk me through the error in PaymentService",
])
def test_bracket_variants_and_walkthrough_questions(message):
    """Bracket variants and explicit walkthrough questions remain read-only."""
    assert classify_agent_intent(message) == AgentIntent.EXPLANATION_REVIEW


# ─── 1c. Two-tier hybrid router (deterministic fast-path + LLM arbiter) ───────

class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, mode, recorder):
        self._mode = mode
        self._recorder = recorder

    def generate_content(self, model, contents, config):
        self._recorder["called"] = True
        return _FakeResp('{"mode": "%s"}' % self._mode)


class _FakeClient:
    def __init__(self, mode, recorder):
        self.models = _FakeModels(mode, recorder)


def test_hybrid_high_confidence_never_calls_llm():
    """A clear message is resolved by the fast-path with zero LLM calls."""
    recorder = {"called": False}
    client = _FakeClient("EXPLANATION_REVIEW", recorder)
    assert classify_agent_intent_hybrid("fix the revoke cascade", client=client) == AgentIntent.AUTONOMOUS_MODIFICATION
    assert recorder["called"] is False


def test_ambiguous_message_is_low_confidence():
    """A message with no cue at all is the ambiguous fall-through the arbiter should resolve."""
    intent, confidence = _classify_deterministic("Look at IssuesController line 85")
    assert intent == AgentIntent.EXPLANATION_REVIEW
    assert confidence == "low"


def test_hybrid_arbiter_resolves_ambiguous_to_modification():
    recorder = {"called": False}
    client = _FakeClient("AUTONOMOUS_MODIFICATION", recorder)
    out = classify_agent_intent_hybrid("Look at IssuesController line 85", client=client)
    assert recorder["called"] is True
    assert out == AgentIntent.AUTONOMOUS_MODIFICATION


def test_hybrid_without_client_falls_back_to_deterministic():
    assert classify_agent_intent_hybrid("Look at IssuesController line 85", client=None) == AgentIntent.EXPLANATION_REVIEW


# ─── 2. Chat → Flow delegation hand-off ──────────────────────────────────────

class _FakeLeanResult:
    def __init__(self, status):
        self.status = status
        self.reason = "all verify rungs passed"
        self.locked_files = ["src/app/foo.component.ts"]
        self.attempts = 2
        self.trace = ["triage: layer=ui", "attempt 1: rung=build", "attempt 2: verified"]
        self.llm_calls = 3


def test_delegation_routes_to_run_lean_path():
    """Chat delegation must hand the ticket to the Flow runner and surface its verdict + diffs."""
    calls = {}
    progress = []

    def fake_prepare(ws, tid, title, desc, labels, meter, run_tier2):
        calls["prepare"] = (ws, tid, title, desc)
        return (lambda *a, **k: [], lambda *a, **k: None, type("S", (), {"layer": "ui", "confidence": 0.9})())

    def fake_runner(tid, title, desc, ws, patch_fn, verify_fn, labels=None, hard_timeout_s=600.0, meter=None):
        calls["runner"] = (tid, title, desc, ws)
        return _FakeLeanResult("verified")

    res = delegate_ticket_to_flow(
        "C:/ws", "Removing a coordinator does not revoke access",
        "Full description of the defect.",
        ticket_id="T-1",
        progress_cb=lambda stage, detail="": progress.append(stage),
        runner=fake_runner,
        prepare=fake_prepare,
    )

    assert calls["runner"][0] == "T-1"
    assert calls["prepare"][1] == "T-1"
    assert res["status"] == "verified"
    assert res["locked_files"] == ["src/app/foo.component.ts"]
    assert res["attempts"] == 2
    # Diffs are the per-attempt / verified trace lines.
    assert any("verified" in d for d in res["diffs"])
    # Progress is streamed start → done.
    assert "flow_start" in progress and "flow_done" in progress


def test_delegation_declines_non_lean_ticket_without_calling_runner():
    """If localization declines, we report out_of_scope and never invent an edit."""
    runner_called = {"v": False}

    def fake_prepare(ws, tid, title, desc, labels, meter, run_tier2):
        return (None, None, type("S", (), {"layer": "unknown", "confidence": 0.1})())

    def fake_runner(*a, **k):
        runner_called["v"] = True
        return _FakeLeanResult("verified")

    res = delegate_ticket_to_flow(
        "C:/ws", "Vague request", "No anchors here.",
        runner=fake_runner, prepare=fake_prepare,
    )

    assert res["status"] == "out_of_scope"
    assert runner_called["v"] is False
    assert res["attempts"] == 0


# ─── 3. Polyglot build verification ──────────────────────────────────────────

def test_detect_build_command_npm_frontend(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts": {"build": "ng build"}}', encoding="utf-8")
    label, cmd = HostEnvironmentTools._detect_build_command(tmp_path, is_win=False)
    assert label == "npm"
    assert cmd == "npm run build"


def test_detect_build_command_npm_test_only(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    label, cmd = HostEnvironmentTools._detect_build_command(tmp_path, is_win=False)
    assert label == "npm"
    assert "test" in cmd


def test_detect_build_command_gradle_posix(tmp_path: Path):
    (tmp_path / "gradlew").write_text("#!/bin/sh\n", encoding="utf-8")
    label, cmd = HostEnvironmentTools._detect_build_command(tmp_path, is_win=False)
    assert label == "gradle"
    assert "compileJava" in cmd


def test_detect_build_command_gradle_windows(tmp_path: Path):
    (tmp_path / "gradlew.bat").write_text("@echo off\n", encoding="utf-8")
    label, cmd = HostEnvironmentTools._detect_build_command(tmp_path, is_win=True)
    assert label == "gradle"
    assert "gradlew.bat" in cmd


def test_detect_build_command_maven(tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    label, cmd = HostEnvironmentTools._detect_build_command(tmp_path, is_win=False)
    assert label == "maven"
    assert "compile" in cmd


def test_detect_build_command_none_when_no_tool(tmp_path: Path):
    (tmp_path / "readme.txt").write_text("nothing", encoding="utf-8")
    assert HostEnvironmentTools._detect_build_command(tmp_path, is_win=False) is None


def test_verify_service_compilation_reports_missing_tool(tmp_path: Path):
    tools = HostEnvironmentTools(str(tmp_path))
    out = tools.verify_service_compilation()
    assert "No supported build tool" in out


def test_kv_cache_prefix_identical_across_dynamic_turns():
    """Verify that PromptAssembler produces byte-identical prefixes up to USER_REQUEST across turns with different cursor/blackboard."""
    from ticket_to_code.runtime.agent_harness import HostStateMetadata, PromptAssembler

    meta1 = HostStateMetadata(
        os_name="windows",
        workspace_path="C:/repo",
        app_data_dir="C:/app",
        active_document="src/Foo.java",
        cursor_line=10,
        local_time_iso="2026-10-11",
    )
    meta2 = HostStateMetadata(
        os_name="windows",
        workspace_path="C:/repo",
        app_data_dir="C:/app",
        active_document="src/Bar.java",
        cursor_line=999,
        local_time_iso="2026-10-11",
    )

    prompt1 = PromptAssembler.build_user_prompt("Fix defect", meta1, ki_summaries="KI-1", arch_block="Arch", blackboard_xml="<facts>1</facts>")
    prompt2 = PromptAssembler.build_user_prompt("Fix defect", meta2, ki_summaries="KI-1", arch_block="Arch", blackboard_xml="<facts>2</facts>")

    prefix1 = prompt1.split("<USER_REQUEST>")[0]
    prefix2 = prompt2.split("<USER_REQUEST>")[0]

    assert prefix1 == prefix2, "Prefix prior to USER_REQUEST must be byte-identical for KV cache prefix reuse"
    assert "<ADDITIONAL_METADATA>" in prompt1.split("</USER_REQUEST>")[1]
    assert "<facts>1</facts>" in prompt1.split("</USER_REQUEST>")[1]


def test_context_compactor_structured_resumption():
    from ticket_to_code.runtime.agent_harness import ContextCompactor

    block = ContextCompactor.compact_history(
        task_id="CC4E-FS-COMPLEX-03",
        user_requests=["What is the endpoint?", "How does Angular consume it?", "Fix the breakdown badge"],
        verified_facts=["IssueCount has criticalCount", "DeliverablesComponent uses Enterprise UX Components"],
        active_context="Workspace: C:/repo\nStatus: in_progress",
    )

    assert "# Resuming from a compaction" in block
    assert "1. What is the endpoint?" in block
    assert "3. Fix the breakdown badge" in block
    assert "- IssueCount has criticalCount" in block
    assert "CC4E-FS-COMPLEX-03" in block
    assert "Enterprise UX Components" in block


def test_react_loop_governor_anti_spinning():
    from ticket_to_code.runtime.agent_harness import ReActLoopGovernor

    gov = ReActLoopGovernor(max_consecutive_redundant=3, hard_stop_limit=5)

    # 2 redundant empty calls -> no alert yet
    assert gov.evaluate_call("search_codebase", "foo", "No matches found") is None
    assert gov.evaluate_call("search_codebase", "foo", "No matches found") is None

    # 3rd redundant call -> CONVERGENCE DIRECTIVE
    g3 = gov.evaluate_call("search_codebase", "foo", "No matches found")
    assert g3 is not None
    assert "CONVERGENCE DIRECTIVE" in g3

    # High signal call resets counter
    assert gov.evaluate_call("search_codebase", "bar", "Found 10 matches: class IssueController...") is None
    assert gov.consecutive_redundant_count == 0


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("/explain why did the fix fail in CC4E-8421", AgentIntent.EXPLANATION_REVIEW),
        ("/ask how does acquireLock work", AgentIntent.EXPLANATION_REVIEW),
        ("/read IssueRepository.java", AgentIntent.EXPLANATION_REVIEW),
        ("/fix CC4E-8421", AgentIntent.AUTONOMOUS_MODIFICATION),
        ("/patch the deadlock in consolidation", AgentIntent.AUTONOMOUS_MODIFICATION),
        ("/flow implement issue consolidation", AgentIntent.AUTONOMOUS_MODIFICATION),
        ("/plan schema migration", AgentIntent.PLANNING_ARCHITECTURE),
        ("How do we resolve the concurrent lock contention?", AgentIntent.EXPLANATION_REVIEW),
        ("What is the recommended way to patch this?", AgentIntent.EXPLANATION_REVIEW),
        ("How should we refactor the controller?", AgentIntent.EXPLANATION_REVIEW),
        ("Can you advise on how to handle the deadlock?", AgentIntent.EXPLANATION_REVIEW),
        ("How can we fix the issue if the database is locked?", AgentIntent.EXPLANATION_REVIEW),
        ("Help me understand why the transaction failed", AgentIntent.EXPLANATION_REVIEW),
        ("--mode=explain Analyze the concurrency model", AgentIntent.EXPLANATION_REVIEW),
        ("[read] check the locking query", AgentIntent.EXPLANATION_REVIEW),
    ]
)
def test_slash_commands_and_consulting_inquiries(msg, expected):
    assert classify_agent_intent(msg) == expected


def test_multi_symbol_inspection(tmp_path):
    from ticket_to_code.runtime.tool_registry import UnifiedToolRegistry
    java_file = tmp_path / "Service.java"
    java_file.write_text(
        "public class Service {\n"
        "    public void methodA() {\n"
        "        System.out.println(\"A\");\n"
        "    }\n"
        "    public void methodB() {\n"
        "        System.out.println(\"B\");\n"
        "    }\n"
        "    public void methodC() {\n"
        "        System.out.println(\"C\");\n"
        "    }\n"
        "}\n",
        encoding="utf-8"
    )
    reg = UnifiedToolRegistry(tmp_path)
    res = reg.inspect_file("Service.java", symbol="methodA, methodC")
    assert "methodA" in res
    assert "methodC" in res
    assert "methodB" not in res


def test_host_state_metadata_build_tool_injection(tmp_path):
    from ticket_to_code.runtime.agent_harness import HostStateMetadata
    # Create mock gradle project
    (tmp_path / "gradlew.bat").write_text("@echo off", encoding="utf-8")
    meta = HostStateMetadata.capture_current(str(tmp_path))
    assert meta.build_tool == "gradle"
    assert "gradlew.bat" in meta.detected_marker
    rendered = meta.render_static_user_info()
    assert "<build_and_runtime_environment>" in rendered
    assert "Detected Build Tool: gradle" in rendered
    assert "CRITICAL CONSTRAINT: You MUST use 'gradle'" in rendered


def test_universal_engineering_invariants_coverage():
    from ticket_to_code.runtime.agent_harness import UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS
    # Truly universal principles stay in the constant (asserted by title, not by number)
    for title in [
        "CALL-CHAIN TRACEABILITY & ENTRYPOINT REACHABILITY",
        "ORM & REPOSITORY NESTED PROPERTY TRAVERSAL",
        "DISTRIBUTED TRANSACTION & SAGA ATOMICITY",
        "POLYGLOT BUILD TOOL CONSTRAINTS",
        "SURGICAL DELTA PRESERVATION & REPOSITORY INTEGRITY",
        "EXHAUSTIVE MULTI-PART TICKET DECOMPOSITION",
    ]:
        assert title in UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS, title
    # Repository-specific  material must NOT leak into the universal prompt
    assert "ContractMember" not in UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS
    assert "CC4E" not in UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS
    assert "HIERARCHICAL RBAC" not in UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS
    assert "8 microservices" not in UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS


def test_repo_invariants_loaded_only_with_marker(tmp_path):
    import json
    from ticket_to_code.skills.loader import SkillLoader
    from ticket_to_code.runtime.agent_harness import PromptAssembler, AgentIntent

    # 1. Bare workspace: no repo invariants, and the system prompt has none either
    assert SkillLoader(tmp_path).load_repo_invariants() == []
    sys_inst = PromptAssembler.build_system_instruction(AgentIntent.EXPLANATION_REVIEW, workspace_path=str(tmp_path))
    assert "HIERARCHICAL RBAC" not in sys_inst
    assert "REPOSITORY-SPECIFIC INVARIANTS" not in sys_inst

    # 2. Marker directory present: packaged CC4E skill is loaded
    (tmp_path / "xchange-ui").mkdir()
    rules = SkillLoader(tmp_path).load_repo_invariants()
    joined = "\n".join(rules)
    for title in [
        "HIERARCHICAL RBAC & LIFECYCLE SEPARATION",
        "SPRING BEAN INJECTION",
        "JPA WRITE-BEHIND",
        "CROSS-ENTITY KEY MAPPING FIDELITY",
    ]:
        assert title in joined, title
    sys_inst = PromptAssembler.build_system_instruction(AgentIntent.EXPLANATION_REVIEW, workspace_path=str(tmp_path))
    assert "REPOSITORY-SPECIFIC INVARIANTS" in sys_inst
    assert "HIERARCHICAL RBAC" in sys_inst

    # 3. Workspace override wins over the packaged skill
    override_dir = tmp_path / ".aviator" / "skills"
    override_dir.mkdir(parents=True)
    (override_dir / "cloudforge_invariants.json").write_text(
        json.dumps({"id": "cloudforge-invariants", "rules": ["CUSTOM OVERRIDE RULE ONLY"]}),
        encoding="utf-8",
    )
    rules = SkillLoader(tmp_path).load_repo_invariants()
    assert rules == ["CUSTOM OVERRIDE RULE ONLY"]
    sys_inst = PromptAssembler.build_system_instruction(AgentIntent.EXPLANATION_REVIEW, workspace_path=str(tmp_path))
    assert "CUSTOM OVERRIDE RULE ONLY" in sys_inst
    assert "HIERARCHICAL RBAC" not in sys_inst



