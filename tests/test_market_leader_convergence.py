"""
Empirical Verification Suite for Market-Leader Convergence Targets.

Verifies:
1. ContextMentionResolver: Cursor-style zero-token @file, @symbol, and @git mentions.
2. InteractiveClarificationGate: Thread-safe HITL decision modals, user response, and fallback.
3. DiffReviewEngine: Multi-hunk unified diff decomposition and selective hunk application.
4. RuntimeTelemetryTracker: Live token economics, KV-cache hit rate, and eval reports.
5. UnifiedToolRegistry: Selective diff and interactive clarification tool invocation.
"""

import json
from pathlib import Path
import pytest

from ticket_to_code.runtime.context_mentions import ContextMentionResolver
from ticket_to_code.runtime.interactive_gate import (
    InteractiveClarificationGate,
    ClarificationResponse,
)
from ticket_to_code.runtime.diff_engine import DiffReviewEngine
from ticket_to_code.runtime.telemetry_evals import RuntimeTelemetryTracker
from ticket_to_code.runtime.deductive_blackboard import DeductiveBlackboard
from ticket_to_code.runtime.tool_registry import UnifiedToolRegistry


def test_context_mention_resolver(tmp_path: Path):
    # Setup test workspace files
    src_dir = tmp_path / "src" / "services"
    src_dir.mkdir(parents=True)
    target_file = src_dir / "ContractService.java"
    target_file.write_text("public class ContractService { int count = 42; }", encoding="utf-8")

    resolver = ContextMentionResolver(workspace_path=tmp_path)

    # 1. Test explicit @file:
    res1 = resolver.resolve_mentions("Please inspect @file:src/services/ContractService.java now.")
    assert len(res1.resolved_mentions) == 1
    m1 = res1.resolved_mentions[0]
    assert m1.resolved
    assert m1.mention_type == "file"
    assert "ContractService { int count = 42; }" in m1.content

    # 2. Test basename implicit @ContractService.java
    res2 = resolver.resolve_mentions("Fix the issue in @ContractService.java immediately.")
    assert len(res2.resolved_mentions) == 1
    m2 = res2.resolved_mentions[0]
    assert m2.resolved
    assert "int count = 42" in m2.content

    # 3. Test git status mention
    res3 = resolver.resolve_mentions("Check @git:status before committing.")
    assert len(res3.resolved_mentions) == 1
    assert res3.resolved_mentions[0].mention_type == "git"


def test_interactive_clarification_gate_live_response():
    bb = DeductiveBlackboard()
    req_question = "Should we soft-delete or hard-delete project members?"
    options = ["Soft Delete (mark isActive=false)", "Hard Delete (remove row)"]

    import threading
    received_event = {}

    def mock_ui_broadcast(payload):
        received_event.update(payload)
        # Simulate user responding after 20ms
        def _respond():
            InteractiveClarificationGate.submit_response(
                request_id=payload["request_id"],
                selected_options=["Hard Delete (remove row)"],
                user_notes="Hard delete is required to avoid orphan records.",
            )
        threading.Timer(0.02, _respond).start()

    InteractiveClarificationGate.set_ui_broadcaster(mock_ui_broadcast)

    result = InteractiveClarificationGate.request_clarification(
        question=req_question,
        options=options,
        recommended_option=options[0],
        timeout_seconds=5.0,
        blackboard=bb,
    )

    assert "Hard Delete (remove row)" in result
    assert "Hard delete is required" in result
    # Verify recorded in DeductiveBlackboard
    assert any("Hard Delete" in f for f in bb.verified_facts)

    # Clean up broadcaster
    InteractiveClarificationGate.set_ui_broadcaster(None)


def test_interactive_clarification_gate_auto_fallback():
    bb = DeductiveBlackboard()
    # In headless mode (no broadcaster), should fallback to recommended default immediately
    result = InteractiveClarificationGate.request_clarification(
        question="Select DB isolation level",
        options=["READ_COMMITTED", "SERIALIZABLE"],
        recommended_option="READ_COMMITTED",
        timeout_seconds=0.01,
        blackboard=bb,
    )
    assert "[AUTO_FALLBACK]" in result
    assert "READ_COMMITTED" in result


def test_diff_review_engine_hunk_decomposition_and_selective_apply():
    orig_lines = [f"line {i}\n" for i in range(1, 25)]
    orig_text = "".join(orig_lines)

    # Modify line 2 and line 20 (2 separate hunks well beyond 3 context lines)
    mod_lines = list(orig_lines)
    mod_lines[1] = "line 2 MODIFIED\n"
    mod_lines[19] = "line 20 MODIFIED\n"
    mod_text = "".join(mod_lines)

    file_diff = DiffReviewEngine.compute_file_diff("test.txt", orig_text, mod_text)
    assert not file_diff.is_noop
    assert len(file_diff.hunks) == 2
    assert file_diff.hunks[0].hunk_index == 0
    assert file_diff.hunks[1].hunk_index == 1

    # Selective apply: accept ONLY hunk 0, reject hunk 1
    selective_result = DiffReviewEngine.apply_selected_hunks(orig_text, file_diff.hunks, {0})
    assert "line 2 MODIFIED" in selective_result
    assert "line 20 MODIFIED" not in selective_result
    assert "line 20\n" in selective_result


def test_runtime_telemetry_evals_tracking(tmp_path: Path):
    tracker = RuntimeTelemetryTracker(session_id="test_session_01", workspace_path=tmp_path)

    tracker.record_token_usage(prompt_tokens=1000, completion_tokens=250, cached_tokens=3000)
    tracker.record_tool_call("grep_search", duration_ms=45, success=True)
    tracker.record_tool_call("grep_search", duration_ms=55, success=True)
    tracker.record_tool_call("compileJava", duration_ms=1200, success=True)
    tracker.record_compiler_check(passed=True, repaired=False)
    tracker.record_clarification("Should we drop table?", "No, keep backup.", was_fallback=False)
    tracker.record_phase_duration("localization", duration_ms=350)

    report = tracker.export_eval_report()

    assert report["session_id"] == "test_session_01"
    assert report["token_economics"]["tokens_in"] == 1000
    assert report["token_economics"]["tokens_cached"] == 3000
    assert report["token_economics"]["cache_hit_rate_pct"] == 75.0
    assert report["tools"]["grep_search"]["calls"] == 2
    assert report["compiler_evals"]["passes"] == 1
    assert report["human_in_loop"]["clarifications_answered"] == 1

    # Verify report written to disk
    eval_file = tmp_path / ".aviator" / "evals" / "test_session_01.json"
    assert eval_file.exists()
    disk_data = json.loads(eval_file.read_text(encoding="utf-8"))
    assert disk_data["session_id"] == "test_session_01"


def test_unified_tool_registry_hunk_diff_and_selective_apply(tmp_path: Path):
    target = tmp_path / "App.java"
    target.write_text("class App {\n  int a = 1;\n  int b = 2;\n}\n", encoding="utf-8")

    registry = UnifiedToolRegistry(str(tmp_path))

    proposed = "class App {\n  int a = 999;\n  int b = 2;\n}\n"
    diff_output = registry.generate_unified_diff("App.java", proposed)
    assert "Unified Diff for 'App.java'" in diff_output
    assert "[HUNK 0]" in diff_output

    # Apply hunk 0
    apply_res = registry.apply_selected_diff_hunks("App.java", proposed, "0")
    assert "Applied 1 of 1 hunks" in apply_res
    assert "int a = 999;" in target.read_text(encoding="utf-8")
