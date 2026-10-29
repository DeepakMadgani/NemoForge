"""
Unit tests for:
1. Targeted Line Slicing (#Lstart-end) in ErrorResolutionAgent, EditLoopAgent, and Tools.
2. Cross-stage KV context caching and token savings calculation in PromptCacheManager.
"""
import pytest
from pathlib import Path
from ticket_to_code.intelligence.prompt_cache_manager import (
    StablePromptStructure,
    PromptCacheManager,
    WorkflowCacheSession,
    get_or_create_cache_session
)
from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent
from ticket_to_code.agents.edit_loop_agent import EditLoopAgent
from ticket_to_code.reasoning.tools import _t_view_file, _t_read_file, ToolContext


def test_error_resolution_agent_line_slicing(tmp_path, caplog):
    # Create sample 300-line file
    test_file = tmp_path / "MyService.ts"
    lines = [f"const line_{i} = {i};" for i in range(1, 301)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    agent = ErrorResolutionAgent(
        llm=object(),
        workspace_path=tmp_path
    )
    agent.pre_edit_snapshots = {"MyService.ts": test_file.read_text(encoding="utf-8")}

    import logging
    with caplog.at_level(logging.INFO):
        # 1. Read slice #L150-160
        res = agent._tool_read_file("MyService.ts", start_line=150, end_line=160)
        assert "#L150-160" in res
        assert "150: const line_150 = 150;" in res
        assert "160: const line_160 = 160;" in res
        assert "161:" not in res
        assert "Analyzed MyService.ts #L150-160" in caplog.text

    # 2. Read original with slicing
    orig_res = agent._tool_read_original("MyService.ts", start_line=10, end_line=15)
    assert "#L10-15" in orig_res
    assert "10: const line_10 = 10;" in orig_res
    assert "15: const line_15 = 15;" in orig_res


def test_edit_loop_agent_read_file_slice(tmp_path, caplog):
    test_file = tmp_path / "SampleComponent.ts"
    lines = [f"// line {i}" for i in range(1, 101)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    agent = EditLoopAgent(workspace_path=str(tmp_path), llm=object())
    import logging
    with caplog.at_level(logging.INFO):
        slice_res = agent.read_file_slice("SampleComponent.ts", start_line=25, end_line=35)
        assert slice_res is not None
        assert "#L25-35" in slice_res
        assert "25: // line 25" in slice_res
        assert "35: // line 35" in slice_res
        assert "36:" not in slice_res
        assert "Analyzed SampleComponent.ts #L25-35" in caplog.text


def test_tools_view_file_slice_header_and_logging(tmp_path, caplog):
    test_file = tmp_path / "UserService.java"
    lines = [f"int field_{i} = {i};" for i in range(1, 101)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    ctx = ToolContext(ticket=None, workspace_path=str(tmp_path), agents=None, brain=None)
    import logging
    with caplog.at_level(logging.INFO):
        view_res = _t_view_file(ctx, {"path": "UserService.java", "start_line": 20, "end_line": 30})
        assert "File: UserService.java #L20-30" in view_res
        assert "20: int field_20 = 20;" in view_res
        assert "30: int field_30 = 30;" in view_res
        assert "Analyzed UserService.java #L20-30" in caplog.text


def test_cross_stage_kv_cache_reuse_and_token_savings():
    static_struct = StablePromptStructure(
        system_instruction="You are an expert autonomous software engineer.",
        tool_schemas=[
            {"name": "read_file", "description": "Read file"},
            {"name": "write_patch", "description": "Write patch"}
        ],
        repo_skeletons=[
            {"path": "src/app.ts", "skeleton": "export class App { run(): void; }"},
            {"path": "src/service.ts", "skeleton": "export class Service { call(): boolean; }"}
        ],
        episodic_memories=[
            "Fixed TS2339 by importing missing module",
            "Fixed NPE by adding null check in controller"
        ]
    )

    session = WorkflowCacheSession(
        workflow_id="wf-test-123",
        model_id="gemini-2.5-pro",
        static_structure=static_struct
    )

    # Stage 1: Investigation
    prompt1 = session.prepare_stage_prompt(
        stage_name="investigate",
        stage_instructions="Investigate candidate files for ticket T-100",
        stage_delta="initial triage"
    )
    t1 = prompt1["telemetry"]
    assert t1["is_cache_hit"] is False
    assert t1["cost_savings_pct"] == 0.0

    # Simulate Stage 1 completing
    session.record_stage_output("investigate", "Found candidate file: src/service.ts at line 42.")

    # Stage 2: Planning with SAME model
    prompt2 = session.prepare_stage_prompt(
        stage_name="plan",
        stage_instructions="Formulate patch plan based on src/service.ts",
        stage_delta="prior findings",
        model_id="gemini-2.5-pro"
    )
    t2 = prompt2["telemetry"]
    # Same model & same static prefix -> CACHE HIT!
    assert t2["is_cache_hit"] is True
    assert t2["cached_tokens"] > 0
    assert t2["cost_savings_pct"] > 50.0  # Big savings!
    assert t2["cached_tokens"] + t2["new_tokens"] == t2["total_tokens"]

    # Stage 3: Switching model -> Cache penalty
    prompt3 = session.prepare_stage_prompt(
        stage_name="generate_code",
        stage_instructions="Write code",
        stage_delta="code plan",
        model_id="claude-3-5-sonnet"
    )
    t3 = prompt3["telemetry"]
    assert t3["is_cache_hit"] is False
    assert t3["cost_savings_pct"] == 0.0
    assert t3["cache_state"] == "RESET_BRIEF"
    assert t3["lossless_store_ref"] is not None


def test_kv_cache_full_eviction_and_structured_reset():
    """Verify that when context approaches capacity (>40k tokens), a structured reset is triggered."""
    static_struct = StablePromptStructure(
        system_instruction="You are Ticket-to-Code.",
        tool_schemas=[{"name": "read_file"}],
        repo_skeletons=[{"path": "foo.ts", "skeleton": "interface Foo {}"}],
    )
    session = WorkflowCacheSession(
        workflow_id="wf-capacity-test",
        model_id="gemini-2.5-pro",
        static_structure=static_struct
    )
    # Simulate massive accumulated turns (over 40,000 tokens ~ 160,000 characters)
    session.stage_history = [
        {"role": "user", "content": "Search repo " * 5000},
        {"role": "assistant", "content": "Found 100 candidate files: " + ("candidate file.ts " * 10000)}
    ]
    prompt = session.prepare_stage_prompt(
        stage_name="fix_build",
        stage_instructions="Fix remaining compiler errors",
        stage_delta="huge logs",
        model_id="gemini-2.5-pro"
    )
    t = prompt["telemetry"]
    assert t["is_near_capacity"] is True
    assert t["cache_state"] == "FULL_EVICTED"
    assert t["lossless_store_ref"] is not None
    # Conversation turns in session should now be compacted to a distilled reset brief
    assert len(session.stage_history) == 1
    assert "Structured Context Reset" in session.stage_history[0]["content"]
