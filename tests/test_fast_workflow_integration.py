import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ticket_to_code.models import (
    BuildResult,
    BuildStatus,
    GeneratedCode,
    TestResult,
    TestStatus,
    ValueEdgeTicket,
)
from ticket_to_code.runtime.fast_ide_engine import FastIDEResult
from ticket_to_code.workflow import run_autonomous_workflow


def test_run_autonomous_workflow_fast_ide_success(tmp_path):
    """Verify run_autonomous_workflow executes FastIDEEngine and returns canonical state."""
    src_file = tmp_path / "App.java"
    src_file.write_text("class App { void run() {} }", encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="TICKET-FAST-1",
        title="Fix App.java run method",
        description="Update App.java",
    )

    fake_fast_result = FastIDEResult(
        success=True,
        status="verified_success",
        duration_seconds=12.4,
        tokens_consumed=3420,
        modified_files=["App.java"],
        diffs={"App.java": "--- a/App.java\n+++ b/App.java\n@@ -1 +1 @@\n-class App { void run() {} }\n+class App { void run() { int x = 1; } }"},
        build_passed=True,
        test_passed=True,
    )

    with patch("ticket_to_code.workflow.WorkflowAgents") as MockAgents:
        mock_agents = MagicMock()
        mock_agents.llm = MagicMock()
        MockAgents.return_value = mock_agents

        with patch("ticket_to_code.runtime.fast_ide_engine.FastIDEEngine.run", return_value=fake_fast_result):
            state = run_autonomous_workflow(
                ticket=ticket,
                workspace_path=str(tmp_path),
                use_fast_ide=True,
            )

    assert state["status"] == "completed"
    assert len(state["generated_code"]) == 1
    assert state["generated_code"][0].file_path == "App.java"
    assert state["build_result"].status == BuildStatus.SUCCESS
    assert state["test_result"].status == TestStatus.ALL_PASSED
    assert state["token_usage"]["total_tokens"] == 3420
    assert state["token_usage"]["latency_seconds"] == 12.4
    assert state["token_usage"]["engine"] == "FastIDEEngine (Cursor/Aider Style)"


def test_run_autonomous_workflow_fallback_on_fast_ide_escalation(tmp_path):
    """Verify run_autonomous_workflow falls through to deep graph if FastIDEEngine escalates."""
    ticket = ValueEdgeTicket(
        ticket_id="TICKET-COMPLEX-1",
        title="Distributed Architecture Refactor",
        description="Refactor across all microservices",
    )

    fake_escalate_result = FastIDEResult(
        success=False,
        status="out_of_scope",
        duration_seconds=0.1,
        tokens_consumed=0,
        modified_files=[],
        escalate_to_full_graph=True,
        error_message="Requires deep architecture graph",
    )

    fallback_called = False

    def mock_deep_graph(*args, **kwargs):
        nonlocal fallback_called
        fallback_called = True
        return {
            "ticket": ticket,
            "status": "completed",
            "generated_code": [],
        }

    with patch("ticket_to_code.workflow.WorkflowAgents"):
        with patch("ticket_to_code.runtime.fast_ide_engine.FastIDEEngine.run", return_value=fake_escalate_result):
            with patch("ticket_to_code.workflow.run_autonomous_workflow_langgraph", side_effect=mock_deep_graph):
                state = run_autonomous_workflow(
                    ticket=ticket,
                    workspace_path=str(tmp_path),
                    use_fast_ide=True,
                )

    assert fallback_called is True
    assert state["status"] == "completed"


@pytest.mark.asyncio
async def test_main_run_workflow_async_fast_ide_execution(tmp_path):
    """Verify chatbot/backend/main.py run_workflow_async calls FastIDEEngine directly and records status."""
    import sys
    sys.path.insert(0, "chatbot/backend")
    from main import run_workflow_async, workflow_status

    src_file = tmp_path / "Service.java"
    src_file.write_text("class Service {}", encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="TASK-ASYNC-01",
        title="Update Service.java",
        description="Update Service.java",
    )
    workflow_id = "wf-test-async-1"
    workflow_status[workflow_id] = {
        "status": "running",
        "ticket_id": ticket.ticket_id,
        "repo_path": str(tmp_path),
        "steps": [],
    }

    fake_result = FastIDEResult(
        success=True,
        status="verified_success",
        duration_seconds=5.2,
        tokens_consumed=1850,
        modified_files=["Service.java"],
        diffs={"Service.java": "+ class Service { void ok() {} }"},
        build_passed=True,
        test_passed=True,
        message="FastIDEEngine completed successfully",
    )

    with patch("ticket_to_code.workflow.WorkflowAgents"):
        with patch("ticket_to_code.runtime.fast_ide_engine.FastIDEEngine.run", return_value=fake_result):
            with patch("main.broadcast_to_workflow", return_value=None):
                with patch("main.manager.broadcast", return_value=None):
                    await run_workflow_async(
                        workflow_id=workflow_id,
                        ticket=ticket,
                        workspace_path=str(tmp_path),
                        execution_mode="pipeline",
                    )

    assert workflow_status[workflow_id]["status"] == "completed"
    assert "Service.java" in workflow_status[workflow_id]["generated_files"]
    assert workflow_status[workflow_id]["token_usage"]["total_tokens"] == 1850
    assert workflow_status[workflow_id]["token_usage"]["engine"] == "FastIDEEngine (4-Stage ReAct Spine)"

