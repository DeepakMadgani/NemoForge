import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ticket_to_code.workflow import (
    patch_gate_node,
    build_node,
    check_build_status,
    _perform_workspace_rollback,
    create_ticket_to_code_graph,
    run_autonomous_workflow_langgraph,
)
from ticket_to_code.models import (
    ValueEdgeTicket,
    BuildResult,
    BuildStatus,
    GeneratedCode,
)
from ticket_to_code.runtime.blast_radius import BlastRadiusAnalyzer, BlastRadiusReport
from ticket_to_code.runtime.diff_engine import DiffReviewEngine
from ticket_to_code.runtime.interactive_gate import InteractiveClarificationGate
from ticket_to_code.runtime.git_sandbox import GitWorktreeSandbox


def test_blast_radius_analyzer_integration_in_patch_gate(tmp_path):
    """
    Verify patch_gate_node runs BlastRadiusAnalyzer and records blast_radius_report.
    """
    svc_dir = tmp_path / "services" / "payment"
    svc_dir.mkdir(parents=True)
    f = svc_dir / "PaymentService.java"
    f.write_text("class PaymentService {}", encoding="utf-8")

    import subprocess
    from unittest.mock import patch

    state = {
        "workspace_path": str(tmp_path),
        "ticket": ValueEdgeTicket(ticket_id="TICKET-BLAST-1", title="Payment Fix", description="Fix payment"),
        "allowed_files": ["services/payment/PaymentService.java"],
        "generated_code": [
            GeneratedCode(
                file_path="services/payment/PaymentService.java",
                content="class PaymentService { void pay() {} }",
                language="java",
                change_type="modify",
            )
        ],
        "_run_generated_map": {"services/payment/PaymentService.java": "class PaymentService { void pay() {} }"},
        "original_file_contents": {"services/payment/PaymentService.java": "class PaymentService {}"},
    }
    mock_agents = MagicMock()

    with patch("ticket_to_code.delivery.patch_gate.PatchGate.current_head", return_value="deadbeef12345678"):
        res = patch_gate_node(state, mock_agents)
    assert "blast_radius_report" in res
    report = res["blast_radius_report"]
    assert isinstance(report, BlastRadiusReport)
    assert report.risk_level in ("low", "medium", "high", "critical")
    assert isinstance(report.modified_services, list)


def test_diff_review_engine_integration_in_rollback(tmp_path):
    """
    Verify _perform_workspace_rollback uses DiffReviewEngine to generate hunk-level diffs.
    """
    f = tmp_path / "app.py"
    f.write_text("def hello():\n    return 'v2'\n", encoding="utf-8")

    state = {
        "workspace_path": str(tmp_path),
        "ticket": ValueEdgeTicket(ticket_id="TICKET-DIFF-1", title="Diff Test", description="Test rollback"),
        "original_file_contents": {"app.py": "def hello():\n    return 'v1'\n"},
        "generated_code": [],
    }

    rollback_res = _perform_workspace_rollback(state=state, patch_filename="test_rollback.patch")

    # Check that diff was generated and archived in traces
    assert rollback_res["rollback_performed"] is True
    patch_dest = rollback_res["patch_file"]
    assert patch_dest is not None
    archived_text = Path(patch_dest).read_text(encoding="utf-8")
    assert "--- a/app.py" in archived_text or "--- " in archived_text
    assert "-    return 'v1'" in archived_text
    assert "+    return 'v2'" in archived_text
    # Verify file was restored
    assert f.read_text(encoding="utf-8") == "def hello():\n    return 'v1'\n"


def test_interactive_clarification_gate_in_check_build_status():
    """
    Verify check_build_status queries InteractiveClarificationGate when pre-existing errors occur.
    """
    state = {
        "build_result": BuildResult(
            status=BuildStatus.FAILURE,
            exit_code=1,
            errors=["Legacy compile error in third_party/lib.py:10"],
        ),
        "build_differential_accept": True,
        "build_infrastructure_only": False,
        "consecutive_identical_build_errors": 0,
        "retry_attempt": 0,
        "max_retry_attempts": 3,
    }

    # In automated test mode without HITL, should default to 'leave' and route to 'test'
    route = check_build_status(state)
    assert route == "test"


def test_interactive_clarification_gate_stop_choice():
    """
    Verify check_build_status respects 'stop' decision from InteractiveClarificationGate.
    """
    state = {
        "build_result": BuildResult(
            status=BuildStatus.FAILURE,
            exit_code=1,
            errors=["Critical failure"],
        ),
        "build_differential_accept": True,
        "pre_existing_decision": "stop",
    }

    route = check_build_status(state)
    assert route == "memory_update"


def test_polyglot_build_discoverer_fallback_in_build_node(tmp_path):
    """
    Verify build_node utilizes PolyglotBuildDiscoverer fallback when standard engine fails.
    """
    # Create a Go project with go.mod
    go_mod = tmp_path / "go.mod"
    go_mod.write_text("module example.com/pkg\n\ngo 1.20\n", encoding="utf-8")
    main_go = tmp_path / "main.go"
    main_go.write_text('package main\nfunc main() {}\n', encoding="utf-8")

    state = {
        "workspace_path": str(tmp_path),
        "ticket": ValueEdgeTicket(ticket_id="TICKET-GO-1", title="Go Task", description="Add Go feature"),
        "generated_code": [
            GeneratedCode(file_path="main.go", content='package main\nfunc main() {}\n', language="go", change_type="modify")
        ],
    }
    mock_agents = MagicMock()

    # Mock shutil.which and subprocess.run to simulate go build passing
    with patch("shutil.which", return_value="go"), patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0, stdout="go build ok", stderr="")
        res = build_node(state, mock_agents)

    assert "build_result" in res
    assert res["build_result"].status == BuildStatus.SUCCESS


def test_git_sandbox_creation_and_cleanup(tmp_path):
    """
    Verify GitWorktreeSandbox creates an isolated shadow workspace and promotes changes cleanly.
    """
    base_file = tmp_path / "service.py"
    base_file.write_text("INITIAL = True\n", encoding="utf-8")

    sandbox_env = GitWorktreeSandbox.create_sandbox(str(tmp_path), "TICKET-SB-1")
    assert sandbox_env.sandbox_path.exists()
    assert (sandbox_env.sandbox_path / "service.py").read_text(encoding="utf-8") == "INITIAL = True\n"

    # Mutate only the sandbox
    (sandbox_env.sandbox_path / "service.py").write_text("INITIAL = False\nUPDATED = True\n", encoding="utf-8")
    # Base workspace must still be initial!
    assert base_file.read_text(encoding="utf-8") == "INITIAL = True\n"

    # Promote
    success = GitWorktreeSandbox.apply_sandbox_to_original(sandbox_env, ["service.py"])
    assert success is True
    assert base_file.read_text(encoding="utf-8") == "INITIAL = False\nUPDATED = True\n"

    # Cleanup
    GitWorktreeSandbox.cleanup_sandbox(sandbox_env)
    assert not sandbox_env.sandbox_path.exists()
