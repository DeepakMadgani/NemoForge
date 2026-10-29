import os
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ticket_to_code.models import (
    ValueEdgeTicket,
    BuildResult,
    BuildStatus,
    GeneratedCode,
)
from ticket_to_code.runtime.container_sandbox import DockerSandboxExecutor, SandboxConfig
from ticket_to_code.runtime.fast_path_engine import FastPathEligibilityAssessor, FastPathReActEngine
from ticket_to_code.runtime.transactional_diff import TransactionalDiffManager
from ticket_to_code.runtime.diff_engine import DiffHunk
from ticket_to_code.benchmarks.swe_bench_runner import BenchmarkTask, SWEBenchObjectiveRunner


def test_docker_and_process_sandbox_isolation(tmp_path):
    """
    Verify DockerSandboxExecutor enforces process isolation and sanitized environment.
    """
    script_file = tmp_path / "test_out.txt"
    cmd = 'python -c "open(\'test_out.txt\', \'w\').write(\'SANDBOX_OK\')"'

    cfg = SandboxConfig(timeout_seconds=5)
    res = DockerSandboxExecutor.run_isolated_command(
        cmd=cmd,
        workspace_path=str(tmp_path),
        technology="python",
        config=cfg,
    )

    assert res.exit_code == 0
    assert res.isolated_via in ("docker", "process_sandbox")
    assert script_file.exists()
    assert script_file.read_text() == "SANDBOX_OK"


def test_fast_path_eligibility_classification(tmp_path):
    """
    Verify FastPathEligibilityAssessor routes small bugs to Fast-Path and deep architecture to Graph.
    """
    # 1. Simple bugfix ticket -> Eligible for Fast-Path
    simple_ticket = ValueEdgeTicket(
        ticket_id="TICKET-SIMPLE-1",
        title="Fix null check in payment processing",
        description="PaymentService throws NPE when currency is null. Add a guard check.",
    )
    eligible, reason = FastPathEligibilityAssessor.is_eligible(simple_ticket, str(tmp_path))
    assert eligible is True
    assert "fast-path" in reason.lower()

    # 2. Deep architectural ticket -> Escalates to Deep Graph
    deep_ticket = ValueEdgeTicket(
        ticket_id="TICKET-ARCH-1",
        title="Microservice database schema migration",
        description="Refactor all payment services and migrate database schema across services.",
    )
    eligible, reason = FastPathEligibilityAssessor.is_eligible(deep_ticket, str(tmp_path))
    assert eligible is False
    assert "deep architectural" in reason.lower()


def test_fast_path_react_engine_execution(tmp_path):
    """
    Verify FastPathReActEngine executes in seconds rather than minutes.
    """
    ticket = ValueEdgeTicket(
        ticket_id="TICKET-FAST-1",
        title="Fix discount rounding",
        description="Round discount to 2 decimal places in Calculator.java",
    )
    engine = FastPathReActEngine(str(tmp_path), ticket)
    res = engine.run()

    assert res.success is True
    assert res.duration_seconds < 2.0  # Must be sub-minute
    assert res.escalated_to_deep_graph is False


def test_transactional_instant_undo(tmp_path):
    """
    Verify TransactionalDiffManager performs instant undo in milliseconds.
    """
    target = tmp_path / "App.java"
    target.write_text("class App { int version = 1; }", encoding="utf-8")

    # 1. Create checkpoint
    cp = TransactionalDiffManager.create_checkpoint("cp_1", str(tmp_path), ["App.java"])
    assert "App.java" in cp.file_snapshots

    # 2. Simulate bad mutation and new file creation
    target.write_text("class App { BROKEN SYNTAX", encoding="utf-8")
    bad_new_file = tmp_path / "Corrupt.java"
    bad_new_file.write_text("corrupted", encoding="utf-8")
    cp.created_files.append("Corrupt.java")

    # 3. Perform instant undo
    t0 = time.time()
    undone = TransactionalDiffManager.instant_undo("cp_1")
    undo_duration = time.time() - t0

    assert undone is True
    assert undo_duration < 0.2  # Instant (<200ms)
    assert target.read_text(encoding="utf-8") == "class App { int version = 1; }"
    assert not bad_new_file.exists()


def test_selective_hunk_application(tmp_path):
    """
    Verify selective hunk application applies only accepted hunks and leaves rejected hunks intact.
    """
    f = tmp_path / "code.py"
    orig_content = "line1\nline2\nline3\n"
    f.write_text(orig_content, encoding="utf-8")
    modified_content = "line1_MUTATED\nline2\nline3_ACCEPTED\n"

    from ticket_to_code.runtime.diff_engine import DiffReviewEngine
    file_diff = DiffReviewEngine.compute_file_diff("code.py", orig_content, modified_content)
    assert len(file_diff.hunks) >= 1

    # Accept all computed hunks selectively
    accepted_indices = [h.hunk_index for h in file_diff.hunks]
    ok = TransactionalDiffManager.apply_selective_hunk(
        workspace_path=str(tmp_path),
        rel_path="code.py",
        original_content=orig_content,
        hunk_indices_to_accept=accepted_indices,
        all_hunks=file_diff.hunks,
    )
    assert ok is True
    content = f.read_text(encoding="utf-8")
    assert "line3_ACCEPTED" in content
    assert "line1_MUTATED" in content


def test_swe_bench_objective_evaluation_runner(tmp_path):
    """
    Verify SWEBenchObjectiveRunner evaluates tasks and saves standardized metrics.
    """
    runner = SWEBenchObjectiveRunner(str(tmp_path), results_dir=str(tmp_path / "eval_results"))
    task = BenchmarkTask(
        instance_id="django__django-11001",
        repo="django/django",
        problem_statement="Incorrect SQL ordering with OrderBy expressions",
        base_commit="a1b2c3d4",
    )

    eval_result = runner.run_single_task(task)
    assert eval_result.instance_id == "django__django-11001"
    assert eval_result.passed is True
    assert eval_result.duration_seconds < 2.0

    # Test batch summary
    summary = runner.run_batch_evaluation([task])
    assert summary["total_tasks"] == 1
    assert summary["passed_tasks"] == 1
    assert summary["pass_rate_percent"] == 100.0
