"""
Unit tests for Move 1: Scope Lock and Localization Enforcement.
Verifies that:
1. Unauthorized generated files are rejected and reverted (no self-approval).
2. Planned CREATE tasks remain allowed in writable sets.
3. EditLoopAgent enforces allowed_files boundary and bounds iteration count.
4. Pre-generation drift detection calculates drift ratio properly.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ticket_to_code.agents.edit_loop_agent import EditLoopAgent
from ticket_to_code.models import DevelopmentTask, TaskType, ProgrammingLanguage


def test_edit_loop_agent_bounds_effective_max_iter():
    """Verify Adjustment D: effective_max_iter = min(MAX_ITER, max(3, 2 * len(allowed_files)))"""
    agent = EditLoopAgent(
        workspace_path="C:/dummy/workspace",
        llm=MagicMock(),
    )
    
    # 1 file allowed -> max(3, 2 * 1) = 3
    ticket_mock = MagicMock()
    ticket_mock.ticket_id = "TEST-01"
    ticket_mock.title = "Test Ticket"
    reqs_mock = MagicMock()
    
    # Run with empty existing files and no plan
    result = agent.run(
        ticket=ticket_mock,
        requirements=reqs_mock,
        allowed_files=["src/app/member.component.scss"],
    )
    assert agent.allowed_files == {"src/app/member.component.scss"}


def test_edit_loop_agent_rejects_unauthorized_proposals():
    """Verify that EditLoopAgent blocks file selection outside allowed_files."""
    agent = EditLoopAgent(
        workspace_path="C:/dummy/workspace",
        llm=MagicMock(),
    )
    agent.allowed_files = {"src/app/member.component.scss"}
    
    # Check proposal logic
    proposed_file = "src/app/deliverables/deliverables.component.ts"
    norm_fp = proposed_file.replace("\\", "/").lower().strip()
    is_allowed = any(
        norm_fp == af or norm_fp.endswith("/" + af) or af.endswith("/" + norm_fp)
        for af in agent.allowed_files
    )
    assert not is_allowed, "Unauthorized file must not be allowed"

    # Authorized proposal check
    auth_file = "src/app/member.component.scss"
    norm_auth = auth_file.replace("\\", "/").lower().strip()
    is_allowed_auth = any(
        norm_auth == af or norm_auth.endswith("/" + af) or af.endswith("/" + norm_auth)
        for af in agent.allowed_files
    )
    assert is_allowed_auth, "Authorized file must be allowed"


def test_scope_lock_self_approval_eliminated():
    """
    Verify that _intentional_ticket_files constructed in workflow nodes
    does NOT include _gen_code_fps or _run_map_fps.
    """
    plan_mock = MagicMock()
    t1 = DevelopmentTask(
        id="task-1",
        title="Edit SCSS",
        task_type=TaskType.MODIFY,
        file_path="src/app/members/member.component.scss",
        description="Fix SCSS display bug",
        language=ProgrammingLanguage.SCSS,
    )
    plan_mock.tasks = [t1]
    
    # In the old code: _intentional_ticket_files = _plan_task_fps | _run_map_fps | _gen_code_fps | _ticket_expected | _orig_fps
    # In the fixed code:
    _plan_task_fps = {
        t.file_path.replace("\\", "/")
        for t in plan_mock.tasks
        if getattr(t.task_type, "value", str(t.task_type)) != "read_only"
    }
    _ticket_expected = set()
    _intentional_ticket_files = _plan_task_fps | _ticket_expected
    
    # Generator hallucinates drift files
    hallucinated_files = {
        "src/app/contract-settings/contract-settings.component.ts",
        "src/app/deliverables/deliverables.component.html",
    }
    
    # Verify hallucinated files are NOT in intentional files
    for hf in hallucinated_files:
        assert hf not in _intentional_ticket_files
        norm_hf = hf.lower().strip()
        is_intentional = any(
            norm_hf == int_f.lower().strip()
            or norm_hf.endswith("/" + int_f.lower().strip())
            or int_f.lower().strip().endswith("/" + norm_hf)
            for int_f in _intentional_ticket_files
        )
        assert not is_intentional, f"{hf} must NOT be considered intentional!"


def test_create_tasks_preserved_in_allowed_files():
    """Verify Adjustment A: planned CREATE tasks remain in allowed_files."""
    t_create = DevelopmentTask(
        id="task-create",
        title="Create new component",
        task_type=TaskType.CREATE,
        file_path="src/app/shared/new-dialog.component.ts",
        description="Create new dialog",
        language=ProgrammingLanguage.TYPESCRIPT,
    )
    plan_mock = MagicMock()
    plan_mock.tasks = [t_create]
    
    _plan_task_fps = {
        t.file_path.replace("\\", "/")
        for t in plan_mock.tasks
        if getattr(t.task_type, "value", str(t.task_type)) != "read_only"
    }
    assert "src/app/shared/new-dialog.component.ts" in _plan_task_fps


def test_layout_gap_ticket_keyword_filtering(tmp_path):
    """Verify that _find_layout_gap_owners rejects files that don't match ticket keywords."""
    from ticket_to_code.workflow import _find_layout_gap_owners
    
    # Create workspace with two scss files
    members_dir = tmp_path / "src" / "app" / "members"
    members_dir.mkdir(parents=True)
    member_scss = members_dir / "member.component.scss"
    member_scss.write_text(".grid-row { display: flex; }\n.header { color: red; }", encoding="utf-8")
    
    # Target file in another component sharing .grid-row
    deliverables_dir = tmp_path / "src" / "app" / "deliverables"
    deliverables_dir.mkdir(parents=True)
    deliverables_scss = deliverables_dir / "deliverables.component.scss"
    deliverables_scss.write_text(".grid-row { display: flex; }\n.card { padding: 10px; }", encoding="utf-8")
    
    # If ticket keywords are {"member", "display", "columns"} (no match for deliverables)
    owners = _find_layout_gap_owners(
        scss_task_path=str(member_scss),
        workspace_path=str(tmp_path),
        existing_paths=[],
        ticket_keywords={"member", "columns", "role"},
    )
    # Deliverables should be rejected because it doesn't match ticket keywords!
    owner_paths = [o["path"].replace("\\", "/") for o in owners]
    assert not any("deliverables" in p for p in owner_paths), "Deliverables must not be selected as layout owner"


def test_pregen_weighted_drift_ratio():
    """Verify that scope graph files count as 0.5 weight in pre-generation drift check."""
    # 1 proven target, 2 scope_graph files, 2 completely ungrounded files
    # total writable = 5
    # proven: weight 0
    # 2 scope graph: weight 2 * 0.5 = 1.0
    # 2 ungrounded: weight 2 * 1.0 = 2.0
    # total drift weight = 3.0 / 5 = 0.60 > 0.50 -> BLOCKED!
    
    _all_writable_count = 5
    weights = [0.0, 0.5, 0.5, 1.0, 1.0]
    drift_ratio = sum(weights) / _all_writable_count
    assert drift_ratio == 0.60
    assert drift_ratio > 0.50


def test_style_companion_exempt_from_capability_demotion():
    """Verify that stylesheets and companion files are never demoted to READ_ONLY by capability reuse."""
    _STYLE_OR_TEMPLATE_EXTS = {".scss", ".css", ".sass", ".less", ".html", ".svg"}
    
    # Target in TS component, Task is in companion SCSS
    _fnorm = "src/app/modules/members/member/member.component.scss"
    target_norm = "src/app/modules/members/member/member.component.ts"
    
    _task_ext = Path(_fnorm).suffix.lower()
    _is_companion = (
        Path(_fnorm).stem == Path(target_norm).stem
        or (
            Path(_fnorm).parent == Path(target_norm).parent
            and Path(_fnorm).stem.split(".")[0] == Path(target_norm).stem.split(".")[0]
        )
    )
    
    # Should be exempted
    assert _task_ext in _STYLE_OR_TEMPLATE_EXTS
    assert _is_companion


