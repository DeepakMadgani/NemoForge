"""Regression tests for Structured Change Authorization, Canonical Path Deduplication,
and Strict Whitelisting.
"""

from pathlib import Path
import pytest

from ticket_to_code.agents.change_authorization import (
    classify_change_role, ChangeRole,
)
from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
from ticket_to_code.models import CrossFileContract, SemanticBlueprint, BlueprintStatus


class _DummyBlueprint:
    def __init__(self, symbol_name: str, created_by_task: str = ""):
        self.symbol_name = symbol_name
        self.created_by_task = created_by_task


class _DummyContract:
    def __init__(self, produces=None, consumes=None):
        self.produces = produces or []
        self.consumes = consumes or []


class _DummyTask:
    def __init__(self, task_id: str, file_path: str, ttype: str = "modify", deps=None, contract=None, allowed_methods=None):
        self.id = task_id
        self.file_path = file_path
        self.task_type = type("_T", (), {"value": ttype})()
        self.dependencies = deps or []
        self.cross_file_contract = contract
        self.allowed_methods = allowed_methods or []
        self.selection_reason = ""


def test_structured_supplier_authorized_when_capability_matches(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    svc = ws / "xchange-ui" / "src" / "member.service.ts"
    svc.parent.mkdir(parents=True, exist_ok=True)
    svc.touch()

    # Consumer: add-members.component.ts (proven target)
    comp_fp = "xchange-ui/src/add-members.component.ts"
    svc_fp = "xchange-ui/src/member.service.ts"

    # Consumer consumes getProjectMembers produced by task-1
    consumer_contract = _DummyContract(consumes=[_DummyBlueprint("getProjectMembers", created_by_task="task-1")])
    consumer_task = _DummyTask("task-2", comp_fp, "modify", deps=["task-1"], contract=consumer_contract)

    # Supplier produces getProjectMembers
    supplier_contract = _DummyContract(produces=[_DummyBlueprint("getProjectMembers")])
    supplier_task = _DummyTask("task-1", svc_fp, "modify", contract=supplier_contract, allowed_methods=["getProjectMembers"])

    res = filter_generation_tasks(
        tasks=[consumer_task, supplier_task],
        proven_targets={comp_fp},
        scope_declared=False,
        workspace_root=ws,
    )

    accepted_paths = {t.file_path for t in res.accepted}
    assert comp_fp in accepted_paths
    assert svc_fp in accepted_paths
    assert not res.hard_block


def test_dependency_edge_alone_insufficient(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    svc = ws / "xchange-ui" / "src" / "member.service.ts"
    svc.parent.mkdir(parents=True, exist_ok=True)
    svc.touch()

    comp_fp = "xchange-ui/src/add-members.component.ts"
    svc_fp = "xchange-ui/src/member.service.ts"

    # Consumer has dependency on task-1 but NO capability match
    consumer_task = _DummyTask("task-2", comp_fp, "modify", deps=["task-1"])
    # Supplier produces an unrelated method
    supplier_contract = _DummyContract(produces=[_DummyBlueprint("unrelatedMethod")])
    supplier_task = _DummyTask("task-1", svc_fp, "modify", contract=supplier_contract)

    res = filter_generation_tasks(
        tasks=[consumer_task, supplier_task],
        proven_targets={comp_fp},
        scope_declared=False,
        workspace_root=ws,
    )

    accepted_paths = {t.file_path for t in res.accepted}
    rejected_paths = {r.file_path for r in res.rejected}
    assert comp_fp in accepted_paths
    assert svc_fp in rejected_paths
    assert any("read_only_reference" in r.reason for r in res.rejected if r.file_path == svc_fp)


def test_shared_service_allowed_as_supplier(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    shared_svc = ws / "src" / "app" / "modules" / "shared" / "services" / "member.service.ts"
    shared_svc.parent.mkdir(parents=True, exist_ok=True)
    shared_svc.touch()

    comp_fp = "src/app/modules/members/add-members/add-members.component.ts"
    shared_svc_fp = "src/app/modules/shared/services/member.service.ts"

    consumer_contract = _DummyContract(consumes=[_DummyBlueprint("getProjectMembers", created_by_task="task-1")])
    consumer_task = _DummyTask("task-2", comp_fp, "modify", deps=["task-1"], contract=consumer_contract)

    supplier_contract = _DummyContract(produces=[_DummyBlueprint("getProjectMembers")])
    supplier_task = _DummyTask("task-1", shared_svc_fp, "modify", contract=supplier_contract, allowed_methods=["getProjectMembers"])

    res = filter_generation_tasks(
        tasks=[consumer_task, supplier_task],
        proven_targets={comp_fp},
        scope_declared=False,
        workspace_root=ws,
    )

    accepted_paths = {t.file_path for t in res.accepted}
    assert comp_fp in accepted_paths
    assert shared_svc_fp in accepted_paths  # Shared service is not blocked by /shared/ marker!


def test_shared_model_remains_protected(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    shared_model = ws / "src" / "app" / "modules" / "shared" / "models" / "project.ts"
    shared_model.parent.mkdir(parents=True, exist_ok=True)
    shared_model.touch()

    comp_fp = "src/app/modules/members/add-members/add-members.component.ts"
    model_fp = "src/app/modules/shared/models/project.ts"

    consumer_contract = _DummyContract(consumes=[_DummyBlueprint("Project", created_by_task="task-1")])
    consumer_task = _DummyTask("task-2", comp_fp, "modify", deps=["task-1"], contract=consumer_contract)

    supplier_contract = _DummyContract(produces=[_DummyBlueprint("Project")])
    supplier_task = _DummyTask("task-1", model_fp, "modify", contract=supplier_contract)

    res = filter_generation_tasks(
        tasks=[consumer_task, supplier_task],
        proven_targets={comp_fp},
        scope_declared=False,
        workspace_root=ws,
    )

    rejected_paths = {r.file_path for r in res.rejected}
    assert model_fp in rejected_paths


def test_scope_declared_strict_whitelist_blocks_supplier(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    svc = ws / "xchange-ui" / "src" / "member.service.ts"
    svc.parent.mkdir(parents=True, exist_ok=True)
    svc.touch()

    comp_fp = "xchange-ui/src/add-members.component.ts"
    svc_fp = "xchange-ui/src/member.service.ts"

    consumer_contract = _DummyContract(consumes=[_DummyBlueprint("getProjectMembers", created_by_task="task-1")])
    consumer_task = _DummyTask("task-2", comp_fp, "modify", deps=["task-1"], contract=consumer_contract)

    supplier_contract = _DummyContract(produces=[_DummyBlueprint("getProjectMembers")])
    supplier_task = _DummyTask("task-1", svc_fp, "modify", contract=supplier_contract)

    # scope_declared=True with ONLY comp_fp in authorized_files
    res = filter_generation_tasks(
        tasks=[consumer_task, supplier_task],
        authorized_files={comp_fp},
        scope_declared=True,
        workspace_root=ws,
    )

    accepted_paths = {t.file_path for t in res.accepted}
    rejected_paths = {r.file_path for r in res.rejected}
    assert comp_fp in accepted_paths
    assert svc_fp in rejected_paths
    assert any("strict whitelist" in r.reason for r in res.rejected if r.file_path == svc_fp)
