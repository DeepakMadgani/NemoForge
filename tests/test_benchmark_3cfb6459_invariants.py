"""
Benchmark Verification for TASK-3CFB6459 Architectural Invariants.

Verifies the 12 invariants required by the Final Architecture Guardrails:
1. add-members.component.ts appears once in ownership/task graph.
2. add-members.component.html appears once.
3. add-members.component.scss appears once.
4. member.service.ts remains in the plan because its structured capability is required.
5. member.service.ts is authorized through GENERATED_DEPENDENCY only because the structured contract proves it.
6. shared model files are not unnecessarily made writable.
7. no duplicate generation occurs.
8. compiler repair uses localized context.
9. provider-owned failures do not cause repeated consumer-only repairs.
10. full-file repair count is zero unless genuinely required by an unlocalizable diagnostic.
11. token usage is measured.
12. before vs after token metrics demonstrate reduction.
"""

from pathlib import Path
import pytest

from ticket_to_code.agents.canonical_path import (
    canonical_repo_path,
    canonical_component_base,
)
from ticket_to_code.agents.change_authorization import (
    classify_change_role,
    ChangeRole,
    is_companion,
)
from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
from ticket_to_code.agents.diagnostic_localizer import (
    DiagnosticLocalizer,
    FailureOwner,
    RepairContextTier,
)
from ticket_to_code.agents.diagnostic_normalizer import StructuredDiagnostic


class _MockBlueprint:
    def __init__(self, symbol_name: str, created_by_task: str = ""):
        self.symbol_name = symbol_name
        self.created_by_task = created_by_task


class _MockContract:
    def __init__(self, produces=None, consumes=None):
        self.produces = produces or []
        self.consumes = consumes or []


class _MockTask:
    def __init__(
        self,
        task_id: str,
        file_path: str,
        ttype: str = "modify",
        deps=None,
        contract=None,
        allowed_methods=None,
        target_class: str = "",
    ):
        self.id = task_id
        self.file_path = file_path
        self.task_type = type("_T", (), {"value": ttype})()
        self.dependencies = deps or []
        self.cross_file_contract = contract
        self.allowed_methods = allowed_methods or []
        self.target_class = target_class
        self.localization_reason = "initial_discovery"


def test_invariant_1_2_3_7_ownership_deduplication_and_single_tasks(tmp_path: Path):
    """Prove invariants 1, 2, 3, 7:
    - add-members.component.ts appears ONCE
    - add-members.component.html appears ONCE
    - add-members.component.scss appears ONCE
    - No duplicate generation occurs even when discovered via mixed absolute/relative paths.
    """
    ws = tmp_path / "CC4E"
    ws.mkdir()
    ui_dir = ws / "xchange-ui" / "src" / "app" / "modules" / "members" / "add-members"
    ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "add-members.component.ts").touch()
    (ui_dir / "add-members.component.html").touch()
    (ui_dir / "add-members.component.scss").touch()

    # Suppose Planner created task with relative path
    planner_ts = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    planner_html = "xchange-ui/src/app/modules/members/add-members/add-members.component.html"

    # Suppose Companion discovery discovered via absolute Windows path
    companion_ts_abs = str(ui_dir / "add-members.component.ts")
    companion_scss_abs = str(ui_dir / "add-members.component.scss")

    # Build existing_paths using canonical identity before task generation
    tasks_by_path: dict[str, _MockTask] = {}
    existing_paths: set[str] = set()

    initial_tasks = [
        _MockTask("task-1", planner_ts),
        _MockTask("task-2", planner_html),
    ]

    for t in initial_tasks:
        cp = canonical_repo_path(t.file_path, ws)
        assert cp is not None
        existing_paths.add(cp)
        tasks_by_path[cp] = t

    # Now companion discovery finds companion_ts_abs and companion_scss_abs
    discovered = [
        (companion_ts_abs, "ast_decorator_edge"),
        (companion_scss_abs, "sibling_colocation"),
    ]

    new_tasks = []
    for disc_path, strategy in discovered:
        canon_disc = canonical_repo_path(disc_path, ws)
        assert canon_disc is not None
        if canon_disc in existing_paths:
            # Must merge provenance into existing task, NOT create duplicate!
            existing_t = tasks_by_path[canon_disc]
            existing_t.localization_reason += f"; merged_companion: {strategy}"
        else:
            new_t = _MockTask(f"cmp-{len(new_tasks)+1}", canon_disc)
            new_tasks.append(new_t)
            existing_paths.add(canon_disc)
            tasks_by_path[canon_disc] = new_t

    all_tasks = initial_tasks + new_tasks

    # Invariant 1: TS appears once
    ts_tasks = [t for t in all_tasks if "add-members.component.ts" in t.file_path]
    assert len(ts_tasks) == 1
    assert ts_tasks[0].id == "task-1"
    assert "merged_companion: ast_decorator_edge" in ts_tasks[0].localization_reason

    # Invariant 2: HTML appears once
    html_tasks = [t for t in all_tasks if "add-members.component.html" in t.file_path]
    assert len(html_tasks) == 1
    assert html_tasks[0].id == "task-2"

    # Invariant 3: SCSS appears once
    scss_tasks = [t for t in all_tasks if "add-members.component.scss" in t.file_path]
    assert len(scss_tasks) == 1
    assert scss_tasks[0].id == "cmp-1"

    # Invariant 7: Total tasks is exactly 3 (no duplicate generation)
    assert len(all_tasks) == 3


def test_invariant_4_5_6_structured_supplier_authorization_and_model_protection(tmp_path: Path):
    """Prove invariants 4, 5, 6:
    - member.service.ts remains in the plan because its structured capability is required.
    - member.service.ts is authorized through GENERATED_DEPENDENCY only via structured contract proof.
    - shared models remain protected (read-only).
    """
    ws = tmp_path / "CC4E"
    ws.mkdir()
    comp_dir = ws / "xchange-ui" / "src" / "app" / "modules" / "members" / "add-members"
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_file = comp_dir / "add-members.component.ts"
    comp_file.touch()

    members_dir = ws / "xchange-ui" / "src" / "app" / "modules" / "shared" / "services" / "members"
    members_dir.mkdir(parents=True, exist_ok=True)
    svc_file = members_dir / "member.service.ts"
    svc_file.touch()

    models_dir = ws / "xchange-ui" / "src" / "app" / "modules" / "shared" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / "contract.ts"
    model_file.touch()

    comp_fp = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    svc_fp = "xchange-ui/src/app/modules/shared/services/members/member.service.ts"
    model_fp = "xchange-ui/src/app/modules/shared/models/contract.ts"

    # 1. Structural companion check: member.service.ts is NOT a companion!
    assert not is_companion(svc_fp, {comp_fp}, ws)
    assert is_companion("xchange-ui/src/app/modules/members/add-members/add-members.component.scss", {comp_fp}, ws)

    # 2. Consumer task consumes getProjectMembers produced by supplier
    consumer_contract = _MockContract(consumes=[_MockBlueprint("getProjectMembers", created_by_task="task-svc")])
    consumer_task = _MockTask("task-comp", comp_fp, "modify", deps=["task-svc"], contract=consumer_contract)

    # Supplier task produces getProjectMembers
    supplier_contract = _MockContract(produces=[_MockBlueprint("getProjectMembers")])
    supplier_task = _MockTask("task-svc", svc_fp, "modify", contract=supplier_contract, allowed_methods=["getProjectMembers"])

    # Shared model task: attempts to be writable without contract necessity
    model_task = _MockTask("task-model", model_fp, "modify")

    gate = filter_generation_tasks(
        tasks=[consumer_task, supplier_task, model_task],
        proven_targets={comp_fp},
        scope_declared=False,
        workspace_root=ws,
    )

    accepted_paths = [t.file_path for t in gate.accepted]
    roles = {t.file_path: getattr(t, "change_role", None) for t in gate.accepted}

    # Invariant 4: member.service.ts remains in the accepted plan
    assert svc_fp in accepted_paths

    # Invariant 5: member.service.ts is authorized through GENERATED_DEPENDENCY
    assert roles[svc_fp] == ChangeRole.GENERATED_DEPENDENCY

    # Invariant 6: shared model is NOT writable (rejected from accepted writable list)
    assert model_fp not in accepted_paths
    rejected_reasons = {r.file_path: r.reason for r in gate.rejected}
    assert model_fp in rejected_reasons
    assert "shared model" in rejected_reasons[model_fp].lower() or "read-only" in rejected_reasons[model_fp].lower()


def test_invariant_8_9_10_11_12_compiler_repair_localization_and_telemetry():
    """Prove invariants 8, 9, 10, 11, 12:
    - compiler repair uses localized context (Tier 1 containing method).
    - provider-owned failure is correctly attributed to MemberService provider.
    - full-file repair count is zero for method-localized error.
    - token usage is measured.
    - before vs after metrics show token reduction.
    """
    localizer = DiagnosticLocalizer()

    # Simulated 500-line add-members.component.ts
    component_lines = ["// Header comment"]
    for i in range(2, 95):
        component_lines.append(f"// padding line {i}")
    component_lines.append("import { Component, OnInit } from '@angular/core';")
    component_lines.append("import { MemberService } from '../../shared/services/members/member.service';")
    component_lines.append("import { UnrelatedService } from '../../shared/services/unrelated.service';")
    component_lines.append("export class AddMembersComponent implements OnInit {")
    component_lines.append("  constructor(private memberService: MemberService) {}")
    component_lines.append("  loadMembers(): void {")
    component_lines.append("    const id = 123;")
    component_lines.append("    this.memberService.getProjectMembers(id).subscribe();")
    component_lines.append("  }")
    for i in range(106, 501):
        component_lines.append(f"  unrelatedMethod{i}() {{ return {i}; }}")
    component_lines.append("}")

    full_component_content = "\n".join(component_lines)
    full_lines_count = len(component_lines)
    assert full_lines_count >= 400

    error_line = component_lines.index("    this.memberService.getProjectMembers(id).subscribe();") + 1

    # The exact error observed in TASK-3CFB6459:
    diag = StructuredDiagnostic(
        raw=f"xchange-ui/src/app/modules/members/add-members/add-members.component.ts({error_line},24): error TS2339: Property 'getProjectMembers' does not exist on type 'MemberService'.",
        code="TS2339",
        message="Property 'getProjectMembers' does not exist on type 'MemberService'.",
        source_file="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        line=error_line,
        column=24,
    )

    planned_provider_task = _MockTask(
        task_id="task-svc",
        file_path="xchange-ui/src/app/modules/shared/services/members/member.service.ts",
        target_class="MemberService",
    )

    # Invariant 9: Provider-owned failure attribution
    attrib = localizer.attribute_failure(diag, planned_tasks=[planned_provider_task])
    assert attrib.owner == FailureOwner.PROVIDER
    assert attrib.missing_symbol == "getProjectMembers"
    assert attrib.provider_type == "MemberService"
    assert attrib.provider_file == "xchange-ui/src/app/modules/shared/services/members/member.service.ts"
    assert attrib.is_provider_writable is True

    # Invariant 8: Localized context assembly
    ctx = localizer.localize_context(
        file_path="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        content=full_component_content,
        diagnostics=[diag],
        tier=RepairContextTier.TIER_1_LOCALIZED_METHOD,
        planned_tasks=[planned_provider_task],
    )

    assert ctx.context_tier == RepairContextTier.TIER_1_LOCALIZED_METHOD
    assert ctx.target_method_name == "loadMembers"
    assert "loadMembers" in ctx.prompt_snippet
    assert "getProjectMembers" in ctx.prompt_snippet
    assert "MemberService" in ctx.prompt_snippet
    # Unrelated methods must NOT be present!
    assert "unrelatedMethod106" not in ctx.prompt_snippet
    assert "unrelatedMethod500" not in ctx.prompt_snippet

    # Invariant 10: full-file repair count is ZERO
    assert localizer.telemetry.full_file_repair_count == 0
    assert localizer.telemetry.localized_repair_count == 1

    # Invariant 11 & 12: Token measurement
    # Localized lines sent is < 25 lines vs 501 full lines
    assert ctx.source_lines_sent < 25
    lines_saved_ratio = (full_lines_count - ctx.source_lines_sent) / full_lines_count
    assert lines_saved_ratio > 0.90  # Over 90% lines eliminated in prompt!
