"""
Integration and Unit Tests for:
1. Scope Isolation (Import != Writable: models & consumers stay READ_ONLY).
2. ChangeTarget Abstraction & Authorization.
3. Target-First Prompt Generation (structural skeleton + target block, omitting unrelated methods).
4. PatchGate: File Scope, Type Safety (unjustified breaking changes), and Cross-Artifact Consistency.
"""

import pytest
from pathlib import Path
from ticket_to_code.models import (
    ChangeTarget,
    DevelopmentTask,
    TaskType,
    ProgrammingLanguage,
    StructuredRequirements,
)
from ticket_to_code.agents.change_target import (
    ChangeTargetResolver,
    authorize_change_target,
)
from ticket_to_code.agents.patch_gate import (
    PatchGate,
    PatchScopeViolation,
    UnjustifiedBreakingTypeChangeError,
    CrossArtifactConsistencyError,
)
from ticket_to_code.workflow import (
    _find_component_companions,
    _make_companion_task,
)
from ticket_to_code.agents.code_generator import CodeGeneratorAgent


# ── Scenario A: Scope Isolation (Import != Writable) ──────────────────────────
def test_scenario_a_imported_models_are_read_only(tmp_path):
    """
    Given a component that imports shared models (project.ts, member.ts, contract.ts),
    verify that companion discovery classifies them as read-only reference context,
    NEVER as writable tasks.
    """
    comp_ts = tmp_path / "add-members.component.ts"
    comp_html = tmp_path / "add-members.component.html"
    comp_scss = tmp_path / "add-members.component.scss"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_proj = models_dir / "project.ts"
    model_mem = models_dir / "member.ts"

    comp_ts.write_text(
        """
        import { Project } from './models/project';
        import { Member } from './models/member';
        @Component({
          selector: 'se-add-members',
          templateUrl: './add-members.component.html',
          styleUrls: ['./add-members.component.scss']
        })
        export class AddMembersComponent {}
        """,
        encoding="utf-8"
    )
    comp_html.write_text("<div>Add Members</div>", encoding="utf-8")
    comp_scss.write_text(".box { display: flex; }", encoding="utf-8")
    model_proj.write_text("export interface Project { id: string; }", encoding="utf-8")
    model_mem.write_text("export interface Member { name: string; }", encoding="utf-8")

    companions = _find_component_companions(
        source_file=str(comp_ts),
        workspace_path=tmp_path,
        existing_paths=set(),
    )

    comp_map = dict(companions)

    # 1. Structural companions (HTML, SCSS) must be discovered as sibling/decorator
    assert "add-members.component.html" in comp_map
    assert comp_map["add-members.component.html"] in ("ast_decorator_edge", "sibling_colocation")

    norm_comp_map = {k.replace("\\", "/"): v for k, v in comp_map.items()}

    # 1. Structural companions (HTML, SCSS) must be discovered as sibling/decorator
    assert any("add-members.component.html" in k for k in norm_comp_map)
    assert any("add-members.component.scss" in k for k in norm_comp_map)

    # 2. Imported models must be discovered as model_import_reference
    assert any("models/project.ts" in k for k in norm_comp_map)
    proj_key = next(k for k in norm_comp_map if "models/project.ts" in k)
    assert norm_comp_map[proj_key] == "model_import_reference"
    
    assert any("models/member.ts" in k for k in norm_comp_map)
    mem_key = next(k for k in norm_comp_map if "models/member.ts" in k)
    assert norm_comp_map[mem_key] == "model_import_reference"

    # 3. Generating tasks via _make_companion_task:
    # HTML/SCSS must be TaskType.MODIFY
    task_html = _make_companion_task("add-members.component.html", 1, str(comp_ts), "ast_decorator_edge")
    assert task_html.task_type == TaskType.MODIFY

    # Models MUST BE TaskType.READ_ONLY
    task_proj = _make_companion_task("models/project.ts", 2, str(comp_ts), norm_comp_map[proj_key])
    assert task_proj.task_type == TaskType.READ_ONLY
    assert task_proj.ownership_type == "READ_ONLY"

    task_mem = _make_companion_task("models/member.ts", 3, str(comp_ts), norm_comp_map[mem_key])
    assert task_mem.task_type == TaskType.READ_ONLY
    assert task_mem.ownership_type == "READ_ONLY"


# ── Scenario B: Target-First Prompt Generation ───────────────────────────────
def test_scenario_b_target_first_prompt_omits_unrelated_methods():
    """
    Verify that when a ChangeTarget / target_method is present:
    1. The target method content is in the prompt.
    2. The structural skeleton is in the prompt.
    3. Unrelated methods are OMITTED from the verbatim editing section.
    """
    large_ts_content = """
import { Injectable } from '@angular/core';

@Injectable()
export class MemberService {
  unrelatedMethodAlpha(x: number): number {
    console.log("This is alpha logic that should be collapsed in skeleton");
    return x * 2;
  }

  onUserSelected(userId: string): void {
    console.log("TARGET METHOD: checking user membership");
    this.checkMembership(userId);
  }

  unrelatedMethodOmega(y: string): string {
    console.log("This is omega logic that should also be collapsed");
    return y.toUpperCase();
  }
}
""" + "\n// PAD" * 5000
    task = DevelopmentTask(
        id="task-1",
        title="Check membership in onUserSelected",
        description="Update onUserSelected to verify project membership",
        file_path="src/app/member.service.ts",
        task_type=TaskType.MODIFY,
        language=ProgrammingLanguage.TYPESCRIPT,
        target_method="onUserSelected",
        allowed_methods=["onUserSelected"],
    )

    reqs = StructuredRequirements(
        ticket_id="TICKET-1",
        title="Test",
        description="Test",
        functional_requirements=[],
        technical_requirements=[],
        edge_cases=[],
        acceptance_tests=[],
        affected_components=[],
    )
    agent = CodeGeneratorAgent.__new__(CodeGeneratorAgent)

    prompt = agent._build_user_prompt(
        task=task,
        requirements=reqs,
        context=[],
        existing_content=large_ts_content,
    )

    # 1. Target method MUST be present in verbatim section
    assert "onUserSelected" in prompt
    assert 'console.log("TARGET METHOD: checking user membership");' in prompt

    # 2. Structural skeleton MUST be present
    assert "=== FILE STRUCTURE" in prompt
    assert "=== EXACT SOURCE" in prompt

    # 3. Unrelated method body MUST NOT be in verbatim edit source
    verbatim_section = prompt.split("=== EXACT SOURCE")[-1]
    assert 'console.log("This is alpha logic' not in verbatim_section
    assert 'console.log("This is omega logic' not in verbatim_section


# ── Scenario C & D: Type Safety (Unjustified Breaking Changes) ───────────────
def test_scenario_c_unjustified_breaking_type_change_rejected():
    """
    Adding a mandatory (non-optional) field to a shared model without
    migration evidence must be rejected by PatchGate.
    """
    patch = """
<<<<<<< SEARCH
export interface ProjectMember {
  organizationName?: string;
}
=======
export interface ProjectMember {
  isProjectMember: boolean;
  organizationName?: string;
}
>>>>>>> REPLACE
"""
    ok, reason = PatchGate.validate_type_changes(
        file_path="src/app/modules/shared/models/project.ts",
        patch_content=patch,
        has_migration_evidence=False,
    )
    assert not ok
    assert "Unjustified breaking type change" in reason
    assert "isProjectMember" in reason


def test_scenario_d_optional_type_change_or_migration_evidence_accepted():
    """
    Adding an optional field ('?') or having explicit migration evidence
    must be accepted by PatchGate.
    """
    # 1. Optional field passes without migration evidence
    patch_optional = """
<<<<<<< SEARCH
export interface ProjectMember {
  organizationName?: string;
}
=======
export interface ProjectMember {
  isProjectMember?: boolean;
  organizationName?: string;
}
>>>>>>> REPLACE
"""
    ok, _ = PatchGate.validate_type_changes(
        file_path="src/app/modules/shared/models/project.ts",
        patch_content=patch_optional,
        has_migration_evidence=False,
    )
    assert ok

    # 2. Required field with migration evidence passes
    patch_required = """
<<<<<<< SEARCH
export interface ProjectMember {
  organizationName?: string;
}
=======
export interface ProjectMember {
  isProjectMember: boolean;
  organizationName?: string;
}
>>>>>>> REPLACE
"""
    ok2, _ = PatchGate.validate_type_changes(
        file_path="src/app/modules/shared/models/project.ts",
        patch_content=patch_required,
        has_migration_evidence=True,
    )
    assert ok2


# ── Scenario E: Cross-Artifact Consistency Gate ──────────────────────────────
def test_scenario_e_cross_artifact_consistency_angular():
    """
    Verify that an Angular template referencing an undeclared controller property
    is caught and rejected before application.
    """
    controller_ts = """
    export class AddMembersComponent {
      isExistingProjectMember = false;
      existingOrganizationName: string;
    }
    """

    # HTML uses mismatched variable names (the exact bug we diagnosed earlier!)
    patch_html_mismatched = """
    <div *ngIf="selectedUserIsExistingProjectMember">
      {{ selectedUserExistingOrganizationName }}
    </div>
    """

    ok, reason = PatchGate.validate_cross_artifact_consistency(
        file_path="src/app/modules/members/add-members/add-members.component.html",
        patch_content=patch_html_mismatched,
        sibling_content=controller_ts,
        sibling_path="add-members.component.ts",
    )
    assert not ok
    assert "Cross-artifact inconsistency" in reason
    assert ("selectedUserExistingOrganizationName" in reason or "selectedUserIsExistingProjectMember" in reason)


def test_scenario_e_cross_artifact_consistency_matching_passes():
    """
    When template variables match controller declarations, it passes.
    """
    controller_ts = """
    export class AddMembersComponent {
      isExistingProjectMember = false;
      existingOrganizationName: string;
    }
    """

    patch_html_matching = """
    <div *ngIf="isExistingProjectMember">
      {{ existingOrganizationName }}
    </div>
    """

    ok, _ = PatchGate.validate_cross_artifact_consistency(
        file_path="src/app/modules/members/add-members/add-members.component.html",
        patch_content=patch_html_matching,
        sibling_content=controller_ts,
    )
    assert ok


# ── Scenario F: Final Security Boundary (Unauthorized File Patch Rejection) ──
def test_scenario_f_unauthorized_imported_file_patch_rejected():
    """
    Verify that at the final security boundary:
    If authorized_files contains only component.ts,
    a patch attempting to modify project.ts is strictly REJECTED.
    """
    authorized_files = {"src/app/modules/members/add-members/add-members.component.ts"}

    # 1. Authorized file patch is accepted
    ok_comp, _ = PatchGate.validate_file_authorization(
        "src/app/modules/members/add-members/add-members.component.ts",
        authorized_files,
    )
    assert ok_comp

    # 2. Unauthorized imported model patch is REJECTED
    ok_model, reason = PatchGate.validate_file_authorization(
        "src/app/modules/shared/models/project.ts",
        authorized_files,
    )
    assert not ok_model
    assert "NOT in authorized writable files list" in reason
