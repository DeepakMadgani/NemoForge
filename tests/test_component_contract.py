"""Tests for Step 3: Component Controller to Template Contract Enforcement.

Verifies:
1. ComponentContract extracts public properties and methods from Angular TypeScript controllers.
2. TemplateContractValidator deterministically flags hallucinated properties
   (e.g., selectedUser.isExistingProjectMember when controller has isExistingMember).
3. TemplateContractValidator approves compliant templates that bind to actual controller fields.
4. Rendered prompt block provides explicit controller contract for template generation.
5. Live validation against real CC4E add-members.component.ts if present.
"""

from pathlib import Path
import pytest

from ticket_to_code.intelligence.contracts.component_contract import (
    ComponentContract,
    TemplateContractValidator,
)


@pytest.fixture
def synthetic_add_members_controller():
    return """
import { Component, OnInit } from '@angular/core';
import { MemberService } from '../../../../shared/services/members/member.service';

@Component({
  selector: 'app-add-members',
  templateUrl: './add-members.component.html',
  styleUrls: ['./add-members.component.scss']
})
export class AddMembersComponent implements OnInit {
  public isExistingMember: boolean = false;
  public existingOrganizationName: string = '';
  public selectedUser: any = null;
  public userExistError: boolean = false;

  constructor(private memberService: MemberService) {}

  ngOnInit(): void {}

  public onUserSelect(searchData: any): void {
    this.selectedUser = searchData;
  }

  public checkMemberExists(email: string): void {
    // checks existence
  }
}
"""


def test_component_contract_extraction(synthetic_add_members_controller):
    contract = ComponentContract.extract_from_ts(
        synthetic_add_members_controller,
        file_path="src/app/components/add-members/add-members.component.ts"
    )

    assert contract.class_name == "AddMembersComponent"
    assert "isExistingMember" in contract.properties
    assert "existingOrganizationName" in contract.properties
    assert "selectedUser" in contract.properties
    assert "userExistError" in contract.properties

    # Methods extracted (lifecycle methods skipped)
    assert "onUserSelect" in contract.methods
    assert "checkMemberExists" in contract.methods
    assert "ngOnInit" not in contract.methods


def test_template_validator_catches_hallucinated_property(synthetic_add_members_controller):
    """CRITICAL PRODUCTION FAILURE REPRODUCTION:
    The template attempted to bind to `selectedUser.isExistingProjectMember`,
    which does not exist on the user object or the controller.
    The controller exposes `isExistingMember`.
    """
    contract = ComponentContract.extract_from_ts(
        synthetic_add_members_controller,
        file_path="src/app/components/add-members/add-members.component.ts"
    )

    # Hallucinated template binding
    bad_template = """
    <div class="user-item">
      <span *ngIf="selectedUser.isExistingProjectMember" class="text-danger">
        User already exists in project!
      </span>
      <span>{{ selectedUser.name }}</span>
    </div>
    """

    violations = TemplateContractValidator.validate(bad_template, contract)
    assert len(violations) > 0, "Validator must reject hallucinated property 'isExistingProjectMember'"
    assert any("isExistingMember" in v for v in violations), "Validator should suggest 'isExistingMember'"


def test_template_validator_approves_correct_bindings(synthetic_add_members_controller):
    contract = ComponentContract.extract_from_ts(
        synthetic_add_members_controller,
        file_path="src/app/components/add-members/add-members.component.ts"
    )

    good_template = """
    <div class="user-item">
      <span *ngIf="isExistingMember" class="text-danger">
        User already exists in {{ existingOrganizationName }}!
      </span>
      <span>{{ selectedUser.name }}</span>
    </div>
    """

    violations = TemplateContractValidator.validate(good_template, contract)
    assert len(violations) == 0, f"Compliant template should have 0 violations, got: {violations}"


def test_template_validator_catches_bare_hallucinated_property():
    """Verify that bare identifiers like selectedUserIsExistingProjectMember
    are caught and mapped to isExistingProjectMember on the controller.
    """
    ts_code = """
    export class AddMembersComponent implements OnInit {
      isExistingProjectMember: boolean = false;
      existingOrganizationName: string;
      onUserSelect(data) {}
    }
    """
    contract = ComponentContract.extract_from_ts(ts_code, file_path="add-members.component.ts")

    bad_template = """
    <ng-container *ngIf="selectedUserIsExistingProjectMember; else orgDropdown">
      <div class="readonly">{{ existingProjectMemberOrganizationName }}</div>
    </ng-container>
    <ng-template #orgDropdown></ng-template>
    """

    violations = TemplateContractValidator.validate(bad_template, contract)
    assert len(violations) == 2, f"Expected 2 violations, got: {violations}"
    assert any("selectedUserIsExistingProjectMember" in v and "isExistingProjectMember" in v for v in violations)
    assert any("existingProjectMemberOrganizationName" in v and "existingOrganizationName" in v for v in violations)



def test_component_contract_prompt_block_rendering(synthetic_add_members_controller):
    contract = ComponentContract.extract_from_ts(
        synthetic_add_members_controller,
        file_path="src/app/components/add-members/add-members.component.ts"
    )

    prompt = contract.render_prompt_block()
    assert "CONTROLLER CONTRACT FOR TEMPLATE: AddMembersComponent" in prompt
    assert "isExistingMember" in prompt
    assert "existingOrganizationName" in prompt
    assert "onUserSelect" in prompt
    assert "Do NOT invent phantom sub-properties" in prompt  # Universal negative warning instructions


def test_live_cc4e_add_members_component_contract():
    cc4e_path = Path(r"C:\CC4E\xchange-ui\src\app\modules\members\add-members\add-members.component.ts")
    if not cc4e_path.exists():
        pytest.skip("CC4E add-members.component.ts not found")

    content = cc4e_path.read_text(encoding="utf-8", errors="ignore")
    contract = ComponentContract.extract_from_ts(content, file_path=str(cc4e_path))
    assert contract.class_name == "AddMembersComponent"
    assert len(contract.properties) > 0
    assert len(contract.methods) > 0


def test_patch_validator_catches_template_contract_violations(tmp_path: Path):
    from ticket_to_code.models import DevelopmentTask, TaskType
    from ticket_to_code.agents.code_generator import GeneratedCode, PatchValidator

    comp_ts = tmp_path / "add-members.component.ts"
    comp_ts.write_text(
        "export class AddMembersComponent {\n"
        "  public isExistingMember: boolean = false;\n"
        "}\n",
        encoding="utf-8"
    )

    comp_html = tmp_path / "add-members.component.html"
    task = DevelopmentTask(
        id="task-1",
        title="Update template",
        description="Template changes",
        file_path=str(comp_html),
        task_type=TaskType.MODIFY,
        language="html",
    )

    bad_code = GeneratedCode(
        task_id="task-1",
        file_path=str(comp_html),
        content='<div *ngIf="selectedUser.isExistingProjectMember">Error</div>',
        language="html",
        change_type="modify",
    )

    validator = PatchValidator()
    result = validator.validate(bad_code, task)
    assert not result.passed
    assert any("TEMPLATE CONTRACT VIOLATION" in v for v in result.violations)

    good_code = GeneratedCode(
        task_id="task-1",
        file_path=str(comp_html),
        content='<div *ngIf="isExistingMember">Error</div>',
        language="html",
        change_type="modify",
    )
    result_good = validator.validate(good_code, task)
    assert result_good.passed


def test_patch_validator_catches_ts_array_shape_violations(tmp_path: Path):
    from ticket_to_code.models import DevelopmentTask, TaskType
    from ticket_to_code.agents.code_generator import GeneratedCode, PatchValidator

    comp_ts = tmp_path / "add-members.component.ts"
    task = DevelopmentTask(
        id="task-2",
        title="Update controller",
        description="Controller changes",
        file_path=str(comp_ts),
        task_type=TaskType.MODIFY,
        language="typescript",
    )

    # WRONG — production failure
    bad_code = GeneratedCode(
        task_id="task-2",
        file_path=str(comp_ts),
        content="const filter = { email: { eq: searchData } };",
        language="typescript",
        change_type="modify",
    )

    validator = PatchValidator()
    result = validator.validate(bad_code, task)
    assert not result.passed
    assert any("TYPE CONTRACT VIOLATION" in v for v in result.violations)

    good_code = GeneratedCode(
        task_id="task-2",
        file_path=str(comp_ts),
        content="const filter = { email: [{ eq: searchData }] };",
        language="typescript",
        change_type="modify",
    )
    result_good = validator.validate(good_code, task)
    assert result_good.passed


def test_code_generator_path_scoping_no_unbound_local(tmp_path: Path, monkeypatch):
    from ticket_to_code.models import DevelopmentTask, TaskType
    from ticket_to_code.agents.code_generator import CodeGeneratorAgent

    agent = CodeGeneratorAgent.__new__(CodeGeneratorAgent)
    agent.llm = "dummy_llm"
    agent._current_allowed_files = set()
    agent._current_readonly_files = set()
    agent._workspace_path = str(tmp_path)

    task = DevelopmentTask(
        id="task-test",
        title="Test task",
        description="Testing",
        file_path="src/app/foo.component.ts",
        task_type=TaskType.MODIFY,
        language="typescript",
    )

    monkeypatch.setattr(agent, "_build_generation_prompt", lambda *args, **kwargs: "sys")
    monkeypatch.setattr(agent, "_build_user_prompt", lambda *args, **kwargs: "user")

    class DummyMsg:
        content = "<<<AVIATOR_CODE_START>>>\nexport class Foo {}\n<<<AVIATOR_CODE_END>>>"
        response_metadata = {}

    monkeypatch.setattr("ticket_to_code.agents.code_generator.llm_invoke", lambda *args, **kwargs: DummyMsg())

    class DummyReq:
        pass

    result = agent.generate_code(task=task, requirements=DummyReq(), context=[])
    assert result is not None
    assert result.file_path == "src/app/foo.component.ts"


def test_template_validator_handles_advanced_syntax_and_locals():
    """Verify that TemplateContractValidator recognizes two-way bindings, 'as' aliases,
    multi-let loops, @for control flow, and global builtins without false positives.
    """
    ts_code = """
    export class UserDashboardComponent {
      users: any[] = [];
      searchQuery: string = '';
      score: number = 95.5;

      saveUser(user: any): void {}
    }
    """
    contract = ComponentContract.extract_from_ts(ts_code, file_path="dashboard.component.ts")

    # Template using:
    # - [(ngModel)]="searchQuery" (two-way binding)
    # - *ngIf="users as userList" ('as' alias)
    # - *ngFor="let u of userList; let idx = index; let isLast = last" (multi-let)
    # - Math.round(score) (global builtin)
    # - @for (u of users; track u.id; let i = $index) (Angular 17+ control flow)
    # - @let total = users.length (Angular 18+ @let)
    advanced_valid_template = """
    <div>
      <input [(ngModel)]="searchQuery" placeholder="Search..." />
      <div *ngIf="users as userList">
        <div *ngFor="let u of userList; let idx = index; let isLast = last">
          <span>#{{ idx + 1 }}: {{ u.name }}</span>
          <span *ngIf="isLast">Total: {{ Math.round(score) }}</span>
          <button (click)="saveUser(u)">Save</button>
        </div>
      </div>
      @for (u of users; track u.id; let i = $index) {
        <span>{{ i }}: {{ u.title }}</span>
      }
      @let total = users.length;
      <span>Count: {{ total }}</span>
    </div>
    """

    violations = TemplateContractValidator.validate(advanced_valid_template, contract)
    assert len(violations) == 0, f"Expected 0 violations for valid advanced template, got: {violations}"

    # Invalid template: hallucinated property inside [(ngModel)]
    invalid_template = """
    <input [(ngModel)]="hallucinatedFilter" />
    """
    bad_violations = TemplateContractValidator.validate(invalid_template, contract)
    assert len(bad_violations) == 1
    assert "hallucinatedFilter" in bad_violations[0]


def test_template_validator_baseline_immunity_and_object_fields():
    """Verify that:
    1. Pre-existing template bindings in original_content are granted baseline immunity.
    2. Common object fields (.options, .placeholder, .length) on known controller objects
       are not falsely flagged as hallucinations.
    """
    ts_code = """
    export class AddMembersComponent {
      organizationItemSelect: any = {};
      displayedMembers: any[] = [];
      isExistingMember: boolean = false;
      existingOrganizationName: string = '';
    }
    """
    contract = ComponentContract.extract_from_ts(ts_code, file_path="add-members.component.ts")

    original_html = """
    <ot-item-select
      [placeholder]="organizationItemSelect.placeholder"
      [options]="organizationItemSelect.options">
    </ot-item-select>
    <div *ngIf="displayedMembers.length > 0"></div>
    """

    # Newly edited template adds valid isExistingMember check and keeps existing bindings
    new_html = """
    <ot-item-select
      *ngIf="!isExistingMember"
      [placeholder]="organizationItemSelect.placeholder"
      [options]="organizationItemSelect.options">
    </ot-item-select>
    <div *ngIf="isExistingMember">{{ existingOrganizationName }}</div>
    <div *ngIf="displayedMembers.length > 0"></div>
    """

    # With baseline immunity, 0 violations
    violations = TemplateContractValidator.validate(new_html, contract, original_content=original_html)
    assert len(violations) == 0, f"Expected 0 violations with baseline immunity, got: {violations}"

    # Even without original_content, common field builtins (.options, .placeholder, .length) on known objects are protected
    violations_no_orig = TemplateContractValidator.validate(new_html, contract)
    assert len(violations_no_orig) == 0, f"Expected 0 violations for protected object fields, got: {violations_no_orig}"





