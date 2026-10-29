from ticket_to_code.agents.patch_gate import PatchGate
from ticket_to_code.agents.template_expression_parser import TemplateExpressionParser


def test_angular_microsyntax_else_then_not_treated_as_controller_member():
    refs = TemplateExpressionParser.extract_references("dmember.isExistingProjectMember; else editableOrg")
    assert len(refs) == 1
    assert refs[0].receiver == "dmember"
    assert refs[0].member == "isExistingProjectMember"


def test_patch_gate_accepts_ngif_with_else_template_ref():
    html = """
<ng-container *ngIf="dmember.isExistingProjectMember; else editableOrg">
  <span>{{ dmember.organization }}</span>
</ng-container>
<ng-template #editableOrg>
  <ot-item-select></ot-item-select>
</ng-template>
"""
    ts = """
export class AddMembersComponent {
  displayedMembers: any[] = [];
}
"""
    ok, msg = PatchGate.validate_cross_artifact_consistency(
        "src/app/add-members.component.html",
        html,
        sibling_content=ts,
        sibling_path="src/app/add-members.component.ts"
    )
    assert ok is True, f"Failed: {msg}"


def test_patch_gate_still_flags_real_missing_controller_property():
    html = """
<div *ngIf="missingPropertyOnController">
  <span>Hello</span>
</div>
"""
    ts = """
export class AddMembersComponent {
  displayedMembers: any[] = [];
}
"""
    ok, msg = PatchGate.validate_cross_artifact_consistency(
        "src/app/add-members.component.html",
        html,
        sibling_content=ts,
        sibling_path="src/app/add-members.component.ts"
    )
    assert ok is False
    assert "missingPropertyOnController" in msg


def test_patch_gate_proportionality_allows_fullstack_contract_expansion():
    """Verify that PatchGate permits legitimate 60-line contract expansions on DTOs/templates for fullstack tickets."""
    search_block = "  public IssueCount(long total) { this.total = total; }"
    replace_block = "\n".join(f"  private long count{i};" for i in range(50))
    patch = f"<<<<<<< SEARCH\n{search_block}\n=======\n{replace_block}\n>>>>>>> REPLACE"
    current = "public class IssueCount {\n" + search_block + "\n}\n"

    # With standard threshold (max_expansion_ratio=4.0, threshold=50), 50 lines / 1 line = 50x expansion trips:
    ok_default, reason_default = PatchGate.validate_patch_proportionality(
        "src/main/IssueCount.java", patch, current, is_fullstack=False
    )
    # But with fullstack / contract scaling on DTO, it is allowed up to 120 lines:
    ok_fs, reason_fs = PatchGate.validate_patch_proportionality(
        "src/main/dto/IssueCount.java", patch, current, is_fullstack=True
    )
    assert ok_fs is True, f"Fullstack contract edit should be allowed, got: {reason_fs}"

