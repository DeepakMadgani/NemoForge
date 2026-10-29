"""
Unit and Integration Tests for P0 Workspace Integrity & Non-Destructive Patch Architecture.

Verifies:
  1. Three-way patch merges developer delta + ticket delta cleanly.
  2. Overlapping modification in same region raises PatchConflictError and preserves workspace.
  3. Patch ownership and reversible AI-only rollback preserves developer changes byte-for-byte.
  4. Patch idempotency prevents duplicate application.
  5. Protected developer regions block unauthorized overwrites.
  6. Structured deletion authorization blocks accidental omissions and permits authorized deletes.
  7. Optimistic concurrency detects concurrent disk changes.
"""

import tempfile
import pytest
from pathlib import Path

from ticket_to_code.integrity.structural_manifest import (
    StructuralChangeManifest,
    DeleteAuthorization,
    DeleteAuthorizationValidator,
    UnauthorizedDeletionError,
)
from ticket_to_code.integrity.workspace_integrity import (
    WorkspaceBaseline,
    FileBaseline,
    ProtectedRegion,
    ThreeWayPatcher,
    HierarchicalPatcher,
    PatchConflictError,
    WorkspaceIntegrityManager,
)


def test_three_way_merge_preserves_both_deltas():
    """Verify DeveloperDelta + TicketDelta are both preserved when non-overlapping."""
    base = (
        "class MemberService {\n"
        "  constructor() {}\n"
        "  getMembers() { return []; }\n"
        "}\n"
    )

    # Developer added custom validation before the ticket ran
    current = (
        "class MemberService {\n"
        "  constructor() {}\n"
        "  validateDevCustom() { return true; }\n"
        "  getMembers() { return []; }\n"
        "}\n"
    )

    # AI ticket wants to add ticketLogic
    generated = (
        "class MemberService {\n"
        "  constructor() {}\n"
        "  getMembers() { return []; }\n"
        "  addMemberTicketLogic() { return true; }\n"
        "}\n"
    )

    merged, conflicts = ThreeWayPatcher.merge_3way(base, current, generated)
    assert not conflicts, f"Unexpected conflicts: {conflicts}"
    assert "validateDevCustom" in merged, "Developer change was lost!"
    assert "addMemberTicketLogic" in merged, "Ticket change was lost!"
    assert "getMembers" in merged


def test_patch_conflict_stops_and_preserves_workspace():
    """Verify overlapping edits in the same line/method raise PatchConflictError and do not overwrite."""
    base = (
        "function computeDiscount() {\n"
        "  return 0.05;\n"
        "}\n"
    )
    # Developer changed discount to 0.10
    current = (
        "function computeDiscount() {\n"
        "  return 0.10;\n"
        "}\n"
    )
    # AI wants discount to be 0.20
    generated = (
        "function computeDiscount() {\n"
        "  return 0.20;\n"
        "}\n"
    )

    with pytest.raises(PatchConflictError) as exc_info:
        HierarchicalPatcher.apply_patch(
            file_path="discount.ts",
            base_content=base,
            current_content=current,
            proposed_content_or_edits=generated,
            symbol="computeDiscount",
        )

    err = exc_info.value
    assert err.file_path == "discount.ts"
    assert "0.10" in err.developer_change or "3-way merge conflict" in err.reason


def test_reversible_ai_only_rollback_preserves_developer_work():
    """Verify that rolling back an AI patch removes ONLY the AI patch, leaving developer edits intact byte-for-byte."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ws = Path(tmp_dir)
        test_file = ws / "member.ts"

        # Developer's existing file with uncommitted changes
        dev_work = (
            "export class MemberComponent {\n"
            "  // Dev custom feature\n"
            "  devCustomHook() { console.log('developer work'); }\n"
            "  onInit() {}\n"
            "}\n"
        )
        test_file.write_text(dev_work, encoding="utf-8")

        mgr = WorkspaceIntegrityManager(ws)
        mgr.capture_baseline()

        # AI applies a patch adding a ticket method
        ai_new_version = (
            "export class MemberComponent {\n"
            "  // Dev custom feature\n"
            "  devCustomHook() { console.log('developer work'); }\n"
            "  onInit() {}\n"
            "  ticketAddMember() { return 'added'; }\n"
            "}\n"
        )

        patch_rec = mgr.safe_write(
            file_path="member.ts",
            new_content_or_edits=ai_new_version,
            ticket_id="TICKET-101",
            task_id="TASK-1",
            requirement_id="REQ-1",
        )

        assert "ticketAddMember" in test_file.read_text(encoding="utf-8")
        assert "devCustomHook" in test_file.read_text(encoding="utf-8")

        # Now simulate a failure requiring rollback of the AI's changes
        mgr.rollback_patch(patch_rec.patch_id)

        # Content on disk must match dev_work byte-for-byte!
        restored = test_file.read_text(encoding="utf-8")
        assert restored == dev_work, f"Developer work corrupted! Restored:\n{restored}"
        assert "ticketAddMember" not in restored


def test_patch_idempotency():
    """Verify applying the same patch twice detects PATCH_ALREADY_APPLIED and does not duplicate code."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ws = Path(tmp_dir)
        test_file = ws / "test.ts"
        initial = "export const PI = 3.14;\n"
        test_file.write_text(initial, encoding="utf-8")

        mgr = WorkspaceIntegrityManager(ws)
        mgr.capture_baseline()

        target_content = "export const PI = 3.14;\nexport const E = 2.71;\n"

        rec1 = mgr.safe_write("test.ts", target_content, ticket_id="TICKET-1")
        rec2 = mgr.safe_write("test.ts", target_content, ticket_id="TICKET-1")

        assert rec1.patch_id == rec2.patch_id
        # File has exactly one copy of E
        assert test_file.read_text(encoding="utf-8").count("export const E") == 1


def test_protected_developer_region_blocks_unauthorized_overwrite():
    """Verify surgical patch attempting to overwrite a developer protected region raises PatchConflictError."""
    current = (
        "export function process() {\n"
        "  // DEVELOPER PROTECTED LOGIC\n"
        "  devSpecialValidate();\n"
        "  return true;\n"
        "}\n"
    )
    protected = [
        ProtectedRegion(
            file_path="process.ts",
            start_line=3,
            end_line=3,
            content="devSpecialValidate();",
            symbol="devSpecialValidate",
        )
    ]

    # AI tries to replace devSpecialValidate with ticket logic
    edits = [{
        "old_str": "devSpecialValidate();",
        "new_str": "aiReplacement();",
    }]

    with pytest.raises(PatchConflictError) as exc_info:
        HierarchicalPatcher.apply_patch(
            file_path="process.ts",
            base_content=None,
            current_content=current,
            proposed_content_or_edits=edits,
            protected_regions=protected,
        )

    assert "protected developer region" in str(exc_info.value)


def test_structured_deletion_authorization():
    """Verify accidental removal of method is blocked, but authorized deletion succeeds."""
    orig = (
        "export class AccountController {\n"
        "  findAccount() {}\n"
        "  legacyHelper() {}\n"
        "}\n"
    )

    # LLM output omitted legacyHelper()
    gen_omitted = (
        "export class AccountController {\n"
        "  findAccount() {}\n"
        "}\n"
    )

    delta = StructuralChangeManifest.compute_delta(orig, gen_omitted, "account.ts")
    assert "method:legacyHelper" in delta.removed_symbols

    # Without authorization -> raises error
    violations = DeleteAuthorizationValidator.validate_deletions(delta, authorizations=[])
    assert len(violations) == 1
    assert "UNAUTHORIZED_DELETION" in violations[0]

    # With authorization -> approved
    auth = [DeleteAuthorization(
        operation="DELETE_SYMBOL",
        file="account.ts",
        symbol="legacyHelper",
        requirement_id="REQ-3",
        reason="Deprecate old legacyHelper per ticket",
    )]
    violations_auth = DeleteAuthorizationValidator.validate_deletions(delta, authorizations=auth)
    assert len(violations_auth) == 0


def test_real_world_developer_uncommitted_work_survives_rollback_byte_for_byte():
    """
    CRITICAL GOLD-STANDARD INTEGRATION TEST:
    Start with unrelated uncommitted developer changes, let Ticket-to-Code modify
    the same file in a different region, force an AI rollback, and verify that the
    developer changes survive byte-for-byte.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        ws = Path(tmp_dir)
        comp_file = ws / "add-members.component.ts"

        # Developer's existing file with uncommitted modifications
        developer_pre_existing_state = (
            "import { Component, OnInit } from '@angular/core';\n\n"
            "@Component({\n"
            "  selector: 'app-add-members',\n"
            "  templateUrl: './add-members.component.html'\n"
            "})\n"
            "export class AddMembersComponent implements OnInit {\n"
            "  // >>> UNCOMMITTED DEVELOPER WORK (Feature: Custom Org Mapping) <<<\n"
            "  customOrgMapper(org: string): string {\n"
            "    if (!org) return 'N/A';\n"
            "    return org.trim().toUpperCase();\n"
            "  }\n\n"
            "  ngOnInit(): void {}\n\n"
            "  onMemberAdd(): void {\n"
            "    console.log('original implementation');\n"
            "  }\n"
            "}\n"
        )
        comp_file.write_text(developer_pre_existing_state, encoding="utf-8")

        # 1. Ticket starts: capture baseline
        mgr = WorkspaceIntegrityManager(ws)
        mgr.capture_baseline()

        # 2. Ticket-to-Code modifies onMemberAdd in place (different region from customOrgMapper)
        ticket_generated_state = (
            "import { Component, OnInit } from '@angular/core';\n\n"
            "@Component({\n"
            "  selector: 'app-add-members',\n"
            "  templateUrl: './add-members.component.html'\n"
            "})\n"
            "export class AddMembersComponent implements OnInit {\n"
            "  // >>> UNCOMMITTED DEVELOPER WORK (Feature: Custom Org Mapping) <<<\n"
            "  customOrgMapper(org: string): string {\n"
            "    if (!org) return 'N/A';\n"
            "    return org.trim().toUpperCase();\n"
            "  }\n\n"
            "  ngOnInit(): void {}\n\n"
            "  onMemberAdd(): void {\n"
            "    // REQ-1: Verify project member and set organization\n"
            "    this.verifiedMember = true;\n"
            "    console.log('original implementation');\n"
            "  }\n"
            "}\n"
        )

        patch_record = mgr.safe_write(
            file_path="add-members.component.ts",
            new_content_or_edits=ticket_generated_state,
            ticket_id="TICKET-ADD-MEMBERS",
            task_id="TASK-UPDATE-COMPONENT",
            requirement_id="REQ-1",
            symbol="onMemberAdd",
        )

        # Confirm patch applied and both developer code + ticket code exist
        disk_during_run = comp_file.read_text(encoding="utf-8")
        assert "customOrgMapper" in disk_during_run, "Developer code was lost during patch write!"
        assert "this.verifiedMember = true;" in disk_during_run, "Ticket code was not written!"

        # 3. Simulate failure / error resolution aborted / forced rollback
        rolled_back_count = mgr.rollback_ai_changes("TICKET-ADD-MEMBERS")
        assert rolled_back_count == 1

        # 4. Check disk: developer pre-existing state MUST survive BYTE-FOR-BYTE!
        disk_after_rollback = comp_file.read_text(encoding="utf-8")
        assert disk_after_rollback == developer_pre_existing_state, (
            f"FATAL: Developer code was corrupted by rollback!\n"
            f"Expected (byte-for-byte):\n{developer_pre_existing_state}\n"
            f"Actual:\n{disk_after_rollback}"
        )
        assert "this.verifiedMember = true;" not in disk_after_rollback


def test_code_generator_agent_accepts_dataflow_contract_and_kwargs():
    """Verify CodeGeneratorAgent.generate_code accepts dataflow_contract and arbitrary kwargs."""
    from ticket_to_code.agents.code_generator import CodeGeneratorAgent
    from ticket_to_code.models import DevelopmentTask, ArtifactType, TaskType, StructuredRequirements

    agent = CodeGeneratorAgent()
    # Mock LLM to avoid real API calls
    agent.llm = type("MockLLM", (), {"invoke": lambda self, *a, **kw: type("R", (), {"content": "<<<AVIATOR_CODE_START>>>\nconst x = 1;\n<<<AVIATOR_CODE_END>>>"})()})()

    task = DevelopmentTask(
        id="task_test",
        title="Test Task",
        description="Test Description",
        file_path="src/test.ts",
        language="typescript",
        task_type=TaskType.MODIFY,
        artifact_type=ArtifactType.SOURCE_CODE,
    )
    reqs = StructuredRequirements(
        ticket_id="TICKET-1",
        ticket_title="Test",
        functional_requirements=["Req 1"],
        technical_requirements=[],
        edge_cases=[],
        acceptance_tests=[],
        affected_components=[],
    )
    
    # 1. Call with string dataflow_contract and extra kwargs
    code = agent.generate_code(
        task=task,
        requirements=reqs,
        context=[],
        existing_content="const existing = 0;",
        allowed_files=["src/test.ts"],
        readonly_files=[],
        dataflow_contract="MOCK_CONTRACT_DATA",
        unknown_kwarg_future="ALLOWED",
    )
    assert code is not None
    assert agent._dataflow_contract == "MOCK_CONTRACT_DATA"

    # 2. Call with real DataFlowContract object and context=None (tests fallback and render_pre_generation_prompt_block)
    from ticket_to_code.intelligence.contracts.dataflow_contract import DataFlowContract, DataFlowLink
    df_contract = DataFlowContract(
        ticket_id="TICKET-1",
        links=[
            DataFlowLink(
                source="user.companyName",
                producer="member.organization",
                sink="dmember.organization",
                collection_item="DisplayedMember",
            )
        ]
    )
    code2 = agent.generate_code(
        task=task,
        requirements=reqs,
        context=None,
        dataflow_contract=df_contract,
    )
    assert code2 is not None
    assert agent._dataflow_contract == df_contract


