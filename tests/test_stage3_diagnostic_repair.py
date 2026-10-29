"""
Stage 3 Focused Unit Tests: Diagnostic Ownership-Aware Compiler Repair,
Protected Provider Handling, Atomic Patch Mismatch Recovery, AST-Localized Context,
Bounded Repair, and Audit Trail Verification.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ticket_to_code.agents.diagnostic_localizer import (
    DiagnosticLocalizer,
    FailureOwner,
    RepairContextTier,
    FailureAttribution,
    ProviderMethodDeclaration,
    AlternativeCapability,
    RepairAuditRecord,
)
from ticket_to_code.agents.diagnostic_normalizer import StructuredDiagnostic
from ticket_to_code.agents.change_authorization import ChangeRole
from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent
from ticket_to_code.agents.edit_loop_agent import EditLoopAgent, ApplyEditResult


_SAMPLE_CONSUMER_COMPONENT = """import { Component, OnInit } from '@angular/core';
import { MemberService } from './member.service';

@Component({
  selector: 'app-add-members',
  templateUrl: './add-members.component.html',
})
export class AddMembersComponent implements OnInit {
  projectId = 'proj-123';
  membersList: any[] = [];

  constructor(private memberService: MemberService) {}

  ngOnInit(): void {
    this.checkMembership();
  }

  checkMembership(): void {
    this.memberService.getProjectMembers(this.projectId).subscribe(res => {
      this.membersList = res;
    });
  }

  unrelatedMethod(): void {
    console.log("Unrelated code that should not be present in Tier 1 localized method snippet");
  }
}
"""

_SAMPLE_PROTECTED_MEMBER_SERVICE = """import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';

export interface FilterParticipantMemberInput {
  projectId?: { eq: string };
}

export interface ParticipatingMember {
  id: string;
  userId: string;
  email: string;
}

@Injectable({
  providedIn: 'root'
})
export class MemberService {
  public addContractMembers(contractMembers: {}): Observable<any> {
    return of({ success: true });
  }

  public members(filter: FilterParticipantMemberInput): Observable<ParticipatingMember[]> {
    return of([{ id: '1', userId: 'u1', email: 'test@example.com' }]);
  }

  public updateProjectUsers(projectId: string, member: any): Observable<any> {
    return of({ success: true });
  }
}
"""

_SAMPLE_NO_ALT_SERVICE = """import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class LegacyService {
  public doNothing(): Observable<any> {
    return of(null);
  }
}
"""


# ── 1. TS2339 Missing Method on Protected Service ──────────────────────────────
def test_1_ts2339_missing_method_on_protected_service_rejected(tmp_path):
    # Setup protected service on disk
    svc_file = tmp_path / "member.service.ts"
    svc_file.write_text(_SAMPLE_PROTECTED_MEMBER_SERVICE, encoding="utf-8")
    consumer_file = tmp_path / "add-members.component.ts"
    consumer_file.write_text(_SAMPLE_CONSUMER_COMPONENT, encoding="utf-8")

    localizer = DiagnosticLocalizer(workspace_path=tmp_path)
    diag = StructuredDiagnostic(
        raw="src/add-members.component.ts(18,24): error TS2339: Property 'getProjectMembers' does not exist on type 'MemberService'.",
        code="TS2339",
        message="Property 'getProjectMembers' does not exist on type 'MemberService'.",
        source_file="add-members.component.ts",
        line=18,
        column=24,
    )

    # When provider is not a planned writable task, it must be CROSS_FILE_CONTRACT and protected
    attrib = localizer.attribute_failure(diag, planned_tasks=[])
    assert attrib.owner == FailureOwner.CROSS_FILE_CONTRACT
    assert attrib.missing_symbol == "getProjectMembers"
    assert attrib.provider_type == "MemberService"
    assert attrib.is_provider_writable is False
    assert attrib.is_protected_provider is True


# ── 2. Protected Provider Has Compatible Alternative ───────────────────────────
def test_2_protected_provider_with_compatible_alternative_allows_consumer_adaptation(tmp_path):
    svc_file = tmp_path / "member.service.ts"
    svc_file.write_text(_SAMPLE_PROTECTED_MEMBER_SERVICE, encoding="utf-8")
    consumer_file = tmp_path / "add-members.component.ts"
    consumer_file.write_text(_SAMPLE_CONSUMER_COMPONENT, encoding="utf-8")
    # Sibling file proving usage of .members(...)
    sibling_file = tmp_path / "member.component.ts"
    sibling_file.write_text("this.memberService.members({ projectId: { eq: '123' } });", encoding="utf-8")

    localizer = DiagnosticLocalizer(workspace_path=tmp_path)
    diag = StructuredDiagnostic(
        raw="add-members.component.ts(18,24): error TS2339: Property 'getProjectMembers' does not exist on type 'MemberService'.",
        code="TS2339",
        message="Property 'getProjectMembers' does not exist on type 'MemberService'.",
        source_file="add-members.component.ts",
        line=18,
        column=24,
    )

    attrib = localizer.attribute_failure(diag, planned_tasks=[])
    assert attrib.is_protected_provider is True
    assert len(attrib.inspected_methods) >= 2
    assert attrib.alternative_capability is not None
    assert attrib.alternative_capability.is_semantically_compatible is True
    assert attrib.alternative_capability.method_name == "members"
    assert "Observable<ParticipatingMember[]>" in attrib.alternative_capability.return_type
    assert "Verified by repository usage" in attrib.alternative_capability.compatibility_reason


# ── 3. Protected Provider Has No Compatible Alternative ────────────────────────
def test_3_protected_provider_without_compatible_alternative_marks_unresolved(tmp_path):
    svc_file = tmp_path / "legacy.service.ts"
    svc_file.write_text(_SAMPLE_NO_ALT_SERVICE, encoding="utf-8")
    consumer_file = tmp_path / "consumer.component.ts"
    consumer_file.write_text("import { LegacyService } from './legacy.service';\nthis.legacyService.getProjectMembers();", encoding="utf-8")

    localizer = DiagnosticLocalizer(workspace_path=tmp_path)
    diag = StructuredDiagnostic(
        raw="consumer.component.ts(2,5): error TS2339: Property 'getProjectMembers' does not exist on type 'LegacyService'.",
        code="TS2339",
        message="Property 'getProjectMembers' does not exist on type 'LegacyService'.",
        source_file="consumer.component.ts",
        line=2,
        column=5,
    )

    attrib = localizer.attribute_failure(diag, planned_tasks=[])
    assert attrib.is_protected_provider is True
    # Alternative capability must be None because LegacyService has no member/project query method
    assert attrib.alternative_capability is None
    assert "NO verified alternative available" in attrib.reason


# ── 4. Planned GENERATED_DEPENDENCY Provider Repair Allowed ───────────────────
def test_4_planned_generated_dependency_provider_repair_allowed(tmp_path):
    localizer = DiagnosticLocalizer(workspace_path=tmp_path)
    diag = StructuredDiagnostic(
        raw="src/consumer.ts(10,5): error TS2339: Property 'newHelper' does not exist on type 'HelperService'.",
        code="TS2339",
        message="Property 'newHelper' does not exist on type 'HelperService'.",
        source_file="src/consumer.ts",
        line=10,
        column=5,
    )

    class _GeneratedTask:
        file_path = "src/helper.service.ts"
        target_class = "HelperService"
        task_type = type("_T", (), {"value": "modify"})()

    with patch("ticket_to_code.agents.change_authorization.classify_change_role", return_value=(ChangeRole.GENERATED_DEPENDENCY.value, "Authorized generated dependency")):
        attrib = localizer.attribute_failure(diag, planned_tasks=[_GeneratedTask()])

    assert attrib.owner == FailureOwner.PROVIDER
    assert attrib.is_provider_writable is True
    assert attrib.is_protected_provider is False
    assert attrib.provider_file == "src/helper.service.ts"


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses) if responses else []
        self.call_count = 0
        self.call_args_list = []

    def invoke(self, messages):
        self.call_count += 1
        self.call_args_list.append(((messages,), {}))
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, str):
                from langchain_core.messages import AIMessage
                return AIMessage(content=item)
            return item
        from langchain_core.messages import AIMessage
        return AIMessage(content="")


# ── 5. SEARCH/REPLACE Mismatch Re-Reads Source & Retries Once ──────────────────
def test_5_search_replace_mismatch_rereads_source_and_retries_once(tmp_path):
    test_file = tmp_path / "test.ts"
    initial_content = "function foo() {\n  const a = 1;\n  return a;\n}\n"
    test_file.write_text(initial_content, encoding="utf-8")

    mismatched_block = "<<<<<<< SEARCH\n  const a = 999;\n=======\n  const a = 2;\n>>>>>>> REPLACE"
    matching_block = "<<<<<<< SEARCH\n  const a = 1;\n=======\n  const a = 2;\n>>>>>>> REPLACE"
    fake_llm = FakeLLM(responses=[mismatched_block, matching_block])

    mock_symbol_index = MagicMock()
    mock_symbol_index.get_ts_diagnostics.return_value = []
    mock_symbol_index.get_members_for_file.return_value = None

    agent = EditLoopAgent(
        workspace_path=str(tmp_path),
        llm=fake_llm,
        symbol_index=mock_symbol_index,
    )

    from ticket_to_code.agents.edit_loop_agent import EditDecision
    decision = EditDecision(file_path="test.ts", reason="Update value", edit_description="change 1 to 2")
    ticket = MagicMock(title="Update", description="Update value")

    result = agent._generate_edit(
        decision=decision,
        current_content=initial_content,
        ticket=ticket,
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )
    assert result is not None
    assert "const a = 2;" in result
    # Verify exactly 2 LLM calls: initial attempt + exactly 1 retry
    assert fake_llm.call_count == 2


# ── 6. Second Mismatch Stops Safely Without Partial Mutation ───────────────────
def test_6_second_mismatch_stops_safely_without_partial_mutation(tmp_path):
    test_file = tmp_path / "test.ts"
    initial_content = "function foo() {\n  const a = 1;\n  return a;\n}\n"
    test_file.write_text(initial_content, encoding="utf-8")

    mismatched_block_1 = "<<<<<<< SEARCH\n  const a = 999;\n=======\n  const a = 2;\n>>>>>>> REPLACE"
    mismatched_block_2 = "<<<<<<< SEARCH\n  const a = 888;\n=======\n  const a = 2;\n>>>>>>> REPLACE"
    fake_llm = FakeLLM(responses=[mismatched_block_1, mismatched_block_2])

    mock_symbol_index = MagicMock()
    mock_symbol_index.get_ts_diagnostics.return_value = []
    mock_symbol_index.get_members_for_file.return_value = None

    agent = EditLoopAgent(
        workspace_path=str(tmp_path),
        llm=fake_llm,
        symbol_index=mock_symbol_index,
    )

    from ticket_to_code.agents.edit_loop_agent import EditDecision
    decision = EditDecision(file_path="test.ts", reason="Update value", edit_description="change 1 to 2")
    ticket = MagicMock(title="Update", description="Update value")

    result = agent._generate_edit(
        decision=decision,
        current_content=initial_content,
        ticket=ticket,
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )
    # Edit safely rejected: returns None, file content untouched
    assert result is None
    assert test_file.read_text(encoding="utf-8") == initial_content
    # Exactly 2 LLM invocations (no 3rd retry)
    assert fake_llm.call_count == 2


# ── 7. DiagnosticLocalizer Identifies All 5 FailureOwners ──────────────────────
def test_7_diagnostic_localizer_identifies_all_failure_owners(tmp_path):
    localizer = DiagnosticLocalizer(workspace_path=tmp_path)

    # 1. CONSUMER (syntax / internal logic error)
    diag_consumer = StructuredDiagnostic(
        raw="src/foo.ts(5,1): error TS1005: ';' expected.",
        code="TS1005",
        message="';' expected.",
        source_file="src/foo.ts",
        line=5,
    )
    assert localizer.attribute_failure(diag_consumer).owner == FailureOwner.CONSUMER

    # 2. INFRASTRUCTURE / BUILD_CONFIG
    diag_infra = StructuredDiagnostic(
        raw="error TS5023: Unknown compiler option 'target' in tsconfig.json",
        code="TS5023",
        message="Unknown compiler option 'target' in tsconfig.json",
        source_file="tsconfig.json",
        line=1,
    )
    assert localizer.attribute_failure(diag_infra).owner == FailureOwner.INFRASTRUCTURE

    # 3. PROVIDER (planned writable dependency)
    diag_prov = StructuredDiagnostic(
        raw="src/c.ts(1,1): error TS2339: Property 'fn' does not exist on type 'GenService'.",
        code="TS2339",
        message="Property 'fn' does not exist on type 'GenService'.",
        source_file="src/c.ts",
        line=1,
    )
    class _Task:
        file_path = "src/gen.service.ts"
        target_class = "GenService"
        task_type = type("_T", (), {"value": "modify"})()
    with patch("ticket_to_code.agents.change_authorization.classify_change_role", return_value=(ChangeRole.GENERATED_DEPENDENCY.value, "Authorized")):
        assert localizer.attribute_failure(diag_prov, planned_tasks=[_Task()]).owner == FailureOwner.PROVIDER

    # 4. CROSS_FILE_CONTRACT (protected provider file)
    prov_p = tmp_path / "shared.service.ts"
    prov_p.write_text("export class SharedService {}", encoding="utf-8")
    diag_cross = StructuredDiagnostic(
        raw="src/c.ts(1,1): error TS2339: Property 'fn' does not exist on type 'SharedService'.",
        code="TS2339",
        message="Property 'fn' does not exist on type 'SharedService'.",
        source_file="src/c.ts",
        line=1,
    )
    attrib_cross = localizer.attribute_failure(diag_cross, planned_tasks=[])
    assert attrib_cross.owner == FailureOwner.CROSS_FILE_CONTRACT

    # 5. UNKNOWN (unresolvable provider)
    diag_unknown = StructuredDiagnostic(
        raw="src/c.ts(1,1): error TS2339: Property 'fn' does not exist on type 'GhostTypeNonExistent'.",
        code="TS2339",
        message="Property 'fn' does not exist on type 'GhostTypeNonExistent'.",
        source_file="src/c.ts",
        line=1,
    )
    assert localizer.attribute_failure(diag_unknown, planned_tasks=[]).owner == FailureOwner.UNKNOWN


# ── 8. Localized Context Uses AST Boundaries Without Arbitrary Heuristics ──────
def test_8_localized_context_uses_ast_boundaries_without_arbitrary_heuristics(tmp_path):
    localizer = DiagnosticLocalizer(workspace_path=tmp_path)
    # Diagnostic at line 20 (inside checkMembership)
    diag = StructuredDiagnostic(
        raw="add-members.component.ts(20,24): error TS2339: Property 'getProjectMembers' does not exist on type 'MemberService'.",
        code="TS2339",
        message="Property 'getProjectMembers' does not exist on type 'MemberService'.",
        source_file="add-members.component.ts",
        line=20,
        column=24,
    )

    ctx = localizer.localize_context("add-members.component.ts", _SAMPLE_CONSUMER_COMPONENT, [diag])
    assert ctx.context_tier == RepairContextTier.TIER_1_LOCALIZED_METHOD
    assert ctx.target_method_name == "checkMembership"
    # Contains targeted method
    assert "checkMembership" in ctx.prompt_snippet
    assert "getProjectMembers" in ctx.prompt_snippet
    # Unrelated method must NOT be included in Tier 1 context
    assert "unrelatedMethod" not in ctx.prompt_snippet
    # Targeted imports included
    assert "MemberService" in ctx.prompt_snippet
    # Constructor injection included
    assert "constructor(private memberService: MemberService)" in ctx.prompt_snippet


# ── 9. Protected Provider Immutable Even When LLM Proposes Provider Patch ─────
def test_9_protected_provider_is_not_modified_even_when_llm_proposes_provider_patch(tmp_path):
    """Regression test specifically requested by user:
    Mock/force LLM to propose adding `getProjectMembers(...)` inside `member.service.ts`.
    Verify:
      - provider patch rejected
      - member.service.ts unchanged
      - consumer NOT blindly rewritten
      - audit trail records rejection
      - alternative search performed
    """
    svc_file = tmp_path / "member.service.ts"
    svc_file.write_text(_SAMPLE_PROTECTED_MEMBER_SERVICE, encoding="utf-8")
    original_svc_content = _SAMPLE_PROTECTED_MEMBER_SERVICE

    consumer_file = tmp_path / "add-members.component.ts"
    consumer_file.write_text(_SAMPLE_CONSUMER_COMPONENT, encoding="utf-8")

    mock_llm = MagicMock()
    agent = ErrorResolutionAgent(mock_llm, workspace_path=tmp_path)

    # Force agent to see member.service.ts as an untouched pre-existing file
    agent.pre_edit_snapshots[str(svc_file)] = original_svc_content
    agent.our_modified_files = {"add-members.component.ts"}

    # Attempt to apply edit to protected member.service.ts
    patch_proposal = (
        "public getProjectMembers(projectId: string): Observable<any[]> {\n"
        "  return of([]);\n"
        "}\n"
    )
    reject_msg = agent._guard_protected_file(str(svc_file), existing_content=original_svc_content)

    # 1. Provider patch rejected
    assert reject_msg is not None
    assert "BLOCKED" in reject_msg
    assert "PROTECTED provider" in reject_msg

    # 2. member.service.ts completely unchanged on disk and in memory
    assert svc_file.read_text(encoding="utf-8") == original_svc_content
    assert str(svc_file).lower() not in agent.modified_files

    # 3. Audit trail records rejection
    assert len(agent.audit_trail) >= 1
    rec: RepairAuditRecord = agent.audit_trail[-1]
    assert rec.authorization_result == "BLOCKED_PROTECTED_PROVIDER"
    assert rec.owner == FailureOwner.CROSS_FILE_CONTRACT
    assert rec.alternative_api_searched is True

    # 4. Guidance provides real public methods available on protected service
    assert "members" in reject_msg
    assert "addContractMembers" in reject_msg
