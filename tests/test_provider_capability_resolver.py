"""
Unit and regression tests for ProviderCapabilityResolver and Fast-Exit Compiler Repair Architecture.

Validates:
1. Public API inspection (TypeScript & Java) without hallucination.
2. Repository usage call-site extraction.
3. Layered capability verification (AST -> signature -> params -> return -> usage -> capability).
4. Deterministic distinction between VERIFIED_ALTERNATIVE, NO_VERIFIED_ALTERNATIVE, and UNKNOWN.
5. Invariant: UNKNOWN does NOT fast-exit (bounded targeted resolution).
6. Fast-exit on NO_VERIFIED_ALTERNATIVE with 0 LLM iterations.
7. Structured JSON evidence injected for VERIFIED_ALTERNATIVE.
8. Unrelated files shielded from modification during consumer adaptation.
"""

import json
from pathlib import Path
from typing import Any, List
import pytest

from ticket_to_code.agents.provider_capability_resolver import (
    ProviderCapabilityResolver,
    CapabilityResolutionStatus,
    ProviderMethodDeclaration,
    VerifiedAlternative,
    CapabilityResolutionResult,
)
from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent
from ticket_to_code.agents.diagnostic_normalizer import StructuredDiagnostic


class FakeLLM:
    """Mock LLM that records prompts and returns predefined responses."""
    def __init__(self, responses: List[str] | None = None):
        self.responses = responses or []
        self.invocations: List[Any] = []
        self.call_count = 0

    def invoke(self, messages, *args, **kwargs):
        self.invocations.append(messages)
        self.call_count += 1
        if self.responses:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            resp = self.responses[idx]
        else:
            resp = '```json\n{"action": "FINISHED", "args": {}}\n```'
        from langchain_core.messages import AIMessage
        return AIMessage(content=resp)


def test_1_inspect_provider_public_api(tmp_path: Path):
    """Test AST public API extraction from TS and Java provider files."""
    resolver = ProviderCapabilityResolver(workspace_path=tmp_path)

    # TypeScript provider
    ts_file = tmp_path / "member.service.ts"
    ts_file.write_text(
        """
        export class MemberService {
          constructor(private http: HttpClient) {}
          private internalHelper(): void {}
          protected secretCheck(): boolean { return false; }
          public members(filter: FilterParticipantMemberInput): Observable<ParticipatingMember[]> {
            return this.http.get('/members');
          }
          async getRoles(): Promise<Role[]> {
            return [];
          }
        }
        """,
        encoding="utf-8",
    )

    methods = resolver.inspect_provider_public_api(ts_file)
    method_names = [m.name for m in methods]
    assert "members" in method_names
    assert "getRoles" in method_names
    assert "internalHelper" not in method_names
    assert "secretCheck" not in method_names
    assert "constructor" not in method_names

    members_decl = next(m for m in methods if m.name == "members")
    assert "FilterParticipantMemberInput" in members_decl.parameters[0]
    assert "Observable<ParticipatingMember[]>" in members_decl.return_type

    # Java provider
    java_file = tmp_path / "MembersService.java"
    java_file.write_text(
        """
        public class MembersService {
          public List<MemberDto> getProjectMembers(Long projectId) {
            return Collections.emptyList();
          }
          private void helper() {}
        }
        """,
        encoding="utf-8",
    )
    java_methods = resolver.inspect_provider_public_api(java_file)
    java_names = [m.name for m in java_methods]
    assert "getProjectMembers" in java_names
    assert "helper" not in java_names


def test_2_repository_usage_discovery(tmp_path: Path):
    """Test discovering actual repository call sites for a candidate method."""
    resolver = ProviderCapabilityResolver(workspace_path=tmp_path)

    members_dir = tmp_path / "src" / "app" / "modules" / "members"
    members_dir.mkdir(parents=True)

    sibling = members_dir / "member.component.ts"
    sibling.write_text(
        """
        export class MemberComponent {
          loadData() {
            this.memberService.members({ projectId: { eq: this.projectId } }).subscribe(res => {
              this.members = res;
            });
          }
        }
        """,
        encoding="utf-8",
    )

    usage = resolver._find_repository_usages(
        method_name="members",
        provider_file="src/app/shared/member.service.ts",
        consumer_file="src/app/modules/members/add-members.component.ts",
    )
    assert usage is not None
    assert "member.component.ts" in usage["location"]
    assert "this.memberService.members" in usage["snippet"]
    assert "{ projectId: { eq: this.projectId } }" in usage["args"]


def test_3_verified_alternative_selected_with_layered_proof(tmp_path: Path):
    """Test that layered capability verification selects verified alternative with evidence."""
    members_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "members"
    shared_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "shared" / "services" / "members"
    members_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    prov = shared_dir / "member.service.ts"
    prov.write_text(
        """
        export class MemberService {
          public members(filter: FilterParticipantMemberInput): Observable<ParticipatingMember[]> {
            return of([]);
          }
        }
        """,
        encoding="utf-8",
    )

    sibling = members_dir / "member.component.ts"
    sibling.write_text(
        """
        this.memberService.members({ projectId: { eq: this.projectId } }).subscribe();
        """,
        encoding="utf-8",
    )

    consumer = members_dir / "add-members.component.ts"
    consumer.write_text(
        """
        import { MemberService } from '../shared/services/members/member.service';
        export class AddMembersComponent {
          constructor(private memberService: MemberService) {}
          check() {
            this.memberService.getProjectMembershipStatus(this.projectId, this.userId);
          }
        }
        """,
        encoding="utf-8",
    )

    resolver = ProviderCapabilityResolver(workspace_path=tmp_path)
    res = resolver.resolve_capability(
        provider_name="MemberService",
        missing_symbol="getProjectMembershipStatus",
        provider_file="xchange-ui/src/app/modules/shared/services/members/member.service.ts",
        consumer_file="xchange-ui/src/app/modules/members/add-members.component.ts",
        ticket_requirement="Check whether user is already a member of current project",
    )

    assert res.status == CapabilityResolutionStatus.VERIFIED_ALTERNATIVE
    assert len(res.verified_alternatives) == 1
    alt = res.verified_alternatives[0]
    assert alt.method_name == "members"
    assert "Observable<ParticipatingMember[]>" in alt.return_type
    assert "member.component.ts" in alt.repository_evidence
    assert "FilterParticipantMemberInput" in alt.parameters[0]


def test_4_no_alternative_returns_no_verified_alternative(tmp_path: Path):
    """When a provider exists and is inspected, but has no compatible capability, returns NO_VERIFIED_ALTERNATIVE."""
    prov = tmp_path / "config.service.ts"
    prov.write_text(
        """
        export class ConfigService {
          public getTheme(): string { return 'dark'; }
          public setLocale(loc: string): void {}
        }
        """,
        encoding="utf-8",
    )

    resolver = ProviderCapabilityResolver(workspace_path=tmp_path)
    res = resolver.resolve_capability(
        provider_name="ConfigService",
        missing_symbol="getProjectMembershipStatus",
        provider_file="config.service.ts",
        consumer_file="add-members.component.ts",
        ticket_requirement="Check member status",
    )

    assert res.status == CapabilityResolutionStatus.NO_VERIFIED_ALTERNATIVE
    assert len(res.verified_alternatives) == 0
    assert "none satisfied all layers of capability proof" in res.unresolved_reason or "none match the domain capability" in res.unresolved_reason


def test_5_unknown_does_not_fast_exit(tmp_path: Path):
    """Scenario: provider location is uncertain. Resolver returns UNKNOWN.

    Invariant: UNKNOWN must NOT fast-exit with 0 LLM iterations.
    It triggers bounded targeted investigation.
    """
    resolver = ProviderCapabilityResolver(workspace_path=tmp_path)
    # File does not exist in workspace -> UNKNOWN
    res = resolver.resolve_capability(
        provider_name="NonExistentService",
        missing_symbol="getUserStatus",
        provider_file="non/existent/path.ts",
        consumer_file="add-members.component.ts",
    )

    assert res.status == CapabilityResolutionStatus.UNKNOWN
    assert res.investigation_needed is True

    # Now verify ErrorResolutionAgent behavior on UNKNOWN:
    # It must NOT fast-exit with 0 iterations!
    fake_llm = FakeLLM(responses=['{"action": "FINISHED", "args": {}}'])
    agent = ErrorResolutionAgent(llm=fake_llm, workspace_path=tmp_path)

    err = "src/app/add.component.ts(10,5): error TS2339: Property 'getUserStatus' does not exist on type 'NonExistentService'."
    agent.resolve_errors([err], max_iter=5)

    # UNKNOWN must proceed to bounded targeted investigation (call_count > 0), NOT 0!
    assert fake_llm.call_count > 0, "UNKNOWN must not fast-exit with zero LLM iterations!"


def test_6_fast_exit_on_no_verified_alternative(tmp_path: Path):
    """When provider has NO_VERIFIED_ALTERNATIVE, ErrorResolutionAgent exits immediately in 0 LLM calls."""
    prov = tmp_path / "theme.service.ts"
    prov.write_text(
        """
        export class ThemeService {
          public getTheme(): string { return 'light'; }
        }
        """,
        encoding="utf-8",
    )

    consumer = tmp_path / "component.ts"
    consumer.write_text(
        """
        import { ThemeService } from './theme.service';
        export class Component {
          check() { this.themeService.getMemberRole(); }
        }
        """,
        encoding="utf-8",
    )

    fake_llm = FakeLLM(responses=['{"action": "FINISHED", "args": {}}'])
    agent = ErrorResolutionAgent(llm=fake_llm, workspace_path=tmp_path)

    err = f"{consumer.name}(4,5): error TS2339: Property 'getMemberRole' does not exist on type 'ThemeService'."
    res = agent.resolve_errors([err], max_iter=10)

    # Fast-exit: 0 LLM calls made!
    assert fake_llm.call_count == 0, f"Expected 0 LLM calls on NO_VERIFIED_ALTERNATIVE, got {fake_llm.call_count}"
    assert res == {}
    assert len(agent.audit_trail) > 0
    assert "no compatible alternative" in agent.audit_trail[0].final_unresolved_reason


def test_7_targeted_consumer_patch_prompt_structure(tmp_path: Path):
    """When VERIFIED_ALTERNATIVE exists, structured JSON evidence is passed into user prompt."""
    shared_dir = tmp_path / "shared"
    members_dir = tmp_path / "members"
    shared_dir.mkdir(parents=True)
    members_dir.mkdir(parents=True)

    prov = shared_dir / "member.service.ts"
    prov.write_text(
        """
        export class MemberService {
          public members(filter: FilterParticipantMemberInput): Observable<ParticipatingMember[]> {
            return of([]);
          }
        }
        """,
        encoding="utf-8",
    )

    sibling = members_dir / "member.component.ts"
    sibling.write_text(
        """
        this.memberService.members({ projectId: { eq: 1 } }).subscribe();
        """,
        encoding="utf-8",
    )

    consumer = members_dir / "add-members.component.ts"
    consumer.write_text(
        """
        import { MemberService } from '../shared/member.service';
        export class AddMembersComponent {
          constructor(private memberService: MemberService) {}
          addMembers() {
            this.memberService.getProjectMembershipStatus(1, 2);
          }
        }
        """,
        encoding="utf-8",
    )

    fake_llm = FakeLLM(responses=['{"action": "FINISHED", "args": {}}'])
    agent = ErrorResolutionAgent(llm=fake_llm, workspace_path=tmp_path)

    err = "members/add-members.component.ts(6,24): error TS2339: Property 'getProjectMembershipStatus' does not exist on type 'MemberService'."
    agent.resolve_errors([err], max_iter=10)

    assert fake_llm.call_count == 1
    # Check prompt sent to LLM contains structured evidence
    prompt_msgs = fake_llm.invocations[0]
    prompt_text = "".join(m.content for m in prompt_msgs)
    assert "STRUCTURED PROVIDER CAPABILITY EVIDENCE" in prompt_text
    assert "VERIFIED_ALTERNATIVE" in prompt_text
    assert "members(filter: FilterParticipantMemberInput): Observable<ParticipatingMember[]>" in prompt_text
    assert "add-members.component.ts" in prompt_text


def test_8_unrelated_files_shielded_during_consumer_adaptation(tmp_path: Path):
    """During consumer adaptation, edits to unrelated files are blocked by the consumer shield."""
    consumer = tmp_path / "consumer.component.ts"
    consumer.write_text("export class Consumer { add() {} }", encoding="utf-8")

    unrelated_java = tmp_path / "MembersService.java"
    unrelated_java.write_text("public class MembersService {}", encoding="utf-8")

    # LLM attempts to edit unrelated Java file first, then finishes
    bad_action = json.dumps({
        "action": "REPLACE_CONTENT",
        "args": {
            "file_path": "MembersService.java",
            "target_content": "public class MembersService {}",
            "replacement_content": "public class MembersService { void extra() {} }"
        }
    })
    finish_action = json.dumps({"action": "FINISHED", "args": {}})

    fake_llm = FakeLLM(responses=[bad_action, finish_action])
    agent = ErrorResolutionAgent(llm=fake_llm, workspace_path=tmp_path)

    # Set mock verified alternative
    from ticket_to_code.agents.diagnostic_localizer import FailureAttribution, FailureOwner
    from ticket_to_code.agents.provider_capability_resolver import CapabilityResolutionResult, VerifiedAlternative
    cap_res = CapabilityResolutionResult(
        status=CapabilityResolutionStatus.VERIFIED_ALTERNATIVE,
        provider_name="MemberService",
        provider_file="member.service.ts",
        missing_symbol="getStatus",
        is_protected=True,
        verified_alternatives=[
            VerifiedAlternative(
                method_name="members",
                signature="members(): Observable<any>",
                parameters=[],
                return_type="Observable<any>",
                repository_evidence="in sibling",
                usage_example="this.members()",
                compatibility_proof="proof",
            )
        ]
    )

    # Force diagnostic to match
    err = "consumer.component.ts(1,1): error TS2339: Property 'getStatus' does not exist on type 'MemberService'."
    agent.resolve_errors([err], max_iter=2)

    # Unrelated Java file must be completely untouched!
    assert unrelated_java.read_text(encoding="utf-8") == "public class MembersService {}"
