"""Unit tests for DiagnosticLocalizer: failure ownership, AST localization,
relevance-driven context, and bounded repair.
"""

from pathlib import Path
import pytest

from ticket_to_code.agents.diagnostic_localizer import (
    DiagnosticLocalizer,
    FailureOwner,
    RepairContextTier,
)
from ticket_to_code.agents.diagnostic_normalizer import StructuredDiagnostic


_SAMPLE_TS_COMPONENT = """import { Component, OnInit } from '@angular/core';
import { MemberService } from './member.service';
import { UnrelatedService } from './unrelated.service';

@Component({
  selector: 'app-members',
  templateUrl: './members.component.html',
})
export class MembersComponent implements OnInit {
  members: any[] = [];

  constructor(private memberService: MemberService) {}

  ngOnInit(): void {
    this.loadMembers();
  }

  loadMembers(): void {
    this.memberService.getProjectMembers(123).subscribe((res) => {
      this.members = res;
    });
  }

  unrelatedMethod(): void {
    console.log("This is 50 lines of unrelated code that should NOT be in repair prompt");
  }
}
"""


def test_failure_ownership_identifies_provider():
    localizer = DiagnosticLocalizer()
    diag = StructuredDiagnostic(
        raw="src/members.component.ts(21,24): error TS2339: Property 'getProjectMembers' does not exist on type 'MemberService'.",
        code="TS2339",
        message="Property 'getProjectMembers' does not exist on type 'MemberService'.",
        source_file="src/members.component.ts",
        line=21,
        column=24,
    )

    class _Task:
        file_path = "src/member.service.ts"
        target_class = "MemberService"
        task_type = type("_T", (), {"value": "modify"})()

    attrib = localizer.attribute_failure(diag, planned_tasks=[_Task()])
    assert attrib.owner == FailureOwner.PROVIDER
    assert attrib.missing_symbol == "getProjectMembers"
    assert attrib.provider_type == "MemberService"
    assert attrib.provider_file == "src/member.service.ts"
    assert attrib.is_provider_writable is True


def test_error_localized_to_containing_method():
    localizer = DiagnosticLocalizer()
    # Line 21 is inside loadMembers()
    diag = StructuredDiagnostic(
        raw="src/members.component.ts(21,24): error TS2339: Property 'getProjectMembers' does not exist on type 'MemberService'.",
        code="TS2339",
        message="Property 'getProjectMembers' does not exist on type 'MemberService'.",
        source_file="src/members.component.ts",
        line=21,
        column=24,
    )

    ctx = localizer.localize_context("src/members.component.ts", _SAMPLE_TS_COMPONENT, [diag])
    assert ctx.context_tier == RepairContextTier.TIER_1_LOCALIZED_METHOD
    assert ctx.target_method_name == "loadMembers"
    # Prompt snippet must contain loadMembers and memberService
    assert "loadMembers" in ctx.prompt_snippet
    assert "getProjectMembers" in ctx.prompt_snippet
    # Unrelated method must NOT be sent in prompt!
    assert "unrelatedMethod" not in ctx.prompt_snippet
    # Lines sent should be small (< 25 lines), not entire file
    assert ctx.source_lines_sent < 25


def test_whole_file_escalation_only_when_unlocalizable():
    localizer = DiagnosticLocalizer()
    # Line 1 is import statement outside any method boundary
    diag = StructuredDiagnostic(
        raw="src/members.component.ts(1,1): error TS1005: ';' expected.",
        code="TS1005",
        message="';' expected.",
        source_file="src/members.component.ts",
        line=1,
        column=1,
    )

    ctx = localizer.localize_context("src/members.component.ts", _SAMPLE_TS_COMPONENT, [diag])
    assert ctx.context_tier == RepairContextTier.TIER_4_WHOLE_FILE_ESCALATION
    assert ctx.escalation_reason is not None
    assert "outside any method" in ctx.escalation_reason


def test_bounded_repair_stops_on_repeated_error():
    localizer = DiagnosticLocalizer()
    diag = StructuredDiagnostic(
        raw="src/members.component.ts(21,24): error TS2339: Property 'getProjectMembers' does not exist on type 'MemberService'.",
        code="TS2339",
        message="Property 'getProjectMembers' does not exist on type 'MemberService'.",
        source_file="src/members.component.ts",
        line=21,
        column=24,
    )

    # Attempt 1: allowed
    assert localizer.can_attempt_repair([diag], max_attempts=2)
    localizer.record_attempt([diag])

    # Attempt 2: allowed
    assert localizer.can_attempt_repair([diag], max_attempts=2)
    localizer.record_attempt([diag])

    # Attempt 3: BLOCKED by bounded budget!
    assert not localizer.can_attempt_repair([diag], max_attempts=2)
    assert localizer.telemetry.repair_attempts == 2
