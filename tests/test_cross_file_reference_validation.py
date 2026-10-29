"""
Cross-File Reference Validation — Regression Tests

Tests the generic cross-file reference validation architecture:
1. AngularTemplateExtractor — syntax-specific reference extraction
2. GeneratedReferenceValidator.validate_template() — generic resolution
3. End-to-end: existingOrganizationName vs organizationName mismatch

Architecture under test:
    SYNTAX-SPECIFIC EXTRACTION → GENERIC RESOLUTION → CONTRACT VERIFICATION

Author: Deepak Madgani
Date: September 2026
"""

import pytest
from ticket_to_code.agents.generated_reference_validator import (
    AngularTemplateExtractor,
    CrossFileReference,
    GeneratedReferenceValidator,
    ReferenceValidationResult,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

CONTROLLER_TS = """\
import { Component, OnInit } from '@angular/core';
import { ProjectMember } from '../../models/project-member.model';

@Component({
  selector: 'se-add-members',
  templateUrl: './add-members.component.html',
  styleUrls: ['./add-members.component.scss']
})
export class AddMembersComponent implements OnInit {
  projectMembers: ProjectMember[] = [];
  selectedMember: ProjectMember | null = null;
  existingOrganizationName: string = '';
  showAddForm: boolean = false;
  isLoading: boolean = false;

  constructor(private memberService: MemberService) {}

  ngOnInit(): void {
    this.loadMembers();
  }

  onMemberAdd(): void {
    if (this.selectedMember) {
      this.memberService.addMember(this.selectedMember);
    }
  }

  loadMembers(): void {
    this.isLoading = true;
    this.memberService.getMembers().subscribe(members => {
      this.projectMembers = members;
      this.isLoading = false;
    });
  }

  toggleAddForm(): void {
    this.showAddForm = !this.showAddForm;
  }
}
"""

HTML_FILE_PATH = "xchange-ui/src/app/modules/members/add-members/add-members.component.html"
TS_FILE_PATH = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"


def _build_written_files(ts_content=CONTROLLER_TS, extra=None):
    """Build a _written_files dict with the controller content."""
    files = {
        TS_FILE_PATH.replace("\\", "/").lower(): ts_content,
    }
    if extra:
        files.update(extra)
    return files


# ═══════════════════════════════════════════════════════════════════════════════
# Test Suite 1: AngularTemplateExtractor — syntax-specific extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestAngularTemplateExtractor:
    """Tests that the Angular template extractor correctly parses
    all Angular template binding syntaxes."""

    def test_interpolation_binding(self):
        """{{ member.existingOrganizationName }} → extracts member.existingOrganizationName"""
        html = '<span>{{ member.existingOrganizationName }}</span>'
        refs = AngularTemplateExtractor.extract(html, HTML_FILE_PATH)
        assert any(
            r.receiver == "member" and r.member == "existingOrganizationName"
            for r in refs
        ), f"Expected member.existingOrganizationName, got: {[(r.receiver, r.member) for r in refs]}"

    def test_property_binding(self):
        """[placeholder]="member.organizationName" → extracts member.organizationName"""
        html = '<input [placeholder]="member.organizationName">'
        refs = AngularTemplateExtractor.extract(html, HTML_FILE_PATH)
        assert any(
            r.receiver == "member" and r.member == "organizationName"
            for r in refs
        )

    def test_event_binding_bare(self):
        """(click)="onMemberAdd()" → extracts this.onMemberAdd"""
        html = '<button (click)="onMemberAdd()">Add</button>'
        refs = AngularTemplateExtractor.extract(html, HTML_FILE_PATH)
        assert any(
            r.receiver == "this" and r.member == "onMemberAdd"
            for r in refs
        ), f"Expected this.onMemberAdd, got: {[(r.receiver, r.member) for r in refs]}"

    def test_event_binding_with_receiver(self):
        """(click)="member.save()" → extracts member.save"""
        html = '<button (click)="member.save()">Save</button>'
        refs = AngularTemplateExtractor.extract(html, HTML_FILE_PATH)
        assert any(
            r.receiver == "member" and r.member == "save"
            for r in refs
        )

    def test_ngif_directive(self):
        """*ngIf="member.isActive" → extracts member.isActive"""
        html = '<div *ngIf="member.isActive">Active</div>'
        refs = AngularTemplateExtractor.extract(html, HTML_FILE_PATH)
        assert any(
            r.receiver == "member" and r.member == "isActive"
            for r in refs
        )

    def test_ngfor_iterable(self):
        """*ngFor="let m of projectMembers" → extracts this.projectMembers"""
        html = '<div *ngFor="let m of projectMembers">{{ m.name }}</div>'
        refs = AngularTemplateExtractor.extract(html, HTML_FILE_PATH)
        assert any(
            r.receiver == "this" and r.member == "projectMembers"
            for r in refs
        ), f"Expected this.projectMembers, got: {[(r.receiver, r.member) for r in refs]}"

    def test_ngfor_element_property(self):
        """*ngFor="let m of projectMembers" then {{ m.name }} → extracts m.name"""
        html = '<div *ngFor="let m of projectMembers">{{ m.name }}</div>'
        refs = AngularTemplateExtractor.extract(html, HTML_FILE_PATH)
        assert any(
            r.receiver == "m" and r.member == "name"
            for r in refs
        )

    def test_empty_content(self):
        """Empty content → no references."""
        refs = AngularTemplateExtractor.extract("", HTML_FILE_PATH)
        assert refs == []

    def test_deduplication(self):
        """Same binding used twice → only one reference extracted."""
        html = """
        <span>{{ member.name }}</span>
        <span>{{ member.name }}</span>
        """
        refs = AngularTemplateExtractor.extract(html, HTML_FILE_PATH)
        member_name_refs = [
            r for r in refs if r.receiver == "member" and r.member == "name"
        ]
        assert len(member_name_refs) == 1

    def test_multiple_bindings(self):
        """Multiple different bindings → all extracted."""
        html = """
        <div *ngFor="let member of projectMembers">
          <span>{{ member.name }}</span>
          <span>{{ member.email }}</span>
          <button (click)="onMemberAdd()">Add</button>
        </div>
        """
        refs = AngularTemplateExtractor.extract(html, HTML_FILE_PATH)
        receivers_members = {(r.receiver, r.member) for r in refs}
        assert ("member", "name") in receivers_members
        assert ("member", "email") in receivers_members
        assert ("this", "onMemberAdd") in receivers_members


# ═══════════════════════════════════════════════════════════════════════════════
# Test Suite 2: GeneratedReferenceValidator.validate_template() — resolution
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateTemplate:
    """Tests the generic resolution engine for Angular template bindings."""

    def test_exact_mismatch_organization_name(self):
        """THE REGRESSION TEST: existingOrganizationName vs organizationName.

        TS has: existingOrganizationName
        HTML uses: member.organizationName
        → MUST BE UNRESOLVED with close_match = existingOrganizationName
        """
        html = """
        <div *ngFor="let member of projectMembers">
          <input [placeholder]="member.organizationName">
        </div>
        """
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content=html,
            html_file_path=HTML_FILE_PATH,
            written_files=_build_written_files(),
        )

        # Must have at least one unresolved reference
        assert not result.is_clean, (
            "Expected unresolved references but validation passed clean! "
            f"Checks: {[(c.receiver, c.member, c.status) for c in result.checks]}"
        )

        # Find the specific organizationName check
        org_checks = [
            c for c in result.checks
            if c.member == "organizationName"
        ]
        assert len(org_checks) > 0, (
            f"Expected check for 'organizationName' but got: "
            f"{[(c.receiver, c.member, c.status) for c in result.checks]}"
        )

        org_check = org_checks[0]
        assert org_check.status == "unresolved", (
            f"Expected 'unresolved' but got '{org_check.status}' for organizationName"
        )
        assert org_check.close_match == "existingOrganizationName", (
            f"Expected close_match='existingOrganizationName' but got "
            f"'{org_check.close_match}'"
        )

    def test_correct_binding_passes(self):
        """TS has existingOrganizationName, HTML uses member.existingOrganizationName → PASS."""
        html = """
        <div *ngFor="let member of projectMembers">
          <input [placeholder]="member.existingOrganizationName">
        </div>
        """
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content=html,
            html_file_path=HTML_FILE_PATH,
            written_files=_build_written_files(),
        )

        # The existingOrganizationName check should be verified
        org_checks = [
            c for c in result.checks
            if c.member == "existingOrganizationName"
        ]
        assert len(org_checks) > 0
        assert org_checks[0].status == "existing_verified", (
            f"Expected 'existing_verified' but got '{org_checks[0].status}'"
        )

    def test_controller_method_resolved(self):
        """(click)="onMemberAdd()" should resolve against controller methods."""
        html = '<button (click)="onMemberAdd()">Add</button>'
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content=html,
            html_file_path=HTML_FILE_PATH,
            written_files=_build_written_files(),
        )

        method_checks = [
            c for c in result.checks
            if c.member == "onMemberAdd"
        ]
        assert len(method_checks) > 0
        assert method_checks[0].status == "existing_verified", (
            f"Expected onMemberAdd to be resolved, got '{method_checks[0].status}'"
        )

    def test_controller_method_mismatch(self):
        """(click)="onMemberAdded()" when controller has onMemberAdd → UNRESOLVED."""
        html = '<button (click)="onMemberAdded()">Add</button>'
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content=html,
            html_file_path=HTML_FILE_PATH,
            written_files=_build_written_files(),
        )

        method_checks = [
            c for c in result.checks
            if c.member == "onMemberAdded"
        ]
        assert len(method_checks) > 0
        assert method_checks[0].status == "unresolved", (
            f"Expected 'unresolved' for onMemberAdded, got '{method_checks[0].status}'"
        )
        assert method_checks[0].close_match == "onMemberAdd"

    def test_controller_property_resolved(self):
        """*ngIf="isLoading" should resolve against controller properties."""
        html = '<div *ngIf="isLoading">Loading...</div>'
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content=html,
            html_file_path=HTML_FILE_PATH,
            written_files=_build_written_files(),
        )

        prop_checks = [
            c for c in result.checks
            if c.member == "isLoading"
        ]
        assert len(prop_checks) > 0
        assert prop_checks[0].status == "existing_verified"

    def test_unresolved_property_no_match(self):
        """member.totallyFakeProperty with no match anywhere → UNRESOLVED."""
        html = """
        <div *ngFor="let member of projectMembers">
          <span>{{ member.totallyFakeProperty }}</span>
        </div>
        """
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content=html,
            html_file_path=HTML_FILE_PATH,
            written_files=_build_written_files(),
        )

        fake_checks = [
            c for c in result.checks
            if c.member == "totallyFakeProperty"
        ]
        assert len(fake_checks) > 0
        assert fake_checks[0].status in ("unresolved", "unknown_receiver")

    def test_property_found_in_written_files(self):
        """Property exists in another written file (model) → PASS."""
        model_content = """\
export interface ProjectMember {
  id: string;
  specialProp: string;
}
"""
        written = _build_written_files(
            extra={
                "xchange-ui/src/app/models/project-member.model.ts": model_content,
            }
        )
        html = """
        <div *ngFor="let member of projectMembers">
          <span>{{ member.specialProp }}</span>
        </div>
        """
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content=html,
            html_file_path=HTML_FILE_PATH,
            written_files=written,
        )

        prop_checks = [
            c for c in result.checks
            if c.member == "specialProp"
        ]
        assert len(prop_checks) > 0
        assert prop_checks[0].status == "existing_verified"

    def test_empty_html(self):
        """Empty HTML → clean result."""
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content="",
            html_file_path=HTML_FILE_PATH,
            written_files=_build_written_files(),
        )
        assert result.is_clean

    def test_no_controller_degrades_gracefully(self):
        """HTML with no sibling controller → unknown_receiver (advisory, not blocking)."""
        html = '<span>{{ member.name }}</span>'
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content=html,
            html_file_path=HTML_FILE_PATH,
            written_files={},  # no controller
        )
        # Should not crash; advisory only
        assert isinstance(result, ReferenceValidationResult)

    def test_ngfor_type_resolution(self):
        """ngFor iterator variable type is resolved from controller declaration."""
        html = """
        <div *ngFor="let member of projectMembers">
          <span>{{ member.existingOrganizationName }}</span>
        </div>
        """
        validator = GeneratedReferenceValidator()
        result = validator.validate_template(
            html_content=html,
            html_file_path=HTML_FILE_PATH,
            written_files=_build_written_files(),
        )
        # existingOrganizationName exists in controller → should verify
        org_checks = [
            c for c in result.checks
            if c.member == "existingOrganizationName"
        ]
        assert len(org_checks) > 0
        assert org_checks[0].status == "existing_verified"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Suite 3: Existing imperative validation (backwards compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExistingValidateBackwardsCompat:
    """Ensure the existing validate() method still works unchanged."""

    def test_validate_no_resolver(self):
        """validate() with no resolver → clean result."""
        validator = GeneratedReferenceValidator()
        result = validator.validate("memberService.addMember(m)")
        assert result.is_clean

    def test_validate_empty_content(self):
        """validate() with empty content → clean result."""
        validator = GeneratedReferenceValidator(symbol_resolver=object())
        result = validator.validate("")
        assert result.is_clean
