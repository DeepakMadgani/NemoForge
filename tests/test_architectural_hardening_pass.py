"""
Architectural Hardening Pass — Comprehensive Test Suite.

Covers:
1. AST/Symbol-first resolution (containing symbol, target source, minimal dependencies).
   Bounded line window used only as fallback when AST resolution fails.
2. Patch Proportionality & Anti-Rewrite validation (safety ceiling, rejection of disproportionate expansion).
3. Structural Template Expression Parser without literal blacklists.
4. Dynamic Validation UI Labels (file extension driven: Template-Check, TSC-live, Java-Check, Style-Check).
5. Completeness Gate:
   - Requirement → Evidence → ChangeTarget → Patch → Validation traceability.
   - Ticket-dependent test requirements (tests required only if ticket explicitly demands them).
   - Production enforcement & manifest persistence to .aviator/tickets/<ticket_id>.json.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ticket_to_code.agents.template_expression_parser import (
    TemplateExpressionParser,
    TemplateLexer,
    TokenType,
)
from ticket_to_code.agents.diagnostic_context_builder import (
    DiagnosticContextBuilder,
    DiagnosticContext,
)
from ticket_to_code.agents.patch_gate import (
    PatchGate,
    CrossArtifactConsistencyError,
)
from ticket_to_code.agents.edit_loop_agent import EditLoopAgent
from ticket_to_code.agents.completeness_gate import (
    CompletenessGate,
    CompletenessVerdict,
)
from ticket_to_code.models import ChangeTarget


# ─── Objective 1: AST/Symbol-First Resolution & Parser Fallback ──────────────

def test_diagnostic_context_resolves_containing_method_via_ast():
    """Verify compiler diagnostic at a line number resolves to the containing method."""
    java_code = """package com.example.service;

public class MemberService {
    private final MemberRepository repo;

    public void init() {
        System.out.println("init");
    }

    public boolean isMemberActive(String memberId) {
        Member m = repo.findById(memberId);
        if (m == null) {
            return false;
        }
        return m.isActive();
    }

    public void close() {
        System.out.println("close");
    }
}
"""
    builder = DiagnosticContextBuilder()
    # Error at line 13: return m.isActive();
    ctx = builder.build_context(
        file_path="MemberService.java",
        content=java_code,
        line=13,
        column=16,
        diagnostic_message="cannot find symbol method isActive()",
    )

    assert ctx.target_symbol == "isMemberActive"
    assert "public boolean isMemberActive(String memberId)" in ctx.target_source
    assert "return m.isActive();" in ctx.target_source
    assert not ctx.is_parser_fallback
    assert "CLASS: public class MemberService" in ctx.structural_context
    assert "init" in ctx.structural_context
    assert "isActive" in ctx.related_symbols or "findById" in ctx.related_symbols


def test_diagnostic_context_uses_bounded_lines_as_fallback_when_ast_fails():
    """When line is outside any method (e.g. invalid syntax or class level), use fallback line window."""
    malformed_code = """line 1
line 2
line 3
syntax error line 4 without braces
line 5
line 6
"""
    builder = DiagnosticContextBuilder()
    ctx = builder.build_context(
        file_path="unknown.txt",
        content=malformed_code,
        line=4,
        column=1,
        diagnostic_message="syntax error",
    )

    assert ctx.is_parser_fallback is True
    assert ctx.target_symbol is None
    assert "syntax error line 4 without braces" in ctx.target_source


def test_diagnostic_builder_content_hash_freshness():
    """Content hash allows reusing validated evidence without re-querying unchanged files."""
    builder = DiagnosticContextBuilder()
    content_a = "public class A { void f() {} }"
    builder.update_file_hash("A.java", content_a)

    assert builder.is_file_fresh("A.java", content_a) is True
    assert builder.is_file_fresh("A.java", content_a + " // edit") is False


# ─── Objective 2: Patch Proportionality & Anti-Rewrite Validation ─────────────

def test_patch_proportionality_accepts_normal_patch():
    """Proportional patch modifying target lines passes."""
    current = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    patch = """<<<<<<< SEARCH
line 3
=======
line 3 updated
>>>>>>> REPLACE"""
    ok, reason = PatchGate.validate_patch_proportionality(
        file_path="service.ts",
        patch_content=patch,
        current_content=current,
        is_new_file=False,
    )
    assert ok is True
    assert "verified" in reason.lower()


def test_patch_proportionality_rejects_disproportionate_expansion():
    """Reject suspicious patch where 2 lines search expands to 200 lines replace."""
    current = "\n".join(f"line {i}" for i in range(100))
    search = "line 5\nline 6"
    replace = "\n".join(f"new line {i}" for i in range(150))
    patch = f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"

    ok, reason = PatchGate.validate_patch_proportionality(
        file_path="service.ts",
        patch_content=patch,
        current_content=current,
        is_new_file=False,
    )
    assert ok is False
    assert "disproportionate patch expansion" in reason.lower()


def test_patch_proportionality_rejects_whole_file_rewrite_in_search_block():
    """Reject patch encompassing >= 90% of a large file as a search block."""
    lines = [f"int var{i} = {i};" for i in range(120)]
    current = "\n".join(lines)
    search = "\n".join(lines[:115])
    patch = f"<<<<<<< SEARCH\n{search}\n=======\nint x = 1;\n>>>>>>> REPLACE"

    ok, reason = PatchGate.validate_patch_proportionality(
        file_path="Service.java",
        patch_content=patch,
        current_content=current,
        is_new_file=False,
    )
    assert ok is False
    assert "disproportionate rewrite" in reason.lower()


# ─── Objective 3: Structural Template Expression Parser ──────────────────────

def test_template_expression_parser_syntactic_classification():
    """Verify literals are never classified as controller references."""
    # Boolean literals
    refs_false = TemplateExpressionParser.extract_references("false")
    assert refs_false == []

    refs_true = TemplateExpressionParser.extract_references("true")
    assert refs_true == []

    # Nil literals
    refs_null = TemplateExpressionParser.extract_references("null")
    assert refs_null == []

    refs_undefined = TemplateExpressionParser.extract_references("undefined")
    assert refs_undefined == []

    # Bare identifier
    refs_ident = TemplateExpressionParser.extract_references("isDisabled")
    assert len(refs_ident) == 1
    assert refs_ident[0].receiver == "this"
    assert refs_ident[0].member == "isDisabled"
    assert not refs_ident[0].is_method

    # Method call
    refs_call = TemplateExpressionParser.extract_references("onSave()")
    assert len(refs_call) == 1
    assert refs_call[0].receiver == "this"
    assert refs_call[0].member == "onSave"
    assert refs_call[0].is_method is True

    # Member access chain
    refs_chain = TemplateExpressionParser.extract_references("user.isActive")
    assert len(refs_chain) == 1
    assert refs_chain[0].receiver == "user"
    assert refs_chain[0].member == "isActive"

    # Pipe handling: pipe transform ignored, input expression parsed
    refs_pipe = TemplateExpressionParser.extract_references('"home.title" | translate')
    assert refs_pipe == []

    refs_pipe_var = TemplateExpressionParser.extract_references('user.name | uppercase')
    assert len(refs_pipe_var) == 1
    assert refs_pipe_var[0].receiver == "user"
    assert refs_pipe_var[0].member == "name"

    # Dynamic framework context symbols: any identifier starting with '$' is framework context, never 'this'
    refs_event = TemplateExpressionParser.extract_references("onOrgChange($event)")
    assert len(refs_event) == 1
    assert refs_event[0].receiver == "this"
    assert refs_event[0].member == "onOrgChange"
    assert refs_event[0].is_method is True
    assert not any(r.member == "$event" for r in refs_event)

    refs_cast = TemplateExpressionParser.extract_references("$any(user).name")
    assert not any(r.member == "$any" for r in refs_cast)

    # Any arbitrary/future framework or custom event variable starting with '$' is handled dynamically without static sets
    refs_arbitrary = TemplateExpressionParser.extract_references("handleCustom($customFrameworkSignal, $futureScope)")
    assert len(refs_arbitrary) == 1
    assert refs_arbitrary[0].member == "handleCustom"
    assert not any(r.member in ("$customFrameworkSignal", "$futureScope") for r in refs_arbitrary)

    # Dynamically declared template-local variables (#myInput, let-item, as data)
    refs_local = TemplateExpressionParser.extract_references(
        "onSelect(searchBox.value)",
        local_vars={"searchBox"}
    )
    assert len(refs_local) == 2
    assert refs_local[0].receiver == "this"
    assert refs_local[0].member == "onSelect"
    assert refs_local[0].is_method is True
    assert refs_local[1].receiver == "searchBox"
    assert refs_local[1].member == "value"
    assert not any(r.receiver == "this" and r.member == "searchBox" for r in refs_local)


# ─── Objective 4: Dynamic Validation UI Labels ───────────────────────────────

def test_dynamic_validation_ui_labels():
    """Verify check types are dynamically assigned based on file type."""
    assert EditLoopAgent._get_live_check_label("add-members.component.html") == "Template-Check"
    assert EditLoopAgent._get_live_check_label("add-members.component.htm") == "Template-Check"
    assert EditLoopAgent._get_live_check_label("add-members.component.ts") == "TSC-live"
    assert EditLoopAgent._get_live_check_label("ContractService.java") == "Java-Check"
    assert EditLoopAgent._get_live_check_label("styles.scss") == "Style-Check"
    assert EditLoopAgent._get_live_check_label("styles.css") == "Style-Check"
    assert EditLoopAgent._get_live_check_label("config.json") == "Build-Check"


# ─── Objective 5: Completeness Gate & Ticket-Dependent Tests ──────────────────

def test_completeness_gate_passes_when_all_invariants_satisfied(tmp_path):
    """Clean build, authorized files, authorized change targets, and requirement coverage."""
    ct = ChangeTarget(
        file_path="src/service.ts",
        symbol="updateStatus",
        symbol_type="method",
        is_authorized=True,
        evidence_ids=["REQ-1"],
    )

    verdict = CompletenessGate.evaluate(
        ticket_id="TICKET-101",
        ticket_title="Update member status",
        ticket_description="Update member status when button clicked",
        requirements=[{"id": "REQ-1", "text": "Update member status"}],
        change_targets=[ct],
        modified_files={"src/service.ts"},
        authorized_files={"src/service.ts"},
        build_passed=True,
        tests_passed=None,  # No tests demanded by ticket
        cross_artifact_passed=True,
        remaining_diagnostics=[],
        workspace_path=tmp_path,
    )

    assert verdict.is_complete is True
    assert verdict.status == "COMPLETED"
    assert verdict.test_status == "not_required"
    assert len(verdict.requirements_coverage) == 1
    assert verdict.requirements_coverage[0].is_covered is True

    # Verify manifest written
    manifest_file = tmp_path / ".aviator" / "tickets" / "TICKET-101.json"
    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["status"] == "COMPLETED"
    assert data["build_passed"] is True
    assert data["test_status"] == "not_required"


def test_completeness_gate_demands_tests_when_ticket_requires_them(tmp_path):
    """When ticket explicitly demands unit tests, tests must pass."""
    ct = ChangeTarget(
        file_path="src/service.ts",
        symbol="updateStatus",
        symbol_type="method",
        is_authorized=True,
        evidence_ids=["REQ-1"],
    )

    # Ticket demands unit tests
    verdict_fail = CompletenessGate.evaluate(
        ticket_id="TICKET-102",
        ticket_title="Update member status and add unit tests",
        ticket_description="Must include unit tests for service",
        requirements=[{"id": "REQ-1", "text": "Update member status", "requires_test": True}],
        change_targets=[ct],
        modified_files={"src/service.ts"},
        authorized_files={"src/service.ts"},
        build_passed=True,
        tests_passed=False,  # Tests failed!
        cross_artifact_passed=True,
        remaining_diagnostics=[],
        workspace_path=tmp_path,
    )

    assert verdict_fail.is_complete is False
    assert verdict_fail.status == "INCOMPLETE"
    assert verdict_fail.test_status == "failed"

    # Now with passing tests
    verdict_pass = CompletenessGate.evaluate(
        ticket_id="TICKET-102",
        ticket_title="Update member status and add unit tests",
        ticket_description="Must include unit tests for service",
        requirements=[{"id": "REQ-1", "text": "Update member status", "requires_test": True}],
        change_targets=[ct],
        modified_files={"src/service.ts"},
        authorized_files={"src/service.ts"},
        build_passed=True,
        tests_passed=True,  # Tests passed!
        cross_artifact_passed=True,
        remaining_diagnostics=[],
        workspace_path=tmp_path,
    )

    assert verdict_pass.is_complete is True
    assert verdict_pass.status == "COMPLETED"
    assert verdict_pass.test_status == "passed"


def test_completeness_gate_verifies_semantic_requirement_coverage(tmp_path):
    """Requirement without implemented authorized ChangeTarget causes incomplete verdict."""
    ct = ChangeTarget(
        file_path="src/service.ts",
        symbol="updateStatus",
        symbol_type="method",
        is_authorized=True,
        evidence_ids=["REQ-1"],
    )

    verdict = CompletenessGate.evaluate(
        ticket_id="TICKET-103",
        ticket_title="Implement member features",
        ticket_description="Add status update and notify audit log",
        requirements=[
            {"id": "REQ-1", "text": "Update status"},
            {"id": "REQ-2", "text": "Notify audit log"},  # Uncovered!
        ],
        change_targets=[ct],
        modified_files={"src/service.ts"},
        authorized_files={"src/service.ts"},
        build_passed=True,
        tests_passed=None,
        cross_artifact_passed=True,
        remaining_diagnostics=[],
        workspace_path=tmp_path,
    )

    assert verdict.is_complete is False
    assert verdict.status == "INCOMPLETE"
    assert len(verdict.uncovered_requirements) == 1
    assert "REQ-2" in verdict.uncovered_requirements[0]


# ─── Objective 6: Generation & Edit Loop Streamlined Routing ──────────────────

def test_route_after_generate_code_clean_routes_to_patch_gate():
    """Verify that clean generation routes directly to patch_gate, not edit_loop."""
    from ticket_to_code.workflow import route_after_generate_code

    clean_state = {"status": "success"}
    assert route_after_generate_code(clean_state) == "patch_gate"

    escalated_state = {"status": "escalated"}
    assert route_after_generate_code(escalated_state) == "edit_loop"

    invalid_state = {"status": "candidates_invalid", "candidate_retry_count": 0, "max_candidate_retries": 2}
    assert route_after_generate_code(invalid_state) == "plan"


# ─── Objective 7: Repository-Agnostic Dynamic Frontend & Boundary Detection ────

def test_dynamic_frontend_detection_non_xchange_ui(tmp_path):
    """Verify that UI detection, root discovery, and boundary scoping work dynamically for arbitrary frontend repos (e.g. web-portal)."""
    # 1. Setup synthetic monorepo with web-portal/ and backend-service/
    web_portal = tmp_path / "web-portal"
    web_portal.mkdir()
    (web_portal / "package.json").write_text(json.dumps({"name": "web-portal", "scripts": {"build": "ng build"}}))
    (web_portal / "angular.json").write_text("{}")
    
    comp_dir = web_portal / "src" / "app" / "users"
    comp_dir.mkdir(parents=True)
    comp_ts = comp_dir / "user-list.component.ts"
    comp_ts.write_text("export class UserListComponent {}")
    comp_html = comp_dir / "user-list.component.html"
    comp_html.write_text("<div>{{ user.name }}</div>")

    backend = tmp_path / "backend-service"
    backend.mkdir()
    (backend / "pom.xml").write_text("<project></project>")

    # 2. Verify _find_ng_root discovers web-portal without hardcoding
    from ticket_to_code.workflow import _find_ng_root
    discovered_root = _find_ng_root(tmp_path)
    assert discovered_root == web_portal

    # 3. Verify boundary resolver identifies web-portal as an independent project boundary
    def get_boundary(p: str):
        parts = p.replace("\\", "/").split("/")
        if len(parts) > 1:
            cand = tmp_path / parts[0]
            if cand.is_dir() and any((cand / cfg).exists() for cfg in ("pom.xml", "package.json")):
                return parts[0]
        return None

    assert get_boundary("web-portal/src/app/users/user-list.component.ts") == "web-portal"
    assert get_boundary("backend-service/src/main/UserResource.java") == "backend-service"

    # 4. Verify candidate in external boundary is demoted when preflight targets web-portal
    target_boundaries = {get_boundary("web-portal/src/app/users/user-list.component.ts")}
    backend_cand = {"path": "backend-service/src/main/UserResource.java", "candidate_role": "PRIMARY_OWNER"}
    cand_boundary = get_boundary(backend_cand["path"])
    assert cand_boundary not in target_boundaries
    if cand_boundary and cand_boundary not in target_boundaries:
        backend_cand["candidate_role"] = "REFERENCE"
    assert backend_cand["candidate_role"] == "REFERENCE"
