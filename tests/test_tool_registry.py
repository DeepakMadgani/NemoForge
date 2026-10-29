"""
Unit tests for UnifiedToolRegistry and HighDensitySanitizer.
Verifies all grounding tools, output density sanitization, and automated blackboard deduction updates.
"""

import tempfile
from pathlib import Path

from ticket_to_code.runtime.deductive_blackboard import DeductiveBlackboard
from ticket_to_code.runtime.tool_registry import HighDensitySanitizer, UnifiedToolRegistry


def test_high_density_sanitizer_caps_output():
    huge_text = "\n".join(f"Line {i}: test data content here" for i in range(500))
    sanitized = HighDensitySanitizer.sanitize(huge_text, max_chars=1000, max_lines=50)
    assert len(sanitized) <= 1200
    assert "Truncated" in sanitized


def test_unified_tool_registry_grep_and_inspect(tmp_path: Path):
    # Setup test workspace
    svc_dir = tmp_path / "issues-service" / "src" / "main" / "java"
    svc_dir.mkdir(parents=True)
    issue_file = svc_dir / "Issue.java"
    issue_file.write_text("""
    package com.cloudforge.services.issues;
    public class Issue {
        private String priority;
        public String getPriority() {
            return priority;
        }
    }
    """, encoding="utf-8")

    bb = DeductiveBlackboard(workspace_path=str(tmp_path))
    registry = UnifiedToolRegistry(workspace_path=str(tmp_path), blackboard=bb)

    # 1. Test grep_search
    grep_res = registry.grep_search("priority")
    assert "Issue.java" in grep_res
    assert "String priority" in grep_res

    # Verify blackboard automatically deduced facts from grep
    assert any("priority" in f and "String" in f for f in bb.verified_facts)
    assert any("IssuePriority" in r for r in bb.rejected_hypotheses)

    # 2. Test inspect_file with symbol
    inspect_sym = registry.inspect_file("issues-service/src/main/java/Issue.java", symbol="getPriority")
    assert "getPriority" in inspect_sym

    # 3. Test inspect_file outline
    inspect_out = registry.inspect_file("issues-service/src/main/java/Issue.java", symbol="outline")
    assert "Structural Outline" in inspect_out


def test_unified_tool_registry_file_mutation(tmp_path: Path):
    test_file = tmp_path / "Test.java"
    test_file.write_text("public class Test { int x = 1; }", encoding="utf-8")

    bb = DeductiveBlackboard(workspace_path=str(tmp_path))
    registry = UnifiedToolRegistry(workspace_path=str(tmp_path), blackboard=bb)

    # Replace content
    res = registry.replace_file_content("Test.java", "int x = 1;", "int x = 2;")
    assert "Successfully updated" in res
    assert test_file.read_text(encoding="utf-8") == "public class Test { int x = 2; }"

    # Target not found safeguard
    err_res = registry.replace_file_content("Test.java", "nonexistent_code", "replacement")
    assert "TargetContent not found" in err_res


def test_all_tools_declared():
    registry = UnifiedToolRegistry(workspace_path=".")
    tool_names = [
        "grep_search",
        "find_files",
        "list_directory",
        "inspect_file",
        "find_references",
        "trace_call_chain",
        "inspect_database_schema",
        "inspect_service_contract",
        "replace_file_content",
        "write_to_file",
        "run_command",
        "verify_service_compilation",
        "run_service_unit_tests",
        "verify_live_ui_with_playwright",
        "request_structured_clarification",
    ]
    for tname in tool_names:
        assert hasattr(registry, tname), f"UnifiedToolRegistry missing tool: {tname}"


def test_inspect_file_skips_import_and_finds_usage(tmp_path):
    f = tmp_path / "IssuesService.java"
    content = (
        "package com.cloudforge.service;\n"
        "import com.cloudforge.exceptions.InvalidInputException;\n"
        "import java.util.List;\n"
        "\n"
        "public class IssuesService {\n"
        "    // Lots of boilerplate\n"
        "    public void doWork() {\n"
        "        if (true) {\n"
        "            throw new InvalidInputException(\"SingleArgumentCall\");\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    f.write_text(content, encoding="utf-8")
    reg = UnifiedToolRegistry(workspace_path=str(tmp_path))
    res = reg.inspect_file("IssuesService.java", symbol="InvalidInputException")
    # Verify it targets the throw usage line, not merely the import line
    assert "throw new InvalidInputException" in res
    assert "SingleArgumentCall" in res


def test_find_references_orders_call_sites_before_imports(tmp_path):
    f1 = tmp_path / "A.java"
    f1.write_text("import com.cloudforge.CustomException;\n", encoding="utf-8")
    f2 = tmp_path / "B.java"
    f2.write_text("throw new CustomException(\"boom\");\n", encoding="utf-8")
    reg = UnifiedToolRegistry(workspace_path=str(tmp_path))
    res = reg.find_references("CustomException")
    lines = [l for l in res.splitlines() if l.strip()]
    # B.java (call site) must appear before A.java (import statement)
    b_idx = next(i for i, l in enumerate(lines) if "B.java" in l)
    a_idx = next(i for i, l in enumerate(lines) if "A.java" in l)
    assert b_idx < a_idx


def test_request_structured_clarification_boundary_rejects_existing_and_approves_absent(tmp_path):
    # 1. File with existing symbol
    f = tmp_path / "ExistingService.java"
    f.write_text("public class ExistingService { void doAction() {} }\n", encoding="utf-8")
    
    # 2. SQL schema file
    sql = tmp_path / "V1__create_issues.sql"
    sql.write_text("CREATE TABLE issue_records (id VARCHAR(36));\n", encoding="utf-8")
    
    reg = UnifiedToolRegistry(workspace_path=str(tmp_path))
    
    # Attempt clarification on existing code symbol -> REJECTED
    res_code = reg.request_structured_clarification(
        missing_concept="ExistingService",
        option_a="Option A",
        option_b="Option B",
        rationale="Unsure about usage"
    )
    assert "🛑 Clarification REJECTED by Strict 0-Interruption Escalation Boundary" in res_code
    assert "active references/call-sites in the codebase" in res_code
    
    # Attempt clarification on table that exists in DB migrations -> REJECTED
    res_db = reg.request_structured_clarification(
        missing_concept="issue_records",
        option_a="Option A",
        option_b="Option B",
        rationale="Unsure about schema"
    )
    assert "🛑 Clarification REJECTED by Strict 0-Interruption Escalation Boundary" in res_db
    assert "exists in database migrations" in res_db
    
    # Concept with verified external absence (0 occurrences in repo and 0 in DB) -> APPROVED
    res_absent = reg.request_structured_clarification(
        missing_concept="CompletelyNovelExternalProtocol",
        option_a="Option A: Use gRPC",
        option_b="Option B: Use WebSocket",
        rationale="Novel 3rd party integration with zero references"
    )
    assert "✅ Tier 3 Structured Clarification Gate APPROVED" in res_absent
    assert "Option A: Use gRPC" in res_absent
    assert "Option B: Use WebSocket" in res_absent


def test_inspect_database_schema_structural_cross_reference(tmp_path):
    # 1. Create a migration file with a specific naming scheme (SemVer)
    mig_dir = tmp_path / "project-service" / "src" / "main" / "resources" / "db" / "migration"
    mig_dir.mkdir(parents=True)
    sql_file = mig_dir / "V10.9__add_project_members.sql"
    sql_file.write_text("""
    CREATE TABLE project_members (
        id VARCHAR(36) PRIMARY KEY,
        otds_user_id VARCHAR(64) NOT NULL,
        project_id VARCHAR(36) NOT NULL,
        joined_at TIMESTAMP
    );
    """, encoding="utf-8")

    # 2. Create an entity model with a field mapping discrepancy
    model_dir = tmp_path / "project-service" / "src" / "main" / "java" / "com" / "cloudforge" / "models"
    model_dir.mkdir(parents=True)
    entity_file = model_dir / "ProjectMember.java"
    entity_file.write_text("""
    package com.cloudforge.models;
    import javax.persistence.*;

    @Entity
    @Table(name = "project_members")
    public class ProjectMember {
        @Id
        private String id;

        @Column(name = "otds_user_id")
        private String userId;

        @Column(name = "project_id")
        private String projectId;
    }
    """, encoding="utf-8")

    reg = UnifiedToolRegistry(workspace_path=str(tmp_path))
    schema_report = reg.inspect_database_schema(service_name="project-service", table_name="project_members")

    # Verify migration scheme discovery
    assert "Database Migration Scheme & Versioning" in schema_report
    assert "V11.0__" in schema_report

    # Verify DDL column parsing
    assert "`otds_user_id`: VARCHAR(64) NOT NULL" in schema_report
    assert "`id`: VARCHAR(36) [PRIMARY KEY]" in schema_report

    # Verify entity cross-referencing and join trap warning
    assert "Entity-to-Schema Cross-Reference" in schema_report
    assert "ProjectMember.java" in schema_report
    assert "Field `userId` -> Column `otds_user_id`" in schema_report
    assert "JOIN / COLUMN TRAP" in schema_report

