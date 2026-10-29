"""Unit tests for the 4 advanced capabilities:
1. Behavioral Verification & Spec Gate (catches runtime-only bugs)
2. Database Schema Migration Runner (Flyway / DDL syntax & safety)
3. Fullstack Two-Stage Pipeline (Java backend + Angular frontend)
4. Clarification Gate (zero-shot novel features & human-in-the-loop decisions)
"""

from pathlib import Path
import pytest

from ticket_to_code.lean.convergence import StructuredFailure
from ticket_to_code.lean.lean_path import Hunk, LockedContext, VerifyOutcome, run_lean_path
from ticket_to_code.lean.scope import triage, ScopeFence


# ─────────────────────────────────────────────────────────────────────────────
# 1. Behavioral Verification & Spec Gate
# ─────────────────────────────────────────────────────────────────────────────

def test_behavioral_ast_invariant_catches_runtime_bug():
    from ticket_to_code.lean.behavioral_verify import (
        BehavioralContract, extract_required_calls_from_ticket, verify_ast_behavior,
    )

    ticket = (
        "When removing a coordinator, clicking Remove must cascade and invoke "
        "`removeContractMembers` and `getContractsByProjectId`."
    )
    reqs = extract_required_calls_from_ticket(ticket)
    assert "removeContractMembers" in reqs
    assert "getContractsByProjectId" in reqs

    contract = BehavioralContract(required_calls=reqs)

    # Broken implementation: compiles cleanly, but does NOT call removeContractMembers
    broken_code = """
    export class DeleteCoordinatorComponent {
        deleteCoordinator() {
            this.memberService.updateProjectUsers(this.currProject.id, this.member);
            this.modalService.close();
        }
    }
    """
    ok, err = verify_ast_behavior(broken_code, "ts", contract)
    assert not ok
    assert "removeContractMembers" in err

    # Fixed implementation: calls both methods and subscribes/pipes them
    fixed_code = """
    export class DeleteCoordinatorComponent {
        deleteCoordinator() {
            this.memberService.updateProjectUsers(this.currProject.id, this.member);
            this.contractService.getContractsByProjectId({ projectId: this.currProject.id }).pipe(
                mergeMap(contracts => {
                    return this.memberService.removeContractMembers({ members: [this.member] });
                })
            ).subscribe(() => this.modalService.close());
        }
    }
    """
    ok, err = verify_ast_behavior(fixed_code, "ts", contract)
    assert ok
    assert err is None


def test_companion_spec_discovery(tmp_path):
    from ticket_to_code.lean.behavioral_verify import find_companion_spec

    root = tmp_path / "repo"
    comp_dir = root / "src/app/project/coordinators"
    comp_dir.mkdir(parents=True)
    ts_file = comp_dir / "delete.component.ts"
    spec_file = comp_dir / "delete.component.spec.ts"
    ts_file.write_text("export class C {}", encoding="utf-8")
    spec_file.write_text("describe('C', () => {});", encoding="utf-8")

    found = find_companion_spec(root, "src/app/project/coordinators/delete.component.ts")
    assert found is not None
    assert found.name == "delete.component.spec.ts"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Database Schema Migration Runner
# ─────────────────────────────────────────────────────────────────────────────

def test_db_migration_flyway_syntax_and_safety(tmp_path):
    from ticket_to_code.lean.db_migration import (
        parse_flyway_version, validate_sql_syntax, verify_db_migration,
    )

    # Test version parsing
    assert parse_flyway_version("V1.10__AddDeliverableAudit.sql") == ((1, 10), "AddDeliverableAudit")
    assert parse_flyway_version("V2_0__CreateTenant.sql") == ((2, 0), "CreateTenant")
    assert parse_flyway_version("bad_migration.sql") is None

    # Test valid DDL syntax
    valid_sql = """
    -- Migration to add status column
    ALTER TABLE deliverables ADD COLUMN priority_level VARCHAR(32) DEFAULT 'NORMAL';
    CREATE INDEX idx_dlv_priority ON deliverables(priority_level);
    """
    assert validate_sql_syntax(valid_sql) == []

    # Test syntax error: unbalanced parentheses
    bad_syntax = "CREATE TABLE test ( id VARCHAR(36) PRIMARY KEY; "
    errs = validate_sql_syntax(bad_syntax)
    assert any("Unbalanced parentheses" in e for e in errs)

    # Test safety check: unsafe DROP TABLE
    unsafe_drop = "DROP TABLE users;"
    errs = validate_sql_syntax(unsafe_drop)
    assert any("Destructive schema operation" in e for e in errs)

    # Test safety check: ADD COLUMN NOT NULL without DEFAULT
    unsafe_not_null = "ALTER TABLE deliverables ADD COLUMN audit_code VARCHAR(16) NOT NULL;"
    errs = validate_sql_syntax(unsafe_not_null)
    assert any("without a DEFAULT clause" in e for e in errs)

    # Test full migration gate on disk
    mig_dir = tmp_path / "src/main/resources/db/migration"
    mig_dir.mkdir(parents=True)
    mig_file = mig_dir / "V1.5__AddAudit.sql"
    mig_file.write_text(valid_sql, encoding="utf-8")

    ok, failure = verify_db_migration(tmp_path, ["src/main/resources/db/migration/V1.5__AddAudit.sql"])
    assert ok
    assert failure is None

    # Test version collision detection
    colliding = mig_dir / "V1.5__Duplicate.sql"
    colliding.write_text("CREATE TABLE dup (id INT);", encoding="utf-8")
    ok_col, fail_col = verify_db_migration(tmp_path, ["src/main/resources/db/migration/V1.5__Duplicate.sql"])
    assert not ok_col
    assert "version collision" in fail_col.actual

    # Test out-of-order regression detection
    colliding.unlink()
    mig_higher = mig_dir / "V2.0__Higher.sql"
    mig_higher.write_text("CREATE TABLE higher (id INT);", encoding="utf-8")
    regressed = mig_dir / "V1.8__Regressed.sql"
    regressed.write_text("CREATE TABLE reg (id INT);", encoding="utf-8")
    ok_reg, fail_reg = verify_db_migration(tmp_path, ["src/main/resources/db/migration/V1.8__Regressed.sql"])
    assert not ok_reg
    assert "Out-of-order migration detected" in fail_reg.actual

    # Test Generic Migration Schemes (SemVer, Timestamp, Sequential)
    from ticket_to_code.lean.db_migration import parse_migration_info, suggest_next_migration_filename
    info_semver = parse_migration_info("V10.9__UpdateContractMembers.sql")
    assert info_semver is not None and info_semver.scheme == "semver" and info_semver.version_tuple == (10, 9)

    info_ts = parse_migration_info("20261012140000_create_users_table.sql")
    assert info_ts is not None and info_ts.scheme == "timestamp" and info_ts.version_tuple == (20261012140000,)

    info_seq = parse_migration_info("0042_add_status_column.sql")
    assert info_seq is not None and info_seq.scheme == "sequential" and info_seq.version_tuple == (42,)

    # Test suggest_next_migration_filename
    next_flyway = suggest_next_migration_filename(["V10.8__Foo.sql", "V10.9__Bar.sql"], "new_feature")
    assert next_flyway == "V11.0__new_feature.sql"

    next_seq = suggest_next_migration_filename(["0001_initial.py", "0002_add_users.py"], "add_roles")
    assert next_seq == "0003_add_roles.sql"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fullstack Two-Stage Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_fullstack_two_stage_pipeline_execution(tmp_path):
    from ticket_to_code.lean.fullstack_pipeline import extract_api_contract_from_java, run_fullstack_pipeline

    ws = tmp_path / "cloudforge"
    ws.mkdir()
    be_root = ws / "issues-service"
    be_root.mkdir()
    (be_root / "build.gradle").write_text("", encoding="utf-8")
    fe_root = ws / "xchange-ui"
    fe_root.mkdir()
    (fe_root / "package.json").write_text("{}", encoding="utf-8")

    # Setup Backend Java Microservice
    dto_dir = be_root / "src/main/java/com/acme/dto"
    dto_dir.mkdir(parents=True)
    dto_file = dto_dir / "IssueCount.java"
    dto_file.write_text("""
    package com.acme.dto;
    public class IssueCount {
        private int total;
        private int criticalCount;
        private int highCount;
    }
    """, encoding="utf-8")

    ctrl_dir = be_root / "src/main/java/com/acme/controller"
    ctrl_dir.mkdir(parents=True)
    ctrl_file = ctrl_dir / "IssuesController.java"
    ctrl_file.write_text("""
    package com.acme.controller;
    import org.springframework.web.bind.annotation.GetMapping;
    public class IssuesController {
        @GetMapping("/api/v1/issues/count")
        public IssueCount getCount() { return null; }
    }
    """, encoding="utf-8")

    # Setup Frontend Angular Client
    fe_app = fe_root / "src/app/deliverables"
    fe_app.mkdir(parents=True)
    fe_comp = fe_app / "deliverables.component.ts"
    fe_comp.write_text("""
    export class DeliverablesComponent {
        issueCount: any;
    }
    """, encoding="utf-8")

    # Test API contract extraction from Java
    contract = extract_api_contract_from_java(be_root, [
        "src/main/java/com/acme/dto/IssueCount.java",
        "src/main/java/com/acme/controller/IssuesController.java"
    ])
    assert contract.endpoint == "/api/v1/issues/count"
    assert contract.dto_name == "IssueCount"
    assert "criticalCount" in contract.fields
    assert "export interface IssueCount" in contract.raw_snippet

    # Test fullstack pipeline execution
    def be_patch(ctx, d, f):
        return [Hunk(ctx.files[0], "private int highCount;", "private int highCount;\n        private int mediumCount;")]

    def be_verify(ctx, w):
        return VerifyOutcome(ok=True, rung="build")

    def fe_patch(ctx, d, f):
        return [Hunk(ctx.files[0], "issueCount: any;", "issueCount: IssueCount;")]

    def fe_verify(ctx, w):
        return VerifyOutcome(ok=True, rung="build")

    fs_res = run_fullstack_pipeline(
        ticket_id="FS-TEST-1",
        title="Issue Count Severity Breakdown",
        description="Add criticalCount, highCount and mediumCount to IssueCount DTO and consume in DeliverablesComponent",
        workspace=str(ws),
        backend_patch_fn=be_patch,
        backend_verify_fn=be_verify,
        frontend_patch_fn=fe_patch,
        frontend_verify_fn=fe_verify,
        backend_service_hint="issues-service",
        frontend_service_hint="xchange-ui"
    )

    assert fs_res.status == "verified"
    assert fs_res.backend_result is not None and fs_res.backend_result.status == "verified"
    assert fs_res.frontend_result is not None and fs_res.frontend_result.status == "verified"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Clarification Gate (Zero-Shot & Novel Features)
# ─────────────────────────────────────────────────────────────────────────────

def test_clarify_gate_zero_shot_resolution():
    from ticket_to_code.lean.clarify_gate import ClarifyGate, ClarifyQuestion

    gate = ClarifyGate(resolver=lambda q: "ContractService.getContractsByProjectId")

    # Ambiguity detected: 2 candidate methods, neither has sibling usage
    q = gate.check_ambiguity(
        concept="retrieve project contracts",
        candidates=["ContractService.getContractsByProjectId", "LegacyProjectService.listContracts"],
        ticket_text="Must fetch all contracts under the project"
    )
    assert q is not None
    assert len(q.options) == 2

    # Resolve decision
    chosen = gate.resolve(q)
    assert chosen == "ContractService.getContractsByProjectId"

    # Verify ground truth formatted string
    gt = gate.format_ground_truth_context()
    assert "CONFIRMED ARCHITECTURAL DECISIONS" in gt
    assert "ContractService.getContractsByProjectId" in gt
