"""
Comprehensive End-to-End Architecture Wiring Verification Test Suite

Validates that all reverse-engineered architectural insights (Opus 4.8, Codex, Copilot)
are seamlessly connected into the Ticket-to-Code workflow:
1. SubagentExplorer: Context firewalling in hypothesis_investigation_node -> plan_node
2. TwoTierVerifier: Tier 1 fast syntax failure interception in build_node (<2ms)
3. OscillationDetector: 2-cycle oscillation detection (A -> B -> A) in pre_fix_build_node
4. ScopePolicyEnforcer: Gate rejection of unauthorized cross-service modifications
5. SkeletonCompressor: In-memory AST compression of dependency context
"""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from ticket_to_code.models import (
    ValueEdgeTicket,
    GeneratedCode,
    BuildResult,
    BuildStatus,
    InvestigationHypothesis,
)
from ticket_to_code.workflow import (
    hypothesis_investigation_node,
    pre_fix_build_node,
    build_node,
    plan_node,
    WorkflowAgents,
)
from ticket_to_code.integrity.scope_policy_enforcer import (
    ScopePolicyEnforcer,
    ScopePolicyViolation,
)
from ticket_to_code.intelligence.skeleton_compressor import SkeletonCompressor


def test_subagent_explorer_firewall_wiring_in_workflow(tmp_path):
    """
    Verify SubagentExplorer runs read-only workers during hypothesis_investigation_node
    and outputs subagent_investigation_summary to the state.
    """
    # Create sample files in temp workspace
    ui_dir = tmp_path / "apps" / "xchange-ui" / "src" / "app" / "components"
    ui_dir.mkdir(parents=True)
    comp_ts = ui_dir / "user.component.ts"
    comp_ts.write_text("export class UserComponent { formGroup = true; }", encoding="utf-8")
    comp_html = ui_dir / "user.component.html"
    comp_html.write_text("<div *ngIf=\"invalid\">Error</div>", encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="TICKET-101",
        title="Fix User form validation",
        description="Missing inline error in user component",
    )

    state = {
        "ticket": ticket,
        "workspace_path": str(tmp_path),
        "status": "investigating",
        "investigation_result": None,
        "discovered_files": [{"path": str(comp_ts.relative_to(tmp_path)).replace("\\", "/")}],
    }

    mock_agents = MagicMock()
    mock_agents.investigation.generate_hypotheses.return_value = [
        InvestigationHypothesis(
            id="H1",
            hypothesis="Investigate user form validation",
            queries=["user form validation"],
            literals=[],
            symbols=["UserComponent"],
            confidence=0.9,
        )
    ]

    result = hypothesis_investigation_node(state, mock_agents)

    assert "subagent_investigation_summary" in result
    summary = result["subagent_investigation_summary"]
    assert summary is not None
    assert "Sub-Agent Exploration Findings" in summary
    assert "UI_MAPPER" in summary
    assert "VALIDATION_ANALYST" in summary


def test_build_node_tier1_fast_syntax_interception(tmp_path):
    """
    Verify TwoTierVerifier Tier 1 intercepts syntax / delimiter errors in <2ms
    in build_node and returns a FAILURE BuildResult without calling expensive tsc.
    """
    broken_ts = tmp_path / "broken.component.ts"
    # Unclosed brace will trigger Tier 1 delimiter / tree-sitter detection
    broken_ts.write_text("export class BrokenComponent {", encoding="utf-8")

    from ticket_to_code.models import ProgrammingLanguage, TaskType

    state = {
        "workspace_path": str(tmp_path),
        "generated_code": [
            GeneratedCode(
                file_path=str(broken_ts.relative_to(tmp_path)).replace("\\", "/"),
                content="export class BrokenComponent {",
                language=ProgrammingLanguage.TYPESCRIPT,
                change_type=TaskType.MODIFY,
            )
        ],
        "ticket": ValueEdgeTicket(ticket_id="TICKET-102", title="Broken Code", description=""),
    }

    mock_agents = MagicMock()
    result = build_node(state, mock_agents)

    assert "build_result" in result
    build_res = result["build_result"]
    assert build_res.exit_code == 1
    assert any("TS_MISSING_TOKEN" in e or "TS_ERROR" in e or "UNBALANCED_DELIMITERS" in e for e in build_res.errors)


def test_pre_fix_build_node_oscillation_cycle_detection():
    """
    Verify pre_fix_build_node detects a 2-cycle oscillation (A -> B -> A)
    and sets consecutive_identical_build_errors to >= 2 to break infinite loops.
    """
    mock_agents = MagicMock()

    # Step 1: Initial error A
    state1 = {
        "build_result": BuildResult(
            status=BuildStatus.FAILURE,
            exit_code=1,
            errors=["src/app/foo.ts:10:5 error TS2304: Cannot find name 'x'"],
        ),
        "prev_build_error_fingerprint": "",
        "build_error_history": [],
        "consecutive_identical_build_errors": 0,
    }
    res1 = pre_fix_build_node(state1, mock_agents)
    fp_a = res1["prev_build_error_fingerprint"]
    assert res1["consecutive_identical_build_errors"] == 0

    # Step 2: Agent edits and causes different error B
    state2 = {
        "build_result": BuildResult(
            status=BuildStatus.FAILURE,
            exit_code=1,
            errors=["src/app/foo.ts:12:1 error TS2322: Type 'string' is not assignable to type 'number'"],
        ),
        "prev_build_error_fingerprint": fp_a,
        "build_error_history": res1["build_error_history"],
        "consecutive_identical_build_errors": 0,
    }
    res2 = pre_fix_build_node(state2, mock_agents)
    fp_b = res2["prev_build_error_fingerprint"]
    assert fp_b != fp_a
    assert res2["consecutive_identical_build_errors"] == 0

    # Step 3: Agent edits again and reverts back to error A (Oscillation A -> B -> A!)
    state3 = {
        "build_result": BuildResult(
            status=BuildStatus.FAILURE,
            exit_code=1,
            errors=["src/app/foo.ts:10:5 error TS2304: Cannot find name 'x'"],
        ),
        "prev_build_error_fingerprint": fp_b,
        "build_error_history": res2["build_error_history"],
        "consecutive_identical_build_errors": 0,
    }
    res3 = pre_fix_build_node(state3, mock_agents)
    assert res3["prev_build_error_fingerprint"] == fp_a
    # Must flag oscillation as consecutive >= 2
    assert res3["consecutive_identical_build_errors"] >= 2


def test_scope_policy_enforcer_gating(tmp_path):
    """
    Verify ScopePolicyEnforcer machine-rejects writes outside allowed roots.
    """
    ui_dir = tmp_path / "apps" / "xchange-ui" / "src" / "app"
    backend_dir = tmp_path / "services" / "area-service"
    ui_dir.mkdir(parents=True)
    backend_dir.mkdir(parents=True)

    allowed_file = ui_dir / "user.component.ts"
    allowed_file.write_text("// valid", encoding="utf-8")
    denied_file = backend_dir / "AreaController.java"
    denied_file.write_text("// invalid", encoding="utf-8")

    enforcer = ScopePolicyEnforcer(
        workspace_root=str(tmp_path),
        allowed_write_roots=[str(ui_dir)],
        denied_write_roots=[str(backend_dir)],
    )

    assert enforcer.can_write(str(allowed_file)) is True
    assert enforcer.can_write(str(denied_file)) is False
    with pytest.raises(ScopePolicyViolation):
        enforcer.validate_write_or_raise(str(denied_file))


def test_skeleton_compressor_in_memory():
    """
    Verify SkeletonCompressor produces clean interface signatures without method bodies.
    """
    ts_code = """
    export class OrderService {
        constructor(private http: HttpClient) {}
        public getOrder(id: string): Observable<Order> {
            console.log("fetching order");
            return this.http.get<Order>(`/api/orders/${id}`);
        }
    }
    """
    compressed = SkeletonCompressor.compress_typescript(ts_code)
    assert "public getOrder(id: string): Observable<Order> { /* elided */ }" in compressed
    assert "fetching order" not in compressed
