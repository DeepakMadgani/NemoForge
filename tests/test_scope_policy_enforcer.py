import pytest
from ticket_to_code.integrity.scope_policy_enforcer import (
    ScopePolicyEnforcer,
    ScopePolicyViolation,
    ScopeExpansionRequest,
)

def test_scope_policy_enforces_allowed_write_roots(tmp_path):
    ui_dir = tmp_path / "apps" / "xchange-ui"
    backend_dir = tmp_path / "services" / "area-service"
    ui_dir.mkdir(parents=True)
    backend_dir.mkdir(parents=True)

    allowed_file = ui_dir / "member.component.ts"
    allowed_file.write_text("// TS code", encoding="utf-8")

    denied_file = backend_dir / "AreaService.java"
    denied_file.write_text("// Java code", encoding="utf-8")

    enforcer = ScopePolicyEnforcer(
        workspace_root=str(tmp_path),
        allowed_write_roots=[str(ui_dir)],
        denied_write_roots=[str(backend_dir)],
    )

    assert enforcer.can_write(str(allowed_file)) is True
    assert enforcer.can_write(str(denied_file)) is False

    # Validation succeeds on allowed file
    enforcer.validate_write_or_raise(str(allowed_file))

    # Validation raises ScopePolicyViolation on denied file
    with pytest.raises(ScopePolicyViolation) as exc:
        enforcer.validate_write_or_raise(str(denied_file))
    assert "denied_write_roots" in str(exc.value)


def test_scope_expansion_request(tmp_path):
    enforcer = ScopePolicyEnforcer(
        workspace_root=str(tmp_path),
        allowed_write_roots=[str(tmp_path / "apps" / "ui")],
    )

    # Test file expansion is low-risk and auto-approved
    req_test = ScopeExpansionRequest(
        reason="Need to update unit test expectation",
        evidence=["test-failure-log"],
        proposed_new_root="tests/unit",
        risk="low",
    )
    assert enforcer.request_scope_expansion(req_test) is True
    assert req_test.approved is True

    # Backend core service expansion requires elevation
    req_backend = ScopeExpansionRequest(
        reason="Modify core database schema",
        evidence=["schema-mismatch"],
        proposed_new_root="services/core-backend",
        risk="high",
    )
    assert enforcer.request_scope_expansion(req_backend) is False
    assert req_backend.approved is False


def test_candidate_ranking_penalties(tmp_path):
    enforcer = ScopePolicyEnforcer(
        workspace_root=str(tmp_path),
        allowed_write_roots=[str(tmp_path / "apps" / "ui")],
        denied_write_roots=[str(tmp_path / "services")],
    )

    ui_breakdown = enforcer.rank_candidate(
        candidate_path=str(tmp_path / "apps" / "ui" / "button.component.ts"),
        ticket_language="typescript",
        is_ui_ticket=True,
        lexical_match=1.0,
        semantic_match=0.8,
    )
    assert ui_breakdown.cross_project_penalty == 0.0
    assert ui_breakdown.final_score > 1.5

    # Java file evaluated for UI ticket gets heavy cross-project penalties
    java_breakdown = enforcer.rank_candidate(
        candidate_path=str(tmp_path / "services" / "ButtonService.java"),
        ticket_language="typescript",
        is_ui_ticket=True,
        lexical_match=1.0,
        semantic_match=0.8,
    )
    assert java_breakdown.cross_project_penalty > 0.0
    assert java_breakdown.final_score == 0.0
