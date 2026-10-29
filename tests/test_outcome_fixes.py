"""Tests for the architectural fixes to the outcome-check pipeline.

Covers:
  1. route_after_outcome_check terminates honestly (no false-success mask).
  2. _build_remediation_plan creates only targeted tasks from findings.
  3. _extract_json_object tolerates prose, nested objects, and fenced blocks.
"""
import pytest
from unittest.mock import MagicMock

from ticket_to_code.models import OutcomeFinding

# ---------------------------------------------------------------------------
# 1. _extract_json_object
# ---------------------------------------------------------------------------


def test_extract_json_from_fenced_block():
    text = """Some prose here.
```json
{"verdict": "PARTIAL", "findings": [{"id": 1}]}
```
More prose."""
    from ticket_to_code.workflow import _extract_json_object
    result = _extract_json_object(text)
    assert result is not None
    assert result["verdict"] == "PARTIAL"
    assert len(result["findings"]) == 1


def test_extract_json_from_bare_object():
    text = (
        "The verdict is PARTIAL because of "
        '{"verdict": "PARTIAL", "findings": []} and some more prose'
    )
    from ticket_to_code.workflow import _extract_json_object
    result = _extract_json_object(text)
    assert result is not None
    assert result["verdict"] == "PARTIAL"


def test_extract_json_nested_object():
    text = (
        "Result: " + '{"verdict": "PARTIAL", "findings": [{"nested": {"deep": true}}]}'
    )
    from ticket_to_code.workflow import _extract_json_object
    result = _extract_json_object(text)
    # Uses balanced-brace scan
    assert result is not None
    assert result["verdict"] == "PARTIAL"


def test_extract_json_no_object():
    from ticket_to_code.workflow import _extract_json_object
    assert _extract_json_object("No JSON here at all.") is None
    assert _extract_json_object("") is None
    assert _extract_json_object(None) is None


def test_extract_json_with_string_containing_braces():
    # String literal with a brace must not desync the balanced scan
    text = (
        '{"verdict": "PARTIAL", "summary": "a } b", "findings": []}'
    )
    from ticket_to_code.workflow import _extract_json_object
    result = _extract_json_object(text)
    assert result is not None
    assert result["verdict"] == "PARTIAL"


# ---------------------------------------------------------------------------
# 2. _build_remediation_plan
# ---------------------------------------------------------------------------


def test_build_remediation_plan_from_findings():
    """Findings for add-members.component.ts should produce one MODIFY task."""
    from ticket_to_code.workflow import _build_remediation_plan

    state = {
        "outcome_findings": [
            OutcomeFinding(
                verdict="PARTIAL",
                requirement_id="REQ-1",
                requirement_text="Check project membership",
                summary="Only handles single selection",
                offending_code="searchData.length === 1",
                affected_file="add-members.component.ts",
                affected_line=262,
                missing_behavior="Use searchData.length > 0 to support multi-select",
                next_action="Fix the quantifier in onUserSelect",
            ).to_dict(),
        ],
        "outcome_check_result": {"status": "PARTIAL", "details": "", "findings": []},
        "generated_code": [
            MagicMock(file_path="src/app/modules/members/add-members/add-members.component.ts", content=""),
        ],
        "change_targets": ["src/app/modules/members/add-members/add-members.component.ts"],
    }

    plan = _build_remediation_plan(state)
    assert plan is not None
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.task_type.value == "modify"
    assert "add-members.component.ts" in task.file_path
    assert "multi-select" in task.description or "length > 0" in task.description


def test_build_remediation_plan_returns_none_for_no_findings():
    """No defect findings → None (caller falls back to original plan)."""
    from ticket_to_code.workflow import _build_remediation_plan

    state = {
        "outcome_findings": [],
        "outcome_check_result": {"status": "CORRECT", "details": "", "findings": []},
        "generated_code": [],
        "change_targets": [],
    }
    plan = _build_remediation_plan(state)
    assert plan is None


def test_build_remediation_plan_multiple_files():
    """Findings spanning two files produce two MODIFY tasks."""
    from ticket_to_code.workflow import _build_remediation_plan

    state = {
        "outcome_findings": [
            OutcomeFinding(
                verdict="PARTIAL", requirement_id="REQ-1",
                requirement_text="Member check", summary="Single-select bug",
                offending_code="length === 1",
                affected_file="add-members.component.ts", affected_line=262,
                missing_behavior="Use length > 0", next_action="Fix quantifier",
            ).to_dict(),
            OutcomeFinding(
                verdict="PARTIAL", requirement_id="REQ-2",
                requirement_text="Org display", summary="Static text wrong",
                offending_code="isExistingProjectMember",
                affected_file="add-members.component.html", affected_line=68,
                missing_behavior="Show per-row state", next_action="Fix template",
            ).to_dict(),
        ],
        "outcome_check_result": {"status": "PARTIAL", "details": "", "findings": []},
        "generated_code": [
            MagicMock(file_path="src/app/modules/members/add-members/add-members.component.ts", content=""),
            MagicMock(file_path="src/app/modules/members/add-members/add-members.component.html", content=""),
            MagicMock(file_path="src/app/modules/members/add-members/add-members.component.scss", content=""),
        ],
        "change_targets": [
            "src/app/modules/members/add-members/add-members.component.ts",
            "src/app/modules/members/add-members/add-members.component.html",
        ],
    }

    plan = _build_remediation_plan(state)
    assert plan is not None
    # Only .ts and .html files implicated; SCSS not named by any finding
    assert len(plan.tasks) == 2
    task_files = {t.file_path for t in plan.tasks}
    assert any("add-members.component.ts" in f for f in task_files)
    assert any("add-members.component.html" in f for f in task_files)


def test_build_remediation_plan_uses_change_targets_when_finding_has_no_file():
    """A finding with an empty affected_file still resolves via change_targets."""
    from ticket_to_code.workflow import _build_remediation_plan

    state = {
        "outcome_findings": [
            {
                "verdict": "PARTIAL", "requirement_id": "REQ-1",
                "requirement_text": "Member check",
                "summary": "Single-select bug",
                "offending_code": "length === 1",
                "affected_file": "",
                "affected_line": None,
                "missing_behavior": "Use length > 0",
                "next_action": "Fix quantifier",
            }
        ],
        "outcome_check_result": {"status": "PARTIAL", "details": "", "findings": []},
        "generated_code": [],
        "change_targets": [
            "src/app/modules/members/add-members/add-members.component.ts",
        ],
    }

    plan = _build_remediation_plan(state)
    assert plan is not None
    assert len(plan.tasks) == 1
    assert "add-members.component.ts" in plan.tasks[0].file_path


# ---------------------------------------------------------------------------
# 3. route_after_outcome_check routing logic
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    base = {
        "outcome_check_result": {"status": "PARTIAL"},
        "outcome_remediation": "fix multi-select",
        "outcome_fix_attempt": 0,
        "max_outcome_fix_attempts": 3,
        "tier2_candidates": [],
        "context_expansion_count": 0,
    }
    base.update(overrides)
    return base


def test_route_correct_goes_to_memory_update():
    """CORRECT verdict always routes to memory_update (success)."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = _make_state(
        outcome_check_result={"status": "CORRECT"},
        outcome_fix_attempt=0,
    )
    assert route_after_outcome_check(state) == "memory_update"


def test_route_partial_with_attempt_goes_to_remediation_plan():
    """PARTIAL with attempts remaining routes to remediation_plan."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = _make_state(
        outcome_check_result={"status": "PARTIAL"},
        outcome_fix_attempt=1,  # 1 <= 3
    )
    assert route_after_outcome_check(state) == "remediation_plan"


def test_route_partial_max_attempts_goes_to_verification_failed():
    """PARTIAL with attempt > max_attempts routes to verification_failed (never success)."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = _make_state(
        outcome_check_result={"status": "PARTIAL"},
        outcome_fix_attempt=4,  # 4 > 3
        max_outcome_fix_attempts=3,
    )
    assert route_after_outcome_check(state) == "verification_failed"


def test_route_partial_first_retry_promotes_tier2():
    """First PARTIAL retry with tier2 candidates promotes to context_expand."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = _make_state(
        outcome_check_result={"status": "PARTIAL"},
        outcome_fix_attempt=1,
        tier2_candidates=[{"path": "some/file.ts"}],
        context_expansion_count=0,
    )
    assert route_after_outcome_check(state) == "context_expand"


def test_route_incomplete_exhausted_goes_to_verification_failed():
    """INCOMPLETE verdict after budget exhaustion routes to verification_failed."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = _make_state(
        outcome_check_result={"status": "INCOMPLETE"},
        outcome_fix_attempt=5,
        max_outcome_fix_attempts=3,
    )
    assert route_after_outcome_check(state) == "verification_failed"


def test_route_repair_required_within_budget():
    """Deterministic repair_required within budget routes to remediation_plan."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = {
        "outcome_check_result": {"status": "repair_required"},
        "outcome_remediation": "fix binding",
        "outcome_fix_attempt": 1,
        "max_outcome_fix_attempts": 3,
    }
    assert route_after_outcome_check(state) == "remediation_plan"


def test_route_repair_required_exhausted_goes_to_verification_failed():
    """Deterministic repair_required exhausted routes to verification_failed."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = {
        "outcome_check_result": {"status": "repair_required"},
        "outcome_remediation": None,
        "outcome_fix_attempt": 4,
        "max_outcome_fix_attempts": 3,
    }
    assert route_after_outcome_check(state) == "verification_failed"


def test_route_repair_required_exhausted_no_hints():
    """Deterministic repair_required exhausted (no hints) routes to verification_failed."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = {
        "outcome_check_result": {"status": "repair_required"},
        "outcome_remediation": None,
        "outcome_fix_attempt": 4,
        "max_outcome_fix_attempts": 3,
    }
    assert route_after_outcome_check(state) == "verification_failed"


def test_route_unknown_goes_to_memory_update():
    """UNKNOWN verdict (skipped/error) routes to memory_update."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = _make_state(
        outcome_check_result={"status": "UNKNOWN"},
        outcome_fix_attempt=0,
    )
    assert route_after_outcome_check(state) == "memory_update"


# ---------------------------------------------------------------------------
# 4. EditDecision target_symbols
# ---------------------------------------------------------------------------


def test_edit_decision_default_target_symbols():
    from ticket_to_code.agents.edit_loop_agent import EditDecision
    d = EditDecision(
        file_path="test.ts", reason="test", edit_description="test",
    )
    assert d.target_symbols == []


def test_edit_decision_with_target_symbols():
    from ticket_to_code.agents.edit_loop_agent import EditDecision
    d = EditDecision(
        file_path="test.ts", reason="test", edit_description="test",
        target_symbols=["onUserSelect", "addMember"],
    )
    assert d.target_symbols == ["onUserSelect", "addMember"]


# ---------------------------------------------------------------------------
# 5. Hierarchical Defect Escalation Ladder (Tier 0 -> Tier 1 -> Tier 2)
# ---------------------------------------------------------------------------


def test_route_partial_tier0_stops_at_remediation_plan():
    """When defect findings belong to already modified files (Tier 0), STOP search
    immediately and route to remediation_plan. Never blindly promote 11 Tier 2 candidates!"""
    from ticket_to_code.workflow import route_after_outcome_check
    state = _make_state(
        outcome_check_result={"status": "PARTIAL"},
        outcome_findings=[{
            "verdict": "PARTIAL",
            "requirement_id": "REQ-2",
            "affected_file": "add-members.component.ts",
            "summary": "Async race condition in project member check",
        }],
        generated_code=[
            type("Gen", (), {"file_path": "src/app/members/add-members.component.ts"})(),
            type("Gen", (), {"file_path": "src/app/members/add-members.component.html"})(),
        ],
        tier2_candidates=[{"path": f"src/app/backup-{i}.ts"} for i in range(11)],
        context_expansion_count=0,
        outcome_fix_attempt=1,
    )
    # MUST stop at Tier 0 and route to remediation_plan, NOT context_expand!
    assert route_after_outcome_check(state) == "remediation_plan"


def test_route_partial_tier1_promotes_tier1_file():
    """When a defect requires a file in Tier 1 candidate pool, stop at Tier 1 and promote only that file."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = _make_state(
        outcome_check_result={"status": "PARTIAL"},
        outcome_findings=[{
            "verdict": "PARTIAL",
            "requirement_id": "REQ-4",
            "affected_file": "project.service.ts",
            "summary": "Need helper method in ProjectService",
        }],
        generated_code=[
            type("Gen", (), {"file_path": "src/app/members/add-members.component.ts"})(),
        ],
        discovered_files=[
            {"path": "src/app/services/project.service.ts"},
            {"path": "src/app/services/other.service.ts"},
        ],
        tier2_candidates=[{"path": f"src/app/backup-{i}.ts"} for i in range(11)],
        context_expansion_count=0,
        outcome_fix_attempt=1,
    )
    assert route_after_outcome_check(state) == "context_expand"
    assert state.get("_targeted_expansion_files") == ["src/app/services/project.service.ts"]


def test_route_partial_tier2_promotes_tier2_file():
    """When a defect requires a file NOT in Tier 0 or Tier 1, promote matching Tier 2 candidate."""
    from ticket_to_code.workflow import route_after_outcome_check
    state = _make_state(
        outcome_check_result={"status": "PARTIAL"},
        outcome_findings=[{
            "verdict": "PARTIAL",
            "requirement_id": "REQ-6",
            "affected_file": "auth.service.ts",
            "summary": "Need token helper in AuthService",
        }],
        generated_code=[
            type("Gen", (), {"file_path": "src/app/members/add-members.component.ts"})(),
        ],
        discovered_files=[
            {"path": "src/app/services/project.service.ts"},
        ],
        tier2_candidates=[
            {"path": "src/app/services/auth.service.ts"},
            {"path": "src/app/backup-1.ts"},
        ],
        context_expansion_count=0,
        outcome_fix_attempt=1,
    )
    assert route_after_outcome_check(state) == "context_expand"
    assert state.get("_targeted_expansion_files") == ["src/app/services/auth.service.ts"]


def test_build_remediation_plan_partitions_satisfied_and_pending():
    """Delta remediation plan must display satisfied requirements (preserve)
    and bundle all pending requirements for the file into a single coordinated task."""
    from ticket_to_code.workflow import _build_remediation_plan
    state = {
        "outcome_findings": [
            {
                "verdict": "CORRECT",
                "requirement_id": "REQ-1",
                "requirement_text": "Verify project membership and company display",
                "summary": "Verified satisfied in add-members.component.ts",
                "affected_file": "",
            },
            {
                "verdict": "PARTIAL",
                "requirement_id": "REQ-2",
                "requirement_text": "Asynchronous race condition check",
                "summary": "Race condition in onUserSelect",
                "affected_file": "src/app/members/add-members.component.ts",
                "missing_behavior": "Await checkProjectMembership before adding to staged list",
                "offending_code": "this.checkProjectMembership(user); this.stagedMembers.push(user);",
            },
            {
                "verdict": "PARTIAL",
                "requirement_id": "REQ-3",
                "requirement_text": "Multi-select per-row state display",
                "summary": "Single component boolean used instead of per-row flag",
                "affected_file": "src/app/members/add-members.component.ts",
                "missing_behavior": "Attach isExistingProjectMember to each staged member object",
                "offending_code": "",
            },
        ],
        "generated_code": [
            type("Gen", (), {"file_path": "src/app/members/add-members.component.ts"})(),
            type("Gen", (), {"file_path": "src/app/members/add-members.component.scss"})(),
        ],
        "change_targets": ["src/app/members/add-members.component.ts"],
    }

    plan = _build_remediation_plan(state)
    assert plan is not None
    # Only 1 file has defects, so only 1 task is created (no useless touching of SCSS!)
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert "add-members.component.ts" in task.file_path
    # Both pending defects are bundled together into the task description
    assert "[REQ-2]" in task.description
    assert "[REQ-3]" in task.description
    # Satisfied requirement is explicitly listed to be preserved
    assert "REQ-1" in task.description
    assert "ALREADY SATISFIED REQUIREMENTS" in task.description
    assert "Already Satisfied: REQ-1" in plan.rationale
    assert "Pending to Remediate: REQ-2, REQ-3" in plan.rationale


def test_remediation_plan_node_emits_ui_plan():
    """Verify remediation_plan_node generates the plan, updates state, and emits UI event."""
    from ticket_to_code.workflow import remediation_plan_node

    emitted = []
    def mock_ui_cb(evt):
        emitted.append(evt)

    state = {
        "outcome_findings": [
            {
                "verdict": "CORRECT",
                "requirement_id": "REQ-1",
                "summary": "Modal open OK",
            },
            {
                "verdict": "PARTIAL",
                "requirement_id": "REQ-2",
                "summary": "Check per user not single",
                "affected_file": "src/app/members/add-members.component.ts",
                "missing_behavior": "Iterate users",
            },
        ],
        "generated_code": [
            type("Gen", (), {"file_path": "src/app/members/add-members.component.ts"})(),
        ],
        "_transient": {"_ui_callback": mock_ui_cb},
    }

    result = remediation_plan_node(state, None)
    assert "architectural_plan" in result
    plan = result["architectural_plan"]
    assert len(plan.tasks) == 1
    assert "add-members.component.ts" in plan.tasks[0].file_path

    # Verify UI event was emitted
    assert len(emitted) == 1
    ui_event = emitted[0]
    assert ui_event["phase"] == "planning"
    assert ui_event["node"] == "remediation_plan"
    assert ui_event["data"]["task_count"] == 1


