import json
import pytest
from ticket_to_code.models import OutcomeFinding


def test_outcome_finding_instantiation_and_dict():
    finding = OutcomeFinding(
        verdict="PARTIAL",
        requirement_id="REQ-1",
        requirement_text="When a user is selected to be added, check if already in project",
        summary="Check is restricted to single selection only",
        offending_code="searchData.length === 1",
        affected_file="src/app/modules/members/add-members/add-members.component.ts",
        affected_line=142,
        missing_behavior="Behavior must apply to every selected user (searchData.length > 0)",
        next_action="Re-evaluating existing evidence and re-planning",
    )
    d = finding.to_dict()
    assert d["verdict"] == "PARTIAL"
    assert d["requirement_id"] == "REQ-1"
    assert d["offending_code"] == "searchData.length === 1"
    assert d["affected_line"] == 142
    assert "searchData.length > 0" in d["missing_behavior"]


def test_outcome_finding_optional_line():
    finding = OutcomeFinding(
        verdict="INCOMPLETE",
        requirement_id="REQ-2",
        requirement_text="Display error toast when user is member",
        summary="Toast call missing from component",
        offending_code="",
        affected_file="src/app/modules/members/add-members/add-members.component.ts",
    )
    assert finding.affected_line is None
    d = finding.to_dict()
    assert d["affected_line"] is None
    assert d["verdict"] == "INCOMPLETE"


def test_outcome_finding_json_serialization():
    finding = OutcomeFinding(
        verdict="PARTIAL",
        requirement_id="REQ-1",
        requirement_text="Cardinality check",
        summary="Fails on multiple selections",
        offending_code="searchData.length === 1",
        affected_file="add-members.component.ts",
        missing_behavior="Support multi-user selection",
    )
    raw_json = json.dumps(finding.to_dict())
    loaded = json.loads(raw_json)
    assert loaded["offending_code"] == "searchData.length === 1"
    assert loaded["requirement_id"] == "REQ-1"
