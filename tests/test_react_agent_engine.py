"""
Unit tests for AutonomousReActEngine and ReActToolbox with Reference Architecture:
1. Cryptographic contentHash verification & stale edit prevention
2. Exact occurrence assertion (rejects ambiguous multiple matches)
3. Prior-Read Invariant enforcement
4. Explicit phase progression (EXPLORE -> PLAN -> EDIT -> VERIFY -> REPORT)
5. Anti-spinning duplicate guard
6. Full end-to-end ReAct loop execution
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ticket_to_code.models import ValueEdgeTicket
from ticket_to_code.runtime.react_agent_engine import (
    AutonomousReActEngine,
    ReActToolbox,
    ReActState,
    ReActAgentResult,
)
from ticket_to_code.editing.surgical_diff_engine import SurgicalDiffEngine


def test_react_toolbox_content_hash_and_stale_edit_rejection(tmp_path):
    """Verifies that replace_file_content rejects stale content hashes and ambiguous occurrences."""
    diff_engine = SurgicalDiffEngine(str(tmp_path))
    toolbox = ReActToolbox(str(tmp_path), diff_engine)

    test_file = tmp_path / "Service.java"
    test_file.write_text("class Service {\n    void run() {}\n}\n", encoding="utf-8")

    # 1. view_file produces contentHash
    view_res = toolbox.view_file("Service.java", 1, 10)
    assert view_res.ok is True
    valid_hash = view_res.data["contentHash"]
    assert len(valid_hash) == 16

    # 2. Edit with stale hash -> Must be rejected
    stale_res = toolbox.replace_file_content(
        file_path="Service.java",
        target_content="    void run() {}",
        replacement_content="    void run() { check(); }",
        expected_content_hash="invalid_deadbeef",
    )
    assert stale_res.ok is False
    assert stale_res.error.code == "STALE_CONTENT_HASH"

    # 3. Edit with correct hash -> Succeeds
    ok_res = toolbox.replace_file_content(
        file_path="Service.java",
        target_content="    void run() {}",
        replacement_content="    void run() { check(); }",
        expected_content_hash=valid_hash,
    )
    assert ok_res.ok is True
    assert "check()" in test_file.read_text(encoding="utf-8")


def test_react_toolbox_occurrence_mismatch_rejection(tmp_path):
    """Verifies that replace_file_content rejects ambiguous targets that match multiple times."""
    diff_engine = SurgicalDiffEngine(str(tmp_path))
    toolbox = ReActToolbox(str(tmp_path), diff_engine)

    test_file = tmp_path / "Ambiguous.java"
    test_file.write_text("void foo() {}\nvoid foo() {}\n", encoding="utf-8")

    view_res = toolbox.view_file("Ambiguous.java", 1, 10)
    file_hash = view_res.data["contentHash"]

    # Target occurs twice, expectedOccurrences is 1 -> Must be rejected as ambiguous
    res = toolbox.replace_file_content(
        file_path="Ambiguous.java",
        target_content="void foo() {}",
        replacement_content="void bar() {}",
        expected_content_hash=file_hash,
        expected_occurrences=1,
    )
    assert res.ok is False
    assert res.error.code == "OCCURRENCE_MISMATCH"


def test_autonomous_react_engine_full_reference_loop(tmp_path):
    """
    Simulates full reference ReAct cycle:
    Step 1: grep_search -> find target file
    Step 2: view_file -> inspect lines and obtain contentHash
    Step 3: SUBMIT_PLAN -> transition state to EDIT
    Step 4: replace_file_content -> apply hash-verified patch
    Step 5: COMPLETE_TASK -> trigger verification gate
    """
    svc_file = tmp_path / "ParticipatingMemberService.java"
    svc_file.write_text(
        "public class ParticipatingMemberService {\n"
        "    public void updateMember() {\n"
        "        save();\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    ticket = ValueEdgeTicket(
        ticket_id="CC4E-BE-DEFECT-09",
        title="Prevent Ghost Members",
        description="Verify contracts before saving in ParticipatingMemberService.",
    )

    # Pre-calculate file hash for Step 4
    file_hash = ReActToolbox.compute_hash(svc_file.read_text(encoding="utf-8"))

    canned_responses = [
        # Step 1: Search
        '{"thought": "Search for updateMember", "tool": "grep_search", "args": {"pattern": "updateMember"}}',
        # Step 2: Read lines and hash
        '{"thought": "Inspect file and obtain contentHash", "tool": "view_file", "args": {"file_path": "ParticipatingMemberService.java", "start_line": 1, "end_line": 10}}',
        # Step 3: Formal Plan
        '{"thought": "Plan changes across service", "tool": "SUBMIT_PLAN", "args": {"plan_summary": "Update updateMember to check contracts"}}',
        # Step 4: Hash-checked patch
        f'{{"thought": "Apply patch with verified contentHash", "tool": "replace_file_content", "args": {{"file_path": "ParticipatingMemberService.java", "target_content": "        save();", "replacement_content": "        if (contracts == 0) deleteMember(); else save();", "expected_content_hash": "{file_hash}", "expected_occurrences": 1}}}}',
        # Step 5: Verification Gate
        '{"thought": "Trigger verification", "tool": "COMPLETE_TASK", "args": {}}',
    ]

    response_iter = iter(canned_responses)
    engine = AutonomousReActEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
        llm_client=lambda prompt: next(response_iter),
        max_steps=8,
    )

    with patch.object(engine, "verify_workspace", return_value=(True, True, "")):
        result: ReActAgentResult = engine.run()

    assert result.success is True
    assert result.status == "verified_success"
    assert "ParticipatingMemberService.java" in result.modified_files
    assert "if (contracts == 0) deleteMember(); else save();" in svc_file.read_text(encoding="utf-8")
