"""
Unit tests for the Stop Flow cancellation mechanism and multi-tier conversation memory.
Verifies that:
1. ConversationSessionManager handles mark_stopped cleanly.
2. Workflow explanations and chat summaries correctly format stopped workflows.
3. _build_fallback_chat_reply answers CloudForge architecture questions (coordinators, roles, contract access).
4. Stopped workflow memory and past task episodic history are preserved.
"""
import sys
from pathlib import Path
import pytest

backend_dir = Path(__file__).resolve().parents[1] / "chatbot" / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ticket_to_code.conversation.session_manager import ConversationSessionManager, SessionStage
from main import (
    _build_workflow_explanation,
    _build_chat_summary,
    _build_fallback_chat_reply,
    WorkflowChatRequest,
)


def test_session_manager_mark_stopped():
    manager = ConversationSessionManager()
    session = manager.create_session(
        workspace_path="/tmp/workspace",
        goal="Fix contract coordinator permissions",
        initial_message="User reports issue with coordinator role removal in CloudForge."
    )
    manager.mark_running(session.session_id)
    assert manager.get_session(session.session_id).stage == SessionStage.RUNNING

    # Stop the session
    stopped_session = manager.mark_stopped(session.session_id, reason="User clicked stop flow")
    assert stopped_session.stage == SessionStage.STOPPED
    assert stopped_session.last_workflow_status == "stopped"
    assert any("Pipeline stopped" in rej for rej in stopped_session.conversation.context.rejected_decisions)
    assert len(stopped_session.conversation.messages) >= 2
    # Verify we can continue adding user messages in the same session without restart
    updated = manager.add_user_message(session.session_id, "Now tell me how roles work instead")
    assert len(updated.conversation.messages) >= 3


def test_build_workflow_explanation_stopped():
    state = {
        "status": "stopped",
        "current_phase": "investigate",
        "steps": [
            {"message": "🔍 Classified as: defect", "status": "completed"},
            {"message": "📂 RAG: 4 chunks retrieved", "status": "completed"}
        ]
    }
    explanation = _build_workflow_explanation(state, {})
    assert "halted by user request" in explanation["summary"]
    assert "investigate" in explanation["why"]
    assert "Continue conversation in chat" in explanation["next_action"]


def test_build_chat_summary_stopped():
    summary = _build_chat_summary(
        workflow_id="wf-test-stopped",
        ticket_title="Add project coordinator window issue fix",
        ticket_description="Fix coordinator modal dropdown",
        status="stopped",
        generated_files=["src/app/modules/project/project-settings/coordinators/coordinators.component.ts"],
        explanation={
            "summary": "The workflow was halted by user request.",
            "why": "Stopped cleanly at phase 'investigate'. Intermediate findings have been saved into memory.",
            "next_action": "Continue conversation in chat."
        },
        error="Stopped by user",
        steps=[],
    )
    assert "🛑 **Workflow stopped by user.**" in summary
    assert "Conversation Memory is Active" in summary
    assert "coordinators.component.ts" in summary


def test_cloudforge_architecture_fallback_response():
    # User asks about coordinator vs administrator and contract access removal
    question = (
        "do project level , it will store coordinator and in backend it will be stored as "
        "administrator and we can assign them 3 role , what are they . and when we remove "
        "any thing from project level coordinator who is also in a contract , then will it "
        "have that contract access or will they lose it?"
    )
    reply = _build_fallback_chat_reply(
        question=question,
        explanation={},
        gen_files=[],
        state={"status": "stopped"},
    )
    # Check that it explains Coordinator vs Administrator mapping
    assert "Coordinator vs. Administrator Mapping" in reply
    assert "ROLE_ADMINISTRATOR" in reply
    # Check that it explains the 3 roles
    assert "Coordinator / Administrator" in reply
    assert "Contributor / Member" in reply
    assert "Consumer / Reviewer" in reply
    # Check that it covers contract access removal scenarios
    assert "Scenario A (Explicit Contract ACL)" in reply
    assert "Scenario B (Inherited Access)" in reply
    assert "Scenario C (Saga Event Synchronization)" in reply


def test_workflow_chat_request_schema():
    req = WorkflowChatRequest(
        message="Explain what you did before stopping",
        workflow_id="wf-123",
        project_id="proj-456",
        chat_id="chat-789",
        messages=[{"role": "user", "content": "hello"}]
    )
    assert req.workflow_id == "wf-123"
    assert req.chat_id == "chat-789"
    assert len(req.messages) == 1


def test_episodic_facts_and_consolidation(tmp_path):
    from ticket_to_code.memory.episodic_memory import EpisodicMemoryStore
    store = EpisodicMemoryStore(str(tmp_path))

    # Test fact upsert & hit count
    store.upsert_fact("cloudforge", "build_fix", "TS2307", "Add @core/auth barrel import", "T-101")
    store.upsert_fact("cloudforge", "build_fix", "TS2307", "Add @core/auth barrel import", "T-102")
    facts = store.lookup_facts("cloudforge", kind="build_fix")
    assert len(facts) == 1
    assert facts[0]["key"] == "TS2307"
    assert facts[0]["hit_count"] == 2

    # Test run consolidation
    state = {
        "ticket_id": "T-103",
        "title": "Fix contract coordinator",
        "description": "Fix coordinator modal",
        "status": "stopped",
        "candidate_diffs": [{"path": "src/app/modules/project/coordinators.component.ts"}],
        "exec_logs": ["Error: TS2307 cannot find module '@bim/auth'"],
        "rejected_approaches": ["Do not edit area-service directly"],
    }
    summary = store.consolidate_run(state, repo="cloudforge")
    assert summary["files_touched"] >= 1
    assert summary["facts_added"] >= 2

    # Test scratchpad priming
    primed = store.prime_scratchpad("cloudforge", "coordinator modal auth error")
    assert any("coordinators.component.ts" in f for f in primed["known_files"])
    assert "TS2307" in primed["known_fixes"]
    assert any("area-service" in p for p in primed["user_preferences"])


def test_resume_workflow_endpoint_logic():
    from main import resume_workflow, workflow_status, ResumeWorkflowRequest
    import asyncio

    wid = "wf-test-resume-unit"
    workflow_status[wid] = {
        "status": "stopped",
        "current_phase": "investigate",
        "rejected_approaches": [],
        "steps": [],
    }

    req = ResumeWorkflowRequest(
        workflow_id=wid,
        resume_from="plan",
        feedback="Focus on coordinators.component.ts only",
    )
    res = asyncio.run(resume_workflow(wid, req))
    assert res["status"] == "resumed"
    assert res["resume_from"] == "plan"
    assert workflow_status[wid]["status"] == "running"
    assert "Focus on coordinators.component.ts only" in workflow_status[wid]["rejected_approaches"]


def test_get_workflow_changes_endpoint():
    from main import get_workflow_changes, workflow_status
    import asyncio

    wid = "wf-test-changes-unit"
    workflow_status[wid] = {
        "status": "stopped",
        "ticket_id": "T-999",
        "state": {
            "generated_code": [
                {
                    "file_path": "src/app/coordinators.component.ts",
                    "content": "export class CoordinatorsComponent {\n  // updated logic\n}\n",
                }
            ],
            "original_file_contents": {
                "src/app/coordinators.component.ts": "export class CoordinatorsComponent {\n  // old logic\n}\n"
            }
        }
    }

    res = asyncio.run(get_workflow_changes(wid))
    assert res["workflow_id"] == wid
    assert res["total_files"] == 1
    assert len(res["files"]) == 1
    assert res["files"][0]["path"] == "src/app/coordinators.component.ts"
    assert res["files"][0]["status"] == "modified"
    assert res["files"][0]["approved"] is True
    assert "--- a/coordinators.component.ts" in res["files"][0]["diff"]
    assert "+  // updated logic" in res["files"][0]["diff"]
    # Check that changes alias is also present for App.jsx compatibility
    assert res["changes"] == res["files"]


def test_selective_approval_endpoint():
    from main import approve_workflow_changes, get_workflow_changes, workflow_status, DiffApprovalRequest
    import asyncio

    wid = "wf-test-selective-approval"
    workflow_status[wid] = {
        "status": "stopped",
        "ticket_id": "T-888",
        "state": {
            "generated_code": [
                {"file_path": "src/app/file1.ts", "content": "class A {}"},
                {"file_path": "src/app/file2.ts", "content": "class B {}"}
            ],
            "original_file_contents": {
                "src/app/file1.ts": "class A_old {}",
                "src/app/file2.ts": "class B_old {}"
            }
        }
    }

    # User selectively approves file1, rejects file2
    req = DiffApprovalRequest(
        approved_files=["src/app/file1.ts"],
        rejected_files=["src/app/file2.ts"]
    )
    approve_res = asyncio.run(approve_workflow_changes(wid, req))
    assert approve_res["status"] == "updated"
    assert approve_res["approved_count"] == 1
    assert approve_res["rejected_count"] == 1

    # Verify get_workflow_changes reflects the selective approval
    changes_res = asyncio.run(get_workflow_changes(wid))
    file1 = next(f for f in changes_res["files"] if f["path"] == "src/app/file1.ts")
    file2 = next(f for f in changes_res["files"] if f["path"] == "src/app/file2.ts")
    assert file1["approved"] is True
    assert file2["approved"] is False


def test_workspace_lock_mutex():
    from main import get_workspace_lock
    import os

    p1 = "c:/Users/test/repo"
    p2 = "C:\\Users\\test\\repo"
    l1 = get_workspace_lock(p1)
    l2 = get_workspace_lock(p2)
    assert l1 is l2, "get_workspace_lock must normalize paths and return the exact same Lock"


def test_safe_add_list_reducer():
    from ticket_to_code.workflow import _safe_add_list

    assert _safe_add_list(None, ["err1"]) == ["err1"]
    assert _safe_add_list(["err1"], None) == ["err1"]
    assert _safe_add_list(["err1"], ["err2"]) == ["err1", "err2"]
    assert _safe_add_list(None, None) == []


def test_workflow_chat_with_state_snapshot():
    from main import workflow_chat, workflow_status, WorkflowChatRequest
    import asyncio

    wid = "wf-test-chat-snapshot"
    workflow_status[wid] = {
        "status": "stopped",
        "ticket_id": "T-SNAP",
        "workflow_explanation": {"summary": "Investigated ticket"},
        "state": {
            "investigation_hypotheses": [
                {"hypothesis": "Service auth token expired", "status": "active"}
            ],
            "evidence_items": [
                {"file_path": "src/services/auth.service.ts", "score": 0.95, "reason": "Exact match"}
            ],
            "blacklisted_files": ["src/legacy/auth.ts"],
            "build_diagnostic_summary": "Clean build",
        }
    }

    req = WorkflowChatRequest(
        message="Why did you look at auth.service.ts?",
        workflow_id=wid,
    )
    res = asyncio.run(workflow_chat(req))
    assert "reply" in res
    assert len(res["reply"]) > 0
