"""
Unit tests for token telemetry persistence.

Verifies that:
1. history_store persists and returns token_usage across tasks.
2. broadcast_to_workflow updates workflow_status with token telemetry snapshots.
3. _build_chat_summary incorporates token usage into conversational summaries.
4. get_workflow_status returns token_usage for completed / historical workflows.
"""

import sys
from pathlib import Path
import pytest

# Ensure chatbot/backend is on path
backend_dir = Path(__file__).resolve().parents[1] / "chatbot" / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import history_store


def test_history_store_persists_token_usage(tmp_path, monkeypatch):
    """Verify record_task persists token_usage and get_task returns it."""
    test_hist_file = tmp_path / "test_history.json"
    monkeypatch.setattr(history_store, "_HISTORY_FILE", test_hist_file)

    sample_tokens = {
        "tokens_in": 12500,
        "tokens_out": 2300,
        "total_tokens": 14800,
        "llm_calls": 5,
        "phase_breakdown": {
            "classification": {"tokens_in": 1500, "tokens_out": 300, "total_tokens": 1800, "llm_calls": 1},
            "patch_generation": {"tokens_in": 11000, "tokens_out": 2000, "total_tokens": 13000, "llm_calls": 4},
        }
    }

    rec = history_store.record_task(
        workflow_id="wf-test-123",
        project_id="proj-1",
        ticket_id="TICKET-42",
        title="Fix Add Members Modal",
        description="Bug fix for add members modal",
        status="completed",
        changed_files=["src/app/add-members.component.ts"],
        execution_mode="pipeline",
        token_usage=sample_tokens,
    )

    assert rec["token_usage"] == sample_tokens
    assert rec["workflow_id"] == "wf-test-123"

    # Now retrieve it via get_task
    fetched = history_store.get_task("wf-test-123")
    assert fetched is not None
    assert fetched["token_usage"] == sample_tokens
    assert fetched["token_usage"]["total_tokens"] == 14800
    assert fetched["token_usage"]["llm_calls"] == 5


def test_chat_summary_includes_token_usage():
    """Verify _build_chat_summary includes formatted token usage."""
    from main import _build_chat_summary

    sample_tokens = {
        "tokens_in": 10500,
        "tokens_out": 1500,
        "total_tokens": 12000,
        "llm_calls": 4,
    }

    summary = _build_chat_summary(
        workflow_id="wf-abc",
        ticket_title="Add member validation",
        ticket_description="Validate member list before add",
        status="completed",
        generated_files=["foo.ts"],
        explanation={"summary": "Added validation logic"},
        error=None,
        steps=[],
        token_usage=sample_tokens,
    )

    assert "🪙 **Token Usage:** 12,000 tokens" in summary
    assert "10,500 prompt" in summary
    assert "1,500 completion" in summary
    assert "4 LLM calls" in summary


import asyncio


def test_broadcast_to_workflow_stores_token_usage():
    """Verify broadcast_to_workflow persists token telemetry in workflow_status."""
    from main import broadcast_to_workflow, workflow_status

    wf_id = "wf-broadcast-telemetry-test"
    workflow_status[wf_id] = {
        "status": "running",
        "current_phase": "classification",
        "steps": [],
    }

    token_event = {
        "phase": "patch_generation",
        "status": "in_progress",
        "message": "",
        "data": {
            "event_type": "token_usage_update",
            "tokens_in": 8000,
            "tokens_out": 1200,
            "total_tokens": 9200,
            "llm_calls": 3,
        }
    }

    asyncio.run(broadcast_to_workflow(wf_id, token_event))

    assert "token_usage" in workflow_status[wf_id]
    assert workflow_status[wf_id]["token_usage"]["total_tokens"] == 9200
    assert workflow_status[wf_id]["token_usage"]["llm_calls"] == 3

    # Clean up
    workflow_status.pop(wf_id, None)


def test_get_workflow_status_returns_token_usage_active_and_fallback(tmp_path, monkeypatch):
    """Verify get_workflow_status endpoint returns token_usage for both active and history fallback."""
    from main import get_workflow_status, workflow_status

    # 1. Active workflow
    active_wf_id = "wf-active-test"
    workflow_status[active_wf_id] = {
        "ticket_id": "TICK-1",
        "status": "completed",
        "current_phase": "completed",
        "candidate_files": [],
        "generated_files": ["foo.ts"],
        "steps": [],
        "token_usage": {
            "tokens_in": 5000,
            "tokens_out": 800,
            "total_tokens": 5800,
            "llm_calls": 2,
        },
    }

    resp = asyncio.run(get_workflow_status(active_wf_id))
    assert resp["token_usage"] is not None
    assert resp["token_usage"]["total_tokens"] == 5800
    workflow_status.pop(active_wf_id, None)

    # 2. History fallback when workflow_id not in workflow_status (e.g. after server restart)
    test_hist_file = tmp_path / "test_hist_fallback.json"
    monkeypatch.setattr(history_store, "_HISTORY_FILE", test_hist_file)

    history_store.record_task(
        workflow_id="wf-hist-fallback",
        project_id="proj-1",
        ticket_id="TICK-2",
        title="Hist ticket",
        description="Hist desc",
        status="completed",
        changed_files=["bar.ts"],
        token_usage={"tokens_in": 9000, "tokens_out": 1000, "total_tokens": 10000, "llm_calls": 4},
    )

    resp_hist = asyncio.run(get_workflow_status("wf-hist-fallback"))
    assert resp_hist["token_usage"] is not None
    assert resp_hist["token_usage"]["total_tokens"] == 10000
    assert resp_hist["token_usage"]["llm_calls"] == 4
