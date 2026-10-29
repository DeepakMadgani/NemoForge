"""
FastAPI Endpoints for Autonomous Ticket-to-Code System

Provides REST API to:
- Submit tickets for autonomous processing
- Check workflow status
- Retrieve generated code
- Trigger builds and tests

Author: Deepak Madgani
Date: April 2026
"""

import logging
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ticket_to_code.models import (
    ValueEdgeTicket,
    TicketPriority,
    WorkflowStatus
)
from ticket_to_code.conversation.session_manager import ConversationSessionManager
from ticket_to_code.workflow import (
    run_autonomous_workflow
)
from ticket_to_code.runtime.intake_contract import GroundTruthTicket
from ticket_to_code.benchmarks.contract_benchmark_runner import run_contract_benchmark

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ticket-to-code", tags=["Autonomous Workflow"])

# In-memory workflow state storage (replace with database in production)
workflow_states: dict[str, Any] = {}

# Run-centric in-memory stores for thin UI / backend-owned decisioning.
run_states: Dict[str, Dict[str, Any]] = {}
run_event_queues: Dict[str, asyncio.Queue] = {}

_TERMINAL_RUN_STATUSES = {"solved", "no_action_required", "failed_closed", "failed", "cancelled", "completed"}

# In-memory interactive conversation sessions
conversation_sessions = ConversationSessionManager()


def _safe_get(state: Any, key: str, default: Any = None) -> Any:
    """Get value from either dict-based state or object-based state."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _infer_contract_ticket_type(ticket: ValueEdgeTicket, labels: List[str]) -> str:
    """Infer contract ticket kind from labels/title/description using stable keyword mapping."""
    text = " ".join(labels + [ticket.title, ticket.description]).lower()
    if any(k in text for k in ("version", "upgrade", "downgrade", "bump")):
        return "version"
    if any(k in text for k in ("permission", "access denied", "forbidden", "unauthorized")):
        return "permission"
    if any(k in text for k in ("refactor", "cleanup", "restructure", "simplify")):
        return "refactor"
    if any(k in text for k in ("config", "configuration", "env", "yaml", "json", "properties")):
        return "config"
    return "bug"


def _ensure_run(run_id: str) -> Dict[str, Any]:
    run = run_states.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"No run found: {run_id}")
    return run


def _emit_run_event(
    run_id: str,
    *,
    event_type: str,
    stage: Optional[str] = None,
    status: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run = _ensure_run(run_id)
    seq = int(run.get("event_seq", 0)) + 1
    run["event_seq"] = seq

    if stage is not None:
        run["current_stage"] = stage
    if status is not None:
        run["status"] = status

    event = {
        "run_id": run_id,
        "seq": seq,
        "type": event_type,
        "stage": run.get("current_stage"),
        "status": run.get("status"),
        "timestamp": _now_iso(),
        "payload": payload or {},
    }
    run.setdefault("events", []).append(event)
    run["updated_at"] = event["timestamp"]

    queue = run_event_queues.get(run_id)
    if queue is not None:
        queue.put_nowait(event)
    return event


def _build_final_report(state: Any, run_id: str) -> Dict[str, Any]:
    contract_report = _safe_get(state, "contract_report")
    if isinstance(contract_report, dict):
        return {
            "run_id": run_id,
            "ticket_id": _ticket_id_from_state(state, run_id),
            "status": str(contract_report.get("status", _status_value(_safe_get(state, "status")))),
            "findings": contract_report.get("findings", []),
            "decisions": contract_report.get("decisions", []),
            "changed_files": contract_report.get("changed_files", []),
            "validation_evidence": contract_report.get("validation_evidence", []),
            "residual_risks": contract_report.get("residual_risks", []),
            "why_changed": [
                {
                    "path": item.get("path"),
                    "reason": item.get("reason", "not_provided"),
                }
                for item in (contract_report.get("changed_files") or [])
                if isinstance(item, dict)
            ],
        }

    generated = _generated_files(state)
    return {
        "run_id": run_id,
        "ticket_id": _ticket_id_from_state(state, run_id),
        "status": _status_value(_safe_get(state, "status")),
        "findings": [],
        "decisions": _safe_get(state, "internal_node_contracts", []) or [],
        "changed_files": [{"path": f.get("path", "unknown"), "reason": "generated_or_updated"} for f in generated],
        "validation_evidence": _safe_get(state, "validation_evidence", []) or [],
        "residual_risks": _safe_get(state, "residual_risks", []) or [],
        "why_changed": [{"path": f.get("path", "unknown"), "reason": "generated_or_updated"} for f in generated],
    }


def _status_value(status: Any) -> str:
    if hasattr(status, "value"):
        return status.value
    if status is None:
        return "unknown"
    return str(status)


def _ticket_id_from_state(state: Any, fallback: str) -> str:
    ticket = _safe_get(state, "ticket")
    if isinstance(ticket, dict):
        return ticket.get("ticket_id", fallback)
    return getattr(ticket, "ticket_id", fallback) if ticket else fallback


def _generated_files(state: Any) -> List[Dict[str, Any]]:
    generated = _safe_get(state, "generated_code", []) or []
    output: List[Dict[str, Any]] = []
    for item in generated:
        if isinstance(item, dict):
            path = item.get("file_path") or item.get("path") or "unknown"
            language = item.get("language", "unknown")
            code = item.get("code") or item.get("content") or ""
            explanation = item.get("explanation") or item.get("documentation") or ""
        else:
            path = getattr(item, "file_path", "unknown")
            language = getattr(item, "language", "unknown")
            code = getattr(item, "code", "")
            explanation = getattr(item, "explanation", "")
        output.append(
            {
                "path": path,
                "language": language,
                "code": code,
                "explanation": explanation,
                "lines": len(code.split("\n")) if code else 0,
            }
        )
    return output


def _build_status_response(state: Any, ticket_id_fallback: str) -> "WorkflowStatusResponse":
    ticket_id = _ticket_id_from_state(state, ticket_id_fallback)
    status = _status_value(_safe_get(state, "status"))
    errors = _safe_get(state, "errors", []) or []
    if not isinstance(errors, list):
        errors = [str(errors)]

    investigation = _safe_get(state, "investigation_result")
    if isinstance(investigation, dict):
        inv = investigation
        investigation_summary = {
            "type": inv.get("ticket_type"),
            "code_needed": inv.get("requires_code_changes"),
            "explanation": inv.get("explanation"),
            "confidence": inv.get("confidence"),
        }
    elif investigation is not None:
        investigation_summary = {
            "type": _status_value(getattr(investigation, "ticket_type", None)),
            "code_needed": getattr(investigation, "requires_code_changes", None),
            "explanation": getattr(investigation, "explanation", None),
            "confidence": getattr(investigation, "confidence", None),
        }
    else:
        investigation_summary = None

    build_result = _safe_get(state, "build_result")
    build_status = None
    if isinstance(build_result, dict):
        build_status = _status_value(build_result.get("status"))
    elif build_result is not None:
        build_status = _status_value(getattr(build_result, "status", None))

    test_result = _safe_get(state, "test_result")
    test_results = None
    if isinstance(test_result, dict):
        test_results = {
            "passed": test_result.get("passed"),
            "failed": test_result.get("failed"),
            "total": test_result.get("total"),
        }
    elif test_result is not None:
        test_results = {
            "passed": getattr(test_result, "passed", None),
            "failed": getattr(test_result, "failed", None),
            "total": getattr(test_result, "total", None),
        }

    generated_files = [
        {"path": f["path"], "language": f["language"], "lines": f["lines"]}
        for f in _generated_files(state)
    ]

    start_time = _safe_get(state, "start_time")
    end_time = _safe_get(state, "end_time")
    duration_seconds = None
    if start_time and end_time:
        try:
            duration_seconds = (end_time - start_time).total_seconds()
        except Exception:
            duration_seconds = None

    return WorkflowStatusResponse(
        ticket_id=ticket_id,
        status=status,
        investigation_summary=investigation_summary,
        generated_files=generated_files,
        build_status=build_status,
        test_results=test_results,
        errors=[str(e) for e in errors],
        duration_seconds=duration_seconds,
    )


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TicketSubmissionRequest(BaseModel):
    """Request to submit a ticket for processing"""
    ticket_id: str = Field(..., description="ValueEdge ticket ID")
    title: str = Field(..., description="Ticket title")
    description: str = Field(..., description="Detailed description")
    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="Acceptance criteria"
    )
    priority: TicketPriority = Field(
        default=TicketPriority.MEDIUM,
        description="Priority level"
    )
    labels: List[str] = Field(default_factory=list, description="Labels/tags")
    workspace_path: str = Field(
        ...,
        description="Path where code should be generated"
    )
    codebase_path: Optional[str] = Field(
        None,
        description="Path to existing codebase (for context)"
    )
    technology: Optional[str] = Field(
        None,
        description="Technology: 'dotnet', 'java', 'nodejs' (auto-detected if None)"
    )
    auto_execute: bool = Field(
        default=False,
        description="Automatically build and test generated code"
    )
    execution_mode: str = Field(
        default="auto",
        description="'auto' (reasoning-first + pipeline fallback), 'reasoning', or 'pipeline'"
    )


class WorkflowStatusResponse(BaseModel):
    """Response with workflow status"""
    ticket_id: str
    status: str
    investigation_summary: Optional[dict] = None
    generated_files: List[dict] = Field(default_factory=list)
    build_status: Optional[str] = None
    test_results: Optional[dict] = None
    errors: List[str] = Field(default_factory=list)
    duration_seconds: Optional[float] = None


class CreateRunRequest(BaseModel):
    ticket_title: str = Field(..., description="Ticket title")
    description: str = Field(..., description="Ticket description")
    constraints: List[str] = Field(default_factory=list, description="Execution constraints")
    repo_scope: str = Field(..., description="Workspace or repository scope path")
    acceptance_criteria: List[str] = Field(default_factory=list, description="Acceptance checks")
    labels: List[str] = Field(default_factory=list, description="Ticket labels")
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM)
    execution_mode: str = Field(default="contract", description="Recommended: contract")
    requires_approval: bool = Field(default=False, description="Pause run for explicit approval before execution")
    ticket_id: Optional[str] = Field(default=None, description="Optional caller-provided ticket id")
    max_fix_attempts: int = Field(default=3, ge=1, le=10)


class CreateRunResponse(BaseModel):
    run_id: str
    status: str
    initial_status: str


class RunStatusResponse(BaseModel):
    run_id: str
    ticket_id: str
    status: str
    current_stage: str
    retries_used: int = 0
    blocking_reason: Optional[str] = None
    requires_clarification: bool = False
    requires_approval: bool = False
    created_at: str
    updated_at: str


class FinalReportResponse(BaseModel):
    run_id: str
    ticket_id: str
    status: str
    findings: List[Any] = Field(default_factory=list)
    decisions: List[Any] = Field(default_factory=list)
    changed_files: List[Any] = Field(default_factory=list)
    validation_evidence: List[Any] = Field(default_factory=list)
    residual_risks: List[Any] = Field(default_factory=list)
    why_changed: List[Any] = Field(default_factory=list)


class ClarificationAnswerRequest(BaseModel):
    answer: str = Field(..., description="Answer to backend clarification question")


class ApprovalDecisionRequest(BaseModel):
    approved: bool = Field(..., description="Human approval decision")
    note: str = Field(default="", description="Optional decision note")


class BenchmarkReplayRequest(BaseModel):
    repo_scope: str = Field(..., description="Workspace scope to run benchmark against")
    tickets: List[Dict[str, Any]] = Field(default_factory=list, description="Ground-truth tickets")


class BenchmarkReplayResponse(BaseModel):
    generated_at: str
    summary: Dict[str, Any]
    benchmark_metrics: Dict[str, Any]
    phase1_thresholds: Dict[str, Any]
    phase1_threshold_eval: Dict[str, Any]
    tickets: List[Dict[str, Any]]
    ticket_metrics: List[Dict[str, Any]]


class GeneratedFileResponse(BaseModel):
    """Response with generated file content"""
    file_path: str
    language: str
    code: str
    explanation: str


class ConversationSessionCreateRequest(BaseModel):
    workspace_path: str = Field(..., description="Target code workspace path")
    goal: str = Field(default="", description="High-level goal for this conversation")
    initial_message: str = Field(default="", description="Optional first user message")


class ConversationMessageRequest(BaseModel):
    text: str = Field(..., description="User message")
    images: List[str] = Field(default_factory=list, description="Optional image references")
    files: List[str] = Field(default_factory=list, description="Optional file path references")


class ConversationFeedbackRequest(BaseModel):
    feedback: str = Field(..., description="Correction feedback about reasoning or output")


class ConversationRunRequest(BaseModel):
    technology: Optional[str] = Field(default=None, description="Optional tech override")
    max_fix_attempts: int = Field(default=3, ge=1, le=10)


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("/runs", response_model=CreateRunResponse)
async def create_run(request: CreateRunRequest, background_tasks: BackgroundTasks):
    """Create a backend-owned run and start autonomous orchestrator in background."""
    scope = Path(request.repo_scope)
    if not scope.exists():
        raise HTTPException(status_code=400, detail=f"Repo scope does not exist: {request.repo_scope}")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    ticket_id = request.ticket_id or f"T2C-{uuid.uuid4().hex[:8].upper()}"

    ticket = ValueEdgeTicket(
        ticket_id=ticket_id,
        title=request.ticket_title,
        description=request.description,
        acceptance_criteria=request.acceptance_criteria,
        priority=request.priority,
        labels=request.labels,
    )

    run_states[run_id] = {
        "run_id": run_id,
        "ticket_id": ticket_id,
        "status": "initialized",
        "current_stage": "intake",
        "retries_used": 0,
        "blocking_reason": None,
        "requires_clarification": False,
        "requires_approval": bool(request.requires_approval),
        "clarification": None,
        "approval": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "event_seq": 0,
        "events": [],
        "final_report": None,
    }
    run_event_queues[run_id] = asyncio.Queue()

    _emit_run_event(
        run_id,
        event_type="run.created",
        stage="intake",
        status="initialized",
        payload={
            "ticket_id": ticket_id,
            "repo_scope": str(scope),
            "execution_mode": request.execution_mode,
            "constraints": request.constraints,
        },
    )

    if request.requires_approval:
        run_states[run_id]["status"] = "awaiting_approval"
        run_states[run_id]["blocking_reason"] = "human_approval_required"
        _emit_run_event(
            run_id,
            event_type="run.awaiting_approval",
            stage="approval_gate",
            status="awaiting_approval",
            payload={"reason": "high_risk_requires_human_approval"},
        )
    else:
        background_tasks.add_task(
            _execute_run_async,
            run_id=run_id,
            ticket=ticket,
            request=request,
        )

    return CreateRunResponse(
        run_id=run_id,
        status=run_states[run_id]["status"],
        initial_status="initialized",
    )


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str):
    """Stream run lifecycle events over Server-Sent Events (SSE)."""
    run = _ensure_run(run_id)

    async def event_stream():
        for event in list(run.get("events", [])):
            yield (
                f"id: {event['seq']}\n"
                f"event: {event['type']}\n"
                f"data: {json.dumps(event, default=_json_default)}\n\n"
            )

        queue = run_event_queues[run_id]
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                heartbeat = {
                    "run_id": run_id,
                    "type": "heartbeat",
                    "timestamp": _now_iso(),
                }
                yield f"event: heartbeat\ndata: {json.dumps(heartbeat)}\n\n"
                if run_states.get(run_id, {}).get("status") in _TERMINAL_RUN_STATUSES:
                    break
                continue

            yield (
                f"id: {event['seq']}\n"
                f"event: {event['type']}\n"
                f"data: {json.dumps(event, default=_json_default)}\n\n"
            )
            if str(event.get("status")) in _TERMINAL_RUN_STATUSES:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/runs/{run_id}/status", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    run = _ensure_run(run_id)
    return RunStatusResponse(
        run_id=run_id,
        ticket_id=run["ticket_id"],
        status=run["status"],
        current_stage=run["current_stage"],
        retries_used=int(run.get("retries_used", 0)),
        blocking_reason=run.get("blocking_reason"),
        requires_clarification=bool(run.get("requires_clarification", False)),
        requires_approval=bool(run.get("requires_approval", False)),
        created_at=run["created_at"],
        updated_at=run["updated_at"],
    )


@router.get("/runs/{run_id}/final-report", response_model=FinalReportResponse)
async def get_final_report(run_id: str):
    run = _ensure_run(run_id)
    if not run.get("final_report"):
        raise HTTPException(status_code=409, detail="Final report not ready yet")
    return FinalReportResponse(**run["final_report"])


@router.get("/runs/{run_id}/clarification")
async def get_clarification(run_id: str):
    run = _ensure_run(run_id)
    question = run.get("clarification")
    if not question:
        raise HTTPException(status_code=404, detail="No clarification pending for this run")
    return question


@router.post("/runs/{run_id}/clarification")
async def answer_clarification(run_id: str, request: ClarificationAnswerRequest, background_tasks: BackgroundTasks):
    run = _ensure_run(run_id)
    if not run.get("requires_clarification"):
        raise HTTPException(status_code=409, detail="Run is not awaiting clarification")

    run["requires_clarification"] = False
    run["blocking_reason"] = None
    run["clarification_answer"] = request.answer
    _emit_run_event(
        run_id,
        event_type="run.clarification_answered",
        stage="resume",
        status="running",
        payload={"answer": request.answer},
    )

    if run.get("pending_resume"):
        resume_payload = run["pending_resume"]
        ticket = resume_payload["ticket"]

        # Inject the user's clarification answer into the ticket description
        # so the cold-restart evidence loop sees the enriched context.
        # The ticket object is an in-memory Pydantic model — this mutation
        # stays local to this run and is NOT persisted back to the ticket system.
        clarification_answer = run.get("clarification_answer")
        if clarification_answer:
            current_desc = getattr(ticket, "description", "") or ""
            ticket.description = (
                current_desc +
                f"\n\n[Clarification from user]: {clarification_answer}"
            )
            logger.info(
                f"Run {run_id}: injected clarification answer into ticket description "
                f"({len(clarification_answer)} chars)"
            )

        background_tasks.add_task(
            _execute_run_async,
            run_id=run_id,
            ticket=ticket,
            request=resume_payload["request"],
        )
        run["pending_resume"] = None

    return {"run_id": run_id, "status": "running"}


@router.post("/runs/{run_id}/approval")
async def submit_approval(run_id: str, request: ApprovalDecisionRequest, background_tasks: BackgroundTasks):
    run = _ensure_run(run_id)
    if run.get("status") != "awaiting_approval":
        raise HTTPException(status_code=409, detail="Run is not awaiting approval")

    run["approval"] = {"approved": request.approved, "note": request.note, "timestamp": _now_iso()}
    if not request.approved:
        run["status"] = "failed_closed"
        run["blocking_reason"] = "human_rejected_high_risk_plan"
        _emit_run_event(
            run_id,
            event_type="run.failed_closed",
            stage="approval_gate",
            status="failed_closed",
            payload={"reason": "approval_rejected", "note": request.note},
        )
        run["final_report"] = {
            "run_id": run_id,
            "ticket_id": run["ticket_id"],
            "status": "failed_closed",
            "findings": ["run_blocked_by_human_rejection"],
            "decisions": [],
            "changed_files": [],
            "validation_evidence": [],
            "residual_risks": ["high_risk_plan_not_approved"],
            "why_changed": [],
        }
        return {"run_id": run_id, "status": "failed_closed"}

    run["blocking_reason"] = None
    run["status"] = "running"
    _emit_run_event(
        run_id,
        event_type="run.approved",
        stage="approval_gate",
        status="running",
        payload={"note": request.note},
    )

    pending = run.get("pending_approval_start")
    if pending:
        background_tasks.add_task(
            _execute_run_async,
            run_id=run_id,
            ticket=pending["ticket"],
            request=pending["request"],
        )
        run["pending_approval_start"] = None

    return {"run_id": run_id, "status": "running"}


@router.post("/benchmark/replay", response_model=BenchmarkReplayResponse)
async def benchmark_replay(request: BenchmarkReplayRequest):
    """Replay a ticket set in contract mode without UI interaction."""
    scope = Path(request.repo_scope)
    if not scope.exists():
        raise HTTPException(status_code=400, detail=f"Repo scope does not exist: {request.repo_scope}")

    if not request.tickets:
        raise HTTPException(status_code=400, detail="At least one benchmark ticket is required")

    try:
        gt_tickets = [GroundTruthTicket.model_validate(item) for item in request.tickets]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid benchmark ticket payload: {exc}") from exc

    result = await asyncio.to_thread(
        run_contract_benchmark,
        workspace_path=request.repo_scope,
        tickets=gt_tickets,
    )
    return BenchmarkReplayResponse(**result)

@router.post("/submit", response_model=WorkflowStatusResponse)
async def submit_ticket(
    request: TicketSubmissionRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a ValueEdge ticket for autonomous processing.
    
    The workflow will:
    1. Investigate the ticket (determine if code is needed)
    2. Analyze requirements
    3. Create architectural plan
    4. Retrieve relevant code context
    5. Generate code
    6. (Optional) Build and test
    
    Returns immediate response with ticket ID. Use /status endpoint to check progress.
    """
    logger.info(f"Received ticket submission: {request.ticket_id}")
    
    # Validate paths
    workspace = Path(request.workspace_path)
    if not workspace.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Workspace path does not exist: {request.workspace_path}"
        )
    
    # Create ticket object
    ticket = ValueEdgeTicket(
        ticket_id=request.ticket_id,
        title=request.title,
        description=request.description,
        acceptance_criteria=request.acceptance_criteria,
        priority=request.priority,
        labels=request.labels
    )
    
    # Start workflow in background
    background_tasks.add_task(
        _execute_workflow_async,
        ticket=ticket,
        workspace_path=request.workspace_path,
        codebase_path=request.codebase_path,
        technology=request.technology,
        auto_execute=request.auto_execute,
        execution_mode=request.execution_mode,
    )
    
    # Return immediate response
    return WorkflowStatusResponse(
        ticket_id=ticket.ticket_id,
        status=WorkflowStatus.INITIALIZED.value,
        generated_files=[],
        errors=[]
    )


@router.post("/conversation/session")
async def create_conversation_session(request: ConversationSessionCreateRequest):
    """Create an interactive ticket-to-code conversation session."""
    workspace = Path(request.workspace_path)
    if not workspace.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Workspace path does not exist: {request.workspace_path}",
        )

    session = conversation_sessions.create_session(
        workspace_path=request.workspace_path,
        goal=request.goal,
        initial_message=request.initial_message,
    )
    snapshot = conversation_sessions.reasoning_snapshot(session.session_id)

    return {
        "session_id": session.session_id,
        "stage": session.stage,
        "workspace_path": session.workspace_path,
        "reasoning": snapshot.model_dump(),
        "conversation": session.conversation.model_dump(),
    }


@router.post("/conversation/{session_id}/message")
async def add_conversation_message(session_id: str, request: ConversationMessageRequest):
    """Add user message to conversation and refresh reasoning."""
    try:
        session = conversation_sessions.add_user_message(
            session_id=session_id,
            text=request.text,
            images=request.images,
            files=request.files,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    snapshot = conversation_sessions.reasoning_snapshot(session_id)
    return {
        "session_id": session.session_id,
        "stage": session.stage,
        "reasoning": snapshot.model_dump(),
        "context": session.conversation.context.model_dump(),
    }


@router.get("/conversation/{session_id}")
async def get_conversation_session(session_id: str):
    """Return full conversation session state."""
    try:
        session = conversation_sessions.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    snapshot = conversation_sessions.reasoning_snapshot(session_id)
    return {
        "session": session.model_dump(),
        "reasoning": snapshot.model_dump(),
    }


@router.get("/conversation/{session_id}/reasoning")
async def get_conversation_reasoning(session_id: str):
    """Return transparent reasoning snapshot (primary/secondary/elimination/next actions)."""
    try:
        snapshot = conversation_sessions.reasoning_snapshot(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return snapshot.model_dump()


@router.post("/conversation/{session_id}/feedback")
async def add_conversation_feedback(session_id: str, request: ConversationFeedbackRequest):
    """Apply user correction feedback and regenerate reasoning."""
    try:
        session = conversation_sessions.apply_feedback(session_id, request.feedback)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    snapshot = conversation_sessions.reasoning_snapshot(session_id)
    return {
        "session_id": session.session_id,
        "stage": session.stage,
        "reasoning": snapshot.model_dump(),
        "rejected_decisions": session.conversation.context.rejected_decisions,
    }


@router.post("/conversation/{session_id}/run")
async def run_from_conversation(session_id: str, request: ConversationRunRequest):
    """Synthesize a ticket from conversation context and run autonomous workflow."""
    try:
        session = conversation_sessions.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    ticket = conversation_sessions.synthesize_ticket(session_id)
    conversation_sessions.mark_running(session_id)

    try:
        state = run_autonomous_workflow(
            ticket=ticket,
            workspace_path=session.workspace_path,
            technology=request.technology,
            max_fix_attempts=request.max_fix_attempts,
        )
        workflow_states[ticket.ticket_id] = state
        status = _status_value(_safe_get(state, "status", "unknown"))
        errors = _safe_get(state, "errors", []) or []
        conversation_sessions.attach_workflow_result(session_id, status=status, errors=[str(e) for e in errors])
        return {
            "session_id": session_id,
            "ticket_id": ticket.ticket_id,
            "status": status,
            "errors": [str(e) for e in errors],
            "generated_files": [
                {"path": f["path"], "language": f["language"], "lines": f["lines"]}
                for f in _generated_files(state)
            ],
            "reasoning": conversation_sessions.reasoning_snapshot(session_id).model_dump(),
        }
    except Exception as exc:
        logger.error("Conversation run failed for %s: %s", session_id, exc, exc_info=True)
        conversation_sessions.attach_workflow_result(session_id, status="failed", errors=[str(exc)])
        raise HTTPException(status_code=500, detail=f"Conversation run failed: {exc}") from exc


@router.get("/status/{ticket_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(ticket_id: str):
    """
    Get the current status of a workflow.
    
    Args:
        ticket_id: ValueEdge ticket ID
        
    Returns:
        Current workflow status and results
    """
    if ticket_id not in workflow_states:
        raise HTTPException(
            status_code=404,
            detail=f"No workflow found for ticket: {ticket_id}"
        )
    
    state = workflow_states[ticket_id]
    return _build_status_response(state, ticket_id_fallback=ticket_id)


@router.get("/files/{ticket_id}", response_model=List[GeneratedFileResponse])
async def get_generated_files(ticket_id: str):
    """
    Get all generated files for a ticket.
    
    Args:
        ticket_id: ValueEdge ticket ID
        
    Returns:
        List of generated files with code
    """
    if ticket_id not in workflow_states:
        raise HTTPException(
            status_code=404,
            detail=f"No workflow found for ticket: {ticket_id}"
        )
    
    state = workflow_states[ticket_id]

    files = _generated_files(state)
    if not files:
        return []

    return [
        GeneratedFileResponse(
            file_path=f["path"],
            language=f["language"],
            code=f["code"],
            explanation=f["explanation"],
        )
        for f in files
    ]


@router.get("/explanation/{ticket_id}")
async def get_investigation_explanation(ticket_id: str):
    """
    Get investigation explanation for tickets that don't need code.
    
    Args:
        ticket_id: ValueEdge ticket ID
        
    Returns:
        Investigation results and explanation
    """
    if ticket_id not in workflow_states:
        raise HTTPException(
            status_code=404,
            detail=f"No workflow found for ticket: {ticket_id}"
        )
    
    state = workflow_states[ticket_id]

    investigation = _safe_get(state, "investigation_result")
    if not investigation:
        raise HTTPException(
            status_code=400,
            detail="Investigation has not been completed yet"
        )

    if isinstance(investigation, dict):
        ticket_type = investigation.get("ticket_type")
        requires_code = investigation.get("requires_code_changes")
        root_cause = investigation.get("root_cause_hypothesis")
        affected_systems = investigation.get("affected_systems")
        possible_causes = investigation.get("possible_causes")
        recommended_action = investigation.get("recommended_action")
        explanation = investigation.get("explanation")
        confidence = investigation.get("confidence")
    else:
        ticket_type = _status_value(getattr(investigation, "ticket_type", None))
        requires_code = getattr(investigation, "requires_code_changes", None)
        root_cause = getattr(investigation, "root_cause_hypothesis", None)
        affected_systems = getattr(investigation, "affected_systems", None)
        possible_causes = getattr(investigation, "possible_causes", None)
        recommended_action = getattr(investigation, "recommended_action", None)
        explanation = getattr(investigation, "explanation", None)
        confidence = getattr(investigation, "confidence", None)

    return {
        "ticket_id": ticket_id,
        "ticket_type": ticket_type,
        "requires_code": requires_code,
        "root_cause": root_cause,
        "affected_systems": affected_systems,
        "possible_causes": possible_causes,
        "recommended_action": recommended_action,
        "explanation": explanation,
        "confidence": confidence,
    }


@router.delete("/workflow/{ticket_id}")
async def cancel_workflow(ticket_id: str):
    """
    Cancel a running workflow.
    
    Args:
        ticket_id: ValueEdge ticket ID
    """
    if ticket_id not in workflow_states:
        raise HTTPException(
            status_code=404,
            detail=f"No workflow found for ticket: {ticket_id}"
        )
    
    state = workflow_states[ticket_id]
    if isinstance(state, dict):
        state["status"] = "cancelled"
    else:
        state.status = WorkflowStatus.CANCELLED
    
    return {"message": f"Workflow cancelled: {ticket_id}"}


# ============================================================================
# BACKGROUND TASK FUNCTIONS
# ============================================================================

async def _execute_workflow_async(
    ticket: ValueEdgeTicket,
    workspace_path: str,
    codebase_path: Optional[str],
    technology: Optional[str],
    auto_execute: bool,
    execution_mode: str = "auto",
):
    """Execute workflow asynchronously in background"""
    try:
        logger.info(f"Starting async workflow for: {ticket.ticket_id}")
        
        # Run workflow
        state = run_autonomous_workflow(
            ticket=ticket,
            workspace_path=workspace_path,
            codebase_path=codebase_path,
            technology=technology,
            auto_execute=auto_execute,
            execution_mode=execution_mode,
        )
        
        # Store state
        workflow_states[ticket.ticket_id] = state
        
        logger.info(
            "Workflow completed: %s - Status: %s",
            ticket.ticket_id,
            _status_value(_safe_get(state, "status")),
        )
        
    except Exception as e:
        logger.error(f"Workflow failed for {ticket.ticket_id}: {e}", exc_info=True)
        
        # Store error state
        if ticket.ticket_id in workflow_states:
            existing = workflow_states[ticket.ticket_id]
            if isinstance(existing, dict):
                existing["status"] = "failed"
                existing.setdefault("errors", []).append(str(e))
            else:
                existing.status = WorkflowStatus.FAILED
                existing.errors.append(str(e))


async def _execute_run_async(
    run_id: str,
    ticket: ValueEdgeTicket,
    request: CreateRunRequest,
):
    """Execute run flow with backend-owned status and event schema."""
    run = _ensure_run(run_id)

    if run.get("status") == "awaiting_approval":
        run["pending_approval_start"] = {"ticket": ticket, "request": request}
        return

    if run.get("requires_clarification"):
        run["pending_resume"] = {"ticket": ticket, "request": request}
        return

    run["status"] = "running"
    run["current_stage"] = "classification"
    _emit_run_event(run_id, event_type="stage.started", stage="classification", status="running")

    acceptance_criteria = request.acceptance_criteria or [
        "Implementation satisfies the ticket description without unrelated changes."
    ]
    contract_ticket = {
        "ticket_id": ticket.ticket_id,
        "title": ticket.title,
        "description": ticket.description,
        "type": _infer_contract_ticket_type(ticket, request.labels),
        "expected_changed_files": [],
        "forbidden_files": [],
        "acceptance_criteria": acceptance_criteria,
    }
    contract_policy = {
        "no_hardcoding": any("hardcoding" in c.lower() for c in request.constraints),
        "require_expected_scope": False,
        "max_retries": request.max_fix_attempts,
    }

    try:
        loop = asyncio.get_running_loop()

        def _workflow_progress(event: Dict[str, Any]) -> None:
            node_name = str(event.get("node", "unknown"))
            node_status = event.get("node_status")
            retry_attempt = event.get("retry_attempt")

            def _emit_node_events() -> None:
                _emit_run_event(
                    run_id,
                    event_type="stage.started",
                    stage=f"node:{node_name}",
                    status="running",
                    payload={"source": "langgraph", "node": node_name},
                )
                payload: Dict[str, Any] = {"source": "langgraph", "node": node_name}
                if node_status is not None:
                    payload["node_status"] = str(node_status)
                if isinstance(retry_attempt, int):
                    payload["retry_attempt"] = retry_attempt
                    if retry_attempt > int(run_states.get(run_id, {}).get("retries_used", 0)):
                        run_states[run_id]["retries_used"] = retry_attempt
                        _emit_run_event(
                            run_id,
                            event_type="run.retry",
                            stage=f"node:{node_name}",
                            status="running",
                            payload={"retry_attempt": retry_attempt, "node": node_name},
                        )
                _emit_run_event(
                    run_id,
                    event_type="stage.completed",
                    stage=f"node:{node_name}",
                    status="running",
                    payload=payload,
                )

            loop.call_soon_threadsafe(_emit_node_events)

        _emit_run_event(run_id, event_type="stage.completed", stage="classification", status="running")
        _emit_run_event(run_id, event_type="stage.started", stage="strategy_selection", status="running")
        _emit_run_event(run_id, event_type="stage.completed", stage="strategy_selection", status="running")
        _emit_run_event(run_id, event_type="stage.started", stage="discovery", status="running")

        state = await asyncio.to_thread(
            run_autonomous_workflow,
            ticket=ticket,
            workspace_path=request.repo_scope,
            codebase_path=request.repo_scope,
            technology=None,
            auto_execute=False,
            max_fix_attempts=request.max_fix_attempts,
            execution_mode=request.execution_mode,
            contract_ticket=contract_ticket,
            contract_policy=contract_policy,
            progress_callback=_workflow_progress,
        )
        workflow_states[ticket.ticket_id] = state

        status = _status_value(_safe_get(state, "status", "unknown"))
        run["final_report"] = _build_final_report(state, run_id)
        run["status"] = status
        run["current_stage"] = "completed" if status in {"solved", "no_action_required", "completed", "success"} else "validation"
        run["retries_used"] = int(_safe_get(state, "retry_count", 0) or 0)

        _emit_run_event(
            run_id,
            event_type="stage.completed",
            stage="validation",
            status=run["status"],
            payload={"final_status": status},
        )
        _emit_run_event(
            run_id,
            event_type="run.completed",
            stage=run["current_stage"],
            status=run["status"],
            payload={"ticket_id": ticket.ticket_id},
        )

        if status == "need_more_info":
            # Track clarification rounds to prevent infinite loops
            clarification_round = run.get("clarification_round", 0) + 1
            run["clarification_round"] = clarification_round
            _MAX_CLARIFICATION_ROUNDS = 3

            if clarification_round > _MAX_CLARIFICATION_ROUNDS:
                logger.warning(
                    f"Run {run_id}: clarification round {clarification_round} "
                    f"exceeds max {_MAX_CLARIFICATION_ROUNDS} — failing"
                )
                run["status"] = "failed"
                run["blocking_reason"] = (
                    f"Exceeded max clarification rounds ({_MAX_CLARIFICATION_ROUNDS}). "
                    f"Ticket may be too vague for automated resolution."
                )
            else:
                run["requires_clarification"] = True
                run["blocking_reason"] = "need_more_info"
                # Read the actual question from the workflow state instead of
                # hardcoding. The agentic loop sets state["clarification_question"]
                # with a specific question tailored to what it couldn't find.
                workflow_question = _safe_get(state, "clarification_question", None)
                run["clarification"] = {
                    "run_id": run_id,
                    "question": workflow_question or "Please provide additional constraints or examples for this ticket.",
                    "requested_at": _now_iso(),
                }
                _emit_run_event(
                    run_id,
                    event_type="run.need_more_info",
                    stage="clarification",
                    status="need_more_info",
                    payload=run["clarification"],
                )

    except Exception as exc:
        logger.error("Run failed for %s: %s", run_id, exc, exc_info=True)
        run["status"] = "failed"
        run["blocking_reason"] = str(exc)
        run["final_report"] = {
            "run_id": run_id,
            "ticket_id": run["ticket_id"],
            "status": "failed",
            "findings": ["orchestrator_exception"],
            "decisions": [],
            "changed_files": [],
            "validation_evidence": [],
            "residual_risks": [str(exc)],
            "why_changed": [],
        }
        _emit_run_event(
            run_id,
            event_type="run.failed",
            stage=run.get("current_stage", "unknown"),
            status="failed",
            payload={"error": str(exc)},
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Autonomous Ticket-to-Code",
        "active_workflows": len([
            s for s in workflow_states.values()
            if _status_value(_safe_get(s, "status")) not in ["completed", "failed", "cancelled", "success"]
        ])
    }
