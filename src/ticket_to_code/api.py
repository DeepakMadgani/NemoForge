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
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ticket_to_code.models import (
    ValueEdgeTicket,
    TicketPriority,
    AutonomousWorkflowState,
    WorkflowStatus
)
from ticket_to_code.workflow import (
    AutonomousWorkflowOrchestrator,
    run_autonomous_workflow
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ticket-to-code", tags=["Autonomous Workflow"])

# In-memory workflow state storage (replace with database in production)
workflow_states: dict[str, AutonomousWorkflowState] = {}


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


class GeneratedFileResponse(BaseModel):
    """Response with generated file content"""
    file_path: str
    language: str
    code: str
    explanation: str


# ============================================================================
# ENDPOINTS
# ============================================================================

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
        auto_execute=request.auto_execute
    )
    
    # Return immediate response
    return WorkflowStatusResponse(
        ticket_id=ticket.ticket_id,
        status=WorkflowStatus.INITIALIZED.value,
        generated_files=[],
        errors=[]
    )


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
    
    # Build response
    response = WorkflowStatusResponse(
        ticket_id=state.ticket.ticket_id,
        status=state.status.value,
        investigation_summary={
            "type": state.investigation_result.ticket_type.value if state.investigation_result else None,
            "code_needed": state.investigation_result.requires_code_changes if state.investigation_result else None,
            "explanation": state.investigation_result.explanation if state.investigation_result else None,
            "confidence": state.investigation_result.confidence if state.investigation_result else None,
        } if state.investigation_result else None,
        generated_files=[
            {
                "path": g.file_path,
                "language": g.language,
                "lines": len(g.code.split('\n'))
            }
            for g in state.generated_code
        ] if state.generated_code else [],
        build_status=state.build_result.status.value if state.build_result else None,
        test_results={
            "passed": state.test_result.passed,
            "failed": state.test_result.failed,
            "total": state.test_result.total,
        } if state.test_result else None,
        errors=state.errors,
        duration_seconds=(
            (state.end_time - state.start_time).total_seconds() 
            if state.end_time else None
        )
    )
    
    return response


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
    
    if not state.generated_code:
        return []
    
    return [
        GeneratedFileResponse(
            file_path=g.file_path,
            language=g.language,
            code=g.code,
            explanation=g.explanation
        )
        for g in state.generated_code
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
    
    if not state.investigation_result:
        raise HTTPException(
            status_code=400,
            detail="Investigation has not been completed yet"
        )
    
    return {
        "ticket_id": ticket_id,
        "ticket_type": state.investigation_result.ticket_type.value,
        "requires_code": state.investigation_result.requires_code_changes,
        "root_cause": state.investigation_result.root_cause_hypothesis,
        "affected_systems": state.investigation_result.affected_systems,
        "possible_causes": state.investigation_result.possible_causes,
        "recommended_action": state.investigation_result.recommended_action,
        "explanation": state.investigation_result.explanation,
        "confidence": state.investigation_result.confidence,
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
    auto_execute: bool
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
            auto_execute=auto_execute
        )
        
        # Store state
        workflow_states[ticket.ticket_id] = state
        
        # run_autonomous_workflow may return a dict-like state or an object;
        # the status may be a WorkflowStatus enum or a plain string.
        _status = state.get("status") if isinstance(state, dict) else getattr(state, "status", None)
        _status_str = getattr(_status, "value", _status)
        logger.info(f"Workflow completed: {ticket.ticket_id} - Status: {_status_str}")
        
    except Exception as e:
        logger.error(f"Workflow failed for {ticket.ticket_id}: {e}", exc_info=True)
        
        # Store error state
        if ticket.ticket_id in workflow_states:
            workflow_states[ticket.ticket_id].status = WorkflowStatus.FAILED
            workflow_states[ticket.ticket_id].errors.append(str(e))


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
            if s.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
        ])
    }
