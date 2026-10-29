"""
Autonomous Ticket-to-Code System

A fully autonomous AI system that converts ValueEdge tickets into production-ready,
tested, and integrated code using multi-agent engineering architecture.

Quick Start:
    >>> from ticket_to_code import run_autonomous_workflow, ValueEdgeTicket, TicketPriority
    >>> 
    >>> ticket = ValueEdgeTicket(
    ...     ticket_id="VE-1234",
    ...     title="Add discount code feature",
    ...     description="Users should be able to apply discount codes...",
    ...     priority=TicketPriority.HIGH
    ... )
    >>> 
    >>> result = run_autonomous_workflow(
    ...     ticket=ticket,
    ...     workspace_path=r"C:\\Projects\\MyApp\\src",
    ...     technology="dotnet",
    ...     auto_execute=True
    ... )
    >>> print(f"Generated {len(result.generated_code)} files!")

Author: Deepak Madgani
Date: April 2026
"""

__version__ = "1.0.0"

import sys
from pathlib import Path

# Standalone runtime imports
from .runtime.llm_factory import LLMFactory, LLMRegistry

# Import models
from .models import (
    ValueEdgeTicket,
    TicketPriority,
    TicketType,
    StructuredRequirements,
    LocalizationResult,
    TargetFile,
    TargetMethod,
    ExecutionPath,
    ImpactAnalysis,
    DevelopmentTask,
    ArchitecturalPlan,
    CodeChunk,
    ContextEvaluation,
    GeneratedCode,
    BuildResult,
    TestResult,
    WorkflowStatus,
    AutonomousWorkflowState,
)

# Import main workflow functions
from .workflow import (
    run_autonomous_workflow,
    run_autonomous_workflow_langgraph,
)

def run_reasoning_agent(*args, **kwargs):
    """Run the dynamic reasoning engine (ReAct-style, tool-orchestrating)."""
    from .reasoning import run_reasoning_agent as _impl
    return _impl(*args, **kwargs)

run_cc4e_reasoning_agent = run_reasoning_agent

# All exports
__all__ = [
    # Main workflow
    "run_autonomous_workflow",
    "run_autonomous_workflow_langgraph",
    "run_reasoning_agent",
    "run_cc4e_reasoning_agent",
    
    # Models
    "ValueEdgeTicket",
    "TicketPriority",
    "TicketType",
    "StructuredRequirements",
    "LocalizationResult",
    "TargetFile",
    "TargetMethod",
    "ExecutionPath",
    "ImpactAnalysis",
    "DevelopmentTask",
    "ArchitecturalPlan",
    "CodeChunk",
    "ContextEvaluation",
    "GeneratedCode",
    "BuildResult",
    "TestResult",
    "WorkflowStatus",
    "AutonomousWorkflowState",
]
