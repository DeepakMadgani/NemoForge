"""
Ticket-to-Code ADT Plugin Tools

Exposes the Autonomous Ticket-to-Code Engine as a native tool
for the Aviator ADT's ContentAviatorAgent. The ADT Assistant can invoke 
this tool to hand over control to the backend Orchestrator.

Author: AI Assistant
Date: April 2026
"""

import logging
import os
import re
from typing import Annotated, List, Dict
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from opentelemetry import trace

from ticket_to_code.runtime.compat_models import StateModel
from ticket_to_code.runtime.settings import settings
from ticket_to_code.models import TicketPriority
from ticket_to_code.conversation.session_manager import ConversationSessionManager
from ticket_to_code.workflow import run_autonomous_workflow

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
conversation_manager = ConversationSessionManager()


def _resolve_workspace_path(task_description: str, messages: List[Dict[str, str]]) -> str:
    """Resolve the best workspace path dynamically from environment or user intent, with safe fallback."""
    # 1. Check configured environment variables for active workspace
    env_ws = os.getenv("AVIATOR_WORKSPACE_PATH") or os.getenv("WORKSPACE_PATH") or os.getenv("TARGET_WORKSPACE_PATH")
    if env_ws and Path(env_ws).exists() and Path(env_ws).is_dir():
        return str(Path(env_ws).resolve())

    settings_base = getattr(settings, "base_dir", None)
    fallback = Path(settings_base) if settings_base else Path.cwd()
    search_text = "\n".join([
        task_description or "",
        *[m.get("content", "") for m in messages],
    ])

    # Prefer explicit Windows or POSIX paths mentioned by the user.
    path_matches = re.findall(r"[A-Za-z]:\\[^\n\r\"']+|/[^\n\r\"']+", search_text)
    for raw_path in path_matches:
        candidate = Path(raw_path.strip())
        if candidate.exists() and candidate.is_dir():
            return str(candidate.resolve())

    return str(fallback.resolve() if isinstance(fallback, Path) else Path(fallback).resolve())


class AviatorConversationAdapter:
    """
    Adapter layer ensuring the Engineering Engine remains 100% frontend-agnostic.
    Converts Aviator's specific StateModel into standard Python dicts/lists.
    """
    @staticmethod
    def extract_messages(state: StateModel) -> List[Dict[str, str]]:
        # In a real environment, this might also extract image base64, attached files, etc.
        return [{"role": m.type, "content": m.content} for m in state.messages]

    @staticmethod
    def extract_acceptance_criteria(messages: List[Dict[str, str]]) -> List[str]:
        criteria: List[str] = []
        for msg in messages:
            text = msg.get("content", "")
            for line in text.splitlines():
                clean = line.strip()
                if clean.startswith("-"):
                    criteria.append(clean[1:].strip())
                elif clean[:2].isdigit() and clean[2:3] in {".", ")"}:
                    criteria.append(clean[3:].strip())
        return criteria[:10]


@tool
@tracer.start_as_current_span("run_autonomous_engineering_engine")
def run_autonomous_engineering_engine(
    task_description: str,
    state: Annotated[StateModel, InjectedState],
) -> str:
    """Execute the full autonomous software engineering pipeline.
    
    Use this tool when the user asks to implement a feature, fix a bug, or execute a coding task.
    This hands over control to the dedicated Ticket-to-Code Orchestrator, which will plan, generate,
    compile, and validate the code changes.
    
    Args:
        task_description: A summary of what the user wants to accomplish.
        state: Injected ADT conversation state (contains full message history and images).
        
    Returns:
        A markdown string detailing the result, any needed clarifications, or the final patch plan.
    """
    logger.info(f"Handing over to Autonomous LangGraph Engine for task: {task_description}")
    
    # 1. Use the Adapter to isolate frontend
    standard_messages = AviatorConversationAdapter.extract_messages(state)
    acceptance_criteria = AviatorConversationAdapter.extract_acceptance_criteria(standard_messages)

    resolved_workspace = _resolve_workspace_path(task_description, standard_messages)
    logger.info("Resolved ticket-to-code workspace: %s", resolved_workspace)

    # 2. Build an interactive conversation session and synthesize an evidence-rich ticket
    session = conversation_manager.create_session(
        workspace_path=resolved_workspace,
        goal=task_description,
        initial_message=task_description,
    )
    for msg in standard_messages[-25:]:
        if msg.get("role") == "user" and msg.get("content"):
            conversation_manager.add_user_message(session.session_id, msg["content"])

    ticket = conversation_manager.synthesize_ticket(session.session_id, priority=TicketPriority.MEDIUM)
    if acceptance_criteria:
        ticket.acceptance_criteria = acceptance_criteria
    ticket.ticket_id = f"ADT-CHAT-{session.session_id}"
    ticket.labels = sorted(set(ticket.labels + ["adt-chat", "autonomous-engine"]))
    
    # 3. Run the autonomous workflow
    try:
        conversation_manager.mark_running(session.session_id)
        final_state = run_autonomous_workflow(
            ticket=ticket,
            workspace_path=resolved_workspace
        )
        
        status = final_state.get('status', 'unknown')
        errors = final_state.get('errors', []) if isinstance(final_state, dict) else []
        conversation_manager.attach_workflow_result(session.session_id, status=status, errors=errors)

        reasoning = conversation_manager.reasoning_snapshot(session.session_id)
        sections = []

        # 1. Explanation / Investigation / Solution Guidance
        inv = final_state.get('investigation_result') if isinstance(final_state, dict) else None
        guidance = getattr(inv, 'solution_guidance', None) if inv else None
        if not guidance and isinstance(final_state, dict):
            guidance = final_state.get('solution_guidance')

        if guidance:
            summary_txt = getattr(guidance, 'issue_summary', '') or getattr(guidance, 'summary', '') or ''
            root_cause = getattr(guidance, 'root_cause', '') or ''
            steps = getattr(guidance, 'step_by_step_solution', []) or getattr(guidance, 'steps', []) or []
            desc_parts = []
            if summary_txt:
                desc_parts.append(summary_txt)
            if root_cause:
                desc_parts.append(f"**Root Cause / Mechanism:** {root_cause}")
            sections.append("### 💡 Architectural Analysis & Guidance\n\n" + "\n\n".join(desc_parts))
            if steps:
                sections.append("**Resolution Steps / Verification:**\n" + "\n".join(f"- {s}" for s in steps))
        elif inv and getattr(inv, 'explanation', None):
            sections.append(f"### 💡 Codebase Investigation\n\n{inv.explanation}")

        # 2. Code Changes & Verification
        code_files = final_state.get('generated_code', []) if isinstance(final_state, dict) else []
        if code_files:
            file_list = []
            for cf in code_files:
                fp = getattr(cf, 'file_path', str(cf))
                doc = getattr(cf, 'documentation', None) or getattr(cf, 'purpose', '')
                file_list.append(f"- `{fp}`" + (f" ({doc})" if doc else ""))
            sections.append(f"### 🛠️ Code Implementation Applied\n\n**Modified/Generated Files ({len(code_files)}):**\n" + "\n".join(file_list))

        if status == "success":
            sections.insert(0, f"## ✅ Task & Analysis Completed Successfully\n**Status:** {status} | **Confidence:** {reasoning.confidence}\n**Primary Hypothesis:** {reasoning.primary_hypothesis}")
            return "\n\n".join(sections)
        elif status == "guidance":
            sections.insert(0, f"## ℹ️ Technical Guidance & Codebase Analysis\n")
            return "\n\n".join(sections)
        elif status == "failed":
            errors = final_state.get('errors', []) if isinstance(final_state, dict) else []
            error_str = "\n".join([str(e) for e in errors]) if errors else "Unknown error."
            return f"**Autonomous Engine Failed:**\n{error_str}\n\n" + "\n\n".join(sections)
        else:
            return f"Engine finished in state: {status}\n\n" + "\n\n".join(sections)
            
    except Exception as e:
        logger.error(f"Engine orchestration failed: {e}", exc_info=True)
        conversation_manager.attach_workflow_result(session.session_id, status="failed", errors=[str(e)])
        return f"Autonomous Engine encountered a fatal error: {str(e)}"
