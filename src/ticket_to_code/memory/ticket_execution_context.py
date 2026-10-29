"""
TicketExecutionContext — Per-ticket in-memory context that accumulates knowledge
across workflow phases and provides it to any agent that needs it.

NOT a new database. This is a lightweight Python object stored in the existing
_transient_store (workflow.py:722-747). It:
1. Collects context as each phase runs (requirements, component maps, type defs)
2. Answers queries from ANY agent at ANY phase
3. Tracks what's been tried and what failed during this execution
4. Dies when the workflow completes (it's transient)

Cross-ticket learning stays in RepositoryMemory (already exists).
Per-ticket context lives here.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FixAttempt:
    """Record of a single fix attempt during error resolution."""
    phase: str              # e.g. "fix_build_errors", "fix_test_failures"
    file: str               # file path that was modified
    action: str             # what was done, e.g. "added property X to interface Y"
    outcome: str            # "success", "failed", "introduced_new_errors"
    errors_before: int = 0  # error count before this fix
    errors_after: int = 0   # error count after this fix
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


class TicketExecutionContext:
    """
    Per-ticket execution context stored in _transient_store.

    Accumulates rich context from every phase so later agents (especially
    ErrorResolutionAgent) can make decisions with full project awareness.

    Usage:
        # In investigate_node():
        ctx = TicketExecutionContext(ticket_id)
        _set_transient(ticket_id, "exec_ctx", ctx)

        # In unified_analysis_node():
        ctx = _get_transient(state, "exec_ctx")
        ctx.set_requirements(reqs.functional_requirements)

        # In fix_build_errors_node():
        ctx = _get_transient(state, "exec_ctx")
        agent.exec_ctx = ctx  # Agent can query full context
    """

    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id

        # ── From unified_analysis_node (Phase 0) ─────────────────────────────
        self.ticket_title: str = ""
        self.ticket_description: str = ""
        self.functional_requirements: List[str] = []
        self.technical_requirements: List[str] = []

        # ── From plan_node (Phase 1) ─────────────────────────────────────────
        self.task_descriptions: List[str] = []
        # Maps file_path → {role, what_to_do, why}
        self.planned_file_roles: Dict[str, Dict[str, str]] = {}

        # ── From evidence_ranking_node (Phase 3B) ────────────────────────────
        # Component groups: related files that form a logical unit
        # e.g., Angular: .ts + .html + .scss; Java: Controller + Service + DTO
        self.component_groups: List[Any] = []

        # ── From dataflow_verification_node ──────────────────────────────────
        # DataFlowReport objects showing template→controller data gaps
        self.dataflow_reports: List[Any] = []

        # ── From grounded_understanding_node (Phase 2G-3) ────────────────────
        self.grounded_understanding_summary: str = ""

        # ── Type definitions discovered during discovery/planning ────────────
        # Maps type_name → {file, properties: [str], methods: [str]}
        self.type_definitions: Dict[str, Dict[str, Any]] = {}

        # ── Fix history: what's been tried this run ──────────────────────────
        self.fix_attempts: List[FixAttempt] = []
        self.build_error_history: List[List[str]] = []  # errors per build attempt

    # ── Setters (called at phase boundaries) ──────────────────────────────────

    def set_requirements(self, functional: List[str], technical: Optional[List[str]] = None):
        """Set the ticket requirements from unified_analysis_node."""
        self.functional_requirements = functional or []
        self.technical_requirements = technical or []
        logger.debug(
            "[ExecCtx] Requirements set: %d functional, %d technical",
            len(self.functional_requirements), len(self.technical_requirements),
        )

    def set_plan(self, tasks: List[Any]):
        """Set the task descriptions from plan_node."""
        self.task_descriptions = []
        self.planned_file_roles = {}
        for t in (tasks or []):
            desc = getattr(t, "description", "") or ""
            file_path = getattr(t, "file_path", "") or ""
            role = getattr(t, "role", "") or ""
            self.task_descriptions.append(desc)
            if file_path:
                self.planned_file_roles[file_path.replace("\\", "/")] = {
                    "role": role,
                    "description": desc,
                }
        logger.debug("[ExecCtx] Plan set: %d tasks", len(self.task_descriptions))

    def set_component_groups(self, groups: List[Any]):
        """Set component groups from evidence_ranking_node."""
        self.component_groups = groups or []
        logger.debug("[ExecCtx] Component groups set: %d groups", len(self.component_groups))

    def set_dataflow_reports(self, reports: List[Any]):
        """Set data flow reports from dataflow_verification_node."""
        self.dataflow_reports = reports or []
        gap_count = sum(1 for r in self.dataflow_reports if getattr(r, "has_gaps", False))
        logger.debug("[ExecCtx] DataFlow reports set: %d reports, %d with gaps", len(self.dataflow_reports), gap_count)

    def set_grounded_understanding(self, gu: Any):
        """Set grounded understanding summary."""
        if gu:
            self.grounded_understanding_summary = str(getattr(gu, "summary", "")) or ""
        logger.debug("[ExecCtx] Grounded understanding set: %d chars", len(self.grounded_understanding_summary))

    def add_type_definition(self, type_name: str, file_path: str, properties: List[str], methods: List[str]):
        """Register a discovered type/interface definition."""
        self.type_definitions[type_name] = {
            "file": file_path,
            "properties": properties,
            "methods": methods,
        }

    # ── Fix tracking ──────────────────────────────────────────────────────────

    def record_attempt(self, phase: str, file: str, action: str, outcome: str,
                       errors_before: int = 0, errors_after: int = 0):
        """Record what was tried so agents don't repeat failed approaches."""
        self.fix_attempts.append(FixAttempt(
            phase=phase, file=file, action=action, outcome=outcome,
            errors_before=errors_before, errors_after=errors_after,
        ))
        logger.debug(
            "[ExecCtx] Fix attempt recorded: %s on %s → %s (%d→%d errors)",
            action[:60], file, outcome, errors_before, errors_after,
        )

    def record_build_errors(self, errors: List[str]):
        """Record the error list from a build attempt."""
        self.build_error_history.append(list(errors))

    def get_failed_approaches(self, file: Optional[str] = None) -> List[FixAttempt]:
        """Return what's been tried and failed, optionally filtered by file."""
        results = [a for a in self.fix_attempts if a.outcome in ("failed", "introduced_new_errors")]
        if file:
            file_norm = file.replace("\\", "/").lower()
            results = [a for a in results if a.file.replace("\\", "/").lower() == file_norm]
        return results

    def get_successful_approaches(self, file: Optional[str] = None) -> List[FixAttempt]:
        """Return what's been tried and succeeded, optionally filtered by file."""
        results = [a for a in self.fix_attempts if a.outcome == "success"]
        if file:
            file_norm = file.replace("\\", "/").lower()
            results = [a for a in results if a.file.replace("\\", "/").lower() == file_norm]
        return results

    # ── Query methods (used by ErrorResolutionAgent) ──────────────────────────

    def get_type_definition(self, type_name: str) -> Optional[Dict[str, Any]]:
        """Look up an interface/class definition by name."""
        return self.type_definitions.get(type_name)

    def get_role_for_file(self, file_path: str) -> Optional[Dict[str, str]]:
        """Get the planned role/purpose for a specific file."""
        norm = file_path.replace("\\", "/")
        return self.planned_file_roles.get(norm)

    def get_dataflow_gaps_for_file(self, file_path: str) -> List[str]:
        """Get data flow gap descriptions relevant to a specific file."""
        norm = file_path.replace("\\", "/").lower()
        gaps = []
        for report in self.dataflow_reports:
            html_norm = getattr(report, "html_file", "").replace("\\", "/").lower()
            ctrl_norm = getattr(report, "controller_file", "").replace("\\", "/").lower()
            if norm in (html_norm, ctrl_norm):
                for gap in getattr(report, "gaps", []):
                    gaps.append(getattr(gap, "reason", str(gap)))
        return gaps

    # ── Prompt formatting ─────────────────────────────────────────────────────

    def to_context_block(self, max_length: int = 4000) -> str:
        """
        Format accumulated context for injection into LLM prompt.
        Truncated to max_length to respect token limits.
        """
        sections = []

        if self.functional_requirements:
            sections.append("TICKET REQUIREMENTS:")
            for i, req in enumerate(self.functional_requirements[:10], 1):
                sections.append(f"  {i}. {req}")

        if self.task_descriptions:
            sections.append("\nPLANNED CHANGES (what the code was trying to achieve):")
            for desc in self.task_descriptions[:15]:
                sections.append(f"  • {desc[:200]}")

        if self.component_groups:
            sections.append(f"\nCOMPONENT GROUPS ({len(self.component_groups)} groups):")
            for group in self.component_groups[:8]:
                primary = getattr(group, "primary_file", "?")
                related = getattr(group, "related_files", [])
                if related:
                    rel_str = ", ".join(str(r) for r in related[:5])
                    sections.append(f"  [{primary}] → {rel_str}")

        if self.dataflow_reports:
            gap_lines = []
            for report in self.dataflow_reports:
                for gap in getattr(report, "gaps", [])[:5]:
                    gap_lines.append(f"  ⚠ {getattr(gap, 'reason', str(gap))[:150]}")
            if gap_lines:
                sections.append("\nDATA FLOW GAPS (template bindings without backing data):")
                sections.extend(gap_lines[:10])

        if self.grounded_understanding_summary:
            sections.append(f"\nGROUNDED UNDERSTANDING:\n  {self.grounded_understanding_summary[:500]}")

        failed = self.get_failed_approaches()
        if failed:
            sections.append(f"\nPREVIOUS FAILED APPROACHES ({len(failed)}):")
            for fa in failed[-5:]:  # last 5 failures
                sections.append(f"  ✗ {fa.action[:120]} on {fa.file} → {fa.outcome}")

        result = "\n".join(sections)
        if len(result) > max_length:
            result = result[:max_length - 20] + "\n... (truncated)"
        return result

    def __repr__(self) -> str:
        return (
            f"TicketExecutionContext(ticket={self.ticket_id}, "
            f"reqs={len(self.functional_requirements)}, "
            f"tasks={len(self.task_descriptions)}, "
            f"groups={len(self.component_groups)}, "
            f"fixes={len(self.fix_attempts)}, "
            f"builds={len(self.build_error_history)})"
        )
