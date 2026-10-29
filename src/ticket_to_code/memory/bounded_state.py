"""DORMANT: not used by any production code path (only by tests) as of 2026-10-29.
Kept because tests pin its behaviour; wire it in or delete it together with its tests.


Bounded Workflow State & Purpose-Built Context Assembly Subsystem for NemoForge.

Eliminates LangGraph context explosion by storing heavy dumps in artifact references
and assembling bounded, prompt-specific context budgets for each agent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ArtifactRef:
    artifact_id: str
    summary: str
    file_path: Optional[str] = None
    line_range: Optional[tuple[int, int]] = None
    byte_size: int = 0
    sha256: str = ""


@dataclass
class ResourceBudget:
    max_input_tokens: int = 32000
    max_output_tokens: int = 4096
    max_retries: int = 3
    remaining_retries: int = 3
    max_wall_time_seconds: int = 600


@dataclass
class CompactStageSummary:
    stage_name: str
    decisions: List[str]
    unresolved_questions: List[str] = field(default_factory=list)
    failure_evidence: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class BoundedTicketToCodeState:
    ticket_id: str
    ticket_title: str
    ticket_description: str
    workspace_root: str
    requirements_summary: str
    hypothesis: Optional[str] = None
    evidence_refs: List[ArtifactRef] = field(default_factory=list)
    candidate_refs: List[ArtifactRef] = field(default_factory=list)
    validation_refs: List[ArtifactRef] = field(default_factory=list)
    stage_summaries: List[CompactStageSummary] = field(default_factory=list)
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    status: str = "initialized"


class ContextAssembler:
    """Assembles strictly bounded, purpose-built prompts for LLM invocations."""

    def __init__(self, artifact_storage_dir: Path):
        self.storage_dir = Path(artifact_storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def store_artifact(
        self,
        content: str,
        summary: str,
        file_path: Optional[str] = None,
        line_range: Optional[tuple[int, int]] = None,
    ) -> ArtifactRef:
        """Persist heavy content to disk and return a compact reference."""
        content_bytes = content.encode("utf-8", errors="replace")
        h = hashlib.sha256(content_bytes).hexdigest()
        artifact_id = f"art_{h[:16]}"
        target_path = self.storage_dir / f"{artifact_id}.txt"
        if not target_path.exists():
            target_path.write_bytes(content_bytes)

        return ArtifactRef(
            artifact_id=artifact_id,
            summary=summary,
            file_path=file_path,
            line_range=line_range,
            byte_size=len(content_bytes),
            sha256=h,
        )

    def assemble_planning_context(self, state: BoundedTicketToCodeState) -> str:
        """Builds a compact prompt for the Planning Agent respecting the token budget."""
        parts = [
            f"# TICKET: {state.ticket_id} - {state.ticket_title}",
            f"## Requirements:\n{state.requirements_summary}",
        ]
        if state.hypothesis:
            parts.append(f"## Current Hypothesis:\n{state.hypothesis}")

        parts.append("## Discovered Evidence (Summaries):")
        for ev in state.evidence_refs[-5:]:  # Limit to 5 most relevant references
            parts.append(f"- [{ev.file_path or 'unknown'}] {ev.summary}")

        if state.validation_refs:
            latest_val = state.validation_refs[-1]
            parts.append(f"## Recent Validation Result:\n{latest_val.summary}")

        return "\n\n".join(parts)

    def assemble_code_repair_context(
        self,
        state: BoundedTicketToCodeState,
        latest_error_snippet: str,
    ) -> str:
        """Builds a compact prompt for Error Resolution Agent without full logs."""
        parts = [
            f"# REPAIR TASK FOR: {state.ticket_id}",
            f"## Target Requirement:\n{state.requirements_summary}",
            f"## Diagnostic Error Snippet:\n```\n{latest_error_snippet[:1500]}\n```",
            f"## Remaining Fix Retries: {state.budget.remaining_retries}",
        ]
        if state.stage_summaries:
            latest_stage = state.stage_summaries[-1]
            parts.append(f"## Prior Decision History: {'; '.join(latest_stage.decisions)}")

        return "\n\n".join(parts)
