"""
Ticket-type strategies (skills) for the CC4E reasoning engine.

A strategy is lightweight prior knowledge that seeds the reasoner for a class
of tickets: which tools to prefer first, typical files, typical validation, and
common mistakes. This is the "Skill System" — the reasoner loads only the skill
relevant to the current ticket instead of using one generic planner.

Strategies are ADVISORY. The reasoner still decides dynamically; strategies just
give it a strong, CC4E-specific starting point so it doesn't rediscover basics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TicketStrategy:
    name: str
    matches: List[str]                       # ticket-type keys / keywords
    preferred_tools: List[str] = field(default_factory=list)
    guidance: str = ""
    common_mistakes: List[str] = field(default_factory=list)


_STRATEGIES: List[TicketStrategy] = [
    TicketStrategy(
        name="version_change",
        matches=["version_bump", "version", "release"],
        preferred_tools=["recall_memory", "find_owner", "search_code", "read_file", "write_patch", "verify_outcome"],
        guidance=(
            "Version tickets change a canonical version literal. Find the OWNER config "
            "(e.g. application.yml/properties), replace the old value everywhere it is the "
            "source of truth, and verify the old literal no longer remains."
        ),
        common_mistakes=[
            "Editing UI/display files instead of the canonical config owner.",
            "Leaving the old version string in some files.",
        ],
    ),
    TicketStrategy(
        name="configuration",
        matches=["configuration", "config", "dependency", "deployment"],
        preferred_tools=["recall_memory", "find_owner", "grep", "read_file", "write_patch", "verify_outcome"],
        guidance=(
            "Configuration tickets modify config files (yml/properties/json/xml/env). "
            "Confirm the exact key and change only the owning config file(s)."
        ),
        common_mistakes=["Changing code when only config needed to change."],
    ),
    TicketStrategy(
        name="api_change",
        matches=["api_change", "api", "endpoint"],
        preferred_tools=["recall_memory", "search_code", "grep", "read_file", "write_patch", "run_build", "verify_outcome"],
        guidance=(
            "API tickets change endpoints/controllers/DTOs. Update the route + controller + "
            "any swagger/spec, then build and verify the old endpoint is gone and the new one exists."
        ),
        common_mistakes=["Renaming the route but leaving old references/tests."],
    ),
    TicketStrategy(
        name="ui",
        matches=["ui", "css", "style", "layout", "angular", "footer", "header"],
        preferred_tools=["recall_memory", "find_owner", "rag", "read_file", "write_patch", "run_build", "verify_outcome"],
        guidance=(
            "UI tickets change Angular templates/components/styles. Locate the component that "
            "renders the target element, edit template/style, then build to catch TS errors."
        ),
        common_mistakes=["Editing the wrong component; not compiling TypeScript."],
    ),
    TicketStrategy(
        name="bug_fix",
        matches=["bug_fix", "bug", "defect", "fix", "race", "performance"],
        preferred_tools=["recall_memory", "find_owner", "rag", "grep", "read_file", "write_patch", "run_build", "verify_outcome"],
        guidance=(
            "Bug tickets need root-cause reasoning. Reproduce the code path, read the relevant "
            "files, form a hypothesis, apply the minimal fix, then build/verify."
        ),
        common_mistakes=["Patching symptoms instead of root cause."],
    ),
]

_GENERIC = TicketStrategy(
    name="generic",
    matches=[],
    preferred_tools=["recall_memory", "find_owner", "rag", "search_code", "read_file", "write_patch", "run_build", "verify_outcome"],
    guidance=(
        "Understand the ticket, gather just enough context by reading files on demand, "
        "make the minimal correct change, then build and verify the outcome."
    ),
    common_mistakes=["Over-reading files; changing more than necessary."],
)


def select_strategy(ticket_type: str, ticket_text: str) -> TicketStrategy:
    tt = (ticket_type or "").lower()
    text = (ticket_text or "").lower()
    for strat in _STRATEGIES:
        for m in strat.matches:
            if m == tt or m in text:
                return strat
    return _GENERIC
