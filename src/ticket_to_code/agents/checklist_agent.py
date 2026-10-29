"""
Phase 0: ChecklistAgent — Convert ticket into a mechanical definition-of-done.

Each checklist item is:
  - A concrete, testable assertion about the code
  - Tied to a specific file or symbol
  - Verifiable by grep/AST inspection (not vibes)

Uses the same LLM as the pipeline. Called after plan_node, before generate_code_node.
Stored in state["definition_of_done"].
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


@dataclass
class BehavioralInvariant:
    """A behavioral/cardinality invariant derived from requirements with explicit provenance."""
    source_requirement_id: str  # e.g., "REQ-1" or "REQ-2"
    source_text: str            # Exact ticket phrase, e.g., "When a user is selected..."
    cardinality: str            # "COLLECTION_AT_LEAST_ONE" | "SINGLETON" | "UNCONSTRAINED"
    rule: str                   # Invariant instruction for generator/reviewer
    confidence: str = "HIGH"    # "HIGH" | "MEDIUM"


@dataclass
class ChecklistItem:
    """A single verifiable requirement derived from the ticket."""
    id: int
    description: str
    file_hint: str = ""          # expected file (e.g., "add-members.component.ts")
    symbol_hint: str = ""        # expected symbol (e.g., "allProjectMembers")
    verification_type: str = ""  # "grep" | "ast" | "semantic"
    status: str = "pending"      # "pending" | "pass" | "fail"
    evidence: str = ""           # citation when verified


@dataclass
class DefinitionOfDone:
    """Complete definition-of-done for a ticket."""
    items: List[ChecklistItem] = field(default_factory=list)
    ticket_title: str = ""
    behavioral_invariants: List[BehavioralInvariant] = field(default_factory=list)
    
    def pending_items(self) -> List[ChecklistItem]:
        return [item for item in self.items if item.status == "pending"]
    
    def failed_items(self) -> List[ChecklistItem]:
        return [item for item in self.items if item.status == "fail"]
    
    def pass_rate(self) -> float:
        if not self.items:
            return 0.0
        passed = sum(1 for item in self.items if item.status == "pass")
        return passed / len(self.items)

    def render_invariants_block(self) -> str:
        """Render behavioral invariants block for pre-injection into code generation prompts."""
        if not self.behavioral_invariants:
            return ""
        lines = [
            "════════════════════════════════════════════════════════════════",
            "BEHAVIORAL INVARIANTS & CARDINALITY RULES (DERIVED FROM REQUIREMENTS):",
            "════════════════════════════════════════════════════════════════",
        ]
        for inv in self.behavioral_invariants:
            lines.append(f"• [{inv.source_requirement_id}] Invariant ({inv.cardinality}, confidence: {inv.confidence}):")
            lines.append(f"  Source: \"{inv.source_text}\"")
            lines.append(f"  Rule: {inv.rule}")
        lines.append("Do NOT write overly-narrow conditions (e.g., length === 1 or single-item only)")
        lines.append("when the invariant demands collection-wide or plural coverage.")
        lines.append("════════════════════════════════════════════════════════════════\n")
        return "\n".join(lines)


class ChecklistAgent:
    """
    Phase 0: Convert ticket + requirements into a numbered definition-of-done
    and explicit behavioral invariants with provenance.
    
    Runs BEFORE code generation. The output is consumed by:
    - CodeGenerator: pre-injects invariants to prevent cardinality defects before writing code
    - Phase 5 (ChecklistVerifier): checks each item against generated code
    - Phase 6 (Escalation): unresolved items trigger escalation instead of silent ship
    """

    SYSTEM_PROMPT = """You are a QA lead converting a development ticket into:
1. Verifiable checklist items (concrete code assertions).
2. Behavioral Invariants with exact provenance (cardinality, collection vs singleton, condition rules).

For behavioral invariants:
- Examine requirements for cardinality: does an action apply to EACH/EVERY/ALL selected items (COLLECTION_AT_LEAST_ONE), or strictly to exactly one item (SINGLETON)?
- If user selection allows multi-select (e.g. dropdown, search items, checkboxes), infer COLLECTION_AT_LEAST_ONE and state that the behavior must apply to all selected items, not only singleton selection.
- Record the exact source requirement text as evidence.

RULES FOR CHECKLIST ITEMS:
- Each item must be independently verifiable by reading ONE file
- If an item verifies a property declaration, use `"verification_type": "declaration"`
- If an item verifies logic that SETS/ASSIGNS a property, use `"verification_type": "assignment"`
- If an item verifies a function call, use `"verification_type": "call"`
- For other logic, use `"verification_type": "logic"`
- Include both the POSITIVE case and NEGATIVE case as separate items
- Maximum 12 items. Minimum 3 items.

Respond with JSON ONLY:
{
  "behavioral_invariants": [
    {
      "source_requirement_id": "REQ-1",
      "source_text": "When a user is selected in the Add Members modal...",
      "cardinality": "COLLECTION_AT_LEAST_ONE",
      "rule": "Must handle any non-empty selection (e.g. searchData.length > 0 or iteration); do NOT restrict logic to singleton length === 1.",
      "confidence": "HIGH"
    }
  ],
  "items": [
    {
      "description": "add-members.component.ts declares isExistingProjectMember property",
      "file_hint": "add-members.component.ts",
      "symbol_hint": "isExistingProjectMember",
      "verification_type": "declaration"
    }
  ]
}

No markdown, no explanations, ONLY valid JSON."""

    def __init__(self, llm):
        self.llm = llm

    def generate_checklist(
        self,
        ticket_title: str,
        ticket_description: str,
        requirements: Optional[object] = None,
        plan_tasks: Optional[list] = None,
    ) -> DefinitionOfDone:
        """
        Generate a definition-of-done checklist from the ticket.
        
        Args:
            ticket_title: The ticket title
            ticket_description: The ticket description
            requirements: StructuredRequirements object (if available)
            plan_tasks: List of DevelopmentTask objects from the planner (if available)
        
        Returns:
            DefinitionOfDone with numbered checklist items
        """
        # Build context from requirements
        req_context = ""
        if requirements:
            func_reqs = getattr(requirements, "functional_requirements", []) or []
            if func_reqs:
                req_context = "\n\nFUNCTIONAL REQUIREMENTS:\n" + "\n".join(
                    f"  {i+1}. {r}" for i, r in enumerate(func_reqs)
                )

        # Build context from plan tasks
        plan_context = ""
        if plan_tasks:
            plan_lines = []
            for t in plan_tasks:
                _type = getattr(getattr(t, "task_type", None), "value", "?")
                plan_lines.append(
                    f"  - [{_type.upper()}] {getattr(t, 'file_path', '?')}: "
                    f"{getattr(t, 'title', '?')}"
                )
            plan_context = "\n\nPLANNED TASKS:\n" + "\n".join(plan_lines)

        user_prompt = (
            f"TICKET: {ticket_title}\n\n"
            f"DESCRIPTION:\n{ticket_description}"
            f"{req_context}"
            f"{plan_context}"
            f"\n\nGenerate the checklist. JSON only."
        )

        try:
            from ticket_to_code.llm_utils import llm_invoke
            response = llm_invoke(
                self.llm,
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_response(raw, ticket_title)

        except Exception as e:
            logger.warning(f"ChecklistAgent failed: {e} — returning empty checklist")
            return DefinitionOfDone(ticket_title=ticket_title)

    def _parse_response(self, raw: str, ticket_title: str) -> DefinitionOfDone:
        """Parse the LLM's JSON response into a DefinitionOfDone."""
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            logger.warning("ChecklistAgent: no JSON found in response")
            return DefinitionOfDone(ticket_title=ticket_title)

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"ChecklistAgent: JSON parse failed: {e}")
            return DefinitionOfDone(ticket_title=ticket_title)

        items = []
        for i, item_data in enumerate(data.get("items", [])):
            items.append(ChecklistItem(
                id=i + 1,
                description=item_data.get("description", ""),
                file_hint=item_data.get("file_hint", ""),
                symbol_hint=item_data.get("symbol_hint", ""),
                verification_type=item_data.get("verification_type", "semantic"),
            ))

        invariants = []
        for inv_data in data.get("behavioral_invariants", []):
            invariants.append(BehavioralInvariant(
                source_requirement_id=inv_data.get("source_requirement_id", "REQ-?"),
                source_text=inv_data.get("source_text", ""),
                cardinality=inv_data.get("cardinality", "COLLECTION_AT_LEAST_ONE"),
                rule=inv_data.get("rule", ""),
                confidence=inv_data.get("confidence", "HIGH"),
            ))

        if not items:
            logger.warning("ChecklistAgent: response had zero items")

        dod = DefinitionOfDone(items=items, ticket_title=ticket_title, behavioral_invariants=invariants)
        logger.info(
            f"  ✅ ChecklistAgent: generated {len(items)} checklist items and "
            f"{len(invariants)} behavioral invariant(s) for '{ticket_title[:60]}'"
        )
        return dod
