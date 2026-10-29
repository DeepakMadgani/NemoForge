"""
Planning Recovery Agent — LLM-driven diagnosis of planning failures.

When candidate validation produces 0 writable tasks, this module diagnoses
WHY from the full failure context and produces a structured PlanningRecoveryAction
that feeds targeted context into the next investigation cycle.

CONSTRAINT 3: ``genuinely_unrecoverable`` requires a HIGH evidence bar.
Planner uncertainty alone can never terminate the ticket.  The LLM must
establish that:
  1. Requirements are understood
  2. Relevant evidence was inspected
  3. Candidate locations were considered
  4. Verification state was understood
  5. No viable writable implementation location exists
  6. No additional investigation can materially resolve the blocker

The recovery node is SOLVE-ORIENTED: if there is a plausible path to
obtaining missing evidence or finding a correct modification point,
choose recovery rather than terminal failure.
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from ticket_to_code.models import PlanningRecoveryAction

logger = logging.getLogger(__name__)


def diagnose_planning_failure(
    ticket: Any,
    requirements: Any,
    evidence_items: List[Any],
    verification_results: List[Dict],
    validation_failure_reason: Optional[str],
    hypotheses: List[Any],
) -> PlanningRecoveryAction:
    """
    LLM-driven diagnosis of why planning produced 0 writable tasks.

    Reasons from the full failure context — not heuristic classifiers.
    Produces a PlanningRecoveryAction with targeted recovery context.

    Args:
        ticket: The original ticket (title, description, ticket_id).
        requirements: Parsed requirements (functional_requirements, technical_requirements).
        evidence_items: All collected evidence items.
        verification_results: Semantic verification feedback (with verification_status).
        validation_failure_reason: Why candidate validation rejected everything.
        hypotheses: Investigation hypotheses that guided evidence collection.

    Returns:
        PlanningRecoveryAction with structured recovery context.
    """
    from ticket_to_code.runtime.llm_factory import LLMRegistry
    from ticket_to_code.llm_utils import llm_invoke
    from langchain_core.messages import HumanMessage

    # ── Build context sections ────────────────────────────────────────────

    ticket_title = getattr(ticket, "title", "") or ""
    ticket_desc = getattr(ticket, "description", "") or ""
    ticket_id = getattr(ticket, "ticket_id", "unknown")

    # Requirements
    req_text = "(no requirements parsed)"
    if requirements:
        func_reqs = getattr(requirements, "functional_requirements", [])
        tech_reqs = getattr(requirements, "technical_requirements", [])
        parts = []
        for r in func_reqs[:8]:
            parts.append(f"  - [FUNC] {r}")
        for r in tech_reqs[:5]:
            parts.append(f"  - [TECH] {r}")
        if parts:
            req_text = "\n".join(parts)

    # Evidence summary
    evidence_text = "(no evidence collected)"
    if evidence_items:
        ev_lines = []
        for e in evidence_items[:20]:
            fp = getattr(e, "file_path", str(e))
            score = getattr(e, "relevance_score", 0.0)
            snippet = getattr(e, "content_snippet", "")[:80]
            provider = getattr(e, "provider", "?")
            ev_lines.append(f"  - [{provider}] {fp} (score={score:.2f}): {snippet}")
        evidence_text = "\n".join(ev_lines)

    # Verification results with infrastructure status
    verification_text = "(no verification results)"
    if verification_results:
        ver_lines = []
        for vr in verification_results[:15]:
            v_status = vr.get("verification_status", "completed")
            decision = vr.get("decision", "?")
            fp = vr.get("file_path", "?")
            reason = vr.get("reason", "")[:100]
            if v_status != "completed":
                ver_lines.append(f"  - ⚠️ UNVERIFIED: {fp} ({v_status})")
            else:
                ver_lines.append(f"  - [{decision.upper()}] {fp}: {reason}")
        verification_text = "\n".join(ver_lines)

    # Hypotheses
    hyp_text = "(no hypotheses)"
    if hypotheses:
        hyp_lines = []
        for h in hypotheses[:5]:
            hyp_str = getattr(h, "hypothesis", str(h))[:150]
            hyp_id = getattr(h, "id", "?")
            hyp_lines.append(f"  - [{hyp_id}] {hyp_str}")
        hyp_text = "\n".join(hyp_lines)

    # Validation failure
    failure_text = validation_failure_reason or "(no specific reason recorded)"

    # ── Count contaminated entries ────────────────────────────────────────
    contaminated = [
        vr.get("file_path", "?")
        for vr in (verification_results or [])
        if vr.get("verification_status", "completed") != "completed"
    ]

    prompt = f"""You are a planning recovery specialist for an autonomous code modification engine.

The system collected evidence and attempted to plan code changes for a ticket,
but the planner produced 0 WRITABLE tasks. Your job is to diagnose WHY and
determine the best recovery action.

TICKET:
  Title: {ticket_title}
  Description: {ticket_desc[:600]}

REQUIREMENTS:
{req_text}

HYPOTHESES (what the investigation agent believed needed to change):
{hyp_text}

EVIDENCE COLLECTED ({len(evidence_items)} items):
{evidence_text}

VERIFICATION RESULTS:
{verification_text}

VALIDATION FAILURE REASON:
  {failure_text}

CONTAMINATED ENTRIES (infrastructure failures during verification):
  {', '.join(contaminated) if contaminated else '(none)'}

TASK: Diagnose the failure by reasoning through these steps IN ORDER:

Step 1: Were any evidence items unverified due to infrastructure failure?
  If yes → the planner may have lacked critical information about those files.
  Recovery type: "evidence_contaminated"

Step 2: Does the evidence cover the ticket's requirements, or are specific aspects missing?
  If specific evidence is missing → identify exactly WHAT file/component/query would fill the gap.
  Recovery type: "evidence_incomplete"
  You MUST provide investigation_target and investigation_query.

Step 3: Were candidates rejected because the planner couldn't identify a modification point
  (not because the files are genuinely immutable)?
  If yes → the discovery scope may need to expand or shift.
  Recovery type: "wrong_candidates"
  You MUST provide alternative_search_terms.

Step 4: Is the ticket's requirement genuinely ambiguous (multiple valid interpretations)?
  Recovery type: "requirement_ambiguous"

Step 5: Is the requested behavior already present in the evidence (no change needed)?
  Recovery type: "already_implemented"

Step 6: ONLY after steps 1-5 have been ruled out with SPECIFIC evidence:
  Recovery type: "genuinely_unrecoverable"

  CRITICAL: This is a HIGH-BAR conclusion. You must establish ALL of these:
    ✓ Requirements are understood (cite the specific requirements)
    ✓ Relevant evidence was inspected (cite specific files and what was found)
    ✓ Candidate locations were considered (cite what was tried and why it failed)
    ✓ Verification state was understood (no infrastructure contamination)
    ✓ No viable writable implementation location exists (explain why)
    ✓ No additional investigation can materially resolve the blocker (explain why not)

  If ANY of these cannot be established, choose a recovery type instead.
  PLANNER UNCERTAINTY ALONE IS NEVER SUFFICIENT FOR TERMINAL FAILURE.

  The system should be SOLVE-ORIENTED: if there is a plausible path to
  obtaining missing evidence or finding a correct modification point,
  choose recovery rather than terminal failure.

Return your diagnosis as a JSON object:
{{
  "recovery_type": "<one of the types above>",
  "reason": "<human-readable diagnosis>",
  "investigation_target": "<file or component to investigate, if applicable>",
  "investigation_query": "<specific search query, if applicable>",
  "alternative_search_terms": ["<term1>", "<term2>"],
  "contaminated_entries": ["<filepath1>", ...],
  "reasoning_points": ["<step1>", "<step2>", ...],
  "confidence": <0.0 to 1.0>
}}

ONLY output valid JSON. DO NOT wrap in markdown."""

    recovery_id = f"R{uuid.uuid4().hex[:6].upper()}"

    try:
        llm = LLMRegistry.get_llm(assistant=False)
        response = llm_invoke(llm, [HumanMessage(content=prompt)])
        text = response.content

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        action = PlanningRecoveryAction.model_validate_json(text)
        action.recovery_id = recovery_id
        action.status = "active"

        logger.info(
            f"  [PlanningRecovery] Diagnosis: {action.recovery_type} "
            f"(confidence={action.confidence:.2f}, id={recovery_id})"
        )
        for pt in action.reasoning_points[:4]:
            logger.info(f"    • {pt}")

        return action

    except Exception as exc:
        logger.warning(f"  [PlanningRecovery] LLM diagnosis failed: {exc}")

        # Fallback: if there are contaminated entries, assume evidence_contaminated.
        # Otherwise assume evidence_incomplete (solve-oriented default).
        if contaminated:
            return PlanningRecoveryAction(
                recovery_type="evidence_contaminated",
                reason=f"LLM diagnosis failed ({exc}); {len(contaminated)} unverified entries detected",
                contaminated_entries=contaminated,
                reasoning_points=["Fallback: LLM diagnosis unavailable", f"Detected {len(contaminated)} contaminated entries"],
                confidence=0.4,
                recovery_id=recovery_id,
                status="active",
            )
        else:
            return PlanningRecoveryAction(
                recovery_type="evidence_incomplete",
                reason=f"LLM diagnosis failed ({exc}); defaulting to evidence_incomplete (solve-oriented)",
                reasoning_points=["Fallback: LLM diagnosis unavailable", "Defaulting to solve-oriented recovery"],
                confidence=0.3,
                recovery_id=recovery_id,
                status="active",
            )
