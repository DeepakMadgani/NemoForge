"""
Investigation & Triage Agent

Performs initial analysis to determine:
- What type of ticket is this? (feature, bug, investigation, config, etc.)
- Does it require code changes?
- What's the root cause hypothesis?
- What action should be taken?

This agent acts like a senior developer doing initial triage - thinking before coding.

Author: Deepak Madgani
Date: April 2026
"""

import json
import logging
import os
import re
from typing import Any, List, Optional

from ticket_to_code.llm_utils import llm_invoke

from ticket_to_code.runtime.llm_factory import LLMRegistry
from langchain_core.messages import SystemMessage, HumanMessage

from ticket_to_code.models import (
    ValueEdgeTicket,
    InvestigationResult,
    InvestigationHypothesis,
    TicketType,
    DevelopmentTask,
    TechnicalFacts,
    StageThought,
    ThinkingChain,
)
from ticket_to_code.json_utils import parse_llm_json


def _extract_thinking(raw: str) -> str:
    """Pull the <thinking>...</thinking> block from an LLM response, if present.
    Falls back to the first 600 chars of the response when no explicit tags exist.
    """
    m = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: first ~600 chars (usually contains the model's opening reasoning)
    return raw[:600].strip()

logger = logging.getLogger(__name__)


def _ticket_text(ticket: "ValueEdgeTicket") -> str:
    criteria = "\n".join(ticket.acceptance_criteria or [])
    return f"{ticket.title}\n{ticket.description}\n{criteria}"


def _explicit_repo_edit_requested(
    ticket: "ValueEdgeTicket",
    technical_facts: Optional["TechnicalFacts"],
) -> tuple[bool, str]:
    """Detect explicit repository edits from structured facts, not ticket-type heuristics.

    The LLM prompt remains the primary judge. This override exists only when the
    deterministic extractor has already found explicit file targets with concrete
    operations such as create/modify/delete.
    """
    if not technical_facts:
        return False, ""

    explicit_file_ops = [
        f for f in (technical_facts.files or [])
        if (f.operation or "unknown") in {"modify", "create", "delete"}
    ]
    if explicit_file_ops:
        targets = ", ".join(
            (f.path or f.name or "unknown") for f in explicit_file_ops[:3]
        )
        return True, f"deterministic file-operation facts extracted for {targets}"

    if technical_facts.versions:
        return True, "deterministic version transition extracted"

    return False, ""


class InvestigationAgent:
    """
    Agent 0: Investigation & Triage (NEW - First Agent in Pipeline)
    
    Makes the system intelligent by:
    - Classifying ticket type
    - Determining if code changes are needed
    - Performing root cause analysis for issues
    - Providing explanations before generating code
    
    This transforms the system from "code generator" to "intelligent assistant"
    """
    
    def __init__(self, workspace_path: Optional[str] = None):
        """Initialize with Aviator LLM.
        
        Args:
            workspace_path: Optional path to the workspace root.
                            When supplied, the agent will load the Macro Architecture
                            Brain from <workspace_path>/brain/knowledge/.
        """
        import os
        self.llm = LLMRegistry.get_llm()
        # Resolve brain_dir: always use the workspace brain if provided
        if workspace_path:
            self.brain_dir = os.path.join(workspace_path, "brain", "knowledge")
        else:
            self.brain_dir = None
        logger.info(f"Investigation & Triage Agent initialized (brain_dir={self.brain_dir})")
    
    def investigate(
        self,
        ticket: ValueEdgeTicket,
        codebase_summary: Optional[str] = None,
        technical_facts: Optional[TechnicalFacts] = None,
        thinking_chain: Optional[ThinkingChain] = None,
    ) -> tuple:
        """Investigate ticket. Returns (InvestigationResult, ThinkingChain)."""
        print("\n" + "="*80)
        print("[ENTER] InvestigationAgent.investigate() in investigation_agent.py")
        print("   Purpose: Analyze ticket and determine if code changes are needed")
        print(f"   Ticket: {ticket.ticket_id} - {ticket.title}")
        if technical_facts and technical_facts.has_facts():
            print(f"   Technical Facts: type={technical_facts.classified_type}, "
                  f"versions={len(technical_facts.versions)}, "
                  f"files={len(technical_facts.files)}, "
                  f"classes={len(technical_facts.classes)}")
        print("="*80)
        """
        Investigate ticket to determine type and required action.

        This is the FIRST step - before any code generation.

        Args:
            ticket: ValueEdge ticket to investigate
            codebase_summary: Optional high-level codebase context
            technical_facts: Deterministically extracted technical facts (zero-LLM)

        Returns:
            Investigation result with classification and recommendations
        """
        logger.info(f"🔍 Investigating ticket: {ticket.ticket_id}")

        try:
            system_prompt = self._build_investigation_prompt()
            user_prompt = self._build_user_prompt(ticket, codebase_summary, technical_facts)

            self.last_system_prompt = system_prompt
            self.last_user_prompt = user_prompt

            # Invoke LLM for intelligent analysis (with retry on transient network errors)
            response = llm_invoke(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # Parse response
            result = self._parse_response(response.content)

            explicit_repo_edit, explicit_repo_edit_reason = _explicit_repo_edit_requested(
                ticket,
                technical_facts,
            )
            if explicit_repo_edit and not result.requires_code_changes:
                logger.warning(
                    "[Investigation] Explicit repo-edit override: forcing requires_code_changes=True (%s)",
                    explicit_repo_edit_reason,
                )
                result = result.model_copy(update={
                    "requires_code_changes": True,
                    "recommended_action": (
                        "Proceed with repository file edits for the explicitly named or constrained targets"
                    ),
                    "explanation": (
                        f"[EXPLICIT REPO-EDIT OVERRIDE] The ticket contains a direct repository mutation request "
                        f"({explicit_repo_edit_reason}). "
                        f"Original LLM explanation: {result.explanation}"
                    ),
                })

            # ── Deterministic Override ────────────────────────────────────────
            # If the KnowledgeExtractor found hard facts (version transitions,
            # explicit files) with high confidence, the LLM classification must
            # not contradict them. This prevents "feature_request" or "bug_fix"
            # from overwriting "VERSION_BUMP" when the ticket explicitly says
            # "update from 260200 to 260300".
            if technical_facts and technical_facts.has_facts():
                det_type = technical_facts.classified_type  # e.g. "VERSION_BUMP"
                lm_type  = result.ticket_type.value         # e.g. "bug_fix"

                # Map deterministic type string → TicketType enum
                _DET_MAP = {
                    "VERSION_BUMP": TicketType.VERSION_BUMP,
                    "BUG":          TicketType.BUG_FIX,
                    "UI":           TicketType.BUG_FIX,
                    "CONFIG":       TicketType.CONFIGURATION,
                    "DATABASE":     TicketType.FEATURE_REQUEST,
                    "SECURITY":     TicketType.FEATURE_REQUEST,
                    "API":          TicketType.FEATURE_REQUEST,
                    "FEATURE":      TicketType.FEATURE_REQUEST,
                    "DEPENDENCY":   TicketType.VERSION_BUMP,
                }
                mapped = _DET_MAP.get(det_type)

                # Only override if:
                # 1. We have a valid mapping
                # 2. Facts confidence is high (versions or explicit files found)
                # 3. The LLM disagreed
                has_strong_facts = bool(technical_facts.versions or len(technical_facts.files) >= 2)
                if mapped and has_strong_facts and mapped != result.ticket_type:
                    force_code_change = mapped in {
                        TicketType.VERSION_BUMP,
                        TicketType.BUG_FIX,
                        TicketType.API_CHANGE,
                        TicketType.UI,
                        TicketType.DEPENDENCY,
                    }
                    logger.warning(
                        f"[Investigation] Deterministic override: "
                        f"LLM said '{lm_type}' but TechnicalFacts says '{det_type}' "
                        f"→ forcing '{mapped.value}'"
                    )
                    result = result.model_copy(update={
                        "ticket_type": mapped,
                        "requires_code_changes": True if force_code_change else result.requires_code_changes,
                        "recommended_action": (
                            "Proceed with code changes using deterministic technical facts as source-of-truth"
                            if force_code_change
                            else result.recommended_action
                        ),
                        "explanation": (
                            f"[DETERMINISTIC OVERRIDE] KnowledgeExtractor found {det_type} signals "
                            f"(versions={len(technical_facts.versions)}, files={len(technical_facts.files)}). "
                            f"LLM classified as '{lm_type}' but deterministic facts take priority. "
                            f"Original LLM explanation: {result.explanation}"
                        )
                    })

            # Guardrail: explicit version transitions always require code edits.
            if technical_facts and technical_facts.versions and not result.requires_code_changes:
                result = result.model_copy(update={
                    "ticket_type": TicketType.VERSION_BUMP,
                    "requires_code_changes": True,
                    "recommended_action": "Locate canonical version owner(s) and update version literals.",
                    "explanation": (
                        "[GUARDRAIL] Version transition literals were deterministically extracted; "
                        "ticket requires code/config updates. " + (result.explanation or "")
                    ),
                })

            # ── Deterministic literal enrichment ─────────────────────────────
            # Guarantee high-signal current-state search literals for retrieval,
            # especially for multi-step UI state regressions across tabs/views.
            if technical_facts and technical_facts.high_priority_literals:
                deterministic_literals = self._select_deterministic_literals(
                    technical_facts.high_priority_literals
                )
                existing_lower = {l.lower() for l in (result.current_state_literals or [])}
                merged_literals = list(result.current_state_literals or [])
                for lit in deterministic_literals:
                    if lit.lower() not in existing_lower:
                        merged_literals.append(lit)
                        existing_lower.add(lit.lower())

                result = result.model_copy(update={
                    "current_state_literals": merged_literals[:20]
                })

            # ── Brain Presence Check ──
            brain_exists = False
            try:
                if self.brain_dir and os.path.exists(self.brain_dir) and any(f.endswith(".json") for f in os.listdir(self.brain_dir)):
                    brain_exists = True
            except Exception:
                pass
                
            if not brain_exists:
                missing_brain_note = (
                    "⚠️ MACRO ARCHITECTURE BRAIN MISSING: "
                    "To generate a Brain for this project, run: "
                    "`python src/ticket_to_code/brain/generate_repository_brain.py --workspace <path>`. "
                    "This creates semantic knowledge mapping folders to business features. "
                    "Without it, the workflow falls back to SQLite exact search."
                )
                areas = list(result.investigation_areas or [])
                areas.insert(0, missing_brain_note)
                result = result.model_copy(update={"investigation_areas": areas})

            logger.info(
                f"Investigation complete: Type={result.ticket_type.value}, "
                f"Code needed={result.requires_code_changes}, "
                f"Confidence={result.confidence}"
            )

            # ── Emit StageThought for downstream agents ───────────────────
            raw_thinking = _extract_thinking(response.content)
            thought = StageThought(
                stage="investigation",
                summary=(
                    f"Ticket classified as '{result.ticket_type.value}'. "
                    f"Root-cause hypothesis: {result.root_cause_hypothesis or 'N/A'}. "
                    f"Code changes required: {result.requires_code_changes}. "
                    f"{result.explanation[:300]}"
                ),
                key_decisions=[
                    f"ticket_type={result.ticket_type.value}",
                    f"requires_code_changes={result.requires_code_changes}",
                ] + (result.possible_causes[:3] if result.possible_causes else []),
                signals_noted=(
                    (result.current_state_literals or [])[:4]
                    + (result.affected_systems or [])[:2]
                ),
                confidence=result.confidence,
                raw_thinking=raw_thinking,
            )
            chain = (thinking_chain or ThinkingChain()).append(thought)
            return result, chain

        except Exception as e:
            logger.error(f"Investigation failed: {e}", exc_info=True)
            # Fallback: if technical_facts available, use deterministic type
            if technical_facts and technical_facts.classified_type:
                _FALLBACK_MAP = {
                    "VERSION_BUMP": TicketType.VERSION_BUMP,
                    "BUG": TicketType.BUG_FIX,
                    "UI": TicketType.BUG_FIX,
                    "CONFIG": TicketType.CONFIGURATION,
                }
                fallback_type = _FALLBACK_MAP.get(
                    technical_facts.classified_type, TicketType.INVESTIGATION
                )
            else:
                title_lower = (ticket.title or "").lower()
                desc_lower  = (ticket.description or "").lower()
                combined    = title_lower + " " + desc_lower
                BUG_SIGNALS = {
                    "bug", "fix", "error", "broken", "not working", "doesn't work",
                    "does not work", "incorrect", "wrong", "missing", "failure",
                    "failed", "crash", "exception", "regression", "defect",
                    "issue", "problem", "unexpected", "should be", "should show",
                }
                is_bug = any(sig in combined for sig in BUG_SIGNALS)
                fallback_type = TicketType.BUG_FIX if is_bug else TicketType.INVESTIGATION

            fallback_result = InvestigationResult(
                ticket_type=fallback_type,
                requires_code_changes=True,
                recommended_action="LLM unavailable — deterministic/keyword fallback used",
                explanation=f"Investigation failed ({str(e)}); fallback → {fallback_type.value}",
                confidence=0.5
            )
            chain = (thinking_chain or ThinkingChain()).append(StageThought(
                stage="investigation",
                summary=f"Investigation fallback due to error: {e}. Type: {fallback_type.value}",
                confidence=0.3,
            ))
            return fallback_result, chain

    def _select_deterministic_literals(self, literals: List[str]) -> List[str]:
        """
        Filter deterministic literals to keep high-signal retrieval terms.

        Prioritizes UI/state assertions and ticket-specific labels while removing
        noisy generic action phrases.
        """
        if not literals:
            return []

        generic = {
            "click on",
            "select button",
            "save button",
            "cancel button",
            "close icon",
            "the application",
            "login to the application",
        }

        scored = []
        for lit in literals:
            t = (lit or "").strip()
            if len(t) < 4:
                continue
            low = t.lower()
            if low in generic:
                continue

            score = 0
            if "enabled" in low or "disabled" in low:
                score += 4
            if " tab" in low or " button" in low or "manage all" in low:
                score += 3
            if "register" in low or "deliverable" in low or "content" in low:
                score += 3
            if len(t) >= 10:
                score += 1

            scored.append((score, t))

        scored.sort(key=lambda x: (-x[0], -len(x[1]), x[1].lower()))

        out = []
        seen = set()
        for _, lit in scored:
            k = lit.lower()
            if k not in seen:
                seen.add(k)
                out.append(lit)
        return out[:15]
    
    def _build_investigation_prompt(self) -> str:
        """Build system prompt for investigation"""
        return """You are a senior software engineer performing initial ticket triage.

YOUR ROLE:
Act like an experienced developer who first UNDERSTANDS the problem before jumping to solutions.

YOUR TASK:
Analyze the ticket and determine:
1. What TYPE of issue is this?
2. Does it REQUIRE code changes?
3. What's the likely ROOT CAUSE?  4. What ACTION should be taken?
  5. What is the GOAL of the ticket?
  6. What CONSTRAINTS are mentioned?
  7. Extract any LITERAL VALUES (e.g. strings, numbers, class names) mentioned.
  8. What is the EXPECTED OUTCOME?
  9. Is CLARIFICATION NEEDED?


TICKET TYPES:
- feature_request: New functionality to build, version updates, or text changes
- bug_fix: Code defect that needs fixing
- investigation: Vague issue that needs analysis first
- configuration: Config/environment problem; may or may not require repository edits
- usage_question: User needs explanation (no code change)
- performance: Performance optimization needed
- refactoring: Code improvement without behavior change
- deployment: Deployment/infrastructure issue

CRITICAL RULES:
- DO NOT assume every ticket needs code changes
- BE SKEPTICAL - investigate before acting
- "User cannot do X" might be config, not code bug
- `requires_code_changes` is about whether the repository must be modified, not whether the target is source code.
- If the ticket explicitly asks to add, update, change, modify, remove, create, rename, align, or set content in repository files, then `requires_code_changes` MUST be true.
- The repository includes source files, config files, scripts, Dockerfiles, manifests, SQL, docs, and other tracked workspace files.
- Use `requires_code_changes=false` only for explanation, setup guidance, diagnosis, or operational advice where no repository file needs to change.

FEATURE & UPDATE DETECTION (HIGH PRIORITY):
- If the ticket requests adding a feature, updating a version, or changing text/labels, classify as "feature_request" or "bug_fix".
- Keywords like "update", "add", "implement", "version update", "change to" strongly indicate code changes are required!
- NEVER classify as "investigation" if the ticket clearly states what needs to be changed (e.g., "Update version to 26.3"). "investigation" is ONLY for vague issues where the problem is entirely unknown.
- If explicit file names or file paths are present and the request says to edit them directly, treat that as a file-editing task even when the files are JSON/YAML/.env/docs/scripts.

BUG DETECTION - HIGH PRIORITY:
- If the ticket title or description contains ANY of these signals -> classify as "bug_fix":
   Keywords: bug, fix, error, broken, not working, doesn't work, incorrect, wrong, missing,
   failure, crash, exception, regression, defect, unexpected behavior, should be, should show,
   not displayed, displays incorrectly, visual issue, alignment, overlap
- UI inconsistency tickets (e.g., "dialog should have grey background", "label is wrong") = bug_fix
- Tickets that describe CURRENT vs EXPECTED behavior = bug_fix
- Tickets that say something "should" look or behave differently from what it does = bug_fix

CURRENT vs DESIRED STATE - MANDATORY FOR ALL TICKETS:
- Extract what the code does TODAY (current_state_literals) separately from what it SHOULD do (desired_state_literals).
- current_state_literals are the PRIMARY search terms -> they are what physically EXISTS in the codebase right now.
- desired_state_literals are NEVER search terms -> they don't exist in code yet.
- The rule is universal:
   - "change label from 'Submit' to 'Confirm'"  -> current=["Submit"],  desired=["Confirm"]
   - "background should be blue, not grey"       -> current=["grey"],    desired=["blue"]
   - "shows 0 instead of actual count"           -> current=["0"],       desired=["actual count"]
   - "Add feature X"                             -> current=[],          desired=["X"]
   - "Fix broken login"                          -> current=["error message if mentioned"], desired=[]
- EXTENDED SEARCH (CRITICAL): If the ticket contains a version number, you MUST generate common codebase permutations for it!
   - "version 26.2 needs to be updated to 26.3"  -> current=["26.2", "260200", "26_2"], desired=["26.3", "260300", "26_3"]
✅ ALWAYS search for what EXISTS today, not what the ticket wants it to become.

ANALYSIS APPROACH:
For issues like "User cannot login" or "Feature not working":
1. Consider: Is this a code bug OR config/environment issue?
2. Think: What could cause this? (missing config, wrong setup, actual bug)
3. Decide: Do we need code OR just explanation/config fix?

REPOSITORY EDIT VS ADVICE:
- "Explain how to set TAX_RATE in docker-compose.yml" -> requires_code_changes=false
- "Add TAX_RATE=0.10 to docker-compose.yml" -> requires_code_changes=true
- "Update config/settings.json to include tax_rate" -> requires_code_changes=true
- "Which file controls TAX_RATE?" -> requires_code_changes=false

RESPOND with JSON in this exact structure:
{
  "ticket_type": "<one of: feature_request, bug_fix, investigation, configuration, usage_question, performance, refactoring, deployment>",
  "requires_code_changes": true or false,
  "root_cause_hypothesis": "<brief hypothesis about what's causing the issue>",
  "affected_systems": ["<system1>", "<system2>"],
  "core_subjects": ["<core entity/value being changed, e.g., 'version string', 'deliverable count'>"],
  "context_locations": ["<contextual locations where issue is displayed, e.g., 'dashboard', 'CC4E app'>"],
  "investigation_areas": ["<area1>", "<area2>"],
  "possible_causes": [
    "<possible cause 1>",
    "<possible cause 2>"
  ],
  "current_state_literals": ["<exact string/value that EXISTS in the codebase RIGHT NOW — primary search targets>"],
  "desired_state_literals": ["<exact string/value the ticket wants as the end result — NOT search terms>"],
  "recommended_action": "<what should be done next>",
  "confidence": 0.0 to 1.0,
  "explanation": "<detailed reasoning for your classification>",
  "needs_more_context": true or false
}

Example for a new feature:
{
  "ticket_type": "feature_request",
  "requires_code_changes": true,
  "root_cause_hypothesis": "New functionality requested by business",
  "affected_systems": ["Logging", "Services"],
  "core_subjects": ["version info", "log format"],
  "context_locations": ["logging output console"],
  "investigation_areas": ["Version management", "Log output"],
  "possible_causes": ["Feature does not exist"],
  "current_state_literals": [],
  "desired_state_literals": ["version info in logs"],
  "recommended_action": "Proceed with feature development - add version info to logs",
  "confidence": 0.95,
  "explanation": "Clear feature request with defined requirements and acceptance criteria",
  "needs_more_context": false
}

NO markdown, NO explanations outside JSON, ONLY JSON."""
    
    def _build_user_prompt(
        self,
        ticket: ValueEdgeTicket,
        codebase_summary: Optional[str],
        technical_facts: Optional[TechnicalFacts] = None,
    ) -> str:
        """Build user prompt with ticket details and structured technical facts."""

        # Build the structured facts block — this is injected BEFORE the raw ticket
        # so the LLM cannot miss critical technical signals even if the description
        # is dominated by business/UX language.
        facts_block = ""
        if technical_facts and technical_facts.has_facts():
            import json as _json
            facts_summary = {
                "detected_ticket_type": technical_facts.classified_type,
            }
            if technical_facts.versions:
                facts_summary["VERSION_TRANSITIONS"] = [
                    {
                        "search_for_in_codebase": v.permutations_old,
                        "replace_with": v.permutations_new,
                    }
                    for v in technical_facts.versions
                ]
            if technical_facts.files:
                facts_summary["EXPLICIT_FILES_IN_TICKET"] = [
                    {"file": f.name, "path": f.path, "operation": f.operation}
                    for f in technical_facts.files
                ]
            if technical_facts.classes:
                facts_summary["CLASS_NAMES_FOUND"] = technical_facts.classes[:8]
            if technical_facts.methods:
                facts_summary["METHOD_NAMES_FOUND"] = technical_facts.methods[:8]
            if technical_facts.config_keys:
                facts_summary["CONFIG_KEYS_FOUND"] = technical_facts.config_keys[:5]
            if technical_facts.error_codes:
                facts_summary["ERROR_CODES_FOUND"] = technical_facts.error_codes[:5]
            if technical_facts.section_map.get("technical_context"):
                facts_summary["TECHNICAL_CONTEXT_SECTION"] = technical_facts.section_map["technical_context"]

            facts_block = f"""

⚠️  DETERMINISTICALLY EXTRACTED TECHNICAL FACTS (HIGH PRIORITY — DO NOT IGNORE):
{_json.dumps(facts_summary, indent=2)}

RULE: The above facts were extracted by a deterministic parser. They MUST be reflected
in your current_state_literals. If VERSION_TRANSITIONS are present, the search literals
MUST include all values in 'search_for_in_codebase'. If EXPLICIT_FILES_IN_TICKET are
present, your investigation_areas MUST reference those files.
"""

        context_section = ""
        if codebase_summary:
            context_section = f"""

CODEBASE CONTEXT:
{codebase_summary}
"""

        return f"""
TICKET ID: {ticket.ticket_id}
TITLE: {ticket.title}
PRIORITY: {ticket.priority.value}
{facts_block}
DESCRIPTION:
{ticket.description}

ACCEPTANCE CRITERIA:
{self._format_criteria(ticket.acceptance_criteria)}

LABELS: {', '.join(ticket.labels) if ticket.labels else 'None'}
{context_section}

Now perform investigation and triage. Respond with JSON only.
        """.strip()
    
    def _format_criteria(self, criteria: List[str]) -> str:
        """Format acceptance criteria"""
        if not criteria:
            return "None specified"
        return "\n".join(f"- {c}" for c in criteria)
    
    def _parse_response(self, response_content: str) -> InvestigationResult:
        """Parse LLM response into InvestigationResult"""
        try:
            data = parse_llm_json(response_content, expect_list=False)
            if not isinstance(data, dict):
                data = {}
            if "ticket_type" not in data:
                data["ticket_type"] = "bug_fix"
            if "requires_code_changes" not in data:
                data["requires_code_changes"] = True
            if "recommended_action" not in data:
                data["recommended_action"] = "Investigate further"
        except json.JSONDecodeError as e:
            logger.error("Failed to parse investigation JSON response")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        
        # Validate with Pydantic
        return InvestigationResult.model_validate(data)

    # =========================================================================
    # HYPOTHESIS GENERATION — feeds the Evidence Collection Loop
    # =========================================================================

    def generate_hypotheses(
        self,
        ticket: ValueEdgeTicket,
        investigation: InvestigationResult,
        localized_tasks: Optional[List[DevelopmentTask]] = None,
    ) -> List[InvestigationHypothesis]:
        """
        Generate investigation hypotheses from the ticket and initial triage.

        Each hypothesis comes with concrete search artefacts the Evidence
        Collection Loop can fire immediately:
          • queries   – natural-language phrases for semantic / RAG search
          • literals  – exact string constants / values to FTS-match in code
          • symbols   – class / method / config-key names to look up in SQLite

        Args:
            ticket:           The originating ticket.
            investigation:    The InvestigationResult from Phase 0 triage.
            localized_tasks:  Optional list of already-localized DevelopmentTasks
                              so the LLM can ground hypotheses in real file names.

        Returns:
            List[InvestigationHypothesis] (2–5 items), fallback to 1 structural
            hypothesis if the LLM call fails.
        """
        print("\n" + "=" * 80)
        print("[ENTER] InvestigationAgent.generate_hypotheses()")
        print(f"   Ticket: {ticket.ticket_id}  Type: {investigation.ticket_type.value}")
        print("=" * 80)
        logger.info("🔬 Generating investigation hypotheses")

        try:
            system_prompt = self._hypothesis_system_prompt()
            user_prompt = self._hypothesis_user_prompt(ticket, investigation, localized_tasks)

            response = llm_invoke(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            hypotheses = self._parse_hypotheses(response.content)

            # Enforce that every hypothesis carries the triaged current-state
            # literals so repository search always includes known-bad behavior
            # strings (deterministic guardrail against LLM omission).
            required_literals = [
                l for l in (investigation.current_state_literals or []) if len((l or "").strip()) >= 4
            ][:12]
            if required_literals:
                for h in hypotheses:
                    existing = {lit.lower() for lit in (h.literals or [])}
                    merged = list(h.literals or [])
                    for lit in required_literals:
                        if lit.lower() not in existing:
                            merged.append(lit)
                            existing.add(lit.lower())
                    h.literals = merged[:20]

            logger.info(f"  Generated {len(hypotheses)} hypothesis(es)")
            return hypotheses

        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}", exc_info=True)
            # Deterministic fallback — build multiple hypotheses so the evidence
            # loop still gets enough search diversity when LLM JSON is malformed.
            fallback_literals = [
                str(l).strip()
                for l in (investigation.current_state_literals or [])
                if str(l).strip()
            ]
            desired_literals = [
                str(l).strip()
                for l in (investigation.desired_state_literals or [])
                if str(l).strip()
            ]
            affected = [
                str(s).strip()
                for s in (investigation.affected_systems or [])
                if str(s).strip()
            ]
            areas = [
                str(a).strip()
                for a in (investigation.investigation_areas or [])
                if str(a).strip()
            ]

            hypotheses: List[InvestigationHypothesis] = []

            hypotheses.append(InvestigationHypothesis(
                id="H1",
                hypothesis=(
                    investigation.root_cause_hypothesis
                    or f"Ticket '{ticket.title}' requires owner-file updates"
                ),
                queries=[
                    ticket.title,
                    f"{ticket.ticket_id} owner implementation",
                ],
                literals=fallback_literals[:10],
                symbols=affected[:5],
                core_subjects=[],
                context_locations=[],
                anchors=[],
                confidence=0.45,
            ))

            if areas:
                hypotheses.append(InvestigationHypothesis(
                    id="H2",
                    hypothesis="Investigation areas contain direct implementation ownership",
                    queries=areas[:3] + [f"{ticket.title} implementation"],
                    literals=(fallback_literals[:6] + desired_literals[:4])[:12],
                    symbols=(affected[:3] + areas[:3])[:6],
                    core_subjects=[],
                    context_locations=[],
                    anchors=[],
                    confidence=0.4,
                ))

            if desired_literals:
                hypotheses.append(InvestigationHypothesis(
                    id=f"H{len(hypotheses) + 1}",
                    hypothesis="Desired-state literals map to validation and consumer paths",
                    queries=[
                        f"{ticket.title} validation",
                        f"{ticket.ticket_id} expected behavior",
                    ],
                    literals=desired_literals[:8],
                    symbols=affected[:4],
                    core_subjects=[],
                    context_locations=[],
                    anchors=[],
                    confidence=0.35,
                ))

            if not hypotheses:
                hypotheses.append(InvestigationHypothesis(
                    id="H1",
                    hypothesis=f"Ticket '{ticket.title}' requires deterministic repository search",
                    queries=[ticket.title],
                    literals=[],
                    symbols=[],
                    core_subjects=[],
                    context_locations=[],
                    anchors=[],
                    confidence=0.3,
                ))

            logger.info(f"  Using deterministic hypothesis fallback with {len(hypotheses)} hypothesis(es)")
            return hypotheses

    def _hypothesis_system_prompt(self) -> str:
        return """You are a senior engineer performing deep-investigation planning.

ROLE:
Given a ticket and initial triage, generate 2-5 GROUNDED investigation hypotheses.
Each hypothesis MUST include concrete search artefacts a code-search tool can use.

RULES:
- Each hypothesis must be falsifiable and specific
- queries   → natural language phrases for semantic search (1–3 per hypothesis)
- literals  → exact string constants, error messages, config keys. Prefer literals that uniquely identify behavior. If a literal is expected to match more than 20 files, drop it and generate a more specific alternative. This must be AI reasoning, not hardcoded stopwords. Extract the longest, most unique distinguishing phrase possible. Ask yourself: 'If I search the entire repository for this literal, will it return exactly 1-3 files?' If it will return dozens of files, the literal is too generic and you MUST NOT include it. (0–3 per hypothesis)
- EXTENDED SEARCH (CRITICAL): If the ticket mentions a version number (e.g., "26.2"), you MUST generate common software variations of that version as literals! For example, for "26.2", generate ["26.2", "260200", "26_2"].
- symbols   → class names, method names, component names, config properties (1–5 per hypothesis)
- anchors   → Domain/page/business nouns from the ticket (e.g. "Reviewer", "Deliverable"). Used to boost files related to specific pages.
- Prefer concrete symbol names over vague descriptions
- Use camelCase / PascalCase for Java/TypeScript symbols
- Use kebab-case for Angular component selectors
- Do NOT include file extensions in symbols

RESPOND with a JSON array only (no markdown):
[
  {
    "id": "H1",
    "hypothesis": "plain-language hypothesis",
    "queries": ["semantic query 1", "semantic query 2"],
    "literals": ["Unique Exact String", "config.key.name"],
    "symbols": ["ClassName", "methodName", "ComponentName"],
    "anchors": ["DomainNoun", "PageName"],
    "confidence": 0.0-1.0
  },
  ...
]"""

    def _hypothesis_user_prompt(
        self,
        ticket: ValueEdgeTicket,
        investigation: InvestigationResult,
        localized_tasks: Optional[List[DevelopmentTask]],
    ) -> str:
        task_context = ""
        if localized_tasks:
            paths = [t.file_path for t in localized_tasks if t.task_type.value != "read_only"]
            task_context = f"\nALREADY LOCALIZED FILES:\n" + "\n".join(f"  - {p}" for p in paths)

        # Inject the Macro Architecture Brain (top relevant folders only, capped to ~1.5K tokens)
        macro_brain_context = ""
        try:
            import os
            import glob
            import json
            import re
            all_macro_brains = []
            if self.brain_dir:
                for brain_file in glob.glob(os.path.join(self.brain_dir, "*_directory_brain.json")):
                    with open(brain_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_macro_brains.extend(data)
            
            if all_macro_brains:
                search_terms = set(re.findall(r'[A-Za-z0-9_]{3,}', f"{ticket.title} {ticket.description}".lower()))
                for s in (investigation.affected_systems or []) + (investigation.investigation_areas or []):
                    search_terms.update(re.findall(r'[A-Za-z0-9_]{3,}', str(s).lower()))
                
                scored_dirs = []
                for entry in all_macro_brains:
                    d_path = str(entry.get("directory_path", "")).lower()
                    d_domain = str(entry.get("business_domain", "")).lower()
                    d_role = str(entry.get("technical_role", "")).lower()
                    d_resp = str(entry.get("core_responsibilities", "")).lower()
                    d_terms = [str(t).lower() for t in entry.get("intent_terms", [])]
                    
                    score = 0
                    for term in search_terms:
                        if term in d_path:
                            score += 5
                        if term in d_domain:
                            score += 4
                        if any(term in dt for dt in d_terms):
                            score += 3
                        if term in d_role:
                            score += 2
                        if term in d_resp:
                            score += 1
                    
                    scored_dirs.append((score, entry))
                
                scored_dirs.sort(key=lambda x: x[0], reverse=True)
                top_entries = [e for s, e in scored_dirs if s > 0][:15]
                if not top_entries:
                    top_entries = [e for _, e in scored_dirs[:10]]
                
                compact_brains = [
                    {
                        "dir": e.get("directory_path"),
                        "domain": e.get("business_domain"),
                        "role": e.get("technical_role"),
                        "terms": e.get("intent_terms", [])[:5],
                    }
                    for e in top_entries
                ]
                macro_brain_context = f"\nMACRO ARCHITECTURE BRAIN (Top Relevant Folders):\n{json.dumps(compact_brains, indent=1)}\n"
        except Exception as e:
            logger.debug(f"Macro brain context formatting skipped: {e}")

        # Inject current_state_literals from the triage result into the prompt
        current_state_hint = ""
        if investigation.current_state_literals:
            current_state_hint = (
                f"\nCURRENT STATE LITERALS (extracted from triage — use as primary literals in hypotheses):\n"
                + "\n".join(f"  - \"{lit}\"" for lit in investigation.current_state_literals)
                + "\n"
            )

        return f"""TICKET ID: {ticket.ticket_id}
TITLE: {ticket.title}
DESCRIPTION:
{ticket.description}

TRIAGE RESULT:
  type: {investigation.ticket_type.value}
  root_cause_hypothesis: {investigation.root_cause_hypothesis or 'N/A'}
  affected_systems: {investigation.affected_systems}
  investigation_areas: {investigation.investigation_areas}
  possible_causes: {investigation.possible_causes}
  current_state_literals: {investigation.current_state_literals}
  desired_state_literals: {investigation.desired_state_literals}
{current_state_hint}
{task_context}
{macro_brain_context}

Generate 2-5 investigation hypotheses with search artefacts. Include the 'macro_directories' field mapping to the most relevant folders from the MACRO ARCHITECTURE BRAIN. Ensure the 'literals' field in each hypothesis includes all current_state_literals from the triage result above. JSON array only."""

    def _parse_hypotheses(self, content: str) -> List[InvestigationHypothesis]:
        """Parse LLM response into InvestigationHypothesis list."""
        data = parse_llm_json(content, expect_list=True)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of hypotheses")

        result = []
        for item in data:
            h = InvestigationHypothesis(
                id=str(item.get("id", f"H{len(result)+1}")),
                hypothesis=str(item.get("hypothesis", "")),
                queries=[str(q) for q in item.get("queries", [])],
                literals=[str(l) for l in item.get("literals", [])],
                symbols=[str(s) for s in item.get("symbols", [])],
                core_subjects=[str(s) for s in item.get("core_subjects", [])],
                context_locations=[str(s) for s in item.get("context_locations", [])],
                macro_directories=[str(m) for m in item.get("macro_directories", [])],
                anchors=[str(a) for a in item.get("anchors", [])],
                confidence=float(item.get("confidence", 0.5)),
            )
            result.append(h)

        return result if result else []

