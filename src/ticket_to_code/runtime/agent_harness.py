"""
Universal Agentic Operating System (Agent Harness) for Ticket-to-Code and IDE Chatbot.

Bridges the LLM with the local operating system, IDE state, and tool environment.
Provides:
  1. 6-Layer Dynamic Prompt Assembly (Identity, Environment, Host-State, Tools, Invariants, KIs)
  2. Instruction-Grounded Intent Decomposition (Explanation vs Autonomous Modification vs Planning)
  3. Dynamic Tool Partitioning (Inhibits mutation tools in explanation mode)
  4. Multi-Turn Autonomous ReAct / AFC Execution Loop
  5. Context Compaction & Resumption Engine (# Resuming from a compaction)
  6. Phase 3 Verification & Self-Correcting Compiler Feedback Loop
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import logging
import subprocess
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from ticket_to_code.editing.surgical_diff_engine import apply_tolerant_replacement
from ticket_to_code.runtime.security import SecurityPolicy, SecurityViolation, run_checked_subprocess
from ticket_to_code.runtime.limits import TOOL_OUTPUT_MAX_CHARS, BUILD_OUTPUT_MAX_CHARS, truncate


# ─── 1. Intent Decomposition & Operational Modes ─────────────────────────────

class AgentIntent(str, Enum):
    EXPLANATION_REVIEW = "explanation_review"
    AUTONOMOUS_MODIFICATION = "autonomous_modification"
    PLANNING_ARCHITECTURE = "planning_architecture"


def _contains_any(text: str, phrases: List[str]) -> bool:
    """Word-boundary cue match so short verbs don't false-fire inside other words
    (e.g. 'edit' must not match inside 'credit')."""
    for p in phrases:
        if re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text):
            return True
    return False


# Interrogative openers that mark a message as an informational question (read-only).
_INTERROGATIVE_PREFIXES = (
    "what ", "where ", "how ", "why ", "which ", "who ", "when ", "whom ", "whose ",
    "is ", "are ", "does ", "do ", "did ", "can you explain", "could you explain",
)

_EXPLAIN_CUES = [
    "what does this button do", "what about this response", "critique of this",
    "where is", "how does", "explain this", "explain the",
    "don't do any code changes", "do not modify", "just analyze", "just explain",
    "provide a comprehensive architectural explanation", "what is the purpose",
    "describe the flow", "walk me through", "can you explain", "could you explain",
]

# Phrases that turn an otherwise-informational question into an implementation request.
_APPLY_CUES = [
    "and apply", "and implement", "and fix", "apply the fix", "fix it",
    "implement it", "solve it", "make the changes", "make it work", "patch it",
]

_MOD_CUES = [
    "fix", "implement", "modify", "patch", "apply changes", "refactor",
    "update the file", "edit", "write code", "create the", "delete the",
    "solve ticket", "solve this ticket", "run ticket to code", "resolve the",
]

# Declarative defect / problem-statement signals. A defect title rarely contains an
# imperative verb ("fix"); it describes broken behaviour, so these mark it as solvable.
_DEFECT_SIGNALS = [
    "nullpointerexception", "null pointer", "exception", "stack trace", "stacktrace",
    "error", "crash", "crashes", "crashing", "fails", "failing", "failed", "broken",
    "bug", "defect", "regression", "regressed", "not revoked", "not revoke",
    "not working", "doesn't work", "does not work", "doesn't", "does not",
    "not updating", "not saving", "not persisted", "not reflected", "not applied",
    "not triggered", "not called", "not enforced", "not cascaded", "not propagated",
    "not removed", "not deleted", "incorrect", "wrong", "unexpected", "missing",
    "unable to", "cannot", "can't", "returns null", "returns empty",
    "internal server error", "leak", "race condition", "deadlock", "throws",
    "no longer", "stopped working", "does nothing", "nothing happens", "no effect",
]

# State-inconsistency / leakage signals: real defects often describe UNWANTED PERSISTENCE
# with affirmative verbs ("still has access", "continues to retain") and carry no negation.
_INCONSISTENCY_SIGNALS = [
    "still has", "still have", "still exists", "still active", "still visible",
    "still shows", "still able", "continues to", "continue to retain", "retains",
    "retained", "remain assigned", "persists", "not cleared", "not cleaned",
    "orphaned", "dangling", "out of sync", "infinite loop", "hangs", "freezes",
    "unhandled", "bypassed", "security hole", "vulnerability", "memory leak",
]

# Typo-tolerant / colloquial defect phrasing (regexes, matched separately from _contains_any).
_COLLOQUIAL_DEFECT_REGEXES = [
    r"\bdoes\s*n'?t\b", r"\bdo\s*n'?t\b", r"\bdid\s*n'?t\b", r"\bwo\s*n'?t\b",
    r"\bdosent\b", r"\bdoesnt\b", r"\bdont\b", r"\bdidnt\b", r"\bwont\b",
    r"\bborked\b", r"\bglitch(?:ed|y)?\b", r"\bhosed\b",
]


def _matches_any_regex(text: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False


def _has_defect_signal(msg_lower: str) -> bool:
    return (
        _contains_any(msg_lower, _DEFECT_SIGNALS)
        or _contains_any(msg_lower, _INCONSISTENCY_SIGNALS)
        or _matches_any_regex(msg_lower, _COLLOQUIAL_DEFECT_REGEXES)
    )


def _classify_deterministic(message: str) -> Tuple[AgentIntent, str]:
    """Deterministic fast-path classifier. Returns (intent, confidence) where confidence is
    'high' for a matched rule and 'low' only for the ambiguous fall-through default — the one
    case the optional semantic arbiter should resolve."""
    msg_clean = (message or "").strip()
    msg_lower = msg_clean.lower()
    if not msg_lower:
        return AgentIntent.EXPLANATION_REVIEW, "high"

    # 0. Check for negated explanation cues ("don't just explain", "stop explaining", "instead of explaining")
    _NEGATED_EXPLAIN = [
        r"don'?t\s+(?:just\s+)?explain",
        r"not\s+(?:just\s+)?explain",
        r"instead\s+of\s+(?:just\s+)?explaining",
        r"stop\s+explaining",
        r"beyond\s+(?:just\s+)?explanation",
        r"no\s+need\s+to\s+explain",
    ]
    is_negated_explain = _matches_any_regex(msg_lower, _NEGATED_EXPLAIN)

    # 0a. Explicit slash command overrides (0ms, 100% deterministic)
    if msg_lower.startswith(("/explain", "/ask", "/query", "/read", "/audit", "/inspect", "/walkthrough")):
        return AgentIntent.EXPLANATION_REVIEW, "high"
    if msg_lower.startswith(("/fix", "/patch", "/flow", "/solve", "/mutate", "/apply")):
        return AgentIntent.AUTONOMOUS_MODIFICATION, "high"

    # 1. Explicit override tags win outright (support brackets or missing brackets, and direct inhibition phrases)
    _EXPLICIT_EXPLAIN_OVERRIDES = [
        "[explain only]", "[read only]", "explain only]", "read only]",
        "[explain-only]", "[read-only]", "explain only",
        "[explain]", "[read]", "[audit]", "[query]",
        "--mode=explain", "--mode=read", "mode: explain", "mode: read",
        "do not trigger", "don't trigger", "no code modification",
        "do not modify the code", "don't modify the code",
        "just analyze the codebase", "architectural explanation in chat",
        "explanation in chat", "solution in chat"
    ]
    if not is_negated_explain and any(cue in msg_lower for cue in _EXPLICIT_EXPLAIN_OVERRIDES):
        return AgentIntent.EXPLANATION_REVIEW, "high"

    # 2. Planning tags / slash commands.
    if msg_lower.startswith(("/plan", "/grill-me", "/design", "/architect")) or "architectural plan" in msg_lower:
        return AgentIntent.PLANNING_ARCHITECTURE, "high"

    # 3. An explicit request to apply a change always wins (even if phrased as a question).
    if any(cue in msg_lower for cue in _APPLY_CUES):
        return AgentIntent.AUTONOMOUS_MODIFICATION, "high"

    # 3b. Negated explanation followed by defect or modification cue wins as modification
    if is_negated_explain and (_contains_any(msg_lower, _MOD_CUES) or _has_defect_signal(msg_lower)):
        return AgentIntent.AUTONOMOUS_MODIFICATION, "high"

    is_question = msg_clean.endswith("?") or msg_lower.startswith(_INTERROGATIVE_PREFIXES)
    _INQUIRY_CUES = (
        "explain", "root cause", "walk me through", "walkthrough", "analyze", "analysis",
        "investigate", "diagnose", "diagnosis", "describe how", "understand why", "understand how",
        "clarify", "review", "audit", "why did", "what caused", "how come this broke",
        "find out why", "check why", "tell me why", "what is the reason", "what happened",
        "how to fix", "how can i", "how do i", "how can we", "how do we", "how should we",
        "how would we", "how to resolve", "how to refactor", "how to patch", "how to handle",
        "how to implement", "can you advise", "can you suggest", "what is the recommended",
        "what is the best practice", "what would happen if", "help me understand",
        "give me a breakdown", "guide me through", "just analyze"
    )
    is_explain = any(cue in msg_lower for cue in _EXPLAIN_CUES) or any(cue in msg_lower for cue in _INQUIRY_CUES)
    has_defect = _has_defect_signal(msg_lower)

    # 4. If the message explicitly asks to explain, analyze root cause, walk through, or clarify,
    #    keep it read-only regardless of defect mentions (e.g. "Explain how to fix this", "Why did the fix fail? Explain the root cause").
    if is_explain:
        return AgentIntent.EXPLANATION_REVIEW, "high"

    # 5. A question that describes a concrete defect without an explanation request is a troubleshooting ticket
    #    (e.g. "Why is contract access still active for removed coordinators?", "How come the button doesn't save?").
    if is_question and has_defect:
        return AgentIntent.AUTONOMOUS_MODIFICATION, "high"

    # 6. Informational questions without defect descriptions stay read-only ("What does this button do?", "Where is X defined?").
    if is_question:
        return AgentIntent.EXPLANATION_REVIEW, "high"

    # 7. Explicit modification verbs directed at the assistant ("fix the...", "implement the...", "refactor...").
    if _contains_any(msg_lower, _MOD_CUES):
        return AgentIntent.AUTONOMOUS_MODIFICATION, "high"

    # 8. Declarative defect / problem / inconsistency statements (e.g. Jira ticket titles pasted into chat).
    if has_defect:
        return AgentIntent.AUTONOMOUS_MODIFICATION, "high"

    # 9. Ambiguous: no cue matched. Safe default, flagged low-confidence for the arbiter.
    return AgentIntent.EXPLANATION_REVIEW, "low"


def classify_agent_intent(message: str) -> AgentIntent:
    """Deterministic intent classifier (0ms, 0 tokens). See `_classify_deterministic`.

    1. EXPLANATION_REVIEW: informational questions, review requests, or [EXPLAIN ONLY] / [READ ONLY].
    2. PLANNING_ARCHITECTURE: /plan, /grill-me, or an explicit architectural-plan request.
    3. AUTONOMOUS_MODIFICATION: explicit change verbs, declarative defect/inconsistency statements,
       or a question that describes a concrete defect.
    """
    return _classify_deterministic(message)[0]


def classify_agent_intent_hybrid(
    message: str,
    client: Optional[Any] = None,
    model_id: Optional[str] = None,
) -> AgentIntent:
    """2-tier router: deterministic fast-path first; a micro-LLM arbiter resolves ONLY the
    ambiguous fall-through case. 90%+ of messages never touch the LLM (0 tokens).

    Poly-Model & Provider Agnostic:
    - Accepts any configured client (Google GenAI, OpenAI, Anthropic, LangChain, or custom).
    - Uses the provided `model_id` or provider-appropriate fast model without locking to a fixed one.
    - If no client is available, or if the model call times out or errors, gracefully falls back
      to the deterministic safe default.
    """
    intent, confidence = _classify_deterministic(message)
    if confidence == "high" or client is None:
        return intent

    probe_prompt = (
        "Classify the developer's intent into exactly one mode.\n"
        "- AUTONOMOUS_MODIFICATION: a bug, defect, broken/unwanted behaviour, code change, or ticket to implement.\n"
        "- EXPLANATION_REVIEW: a conceptual question, code walkthrough, or read-only inquiry.\n"
        "- PLANNING_ARCHITECTURE: high-level design or migration planning without edits.\n"
        f"Developer message: {message[:300]}"
    )

    # Resolve target model for the probe
    active_model = model_id or os.environ.get("AVIATOR_ARBITER_MODEL", "gemini-2.5-flash")

    try:
        raw_text = ""

        # 1. Google GenAI Client (Vertex AI or AI Studio)
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            from google.genai import types
            resp = client.models.generate_content(
                model=active_model,
                contents=probe_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {"mode": {"type": "STRING"}},
                        "required": ["mode"],
                    },
                    temperature=0.0,
                    max_output_tokens=25,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw_text = getattr(resp, "text", "") or ""

        # 2. OpenAI / LiteLLM / Azure OpenAI Client
        elif hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = client.chat.completions.create(
                model=active_model,
                messages=[{"role": "user", "content": probe_prompt}],
                max_tokens=25,
                temperature=0.0,
            )
            raw_text = resp.choices[0].message.content or ""

        # 3. Anthropic Client
        elif hasattr(client, "messages") and hasattr(client.messages, "create"):
            resp = client.messages.create(
                model=active_model,
                max_tokens=25,
                temperature=0.0,
                messages=[{"role": "user", "content": probe_prompt}],
            )
            if resp.content and hasattr(resp.content[0], "text"):
                raw_text = resp.content[0].text

        # 4. LangChain / Callable fallback
        elif hasattr(client, "invoke"):
            resp = client.invoke(probe_prompt)
            raw_text = getattr(resp, "content", str(resp))
        elif callable(client):
            raw_text = str(client(probe_prompt))

        # Parse extracted mode
        if raw_text:
            text_lower = raw_text.lower()
            if "modification" in text_lower:
                return AgentIntent.AUTONOMOUS_MODIFICATION
            if "planning" in text_lower:
                return AgentIntent.PLANNING_ARCHITECTURE
            if "explanation" in text_lower or "review" in text_lower:
                return AgentIntent.EXPLANATION_REVIEW

    except Exception as exc:  # noqa: BLE001
        logger.debug("Semantic intent arbiter fell back to deterministic default: %s", exc)

    return intent



# ─── 2. Host-State Metadata ──────────────────────────────────────────────────

@dataclass
class HostStateMetadata:
    """Dynamic IDE metadata passed into the harness on every invocation."""
    os_name: str = "windows"
    shell: str = "powershell"
    workspace_path: str = ""
    active_document: str = ""
    cursor_line: int = 1
    open_documents: List[str] = field(default_factory=list)
    local_time_iso: str = ""
    conversation_id: str = ""
    app_data_dir: str = ""
    build_tool: str = ""
    build_command: str = ""
    test_command: str = ""
    detected_marker: str = ""

    @classmethod
    def capture_current(
        cls,
        workspace_path: str,
        active_document: str = "",
        cursor_line: int = 1,
        open_documents: Optional[List[str]] = None,
        conversation_id: str = "",
    ) -> HostStateMetadata:
        is_win = sys.platform == "win32"
        build_tool = ""
        build_command = ""
        test_command = ""
        detected_marker = ""
        try:
            from ticket_to_code.runtime.polyglot_build import PolyglotBuildDiscoverer
            ws_p = Path(workspace_path).resolve()
            target_p = ws_p
            if active_document:
                doc_p = Path(active_document)
                if doc_p.is_file():
                    target_p = doc_p.parent
            b_cfg = PolyglotBuildDiscoverer.discover(target_p, ws_p)
            if b_cfg:
                build_tool = b_cfg.build_tool
                build_command = b_cfg.compile_cmd
                test_command = b_cfg.test_cmd
                detected_marker = b_cfg.detected_marker
        except Exception:
            pass

        return cls(
            os_name="windows" if is_win else "linux",
            shell="powershell" if is_win else "bash",
            workspace_path=str(Path(workspace_path).resolve()),
            active_document=active_document,
            cursor_line=cursor_line,
            open_documents=open_documents or [],
            local_time_iso=time.strftime("%Y-%m-%d"),
            conversation_id=conversation_id,
            app_data_dir=os.environ.get("APPDATA", "") or str(Path.home() / ".aviator"),
            build_tool=build_tool,
            build_command=build_command,
            test_command=test_command,
            detected_marker=detected_marker,
        )

    def render_static_user_info(self) -> str:
        """Renders cache-stable <user_information> XML block."""
        ws_display = self.workspace_path.replace("\\", "/")
        build_env_block = ""
        if self.build_tool:
            build_env_block = (
                "\n<build_and_runtime_environment>\n"
                f"Detected Build Tool: {self.build_tool} (wrapper/marker: '{self.detected_marker}')\n"
                f"Compile Command: {self.build_command}\n"
                f"Test Command: {self.test_command}\n"
                f"CRITICAL CONSTRAINT: You MUST use '{self.build_tool}' for all build/test commands. "
                f"Do NOT hallucinate or suggest incompatible tools (e.g. do NOT use 'mvn' or Maven if Gradle is detected, and vice-versa).\n"
                "</build_and_runtime_environment>\n"
            )

        return (
            "<user_information>\n"
            f"The USER's OS version is {self.os_name}. Shell: {self.shell}.\n"
            f"The user has 1 active workspaces:\n"
            f"{self.workspace_path} -> {ws_display}\n"
            "Code relating to the user's requests should be written in the locations listed above. "
            "Avoid writing project code files to tmp or outside the workspace.\n"
            f"App Data Directory: {self.app_data_dir}\n"
            f"Conversation ID: {self.conversation_id or 'default-session'}\n"
            f"{build_env_block}"
            "</user_information>"
        )

    def render_dynamic_metadata(self) -> str:
        """Renders volatile <ADDITIONAL_METADATA> XML block strictly positioned at prompt tail."""
        open_docs_block = ""
        if self.open_documents:
            open_docs_block = "Other open documents:\n" + "\n".join(f"- {d}" for d in self.open_documents)

        active_lang = "PYTHON"
        if self.active_document.endswith((".java", ".class")):
            active_lang = "JAVA"
        elif self.active_document.endswith((".ts", ".tsx")):
            active_lang = "TYPESCRIPT"
        elif self.active_document.endswith((".js", ".jsx")):
            active_lang = "JAVASCRIPT"

        return (
            "<ADDITIONAL_METADATA>\n"
            f"The current local date is: {self.local_time_iso}.\n\n"
            "The user's current state is as follows:\n"
            f"Active Document: {self.active_document or 'None'} (LANGUAGE_{active_lang})\n"
            f"Cursor is on line: {self.cursor_line}\n"
            f"{open_docs_block}\n"
            "</ADDITIONAL_METADATA>"
        )

    def render_xml(self) -> str:
        """Renders combined XML blocks (backward-compatible)."""
        return f"{self.render_static_user_info()}\n\n{self.render_dynamic_metadata()}"


# ─── 3. Universal Software Engineering Invariants ────────────────────────────

UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS = (
    "### UNIVERSAL SOFTWARE ENGINEERING INVARIANTS (MANDATORY ACROSS ALL CODE & ARCHITECTURE):\n\n"
    "1. REPOSITORY GROUND TRUTH & LIVE INSPECTION (ZERO SPECULATION):\n"
    "   - All class names, interfaces, methods, database tables, queue topics, endpoints, and file paths MUST be verified from active workspace files.\n"
    "   - NEVER hypothesize, guess, or invent names. Use live inspection tools (search_codebase, find_files, inspect_file, trace_call_chain, find_references) to discover exact identifiers.\n"
    "   - Outlines only show signatures without method bodies. To implement, modify, or reference any method, you MUST inspect its actual body using inspect_file(path, symbol='methodName') so you read real constructors, injected dependencies, and exception signatures directly from code.\n\n"
    "2. CONTRACT & SIGNATURE INTEGRITY (INTERFACE VS IMPLEMENTATION & ZERO @Override HALLUCINATION):\n"
    "   - When referencing or implementing methods declared on interfaces, base classes, or public APIs, you MUST preserve exact method signatures: parameter names, parameter types, parameter order, and return types.\n"
    "   - Never mutate existing signatures unless explicitly mandated by the ticket.\n"
    "   - CRITICAL: NEVER declare '@Override' on a method in an implementation class unless that EXACT method signature is declared on the corresponding interface or superclass. Inspect the interface to verify whether the method is part of the public contract or merely an internal private/protected helper method.\n"
    "   - Never rename existing public constants or fields that other components or tests depend upon.\n\n"
    "3. SCOPE & PARAMETER UTILIZATION FIRST (ZERO REDUNDANT PROVIDER CALLS):\n"
    "   - When analyzing or implementing a method, ALWAYS inspect its own parameter list first!\n"
    "   - If a method already receives a resolved context object passed as a parameter, use its fields directly.\n"
    "   - Never invent calls on external auth/context providers when the caller already passed the resolved context in parameters.\n\n"
    "4. CALLER CROSS-REFERENCING & CONSTANT PRESERVATION (find_references):\n"
    "   - Before modifying, renaming, or refactoring any shared constant or public method, ALWAYS run find_references(symbol) across the entire workspace.\n"
    "   - Never rename or delete shared constants when dependent services or callers import them directly.\n\n"
    "5. ENUM & DOMAIN STATUS GROUNDING (TICKET REQUIREMENTS VS REPOSITORY CONSTANTS):\n"
    "   - When tickets or user requirements mention business terms, statuses, or phases (e.g. 'under review', 'resolved', 'approved', 'pending', 'in creation'), NEVER assume or invent enum constants matching those English words.\n"
    "   - You MUST inspect the Enum definition files in the codebase to discover the exact existing constants. Map the ticket's business requirements strictly to the codebase's real domain states.\n"
    "   - Respect the exact casing of enum definitions and map keys. Never invent non-existent enum constants unless the prompt explicitly instructs you to edit the enum file itself to declare new constants.\n\n"
    "6. EXCEPTION CONSTRUCTOR & ARITY FIDELITY (ZERO CONSTRUCTOR HALLUCINATION):\n"
    "   - Before instantiating or throwing domain or platform exceptions, verify constructor arity and argument types against their class declarations or existing usages in the workspace.\n"
    "   - If an exception only defines a single-argument constructor new SomeException('ERR_CODE'), NEVER call it with 2 arguments new SomeException('ERR_CODE', 'message').\n"
    "   - If an authorization exception defines a no-arg constructor, never pass error messages to it.\n"
    "   - CRITICAL EXTERNAL DEPENDENCY RULE: If an exception class originates from an external package with no source file in the repository, you MUST search active workspace call sites using `search_codebase` or `inspect_file` to observe the EXACT constructor invocation used in the repo. If existing call sites use a single-argument constructor, you are STRICTLY FORBIDDEN from adding a second argument.\n"
    "   - Check existing usages across the codebase via search_codebase or inspect the exception class before writing code.\n\n"
    "7. DTO & REQUEST MODEL FIDELITY (EXACT FIELDS & NO HALLUCINATED GETTERS):\n"
    "   - When extracting data from DTOs, request payloads, or command objects, inspect the DTO class definition to verify its actual getters and fields.\n"
    "   - NEVER hallucinate methods or fields on DTOs; do not call a getter that is not declared on the DTO.\n"
    "   - If the DTO supplies raw content (such as a string comment) rather than an entity ID, trace how existing code handles it (e.g., invoking a service to create the entity and obtaining its generated identifier).\n\n"
    "8. DATABASE & PERSISTENCE GROUNDING (EXACT SCHEMA FIDELITY & MIGRATION SEQUENCING):\n"
    "   - When working with database entities, models, DTOs, or repositories, ground your implementation strictly in the actual database schema definitions (inspect migration files, SQL DDL, or existing entity definitions in the workspace).\n"
    "   - NEVER hallucinate non-existent database columns or entity fields.\n"
    "   - Strictly honor schema nullability constraints, foreign keys, and exact types (e.g. UUID vs String vs Long, primitive vs wrapper).\n"
    "   - UNIVERSAL MIGRATION SEQUENCING: When adding or modifying database migrations in ANY framework (Flyway, Alembic, Django, EF Core, Prisma, Liquibase, Go Migrate), NEVER invent version prefixes or filenames blind. Always inspect the migration directory first, identify the repository's active versioning convention (semver prefix, timestamp prefix, or sequential integer), and select the next strictly monotonic, non-colliding version identifier.\n\n"
    "9. DEFENSIVE STREAM & CONTEXT LOOKUP NULL-GUARDS (ZERO UNCHECKED NULL-RPC):\n"
    "   - When extracting context identifiers (e.g. projectId, contractId, parentId) using Stream filters (e.g. contexts.stream().filter(...).map(...).findFirst().orElse(null)), remember that entities frequently belong to other context types and the resulting ID CAN BE NULL!\n"
    "   - NEVER pass an ID extracted from a nullable stream or optional directly into a downstream remote client, GraphQL query, REST service, or helper method without an explicit null-guard: `if (contextId != null && downstreamCheck(contextId))`.\n"
    "   - Passing null into downstream GraphQL or REST service filters causes immediate 400 Bad Request or unhandled NullPointerException in production.\n\n"
    "10. SURGICAL REFACTORING & OPERATIONAL GUARD PRESERVATION (ZERO DELETED INVARIANTS):\n"
    "   - When refactoring existing methods, NEVER delete or overwrite existing operational mode checks, state flags, or business guards unless the prompt or ticket explicitly and specifically orders their deletion.\n"
    "   - Refactoring must be additive and surgical: preserve all existing defensive preconditions at the method head, and only update the specific authorization or state check targeted by the ticket.\n\n"
    "11. STRICT 0-INTERRUPTION ESCALATION BOUNDARY (ZERO CHEAP CLARIFICATION INTERRUPTIONS):\n"
    "   - You are strictly barred from stopping and asking the developer questions for missing information (such as exception constructors, method signatures, service URLs, enum constants, DTO fields, or table schemas) if that information can be discovered in the codebase.\n"
    "   - In 90% of cases where an AI agent stops and asks the user for missing information, that information already exists in the repository. Avoid learned helplessness.\n"
    "   - You may ONLY escalate to asking the human if `find_references` returns 0 hits across the entire workspace AND the requirement cannot be derived from existing database schemas or database migrations.\n"
    "   - When an external absence is mathematically confirmed (0 occurrences across the entire repository and no database schema match), activate Tier 3 Structured Clarification Gate: present the developer with structured Option A vs Option B choices with concrete architectural trade-offs, never open-ended helpless questions.\n\n"
    "12. WORKING TREE INTEGRITY & COLLATERAL MUTATION PREVENTION (ZERO UNINTENDED CONFIG COMMITS):\n"
    "   - When correcting or auditing a branch with existing changes, ALWAYS run `git status` and `git diff` first to establish ground truth on the working tree.\n"
    "   - Never commit or dirty local developer configurations (e.g. `application*.yml`, `.env`, `appsettings.json`, config files containing local secrets or debug flags). If local configuration was dirtied, revert it via `git checkout <config-file>`.\n"
    "   - After applying edits, conduct a mandatory post-edit audit (`git status` and `git diff`) to confirm that ONLY the intended files were modified and zero extraneous debug lines or colliding migration files remain.\n\n"
    "13. CALL-CHAIN TRACEABILITY & ENTRYPOINT REACHABILITY (DEAD-CODE AVOIDANCE):\n"
    "   - Before modifying or proposing changes to any internal service, helper function, or repository method, you MUST trace callers backward to an active entrypoint (e.g., HTTP Controller, CLI Command, Saga/Workflow Handler, or Message Consumer) using `find_references` or call graph tools.\n"
    "   - Never patch an unreachable, dormant, or alternate helper method when the actual entrypoint routes through a different service path. Always document the complete execution path: `Entrypoint -> Service -> Target Method`.\n\n"
    "14. ORM & REPOSITORY NESTED PROPERTY TRAVERSAL:\n"
    "   - When declaring repository queries that filter through nested entity relationships (e.g., Spring Data JPA, Django ORM, EF Core), adhere strictly to framework relationship traversal conventions (e.g., property traversal delimiter `_` in Spring Data, `__` in Django, or explicit query annotations like `@Query`).\n"
    "   - Never blindly concatenate entity and child property names without the framework-mandated relation separator, as ORM entity metamodel parsers fail on startup when nested properties cannot be resolved.\n\n"
    "15. DISTRIBUTED TRANSACTION & SAGA ATOMICITY:\n"
    "   - In distributed workflows or saga patterns, all multi-step state mutations must participate in transactional boundaries or register explicit compensation handlers.\n"
    "   - Never silently swallow exceptions (`try { ... } catch (Exception e) { log.error(...); }`) to bypass rollback, as this introduces irrecoverable data desynchronization between services.\n\n"
    "16. POLYGLOT BUILD TOOL CONSTRAINTS:\n"
    "   - Never guess or assume build or test tools based on file extensions alone (e.g., never assume Maven for Java or npm for TypeScript without verifying the repository's build configuration).\n"
    "   - Always inspect and adhere strictly to the build system declared in `<build_and_runtime_environment>`.\n\n"
    "17. SURGICAL DELTA PRESERVATION & REPOSITORY INTEGRITY (ZERO INTERFACE WIPEOUTS):\n"
    "   - When presenting architectural blueprints, code reviews, or proposed changes for existing interfaces or classes (especially repository interfaces and service classes), NEVER output a truncated whole-file replacement that omits existing methods!\n"
    "   - Replacing an existing interface with only a newly proposed method destroys all existing production queries and causes catastrophic compilation failures across the entire service.\n"
    "   - You MUST output either:\n"
    "     (a) A surgical diff / search-replace block, or\n"
    "     (b) The exact method signature/snippet to add, explicitly stating `// Preserving all existing repository methods...`.\n"
    "   - Never alter existing repository generic type parameters (keep the declared entity and ID types exactly as they are).\n\n"
    "18. EXHAUSTIVE MULTI-PART TICKET DECOMPOSITION (ZERO DROPPED QUESTIONS):\n"
    "   - When a ticket or inquiry contains multiple questions, sub-tasks, or clauses (e.g. 'What is X, how many Y, and are any Z?'), you MUST decompose the prompt into its individual requirements.\n"
    "   - You MUST explicitly address and answer EVERY single question asked by the developer. Dropping, truncating, or answering only the first question is a catastrophic failure.\n"
    "   - Structure your final response with explicit numbered or bulleted sections corresponding to each question asked.\n"
    "   - If tools are invoked to inspect state (e.g. Playwright live UI inspection or codebase searches), extract ALL data points needed for all questions before synthesizing your authoritative response."
)


# ─── 4. Dynamic Prompt Assembler ─────────────────────────────────────────────

class PromptAssembler:
    """Assembles the 6-layer prompt required by the Agentic Operating System."""

    @staticmethod
    def build_system_instruction(
        intent: AgentIntent,
        workspace_path: Optional[str] = None,
        repo_invariants: Optional[List[str]] = None,
    ) -> str:
        """Assemble the system instruction.

        Repository-specific invariants are NOT baked into the universal constant any more; they are
        loaded from the skills layer (`SkillLoader.load_repo_invariants`) only when the workspace
        matches the repo's marker directories or ships an explicit `.aviator/skills` override.
        """
        if repo_invariants is None and workspace_path:
            try:
                from ticket_to_code.skills.loader import SkillLoader
                repo_invariants = SkillLoader(Path(workspace_path)).load_repo_invariants()
            except Exception as _skill_err:
                logger.debug(f"Repo invariant skill load skipped: {_skill_err}")
                repo_invariants = []
        identity_layer = (
            "<identity>\n"
            "You are an advanced agentic AI coding assistant designed for deep enterprise software engineering.\n"
            "You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, "
            "modifying or debugging an existing codebase, or providing architectural reviews.\n"
            "The USER will send you requests, which you must always prioritize addressing. User requests are enclosed within <USER_REQUEST> tags. "
            "Along with each USER request, we attach additional metadata about their current state, such as what files they have open and where their cursor is.\n"
            "</identity>"
        )

        guidelines_layer = (
            "<guidelines>\n"
            "- Maintain documentation integrity. Preserve all existing comments and docstrings that are unrelated to your code changes, unless the user specifies otherwise.\n"
            "- Never guess class constructors, method signatures, or database schemas. Always inspect the source code or existing tests first.\n"
            "- If in explanatory mode, do not execute code modification tools. If in autonomous implementation mode, verify changes with local tests or compiler checks before concluding.\n"
            "</guidelines>"
        )

        comm_style_layer = (
            "<communication_style>\n"
            "- Keep your responses concise and technically rigorous.\n"
            "- Format your responses in github-style markdown.\n"
            "- STRICT 0-INTERRUPTION ESCALATION BOUNDARY: You are strictly barred from asking the human for missing code details (exception constructors, method signatures, service URLs, enum constants, DTO fields, or DB columns) if that information can be found in the repository. You may ONLY escalate with questions to the human if `find_references` returns 0 hits across the entire workspace AND the requirement cannot be derived from existing database schemas. When escalating, provide structured Option A vs Option B technical choices.\n"
            "- You MUST create clickable links for all files and code symbols (classes, types, functions, structs). "
            "Use github style markdown links with the file:// scheme (e.g., [filename](file:///path/to/file) or [ClassName](file:///path/to/file#L10-L20)). For Windows, use forward slashes for paths.\n"
            "</communication_style>"
        )

        methodology_layer = (
            "AUTONOMOUS INVESTIGATION METHODOLOGY:\n"
            "- Tool Efficiency Best Practices:\n"
            "  * `find_references`: Finds all callers, usages, and import sites across all microservices. ALWAYS invoke this before modifying or renaming any shared constant or method!\n"
            "  * `verify_service_compilation`: Runs a fast incremental compile check on the service (e.g. gradlew compileJava) to verify compiler sanity.\n"
            "  * `inspect_file`: Supports AST-aware symbol extraction and comma-separated batching: pass `symbol='methodName'` to inspect complete method bodies. CRITICAL: If you use `symbol='outline'`, outlines only list method signatures. To implement or modify a method, you MUST inspect its actual body using `symbol='methodName'` so you read real constructors, injected beans, and exception signatures directly from code! NEVER guess constructors or bean calls from outlines alone.\n"
            "  * `search_codebase`: Supports broad text & regex across all file types: use it to discover tables, symbols, or annotations across services.\n"
            "  * `find_files`: Returns cross-service distributions: use broad patterns to survey all services in 1 query.\n"
            "  * `trace_call_chain`: Traces end-to-end call flow: from UI button/dialog/i18n key -> component -> service -> backend handler/API -> database.\n"
            "- For focused questions, inspect only the specific files necessary, then synthesize immediately without unnecessary tool calls.\n\n"
            f"{UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS}"
        )

        mode_behavior = (
            f"OPERATIONAL INTENT MODE: {intent.value.upper()}\n"
            "- EXPLANATION_REVIEW: This is a strictly read-only informational and diagnostic inquiry. Explore the workspace using read-only tools (search_codebase, inspect_file, find_references, trace_call_chain, verify_service_compilation) to verify ground truth and compiler sanity, then synthesize a complete, technically rigorous explanation, root-cause diagnosis, and verified code diffs directly in chat. When proposing code solutions or diffs, ensure all exception constructors and method signatures match real active call sites discovered via find_references/inspect_file. Do NOT call solve_ticket_with_flow, escalate_to_ticket_fix, replace_file_content, or write_to_file. Synthesize your final answer directly in conversational markdown.\n"
            "- AUTONOMOUS_MODIFICATION: For any defect, bug report, or ticket implementation, FIRST delegate execution to the `solve_ticket_with_flow` tool. It runs the verified multi-stage Flow (deterministic localization, locked-context patching, and polyglot build/test verification via `npm run build` for frontend and `gradlew`/`mvn` for backend) and returns the verification status and diffs. Only use `replace_file_content` / `write_to_file` directly for a small, surgical follow-up tweak the Flow did not cover. Always finish by reporting the Flow's verification status (verified / unverified / needs_clarification).\n"
            "- PLANNING_ARCHITECTURE: Produce a structured Markdown plan artifact detailing contract changes, migrations, and test matrices.\n"
        )

        timing_protocol = (
            "EXECUTION TIMING & TOOL SEQUENCING PROTOCOL (THE RIGHT TIME & RIGHT PLACE):\n"
            "1. INVESTIGATION PHASE (FIRST): NEVER propose or write code before inspecting real files. You MUST call `inspect_file` or `search_codebase` to discover real entity fields, DTO constructors, and UI frameworks. Memory-based speculation is strictly forbidden.\n"
            "2. CROSS-SERVICE PHASE (IN ORDER): For full-stack tickets spanning multiple layers (Backend, Gateway, Frontend), inspect in strict sequence:\n"
            "   (A) Backend: Inspect Entity (verify String vs Enum) -> DTO (verify constructor) -> Service JPQL -> Controller.\n"
            "   (B) Gateway: Inspect application.yml to verify route forwarding.\n"
            "   (C) Frontend: Inspect model -> service -> component template (verify which UI component library is actually in use) -> SCSS.\n"
            "3. PARALLEL TOOL EXECUTION (TURN BATCHING): Whenever investigating multiple files, endpoints, or layers, EMIT MULTIPLE TOOL CALLS SIMULTANEOUSLY in a single turn (e.g. call search_codebase on backend and inspect_file on frontend together). Do not wait for separate turns when queries are independent.\n"
            "4. MUTATION PHASE (WITH INSTANT COMPILATION): Apply edits using `replace_file_content`. Edits are automatically compiler-verified.\n"
            "5. SELF-HEALING PHASE: If the compiler returns any diagnostic (cannot find symbol, type mismatch), you must repair the issue immediately in the next turn.\n"
            "6. CHAT BLUEPRINT & CODE SPECIFICATION INTEGRITY:\n"
            "   When proposing code blueprints or architecture plans in conversational chat:\n"
            "   - NEVER output a truncated interface that wipes out existing methods! Always present surgical additions or diff blocks, preserving all existing method signatures and generic types (e.g. JpaRepository<Entity, UUID>).\n"
            "   - Follow the dependency-injection, transaction and join-key conventions observed in the inspected code; never assume them.\n"
        )

        repo_layer = ""
        if repo_invariants:
            try:
                from ticket_to_code.skills.loader import SkillLoader
                repo_layer = SkillLoader.format_repo_invariants(list(repo_invariants))
            except Exception:
                repo_layer = ""

        assembled = f"{identity_layer}\n\n{guidelines_layer}\n\n{comm_style_layer}\n\n{mode_behavior}\n\n{timing_protocol}\n\n{methodology_layer}"
        if repo_layer:
            assembled = f"{assembled}\n\n{repo_layer}"
        return assembled

    @staticmethod
    def build_user_prompt(
        user_message: str,
        host_meta: HostStateMetadata,
        ki_summaries: str = "",
        arch_block: str = "",
        conv_history: str = "",
        blackboard_xml: str = "",
    ) -> str:
        ki_block = ""
        if ki_summaries:
            ki_block = (
                "<knowledge_items>\n"
                "# Knowledge Items (KI) System\n"
                "### MANDATORY FIRST STEP: Check KI Summaries Before Any Research\n"
                f"{ki_summaries}\n"
                "Critical Rule: KIs are Starting Points, Not Ground Truth. Cross-reference them against active code before making edits.\n"
                "</knowledge_items>\n\n"
            )

        conv_section = ""
        if conv_history:
            conv_section = f"## Previous Conversation Context\n{conv_history}\n\n"

        arch_section = ""
        if arch_block:
            arch_section = f"## Workspace Architecture & Live Code Graph\n{arch_block}\n\n"

        bb_section = ""
        if blackboard_xml:
            bb_section = f"{blackboard_xml}\n\n"

        static_meta = host_meta.render_static_user_info()
        dynamic_meta = host_meta.render_dynamic_metadata()

        tail_parts = [p for p in (dynamic_meta, bb_section.strip()) if p]
        tail_block = ("\n\n" + "\n\n".join(tail_parts)) if tail_parts else ""

        return (
            f"{static_meta}\n\n"
            f"{ki_block}"
            f"{arch_section}"
            f"{conv_section}"
            "<USER_REQUEST>\n"
            f"{user_message}\n"
            "</USER_REQUEST>"
            f"{tail_block}\n\n"
            "Please investigate the codebase using your tools and provide an authoritative, grounded response."
        )


# ─── 5. Context Compaction & Resumption Engine ───────────────────────────────

class ContextCompactor:
    """Compacts conversation history into structured resumption blocks to survive token limits."""

    @staticmethod
    def compact_history(
        task_id: str,
        user_requests: List[str],
        verified_facts: List[str],
        active_context: str,
    ) -> str:
        req_lines = [f"{i+1}. {r}" for i, r in enumerate(user_requests)]
        fact_lines = [f"- {f}" for f in verified_facts]

        return (
            "# Resuming from a compaction\n\n"
            "You are continuing work on the task described above, but you have lost access to the full conversation history, "
            "and need to resume work efficiently using the progress summary below:\n\n"
            "# User Requests\n"
            "The following were user requests from the truncated conversation in chronological order:\n"
            + "\n".join(req_lines) + "\n\n"
            "<summary>\n"
            "### 1. Task Overview\n"
            f"- **Task / Ticket ID**: {task_id}\n"
            "- **Status**: In progress.\n\n"
            "### 2. Progress & Verified Ground Truth\n"
            + "\n".join(fact_lines) + "\n\n"
            "### 3. Active Context\n"
            f"{active_context}\n"
            "</summary>"
        )


# ─── 5b. ReAct Loop Governor (Exploration Leash & Anti-Spinning) ──────────────

# Tools the governor withdraws once the hard limit is hit. `inspect_file` stays available because the
# convergence directive explicitly tells the model to inspect specific candidate files.
EXPLORATION_TOOL_NAMES = frozenset({
    "search_codebase", "find_references", "find_files", "list_directory", "grep_search",
    "trace_call_chain", "crawl_sibling_and_rosetta_context", "invoke_subagent_explorer",
})


class ReActLoopGovernor:
    """Monitors tool calls during the multi-turn ReAct loop to prevent repetitive searching

    and force semantic convergence or synthesis when exploratory tools yield empty or duplicate results.
    """

    def __init__(self, max_consecutive_redundant: int = 3, hard_stop_limit: int = 6):
        self.max_consecutive_redundant = max_consecutive_redundant
        self.hard_stop_limit = hard_stop_limit
        self.consecutive_redundant_count: int = 0
        self.query_history: List[Tuple[str, str, int]] = []
        self.seen_queries: Set[str] = set()
        self.exploration_disabled: bool = False

    def should_disable_exploration(self) -> bool:
        """True once the hard limit has been reached; callers must stop offering exploration tools."""
        return self.exploration_disabled

    def filter_tools(self, tools: List[Any], name_of: Callable[[Any], str] = lambda t: getattr(t, "__name__", str(t))) -> List[Any]:
        """Return `tools` minus the exploration tools when exploration is disabled."""
        return [t for t in tools if not (self.exploration_disabled and name_of(t) in EXPLORATION_TOOL_NAMES)]

    def evaluate_call(self, tool_name: str, query: str, output: str) -> Optional[str]:
        """Evaluates a tool call observation. If redundant or empty spinning is detected,

        returns an authoritative governor guidance instruction to append to the tool response.
        """
        clean_q = query.strip().lower()
        query_key = f"{tool_name}:{clean_q}"
        is_duplicate = query_key in self.seen_queries and len(clean_q) > 2
        self.seen_queries.add(query_key)

        is_empty = (
            not output
            or "no matches found" in output.lower()
            or "no references found" in output.lower()
            or bool(re.search(r'\b0\s+matches\b', output, re.IGNORECASE))
            or len(output.strip()) < 40
        )

        if is_duplicate or is_empty:
            self.consecutive_redundant_count += 1
        else:
            self.consecutive_redundant_count = 0

        self.query_history.append((tool_name, clean_q, len(output)))

        if self.consecutive_redundant_count >= self.hard_stop_limit:
            self.exploration_disabled = True
            return (
                "\n\n[REACT_LOOP_GOVERNOR: HARD LIMIT REACHED]\n"
                f"You have executed {self.consecutive_redundant_count} consecutive exploratory queries that returned empty or duplicate results. "
                "Further searching is disabled for this turn. Synthesize your final grounded engineering response immediately using the facts recorded on the Blackboard."
            )
        elif self.consecutive_redundant_count >= self.max_consecutive_redundant:
            return (
                "\n\n[REACT_LOOP_GOVERNOR: CONVERGENCE DIRECTIVE]\n"
                f"Detected {self.consecutive_redundant_count} consecutive queries returning empty or duplicate observations ({tool_name} for '{query}'). "
                "Stop speculative searching. Either inspect specific candidate source files directly or synthesize your authoritative answer from current findings."
            )

        return None


# ─── 6. Host Environment Tool Implementations ────────────────────────────────

def apply_tolerant_replacement(raw: str, target_content: str, replacement_content: str) -> Optional[str]:
    """Applies code replacement with multi-stage tolerance:
    1. Exact string match
    2. Line-ending normalized match (CRLF <-> LF)
    3. Indentation & whitespace tolerant slice matching with blank line preservation
    """
    if not raw or not target_content:
        return None

    # 1. Exact match
    if target_content in raw:
        return raw.replace(target_content, replacement_content, 1)

    # 2. Line-ending normalized match (CRLF <-> LF)
    target_lf = target_content.replace("\r\n", "\n")
    raw_lf = raw.replace("\r\n", "\n")
    repl_lf = replacement_content.replace("\r\n", "\n")
    if target_lf in raw_lf:
        res_lf = raw_lf.replace(target_lf, repl_lf, 1)
        return res_lf.replace("\n", "\r\n") if "\r\n" in raw else res_lf

    # 3. Trimmed line-slice matching with blank line preservation and indentation alignment
    raw_lines = raw_lf.split("\n")
    target_lines = target_lf.split("\n")
    target_len = len(target_lines)

    if target_len > 0 and len(raw_lines) >= target_len:
        target_trimmed = [l.strip() for l in target_lines]
        for i in range(len(raw_lines) - target_len + 1):
            cand = [raw_lines[i + j].strip() for j in range(target_len)]
            if cand == target_trimmed:
                # Find indentation of the first non-empty line of the target match in raw
                orig_first_line = next((raw_lines[i + j] for j in range(target_len) if raw_lines[i + j].strip()), "")
                orig_indent = orig_first_line[:len(orig_first_line) - len(orig_first_line.lstrip())]

                repl_lines = repl_lf.split("\n")
                repl_first_line = next((l for l in repl_lines if l.strip()), "")
                repl_indent = repl_first_line[:len(repl_first_line) - len(repl_first_line.lstrip())]

                # Align replacement indentation to the destination code's indentation
                aligned_repl_lines = []
                for l in repl_lines:
                    if not l.strip():
                        aligned_repl_lines.append("")
                    elif repl_indent and l.startswith(repl_indent):
                        aligned_repl_lines.append(orig_indent + l[len(repl_indent):])
                    elif not repl_indent:
                        aligned_repl_lines.append(orig_indent + l)
                    else:
                        aligned_repl_lines.append(l)

                merged = raw_lines[:i] + aligned_repl_lines + raw_lines[i + target_len:]
                res = "\n".join(merged)
                return res.replace("\n", "\r\n") if "\r\n" in raw else res

    return None


class HostEnvironmentTools:
    """Executes tools on the local machine and returns observations to the LLM."""

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path).resolve()
        from ticket_to_code.editing.surgical_diff_engine import SurgicalDiffEngine
        self.diff_engine = SurgicalDiffEngine(str(self.workspace_path))

    def replace_file_content(
        self,
        file_path: str,
        target_content: str,
        replacement_content: str,
        expected_content_hash: str = "",
        expected_occurrences: int = 1,
    ) -> str:
        """Surgical, hash-checked replacement using SurgicalDiffEngine with rollback protection."""
        ok, why = SecurityPolicy.validate_path_containment(file_path, self.workspace_path)
        if not ok:
            return f"Security policy rejection: {why}"
        try:
            res = self.diff_engine.apply_anchored_replacement(
                rel_path=file_path,
                old_block=target_content,
                new_block=replacement_content,
                expected_match_count=expected_occurrences,
                expected_content_hash=expected_content_hash or None,
            )
            if res.success:
                return f"Successfully updated {file_path}. Lines: +{res.lines_added}, -{res.lines_removed}. New contentHash: {res.new_content_hash}"
            return f"Error [{res.status.value}]: {res.message}"
        except Exception as e:
            return f"Error replacing content: {e}"

    def rollback_all(self) -> List[str]:
        """Rollback all surgical edits made during this session."""
        return self.diff_engine.rollback_all()

    def write_to_file(self, file_path: str, code_content: str, overwrite: bool = False) -> str:
        """Safe file creation."""
        ok, why = SecurityPolicy.validate_path_containment(file_path, self.workspace_path)
        if not ok:
            return f"Security policy rejection: {why}"
        try:
            full_p = (self.workspace_path / file_path.lstrip("/\\")).resolve()
            if full_p.exists() and not overwrite:
                return f"Error: File '{file_path}' already exists. Pass overwrite=True to overwrite."
            full_p.parent.mkdir(parents=True, exist_ok=True)
            full_p.write_text(code_content, encoding="utf-8")
            return f"Successfully wrote {len(code_content)} chars to {full_p.name}."
        except Exception as e:
            return f"Error writing file: {e}"

    def run_command(self, command: str, cwd: str = "", timeout_seconds: int = 45) -> str:
        """Executes a shell command in workspace (policy-checked, env-scrubbed, process-tree bounded)."""
        try:
            rc, out, err = run_checked_subprocess(
                command, cwd=cwd, timeout=timeout_seconds, workspace=self.workspace_path
            )
            combined = (out + "\n" + err).strip()
            return f"[Exit code {rc}]\n{truncate(combined, TOOL_OUTPUT_MAX_CHARS)}"
        except SecurityViolation as sv:
            return f"Security policy rejection: {sv}"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout_seconds}s (process tree terminated)."
        except Exception as e:
            return f"Command error: {e}"

    def verify_service_compilation(self, service_name: str = "") -> str:
        """Phase 3 Compiler Feedback Loop: polyglot build/compile gate.

        Frontend (package.json 'build' script -> `npm run build`, else `npm test`) and backend
        (gradle `compileJava` / maven `compile`) are both supported. No service name is hardcoded;
        if a named sub-service has no build tool, falls back to the workspace root."""
        is_win = sys.platform == "win32"
        sname = service_name.strip()
        target_dir = (
            self.workspace_path / sname
            if sname and (self.workspace_path / sname).is_dir()
            else self.workspace_path
        )
        detected = self._detect_build_command(target_dir, is_win)
        if detected is None and target_dir != self.workspace_path:
            target_dir = self.workspace_path
            detected = self._detect_build_command(target_dir, is_win)
        if detected is None:
            return f"No supported build tool (npm/gradle/maven) found in '{target_dir.name}'."

        label, cmd = detected
        try:
            rc, out, err = run_checked_subprocess(
                cmd, cwd=target_dir, timeout=1800, workspace=self.workspace_path
            )
        except SecurityViolation as sv:
            return f"Security policy rejection: {sv}"
        except subprocess.TimeoutExpired:
            return f"{label} verification timed out in '{target_dir.name}' (process tree terminated)."
        except Exception as e:
            return f"Compilation execution error: {e}"

        if rc == 0:
            return f"[OK] {label} verification SUCCESS on {target_dir.name}: clean build, 0 errors."
        err_text = (err or out or "").strip()
        try:
            from ticket_to_code.runtime.compiler_gate import CompilerGate
            comp_obj = CompilerGate.parse_compiler_output(target_dir.name, rc, err_text)
            if comp_obj.diagnostics:
                return comp_obj.get_repair_instruction()
        except Exception:
            pass
        return f"[FAIL] {label} verification FAILED on {target_dir.name}:\n{truncate(err_text, BUILD_OUTPUT_MAX_CHARS)}"

    @staticmethod
    def _detect_build_command(target_dir: Path, is_win: bool) -> Optional[Tuple[str, str]]:
        """Resolve (label, shell_command) for the project's build tool, or None.

        Covers frontend (npm/Angular) and backend (gradle/maven) so verification is polyglot
        rather than JVM-only. Detection is file-driven and independent of any project name."""
        pkg = target_dir / "package.json"
        if pkg.is_file():
            try:
                scripts = json.loads(pkg.read_text(encoding="utf-8", errors="ignore")).get("scripts") or {}
            except Exception:
                scripts = {}
            if "build" in scripts:
                return ("npm", "npm run build")
            if "test" in scripts:
                return ("npm", "npm test -- --watch=false")
        if (target_dir / "gradlew.bat").is_file() and is_win:
            return ("gradle", ".\\gradlew.bat compileJava --daemon -Dorg.gradle.parallel=true -Dorg.gradle.caching=true -q")
        if (target_dir / "gradlew").is_file():
            return ("gradle", "./gradlew compileJava --daemon -Dorg.gradle.parallel=true -Dorg.gradle.caching=true -q")
        if (target_dir / "mvnw.cmd").is_file() and is_win:
            return ("maven", ".\\mvnw.cmd -q -DskipTests compile")
        if (target_dir / "mvnw").is_file():
            return ("maven", "./mvnw -q -DskipTests compile")
        if (target_dir / "pom.xml").is_file():
            return ("maven", "mvn -q -DskipTests compile")
        if (target_dir / "build.gradle").is_file() or (target_dir / "build.gradle.kts").is_file():
            return ("gradle", "gradle compileJava --daemon -Dorg.gradle.parallel=true -Dorg.gradle.caching=true -q")
        return None


# ─── 7. The Reusable AgenticHarness ──────────────────────────────────────────

class AgenticHarness:
    """The Autonomous Agent Harness bridging LLM with host operating system."""

    def __init__(
        self,
        workspace_path: str,
        model_client: Any = None,
        tools: Optional[Dict[str, Callable]] = None,
        host_meta: Optional[HostStateMetadata] = None,
    ):
        self.workspace_path = Path(workspace_path).resolve()
        self.model_client = model_client
        self.host_tools = HostEnvironmentTools(str(self.workspace_path))
        self.tools = tools or {}
        self.host_meta = host_meta or HostStateMetadata.capture_current(str(self.workspace_path))

    def get_toolset_for_intent(self, intent: AgentIntent) -> List[Callable]:
        """Dynamically partitions tools based on intent."""
        # Read-only tools available across all modes
        active_tools = [v for k, v in self.tools.items() if k not in ("replace_file_content", "write_to_file", "run_command")]

        # Mutation tools only in autonomous modification mode
        if intent == AgentIntent.AUTONOMOUS_MODIFICATION:
            active_tools.extend([
                self.host_tools.replace_file_content,
                self.host_tools.write_to_file,
                self.host_tools.run_command,
            ])

        return active_tools

    def build_full_prompt(self, user_message: str, intent: Optional[AgentIntent] = None) -> Tuple[str, str]:
        """Builds (system_instruction, user_prompt) adhering to the 6 prompt layers."""
        effective_intent = intent or classify_agent_intent(user_message)
        sys_inst = PromptAssembler.build_system_instruction(effective_intent, workspace_path=str(self.workspace_path))
        user_p = PromptAssembler.build_user_prompt(
            user_message=user_message,
            host_meta=self.host_meta,
        )
        return sys_inst, user_p

    def run_turn(
        self,
        user_message: str,
        max_steps: int = 25,
        target_model: str = "gemini-2.5-pro",
        emit_fn: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        """Executes the multi-turn agent loop with AFC/ReAct, tool execution, and compiler self-correction.
        
        Iterative dialogue between model and host environment:
        1. Classifies intent (Explanation vs Autonomous Modification vs Planning).
        2. Injects 6-layer prompt with HostStateMetadata and Universal Invariants.
        3. Dynamically partitions toolset (inhibits mutation tools in explanation mode).
        4. Lets model call tools, executes them on local OS, feeds observations back.
        5. In modification mode, performs closed-loop compiler check.
        """
        intent = classify_agent_intent(user_message)
        sys_inst, user_prompt = self.build_full_prompt(user_message, intent)
        tools = self.get_toolset_for_intent(intent)

        client = self.model_client
        if client is None:
            from google import genai
            from google.genai import types
            _gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
            if not _gcp_project:
                raise RuntimeError(
                    "AgentHarness needs a Vertex AI client but GOOGLE_CLOUD_PROJECT is not set "
                    "(no default project is assumed). Set GOOGLE_CLOUD_PROJECT or inject model_client."
                )
            client = genai.Client(
                vertexai=True,
                project=_gcp_project,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west4"),
                http_options=types.HttpOptions(timeout=720000),
            )
        else:
            from google.genai import types

        model_thinking_config = (
            types.ThinkingConfig(thinking_budget=0)
            if "flash" in target_model.lower()
            else types.ThinkingConfig(thinking_budget=1024)
        )

        chat = client.chats.create(
            model=target_model,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=max_steps
                ),
                temperature=0.1,
                max_output_tokens=8192,
                thinking_config=model_thinking_config,
            ),
        )

        reply_chunks = []
        for chunk in chat.send_message_stream(user_prompt):
            txt = getattr(chunk, "text", None)
            if txt:
                reply_chunks.append(txt)
                if emit_fn:
                    emit_fn("text_delta", {"delta": txt})

        final_text = "".join(reply_chunks).strip()
        return final_text


# ─── 8. Chat → Flow Delegation (verified execution engine) ───────────────────

def _default_prepare_flow(
    workspace_path: str,
    ticket_id: str,
    title: str,
    description: str,
    labels: Optional[List[str]],
    meter: Any,
    run_tier2: bool,
    blackboard: Optional[Any] = None,
) -> Tuple[Optional[Callable], Optional[Callable], Any]:
    """Build (patch_fn, verify_fn, scope) via the proven lean pipeline
    (triage → locate → lock → bound LLM patcher + polyglot compile/build verifier).

    Returns (None, None, scope) when the ticket is not lean-solvable so the caller can
    report that cleanly instead of inventing an edit."""
    from ticket_to_code.lean.scope import triage
    from ticket_to_code.lean.locate import locate
    from ticket_to_code.lean.lean_path import build_locked_context, should_use_lean
    from ticket_to_code.lean.live import make_patch_fn, make_compile_verify_fn

    # Enhance ticket text with any verified facts discovered during Chat exploration
    augmented_description = description
    if blackboard is not None and getattr(blackboard, "verified_facts", None):
        known_facts = "\n".join(f"- {fact}" for fact in blackboard.verified_facts[:8])
        augmented_description = f"{description}\n\nVerified Codebase Facts from Investigation:\n{known_facts}"

    scope = triage(ticket_id, title, augmented_description, workspace_path, None, labels)
    if not should_use_lean(scope):
        return None, None, scope
    cands, fence = locate(scope)
    if not cands:
        return None, None, scope
    ctx = build_locked_context(scope, cands, fence)
    ticket_text = f"{title}\n{augmented_description}"
    patch_fn = make_patch_fn(ctx, cands, ticket_text, meter)
    verify_fn = make_compile_verify_fn(run_tier2=run_tier2)
    return patch_fn, verify_fn, scope


def delegate_ticket_to_flow(
    workspace_path: str,
    title: str,
    description: str,
    ticket_id: str = "CHAT-TICKET",
    labels: Optional[List[str]] = None,
    progress_cb: Optional[Callable[[str, str], None]] = None,
    run_tier2: bool = True,
    hard_timeout_s: float = 600.0,
    runner: Optional[Callable] = None,
    prepare: Optional[Callable] = None,
    blackboard: Optional[Any] = None,
) -> Dict[str, Any]:
    """Hand a defect / bug / ticket to the verified Flow (`run_lean_path`) and stream progress.

    The Chat agent is the front door; the Flow is the single execution engine. Binds Chat's
    DeductiveBlackboard bidirectional state to the Flow so exploration facts and compiler outcomes
    are unified into a single shared execution brain.
    """

    def _emit(stage: str, detail: str = "") -> None:
        if progress_cb:
            try:
                progress_cb(stage, detail)
            except Exception:
                pass
        if blackboard is not None:
            try:
                if stage in ("localized", "patching", "flow_start"):
                    blackboard.record_fact(f"Flow [{stage}]: {detail}")
                elif any(k in detail.lower() for k in ("failed", "rejected", "error", "regression", "declined")):
                    blackboard.record_rejection(f"Flow [{stage}]: {detail}")
            except Exception:
                pass

    from ticket_to_code.lean.lean_path import TokenMeter

    meter = TokenMeter()
    prepare = prepare or _default_prepare_flow
    if runner is None:
        from ticket_to_code.lean.lean_path import run_lean_path as runner

    _emit("flow_start", f"{ticket_id}: {title}")
    
    # Try prepare with blackboard support
    try:
        patch_fn, verify_fn, scope = prepare(
            workspace_path, ticket_id, title, description, labels, meter, run_tier2, blackboard=blackboard
        )
    except TypeError:
        patch_fn, verify_fn, scope = prepare(
            workspace_path, ticket_id, title, description, labels, meter, run_tier2
        )
        
    layer = getattr(scope, "layer", "?")
    conf = getattr(scope, "confidence", "?")
    _emit("localized", f"layer={layer} conf={conf}")

    if patch_fn is None or verify_fn is None:
        _emit("react_escalation", f"Escalating multi-file ticket (layer={layer}) to AutonomousReActEngine")
        try:
            from ticket_to_code.runtime.react_agent_engine import AutonomousReActEngine
            from ticket_to_code.models import ValueEdgeTicket
            
            t = ValueEdgeTicket(ticket_id=ticket_id, title=title, description=description, labels=labels or [])
            engine = AutonomousReActEngine(workspace_path=workspace_path, ticket=t)
            react_res = engine.run()
            if react_res.success:
                _emit("flow_done", "status=verified (via AutonomousReActEngine)")
                return {
                    "status": "verified",
                    "reason": react_res.message,
                    "locked_files": react_res.modified_files,
                    "attempts": react_res.total_steps,
                    "diffs": list(react_res.diffs.values()),
                    "trace": [f"step {react_res.total_steps}: {react_res.status}"],
                    "llm_calls": react_res.total_steps,
                }
            else:
                _emit("declined", f"AutonomousReActEngine could not converge: {react_res.error_message}")
                return {
                    "status": "out_of_scope",
                    "reason": f"AutonomousReActEngine failed: {react_res.error_message}",
                    "locked_files": [],
                    "attempts": 0,
                    "diffs": [],
                    "trace": [],
                    "llm_calls": 0,
                }
        except Exception as exc:
            logger.error(f"[AgentHarness] AutonomousReActEngine escalation failed: {exc}")
            _emit("declined", f"not a lean ticket (layer={layer})")
            return {
                "status": "out_of_scope",
                "reason": f"Flow declined: not a lean-solvable ticket (layer={layer}, conf={conf}).",
                "locked_files": [],
                "attempts": 0,
                "diffs": [],
                "trace": [],
                "llm_calls": 0,
            }

    # ── Out-of-Tree Git Sandbox Isolation ──
    from ticket_to_code.runtime.git_sandbox import GitWorktreeSandbox
    sandbox_env = None
    ws_dir = Path(workspace_path)
    if ws_dir.is_dir() and (ws_dir / ".git").exists():
        try:
            sandbox_env = GitWorktreeSandbox.create_sandbox(ws_dir, ticket_id)
            _emit("sandbox_created", f"Isolated in {sandbox_env.sandbox_path.name}")
        except Exception:
            sandbox_env = None

    exec_path = str(sandbox_env.sandbox_path) if sandbox_env else workspace_path

    _emit("patching", "running convergent patch -> verify loop")
    result = runner(
        ticket_id, title, description, exec_path, patch_fn, verify_fn,
        labels=labels, hard_timeout_s=hard_timeout_s, meter=meter,
    )
    status = getattr(result, "status", "error")
    locked_files = list(getattr(result, "locked_files", []) or [])

    if sandbox_env:
        try:
            if status == "verified" and locked_files:
                GitWorktreeSandbox.apply_sandbox_to_original(sandbox_env, locked_files)
                _emit("sandbox_promoted", f"Promoted verified changes for {len(locked_files)} file(s)")
            GitWorktreeSandbox.cleanup_sandbox(sandbox_env)
        except Exception:
            pass

    _emit("flow_done", f"status={status}")

    trace = list(getattr(result, "trace", []) or [])
    diffs = [t for t in trace if t.startswith("attempt") or "verified" in t]
    locked_files = list(getattr(result, "locked_files", []) or [])
    reason = getattr(result, "reason", "")

    # Hydrate Chat's DeductiveBlackboard with Flow outcomes (FLAW-01 Resolution)
    if blackboard is not None:
        try:
            if status == "verified":
                for f in locked_files:
                    blackboard.record_fact(f"Flow verified patch applied to '{f}'")
                blackboard.record_fact(f"Flow verified ticket '{ticket_id}' with clean polyglot compilation")
            elif status in ("failed", "unverified", "out_of_scope"):
                if reason:
                    blackboard.record_rejection(f"Flow execution for '{ticket_id}' unverified: {reason}")
        except Exception:
            pass

    return {
        "status": status,
        "reason": reason,
        "locked_files": locked_files,
        "attempts": getattr(result, "attempts", 0),
        "diffs": diffs,
        "trace": trace,
        "llm_calls": getattr(result, "llm_calls", 0),
    }
