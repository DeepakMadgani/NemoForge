"""
Planning Agent

Decomposes features into modules, tasks, and architectural decisions.

This is Agent 2 in the autonomous pipeline - the "System Intelligence Layer"

Author: Deepak Madgani
Date: April 2026
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Any, Union
import time
from datetime import datetime
from ticket_to_code.llm_utils import llm_invoke, extract_token_usage

from ticket_to_code.runtime.llm_factory import LLMRegistry
from langchain_core.messages import SystemMessage, HumanMessage

import re as _re

from ticket_to_code.models import (
    ValueEdgeTicket,
    StructuredRequirements,
    ArchitecturalPlan,
    DevelopmentTask,
    APIChange,
    DatabaseChange,
    ArchitecturalPattern,
    TechnicalFacts,
    StageThought,
    ThinkingChain,
    PlanReviewResult,
    SignatureBlueprint,
    CrossFileContract,
    BlueprintStatus,
)


def _extract_thinking(raw: str) -> str:
    m = _re.search(r"<thinking>(.*?)</thinking>", raw, _re.DOTALL | _re.IGNORECASE)
    return m.group(1).strip() if m else raw[:600].strip()
logger = logging.getLogger(__name__)


class PlanningAgent:
    """
    Agent 2: Planning & Task Decomposition
    
    Acts like a senior engineer - breaks down features into:
    - Architectural decisions
    - Module decomposition
    - Task sequencing
    - Dependency analysis
    """
    
    def __init__(self, workspace_path: Optional[Union[str, Path]] = None):
        # Always use the smarter assistant model (gemini-1.5-pro / gemini-2.5-flash) for planning
        # Use a lower max_output_tokens for planning — plans are structured JSON, not full files
        self.llm = LLMRegistry.get_llm(assistant=True)
        self._workspace_path: Optional[str] = str(workspace_path) if workspace_path else None
        # Cap planner output to 16K tokens — enough for a detailed plan JSON
        # but prevents the LLM from spending 10 minutes on massive outputs
        try:
            if hasattr(self.llm, 'max_output_tokens'):
                self.llm.max_output_tokens = 16384
            elif hasattr(self.llm, 'max_tokens'):
                self.llm.max_tokens = 16384
        except Exception:
            pass
        logger.info("Planning Agent initialized")

    @property
    def workspace_path(self) -> Optional[Path]:
        ws = getattr(self, "_workspace_path", None)
        return Path(ws) if ws else None

    @workspace_path.setter
    def workspace_path(self, val: Optional[Union[str, Path]]) -> None:
        self._workspace_path = str(val) if val else None
    
    def create_plan(
        self,
        ticket: ValueEdgeTicket,
        requirements: StructuredRequirements,
        retrieved_capabilities: Optional[Any] = None,
        workspace_path: Optional[str] = None,
        workspace_knowledge: Optional[Any] = None,
        technical_facts: Optional["TechnicalFacts"] = None,
        code_facts: Optional[str] = None,
        thinking_chain: Optional[ThinkingChain] = None,
        **kwargs
    ) -> Any:
        """
        Create architectural plan using retrieved semantic capabilities.
        
        Args:
            ticket: Original ticket
            requirements: Structured requirements from Analyzer
            retrieved_capabilities: Filtered semantic graph nodes
            
        Returns:
            Complete architectural plan with tasks
        """
        print("\n" + "="*80)
        print("[ENTER] PlanningAgent.create_plan() in planning_agent.py")
        print("   Purpose: Design architecture and break down into development tasks")
        print(f"   Ticket: {ticket.ticket_id}")
        print("="*80)
        logger.info(f"Creating plan for ticket: {ticket.ticket_id}")
        
        start_time_ts = time.time()
        start_time_iso = datetime.utcnow().isoformat()
        trace = {
            "stage_name": "Behavior Planning",
            "start_time": start_time_iso,
            "end_time": "",
            "duration_ms": 0,
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "accepted": [],
            "rejected": [],
            "skipped": [],
            "metrics": {},
            "decision_reasoning": {}
        }
        
        def _empty_return(error_msg=None):
            if error_msg:
                trace["decision_reasoning"]["error"] = error_msg
            trace["end_time"] = datetime.utcnow().isoformat()
            trace["duration_ms"] = int((time.time() - start_time_ts) * 1000)
            plan = ArchitecturalPlan(
                pattern="Layered Architecture", # type: ignore (or ArchitecturalPattern.LAYERED if imported)
                affected_modules=[],
                api_changes=[],
                database_changes=[],
                tasks=[],
                estimated_total_complexity=0
            )
            return plan if kwargs else (plan, trace)

        try:
            # Store workspace_path so _build_planning_prompt can load brain knowledge
            self._workspace_path = workspace_path
            system_prompt = self._build_planning_prompt(technical_facts)
            
            # Extract Pipeline A arguments from kwargs if present
            discovered_files = kwargs.pop("discovered_files", None)
            blacklisted_files = kwargs.pop("blacklisted_files", None)
            
            user_prompt = self._build_user_prompt(
                ticket, 
                requirements, 
                retrieved_capabilities=retrieved_capabilities,
                workspace_path=workspace_path,
                workspace_knowledge=workspace_knowledge,
                technical_facts=technical_facts,
                discovered_files=discovered_files,
                blacklisted_files=blacklisted_files,
                code_facts=code_facts,
                **kwargs
            )

            # ── Inject accumulated reasoning from upstream agents ──────────
            if thinking_chain and thinking_chain.thoughts:
                user_prompt = thinking_chain.to_prompt_block() + "\n\n" + user_prompt
            
            self.last_system_prompt = system_prompt
            self.last_user_prompt = user_prompt

            from langchain_core.messages import AIMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            max_llm_retries = 3
            final_plan = None
            response = None
            
            for attempt in range(max_llm_retries):
                response = llm_invoke(self.llm, messages)
                
                # Extract Tokens
                token_usage = extract_token_usage(response)
                # Simple accumulation for trace
                for k, v in token_usage.items():
                    if isinstance(v, int):
                        trace[k] = trace.get(k, 0) + v
                        
                try:
                    # Parse response into final plan
                    final_plan = self._parse_response(response.content)
                    break  # Success
                except ValueError as ve:
                    logger.warning(f"Planner LLM failed validation (attempt {attempt+1}/{max_llm_retries}): {ve}")
                    if attempt < max_llm_retries - 1:
                        messages.append(AIMessage(content=response.content))
                        messages.append(HumanMessage(content=f"Your previous response failed validation with error:\n{ve}\n\nPlease fix the JSON and try again. Do not output anything other than valid JSON."))
                    else:
                        raise ValueError(f"Failed to generate valid plan after {max_llm_retries} attempts: {ve}")
            
            # Calculate total complexity
            final_plan.estimated_total_complexity = sum(
                task.estimated_complexity for task in final_plan.tasks
            )
            
            selected_files = [t.file_path for t in final_plan.tasks]
            trace["accepted"] = selected_files
            trace["decision_reasoning"]["why_selected"] = "Generated from capability graph"
            
            if retrieved_capabilities and hasattr(retrieved_capabilities, "relevant_nodes"):
                candidate_files = set()
                for rc in retrieved_capabilities.relevant_nodes:
                    if hasattr(rc.capability, "participating_files"):
                        for pf in rc.capability.participating_files:
                            candidate_files.add(pf.file_path)
                            
                rejected_files = candidate_files - set(selected_files)
                trace["rejected"] = [{"file": f, "reason": "Not selected by planner"} for f in rejected_files]
                
                trace["metrics"]["candidate_files_count"] = len(candidate_files)
                trace["metrics"]["selected_files_count"] = len(selected_files)
            
            logger.info(
                f"Planning complete: {len(final_plan.tasks)} total tasks, "
                f"complexity={final_plan.estimated_total_complexity}"
            )
            
            trace["decision_reasoning"]["raw_response"] = response.content
            
            # DEBUG: dump raw response
            if workspace_path:
                try:
                    from ticket_to_code.tracing import get_trace_dir
                    debug_dir = get_trace_dir(ticket.ticket_id, create=True)
                    with open(debug_dir / "planner_raw.txt", "w", encoding="utf-8") as f:
                        f.write(response.content)
                except Exception:
                    pass
            
            plan = self._parse_response(response.content)
            
            trace["end_time"] = datetime.utcnow().isoformat()
            trace["duration_ms"] = int((time.time() - start_time_ts) * 1000)

            # ── Emit StageThought for downstream agents ───────────────────
            raw_thinking = _extract_thinking(response.content)
            thought = StageThought(
                stage="planning",
                summary=(
                    f"Created {len(plan.tasks)} task(s) using pattern '{plan.pattern}'. "
                    f"Selected files: {', '.join(t.file_path for t in plan.tasks[:4])}."
                ),
                key_decisions=[
                    f"[{t.id}] {t.task_type.value}: {t.file_path}  — {t.selection_reason[:80] if t.selection_reason else ''}"
                    for t in plan.tasks[:6]
                ],
                signals_noted=[
                    f"candidates_seen={len(discovered_files) if discovered_files else 'N/A'}",
                    f"blacklisted={len(blacklisted_files) if blacklisted_files else 0}",
                ],
                confidence=0.8 if plan.tasks else 0.3,
                raw_thinking=raw_thinking,
            )
            updated_chain = (thinking_chain or ThinkingChain()).append(thought)

            # Return: if called from old code paths that don't pass kwargs,
            # include chain in return; otherwise legacy tuple (plan, trace) preserved.
            if not kwargs:
                return plan, trace, updated_chain
            return plan

        except Exception as e:
            logger.error(f"Failed to generate architectural plan: {e}")
            return _empty_return(str(e))

    def evaluate_shallow_candidates(self, shallow_code_facts: str, ticket: ValueEdgeTicket) -> List[str]:
        """
        Evaluate shallow snippets and select relevant files for full extraction.
        Returns a list of file paths.
        """
        system_prompt = f"""You are an Expert Code Investigator.
Your ONLY job is to review a set of file snippets (SHALLOW CONTEXT) and decide WHICH of these files need to be investigated in FULL to solve the given ticket.

TICKET: {ticket.title}
DESCRIPTION: {ticket.description}

RULES:
1. Review each snippet provided. 
2. Identify ONLY the files that MUST be modified or directly used to resolve the issue. Do NOT include files that merely contain related context or background information.
3. If it is clearly irrelevant or just an import statement, ignore it.
4. Output a raw JSON list of file paths. ONLY JSON. No markdown, no explanations.

Example output:
[
  "src/app/app.component.ts",
  "src/app/services/auth.service.ts"
]"""
        user_prompt = f"=== SHALLOW CONTEXT ===\n{shallow_code_facts}"
        
        try:
            response = llm_invoke(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            cleaned = response.content.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
            paths = json.loads(cleaned)
            if isinstance(paths, list):
                return [str(p) for p in paths]
            return []
        except Exception as e:
            logger.error(f"Failed to evaluate shallow candidates: {e}")
            return []

    def _build_planning_prompt(self, technical_facts: Optional["TechnicalFacts"] = None) -> str:
        """Build system prompt — authoritative about MUST_USE files from TechnicalFacts."""

        ticket_type = "UNKNOWN"
        if hasattr(self, "ticket") and self.ticket:
            labels = [lbl.upper() for lbl in (self.ticket.labels or [])]
            if any("BUG" in l for l in labels):
                ticket_type = "BUG_FIX"
            elif any("FEATURE" in l for l in labels):
                ticket_type = "FEATURE"
            elif any("REFACTOR" in l for l in labels):
                ticket_type = "REFACTOR"
        
        if ticket_type == "BUG_FIX":
            task_limit_rule = "- Max 8 writable tasks total. A bug fix should be focused and minimal."
        elif ticket_type == "FEATURE":
            task_limit_rule = "- Max 15 writable tasks total. A feature may span multiple layers (UI, API, DB)."
        elif ticket_type == "REFACTOR":
            task_limit_rule = "- Max 20 writable tasks total. Refactoring can touch many files."
        else:
            task_limit_rule = "- Max 10 writable tasks total (prefer 5-7)."

        must_use_block = ""
        ui_hints_block = ""
        if technical_facts:
            must_use_files = [f.name for f in (technical_facts.files or [])]
            if must_use_files:
                must_use_block = (
                    "\\n⚠️  HIGH CONFIDENCE FILES (evidence confidence ≥ 0.90):\\n"
                    + "\\n".join(f"  • {fn}" for fn in must_use_files)
                    + "\\nThe search engine prioritized these files. Review their contents carefully."
                    + "\\nGenerate a `modify` task for these files unless you have a critical reason not to.\\n"
                )
            if technical_facts.versions:
                v = technical_facts.versions[0]
                must_use_block += (
                    f"\\n⚠️  VERSION TRANSITION (from ticket): {v.old_value} → {v.new_value}\\n"
                    f"   Permutations of old value to search for: {', '.join(v.permutations_old[:4])}\\n"
                    f"   Replace with new value: {v.new_value}\\n"
                )
            if getattr(technical_facts, 'classified_type', None) == "UI_STYLING":
                ui_hints_block = (
                    "\\n🎨 ARCHITECTURAL HINTS (UI_STYLING):\\n"
                    " - For UI alignment or layout issues, prefer flexible layouts (e.g. `flex-grow: 1`) over static absolute heights (e.g. `height: 28.5rem`).\\n"
                    " - Avoid truncating or arbitrarily limiting array lengths in business logic (.ts) if the issue is just visual layout.\\n"
                    " - Select SCSS/CSS files for layout/alignment tasks, but verify HTML templates if DOM structure changes are needed.\\n"
                )

        # Load Architecture knowledge from brain folder (dynamic — works for ANY project)
        architecture_map_block = ""
        try:
            import os, glob, json
            # Use workspace_path passed to create_plan() — stored as instance attr
            ws = getattr(self, '_workspace_path', None)
            if ws:
                brain_knowledge_dir = os.path.join(ws, "brain", "knowledge")
                if os.path.isdir(brain_knowledge_dir):
                    arch_parts = []
                    # Load all .md files from brain knowledge (architecture maps, docs)
                    for md_file in sorted(glob.glob(os.path.join(brain_knowledge_dir, "*.md"))):
                        try:
                            with open(md_file, "r", encoding="utf-8") as f:
                                content = f.read().strip()
                            if content and len(content) < 5000:  # Skip huge docs
                                arch_parts.append(f"--- {os.path.basename(md_file)} ---\\n{content}")
                        except Exception:
                            pass
                    # Also load directory brain JSON as compact context
                    brain_json = os.path.join(brain_knowledge_dir, "generated_directory_brain.json")
                    if os.path.exists(brain_json):
                        try:
                            with open(brain_json, "r", encoding="utf-8") as f:
                                brain_data = json.load(f)
                            # Only include folder-level entries (compact)
                            folders = [e.get("path", "") for e in brain_data if isinstance(e, dict) and e.get("type") == "directory"]
                            if folders:
                                arch_parts.append(
                                    "--- Repository Structure ---\\n"
                                    + "\\n".join(f"  {p}" for p in folders[:50])
                                )
                        except Exception:
                            pass
                    if arch_parts:
                        architecture_map_block = (
                            "\n=== PROJECT ARCHITECTURE KNOWLEDGE ===\n"
                            + "\n\n".join(arch_parts)
                            + "\n=== END ARCHITECTURE KNOWLEDGE ===\n"
                            "\nUse this architecture knowledge to select the CORRECT files. "
                            "Match the ticket's domain to the right module/service.\n"
                        )
        except Exception:
            pass

        # Load Flyway migration knowledge dynamically from target module
        flyway_migration_block = ""
        try:
            ws = getattr(self, '_workspace_path', None)
            if ws:
                mig_dirs = glob.glob(os.path.join(ws, "**/src/main/resources/db/migration"), recursive=True)
                for md in mig_dirs:
                    def _ver_key(fpath):
                        fname = os.path.basename(fpath)
                        nums = [int(n) for n in re.findall(r'\d+', fname)]
                        return nums or [0]
                    sqls = sorted(glob.glob(os.path.join(md, "*.sql")), key=_ver_key)
                    if sqls:
                        latest_sql = os.path.basename(sqls[-1])
                        rel_md = os.path.relpath(md, ws).replace("\\", "/")
                        m = re.match(r'V(\d+)(?:\.(\d+))?__', latest_sql)
                        if m:
                            major = int(m.group(1))
                            next_ver = f"V{major + 1}.0"
                        else:
                            next_ver = "V11.0"
                        flyway_migration_block += (
                            f"\n=== FLYWAY DATABASE MIGRATION SPECIFICATION ===\n"
                            f"Migration directory: {rel_md}\n"
                            f"Latest applied script: {latest_sql}\n"
                            f"If ticket requires database cleanup, schema change, or orphaned row deletion, you MUST create a task with:\n"
                            f"  - task_type: 'create'\n"
                            f"  - language: 'sql'\n"
                            f"  - file_path: '{rel_md}/{next_ver}__<Descriptive_Action_Name>.sql'\n"
                            f"  - new_file_creation_allowed: true\n"
                            f"  - selection_reason: 'Flyway forward-only cleanup migration'\n"
                            f"=== END FLYWAY DATABASE MIGRATION SPECIFICATION ===\n"
                        )
        except Exception:
            pass

        return f"""You are a Precise Software Planner working on a real production codebase.

YOUR ONLY JOB:
Given a ticket and a ranked list of candidate files, produce the minimal set of tasks needed.
Do NOT hallucinate file paths for 'modify' tasks. Use ONLY the exact paths shown in CANDIDATE FILES.
If a required file does not exist in CANDIDATE FILES:
1. Prioritize modifying existing files whenever possible. You MUST choose the closest existing file from the candidates to implement the required behavior.
2. ONLY use the 'create' task_type when creation is truly necessary (no existing file can safely host the behavior) or explicitly requested in the ticket.
3. Never emit a non-existent path as 'modify'. For any net-new path, emit task_type='create' and set new_file_creation_allowed=true.
4. For every create task, include a concrete selection_reason explaining why existing files were insufficient.

{must_use_block}
{ui_hints_block}
{architecture_map_block}
{flyway_migration_block}
FILE SELECTION RULES (in priority order):
1. HIGH CONFIDENCE FILES & EXISTING REPOSITORY REUSE:
   Evaluate candidate files carefully. If a Spring Data repository or service already exists in CANDIDATE FILES for the target entity (e.g. ContractMemberRepository for contract members, ParticipatingMemberService for members), you MUST use a 'modify' task on the EXISTING repository/service rather than creating a new repository or service interface.
   CRITICAL: If a handler or service requires querying an entity (e.g. counting contracts for a user in a project via countByUserIdAndProjectId), and the query method does not exist in the candidate Repository interface, you MUST create a 'modify' task for that Repository (*Repository.java) to declare the required query method. Adding query methods to an existing repository is always preferred over creating new interfaces.
3. Files with STYLE role (*.scss, *.css) → only include if ticket is a CSS/styling ticket
4. LOCK_FILE / GENERATED files → NEVER include
5. DEPENDENCY & FRAMEWORK INTEGRITY:
   - NEVER add @PreAuthorize, @Secured, or Spring Security annotations unless the ticket explicitly specifies role-based authorization requirements and spring-security is in the project build file.
   - Do NOT modify endpoints that already exist in the codebase unless their signature or behavior is explicitly requested to change.
6. TEST FILE DETECTION & CO-INCLUSION:
   If a planned change to an implementation class (e.g. Saga Handler, Service, Controller) introduces a NEW constructor dependency, injected repository/service, or modifies behavior covered by existing tests, you MUST include the companion unit test file (*Test.java / *.spec.ts) as a 'modify' task to register the new mock (e.g. @Mock ContractMemberRepository) and update assertions.
   Feature implementation tickets that do not alter tested signatures do not require test changes.
6. EVIDENCE-BASED SELECTION: You MUST verify the candidate file contents in the CODE FACTS section before selecting them. Do NOT blindly accept files just because they are ranked highly. If a top-ranked file's contents do not match the ticket's functional requirements, IGNORE IT and look for better matches.d file's contents do not match the ticket's functional requirements, IGNORE IT and look for better matches.

DTO / MODEL / INTERFACE CO-INCLUSION RULE (Fix E — MINIMAL SCOPE):
Only include a DTO/Model/Interface file if the NEW property or method being added DOES NOT ALREADY EXIST in that file.
- DO include: if your service task calls `member.getNewField()` and `getNewField()` is absent from the entity class
- DO NOT include: if you are only using EXISTING fields (even via new methods) — the model is already correct
- DO NOT include: separate interface files (*.types.ts, *.interface.ts) just to "organize" new types — add inline to the component instead
- DO NOT include: shared model files (project.ts, saga.types.ts) unless a field STRUCTURALLY missing from them is required
- WHEN IN DOUBT: do NOT include the model file. Prefer adding the field inline in the component .ts.

ANGULAR COMPONENT CO-INCLUSION RULE (critical):
If a task modifies a `*.component.html` template to read a new property (e.g. `*ngIf="isExistingMember"`,
`{{ existingOrganizationName }}`), you MUST also include the sibling `*.component.ts` as a MODIFY task —
because the TypeScript controller is the ONLY place that declares that property.
The property name in the HTML MUST exactly match what the .ts declares.
The `.ts` task must come BEFORE the `.html` task in the tasks array.
NEVER use a property from a sub-object (e.g. `dmember.organizationName`) when the property lives
on the component class itself (e.g. `this.existingOrganizationName`). Use the component-level flag.

CHANGE INTENT & DEPENDENCY RULES (CRITICAL):
Files in CANDIDATE FILES may have explicit ChangeIntent markings:
1. IMPLEMENTATION TARGETS (ChangeIntent: MODIFY or CREATE):
   - Direct feature implementation files (e.g. component controllers, templates, specific domain classes).
2. REFERENCE DEPENDENCIES (ChangeIntent: READ_ONLY):
   - Existing services, APIs, models, and shared utilities (e.g. organization.service.ts, member.service.ts).
   - Relevant dependency ≠ writable file!
   - You MUST NOT create modify tasks for files marked READ_ONLY. Only inspect their public APIs to CALL them.

UI DATA-FLOW & SERVICE REUSE RULE:
If the ticket requires displaying data in the UI:
  1. FIRST, check if existing services or models ALREADY provide the data (e.g. memberService.members() with projectId already returns companyName).
     If an existing service method provides the data, CALL IT in the component (.ts) and bind to template (.html). Do NOT modify the service!
  2. Existing services are READ_ONLY unless the ticket title or description EXPLICITLY asks to alter a backend API or service interface.
  3. Never create modify tasks for a service just because the UI component calls it. Use the service as a read-only dependency.

DOMAIN OWNERSHIP GUIDANCE:
The context may include DOMAIN OWNERSHIP information describing the likely architectural owner of a business capability.
Use this information as architectural evidence, not as a substitute for repository investigation.
When selecting files:
1. Prefer files in the identified owning service when repository evidence supports that ownership.
2. Treat REFERENCE_ONLY classifications as a warning that the file may belong to another bounded context.
3. Verify the classification against actual implementations, imports, callers, dependencies, and data flow.
4. Do not modify a file solely because RAG considers it relevant.
5. Do not reject a file solely because the architecture model says another service is the preferred owner.
6. If repository evidence contradicts the architecture guidance, investigate the contradiction and use the stronger repository evidence.
7. Select the files that actually implement the ticket's behavior.

CAPABILITY REUSE RULE (CRITICAL — apply BEFORE creating any new method or API):
Before proposing a new method, service call, or API endpoint, you MUST check the
VERIFIED EVIDENCE section for methods that already provide the required functionality.

For each proposed new capability, follow this decision process:
1. REQUIRED: What capability does the ticket need? (e.g. "check membership")
2. EXISTING: Does a verified existing method already provide this? Check ACTUAL CODE.
3. PATH: Is the existing method connected to the relevant execution path?
   (e.g. AddMembersComponent → Service → API → Backend → Repository)
4. SUFFICIENT: Does the existing method's implementation satisfy the requirement?

Decision:
- If EXISTING + PATH VERIFIED + SUFFICIENT → REUSE the existing method.
  Create a task that CALLS the existing method, not a duplicate.
- If EXISTING + PARTIALLY sufficient → MODIFY the existing method to add
  the missing behavior. Do NOT create a parallel method.
- If EXISTING but PATH NOT VERIFIED → note this in selection_reason.
  State what relationship evidence is missing before assuming reuse.
- If NO existing method → create a new method. Explain in selection_reason
  exactly why no existing method was sufficient.

IMPORTANT: Do NOT assume a method is reusable just because its name sounds
related. You must verify from the actual code and relationship path that it
serves the same purpose in the same execution flow.

EVIDENCE AUTHORITY (CRITICAL — prevents inventing capabilities that already exist):
- VERIFIED EVIDENCE (relationship-grounded, progressively inspected code) is
  AUTHORITATIVE. If it contains a connected capability that satisfies the
  requirement, REUSE it — even if its name differs from the ticket's wording
  (e.g. reuse an existing members() flow instead of inventing
  checkProjectMembership()/isProjectMember()).
- Broad semantic/RAG candidates and keyword matches are SUPPLEMENTARY DISCOVERY
  ONLY. NEVER conclude a capability is absent — and NEVER propose CREATE_NEW —
  because RAG candidates (possibly from an unrelated service/module) do not
  contain it. RAG cannot override verified repository evidence.
- Only propose CREATE_NEW when the VERIFIED EVIDENCE itself shows no connected
  capability is sufficient. State exactly which verified capability you inspected
  and why it is insufficient.
- The BEHAVIORAL UNDERSTANDING block (if present) lists authoritative REUSE
  DECISIONS and CHANGE CANDIDATES vs READ-ONLY REFERENCES. Honor them: do NOT
  create a task for a READ-ONLY REFERENCE, and do NOT create a new capability
  where a REUSE decision is stated.

TASK DECOMPOSITION RULES:
- Each task modifies ONE file
{task_limit_rule} For version bumps: 2-4 tasks.
- MINIMUM SCOPE: if the ticket only touches Add Members modal, do NOT modify shared models, shared types, or test files.

⛔ NECESSITY GATE (CRITICAL — apply this test to EVERY file before including it):
For EACH candidate file, ask yourself this ONE question:
  "If I do NOT modify this file, will the bug still exist / will the feature still be incomplete?"
  - If YES → include it as a task (it is NECESSARY)
  - If NO  → DO NOT include it (it is merely RELATED, not necessary)

COMMON MISTAKES this gate prevents:
  ❌ Including an interface file (e.g. DeliverablesService.java) when you're only changing the implementation (DeliverablesServiceImpl.java) and the interface signature stays the same
  ❌ Including UI components that DISPLAY deliverables but don't VALIDATE the name
  ❌ Including service files that CALL the validation but don't CONTAIN the validation logic
  ❌ Including files just because they have "deliverable" or the ticket keyword in their name

EXAMPLE — Bug: "duplicate name error when saving deliverable":
  ✅ NECESSARY: The component that shows "Name has already been used" error (contains the validation)
  ✅ NECESSARY: The backend method that throws the validation exception (if backend fix needed)
  ❌ NOT NECESSARY: Other deliverable components that just list/display/submit deliverables
  ❌ NOT NECESSARY: The interface file if the method signature doesn't change
  ❌ NOT NECESSARY: Custom properties component (unrelated to name validation)

`allowed_methods` RULES (critical — wrong values here cause generation to fail silently):
- For MODIFY tasks that only change EXISTING methods: list the existing method name(s) to change.
- For MODIFY tasks that must ADD A NEW METHOD: list the NEW method name you expect the generator to create (e.g. `checkProjectMembership`). Do NOT list only existing methods when the task requires creating a new one — the validator will reject the output.
  BUT FIRST: check the VERIFIED EVIDENCE for an existing method that already does this.
  Only propose a NEW method name if no existing method is sufficient.
- For Angular component tasks that modify member-staging logic: the method to list is `onMemberAdd` or `onSave`, NOT `ngOnInit`. Use `ngOnInit` only when the initialization lifecycle hook itself must change.
- When unsure, list BOTH the existing method name AND the expected new method name.
- Never leave `allowed_methods` empty for a MODIFY task that is expected to add behavior.
- Tasks must be in dependency order (infrastructure first)
- `selection_reason` must explain WHY this file needs changing (not just "it exists")
  For files with VERIFIED EVIDENCE: selection_reason MUST reference the verified
  capabilities and explain whether they are being reused, modified, or why they
  are insufficient.
- `allowed_methods` must list SPECIFIC method or property names to change
- **rationale**: You MUST provide a substantive plan-level rationale explaining the overall strategy: what problem is being solved, why this specific shape of solution (why these specific 5 modifies + 1 create) was chosen, and how the tasks fit together. Do NOT write generic boilerplate ("implements the ticket"). This rationale will be used to mechanically gate the tasks.
  The rationale MUST address any verified existing capabilities and explain
  whether they are reused or why new capabilities are needed.

`edit_anchors` RULES (tells the generator WHERE to place code, not just WHAT to write):
- MANDATORY: You MUST provide `edit_anchors` for any task that adds new methods, properties, or modifies existing logic where placement matters.
- For ADD tasks (new method/property): specify `action: "add_method"`, `placement: "sibling_after"`, `anchor_method: "<existing method name>"` — this means "add as a sibling method right after the named method, NOT inside it."
- For MODIFY tasks: specify `action: "modify_method"`, `anchor_method: "<method being modified>"` — this scopes the edit to that method's body.
- For ADD PROPERTY: specify `action: "add_property"`, `placement: "class_body_top"` or `"after_last_property"`.
- The `anchor_method` field is used by the code generator to resolve ambiguous edits — if the same code snippet appears in multiple methods, `anchor_method` tells it which one to modify.

GROUNDED IMPLEMENTATION DECISION RULES:
- If a GROUNDED IMPLEMENTATION DECISION section is present in the context, you MUST obey it:
  - MODIFIABLE FILES → must become REQUIRED tasks (they are the canonical source of the change)
  - VERIFICATION TARGETS → must NOT become tasks (they are downstream consumers — verify, don't modify)
  - BLOCKED FILES → MUST NEVER be modified (generated code, locks, swagger)
  - IGNORED FILES → must NEVER appear as tasks
  - SUPPORTING FILES → may be included ONLY if needed for consistency (imports, interfaces)
- A VERIFICATION TARGET or BLOCKED file CANNOT be a REQUIRED task
- MODIFIABLE files take priority — create tasks for them first


LANGUAGE MAP:
- .ts, .html → typescript
- .js → javascript
- .java, .kt → java
- .py → python
- .sh, .bash, .ps1, .bat → shell
- .json, .yml, .yaml, .xml → json
- .scss, .css, .less → scss
- .sql → sql

CROSS-FILE RELATIONSHIPS (for multi-file changes):
When tasks share data or functionality across file boundaries, describe the INTENT:
- `cross_file_contract` on each task:
  - produces: what capability/data this task provides to other tasks
  - consumes: what capability/data this task needs from earlier tasks
For each cross-file relationship, specify:
  - capability: what crosses the boundary (e.g. "project membership check result")
  - data_shape: shape of the data (e.g. "boolean + organization name")
  - relationship_type: kind of relationship (data, trigger, precondition, consumer, side_effect, shared_type)
  - from_task: (for consumes only) which task provides this capability
Describe capabilities in natural language. Do NOT specify exact method names,
signatures, or import paths — those will be determined by the code generator
based on actual repository evidence.
You CAN still observe and reference actual repository patterns in your descriptions.

RESPOND with JSON ONLY:
{{
  "pattern": "Layered Architecture",
  "rationale": "Centralizing the app version into a single constant. Created one new constant file and modifying 5 consumer components to import this new canonical source instead of hardcoding the value.",
  "affected_modules": ["UI Layer"],
  "tasks": [
    {{
      "id": "task-1",
      "title": "Update version in app.component.ts",
      "description": "Change helpVersion from 260200 to 260300",
      "requirement_ids": ["REQ-1"],
      "file_path": "<exact-path-from-candidates>",
      "task_type": "modify",
      "language": "typescript",
      "selection_reason": "Stores the helpVersion constant that needs updating from 260200 to 260300",
      "target_class": "AppComponent",
      "allowed_methods": ["helpVersion"],
      "new_file_creation_allowed": false,
      "dependencies": [],
      "estimated_complexity": 1,
      "requires_testing": false,
      "edit_anchors": [
        {{
          "action": "modify_method",
          "method_name": "helpVersion",
          "placement": "inside",
          "anchor_method": "helpVersion",
          "description": "Modify the helpVersion constant value"
        }}
      ],
      "cross_file_contract": {{
        "produces": [{{
          "capability": "updated version constant",
          "data_shape": "integer constant",
          "relationship_type": "data"
        }}],
        "consumes": []
      }}
    }}
  ],
  "signature_blueprints": [],
  "api_changes": [],
  "database_changes": [],
  "external_dependencies": []
}}

NO markdown, NO explanations, ONLY valid JSON."""

    def _extract_content_snippet(self, *args, **kwargs) -> str:
        return ""

    def _build_user_prompt(
        self,
        ticket: ValueEdgeTicket,
        requirements: StructuredRequirements,
        retrieved_capabilities: Optional[Any] = None,
        workspace_path: Optional[str] = None,
        workspace_knowledge: Optional[Any] = None,
        technical_facts: Optional["TechnicalFacts"] = None,
        discovered_files: Optional[List[dict]] = None,
        blacklisted_files: Optional[List[str]] = None,
        code_facts: Optional[str] = None,
        verification_feedback: Optional[List[dict]] = None,
        verified_evidence: Optional[str] = None,
        **kwargs,
    ) -> str:
        
        # Format codebase context from RAG
        context_section = ""
        
        # ── Code Facts block (Contents of matched files) ──────────────────────
        code_facts_section = ""
        if code_facts:
            code_facts_section = f"\n=== CODE FACTS (Contents of matched files) ===\n{code_facts}\n"
        
        # ── B7: Grounded Evidence and Hypotheses ──────────────────────────────
        evidence_items = kwargs.get("evidence_items", [])
        investigation_hypotheses = kwargs.get("investigation_hypotheses", [])
        
        evidence_section = ""
        if evidence_items or investigation_hypotheses:
            evidence_section += "\n=== GROUNDED EVIDENCE (MUST READ) ===\n"
            evidence_section += "You MUST align your plan with this validated evidence.\n\n"
            
            if investigation_hypotheses:
                evidence_section += "Confirmed Hypotheses:\n"
                for h in investigation_hypotheses:
                    conf = getattr(h, "confidence", 0) if not isinstance(h, dict) else h.get("confidence", 0)
                    text = getattr(h, "hypothesis", "Unknown") if not isinstance(h, dict) else h.get("hypothesis", "Unknown")
                    if conf > 0.5:
                        evidence_section += f"- {text}\n"
                evidence_section += "\n"
            
            if evidence_items:
                evidence_section += "Promoted Evidence Snippets:\n"
                for idx, ev in enumerate(evidence_items, 1):
                    evidence_section += f"{idx}. File: {ev.file_path}\n"
                    if hasattr(ev, "content_snippet") and ev.content_snippet:
                        evidence_section += f"   Snippet: {ev.content_snippet}\n"
                    elif isinstance(ev, dict) and ev.get("content_snippet"):
                        evidence_section += f"   Snippet: {ev.get('content_snippet')}\n"
                evidence_section += "\n"
        
        # Format Discovered Files (Pipeline A Context)
        discovered_section = ""
        if discovered_files:
            try:
                sorted_discovered = sorted(
                    discovered_files,
                    key=lambda d: d.get("confidence", 0.0),
                    reverse=True,
                )
                candidates_for_prompt = []
                for idx, d in enumerate(sorted_discovered, 1):
                    # Translate signals into human-readable reasons for the LLM
                    reasons = []
                    _role = d.get("candidate_role", "unknown")
                    if _role != "unknown":
                        reasons.append(f"Role: {_role}")
                    
                    _lits = d.get("matched_literals", [])
                    if _lits:
                        reasons.append(f"Ticket literals found: {', '.join(_lits[:3])}")
                        
                    _sigs = d.get("signals", [])
                    if any("neo4j" in s.lower() or "graphify" in s.lower() for s in _sigs):
                        reasons.append("Graph match (Neo4j/Graphify)")
                    if any("semantic_brain" in s.lower() or "capability" in s.lower() for s in _sigs):
                        reasons.append("Semantic capability match")
                    if any("workspace" in s.lower() for s in _sigs):
                        reasons.append("Workspace knowledge match")
                        
                    _intent = d.get("change_intent")
                    if not _intent:
                        _norm_p = str(d.get("path", "")).lower().replace("\\", "/")
                        if any(_norm_p.endswith(suf) for suf in ("service.ts", "service.java", "service.py", "repository.java", "models.ts", "types.ts")):
                            _intent = "READ_ONLY"
                        else:
                            _intent = "MODIFY"
                    if _intent == "READ_ONLY":
                        reasons.append("Reference Dependency (ChangeIntent: READ_ONLY — do NOT modify)")

                    c = {
                        "rank": idx,
                        "path": d.get("path"),
                        "change_intent": _intent,
                        "confidence": round(float(d.get("confidence", 0.0)), 3),
                        "reasons": reasons
                    }
                    candidates_for_prompt.append(c)
                
                discovered_json = json.dumps(candidates_for_prompt, indent=2)
                discovered_section = (
                    "\nDISCOVERED CANDIDATE FILES (RANKED BY CONFIDENCE, YOU MUST SELECT FROM THESE):\n"
                    f"{discovered_json}\n"
                )

                # FIX 3.1 & 3.2: Read the actual file contents for the top candidates
                # so the planner isn't guessing structure from filenames alone.
                if workspace_path and candidates_for_prompt:
                    from pathlib import Path
                    ws_path = Path(workspace_path)
                    evidence_items = kwargs.get("evidence_items", [])
                    line_starts = {}
                    for item in evidence_items:
                        if getattr(item, "file_path", None) and getattr(item, "line_start", None):
                            line_starts[item.file_path] = item.line_start
                            
                    # FIX 3.1 & 3.2: Sort candidates by confidence (highest first)
                    sorted_candidates = sorted(
                        candidates_for_prompt, 
                        key=lambda c: c.get("confidence", 0.0), 
                        reverse=True
                    )
                    
                    # Set a hard global character budget for all candidate file contents (~15k tokens)
                    TOTAL_CHAR_BUDGET = 60000
                    chars_used = 0
                    
                    for c in sorted_candidates:
                        if chars_used >= TOTAL_CHAR_BUDGET:
                            logger.info(f"Reached planner context budget ({TOTAL_CHAR_BUDGET} chars). Dropping remaining {len(sorted_candidates)} lower-confidence candidates.")
                            break
                            
                        fp = c["path"]
                        try:
                            full_path = ws_path / fp
                            if full_path.exists() and full_path.is_file():
                                content = full_path.read_text(encoding="utf-8")
                                if "=== CODE FACTS" not in code_facts_section:
                                    code_facts_section += "\n=== CODE FACTS (Contents of matched files) ===\n"
                                
                                line_start = line_starts.get(fp)
                                snippet = ""
                                if line_start:
                                    lines = content.splitlines()
                                    start_idx = max(0, line_start - 20)
                                    end_idx = min(len(lines), line_start + 20)
                                    snippet = "\n".join(lines[start_idx:end_idx])
                                    snippet = f"\n--- {fp} (Lines {start_idx+1}-{end_idx}) ---\n{snippet}\n"
                                else:
                                    # Fallback: Top of the file (up to 3000 chars)
                                    snippet = f"\n--- {fp} ---\n{content[:3000]}\n"
                                
                                code_facts_section += snippet
                                chars_used += len(snippet)
                        except Exception as e:
                            logger.warning(f"Could not read {fp} for planner prompt: {e}")
            except Exception:
                pass
                
        if blacklisted_files:
            discovered_section += f"\nBLACKLISTED FILES (DO NOT USE):\n" + "\n".join(blacklisted_files) + "\n"

        # Format the Semantic Capabilities Context (Pipeline B Context)
        capabilities_json = "[]"
        if retrieved_capabilities:
            try:
                # We expect retrieved_capabilities to be a RetrievedCapabilities object
                # which has a list of relevant_nodes (CapabilityGraphNode).
                # We need to serialize them for the LLM.
                if hasattr(retrieved_capabilities, "relevant_nodes"):
                    # We just dump the nodes, which already follow the Capability -> Goal -> Files -> Change Surfaces structure
                    nodes_data = [node.model_dump() for node in retrieved_capabilities.relevant_nodes]
                    capabilities_json = json.dumps(nodes_data, indent=2)
                elif isinstance(retrieved_capabilities, list):
                    nodes_data = [node.model_dump() if hasattr(node, "model_dump") else node for node in retrieved_capabilities]
                    capabilities_json = json.dumps(nodes_data, indent=2)
            except Exception as e:
                logger.warning(f"Failed to serialize retrieved capabilities: {e}")

        # ── LOCKED files from TechnicalFacts (ticket-explicit + high-confidence evidence) ──
        must_use_section = ""
        if technical_facts and technical_facts.files:
            must_use_names = [f.name for f in technical_facts.files]
            must_use_section = (
                "\n=== LOCKED FILES (confidence ≥ 0.90 — you MUST include all of these) ===\n"
                + "\n".join(f"  • {fn}" for fn in must_use_names)
                + "\n\nFind the exact path for each filename in CANDIDATE FILES below and create a modify task."
                "\nDo NOT skip any of them. If you skip one, explain why in another task's selection_reason.\n"
            )

        # ── Workspace knowledge block ─────────────────────────────────────────
        workspace_block = ""
        if workspace_knowledge and hasattr(workspace_knowledge, "to_prompt_block"):
            workspace_block = f"\n{workspace_knowledge.to_prompt_block()}\n"

        # ── Verification Feedback ─────────────────────────────────────────────
        feedback_section = ""
        if verification_feedback:
            feedback_text = "\n=== SEMANTIC VERIFICATION FEEDBACK (LLM-verified file relevance) ===\n"
            for fb in verification_feedback:
                v_status = fb.get("verification_status", "completed")

                if v_status != "completed":
                    # Infrastructure failure — this is NOT evidence about the file
                    feedback_text += (
                        f"- UNVERIFIED: {fb.get('file_path')} "
                        f"(verification infrastructure did not complete — treat as valid candidate)\n"
                    )
                elif fb.get("decision") in ("REJECT", "exclude"):
                    feedback_text += f"- EXCLUDED: {fb.get('file_path')} | Reason: {fb.get('reason')}\n"
                elif fb.get("decision") in ("include",):
                    feedback_text += f"- VERIFIED: {fb.get('file_path')} | Reason: {fb.get('reason')}\n"
            if "EXCLUDED:" in feedback_text or "VERIFIED:" in feedback_text or "UNVERIFIED:" in feedback_text:
                feedback_section = feedback_text

        # ── Interactive Loop Context ─────────────────────────────────────────
        loop_context_section = ""
        previous_tasks = kwargs.get("previous_tasks")
        if previous_tasks:
            loop_context_section = "\n=== PREVIOUS PLANNING DECISIONS ===\n"
            loop_context_section += "You already started planning. Here are the tasks you decided on so far:\n"
            for t in previous_tasks:
                loop_context_section += f" - [{t.task_type.value}] {t.file_path}: {t.description}\n"
            loop_context_section += "\nContinue building your plan. You do not need to output these tasks again unless you are changing them.\n"

        # ── Strategy + Skills Context (P1/P2) ───────────────────────────────
        strategy_section = ""
        strategy_decision = kwargs.get("strategy_decision")
        active_skills = kwargs.get("active_skills") or []
        skill_guidance = kwargs.get("skill_guidance") or ""
        if strategy_decision:
            strategy_section += "\n=== STRATEGY DECISION (MANDATORY) ===\n"
            strategy_section += f"Category: {getattr(strategy_decision, 'category', '')}\n"
            strategy_section += f"Profile: {getattr(strategy_decision, 'strategy_profile', '')}\n"
            req_ev = getattr(strategy_decision, "required_evidence", []) or []
            req_val = getattr(strategy_decision, "required_validations", []) or []
            stop_conds = getattr(strategy_decision, "stop_conditions", []) or []
            skill_tags = getattr(strategy_decision, "skill_tags", []) or []
            if req_ev:
                strategy_section += "Required evidence:\n" + "\n".join(f"- {x}" for x in req_ev[:6]) + "\n"
            if req_val:
                strategy_section += "Required validations:\n" + "\n".join(f"- {x}" for x in req_val[:6]) + "\n"
            if stop_conds:
                strategy_section += "Stop conditions:\n" + "\n".join(f"- {x}" for x in stop_conds[:6]) + "\n"
            if skill_tags:
                strategy_section += "Skill tags: " + ", ".join(skill_tags[:8]) + "\n"

        if active_skills or skill_guidance:
            strategy_section += "\n=== ACTIVE SKILLS GUIDANCE ===\n"
            if skill_guidance:
                strategy_section += skill_guidance + "\n"
            else:
                for s in active_skills[:6]:
                    strategy_section += f"- {s.get('id', 'unknown')}: {s.get('title', '')}\n"

        # ── Validation Failure Reason (from gating) ───────────────────────────
        validation_failure_reason = kwargs.get("validation_failure_reason")
        validation_failure_section = ""
        if validation_failure_reason:
            validation_failure_section = (
                "\n> [!WARNING]\n"
                "> PREVIOUS PLAN FAILED VALIDATION\n"
                "> Your previous architectural plan was rejected by the safety gates for the following reason:\n"
                f"> {validation_failure_reason}\n"
                ">\n"
                "> You MUST fix this issue. If a specific file was rejected for an architectural boundary violation, "
                "you must either find a valid way to implement the logic without crossing that boundary, OR choose a different file entirely.\n"
            )

        # ── Phase 5: Verified Evidence section ─────────────────────────────
        # This section contains structurally inspected capabilities with actual
        # source code, facts, questions, and relationship provenance. It appears
        # BEFORE code_facts so the planner sees structured evidence first.
        verified_evidence_section = ""
        if verified_evidence:
            verified_evidence_section = (
                "\n=== VERIFIED EVIDENCE (Inspected Capabilities — READ BEFORE PLANNING) ===\n"
                "The following files were progressively inspected during evidence collection.\n"
                "For each file, you can see the actual class structure, methods, relationships,\n"
                "source code, and verified facts. Use this to determine whether existing\n"
                "capabilities satisfy the ticket's requirements BEFORE proposing new methods.\n"
                "\n"
                "KEY: 'RELEVANT TO TICKET' marks methods the evidence agent identified as\n"
                "potentially relevant. 'ACTUAL CODE' shows the real implementation.\n"
                "'Discovery reason' explains how this file was found (relationship path).\n"
                "'Open questions' indicate unverified aspects you should consider.\n"
                f"{verified_evidence}\n"
                "=== END VERIFIED EVIDENCE ===\n"
            )

        return f"""
TICKET: {ticket.ticket_id}
TITLE: {ticket.title}
PRIORITY: {ticket.priority.value}
{must_use_section}{workspace_block}
REQUIREMENTS:
Functional: {self._format_list(requirements.functional_requirements)}
Technical:  {self._format_list(requirements.technical_requirements)}
Affected:   {self._format_list(requirements.affected_components)}
{verified_evidence_section}
{code_facts_section}
{discovered_section}
{feedback_section}
{strategy_section}
{loop_context_section}
{validation_failure_section}
Now produce the JSON plan. Select from CANDIDATE FILES only. Must_use files must appear in tasks.
Before creating any new method or API, check the VERIFIED EVIDENCE section for existing capabilities.
        """.strip()
    
    def _format_list(self, items: List[str]) -> str:
        """Format list items with bullets and numbered [REQ-x] identifiers if absent"""
        if not items:
            return "- None"
        out = []
        for idx, item in enumerate(items, 1):
            stripped = item.strip()
            if not re.match(r"^\[REQ-\d+\]", stripped, re.IGNORECASE):
                out.append(f"- [REQ-{idx}] {stripped}")
            else:
                out.append(f"- {stripped}")
        return "\n".join(out)

    # =========================================================================
    # P0: PLAN SELF-REVIEW (inside the same node, not a separate graph node)
    # =========================================================================

    def self_review_plan(
        self,
        plan: ArchitecturalPlan,
        ticket: ValueEdgeTicket,
        requirements: StructuredRequirements,
        evidence_items: Optional[list] = None,
        discovered_files: Optional[list] = None,
        grounded_decision: Optional[Any] = None,
    ) -> PlanReviewResult:
        """
        Self-review: the planner critiques its own plan point-by-point.

        Uses a cheaper model (gemini-2.0-flash) to evaluate:
        1. Requirements coverage — does the plan address all requirements?
        2. Evidence alignment — are selected files backed by evidence?
        3. Scope — is the plan too narrow or too broad?
        4. Missing files — are there obvious files that should be included?

        Returns PlanReviewResult. If rejected, the plan_node will regenerate.
        """
        from ticket_to_code.runtime.llm_factory import LLMRegistry
        from ticket_to_code.llm_utils import llm_invoke
        from langchain_core.messages import HumanMessage
        import re

        # Build plan summary
        task_lines = []
        for t in plan.tasks:
            task_lines.append(
                f"  - [{t.task_type.value}] {t.file_path} "
                f"(class={t.target_class or '?'}, method={t.target_method or '?'})"
            )

        # Build requirements summary
        func_reqs = "\n".join(f"  - {r}" for r in requirements.functional_requirements[:10])
        tech_reqs = "\n".join(f"  - {r}" for r in requirements.technical_requirements[:5])

        # Build evidence summary
        evidence_summary = ""
        if evidence_items:
            ev_files = set()
            for e in evidence_items[:20]:
                fp = getattr(e, "file_path", "")
                if fp:
                    ev_files.add(fp)
            evidence_summary = f"Evidence-backed files ({len(ev_files)}):\n" + "\n".join(
                f"  - {f}" for f in sorted(ev_files)[:15]
            )

        # Build grounded decision summary
        grounded_summary = ""
        if grounded_decision:
            grounded_summary = "\n=== GROUNDED IMPLEMENTATION DECISION ===\n"
            if getattr(grounded_decision, "modifiable_files", None):
                grounded_summary += f"MODIFIABLE OWNERS (MUST have tasks): {', '.join(grounded_decision.modifiable_files)}\n"
            if getattr(grounded_decision, "verification_targets", None):
                grounded_summary += f"VERIFICATION TARGETS (MUST NOT have tasks): {', '.join(grounded_decision.verification_targets)}\n"
            if getattr(grounded_decision, "ignored_files", None):
                grounded_summary += f"IGNORED FILES (MUST NOT have tasks): {', '.join(grounded_decision.ignored_files)}\n"
            if getattr(grounded_decision, "blocked_files", None):
                grounded_summary += f"BLOCKED FILES (MUST NOT have tasks): {', '.join(grounded_decision.blocked_files)}\n"
            grounded_summary += "========================================\n"

        prompt = f"""You are a Plan Reviewer for an autonomous code modification engine.

TICKET: {ticket.title}
DESCRIPTION: {ticket.description[:400]}

FUNCTIONAL REQUIREMENTS:
{func_reqs}

TECHNICAL REQUIREMENTS:
{tech_reqs}

PROPOSED PLAN ({len(plan.tasks)} tasks):
{chr(10).join(task_lines)}

{evidence_summary}
{grounded_summary}

TASK: Review this plan point-by-point. For each point, evaluate:

1. REQUIREMENTS COVERAGE
   - For each functional requirement, state whether the plan addresses it
   - List any requirements that are NOT addressed

2. EVIDENCE ALIGNMENT
   - For each planned file, state whether it is backed by evidence
   - Flag any planned files that have NO evidence support

3. GROUNDED DECISION ALIGNMENT
   - Check if any "VERIFICATION TARGET", "IGNORED", or "BLOCKED" files were incorrectly scheduled as tasks
   - Check if any "MODIFIABLE OWNER" files are missing from the plan

4. SCOPE ASSESSMENT
   - Is the plan too narrow (missing obvious files)?
   - Is the plan too broad (including unnecessary files)?
   - Is the scope appropriate?

5. DECISION
   - approved=true if the plan is good enough to proceed
   - approved=false if critical requirements are missed, files are wrong, or grounded decisions are violated

Rules:
- REJECT IMMEDIATELY (approved=false) if the plan includes tasks for VERIFICATION TARGET, IGNORED, or BLOCKED files
- REJECT IMMEDIATELY if the plan is missing tasks for MODIFIABLE OWNER files
- Otherwise, bias toward approval if the plan covers >70% of requirements
- Point out improvements but don't reject for non-critical issues

Return your assessment as a JSON object matching this schema:
{{
  "approved": bool,
  "review_points": ["string", "string"],
  "missing_coverage": ["string", "string"],
  "scope_assessment": "string",
  "rejection_reason": "string" (or null if approved),
  "improvement_suggestions": ["string", "string"]
}}

ONLY output valid JSON. DO NOT wrap it in markdown block quotes."""

        try:
            llm = LLMRegistry.get_llm(assistant=False)
            response = llm_invoke(llm, [HumanMessage(content=prompt)])
            text = response.content
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group(0)
            review = PlanReviewResult.model_validate_json(text)
            logger.info(
                f"  [PlanSelfReview] approved={review.approved} "
                f"scope={review.scope_assessment[:60]}"
            )
            for pt in review.review_points[:5]:
                logger.info(f"    • {pt}")
            if review.missing_coverage:
                for mc in review.missing_coverage[:3]:
                    logger.info(f"    ✗ Missing: {mc}")
            return review

        except Exception as exc:
            logger.warning(f"  [PlanSelfReview] LLM call failed: {exc} — auto-approving")
            return PlanReviewResult(
                approved=True,
                review_points=[f"Self-review failed ({exc}), auto-approved"],
                scope_assessment="unknown (review failed)",
            )
    
    def _parse_response(self, response_content: str) -> ArchitecturalPlan:
        """
        Parse LLM response into ArchitecturalPlan.
        
        Args:
            response_content: Raw LLM response
            
        Returns:
            Validated ArchitecturalPlan object
        """
        # Clean response
        cleaned = response_content.strip()
        
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Fallback: Try to repair truncated JSON by appending closing braces
            logger.warning(f"  ⚠️ Standard JSON parse failed ({e}). Attempting to repair truncated JSON string.")
            repaired = cleaned
            
            # Simple repair: if it ends mid-string, close the string
            if repaired.rfind('"') % 2 != 0:
                repaired += '"'
                
            # Close arrays and objects based on open braces count
            open_braces = repaired.count('{') - repaired.count('}')
            open_brackets = repaired.count('[') - repaired.count(']')
            
            if open_braces > 0 or open_brackets > 0:
                # Naive closing strategy: just append closing brackets in reverse order they typically appear at the end
                if open_braces > open_brackets:
                    repaired += '}' * (open_braces - open_brackets)
                    repaired += ']' * open_brackets
                    repaired += '}' * open_brackets
                else:
                    repaired += '}' * open_braces
                    repaired += ']' * open_brackets
                    repaired += '}' 
            
            # Use partial regex extraction if that fails
            try:
                data = json.loads(repaired)
                logger.info(f"  ✅ Basic JSON repair succeeded.")
            except Exception:
                # If naive repair fails, just extract the tasks array using regex
                import re
                logger.warning("  ⚠️ Basic repair failed, attempting regex extraction of tasks array.")
                tasks = []
                task_matches = re.findall(r'\{\s*"id"\s*:\s*[^}]+\}', cleaned)
                if not task_matches:
                    task_matches = re.findall(r'\{\s*"task_type"\s*:\s*[^}]+(?:edit_anchors|file_path)[^}]*\}', cleaned, re.DOTALL)
                
                for match in task_matches:
                    try:
                        tasks.append(json.loads(match))
                    except:
                        pass
                
                if tasks:
                    data = {
                        "pattern": "Layered Architecture",
                        "rationale": "Recovered via regex",
                        "affected_modules": [],
                        "tasks": tasks
                    }
                    logger.info(f"  ✅ Regex recovery salvaged {len(tasks)} tasks.")
                else:
                    logger.error(f"Failed to parse or repair JSON response: {cleaned}")
                    raise ValueError(f"Invalid JSON response from LLM: {e}")

        # ── Normalize pattern ──────────────────────────────────────────────
        _VALID_PATTERNS = {"MVC", "MVVM", "Feature-Sliced Design",
                           "Layered Architecture", "Microservices",
                           "Clean Architecture", "Monolithic"}
        _PATTERN_KEYWORDS = [
            ("microservice", "Microservices"),
            ("layered", "Layered Architecture"),
            ("clean", "Clean Architecture"),
            ("mvc", "MVC"),
            ("mvvm", "MVVM"),
            ("feature", "Feature-Sliced Design"),
            ("monolith", "Monolithic"),
            ("component", "Layered Architecture"),
            ("frontend", "MVC"),
        ]
        if "pattern" in data:
            raw = str(data["pattern"]).strip()
            if raw not in _VALID_PATTERNS:
                low = raw.lower()
                matched = next((v for k, v in _PATTERN_KEYWORDS if k in low), "Microservices")
                logger.warning(f"Unknown pattern '{raw}' → '{matched}'")
                data["pattern"] = matched

        # ── Normalize task_type for every task ─────────────────────────────
        _VALID_TASK_TYPES = {"read_only", "create", "modify", "delete", "refactor"}
        _TASK_TYPE_MAP = {
            # Analysis verbs → read_only (no file writes ever)
            "locate": "read_only", "find": "read_only", "identify": "read_only",
            "inspect": "read_only", "check": "read_only", "verify": "read_only",
            "analyse": "read_only", "analyze": "read_only", "read": "read_only",
            "extract": "read_only", "gather": "read_only", "review": "read_only",
            # Write verbs
            "test": "read_only",  # FIX 2B: spec/test tasks must never silently become writable modify
            "update": "modify", "change": "modify", "edit": "modify",
            "add": "create", "new": "create", "implement": "create",
            "remove": "delete", "drop": "delete",
            "refactor": "refactor", "clean": "refactor", "improve": "refactor",
        }
        for i, task in enumerate(data.get("tasks", [])):
            if not isinstance(task, dict):
                continue
            if not task.get("id"):
                task["id"] = f"task-{i+1}"
            if not task.get("title"):
                task["title"] = task.get("description") or task.get("selection_reason") or f"Task {task['id']}"
            if not task.get("description"):
                task["description"] = task.get("title") or f"Execute {task['id']}"

            req_ids = task.get("requirement_ids", [])
            if isinstance(req_ids, str):
                req_ids = [r.strip() for r in req_ids.split(",") if r.strip()]
            elif not isinstance(req_ids, list):
                req_ids = []
            task["requirement_ids"] = [str(r) for r in req_ids]

            fp = task.get("file_path") or task.get("path") or task.get("target_file") or task.get("file")
            if hasattr(fp, "file_path"):
                task["file_path"] = str(fp.file_path)
            elif hasattr(fp, "path"):
                task["file_path"] = str(fp.path)
            elif isinstance(fp, dict):
                task["file_path"] = str(fp.get("file_path") or fp.get("path") or "")
            elif fp is not None:
                task["file_path"] = str(fp)

            raw_type = str(task.get("task_type", "modify")).strip().lower()
            if raw_type not in _VALID_TASK_TYPES:
                task["task_type"] = _TASK_TYPE_MAP.get(raw_type, "modify")
                logger.warning(f"Unknown task_type '{raw_type}' → '{task['task_type']}'")
            else:
                task["task_type"] = raw_type
        # ── Normalize language for every task ──────────────────────────────
        _VALID_LANGS = {"csharp", "csharp_test", "typescript", "javascript",
                        "python", "java", "go", "xml", "json", "shell", "scss",
                        "html", "yaml", "properties", "sql"}
        _LANG_MAP = {
            "htm": "html", "css": "scss", "less": "scss", "sass": "scss",
            "jsx": "javascript", "tsx": "typescript", "kt": "java",
            "kotlin": "java", "groovy": "java", "yml": "yaml",
            "shell": "shell", "bash": "shell", "sh": "shell",
            "ps1": "shell", "bat": "shell", "cmd": "shell",
            "c#": "csharp", "cs": "csharp", "env": "properties",
            "ts": "typescript", "js": "javascript", "py": "python",
            "scss": "scss", "sql": "sql",
        }
        for task in data.get("tasks", []):
            raw_lang = str(task.get("language", "java")).strip().lower()
            if raw_lang not in _VALID_LANGS:
                task["language"] = _LANG_MAP.get(raw_lang, "java")
                logger.warning(f"Unknown language '{raw_lang}' → '{task['language']}'")
            else:
                # Always write back the lowercase form — LLM may return 'Java' or 'TypeScript'
                task["language"] = raw_lang
        # ── Normalize api_changes ──────────────────────────────────────────
        def _coerce_json_object(value: Any) -> Optional[dict[str, Any]]:
            if value is None:
                return None
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                txt = value.strip()
                if not txt:
                    return None
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
                return {"value": value}
            return {"value": str(value)}

        norm_api_changes = []
        for item in data.get("api_changes", []):
            if isinstance(item, str):
                norm_api_changes.append({
                    "endpoint": "custom",
                    "method": "OTHER",
                    "description": item,
                    "required": True,
                })
            elif isinstance(item, dict):
                norm_item = dict(item)
                if "endpoint" not in norm_item:
                    norm_item["endpoint"] = "custom"
                if "method" not in norm_item:
                    norm_item["method"] = "OTHER"
                if "description" not in norm_item:
                    norm_item["description"] = str(item)
                norm_item["request_body"] = _coerce_json_object(norm_item.get("request_body"))
                norm_item["response_body"] = _coerce_json_object(norm_item.get("response_body"))
                norm_api_changes.append(norm_item)
        data["api_changes"] = norm_api_changes

        # ── Normalize database_changes ─────────────────────────────────────
        norm_db_changes = []
        for item in data.get("database_changes", []):
            if isinstance(item, str):
                norm_db_changes.append({
                    "table_name": "unknown",
                    "change_type": "ALTER",
                    "migration_script": item,
                })
            elif isinstance(item, dict):
                norm_item = dict(item)
                if "table_name" not in norm_item:
                    norm_item["table_name"] = "unknown"
                if "change_type" not in norm_item:
                    norm_item["change_type"] = "ALTER"
                norm_db_changes.append(norm_item)
        data["database_changes"] = norm_db_changes

        # ── Component 4: edit_anchors validation ───────────────────────────
        for i, t in enumerate(data.get("tasks", [])):
            if t.get("task_type") == "modify" and not t.get("edit_anchors"):
                raise ValueError(
                    f"Task '{t.get('title', f'task-{i}')}' is a 'modify' task but is missing 'edit_anchors'. "
                    f"You MUST provide edit_anchors for modify tasks to tell the generator where to place code."
                )

        # ── Normalize cross_file_contract on each task (backward-compat) ───
        for t in data.get("tasks", []):
            contract = t.get("cross_file_contract")
            if contract is None:
                # Older plan format — no cross-file contract emitted
                t["cross_file_contract"] = None
            elif isinstance(contract, dict):
                # Normalize semantic blueprints inside the contract
                for key in ("produces", "consumes"):
                    items = contract.get(key, [])
                    if not isinstance(items, list):
                        contract[key] = []
                    else:
                        # Backward-compat: if items use old SignatureBlueprint format,
                        # convert to SemanticBlueprint format
                        normalized = []
                        for item in items:
                            if isinstance(item, dict):
                                if "capability" in item and item["capability"]:
                                    # Already semantic format
                                    norm_b = dict(item)
                                    if "relationship_type" not in norm_b or not norm_b["relationship_type"]:
                                        norm_b["relationship_type"] = "data"
                                    normalized.append(norm_b)
                                elif "symbol_name" in item:
                                    # Old exact-signature format → convert to semantic
                                    owner = item.get("owner_class", "")
                                    sym = item.get("symbol_name", "")
                                    sig = item.get("signature", "")
                                    normalized.append({
                                        "capability": f"{owner}.{sym}" if owner else sym or "default_capability",
                                        "data_shape": sig or "",
                                        "from_task": item.get("created_by_task", ""),
                                        "relationship_type": item.get("relationship_type", "data") or "data",
                                    })
                                    logger.info(
                                        f"  [Planner] Converted legacy blueprint to semantic: "
                                        f"{owner}.{sym} → capability"
                                    )
                                else:
                                    norm_b = dict(item)
                                    if "capacity" in norm_b and norm_b["capacity"]:
                                        norm_b["capability"] = norm_b.pop("capacity")
                                    elif "name" in norm_b and norm_b["name"]:
                                        norm_b["capability"] = norm_b.get("name")
                                    elif "description" in norm_b and norm_b["description"]:
                                        norm_b["capability"] = norm_b.get("description")
                                    else:
                                        norm_b["capability"] = "default_capability"
                                    if "relationship_type" not in norm_b or not norm_b["relationship_type"]:
                                        norm_b["relationship_type"] = "data"
                                    normalized.append(norm_b)
                            else:
                                normalized.append(item)
                        contract[key] = normalized

        # ── Normalize top-level signature_blueprints (backward-compat) ─────
        # The planner no longer produces these (semantic planner uses
        # SemanticBlueprint in cross_file_contract instead), but old plans
        # may still contain them.
        raw_blueprints = data.get("signature_blueprints", [])
        if not isinstance(raw_blueprints, list):
            raw_blueprints = []
        data["signature_blueprints"] = raw_blueprints
        if raw_blueprints:
            logger.info(f"  [Planner] Emitted {len(raw_blueprints)} legacy signature blueprint(s)")

        # ── Validate with Pydantic (with final safety net) ─────────────────
        try:
            return ArchitecturalPlan.model_validate(data)
        except Exception as e:
            logger.error(f"Pydantic validation failed after normalization: {e}")
            logger.error(f"Data was: {data}")
            # Try to salvage valid tasks so planning does not fail completely
            salvaged_tasks = []
            for i, t in enumerate(data.get("tasks", [])):
                try:
                    if isinstance(t, dict):
                        if not t.get("id"):
                            t["id"] = f"task-{i+1}"
                        if not t.get("title"):
                            t["title"] = t.get("description") or t.get("selection_reason") or f"Task {t['id']}"
                        if not t.get("description"):
                            t["description"] = t.get("title") or f"Execute {t['id']}"
                        if isinstance(t, dict):
                            if not t.get("file_path"):
                                t["file_path"] = t.get("path") or t.get("target_file") or t.get("file") or ""
                            if not t.get("task_type"):
                                t["task_type"] = "modify"
                        salvaged_tasks.append(DevelopmentTask.model_validate(t))
                    elif isinstance(t, DevelopmentTask):
                        salvaged_tasks.append(t)
                except Exception as te:
                    logger.warning(f"Could not salvage task {t}: {te}")
            return ArchitecturalPlan(
                pattern=ArchitecturalPattern.MICROSERVICES,
                rationale=data.get("rationale", ""),
                affected_modules=data.get("affected_modules", ["unknown"]),
                tasks=salvaged_tasks,
                api_changes=[],
                database_changes=[],
            )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_plan(
    ticket: ValueEdgeTicket,
    requirements: StructuredRequirements
) -> ArchitecturalPlan:
    """
    Convenience function to create architectural plan.
    
    Args:
        ticket: ValueEdge ticket
        requirements: Structured requirements
        
    Returns:
        Architectural plan
    """
    agent = PlanningAgent()
    return agent.create_plan(ticket, requirements)
