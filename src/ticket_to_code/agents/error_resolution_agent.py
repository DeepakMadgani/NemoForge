import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ticket_to_code.llm_utils import llm_invoke
from ticket_to_code.runtime.limits import BUILD_OUTPUT_MAX_CHARS, TOOL_OUTPUT_MAX_CHARS, truncate
from ticket_to_code.agents.diagnostic_context_builder import DiagnosticContextBuilder, DiagnosticContext

if TYPE_CHECKING:
    from ticket_to_code.memory.ticket_execution_context import TicketExecutionContext
    from ticket_to_code.agents.component_structure_provider import ComponentStructureProvider
    from ticket_to_code.agents.symbol_resolver import SymbolResolver
    from ticket_to_code.agents.relationship_analyzer import RelationshipAnalyzer
    from ticket_to_code.agents.fix_localizer import FixLocalizer
    from ticket_to_code.agents.fix_hypothesis_builder import FixHypothesisBuilder

logger = logging.getLogger(__name__)

class ErrorResolutionAgent:
    """
    An agentic loop that resolves build/compiler errors.
    Instead of hardcoded regexes, it gives the LLM tools to investigate and fix errors dynamically.
    """

    def __init__(self, llm, workspace_path: Path, sqlite_store=None):
        self.llm = llm
        self.workspace_path = workspace_path
        self.sqlite_store = sqlite_store
        
        # Keep track of which files we've modified during this loop
        self.modified_files: dict[str, str] = {}
        self._file_case_map: dict[str, str] = {}
        # Structured audit trail for diagnostic repair decisions
        self.audit_trail: list[Any] = []
        # Pre-existing files the USER explicitly authorized us to fix this run.
        # Normalized (lowercased, forward-slashed). Empty = protection fully on.
        self.authorized_pre_existing: set[str] = set()
        # TicketScopeProof instance enforcing that compiler diagnostics cannot expand write authorization
        self.scope_proof: Any = None
        # Structured error-to-method map for patch target validation
        # Maps normalized file path → list of (error_line, method_name, method_start, method_end)
        self._error_method_map: dict[str, list[tuple[int, str, int, int]]] = {}

        # ── v2 Context (set by fix_build_errors_node before calling resolve_errors) ──
        # Pre-edit file snapshots (from state["original_file_contents"])
        self.pre_edit_snapshots: dict[str, str] = {}
        # Set of file paths (lowercase, forward-slashed) that our workflow modified
        self.our_modified_files: set[str] = set()
        # Original ticket description so the agent understands intent
        self.ticket_description: str = ""
        # Root directory for running tsc/compiler (e.g. the Angular project root)
        self.compile_root: Optional[Path] = None
        # Which compiler to use: "tsc" (default, includes Angular AOT),
        # "gradle", or "maven"
        self.compile_tool: str = "tsc"

        # ── v3 Context: Rich pipeline context from earlier phases ──────────────
        # TicketExecutionContext accumulates knowledge from investigation through planning.
        # Provides: functional requirements, planned file roles, component groups,
        # data flow gaps, grounded understanding, fix history.
        self.exec_ctx: Optional["TicketExecutionContext"] = None
        # ComponentStructureProvider wraps DataFlowTracer + LSP + ComponentGroups
        # to give language-agnostic file relationship queries.
        self.component_provider: Optional["ComponentStructureProvider"] = None

        # ── v4 Context: Diagnostic Intelligence Pipeline ──────────────────
        # These are wired by fix_build_errors_node in workflow.py.
        # When present, the agent receives structured evidence instead of
        # raw error text — enabling engineering reasoning, not file guessing.
        self.symbol_resolver: Optional["SymbolResolver"] = None
        self.relationship_analyzer: Optional["RelationshipAnalyzer"] = None
        self.fix_localizer: Optional["FixLocalizer"] = None
        self.hypothesis_builder: Optional["FixHypothesisBuilder"] = None

        # ── Version-aware editing (hash-based staleness detection) ────────
        self._file_versions: dict[str, str] = {}  # normalized_path → content_hash

        # ── Semantic Contract (from code generation) ──────────────────────
        self.semantic_contract: Optional["SemanticContract"] = None

        # ── Oscillation Detector ──────────────────────────────────────────
        from ticket_to_code.verification.oscillation_detector import OscillationDetector
        self.oscillation_detector = OscillationDetector(max_recovery_turns=3)

    @staticmethod
    def _consumer_guard_blocks(target_consumer_file: Optional[str], requested_path: str) -> bool:
        """True when an edit to ``requested_path`` must be blocked because the run is
        restricted to the single authorized consumer file ``target_consumer_file``.

        Returns False (edit allowed) when no consumer restriction is active or when the
        requested path is the consumer file (exact, or suffix-match in either direction).
        """
        if not target_consumer_file:
            return False
        norm_target = target_consumer_file.replace("\\", "/").lower()
        norm_req = (requested_path or "").replace("\\", "/").lower()
        if norm_req == norm_target:
            return False
        if norm_req.endswith("/" + norm_target) or norm_target.endswith("/" + norm_req):
            return False
        return True

    def resolve_errors(self, build_errors: list[str], max_iter: int = 0) -> dict[str, str]:
        """
        Takes raw compiler errors, extracts file contexts, and loops with the LLM until FINISHED.
        Returns a dict mapping relative file paths to their new fixed content.

        max_iter=0 means auto-scale based on error count.
        """
        # ── Deterministic pre-repair: strip hallucinated uninstalled security annotations ──
        if any("org.springframework.security" in str(e) or "PreAuthorize" in str(e) for e in build_errors):
            for fp, snap in list(self.modified_files.items()):
                if "import org.springframework.security" in snap or "@PreAuthorize" in snap:
                    cleaned = re.sub(r"import\s+org\.springframework\.security.*\n?", "", snap)
                    cleaned = re.sub(r"@PreAuthorize\([^\)]*\)\s*", "", cleaned)
                    self.modified_files[fp] = cleaned
                    logger.info(f"  🛡️ Deterministic pre-repair: stripped uninstalled Spring Security from {fp}")

        # ── Dynamic iteration count (Market AI IDE standard: bounded fail-fast) ──
        if max_iter <= 0:
            max_iter = max(min(max(2, len(build_errors)), 5), 4)

        logger.info("\n" + "=" * 70)
        logger.info(" ERROR RESOLUTION AGENT v2: Compile-in-the-loop fix")
        logger.info(f"   Errors Count: {len(build_errors)}")
        logger.info(f"   Max Iterations: {max_iter}")
        logger.info(f"   Pre-edit snapshots: {len(self.pre_edit_snapshots)} file(s)")
        logger.info(f"   Our modified files: {len(self.our_modified_files)} file(s)")
        logger.info(f"   Compile root: {self.compile_root}")
        logger.info("=" * 70)

        error_text = "\n".join(build_errors)

        # 1. Target-First Diagnostic Context (AST/symbol resolution, minimal dependencies)
        diag_builder = DiagnosticContextBuilder()
        diag_contexts: list[DiagnosticContext] = []
        error_locations = self._parse_error_locations(error_text)
        preloaded_files = {}

        for raw_path, line_num in error_locations[:8]:
            fp, content = self._resolve_and_read(raw_path)
            if content:
                preloaded_files[fp] = content
                err_msg = ""
                for el in build_errors:
                    if str(line_num) in el and (Path(raw_path).name in el or raw_path in el):
                        err_msg = el.strip()
                        break
                ctx = diag_builder.build_context(fp, content, line_num, 0, err_msg)
                diag_contexts.append(ctx)
                diag_builder.update_file_hash(fp, content)
                self._file_versions[fp.replace('\\', '/').lower()] = diag_builder.compute_hash(content)

        # Fallback: if no files preloaded, inspect our modified files against error_text
        if not preloaded_files and self.our_modified_files:
            for mod_fp in sorted(self.our_modified_files):
                mod_name = Path(mod_fp).name
                if mod_name.lower() in error_text.lower() or mod_fp.lower() in error_text.lower():
                    fp, content = self._resolve_and_read(mod_fp)
                    if content:
                        preloaded_files[fp] = content
                        found_line = 1
                        try:
                            ext = Path(fp).suffix.lower()
                            if ext in (".ts", ".tsx", ".js", ".jsx"):
                                import tree_sitter
                                import tree_sitter_typescript
                                if ext in (".tsx", ".jsx") and hasattr(tree_sitter_typescript, "language_tsx"):
                                    lang = tree_sitter.Language(tree_sitter_typescript.language_tsx())
                                else:
                                    lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
                                parser = tree_sitter.Parser(lang)
                                tree = parser.parse(content.encode("utf-8"))
                                if tree.root_node.has_error:
                                    from ticket_to_code.verification.two_tier_verifier import TwoTierVerifier
                                    ts_errs = TwoTierVerifier._extract_tree_sitter_errors(tree.root_node)
                                    if ts_errs:
                                        found_line = ts_errs[0][1]
                        except Exception:
                            pass
                        ctx = diag_builder.build_context(fp, content, found_line, 0, f"Errors reported in {mod_name}")
                        diag_contexts.append(ctx)
                        diag_builder.update_file_hash(fp, content)
                        self._file_versions[fp.replace('\\', '/').lower()] = diag_builder.compute_hash(content)
                        error_locations.append((fp, found_line))

        # ── Step 1c: Diagnostic Ownership & Provider Capability Resolution Pre-Flight ──
        target_consumer_file: Optional[str] = None
        has_verified_alternative = False
        verified_evidence_json = ""

        try:
            from ticket_to_code.agents.diagnostic_localizer import (
                DiagnosticLocalizer,
                FailureOwner,
                RepairAuditRecord,
            )
            from ticket_to_code.agents.provider_capability_resolver import (
                CapabilityResolutionStatus,
                ProviderCapabilityResolver,
            )
            from ticket_to_code.agents.diagnostic_normalizer import normalize_diagnostics

            localizer = DiagnosticLocalizer(self.workspace_path)
            normalized_diags = normalize_diagnostics(build_errors)
            protected_attributions = []

            for diag in normalized_diags:
                attrib = localizer.attribute_failure(diag)
                if attrib.is_protected_provider and attrib.owner == FailureOwner.CROSS_FILE_CONTRACT:
                    protected_attributions.append(attrib)

            if protected_attributions:
                # 1. Check if all errors are on protected providers with NO_VERIFIED_ALTERNATIVE
                all_no_alt = len(protected_attributions) == len(normalized_diags) and all(
                    a.capability_resolution and a.capability_resolution.status == CapabilityResolutionStatus.NO_VERIFIED_ALTERNATIVE
                    for a in protected_attributions
                )
                if all_no_alt:
                    logger.warning(
                        "  🛡️ [Fast-Exit] All diagnostics are on protected providers with NO_VERIFIED_ALTERNATIVE. "
                        "Exiting repair loop immediately (0 LLM iterations) to REPLAN."
                    )
                    for a in protected_attributions:
                        if hasattr(self, "audit_trail"):
                            self.audit_trail.append(
                                RepairAuditRecord(
                                    diagnostic=a.diagnostic.raw,
                                    diagnostic_fingerprint=f"no_alt_{a.provider_type}",
                                    owner=FailureOwner.CROSS_FILE_CONTRACT,
                                    authorization_result="BLOCKED_PROTECTED_PROVIDER",
                                    attempted_repair_target=a.provider_file or a.provider_type or "unknown",
                                    reason_for_rejection="Protected provider lacks verified alternative capability.",
                                    alternative_api_searched=True,
                                    alternative_api_found=None,
                                    final_unresolved_reason="Protected provider inspected; no compatible alternative capability exists. Immediate REPLAN.",
                                )
                            )
                    return {}

                # 2. Check if a VERIFIED_ALTERNATIVE exists
                for a in protected_attributions:
                    if a.capability_resolution and a.capability_resolution.status == CapabilityResolutionStatus.VERIFIED_ALTERNATIVE:
                        has_verified_alternative = True
                        target_consumer_file = a.consumer_file
                        alt = a.capability_resolution.verified_alternatives[0]
                        evidence_dict = {
                            "provider": a.capability_resolution.provider_name,
                            "provider_file": a.capability_resolution.provider_file,
                            "missing_symbol": a.capability_resolution.missing_symbol,
                            "protected": True,
                            "status": "VERIFIED_ALTERNATIVE",
                            "target_consumer_file": a.consumer_file,
                            "verified_alternative": {
                                "method": alt.method_name,
                                "signature": alt.signature,
                                "parameters": alt.parameters,
                                "return_type": alt.return_type,
                                "repository_evidence": alt.repository_evidence,
                                "usage_example": alt.usage_example,
                                "compatibility_proof": alt.compatibility_proof,
                            },
                            "instruction": (
                                f"Protected provider '{a.capability_resolution.provider_file}' is IMMUTABLE. "
                                f"Adapt ONLY consumer file '{a.consumer_file}' to call verified method '{alt.method_name}'. "
                                f"Do NOT attempt to modify '{a.capability_resolution.provider_file}' or any other file."
                            ),
                        }
                        verified_evidence_json = json.dumps(evidence_dict, indent=2)
                        break

                # 3. Iteration bounds:
                # If verified alternative: 1 targeted consumer patch + at most 1 compiler retry
                # If unknown: bounded targeted investigation (max 3 turns)
                if has_verified_alternative:
                    max_iter = min(max_iter, 2)
                else:
                    max_iter = min(max_iter, 3)
        except Exception as pre_exc:
            logger.debug(f"Capability resolution pre-flight failed (non-fatal): {pre_exc}")

        # 2. Build the initial prompt
        system_prompt = self._build_system_prompt()

        if diag_contexts:
            context_str = "TARGETED DIAGNOSTIC CONTEXTS (AST/Symbol-First):\n" + "\n\n".join(
                c.to_prompt_block() for c in diag_contexts
            ) + "\n"
        else:
            context_str = "(No containing symbols could be automatically resolved from errors. You must use SEARCH_WORKSPACE to find them.)\n"

        # 2b. Map error line numbers to containing methods
        error_method_map = self._map_errors_to_methods(error_text, preloaded_files)

        # 2c. Build context about which files we modified and the ticket intent
        context_meta = ""
        if self.our_modified_files:
            context_meta += "\nFILES MODIFIED BY OUR WORKFLOW (we changed these):\n"
            for fp in sorted(self.our_modified_files):
                context_meta += f"  - {fp}\n"
        if self.ticket_description:
            context_meta += f"\nORIGINAL TICKET INTENT:\n{self.ticket_description[:1000]}\n"
        if self.pre_edit_snapshots:
            context_meta += f"\nPRE-EDIT SNAPSHOTS AVAILABLE ({len(self.pre_edit_snapshots)} files) — use READ_ORIGINAL to see what a file looked like before our edits.\n"

        # ── v4: Diagnostic Intelligence — evidence-based reasoning ─────────
        evidence_block = self._build_diagnostic_evidence(build_errors)

        # ── v4: Semantic Contract — pre-identified broken invariants ──────
        contract_block = ""
        user_content = (
            f"COMPILER ERRORS:\n{error_text}\n\n"
            f"{evidence_block}{contract_block}{error_method_map}{context_meta}\n{context_str}\n"
            "⚡ ACTION DIRECTIVE (MARKET AI IDE STANDARD):\n"
            "1. The TARGET SOURCE and surrounding code are preloaded for you above.\n"
            "2. Do NOT call READ_FILE, VIEW_FILE, READ_COMPONENT, or SEARCH_WORKSPACE when the target file and line are preloaded above. Output REPLACE_CONTENT immediately to apply the fix.\n"
            "3. DO NOT OUTPUT ANY PREAMBLE, ANALYSIS, EXPLANATION, OR INTRODUCTORY PROSE.\n"
            "4. You MUST start your response IMMEDIATELY with a JSON action block.\n"
            '   Example: {"action": "REPLACE_CONTENT", "args": {"file_path": "...", "old_content": "...", "new_content": "..."}}\n'
            "CRITICAL INSTRUCTION:\n"
            "DO NOT OUTPUT ANY PREAMBLE, EXPLANATION, OR REASONING TEXT.\n"
            "YOUR RESPONSE MUST BE ONLY A JSON ACTION OBJECT.\n"
            "START YOUR RESPONSE IMMEDIATELY WITH {\"action\": \"REPLACE_CONTENT\", ...}.\n"
        )
        if verified_evidence_json:
            user_content = (
                f"STRUCTURED PROVIDER CAPABILITY EVIDENCE (Layered Proof):\n"
                f"```json\n{verified_evidence_json}\n```\n\n"
                + user_content
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]

        # 3. Enter Tool-Calling Loop
        last_compile_passed = False
        edits_since_last_compile = False

        for attempt in range(max_iter):
            logger.info(f"\n--- Error Resolution Iteration {attempt + 1}/{max_iter} ---")
            
            try:
                # Market AI IDE standard: allow adequate tokens (2500) for surgical repair blocks without truncation
                response = llm_invoke(self.llm, messages, max_tokens=4096)
                response_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                break

            logger.info(f"LLM Response:\n{response_text[:500]}...")
            
            # Append LLM's response to history
            messages.append(AIMessage(content=response_text))

            # Parse JSON action blocks
            actions = self._parse_json_actions(response_text)
            if not actions:
                logger.warning("No JSON actions found in response. Prompting LLM to use proper format.")
                messages.append(HumanMessage(content="ERROR: No valid JSON action block found. Do NOT output analysis or text. Start your response IMMEDIATELY with: {\"action\": \"REPLACE_CONTENT\", \"args\": {\"file_path\": \"...\", \"old_content\": \"...\", \"new_content\": \"...\"}}"))
                continue

            tool_responses = []
            finished = False
            batch_had_error_or_block = False

            for action in actions:
                cmd = action.get("action")
                args = action.get("args", {})
                
                if cmd == "FINISHED":
                    if batch_had_error_or_block:
                        logger.warning("Agent declared FINISHED, but previous action in batch was blocked or failed.")
                        tool_responses.append(
                            "Cannot declare FINISHED: One or more edits in your current request failed or were blocked. "
                            "You must adapt the authorized consumer file and verify the build passes before calling FINISHED."
                        )
                        continue
                    if edits_since_last_compile:
                        logger.warning("Agent declared FINISHED without compiling after edits.")
                        tool_responses.append(
                            "Cannot declare FINISHED: You have applied edits since the last compilation. "
                            "You MUST run COMPILE or QUICK_CHECK first to verify that 0 errors remain before calling FINISHED."
                        )
                        continue
                    if not last_compile_passed and (self.modified_files or edits_since_last_compile):
                        logger.warning("Agent declared FINISHED, but compilation has not succeeded.")
                        tool_responses.append(
                            "Cannot declare FINISHED: Compilation has not passed or unresolved compiler errors remain. "
                            "Run COMPILE to verify your fixes. You can only call FINISHED when COMPILE reports 0 errors."
                        )
                        continue

                    logger.info("Agent declared FINISHED with verified clean compilation.")
                    finished = True
                    break

                # Consumer-only guard is a PRE-CHECK: a blocked edit skips the action,
                # an authorized edit falls through to the REPLACE_CONTENT /
                # REPLACE_MULTI / CREATE_FILE handlers below (the old elif-chain
                # silently dropped authorized edits here).
                if target_consumer_file and cmd in ("REPLACE_CONTENT", "REPLACE_MULTI", "CREATE_FILE"):
                    req_fp = args.get("file_path", "")
                    if self._consumer_guard_blocks(target_consumer_file, req_fp):
                        logger.warning(f"  🛡️ Guardrail BLOCKED edit to non-consumer file '{req_fp}' (authorized consumer: '{target_consumer_file}')")
                        tool_responses.append(
                            f"BLOCKED: You are authorized to modify ONLY consumer file '{target_consumer_file}' to adapt it to the verified provider API. "
                            f"Modifying '{req_fp}' is prohibited."
                        )
                        batch_had_error_or_block = True
                        continue
                if cmd == "SEARCH_WORKSPACE":
                    res = self._tool_search_workspace(args.get("query", ""))
                    tool_responses.append(f"Result for SEARCH_WORKSPACE('{args.get('query')}'):\n{res}")
                elif cmd in ("READ_FILE", "VIEW_FILE"):
                    fp = args.get("file_path", args.get("path", ""))
                    s_line = args.get("start_line", args.get("start"))
                    e_line = args.get("end_line", args.get("end"))
                    res = self._tool_read_file(fp, start_line=s_line, end_line=e_line)
                    tag = f" #L{s_line}-{e_line}" if s_line else ""
                    tool_responses.append(f"Result for READ_FILE('{fp}'{tag}):\n{res}")
                elif cmd == "READ_ORIGINAL":
                    fp = args.get("file_path", args.get("path", ""))
                    s_line = args.get("start_line", args.get("start"))
                    e_line = args.get("end_line", args.get("end"))
                    res = self._tool_read_original(fp, start_line=s_line, end_line=e_line)
                    tag = f" #L{s_line}-{e_line}" if s_line else ""
                    tool_responses.append(f"Result for READ_ORIGINAL('{fp}'{tag}):\n{res}")
                elif cmd == "READ_COMPONENT":
                    res = self._tool_read_component(args.get("file", args.get("file_path", "")))
                    tool_responses.append(f"Result for READ_COMPONENT:\n{res}")
                elif cmd == "CREATE_FILE":
                    res = self._tool_edit_file(args.get("file_path", ""), args.get("content", ""))
                    if "BLOCKED" in res or "Error:" in res:
                        batch_had_error_or_block = True
                    else:
                        edits_since_last_compile = True
                        last_compile_passed = False
                    tool_responses.append(f"Result for CREATE_FILE('{args.get('file_path')}'):\n{res}")
                elif cmd == "REPLACE_CONTENT":
                    file_path = args.get("file_path", "")
                    target_content = (
                        args.get("target_content")
                        or args.get("old_content")
                        or args.get("old_str")
                        or args.get("search")
                        or args.get("original")
                        or ""
                    )
                    replacement_content = (
                        args.get("replacement_content")
                        or args.get("new_content")
                        or args.get("new_str")
                        or args.get("replace")
                        or args.get("replacement")
                        or ""
                    )
                    
                    # Validate patch targets the correct method
                    validation = self._validate_patch_target(file_path, target_content)
                    if validation:  # validation is non-empty string = rejection
                        logger.warning(f"  ⛔ Patch target validation FAILED: {validation[:200]}")
                        batch_had_error_or_block = True
                        tool_responses.append(validation)
                    else:
                        res = self._tool_replace_content(file_path, target_content, replacement_content)
                        if "BLOCKED" in res or "Error:" in res:
                            batch_had_error_or_block = True
                        else:
                            edits_since_last_compile = True
                            last_compile_passed = False
                        # v4: Post-patch invariant verification
                        inv_hint = self._verify_post_patch(file_path)
                        tool_responses.append(f"Result for REPLACE_CONTENT('{file_path}'):\n{res}{inv_hint}")
                elif cmd == "COMPILE":
                    res = self._tool_compile()
                    edits_since_last_compile = False
                    if "BUILD SUCCESS" in res or "0 error" in res.lower() or "0 errors" in res.lower():
                        last_compile_passed = True
                    else:
                        last_compile_passed = False
                    tool_responses.append(f"Result for COMPILE:\n{res}")
                elif cmd == "QUICK_CHECK":
                    res = self._tool_quick_check()
                    edits_since_last_compile = False
                    if "0 error" in res.lower() or "no error" in res.lower() or "clean" in res.lower():
                        last_compile_passed = True
                    else:
                        last_compile_passed = False
                    tool_responses.append(f"Result for QUICK_CHECK:\n{res}")
                elif cmd == "REPLACE_MULTI":
                    file_path = args.get("file_path", "")
                    replacements = args.get("replacements", [])
                    if not replacements:
                        batch_had_error_or_block = True
                        tool_responses.append("Error: REPLACE_MULTI requires a 'replacements' array.")
                    else:
                        res = self._tool_replace_multi(file_path, replacements)
                        if "BLOCKED" in res or "Error:" in res:
                            batch_had_error_or_block = True
                        else:
                            edits_since_last_compile = True
                            last_compile_passed = False
                        # v4: Post-patch invariant verification
                        inv_hint = self._verify_post_patch(file_path)
                        tool_responses.append(f"Result for REPLACE_MULTI('{file_path}'):\n{res}{inv_hint}")
                else:
                    tool_responses.append(f"Error: Unknown action '{cmd}'. Allowed actions: SEARCH_WORKSPACE, READ_FILE, READ_ORIGINAL, READ_COMPONENT, REPLACE_CONTENT, REPLACE_MULTI, CREATE_FILE, COMPILE, QUICK_CHECK, FINISHED.")

            if finished:
                break
                
            if tool_responses:
                # Bound individual tool responses: compiler output gets the build
                # budget (errors are the whole point of this loop), everything
                # else the generic tool-output budget. Shared limits live in
                # runtime/limits.py so the caps cannot drift between agents.
                bounded_responses = []
                for tr in tool_responses:
                    limit = (
                        BUILD_OUTPUT_MAX_CHARS
                        if tr.startswith(("Result for COMPILE", "Result for QUICK_CHECK"))
                        else TOOL_OUTPUT_MAX_CHARS
                    )
                    bounded_responses.append(truncate(tr, limit))
                feedback = "\n\n".join(bounded_responses)
                # Structured compaction (bounded repair context): retains objective, errors, modified files, and feedback
                applied_summary = "\n".join(
                    f"- {fp} ({len(c_str.splitlines())} lines)"
                    for fp, c_str in self.modified_files.items()
                ) or "None yet"
                compact_content = (
                    f"ITERATION {attempt + 1}/{max_iter}\n\n"
                    f"CURRENT COMPILER ERRORS TO FIX:\n{error_text[:1200]}\n\n"
                    f"FILES CURRENTLY MODIFIED:\n{applied_summary}\n\n"
                    f"LATEST TOOL RESULTS / COMPILER FEEDBACK:\n{feedback}\n\n"
                    '⚡ DIRECTIVE: Output ONLY a JSON action block (e.g. REPLACE_CONTENT or FINISHED). Start IMMEDIATELY with {"action": ...}.'
                )
                messages = [
                    messages[0],
                    HumanMessage(content=compact_content)
                ]

        final_files = {self._file_case_map.get(k, k): v for k, v in self.modified_files.items()}
        logger.info(f"Agent finished. Total files modified: {len(final_files)}")
        return final_files

    def _build_system_prompt(self) -> str:
        # ── Build context injection from earlier pipeline phases ────────────
        context_block = ""
        if self.exec_ctx:
            ctx_text = self.exec_ctx.to_context_block(max_length=3500)
            if ctx_text.strip():
                context_block = f"""

═══════════════════════════════════════════════════════════════════════════
PIPELINE CONTEXT (from investigation, planning, and discovery phases):
═══════════════════════════════════════════════════════════════════════════
{ctx_text}
═══════════════════════════════════════════════════════════════════════════

CRITICAL REASONING RULES based on the above context:
1. When you see "Property X does not exist on type Y":
   - FIRST use READ_COMPONENT to understand how the files relate to each other
   - Check if X is used in the CORRECT lifecycle hook / function
   - If the code is in the WRONG place, MOVE it — don't just add the missing property
   - Check COMPONENT GROUPS above to understand which .ts/.html/.model files are related
2. When fixing an interface/model, check the PLANNED CHANGES to understand WHY properties
   were added — don't blindly remove them, check if they need to be in a DIFFERENT interface
3. When you see PREVIOUS FAILED APPROACHES, do NOT repeat the same fix strategy
4. Use DATA FLOW GAPS to verify your fix actually connects template bindings to controller data
"""

        plan_block = ""
        if hasattr(self, "plan") and self.plan:
            tasks = getattr(self.plan, "tasks", []) or []
            if tasks:
                t_lines = [f"  - [{getattr(t, 'file_path', '')}] {getattr(t, 'title', '')}: {getattr(t, 'description', '')}" for t in tasks]
                plan_block = f"""
═══════════════════════════════════════════════════════════════════════════
ARCHITECTURAL PLAN (Ensure fixes align with this intended architecture):
═══════════════════════════════════════════════════════════════════════════
{chr(10).join(t_lines)}
═══════════════════════════════════════════════════════════════════════════
"""
        context_block = plan_block + context_block

        # ── Coupled caller-callee co-fix pairs (P0: multi-file healing) ────
        # TS2339-style errors couple a CALLER with a DECLARATION file; fixing
        # the caller alone produces local mocks that break the runtime
        # dataflow.  Pairs resolved by the fix node land in self.cofix_pairs
        # and are surfaced here so both files are healed in one session.
        _cofix_pairs = getattr(self, "cofix_pairs", None) or []
        if _cofix_pairs:
            _pairs_txt = "\n\n".join(
                f"  • '{p.get('member', '?')}' missing on type '{p.get('type', '?')}' — "
                "heal BOTH files together:\n"
                f"      caller:      {p.get('caller', '?')}\n"
                f"      declaration: {p.get('callee', '?')}"
                for p in _cofix_pairs
            )
            context_block += (
                "\n\n═══════════════ CO-FIX PAIRS (multi-file healing) ═══════════════\n"
                "These compiler errors couple a CALLER with a DECLARATION file. Edit BOTH in "
                "this session: add the missing member to the declaration file with the correct "
                "signature AND keep the caller's usage clean. NEVER mock the member locally "
                "inside the caller.\n"
                f"{_pairs_txt}\n"
                "═════════════════════════════════════════════════════════════════"
            )

        # ── v4: Diagnostic Intelligence reasoning instructions ────────────
        evidence_reasoning = ""
        if self.hypothesis_builder is not None:
            evidence_reasoning = """

═══════════════════════════════════════════════════════════════════════════
DIAGNOSTIC INTELLIGENCE — HOW TO USE THE EVIDENCE BELOW
═══════════════════════════════════════════════════════════════════════════

The COMPILER ERRORS section includes structured HYPOTHESES with:
- VIOLATED INVARIANT: The program relationship that is broken
- EVIDENCE: Facts gathered from LSP, component analysis, and symbol resolution
- CANDIDATE FIX LOCATIONS: Ranked possible files to fix, with confidence scores

REASONING PROTOCOL (you MUST follow this):
1. READ the hypothesis evidence BEFORE touching any file
2. The DEFINITION FILE is not always the right fix location:
   - "Property X not on type Y" could mean: add X to Y's definition, OR fix the
     reference in the source file (typo, wrong type, wrong property name)
   - The CANDIDATES section lists ALL possibilities with confidence scores
3. Pick the candidate whose FIX aligns with the TICKET CONTEXT
   - If the ticket requires adding a feature → add the missing symbol to the definition
   - If the code was auto-generated incorrectly → fix the source reference
4. After fixing, use QUICK_CHECK to verify the invariant is now satisfied
5. NEVER blindly trust confidence scores — they are structural heuristics, not certainty
6. NEVER comment out, delete, or disable functional logic or ticket requirements just to make the compiler pass. If a class or symbol is not found, use SEARCH_WORKSPACE to locate the actual repository/service/model in the project and adapt the code to call the correct existing class or method.
7. JAVA COMPILER ERROR DISCRIMINATION:
   - When seeing 'cannot find symbol: variable <var>' on an existing class: ALWAYS check the class field declarations. If an existing field had an unusual name or misspelling in the original code (e.g. participantingMemberTypeRef), preserve that exact name or ensure ALL references throughout the file match. NEVER rename a field definition without updating every call site in the file.
   - When seeing 'cannot find symbol: method <method>' on a service/repository: NEVER guess or hallucinate that the method exists on an unrelated service. Check if the entity has an existing Spring Data Repository (e.g. ContractMemberRepository) and inject/call the repository query method directly.
   - When seeing 'cannot find symbol: class <Class>':
     a) Check if the symbol is actually used! If a newly injected field (e.g. 'private final SomeRepository someRepo;') is NOT called anywhere in the methods of that file, REMOVE the unused import and field declaration. Do NOT create dummy files or new repositories for unused fields.
     b) Search the repository/service package using SEARCH_WORKSPACE. Check for plural/singular differences (e.g. 'ParticipatingMembersRepository' vs 'ParticipatingMemberRepository'). ALWAYS import and use the real existing class in the codebase; NEVER invent a duplicate repository or entity!
     c) When creating or editing Java files, the file name MUST strictly match the class/interface name with exact casing (e.g. 'ParticipatingMembersRepository.java').
   - When seeing constructor parameter mismatch in unit tests: If a production class with @RequiredArgsConstructor added a new final field, update the corresponding unit test to add a @Mock for that field and pass it into the constructor.
"""

        return f"""You are a senior software engineer resolving compiler/build errors with FULL PROJECT CONTEXT.
You understand not just the syntax errors but WHY the code was written, what the ticket requires,
and how the components relate to each other.
{context_block}{evidence_reasoning}
You interact with the workspace by outputting JSON blocks.
You can use the following tools:

1. SEARCH_WORKSPACE
Search for a class, interface, or file name.
```json
{{
  "action": "SEARCH_WORKSPACE",
  "args": {{
    "query": "MemberService"
  }}
}}
```

2. READ_FILE
Read the CURRENT contents of a file. Supports line range slicing to save tokens and inspect exact lines.
```json
{{
  "action": "READ_FILE",
  "args": {{
    "file_path": "path/to/file.ts",
    "start_line": 150,
    "end_line": 250
  }}
}}
```
*(Tip: Always use `start_line` and `end_line` on large files to focus on the target method or error lines #L150-250)*

3. READ_ORIGINAL
Read the ORIGINAL contents of a file BEFORE our workflow modified it. Supports line slicing.
```json
{{
  "action": "READ_ORIGINAL",
  "args": {{
    "file_path": "path/to/file.ts",
    "start_line": 150,
    "end_line": 250
  }}
}}
```

4. READ_COMPONENT
Get the FULL component structure for a file: related files (template, model, service), methods, properties, event bindings, and data flow gaps. Use this BEFORE fixing "Property does not exist" errors!
```json
{{
  "action": "READ_COMPONENT",
  "args": {{
    "file": "src/app/members/add-members/add-members.component.ts"
  }}
}}
```

5. REPLACE_CONTENT
Replace a specific block of code in an existing file.
```json
{{
  "action": "REPLACE_CONTENT",
  "args": {{
    "file_path": "path/to/file.ts",
    "target_content": "    private translateService: TranslateService,",
    "replacement_content": "    private translateService: TranslateService,\\n    private newService: NewService,"
  }}
}}
```
*Rule: `target_content` must be an exact, unique string match from the existing file.*

5b. REPLACE_MULTI
Apply MULTIPLE replacements to the SAME file in one tool call. Use when you need to fix several places in one file. Each replacement is applied sequentially.
```json
{{
  "action": "REPLACE_MULTI",
  "args": {{
    "file_path": "path/to/file.ts",
    "replacements": [
      {{"target": "old code 1", "replacement": "new code 1"}},
      {{"target": "old code 2", "replacement": "new code 2"}}
    ]
  }}
}}
```

6. CREATE_FILE
Create a completely new file (use only if the file does not exist).
```json
{{
  "action": "CREATE_FILE",
  "args": {{
    "file_path": "path/to/file.ts",
    "content": "// COMPLETE new file content here..."
  }}
}}
```

7. QUICK_CHECK
Run a FAST type-check (tsc --noEmit for TS, mvn compile -q for Java, py_compile for Python) without full build. Use between individual fixes to check progress without waiting for full compilation. ~10 seconds instead of ~3 minutes.
```json
{{
  "action": "QUICK_CHECK"
}}
```

8. COMPILE
Run the FULL project build (ng build for Angular, mvn install for Java, etc.). Use after all fixes are applied to get final verification. This is slower but catches template errors that QUICK_CHECK misses.
```json
{{
  "action": "COMPILE"
}}
```

9. FINISHED
When the last COMPILE returned 0 errors, declare done.
```json
{{
  "action": "FINISHED"
}}
```

Rules:
- You may output multiple JSON action blocks in a single response.
- Do not use markdown wrappers around the JSON unless it is exactly ```json ... ```.
- Use REPLACE_CONTENT to modify existing files. Provide enough lines in `target_content` to make it unique (typically 3 to 8 lines).
- Write ONLY the specific part to replace — never rewrite whole functions or large blocks.
- Directly replace deprecated code with the updated implementation in `replacement_content` without leaving dead code or invalid comment tokens.
- Only use CREATE_FILE for brand new files, and provide the FULL file content.
- If you see an error about a missing file, SEARCH for it before blindly creating it!
- IMPORTANT: When the ERROR LINE MAPPING section identifies which method contains an error, FIX THAT SPECIFIC METHOD. Do NOT fix a different method that calls it.
- CRITICAL WORKFLOW: Fix errors → QUICK_CHECK → check remaining → fix more → QUICK_CHECK → when QUICK_CHECK passes → COMPILE for full verification → repeat until 0 errors.
- When many errors reference the same type/interface/class, fix the TYPE DEFINITION first — a single fix there may resolve many errors at once. Then QUICK_CHECK to verify.
- When you see errors in files you did NOT modify, check if a file you DID modify broke them (use READ_ORIGINAL to compare). Fix your file correctly rather than editing many consumer files.
- When fixing a shared model/interface file, ensure your fix preserves backward compatibility with existing consumers unless the ticket explicitly requires removing something.
- BEFORE fixing "Property X does not exist on type Y" errors, ALWAYS use READ_COMPONENT to understand the full component structure. This prevents tunnel-vision fixes.
- When DIAGNOSTIC INTELLIGENCE hypotheses are provided, use the CANDIDATE FIX LOCATIONS to guide your first action — don't search for files that the evidence already identified.

COMMON TYPESCRIPT ERROR QUICK-FIXES:
- TS2322 'Type "String" is not assignable to type "string"': Change uppercase `String` to lowercase `string`. Same for `Number`→`number`, `Boolean`→`boolean`, `Object`→`object`.
- TS2554 'Expected N arguments, but got M': Check if a function parameter was added/removed by our edits. Either add the missing parameter to the function signature or remove the extra argument from the call site.
- TS2304 'Cannot find name X': Add the missing import statement.
- TS2339 'Property X does not exist on type Y':
  * If the error is reported in an HTML template (*.component.html):
    DO NOT ADD THE PROPERTY TO THE CONTROLLER (.ts) CLASS!
    Check the sibling .ts controller first — it almost certainly already declares a property representing this state under a slightly different name (e.g. `isExistingProjectMember` vs `selectedUserIsExistingProjectMember`).
    FIX THE HTML TEMPLATE to bind to the existing controller property! Blindly declaring new properties on the .ts controller causes duplicate identifier (TS2300) errors and breaks logic.
  * If the error is in a TypeScript file (.ts):
    Use READ_COMPONENT first! Check if the property belongs on a different interface (e.g., DisplayedMember vs Member). Don't just add it blindly.
- TS2300 'Duplicate identifier X': Remove the redundant duplicate declaration of X. Keep only the canonical declaration with proper type and initialization.
- TS2345 'Argument of type X is not assignable to parameter of type Y': Check if the function expects a different type. Fix the caller or add proper type conversion.
- TS2353 'Object literal may only specify known properties, and X does not exist in type Y':
  * If Y is an imported or shared model that cannot be modified, do NOT attempt to edit the model file.
  * Instead, adapt the consumer file: declare a local extended interface (`interface LocalY extends Y {{ X?: type; }}`), use an intersection type (`Y & {{ X?: type }}`), or cast/assign the property after instantiation (`(obj as any).X = ...`).

CRITICAL: For simple type errors (TS2322, TS2554), fix them IMMEDIATELY with REPLACE_CONTENT — do NOT waste iterations reading files or searching. You already have the file content and error line number."""

    def _parse_json_actions(self, text: str) -> list[dict]:
        actions = []
        # 1. Find all fenced JSON blocks
        matches = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    parsed = json.loads(match.strip())
                    if isinstance(parsed, list):
                        actions.extend(parsed)
                    elif isinstance(parsed, dict):
                        actions.append(parsed)
                except json.JSONDecodeError:
                    pass
            if actions:
                return actions

        # 2. Bracket-balanced JSON extractor (handles nested objects and text around JSON)
        start_indices = [m.start() for m in re.finditer(r'\{\s*"action"\s*:', text)]
        for s_idx in start_indices:
            depth = 0
            in_string = False
            escape = False
            end_idx = -1
            for i in range(s_idx, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
            if end_idx != -1:
                candidate = text[s_idx:end_idx].strip()
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "action" in parsed:
                        actions.append(parsed)
                except json.JSONDecodeError:
                    pass

        return actions

    # ── v4: Diagnostic Intelligence Pipeline ──────────────────────────────

    def _content_hash(self, content: str) -> str:
        """Generate a short content hash for staleness detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _build_diagnostic_evidence(self, build_errors: list[str]) -> str:
        """Run the full diagnostic intelligence pipeline if available.

        Pipeline: normalize → localize → hypothesize → format

        Falls back gracefully if any component is not wired.
        Returns empty string if the pipeline is not available.
        """
        if not self.fix_localizer or not self.hypothesis_builder:
            return ""

        try:
            # Step 1: Normalize raw errors into structured diagnostics
            from ticket_to_code.agents.diagnostic_normalizer import normalize_diagnostics
            diagnostics = normalize_diagnostics(build_errors)
            if not diagnostics:
                return ""

            # Step 2: Localize — find candidate fix locations
            localization_results = self.fix_localizer.localize(diagnostics)

            # Step 3: Analyze relationships (if analyzer is available)
            rel_ctx = None
            if self.relationship_analyzer:
                source_files = list({d.source_file for d in diagnostics if d.source_file})
                if source_files:
                    try:
                        rel_ctx = self.relationship_analyzer.analyze_files(source_files)
                        # Check invariants to find broken ones
                        self.relationship_analyzer.check_all_invariants(rel_ctx)
                    except Exception as exc:
                        logger.debug(f"  [v4] Relationship analysis failed: {exc}")

            # Step 4: Build hypotheses with evidence
            hypotheses = self.hypothesis_builder.build_hypotheses(
                diagnostics=diagnostics,
                localization_results=localization_results,
                relationship_context=rel_ctx,
                ticket_intent=self.ticket_description,
            )

            # Step 5: Format for the LLM prompt
            evidence = self.hypothesis_builder.format_for_prompt(hypotheses)
            if evidence:
                logger.info(
                    f"  [v4] Diagnostic Intelligence: {len(hypotheses)} hypotheses, "
                    f"{sum(len(h.candidates) for h in hypotheses)} candidates"
                )
            return evidence + "\n\n" if evidence else ""

        except Exception as exc:
            logger.warning(f"  [v4] Diagnostic intelligence pipeline failed (non-fatal): {exc}")
            return ""

    def _verify_post_patch(self, file_path: str) -> str:
        """Post-patch invariant re-check (Step 8).

        After a successful REPLACE_CONTENT or REPLACE_MULTI, re-check any
        invariants that involve the patched file to give the LLM immediate
        feedback on whether the fix resolved the underlying relationship violation.

        Returns a hint string to append to the tool response, or empty string.
        """
        if not self.relationship_analyzer:
            return ""

        try:
            # Re-analyze relationships for the patched file
            symbols = []  # We don't have specific symbols here, so use file-level analysis
            rel_ctx = self.relationship_analyzer.analyze_for_diagnostic(file_path, symbols)

            # Check all invariants
            broken = self.relationship_analyzer.check_all_invariants(rel_ctx)

            if broken:
                hint_lines = ["\n[INVARIANT CHECK] After your patch, these invariants are still broken:"]
                for inv in broken[:5]:
                    hint_lines.append(f"  {inv}")
                return "\n".join(hint_lines)
            elif rel_ctx.invariants:
                satisfied = [i for i in rel_ctx.invariants if i.is_satisfied]
                if satisfied:
                    return f"\n[INVARIANT CHECK] ✓ {len(satisfied)} invariant(s) now satisfied."
        except Exception as exc:
            logger.debug(f"  [v4] Post-patch invariant check failed (non-fatal): {exc}")

        return ""

    def _parse_error_locations(self, error_text: str) -> list[tuple[str, int]]:
        """Parse (file_path, line_number) pairs from compiler errors."""
        line_rx = re.compile(
            r'(?P<path>[^\s:]+\.(?:java|ts|tsx|js|jsx|py|kt|scala|html|scss|css|cs))'
            r'(?:'
            r':\[(?P<line1>\d+)'          # Maven format:  file.java:[952,57]
            r'|'
            r'\((?P<line2>\d+)'           # TS/CS format:  file.ts(12,3)
            r'|'
            r':(?P<line3>\d+)'            # Simple format: file.java:952
            r'|'
            r':\s*\[[A-Za-z0-9_-]+\]\s*(?:line\s+)?(?P<line4>\d+)' # Fast syntax check format
            r')',
            re.IGNORECASE
        )
        error_locations: list[tuple[str, int]] = []
        seen = set()
        for m in line_rx.finditer(error_text):
            raw_path = m.group('path')
            line_str = m.group('line1') or m.group('line2') or m.group('line3') or m.group('line4')
            if line_str:
                loc = (raw_path, int(line_str))
                if loc not in seen:
                    seen.add(loc)
                    error_locations.append(loc)
        return error_locations

    def _preload_error_files(self, error_text: str) -> dict[str, str]:
        """Extract file paths from compiler errors and preload their contents."""
        preloaded = {}
        # Regex to catch paths with line numbers (e.g. src/app/file.ts(12,3) or src/app/file.java:[12,3])
        path_rx = re.compile(
            r"(?P<path>"
            r"(?:(?:[A-Za-z]:[/\\]|/[A-Za-z]:/)[^\n\r\s]*?"
            r"|"
            r"[^:\n\r\[\s]+?)"
            r"\.(?:ts|tsx|js|jsx|html|scss|css|cs|java|py|kt|scala))"
            r"(?:\(|:\[|:)",
            re.IGNORECASE
        )
        
        matches = path_rx.findall(error_text)
        # Deduplicate while preserving order
        unique_paths = list(dict.fromkeys(matches))
        
        for p in unique_paths[:10]:  # Limit to avoid massive context
            fp, content = self._resolve_and_read(p)
            if content:
                preloaded[fp] = content
        return preloaded

    def _map_errors_to_methods(self, error_text: str, preloaded_files: dict[str, str]) -> str:
        """Parse error line numbers and map each to the method that contains it.

        Compiler errors like 'File.java:[952,57] unreported exception' tell us
        the line number but not which method that line belongs to.  The LLM may
        then edit the wrong method (e.g. a caller instead of the method that
        actually contains line 952).

        This method builds an explicit mapping:
            ERROR LINE MAPPING:
              File.java:952 → inside method `isDeliverableNameUnique` (lines 943-961)
              File.java:945 → inside method `isDeliverableNameUnique` (lines 943-961)

        so the LLM knows exactly which method to target its fix on.
        """
        from ticket_to_code.agents.smart_extract import _parse_java_ts_boundaries, _parse_python_boundaries

        # Parse file:line pairs from error text
        # Handles Maven:      File.java:[952,57]
        # Handles TypeScript:  File.ts(12,3)
        # Handles simple:     File.java:952
        line_rx = re.compile(
            r'(?P<path>[^\s]+\.(?:java|ts|tsx|js|jsx|py|kt|scala))'
            r'(?:'
            r':\[(?P<line1>\d+)'          # Maven format:  file.java:[952,57]
            r'|'
            r'\((?P<line2>\d+)'           # TS format:     file.ts(12,3)
            r'|'
            r':(?P<line3>\d+)'            # Simple format: file.java:952
            r')',
            re.IGNORECASE
        )

        error_locations: list[tuple[str, int]] = []
        for m in line_rx.finditer(error_text):
            raw_path = m.group('path')
            line_str = m.group('line1') or m.group('line2') or m.group('line3')
            if line_str:
                error_locations.append((raw_path, int(line_str)))

        if not error_locations:
            return ""

        # Build method boundaries for each file we have preloaded
        file_boundaries: dict[str, list] = {}
        for fp, content in preloaded_files.items():
            file_lines = content.splitlines()
            ext = fp.rsplit('.', 1)[-1].lower() if '.' in fp else ''
            if ext in ('java', 'ts', 'tsx', 'js', 'jsx', 'kt', 'scala'):
                file_boundaries[fp] = _parse_java_ts_boundaries(file_lines)
            elif ext == 'py':
                file_boundaries[fp] = _parse_python_boundaries(file_lines)

        # Map each error location to its containing method
        mappings: list[str] = []
        seen = set()
        for raw_path, line_num in error_locations:
            # Normalize path for matching
            norm = raw_path.replace('\\', '/').lower()
            matched_fp = None
            for fp in file_boundaries:
                if norm.endswith(fp.lower()) or fp.lower().endswith(norm):
                    matched_fp = fp
                    break
                # Also try basename match
                if norm.split('/')[-1] == fp.split('/')[-1]:
                    matched_fp = fp
                    break

            if not matched_fp:
                continue

            boundaries = file_boundaries[matched_fp]
            # Find which method contains this line (0-indexed internally, errors are 1-indexed)
            error_line_0 = line_num - 1
            containing_method = None
            for mb in boundaries:
                if mb.kind == "class":
                    continue
                if mb.start_line <= error_line_0 <= mb.end_line:
                    containing_method = mb
                    break

            if containing_method:
                key = (matched_fp, containing_method.name, line_num)
                if key not in seen:
                    seen.add(key)
                    sig_preview = containing_method.signature[:200] if containing_method.signature else ""
                    mappings.append(
                        f"  {raw_path}:{line_num} → inside method `{containing_method.name}` "
                        f"(lines {containing_method.start_line+1}-{containing_method.end_line+1})"
                        f"\n    Signature: {sig_preview}"
                    )
                    # Store structured data for patch target validation
                    norm_fp = matched_fp.replace('\\', '/').lower()
                    if norm_fp not in self._error_method_map:
                        self._error_method_map[norm_fp] = []
                    self._error_method_map[norm_fp].append(
                        (line_num, containing_method.name,
                         containing_method.start_line, containing_method.end_line)
                    )

        if not mappings:
            return ""

        return (
            "ERROR LINE MAPPING (which method contains each error line — "
            "fix THESE methods, not their callers):\n"
            + "\n".join(mappings) + "\n\n"
        )

    def _validate_patch_target(self, file_path: str, target_content: str) -> str:
        """Validate that a REPLACE_CONTENT patch targets a method containing an error line.

        Returns empty string if valid (or if validation is not applicable).
        Returns rejection message string if the patch targets the wrong method.

        This is defense-in-depth: even if the LLM ignores the prompt instruction
        to fix the correct method, this deterministic check catches it.
        """
        if not self._error_method_map:
            return ""  # No mappings available, skip validation

        # Normalize the patch file path
        norm_fp = file_path.replace('\\', '/').lower()

        # Find matching file in our error map
        matched_key = None
        for key in self._error_method_map:
            if norm_fp.endswith(key) or key.endswith(norm_fp):
                matched_key = key
                break
            # Basename match
            if norm_fp.split('/')[-1] == key.split('/')[-1]:
                matched_key = key
                break

        if not matched_key:
            return ""  # This file has no error mappings, allow the patch

        # Read the current file content to find WHERE the target_content falls
        fp_resolved, content = self._resolve_and_read(file_path)
        if not content or target_content not in content:
            return ""  # Can't validate if we can't read or find the target

        # Find the line number where target_content starts
        lines_before = content[:content.index(target_content)].count('\n')
        target_start_line = lines_before  # 0-indexed

        # Get the error-containing methods for this file
        error_methods = self._error_method_map[matched_key]
        expected_method_names = set()
        for error_line, method_name, method_start, method_end in error_methods:
            expected_method_names.add(method_name)

        # Check: does target_content fall inside ANY of the error-containing methods?
        target_in_error_method = False
        for error_line, method_name, method_start, method_end in error_methods:
            if method_start <= target_start_line <= method_end:
                target_in_error_method = True
                break

        if target_in_error_method:
            return ""  # Patch targets the correct method

        # Determine what method the patch IS targeting (for the rejection message)
        from ticket_to_code.agents.smart_extract import _parse_java_ts_boundaries, _parse_python_boundaries
        file_lines = content.splitlines()
        ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
        if ext in ('java', 'ts', 'tsx', 'js', 'jsx', 'kt', 'scala'):
            boundaries = _parse_java_ts_boundaries(file_lines)
        elif ext == 'py':
            boundaries = _parse_python_boundaries(file_lines)
        else:
            return ""  # Can't validate unknown file types

        actual_target_method = "unknown"
        for mb in boundaries:
            if mb.kind == "class":
                continue
            if mb.start_line <= target_start_line <= mb.end_line:
                actual_target_method = mb.name
                break

        if actual_target_method == "unknown":
            # If target falls outside recognized method boundaries (e.g. imports, class body, syntax error recovery),
            # allow the patch rather than blocking it.
            return ""

        expected_names = ", ".join(f"`{n}`" for n in sorted(expected_method_names))
        return (
            f"PATCH TARGET VALIDATION FAILED.\n\n"
            f"Your patch modifies method `{actual_target_method}`, but the compiler error "
            f"is inside method {expected_names}.\n\n"
            f"You must fix {expected_names} directly — for example by adding a `throws` "
            f"clause to its signature, or wrapping the call in a try-catch. "
            f"Do NOT propagate the exception to callers.\n\n"
            f"Please generate a new REPLACE_CONTENT that targets {expected_names}."
        )


    def _resolve_and_read(self, raw_path: str) -> tuple[str, Optional[str]]:
        fp = str(raw_path).strip().replace("\\", "/")
        # Strip leading slash if followed by Windows drive letter (e.g. /C:/ -> C:/)
        if fp.startswith("/") and len(fp) > 2 and fp[1].isalpha() and fp[2] == ":":
            fp = fp[1:]
        
        # Try absolute
        abs_path = Path(fp)
        if not abs_path.is_absolute():
            abs_path = self.workspace_path / fp
        elif self.workspace_path:
            try:
                fp = str(abs_path.relative_to(self.workspace_path)).replace("\\", "/")
            except ValueError:
                pass
            
        # Try finding it if the compiler used a relative path not rooted at workspace
        if not abs_path.exists():
            target_name = Path(fp).name.lower()
            import os
            for root, dirs, files in os.walk(str(self.workspace_path)):
                dirs[:] = [d for d in dirs if d not in {"node_modules", "dist", ".git", "build", "target", ".venv"}]
                for f in files:
                    if f.lower() == target_name:
                        rel = str(Path(root, f).relative_to(self.workspace_path)).replace("\\", "/")
                        if fp.endswith(rel.split("/", 1)[-1] if "/" in rel else rel):
                            abs_path = self.workspace_path / rel
                            fp = rel
                            break
        
        if abs_path.exists():
            # If we've already modified it in this run, return our modified version
            if fp.lower() in self.modified_files:
                return fp, self.modified_files[fp.lower()]
            try:
                return fp, abs_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
        return fp, None

    def _tool_search_workspace(self, query: str) -> str:
        logger.info(f"Tool SEARCH_WORKSPACE: {query}")
        matches = set()
        
        # Tier 1: SQLite Store
        if self.sqlite_store:
            try:
                rows = self.sqlite_store._conn.execute(
                    "SELECT DISTINCT path FROM symbols WHERE name LIKE ? OR path LIKE ? LIMIT 10",
                    (f"%{query}%", f"%{query}%")
                ).fetchall()
                for (path,) in rows:
                    if path:
                        matches.add(path.replace("\\", "/"))
            except Exception as e:
                logger.debug(f"SQLite search failed: {e}")
                
        # Tier 2: rglob (if not enough matches)
        if not matches:
            try:
                # Basic kebab conversion for Angular paths
                kebab = re.sub(r'(?<=[a-z])(?=[A-Z])', '-', query).lower()
                stem = kebab.replace('-', '')
                
                skip_dirs = {'node_modules', 'dist', '.git', 'build', 'target'}
                for p in self.workspace_path.rglob('*'):
                    if not p.is_file(): continue
                    if skip_dirs.intersection(p.parts): continue
                    
                    p_stem = p.stem.replace('-', '').replace('.', '').lower()
                    p_name = p.name.lower()
                    clean_query = query.lower().replace('-', '').replace('.', '')
                    if stem in p_stem or query.lower() in p_name or clean_query in p_stem or query.lower() in p_stem:
                        rel = str(p.relative_to(self.workspace_path)).replace('\\', '/')
                        matches.add(rel)
                        if len(matches) > 10:
                            break
            except Exception:
                pass
                
        if not matches:
            return f"No matches found in workspace for '{query}'."
        return "Found the following matching files:\n" + "\n".join(sorted(matches))

    def _tool_read_file(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> str:
        fp, content = self._resolve_and_read(file_path)
        if content is None:
            return f"Error: Could not read file '{file_path}'. It may not exist."

        lines = content.splitlines()
        total_lines = len(lines)

        # If line range is specified, slice accurately
        if start_line is not None or end_line is not None:
            try:
                s_line = max(1, int(start_line or 1))
            except Exception:
                s_line = 1
            try:
                e_line = min(total_lines, int(end_line or (s_line + 150)))
            except Exception:
                e_line = min(total_lines, s_line + 150)

            logger.info(f"Analyzed {file_path} #L{s_line}-{e_line}")
            start_idx = max(0, s_line - 1)
            end_idx = min(total_lines, e_line)
            numbered = [f"{i + 1}: {lines[i]}" for i in range(start_idx, end_idx)]
            header = f"File: {file_path} #L{s_line}-{e_line} (Total lines: {total_lines}):\n"
            return header + "\n".join(numbered)

        logger.info(f"Analyzed {file_path} #L1-{min(total_lines, 200)}")

        if len(content) > 25000 or total_lines > 400:
            from ticket_to_code.agents.smart_extract import smart_extract
            extracted = smart_extract(content, file_path=file_path)
            # Also provide the first 120 lines directly
            preview_lines = [f"{i + 1}: {lines[i]}" for i in range(min(total_lines, 120))]
            return (
                f"(File is large: {total_lines} lines / {len(content)} chars. "
                f"Showing #L1-{min(total_lines, 120)}. Use start_line & end_line args to inspect specific ranges):\n"
                + "\n".join(preview_lines) + "\n\n"
                f"--- STRUCTURAL OUTLINE ---\n{extracted}"
            )

        # For normal sized files, return with line numbers for exact patch anchoring
        numbered = [f"{i + 1}: {lines[i]}" for i in range(total_lines)]
        return f"File: {file_path} #L1-{total_lines} (Total lines: {total_lines}):\n" + "\n".join(numbered)

    def _tool_read_original(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> str:
        """Read the ORIGINAL content of a file before our workflow modified it."""
        if not self.pre_edit_snapshots:
            return "No pre-edit snapshots available. Cannot compare with original."

        fp_norm = file_path.replace("\\", "/").lower()
        matched_content = None
        # Try exact match first
        for snap_path, snap_content in self.pre_edit_snapshots.items():
            if snap_path.replace("\\", "/").lower() == fp_norm:
                matched_content = snap_content
                break
        if matched_content is None:
            # Try basename match
            fp_basename = fp_norm.split("/")[-1]
            for snap_path, snap_content in self.pre_edit_snapshots.items():
                snap_basename = snap_path.replace("\\", "/").split("/")[-1].lower()
                if snap_basename == fp_basename:
                    matched_content = snap_content
                    break
        if matched_content is None:
            # Try suffix match
            for snap_path, snap_content in self.pre_edit_snapshots.items():
                snap_norm = snap_path.replace("\\", "/").lower()
                if fp_norm.endswith(snap_norm) or snap_norm.endswith(fp_norm):
                    matched_content = snap_content
                    break

        if matched_content is None:
            return f"No pre-edit snapshot found for '{file_path}'. This file may not have been modified by our workflow."

        lines = matched_content.splitlines()
        total_lines = len(lines)

        if start_line is not None or end_line is not None:
            try:
                s_line = max(1, int(start_line or 1))
            except Exception:
                s_line = 1
            try:
                e_line = min(total_lines, int(end_line or (s_line + 150)))
            except Exception:
                e_line = min(total_lines, s_line + 150)

            logger.info(f"Analyzed (original) {file_path} #L{s_line}-{e_line}")
            start_idx = max(0, s_line - 1)
            end_idx = min(total_lines, e_line)
            numbered = [f"{i + 1}: {lines[i]}" for i in range(start_idx, end_idx)]
            header = f"ORIGINAL File: {file_path} #L{s_line}-{e_line} (Total lines: {total_lines}):\n"
            return header + "\n".join(numbered)

        logger.info(f"Analyzed (original) {file_path} #L1-{min(total_lines, 200)}")
        numbered = [f"{i + 1}: {lines[i]}" for i in range(total_lines)]
        return f"ORIGINAL File: {file_path} #L1-{total_lines} (Total lines: {total_lines}):\n" + "\n".join(numbered)

    def _matches_our_modified(self, norm_path: str) -> bool:
        """True if this path is one our workflow generated/modified this run."""
        owned = set(self.our_modified_files) | set(self.modified_files.keys())
        for mod in owned:
            if not mod:
                continue
            if norm_path == mod or norm_path.endswith(mod) or mod.endswith(norm_path):
                return True
        return False

    def _is_user_authorized(self, norm_path: str) -> bool:
        """True if the user explicitly authorized fixing this pre-existing file."""
        for auth in self.authorized_pre_existing:
            if not auth:
                continue
            if norm_path == auth or norm_path.endswith(auth) or auth.endswith(norm_path):
                return True
        return False

    def classify_error_file(self, file_path: str, existing_content: Optional[str] = None) -> str:
        """Attribute a file's errors relative to the ticket.

        Returns an ErrorCategory value. The decisive, language-agnostic signal:
        a file our workflow never generated/modified, that already existed
        before our run, owns PRE_EXISTING errors and must NOT be edited.
        """
        from ticket_to_code.models import ErrorCategory

        norm = file_path.replace("\\", "/").lower()
        if self._matches_our_modified(norm):
            return ErrorCategory.GENERATED_COMPANION.value
        # Does it exist on disk (i.e. not a brand-new file being created)?
        exists = existing_content is not None
        if not exists:
            _fp, _content = self._resolve_and_read(file_path)
            exists = _content is not None
        if exists:
            return ErrorCategory.PRE_EXISTING.value  # untouched existing file
        return ErrorCategory.UNKNOWN.value           # new file → allow create

    def _guard_protected_file(self, file_path: str, existing_content: Optional[str] = None) -> Optional[str]:
        """Refuse edits to pre-existing files our ticket never touched.

        Returns a rejection message when blocked, else None. This is what
        prevents the resolver from 'fixing' unrelated baseline errors (e.g.
        issue.ts) with casts/suppressions.
        """
        # Hard TicketScopeProof Gate: Compiler diagnostics are evidence, not write authorization
        if getattr(self, "scope_proof", None) is not None and hasattr(self.scope_proof, "is_file_writable"):
            if not self.scope_proof.is_file_writable(file_path, workspace_root=self.workspace_path):
                role_val = "unauthorized"
                if hasattr(self.scope_proof, "get_file_role"):
                    try:
                        role_val = self.scope_proof.get_file_role(file_path, workspace_root=self.workspace_path).value
                    except Exception:
                        role_val = "unauthorized"
                logger.warning(
                    f"  🛡️ SCOPE_PROOF_VIOLATION in ErrorResolutionAgent: '{file_path}' (role: {role_val}) "
                    f"is not in approved writable scope."
                )
                return (
                    f"SCOPE_PROOF_VIOLATION: File '{file_path}' (role: {role_val}) is not in approved writable scope. "
                    f"Compiler diagnostics are evidence, not write authorization. "
                    f"You are strictly forbidden from editing this file. Adapt the authorized consumer file instead."
                )

        from ticket_to_code.models import ErrorCategory

        category = self.classify_error_file(file_path, existing_content=existing_content)
        if category == ErrorCategory.PRE_EXISTING.value:
            norm = file_path.replace("\\", "/").lower()
            if self._is_user_authorized(norm):
                logger.info(
                    f"  ✅ ErrorAttribution: user AUTHORIZED fixing pre-existing file '{file_path}'."
                )
                return None
            # Companion authorization: if the file is a companion repository, test, or migration, authorize it
            is_companion = (
                norm.endswith("repository.java") or
                norm.endswith("test.java") or
                norm.endswith(".sql")
            )
            if is_companion:
                logger.info(
                    f"  ✅ ErrorAttribution: AUTHORIZED companion file '{file_path}' for error resolution."
                )
                return None
            logger.warning(
                f"  🛡️ ErrorAttribution: BLOCKED edit to PRE_EXISTING file "
                f"'{file_path}' — its errors predate this ticket and must not be changed."
            )
            # Record audit entry for protected provider rejection
            try:
                from ticket_to_code.agents.diagnostic_localizer import RepairAuditRecord, FailureOwner
                if hasattr(self, "audit_trail"):
                    self.audit_trail.append(
                        RepairAuditRecord(
                            diagnostic=f"Blocked edit to protected provider '{file_path}'",
                            diagnostic_fingerprint=f"blocked_edit_{norm}",
                            owner=FailureOwner.CROSS_FILE_CONTRACT,
                            authorization_result="BLOCKED_PROTECTED_PROVIDER",
                            attempted_repair_target=file_path,
                            reason_for_rejection=f"File '{file_path}' is PRE_EXISTING / protected and unauthorized for modification.",
                            alternative_api_searched=True,
                            alternative_api_found=None,
                            final_unresolved_reason="Protected provider is immutable. Cannot modify provider without authorization.",
                        )
                    )
            except Exception:
                pass

            # Inspect real public methods to guide consumer adaptation
            alt_guidance = ""
            try:
                from ticket_to_code.agents.diagnostic_localizer import DiagnosticLocalizer
                loc = DiagnosticLocalizer(self.workspace_path)
                inspected = loc.inspect_provider_public_api(file_path)
                if inspected:
                    alt_guidance = (
                        f"\nReal public methods available on protected '{file_path}':\n"
                        + "\n".join(f"  - {m.signature}" for m in inspected[:6])
                        + "\nAdapt the caller/consumer to call an existing verified method instead of modifying this protected file."
                    )
            except Exception:
                pass

            return (
                f"BLOCKED: '{file_path}' was NOT modified or generated by this ticket. "
                f"Its errors are PRE_EXISTING and it is a PROTECTED provider. "
                f"Do NOT edit it (no new methods, casts, <any>, suppressions, or refactors). "
                f"Protected providers must remain immutable.{alt_guidance}\n"
                f"Skip editing this file and adapt the ticket-owned consumer file instead. "
                f"If a consumer component failed with TS2339 (property does not exist on provider constant/enum/class), "
                f"do NOT add properties to '{file_path}'. Modify the caller/consumer file to use its existing exported properties "
                f"or existing routes (e.g., RoutingConstant.HOME_PATH or '/dashboard'). "
                f"For TypeScript type/interface mismatches: in the consumer file, extend the interface locally "
                f"(e.g. `interface LocalType extends ... {{ extraProp?: boolean }}`), use type intersection "
                f"(`Type & {{ extraProp?: boolean }}`), or use type casting instead of modifying the protected provider."
            )
        return None

    def _tool_edit_file(self, file_path: str, content: str) -> str:
        logger.info(f"Tool EDIT_FILE: {file_path} ({len(content)} chars)")
        _blocked = self._guard_protected_file(file_path)
        if _blocked:
            return _blocked
        if not content.strip():
            return "Error: Content provided was empty."

        fp = file_path.replace("\\", "/")
        norm_key = fp.lower()
        self._file_case_map[norm_key] = fp
        self.modified_files[norm_key] = content
        return f"Successfully updated (in memory): {fp}"

    def _tool_replace_content(self, file_path: str, target_content: str, replacement_content: str) -> str:
        """Apply a single str_replace edit using the 6-Tier Matching Cascade."""
        logger.info(f"Tool REPLACE_CONTENT: {file_path}")
        fp, content = self._resolve_and_read(file_path)
        if content is None:
            return f"Error: Could not read file '{file_path}'. It may not exist."

        _blocked = self._guard_protected_file(file_path, existing_content=content)
        if _blocked:
            return _blocked

        # ── v4: Staleness detection ───────────────────────────────────────
        norm_key = fp.replace('\\', '/').lower()
        current_hash = self._content_hash(content)
        stored_hash = self._file_versions.get(norm_key)
        if stored_hash and stored_hash != current_hash:
            logger.info(f"  [v4] File {fp} has changed since last seen (stale context)")

        # ── Delegate to 6-Tier Matching Cascade ───────────────────────────
        try:
            from ticket_to_code.agents.code_generator import _apply_str_replace_edits
            edits = [{"old_str": target_content, "new_str": replacement_content}]
            new_content = _apply_str_replace_edits(edits, content, fp)
        except ValueError as exc:
            stale_hint = ""
            if stored_hash and stored_hash != current_hash:
                stale_hint = (
                    " NOTE: This file was modified since you last read it. "
                    "Use READ_FILE to get the current contents before retrying."
                )
            return f"Error: {exc}{stale_hint}"

        self._file_case_map[norm_key] = fp
        self.modified_files[norm_key] = new_content
        self._file_versions[norm_key] = self._content_hash(new_content)
        return f"Successfully updated (in memory): {fp}"


    def _tool_replace_multi(self, file_path: str, replacements: list[dict]) -> str:
        """Apply MULTIPLE replacements using strict all-or-nothing atomicity.

        Invariant: Never partially apply a multi-edit patch. If ANY replacement fails,
        the entire batch is rejected and the file remains completely unchanged.
        """
        logger.info(f"Tool REPLACE_MULTI: {file_path} ({len(replacements)} replacements)")
        fp, content = self._resolve_and_read(file_path)
        if content is None:
            return f"Error: Could not read file '{file_path}'. It may not exist."

        _blocked = self._guard_protected_file(file_path, existing_content=content)
        if _blocked:
            return _blocked

        from ticket_to_code.agents.code_generator import _apply_str_replace_edits

        edits = []
        for repl in replacements:
            target = repl.get("target", "")
            replacement = repl.get("replacement", "")
            if target:
                edits.append({"old_str": target, "new_str": replacement})

        if not edits:
            return f"REPLACE_MULTI: no valid edits provided for {fp}"

        # ── Atomic batch (all-or-nothing): NEVER partially apply ───────────
        try:
            new_content = _apply_str_replace_edits(edits, content, fp)
            norm_key = fp.replace('\\', '/').lower()
            self._file_case_map[norm_key] = fp
            self.modified_files[norm_key] = new_content
            self._file_versions[norm_key] = self._content_hash(new_content)
            return (
                f"REPLACE_MULTI: all {len(edits)} edits applied successfully for {fp}"
            )
        except ValueError as exc:
            logger.warning(
                f"  ❌ REPLACE_MULTI atomic batch failed for {fp}: {exc}. "
                f"Rejecting entire batch — existing file remains untouched."
            )
            return (
                f"REPLACE_MULTI REJECTED: Atomic batch failed ({exc}). "
                f"None of the {len(edits)} edits were applied and {fp} remains completely untouched. "
                f"Re-read the file with READ_FILE to obtain fresh source context before retrying."
            )


    def _tool_compile(self) -> str:
        """Run the project compiler and return fresh error output.

        First writes all in-memory modifications to disk so the compiler sees them,
        then runs the appropriate compiler (tsc+Angular AOT, Gradle, or Maven).
        """
        logger.info(f"Tool COMPILE: Running project compiler (tool={self.compile_tool})...")

        if not self.compile_root:
            return "Error: No compile root configured. Cannot run compiler."

        # Write in-memory modifications to disk before compiling via WorkspaceIntegrity
        for fp_lower, content in self.modified_files.items():
            orig_fp = self._file_case_map.get(fp_lower, fp_lower)
            fp = orig_fp.replace("/", "\\" if sys.platform == "win32" else "/")
            abs_path = self.workspace_path / fp
            if not abs_path.is_absolute():
                abs_path = self.workspace_path / fp
            try:
                from ticket_to_code.integrity.workspace_integrity import get_workspace_integrity_manager
                mgr = get_workspace_integrity_manager(self.workspace_path)
                mgr.safe_write(
                    file_path=orig_fp,
                    new_content_or_edits=content,
                    ticket_id=getattr(self, "ticket_id", "") or "",
                    task_id="error_resolution",
                    requirement_id="",
                )
                logger.info(f"  COMPILE: Wrote {orig_fp} safely via WorkspaceIntegrity")
            except Exception as e:
                logger.warning(f"  COMPILE: Integrity safe_write fallback for {orig_fp}: {e}")
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(content, encoding="utf-8")


        use_shell = sys.platform == "win32"

        if self.compile_tool == "gradle":
            res = self._compile_gradle(use_shell)
        elif self.compile_tool == "maven":
            res = self._compile_maven(use_shell)
        else:
            res = self._compile_tsc_and_angular(use_shell)

        # ── Oscillation & Stagnation Detection ────────────────────────────
        try:
            from ticket_to_code.verification.oscillation_detector import (
                NormalizedDiagnostic,
            )
            raw_lines = [l.strip() for l in res.splitlines() if l.strip()]
            diags = [
                NormalizedDiagnostic.create(
                    file=self.compile_root.name if self.compile_root else "project",
                    code="COMPILER_ERR",
                    raw_message=l,
                )
                for l in raw_lines if any(k in l.lower() for k in ("error", "failed", "cannot find", "not found"))
            ]
            if diags and hasattr(self, "oscillation_detector") and self.oscillation_detector:
                eval_res = self.oscillation_detector.evaluate(diags)
                if eval_res.requires_rollback:
                    logger.warning(f"  🛑 [OscillationDetector] Tripped: {eval_res.diagnostic_summary}")
                    res = (
                        f"🛑 REPAIR LOOP HALTED (Oscillation/Stagnation Detected):\n"
                        f"{eval_res.diagnostic_summary}\n\n"
                        f"Targeted error output:\n{res}"
                    )
        except Exception as _osc_err:
            logger.debug(f"Oscillation check skipped: {_osc_err}")

        return res

    def _compile_tsc_and_angular(self, use_shell: bool) -> str:
        """Run tsc --noEmit AND Angular AOT (ng build) to catch template errors."""
        # Step 1: tsc --noEmit (fast check for .ts errors)
        tsconfig_app = self.compile_root / "tsconfig.app.json"
        if tsconfig_app.exists():
            cmd: Any = (
                "npx --no-install tsc --noEmit --pretty false -p tsconfig.app.json"
                if use_shell
                else ["npx", "--no-install", "tsc", "--noEmit", "--pretty", "false", "-p", "tsconfig.app.json"]
            )
        else:
            cmd = (
                "npx --no-install tsc --noEmit --pretty false"
                if use_shell
                else ["npx", "--no-install", "tsc", "--noEmit", "--pretty", "false"]
            )

        try:
            result = subprocess.run(
                cmd, cwd=str(self.compile_root),
                capture_output=True, text=True, timeout=180,
                shell=use_shell, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return "COMPILE: tsc timed out after 180 seconds."
        except Exception as e:
            return f"COMPILE ERROR (tsc): {e}"

        if result.returncode != 0:
            return self._format_compile_errors(result)

        # Step 2: Angular AOT template type-check (catches .html template errors)
        angular_json = self.compile_root / "angular.json"
        if angular_json.exists():
            logger.info("  COMPILE: tsc OK, running Angular AOT template check (npm run build)...")
            ng_cmd: Any = "npm run build" if use_shell else ["npm", "run", "build"]
            try:
                ng_result = subprocess.run(
                    ng_cmd, cwd=str(self.compile_root),
                    capture_output=True, text=True, timeout=300,
                    shell=use_shell, encoding="utf-8", errors="replace",
                )
            except subprocess.TimeoutExpired:
                return "COMPILE: Angular AOT timed out after 300 seconds."
            except Exception as e:
                return f"COMPILE ERROR (ng build): {e}"

            if ng_result.returncode != 0:
                # Extract only error lines from Angular build output
                raw = (ng_result.stdout + "\n" + ng_result.stderr).strip()
                ansi_rx = re.compile(r'\x1b\[[0-9;]*m')
                clean = ansi_rx.sub('', raw)
                all_lines = clean.splitlines()
                ng_errors = [
                    line for line in all_lines
                    if ("Error:" in line or "error TS" in line or "error NG" in line)
                ]
                if not ng_errors:
                    ng_errors = [line for line in all_lines if line.strip() and not line.startswith("Warning:")]

                logger.info(f"  COMPILE: ❌ Angular AOT FAILED — {len(ng_errors)} error(s)")
                truncated = ng_errors[:40]
                remaining_msg = ""
                if len(ng_errors) > 40:
                    remaining_msg = f"\n... and {len(ng_errors) - 40} more errors"
                return (
                    f"❌ Angular AOT BUILD FAILED — {len(ng_errors)} template error(s) remaining:\n"
                    + "\n".join(truncated)
                    + remaining_msg
                    + "\n\nThese are TEMPLATE errors in .html files. Fix the model/interface "
                    + "to add missing properties, then COMPILE again."
                )

        logger.info("  COMPILE: ✅ BUILD SUCCESS — 0 errors (tsc + Angular AOT)!")
        return "✅ BUILD SUCCESS — 0 errors! You can now call FINISHED."

    def _compile_gradle(self, use_shell: bool) -> str:
        """Run Gradle compileJava."""
        gradlew = self.compile_root / ("gradlew.bat" if sys.platform == "win32" else "gradlew")
        if gradlew.exists():
            cmd: Any = (f"{gradlew} compileJava --no-daemon" if use_shell
                        else [str(gradlew), "compileJava", "--no-daemon"])
        else:
            cmd = ("gradle compileJava --no-daemon" if use_shell
                   else ["gradle", "compileJava", "--no-daemon"])
        try:
            result = subprocess.run(
                cmd, cwd=str(self.compile_root),
                capture_output=True, text=True, timeout=300,
                shell=use_shell, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return "COMPILE: Gradle timed out after 300 seconds."
        except Exception as e:
            return f"COMPILE ERROR (gradle): {e}"

        if result.returncode == 0:
            logger.info("  COMPILE: ✅ Gradle BUILD SUCCESS — 0 errors!")
            return "✅ BUILD SUCCESS — 0 errors! You can now call FINISHED."
        return self._format_compile_errors(result)

    def _compile_maven(self, use_shell: bool) -> str:
        """Run Maven compile."""
        cmd: Any = ("mvn compile -q" if use_shell else ["mvn", "compile", "-q"])
        try:
            result = subprocess.run(
                cmd, cwd=str(self.compile_root),
                capture_output=True, text=True, timeout=300,
                shell=use_shell, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return "COMPILE: Maven timed out after 300 seconds."
        except Exception as e:
            return f"COMPILE ERROR (maven): {e}"

        if result.returncode == 0:
            logger.info("  COMPILE: ✅ Maven BUILD SUCCESS — 0 errors!")
            return "✅ BUILD SUCCESS — 0 errors! You can now call FINISHED."
        return self._format_compile_errors(result)

    def _format_compile_errors(self, result) -> str:
        """Format compiler output into a structured error message."""
        raw_output = (result.stdout + "\n" + result.stderr).strip()
        ansi_rx = re.compile(r'\x1b\[[0-9;]*m')
        clean_output = ansi_rx.sub('', raw_output)
        error_lines = [line for line in clean_output.splitlines() if line.strip()]

        logger.info(f"  COMPILE: ❌ {len(error_lines)} error line(s) remaining")
        truncated = error_lines[:40]
        remaining_msg = ""
        if len(error_lines) > 40:
            remaining_msg = f"\n... and {len(error_lines) - 40} more errors (fix the above first)"

        return (
            f"❌ BUILD FAILED — {len(error_lines)} error(s) remaining:\n"
            + "\n".join(truncated)
            + remaining_msg
            + "\n\nFix these errors and COMPILE again."
        )

    # ── v3 Tools: Component structure + Quick check ────────────────────────

    def _tool_read_component(self, file_path: str) -> str:
        """READ_COMPONENT: Get full component structure (related files, methods, bindings, gaps)."""
        if not self.component_provider:
            return "READ_COMPONENT unavailable: ComponentStructureProvider not initialized."

        try:
            rel_path = file_path.replace("\\", "/")
            structure = self.component_provider.get_component_structure(rel_path)
            result = structure.to_prompt_block()
            logger.info(f"  READ_COMPONENT: {file_path} → {len(structure.related_files)} related, {len(structure.methods)} methods, {len(structure.data_gaps)} gaps")
            return result
        except Exception as exc:
            logger.warning(f"  READ_COMPONENT failed for {file_path}: {exc}")
            return f"READ_COMPONENT error: {exc}"

    def _tool_quick_check(self) -> str:
        """QUICK_CHECK: Run fast type-check only (no full build/AOT).

        For TypeScript: tsc --noEmit (~10 seconds)
        For Java/Gradle: gradle compileJava (~15 seconds)
        For Java/Maven: mvn compile -q (~15 seconds)
        For Python: py_compile + mypy (~5 seconds)
        """
        # First, flush any pending file modifications to disk via WorkspaceIntegrity
        for fp_lower, content in self.modified_files.items():
            orig_fp = self._file_case_map.get(fp_lower, fp_lower)
            fp = orig_fp.replace("/", "\\" if sys.platform == "win32" else "/")
            abs_path = self.workspace_path / fp
            try:
                from ticket_to_code.integrity.workspace_integrity import get_workspace_integrity_manager
                mgr = get_workspace_integrity_manager(self.workspace_path)
                mgr.safe_write(
                    file_path=orig_fp,
                    new_content_or_edits=content,
                    ticket_id=getattr(self, "ticket_id", "") or "",
                    task_id="error_resolution",
                    requirement_id="",
                )
            except Exception as exc:
                logger.warning(f"  QUICK_CHECK: Integrity safe_write fallback for {orig_fp}: {exc}")
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(content, encoding="utf-8")


        use_shell = sys.platform == "win32"

        if self.compile_tool == "tsc":
            return self._quick_check_tsc(use_shell)
        elif self.compile_tool == "gradle":
            return self._quick_check_gradle(use_shell)
        elif self.compile_tool == "maven":
            return self._quick_check_maven(use_shell)
        else:
            return self._quick_check_tsc(use_shell)

    def _quick_check_tsc(self, use_shell: bool) -> str:
        """Fast TypeScript type check: tsc --noEmit only (no Angular AOT)."""
        compile_root = self.compile_root or self.workspace_path
        tsc_path = compile_root / "node_modules" / ".bin" / ("tsc.cmd" if sys.platform == "win32" else "tsc")

        project_flag = ""
        project_args: list[str] = []
        if (compile_root / "tsconfig.app.json").exists():
            project_flag = " -p tsconfig.app.json"
            project_args = ["-p", "tsconfig.app.json"]
        elif (compile_root / "tsconfig.json").exists():
            project_flag = " -p tsconfig.json"
            project_args = ["-p", "tsconfig.json"]

        if tsc_path.exists():
            cmd = f"\"{tsc_path}\"{project_flag} --noEmit" if use_shell else ([str(tsc_path)] + project_args + ["--noEmit"])
        else:
            cmd = f"npx tsc{project_flag} --noEmit" if use_shell else (["npx", "tsc"] + project_args + ["--noEmit"])

        logger.info(f"  QUICK_CHECK: Running tsc --noEmit (fast type-check)...")
        try:
            result = subprocess.run(
                cmd, cwd=str(compile_root),
                capture_output=True, text=True, timeout=60,
                shell=use_shell, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return "QUICK_CHECK: tsc timed out after 60 seconds."
        except Exception as e:
            return f"QUICK_CHECK ERROR: {e}"

        if result.returncode == 0:
            logger.info("  QUICK_CHECK: ✅ Type check passed — 0 errors")
            return "✅ QUICK_CHECK PASSED — 0 type errors. Run COMPILE for full build verification (catches template/AOT errors too)."

        return self._format_compile_errors(result).replace("BUILD FAILED", "QUICK_CHECK FAILED (type errors)")

    def _quick_check_gradle(self, use_shell: bool) -> str:
        """Fast Gradle compile check."""
        compile_root = self.compile_root or self.workspace_path
        gradlew = compile_root / ("gradlew.bat" if sys.platform == "win32" else "gradlew")

        if gradlew.exists():
            cmd = f"\"{gradlew}\" compileJava --no-daemon -q" if use_shell else [str(gradlew), "compileJava", "--no-daemon", "-q"]
        else:
            cmd = "gradle compileJava --no-daemon -q" if use_shell else ["gradle", "compileJava", "--no-daemon", "-q"]

        logger.info(f"  QUICK_CHECK: Running Gradle compileJava (fast)...")
        try:
            result = subprocess.run(
                cmd, cwd=str(compile_root),
                capture_output=True, text=True, timeout=120,
                shell=use_shell, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return "QUICK_CHECK: Gradle timed out after 120 seconds."
        except Exception as e:
            return f"QUICK_CHECK ERROR: {e}"

        if result.returncode == 0:
            logger.info("  QUICK_CHECK: ✅ Gradle compile passed")
            return "✅ QUICK_CHECK PASSED — 0 compile errors. Run COMPILE for full build verification."

        return self._format_compile_errors(result).replace("BUILD FAILED", "QUICK_CHECK FAILED")

    def _quick_check_maven(self, use_shell: bool) -> str:
        """Fast Maven compile check."""
        compile_root = self.compile_root or self.workspace_path
        cmd = "mvn compile -q" if use_shell else ["mvn", "compile", "-q"]

        logger.info(f"  QUICK_CHECK: Running Maven compile (fast)...")
        try:
            result = subprocess.run(
                cmd, cwd=str(compile_root),
                capture_output=True, text=True, timeout=120,
                shell=use_shell, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return "QUICK_CHECK: Maven timed out after 120 seconds."
        except Exception as e:
            return f"QUICK_CHECK ERROR: {e}"

        if result.returncode == 0:
            logger.info("  QUICK_CHECK: ✅ Maven compile passed")
            return "✅ QUICK_CHECK PASSED — 0 compile errors. Run COMPILE for full build verification."

        return self._format_compile_errors(result).replace("BUILD FAILED", "QUICK_CHECK FAILED")
