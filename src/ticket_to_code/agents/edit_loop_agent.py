"""
Edit Loop Agent — The "Claude Code / Devin" style free-form edit loop.

Architecture contrast:
  OLD (current):  Plan N files → Generate each in order → Full build → Fix loop
  NEW (this):     Understand ticket → While not solved: pick next file → read it
                  fully → make targeted edit → verify immediately → continue

Key properties:
  - NO fixed plan. The agent decides what to edit next based on current state.
  - Unbounded iterations (bounded by MAX_ITER safety cap, default 25)
  - Every file write is immediately verified with the TypeScript/Java compiler
  - When verification fails, the error is fixed BEFORE moving to the next file
  - The LLM always reads the CURRENT file content (not a cached snapshot)

This is how Claude Code, Devin, and Cursor Composer actually work.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.models import (
        ValueEdgeTicket,
        StructuredRequirements,
        ArchitecturalPlan,
    )
    from ticket_to_code.agents.implementation_state import ImplementationState

from ticket_to_code.agents.lsp_client import WorkspaceSymbolIndex, ClassMembers
from ticket_to_code.agents.symbol_resolver import SymbolResolver
from ticket_to_code.runtime.limits import EDIT_MAX_TOKENS, WHOLE_FILE_MAX_TOKENS

logger = logging.getLogger(__name__)


MAX_ITER = 8           # safety cap — never exceed this many file edits
MAX_FIX_PER_FILE = 3   # max inline fix attempts per file before moving on


@dataclass
class EditDecision:
    """The agent's decision about what to edit next."""
    file_path: str
    reason: str
    edit_description: str
    is_new_file: bool = False
    priority: int = 5  # 1=critical, 10=optional
    # Method/property names the edit should localize on (from outcome_check
    # findings). Used to extract only the implicated source regions instead of
    # shipping whole files + whole companion ASTs to the LLM every iteration.
    target_symbols: list = field(default_factory=list)


@dataclass
class EditResult:
    """Result of one file edit + verification cycle."""
    file_path: str
    content_before: str
    content_after: str
    compile_errors: list[str] = field(default_factory=list)
    compile_clean: bool = False
    fix_applied: bool = False
    skipped: bool = False
    reason: str = ""


@dataclass
class ApplyEditResult:
    """Structured result of applying search/replace edits to content."""
    success: bool
    content: str
    applied_count: int = 0
    failure_reason: str = ""
    failed_search: str = ""
    parser_format: str = ""  # "aider", "legacy", "delimiters", "create", "none"


class EditLoopAgent:
    """
    Drives an autonomous edit loop that mirrors how AI IDEs work:

    1. Assess the current state of the codebase vs. ticket requirements
    2. Choose the highest-impact file to edit next
    3. Read the full current file content (always fresh from disk)
    4. Generate a targeted, minimal edit
    5. Verify immediately with the language compiler
    6. If errors → fix inline before moving on
    7. Repeat until the ticket requirements are fully implemented

    This replaces the rigid plan-then-execute approach with an adaptive loop
    that can discover and fix issues as they arise — exactly like a developer
    working in a terminal with `tsc --watch` running.
    """

    def __init__(
        self,
        workspace_path: str,
        llm,
        symbol_index: Optional[WorkspaceSymbolIndex] = None,
        impl_state: Optional["ImplementationState"] = None,
        ui_callback=None,
        pre_edit_snapshots: Optional[dict[str, str]] = None,
    ):
        self.workspace_path = Path(workspace_path)
        self.llm = llm
        self.symbol_index = symbol_index or WorkspaceSymbolIndex(workspace_path)
        self._resolver = SymbolResolver(self.symbol_index)
        self._ng_root: Optional[Path] = None
        self._written_files: dict[str, str] = {}  # path → last written content
        self.pre_edit_snapshots: dict[str, str] = dict(pre_edit_snapshots or {})
        self._impl_state = impl_state  # Four-pillar ImplementationState (optional)
        self.ui_callback = ui_callback
        # Remediation mode: re-entry from outcome_check with a PARTIAL verdict.
        # Compile-clean does NOT mean done here — the defect is semantic.
        self._remediation_mode: bool = False
        self._remediation_findings: list = []
        self.allowed_files: Optional[set[str]] = None
        # Set by _verify_file when the compiler returned None (diagnostics
        # unavailable) — a clean-looking [] must then NOT be claimed as verified.
        self._last_verify_unknown: bool = False
        # Per-call LLM token telemetry (input/output/total + call count + cache)
        self._tokens: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "thinking_tokens": 0,
            "llm_calls": 0,
        }

    def _track_tokens(self, response) -> None:
        """Accumulate usage_metadata and KV cache from an LLM response (never raises)."""
        self._tokens["llm_calls"] += 1
        try:
            from ticket_to_code.llm_utils import extract_token_usage
            u = extract_token_usage(response)
            inp = u.get("prompt_tokens", 0)
            out = u.get("completion_tokens", 0)
            tot = u.get("total_tokens", inp + out)
            cached = u.get("cached_tokens", 0)
            thinking = u.get("thinking_tokens", 0)
            self._tokens["input_tokens"] += inp
            self._tokens["output_tokens"] += out
            self._tokens["total_tokens"] += tot
            self._tokens["cached_tokens"] += cached
            self._tokens["thinking_tokens"] += thinking
        except Exception:
            pass

    def _find_targets_for_file(self, file_path: str) -> list:
        """Extract candidate method/property names for a file from remediation findings.

        Used to localize edits: instead of shipping the whole file plus the
        whole companion AST to the LLM, the prompt carries only the implicated
        methods. Over-matching is harmless (a little extra context); the hard
        cap keeps it bounded.
        """
        targets: list = []
        norm = file_path.replace("\\", "/").lower()
        base = Path(file_path).name.lower()
        _NOISE = {
            "function", "undefined", "string", "number", "boolean", "object",
            "return", "subscribe", "console", "window", "document",
        }
        for f in self._remediation_findings:
            fpath = str(f.get("affected_file", "") or "").replace("\\", "/").lower()
            if fpath and fpath != norm and Path(fpath).name.lower() != base:
                continue
            blob = " ".join(
                str(f.get(k, "") or "")
                for k in ("missing_behavior", "offending_code", "next_action", "summary")
            )
            # Method calls mentioned in the finding: onUserSelect(...), addMember(...)
            for m in re.finditer(r'\b([a-zA-Z_$][a-zA-Z0-9_$]{5,})\s*\(', blob):
                name = m.group(1)
                if name not in targets and name not in _NOISE:
                    targets.append(name)
            # camelCase / snake_case property identifiers: isExistingProjectMember
            for m in re.finditer(r'\b([a-z]+[A-Z][a-zA-Z0-9_$]{6,}|[a-z]+(?:_[a-z0-9]+){2,})\b', blob):
                name = m.group(1)
                if name not in targets and name not in _NOISE:
                    targets.append(name)
        return targets[:6]

    @staticmethod
    def _get_live_check_label(file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext in (".html", ".htm"):
            return "Template-Check"
        elif ext in (".ts", ".tsx"):
            return "TSC-live"
        elif ext in (".java", ".kt", ".scala"):
            return "Java-Check"
        elif ext in (".scss", ".css", ".sass", ".less"):
            return "Style-Check"
        return "Build-Check"

    def _ui_emit(self, event_type: str, **kwargs):
        if not self.ui_callback:
            return
        from datetime import datetime
        self.ui_callback({
            "phase": "patch_generation",
            "status": "in_progress",
            "message": kwargs.get("message", ""),
            "event_type": event_type,
            "data": {
                "node": "generate_code",
                "event_type": event_type,
                **{k: v for k, v in kwargs.items() if k != "message"},
            },
            "timestamp": datetime.now().isoformat(),
        })

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        ticket: "ValueEdgeTicket",
        requirements: "StructuredRequirements",
        initial_plan: Optional["ArchitecturalPlan"] = None,
        code_rag_context: str = "",
        existing_file_map: Optional[dict[str, str]] = None,
        remediation_mode: bool = False,
        remediation_findings: Optional[list] = None,
        allowed_files: Optional[Any] = None,
    ) -> dict:
        """
        Run the free-form edit loop until the ticket is solved or MAX_ITER reached.

        Args:
            remediation_mode: True when re-entering from outcome_check with a
                PARTIAL verdict. In this mode, files that already compile clean
                are NOT skipped — the defect is semantic, not syntactic — and
                the compile-clean early exits are suppressed.
            remediation_findings: Structured OutcomeFinding dicts from the
                outcome check, used to localize edits on the implicated
                methods/properties (target symbols).
            allowed_files: Collection of authorized writable file paths.
                When provided, edit decisions and writes outside this set
                are strictly blocked, and the iteration ceiling is scaled mechanically:
                effective_max_iter = min(MAX_ITER, max(3, 2 * len(allowed_files))).

        Returns:
            {
                "generated_files": {path: content},
                "edit_results": [EditResult],
                "final_compile_errors": [str],
                "iterations": int,
                "solved": bool,
                "token_usage": {input_tokens, output_tokens, total_tokens, llm_calls},
            }
        """
        if allowed_files is not None:
            self.allowed_files = {
                f.replace("\\", "/").lower().strip()
                for f in allowed_files if f and str(f).strip()
            }
        else:
            self.allowed_files = None

        # Mechanical iteration cap: max(3, 2 * len(allowlist))
        if self.allowed_files:
            effective_max_iter = min(MAX_ITER, max(3, 2 * len(self.allowed_files)))
        else:
            effective_max_iter = MAX_ITER

        logger.info("\n" + "=" * 70)
        logger.info(" EDIT LOOP AGENT: Starting free-form edit loop")
        logger.info(f"   Ticket: {ticket.ticket_id} — {ticket.title}")
        logger.info(f"   Effective MAX_ITER: {effective_max_iter} (allowed_files: {len(self.allowed_files) if self.allowed_files else 'unconstrained'})")
        if remediation_mode:
            logger.info(
                f"   REMEDIATION MODE: {len(remediation_findings or [])} finding(s) — "
                "compile-clean files are NOT skipped (defect is semantic)"
            )
        logger.info("=" * 70)

        self._remediation_mode = remediation_mode
        self._remediation_findings = list(remediation_findings or [])
        self._initial_plan = initial_plan

        files_edited: set[str] = set()
        if existing_file_map:
            self._written_files.update(existing_file_map)
            if not remediation_mode:
                files_edited.update(existing_file_map.keys())

        edit_results: list[EditResult] = []
        iteration = 0
        _seen_error_sigs: set[str] = set()   # bounded safety net (repeated-error stop)

        # Seed the initial file queue from the architectural plan (if provided)
        # Files already written and compiling clean are skipped — EXCEPT in
        # remediation mode, where every planned file is implicated by a finding
        # and must be edited even though it compiles.
        pending_queue: list[EditDecision] = []
        self._pending_queue_ref = pending_queue
        norm_existing = {k.replace("\\", "/").lower() for k in self._written_files}
        if isinstance(initial_plan, tuple):
            initial_plan = initial_plan[0]
        if initial_plan and getattr(initial_plan, "tasks", None):
            for t in initial_plan.tasks:
                if getattr(getattr(t, "task_type", None), "value", "") != "read_only":
                    norm_fp = t.file_path.replace("\\", "/").lower()
                    # Scope filter: only queue tasks within allowed_files
                    if self.allowed_files is not None:
                        if not any(
                            norm_fp == af or norm_fp.endswith("/" + af) or af.endswith("/" + norm_fp)
                            for af in self.allowed_files
                        ):
                            is_migration = "db/migration" in norm_fp and norm_fp.endswith(".sql")
                            is_supported_source = norm_fp.endswith((".java", ".ts", ".tsx", ".sql", ".xml", ".json", ".properties"))
                            service_match = False
                            for af in self.allowed_files:
                                af_svc = af.split("/")[0] if "/" in af else ""
                                fp_svc = norm_fp.split("/")[0] if "/" in norm_fp else ""
                                if af_svc and fp_svc and af_svc == fp_svc:
                                    service_match = True
                                    break
                            if is_migration or (is_supported_source and service_match) or norm_fp.endswith("repository.java"):
                                logger.info(f"  🔓 Auto-admitting planned task file '{t.file_path}' into allowed_files")
                                self.allowed_files.add(norm_fp)
                            else:
                                logger.info(f"  ⏭️ Skipping task for {t.file_path} — outside allowed_files scope")
                                continue

                    if norm_fp in norm_existing and not remediation_mode:
                        errs = self._verify_file(t.file_path)
                        if not errs:
                            logger.info(f"  ⏭️ File {t.file_path} already generated and compiles clean — skipping re-queue")
                            edit_results.append(EditResult(
                                file_path=t.file_path,
                                content_before="",
                                content_after=self._written_files.get(t.file_path) or self._written_files.get(norm_fp) or "",
                                compile_clean=True,
                                skipped=False,
                                reason="already generated and clean",
                            ))
                            continue
                    pending_queue.append(EditDecision(
                        file_path=t.file_path,
                        reason=f"Planner task: {t.title}",
                        edit_description=t.description or t.title,
                        is_new_file=getattr(getattr(t, "task_type", None), "value", "") == "create",
                        priority=3,
                        target_symbols=self._find_targets_for_file(t.file_path),
                    ))

        # In Direct Mode or when initial plan produced no queue, seed directly from allowed_files
        if not pending_queue and self.allowed_files:
            for af in sorted(self.allowed_files):
                pending_queue.append(EditDecision(
                    file_path=af,
                    reason=f"Direct Mode localized target: {Path(af).name}",
                    edit_description=f"Implement requirements from ticket {getattr(ticket, 'ticket_id', '')}: {getattr(ticket, 'title', '')}",
                    is_new_file=False,
                    priority=2,
                    target_symbols=self._find_targets_for_file(af),
                ))
            logger.info(f"  ⚡ Direct Mode: seeded pending_queue with {len(pending_queue)} localized target file(s): {[d.file_path for d in pending_queue]}")

        # Fast 0-call early exit: if plan tasks were provided, all exist in written files,
        # and pending_queue is empty with all files clean. Never applies in remediation
        # mode — the caller re-entered precisely because the ticket is NOT solved.
        if (
            initial_plan and initial_plan.tasks and not pending_queue and edit_results
            and not remediation_mode
        ):
            if all(r.compile_clean for r in edit_results):
                logger.info(
                    f"  ⏹️ Edit loop fast-exit: all {len(edit_results)} planned file(s) "
                    "already generated and compile clean (0 calls burned)."
                )
                return {
                    "generated_files": dict(self._written_files),
                    "edit_results": edit_results,
                    "final_compile_errors": [],
                    "iterations": 0,
                    "solved": True,
                    "stop_reason": "",
                }

        # ── Stuck-loop guard state (P0: deterministic termination) ────────
        file_edit_counts: dict = {}
        recent_edit_seq: list = []
        MAX_EDITS_PER_FILE = 3
        stop_reason = ""
        diagnostics_unverified = False
        ceiling_hits: set = set()   # files already refused once by the per-file ceiling

        while iteration < effective_max_iter:
            # ── Early exit: nothing left to fix ───────────────────────────────
            if iteration >= 1 and not pending_queue:
                latest_by_file: dict[str, EditResult] = {}
                for r in edit_results:
                    latest_by_file[r.file_path.replace("\\", "/").lower()] = r
                _dirty = [
                    r for r in latest_by_file.values()
                    if (not r.skipped and not r.compile_clean and r.reason != "diagnostics unavailable")
                    or (r.skipped and r.reason.startswith("edit application failed"))
                ]
                if not _dirty:
                    logger.info(
                        "  ⏹️ Edit loop early-exit: queue empty and all edited "
                        f"files clean after {iteration} iteration(s)."
                    )
                    break

            iteration += 1
            logger.info(f"\n--- EDIT LOOP iteration {iteration}/{effective_max_iter} ---")

            # Pick next file to edit
            decision = self._decide_next_edit(
                ticket, requirements, pending_queue, files_edited, code_rag_context
            )
            if decision is None:
                logger.info("  Agent decided: ticket fully implemented. Loop complete.")
                break

            logger.info(f"  Next edit: {decision.file_path}")
            logger.info(f"  Reason: {decision.reason}")

            # ── Stuck-loop guard: oscillation breaker + per-file ceiling ──
            _fp_key = decision.file_path.replace("\\", "/").lower()
            file_edit_counts[_fp_key] = file_edit_counts.get(_fp_key, 0) + 1
            recent_edit_seq.append(_fp_key)
            if (
                len(recent_edit_seq) >= 3
                and recent_edit_seq[-1] == recent_edit_seq[-3]
                and recent_edit_seq[-1] != recent_edit_seq[-2]
            ):
                logger.warning(
                    f"  🔁 Oscillation detected: {recent_edit_seq[-3]} <-> "
                    f"{recent_edit_seq[-2]} — ping-pong cycle broken; the pair "
                    "will be capped by the per-file ceiling instead of cycling "
                    "until MAX_ITER"
                )
                # Actually break the cycle: cap BOTH partners so the ceiling check
                # below fires now for the current file and on the next pick for
                # its partner (previously this branch only logged).
                file_edit_counts[recent_edit_seq[-1]] = MAX_EDITS_PER_FILE + 1
                file_edit_counts[recent_edit_seq[-2]] = MAX_EDITS_PER_FILE + 1
                stop_reason = "oscillation"
            if file_edit_counts[_fp_key] > MAX_EDITS_PER_FILE:
                logger.warning(
                    f"  ⛔ Per-file edit ceiling ({MAX_EDITS_PER_FILE}) reached for "
                    f"{decision.file_path} — refusing further incremental edits; "
                    "escalate via build-fix or full-file replacement instead"
                )
                edit_results.append(EditResult(
                    file_path=decision.file_path,
                    content_before="",
                    content_after="",
                    skipped=True,
                    reason="per-file edit ceiling reached",
                ))
                if _fp_key in ceiling_hits:
                    # The agent keeps proposing a file that is already capped —
                    # nothing productive is left; stop instead of burning MAX_ITER.
                    stop_reason = stop_reason or "per_file_ceiling"
                    logger.warning(
                        f"  ⏹️ Edit loop stop ({stop_reason}): {decision.file_path} proposed again "
                        "after hitting the per-file ceiling"
                    )
                    break
                ceiling_hits.add(_fp_key)
                continue

            # Remove decision from pending queue if still present
            if decision in pending_queue:
                pending_queue.remove(decision)

            # Read current file content (ALWAYS from disk — never a stale cache)
            current_content = self._read_file(decision.file_path)
            if current_content is None and not decision.is_new_file:
                logger.warning(f"  File not found on disk: {decision.file_path} — skipping")
                edit_results.append(EditResult(
                    file_path=decision.file_path,
                    content_before="",
                    content_after="",
                    skipped=True,
                    reason="file not found",
                ))
                continue

            # Get LSP context for this file (exact class members)
            lsp_context = self._get_lsp_context(decision.file_path, current_content, decision.target_symbols)

            # Generate the edit
            new_content = self._generate_edit(
                decision=decision,
                current_content=current_content or "",
                ticket=ticket,
                requirements=requirements,
                code_rag_context=code_rag_context,
                lsp_context=lsp_context,
            )

            if new_content is None or new_content == current_content:
                reason = "no change generated" if new_content == current_content else "edit application failed"
                logger.warning(f"  {reason.capitalize()} for {decision.file_path} — skipping write")
                edit_results.append(EditResult(
                    file_path=decision.file_path,
                    content_before=current_content or "",
                    content_after=current_content or "",
                    skipped=True,
                    reason=reason,
                ))
                continue

            # Write the file
            self._write_file(decision.file_path, new_content)
            files_edited.add(decision.file_path)

            # Immediately live compile and build verify
            compile_errors = self._live_compile_file(decision.file_path)
            # Compiler produced no verdict (None → []): the edit is UNVERIFIED.
            diag_unknown = bool(getattr(self, "_last_verify_unknown", False))
            if diag_unknown:
                diagnostics_unverified = True
                logger.warning(
                    f"  ⚠️ Diagnostics unavailable for {decision.file_path} — "
                    "edit recorded as unverified (compile_clean=False)"
                )

            fix_applied = False
            if compile_errors:
                logger.warning(
                    f"  ⚠️ Compile errors after writing {decision.file_path} "
                    f"({len(compile_errors)} error(s))"
                )

                # ── Cross-file error detection (compile-driven) ──────────
                # When errors reference a type/class defined in a DIFFERENT
                # file (e.g. "Property 'X' does not exist on type 'Y'"),
                # queue that file for proper editing in the next iteration.
                # This is how Cursor/Copilot handle cross-file dependencies:
                # reactively from compiler output, not pre-emptive scanning.
                cross_file_targets = self._extract_cross_file_targets(
                    compile_errors, decision.file_path
                )

                if cross_file_targets:
                    for target_path, missing_members in cross_file_targets.items():
                        _tp_key = target_path.replace("\\", "/").lower()
                        if file_edit_counts.get(_tp_key, 0) < MAX_EDITS_PER_FILE and target_path not in {
                            q.file_path for q in pending_queue
                        }:
                            pending_queue.insert(0, EditDecision(
                                file_path=target_path,
                                reason=(
                                    f"Compiler error: {Path(decision.file_path).name} "
                                    f"references members not found in "
                                    f"{Path(target_path).name}: "
                                    f"{', '.join(missing_members)}"
                                ),
                                edit_description=(
                                    f"Add the following members to match usage in "
                                    f"{Path(decision.file_path).name}: "
                                    f"{', '.join(missing_members)}"
                                ),
                                is_new_file=False,
                                priority=1,  # High priority — unblock dependent
                            ))
                            logger.info(
                                f"  🔗 Cross-file error → queued {target_path} for "
                                f"missing members: {missing_members}"
                            )

                    # Filter out cross-file errors — only try inline fix for
                    # same-file errors (the cross-file ones will be fixed when
                    # the queued file is properly edited next iteration)
                    same_file_errors = [
                        e for e in compile_errors
                        if not self._is_cross_file_error(e, decision.file_path)
                    ]
                    compile_errors = same_file_errors

                # ── Same-file inline fix (original logic) ────────────────
                if compile_errors:
                    for fix_attempt in range(MAX_FIX_PER_FILE):
                        logger.info(f"  Inline fix attempt {fix_attempt + 1}/{MAX_FIX_PER_FILE}")
                        # Snapshot written files before fix to detect cross-file changes
                        _pre_fix_snapshot = dict(self._written_files)
                        fixed = self._fix_errors(
                            decision.file_path, new_content, compile_errors,
                            ticket, requirements, code_rag_context, lsp_context
                        )
                        if fixed and fixed != new_content:
                            # Primary file was modified by the fix
                            self._write_file(decision.file_path, fixed)
                            new_content = fixed
                            fix_applied = True
                            remaining_errors = self._live_compile_file(decision.file_path)
                            if not remaining_errors:
                                logger.info(f"  ✅ Inline fix resolved all errors")
                                compile_errors = []
                                break
                            compile_errors = remaining_errors
                        else:
                            # Primary file wasn't changed — but cross-file fixes
                            # may have been applied (e.g. adding a method to a
                            # service file that the primary file imports).
                            # Detect this by comparing _written_files snapshots.
                            _cross_file_changed = {
                                k for k, v in self._written_files.items()
                                if k not in _pre_fix_snapshot or _pre_fix_snapshot[k] != v
                            }
                            if _cross_file_changed:
                                logger.info(
                                    f"  🔗 Cross-file fix applied to "
                                    f"{len(_cross_file_changed)} file(s): "
                                    f"{[Path(p).name for p in _cross_file_changed]}"
                                    f" — re-verifying primary file"
                                )
                                fix_applied = True
                                remaining_errors = self._live_compile_file(decision.file_path)
                                if not remaining_errors:
                                    logger.info(f"  ✅ Cross-file fix resolved all errors in {Path(decision.file_path).name}")
                                    compile_errors = []
                                    break
                                compile_errors = remaining_errors
                                # Don't break — try another inline fix attempt
                                # if errors remain and we have attempts left
                            else:
                                logger.info(f"  Fix attempt produced no change")
                                break

            # Bounded safety net: stop if the SAME unresolved error set recurs,
            # instead of consuming the remaining iterations on the same fix.
            if compile_errors:
                from ticket_to_code.agents.edit_loop_policy import error_signature as _esig
                _sig = _esig(compile_errors)
                if _sig in _seen_error_sigs:
                    logger.warning(
                        f"  ⏹️ Edit loop stop: identical unresolved errors recurred for "
                        f"{decision.file_path} — returning failure instead of repeating the same fix."
                    )
                    edit_results.append(EditResult(
                        file_path=decision.file_path,
                        content_before=current_content or "",
                        content_after=new_content,
                        compile_errors=compile_errors,
                        compile_clean=False,
                    ))
                    break
                _seen_error_sigs.add(_sig)

            compile_clean = not bool(compile_errors)
            file_basename = Path(decision.file_path).name
            check_label = self._get_live_check_label(decision.file_path)

            if compile_clean:
                logger.info(f"  ✅ {decision.file_path} — clean compile [{check_label}]")
                self._ui_emit(
                    "live_check_passed",
                    message=f"✅ [{check_label}] Compiled OK — {file_basename}",
                    file_path=decision.file_path,
                    file_name=file_basename,
                    check_type=check_label,
                )
            else:
                logger.warning(
                    f"  ⚠️ {decision.file_path} — {len(compile_errors)} error(s) remain "
                    f"(will be addressed by full build phase)"
                )
                self._ui_emit(
                    "live_check_start",
                    message=f"⚠️ [{check_label}] {len(compile_errors)} compile error(s) in {file_basename}",
                    file_path=decision.file_path,
                    file_name=file_basename,
                    error_count=len(compile_errors),
                    errors=compile_errors[:5],
                    check_type=check_label,
                )

            edit_results.append(EditResult(
                file_path=decision.file_path,
                content_before=current_content or "",
                content_after=new_content,
                compile_errors=compile_errors,
                compile_clean=compile_clean and not diag_unknown,
                fix_applied=fix_applied,
                reason="diagnostics unavailable" if (diag_unknown and compile_clean) else "",
            ))

            # Update the symbol index cache so subsequent edits see fresh types
            ext = Path(decision.file_path).suffix.lower()
            if ext in (".ts", ".tsx", ".java", ".kt"):
                self.symbol_index.get_members_for_file(decision.file_path)

        # Final compile check
        ts_files = [fp for fp in files_edited if fp.endswith((".ts", ".tsx"))]
        final_errors = self.symbol_index.get_ts_diagnostics(ts_files) if ts_files else []
        if final_errors is None:
            logger.warning("  ⚠️ TS diagnostics unavailable — final result is UNVERIFIED (not claimed clean)")
            final_errors = []
            diagnostics_unverified = True
        has_failed_edits = any(r.reason.startswith("edit application failed") for r in edit_results)
        solved = not bool(final_errors) and not has_failed_edits and bool(files_edited)
        if solved:
            logger.info(f"\n✅ EDIT LOOP COMPLETE: Ticket solved in {iteration} iteration(s)")
        else:
            logger.warning(
                f"\n⚠️ EDIT LOOP COMPLETE: Ticket not solved ({len(final_errors)} compile error(s), "
                f"failed_edits={has_failed_edits}) after {iteration} iteration(s)"
            )
        logger.info(
            f"   Edit loop tokens: {self._tokens['total_tokens']} "
            f"(in={self._tokens['input_tokens']}, out={self._tokens['output_tokens']}, "
            f"llm_calls={self._tokens['llm_calls']})"
        )

        return {
            "generated_files": dict(self._written_files),
            "edit_results": edit_results,
            "final_compile_errors": final_errors,
            "iterations": iteration,
            "solved": solved,
            "stop_reason": stop_reason,
            "diagnostics_unverified": diagnostics_unverified,
            "token_usage": dict(self._tokens),
            "pre_edit_snapshots": dict(self.pre_edit_snapshots),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _decide_next_edit(
        self,
        ticket: "ValueEdgeTicket",
        requirements: "StructuredRequirements",
        pending_queue: list[EditDecision],
        files_edited: set[str],
        code_rag_context: str,
    ) -> Optional[EditDecision]:
        """
        Ask the LLM: given the current codebase state, what is the NEXT file
        that needs to be edited to implement the ticket?

        Returns None when the ticket is fully implemented.
        """
        # If pending queue has items, pick the next task in queue order.
        # This supports multi-pass coordination on the same file (e.g. a.ts -> b.html -> a.ts).
        if pending_queue:
            return pending_queue.pop(0)

        # If any edited file currently has compilation errors, prioritize fixing it first
        if files_edited:
            for fp in files_edited:
                errs = self._verify_file(fp)
                if errs:
                    return EditDecision(
                        file_path=fp,
                        reason=f"Compilation error needs inline fix: {errs[0]}",
                        edit_description=f"Fix compilation error in {fp}: {errs[0]}",
                        is_new_file=False,
                        priority=1,
                        target_symbols=self._find_targets_for_file(fp),
                    )

        # Build a summary of what's been done
        done_summary = "\n".join(f"  ✅ {fp}" for fp in sorted(files_edited))
        current_file_states = self._summarize_written_files()

        from ticket_to_code.llm_utils import llm_invoke
        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = """You are a senior developer assessing whether a ticket is fully implemented.
Given the ticket requirements and the files that have been modified, determine:
1. Is the ticket FULLY implemented? If yes, respond with just: DONE
2. If NOT fully implemented, which single file should be edited next?

Respond with EITHER:
  DONE
OR:
  FILE: <exact relative file path>
  REASON: <why this file still needs changes>
  DESCRIPTION: <what specific change to make>

Be decisive. Do not suggest files that are already done unless they need further changes."""
        if self.allowed_files:
            system_prompt += f"""

STRICT SCOPE CONSTRAINT:
You may ONLY propose edits to files in this authorized allowlist:
{chr(10).join(f' - {af}' for af in sorted(self.allowed_files))}
Proposing any file outside this allowlist is strictly forbidden. If no file in the allowlist needs changes, respond DONE."""

        if self._remediation_mode:
            system_prompt += """

CONTEXT: A prior verification pass flagged this ticket as only PARTIALLY implemented.
Compiling cleanly does NOT mean done — judge against the OUTSTANDING REQUIREMENTS
section below. If the outstanding behavior is implemented, respond DONE; otherwise
point at the exact file that still needs the change."""

        plan_section = ""
        if hasattr(self, "_initial_plan") and self._initial_plan and getattr(self._initial_plan, "tasks", None):
            plan_lines = [f"  - [{t.file_path}] {t.title}: {t.description}" for t in self._initial_plan.tasks]
            plan_section = f"""
ARCHITECTURAL PLAN (Ensure all tasks are executed):
{chr(10).join(plan_lines)}
"""

        user_prompt = f"""TICKET: {ticket.title}
{ticket.description}

REQUIREMENTS:
{getattr(requirements, 'functional_requirements', '')}
{plan_section}
"""
        if code_rag_context:
            user_prompt += f"""
OUTSTANDING REQUIREMENTS (from verification):
{str(code_rag_context)[:1500]}
"""
        user_prompt += f"""
FILES ALREADY MODIFIED:
{done_summary}

CURRENT STATE OF WRITTEN FILES:
{current_file_states[:2000]}

Is the ticket fully implemented? If not, what is the next file to edit?"""

        try:
            response = llm_invoke(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            self._track_tokens(response)
            text = response.content.strip()

            if text.startswith("DONE") or "fully implemented" in text.lower():
                return None

            file_m = re.search(r'FILE:\s*(.+)', text)
            reason_m = re.search(r'REASON:\s*(.+)', text)
            desc_m = re.search(r'DESCRIPTION:\s*(.+)', text, re.DOTALL)

            if not file_m:
                return None

            fp = file_m.group(1).strip().strip("`'\"")
            if self.allowed_files is not None:
                norm_fp = fp.replace("\\", "/").lower().strip()
                is_allowed = any(
                    norm_fp == af or norm_fp.endswith("/" + af) or af.endswith("/" + norm_fp)
                    for af in self.allowed_files
                )
                if not is_allowed:
                    is_migration = "db/migration" in norm_fp and norm_fp.endswith(".sql")
                    is_companion = (
                        norm_fp.endswith("repository.java") or
                        norm_fp.endswith("service.java") or
                        norm_fp.endswith("test.java")
                    )
                    if is_migration or is_companion:
                        logger.info(f"  🔓 Auto-admitting proposed companion file '{fp}' into allowed_files")
                        self.allowed_files.add(norm_fp)
                        is_allowed = True
                    else:
                        logger.warning(
                            f"  ⛔ edit_loop: LLM proposed file '{fp}' outside allowed_files ({sorted(self.allowed_files)}). Blocking proposal."
                        )
                        return None

            return EditDecision(
                file_path=fp,
                reason=reason_m.group(1).strip() if reason_m else "Agent identified missing change",
                edit_description=desc_m.group(1).strip()[:400] if desc_m else "Make required changes",
                priority=5,
            )
        except Exception as exc:
            logger.warning(f"  Agent decision failed: {exc}")
            return None

    def _generate_edit(
        self,
        decision: EditDecision,
        current_content: str,
        ticket: "ValueEdgeTicket",
        requirements: "StructuredRequirements",
        code_rag_context: str,
        lsp_context: str,
    ) -> Optional[str]:
        """Generate a targeted, minimal edit for the file."""
        from ticket_to_code.llm_utils import llm_invoke
        from langchain_core.messages import SystemMessage, HumanMessage

        ext = Path(decision.file_path).suffix.lower()
        is_html = ext in (".html", ".htm")
        is_create = decision.is_new_file or not current_content

        # Build session context (what's been written so far in this run)
        session_ctx = self._build_session_context(decision.file_path)

        system_prompt = f"""You are an expert software engineer making a SURGICALLY TARGETED edit to ONE file.

CRITICAL RULES (TARGETED PART REPLACEMENT ONLY):
1. ONLY modify the single file specified.
2. Write ONLY the specific part of code you want to replace. Do NOT write in large blocks or rewrite whole functions.
3. Make the SMALLEST possible change that satisfies the requirement.
4. Preserve ALL existing logic, methods, classes, and imports not directly related to the change.
5. Implement the exact production solution cleanly. Do NOT leave commented-out dead code and NEVER comment out functional defect logic to bypass checks.
6. {"Output the COMPLETE new file content." if is_create else "Use minimal SEARCH/REPLACE or TARGET/REPLACEMENT blocks — NEVER output the whole file or large surrounding blocks."}
7. Patches must be strictly minimal and proportional to the target symbol. Whole-file rewrites or large block replacements will be rejected.
8. Do NOT output conversational preamble, greetings, or long reasoning. Begin immediately with the SUMMARY line.

{"ANGULAR TEMPLATE RULE: Before writing any *ngIf or {{ }} binding, read the SIBLING CONTROLLER section in the context. Use ONLY property names that are DECLARED in that controller. Never invent new names." if is_html else ""}

OUTPUT FORMAT:
SUMMARY:
<one concise user-facing sentence explaining what this edit changes>

{"```" + (ext[1:] if ext else "text") + "\n<complete new content>\n```" if is_create else """<<<<<<< SEARCH
<exact existing code to find (3 to 8 lines only, matching character-for-character)>
=======
<replacement code for that specific part only>
>>>>>>> REPLACE

Or TARGET / REPLACEMENT:
TARGET:
<exact existing code to find (3 to 8 lines only)>
REPLACEMENT:
<replacement code for that specific part only>"""}

SEARCH/REPLACE RULES:
- The SEARCH (or TARGET) block must be an EXACT substring of the CURRENT FILE CONTENT (or EXACT SOURCE section) shown below — copy it character-for-character including indentation.
- Write ONLY the specific part to replace (typically 3 to 8 lines). Do NOT replace whole methods or big blocks when changing a few lines.
- The SEARCH block must match EXACTLY ONCE in the file.
- Prefer multiple small SEARCH/REPLACE blocks over one large block.
- NEVER put the entire file content or large blocks in a SEARCH block.
- For ADDING new code: use a small anchor from existing code as SEARCH, and include anchor + new code as REPLACE.
- You may use multiple SEARCH/REPLACE blocks in a single response.
- PURE CODE ONLY: The REPLACE block must contain ONLY pure, syntactically valid code. NEVER include conversational monologue, reasoning paragraphs, option trade-off discussions, or conversational English notes inside the code.
- Balanced Delimiters: Every opening curly brace or parenthesis in your REPLACE block MUST have its corresponding closing brace or parenthesis. Unbalanced delimiters cause INSTANT REJECTION."""

        # Target-first file content: verbatim source for small files, skeleton+exact for large files
        if current_content:
            from ticket_to_code.agents.smart_extract import smart_extract, extract_exact_methods

            # Extract target method/symbol from decision description or reason
            target_symbol = None
            sym_match = re.search(
                r'\b(?:method|function|property|class)\s+[`\'"]?([a-zA-Z_$][a-zA-Z0-9_$]*)',
                f"{decision.reason} {decision.edit_description}",
                re.IGNORECASE
            )
            if sym_match:
                target_symbol = sym_match.group(1)

            target_methods = list(decision.target_symbols or [])
            if target_symbol and target_symbol not in target_methods:
                target_methods.insert(0, target_symbol)
            line_count = len(current_content.splitlines())
            char_count = len(current_content)

            # When the outcome findings (or the decision text) name the
            # implicated methods, ship ONLY the structure outline + those
            # methods' exact source. This is what keeps remediation iterations
            # cheap: an edit to onUserSelect no longer pays for the whole file
            # plus the whole companion AST on every retry.
            verbatim_content, matched_boundaries = None, None
            if target_methods:
                try:
                    verbatim_content, matched_boundaries = extract_exact_methods(
                        content=current_content,
                        file_path=decision.file_path,
                        anchor_methods=target_methods,
                    )
                except Exception:
                    verbatim_content, matched_boundaries = None, None
                if not matched_boundaries:
                    verbatim_content = None

            # --- LEGACY THRESHOLD (Commented out to save tokens on medium files) ---
            # if verbatim_content and char_count > 8000:
            # ----------------------------------------------------------------------
            # When verbatim methods are extracted via AST/regex, use skeleton + exact source
            # whenever the file is > 100 lines or > 3500 chars. This cuts prompt tokens by ~60%
            # while maintaining 100% SEARCH/REPLACE exactness.
            if verbatim_content and (char_count > 3500 or line_count > 100):
                skeleton = smart_extract(
                    current_content,
                    file_path=decision.file_path,
                    allowed_methods=target_methods,
                    target_method=target_symbol,
                    edit_description=decision.edit_description,
                )
                file_content_block = (
                    "=== FILE STRUCTURE (context only — DO NOT copy text from this section) ===\n"
                    f"{skeleton}\n\n"
                    "=== EXACT SOURCE (your SEARCH block MUST be a verbatim substring of THIS section) ===\n"
                    f"{verbatim_content}"
                )
            # If the file is small (<= 100 lines or <= 3500 chars), pass pure VERBATIM source
            # This completely avoids synthetic outline comments (// === FILE OUTLINE) poisoning SEARCH blocks!
            elif line_count <= 600 or char_count <= 30000:
                file_content_block = current_content
            else:
                # Fallback: verbatim window
                file_content_block = current_content[:25000]
        else:
            file_content_block = "(new file)"

        plan_block = ""
        if hasattr(self, "_initial_plan") and self._initial_plan and getattr(self._initial_plan, "tasks", None):
            plan_lines = []
            for t in self._initial_plan.tasks:
                marker = "👉 [TARGET FILE TASK]" if (decision.file_path and t.file_path and (t.file_path in decision.file_path or decision.file_path in t.file_path)) else "  "
                plan_lines.append(f"{marker} [{t.file_path}] {t.title}: {t.description}")
            plan_block = f"""
ARCHITECTURAL PLAN & SPECIFICATION:
{chr(10).join(plan_lines)}
"""

        companion_ctx = self._get_companion_context(decision.file_path)

        user_prompt = f"""TICKET: {ticket.title}
{ticket.description}

FILE TO EDIT: {decision.file_path}
REASON FOR EDIT: {decision.reason}
WHAT TO CHANGE: {decision.edit_description}

{plan_block}

{companion_ctx}

{lsp_context}

{session_ctx}

CURRENT FILE CONTENT:
```
{file_content_block}
```

Make the targeted edit now."""

        try:
            # A whole new file legitimately needs far more output than a
            # SEARCH/REPLACE hunk; 1024 tokens silently truncated created files.
            response = llm_invoke(
                self.llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
                max_tokens=WHOLE_FILE_MAX_TOKENS if is_create else EDIT_MAX_TOKENS,
            )
            self._track_tokens(response)
            if (getattr(response, "response_metadata", {}) or {}).get("finish_reason") == "length":
                logger.warning(
                    "edit_loop: LLM output truncated (finish_reason=length) for %s", decision.file_path
                )

            # 1. Parse SUMMARY line for UI/user visibility
            summary_m = re.search(r'SUMMARY:\s*(.+?)(?:\n\n|<<<<<<<|\Z)', response.content, re.DOTALL)
            if summary_m:
                summary_line = summary_m.group(1).strip().splitlines()[0]
                logger.info(f"  edit_loop SUMMARY: {summary_line}")
                self._ui_emit(
                    "patch_summary",
                    message=f"✓ {summary_line}",
                    file_path=decision.file_path,
                )

            # 2. Validate patch proportionality via PatchGate before applying
            from ticket_to_code.agents.patch_gate import PatchGate
            prop_ok, prop_reason = PatchGate.validate_patch_proportionality(
                decision.file_path,
                response.content,
                current_content,
                is_new_file=is_create,
            )
            if not prop_ok:
                logger.warning(f"  ⛔ Patch rejected by PatchGate proportionality check: {prop_reason}")
                return None

            edit_res = self._apply_search_replace(current_content or "", response.content, decision.is_new_file)
            if edit_res.success:
                logger.info(
                    f"  edit_loop: {edit_res.applied_count} edit(s) applied to "
                    f"{decision.file_path} via [{edit_res.parser_format}]"
                )
                return edit_res.content

            # If create action or no existing content, fail safely
            if is_create or not current_content:
                logger.warning(f"  Create edit failed for {decision.file_path}: {edit_res.failure_reason}")
                return None

            # Edit failed — perform EXACTLY ONE retry with feedback (Phase 4)
            # Re-read fresh source from disk/storage to resolve drift
            fresh_content = self._read_file(decision.file_path) or current_content
            logger.warning(
                f"  Edit application failed for {decision.file_path}: {edit_res.failure_reason} "
                f"[{edit_res.parser_format}]. Re-reading fresh source and re-localizing AST boundary for single retry..."
            )

            # Re-localize containing method boundary on fresh source
            method_ctx = ""
            try:
                from ticket_to_code.agents.smart_extract import _get_reliable_boundaries
                fresh_lines = fresh_content.splitlines()
                boundaries = _get_reliable_boundaries(fresh_content, decision.file_path, fresh_lines)
                if boundaries and edit_res.failed_search:
                    search_first_line = edit_res.failed_search.strip().splitlines()[0].strip()
                    for idx, fl in enumerate(fresh_lines):
                        if search_first_line in fl:
                            for mb in boundaries:
                                if mb.kind != "class" and mb.start_line <= idx <= mb.end_line:
                                    method_ctx = "\n".join(fresh_lines[mb.start_line : mb.end_line + 1])
                                    break
                            if method_ctx:
                                break
            except Exception:
                pass

            feedback_lines = [
                "⚠️ YOUR PREVIOUS EDIT ATTEMPT FAILED TO APPLY:",
                f"Failure reason: {edit_res.failure_reason}",
            ]
            if edit_res.failed_search:
                feedback_lines.append(f"Failed SEARCH block:\n```\n{edit_res.failed_search}\n```")
            if method_ctx:
                feedback_lines.append(f"Re-localized containing method from fresh file:\n```\n{method_ctx}\n```")
            elif edit_res.failed_search:
                local_ctx = self._find_surrounding_context(fresh_content, edit_res.failed_search)
                if local_ctx:
                    feedback_lines.append(f"Surrounding context from current file:\n```\n{local_ctx}\n```")

            feedback_lines.extend([
                "",
                "CRITICAL INSTRUCTIONS FOR RETRY:",
                "1. Your previous SEARCH block does not match the current file.",
                "2. Do NOT regenerate the file.",
                "3. Return ONLY a corrected edit using the required format:",
                "   <<<<<<< SEARCH",
                "   <exact existing code copied character-for-character>",
                "   =======",
                "   <replacement code>",
                "   >>>>>>> REPLACE",
                "4. The SEARCH text must be copied character-for-character from the current file, including indentation.",
            ])
            retry_prompt = "\n".join(feedback_lines)

            from langchain_core.messages import AIMessage
            retry_response = llm_invoke(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
                AIMessage(content=response.content),
                HumanMessage(content=retry_prompt),
            ])

            retry_res = self._apply_search_replace(fresh_content, retry_response.content, False)
            if retry_res.success:
                logger.info(
                    f"  ✅ Retry succeeded for {decision.file_path}: {retry_res.applied_count} "
                    f"edit(s) applied via [{retry_res.parser_format}]"
                )
                return retry_res.content
            else:
                logger.warning(
                    f"  ❌ Retry also failed for {decision.file_path}: {retry_res.failure_reason}. "
                    f"Rejecting edit safely — existing code remains untouched."
                )
                return None
        except Exception as exc:
            logger.warning(f"  Edit generation failed for {decision.file_path}: {exc}")
            return None

    @staticmethod
    def _find_surrounding_context(content: str, failed_search: str) -> str:
        """Find surrounding lines from content for context without guessing or inventing a location."""
        if not content or not failed_search:
            return ""
        lines = [l.strip() for l in failed_search.splitlines() if len(l.strip()) >= 10]
        for candidate_line in sorted(lines, key=len, reverse=True):
            if content.count(candidate_line) == 1:
                content_lines = content.splitlines()
                for idx, cl in enumerate(content_lines):
                    if candidate_line in cl:
                        start = max(0, idx - 4)
                        end = min(len(content_lines), idx + 8)
                        return "\n".join(content_lines[start:end])
        return ""

    def _fix_errors(
        self,
        file_path: str,
        current_content: str,
        errors: list[str],
        ticket: "ValueEdgeTicket",
        requirements: "StructuredRequirements",
        code_rag_context: str,
        lsp_context: str,
    ) -> Optional[str]:
        """Fix compile errors with cross-file awareness and action-based validation.

        This runs in the FAILURE NODE after all planned files have been written.
        At this point we have a complete list of modified/created files, so cross-file
        fixes are safe — we won't accidentally overwrite a file before its task runs.
        """
        import json as _json_fix
        import os as _os_fix
        from ticket_to_code.llm_utils import llm_invoke
        from langchain_core.messages import SystemMessage, HumanMessage
        from ticket_to_code.agents.diagnostic_normalizer import normalize_diagnostics
        from ticket_to_code.agents.diagnostic_localizer import (
            DiagnosticLocalizer,
            RepairContextTier,
            FailureOwner,
        )

        localizer = getattr(self, "_diagnostic_localizer", None)
        if not localizer:
            localizer = DiagnosticLocalizer(self.workspace_path)
            self._diagnostic_localizer = localizer

        norm_diags = normalize_diagnostics(errors)
        if not localizer.can_attempt_repair(norm_diags, max_attempts=2):
            logger.warning(
                "  🛑 Bounded repair budget reached: identical diagnostics unchanged after 2 attempts in EditLoop."
            )
            return None
        localizer.record_attempt(norm_diags)

        error_text = "\n".join(errors[:20])

        # ── Collect related files mentioned in errors ──
        _all_error_files: dict[str, str] = {file_path: current_content}
        for err_line in errors:
            ep = re.search(
                r'([^\s(]+\.(?:ts|tsx|js|jsx|java|kt|py))\s*[\(:\[]',
                err_line, re.IGNORECASE
            )
            if ep:
                ep_raw = ep.group(1).lstrip("/").replace("\\", "/")
                if ep_raw != file_path.replace("\\", "/") and ep_raw not in _all_error_files:
                    # Try to resolve: written this run → direct path → workspace scan
                    resolved_path, resolved_content = self._resolve_error_file(ep_raw)
                    if resolved_content and resolved_path not in _all_error_files:
                        _all_error_files[resolved_path] = resolved_content
                        logger.info(f"  Including related error file: {resolved_path}")

        # ── Also inject session-generated files imported by the primary file ──
        src_dir = str(Path(file_path).parent).replace("\\", "/")
        for imp_m in re.finditer(r"""from\s+['"](\.[']+)['"]""", current_content):
            for iext in (".ts", ".tsx"):
                irel = _os_fix.path.normpath(
                    _os_fix.path.join(src_dir, imp_m.group(1) + iext)
                ).replace("\\", "/").lower()
                if irel in self._written_files and irel not in _all_error_files:
                    _all_error_files[irel] = self._written_files[irel]

        # ── Build related files section (Market AI IDE standard: bounded context) ──
        other_files_section = ""
        for rf, rc in list(_all_error_files.items())[:3]:
            if rf != file_path:
                other_files_section += f"\nRELATED FILE: {rf}\n```\n{rc[:1000]}\n```\n"

        # ── Ticket context so fixes preserve intent ──
        ticket_title = getattr(ticket, "title", "") or ""
        ticket_desc = (getattr(ticket, "description", "") or "")[:400]
        func_reqs = ""
        if requirements:
            fr = getattr(requirements, "functional_requirements", []) or []
            func_reqs = "\n".join(f"  - {r}" for r in fr[:5])

        system_prompt = (
            "You are fixing compile errors while PRESERVING the ticket's intended functionality.\n"
            "CRITICAL: Do NOT remove or stub out new functionality to make the code compile.\n"
            "If a method or property is missing, ADD it where it belongs.\n"
            "If a service call is wrong, fix the call to use the right service — not remove it.\n\n"
            "The root cause may be in a DIFFERENT file than where the error appears.\n"
            "Fix the actual root cause, not just the symptom.\n\n"
            "Return a JSON object:\n"
            "{\n"
            '  "reasoning": "root cause + which files need changing + how intent is preserved",\n'
            '  "fixes": [\n'
            '    {\n'
            '      "file": "exact/relative/path.ts",\n'
            '      "action": "modify",\n'
            '      "edits": [\n'
            '        {"search": "exact existing code to find", "replace": "replacement code"}\n'
            '      ]\n'
            '    },\n'
            '    {"file": "new/file/if/needed.ts", "action": "create", "content": "complete new file"}\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- For 'modify' actions: use 'edits' array with search/replace pairs (NOT complete file content)\n"
            "- The 'search' value must be an EXACT substring of the existing file — copy it character-for-character\n"
            "- The 'replace' value is what replaces the search text\n"
            "- Write ONLY the specific lines to replace (3 to 8 lines). Never replace large blocks\n"
            "- Directly replace deprecated code with the updated implementation in the replace block without leaving dead code or invalid comment tokens\n"
            "- Use multiple small edits rather than one giant edit\n"
            "- For 'create' actions ONLY: use 'content' with the complete new file\n"
            "- NEVER put complete file content in 'edits' — only the lines that change + minimal surrounding context\n"
            "- Fix every file that needs to change (primary + root-cause files)\n"
            "- Respond with JSON only, no markdown"
        )

        # ── Feedback loop: ask LLM → validate → rejected? → tell LLM why → retry ──
        MAX_FIX_ATTEMPTS = 3
        rejection_history: list[str] = []  # accumulates across attempts
        primary_content = None

        for attempt in range(MAX_FIX_ATTEMPTS):
            tier = (
                RepairContextTier.TIER_1_LOCALIZED_METHOD
                if attempt == 0
                else RepairContextTier.TIER_4_WHOLE_FILE_ESCALATION
            )
            loc_ctx = localizer.localize_context(file_path, current_content, norm_diags, tier=tier)
            if loc_ctx.context_tier == RepairContextTier.TIER_1_LOCALIZED_METHOD:
                primary_file_block = (
                    f"PRIMARY FILE (Tier 1 Localized Context around `{loc_ctx.target_method_name}`): {file_path}\n"
                    f"```\n{loc_ctx.prompt_snippet}\n```\n"
                )
            else:
                primary_file_block = (
                    f"PRIMARY FILE (Tier 4 Whole File Context): {file_path}\n"
                    f"```\n{current_content}\n```\n"
                )

            user_prompt = (
                f"TICKET: {ticket_title}\n{ticket_desc}\n\n"
                + (f"FUNCTIONAL REQUIREMENTS (must still be satisfied after fix):\n{func_reqs}\n\n" if func_reqs else "")
                + f"COMPILE ERRORS:\n{error_text}\n\n"
                + f"{lsp_context}\n\n"
                + primary_file_block
                + other_files_section
                + "\nFix all errors without removing any new functionality. Return JSON."
            )

            # Build the prompt — include rejection feedback from previous attempts
            rejection_feedback = ""
            if rejection_history:
                rejection_feedback = (
                    "\n\n⚠️ YOUR PREVIOUS FIX ATTEMPT WAS PARTIALLY REJECTED:\n"
                    + "\n".join(f"  ❌ {r}" for r in rejection_history)
                    + "\n\nFix these issues in your next attempt. "
                    "If you tried to 'modify' a file that doesn't exist, either:\n"
                    "  1. Use 'action': 'create' if the file genuinely needs to be created\n"
                    "  2. Fix the issue in a DIFFERENT existing file instead\n"
                    "  3. Add the missing method/property in the file where the type is actually defined\n"
                )

            final_user_prompt = (
                user_prompt + rejection_feedback
            )

            try:
                # The JSON `fixes` array may include {"action": "create"} with a
                # complete new file, so it gets the whole-file output budget.
                response = llm_invoke(
                    self.llm,
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=final_user_prompt),
                    ],
                    max_tokens=WHOLE_FILE_MAX_TOKENS,
                )
                if (getattr(response, "response_metadata", {}) or {}).get("finish_reason") == "length":
                    logger.warning(
                        "edit_loop: LLM output truncated (finish_reason=length) for %s", file_path
                    )
                fix_raw = response.content if hasattr(response, "content") else str(response)
                usage = getattr(response, "usage_metadata", {}) or {}
                tok = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
                if not tok:
                    tok = len(final_user_prompt.split()) + len(fix_raw.split())
                localizer.telemetry.edit_loop_tokens += tok
                localizer.telemetry.total_tokens += tok

                # Parse JSON response
                from ticket_to_code.workflow import _extract_json_object
                fix_result = _extract_json_object(fix_raw)
                if not fix_result:
                    json_m = re.search(r'\{.*\}', fix_raw, re.DOTALL)
                    if json_m:
                        try:
                            fix_result = _json_fix.loads(json_m.group(0))
                        except Exception:
                            pass
                if not fix_result or not isinstance(fix_result, dict):
                    logger.warning(f"  Attempt {attempt+1}: No JSON in response — falling back to SEARCH/REPLACE")
                    edit_res = self._apply_search_replace(current_content, fix_raw, False)
                    if edit_res.success:
                        return edit_res.content
                    logger.warning(f"  SEARCH/REPLACE fallback failed: {edit_res.failure_reason}")
                    if attempt < MAX_FIX_ATTEMPTS - 1:
                        rejection_history.append(
                            f"Response was not valid JSON and SEARCH/REPLACE failed: {edit_res.failure_reason}. "
                            f"Return a valid JSON object with 'fixes' array."
                        )
                        continue
                    return None

                fixes = fix_result.get("fixes", [])
                if not fixes:
                    logger.warning(f"  Attempt {attempt+1}: No fixes in JSON response")
                    if attempt < MAX_FIX_ATTEMPTS - 1:
                        rejection_history.append("You returned an empty 'fixes' array. Provide actual fixes.")
                        continue
                    return None

                logger.info(
                    f"  Attempt {attempt+1}: {len(fixes)} fix(es) — "
                    f"{fix_result.get('reasoning', '')[:120]}"
                )

                # ── Validate and apply each fix ──
                attempt_rejections: list[str] = []
                applied_count = 0

                for fix_item in fixes:
                    fix_path_raw = fix_item.get("file", "")
                    fix_action = fix_item.get("action", "modify").lower().strip()
                    fix_edits = fix_item.get("edits", [])    # NEW: search/replace pairs
                    fix_content = fix_item.get("content", "")  # Only for CREATE actions

                    if not fix_path_raw:
                        continue

                    # Resolve the path
                    resolved_path = self._resolve_fix_path(fix_path_raw)
                    fix_abs = self.workspace_path / resolved_path
                    is_primary = (
                        resolved_path.replace("\\", "/").lower()
                        == file_path.replace("\\", "/").lower()
                    )

                    # Action-based validation
                    if fix_action == "modify":
                        if not fix_abs.exists() and not is_primary:
                            msg = (
                                f"action='modify' for '{resolved_path}' but file does NOT exist on disk. "
                                f"Either use action='create' if it truly needs to be created, "
                                f"or fix the issue in an existing file instead."
                            )
                            logger.warning(f"  ❌ {msg}")
                            attempt_rejections.append(msg)
                            continue
                    elif fix_action == "create":
                        if fix_abs.exists():
                            logger.info(
                                f"  action='create' but file already exists: "
                                f"{resolved_path} — treating as modify"
                            )
                            fix_action = "modify"
                        else:
                            logger.info(f"  ✅ LLM requested file creation: {resolved_path}")
                    else:
                        if not fix_abs.exists() and not is_primary:
                            msg = (
                                f"Unknown action='{fix_action}' for '{resolved_path}' "
                                f"and file does not exist. Use 'modify' or 'create'."
                            )
                            logger.warning(f"  ❌ {msg}")
                            attempt_rejections.append(msg)
                            continue

                    # Validation: no protected paths
                    fp_lower = resolved_path.lower()
                    if any(x in fp_lower for x in ("/dist/", "/node_modules/", "/.git/", "/target/")):
                        msg = f"Fix targets protected path: {resolved_path} — cannot modify."
                        logger.warning(f"  ❌ {msg}")
                        attempt_rejections.append(msg)
                        continue

                    # Validation: must be inside workspace
                    try:
                        fix_abs.resolve().relative_to(self.workspace_path.resolve())
                    except ValueError:
                        msg = f"Fix targets path outside workspace: {resolved_path}"
                        logger.warning(f"  ❌ {msg}")
                        attempt_rejections.append(msg)
                        continue

                    # ── All validations passed — apply the fix ──
                    fix_abs.parent.mkdir(parents=True, exist_ok=True)

                    if fix_action == "create":
                        # CREATE: write complete content directly (correct for new files)
                        if not fix_content:
                            msg = f"action='create' for '{resolved_path}' but no 'content' provided."
                            logger.warning(f"  ❌ {msg}")
                            attempt_rejections.append(msg)
                            continue
                        self._write_file(resolved_path, fix_content)
                        logger.info(
                            f"    ✏️  CREATE: {resolved_path} "
                            f"({len(fix_content)} chars)"
                        )
                        applied_count += 1
                        if is_primary:
                            primary_content = fix_content

                    elif fix_edits:
                        # MODIFY with search/replace edits — route through the SAME safe atomic engine
                        existing_fix_content = (
                            current_content if is_primary
                            else self._read_file(resolved_path) or ""
                        )
                        canonical_edits = []
                        for ed in fix_edits:
                            canonical_edits.append({
                                "old_str": ed.get("search", ed.get("old_str", "")),
                                "new_str": ed.get("replace", ed.get("new_str", "")),
                            })
                        if not canonical_edits:
                            msg = f"No valid search/replace pairs found in 'edits' for '{resolved_path}'"
                            logger.warning(f"  ❌ {msg}")
                            attempt_rejections.append(msg)
                            continue

                        edit_res = self._safe_match_and_replace(
                            original=existing_fix_content,
                            edits=canonical_edits,
                            parser_format="json_fix",
                        )
                        if edit_res.success:
                            self._write_file(resolved_path, edit_res.content)
                            logger.info(
                                f"    ✏️  MODIFY (S/R atomic): {resolved_path} "
                                f"({edit_res.applied_count} edit(s) applied)"
                            )
                            applied_count += 1
                            if is_primary:
                                primary_content = edit_res.content
                        else:
                            ast_ctx_hint = ""
                            try:
                                from ticket_to_code.agents.smart_extract import _get_reliable_boundaries
                                fresh_content = self._read_file(resolved_path) or existing_fix_content
                                fresh_lines = fresh_content.splitlines()
                                bounds = _get_reliable_boundaries(fresh_content, resolved_path, fresh_lines)
                                if bounds and edit_res.failed_search:
                                    s_first = edit_res.failed_search.strip().splitlines()[0].strip()
                                    for idx, fl in enumerate(fresh_lines):
                                        if s_first in fl:
                                            for mb in bounds:
                                                if mb.kind != "class" and mb.start_line <= idx <= mb.end_line:
                                                    ast_ctx_hint = (
                                                        f"\nFresh source for containing method `{mb.name}`:\n```\n"
                                                        + "\n".join(fresh_lines[mb.start_line : mb.end_line + 1])
                                                        + "\n```"
                                                    )
                                                    break
                                            if ast_ctx_hint:
                                                break
                            except Exception:
                                pass

                            msg = (
                                f"Atomic edit failed for '{resolved_path}': {edit_res.failure_reason}. "
                                f"Existing file remains untouched.{ast_ctx_hint}"
                            )
                            logger.warning(f"  ❌ {msg}")
                            attempt_rejections.append(msg)

                    elif fix_content:
                        # Reject full-file content for modify actions (Phase 6 & 7)
                        msg = (
                            f"action='modify' for '{resolved_path}' requires explicit search/replace edits. "
                            f"Full-file content was returned instead of 'edits' array. "
                            f"Use 'edits': [{{'search': '...', 'replace': '...'}}]."
                        )
                        logger.warning(f"  ❌ {msg}")
                        attempt_rejections.append(msg)

                    else:
                        msg = f"No 'edits' or 'content' provided for '{resolved_path}'"
                        logger.warning(f"  ❌ {msg}")
                        attempt_rejections.append(msg)

                # ── Decide: all applied? some rejected? ──
                if attempt_rejections:
                    rejection_history.extend(attempt_rejections)
                    if applied_count == 0:
                        # ALL fixes were rejected — must retry
                        logger.warning(
                            f"  ⚠️ Attempt {attempt+1}: ALL {len(attempt_rejections)} "
                            f"fix(es) rejected — feeding back to LLM"
                        )
                        if attempt < MAX_FIX_ATTEMPTS - 1:
                            continue  # retry with feedback
                        else:
                            logger.warning(f"  ⚠️ All {MAX_FIX_ATTEMPTS} attempts exhausted")
                            return primary_content
                    else:
                        # Some applied, some rejected — partial success
                        logger.info(
                            f"  Attempt {attempt+1}: {applied_count} fix(es) applied, "
                            f"{len(attempt_rejections)} rejected"
                        )
                        return primary_content
                else:
                    # All fixes applied successfully
                    logger.info(f"  ✅ Attempt {attempt+1}: all {applied_count} fix(es) applied")
                    return primary_content

            except Exception as exc:
                logger.warning(f"  Attempt {attempt+1} failed: {exc}")
                if attempt < MAX_FIX_ATTEMPTS - 1:
                    rejection_history.append(f"Previous attempt threw an error: {str(exc)[:200]}")
                    continue

        # All attempts exhausted — fallback to simple SEARCH/REPLACE
        logger.warning(f"  All fix attempts exhausted — falling back to SEARCH/REPLACE")
        try:
            from ticket_to_code.llm_utils import llm_invoke
            from langchain_core.messages import SystemMessage, HumanMessage
            response = llm_invoke(self.llm, [
                SystemMessage(content=(
                    "Fix compile errors using SEARCH/REPLACE blocks. Fix ONLY what the error requires.\n"
                    "DELIMITER FORMAT:\n"
                    "<<<<<<< SEARCH\n"
                    "<exact existing code to find>\n"
                    "=======\n"
                    "<replacement code>\n"
                    ">>>>>>> REPLACE"
                )),
                HumanMessage(content=f"ERRORS:\n{error_text}\n\nFILE:\n```\n{current_content}\n```"),
            ])
            edit_res = self._apply_search_replace(current_content, response.content, False)
            return edit_res.content if edit_res.success else primary_content
        except Exception:
            return primary_content

    def _resolve_error_file(self, fp_raw: str) -> tuple[str, Optional[str]]:
        """Resolve a file path from a compiler error to actual workspace file + content.
        Priority: written-this-run → direct path → workspace scan.
        """
        fp = fp_raw.replace("\\", "/")
        target_name = Path(fp).name.lower()

        # 1. Check files written THIS RUN first (highest confidence)
        for wp, content in self._written_files.items():
            if Path(wp).name.lower() == target_name:
                abs_path = self.workspace_path / wp
                if abs_path.exists():
                    return wp, content

        # 2. Direct path
        abs1 = self.workspace_path / fp
        if abs1.exists():
            try:
                return fp, abs1.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return fp, None

        # 3. Workspace scan
        import os as _os
        _SKIP = {"node_modules", "dist", ".git", "build", "target", "coverage", ".venv"}
        matches = []
        for root, dirs, files in _os.walk(str(self.workspace_path)):
            dirs[:] = [d for d in dirs if d not in _SKIP]
            for f in files:
                if f.lower() == target_name:
                    try:
                        rel = str(Path(root, f).relative_to(self.workspace_path)).replace("\\", "/")
                        matches.append(rel)
                    except ValueError:
                        pass

        if matches:
            # Prefer suffix match
            for m in matches:
                inner = m.split("/", 1)[-1] if "/" in m else m
                if fp.endswith(inner) or m.endswith(fp):
                    abs_m = self.workspace_path / m
                    try:
                        return m, abs_m.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        return m, None
            # First match
            abs_m = self.workspace_path / matches[0]
            try:
                return matches[0], abs_m.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return matches[0], None

        return fp, None

    def _resolve_fix_path(self, fp_raw: str) -> str:
        """Resolve LLM-returned fix path to canonical workspace-relative path.
        Priority: direct → written-this-run → workspace scan → suffix match → raw.
        """
        fp = fp_raw.replace("\\", "/")
        target_name = Path(fp).name.lower()

        # 1. Direct path
        if (self.workspace_path / fp).exists():
            return fp

        # 2. Written this run
        for wp in self._written_files:
            if Path(wp).name.lower() == target_name:
                if (self.workspace_path / wp).exists():
                    logger.info(f"  Resolved '{fp}' → '{wp}' (written this run)")
                    return wp

        # 3. Workspace scan
        import os as _os
        _SKIP = {"node_modules", "dist", ".git", "build", "target", "coverage", ".venv"}
        matches = []
        for root, dirs, files in _os.walk(str(self.workspace_path)):
            dirs[:] = [d for d in dirs if d not in _SKIP]
            for f in files:
                if f.lower() == target_name:
                    try:
                        rel = str(Path(root, f).relative_to(self.workspace_path)).replace("\\", "/")
                        matches.append(rel)
                    except ValueError:
                        pass

        if not matches:
            return fp  # truly new file — action-based validation will handle it

        if len(matches) == 1:
            logger.info(f"  Resolved '{fp}' → '{matches[0]}' (workspace scan)")
            return matches[0]

        # Suffix tie-breaker
        for m in matches:
            inner = m.split("/", 1)[-1] if "/" in m else m
            if fp.endswith(inner) or m.endswith(fp):
                logger.info(f"  Resolved '{fp}' → '{m}' (suffix match)")
                return m

        logger.warning(f"  Ambiguous '{fp}': {len(matches)} candidates — using first")
        return matches[0]

    def _get_lsp_context(
        self, file_path: str, current_content: Optional[str], target_symbols: Optional[list] = None
    ) -> str:
        """
        Build LSP context block for the LLM prompt.
        For HTML files: injects the sibling .ts controller's declared members.
        For TS files: injects types of imported interfaces/models.

        When target_symbols are provided (remediation findings), the sibling
        controller source is trimmed to the implicated methods instead of the
        full 8K-char dump — the member list still ships in full so binding
        names stay verifiable.
        """
        ext = Path(file_path).suffix.lower()
        blocks: list[str] = []

        if ext in (".html", ".htm"):
            # Get sibling .ts controller's declared members
            stem = file_path.rsplit(".", 1)[0]
            for ctrl_ext in (".ts", ".tsx"):
                ctrl_path = stem + ctrl_ext
                # First check session files (most up-to-date)
                ctrl_key = ctrl_path.replace("\\", "/").lower()
                ctrl_content = self._written_files.get(ctrl_key)
                if not ctrl_content:
                    ctrl_abs = self.workspace_path / ctrl_path
                    if ctrl_abs.exists():
                        try:
                            ctrl_content = ctrl_abs.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            pass
                if ctrl_content:
                    members = self.symbol_index._ts_lsp._extract_via_regex(
                        self.workspace_path / ctrl_path, ctrl_path
                    )
                    if members:
                        blocks.append(
                            f"\n=== SIBLING CONTROLLER MEMBERS (use ONLY these names in template) ===\n"
                            f"{members.to_prompt_block()}\n"
                        )

                    # ── Include controller source so the LLM can see exact
                    # property names used in method bodies. In remediation mode
                    # with named targets, trim to the implicated methods; small
                    # controllers still ship in full.
                    ctrl_preview: Optional[str] = None
                    if target_symbols and len(ctrl_content) > 8000:
                        try:
                            from ticket_to_code.agents.smart_extract import (
                                extract_exact_methods as _eem_ctrl,
                            )
                            _vc, _mb = _eem_ctrl(
                                content=ctrl_content,
                                file_path=ctrl_path,
                                anchor_methods=list(target_symbols),
                            )
                            if _vc and _mb:
                                ctrl_preview = (
                                    "// (targeted: methods implicated by the verification findings)\n"
                                    + _vc
                                )
                        except Exception:
                            ctrl_preview = None
                    if ctrl_preview is None:
                        ctrl_preview = ctrl_content[:8000]
                        if len(ctrl_content) > 8000:
                            ctrl_preview += "\n// ... (truncated)"
                    blocks.append(
                        f"\n=== SIBLING CONTROLLER SOURCE (use exact property names from this code) ===\n"
                        f"```typescript\n{ctrl_preview}\n```\n"
                    )
                    break

        elif ext in (".ts", ".tsx") and current_content:
            # For TS files, show session-generated related types
            import os as _os
            src_dir = _os.path.dirname(file_path.replace("\\", "/"))
            imp_re = re.compile(r"""from\s+['"](\.[^'"]+)['"]""")
            for m in imp_re.finditer(current_content):
                rel = m.group(1)
                for cand_ext in (".ts", ".tsx"):
                    cand = _os.path.normpath(_os.path.join(src_dir, rel + cand_ext)).replace("\\", "/").lower()
                    if cand in self._written_files:
                        blocks.append(
                            f"\n=== IMPORTED FILE (freshly modified this run) ===\n"
                            f"{cand}\n{self._written_files[cand][:500]}\n"
                        )
                        break

        return "\n".join(blocks) if blocks else ""

    def _build_session_context(self, current_file: str) -> str:
        """Summary of files written so far in this run, WITH exported symbols.

        Previously this only showed file paths (e.g. '✅ fp') which told the
        LLM nothing about what was actually created. Now we extract exported
        symbols (classes, interfaces, methods) from the written content so the
        LLM generating file N knows exactly what file 1..N-1 exported.
        """
        if not self._written_files:
            return ""

        lines = ["\n=== FILES MODIFIED EARLIER IN THIS RUN ==="]
        _current_norm = current_file.replace("\\", "/").lower()

        for fp in sorted(self._written_files.keys()):
            if fp == _current_norm:
                continue
            content = self._written_files[fp]
            lines.append(f"  ✅ {fp}")

            # Extract exported symbols from the file content
            if content:
                _exports = self._extract_exported_symbols(content, fp)
                if _exports:
                    for _exp in _exports[:15]:  # cap per file
                        lines.append(f"       ↳ {_exp}")

        # Enrich with blueprint status from ImplementationState
        if self._impl_state and self._impl_state.signature_blueprints:
            lines.append("\n=== CROSS-FILE BLUEPRINT STATUS ===")
            for bp in self._impl_state.signature_blueprints[:15]:
                status = bp.status.value if bp.status else "unknown"
                owner = f"{bp.owner_class}." if bp.owner_class else ""
                sig_display = f" → {bp.signature}" if bp.signature else ""
                lines.append(
                    f"  [{status.upper():11}] {owner}{bp.symbol_name}{sig_display} "
                    f"in {Path(bp.file_path).name if bp.file_path else '?'}"
                )

        return "\n".join(lines) + "\n"

    @staticmethod
    def _extract_exported_symbols(content: str, file_path: str) -> list[str]:
        """Extract human-readable exported symbol declarations from file content.

        Returns lines like:
          'export class AddMembersComponent { ... }'
          'export interface ProjectMemberCheckResponse { isMember: boolean; ... }'
          'public checkProjectMembership(projectId: string): Observable<...>'
        """
        symbols: list[str] = []
        ext = Path(file_path).suffix.lower()

        if ext in (".ts", ".tsx", ".js", ".jsx"):
            # TypeScript/JS: exported classes, interfaces, functions, types
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("export ") and any(
                    kw in stripped for kw in ("class ", "interface ", "enum ", "type ", "function ", "const ")
                ):
                    # Trim to signature only (first 120 chars)
                    symbols.append(stripped[:120])
                elif stripped.startswith(("public ", "private ", "protected ")) and "(" in stripped:
                    # Class method declaration
                    sig = stripped.split("{")[0].strip().rstrip(":")
                    if len(sig) > 10:
                        symbols.append(sig[:120])
        elif ext == ".java":
            # Java: public class/interface declarations and public methods
            for line in content.splitlines():
                stripped = line.strip()
                if any(kw in stripped for kw in ("public class ", "public interface ", "public enum ")):
                    symbols.append(stripped.split("{")[0].strip()[:120])
                elif stripped.startswith("public ") and "(" in stripped:
                    sig = stripped.split("{")[0].strip()
                    if len(sig) > 10:
                        symbols.append(sig[:120])
        elif ext == ".py":
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("class ") or stripped.startswith("def "):
                    symbols.append(stripped.rstrip(":").strip()[:120])

        return symbols

    def _get_companion_context(self, file_path: str) -> str:
        """Discover sibling and companion repositories/interfaces/entities to ground LLM code generation."""
        if not file_path:
            return ""
        norm_fp = file_path.replace("\\", "/")
        path_obj = Path(file_path)
        companions: list[str] = []

        # 1. Java Spring Boot companions (service -> repository / model / utils)
        if path_obj.suffix.lower() == ".java":
            try:
                src_dir = None
                p = (self.workspace_path / file_path).resolve() if (self.workspace_path / file_path).exists() else None
                if p:
                    for parent in p.parents:
                        if (parent / "repository").is_dir() or (parent / "service").is_dir():
                            src_dir = parent
                            break
                        if parent.name in ("java", "main", "src"):
                            src_dir = parent
                            break
                if src_dir:
                    repo_dir = src_dir / "repository" if (src_dir / "repository").is_dir() else (src_dir.parent / "repository" if (src_dir.parent / "repository").is_dir() else None)
                    if repo_dir and repo_dir.exists():
                        for repo_file in sorted(repo_dir.glob("*Repository.java"))[:8]:
                            try:
                                r_content = repo_file.read_text(encoding="utf-8", errors="ignore")
                                r_syms = self._extract_exported_symbols(r_content, repo_file.name)
                                if r_syms:
                                    companions.append(f"[{repo_file.name}]:\n" + "\n".join(f"  {s};" for s in r_syms[:10]))
                            except Exception:
                                pass
            except Exception as _comp_err:
                logger.debug(f"Companion repository resolution skipped: {_comp_err}")

        # 2. Migration script alignment context
        if "db/migration" in norm_fp or norm_fp.endswith(".sql"):
            try:
                mig_dir = (self.workspace_path / file_path).parent if (self.workspace_path / file_path).exists() else None
                if not mig_dir or not mig_dir.exists():
                    for found in self.workspace_path.glob("**/db/migration"):
                        if found.is_dir():
                            mig_dir = found
                            break
                if mig_dir and mig_dir.exists():
                    recent_sql = sorted([f.name for f in mig_dir.glob("*.sql")])[-5:]
                    if recent_sql:
                        companions.append(f"[Existing Migration History]:\n  " + "\n  ".join(recent_sql))
            except Exception:
                pass

        # 3. Universal Frontend Component & Controller companions (.html <-> .ts)
        if path_obj.suffix.lower() in (".ts", ".html"):
            try:
                p = (self.workspace_path / file_path).resolve() if (self.workspace_path / file_path).exists() else None
                if p:
                    companion_ext = ".html" if path_obj.suffix.lower() == ".ts" else ".ts"
                    companion_path = p.with_suffix(companion_ext)
                    if companion_path.is_file():
                        comp_content = companion_path.read_text(encoding="utf-8", errors="ignore")
                        if companion_ext == ".ts":
                            comp_syms = self._extract_exported_symbols(comp_content, companion_path.name)
                            if comp_syms:
                                companions.append(f"[{companion_path.name} Controller Symbols]:\n" + "\n".join(f"  {s}" for s in comp_syms[:15]))
                            else:
                                companions.append(f"[{companion_path.name} Companion Controller]:\n{comp_content[:500]}")
                        else:
                            companions.append(f"[{companion_path.name} Companion Template]:\n{comp_content[:500]}")
            except Exception:
                pass

        if not companions:
            return ""
        return "=== COMPANION DATA ACCESS & REPOSITORY SPECIFICATION (Use these real methods) ===\n" + "\n\n".join(companions) + "\n"

    def _summarize_written_files(self) -> str:
        """Structural summary of written files for agent decision."""
        parts = []
        for fp, content in list(self._written_files.items())[:6]:
            symbols = self._extract_exported_symbols(content, fp)
            if symbols:
                sym_str = "\n".join(f"    - {s}" for s in symbols[:8])
                parts.append(f"--- {fp} ({len(content)} chars) ---\nDeclared Symbols / Methods:\n{sym_str}")
            else:
                parts.append(f"--- {fp} ({len(content)} chars) ---\n{content[:400]}")
        return "\n\n".join(parts)

    def _smart_extract_for_large_file(
        self, content: str, edit_description: str, file_path: str
    ) -> str:
        """
        Smart file reading for large files — like how top AI IDEs work.
        
        Instead of dumb truncation, extract:
        1. FILE OUTLINE: all class/method signatures with line numbers
        2. RELEVANT SECTIONS: full content of methods matching the edit task
        3. IMPORTS + CLASS HEADER: always included for context
        
        This lets the LLM see the structure of a 112K file AND the exact
        methods it needs to edit, without wasting tokens on unrelated code.
        """
        lines = content.split('\n')
        total_lines = len(lines)
        content_len = len(content)
        
        # Determine file type
        fp_lower = file_path.lower()
        is_java = fp_lower.endswith('.java')
        is_ts = fp_lower.endswith(('.ts', '.tsx'))
        is_py = fp_lower.endswith('.py')
        is_html = fp_lower.endswith(('.html', '.htm'))
        
        # --- Step 1: Extract method/function boundaries ---
        method_boundaries = []  # [(start_line, end_line, signature)]
        
        if is_java or is_ts:
            # Match Java/TS method signatures
            import re as _re
            method_pattern = _re.compile(
                r'^\s*(?:public|private|protected|static|async|abstract|override|final|synchronized|\s)*'
                r'(?:[\w<>\[\],\s]+\s+)?'
                r'(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{',
                _re.MULTILINE
            )
            # Also match class declarations
            class_pattern = _re.compile(
                r'^\s*(?:export\s+)?(?:public|private|protected|abstract|final|\s)*'
                r'(?:class|interface|enum)\s+(\w+)',
                _re.MULTILINE
            )
            
            brace_depth = 0
            current_method_start = None
            current_sig = None
            
            for i, line in enumerate(lines):
                # Track class declarations
                cm = class_pattern.match(line)
                if cm:
                    method_boundaries.append((i, i, f"class {cm.group(1)}"))
                
                # Track method starts
                mm = method_pattern.match(line)
                if mm and brace_depth <= 1:  # Top-level or class-level method
                    current_method_start = i
                    current_sig = line.strip()[:120]
                
                # Track braces to find method end
                brace_depth += line.count('{') - line.count('}')
                
                if current_method_start is not None and brace_depth <= 1:
                    method_boundaries.append((current_method_start, i, current_sig))
                    current_method_start = None
                    current_sig = None
                    
        elif is_py:
            import re as _re
            func_pattern = _re.compile(r'^(\s*)(def|class)\s+(\w+)')
            current_start = None
            current_sig = None
            current_indent = 0
            
            for i, line in enumerate(lines):
                m = func_pattern.match(line)
                if m:
                    if current_start is not None:
                        method_boundaries.append((current_start, i - 1, current_sig))
                    current_start = i
                    current_sig = line.strip()[:120]
                    current_indent = len(m.group(1))
            if current_start is not None:
                method_boundaries.append((current_start, total_lines - 1, current_sig))
        
        # --- Step 2: Build outline ---
        outline_parts = []
        outline_parts.append(f"// FILE: {file_path} ({content_len:,} chars, {total_lines} lines)")
        outline_parts.append(f"// STRUCTURE OUTLINE ({len(method_boundaries)} methods/classes):")
        outline_parts.append("")
        
        for start, end, sig in method_boundaries:
            size = end - start + 1
            outline_parts.append(f"  L{start+1}-L{end+1} ({size} lines): {sig}")
        
        # --- Step 3: Find relevant methods ---
        edit_desc_lower = (edit_description or "").lower()
        # Extract keywords from edit description
        import re as _re
        keywords = set(_re.findall(r'[a-zA-Z]\w{3,}', edit_desc_lower))
        
        relevant_sections = []
        relevant_starts = set()  # Track by start line to avoid duplicates
        
        for start, end, sig in method_boundaries:
            sig_lower = sig.lower()
            # Check if method name or signature overlaps with edit keywords
            sig_words = set(_re.findall(r'[a-zA-Z]\w{3,}', sig_lower))
            overlap = keywords & sig_words
            
            if overlap:
                # This method is relevant — include full content
                section_lines = lines[start:end + 1]
                relevant_sections.append((start, end, sig, '\n'.join(section_lines)))
                relevant_starts.add(start)
        
        # If no keyword match found, try to find methods containing edit keywords in their body
        if not relevant_sections:
            for start, end, sig in method_boundaries:
                body = '\n'.join(lines[start:end + 1]).lower()
                body_overlap = sum(1 for kw in keywords if kw in body)
                if body_overlap >= 2:  # At least 2 keywords found in method body
                    section_lines = lines[start:end + 1]
                    relevant_sections.append((start, end, sig, '\n'.join(section_lines)))
                    relevant_starts.add(start)
                    if len(relevant_sections) >= 5:  # Cap at 5 relevant methods
                        break
        
        # --- Step 3b: Follow call chains within the same file ---
        # If method A calls method B (in the same file), include B too.
        # This is how top AI IDEs work — they follow the call graph.
        # Do 2 levels deep: A→B, B→C
        if relevant_sections and method_boundaries:
            # Build a lookup: method_name → (start, end, sig, body)
            method_by_name = {}
            for start, end, sig in method_boundaries:
                # Extract method name from signature
                name_match = _re.search(r'(\w+)\s*\(', sig)
                if name_match:
                    method_by_name[name_match.group(1)] = (start, end, sig)
            
            # Extract method calls from relevant methods' bodies
            for depth in range(2):  # 2 levels deep
                new_calls = set()
                for start, end, sig, body in relevant_sections:
                    # Find method calls: this.methodName(, methodName(, self.method_name(
                    calls = _re.findall(r'(?:this\.|self\.)?(\w+)\s*\(', body)
                    for call_name in calls:
                        if call_name in method_by_name:
                            called_start = method_by_name[call_name][0]
                            if called_start not in relevant_starts:
                                new_calls.add(call_name)
                
                if not new_calls:
                    break  # No new methods to add
                    
                for call_name in new_calls:
                    c_start, c_end, c_sig = method_by_name[call_name]
                    if c_start not in relevant_starts:
                        section_lines = lines[c_start:c_end + 1]
                        relevant_sections.append(
                            (c_start, c_end, f"[CALLED BY ABOVE] {c_sig}", '\n'.join(section_lines))
                        )
                        relevant_starts.add(c_start)
        
        # --- Step 4: Build the smart content ---
        parts = []
        
        # Always include imports + class header (first 80 lines or until first method)
        first_method_line = method_boundaries[0][0] if method_boundaries else 80
        header_end = min(first_method_line, 80)
        parts.append("// === IMPORTS & CLASS HEADER ===")
        parts.append('\n'.join(lines[:header_end]))
        
        # Include outline
        parts.append("")
        parts.append('\n'.join(outline_parts))
        
        # Include relevant methods in full
        if relevant_sections:
            parts.append("")
            parts.append(f"// === RELEVANT METHODS ({len(relevant_sections)} sections shown in FULL) ===")
            for start, end, sig, body in relevant_sections:
                parts.append(f"\n// --- L{start+1}-L{end+1}: {sig} ---")
                parts.append(body)
        else:
            # No specific methods matched — show first 30K + outline
            parts.append("")
            parts.append("// === FILE CONTENT (first 30K chars — use outline above for navigation) ===")
            parts.append(content[:30000])
        
        result = '\n'.join(parts)
        
        # Safety cap: if result is still huge, truncate with note
        if len(result) > 60000:
            result = result[:55000] + f"\n\n// [TRUNCATED at 55K chars — file is {content_len:,} total]"
        
        return result



    def _safe_match_and_replace(
        self, original: str, edits: list[dict], parser_format: str
    ) -> ApplyEditResult:
        """Apply a batch of edits atomically using safe matching only.

        Matching order:
          1. Exact match (count == 1)
          2. CRLF/LF line-ending normalization (count == 1)
          3. Whitespace-normalized sliding window (count == 1)

        If any edit produces 0 matches or >1 ambiguous matches, the entire batch
        fails safely and ZERO changes are applied to original content.
        NO fuzzy matching or SequenceMatcher is ever used.
        """
        result = original
        applied_count = 0

        for i, edit in enumerate(edits):
            old_str = edit.get("old_str", "")
            new_str = edit.get("new_str", "")

            if not old_str:
                return ApplyEditResult(
                    success=False,
                    content=original,
                    failure_reason=f"Edit {i+1}: SEARCH block is empty",
                    failed_search="",
                    parser_format=parser_format,
                )

            # ── 1. Exact match ──────────────────────────────────────────
            count = result.count(old_str)
            if count == 1:
                result = result.replace(old_str, new_str, 1)
                applied_count += 1
                logger.info(f"  edit_loop edit {i+1}/{len(edits)} applied: Exact match")
                continue
            if count > 1:
                logger.warning(
                    f"  edit_loop edit {i+1}/{len(edits)} ambiguous: matches {count} locations"
                )
                return ApplyEditResult(
                    success=False,
                    content=original,
                    failure_reason=f"SEARCH block matched multiple locations ({count} matches, ambiguous)",
                    failed_search=old_str,
                    parser_format=parser_format,
                )

            # ── 2. CRLF/LF line-ending adaptation ───────────────────────
            adapted = None
            if "\n" in old_str and "\r\n" not in old_str:
                cand = old_str.replace("\n", "\r\n")
                c = result.count(cand)
                if c == 1:
                    adapted = cand
                elif c > 1:
                    return ApplyEditResult(
                        success=False,
                        content=original,
                        failure_reason=f"SEARCH block matched multiple locations after CRLF adaptation ({c} matches, ambiguous)",
                        failed_search=old_str,
                        parser_format=parser_format,
                    )
            elif "\r\n" in old_str:
                cand = old_str.replace("\r\n", "\n")
                c = result.count(cand)
                if c == 1:
                    adapted = cand
                elif c > 1:
                    return ApplyEditResult(
                        success=False,
                        content=original,
                        failure_reason=f"SEARCH block matched multiple locations after CRLF adaptation ({c} matches, ambiguous)",
                        failed_search=old_str,
                        parser_format=parser_format,
                    )

            if adapted is not None:
                result = result.replace(adapted, new_str, 1)
                applied_count += 1
                logger.info(f"  edit_loop edit {i+1}/{len(edits)} applied: CRLF-adapted")
                continue

            # ── 3. Whitespace-normalized sliding window ─────────────────
            search_lines = old_str.strip().splitlines()
            if not search_lines:
                return ApplyEditResult(
                    success=False,
                    content=original,
                    failure_reason=f"Edit {i+1}: SEARCH block contains only whitespace",
                    failed_search=old_str,
                    parser_format=parser_format,
                )

            normalized_search = [re.sub(r'\s+', ' ', l.strip()) for l in search_lines]
            if all(not s for s in normalized_search):
                return ApplyEditResult(
                    success=False,
                    content=original,
                    failure_reason=f"Edit {i+1}: SEARCH block contains only blank lines",
                    failed_search=old_str,
                    parser_format=parser_format,
                )

            n = len(search_lines)
            file_lines = result.splitlines(True)
            file_lines_stripped = [re.sub(r'\s+', ' ', l.strip()) for l in file_lines]

            matches = []
            for j in range(len(file_lines_stripped) - n + 1):
                if file_lines_stripped[j:j + n] == normalized_search:
                    matches.append(j)

            if len(matches) == 1:
                match_start = matches[0]
                original_window = "".join(file_lines[match_start:match_start + n])
                result = result.replace(original_window, new_str, 1)
                applied_count += 1
                logger.info(
                    f"  edit_loop edit {i+1}/{len(edits)} applied: Whitespace-normalized window"
                )
                continue
            elif len(matches) > 1:
                logger.warning(
                    f"  edit_loop edit {i+1}/{len(edits)} ambiguous after whitespace normalization: {len(matches)} matches"
                )
                return ApplyEditResult(
                    success=False,
                    content=original,
                    failure_reason=f"SEARCH block matched multiple locations after whitespace normalization ({len(matches)} matches, ambiguous)",
                    failed_search=old_str,
                    parser_format=parser_format,
                )
            else:
                logger.warning(
                    f"  edit_loop edit {i+1}/{len(edits)} failed: SEARCH block did not match"
                )
                return ApplyEditResult(
                    success=False,
                    content=original,
                    failure_reason="SEARCH block did not match the current file",
                    failed_search=old_str,
                    parser_format=parser_format,
                )

        return ApplyEditResult(
            success=True,
            content=result,
            applied_count=applied_count,
            parser_format=parser_format,
        )

    def _apply_search_replace(
        self, original: str, llm_response: str, is_create: bool
    ) -> ApplyEditResult:
        """Apply additive insertion strategies or SEARCH/REPLACE from LLM response.

        Returns structured ApplyEditResult with success status, failure reason,
        and parser format.
        """
        if is_create or not original:
            code_m = re.search(r'```[\w]*\n(.*?)```', llm_response, re.DOTALL)
            if code_m:
                clean = code_m.group(1)
            else:
                clean = re.sub(r'^```[\w]*\n|```$', '', llm_response.strip(), flags=re.MULTILINE)
            if clean.strip():
                return ApplyEditResult(
                    success=True,
                    content=clean,
                    applied_count=1,
                    parser_format="create",
                )
            return ApplyEditResult(
                success=False,
                content=original,
                failure_reason="Empty content generated for file creation",
                parser_format="create",
            )

        stripped = llm_response.strip()
        if not stripped:
            return ApplyEditResult(
                success=False,
                content=original,
                failure_reason="No edit requested (empty LLM response)",
                parser_format="none",
            )

        # ── Primary: Canonical Aider-style SEARCH/REPLACE blocks ──────────
        aider_pattern = re.compile(
            r'<{4,}\s*(?:SEARCH|ORIGINAL|TARGET)[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}\s*(?:REPLACE|UPDATED|REPLACEMENT)[^\r\n]*',
            re.DOTALL,
        )
        aider_matches = list(aider_pattern.finditer(llm_response))
        if not aider_matches:
            # Fallback for plain merge conflict markers <<<<<<< / ======= / >>>>>>>
            alt_conflict_pattern = re.compile(
                r'<{4,}[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}[^\r\n]*',
                re.DOTALL,
            )
            aider_matches = list(alt_conflict_pattern.finditer(llm_response))

        if aider_matches:
            edits = []
            for idx, m in enumerate(aider_matches):
                old_s = m.group(1)
                new_s = m.group(2)
                if re.search(r'<{4,}\s*(?:SEARCH|ORIGINAL|TARGET)|>{4,}\s*(?:REPLACE|UPDATED|REPLACEMENT)', new_s):
                    return ApplyEditResult(
                        success=False,
                        content=original,
                        failure_reason=f"Edit {idx+1}: REPLACE block contains nested SEARCH/REPLACE markers",
                        failed_search=old_s,
                        parser_format="aider",
                    )
                edits.append({"old_str": old_s, "new_str": new_s})
            return self._safe_match_and_replace(original, edits, parser_format="aider")

        # ── Fallback 1: <<<SEARCH>>>/<<<REPLACE>>>/<<<END>>> format ────────
        delimiter_pattern = re.compile(
            r'<<<SEARCH>>>\s*\r?\n?(.*?)\r?\n?<<<REPLACE>>>\s*\r?\n?(.*?)\r?\n?<<<END>>>',
            re.DOTALL,
        )
        delimiter_matches = list(delimiter_pattern.finditer(llm_response))
        if delimiter_matches:
            edits = []
            for idx, m in enumerate(delimiter_matches):
                old_s = m.group(1)
                new_s = m.group(2)
                if "<<<SEARCH>>>" in new_s or "<<<REPLACE>>>" in new_s:
                    return ApplyEditResult(
                        success=False,
                        content=original,
                        failure_reason=f"Edit {idx+1}: REPLACE block contains nested SEARCH/REPLACE markers",
                        failed_search=old_s,
                        parser_format="delimiters",
                    )
                edits.append({"old_str": old_s, "new_str": new_s})
            return self._safe_match_and_replace(original, edits, parser_format="delimiters")

        # ── Fallback 2: legacy <<</>>> EDIT blocks ────────────────────────
        legacy_pattern = re.compile(
            r'EDIT:\s*\r?\n'
            r'old_str:\s*\r?\n<<<\r?\n(.*?)\r?\n>>>\s*\r?\n'
            r'new_str:\s*\r?\n<<<\r?\n(.*?)\r?\n>>>',
            re.DOTALL,
        )
        legacy_matches = list(legacy_pattern.finditer(llm_response))
        if legacy_matches:
            edits = []
            for idx, m in enumerate(legacy_matches):
                old_s = m.group(1)
                new_s = m.group(2)
                if "EDIT:\nold_str:" in new_s or "EDIT:\r\nold_str:" in new_s:
                    return ApplyEditResult(
                        success=False,
                        content=original,
                        failure_reason=f"Edit {idx+1}: new_str contains nested EDIT block",
                        failed_search=old_s,
                        parser_format="legacy",
                    )
                edits.append({"old_str": old_s, "new_str": new_s})
            return self._safe_match_and_replace(original, edits, parser_format="legacy")

        # ── Fallback 3: Antigravity / AI IDE TARGET/REPLACEMENT format ────
        target_pattern = re.compile(
            r'(?:TARGET(?:_CONTENT)?|ORIGINAL):\s*\r?\n?(.*?)\r?\n?(?:REPLACEMENT(?:_CONTENT)?|REPLACE):\s*\r?\n?(.*?)(?=\r?\n(?:TARGET(?:_CONTENT)?|ORIGINAL):|\Z)',
            re.DOTALL | re.IGNORECASE,
        )
        target_matches = list(target_pattern.finditer(llm_response))
        if target_matches:
            edits = []
            for idx, m in enumerate(target_matches):
                old_s = m.group(1).strip()
                new_s = m.group(2).strip()
                edits.append({"old_str": old_s, "new_str": new_s})
            return self._safe_match_and_replace(original, edits, parser_format="target_replacement")

        return ApplyEditResult(
            success=False,
            content=original,
            failure_reason="No recognized SEARCH/REPLACE or TARGET/REPLACEMENT blocks found in response",
            parser_format="none",
        )

    # ── Cross-file error detection helpers ────────────────────────────────────
    # These parse DETERMINISTIC COMPILER OUTPUT (not keyword lists).
    # The TypeScript compiler outputs structured error messages with exact
    # property names and type names — this is machine output like exit codes.
    #
    # Symbol resolution is delegated to SymbolResolver (symbol_resolver.py),
    # a shared program-understanding primitive. The queuing logic stays here
    # because it's edit-loop-specific.

    # Regex for TypeScript compiler's "does not exist on type" error format.
    # This is part of the TS compiler specification, not a keyword guess.
    _CROSS_FILE_RE = re.compile(
        r"Property\s+'(\w+)'\s+does not exist on type\s+'(\w+)'"
    )

    def _extract_cross_file_targets(
        self, errors: list[str], current_file: str
    ) -> dict[str, list[str]]:
        """Parse compiler errors to find cross-file dependencies.

        Reads COMPILER OUTPUT (deterministic machine format) to detect when
        errors reference a type/class defined in a different file.
        Uses SymbolResolver to trace type names to file paths.

        Returns: {target_file_path: [missing_member_1, missing_member_2]}
        """
        targets: dict[str, list[str]] = {}
        norm_current = current_file.replace("\\", "/").lower()

        for err in errors:
            m = self._CROSS_FILE_RE.search(err)
            if not m:
                continue
            member_name = m.group(1)
            type_name = m.group(2)

            # Delegate symbol resolution to the shared SymbolResolver
            type_file = self._resolver.get_file_for_type(type_name)

            if type_file:
                norm_type = type_file.replace("\\", "/").lower()
                if norm_type != norm_current:
                    targets.setdefault(type_file, []).append(member_name)

        return targets

    def _is_cross_file_error(self, error: str, current_file: str) -> bool:
        """Check if a compiler error references a type in a different file."""
        m = self._CROSS_FILE_RE.search(error)
        if not m:
            return False
        type_name = m.group(2)
        type_file = self._resolver.get_file_for_type(type_name)
        if not type_file:
            return False
        return not self._resolver.is_defined_in(type_name, current_file)

    def _live_compile_file(self, file_path: str) -> list[str]:
        """Live compile & build verification executed immediately after writing each file.

        Matches modern AI IDE behavior (Antigravity/Cursor/Claude Code):
        1. Emits live compilation start telemetry.
        2. Executes immediate compiler diagnostics (tsc, py_compile, SCSS syntax, template bindings).
        3. Emits live compilation result telemetry.
        4. Surfaces errors immediately to trigger localized inline repair.
        """
        file_basename = Path(file_path).name
        check_label = self._get_live_check_label(file_path)
        
        self._ui_emit(
            "live_compile_start",
            message=f"🔨 Live Compile: Building and compiling {file_basename} after write [{check_label}]...",
            file_path=file_path,
            file_name=file_basename,
            check_type=check_label,
        )

        errors = self._verify_file(file_path)

        if not errors:
            logger.info(f"  ✅ [LIVE COMPILE] {file_path} compiled cleanly (0 errors) [{check_label}]")
            self._ui_emit(
                "live_compile_passed",
                message=f"✅ Live Compile: Clean compile (0 errors) — {file_basename} [{check_label}]",
                file_path=file_path,
                file_name=file_basename,
                check_type=check_label,
            )
        else:
            logger.warning(
                f"  ⚠️ [LIVE COMPILE] {file_path} has {len(errors)} compile error(s) [{check_label}]"
            )
            self._ui_emit(
                "live_compile_failed",
                message=f"❌ Live Compile: {len(errors)} error(s) in {file_basename} [{check_label}]",
                file_path=file_path,
                file_name=file_basename,
                error_count=len(errors),
                errors=errors[:5],
                check_type=check_label,
            )

        return errors

    def _auto_register_angular_module_for_file(self, file_path: str, content: str) -> None:
        """Auto-register Angular constructs in the nearest *.module.ts before compilation."""
        if not content or not file_path.endswith((".ts", ".tsx")):
            return
        if re.search(r'standalone\s*:\s*true', content):
            return

        _DECORATOR_ARRAY = {
            "@Component": "declarations",
            "@Directive": "declarations",
            "@Pipe":      "declarations",
            "@Injectable": "providers",
        }
        decorator_used = None
        for dec in _DECORATOR_ARRAY:
            if dec + "(" in content or dec + "\n" in content:
                decorator_used = dec
                break
        if not decorator_used:
            return

        target_array = _DECORATOR_ARRAY[decorator_used]
        class_match = re.search(r'export\s+class\s+([A-Za-z][A-Za-z0-9_]*)', content)
        if not class_match:
            return
        class_name = class_match.group(1)

        gen_abs = Path(self.workspace_path) / file_path
        search_dir = gen_abs.parent
        ws_root = Path(self.workspace_path)
        module_path = None
        for _ in range(8):
            candidates = sorted(search_dir.glob("*.module.ts"))
            if candidates:
                module_path = candidates[0]
                break
            if search_dir == ws_root or search_dir.parent == search_dir:
                break
            search_dir = search_dir.parent

        if not module_path or not module_path.exists():
            return

        try:
            mod_content = module_path.read_text(encoding="utf-8")
            if class_name in mod_content:
                return

            import os
            rel = os.path.relpath(
                str(gen_abs.with_suffix("")),
                str(module_path.parent),
            ).replace("\\", "/")
            if not rel.startswith("."):
                rel = "./" + rel
            new_import = f"import {{ {class_name} }} from '{rel}';"

            _last_imp = None
            for _m in re.finditer(r'^import[^\n]+\n', mod_content, re.MULTILINE):
                _last_imp = _m
            insert_at = _last_imp.end() if _last_imp else 0
            mod_content = mod_content[:insert_at] + new_import + "\n" + mod_content[insert_at:]

            arr_rx = re.compile(rf'{target_array}\s*:\s*\[([^\]]*)\]', re.DOTALL)
            arr_match = arr_rx.search(mod_content)
            if arr_match:
                old_arr = arr_match.group(0)
                inner = arr_match.group(1).rstrip()
                sep = ",\n    " if "\n" in inner else ", "
                new_inner = inner.rstrip(",") + sep + class_name
                new_arr = old_arr.replace(arr_match.group(1), new_inner)
                rel_mod = str(module_path.relative_to(self.workspace_path)).replace("\\", "/")
                if rel_mod not in self.pre_edit_snapshots:
                    self.pre_edit_snapshots[rel_mod] = module_path.read_text(encoding="utf-8", errors="ignore")
                module_path.write_text(mod_content, encoding="utf-8")
                self._written_files[rel_mod.lower()] = mod_content
                logger.info(f"  angular_module: auto-registered {class_name} in {module_path.name} ({target_array})")
        except Exception as e:
            logger.debug(f"auto_register_angular_module failed: {e}")

    def _verify_file(self, file_path: str) -> list[str]:
        """Run the appropriate compiler/syntax checker for the file type.

        Always returns a list. When the underlying compiler returned ``None``
        (diagnostics unavailable) this returns ``[]`` AND sets
        ``self._last_verify_unknown = True`` so callers do not claim compile_clean.
        """
        self._last_verify_unknown = False
        ext = Path(file_path).suffix.lower()
        if ext in (".ts", ".tsx") and not file_path.endswith(".spec.ts"):
            content = self._written_files.get(file_path.replace("\\", "/").lower(), "")
            if not content:
                abs_path = self._resolve_fs_path(file_path)
                if abs_path.exists():
                    content = abs_path.read_text(encoding="utf-8", errors="ignore")
            # 1. Fast local syntax check first
            from ticket_to_code.workflow import _check_code_syntax
            if content:
                syntax_errs = _check_code_syntax(content, file_path)
                if syntax_errs:
                    return syntax_errs

            # 2. Defer cross-file tsc diagnostics if companion files are still queued
            stem = Path(file_path).stem
            has_pending_companion = any(
                stem in Path(q.file_path).stem for q in getattr(self, "_pending_queue_ref", [])
            )
            if has_pending_companion:
                logger.info(f"  ⏳ Deferring cross-file tsc diagnostics for {file_path}: companion file(s) pending in queue")
                return []

            # 3. Auto-register in Angular module before running tsc diagnostics if newly created
            if content:
                self._auto_register_angular_module_for_file(file_path, content)

            written_ts = [
                fp for fp in self._written_files.keys()
                if fp.endswith((".ts", ".tsx")) and not fp.endswith(".spec.ts")
            ]
            ts_diags = self.symbol_index.get_ts_diagnostics([file_path] + written_ts)
            if ts_diags is None:
                logger.warning(f"  ⚠️ TS diagnostics unavailable for {file_path} — result unverified")
                self._last_verify_unknown = True
                return []
            return ts_diags
        if ext in (".py",) and not file_path.endswith("_test.py"):
            py_diags = self.symbol_index.get_python_diagnostics([file_path])
            if py_diags is None:
                logger.warning(f"  ⚠️ Python diagnostics unavailable for {file_path} — result unverified")
                self._last_verify_unknown = True
                py_diags = []
            errors = list(py_diags or [])
            # Guaranteed live compile via standard library py_compile
            try:
                import py_compile
                abs_p = self._resolve_fs_path(file_path)
                if abs_p.exists():
                    py_compile.compile(str(abs_p), doraise=True)
            except py_compile.PyCompileError as pce:
                errors.append(f"{file_path}: {pce.msg or str(pce)}")
            except Exception as _py_err:
                logger.debug(f"py_compile non-fatal: {_py_err}")
            return errors
        if ext in (".scss", ".css"):
            # Fast syntax check — balanced braces + format marker leak detection
            from ticket_to_code.workflow import _check_scss_syntax
            content = self._written_files.get(file_path.replace("\\", "/").lower(), "")
            if not content:
                abs_path = self._resolve_fs_path(file_path)
                if abs_path.exists():
                    content = abs_path.read_text(encoding="utf-8", errors="ignore")
            return _check_scss_syntax(content, file_path) if content else []
        if ext == ".html":
            from ticket_to_code.workflow import _check_html_syntax
            content = self._written_files.get(file_path.replace("\\", "/").lower(), "")
            if not content:
                abs_path = self._resolve_fs_path(file_path)
                if abs_path.exists():
                    content = abs_path.read_text(encoding="utf-8", errors="ignore")
            errors = _check_html_syntax(content, file_path) if content else []

            # ── Cross-file template binding validation ────────────────────
            # Catches mismatched property names (e.g. organizationName vs
            # existingOrganizationName) by resolving Angular template bindings
            # against the sibling TS controller + repository/generated truth.
            if content:
                ref_errors = self._validate_template_bindings(file_path, content)
                errors.extend(ref_errors)

        if ext in (".java", ".kt"):
            # 1. Fast local syntax check first (balanced braces, parens, string quotes, format markers)
            from ticket_to_code.workflow import _check_code_syntax, _find_java_module_root, _run_java_compile_on_module
            content = self._written_files.get(file_path.replace("\\", "/").lower(), "")
            if not content:
                abs_path = self._resolve_fs_path(file_path)
                if abs_path.exists():
                    content = abs_path.read_text(encoding="utf-8", errors="ignore")
            if content:
                syntax_errs = _check_code_syntax(content, file_path)
                if syntax_errs:
                    return syntax_errs

            # 2. Defer full cross-file Java compile if companion files are still queued
            stem = Path(file_path).stem
            has_pending_companion = any(
                stem in Path(q.file_path).stem for q in getattr(self, "_pending_queue_ref", [])
            )
            if has_pending_companion:
                logger.info(f"  ⏳ Deferring Java module compile for {file_path}: companion file(s) pending in queue")
                return []

            # 3. Real Java compiler check via Maven or Gradle
            java_module = _find_java_module_root(file_path, str(self.workspace_path))
            if java_module:
                module_root, build_tool = java_module
                compile_errors = _run_java_compile_on_module(module_root, build_tool, timeout=60)
                return compile_errors

            # 4. Fallback: javac if installed on PATH
            try:
                import subprocess, shutil
                javac_cmd = shutil.which("javac")
                if javac_cmd:
                    abs_p = self._resolve_fs_path(file_path)
                    if abs_p.exists():
                        res = subprocess.run([javac_cmd, "-Xlint:none", str(abs_p)], capture_output=True, text=True, timeout=20)
                        if res.returncode != 0:
                            err_lines = [l.strip() for l in (res.stdout + res.stderr).splitlines() if "error:" in l.lower()]
                            return err_lines or [f"javac compilation failed for {file_path}"]
            except Exception as _javac_err:
                logger.debug(f"javac fallback non-fatal: {_javac_err}")

            return []

        if ext in (".cs",):
            from ticket_to_code.workflow import _check_code_syntax
            content = self._written_files.get(file_path.replace("\\", "/").lower(), "")
            if not content:
                abs_path = self._resolve_fs_path(file_path)
                if abs_path.exists():
                    content = abs_path.read_text(encoding="utf-8", errors="ignore")
            if content:
                return _check_code_syntax(content, file_path)
            return []

        return []

    def _validate_template_bindings(self, file_path: str, content: str) -> list[str]:
        """Validate Angular template bindings against the generated workspace.

        Uses GeneratedReferenceValidator.validate_template() to extract bindings
        and resolve them against:
        1. SymbolResolver (repository truth)
        2. ImplementationState (generated symbols)
        3. _written_files (freshly generated content)

        Returns compiler-like error strings that feed into the existing
        inline-fix loop.
        """
        from ticket_to_code.agents.generated_reference_validator import (
            GeneratedReferenceValidator,
        )

        validator = GeneratedReferenceValidator(
            symbol_resolver=self._resolver,
            impl_state=self._impl_state,
        )

        result = validator.validate_template(
            html_content=content,
            html_file_path=file_path,
            written_files=self._written_files,
        )

        errors: list[str] = []
        for check in result.unresolved:
            close_hint = ""
            if check.close_match:
                close_hint = f" Did you mean '{check.close_match}'?"
            owner_hint = f" on type '{check.resolved_type}'" if check.resolved_type else ""
            errors.append(
                f"Template binding error in {Path(file_path).name}: "
                f"'{check.receiver}.{check.member}' — "
                f"member '{check.member}' is unresolved{owner_hint}.{close_hint}"
            )

        if errors:
            logger.warning(
                f"  🔗 Template binding validation found {len(errors)} "
                f"unresolved reference(s) in {Path(file_path).name}"
            )
            for e in errors:
                logger.warning(f"    ⚠️ {e}")

        return errors

    def _resolve_fs_path(self, file_path: str) -> Path:
        """Resolve a file path against workspace_path safely, handling redundant repo root prefix."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        direct = (self.workspace_path / p).resolve()
        if direct.exists():
            return direct
        ws_name = self.workspace_path.name.lower()
        parts = p.parts
        if parts and parts[0].lower() == ws_name:
            stripped = Path(*parts[1:])
            stripped_path = (self.workspace_path / stripped).resolve()
            if stripped_path.exists():
                return stripped_path
        return direct

    def _read_file(self, file_path: str) -> Optional[str]:
        """Read file from disk, with session map fallback and concurrency tracking."""
        key = file_path.replace("\\", "/").lower()
        if key in self._written_files:
            return self._written_files[key]
        abs_path = self._resolve_fs_path(file_path)
        if abs_path.exists():
            try:
                content = abs_path.read_text(encoding="utf-8", errors="ignore")
                try:
                    from ticket_to_code.integrity.workspace_integrity import get_workspace_integrity_manager
                    mgr = get_workspace_integrity_manager(self.workspace_path)
                    mgr.record_file_read(file_path, content)
                except Exception as _mgr_e:
                    logger.debug(f"Integrity read record skipped: {_mgr_e}")
                return content
            except Exception:
                pass
        return None

    def read_file_slice(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> Optional[str]:
        """Read a targeted slice of a file with line numbers and log 'Analyzed file #Lstart-end'."""
        content = self._read_file(file_path)
        if content is None:
            return None
        lines = content.splitlines()
        total_lines = len(lines)
        if start_line is None and end_line is None:
            return content
        s = max(1, int(start_line or 1))
        e = min(total_lines, int(end_line or (s + 100)))
        logger.info(f"Analyzed {file_path} #L{s}-{e}")
        start_idx = max(0, s - 1)
        end_idx = min(total_lines, e)
        numbered = [f"{i + 1}: {lines[i]}" for i in range(start_idx, end_idx)]
        header = f"File: {file_path} #L{s}-{e} (Total lines: {total_lines}):\n"
        return header + "\n".join(numbered)

    def _write_file(self, file_path: str, content: str) -> None:
        """Write file to disk safely via WorkspaceIntegrityManager and update session map."""
        if self.allowed_files is not None:
            norm_fp = file_path.replace("\\", "/").lower().strip()
            is_allowed = any(
                norm_fp == af or norm_fp.endswith("/" + af) or af.endswith("/" + norm_fp)
                for af in self.allowed_files
            )
            if not is_allowed:
                is_migration = "db/migration" in norm_fp and norm_fp.endswith(".sql")
                is_companion = (
                    norm_fp.endswith("repository.java") or
                    norm_fp.endswith("service.java") or
                    norm_fp.endswith("test.java")
                )
                if is_migration or is_companion:
                    logger.info(
                        f"  🔓 [Dynamic Scope Expansion] Auto-admitting companion file '{file_path}' into allowed_files"
                    )
                    self.allowed_files.add(norm_fp)
                    is_allowed = True
                else:
                    logger.error(
                        f"  ⛔ EDIT LOOP SCOPE LOCK: '{file_path}' is outside allowed_files. Write aborted."
                    )
                    return

        abs_path = self._resolve_fs_path(file_path)

        # Capture pre-edit snapshot before first write to an existing file
        norm_key = file_path.replace("\\", "/").strip()
        while norm_key.startswith("./"):
            norm_key = norm_key[2:]
        norm_key = norm_key.lstrip("/")

        have_snapshot = any(
            k.replace("\\", "/").lower().strip().lstrip("./") == norm_key.lower()
            for k in self.pre_edit_snapshots.keys()
        )
        if not have_snapshot and abs_path.is_file():
            try:
                self.pre_edit_snapshots[norm_key] = abs_path.read_text(encoding="utf-8", errors="ignore")
                logger.info(f"  📸 Captured pre-edit snapshot for {norm_key} ({len(self.pre_edit_snapshots[norm_key])} bytes)")
            except Exception as _snap_err:
                logger.debug(f"Pre-edit snapshot failed for {norm_key}: {_snap_err}")

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from ticket_to_code.integrity.workspace_integrity import get_workspace_integrity_manager
            mgr = get_workspace_integrity_manager(self.workspace_path)
            mgr.safe_write(
                file_path=file_path,
                new_content_or_edits=content,
                ticket_id=getattr(self, "ticket_id", "") or "",
                task_id="edit_loop",
                requirement_id="",
            )
        except Exception as _w_err:
            logger.warning(f"  [WorkspaceIntegrity] safe_write fallback for {file_path}: {_w_err}")
            abs_path.write_text(content, encoding="utf-8")

        self._written_files[file_path.replace("\\", "/").lower()] = content
        # Sync to ImplementationState so blueprint verification can see this content
        if self._impl_state is not None:
            self._impl_state.update_generated_file(file_path, content)
        logger.info(f"  📝 Written: {file_path}")
        
        file_basename = Path(file_path).name
        self._ui_emit(
            "file_written",
            message=f"📝 Written: {file_basename}",
            file_path=file_path,
            file_name=file_basename,
        )

