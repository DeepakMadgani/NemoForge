"""
Autonomous ReAct Agent Engine for Ticket-to-Code.

Implements the Reference Architecture for Autonomous Ticket Resolution:
1. Reference Tool Interfaces (Typed, Bounded, Canonical Boundaries):
   - grep_search: Bounded ripgrep with line-number snippets
   - view_file: Line-bounded 1-indexed file reader emitting cryptographic contentHash
   - list_dir: Directory child scanner
   - replace_file_content: Hash-checked, occurrence-verified surgical editing
   - run_command: Bounded shell execution for compiler/test verification
2. Explicit Controller State Machine:
   UNDERSTAND -> EXPLORE -> PLAN -> EDIT -> VERIFY -> REPORT
3. Operational Safety Policies:
   - Action signature tracking (prevents duplicate tool calls and empty query loops)
   - Hash-checked writes (rejects stale edits if contentHash changed)
   - Expected occurrence verification (rejects ambiguous multi-matches)
   - Read-Before-Write enforcement (cannot patch a file without prior view_file)
   - Strict non-zero test assertion gate (rejects 0 tests executed)
   - Scoped atomic rollback via SurgicalDiffEngine (never resets entire worktree)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ticket_to_code.models import ValueEdgeTicket
from ticket_to_code.editing.surgical_diff_engine import SurgicalDiffEngine, EditStatus, EditResult
from ticket_to_code.runtime.diagnostic_error_slicer import DiagnosticErrorSlicer
from ticket_to_code.runtime.polyglot_build import BuildConfig, PolyglotBuildDiscoverer
from ticket_to_code.runtime.security import CommandSafetyValidator
from ticket_to_code.runtime.limits import OBSERVATION_MAX_CHARS, LATEST_READ_MAX_CHARS, truncate
from ticket_to_code.llm_utils import llm_invoke

logger = logging.getLogger(__name__)

# Tools whose repeated invocation is legitimate once the workspace has changed
# (their duplicate signature is salted with the write epoch).
_READ_ONLY_TOOLS = {"view_file", "grep_search", "list_dir", "read", "search"}
# Tools that mutate the workspace (or may): every dispatch bumps the write epoch.
_WRITE_TOOLS = {"replace_file_content", "run_command"}


# ─── 1. Controller State Machine & Data Models ───────────────────────────────

class ReActState(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    EXPLORE = "EXPLORE"
    PLAN = "PLAN"
    EDIT = "EDIT"
    VERIFY = "VERIFY"
    REPORT = "REPORT"
    FAILED = "FAILED"


@dataclass
class ToolError:
    code: str
    message: str
    retryable: bool = True


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[ToolError] = None
    truncated: bool = False


@dataclass
class ReActAgentResult:
    success: bool
    status: str
    duration_seconds: float
    total_steps: int
    modified_files: List[str]
    diffs: Dict[str, str] = field(default_factory=dict)
    build_passed: bool = False
    test_passed: bool = False
    message: str = ""
    error_message: str = ""


# ─── 2. Reference Typed Exploration & Mutation Toolbox ───────────────────────

class ReActToolbox:
    """
    Typed, boundary-checked tools for autonomous repository exploration and mutation.
    Enforces canonical absolute workspace boundaries and bounded output buffers.
    """

    def __init__(self, workspace_path: str, diff_engine: SurgicalDiffEngine):
        self.workspace_path = Path(workspace_path).resolve()
        self.diff_engine = diff_engine
        # Maps rel_path -> last read contentHash
        self.read_hashes: Dict[str, str] = {}

    def canonicalize_path(self, rel_or_abs: str) -> Optional[Path]:
        """Resolves path and asserts it resides strictly within workspace boundaries and containment policy."""
        p = Path(rel_or_abs)
        full = (self.workspace_path / p).resolve() if not p.is_absolute() else p.resolve()
        ok, _ = CommandSafetyValidator.validate_path_containment(full, self.workspace_path)
        if not ok:
            return None
        return full

    @staticmethod
    def compute_hash(text: str) -> str:
        """Computes SHA-256 hash normalized for line endings."""
        normalized = text.replace("\r\n", "\n")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def grep_search(
        self,
        pattern: str,
        search_path: str = "",
        file_glob: str = "",
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> ToolResult:
        """Bounded lexical pattern matching across directory or file globs."""
        target_dir = self.workspace_path
        if search_path:
            resolved = self.canonicalize_path(search_path)
            if not resolved or not resolved.exists():
                return ToolResult(
                    ok=False,
                    error=ToolError(code="PATH_NOT_FOUND", message=f"Search path does not exist: {search_path}", retryable=False),
                )
            target_dir = resolved

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error:
            compiled = re.compile(re.escape(pattern), flags)

        matches: List[Dict[str, Any]] = []
        truncated = False

        for root, _, files in os.walk(target_dir):
            norm_root = root.replace("\\", "/")
            if any(skip in norm_root for skip in ["/.git", "/node_modules", "/build", "/target", "/.gradle", "/dist"]):
                continue

            for f in files:
                if file_glob and not Path(f).match(file_glob):
                    continue

                full_p = Path(root) / f
                try:
                    lines = full_p.read_text(encoding="utf-8", errors="ignore").splitlines()
                    for idx, line in enumerate(lines, start=1):
                        if compiled.search(line):
                            rel = str(full_p.relative_to(self.workspace_path)).replace("\\", "/")
                            matches.append({
                                "path": rel,
                                "line": idx,
                                "text": line.strip()[:200],
                            })
                            if len(matches) >= max_results:
                                truncated = True
                                break
                except Exception:
                    continue

                if truncated:
                    break
            if truncated:
                break

        return ToolResult(ok=True, data={"matches": matches, "truncated": truncated})

    def view_file(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: int = 800,
    ) -> ToolResult:
        """Bounded line slicing reader with 1-indexed markers and contentHash."""
        resolved = self.canonicalize_path(file_path)
        if not resolved or not resolved.is_file():
            return ToolResult(
                ok=False,
                error=ToolError(code="FILE_NOT_FOUND", message=f"File not found: {file_path}", retryable=False),
            )

        try:
            raw_text = resolved.read_text(encoding="utf-8", errors="ignore")
            content_hash = self.compute_hash(raw_text)

            all_lines = raw_text.splitlines()
            total_lines = len(all_lines)
            s_idx = max(1, start_line)
            e_idx = min(total_lines, end_line if end_line > 0 else s_idx + 800)

            sliced = all_lines[s_idx - 1:e_idx]
            annotated = [f"{s_idx + i}: {line}" for i, line in enumerate(sliced)]

            rel = str(resolved.relative_to(self.workspace_path)).replace("\\", "/")
            self.read_hashes[rel] = content_hash

            return ToolResult(
                ok=True,
                data={
                    "path": rel,
                    "content": "\n".join(annotated),
                    "contentHash": content_hash,
                    "startLine": s_idx,
                    "endLine": e_idx,
                    "totalLines": total_lines,
                    "truncated": (e_idx < total_lines),
                },
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                error=ToolError(code="READ_ERROR", message=f"Failed to read file {file_path}: {exc}"),
            )

    def list_dir(self, dir_path: str = "", max_entries: int = 60) -> ToolResult:
        """Inspect directory children."""
        resolved = self.canonicalize_path(dir_path) if dir_path else self.workspace_path
        if not resolved or not resolved.is_dir():
            return ToolResult(
                ok=False,
                error=ToolError(code="DIR_NOT_FOUND", message=f"Directory not found: {dir_path}", retryable=False),
            )

        entries = []
        truncated = False
        try:
            for item in sorted(resolved.iterdir()):
                norm = str(item).replace("\\", "/")
                if any(skip in norm for skip in ["/.git", "/node_modules", "/build", "/target"]):
                    continue
                entries.append({
                    "path": item.name,
                    "kind": "directory" if item.is_dir() else "file",
                })
                if len(entries) >= max_entries:
                    truncated = True
                    break
            return ToolResult(ok=True, data={"entries": entries, "truncated": truncated})
        except Exception as exc:
            return ToolResult(ok=False, error=ToolError(code="LIST_ERROR", message=str(exc)))

    def replace_file_content(
        self,
        file_path: str,
        target_content: str,
        replacement_content: str,
        expected_content_hash: str = "",
        expected_occurrences: int = 1,
    ) -> ToolResult:
        """
        Surgically patches code using SurgicalDiffEngine.
        Enforces Prior-Read Invariant, Hash Check, and Exact Occurrences.
        """
        resolved = self.canonicalize_path(file_path)
        if not resolved:
            return ToolResult(
                ok=False,
                error=ToolError(code="OUT_OF_SCOPE", message=f"Out of scope path: {file_path}", retryable=False),
            )

        rel = str(resolved.relative_to(self.workspace_path)).replace("\\", "/")

        # 1. New file creation branch
        if not target_content and not resolved.is_file():
            res = self.diff_engine.apply_anchored_replacement(
                rel_path=rel,
                old_block="",
                new_block=replacement_content,
            )
            if not res.success:
                return ToolResult(ok=False, error=ToolError(code="CREATE_FAILED", message=res.message))
            new_hash = self.compute_hash(replacement_content)
            self.read_hashes[rel] = new_hash
            return ToolResult(
                ok=True,
                data={"path": rel, "replacements": 1, "newContentHash": new_hash, "diff": res.unified_diff},
            )

        # 2. Prior-Read Invariant Check
        if rel not in self.read_hashes:
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="UNINSPECTED_FILE",
                    message=f"Safety Rejection: You must inspect '{rel}' using view_file before editing it.",
                    retryable=True,
                ),
            )

        # 3. Cryptographic Hash Validation
        try:
            current_raw = resolved.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(ok=False, error=ToolError(code="READ_ERROR", message=str(exc)))

        current_hash = self.compute_hash(current_raw)
        if expected_content_hash and expected_content_hash != current_hash:
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="STALE_CONTENT_HASH",
                    message=f"Stale Edit: File '{rel}' hash changed ({current_hash} != expected {expected_content_hash}). Re-read via view_file.",
                    retryable=True,
                ),
            )

        # 4. Expected Occurrences Verification
        normalized_content = current_raw.replace("\r\n", "\n")
        normalized_target = target_content.replace("\r\n", "\n")
        actual_occurrences = normalized_content.count(normalized_target)

        if actual_occurrences != expected_occurrences:
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="OCCURRENCE_MISMATCH",
                    message=f"Occurrence Mismatch in '{rel}': target matches {actual_occurrences} times (expected {expected_occurrences}). Provide unique context lines.",
                    retryable=True,
                ),
            )

        # 5. Apply surgical diff
        res: EditResult = self.diff_engine.apply_anchored_replacement(
            rel_path=rel,
            old_block=target_content,
            new_block=replacement_content,
            expected_match_count=expected_occurrences,
        )

        if not res.success:
            return ToolResult(ok=False, error=ToolError(code="PATCH_FAILED", message=res.message))

        new_raw = resolved.read_text(encoding="utf-8")
        new_hash = self.compute_hash(new_raw)
        self.read_hashes[rel] = new_hash

        return ToolResult(
            ok=True,
            data={
                "path": rel,
                "replacements": 1,
                "newContentHash": new_hash,
                "diff": res.unified_diff,
            },
        )

    def run_command(self, command: str, cwd: str = "", timeout: int = 60) -> ToolResult:
        """Executes shell command strictly within workspace with containment validation."""
        work_dir = self.canonicalize_path(cwd) if cwd else self.workspace_path
        if not work_dir or not work_dir.is_dir():
            work_dir = self.workspace_path

        safe, reason = CommandSafetyValidator.validate_command(command, self.workspace_path)
        if not safe:
            return ToolResult(
                ok=False,
                error=ToolError(code="SECURITY_VIOLATION", message=f"Command rejected by security policy: {reason}", retryable=False),
                data={"command": command, "violation": reason},
            )

        try:
            # Validated, env-scrubbed, process-tree-killed on timeout, no shell on POSIX.
            from ticket_to_code.runtime.security import run_checked_subprocess
            from ticket_to_code.runtime.limits import TOOL_OUTPUT_MAX_CHARS, truncate
            returncode, stdout, stderr = run_checked_subprocess(
                command, cwd=str(work_dir), timeout=timeout, workspace=str(self.workspace_path)
            )
            return ToolResult(
                ok=True,
                data={
                    "command": command,
                    "exit_code": returncode,
                    "stdout": truncate(stdout or "", TOOL_OUTPUT_MAX_CHARS),
                    "stderr": truncate(stderr or "", TOOL_OUTPUT_MAX_CHARS),
                },
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                error=ToolError(code="EXEC_ERROR", message=f"Command execution error: {exc}"),
            )


# ─── 3. Autonomous ReAct Controller ──────────────────────────────────────────

class AutonomousReActEngine:
    """
    Autonomous ReAct execution controller modeled after Copilot Workspace and Antigravity.
    Drives iterative Reason + Act loop through explicit phases:
    UNDERSTAND -> EXPLORE -> PLAN -> EDIT -> VERIFY -> REPORT
    """

    def __init__(
        self,
        workspace_path: str,
        ticket: ValueEdgeTicket,
        llm_client: Optional[Any] = None,
        max_steps: int = 15,
        max_repair_turns: int = 3,
    ):
        self.workspace_path = Path(workspace_path).resolve()
        self.ticket = ticket
        self.llm = llm_client
        if self.llm is None:
            # Fall back to the registry's configured model; discard test mocks so
            # a missing client still surfaces as `no_llm_client`.
            try:
                from ticket_to_code.runtime.llm_factory import LLMRegistry
                cand = LLMRegistry.get_llm()
                if cand is not None and not type(cand).__module__.startswith("unittest.mock"):
                    self.llm = cand
            except Exception:
                self.llm = None
        self.max_steps = max_steps
        self.max_repair_turns = max_repair_turns

        self.diff_engine = SurgicalDiffEngine(str(self.workspace_path))
        self.toolbox = ReActToolbox(str(self.workspace_path), self.diff_engine)

        self.state = ReActState.UNDERSTAND
        # Incremented after every workspace mutation; read-only tool signatures
        # include it so re-reading a file after an edit is not a "duplicate".
        self.write_epoch: int = 0
        self.seen_signatures: Set[str] = set()
        self.step_history: List[Dict[str, Any]] = []
        self.modified_files: Dict[str, str] = {}
        self.consecutive_empty_searches: int = 0
        self.staged_plan: Optional[str] = None
        self.inspected_files_ledger: Dict[str, Dict[str, Any]] = {}

    def _hash_action(self, tool_name: str, args: Dict[str, Any]) -> str:
        payload: Dict[str, Any] = {"t": tool_name, "a": args}
        if tool_name in _READ_ONLY_TOOLS:
            payload["e"] = self.write_epoch
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _execute_tool_call(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        sig = self._hash_action(tool_name, args)
        if sig in self.seen_signatures:
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="DUPLICATE_ACTION",
                    message="Anti-Spinning Guard: You have already executed this exact tool call. Pick a different action or proceed to planning.",
                    retryable=True,
                ),
            )
        self.seen_signatures.add(sig)

        if tool_name == "grep_search":
            res = self.toolbox.grep_search(
                pattern=args.get("pattern", ""),
                search_path=args.get("search_path", ""),
                file_glob=args.get("file_glob", ""),
                case_sensitive=args.get("case_sensitive", False),
                max_results=args.get("max_results", 50),
            )
            matches = res.data.get("matches", []) if res.ok and isinstance(res.data, dict) else []
            if not matches:
                self.consecutive_empty_searches += 1
            else:
                self.consecutive_empty_searches = 0
            return res

        elif tool_name == "view_file":
            res = self.toolbox.view_file(
                file_path=args.get("file_path", ""),
                start_line=args.get("start_line", 1),
                end_line=args.get("end_line", 800),
            )
            if res.ok and isinstance(res.data, dict):
                self.inspected_files_ledger[res.data["path"]] = {
                    "content_hash": res.data["contentHash"],
                    "lines": f"{res.data['startLine']}-{res.data['endLine']}",
                    "total_lines": res.data["totalLines"],
                }
            return res

        elif tool_name == "list_dir":
            return self.toolbox.list_dir(
                dir_path=args.get("dir_path", ""),
                max_entries=args.get("max_entries", 60),
            )

        elif tool_name == "replace_file_content":
            fp = args.get("file_path", "")
            res = self.toolbox.replace_file_content(
                file_path=fp,
                target_content=args.get("target_content", ""),
                replacement_content=args.get("replacement_content", ""),
                expected_content_hash=args.get("expected_content_hash", ""),
                expected_occurrences=args.get("expected_occurrences", 1),
            )
            if res.ok and isinstance(res.data, dict):
                rel = res.data.get("path", fp)
                self.modified_files[rel] = res.data.get("diff", "")
                if rel in self.inspected_files_ledger and "newContentHash" in res.data:
                    self.inspected_files_ledger[rel]["content_hash"] = res.data["newContentHash"]
                self.write_epoch += 1
            return res

        elif tool_name == "run_command":
            # Any command may mutate the workspace (build, codegen, formatter):
            # invalidate read-only duplicate signatures regardless of outcome.
            self.write_epoch += 1
            return self.toolbox.run_command(
                command=args.get("command", ""),
                cwd=args.get("cwd", ""),
                timeout=args.get("timeout", 60),
            )

        return ToolResult(
            ok=False,
            error=ToolError(code="UNKNOWN_TOOL", message=f"Unknown tool: {tool_name}", retryable=False),
        )

    def verify_workspace(self, primary_file: str) -> Tuple[bool, bool, str]:
        """
        Runs compilation and strict non-zero test verification gates.
        Uses focused test slicing to execute companion tests fast and prevent monorepo pollution.
        Returns (build_passed, test_passed, error_diagnostic).
        """
        sample_path = self.workspace_path / primary_file
        cfg = PolyglotBuildDiscoverer.discover(sample_path.parent, self.workspace_path)
        if not cfg:
            return True, True, ""

        # 1. Compiler Gate
        if cfg.compile_cmd:
            comp_res = self.toolbox.run_command(cfg.compile_cmd, cwd=str(cfg.work_dir), timeout=60)
            if comp_res.ok and isinstance(comp_res.data, dict):
                exit_code = comp_res.data.get("exit_code", 1)
                if exit_code != 0:
                    err_text = comp_res.data.get("stdout", "") + "\n" + comp_res.data.get("stderr", "")
                    sliced = DiagnosticErrorSlicer.slice_errors(err_text, max_tokens_budget=300)
                    return False, False, f"Compiler Verification Failed:\n{sliced}"

        # 2. Strict Non-Zero Test Gate (Focused Companion Test Priority)
        focused_info = PolyglotBuildDiscoverer.get_focused_test_command(sample_path, self.workspace_path)
        test_cmd, test_dir = focused_info if focused_info else (cfg.test_cmd, cfg.work_dir)

        if test_cmd:
            test_res = self.toolbox.run_command(test_cmd, cwd=str(test_dir), timeout=90)
            if test_res.ok and isinstance(test_res.data, dict):
                exit_code = test_res.data.get("exit_code", 1)
                combined = test_res.data.get("stdout", "") + "\n" + test_res.data.get("stderr", "")
                lower = combined.lower()

                # Assert non-zero test execution
                if "0 tests completed" in lower or "collected 0 items" in lower or "no tests found" in lower:
                    return True, False, "Strict Test Gate Rejection: 0 unit tests were executed. Test suite did not target the modified service."

                if exit_code != 0:
                    sliced = DiagnosticErrorSlicer.slice_errors(combined, max_tokens_budget=300)
                    return True, False, f"Test Suite Failure:\n{sliced}"

        return True, True, ""

    def run(self) -> ReActAgentResult:
        """Executes the full ReAct controller loop."""
        start_time = time.time()
        logger.info(f"🚀 [AutonomousReActEngine] Initiating ReAct execution for: {self.ticket.ticket_id}")

        self.state = ReActState.EXPLORE
        step = 0

        while step < self.max_steps:
            step += 1

            if not self.llm:
                return ReActAgentResult(
                    success=False,
                    status="no_llm_client",
                    duration_seconds=time.time() - start_time,
                    total_steps=step,
                    modified_files=[],
                    error_message="No LLM client configured for AutonomousReActEngine",
                )

            prompt = self._build_step_prompt()

            try:
                if hasattr(self.llm, "invoke"):
                    resp = llm_invoke(self.llm, prompt)
                    raw_text = getattr(resp, "content", str(resp))
                elif callable(self.llm):
                    raw_text = str(self.llm(prompt))
                else:
                    raw_text = str(self.llm)
            except Exception as exc:
                return ReActAgentResult(
                    success=False,
                    status="llm_error",
                    duration_seconds=time.time() - start_time,
                    total_steps=step,
                    modified_files=[],
                    error_message=f"LLM call failed: {exc}",
                )

            tool_name, tool_args, finish_intent, plan_intent = self._parse_agent_action(raw_text)

            if plan_intent:
                self.staged_plan = plan_intent
                self.state = ReActState.EDIT
                self.step_history.append({
                    "role": "plan",
                    "content": f"PLAN ACCEPTED: {self.staged_plan}\nProceed with hash-checked edits.",
                })
                continue

            if finish_intent:
                self.state = ReActState.VERIFY
                if not self.modified_files:
                    return ReActAgentResult(
                        success=False,
                        status="no_modifications",
                        duration_seconds=time.time() - start_time,
                        total_steps=step,
                        modified_files=[],
                        error_message="Agent terminated without modifying any files.",
                    )

                primary = list(self.modified_files.keys())[0]
                build_ok, test_ok, diag = self.verify_workspace(primary)

                if build_ok and test_ok:
                    self.state = ReActState.REPORT
                    return ReActAgentResult(
                        success=True,
                        status="verified_success",
                        duration_seconds=time.time() - start_time,
                        total_steps=step,
                        modified_files=list(self.modified_files.keys()),
                        diffs=self.modified_files,
                        build_passed=True,
                        test_passed=True,
                        message=f"Ticket {self.ticket.ticket_id} resolved and verified via ReAct loop.",
                    )
                else:
                    self.step_history.append({
                        "role": "environment",
                        "content": f"VERIFICATION FAILED: {diag}\nYou must repair this error.",
                    })
                    continue

            if tool_name:
                res = self._execute_tool_call(tool_name, tool_args)
                err_dict = {"code": res.error.code, "message": res.error.message} if res.error else None
                self.step_history.append({
                    "role": "tool_call",
                    "name": tool_name,
                    "args": tool_args,
                    "result": res.data if res.ok else {"error": err_dict},
                })
            else:
                self.step_history.append({
                    "role": "assistant",
                    "content": raw_text[:500],
                })

        # Max steps exceeded: Scoped atomic rollback
        logger.warning(f"⚠️ [AutonomousReActEngine] Max steps ({self.max_steps}) exceeded. Rolling back.")
        self.diff_engine.rollback_all()
        return ReActAgentResult(
            success=False,
            status="max_steps_exceeded",
            duration_seconds=time.time() - start_time,
            total_steps=step,
            modified_files=[],
            error_message="ReAct loop exceeded step budget without passing verification. Changes rolled back cleanly.",
        )

    def _build_step_prompt(self) -> str:
        history_summary = []
        window = self.step_history[-6:]
        # The most recent read/view observation is kept (nearly) verbatim so the
        # model can write an exact-match edit against it; older observations are
        # compacted to the standard observation budget.
        latest_read_idx = -1
        for idx in range(len(window) - 1, -1, -1):
            entry = window[idx]
            if entry.get("role") == "tool_call" and entry.get("name") in _READ_ONLY_TOOLS:
                latest_read_idx = idx
                break
        for idx, s in enumerate(window):
            if s.get("role") == "tool_call":
                limit = LATEST_READ_MAX_CHARS if idx == latest_read_idx else OBSERVATION_MAX_CHARS
                res_str = truncate(json.dumps(s.get("result", ""), default=str), limit)
                history_summary.append(f"Action: {s.get('name')}({s.get('args')})\nObservation: {res_str}")
            elif s.get("role") == "environment":
                history_summary.append(s.get("content", ""))
            elif s.get("role") == "plan":
                history_summary.append(s.get("content", ""))

        history_block = "\n---\n".join(history_summary) if history_summary else "No prior actions yet."

        ledger_lines = []
        for fp, meta in self.inspected_files_ledger.items():
            ledger_lines.append(f"- {fp}: [lines {meta['lines']} of {meta['total_lines']}, contentHash: {meta['content_hash']}]")
        ledger_block = "\n".join(ledger_lines) if ledger_lines else "None yet (explore with view_file)."

        return f"""You are an autonomous ReAct software engineer resolving a defect.
Current Controller State: {self.state.value}

### TICKET:
ID: {self.ticket.ticket_id}
Title: {self.ticket.title}
Description: {self.ticket.description}

### ACTIVE WORKING SET & CONTENT HASH LEDGER:
{ledger_block}

### PREVIOUS STEPS & OBSERVATIONS:
{history_block}

### REFERENCE TOOLS:
1. grep_search(pattern, search_path, file_glob) -> find definitions & call sites
2. view_file(file_path, start_line, end_line) -> returns content AND contentHash (MUST view before editing)
3. list_dir(dir_path) -> view directory children
4. SUBMIT_PLAN(plan_summary, target_files) -> locks working set before edits
5. replace_file_content(file_path, target_content, replacement_content, expected_content_hash, expected_occurrences) -> hash-checked surgical patch
6. COMPLETE_TASK() -> trigger compiler and test verification gate

### FORMAT:
Output your step strictly in JSON format:
{{
  "thought": "Your reasoning about what to explore, plan, or patch",
  "tool": "tool_name",
  "args": {{ ... }}
}}
"""

    def _parse_agent_action(self, text: str) -> Tuple[Optional[str], Dict[str, Any], bool, Optional[str]]:
        """Extracts JSON tool call, SUBMIT_PLAN, or COMPLETE_TASK command."""
        text_clean = text.strip()
        match = re.search(r"\{.*\}", text_clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                tool = data.get("tool", "")
                args = data.get("args", {})
                if tool == "COMPLETE_TASK":
                    return None, {}, True, None
                if tool == "SUBMIT_PLAN":
                    return None, {}, False, args.get("plan_summary", "Plan submitted")
                if tool:
                    return tool, args, False, None
            except Exception:
                pass
        return None, {}, False, None
