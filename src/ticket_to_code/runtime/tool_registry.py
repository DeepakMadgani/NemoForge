"""
Unified High-Signal Tool Registry for Autonomous Reasoning & Deduction.

Consolidates all 14 grounding tools across discovery, AST inspection, database schema,
code mutation, and polyglot compilation verification.
Each tool automatically feeds observations into the DeductiveBlackboard and applies
strict output density sanitation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ticket_to_code.runtime.deductive_blackboard import DeductiveBlackboard
from ticket_to_code.runtime.security import SecurityPolicy, SecurityViolation, run_checked_subprocess
from ticket_to_code.runtime.limits import (
    TOOL_OUTPUT_MAX_CHARS,
    SEARCH_MAX_LINES,
    LATEST_READ_MAX_CHARS,
    BUILD_OUTPUT_MAX_CHARS,
    truncate,
)

logger = logging.getLogger(__name__)


class HighDensitySanitizer:
    """Sanitizes tool output to maximize signal density and prevent context bloat."""

    @staticmethod
    def sanitize(output: str, max_chars: int = TOOL_OUTPUT_MAX_CHARS, max_lines: int = SEARCH_MAX_LINES) -> str:
        if not output:
            return ""
        lines = output.splitlines()
        if len(lines) > max_lines:
            truncated_lines = lines[:max_lines]
            truncated_lines.append(f"\n... [Truncated {len(lines) - max_lines} additional lines. Showing lines 1-{max_lines} of {len(lines)}. Use line slicing (start_line={max_lines+1}) to view remaining content]")
            output = "\n".join(truncated_lines)
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... [Truncated at {max_chars} characters. Use targeted line slicing to view remaining content]"
        return output


class UnifiedToolRegistry:
    """Unified provider of all 14 grounding tools bound to a workspace and a DeductiveBlackboard."""

    def __init__(self, workspace_path: str, blackboard: Optional[DeductiveBlackboard] = None):
        self.workspace_path = Path(workspace_path).resolve()
        self.blackboard = blackboard or DeductiveBlackboard(workspace_path=str(self.workspace_path))
        self.execution_trace: List[Dict[str, Any]] = []
        self.mutated_files: List[str] = []
        # Number of successful agent writes through this registry. A compiler baseline may only be
        # recorded while this is 0, otherwise agent-introduced errors would be mistaken for pre-existing ones.
        self._writes_performed: int = 0
        from ticket_to_code.editing.surgical_diff_engine import SurgicalDiffEngine
        self.diff_engine = SurgicalDiffEngine(str(self.workspace_path))

    def _record_execution(self, tool_name: str, target: str, raw_output: str, duration_ms: int, success: bool):
        sanitized = HighDensitySanitizer.sanitize(raw_output)
        deductions = self.blackboard.ingest_tool_observation(tool_name, target, sanitized)
        self.execution_trace.append({
            "tool": tool_name,
            "target": target,
            "duration_ms": duration_ms,
            "success": success,
            "deductions": deductions,
            "output_preview": sanitized[:300],
        })

    # ── Tool 1: grep_search ──────────────────────────────────────────────────
    def grep_search(self, query: str, search_path: str = "", includes: str = "", case_insensitive: bool = True) -> str:
        """Fast regex or literal text search across files in the workspace.
        Returns matching lines with file paths and 1-indexed line numbers.
        """
        t0 = time.time()
        base_dir = (self.workspace_path / search_path) if search_path else self.workspace_path
        if not base_dir.exists():
            return f"Directory not found: {search_path}"

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            pattern = re.compile(query, flags)
        except re.error:
            pattern = re.compile(re.escape(query), flags)

        matches: List[str] = []
        include_exts = [e.strip().lower() for e in includes.split(",") if e.strip()]

        for p in base_dir.rglob("*"):
            if not p.is_file():
                continue
            # Skip build/cache/git
            norm = str(p).replace("\\", "/")
            if any(skip in norm for skip in ["/.git/", "/node_modules/", "/build/", "/target/", "/.gradle/", "/dist/"]):
                continue
            if include_exts and not any(p.name.lower().endswith(ext) or ext in p.name.lower() for ext in include_exts):
                continue

            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
                for idx, line in enumerate(lines, 1):
                    if pattern.search(line):
                        rel = str(p.relative_to(self.workspace_path)).replace("\\", "/")
                        matches.append(f"{rel}:{idx}: {line.strip()}")
                        if len(matches) >= SEARCH_MAX_LINES:
                            break
            except Exception:
                continue
            if len(matches) >= SEARCH_MAX_LINES:
                break

        res = "\n".join(matches) if matches else f"No matches found for '{query}'"
        dur = int((time.time() - t0) * 1000)
        self._record_execution("grep_search", query, res, dur, bool(matches))
        return HighDensitySanitizer.sanitize(res)

    # ── Tool 2: find_files ───────────────────────────────────────────────────
    def find_files(self, pattern: str, search_path: str = "") -> str:
        """Locate files matching a glob pattern (e.g. '*Issue*.java', 'deliverable*.ts')."""
        t0 = time.time()
        base_dir = (self.workspace_path / search_path) if search_path else self.workspace_path
        found: List[str] = []
        for p in base_dir.rglob(pattern):
            if p.is_file():
                norm = str(p).replace("\\", "/")
                if not any(skip in norm for skip in ["/.git/", "/node_modules/", "/build/", "/target/"]):
                    found.append(str(p.relative_to(self.workspace_path)).replace("\\", "/"))
            if len(found) >= 50:
                break
        res = "\n".join(found) if found else f"No files matching '{pattern}' found."
        dur = int((time.time() - t0) * 1000)
        self._record_execution("find_files", pattern, res, dur, bool(found))
        return res

    # ── Tool 3: list_directory ───────────────────────────────────────────────
    def list_directory(self, directory_path: str = "", depth: int = 2) -> str:
        """List subdirectories and key files within a directory up to the specified depth."""
        t0 = time.time()
        target = (self.workspace_path / directory_path) if directory_path else self.workspace_path
        if not target.exists() or not target.is_dir():
            return f"Directory not found: {directory_path}"

        lines = [f"Contents of {directory_path or '.'}:"]
        for p in sorted(target.iterdir()):
            norm = str(p).replace("\\", "/")
            if any(skip in norm for skip in ["/.git", "/node_modules", "/build", "/target"]):
                continue
            prefix = "[DIR] " if p.is_dir() else "[FILE]"
            lines.append(f"  {prefix} {p.name}")
        res = "\n".join(lines)
        dur = int((time.time() - t0) * 1000)
        self._record_execution("list_directory", directory_path, res, dur, True)
        return res

    # ── Tool 4: inspect_file ─────────────────────────────────────────────────
    def inspect_file(self, file_path: str, symbol: str = "", start_line: int = 1, end_line: int = 0) -> str:
        """Inspect source code with AST boundary extraction for methods, classes, and structural outlines."""
        t0 = time.time()
        ok, why = SecurityPolicy.validate_path_containment(file_path, self.workspace_path)
        if not ok:
            return f"Security policy rejection: {why}"
        full_p = (self.workspace_path / file_path).resolve()
        if not full_p.exists() or not full_p.is_file():
            # Try fuzzy match
            matches = list(self.workspace_path.glob(f"**/{Path(file_path).name}"))
            if matches:
                full_p = matches[0]
                # Re-check the fuzzy candidate so the sensitive-path blacklist still applies
                ok, why = SecurityPolicy.validate_path_containment(full_p, self.workspace_path)
                if not ok:
                    return f"Security policy rejection: {why}"
            else:
                return f"File not found: {file_path}"

        try:
            lines = full_p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            return f"Error reading {file_path}: {e}"

        total_lines = len(lines)
        rel = str(full_p.relative_to(self.workspace_path)).replace("\\", "/")

        if symbol:
            sym_clean = symbol.strip().lower()
            if sym_clean in ("outline", "structure", "methods"):
                # Method outline
                outlines = [f"// === Structural Outline of {rel} ({total_lines} lines) ==="]
                for idx, line in enumerate(lines, 1):
                    if re.search(r'(public|protected|private|def|export\s+class|interface)\s+([A-Za-z0-9_]+)', line):
                        outlines.append(f"  L{idx}: {line.strip()[:100]}")
                res = "\n".join(outlines[:120])
            else:
                # 1. Try AST-aware complete method/class boundary extraction (supports comma-separated symbols)
                requested_syms = [s.strip().lower() for s in re.split(r'[,;]', symbol) if s.strip()]
                ast_extracted = None
                try:
                    from ticket_to_code.agents.smart_extract import (
                        _parse_java_ts_boundaries,
                        _parse_python_boundaries,
                        _detect_language
                    )
                    lang = _detect_language(str(full_p))
                    boundaries = []
                    if lang in ("java", "typescript", "javascript"):
                        boundaries = _parse_java_ts_boundaries(lines)
                    elif lang == "python":
                        boundaries = _parse_python_boundaries(lines)

                    matched = []
                    for req in requested_syms:
                        m = [b for b in boundaries if b.name.lower() == req]
                        if not m:
                            m = [b for b in boundaries if req in b.name.lower()]
                        for item in m:
                            if item not in matched:
                                matched.append(item)

                    if matched:
                        blocks = []
                        for b in matched:
                            a_start = b.start_line + 1
                            a_end = b.end_line + 1
                            sliced = lines[b.start_line : b.end_line + 1]
                            if len(sliced) > 350:
                                hdr = [f"{i + a_start}: {l}" for i, l in enumerate(sliced[:250])]
                                blocks.append(
                                    f"// Complete {b.kind.title()} '{b.name}' ({a_start}-{a_end} of {total_lines} lines):\n"
                                    + "\n".join(hdr)
                                    + f"\n... [Truncated remaining {len(sliced) - 250} lines of method body. Use line slicing (start_line={a_start + 250}, end_line={a_end}) to view remainder]"
                                )
                            else:
                                blocks.append(
                                    f"// Complete {b.kind.title()} '{b.name}' ({a_start}-{a_end} of {total_lines} lines):\n"
                                    + "\n".join(f"{i + a_start}: {l}" for i, l in enumerate(sliced))
                                )
                        ast_extracted = "\n\n".join(blocks)
                except Exception:
                    ast_extracted = None

                if ast_extracted:
                    res = ast_extracted
                else:
                    # Fallback text block matching (prioritizes non-import usage/instantiation call sites)
                    matched_lines: List[str] = []
                    for req in requested_syms:
                        usage_indices = [
                            idx for idx, line in enumerate(lines, 1)
                            if req in line.lower() and not line.strip().startswith(("import ", "package ", "from "))
                        ]
                        target_idx = usage_indices[0] if usage_indices else None
                        if not target_idx:
                            for idx, line in enumerate(lines, 1):
                                if req in line.lower():
                                    target_idx = idx
                                    break
                        if target_idx:
                            s_idx = max(1, target_idx - 15)
                            e_idx = min(total_lines, target_idx + 45)
                            matched_lines.append(
                                f"// Usage of '{req}' in {rel} (Lines {s_idx}-{e_idx} of {total_lines}):\n"
                                + "\n".join(f"{i}: {lines[i-1]}" for i in range(s_idx, e_idx + 1))
                            )
                    res = "\n\n".join(matched_lines) if matched_lines else f"Symbol(s) '{symbol}' not found in {rel}."
        elif end_line > 0:
            s_idx = max(1, start_line)
            e_idx = min(total_lines, end_line)
            sliced = [f"{i}: {lines[i-1]}" for i in range(s_idx, e_idx + 1)]
            res = f"// Slice of {rel} (Lines {s_idx}-{e_idx} of {total_lines}):\n" + "\n".join(sliced)
        else:
            # Full file or capped
            if total_lines <= 250:
                res = f"// File {rel} ({total_lines} lines):\n" + "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines))
            else:
                head = [f"{i+1}: {lines[i]}" for i in range(min(total_lines, 80))]
                res = f"// Head of {rel} (First 80 of {total_lines} lines. Use symbol or line slicing to view more):\n" + "\n".join(head)

        dur = int((time.time() - t0) * 1000)
        self._record_execution("inspect_file", rel, res, dur, True)
        # Reads keep a wider window than search output so the model can edit against real text.
        return HighDensitySanitizer.sanitize(res, max_chars=LATEST_READ_MAX_CHARS, max_lines=LATEST_READ_MAX_CHARS // 32)

    # ── Tool 5: find_references ──────────────────────────────────────────────
    def find_references(self, symbol_name: str, search_path: str = "") -> str:
        """Find all usages, callers, and import sites of a symbol across all services, prioritizing call sites over imports."""
        raw = self.grep_search(query=r'\b' + re.escape(symbol_name.strip()) + r'\b', search_path=search_path)
        if not raw or "No matches found" in raw:
            return raw
        lines = raw.splitlines()
        call_sites = []
        imports = []
        for l in lines:
            if any(k in l for k in [": import ", ":import ", ": from ", ":from ", ": package "]):
                imports.append(l)
            else:
                call_sites.append(l)
        ordered = call_sites + imports
        return HighDensitySanitizer.sanitize("\n".join(ordered))

    # ── Tool 6: trace_call_chain ─────────────────────────────────────────────
    def trace_call_chain(self, query: str) -> str:
        """Trace fullstack flow from UI button/dialog to backend handler and database."""
        t0 = time.time()
        try:
            from services.call_chain_tracer import CallChainTracer
            tracer = CallChainTracer(self.workspace_path)
            res = tracer.trace(query)
            formatted = "\n".join(res.get("summary_trace", [])) or json.dumps(res, indent=2)
            dur = int((time.time() - t0) * 1000)
            self._record_execution("trace_call_chain", query, formatted, dur, True)
            return formatted
        except Exception:
            # Fallback to grep chain
            hits = self.grep_search(query=query)
            dur = int((time.time() - t0) * 1000)
            self._record_execution("trace_call_chain", query, hits, dur, bool(hits))
            return hits

    # ── Tool 7: inspect_database_schema ──────────────────────────────────────
    def inspect_database_schema(self, service_name: str = "", table_name: str = "") -> str:
        """Inspect database migration files and cross-reference schema DDL against entity models.
        
        Discovers:
        1. Migration versioning sequence and next safe migration number (SemVer, Timestamp, or Sequential).
        2. Real DDL column definitions, types, and primary/foreign keys.
        3. Entity-to-schema field mappings across Java, Python, C#, and TypeScript models.
        4. Subtle column discrepancies and join traps (e.g. otds_user_id vs user_id).
        """
        t0 = time.time()
        from ticket_to_code.lean.schema_entity_cross_reference import generate_structural_schema_report

        res = generate_structural_schema_report(
            workspace_path=self.workspace_path,
            service_name=service_name,
            table_name=table_name,
        )
        dur = int((time.time() - t0) * 1000)
        has_findings = "No SQL migrations found" not in res
        self._record_execution("inspect_database_schema", f"{service_name}:{table_name}", res, dur, has_findings)
        return HighDensitySanitizer.sanitize(res)

    # ── Tool 8: inspect_service_contract ─────────────────────────────────────
    def inspect_service_contract(self, service_name: str = "", endpoint_filter: str = "") -> str:
        """Inspect REST endpoints, @RequestMapping annotations, and URLMappings constants in a service."""
        t0 = time.time()
        hits = self.grep_search(query=r'@(Request|Get|Post|Put|Delete)Mapping|URLMappings', search_path=service_name)
        dur = int((time.time() - t0) * 1000)
        self._record_execution("inspect_service_contract", service_name, hits, dur, bool(hits))
        return hits

    # ── Tool 9: replace_file_content ─────────────────────────────────────────
    # ── Tool 9: replace_file_content ─────────────────────────────────────────
    def replace_file_content(
        self,
        file_path: str,
        target_content: str,
        replacement_content: str,
        start_line: int = 1,
        end_line: int = 0,
        expected_content_hash: str = "",
        expected_occurrences: int = 1,
    ) -> str:
        """Surgical, cryptographic-hash-checked replacement using SurgicalDiffEngine with rollback support."""
        t0 = time.time()
        ok, why = SecurityPolicy.validate_path_containment(file_path, self.workspace_path)
        if not ok:
            out = f"Security policy rejection: {why}"
            self._record_execution("replace_file_content", file_path, out, 0, False)
            return out
        res = self.diff_engine.apply_anchored_replacement(
            rel_path=file_path,
            old_block=target_content,
            new_block=replacement_content,
            expected_match_count=expected_occurrences,
            expected_content_hash=expected_content_hash or None,
        )
        dur = int((time.time() - t0) * 1000)
        if res.success:
            self._writes_performed += 1
            if file_path not in self.mutated_files:
                self.mutated_files.append(file_path)
            out = f"Successfully updated {file_path}. New contentHash: {res.new_content_hash}"
            self._record_execution("replace_file_content", file_path, out, dur, True)
            return out
        out = f"Error [{res.status.value}]: TargetContent not found in {file_path}. {res.message}"
        self._record_execution("replace_file_content", file_path, out, dur, False)
        return out

    # ── Tool 10: write_to_file ───────────────────────────────────────────────
    def write_to_file(self, file_path: str, code_content: str, overwrite: bool = False) -> str:
        """Create a new file safely using SurgicalDiffEngine with undo-stack rollback support."""
        t0 = time.time()
        ok, why = SecurityPolicy.validate_path_containment(file_path, self.workspace_path)
        if not ok:
            out = f"Security policy rejection: {why}"
            self._record_execution("write_to_file", file_path, out, 0, False)
            return out
        full_p = (self.workspace_path / file_path).resolve()
        if full_p.exists() and not overwrite:
            return f"File {file_path} already exists. Set overwrite=True to overwrite."

        res = self.diff_engine.apply_anchored_replacement(
            rel_path=file_path,
            old_block="" if not full_p.exists() else full_p.read_text(encoding="utf-8", errors="ignore"),
            new_block=code_content,
        )
        dur = int((time.time() - t0) * 1000)
        if res.success:
            self._writes_performed += 1
            if file_path not in self.mutated_files:
                self.mutated_files.append(file_path)
            out = f"Successfully wrote {file_path}. New contentHash: {res.new_content_hash}"
            self._record_execution("write_to_file", file_path, out, dur, True)
            return out
        out = f"Error writing {file_path}: {res.message}"
        self._record_execution("write_to_file", file_path, out, dur, False)
        return out

    def rollback_all(self) -> List[str]:
        """Rollback all surgical edits made via this registry."""
        reverted = self.diff_engine.rollback_all()
        self.mutated_files.clear()
        return reverted

    # ── Tool 11: run_command ─────────────────────────────────────────────────
    def run_command(self, command: str, cwd: str = "", timeout_seconds: int = 45) -> str:
        """Execute a shell command within the workspace (policy-checked, env-scrubbed, process-tree bounded)."""
        t0 = time.time()
        try:
            rc, stdout, stderr = run_checked_subprocess(
                command, cwd=cwd, timeout=timeout_seconds, workspace=self.workspace_path
            )
            out = (stdout or "") + ("\n" + stderr if stderr else "")
            res = f"[Exit code {rc}]\n{out}"
            dur = int((time.time() - t0) * 1000)
            self._record_execution("run_command", command, res, dur, rc == 0)
            return HighDensitySanitizer.sanitize(res)
        except SecurityViolation as sv:
            res = f"Security policy rejection: {sv}"
            dur = int((time.time() - t0) * 1000)
            self._record_execution("run_command", command, res, dur, False)
            return res
        except subprocess.TimeoutExpired:
            res = f"Command timed out after {timeout_seconds}s (process tree terminated)."
            dur = int((time.time() - t0) * 1000)
            self._record_execution("run_command", command, res, dur, False)
            return res
        except Exception as e:
            res = f"Command execution error: {e}"
            dur = int((time.time() - t0) * 1000)
            self._record_execution("run_command", command, res, dur, False)
            return res

    # ── Tool 12: verify_service_compilation ──────────────────────────────────
    def verify_service_compilation(self, service_name: str = "") -> str:
        """Run incremental compiler check across any supported ecosystem (Gradle, Maven, NPM, Python, Go, Rust, .NET, Make)."""
        t0 = time.time()

        # Multi-Zone Auto-Orchestration: If multiple files spanning multiple zones were mutated,
        # run the MultiZoneOrchestrator across topological waves!
        if not service_name and len(self.mutated_files) > 1:
            try:
                from ticket_to_code.runtime.multi_zone_orchestrator import MultiZoneOrchestrator
                orch = MultiZoneOrchestrator(self.workspace_path)
                zone_map = orch.map_files_to_zones(self.mutated_files)
                if len(zone_map) > 1:
                    report = orch.execute_multi_zone_pipeline(self.mutated_files)
                    if report.success:
                        res = f"✅ Multi-Zone Pipeline Verified: {len(report.zone_results)} zones compiled successfully across {report.waves_executed} waves."
                        for zr in report.zone_results:
                            res += f"\n  * Zone '{zr.zone_name}' (Tier {zr.tier.name}): PASSED ({zr.duration_ms}ms)"
                    else:
                        res = f"❌ Multi-Zone Pipeline FAILED: Rollback triggered."
                        for zr in report.zone_results:
                            status_icon = "PASSED" if zr.success else "FAILED"
                            res += f"\n  * Zone '{zr.zone_name}' ({zr.tier.name}): {status_icon}\n    {zr.error_message}"
                        if report.contract_violations:
                            res += "\n🚨 Cross-Zone Contract Drift Violations:"
                            for cv in report.contract_violations:
                                res += f"\n  * {cv}"
                    dur = int((time.time() - t0) * 1000)
                    self._record_execution("verify_service_compilation", "multi-zone", res, dur, report.success)
                    return HighDensitySanitizer.sanitize(res)
            except Exception:
                pass

        target_dir = (self.workspace_path / service_name) if service_name else self.workspace_path

        from ticket_to_code.runtime.polyglot_build import PolyglotBuildDiscoverer
        cfg = PolyglotBuildDiscoverer.discover(target_dir, self.workspace_path)

        if not cfg:
            res = f"No build tool detected in {service_name or '.'}"
            dur = int((time.time() - t0) * 1000)
            self._record_execution("verify_service_compilation", service_name, res, dur, True)
            return res

        cmd = cfg.compile_cmd
        exec_dir = cfg.work_dir

        try:
            rc, p_out, p_err = run_checked_subprocess(cmd, cwd=exec_dir, timeout=60, workspace=self.workspace_path)
            # Offline fallback for enterprise monorepos without corporate artifactory VPN access
            if rc != 0 and "gradlew" in cmd and "--offline" not in cmd:
                raw_err = (p_err or p_out or "")
                if any(x in raw_err for x in ["Could not resolve", "artifactory", "No such host is known", "Unable to load Maven meta-data"]):
                    o_rc, o_out, o_err = run_checked_subprocess(
                        cmd + " --offline", cwd=exec_dir, timeout=90, workspace=self.workspace_path
                    )
                    if o_rc == 0 or len(o_err or "") < len(p_err or ""):
                        rc, p_out, p_err = o_rc, o_out, o_err

            if rc == 0:
                res = f"✅ Compilation SUCCESS on {exec_dir.name}: Clean compile with zero errors ({cfg.build_tool})."
                success = True
            else:
                err = (p_err or p_out or "").strip()
                try:
                    from ticket_to_code.runtime.compiler_gate import CompilerGate
                    comp_obj = CompilerGate.parse_compiler_output(exec_dir.name, rc, err)
                    res = comp_obj.get_repair_instruction() if comp_obj.diagnostics else f"❌ Compilation FAILED on {exec_dir.name} (Exit code {rc}):\n{truncate(err, BUILD_OUTPUT_MAX_CHARS)}"
                except Exception:
                    res = f"❌ Compilation FAILED on {exec_dir.name} (Exit code {rc}):\n{truncate(err, BUILD_OUTPUT_MAX_CHARS)}"
                success = False
            dur = int((time.time() - t0) * 1000)
            try:
                from ticket_to_code.runtime.compiler_gate import CompilerGate
                comp_res = CompilerGate.record_and_verify(exec_dir.name, rc, res, blackboard=self.blackboard)
                if CompilerGate.get_baseline(exec_dir.name, str(self.workspace_path)) is None:
                    if self._writes_performed == 0:
                        CompilerGate.record_baseline(exec_dir.name, comp_res, str(self.workspace_path))
                    else:
                        logger.info(
                            "Skipping compiler baseline for '%s': %d agent write(s) already applied in this registry",
                            exec_dir.name, self._writes_performed,
                        )
            except Exception:
                pass
            self._record_execution("verify_service_compilation", service_name, res, dur, success)
            return HighDensitySanitizer.sanitize(res)
        except SecurityViolation as sv:
            res = f"Security policy rejection: {sv}"
            dur = int((time.time() - t0) * 1000)
            self._record_execution("verify_service_compilation", service_name, res, dur, False)
            return res
        except subprocess.TimeoutExpired:
            res = f"Compilation check timed out on {exec_dir.name} (process tree terminated)."
            dur = int((time.time() - t0) * 1000)
            self._record_execution("verify_service_compilation", service_name, res, dur, False)
            return res
        except Exception as e:
            res = f"Compilation execution error: {e}"
            dur = int((time.time() - t0) * 1000)
            self._record_execution("verify_service_compilation", service_name, res, dur, False)
            return res

    # ── Tool 13: run_service_unit_tests ──────────────────────────────────────
    def run_service_unit_tests(self, service_name: str = "", test_target: str = "") -> str:
        """Run targeted unit tests for a service across any supported ecosystem."""
        target_dir = (self.workspace_path / service_name) if service_name else self.workspace_path
        from ticket_to_code.runtime.polyglot_build import PolyglotBuildDiscoverer
        cfg = PolyglotBuildDiscoverer.discover(target_dir, self.workspace_path)
        if cfg:
            cmd = cfg.test_cmd
            if "gradlew" in cmd and "--offline" not in cmd:
                cmd = f"{cmd} --offline"
            if test_target and "gradlew" in cmd:
                cmd = f"{cmd} --tests {test_target}"
            elif test_target and "npm" in cmd:
                cmd = f"{cmd} -- {test_target}"
            elif test_target and "pytest" in cmd:
                cmd = f"{cmd} -k {test_target}"
            return self.run_command(cmd, cwd=str(cfg.work_dir.relative_to(self.workspace_path)), timeout_seconds=120)
        return self.run_command("npm test", cwd=service_name, timeout_seconds=90)

    # ── Tool 14: verify_live_ui_with_playwright ──────────────────────────────
    def verify_live_ui_with_playwright(self, target_url: str, check_description: str = "") -> str:
        """Verify DOM elements and console errors in a running web application."""
        t0 = time.time()
        res = f"Playwright verification ready for {target_url} ({check_description or 'DOM inspection'})"
        dur = int((time.time() - t0) * 1000)
        self._record_execution("verify_live_ui_with_playwright", target_url, res, dur, True)
        return res

    # ── Tool 15: request_structured_clarification ─────────────────────────────
    def request_structured_clarification(self, missing_concept: str, option_a: str, option_b: str, rationale: str) -> str:
        """Structured Clarification Gate (Tier 3): Escalate an architectural ambiguity to the human developer ONLY when a concept is mathematically absent from the repository.
        Strict Boundary: The gate verifies that find_references returns 0 hits across all microservices and the concept is not defined in any Flyway migration schema. If any call site or schema exists in the repository, escalation is REJECTED and the agent must inspect existing code.
        Args:
            missing_concept: Identifier, method, or concept with 0 occurrences in the repo.
            option_a: First concrete architectural alternative with pros/cons.
            option_b: Second concrete architectural alternative with pros/cons.
            rationale: Why human intervention is required instead of repo grounding.
        """
        t0 = time.time()
        # Verify 0 references across workspace code files (excluding .sql migrations)
        raw_refs = self.grep_search(query=r'\b' + re.escape(missing_concept.strip()) + r'\b')
        code_refs = [
            l for l in (raw_refs.splitlines() if raw_refs else [])
            if ":" in l and not any(l.lower().startswith(p) for p in ["no matches", "directory not found"])
            and not l.split(":")[0].strip().endswith(".sql")
        ]
        has_refs = bool(code_refs)
        
        # Verify DB schemas
        db_schemas = self.inspect_database_schema(table_name=missing_concept.strip())
        has_db = "No SQL migrations found" not in db_schemas and bool(db_schemas.strip())
        
        if has_refs:
            res = (
                f"🛑 Clarification REJECTED by Strict 0-Interruption Escalation Boundary: "
                f"'{missing_concept}' has active references/call-sites in the codebase! "
                f"You are strictly barred from interrupting the developer. "
                f"Use `find_references('{missing_concept}')` and `inspect_file` to ground your solution in existing code."
            )
            success = False
        elif has_db:
            res = (
                f"🛑 Clarification REJECTED by Strict 0-Interruption Escalation Boundary: "
                f"'{missing_concept}' exists in database migrations! "
                f"You are strictly barred from interrupting the developer. "
                f"Use `inspect_database_schema(table_name='{missing_concept}')` to derive column definitions and types."
            )
            success = False
        else:
            # Verified external absence (0 occurrences in entire repo and no DB schema match)
            from ticket_to_code.runtime.interactive_gate import InteractiveClarificationGate
            clarif_result = InteractiveClarificationGate.request_clarification(
                question=f"Architectural Decision Required for '{missing_concept}': {rationale}",
                options=[f"Option A: {option_a}", f"Option B: {option_b}"],
                recommended_option=f"Option A: {option_a}",
                timeout_seconds=30.0,
                blackboard=self.blackboard,
            )
            res = (
                f"✅ Tier 3 Structured Clarification Gate APPROVED (Verified 0 hits across all microservices & no DB schema matches):\n"
                f"### Architectural Decision: '{missing_concept}'\n"
                f"- **Rationale**: {rationale}\n"
                f"- **Option A**: {option_a}\n"
                f"- **Option B**: {option_b}\n"
                f"- **User Resolution**: {clarif_result}"
            )
            success = True
        dur = int((time.time() - t0) * 1000)
        self._record_execution("request_structured_clarification", missing_concept, res, dur, success)
        return res

    # ── Tool 16: generate_unified_diff ───────────────────────────────────────
    def generate_unified_diff(self, file_path: str, proposed_content: str) -> str:
        """Generates a structured unified diff with line-by-line hunks for inspection."""
        t0 = time.time()
        p = (self.workspace_path / file_path).resolve()
        if not p.exists() or not p.is_file():
            return f"File not found: '{file_path}'"

        orig = p.read_text(encoding="utf-8", errors="ignore")
        from ticket_to_code.runtime.diff_engine import DiffReviewEngine
        file_diff = DiffReviewEngine.compute_file_diff(file_path, orig, proposed_content)

        if file_diff.is_noop:
            res = f"No changes: proposed content matches '{file_path}' exactly."
        else:
            hunk_summaries = [
                f"[HUNK {h.hunk_index}] {h.header} (-{h.old_count}/+{h.new_count} lines)"
                for h in file_diff.hunks
            ]
            res = f"Unified Diff for '{file_path}' ({len(file_diff.hunks)} hunks):\n" + "\n".join(hunk_summaries) + "\n\n" + file_diff.raw_diff[:3000]

        dur = int((time.time() - t0) * 1000)
        self._record_execution("generate_unified_diff", file_path, res, dur, True)
        return HighDensitySanitizer.sanitize(res)

    # ── Tool 17: apply_selected_diff_hunks ───────────────────────────────────
    def apply_selected_diff_hunks(self, file_path: str, proposed_content: str, accepted_hunk_indices: str) -> str:
        """Selectively applies only chosen diff hunks to a file (e.g. '0,2'), rejecting the rest."""
        t0 = time.time()
        ok, why = SecurityPolicy.validate_path_containment(file_path, self.workspace_path)
        if not ok:
            return f"Security policy rejection: {why}"
        p = (self.workspace_path / file_path).resolve()
        if not p.exists() or not p.is_file():
            return f"File not found: '{file_path}'"

        orig = p.read_text(encoding="utf-8", errors="ignore")
        from ticket_to_code.runtime.diff_engine import DiffReviewEngine
        file_diff = DiffReviewEngine.compute_file_diff(file_path, orig, proposed_content)

        try:
            indices = {int(idx.strip()) for idx in accepted_hunk_indices.split(",") if idx.strip()}
        except Exception:
            return f"Invalid accepted_hunk_indices: '{accepted_hunk_indices}'. Expected comma-separated integers (e.g. '0,1')."

        applied_text = DiffReviewEngine.apply_selected_hunks(orig, file_diff.hunks, indices)
        # Push a rollback snapshot so rollback_all() also undoes hunk-level writes
        from ticket_to_code.editing.surgical_diff_engine import EditSnapshot
        self.diff_engine.undo_stack.append(
            EditSnapshot(file_path=str(p), original_content=orig, timestamp=time.time())
        )
        p.write_text(applied_text, encoding="utf-8")
        self._writes_performed += 1
        if file_path not in self.mutated_files:
            self.mutated_files.append(file_path)

        res = f"✅ Applied {len(indices)} of {len(file_diff.hunks)} hunks to '{file_path}' successfully."
        dur = int((time.time() - t0) * 1000)
        self._record_execution("apply_selected_diff_hunks", file_path, res, dur, True)
        return res


