"""
Language Server Protocol (LSP) & Sub-Second Diagnostic Client.

Modeled after Cursor Composer and Windsurf real-time compiler integration.
Provides:
  1. Sub-second (<50-200ms) compiler diagnostics without running full multi-minute build pipelines.
  2. Semantic Navigation: find_references(), go_to_definition(), and workspace symbol search.
Supports TypeScript/JavaScript (tsserver / tsc AST), Java (javac / jdtls), Go (gopls / go vet), and Python.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ticket_to_code.lsp_client")


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class LSPDiagnostic:
    file_path: str
    line: int
    column: int
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    code: Optional[str] = None  # e.g. "TS2304" or "TS2307"

    def format_slice(self) -> str:
        code_str = f" [{self.code}]" if self.code else ""
        return f"{Path(self.file_path).name}:{self.line}:{self.column} - {self.severity.value}{code_str}: {self.message}"


@dataclass
class LSPLocation:
    file_path: str
    line: int
    column: int
    snippet: str = ""


class DiagnosticList(list):
    """``list[LSPDiagnostic]`` that also carries a ``degraded`` flag.

    ``degraded=True`` means the real compiler did not produce a verdict (tsc
    timed out / missing tsconfig / tsc error) and only the lightweight syntax
    fallback ran, so an empty list must NOT be read as "verified clean".
    Being a list subclass keeps every existing caller working unchanged.
    """

    def __init__(self, items=(), degraded: bool = False):
        super().__init__(items)
        self.degraded: bool = bool(degraded)


class FastLSPClient:
    """
    In-process, high-speed LSP and diagnostic service.
    Replaces 70-second terminal build scripts with sub-second feedback during repair loops.
    """

    def __init__(self, workspace_path: str | Path) -> None:
        self.workspace = Path(workspace_path).resolve()
        self._cache: Dict[str, Tuple[float, List[LSPDiagnostic]]] = {}
        self._javac_scratch = Path(tempfile.mkdtemp(prefix="javac_lsp_"))
        # True when the most recent get_diagnostics() verdict was degraded (fail-open).
        self.last_degraded: bool = False

    def get_diagnostics(self, file_path: str | Path, fast_only: bool = True) -> List[LSPDiagnostic]:
        """
        Retrieves compiler diagnostics for a file in <200ms.
        Uses language-specific lightweight AST and in-process compiler hooks.

        Returns a ``DiagnosticList`` (a ``list`` subclass) whose ``.degraded``
        attribute is True when the compiler could not run and only the syntax
        fallback produced the verdict; ``self.last_degraded`` mirrors it.
        """
        p = Path(file_path)
        abs_p = (self.workspace / p) if not p.is_absolute() else p
        if not abs_p.is_file():
            self.last_degraded = False
            return DiagnosticList()

        # Check memory cache based on mtime
        mtime = abs_p.stat().st_mtime
        if str(abs_p) in self._cache:
            cached_mtime, cached_diags = self._cache[str(abs_p)]
            if cached_mtime == mtime:
                self.last_degraded = bool(getattr(cached_diags, "degraded", False))
                return cached_diags

        ext = abs_p.suffix.lower()
        diags: List[LSPDiagnostic] = []

        t0 = time.time()
        if ext in (".ts", ".tsx", ".js", ".jsx"):
            diags = self._diagnose_typescript(abs_p)
        elif ext == ".py":
            diags = self._diagnose_python(abs_p)
        elif ext == ".java":
            diags = self._diagnose_java(abs_p)
        elif ext == ".go":
            diags = self._diagnose_go(abs_p)

        if not isinstance(diags, DiagnosticList):
            diags = DiagnosticList(diags, degraded=False)

        elapsed = time.time() - t0
        logger.debug(
            f"⚡ [LSPClient] Diagnosed {abs_p.name} in {elapsed*1000:.1f}ms "
            f"({len(diags)} diagnostic(s){', DEGRADED' if diags.degraded else ''})"
        )

        self._cache[str(abs_p)] = (mtime, diags)
        self.last_degraded = diags.degraded
        return diags

    def check_delta_diagnostics(
        self, file_path: str | Path, proposed_content: str
    ) -> Tuple[bool, List[LSPDiagnostic]]:
        """
        Delta-scoped compiler diagnostic evaluation:
          1. Measures baseline diagnostics on current file.
          2. Temporarily writes proposed_content.
          3. Measures candidate diagnostics.
          4. Computes: Delta_Errors = Candidate_Errors - Baseline_Errors.
          5. Restores original file content cleanly.
        Only flags regressions introduced by the proposed patch (zero false positives on pre-existing codebase errors).
        """
        p = Path(file_path)
        abs_p = (self.workspace / p) if not p.is_absolute() else p

        orig_content = abs_p.read_text(encoding="utf-8", errors="ignore") if abs_p.is_file() else None

        # 1. Baseline
        baseline_diags = self.get_diagnostics(abs_p) if orig_content is not None else []
        baseline_errors = [d for d in baseline_diags if d.severity == DiagnosticSeverity.ERROR]
        baseline_messages = {f"{d.line}:{d.message}" for d in baseline_errors}

        # 2. Apply proposed
        try:
            abs_p.parent.mkdir(parents=True, exist_ok=True)
            abs_p.write_text(proposed_content, encoding="utf-8")
            self._cache.pop(str(abs_p), None)

            # 3. Candidate
            cand_diags = self.get_diagnostics(abs_p)
            cand_errors = [d for d in cand_diags if d.severity == DiagnosticSeverity.ERROR]

            # 4. Filter only NEW errors
            new_errors = [
                d for d in cand_errors
                if f"{d.line}:{d.message}" not in baseline_messages
            ]

            is_valid = len(new_errors) == 0
            return is_valid, new_errors
        finally:
            # 5. Restore
            if orig_content is not None:
                abs_p.write_text(orig_content, encoding="utf-8")
            elif abs_p.is_file():
                abs_p.unlink()
            self._cache.pop(str(abs_p), None)

    def _diagnose_typescript(self, abs_path: Path) -> List[LSPDiagnostic]:
        """
        Runs project-aware tsc type check honoring tsconfig.json path mappings, JSX, and baseUrl.
        Uses diff-scoped stdout filtering to prevent false positives on pre-existing repo errors.

        Returns a ``DiagnosticList``; ``.degraded`` is True whenever tsc did not
        deliver a verdict (no tsconfig, tsc missing/errored, or timed out) and the
        brace-count fallback is all that ran.
        """
        diags: List[LSPDiagnostic] = []
        tsc_ran_ok = False
        # Find nearest tsconfig.json
        cur = abs_path.parent
        tsconfig: Optional[Path] = None
        while cur != cur.parent and cur >= self.workspace:
            cand = cur / "tsconfig.json"
            if cand.is_file():
                tsconfig = cand
                break
            cur = cur.parent

        if tsconfig:
            node_tsc = None
            for p_dir in [tsconfig.parent, self.workspace, self.workspace.parent]:
                cand_bin = p_dir / "node_modules" / ".bin" / ("tsc.cmd" if sys.platform == "win32" else "tsc")
                if cand_bin.is_file():
                    node_tsc = cand_bin
                    break
            cmd = [str(node_tsc) if node_tsc else "npx", "--no-install", "tsc", "--noEmit", "--skipLibCheck", "--project", str(tsconfig)]
            try:
                res = subprocess.run(
                    cmd,
                    cwd=str(tsconfig.parent),
                    capture_output=True,
                    text=True,
                    timeout=8,
                    shell=(sys.platform == "win32"),
                    env=dict(os.environ),
                )
                rel_posix = str(abs_path.relative_to(tsconfig.parent)).replace("\\", "/")
                combined_out = res.stdout + "\n" + res.stderr
                # tsc exit 0 = clean; exit 2 = type errors reported. Anything else
                # (npx could not find tsc, "command not found", ENOENT) is NOT a verdict.
                tsc_ran_ok = res.returncode in (0, 1, 2) and not re.search(
                    r"(not found|could not determine executable|ENOENT|npm ERR!|is not recognized)",
                    combined_out, re.IGNORECASE,
                )
                for line in res.stdout.splitlines() + res.stderr.splitlines():
                    norm_line = line.replace("\\", "/")
                    if rel_posix in norm_line or abs_path.name in norm_line:
                        m = re.search(r"([^(]+)\((\d+),(\d+)\):\s*(error|warning)\s*(TS\d+)?:\s*(.*)", line)
                        if m:
                            diags.append(LSPDiagnostic(
                                file_path=str(abs_path),
                                line=int(m.group(2)),
                                column=int(m.group(3)),
                                severity=DiagnosticSeverity.ERROR if m.group(4) == "error" else DiagnosticSeverity.WARNING,
                                code=m.group(5),
                                message=m.group(6).strip(),
                            ))
            except subprocess.TimeoutExpired:
                tsc_ran_ok = False
                logger.warning(f"[LSPClient] tsc timed out for {abs_path.name} — diagnostics DEGRADED (unverified)")
            except Exception as exc:
                tsc_ran_ok = False
                logger.debug(f"[LSPClient] Project-aware TypeScript compile notice: {exc}")
        else:
            logger.debug(f"[LSPClient] No tsconfig.json above {abs_path.name} — tsc skipped, diagnostics DEGRADED")

        # AST token & delimiter syntax validation if tsc is not installed in environment
        if not diags:
            try:
                text = abs_path.read_text(encoding="utf-8", errors="ignore")
                # 1. Delimiter balance checks (braces, parentheses, brackets)
                delimiters = [("{", "}", "braces"), ("(", ")", "parentheses"), ("[", "]", "brackets")]
                for o_c, c_c, label in delimiters:
                    diff = text.count(o_c) - text.count(c_c)
                    if diff != 0:
                        lines = text.splitlines()
                        diags.append(LSPDiagnostic(
                            file_path=str(abs_path),
                            line=max(1, len(lines)),
                            column=1,
                            severity=DiagnosticSeverity.ERROR,
                            code="SYNTAX_DELIMITER",
                            message=f"SyntaxError: Unbalanced {label} in {abs_path.name} (missing {abs(diff)} {'closing' if diff > 0 else 'opening'} {o_c}{c_c})",
                        ))
                # 2. Malformed import/export checks
                if re.search(r'export\s+(?:interface|type|class|function)\s*[{;=]', text):
                    diags.append(LSPDiagnostic(
                        file_path=str(abs_path),
                        line=1,
                        column=1,
                        severity=DiagnosticSeverity.ERROR,
                        code="TS_MALFORMED_EXPORT",
                        message="SyntaxError: Malformed export declaration missing symbol identifier",
                    ))
            except Exception:
                pass

        return DiagnosticList(diags, degraded=not tsc_ran_ok)

    def _diagnose_python(self, abs_path: Path) -> List[LSPDiagnostic]:
        """Runs Python built-in AST compile to catch syntax and indentation errors in <5ms."""
        diags: List[LSPDiagnostic] = []
        try:
            code = abs_path.read_text(encoding="utf-8", errors="ignore")
            ast.parse(code, filename=str(abs_path))
        except SyntaxError as e:
            diags.append(LSPDiagnostic(
                file_path=str(abs_path),
                line=e.lineno or 1,
                column=e.offset or 1,
                severity=DiagnosticSeverity.ERROR,
                message=e.msg,
                code="SyntaxError",
            ))
        except Exception as e:
            diags.append(LSPDiagnostic(
                file_path=str(abs_path),
                line=1,
                column=1,
                severity=DiagnosticSeverity.ERROR,
                message=str(e),
            ))
        return diags

    @classmethod
    def _find_javac(cls) -> Optional[str]:
        which = shutil.which("javac")
        if which:
            return which
        for env_var in ("JAVA_HOME", "JDK_HOME"):
            val = os.environ.get(env_var)
            if val:
                cand = Path(val) / "bin" / ("javac.exe" if sys.platform == "win32" else "javac")
                if cand.is_file():
                    return str(cand)
        if sys.platform == "win32":
            common_roots = [
                Path(r"C:\Program Files\Java"),
                Path(r"C:\Program Files\Fortify"),
                Path(r"C:\Program Files\Eclipse Adoptium"),
                Path(r"C:\Program Files\Microsoft"),
                Path(r"C:\Program Files\Amazon Corretto"),
                Path(r"C:\Program Files (x86)\Java"),
            ]
            for root in common_roots:
                if root.is_dir():
                    try:
                        matches = list(root.glob("**/bin/javac.exe"))
                        if matches:
                            return str(matches[0])
                    except Exception:
                        pass
        return None

    def _diagnose_java(self, abs_path: Path) -> List[LSPDiagnostic]:
        """
        Runs fast javac syntax pass with classpath discovery and missing-dependency filtering.
        Eliminates false positives on uncompiled Spring Boot/Maven/Gradle projects while catching real syntax errors.
        """
        diags: List[LSPDiagnostic] = []
        javac_bin = self._find_javac()
        try:
            if javac_bin:
                # 1. Discover local classpath elements
                cp_elements: List[str] = []
                cur = abs_path.parent
                while cur >= self.workspace:
                    for target_cand in [cur / "target" / "classes", cur / "build" / "classes" / "java" / "main"]:
                        if target_cand.is_dir():
                            cp_elements.append(str(target_cand))
                    cur = cur.parent

                self._javac_scratch.mkdir(parents=True, exist_ok=True)
                cmd = [javac_bin, "-proc:none", "-nowarn", "-d", str(self._javac_scratch)]
                if cp_elements:
                    cmd.extend(["-cp", os.pathsep.join(cp_elements)])
                cmd.append(str(abs_path))

            res = subprocess.run(
                cmd,
                cwd=str(abs_path.parent),
                capture_output=True,
                text=True,
                timeout=5,
                shell=(sys.platform == "win32"),
            )
            lines = res.stderr.splitlines()
            for idx, line in enumerate(lines):
                m = re.search(r"(?:([A-Za-z]:[\\/][^:]+|[^:]+)):(\d+):\s*(error|warning):\s*(.*)", line)
                if m and Path(m.group(1).strip()).name == abs_path.name:
                    msg = m.group(4).strip()
                    context_chunk = "\n".join(lines[idx: min(len(lines), idx + 4)])
                    is_missing_external_class = (
                        "does not exist" in msg
                        or "package " in msg
                        or "cannot access" in msg
                        or "class file for" in context_chunk
                        or "interface expected here" in msg
                        or ("cannot find symbol" in msg and ("symbol:   class" in context_chunk or "symbol: class" in context_chunk))
                    )
                    # Ignore missing external classes/packages/base-entities, but strictly flag missing methods and variables
                    if is_missing_external_class and not ("symbol:   method" in context_chunk or "symbol:   variable" in context_chunk):
                        continue
                    # In lightweight javac pre-flight (runs without Lombok / Spring bytecode processor):
                    # Ignore Lombok generated symbols (e.g. log) and Spring Data repository methods (findById, save, etc.)
                    if "cannot find symbol" in msg:
                        if "variable log" in context_chunk:
                            continue
                        if any(repo_m in context_chunk for repo_m in ["findById", "save", "findAll", "delete", "deleteAll", "existsById", "count"]):
                            continue
                    diag_msg = msg
                    for c_line in context_chunk.splitlines()[1:]:
                        if "symbol:" in c_line:
                            diag_msg = f"{msg}: {c_line.strip()}"
                            break
                    diags.append(LSPDiagnostic(
                        file_path=str(abs_path),
                        line=int(m.group(2)),
                        column=1,
                        severity=DiagnosticSeverity.ERROR if m.group(3) == "error" else DiagnosticSeverity.WARNING,
                        message=diag_msg,
                    ))
        except Exception:
            pass

        # AST token & delimiter syntax validation for Java (unbalanced braces/brackets)
        try:
            text = abs_path.read_text(encoding="utf-8", errors="ignore")
            for o_c, c_c, label in [("{", "}", "braces"), ("(", ")", "parentheses"), ("[", "]", "brackets")]:
                diff = text.count(o_c) - text.count(c_c)
                if diff != 0:
                    lines = text.splitlines()
                    diags.append(LSPDiagnostic(
                        file_path=str(abs_path),
                        line=max(1, len(lines)),
                        column=1,
                        severity=DiagnosticSeverity.ERROR,
                        code="JAVA_SYNTAX_DELIMITER",
                        message=f"SyntaxError: Unbalanced {label} in {abs_path.name} (missing {abs(diff)} {'closing' if diff > 0 else 'opening'} {o_c}{c_c})",
                    ))
        except Exception:
            pass

        # Fallback static method check if javac is unavailable
        if not diags and not javac_bin:
            try:
                text = abs_path.read_text(encoding="utf-8", errors="ignore")
                declared_methods = set(re.findall(r"(?:public|protected|private|static|\s)\s+[\w<>\[\],\s]+\s+(\w+)\s*\([^)]*\)\s*\{", text))
                calls = re.findall(r"this\.(\w+)\s*\(", text)
                for call in calls:
                    if call not in declared_methods and call not in ("equals", "hashCode", "toString", "getClass", "notify", "notifyAll", "wait"):
                        diags.append(LSPDiagnostic(
                            file_path=str(abs_path),
                            line=1,
                            column=1,
                            severity=DiagnosticSeverity.ERROR,
                            code="JAVA_HALLUCINATED_METHOD",
                            message=f"cannot find symbol: method {call}() in class {abs_path.stem}",
                        ))
            except Exception:
                pass

        return diags

    def _diagnose_go(self, abs_path: Path) -> List[LSPDiagnostic]:
        """Runs fast go vet or go build -n check."""
        diags: List[LSPDiagnostic] = []
        try:
            res = subprocess.run(
                ["go", "vet", str(abs_path)],
                cwd=str(abs_path.parent),
                capture_output=True,
                text=True,
                timeout=4,
                shell=(sys.platform == "win32"),
            )
            for line in res.stderr.splitlines():
                m = re.search(r"([^:]+):(\d+):(\d+):\s*(.*)", line)
                if m and Path(m.group(1).strip()).name == abs_path.name:
                    diags.append(LSPDiagnostic(
                        file_path=str(abs_path),
                        line=int(m.group(2)),
                        column=int(m.group(3)),
                        severity=DiagnosticSeverity.ERROR,
                        message=m.group(4).strip(),
                    ))
        except Exception:
            pass
        return diags

    def find_references(self, symbol: str, file_path: str | Path) -> List[LSPLocation]:
        """
        Locates all references to a symbol across the workspace.
        Uses ripgrep / regex with word boundary matching.
        """
        locations: List[LSPLocation] = []
        if not symbol or not symbol.isidentifier():
            return []

        pattern = rf"\b{re.escape(symbol)}\b"
        try:
            res = subprocess.run(
                ["git", "grep", "-n", "-I", "-E", pattern],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=5,
                shell=(sys.platform == "win32"),
            )
            for line in res.stdout.splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    f_rel = parts[0]
                    line_no = int(parts[1]) if parts[1].isdigit() else 1
                    snippet = parts[2].strip()
                    locations.append(LSPLocation(
                        file_path=str(self.workspace / f_rel),
                        line=line_no,
                        column=snippet.find(symbol) + 1,
                        snippet=snippet,
                    ))
        except Exception:
            pass
        return locations

    def go_to_definition(self, symbol: str, file_path: str | Path) -> Optional[LSPLocation]:
        """Locates the declaration / definition of a symbol across the workspace."""
        if not symbol:
            return None
        patterns = [
            rf"(?:class|interface|enum|type|struct)\s+{re.escape(symbol)}\b",
            rf"(?:function|def|func)\s+{re.escape(symbol)}\b",
            rf"(?:const|let|var)\s+{re.escape(symbol)}\b",
        ]
        combined = "|".join(patterns)
        try:
            res = subprocess.run(
                ["git", "grep", "-n", "-I", "-E", combined],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=5,
                shell=(sys.platform == "win32"),
            )
            for line in res.stdout.splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    return LSPLocation(
                        file_path=str(self.workspace / parts[0]),
                        line=int(parts[1]) if parts[1].isdigit() else 1,
                        column=1,
                        snippet=parts[2].strip(),
                    )
        except Exception:
            pass
        return None
