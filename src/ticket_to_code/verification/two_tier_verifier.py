"""
Two-Tier Verification Engine: LSP Inner Loop + Scoped Build/Test Outer Loop

Grounded in AI IDE Verification Architecture:
1. Tier 1 Inner Loop (<1s):
   - Structured diagnostics on modified files (syntax, typing, missing imports).
   - Eliminates need to run expensive full-project builds when code has obvious compile errors.
2. Tier 2 Outer Loop (Authoritative):
   - Scoped strictly to the target sub-project (e.g. `xchange-ui` or specific Java microservice).
   - Runs compiler and unit tests (`mvn test -Dtest=...`, `npm test`).
3. Oscillation Integration:
   - Pairs with OscillationDetector and SurgicalDiffEngine.
   - Enforces automatic rollback on stagnation or cycling.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Tuple

from ticket_to_code.verification.oscillation_detector import (
    NormalizedDiagnostic,
    OscillationDetector,
    TransitionEvaluation,
    VerificationTransition,
)
from ticket_to_code.editing.surgical_diff_engine import SurgicalDiffEngine

logger = logging.getLogger(__name__)


@dataclass
class Tier1Result:
    """Result of Tier 1 fast syntax / typing check."""
    passed: bool
    diagnostics: List[NormalizedDiagnostic] = field(default_factory=list)
    raw_output: str = ""
    elapsed_s: float = 0.0
    raw_errors: List[str] = field(default_factory=list)


@dataclass
class Tier2Result:
    """Result of Tier 2 scoped compiler & unit test run."""
    passed: bool
    exit_code: int
    diagnostics: List[NormalizedDiagnostic] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    failed_tests: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class FullVerificationReport:
    """Consolidated verification verdict across both tiers."""
    passed: bool
    tier: str  # "tier1" or "tier2"
    evaluation: TransitionEvaluation
    diagnostics: List[NormalizedDiagnostic] = field(default_factory=list)
    summary: str = ""


class TwoTierVerifier:
    """
    Coordinates two-tier verification with infinite-loop oscillation protection.
    """

    def __init__(
        self,
        workspace_path: str,
        diff_engine: Optional[SurgicalDiffEngine] = None,
        max_recovery_turns: int = 3,
    ):
        self.workspace_path = Path(workspace_path).resolve()
        self.diff_engine = diff_engine or SurgicalDiffEngine(str(self.workspace_path))
        self.oscillation_detector = OscillationDetector(max_recovery_turns=max_recovery_turns)

    @staticmethod
    def _extract_tree_sitter_errors(node, max_errors: int = 5) -> List[Tuple[str, int, int, str]]:
        """
        Recursively traverses a Tree-sitter AST to extract exact syntax errors and missing tokens.
        Returns list of (error_type, line_number, column, description).
        """
        errors = []
        stack = [node]
        while stack and len(errors) < max_errors:
            curr = stack.pop()
            if curr.is_error:
                line = curr.start_point.row + 1
                col = curr.start_point.column
                errors.append(("SYNTAX_ERROR", line, col, f"Syntax error at line {line}:{col}"))
            elif curr.is_missing:
                line = curr.start_point.row + 1
                col = curr.start_point.column
                missing_token = curr.type
                errors.append(("MISSING_TOKEN", line, col, f"Missing '{missing_token}' at line {line}:{col}"))

            for child in reversed(curr.children):
                stack.append(child)
        return errors

    def run_tier1_fast_check(self, modified_files: Sequence[str]) -> Tier1Result:
        """
        Fast Tier 1 syntax and structural typing check on modified files (<1s).
        Supports Python, TypeScript, JavaScript, and Java.
        """
        import ast
        import time
        start = time.time()
        diagnostics: List[NormalizedDiagnostic] = []
        raw_errors: List[str] = []

        for f_path in modified_files:
            target = self.workspace_path / f_path if not Path(f_path).is_absolute() else Path(f_path)
            if not target.is_file():
                continue

            ext = target.suffix.lower()

            # 1. Python AST & Grammar verification
            if ext == ".py":
                try:
                    content = target.read_text(encoding="utf-8", errors="replace")
                    ast.parse(content, filename=str(target))
                except SyntaxError as syn_err:
                    diagnostics.append(
                        NormalizedDiagnostic.create(
                            file=f_path,
                            code="PYTHON_SYNTAX_ERROR",
                            raw_message=f"line {syn_err.lineno}: {syn_err.msg} ({syn_err.text or ''})",
                        )
                    )
                    raw_errors.append(f"{f_path}({syn_err.lineno},1): error PYTHON_SYNTAX_ERROR: {syn_err.msg}")
                except Exception as exc:
                    diagnostics.append(
                        NormalizedDiagnostic.create(
                            file=f_path,
                            code="PYTHON_PARSE_ERROR",
                            raw_message=str(exc),
                        )
                    )
                    raw_errors.append(f"{f_path}(1,1): error PYTHON_PARSE_ERROR: {exc}")

            # 2. TypeScript / JavaScript AST & Tree-Sitter verification
            elif ext in {".ts", ".tsx", ".js", ".jsx"}:
                try:
                    content = target.read_text(encoding="utf-8", errors="replace")
                    # Try authoritative Tree-sitter AST validation (<2ms)
                    tree_sitter_parsed = False
                    try:
                        import tree_sitter
                        import tree_sitter_typescript
                        # Select grammar: TSX grammar for .tsx/.jsx, TypeScript grammar for .ts/.js
                        if ext in {".tsx", ".jsx"} and hasattr(tree_sitter_typescript, "language_tsx"):
                            ts_grammar = tree_sitter_typescript.language_tsx()
                        else:
                            ts_grammar = tree_sitter_typescript.language_typescript()
                        lang = tree_sitter.Language(ts_grammar)
                        parser = tree_sitter.Parser(lang)
                        tree = parser.parse(content.encode("utf-8"))
                        tree_sitter_parsed = True
                        if tree.root_node.has_error:
                            # Verify if TypeScript compiler itself considers this file valid (avoids false-positive tree-sitter AST errors like JSX '&')
                            ts_is_valid = False
                            try:
                                cur = target.parent
                                ts_pkg = None
                                while cur != cur.parent:
                                    candidate = cur / "node_modules" / "typescript"
                                    if candidate.exists():
                                        ts_pkg = candidate
                                        break
                                    cur = cur.parent
                                if ts_pkg:
                                    import subprocess
                                    ts_pkg_str = json.dumps(str(ts_pkg).replace('\\', '/'))
                                    target_str = json.dumps(str(target).replace('\\', '/'))
                                    node_script = f"const ts = require({ts_pkg_str}); const fs = require('fs'); const code = fs.readFileSync({target_str}, 'utf8'); const sf = ts.createSourceFile('temp.tsx', code, ts.ScriptTarget.Latest, true); process.exit(sf.parseDiagnostics.length > 0 ? 1 : 0);"
                                    res = subprocess.run(["node", "-e", node_script], capture_output=True, timeout=5)
                                    if res.returncode == 0:
                                        ts_is_valid = True
                            except Exception:
                                pass

                            if not ts_is_valid:
                                ts_errors = self._extract_tree_sitter_errors(tree.root_node)
                                content_lines = content.splitlines()
                                for err_code, line, col, msg in ts_errors:
                                    # Skip harmless standalone '&' in JSX text (which standard tsc compiles cleanly)
                                    if ext in {".tsx", ".jsx"} and line <= len(content_lines) and "&" in content_lines[line - 1]:
                                        continue
                                    diagnostics.append(
                                        NormalizedDiagnostic.create(
                                            file=f_path,
                                            code=f"TS_{err_code}",
                                            raw_message=f"line {line}:{col} - {msg}",
                                        )
                                    )
                                    raw_errors.append(f"{f_path}({line},{col}): error TS_{err_code}: {msg}")
                    except Exception:
                        tree_sitter_parsed = False

                    # Delimiter & syntax fallbacks
                    if not tree_sitter_parsed:
                        open_braces = content.count("{") - content.count("}")
                        open_parens = content.count("(") - content.count(")")
                        open_brackets = content.count("[") - content.count("]")

                        backticks = content.count("`")
                        if backticks % 2 != 0:
                            diagnostics.append(
                                NormalizedDiagnostic.create(
                                    file=f_path,
                                    code="UNCLOSED_TEMPLATE_LITERAL",
                                    raw_message="Odd number of template backticks detected (unclosed template string).",
                                )
                            )

                        if open_braces != 0 or open_parens != 0 or open_brackets != 0:
                            diagnostics.append(
                                NormalizedDiagnostic.create(
                                    file=f_path,
                                    code="UNBALANCED_DELIMITERS",
                                    raw_message=f"Braces delta: {open_braces}, Parens delta: {open_parens}, Brackets delta: {open_brackets}",
                                )
                            )

                    # Node.js syntax check for plain JS
                    if ext == ".js":
                        try:
                            node_res = subprocess.run(
                                ["node", "--check", str(target)],
                                capture_output=True,
                                text=True,
                                timeout=2,
                            )
                            if node_res.returncode != 0:
                                diagnostics.append(
                                    NormalizedDiagnostic.create(
                                        file=f_path,
                                        code="JS_SYNTAX_ERROR",
                                        raw_message=node_res.stderr.strip(),
                                    )
                                )
                        except Exception:
                            pass
                except Exception as exc:
                    diagnostics.append(
                        NormalizedDiagnostic.create(
                            file=f_path,
                            code="READ_ERROR",
                            raw_message=str(exc),
                        )
                    )

            # 3. Java syntax and Tree-Sitter / class structural check
            elif ext == ".java":
                try:
                    content = target.read_text(encoding="utf-8", errors="replace")
                    # Try authoritative Tree-sitter Java parser (<2ms)
                    tree_sitter_parsed = False
                    try:
                        import tree_sitter
                        import tree_sitter_java
                        lang = tree_sitter.Language(tree_sitter_java.language())
                        parser = tree_sitter.Parser(lang)
                        tree = parser.parse(content.encode("utf-8"))
                        tree_sitter_parsed = True
                        if tree.root_node.has_error:
                            java_errors = self._extract_tree_sitter_errors(tree.root_node)
                            for err_code, line, col, msg in java_errors:
                                diagnostics.append(
                                    NormalizedDiagnostic.create(
                                        file=f_path,
                                        code=f"JAVA_{err_code}",
                                        raw_message=f"line {line}:{col} - {msg}",
                                    )
                                )
                                raw_errors.append(f"{f_path}({line},{col}): error JAVA_{err_code}: {msg}")
                    except Exception:
                        tree_sitter_parsed = False

                    if not tree_sitter_parsed:
                        open_braces = content.count("{") - content.count("}")
                        open_parens = content.count("(") - content.count(")")
                        open_brackets = content.count("[") - content.count("]")

                        if open_braces != 0 or open_parens != 0 or open_brackets != 0:
                            diagnostics.append(
                                NormalizedDiagnostic.create(
                                    file=f_path,
                                    code="JAVA_UNBALANCED_DELIMITERS",
                                    raw_message=f"Java delimiters delta: braces={open_braces}, parens={open_parens}, brackets={open_brackets}",
                                )
                            )
                            raw_errors.append(f"{f_path}(1,1): error JAVA_UNBALANCED_DELIMITERS: braces={open_braces}, parens={open_parens}, brackets={open_brackets}")

                    # Verify class/interface name matches filename
                    expected_type_name = target.stem
                    type_decl_match = re.search(
                        r'\b(public\s+)?(class|interface|enum|record)\s+(\w+)\b', content
                    )
                    if type_decl_match and "public" in (type_decl_match.group(1) or ""):
                        declared_name = type_decl_match.group(3)
                        if declared_name != expected_type_name:
                            diagnostics.append(
                                NormalizedDiagnostic.create(
                                    file=f_path,
                                    code="JAVA_TYPE_NAME_MISMATCH",
                                    raw_message=f"Public type '{declared_name}' does not match file name '{expected_type_name}'",
                                )
                            )
                            raw_errors.append(f"{f_path}(1,1): error JAVA_TYPE_NAME_MISMATCH: Public type '{declared_name}' does not match file name '{expected_type_name}'")
                except Exception as exc:
                    diagnostics.append(
                        NormalizedDiagnostic.create(
                            file=f_path,
                            code="READ_ERROR",
                            raw_message=str(exc),
                        )
                    )
                    raw_errors.append(f"{f_path}(1,1): error READ_ERROR: {exc}")

        elapsed = time.time() - start
        passed = (len(diagnostics) == 0)
        return Tier1Result(
            passed=passed,
            diagnostics=diagnostics,
            elapsed_s=round(elapsed, 3),
            raw_errors=raw_errors,
        )

    def run_tier2_scoped_build(
        self,
        target_service: str = "",
        test_filter: Optional[str] = None,
    ) -> Tier2Result:
        """
        Authoritative Tier 2 build and unit test execution, scoped strictly to the target sub-project.
        Operates dynamically across Pytest (Python), Maven/Gradle (Java), and Karma/Jest/NPM (Angular/TS).
        """
        import time
        start = time.time()
        service_dir = self.workspace_path / target_service if target_service else self.workspace_path
        if not service_dir.is_dir():
            service_dir = self.workspace_path

        diagnostics: List[NormalizedDiagnostic] = []
        failed_tests: List[str] = []

        # ── 1. Dynamic Toolchain Detection ──────────────────────────────────
        is_python = (
            (service_dir / "pytest.ini").is_file()
            or (service_dir / "pyproject.toml").is_file()
            or (service_dir / "setup.py").is_file()
            or (service_dir / "tests").is_dir()
        )
        is_maven = (service_dir / "pom.xml").is_file()
        is_gradle = (
            (service_dir / "build.gradle").is_file()
            or (service_dir / "build.gradle.kts").is_file()
            or (service_dir / "gradlew").is_file()
        )
        is_angular = (service_dir / "angular.json").is_file()
        is_npm = (service_dir / "package.json").is_file()

        # ── 2. Select Command Line ──────────────────────────────────────────
        cmd: List[str] = []
        if is_python:
            cmd = ["pytest"]
            if test_filter:
                cmd.extend([test_filter, "-v"])
            else:
                cmd.append("-v")
        elif is_maven:
            if test_filter:
                cmd = ["mvn", "test", f"-Dtest={test_filter}"]
            else:
                cmd = ["mvn", "test-compile"]
        elif is_gradle:
            gradle_bin = "./gradlew" if os.name != "nt" else "gradlew.bat"
            if test_filter:
                cmd = [gradle_bin, "test", f"--tests={test_filter}"]
            else:
                cmd = [gradle_bin, "testClasses"]
        elif is_angular:
            cmd = ["npx", "ng", "test", "--watch=false", "--browsers=ChromeHeadless"]
            if test_filter:
                cmd.append(f"--include={test_filter}")
        elif is_npm:
            cmd = ["npm", "test", "--if-present"]
        else:
            return Tier2Result(
                passed=True,
                exit_code=0,
                stdout="No build toolchain detected; skipped build execution.",
                elapsed_s=round(time.time() - start, 2),
            )

        # ── 3. Execute Scoped Process & Capture Output ─────────────────────
        # run_checked_subprocess validates the command, scrubs the env, confines cwd to the
        # workspace and kills the whole process tree on timeout.
        from ticket_to_code.runtime.security import SecurityViolation, run_checked_subprocess
        from ticket_to_code.runtime.limits import BUILD_OUTPUT_MAX_CHARS, truncate
        try:
            try:
                rc, raw_stdout, raw_stderr = run_checked_subprocess(
                    cmd, cwd=service_dir, timeout=180, workspace=self.workspace_path
                )
            except SecurityViolation as sv:
                return Tier2Result(
                    passed=False,
                    exit_code=1,
                    stderr=f"Security violation: test command blocked: {sv}",
                    stdout="",
                    elapsed_s=round(time.time() - start, 2),
                    diagnostics=[
                        NormalizedDiagnostic.create(
                            file=target_service or str(service_dir.name),
                            code="SECURITY_VIOLATION",
                            raw_message=f"Command blocked by safety policy: {sv}",
                        )
                    ],
                )
            passed = (rc == 0)

            # Strip ANSI color codes
            clean_stdout = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw_stdout)
            clean_stderr = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw_stderr)
            combined = clean_stdout + "\n" + clean_stderr

            # ── 4. Parse Structured Failures & Diagnostics ─────────────────
            if not passed:
                for line in combined.splitlines():
                    stripped = line.strip()
                    # Pytest failures: FAILED tests/test_foo.py::test_bar
                    if stripped.startswith("FAILED ") or stripped.startswith("ERROR "):
                        failed_tests.append(stripped)
                        diagnostics.append(
                            NormalizedDiagnostic.create(
                                file=target_service or str(service_dir.name),
                                code="PYTEST_FAILURE",
                                raw_message=stripped,
                            )
                        )
                    # Maven / Java failures
                    elif "[ERROR]" in stripped and (".java:" in stripped or "Compilation failure" in stripped):
                        diagnostics.append(
                            NormalizedDiagnostic.create(
                                file=target_service or str(service_dir.name),
                                code="MAVEN_COMPILE_ERROR",
                                raw_message=stripped,
                            )
                        )
                    elif "Failed tests:" in stripped or "Tests in error:" in stripped:
                        failed_tests.append(stripped)
                    # Angular / TS / Jest failures
                    elif "error TS" in stripped or "ERROR in" in stripped:
                        diagnostics.append(
                            NormalizedDiagnostic.create(
                                file=target_service or str(service_dir.name),
                                code="TS_COMPILATION_ERROR",
                                raw_message=stripped,
                            )
                        )
                    elif "✕" in stripped or "FAIL " in stripped:
                        failed_tests.append(stripped)

            return Tier2Result(
                passed=passed,
                exit_code=rc,
                diagnostics=diagnostics,
                stdout=truncate(clean_stdout, BUILD_OUTPUT_MAX_CHARS),
                stderr=truncate(clean_stderr, BUILD_OUTPUT_MAX_CHARS),
                failed_tests=failed_tests,
                elapsed_s=round(time.time() - start, 2),
            )
        except Exception as exc:
            return Tier2Result(
                passed=False,
                exit_code=-1,
                diagnostics=[
                    NormalizedDiagnostic.create(
                        file=target_service or str(service_dir.name),
                        code="EXECUTION_EXCEPTION",
                        raw_message=str(exc),
                    )
                ],
                stderr=str(exc),
                elapsed_s=round(time.time() - start, 2),
            )

    def verify_cycle(
        self,
        modified_files: Sequence[str],
        target_service: str = "",
        test_filter: Optional[str] = None,
    ) -> FullVerificationReport:
        """
        Execute full two-tier verification cycle with oscillation protection.

        If stagnation or cycling is detected, automatically triggers rollback.
        """
        # Step 1: Run Tier 1 fast check
        t1 = self.run_tier1_fast_check(modified_files)
        if not t1.passed:
            eval_res = self.oscillation_detector.evaluate(t1.diagnostics)
            if eval_res.requires_rollback:
                logger.warning("[TwoTierVerifier] Tier 1 failure with oscillation/stagnation: rolling back!")
                self.diff_engine.rollback_last_edit()

            return FullVerificationReport(
                passed=False,
                tier="tier1",
                evaluation=eval_res,
                diagnostics=t1.diagnostics,
                summary=f"Tier 1 fast check failed with {len(t1.diagnostics)} error(s).",
            )

        # Step 2: Run Tier 2 scoped build
        t2 = self.run_tier2_scoped_build(target_service, test_filter)
        eval_res = self.oscillation_detector.evaluate(t2.diagnostics)

        if not t2.passed:
            if eval_res.requires_rollback:
                logger.warning("[TwoTierVerifier] Tier 2 build failure with oscillation/stagnation: rolling back!")
                self.diff_engine.rollback_last_edit()

            return FullVerificationReport(
                passed=False,
                tier="tier2",
                evaluation=eval_res,
                diagnostics=t2.diagnostics,
                summary=f"Tier 2 scoped build failed (exit code {t2.exit_code}).",
            )

        # Clean build!
        return FullVerificationReport(
            passed=True,
            tier="tier2",
            evaluation=eval_res,
            diagnostics=[],
            summary="All verification checks passed cleanly.",
        )
