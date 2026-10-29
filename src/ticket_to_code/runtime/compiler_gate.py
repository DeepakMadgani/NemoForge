"""
Closed-Loop Automated Compiler & Self-Repair Gate.

Intercepts code mutations, executes polyglot compiler checks (Gradle, Maven, NPM),
parses compilation diagnostics down to exact file:line coordinates,
and generates grounded self-repair feedback loops for the LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ticket_to_code.runtime.deductive_blackboard import DeductiveBlackboard


@dataclass
class CompilationDiagnostic:
    file_path: str
    line_number: int
    column_number: int
    error_code: str
    message: str
    raw_line: str


@dataclass
class CompilationResult:
    service_name: str
    passed: bool
    exit_code: int
    diagnostics: List[CompilationDiagnostic] = field(default_factory=list)
    raw_output: str = ""

    def get_repair_instruction(self) -> str:
        """Generates a targeted self-healing prompt instruction for the LLM."""
        if self.passed:
            return "✅ Compilation succeeded with 0 errors."

        lines = [
            f"❌ COMPILATION FAILED on {self.service_name} (Exit code {self.exit_code}):",
            "The compiler rejected your previous modification with the following concrete errors:",
        ]
        for d in self.diagnostics[:5]:
            pos = f":{d.line_number}" if d.line_number > 0 else ""
            lines.append(f"  • `{Path(d.file_path).name}{pos}`: {d.message}")

        lines.append("\n⚠️ ACTION REQUIRED: You must inspect the target file using `inspect_file` or `grep_search` to verify real constructors, types, or imports, and apply a corrective edit with `replace_file_content`.")
        return "\n".join(lines)


class CompilerGate:
    """Parses compiler errors across Java, TypeScript, Python, Go, Rust, and .NET to drive the self-repair loop."""

    # Java: /path/to/File.java:123: error: cannot find symbol or C:\path\File.java:123: error: ...
    JAVA_ERROR_RE = re.compile(
        r'^\s*(?:\[ERROR\]\s+)?(?:Error:\s+)?([A-Za-z0-9_:/\\.-]+\.java):(\d+):(?:\s*(\d+):)?\s*error:\s*(.+)$',
        re.MULTILINE | re.IGNORECASE
    )

    # TypeScript: src/app/file.ts:42:15 - error TS2304: Cannot find name 'IssuePriority'
    # or file.ts(245,18): error TS2304: ...
    TS_ERROR_RE = re.compile(
        r'^\s*(?:\[ERROR\]\s+)?(?:Error:\s+)?([A-Za-z0-9_:/\\.-]+\.ts)(?:\((\d+),(\d+)\)|:(\d+):(\d+))\s*[:-]\s*error\s*(TS\d+)?:\s*(.+)$',
        re.MULTILINE | re.IGNORECASE
    )

    # Python: file.py:42: error: ... or SyntaxError: ...
    PY_ERROR_RE = re.compile(
        r'^\s*([A-Za-z0-9_:/\\.-]+\.py):(\d+):(?:\s*(\d+):)?\s*(?:error|syntaxerror)?:\s*(.+)$',
        re.MULTILINE | re.IGNORECASE
    )

    # Python Traceback: File "file.py", line 42, in ... \n Error: ...
    PY_TRACEBACK_RE = re.compile(
        r'^\s*File\s+["\']([^"\']+\.py)["\'],\s+line\s+(\d+)(?:,\s+in\s+[^\n\r]+)?(?:\r?\n\s+.*)?\r?\n\s*([A-Za-z0-9_]+Error:\s*.+)$',
        re.MULTILINE
    )

    # Go: path/file.go:42:15: undefined: Foo
    GO_ERROR_RE = re.compile(
        r'^\s*([A-Za-z0-9_:/\\.-]+\.go):(\d+):(\d+):\s*(.+)$',
        re.MULTILINE
    )

    # Rust: error[E0425]: ... --> src/main.rs:42:15
    RUST_ERROR_RE = re.compile(
        r'^\s*error(?:\[([A-Z0-9]+)\])?:\s*(.+)\n\s*-->\s*([A-Za-z0-9_:/\\.-]+\.rs):(\d+):(\d+)',
        re.MULTILINE
    )

    # .NET / C#: File.cs(42,15): error CS0103: ...
    DOTNET_ERROR_RE = re.compile(
        r'^\s*([A-Za-z0-9_:/\\.-]+\.cs)(?:\((\d+),(\d+)\)):\s*error\s*([A-Z0-9]+)?:\s*(.+)$',
        re.MULTILINE | re.IGNORECASE
    )

    # Generic: [ERROR] ... at line X
    GENERIC_ERROR_RE = re.compile(r'\[ERROR\]\s*(.+)$', re.MULTILINE)

    @classmethod
    def parse_compiler_output(cls, service_name: str, exit_code: int, output: str) -> CompilationResult:
        """Parse raw compiler stdout/stderr into structured CompilationResult."""
        if exit_code == 0:
            return CompilationResult(
                service_name=service_name,
                passed=True,
                exit_code=0,
                raw_output=output,
            )

        diagnostics: List[CompilationDiagnostic] = []

        # 1. Try Java patterns
        for m in cls.JAVA_ERROR_RE.finditer(output):
            fpath, line_str, col_str, msg = m.group(1), m.group(2), m.group(3) or "0", m.group(4)
            diagnostics.append(CompilationDiagnostic(
                file_path=fpath.strip(),
                line_number=int(line_str),
                column_number=int(col_str),
                error_code="javac",
                message=msg.strip(),
                raw_line=m.group(0),
            ))

        # 2. Try TypeScript patterns
        for m in cls.TS_ERROR_RE.finditer(output):
            fpath = m.group(1)
            line_str = m.group(2) or m.group(4) or "0"
            col_str = m.group(3) or m.group(5) or "0"
            code = m.group(6)
            msg = m.group(7)
            diagnostics.append(CompilationDiagnostic(
                file_path=fpath.strip(),
                line_number=int(line_str),
                column_number=int(col_str),
                error_code=code,
                message=msg.strip(),
                raw_line=m.group(0),
            ))

        # 3. Try Python patterns
        if not diagnostics:
            for m in cls.PY_TRACEBACK_RE.finditer(output):
                fpath, line_str, msg = m.group(1), m.group(2), m.group(3)
                diagnostics.append(CompilationDiagnostic(
                    file_path=fpath.strip(),
                    line_number=int(line_str),
                    column_number=0,
                    error_code="python",
                    message=msg.strip(),
                    raw_line=m.group(0),
                ))
            if not diagnostics:
                for m in cls.PY_ERROR_RE.finditer(output):
                    fpath, line_str, col_str, msg = m.group(1), m.group(2), m.group(3) or "0", m.group(4)
                    diagnostics.append(CompilationDiagnostic(
                        file_path=fpath.strip(),
                        line_number=int(line_str),
                        column_number=int(col_str),
                        error_code="python",
                        message=msg.strip(),
                        raw_line=m.group(0),
                    ))

        # 4. Try Go patterns
        if not diagnostics:
            for m in cls.GO_ERROR_RE.finditer(output):
                fpath, line_str, col_str, msg = m.group(1), m.group(2), m.group(3) or "0", m.group(4)
                diagnostics.append(CompilationDiagnostic(
                    file_path=fpath.strip(),
                    line_number=int(line_str),
                    column_number=int(col_str),
                    error_code="go",
                    message=msg.strip(),
                    raw_line=m.group(0),
                ))

        # 5. Try Rust patterns
        if not diagnostics:
            for m in cls.RUST_ERROR_RE.finditer(output):
                code, msg, fpath, line_str, col_str = m.group(1) or "rustc", m.group(2), m.group(3), m.group(4), m.group(5)
                diagnostics.append(CompilationDiagnostic(
                    file_path=fpath.strip(),
                    line_number=int(line_str),
                    column_number=int(col_str),
                    error_code=code,
                    message=msg.strip(),
                    raw_line=m.group(0),
                ))

        # 6. Try .NET / C# patterns
        if not diagnostics:
            for m in cls.DOTNET_ERROR_RE.finditer(output):
                fpath, line_str, col_str, code, msg = m.group(1), m.group(2), m.group(3) or "0", m.group(4) or "cs", m.group(5)
                diagnostics.append(CompilationDiagnostic(
                    file_path=fpath.strip(),
                    line_number=int(line_str),
                    column_number=int(col_str),
                    error_code=code,
                    message=msg.strip(),
                    raw_line=m.group(0),
                ))

        # 7. Fallback generic errors if nothing matched
        if not diagnostics:
            for m in cls.GENERIC_ERROR_RE.finditer(output):
                msg = m.group(1).strip()
                if not any(skip in msg.lower() for skip in ["to see the full stack trace", "re-run with"]):
                    diagnostics.append(CompilationDiagnostic(
                        file_path=service_name,
                        line_number=0,
                        column_number=0,
                        error_code="generic",
                        message=msg,
                        raw_line=m.group(0),
                    ))

        return CompilationResult(
            service_name=service_name,
            passed=False,
            exit_code=exit_code,
            diagnostics=diagnostics,
            raw_output=output,
        )

    @classmethod
    def record_and_verify(
        cls,
        service_name: str,
        exit_code: int,
        output: str,
        blackboard: Optional[DeductiveBlackboard] = None
    ) -> CompilationResult:
        """Parses output and immediately updates DeductiveBlackboard with compile facts or rejected assumptions."""
        res = cls.parse_compiler_output(service_name, exit_code, output)
        if blackboard:
            if res.passed:
                blackboard.record_fact(f"Service '{service_name}' compiled cleanly with 0 errors.")
            else:
                for d in res.diagnostics:
                    blackboard.record_rejection(
                        f"Compiler rejected edit in '{Path(d.file_path).name}' (line {d.line_number}): {d.message}"
                    )
        return res

    # ── Service Baseline Tracking & Delta Regression Gate ──────────────────
    # Keyed by (resolved workspace path | "", service_name) so two workspaces that both contain a
    # service with the same directory name (e.g. "api") never share a baseline. The "" workspace key
    # is the legacy/service-only namespace used by callers that never pass a workspace.
    _baselines: Dict[Tuple[str, str], CompilationResult] = {}

    @staticmethod
    def _key(service_name: str, workspace_path: Optional[str]) -> Tuple[str, str]:
        ws = str(Path(workspace_path).resolve()) if workspace_path else ""
        return (ws, service_name)

    @classmethod
    def record_baseline(cls, service_name: str, result: CompilationResult, workspace_path: Optional[str] = None) -> None:
        """Stores the initial pre-edit compilation state for a service and optionally saves to disk."""
        cls._baselines[cls._key(service_name, workspace_path)] = result
        if workspace_path:
            cls.save_baselines_to_disk(workspace_path)

    @classmethod
    def get_baseline(cls, service_name: str, workspace_path: Optional[str] = None) -> Optional[CompilationResult]:
        """Retrieves the recorded baseline compilation state for a service, loading from disk if needed.

        Lookup order: exact (workspace, service) key -> disk for that workspace -> legacy ("", service)
        -> backward-compatible service-only match (only when unambiguous or no workspace was given).
        """
        key = cls._key(service_name, workspace_path)
        if key in cls._baselines:
            return cls._baselines[key]
        if workspace_path:
            cls.load_baselines_from_disk(workspace_path)
            if key in cls._baselines:
                return cls._baselines[key]
        legacy = cls._baselines.get(("", service_name))
        if legacy is not None:
            return legacy
        candidates = [res for (ws, sname), res in cls._baselines.items() if sname == service_name]
        if candidates and (workspace_path is None or len(candidates) == 1):
            return candidates[0]
        return None

    @classmethod
    def save_baselines_to_disk(cls, workspace_path: str) -> Optional[str]:
        """Persists this workspace's baselines to disk in .nemoforge/compiler_baselines.json and .aviator/compiler_baselines.json.

        Only entries recorded for `workspace_path` (or without a workspace) are written; the on-disk
        JSON format is unchanged (keyed by service name)."""
        try:
            ws_key = str(Path(workspace_path).resolve())
            data = {}
            for (ws, sname), res in cls._baselines.items():
                if ws not in (ws_key, ""):
                    continue
                data[sname] = {
                    "service_name": res.service_name,
                    "passed": res.passed,
                    "exit_code": res.exit_code,
                    "raw_output": res.raw_output[:4000],
                    "diagnostics": [
                        {
                            "file_path": d.file_path,
                            "line_number": d.line_number,
                            "column_number": d.column_number,
                            "error_code": d.error_code,
                            "message": d.message,
                            "raw_line": d.raw_line,
                        }
                        for d in res.diagnostics
                    ],
                }
            for dirname in (".nemoforge", ".aviator"):
                target_dir = Path(workspace_path) / dirname
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / "compiler_baselines.json"
                target_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            return str(Path(workspace_path) / ".nemoforge" / "compiler_baselines.json")
        except Exception:
            return None

    @classmethod
    def load_baselines_from_disk(cls, workspace_path: str) -> bool:
        """Loads baselines from .nemoforge/compiler_baselines.json with fallback."""
        try:
            target_file = Path(workspace_path) / ".nemoforge" / "compiler_baselines.json"
            if not target_file.exists():
                target_file = Path(workspace_path) / ".aviator" / "compiler_baselines.json"
            if not target_file.exists():
                return False
            data = json.loads(target_file.read_text(encoding="utf-8"))
            ws_key = str(Path(workspace_path).resolve())
            for sname, item in data.items():
                diags = [
                    CompilationDiagnostic(
                        file_path=d["file_path"],
                        line_number=d["line_number"],
                        column_number=d.get("column_number", 0),
                        error_code=d.get("error_code", ""),
                        message=d["message"],
                        raw_line=d.get("raw_line", ""),
                    )
                    for d in item.get("diagnostics", [])
                ]
                cls._baselines[(ws_key, sname)] = CompilationResult(
                    service_name=sname,
                    passed=item.get("passed", False),
                    exit_code=item.get("exit_code", 0),
                    diagnostics=diags,
                    raw_output=item.get("raw_output", ""),
                )
            return True
        except Exception:
            return False

    @classmethod
    def clear_baselines(cls) -> None:
        """Clears all recorded baselines."""
        cls._baselines.clear()

    @classmethod
    def evaluate_mutation(cls, service_name: str, current_res: CompilationResult, workspace_path: Optional[str] = None) -> Dict[str, Any]:
        """Compares current compilation result against recorded baseline to differentiate
        between new regressions introduced by the edit and pre-existing repository errors.
        """
        baseline = cls.get_baseline(service_name, workspace_path)
        if current_res.passed:
            return {
                "passed": True,
                "is_regression": False,
                "new_diagnostics": [],
                "pre_existing_diagnostics": [],
                "fixed_diagnostics": baseline.diagnostics if (baseline and not baseline.passed) else [],
                "message": f"✅ Compilation SUCCESS on {service_name}: Clean compile with 0 errors.",
            }

        if not baseline or baseline.passed:
            # Baseline was clean or not recorded; all current errors are treated as regressions
            return {
                "passed": False,
                "is_regression": True,
                "new_diagnostics": current_res.diagnostics,
                "pre_existing_diagnostics": [],
                "fixed_diagnostics": [],
                "message": current_res.get_repair_instruction(),
            }

        # Compare diagnostics
        baseline_keys = {
            (Path(d.file_path).name, d.line_number, d.message[:40])
            for d in baseline.diagnostics
        }
        current_keys = {
            (Path(d.file_path).name, d.line_number, d.message[:40])
            for d in current_res.diagnostics
        }

        new_diags = [
            d for d in current_res.diagnostics
            if (Path(d.file_path).name, d.line_number, d.message[:40]) not in baseline_keys
        ]
        pre_existing = [
            d for d in current_res.diagnostics
            if (Path(d.file_path).name, d.line_number, d.message[:40]) in baseline_keys
        ]
        fixed = [
            d for d in baseline.diagnostics
            if (Path(d.file_path).name, d.line_number, d.message[:40]) not in current_keys
        ]

        if not new_diags:
            msg = (
                f"ℹ️ COMPILATION NOTICE on {service_name}: Build has {len(pre_existing)} pre-existing error(s), "
                f"but your modification introduced 0 new regressions."
            )
            return {
                "passed": False,
                "is_regression": False,
                "new_diagnostics": [],
                "pre_existing_diagnostics": pre_existing,
                "fixed_diagnostics": fixed,
                "message": msg,
            }

        lines = [
            f"❌ COMPILATION REGRESSION on {service_name} (Exit code {current_res.exit_code}):",
            f"Your modification introduced {len(new_diags)} new error(s) (plus {len(pre_existing)} pre-existing):",
        ]
        for d in new_diags[:5]:
            pos = f":{d.line_number}" if d.line_number > 0 else ""
            lines.append(f"  • [NEW] `{Path(d.file_path).name}{pos}`: {d.message}")
        lines.append("\n⚠️ ACTION REQUIRED: You must fix the new errors introduced by your edit.")
        return {
            "passed": False,
            "is_regression": True,
            "new_diagnostics": new_diags,
            "pre_existing_diagnostics": pre_existing,
            "fixed_diagnostics": fixed,
            "message": "\n".join(lines),
        }
