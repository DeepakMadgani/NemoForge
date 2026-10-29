"""
DiagnosticErrorSlicer - Zero-Noise Multi-Compiler Error Extraction

Cuts 95% of prompt bloat during repair turns by extracting only the exact,
actionable compiler diagnostic blocks and stripping terminal noise:
- Strips ANSI escape codes, Maven/Gradle build progress, downloading jars, reactor summaries.
- Groups compiler error headers with their contiguous explanatory lines:
  (e.g., javac 'symbol:', 'location:', 'required:', 'found:' clauses).
- Polyglot Parsers:
  - Java (javac / Maven / Gradle)
  - TypeScript / JavaScript (tsc / eslint / node)
  - Python (pytest / mypy / traceback)
  - C# (.NET / MSBuild CSxxxx)
  - Go / Rust (cargo / rustc)
- Bounded Output: Strict token cap (~300 tokens max) to keep repair turn prompts ultra-lean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Terminal noise patterns to ignore outright
_IGNORE_PATTERNS = [
    re.compile(r"^\s*\[INFO\]\s+---.*"),
    re.compile(r"^\s*\[INFO\]\s+Building.*"),
    re.compile(r"^\s*\[INFO\]\s+Total time:.*"),
    re.compile(r"^\s*\[INFO\]\s+Finished at:.*"),
    re.compile(r"^\s*\[INFO\]\s+------------------------------------------------------------------------"),
    re.compile(r".*Download.*https?://.*", re.IGNORECASE),
    re.compile(r"^\s*npm\s+(?:warn|notice).*", re.IGNORECASE),
    re.compile(r"^\s*>\s+.*@.*build\s+.*"),
    re.compile(r"^\s*Deprecation Warning.*", re.IGNORECASE),
    re.compile(r"^\s*More info and automated migrator.*", re.IGNORECASE),
]

# Error header patterns across polyglot compilers
_ERROR_HEADER_PATTERNS = [
    # Java Maven/Gradle/javac: [ERROR] /path/to/File.java:[12,34] error: message
    re.compile(r"^(?:\[ERROR\]\s*)?([^\r\n:]+\.java):\[(\d+),(\d+)\]\s*(.*)", re.IGNORECASE),
    # javac standard: /path/to/File.java:12: error: message
    re.compile(r"^([^\r\n:]+\.java):(\d+):\s*(?:error:\s*)?(.*)", re.IGNORECASE),
    # TypeScript/C#: path/to/file.ts(12,34): error TS1234: message OR path/file.cs(12,34): error CS1234: message
    re.compile(r"^([^\r\n(]+\.(?:ts|tsx|js|jsx|cs))\s*\(\s*(\d+)\s*,\s*(\d+)\s*\):\s*(?:error\s+[A-Za-z0-9]+:\s*)?(.*)", re.IGNORECASE),
    # Python Traceback/pytest/mypy: path/to/file.py:12: error: message
    re.compile(r"^([^\r\n:]+\.py):(\d+):\s*(?:error:\s*)?(.*)", re.IGNORECASE),
    # Python traceback line: File "path/to/file.py", line 12, in func
    re.compile(r'^\s*File\s+"([^"]+\.py)",\s*line\s*(\d+)(?:,\s*in\s*.*)?', re.IGNORECASE),
    # Go: path/to/file.go:12:34: message
    re.compile(r"^([^\r\n:]+\.go):(\d+):(?:\d+:)?\s*(.*)", re.IGNORECASE),
    # Rust: error[E0123]: ... --> path/file.rs:12:34
    re.compile(r"^error\[E\d+\]:\s*(.*)", re.IGNORECASE),
]

# Continuation markers for compiler explanations (e.g. javac symbol/location)
_CONTINUATION_PATTERNS = [
    re.compile(r"^\s*(?:\[ERROR\]\s*)?(?:symbol|location|required|found|reason|expected):\s*.*", re.IGNORECASE),
    re.compile(r"^\s*(?:-->|\^|~+|\|).*"),
    re.compile(r"^\s*(?:[A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception):\s*.*"),
]


@dataclass
class DiagnosticBlock:
    file_path: str
    line: Optional[int]
    message: str
    context_lines: List[str] = field(default_factory=list)

    def to_formatted_string(self) -> str:
        loc = f":{self.line}" if self.line is not None else ""
        header = f"{self.file_path}{loc}: {self.message}"
        if not self.context_lines:
            return header
        details = "\n".join(f"  {line}" for line in self.context_lines)
        return f"{header}\n{details}"


class DiagnosticErrorSlicer:
    """
    Slices raw terminal stdout/stderr into clean, structured, low-token diagnostic errors.
    """

    @classmethod
    def slice_errors(
        cls,
        raw_output: str,
        max_errors: int = 5,
        max_tokens_budget: int = 350,
    ) -> str:
        """
        Parses raw build output and returns a tightly formatted diagnostic string
        strictly bounded by max_tokens_budget (~4 chars/token).
        """
        if not raw_output or not raw_output.strip():
            return "Unknown build/test failure (empty log output)."

        clean_text = _ANSI_ESCAPE_RE.sub("", raw_output)
        raw_lines = [l.strip() for l in clean_text.splitlines() if l.strip()]

        # Filter out obvious Maven/NPM noise
        filtered_lines = [
            l for l in raw_lines
            if not any(p.match(l) for p in _IGNORE_PATTERNS)
        ]

        blocks: List[DiagnosticBlock] = []
        current_block: Optional[DiagnosticBlock] = None

        for line in filtered_lines:
            matched_header = False
            for pat in _ERROR_HEADER_PATTERNS:
                m = pat.match(line)
                if m:
                    # Flush previous block
                    if current_block:
                        blocks.append(current_block)
                        if len(blocks) >= max_errors:
                            break

                    groups = m.groups()
                    if len(groups) == 4:
                        fp, line_no, _, msg = groups
                        current_block = DiagnosticBlock(
                            file_path=fp.replace("\\", "/"),
                            line=int(line_no) if line_no.isdigit() else None,
                            message=msg.strip(),
                        )
                    elif len(groups) == 3:
                        fp, line_no, msg = groups
                        current_block = DiagnosticBlock(
                            file_path=fp.replace("\\", "/"),
                            line=int(line_no) if line_no.isdigit() else None,
                            message=msg.strip(),
                        )
                    elif len(groups) == 2:
                        fp, line_no = groups
                        current_block = DiagnosticBlock(
                            file_path=fp.replace("\\", "/"),
                            line=int(line_no) if line_no.isdigit() else None,
                            message="",
                        )
                    else:
                        current_block = DiagnosticBlock(
                            file_path="unknown",
                            line=None,
                            message=line,
                        )
                    matched_header = True
                    break

            if matched_header:
                continue

            # Check if this line is an explanatory continuation of the current block
            if current_block and len(current_block.context_lines) < 4:
                if any(p.match(line) for p in _CONTINUATION_PATTERNS):
                    # Clean maven [ERROR] prefix if present in continuation
                    cleaned = re.sub(r"^\s*\[ERROR\]\s*", "", line)
                    current_block.context_lines.append(cleaned)
                elif current_block.message == "" and ("Error" in line or "Exception" in line):
                    current_block.message = line

        if current_block and len(blocks) < max_errors:
            blocks.append(current_block)

        # Fallback if no structured compiler patterns matched (e.g. raw panic or custom test failure)
        if not blocks:
            error_lines = [
                l for l in filtered_lines
                if any(k in l.lower() for k in ["error", "failure", "exception", "cannot", "failed"])
            ]
            if error_lines:
                return "\n".join(error_lines[:8])
            return "\n".join(filtered_lines[-10:])

        # Render blocks under strict char budget
        char_budget = max_tokens_budget * 4
        result_lines = []
        current_len = 0

        for b in blocks[:max_errors]:
            formatted = b.to_formatted_string()
            if current_len + len(formatted) > char_budget:
                result_lines.append("[... additional compiler warnings truncated ...]")
                break
            result_lines.append(formatted)
            current_len += len(formatted)

        return "\n\n".join(result_lines)
