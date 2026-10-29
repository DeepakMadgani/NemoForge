"""
Diagnostic Normalizer — Structures raw compiler output into typed objects.

Parses deterministic compiler output formats from multiple languages into
StructuredDiagnostic objects. This module is ONLY responsible for parsing
the structured format of compiler output — it does NOT interpret semantics,
determine fix locations, or understand program relationships.

Separation of responsibilities:
  - DiagnosticNormalizer: "What did the compiler say?" (this module)
  - SymbolResolver:       "Where are those symbols defined?" (symbol_resolver.py)
  - FixLocalizer:         "Where might the fix be?"          (fix_localizer.py)
  - FixHypothesisBuilder: "What evidence supports each fix?" (fix_hypothesis_builder.py)

Supported compiler output formats:
  TypeScript:  file.ts(line,col): error TS2339: ...
  Angular AOT: file.html:line:col - error NG8002: ...
  Java/Maven:  [ERROR] File.java:[line,col] error: ...
  Java/Gradle: File.java:line: error: ...
  Python:      File.py:line: SyntaxError: ... / ImportError: ...
  C#:          File.cs(line,col): error CS1234: ...

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from ticket_to_code.utils.error_normalization import strip_ansi

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StructuredDiagnostic:
    """Language-agnostic normalized compiler diagnostic.

    The normalizer extracts ONLY what the compiler explicitly provides
    in its structured output format.

    Semantic understanding (which symbol is wrong, what invariant is broken,
    where the fix should go) is the job of downstream layers.
    """
    raw: str                     # Original compiler output line
    code: str                    # "TS2339", "NG8002", "CS1234", "" if unknown
    message: str                 # Human-readable error message
    source_file: str             # File where error was REPORTED (not necessarily fix location)
    line: int = 0                # Line number (0 if unknown)
    column: int = 0              # Column (0 if unknown)
    language: str = "unknown"    # "typescript", "java", "python", "csharp", "angular"
    severity: str = "error"      # "error", "warning"

    # Best-effort entity extraction.
    # The normalizer extracts what it CAN from well-known patterns,
    # but does NOT guarantee completeness. Downstream layers enrich these.
    extracted_entities: list[str] = field(default_factory=list)


# ── Compiler-specific parsers ─────────────────────────────────────────────────
# Each parser handles one compiler's output format.
# They extract ONLY what the compiler explicitly provides.

# TypeScript / Angular AOT
# Format: file.ts(line,col): error TS2339: Property 'X' does not exist on type 'Y'
# Format: file.html:line:col - error NG8002: Can't bind to 'ngModel'...
_TS_DIAG_RE = re.compile(
    r"(?P<file>[^\s(]+\.(?:ts|tsx|js|jsx|mts|cts))"
    r"\((?P<line>\d+),(?P<col>\d+)\)"
    r":\s*(?P<severity>error|warning)\s+"
    r"(?P<code>TS[A-Za-z0-9_]+)"
    r":\s*(?P<message>.+)",
    re.IGNORECASE,
)

_ANGULAR_DIAG_RE = re.compile(
    r"(?P<file>[^\s]+\.html)"
    r"[:（](?P<line>\d+)[,:](?P<col>\d+)[)）]?\s*"
    r"[-–]\s*(?P<severity>error|warning)\s+"
    r"(?P<code>NG\d+)"
    r":\s*(?P<message>.+)",
    re.IGNORECASE,
)

# Java / Maven
# Format: [ERROR] /path/File.java:[line,col] error: cannot find symbol
_JAVA_MAVEN_RE = re.compile(
    r"(?:\[ERROR\]\s+)?(?P<file>[^\s]+\.(?:java|kt|scala))"
    r":\[(?P<line>\d+),(?P<col>\d+)\]\s*"
    r"(?:error:\s*)?(?P<message>.+)",
    re.IGNORECASE,
)

# Java / Gradle
# Format: File.java:line: error: cannot find symbol
_JAVA_GRADLE_RE = re.compile(
    r"(?P<file>[^\s]+\.(?:java|kt|scala))"
    r":(?P<line>\d+):\s*"
    r"(?:error:\s*)?(?P<message>.+)",
    re.IGNORECASE,
)

# Python
# Format: File "file.py", line N / SyntaxError: ...
_PYTHON_FILE_LINE_RE = re.compile(
    r'File\s+"(?P<file>[^"]+\.py)"'
    r",\s*line\s+(?P<line>\d+)",
    re.IGNORECASE,
)

_PYTHON_ERROR_RE = re.compile(
    r"(?P<kind>SyntaxError|IndentationError|ImportError|ModuleNotFoundError|NameError|TypeError|AttributeError)"
    r":\s*(?P<message>.+)",
)

# C#
# Format: File.cs(line,col): error CS1234: ...
_CSHARP_DIAG_RE = re.compile(
    r"(?P<file>[^\s(]+\.cs)"
    r"\((?P<line>\d+),(?P<col>\d+)\)"
    r":\s*(?P<severity>error|warning)\s+"
    r"(?P<code>CS\d+)"
    r":\s*(?P<message>.+)",
    re.IGNORECASE,
)

# ── Entity extraction patterns (best-effort) ─────────────────────────────────
# These extract symbol names from WELL-KNOWN error message formats.
# They are NOT comprehensive — that's the job of downstream layers.

_ENTITY_PATTERNS = [
    # TS2339: Property 'X' does not exist on type 'Y'
    re.compile(r"Property\s+'(\w+)'\s+does not exist on type\s+'(\w+)'"),
    # TS2304: Cannot find name 'X'
    re.compile(r"Cannot find name\s+'(\w+)'"),
    # TS2345: Argument of type 'X' is not assignable to parameter of type 'Y'
    re.compile(r"Argument of type\s+'([^']+)'\s+is not assignable to parameter of type\s+'([^']+)'"),
    # TS2322: Type 'X' is not assignable to type 'Y'
    re.compile(r"Type\s+'([^']+)'\s+is not assignable to type\s+'([^']+)'"),
    # TS2554: Expected N arguments, but got M
    re.compile(r"Expected\s+(\d+)\s+arguments?,\s+but got\s+(\d+)"),
    # Java: cannot find symbol: method X() / variable X / class X
    re.compile(r"cannot find symbol.*?(?:method|variable|class)\s+(\w+)"),
    # Java: incompatible types
    re.compile(r"incompatible types.*?(\w+)"),
    # Python: cannot import name 'X' from 'Y'
    re.compile(r"cannot import name\s+'(\w+)'\s+from\s+'([^']+)'"),
    # Python: has no attribute 'X'
    re.compile(r"has no attribute\s+'(\w+)'"),
    # Python: No module named 'X'
    re.compile(r"No module named\s+'([^']+)'"),
    # C#: The type or namespace name 'X' could not be found
    re.compile(r"type or namespace name\s+'(\w+)'.*?could not be found"),
    # Angular: Can't bind to 'X' since it isn't a known property
    re.compile(r"Can't bind to\s+'(\w+)'"),
    # Angular: There is no directive with exportAs 'X'
    re.compile(r"no directive with\s+exportAs\s+'(\w+)'"),
]


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_diagnostics(raw_errors: list[str]) -> list[StructuredDiagnostic]:
    """Parse raw compiler output lines into structured diagnostics.

    Each input line is matched against known compiler output formats.
    Lines that don't match any known format are still included with
    reduced metadata (raw text + severity).

    Entity extraction is BEST-EFFORT — the normalizer extracts what it
    can from well-known patterns but does not guarantee completeness.

    Args:
        raw_errors: List of raw compiler output lines (may contain ANSI codes).

    Returns:
        List of StructuredDiagnostic objects, one per parseable error line.
    """
    diagnostics: list[StructuredDiagnostic] = []

    for raw_line in raw_errors:
        plain = strip_ansi(raw_line).strip()
        if not plain:
            continue

        diag = _try_parse(plain)
        if diag:
            # Best-effort entity extraction
            diag.extracted_entities = _extract_entities(diag.message)
            diagnostics.append(diag)
        else:
            # Unparseable line — still record it as a diagnostic with minimal info
            # Check if it looks like an error continuation or secondary line
            if _is_likely_error(plain):
                diagnostics.append(StructuredDiagnostic(
                    raw=plain,
                    code="",
                    message=plain,
                    source_file="",
                    severity="error",
                ))

    logger.info(
        f"  [DiagnosticNormalizer] Parsed {len(diagnostics)} diagnostics "
        f"from {len(raw_errors)} raw lines"
    )
    return diagnostics


def group_by_file(diagnostics: list[StructuredDiagnostic]) -> dict[str, list[StructuredDiagnostic]]:
    """Group diagnostics by their source file for efficient processing."""
    groups: dict[str, list[StructuredDiagnostic]] = {}
    for d in diagnostics:
        if d.source_file:
            groups.setdefault(d.source_file, []).append(d)
    return groups


def group_by_code(diagnostics: list[StructuredDiagnostic]) -> dict[str, list[StructuredDiagnostic]]:
    """Group diagnostics by error code (e.g., all TS2339 errors together)."""
    groups: dict[str, list[StructuredDiagnostic]] = {}
    for d in diagnostics:
        key = d.code or "UNKNOWN"
        groups.setdefault(key, []).append(d)
    return groups


def diagnostic_fingerprint(diagnostics: list[StructuredDiagnostic]) -> str:
    """Compute a stable identity fingerprint from structured diagnostics.

    Fingerprints on NORMALIZED diagnostic identity, NOT raw strings:
      - diagnostic code (TS2339, NG8002, CS1234)
      - normalized source file (basename only — ignores absolute paths)
      - extracted entities/symbols

    Deliberately IGNORES:
      - Line/column numbers (change with harmless code movement)
      - Absolute paths (vary by workspace location)
      - ANSI formatting codes (already stripped by normalize_diagnostics)
      - Timestamps and ordering

    Example fingerprint components:
      "TS2339|add-members.component.html|selectedUserIsExistingProjectMember"
    rather than:
      "C:\\workspace\\src\\app\\...\\add-members.component.html:25:31 - error TS2339: ..."

    The fingerprint is order-independent (sorted set) so that the same errors
    reported in different order produce the same fingerprint.

    Args:
        diagnostics: Normalized diagnostics from normalize_diagnostics().

    Returns:
        A stable string fingerprint. Empty string if no diagnostics.
    """
    if not diagnostics:
        return ""

    import hashlib
    import os

    components: set[str] = set()

    for d in diagnostics:
        # Build a stable identity from: code + basename + entities
        parts = []

        # Error code is the strongest identity signal
        if d.code:
            parts.append(d.code)

        # Use basename only — ignore directory structure which varies
        if d.source_file:
            basename = os.path.basename(d.source_file.replace("\\", "/"))
            parts.append(basename.lower())

        # Extracted entities (symbol names, types) are stable across edits
        if d.extracted_entities:
            for entity in sorted(d.extracted_entities):
                parts.append(entity.lower())

        # Fall back to a normalized message snippet if no code/entities
        if not parts and d.message:
            # Strip numbers and paths, keep only keyword tokens
            msg = re.sub(r"\d+", "", d.message).lower()
            msg = re.sub(r"[^a-z ]+", " ", msg)
            tokens = " ".join(w for w in msg.split() if len(w) > 2)
            if tokens:
                parts.append(tokens)

        if parts:
            components.add("|".join(parts))

    if not components:
        return ""

    # Sort for order-independence, then hash for compactness
    canonical = "\n".join(sorted(components))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]




# ── Internal parsing ──────────────────────────────────────────────────────────

def _normalize_diagnostic_path(path_str: str) -> str:
    if not path_str:
        return ""
    p = str(path_str).strip().replace("\\", "/")
    if p.startswith("/") and len(p) > 2 and p[1].isalpha() and p[2] == ":":
        p = p[1:]
    return p

def _try_parse(line: str) -> Optional[StructuredDiagnostic]:
    """Try each compiler-specific parser in priority order."""

    # TypeScript
    m = _TS_DIAG_RE.match(line)
    if m:
        return StructuredDiagnostic(
            raw=line,
            code=m.group("code"),
            message=m.group("message").strip(),
            source_file=_normalize_diagnostic_path(m.group("file")),
            line=int(m.group("line")),
            column=int(m.group("col")),
            language="typescript",
            severity=m.group("severity").lower(),
        )

    # Angular AOT
    m = _ANGULAR_DIAG_RE.match(line)
    if m:
        return StructuredDiagnostic(
            raw=line,
            code=m.group("code"),
            message=m.group("message").strip(),
            source_file=_normalize_diagnostic_path(m.group("file")),
            line=int(m.group("line")),
            column=int(m.group("col")),
            language="angular",
            severity=m.group("severity").lower(),
        )

    # C# (before Java, since .cs(line,col) is unambiguous)
    m = _CSHARP_DIAG_RE.match(line)
    if m:
        return StructuredDiagnostic(
            raw=line,
            code=m.group("code"),
            message=m.group("message").strip(),
            source_file=_normalize_diagnostic_path(m.group("file")),
            line=int(m.group("line")),
            column=int(m.group("col")),
            language="csharp",
            severity=m.group("severity").lower(),
        )

    # Java Maven
    m = _JAVA_MAVEN_RE.match(line)
    if m:
        return StructuredDiagnostic(
            raw=line,
            code="",
            message=m.group("message").strip(),
            source_file=_normalize_diagnostic_path(m.group("file")),
            line=int(m.group("line")),
            column=int(m.group("col")),
            language="java",
            severity="error",
        )

    # Java Gradle
    m = _JAVA_GRADLE_RE.match(line)
    if m:
        return StructuredDiagnostic(
            raw=line,
            code="",
            message=m.group("message").strip(),
            source_file=_normalize_diagnostic_path(m.group("file")),
            line=int(m.group("line")),
            language="java",
            severity="error",
        )

    # Python
    m = _PYTHON_FILE_LINE_RE.search(line)
    if m:
        # Try to find the error type in the same or adjacent text
        err_m = _PYTHON_ERROR_RE.search(line)
        msg = err_m.group("message") if err_m else line
        code = err_m.group("kind") if err_m else ""
        return StructuredDiagnostic(
            raw=line,
            code=code,
            message=msg.strip(),
            source_file=m.group("file"),
            line=int(m.group("line")),
            language="python",
            severity="error",
        )

    # Python error without file reference (e.g., standalone "ImportError: ...")
    m = _PYTHON_ERROR_RE.match(line)
    if m:
        return StructuredDiagnostic(
            raw=line,
            code=m.group("kind"),
            message=m.group("message").strip(),
            source_file="",
            language="python",
            severity="error",
        )

    # Generic compiler diagnostic fallback (handles file:line:col, file(line,col), and [CODE] syntax errors)
    generic_m = re.search(
        r"(?P<file>[^\s:]+\.(?:ts|tsx|js|jsx|mts|cts|java|kt|scala|py|html|scss|css|cs))"
        r"(?:"
        r"\((?P<line1>\d+)(?:,(?P<col1>\d+))?\)"
        r"|:\[(?P<line2>\d+)(?:,(?P<col2>\d+))?\]"
        r"|:(?P<line3>\d+)(?::(?P<col3>\d+))?"
        r"|:\s*\[(?P<code>[A-Za-z0-9_-]+)\]\s*(?:line\s+(?P<line4>\d+)(?::(?P<col4>\d+))?)?"
        r")"
        r"(?:[:\s]*(?P<severity>error|warning|fatal))?"
        r"[:\s]*(?P<message>.+)",
        line,
        re.IGNORECASE,
    )
    if generic_m:
        line_num = int(generic_m.group("line1") or generic_m.group("line2") or generic_m.group("line3") or generic_m.group("line4") or 0)
        col_num = int(generic_m.group("col1") or generic_m.group("col2") or generic_m.group("col3") or generic_m.group("col4") or 0)
        code_str = generic_m.group("code") or ""
        sev = (generic_m.group("severity") or "error").lower()
        fp = generic_m.group("file")
        ext = fp.rsplit(".", 1)[-1].lower()
        lang_map = {"ts": "typescript", "tsx": "typescript", "js": "javascript", "jsx": "javascript", "java": "java", "kt": "java", "py": "python", "cs": "csharp", "html": "angular"}
        return StructuredDiagnostic(
            raw=line,
            code=code_str,
            message=generic_m.group("message").strip(),
            source_file=fp,
            line=line_num,
            column=col_num,
            language=lang_map.get(ext, "unknown"),
            severity=sev,
        )

    return None


def _extract_entities(message: str) -> list[str]:
    """Best-effort extraction of symbol/type names from error messages.

    Uses known patterns to pull out identifiers. This is NOT comprehensive —
    downstream layers (SymbolResolver, FixLocalizer) enrich these.
    """
    entities: list[str] = []
    for pattern in _ENTITY_PATTERNS:
        for m in pattern.finditer(message):
            for group_val in m.groups():
                if group_val and group_val not in entities:
                    entities.append(group_val)
    return entities


def _is_likely_error(line: str) -> bool:
    """Heuristic: does this line look like a compiler error?"""
    lower = line.lower()
    return any(indicator in lower for indicator in (
        "error",
        "cannot find",
        "does not exist",
        "not assignable",
        "is not a",
        "unexpected",
        "failed to",
        "incompatible",
        "missing",
        "unresolved",
    ))
