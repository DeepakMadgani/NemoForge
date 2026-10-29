"""Surgical anchored hunks: search/replace only, no full-file rewrites.

Each hunk's search text must match exactly once in the target, or the hunk is rejected
before any file is written. This bounds patch tokens to O(hunk) by construction and makes
"model rewrote 1,700 lines" impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

FULL_REWRITE_LINE_LIMIT = 200  # files larger than this may ONLY be edited via hunks


@dataclass
class Hunk:
    file: str          # relative path
    search: str
    replace: str


@dataclass
class ApplyResult:
    ok: bool
    applied: List[Hunk] = field(default_factory=list)
    rejected: List[Tuple[Hunk, str]] = field(default_factory=list)
    new_contents: Dict[str, str] = field(default_factory=dict)   # rel -> content (not yet written)


def _match_count(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    count = start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return count
        count += 1
        start = idx + len(needle)


def _ws_tolerant_span(content: str, search: str):
    """Fallback match ignoring per-line leading/trailing whitespace. The model often copies a block
    correctly but with different indentation; this locates a UNIQUE contiguous run of lines whose
    stripped text equals the search's stripped lines. Returns (start_line, end_line_exclusive) or None."""
    c_lines = content.split("\n")
    s_lines = search.split("\n")
    while s_lines and s_lines[0].strip() == "":
        s_lines.pop(0)
    while s_lines and s_lines[-1].strip() == "":
        s_lines.pop()
    if not s_lines:
        return None
    s_norm = [l.strip() for l in s_lines]
    m = len(s_norm)
    spans = [i for i in range(0, len(c_lines) - m + 1)
             if [c_lines[i + j].strip() for j in range(m)] == s_norm]
    return (spans[0], spans[0] + m) if len(spans) == 1 else None


def plan_hunks(root: str, hunks: List[Hunk], allowed_files: "set[str] | None" = None) -> ApplyResult:
    """Validate hunks against current disk content. Returns new contents without writing them.

    Matching is exact first; if that misses, a whitespace-tolerant line match is tried so a correct
    edit is not rejected purely over indentation. Ambiguous (multi-match) hunks are still rejected.
    If allowed_files is given, a hunk targeting any file outside the locked set is rejected — the
    lock is a one-way door, so the patcher can never escape it to edit an unrelated file."""
    result = ApplyResult(ok=True)
    working: Dict[str, str] = {}
    for h in hunks:
        if allowed_files is not None and h.file not in allowed_files:
            result.rejected.append((h, "file is outside the locked set — edit only the locked files"))
            result.ok = False
            continue
        abs_path = Path(root) / h.file
        if h.file not in working:
            if not abs_path.is_file():
                result.rejected.append((h, "file does not exist"))
                result.ok = False
                continue
            working[h.file] = abs_path.read_text(encoding="utf-8", errors="ignore")
        content = working[h.file]
        hits = _match_count(content, h.search)
        if hits == 1:
            working[h.file] = content.replace(h.search, h.replace, 1)
            result.applied.append(h)
            continue
        if hits > 1:
            result.rejected.append((h, f"search text is ambiguous ({hits} matches) — add context"))
            result.ok = False
            continue
        # hits == 0: try the whitespace-tolerant line fallback before rejecting.
        span = _ws_tolerant_span(content, h.search)
        if span is not None:
            c_lines = content.split("\n")
            new_lines = c_lines[:span[0]] + h.replace.split("\n") + c_lines[span[1]:]
            working[h.file] = "\n".join(new_lines)
            result.applied.append(h)
            continue
        result.rejected.append((h, "search text not found"))
        result.ok = False
    result.new_contents = working
    return result


def write_result(root: str, result: ApplyResult) -> List[str]:
    """Persist planned contents to disk. Returns the relative paths written."""
    written: List[str] = []
    for rel, content in result.new_contents.items():
        (Path(root) / rel).write_text(content, encoding="utf-8")
        written.append(rel)
    return written
