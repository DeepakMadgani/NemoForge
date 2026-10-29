"""Bounded evidence windows and token estimation.

Relevance is judged on a window around each anchor hit, never the whole file. This is what
keeps per-file tokens O(window) instead of O(file) — a 1,571-line file contributes ~80 lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

WINDOW = 40  # lines on each side of an anchor hit
MAX_SNIPPET_LINES = 180  # hard cap so an anchor-dense file can never expand to the whole file


def estimate_tokens(text: str) -> int:
    """Rough GPT-style estimate (~4 chars/token); good enough for budget accounting."""
    return max(1, (len(text) + 3) // 4)


def _merge(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    spans = sorted(spans)
    out: List[Tuple[int, int]] = []
    for lo, hi in spans:
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _bounded_spans(anchor_lines: List[int], n: int, window: int, max_lines: int) -> List[Tuple[int, int]]:
    """Densest-first span selection, capped at max_lines. A dense file shrinks the window instead of
    returning the whole file."""
    if not anchor_lines:
        return [(1, min(n, 2 * window))]
    spans = _merge([(max(1, l - window), min(n, l + window)) for l in sorted(set(anchor_lines))])
    if sum(hi - lo + 1 for lo, hi in spans) <= max_lines:
        return spans
    # Too big: shrink the window until it fits, then take the most anchor-dense spans.
    counts = {}
    for l in anchor_lines:
        counts[l] = counts.get(l, 0) + 1
    for w in (20, 10, 5, 3):
        spans = _merge([(max(1, l - w), min(n, l + w)) for l in sorted(set(anchor_lines))])
        if sum(hi - lo + 1 for lo, hi in spans) <= max_lines:
            return spans
    # Still too big: keep highest-density spans until the budget is spent.
    density = sorted(spans, key=lambda s: -sum(1 for l in anchor_lines if s[0] <= l <= s[1]))
    chosen: List[Tuple[int, int]] = []
    used = 0
    for s in density:
        size = s[1] - s[0] + 1
        if used + size > max_lines:
            continue
        chosen.append(s)
        used += size
    return sorted(chosen) or [(anchor_lines[0], min(n, anchor_lines[0] + max_lines - 1))]



@dataclass
class Snippet:
    rel: str
    text: str
    line_count: int
    tokens: int


def window_snippet(path: str, rel: str, anchor_lines: List[int], window: int = WINDOW,
                   max_lines: int = MAX_SNIPPET_LINES, numbered: bool = True) -> Snippet:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    n = len(lines)
    spans = _bounded_spans(anchor_lines, n, window, max_lines)
    chunks: List[str] = []
    count = 0
    for lo, hi in spans:
        chunks.append(f"... lines {lo}-{hi} ...")
        for i in range(lo, hi + 1):
            # numbered windows help the model locate; raw windows let it copy exact search text.
            chunks.append(f"{i}: {lines[i - 1]}" if numbered else lines[i - 1])
            count += 1
    text = "\n".join(chunks)
    return Snippet(rel=rel, text=text, line_count=count, tokens=estimate_tokens(text))
