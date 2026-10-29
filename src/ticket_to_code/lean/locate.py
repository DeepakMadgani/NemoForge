"""Deterministic anchor-based localization — zero LLM calls.

A file qualifies only if it contains at least MIN_DISTINCT_ANCHORS of the ticket's exact
strings. Rarer anchors (lower document frequency) weigh more, so a unique header like
'Revision code' outranks a common word. All reads go through the ScopeFence, so a denied
file is never scored.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ticket_to_code.lean.scope import ScopeContract, ScopeFence

MIN_DISTINCT_ANCHORS = 2
# YAML/properties are configuration, not logic — they matched generic words and beat the real Java
# owner, so they are not scanned as candidate owners (a config-only ticket would name the file).
_CODE_EXT = {".ts", ".tsx", ".js", ".html", ".scss", ".css", ".java", ".sql", ".json", ".py"}
# i18n holds the on-screen label words (e.g. 'Role', 'Modified By') as data, so it matches every
# header anchor and would outrank the real component. It is never the owner of a UI-structure ticket.
# build/bin/out hold compiled copies of the same sources — never edit those.
_SKIP_DIRS = {"node_modules", "dist", "target", ".git", ".angular", "coverage", "__pycache__",
              "graphify-out", "i18n", "build", "bin", "out"}

# Structural signal: the file that renders the shared table (template) or declares its column
# model (component .ts). Anchors match translate-KEY substrings everywhere; this marker is what
_GRID_MARKERS = re.compile(r"<table\b|cdkColumnDef|matColumnDef|ColumnDef|\b\w*table\b|\b\w*grid\b", re.I)


@dataclass
class Candidate:
    path: str                      # absolute
    rel: str
    score: float
    distinct_anchors: int
    anchor_lines: Dict[str, List[int]]

    def all_lines(self) -> List[int]:
        seen = sorted({l for ls in self.anchor_lines.values() for l in ls})
        return seen


def _iter_files(fence: ScopeFence) -> List[Path]:
    root = Path(fence.scope.root)
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _CODE_EXT:
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if fence.allows(str(p)):
            out.append(p)
    return out


def _document_frequency(files: List[Path], anchors: Tuple[str, ...]) -> Dict[str, int]:
    df = {a: 0 for a in anchors}
    compiled = {a: re.compile(re.escape(a), re.I) for a in anchors}
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for a, pat in compiled.items():
            if pat.search(text):
                df[a] += 1
    return df


def locate(scope: ScopeContract, fence: Optional[ScopeFence] = None,
           max_candidates: int = 8) -> Tuple[List[Candidate], ScopeFence]:
    """Return anchor-matched candidate files, ranked, with zero LLM calls."""
    fence = fence or ScopeFence(scope)
    if not scope.anchors:
        return [], fence
    files = _iter_files(fence)
    total = max(len(files), 1)
    df = _document_frequency(files, scope.anchors)
    # idf-style weight: rarer anchors weigh more; an anchor in every file weighs ~0.
    weight = {a: math.log((total + 1) / (df.get(a, 0) + 1)) + 0.1 for a in scope.anchors}
    compiled = {a: re.compile(re.escape(a), re.I) for a in scope.anchors}

    cands: List[Candidate] = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        anchor_lines: Dict[str, List[int]] = {}
        score = 0.0
        for a, pat in compiled.items():
            hits = [i + 1 for i, line in enumerate(lines) if pat.search(line)]
            if hits:
                anchor_lines[a] = hits
                score += weight[a] * (1 + math.log(len(hits)))
                # Smoking-gun bonus: an anchor that occurs in at most two files repo-wide (a method
                # name, dotted key or filename named in the ticket) is a near-unique locator; reward it
                # strongly so it beats a file that merely repeats common words many times.
                if df.get(a, 0) <= 2:
                    score += weight[a] * 3.0
        distinct = len(anchor_lines)
        if distinct >= MIN_DISTINCT_ANCHORS:
            rel = fence.scope.relative(str(f)) or f.name
            # Structural affinity: the real owner of a GRID/COLUMN ticket renders the shared table or
            # declares its column model. Gated on grid_hint so it never inflates a grid component for a
            # non-grid ticket (e.g. a confirmation-dialog defect) that merely shares generic words.
            if getattr(scope, "grid_hint", False) and _GRID_MARKERS.search("\n".join(lines)):
                score *= 4.0
            # Path affinity: a file under the screen-noun feature folder is the intended screen.
            noun = scope.screen_noun
            if noun and noun in rel.lower():
                score *= 3.0
            # Test files match the same anchors as the code they exercise; deprioritize them so the
            # primary owner is the production source, not its test (the test may still be a candidate).
            low = rel.lower()
            if "/test/" in low or low.endswith((".spec.ts", "test.java", "tests.java", "spec.java")):
                score *= 0.3
            cands.append(Candidate(str(f), rel, round(score, 3), distinct, anchor_lines))

    # Rank by weighted score (rare, specific anchors dominate), then by breadth of distinct anchors.
    # Ranking by raw anchor COUNT first would let a file matching many generic words beat the real
    # owner that matches a few highly specific ones.
    cands.sort(key=lambda c: (c.score, c.distinct_anchors), reverse=True)
    return cands[:max_candidates], fence


def companions(candidate: Candidate) -> List[str]:
    """Sibling files that share the component stem (.ts/.html/.scss), so a locked edit is complete.
    Spec files are excluded — a presentation edit must not rewrite tests."""
    p = Path(candidate.path)
    stem = re.sub(r"\.(component\.)?(ts|html|scss|css)$", "", p.name)
    out: List[str] = []
    for sib in p.parent.iterdir():
        if (sib.is_file() and sib.name.startswith(stem)
                and sib.suffix.lower() in {".ts", ".html", ".scss", ".css"}
                and not sib.name.endswith(".spec.ts")):
            out.append(str(sib))
    return sorted(set(out))
