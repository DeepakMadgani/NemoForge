"""
Ticket-aware outcome verifiers for the CC4E reasoning engine.

Distinct from semantic validation ("did I edit the intended owner?"), these
verify the ACTUAL requested change is now true in the repository.

Pluggable by ticket type:
  - literal      : version bumps, renames, string replacements
  - configuration: config/yaml/properties changes
  - api          : endpoint renames / API changes
  - generic      : requirement/acceptance-criteria coverage (fallback)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_IGNORE_DIRS = {".git", "node_modules", "dist", "build", "target", "out", "trace",
                "logs", "coverage", ".venv", "venv"}


def _iter_files(workspace: Path, extensions: Optional[set] = None):
    for p in workspace.rglob("*"):
        try:
            if not p.is_file():
                continue
            if any(part in _IGNORE_DIRS for part in p.parts):
                continue
            if p.stat().st_size > 1_000_000:
                continue
            if extensions and p.suffix.lower() not in extensions:
                continue
            yield p
        except Exception:
            continue


def _scan(workspace: Path, old_literals: List[str], new_literals: List[str],
          extensions: Optional[set] = None) -> dict:
    old_hits = {lit: 0 for lit in old_literals[:15] if lit}
    new_hits = {lit: 0 for lit in new_literals[:15] if lit}
    checked = 0
    for p in _iter_files(workspace, extensions):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        checked += 1
        for lit in old_hits:
            if lit in text:
                old_hits[lit] += 1
        for lit in new_hits:
            if lit in text:
                new_hits[lit] += 1
    old_remaining = [l for l, c in old_hits.items() if c > 0]
    new_missing = [l for l, c in new_hits.items() if c == 0]
    return {
        "verified": (not old_remaining) and (not new_missing),
        "checked_files": checked,
        "old_literals_remaining": old_remaining,
        "new_literals_missing": new_missing,
    }


def verify_outcome(ctx: Any) -> dict:
    """Select and run the appropriate verifier for this ticket."""
    workspace = Path(ctx.workspace_path)
    investigation = getattr(ctx, "investigation", None)
    old_literals: List[str] = []
    new_literals: List[str] = []
    if investigation is not None:
        old_literals = [str(x).strip() for x in (getattr(investigation, "current_state_literals", []) or []) if str(x).strip()]
        new_literals = [str(x).strip() for x in (getattr(investigation, "desired_state_literals", []) or []) if str(x).strip()]

    ticket_type = (ctx.ticket_type or "").lower()

    if ticket_type in {"api_change", "api"}:
        routes_old = [l for l in old_literals if "/" in l]
        routes_new = [l for l in new_literals if "/" in l]
        if routes_old or routes_new:
            res = _scan(workspace, routes_old, routes_new)
            res.update(mode="api_verifier", reason=_reason(res))
            return res

    if ticket_type in {"configuration", "config", "dependency", "deployment"}:
        res = _scan(workspace, old_literals, new_literals,
                    extensions={".yml", ".yaml", ".properties", ".json", ".xml", ".env"})
        res.update(mode="configuration_verifier", reason=_reason(res))
        return res

    if old_literals or new_literals:
        res = _scan(workspace, old_literals, new_literals)
        res.update(mode="literal_verifier", reason=_reason(res))
        return res

    # Generic fallback: acceptance-criteria coverage against written content.
    return _generic(ctx)


def _reason(res: dict) -> str:
    parts = []
    if res.get("old_literals_remaining"):
        parts.append(f"old still present={res['old_literals_remaining'][:5]}")
    if res.get("new_literals_missing"):
        parts.append(f"new missing={res['new_literals_missing'][:5]}")
    return "; ".join(parts) if parts else "requested state observed"


def _generic(ctx: Any) -> dict:
    written = getattr(ctx, "written_files", {}) or {}
    ticket = ctx.ticket
    acceptance: List[str] = []
    acceptance += [str(x) for x in (getattr(ticket, "acceptance_criteria", []) or []) if str(x).strip()]
    corpus = "\n".join(written.values()).lower()
    matched = checked = 0
    for crit in acceptance[:10]:
        toks = [t.lower() for t in re.findall(r"[A-Za-z0-9_]{4,}", crit)[:6]]
        if not toks:
            continue
        checked += 1
        if any(t in corpus for t in toks):
            matched += 1
    coverage = (matched / checked) if checked else (1.0 if written else 0.0)
    verified = bool(written) and coverage >= 0.5
    return {
        "verified": verified,
        "mode": "generic_requirement_verifier",
        "coverage": round(coverage, 3),
        "reason": f"acceptance coverage {matched}/{checked}" if checked else
                  ("changes written" if written else "no changes written"),
    }
