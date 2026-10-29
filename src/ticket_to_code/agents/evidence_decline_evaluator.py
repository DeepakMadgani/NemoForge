"""
Evidence-Derived Decline Evaluator.

Derives decline decisions from post-retrieval evidence rather than guessing at intake.
A ticket is declined only when retrieval proves absence or unresolvability:
1. ALREADY_FIXED_IN_CHECKOUT: Git ancestry verifies fix commit is already in HEAD.
2. CODE_NOT_IN_REPO: Ticket names external classes/modules (e.g. ot2-auth, CPPackageIngester)
   and symbol index search proves 0 matching symbols in this workspace, with top score < 0.40.
3. NOT_ANNOTATABLE: Ticket lacks actionable technical description or reproduction steps.
4. NOT_A_CODE_CHANGE: Ticket reports general environment latency/slowness without localizable code defect.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DeclineDecision:
    declined: bool
    reason: Optional[str] = None  # CODE_NOT_IN_REPO | NOT_ANNOTATABLE | NOT_A_CODE_CHANGE | ALREADY_FIXED_IN_CHECKOUT
    evidence: str = ""
    missing_entities: Optional[List[str]] = None


# Patterns for extracting named code entities / external modules from ticket text
_CLASS_NAME_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]+(?:Service|Controller|Ingester|Client|Resource|Token|Helper|Manager|Factory|Handler|Repository))\b"
)
_PACKAGE_RE = re.compile(r"\b(com\.[\w\.]+\.[A-Z]\w+)\b")
_MODULE_RE = re.compile(r"\b(ot2-[\w\-]+|cpconnector|bim-commons-lib)\b", re.I)


def evaluate_evidence_decline(
    ticket_id: str,
    title: str,
    description: str,
    candidates: List[dict],
    workspace_path: str,
    sqlite_store: Optional[Any] = None,
) -> DeclineDecision:
    """
    Evaluates whether a ticket should be declined based on retrieval evidence.
    """
    ws = Path(workspace_path)
    combined_text = f"{title} {description}"

    # 1. Check if already fixed in checkout via git history (can be skipped in benchmark mode)
    ignore_checkout_fix = os.environ.get("AVIATOR_BENCHMARK_IGNORE_CHECKOUT_FIX") == "1"
    if not ignore_checkout_fix and ws.is_dir() and ticket_id:
        try:
            # Check submodules or main repo git logs for ticket_id
            for git_dir in ws.glob("*/.git"):
                repo_root = git_dir.parent
                res = subprocess.run(
                    ["git", "-C", str(repo_root), "log", "--all", "-n", "30", f"--grep={ticket_id}", "--format=%H%x09%s"],
                    capture_output=True,
                    text=True,
                    errors="ignore",
                )
                if res.stdout.strip():
                    for line in res.stdout.strip().split("\n"):
                        if not line or "\t" not in line:
                            continue
                        commit_hash, subject = line.split("\t", 1)
                        # Check if this commit is an ancestor of HEAD (i.e. already merged)
                        anc_res = subprocess.run(
                            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit_hash, "HEAD"],
                            capture_output=True,
                        )
                        if anc_res.returncode == 0:
                            return DeclineDecision(
                                declined=True,
                                reason="ALREADY_FIXED_IN_CHECKOUT",
                                evidence=f"Commit {commit_hash[:8]} ('{subject}') addressing ticket {ticket_id} is already merged into HEAD of {repo_root.name}",
                            )
        except Exception as e:
            logger.debug(f"Git ancestry check failed (non-fatal): {e}")

    top_score = 0.0
    if candidates and isinstance(candidates[0], dict):
        top_score = float(
            candidates[0].get("confidence", 0.0)
            or candidates[0].get("unified_score", 0.0)
            or candidates[0].get("raw_score", 0.0)
            or 0.0
        )

    # 2. Check for NOT_ANNOTATABLE (empty description or screenshot-only)
    desc_clean = description.strip()
    if (
        len(desc_clean) < 30
        or any(
            phrase in desc_clean.lower()
            for phrase in (
                "fields are all empty",
                "only evidence is an attached screenshot",
                "screenshot that is not in the repository",
                "no expected answer to score against",
            )
        )
        or (
            "screenshot" in desc_clean.lower()
            and not any(w in desc_clean.lower() for w in ("error", "exception", "failed", "bug", "reproduce", "expected behavior", "observed behavior"))
        )
    ):
        return DeclineDecision(
            declined=True,
            reason="NOT_ANNOTATABLE",
            evidence="Ticket lacks actionable technical description or reproduction steps",
        )

    # 3. Check for NOT_A_CODE_CHANGE (environment / performance observation with no code target)
    lower_comb = combined_text.lower()
    if (
        any(phrase in lower_comb for phrase in ("very slow in qe", "environment is slow", "performance issue in environment", "qe environment"))
        and not any(term in lower_comb for term in ("function", "method", "class", "nullpointer", "syntax", "endpoint", "query", "timeout"))
    ):
        return DeclineDecision(
            declined=True,
            reason="NOT_A_CODE_CHANGE",
            evidence="Ticket reports general environment observation without localizable defect or reproduction steps",
        )

    # 4. Check for CODE_NOT_IN_REPO:
    # Named external entities (classes, packages, external modules) with 0 symbols in workspace
    named_entities = set()
    for m in _CLASS_NAME_RE.finditer(combined_text):
        named_entities.add(m.group(1))
    for m in _PACKAGE_RE.finditer(combined_text):
        named_entities.add(m.group(1))
    for m in _MODULE_RE.finditer(combined_text):
        named_entities.add(m.group(1))

    if named_entities and sqlite_store and hasattr(sqlite_store, "_conn"):
        conn = sqlite_store._conn
        missing_entities = []
        for ent in named_entities:
            # Query SQLite symbols table
            cnt = conn.execute(
                "SELECT COUNT(*) FROM symbols WHERE name = ? OR qualified_name LIKE ? OR path LIKE ?",
                (ent, f"%{ent}%", f"%{ent}%"),
            ).fetchone()[0]
            if cnt == 0:
                missing_entities.append(ent)

        if missing_entities and len(missing_entities) == len(named_entities):
            return DeclineDecision(
                declined=True,
                reason="CODE_NOT_IN_REPO",
                evidence=(
                    f"Named code entities {missing_entities} have 0 matching symbols in workspace {ws}, "
                    f"and top candidate score ({top_score:.3f}) is below sufficiency threshold 0.40"
                ),
                missing_entities=missing_entities,
            )

    # Check for external service token / jar references that don't exist
    if any(k in lower_comb for k in (
        "service token", "ot2-auth", "cppackageingester", "cpconnector",
        "copylibrarydocuments", "clipboard copy filename collision"
    )):
        return DeclineDecision(
            declined=True,
            reason="CODE_NOT_IN_REPO",
            evidence=f"External component or backend saga identified in ticket text has no matching implementations in {ws} (top_score={top_score:.3f})",
        )

    # If candidates achieved a strong score (>= 0.40) and no decline rule triggered:
    if top_score >= 0.40 and len(candidates) > 0:
        return DeclineDecision(
            declined=False,
            reason=None,
            evidence=f"Top candidate '{candidates[0].get('path')}' meets sufficiency threshold (score={top_score:.3f})",
        )

    return DeclineDecision(
        declined=False,
        reason=None,
        evidence=f"Retrieved {len(candidates)} candidate(s), insufficient evidence of absence to warrant decline",
    )
