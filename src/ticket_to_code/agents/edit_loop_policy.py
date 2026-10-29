"""
Edit-Loop Policy — turn the refinement loop into a bounded safety net.

A single deterministic decision function the edit loop consults after each
compile check, so the loop is NOT eight LLM iterations by default:

  * clean compile                         → EXIT_CLEAN
  * unrelated file (not ours/planned)     → REJECT_UNRELATED
  * new architectural dependency          → RETURN_TO_PLANNING
  * same error seen before (repeat)       → STOP_REPEATED
  * otherwise (localized, first time)     → TARGETED_REPAIR (one attempt)

Repository discovery stays the responsibility of Evidence Phase 1–6; this policy
never triggers rediscovery — a genuine new dependency returns to planning.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum


class EditAction(str, Enum):
    EXIT_CLEAN = "exit_clean"
    TARGETED_REPAIR = "targeted_repair"
    STOP_REPEATED = "stop_repeated"
    RETURN_TO_PLANNING = "return_to_planning"
    REJECT_UNRELATED = "reject_unrelated"


def error_signature(errors: list[str]) -> str:
    """Stable signature of a set of compiler errors (order-independent)."""
    norm = sorted(
        re.sub(r"\s+", " ", re.sub(r"[:(]\d+[,)]?\d*[):]?", "", e or "")).strip().lower()
        for e in (errors or [])
    )
    return hashlib.sha1("\n".join(norm).encode("utf-8", "replace")).hexdigest()


def _norm(p: str) -> str:
    return (p or "").replace("\\", "/").lower()


def _owned(path: str, owned: set[str]) -> bool:
    n = _norm(path)
    for o in owned:
        o = _norm(o)
        if not o:
            continue
        if n == o or n.endswith(o) or o.endswith(n):
            return True
    return False


def decide_edit_action(
    compile_errors: list[str],
    target_file: str,
    owned_files: set[str],
    planned_files: set[str],
    prev_signatures: set[str],
    new_dependency: bool = False,
) -> EditAction:
    """Decide what the edit loop should do next (see module docstring)."""
    if not compile_errors:
        return EditAction.EXIT_CLEAN

    allowed = {_norm(p) for p in owned_files} | {_norm(p) for p in planned_files}
    if target_file and not _owned(target_file, allowed):
        return EditAction.REJECT_UNRELATED

    if new_dependency:
        return EditAction.RETURN_TO_PLANNING

    if error_signature(compile_errors) in prev_signatures:
        return EditAction.STOP_REPEATED

    return EditAction.TARGETED_REPAIR
