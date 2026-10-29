"""
User Decision Gate — bounded human-in-the-loop for safety decisions.

Provides a single, transport-agnostic primitive for asking the user a bounded
question and falling back to a SAFE DEFAULT on timeout or when no interactive
channel is wired.

Design goals:
  * Never block a non-interactive run (benchmark/CI): when no provider is wired,
    return the safe default immediately.
  * Never hang: even a misbehaving provider is capped by ``timeout`` via a
    daemon thread; on timeout the safe default is returned.
  * Never silently authorize unrelated modifications: the caller decides the
    safe default (LEAVE for pre-existing errors).

The ``provider`` is any callable ``(payload: dict) -> Optional[str]`` that
returns the chosen option (case-insensitive) or None. It is supplied per-run via
the workflow transient store; the interactive backend wires a real provider,
the benchmark wires none.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DecisionProvider = Callable[[dict], Optional[str]]


def request_user_decision(
    provider: Optional[DecisionProvider],
    payload: dict,
    options: list[str],
    default: str,
    timeout: float = 60.0,
) -> str:
    """Ask the user to choose one of ``options``; return the choice or ``default``.

    Args:
        provider: callable that surfaces the payload and returns a choice, or
            None for non-interactive runs.
        payload: human-facing context (affected files, counts, diagnostics …).
        options: allowed choices (lowercased comparison).
        default: safe fallback used on timeout / no provider / invalid response.
        timeout: max seconds to wait for the provider.
    """
    opts = [o.strip().lower() for o in options]
    default = default.strip().lower()
    if provider is None:
        logger.info(
            f"  [decision] no interactive provider — using safe default '{default}'"
        )
        return default

    result: dict = {"choice": None}

    def _run() -> None:
        try:
            raw = provider({**payload, "options": opts, "default": default, "timeout": timeout})
            if isinstance(raw, str) and raw.strip().lower() in opts:
                result["choice"] = raw.strip().lower()
        except Exception as exc:  # provider must never crash the workflow
            logger.debug(f"  [decision] provider raised (ignored): {exc}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)

    choice = result["choice"] or default
    if result["choice"] is None:
        logger.info(
            f"  [decision] no response within {timeout:.0f}s — using safe default '{default}'"
        )
    else:
        logger.info(f"  [decision] user chose '{choice}'")
    return choice


def build_pre_existing_payload(report) -> dict:
    """Payload describing PRE_EXISTING build errors for the decision UI."""
    pre = report.pre_existing
    files: dict[str, int] = {}
    for d in pre:
        fp = d.file_path or "(unknown)"
        files[fp] = files.get(fp, 0) + 1
    return {
        "kind": "pre_existing_errors",
        "title": "Pre-existing build errors detected",
        "message": (
            "These build errors existed BEFORE this ticket and are unrelated "
            "unless evidence says otherwise. Choose how to proceed."
        ),
        "affected_files": [{"file": f, "error_count": c} for f, c in sorted(files.items())],
        "total_error_count": len(pre),
        "representative_diagnostics": [d.raw[:300] for d in pre[:5]],
        "options": ["fix", "leave", "stop"],
        "default": "leave",
    }


def build_import_payload(file_path: str, added_imports: list[str]) -> dict:
    """Payload describing deterministic, repository-verified imports to add."""
    return {
        "kind": "add_verified_imports",
        "title": "Add repository-verified imports?",
        "message": (
            f"{len(added_imports)} referenced symbol(s) exist in the repository "
            f"but are not imported in {file_path}. Add the verified import(s)?"
        ),
        "file": file_path,
        "imports": list(added_imports),
        "options": ["apply", "skip"],
        "default": "apply",
    }
