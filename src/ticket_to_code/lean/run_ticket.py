"""Run one ticket through the lean path with live LLM patching and CDP verification.

Example:
  python -m ticket_to_code.lean.run_ticket --ticket src/.../benchmark_cc4e_deliverables_easy.json \
      --workspace C:/CC4E --cdp http://127.0.0.1:9222 \
      --expected "status,info,name,issues,content,list,revision code"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from ticket_to_code.lean.lean_path import (
    LeanResult, LockedContext, TokenMeter, run_lean_path, should_use_lean,
)
from ticket_to_code.lean.scope import triage
from ticket_to_code.lean.locate import locate, companions
from ticket_to_code.lean.live import make_patch_fn, make_verify_fn


def _load_ticket(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = data["tickets"] if isinstance(data, dict) and "tickets" in data else data
    return rows[0] if isinstance(rows, list) else rows


def run(ticket_path: str, workspace: str, cdp: str, expected: List[str],
        page_path: str, ignore: List[str], hard_timeout_s: float = 600.0,
        use_firefox: bool = False, firefox_url: str = "", fast: bool = False) -> LeanResult:
    row = _load_ticket(ticket_path)
    tid = row.get("ticket_id", "LEAN-1")
    title = row.get("title", "")
    desc = row.get("description", "")
    explicit_root = row.get("repo_scope")
    meter = TokenMeter()

    scope = triage(tid, title, desc, workspace, explicit_root, row.get("labels"))
    if scope.layer == "fullstack":
        from ticket_to_code.lean.fullstack_pipeline import run_fullstack_pipeline
        print(f"[lean] fullstack ticket detected: delegating to fullstack two-stage pipeline")
        from ticket_to_code.lean.live import make_compile_verify_fn
        # Create default verify and patch functions for both stages
        be_verify = make_compile_verify_fn(run_tier2=not fast)
        fe_verify = make_compile_verify_fn(run_tier2=not fast)
        fs_res = run_fullstack_pipeline(
            ticket_id=tid, title=title, description=desc,
            workspace=workspace,
            backend_patch_fn=lambda c, d, f: [], # default
            backend_verify_fn=be_verify,
            frontend_patch_fn=lambda c, d, f: [],
            frontend_verify_fn=fe_verify,
            hard_timeout_s=hard_timeout_s
        )
        return LeanResult(
            status=fs_res.status,
            scope=scope,
            locked_files=(fs_res.backend_result.locked_files if fs_res.backend_result else []) +
                         (fs_res.frontend_result.locked_files if fs_res.frontend_result else []),
            trace=fs_res.trace,
            elapsed_s=fs_res.elapsed_s
        )

    if not should_use_lean(scope):
        print(f"[lean] declined: layer={scope.layer} conf={scope.confidence}; fall back to full graph")
        return LeanResult("out_of_scope", scope=scope, reason="declined by should_use_lean")

    cands, fence = locate(scope)
    if not cands:
        return LeanResult("no_candidates", scope=scope, reason="no anchor match")

    # Build the locked context identically to the orchestrator so patch_fn sees the same windows.
    from ticket_to_code.lean.lean_path import build_locked_context
    ctx = build_locked_context(scope, cands, fence)

    ticket_text = f"{title}\n{desc}"
    patch_fn = make_patch_fn(ctx, cands, ticket_text, meter)
    ui_url = firefox_url or row.get("ui_url", "")
    ui_expected = expected or [s.strip().lower() for s in row.get("expected_order", []) if s.strip()]
    if use_firefox:
        from ticket_to_code.lean.live import make_firefox_verify_fn
        from ticket_to_code.lean.firefox_session import DEFAULT_PROFILE
        verify_fn = make_firefox_verify_fn(ui_url or page_path, ui_expected, DEFAULT_PROFILE, ignore=ignore)
    elif scope.layer == "ui" and ui_url and ui_expected and not fast:
        # Default for UI grid tickets: build + live header-order read. Compile-only verification
        # cannot see column ORDER, so a wrong order would pass; this makes the live order the gate.
        from ticket_to_code.lean.live import make_ui_order_verify_fn
        from ticket_to_code.lean.firefox_session import DEFAULT_PROFILE
        verify_fn = make_ui_order_verify_fn(ui_url, ui_expected, DEFAULT_PROFILE, ignore=ignore)
    else:
        # Browser-free fallback: deterministic compile/build verification (reflects edits, no DOM).
        from ticket_to_code.lean.live import make_compile_verify_fn
        verify_fn = make_compile_verify_fn(run_tier2=not fast)

    # Delegate to the orchestrator (it re-locks identically and runs the convergent loop).
    t0 = time.time()
    result = run_lean_path(tid, title, desc, workspace, patch_fn, verify_fn,
                           explicit_root=explicit_root, labels=row.get("labels"),
                           hard_timeout_s=hard_timeout_s, meter=meter)
    result.elapsed_s = round(time.time() - t0, 1)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticket", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--cdp", default="http://127.0.0.1:9222")
    ap.add_argument("--expected", default="", help="comma-separated expected header order (UI/firefox mode)")
    ap.add_argument("--page-path", default="/")
    ap.add_argument("--ignore", default="/config,NG04001")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--firefox", action="store_true", help="verify via Firefox persistent session (auto-login) instead of CDP")
    ap.add_argument("--firefox-url", default="", help="full grid URL for Firefox verification")
    ap.add_argument("--fast", action="store_true", help="compile mode: Tier-1 AST syntax only, skip scoped build")
    args = ap.parse_args(argv)

    expected = [s.strip().lower() for s in args.expected.split(",") if s.strip()]
    ignore = [s.strip() for s in args.ignore.split(",") if s.strip()]
    res = run(args.ticket, args.workspace, args.cdp, expected, args.page_path, ignore, args.timeout,
              use_firefox=args.firefox, firefox_url=args.firefox_url, fast=args.fast)

    print("\n================ LEAN RESULT ================")
    print(f"status       : {res.status}")
    print(f"reason       : {res.reason}")
    print(f"locked files : {res.locked_files}")
    print(f"attempts     : {res.attempts}")
    print(f"llm_calls    : {res.llm_calls}")
    print(f"tokens(est)  : {res.tokens}")
    print(f"elapsed_s    : {res.elapsed_s}")
    print("trace:")
    for t in res.trace:
        print(f"  - {t}")
    if res.status == "needs_clarification":
        print("\n>>> HARNESS NEEDS CLARIFICATION (it stopped instead of guessing an API):")
        print(f"    {res.reason}")
    return 0 if res.status == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
