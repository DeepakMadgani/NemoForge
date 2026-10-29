"""Live wiring for the lean path: a cached-prefix LLM patcher and a scoped verify ladder.

patch_fn  — one LLM call per attempt. The static prefix (role + output contract + framework
            facts + locked file windows + baselines) is byte-stable across attempts so it is
            cache-eligible; only the failure tail changes.
verify_fn — cheapest rung first: syntax → scoped ng build (optional) → component unit test
            (optional) → live Playwright over CDP. Stops at the first failing rung.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from ticket_to_code.lean.convergence import StructuredFailure
from ticket_to_code.lean.hunks import Hunk
from ticket_to_code.lean.lean_path import LockedContext, TokenMeter, VerifyOutcome
from ticket_to_code.lean.snippet import estimate_tokens, window_snippet
from ticket_to_code.lean.locate import Candidate

logger = logging.getLogger(__name__)

_PATCH_CONTRACT = (
    "You are a precise code editor. You change ONLY what the ticket asks.\n"
    "Return a single JSON object, no prose:\n"
    '{\"hunks\":[{\"file\":\"<relative path>\",\"search\":\"<exact text present once>\",'
    '\"replace\":\"<new text>\"}]}\n'
    "Rules: each search string MUST appear verbatim exactly once in that file; keep hunks minimal; "
    "never output a whole file; preserve surrounding code and indentation exactly.\n"
    "API HONESTY: use ONLY methods, fields, imports and commands that actually appear in the LOCKED FILES "
    "or the READ-ONLY API REFERENCE. NEVER invent or guess a method/field/command name just to make the "
    "code compile — a call that type-checks but does not exist at runtime is a defect. If the exact API "
    "needed to finish the ticket is NOT in the provided context, do NOT guess: output exactly "
    '{\"clarify\":\"<one precise question naming the method/service/decision you need>\"} and no hunks.\n'
    "FRAMEWORK FIDELITY: use existing injected services and patterns from the locked files. "
    "Do NOT introduce Angular Material (MatSnackBar, MatDialog) or this.data when the component uses @Input() config "
    "and custom services (ModalsService, NotificationService). For sagas, dispatch via this.sagaService.doCommand(...). "
    "In Java handlers, call setFieldForNextStep(key, val, state) on the handler base class taking state as an argument. "
    "Respect entity and repository types (UUID vs String).\n"
    "SCOPE: edit ONLY the specific columns/fields the ticket names. Do NOT add, remove, renumber, or "
    "reposition any other column; in particular, leave every unmentioned column's numeric cdkColumnDef id "
    "exactly as-is. For a column-order change, prefer the single smallest edit — usually reordering the "
    "entries in the component's columns array — and touch a cdkColumnDef id only for a column the ticket "
    "explicitly asks to move."
)


def _static_prefix(ctx: LockedContext, windows: dict) -> str:
    facts = "\n".join(f"- {f}" for f in ctx.framework_facts) or "- (none)"
    blocks = []
    for rel in ctx.files:
        win = windows.get(rel, "")
        blocks.append(f"=== {rel} ===\n{win}")
    deps = getattr(ctx, "dependency_context", "") or ""
    deps_section = (f"\n\nREAD-ONLY API REFERENCE (use these exact names and import paths; these files are "
                    f"NOT in the locked set and must NOT be edited):\n{deps}\n") if deps else ""
    return (
        f"{_PATCH_CONTRACT}\n\nFRAMEWORK FACTS (authoritative, do not contradict):\n{facts}\n"
        f"{deps_section}\n"
        f"LOCKED FILES (only these may change):\n" + "\n\n".join(blocks)
    )


def _parse_hunks(text: str) -> List[Hunk]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        # tolerate ```json fences
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
    out = []
    for h in data.get("hunks", []):
        if h.get("file") and h.get("search") is not None and h.get("replace") is not None:
            out.append(Hunk(file=h["file"], search=h["search"], replace=h["replace"]))
    return out


def make_patch_fn(ctx: LockedContext, candidates: List[Candidate], ticket_text: str,
                  meter: TokenMeter, llm=None, full_file_max_lines: int = 2500):
    """Build a patch_fn bound to the locked context. The static prefix (full file bytes for exact
    search/replace, windowed only for very large files) is computed once so it stays cache-stable."""
    anchor_lines = {c.rel: c.all_lines() for c in candidates}
    windows = {}
    for rel in ctx.files:
        raw = (Path(ctx.scope.root) / rel).read_text(encoding="utf-8", errors="ignore")
        if raw.count("\n") + 1 <= full_file_max_lines:
            # Exact bytes so the model's search text matches verbatim. Output is still hunks only.
            windows[rel] = raw
        else:
            lines = anchor_lines.get(rel) or [1]
            windows[rel] = window_snippet(str(Path(ctx.scope.root) / rel), rel, lines, numbered=False).text
    prefix = _static_prefix(ctx, windows)

    def patch_fn(_ctx: LockedContext, last_diff: str, failure: Optional[StructuredFailure]) -> List[Hunk]:
        from ticket_to_code.llm_utils import llm_invoke
        from langchain_core.messages import SystemMessage, HumanMessage
        from ticket_to_code.runtime.llm_factory import LLMRegistry

        model = llm or LLMRegistry.get_llm(assistant=True)
        tail = f"TICKET:\n{ticket_text}\n"
        if failure is not None:
            tail += (f"\nPREVIOUS ATTEMPT FAILED ({failure.kind}).\n"
                     f"Previous change: {last_diff}\n"
                     f"Failure detail: {failure.repair_hint()}\n"
                     f"Fix it with new hunks. Do not repeat the same change.\n")
        resp = llm_invoke(model, [SystemMessage(content=prefix), HumanMessage(content=tail)])
        content = getattr(resp, "content", str(resp))
        meter.add(estimate_tokens(prefix) + estimate_tokens(tail) + estimate_tokens(content),
                  calls=1, label="patch")
        # The model may ask for a missing API/decision rather than invent one that only compiles.
        mm = re.search(r"\{.*\}", content, re.DOTALL)
        if mm:
            try:
                data = json.loads(mm.group(0))
                if isinstance(data, dict) and data.get("clarify") and not data.get("hunks"):
                    from ticket_to_code.lean.lean_path import ClarifyRequest
                    raise ClarifyRequest(str(data["clarify"]))
            except json.JSONDecodeError:
                pass
        return _parse_hunks(content)

    return patch_fn


# ── verify ladder ─────────────────────────────────────────────────────────────

def _syntax_ok(root: str, rel: str) -> Optional[str]:
    """Cheapest rung: grammar-correct AST check via the existing TwoTierVerifier (tree-sitter for
    TS/JS, ast for Python); falls back to a brace/tag balance check for html/scss."""
    abs_path = Path(root) / rel
    ext = Path(rel).suffix.lower()
    if ext in (".ts", ".tsx", ".js", ".jsx", ".py"):
        try:
            from ticket_to_code.verification.two_tier_verifier import TwoTierVerifier
            res = TwoTierVerifier(root).run_tier1_fast_check([str(abs_path)])
            if not res.passed and res.diagnostics:
                d = res.diagnostics[0]
                return f"{getattr(d, 'code', 'syntax')}: {getattr(d, 'raw_message', str(d))}"
            return None
        except Exception:
            pass  # fall through to the structural check
    text = abs_path.read_text(encoding="utf-8", errors="ignore")
    if ext in (".ts", ".js", ".scss", ".css"):
        if text.count("{") != text.count("}"):
            return f"unbalanced braces in {rel}"
        if text.count("(") != text.count(")"):
            return f"unbalanced parentheses in {rel}"
    if ext == ".html":
        opens = len(re.findall(r"<ng-container", text))
        closes = len(re.findall(r"</ng-container>", text))
        if opens != closes:
            return f"unbalanced <ng-container> in {rel} ({opens} open / {closes} close)"
    return None


def make_verify_fn(cdp_url: str, page_path: str, expected_order: List[str],
                   ignore: Optional[List[str]] = None, run_build: bool = False,
                   build_cwd: Optional[str] = None):
    """Scoped ladder. By default build/unit rungs are skipped (syntax + live UI only)."""
    from ticket_to_code.verification.playwright_verifier import build_verifier_from_env

    def verify_fn(ctx: LockedContext, written: List[str]) -> VerifyOutcome:
        # Rung 1 — syntax (ms, 0 tokens)
        for rel in written:
            err = _syntax_ok(ctx.scope.root, rel)
            if err:
                return VerifyOutcome(ok=False, rung="syntax",
                                     failure=StructuredFailure(kind="compile", file=rel, actual=err))

        # Rung 2 — scoped build (optional; never Maven)
        if run_build and build_cwd:
            proc = subprocess.run(["npx", "ng", "build", "--configuration", "development"],
                                  cwd=build_cwd, capture_output=True, text=True, timeout=1200)
            if proc.returncode != 0:
                tail = (proc.stdout + proc.stderr)[-800:]
                return VerifyOutcome(ok=False, rung="build",
                                     failure=StructuredFailure(kind="compile", actual=tail))

        # Rung 4 — live UI over CDP (attach to the logged-in page)
        import asyncio

        async def _check():
            v = build_verifier_from_env()
            return await v.verify_dom_scenario(
                "about:blank", None,
                [{"type": "column_order", "selector": "table th", "expected": expected_order, "timeout_ms": 20000},
                 {"type": "no_console_errors"}],
                timeout_ms=30000,
                options={"cdp_url": cdp_url, "reuse_page": True, "reload": True,
                         "ignore": ignore or [],
                         "observe": [{"name": "headers", "selector": "table th", "kind": "texts"}]},
            )

        res = asyncio.run(_check())
        if res.infra_error:
            return VerifyOutcome(ok=False, rung="ui",
                                 failure=StructuredFailure(kind="infra", actual=res.summary))
        if res.passed:
            return VerifyOutcome(ok=True, rung="ui")
        f0 = res.failures[0]
        kind = "ui_order" if f0.get("type") == "column_order_failed" else "console"
        return VerifyOutcome(ok=False, rung="ui",
                             failure=StructuredFailure(kind=kind, expected=f0.get("expected"),
                                                       actual=f0.get("actual"),
                                                       hint="the live grid order is wrong; consult the FRAMEWORK FACTS for how this table orders columns"))

    return verify_fn


def make_compile_verify_fn(run_tier2: bool = True, target_service: str = ""):
    """Deterministic, browser-free verify ladder: Tier-1 AST syntax on changed files, then an
    optional Tier-2 scoped compile/build (reuses TwoTierVerifier). Reflects source edits, no DOM."""
    from ticket_to_code.verification.two_tier_verifier import TwoTierVerifier

    def verify_fn(ctx: LockedContext, written: List[str]) -> VerifyOutcome:
        root = ctx.scope.root
        # Rung 1 — grammar/AST syntax on the files we changed (ms, 0 tokens).
        for rel in written:
            err = _syntax_ok(root, rel)
            if err:
                return VerifyOutcome(ok=False, rung="syntax",
                                     failure=StructuredFailure(kind="compile", file=rel, actual=err))
        # Rung 1b — Database migration verification for .sql files
        from ticket_to_code.lean.db_migration import verify_db_migration
        sql_ok, sql_failure = verify_db_migration(Path(root), written)
        if not sql_ok and sql_failure:
            return VerifyOutcome(ok=False, rung="db_migration", failure=sql_failure)

        if not run_tier2:
            return VerifyOutcome(ok=True, rung="syntax")
        # Rung 2 — prefer the project's own `npm run build`: it excludes .spec files and is the real
        # deploy gate, so a pre-existing error in an unrelated test file can't block an unrelated edit.
        import json as _json
        pkg = Path(root) / "package.json"
        has_npm_build = False
        if pkg.is_file():
            try:
                scripts = _json.loads(pkg.read_text(encoding="utf-8", errors="ignore")).get("scripts") or {}
                has_npm_build = "build" in scripts
            except Exception:
                has_npm_build = False
        if has_npm_build:
            import re as _re
            import subprocess
            try:
                proc = subprocess.run("npm run build", cwd=root, capture_output=True, text=True,
                                      shell=True, timeout=1800)
            except subprocess.TimeoutExpired:
                return VerifyOutcome(ok=False, rung="build",
                                     failure=StructuredFailure(kind="infra", actual="npm run build timed out"))
            if proc.returncode != 0:
                tail = (proc.stderr or "") + (proc.stdout or "")
                errs = _re.findall(r"(?mi)^.*\berror\b.*$", tail)
                msg = ("\n".join(errs[:3]) or tail[-600:])[:600]
                return VerifyOutcome(ok=False, rung="build",
                                     failure=StructuredFailure(kind="compile", actual=msg))
            # Rung 3 (Behavioral Invariants) — ensure mandated runtime methods are actually invoked
            from ticket_to_code.lean.behavioral_verify import extract_required_calls_from_ticket, verify_ast_behavior, BehavioralContract
            ticket_text = getattr(ctx.scope, "ticket_text", "") or ""
            req_calls = extract_required_calls_from_ticket(ticket_text)
            if req_calls:
                contract = BehavioralContract(required_calls=req_calls)
                for rel in written:
                    abs_p = Path(root) / rel
                    lang = "java" if rel.endswith(".java") else "ts" if rel.endswith(".ts") else ""
                    if lang and abs_p.is_file():
                        ok, err = verify_ast_behavior(abs_p.read_text(encoding="utf-8", errors="ignore"), lang, contract)
                        if not ok:
                            return VerifyOutcome(ok=False, rung="behavioral",
                                                 failure=StructuredFailure(kind="runtime_behavior", file=rel, actual=err,
                                                                           hint=f"Runtime behavior error: {err}"))
            return VerifyOutcome(ok=True, rung="build")
        # Rung 2 (gradle) — a Java/Gradle service is gated by `gradlew test`: it compiles main + test
        # and runs the suite, so both compile errors and failing assertions drive the repair loop.
        gradlew = Path(root) / ("gradlew.bat" if os.name == "nt" else "gradlew")
        if gradlew.is_file():
            import re as _re
            import subprocess
            cmd = (f'"{gradlew}" test --console=plain' if os.name == "nt"
                   else f'"{gradlew}" test --console=plain')
            try:
                proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                                      shell=True, timeout=1800)
            except subprocess.TimeoutExpired:
                return VerifyOutcome(ok=False, rung="build",
                                     failure=StructuredFailure(kind="infra", actual="gradle test timed out"))
            if proc.returncode != 0:
                tail = (proc.stdout or "") + (proc.stderr or "")
                errs = _re.findall(r"(?mi)^.*(?:error:|\bFAILED\b|\w+Exception|expected:.*but was).*$", tail)
                msg = ("\n".join(errs[:6]) or tail[-700:])[:700]
                return VerifyOutcome(ok=False, rung="build",
                                     failure=StructuredFailure(kind="compile", actual=msg))
            return VerifyOutcome(ok=True, rung="build")
        # Fallback — non-npm projects: scoped compile/build via TwoTierVerifier.
        ttv = TwoTierVerifier(root)
        abs_written = [str(Path(root) / r) for r in written]
        t1 = ttv.run_tier1_fast_check(abs_written)
        if not t1.passed and t1.diagnostics:
            d = t1.diagnostics[0]
            return VerifyOutcome(ok=False, rung="syntax",
                                 failure=StructuredFailure(kind="compile",
                                                           actual=f"{getattr(d,'code','')}: {getattr(d,'raw_message','')}"))
        t2 = ttv.run_tier2_scoped_build(target_service)
        if not t2.passed:
            d0 = t2.diagnostics[0] if t2.diagnostics else None
            dmsg = (getattr(d0, "raw_message", None) or getattr(d0, "message", None) or str(d0)) if d0 else ""
            msg = (dmsg or (t2.stderr or "") or (t2.stdout or ""))[:500]
            return VerifyOutcome(ok=False, rung="build",
                                 failure=StructuredFailure(kind="compile", actual=msg))
        return VerifyOutcome(ok=True, rung="build")

    return verify_fn


def make_firefox_verify_fn(url: str, expected_order: List[str], profile_dir: str,
                          ignore: Optional[List[str]] = None, rebuild_wait_s: float = 18.0,
                          infra_retries: int = 3):
    """Verify ladder using the Firefox persistent-session verifier (auto-login, no CDP).
    Rung 1 syntax, then a fresh Firefox navigation that picks up the ng-serve rebuild.
    An 'unavailable' (grid did not render) result is RETRIED — it is an environment hiccup,
    never a code defect, and must not pollute the convergence tracker."""
    import time
    from ticket_to_code.lean.firefox_session import verify_order

    def verify_fn(ctx: LockedContext, written: List[str]) -> VerifyOutcome:
        for rel in written:
            err = _syntax_ok(ctx.scope.root, rel)
            if err:
                return VerifyOutcome(ok=False, rung="syntax",
                                     failure=StructuredFailure(kind="compile", file=rel, actual=err))
        import asyncio
        res = {}
        for attempt in range(infra_retries):
            time.sleep(rebuild_wait_s if attempt == 0 else 10.0)  # let ng serve rebuild, then settle
            res = asyncio.run(verify_order(url, expected_order, profile_dir, ignore=ignore))
            if res.get("status") != "unavailable":
                break  # got a real read (passed or ui_order)
        status = res.get("status")
        if status == "unavailable":
            return VerifyOutcome(ok=False, rung="ui",
                                 failure=StructuredFailure(kind="infra", actual=res.get("reason")))
        if status == "passed":
            return VerifyOutcome(ok=True, rung="ui")
        return VerifyOutcome(ok=False, rung="ui",
                             failure=StructuredFailure(kind="ui_order", expected=res.get("expected"),
                                                       actual=res.get("headers"),
                                                       hint=(f"the live grid header order is {res.get('headers')}, but it "
                                                             f"must read {res.get('expected')}. Produce a NEW, different edit "
                                                             "that reorders the columns to match exactly. Change ONLY the "
                                                             "columns the ticket names; do not renumber any other column's "
                                                             "cdkColumnDef id. Consult the FRAMEWORK FACTS for how this table "
                                                             "orders columns.")))

    return verify_fn


def make_ui_order_verify_fn(url: str, expected_order: List[str], profile_dir: str,
                            ignore: Optional[List[str]] = None, do_build: bool = True,
                            build_timeout_s: int = 1200, infra_retries: int = 3):
    """Default verifier for UI grid tickets: syntax gate → scoped build → live header-order read.

    Compile-only verification cannot see column ORDER, so a wrong order would pass. This verifier
    makes the live order the authoritative gate: it builds the sub-project (so the served bundle
    reflects the edit) and then reads the real grid headers. A wrong order is a hard ui_order
    failure that drives the repair loop; a non-rendering grid is retried as an environment hiccup.
    """
    import asyncio
    import subprocess
    import time
    from ticket_to_code.lean.firefox_session import verify_order

    def verify_fn(ctx: LockedContext, written: List[str]) -> VerifyOutcome:
        root = ctx.scope.root
        for rel in written:
            err = _syntax_ok(root, rel)
            if err:
                return VerifyOutcome(ok=False, rung="syntax",
                                     failure=StructuredFailure(kind="compile", file=rel, actual=err))
        if do_build:
            try:
                proc = subprocess.run("npm run build", cwd=root, capture_output=True, text=True,
                                      shell=True, timeout=build_timeout_s)
            except subprocess.TimeoutExpired:
                return VerifyOutcome(ok=False, rung="build",
                                     failure=StructuredFailure(kind="infra", actual="build timed out"))
            if proc.returncode != 0:
                tail = ((proc.stderr or "") + (proc.stdout or ""))[-600:]
                return VerifyOutcome(ok=False, rung="build",
                                     failure=StructuredFailure(kind="compile", actual=tail))
        res = {}
        for attempt in range(infra_retries):
            time.sleep(2.0 if attempt == 0 else 8.0)
            res = asyncio.run(verify_order(url, expected_order, profile_dir, ignore=ignore))
            if res.get("status") != "unavailable":
                break
        status = res.get("status")
        if status == "unavailable":
            return VerifyOutcome(ok=False, rung="ui",
                                 failure=StructuredFailure(kind="infra", actual=res.get("reason")))
        if status == "passed":
            return VerifyOutcome(ok=True, rung="ui")
        return VerifyOutcome(ok=False, rung="ui",
                             failure=StructuredFailure(kind="ui_order", expected=res.get("expected"),
                                                       actual=res.get("headers"),
                                                       hint=(f"the live grid header order is {res.get('headers')}, but it "
                                                             f"must read {res.get('expected')}. Produce a NEW, different edit "
                                                             "that reorders the columns to match exactly. Change ONLY the "
                                                             "columns the ticket names; do not renumber any other column's "
                                                             "cdkColumnDef id. Consult the FRAMEWORK FACTS for how this table "
                                                             "orders columns.")))

    return verify_fn
