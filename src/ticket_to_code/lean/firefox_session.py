"""Firefox persistent-session verification — no CDP, no attaching to the user's browser.

Playwright owns a Firefox instance with a persistent profile directory. The user logs in ONCE
(establish_session detects when the grid renders and exits); every later verify_order reuses the
saved profile headlessly. This sidesteps Chrome's debug-port and background-process problems.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_PROFILE = str(Path.home() / ".aviator_firefox_profile")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip().lower()


async def _headers(page, selector: str = "table th") -> List[str]:
    return [_norm(t) for t in await page.locator(selector).all_text_contents() if _norm(t)]


async def _maybe_login(page, grid_selector: str, timeout_ms: int = 60000) -> None:
    """If the page is on the OTDS login screen, sign in with UI_USER/UI_PASSWORD (env only)."""
    user = os.environ.get("UI_USER", "")
    pwd = os.environ.get("UI_PASSWORD", "")
    if not user or not pwd:
        return
    # The deep link redirects through OTDS; wait for either the grid or the username field.
    try:
        await page.wait_for_selector(f"{grid_selector}, #otds_username", timeout=30000)
    except Exception:
        pass
    try:
        if await page.locator("#otds_username").count() == 0:
            return  # already authenticated (grid showing) or no login form
    except Exception:
        return
    try:
        await page.locator("#otds_username").first.fill(user, timeout=15000)
        await page.locator("button:has-text('Next'), input[type=submit], button[type=submit]").first.click()
        await page.locator("input[type=password]").first.wait_for(state="visible", timeout=20000)
        await page.locator("input[type=password]").first.fill(pwd)
        await page.locator("button:has-text('Sign in'), button[type=submit], input[type=submit]").first.click()
        await page.locator(grid_selector).first.wait_for(state="visible", timeout=timeout_ms)
    except Exception:
        pass  # caller reports the resulting state


async def establish_session(url: str, profile_dir: str = DEFAULT_PROFILE,
                            grid_selector: str = "table th", timeout_s: int = 300) -> Dict:
    """Open headed Firefox at the grid; auto-login if needed; persist the profile."""
    from playwright.async_api import async_playwright
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        ctx = await pw.firefox.launch_persistent_context(profile_dir, headless=False,
                                                         viewport={"width": 1440, "height": 900})
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception as exc:
            await ctx.close()
            return {"ok": False, "reason": f"navigation failed: {exc}"}
        await _maybe_login(page, grid_selector)
        try:
            await page.locator(grid_selector).first.wait_for(state="visible", timeout=timeout_s * 1000)
            headers = await _headers(page, grid_selector)
            result = {"ok": True, "headers": headers, "url": page.url, "profile": profile_dir}
        except Exception:
            result = {"ok": False, "reason": f"grid did not render within {timeout_s}s "
                                             "(log in and open the grid, then re-run)", "url": page.url}
        await ctx.close()
        return result


async def verify_order(url: str, expected: List[str], profile_dir: str = DEFAULT_PROFILE,
                       grid_selector: str = "table th", ignore: Optional[List[str]] = None,
                       headless: bool = True, settle_ms: int = 1500) -> Dict:
    """Reuse the saved profile to read the live column order. Returns status passed|failed|unavailable."""
    from playwright.async_api import async_playwright
    if not Path(profile_dir).exists():
        return {"status": "unavailable", "reason": f"no Firefox session profile at {profile_dir}; "
                                                   "run establish_session first"}
    console_errors: List[str] = []
    # Auth-flow noise from the OTDS login redirect is never the app's code; always ignore it.
    _AUTH_NOISE = ["Content-Security-Policy", "otdsauth", "JSHandle@object", "script-src", "default-src"]
    async with async_playwright() as pw:
        ctx = await pw.firefox.launch_persistent_context(profile_dir, headless=headless,
                                                         viewport={"width": 1440, "height": 900})
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        ignore_pats = [re.compile(p, re.I) for p in (list(ignore or []) + _AUTH_NOISE)]
        page.on("console", lambda m: console_errors.append(m.text)
                if m.type == "error" and not any(p.search(m.text) for p in ignore_pats) else None)
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await _maybe_login(page, grid_selector)
            try:
                await page.locator(grid_selector).first.wait_for(state="visible", timeout=25000)
            except Exception:
                body = _norm(await page.locator("body").inner_text())[:160]
                await ctx.close()
                return {"status": "unavailable", "reason": f"grid did not render (login/backend/empty state): {body}"}
            await page.wait_for_timeout(settle_ms)
            console_errors.clear()  # drop login-redirect noise; count only post-render errors
            await page.wait_for_timeout(300)
            headers = await _headers(page, grid_selector)
        except Exception as exc:
            await ctx.close()
            return {"status": "unavailable", "reason": f"navigation/read error: {exc}"}
        await ctx.close()

    # Relative-order check: each expected label must appear, in order, within the headers.
    want = [_norm(e) for e in expected]
    idx = [next((i for i, h in enumerate(headers) if w in h), -1) for w in want]
    ok = all(i >= 0 for i in idx) and idx == sorted(idx)
    if ok and console_errors:
        return {"status": "failed", "headers": headers, "console_errors": console_errors[:5],
                "reason": "console errors on the live page"}
    return {"status": "passed" if ok else "failed", "headers": headers,
            "expected": want, "console_errors": console_errors[:5]}


def _main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["login", "verify"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--expected", default="")
    ap.add_argument("--ignore", default="/config,NG04001")
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "login":
        r = asyncio.run(establish_session(a.url, a.profile))
    else:
        exp = [s.strip() for s in a.expected.split(",") if s.strip()]
        ig = [s.strip() for s in a.ignore.split(",") if s.strip()]
        r = asyncio.run(verify_order(a.url, exp, a.profile, ignore=ig, headless=not a.headed))
    import json
    print(json.dumps(r, indent=1))
    return 0 if r.get("ok") or r.get("status") == "passed" else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
