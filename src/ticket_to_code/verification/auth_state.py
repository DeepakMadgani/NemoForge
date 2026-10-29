"""Generic UI login helper: logs in once and saves a Playwright storage_state for later verification runs.

Credentials are read from arguments/env only and are never written to disk; the saved state holds session
cookies/tokens, so keep it outside the repository.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

_USER_SELECTORS = (
    "input[type=email]", "input[name*=user i]", "input[id*=user i]", "input[name*=email i]",
    "input[id*=email i]", "input[name*=login i]", "input[type=text]",
)
_SUBMIT_SELECTORS = (
    "button[type=submit]", "input[type=submit]", "button:has-text('Sign in')", "button:has-text('Log in')",
    "button:has-text('Login')", "button:has-text('Next')", "button:has-text('Continue')",
)


async def save_login_state(
    url: str, username: str, password: str, out_path: str, headless: bool = True, timeout_ms: int = 60000,
) -> str:
    """Returns the final URL after login; raises RuntimeError when no login form is usable."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        try:
            context = await browser.new_context(viewport={"width": 1440, "height": 900})
            page = await context.new_page()
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            await _fill_first(page, _USER_SELECTORS, username, timeout_ms)
            if await page.locator("input[type=password]:visible").count() == 0:
                await _click_first(page, _SUBMIT_SELECTORS)  # two-step login: username first
            await page.locator("input[type=password]:visible").first.fill(password, timeout=timeout_ms)
            await _click_first(page, _SUBMIT_SELECTORS)
            origin = "/".join(url.split("/")[:3])
            try:
                await page.wait_for_url(f"{origin}/**", timeout=30000)
            except Exception:
                text = " ".join((await page.locator("body").inner_text()).split())[:200]
                raise RuntimeError(f"login did not complete; page says: {text}")
            try:
                await page.wait_for_load_state("networkidle", timeout=20000)
            except Exception:
                pass
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            await context.storage_state(path=out_path)
            return page.url
        finally:
            await browser.close()


async def _fill_first(page, selectors, value: str, timeout_ms: int) -> None:
    for sel in selectors:
        loc = page.locator(f"{sel}:visible")
        try:
            await loc.first.wait_for(state="visible", timeout=min(timeout_ms, 15000))
            await loc.first.fill(value)
            return
        except Exception:
            continue
    raise RuntimeError("no visible username field found")


async def _click_first(page, selectors) -> None:
    for sel in selectors:
        loc = page.locator(f"{sel}:visible")
        if await loc.count():
            await loc.first.click()
            return
    await page.keyboard.press("Enter")


def main(argv: Optional[list] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m ticket_to_code.verification.auth_state <url> <out_state.json>  (env UI_USER, UI_PASSWORD)")
        return 2
    user, pwd = os.environ.get("UI_USER", ""), os.environ.get("UI_PASSWORD", "")
    if not user or not pwd:
        print("set UI_USER and UI_PASSWORD")
        return 2
    final = asyncio.run(save_login_state(args[0], user, pwd, args[1]))
    print(f"saved state to {args[1]}; landed on {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
