"""
Playwright live-UI verifier shared by the chat backend and the ticket-to-code pipeline.

Design rules:
  - Never pass silently: unknown assertion/action types and missing arguments are failures.
  - Separate infrastructure problems (server down, browser missing) from real UI failures
    via ``UIVerificationResult.infra_error`` so callers do not "repair" code for an env issue.
  - Wait on real conditions (selector/text polling, network idle) instead of fixed sleeps.
  - Always release the browser; optionally keep a screenshot + trace on failure.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Console noise that is never caused by generated code.
_IGNORED_CONSOLE_PATTERNS = (
    re.compile(r"favicon\.ico", re.IGNORECASE),
    re.compile(r"Download the (React|Angular) DevTools", re.IGNORECASE),
    re.compile(r"\[vite\]|\[webpack-dev-server\]|\[HMR\]", re.IGNORECASE),
)

_SUPPORTED_ASSERTIONS = (
    "element_visible", "element_hidden", "element_count", "text_contains", "text_equals",
    "element_disabled", "computed_style", "percentage_dimension", "attribute_equals",
    "url_contains", "no_console_errors", "no_network_errors", "column_order", "aria_contains",
)
_SUPPORTED_ACTIONS = (
    "click", "type", "press", "select", "hover", "upload_file", "wait",
    "wait_for_selector", "goto", "set_viewport",
)


@dataclass
class UIAssertion:
    assertion_type: str
    selector: str
    expected_value: Any = None
    description: str = ""


@dataclass
class UIVerificationResult:
    passed: bool
    summary: str
    target_url: str
    assertions_passed: int
    assertions_failed: int
    failures: List[Dict[str, Any]] = field(default_factory=list)
    dom_snapshot: Optional[str] = None
    console_logs: List[str] = field(default_factory=list)
    network_errors: List[str] = field(default_factory=list)
    soft_failures: List[Dict[str, Any]] = field(default_factory=list)
    observations: Dict[str, Any] = field(default_factory=dict)
    infra_error: bool = False
    screenshot_path: Optional[str] = None
    trace_path: Optional[str] = None
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "target_url": self.target_url,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
            "failures": self.failures,
            "dom_snapshot": self.dom_snapshot,
            "console_logs": self.console_logs,
            "network_errors": self.network_errors,
            "soft_failures": self.soft_failures,
            "observations": self.observations,
            "infra_error": self.infra_error,
            "screenshot_path": self.screenshot_path,
            "trace_path": self.trace_path,
            "duration_ms": self.duration_ms,
        }


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _is_console_error(line: str) -> bool:
    low = line.lower()
    if not ("[error]" in low or "[page_error]" in low):
        return False
    return not any(p.search(line) for p in _IGNORED_CONSOLE_PATTERNS)


async def _eventually(
    check: Callable[[], Awaitable[Tuple[bool, Any]]],
    timeout_ms: int,
    interval_ms: int = 150,
) -> Tuple[bool, Any]:
    """Poll ``check`` until it returns ok or the timeout expires; returns the last result."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_ms / 1000.0
    last: Tuple[bool, Any] = (False, None)
    while True:
        try:
            last = await check()
        except Exception as exc:  # transient DOM/navigation races are retried
            last = (False, f"{type(exc).__name__}: {exc}")
        if last[0] or loop.time() >= deadline:
            return last
        await asyncio.sleep(interval_ms / 1000.0)


class PlaywrightUIVerifier:
    def __init__(
        self,
        headless: bool = False,
        slow_mo: int = 500,
        browser_channel: str = "msedge",
        artifacts_dir: Optional[str] = None,
        viewport: Tuple[int, int] = (1280, 800),
        assertion_timeout_ms: int = 5000,
        overall_timeout_s: int = 120,
        ignore_patterns: Optional[List[str]] = None,
    ):
        self.headless = headless
        self.ignore_patterns = [re.compile(p, re.IGNORECASE) for p in (ignore_patterns or [])]
        self.slow_mo = slow_mo
        self.browser_channel = browser_channel
        self.artifacts_dir = artifacts_dir
        self.viewport = viewport
        self.assertion_timeout_ms = assertion_timeout_ms
        self.overall_timeout_s = overall_timeout_s

    # ------------------------------------------------------------------ public

    async def verify_dom_scenario(
        self,
        target_url: str,
        actions: Optional[List[Dict[str, Any]]] = None,
        assertions: Optional[List[Dict[str, Any]]] = None,
        timeout_ms: int = 30000,
        options: Optional[Dict[str, Any]] = None,
    ) -> UIVerificationResult:
        """options: mock_routes [{url, status, json|body, content_type, method}], storage_state (path),
        headers {name: value}, cdp_url (attach to a running, already-logged-in Chromium/Edge), observe [{name, selector, kind: texts|count|attrs, attribute}],
        preconditions [assertion specs; failing one returns an infra error instead of a UI failure]."""
        started = time.time()
        try:
            result = await asyncio.wait_for(
                self._run(target_url, actions or [], assertions or [], timeout_ms, options or {}),
                timeout=self.overall_timeout_s,
            )
        except asyncio.TimeoutError:
            result = self._infra(target_url, f"UI verification exceeded {self.overall_timeout_s}s overall timeout")
        except Exception as exc:
            logger.error("Playwright verification crashed: %s", exc, exc_info=True)
            result = self._infra(target_url, f"Playwright execution error: {exc or repr(exc)}")
        result.duration_ms = int((time.time() - started) * 1000)
        return result

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _infra(url: str, message: str, logs: Optional[List[str]] = None) -> UIVerificationResult:
        return UIVerificationResult(
            passed=False,
            summary=message,
            target_url=url,
            assertions_passed=0,
            assertions_failed=1,
            failures=[{"type": "infrastructure_error", "error": message}],
            console_logs=logs or [],
            infra_error=True,
        )

    async def _launch(self, pw: Any) -> Any:
        attempts: List[Optional[str]] = []
        for ch in (self.browser_channel or None, None, "msedge", "chrome"):
            if ch not in attempts:
                attempts.append(ch)
        last_err: Optional[Exception] = None
        for ch in attempts:
            kwargs: Dict[str, Any] = {"headless": self.headless, "slow_mo": self.slow_mo}
            if ch:
                kwargs["channel"] = ch
            try:
                return await pw.chromium.launch(**kwargs)
            except Exception as exc:
                last_err = exc
                logger.warning("Browser launch failed (channel=%s): %s", ch, exc)
        raise RuntimeError(f"No usable Chromium browser ({last_err}). Run: playwright install chromium")

    # -------------------------------------------------------------------- core

    async def _run(
        self,
        target_url: str,
        actions: List[Dict[str, Any]],
        assertions: List[Dict[str, Any]],
        timeout_ms: int,
        options: Dict[str, Any],
    ) -> UIVerificationResult:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return self._infra(target_url, "Playwright is not installed (pip install playwright)")

        console_logs: List[str] = []
        network_errors: List[str] = []
        failures: List[Dict[str, Any]] = []
        observations: Dict[str, Any] = {}
        cdp_url = options.get("cdp_url")
        passed = 0
        screenshot_path = trace_path = None
        dom_snippet = ""

        logger.info("Playwright UI verification: %s (headless=%s)", target_url, self.headless)

        async with async_playwright() as pw:
            try:
                browser = (
                    await pw.chromium.connect_over_cdp(cdp_url) if cdp_url else await self._launch(pw)
                )
            except Exception as exc:
                return self._infra(
                    target_url,
                    f"Could not attach to browser at {cdp_url}: {exc}" if cdp_url else str(exc),
                )

            context = None
            page = None
            reuse = False
            tracing = False
            try:
                ctx_kwargs: Dict[str, Any] = {"viewport": {"width": self.viewport[0], "height": self.viewport[1]}}
                state_path = options.get("storage_state")
                if state_path and Path(state_path).is_file():
                    ctx_kwargs["storage_state"] = state_path
                if options.get("headers"):
                    ctx_kwargs["extra_http_headers"] = dict(options["headers"])
                if cdp_url and browser.contexts:
                    # Reuse the user's authenticated context; mocks are skipped so their browser is never altered.
                    context = browser.contexts[0]
                else:
                    context = await browser.new_context(**ctx_kwargs)
                    for mock in options.get("mock_routes") or []:
                        await self._install_mock(context, mock)
                if self.artifacts_dir:
                    try:
                        await context.tracing.start(screenshots=True, snapshots=True)
                        tracing = True
                    except Exception as exc:
                        logger.debug("Tracing unavailable: %s", exc)
                reuse = bool(cdp_url and options.get("reuse_page") and context.pages)
                page = context.pages[0] if reuse else await context.new_page()
                def _on_console(m: Any) -> None:
                    loc = getattr(m, "location", {}) or {}
                    url = loc.get("url", "") if isinstance(loc, dict) else ""
                    msg = f"[{m.type}] {m.text} ({url})" if url else f"[{m.type}] {m.text}"
                    console_logs.append(msg)

                page.on("console", _on_console)
                page.on("pageerror", lambda e: console_logs.append(f"[PAGE_ERROR] {e}"))
                page.on(
                    "requestfailed",
                    lambda r: network_errors.append(f"FAILED {r.method} {r.url} ({r.failure})"),
                )
                page.on(
                    "response",
                    lambda r: network_errors.append(f"HTTP {r.status} {r.request.method} {r.url}")
                    if r.status >= 400 and "favicon" not in r.url else None,
                )

                # 1. Navigate
                try:
                    if reuse:
                        resp = None
                        if options.get("reload"):
                            resp = await page.reload(timeout=timeout_ms, wait_until="domcontentloaded")
                    else:
                        resp = await page.goto(target_url, timeout=timeout_ms, wait_until="domcontentloaded")
                except Exception as exc:
                    return self._infra(
                        target_url,
                        f"Failed to connect to application at {target_url}: {exc}. Ensure the dev server is running.",
                        console_logs,
                    )
                if resp is not None and resp.status >= 500:
                    return self._infra(target_url, f"Server error HTTP {resp.status} at {target_url}", console_logs)
                try:
                    await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 8000))
                except Exception:
                    pass  # long-polling/websocket apps never go idle; selector waits cover readiness

                # 2. Actions
                for action in actions:
                    err = await self._do_action(page, action, timeout_ms)
                    if err:
                        failures.append(err)

                # 3. Assertions (preconditions first: unreachable app state is environmental, not a code defect)
                for spec in options.get("preconditions") or []:
                    ok, failure = await self._check_assertion(page, spec, console_logs, network_errors)
                    if not ok:
                        return self._infra(
                            target_url,
                            f"Precondition not met ({failure.get('type')} on {failure.get('selector')}): app did not reach the state under test",
                            console_logs,
                        )
                for spec in assertions:
                    ok, failure = await self._check_assertion(page, spec, console_logs, network_errors)
                    if ok:
                        passed += 1
                    else:
                        failures.append(failure)

                try:
                    inner_txt = await page.inner_text("body")
                    dom_snippet = inner_txt if inner_txt else await page.content()
                    if len(dom_snippet) > 4000:
                        dom_snippet = dom_snippet[:4000] + "... [truncated]"
                except Exception:
                    pass
                observations = await self._observe(page, options.get("observe") or [])

                if failures and self.artifacts_dir:
                    out = Path(self.artifacts_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    try:
                        screenshot_path = str(out / f"ui_fail_{stamp}.png")
                        await page.screenshot(path=screenshot_path, full_page=True)
                    except Exception:
                        screenshot_path = None
                    if tracing:
                        try:
                            trace_path = str(out / f"ui_fail_{stamp}.zip")
                            await context.tracing.stop(path=trace_path)
                            tracing = False
                        except Exception:
                            trace_path = None
            finally:
                if context is not None and tracing:
                    try:
                        await context.tracing.stop()
                    except Exception:
                        pass
                try:
                    if cdp_url:
                        if page is not None and not reuse:
                            await page.close()  # leave the user's browser and page running
                    else:
                        await browser.close()
                except Exception:
                    pass

        soft_failures = [f for f in failures if f.get("soft")]
        failures = [f for f in failures if not f.get("soft")]
        is_passed = not failures
        summary = (
            f"UI verification passed: {passed} assertion(s) verified in live DOM."
            if is_passed
            else f"UI verification failed: {len(failures)} failure(s), {passed} assertion(s) passed."
        )
        return UIVerificationResult(
            passed=is_passed,
            summary=summary,
            target_url=target_url,
            assertions_passed=passed,
            assertions_failed=len(failures),
            failures=failures,
            soft_failures=soft_failures,
            observations=observations,
            dom_snapshot=dom_snippet,
            console_logs=console_logs,
            network_errors=network_errors,
            screenshot_path=screenshot_path,
            trace_path=trace_path,
        )

    # ----------------------------------------------------------------- actions

    @staticmethod
    async def _install_mock(context: Any, mock: Dict[str, Any]) -> None:
        """Fulfil matching requests locally so UI can be verified without a live backend."""
        pattern: Any = mock.get("url", "**/*")
        if isinstance(pattern, str) and pattern.startswith("re:"):
            pattern = re.compile(pattern[3:])
        method = str(mock.get("method", "")).upper()

        async def handler(route: Any) -> None:
            if method and route.request.method.upper() != method:
                await route.fallback()
                return
            kwargs: Dict[str, Any] = {"status": int(mock.get("status", 200))}
            if "json" in mock:
                kwargs["json"] = mock["json"]
            else:
                kwargs["body"] = str(mock.get("body", ""))
                kwargs["content_type"] = mock.get("content_type", "text/plain")
            if mock.get("headers"):
                kwargs["headers"] = mock["headers"]
            await route.fulfill(**kwargs)

        await context.route(pattern, handler)

    @staticmethod
    async def _observe(page: Any, specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Structured evidence beyond screenshots: URL, title, accessibility tree, named DOM probes."""
        obs: Dict[str, Any] = {"url": page.url}
        try:
            obs["title"] = await page.title()
        except Exception:
            pass
        try:
            obs["aria_snapshot"] = (await page.locator("body").aria_snapshot())[:6000]
        except Exception:
            pass
        for spec in specs:
            name, sel = spec.get("name"), spec.get("selector")
            if not name or not sel:
                continue
            try:
                loc = page.locator(sel)
                kind = spec.get("kind", "texts")
                if kind == "count":
                    obs[name] = await loc.count()
                elif kind == "attrs":
                    obs[name] = [await loc.nth(i).get_attribute(spec.get("attribute", "class")) for i in range(min(await loc.count(), 50))]
                else:
                    obs[name] = [_norm(t) for t in await loc.all_text_contents()][:100]
            except Exception as exc:
                obs[name] = f"error: {exc}"
        return obs

    # ----------------------------------------------------------------- actions

    async def _do_action(self, page: Any, action: Dict[str, Any], timeout_ms: int) -> Optional[Dict[str, Any]]:
        kind = action.get("type")
        selector = action.get("selector")
        wait_ms = min(timeout_ms, 8000)

        def fail(error: str) -> Dict[str, Any]:
            return {"type": "action_failure", "action": kind, "selector": selector, "error": error}

        if kind not in _SUPPORTED_ACTIONS:
            return fail(f"unsupported action type '{kind}' (supported: {', '.join(_SUPPORTED_ACTIONS)})")
        needs_selector = kind in ("click", "type", "press", "select", "hover", "upload_file", "wait_for_selector")
        if needs_selector and not selector:
            return fail("selector is required")

        try:
            if kind == "click":
                await page.locator(selector).first.click(timeout=wait_ms)
            elif kind == "type":
                await page.locator(selector).first.fill(str(action.get("text", "")), timeout=wait_ms)
            elif kind == "press":
                await page.locator(selector).first.press(str(action.get("key", "Enter")), timeout=wait_ms)
            elif kind == "select":
                await page.locator(selector).first.select_option(action.get("value"), timeout=wait_ms)
            elif kind == "hover":
                await page.locator(selector).first.hover(timeout=wait_ms)
            elif kind == "upload_file":
                path = action.get("file_path")
                if not path or not Path(path).is_file():
                    return fail(f"file_path does not exist: {path}")
                await page.locator(selector).first.set_input_files(path, timeout=wait_ms)
            elif kind == "wait":
                await asyncio.sleep(int(action.get("duration_ms", 1000)) / 1000.0)
            elif kind == "wait_for_selector":
                await page.locator(selector).first.wait_for(
                    state=action.get("state", "visible"), timeout=wait_ms
                )
            elif kind == "goto":
                await page.goto(str(action.get("url")), timeout=timeout_ms, wait_until="domcontentloaded")
            elif kind == "set_viewport":
                await page.set_viewport_size(
                    {"width": int(action.get("width", 1280)), "height": int(action.get("height", 800))}
                )
        except Exception as exc:
            logger.warning("Action '%s' on '%s' failed: %s", kind, selector, exc)
            return fail(str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
        return None

    # -------------------------------------------------------------- assertions

    async def _check_assertion(
        self,
        page: Any,
        spec: Dict[str, Any],
        console_logs: List[str],
        network_errors: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        kind = spec.get("type")
        selector = spec.get("selector")
        expected = spec.get("expected")
        timeout = int(spec.get("timeout_ms", self.assertion_timeout_ms))

        def fail(ftype: str, actual: Any = None, **extra: Any) -> Tuple[bool, Dict[str, Any]]:
            data = {"type": ftype, "selector": selector, "expected": expected, "actual": actual}
            if spec.get("soft"):
                data["soft"] = True
            data.update(extra)
            return False, data

        if kind not in _SUPPORTED_ASSERTIONS:
            return fail("unsupported_assertion", f"unknown type '{kind}'")

        if kind == "no_console_errors":
            errors = [
                l for l in console_logs
                if _is_console_error(l) and not any(p.search(l) for p in self.ignore_patterns)
            ]
            return (True, {}) if not errors else fail("console_errors_detected", errors, expected="Zero console errors")
        if kind == "no_network_errors":
            net = [n for n in network_errors if not any(p.search(n) for p in self.ignore_patterns)]
            return (True, {}) if not net else fail(
                "network_errors_detected", net[:10], expected="No failed requests"
            )
        if kind == "url_contains":
            if not expected:
                return fail("invalid_assertion", "expected is required")

            async def check_url() -> Tuple[bool, Any]:
                return str(expected) in page.url, page.url
            ok, actual = await _eventually(check_url, timeout)
            return (True, {}) if ok else fail("url_mismatch", actual)

        if not selector:
            return fail("invalid_assertion", "selector is required")
        loc = page.locator(selector)

        async def check() -> Tuple[bool, Any]:
            count = await loc.count()
            if kind == "element_count":
                target = int(expected if expected is not None else 1)
                return count == target, count
            if kind == "column_order":
                want = [_norm(x) for x in (expected or [])]
                if not want:
                    return False, "expected must be a non-empty list of header labels"
                headers = [_norm(t) for t in await loc.all_text_contents()]
                headers = [h for h in headers if h]
                idx, prev = [], -1
                for label in want:
                    pos = next((i for i, h in enumerate(headers) if label in h), -1)
                    idx.append(pos)
                strict = spec.get("mode", "relative") == "exact"
                ok = all(i >= 0 for i in idx) and idx == sorted(idx) and len(set(idx)) == len(idx)
                if ok and strict:
                    ok = idx == list(range(idx[0], idx[0] + len(idx)))
                return ok, headers
            if kind == "aria_contains":
                snap = await loc.first.aria_snapshot() if count else ""
                return _norm(expected) in _norm(snap), _norm(snap)[:300]
            if kind == "element_hidden":
                return count == 0 or not await loc.first.is_visible(), count
            if count == 0:
                return False, "element not found"
            first = loc.first
            if kind == "element_visible":
                return await first.is_visible(), "not visible"
            if kind in ("text_contains", "text_equals"):
                if expected is None:
                    return False, "expected is required"
                text = _norm(await first.text_content())
                want = _norm(expected)
                return (want in text if kind == "text_contains" else text == want), text[:300]
            if kind == "element_disabled":
                want_disabled = True if expected is None else bool(expected)
                is_dis = await first.is_disabled()
                return is_dis == want_disabled, f"disabled={is_dis}"
            if kind == "attribute_equals":
                attr = spec.get("attribute")
                if not attr:
                    return False, "attribute is required"
                val = await first.get_attribute(attr)
                return (val is not None and (val == expected or str(expected) in val)), val
            if kind == "computed_style":
                prop = spec.get("property")
                if not prop:
                    return False, "property is required"
                val = await first.evaluate("(el, p) => getComputedStyle(el).getPropertyValue(p)", prop)
                return (_norm(expected) in _norm(val) if val else False), val
            if kind == "percentage_dimension":
                dim = str(spec.get("dimension", "width")).lower()
                want = float(spec.get("expected_percent", expected or 0.0))
                tol = float(spec.get("tolerance", 1.0))
                pct = await page.evaluate(
                    """([sel, pSel, d]) => {
                        const el = document.querySelector(sel);
                        const parent = pSel ? document.querySelector(pSel) : (el && el.parentElement);
                        if (!el || !parent) return null;
                        const a = el.getBoundingClientRect(), b = parent.getBoundingClientRect();
                        const x = d === 'height' ? a.height : a.width;
                        const y = d === 'height' ? b.height : b.width;
                        return y === 0 ? 0 : parseFloat(((x / y) * 100).toFixed(2));
                    }""",
                    [selector, spec.get("parent_selector"), dim],
                )
                return (pct is not None and abs(pct - want) <= tol), pct
            return False, "unhandled"

        ok, actual = await _eventually(check, timeout)
        return (True, {}) if ok else fail(f"{kind}_failed", actual)


def build_verifier_from_env(**overrides: Any) -> PlaywrightUIVerifier:
    """Headless, no slow-mo defaults for automation; env AVIATOR_PW_HEADLESS=0 shows the browser."""
    headless = os.environ.get("AVIATOR_PW_HEADLESS", "1").lower() not in ("0", "false", "no")
    params: Dict[str, Any] = {"headless": headless, "slow_mo": 0, "browser_channel": ""}
    params.update(overrides)
    return PlaywrightUIVerifier(**params)
