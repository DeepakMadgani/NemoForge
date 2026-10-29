"""Real (non-mocked) tests for the Playwright verifier and the pipeline UI-verification helpers."""

import functools
import http.server
import socket
import threading
from types import SimpleNamespace

import pytest

from ticket_to_code.verification import ui_verification as uv
from ticket_to_code.verification.playwright_verifier import (
    PlaywrightUIVerifier,
    UIVerificationResult,
)

PAGE = """<!doctype html><html><head><title>t</title></head><body>
<h1 id="title">Export Center</h1>
<button id="go" disabled>Go</button>
<button id="add" onclick="document.getElementById('out').textContent='Added item'">Add</button>
<div id="out"></div>
<script>setTimeout(()=>{document.body.insertAdjacentHTML('beforeend','<p class=late>late</p>')},300)</script>
</body></html>"""

BAD_PAGE = "<html><body><h1>Broken</h1><script>throw new Error('boom')</script></body></html>"

GRID_PAGE = """<!doctype html><html><body><table><thead><tr id="hdr"></tr></thead><tbody id="rows"></tbody></table>
<script>fetch('/api/grid').then(r=>r.json()).then(d=>{
 document.getElementById('hdr').innerHTML=d.columns.map(c=>'<th>'+c+'</th>').join('');
 document.getElementById('rows').innerHTML=d.rows.map(r=>'<tr><td>'+r+'</td></tr>').join('');});</script></body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("site")
    (root / "index.html").write_text(PAGE, encoding="utf-8")
    (root / "bad.html").write_text(BAD_PAGE, encoding="utf-8")
    (root / "grid.html").write_text(GRID_PAGE, encoding="utf-8")
    (root / "favicon.ico").write_bytes(b"")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _verifier(tmp_path):
    return PlaywrightUIVerifier(
        headless=True, slow_mo=0, browser_channel="", artifacts_dir=str(tmp_path), overall_timeout_s=60
    )


async def _run(tmp_path, url, actions=None, assertions=None):
    res = await _verifier(tmp_path).verify_dom_scenario(url, actions, assertions, timeout_ms=10000)
    if res.infra_error and "No usable Chromium" in res.summary:
        pytest.skip("Chromium not installed for Playwright")
    return res


async def test_passes_on_valid_page_with_late_rendered_content(site, tmp_path):
    res = await _run(
        tmp_path, site + "/index.html",
        actions=[{"type": "click", "selector": "#add"}],
        assertions=[
            {"type": "text_contains", "selector": "#title", "expected": "export"},
            {"type": "text_equals", "selector": "#out", "expected": "Added item"},
            {"type": "element_visible", "selector": ".late"},
            {"type": "element_disabled", "selector": "#go", "expected": True},
            {"type": "element_count", "selector": "button", "expected": 2},
            {"type": "no_console_errors"},
            {"type": "no_network_errors"},
        ],
    )
    assert res.passed, res.failures
    assert res.assertions_passed == 7


async def test_unknown_assertion_and_action_fail_instead_of_passing_silently(site, tmp_path):
    res = await _run(
        tmp_path, site + "/index.html",
        actions=[{"type": "teleport", "selector": "#x"}],
        assertions=[{"type": "made_up", "selector": "#title"}],
    )
    assert not res.passed
    kinds = {f["type"] for f in res.failures}
    assert {"action_failure", "unsupported_assertion"} <= kinds


async def test_missing_element_and_wrong_text_fail_with_screenshot(site, tmp_path):
    res = await _run(
        tmp_path, site + "/index.html",
        assertions=[
            {"type": "element_visible", "selector": "#nope", "timeout_ms": 500},
            {"type": "text_contains", "selector": "#title", "expected": "Import", "timeout_ms": 500},
        ],
    )
    assert not res.passed and res.assertions_failed == 2
    assert res.screenshot_path and res.screenshot_path.endswith(".png")


async def test_console_error_detected(site, tmp_path):
    res = await _run(tmp_path, site + "/bad.html", assertions=[{"type": "no_console_errors"}])
    assert not res.passed
    assert res.failures[0]["type"] == "console_errors_detected"


async def test_soft_failures_do_not_fail_run(site, tmp_path):
    res = await _run(
        tmp_path, site + "/index.html",
        assertions=[{"type": "text_contains", "selector": "body", "expected": "Nonexistent", "soft": True, "timeout_ms": 300}],
    )
    assert res.passed and len(res.soft_failures) == 1


async def test_ignore_patterns_suppress_known_baseline_errors(site, tmp_path):
    v = PlaywrightUIVerifier(
        headless=True, slow_mo=0, browser_channel="", artifacts_dir=str(tmp_path), ignore_patterns=["boom"]
    )
    res = await v.verify_dom_scenario(site + "/bad.html", None, [{"type": "no_console_errors"}], timeout_ms=10000)
    if res.infra_error and "No usable Chromium" in res.summary:
        pytest.skip("Chromium not installed for Playwright")
    assert res.passed, res.failures


async def test_unreachable_server_is_infra_error_not_ui_failure(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    res = await _run(tmp_path, f"http://127.0.0.1:{port}/")
    assert res.infra_error and not res.passed


async def _grid(tmp_path, site, columns, assertions, **opts):
    mock = {"url": "**/api/grid", "json": {"columns": columns, "rows": ["DB1", "DB2"]}}
    res = await _verifier(tmp_path).verify_dom_scenario(
        site + "/grid.html", None, assertions, timeout_ms=10000,
        options={"mock_routes": [mock], "observe": [{"name": "headers", "selector": "th"}], **opts},
    )
    if res.infra_error and "No usable Chromium" in res.summary:
        pytest.skip("Chromium not installed for Playwright")
    return res


ORDER = [{"type": "column_order", "selector": "th", "expected": ["Name", "Issues", "Content"], "timeout_ms": 1500}]


async def test_mocked_backend_and_column_order_pass_and_expose_observations(site, tmp_path):
    res = await _grid(tmp_path, site, ["Status", "Info", "Name", "Issues", "Content", "List"], ORDER)
    assert res.passed, res.failures
    assert res.observations["headers"] == ["status", "info", "name", "issues", "content", "list"]
    assert "aria_snapshot" in res.observations


async def test_column_order_detects_old_order(site, tmp_path):
    res = await _grid(tmp_path, site, ["Status", "Info", "Content", "Issues", "Name"], ORDER)
    assert not res.passed
    assert res.failures[0]["type"] == "column_order_failed"


async def test_column_order_exact_mode_requires_adjacent_columns(site, tmp_path):
    exact = [{**ORDER[0], "mode": "exact"}]
    res = await _grid(tmp_path, site, ["Name", "List", "Issues", "Content"], exact)
    assert not res.passed


def test_result_to_dict_has_new_fields():
    d = UIVerificationResult(True, "ok", "u", 1, 0).to_dict()
    assert {"network_errors", "soft_failures", "infra_error", "screenshot_path", "duration_ms"} <= d.keys()


def test_changed_ui_files_and_assertion_derivation():
    gen = [SimpleNamespace(file_path="a/b/x.component.html"), SimpleNamespace(file_path="svc/Foo.java")]
    assert uv.changed_ui_files(gen) == ["a/b/x.component.html"]
    asserts = uv.derive_assertions(["Show an 'Export' button", "No quotes here"])
    hard = [a for a in asserts if not a.get("soft")]
    soft = [a for a in asserts if a.get("soft")]
    assert {a["type"] for a in hard} == {"element_visible", "no_console_errors", "no_network_errors"}
    assert [a["expected"] for a in soft] == ["Export"]


def test_run_ui_verification_is_skipped_by_default_and_without_ui_files(monkeypatch, tmp_path):
    monkeypatch.delenv("TICKET_TO_CODE_UI_VERIFY", raising=False)
    assert uv.run_ui_verification(str(tmp_path), [])["status"] == "skipped"
    monkeypatch.setenv("TICKET_TO_CODE_UI_VERIFY", "1")
    res = uv.run_ui_verification(str(tmp_path), [SimpleNamespace(file_path="Foo.java")])
    assert res["status"] == "skipped" and "no frontend" in res["reason"]


def test_find_frontend_root_walks_up_to_package_json(tmp_path):
    (tmp_path / "web" / "src" / "app").mkdir(parents=True)
    (tmp_path / "web" / "package.json").write_text("{}", encoding="utf-8")
    assert uv.find_frontend_root(tmp_path, ["web/src/app/x.html"]) == (tmp_path / "web").resolve()
    assert uv.find_frontend_root(tmp_path, ["other/x.html"]) is None


def test_end_to_end_with_existing_url(site, monkeypatch, tmp_path):
    monkeypatch.setenv("TICKET_TO_CODE_UI_VERIFY", "1")
    monkeypatch.setenv("TICKET_TO_CODE_UI_URL", site)
    monkeypatch.setenv("TICKET_TO_CODE_UI_PATH", "/index.html")
    monkeypatch.setenv("TICKET_TO_CODE_UI_ARTIFACTS", str(tmp_path))
    gen = [SimpleNamespace(file_path="web/x.component.html")]
    res = uv.run_ui_verification(str(tmp_path), gen, ["Show 'Export Center' heading"])
    if res["status"] == "unavailable" and "Chromium" in res["reason"]:
        pytest.skip("Chromium not installed for Playwright")
    assert res["status"] == "passed", res
    monkeypatch.setenv("TICKET_TO_CODE_UI_PATH", "/bad.html")
    res = uv.run_ui_verification(str(tmp_path), gen, [])
    assert res["status"] == "failed" and res["repair_hints"]
