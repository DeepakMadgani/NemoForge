"""
Post-build live-UI verification for ticket-to-code.

Starts the project's dev server (or uses an already running one), drives it with the shared
Playwright verifier and returns a plain dict the workflow can store in state.

Statuses:
  skipped      - disabled, or no frontend files were generated
  unavailable  - environment problem (no server, no browser); NEVER triggers code repair
  passed       - hard assertions held
  failed       - real UI failure; fed to the repair loop as deterministic hints

Configuration (env):
  TICKET_TO_CODE_UI_VERIFY        1/true to enable (default off)
  TICKET_TO_CODE_UI_URL           use an already running app instead of starting one
  TICKET_TO_CODE_UI_START_CMD     custom start command (use {port} placeholder)
  TICKET_TO_CODE_UI_PATH          path to open, default "/"
  TICKET_TO_CODE_UI_START_TIMEOUT seconds to wait for the server, default 120
  TICKET_TO_CODE_UI_SCENARIO      JSON file: {"path", "actions", "assertions", "ignore": [regex],
                                  "mock_routes": [{url, method, status, json|body}], "storage_state", "headers",
                                  "observe": [{name, selector, kind: texts|count|attrs}]}
  TICKET_TO_CODE_UI_IGNORE        comma-separated regexes of known baseline console/network noise
  TICKET_TO_CODE_UI_CDP           e.g. http://127.0.0.1:9222 - attach to a running logged-in browser
  TICKET_TO_CODE_UI_ARTIFACTS     screenshot/trace output dir
"""

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_UI_EXTENSIONS = (".html", ".tsx", ".jsx", ".vue", ".svelte", ".scss", ".css", ".less", ".component.ts")
_PKG_MARKERS = ("angular.json", "package.json")


def is_enabled() -> bool:
    return os.environ.get("TICKET_TO_CODE_UI_VERIFY", "").strip().lower() in ("1", "true", "yes", "on")


def changed_ui_files(generated_code: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for g in generated_code or []:
        fp = (getattr(g, "file_path", "") or "").replace("\\", "/")
        if fp.lower().endswith(_UI_EXTENSIONS) and fp not in out:
            out.append(fp)
    return out


def find_frontend_root(workspace: Path, ui_files: List[str]) -> Optional[Path]:
    """Nearest ancestor of a changed UI file that owns a package.json/angular.json."""
    ws = workspace.resolve()
    for rel in ui_files:
        cur = (ws / rel).parent
        while True:
            try:
                cur.relative_to(ws)
            except ValueError:
                break
            if any((cur / m).is_file() for m in _PKG_MARKERS):
                return cur
            if cur == ws:
                break
            cur = cur.parent
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_command(root: Path, port: int) -> Optional[List[str]]:
    custom = os.environ.get("TICKET_TO_CODE_UI_START_CMD", "").strip()
    if custom:
        return shlex.split(custom.replace("{port}", str(port)), posix=os.name != "nt")
    npx, npm = shutil.which("npx"), shutil.which("npm")
    if (root / "angular.json").is_file() and npx:
        return [npx, "ng", "serve", "--port", str(port), "--host", "127.0.0.1"]
    pkg = root / "package.json"
    if not (pkg.is_file() and npm):
        return None
    try:
        scripts = (json.loads(pkg.read_text(encoding="utf-8", errors="ignore")) or {}).get("scripts", {})
    except Exception:
        return None
    if "dev" in scripts and re.search(r"vite|next|nuxt", scripts["dev"]):
        return [npm, "run", "dev", "--", "--port", str(port)]
    if "start" in scripts:
        return [npm, "start", "--", "--port", str(port)]
    return None


class _DevServer:
    def __init__(self, root: Path, cmd: List[str], port: int):
        self.root, self.cmd, self.port = root, cmd, port
        self.proc: Optional[subprocess.Popen] = None
        self.log: List[str] = []

    def _drain(self) -> None:
        try:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                self.log.append(line.rstrip())
                if len(self.log) > 200:
                    del self.log[:100]
        except Exception:
            pass

    def start(self, timeout_s: int) -> Optional[str]:
        """Return None when ready, otherwise an error message."""
        env = {**os.environ, "PORT": str(self.port), "BROWSER": "none", "CI": "true"}
        kwargs: Dict[str, Any] = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        else:
            # Real process group on Windows too, so kill_process_tree() reaps the whole tree.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self.proc = subprocess.Popen(
                self.cmd, cwd=str(self.root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env, **kwargs,
            )
        except Exception as exc:
            return f"could not start dev server: {exc}"
        threading.Thread(target=self._drain, daemon=True).start()
        deadline = time.time() + timeout_s
        url = f"http://127.0.0.1:{self.port}/"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return f"dev server exited with code {self.proc.returncode}: {' | '.join(self.log[-5:])}"
            try:
                urllib.request.urlopen(url, timeout=3).close()
                return None
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return None
            except Exception:
                pass
            time.sleep(1)
        return f"dev server not ready after {timeout_s}s: {' | '.join(self.log[-5:])}"

    def stop(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        from ticket_to_code.runtime.security import kill_process_tree
        kill_process_tree(self.proc)


def derive_assertions(functional_requirements: Iterable[str]) -> List[Dict[str, Any]]:
    """Hard: page renders, no console/network errors. Soft: quoted UI text from requirements."""
    assertions: List[Dict[str, Any]] = [
        {"type": "element_visible", "selector": "body"},
        {"type": "no_console_errors"},
        {"type": "no_network_errors"},
    ]
    seen = set()
    for req in functional_requirements or []:
        for phrase in re.findall(r"['\"“‘]([^'\"”’]{2,60})['\"”’]", str(req)):
            key = phrase.strip().lower()
            if key and key not in seen and len(seen) < 8:
                seen.add(key)
                assertions.append({"type": "text_contains", "selector": "body", "expected": phrase, "soft": True})
    return assertions


def _load_scenario() -> Optional[Dict[str, Any]]:
    path = os.environ.get("TICKET_TO_CODE_UI_SCENARIO", "").strip()
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("UI scenario file unreadable (%s): %s", path, exc)
        return None


def _run_async(coro_factory) -> Any:
    """Run a coroutine on a private loop in a worker thread (safe inside any caller's loop)."""
    box: Dict[str, Any] = {}

    def _worker() -> None:
        import sys
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)
        else:
            loop = asyncio.new_event_loop()
        try:
            box["value"] = loop.run_until_complete(coro_factory())
        except Exception as exc:
            box["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def run_ui_verification(
    workspace_path: str,
    generated_code: Iterable[Any],
    functional_requirements: Iterable[str] = (),
) -> Dict[str, Any]:
    if not is_enabled():
        return {"status": "skipped", "reason": "disabled (set TICKET_TO_CODE_UI_VERIFY=1)"}
    ui_files = changed_ui_files(generated_code)
    if not ui_files:
        return {"status": "skipped", "reason": "no frontend files generated"}

    workspace = Path(workspace_path)
    scenario = _load_scenario() or {}
    base_url = os.environ.get("TICKET_TO_CODE_UI_URL", "").strip().rstrip("/")
    server: Optional[_DevServer] = None

    try:
        if not base_url:
            root = find_frontend_root(workspace, ui_files)
            if root is None:
                return {"status": "unavailable", "reason": "no package.json/angular.json for changed UI files"}
            port = _free_port()
            cmd = _start_command(root, port)
            if not cmd:
                return {"status": "unavailable", "reason": f"no start command detected in {root}"}
            server = _DevServer(root, cmd, port)
            err = server.start(int(os.environ.get("TICKET_TO_CODE_UI_START_TIMEOUT", "120")))
            if err:
                return {"status": "unavailable", "reason": err}
            base_url = f"http://127.0.0.1:{port}"

        path = scenario.get("path") or os.environ.get("TICKET_TO_CODE_UI_PATH", "/")
        url = base_url + (path if str(path).startswith("/") else "/" + str(path))
        assertions = scenario.get("assertions") or derive_assertions(functional_requirements)
        actions = scenario.get("actions") or []

        from ticket_to_code.verification.playwright_verifier import build_verifier_from_env

        artifacts = os.environ.get("TICKET_TO_CODE_UI_ARTIFACTS") or str(workspace / ".ticket_to_code_ui_artifacts")
        verifier = build_verifier_from_env(
            artifacts_dir=artifacts,
            ignore_patterns=[p for p in (list(scenario.get("ignore") or []) + [
                s.strip() for s in os.environ.get("TICKET_TO_CODE_UI_IGNORE", "").split(",")
            ]) if p],
        )
        options = {k: scenario[k] for k in ("mock_routes", "storage_state", "headers", "observe", "preconditions", "cdp_url", "reuse_page") if k in scenario}
        if os.environ.get("TICKET_TO_CODE_UI_CDP"):
            options["cdp_url"] = os.environ["TICKET_TO_CODE_UI_CDP"]
        if os.environ.get("TICKET_TO_CODE_UI_REUSE_PAGE", "").lower() in ("1", "true"):
            options["reuse_page"] = True
        res = _run_async(lambda: verifier.verify_dom_scenario(
            url, actions=actions, assertions=assertions, timeout_ms=30000, options=options,
        ))
    except Exception as exc:
        logger.warning("UI verification unavailable: %s", exc)
        return {"status": "unavailable", "reason": str(exc) or repr(exc)}
    finally:
        if server is not None:
            server.stop()

    if res.infra_error:
        return {"status": "unavailable", "reason": res.summary, "url": url}
    hints = [_failure_hint(f, url) for f in res.failures]
    signature = "|".join(sorted(f"{f.get('type')}:{str(f.get('actual'))[:120]}" for f in res.failures))
    return {
        "status": "passed" if res.passed else "failed",
        "signature": signature,
        "url": url,
        "summary": res.summary,
        "failures": res.failures,
        "soft_failures": res.soft_failures,
        "observations": res.observations,
        "repair_hints": hints,
        "screenshot_path": res.screenshot_path,
        "trace_path": res.trace_path,
        "ui_files": ui_files,
    }


def _failure_hint(f: Dict[str, Any], url: str) -> str:
    kind = f.get("type", "ui_failure")
    if kind == "console_errors_detected":
        return f"live UI at {url} logged console errors: {str(f.get('actual'))[:300]}"
    if kind == "network_errors_detected":
        return f"live UI at {url} had failed network requests: {str(f.get('actual'))[:300]}"
    if kind == "action_failure":
        return f"UI action '{f.get('action')}' on '{f.get('selector')}' failed: {f.get('error')}"
    if kind == "column_order_failed":
        return f"grid column order is wrong: expected relative order {f.get('expected')}, actual headers {f.get('actual')}"
    return (
        f"live UI check '{kind}' failed for selector '{f.get('selector')}': "
        f"expected {f.get('expected')!r}, got {str(f.get('actual'))[:200]!r}"
    )
