"""
Backend security hardening regression tests.

Covers:
  * /api/settings/keys never returns raw API keys (item 15)
  * MCP command allow-list / shell-metachar rejection / ${VAR} env expansion (item 16, 20)
  * /api/file-tree refuses paths outside registered project roots (item 17)
  * Upload filename sanitisation (item 18)
  * Playwright verification host allow-list (item 19)
  * DurableWorkflowStateStore.mark_stale_running + bounded cache preload (item 22)
"""

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_backend_dir = Path(__file__).resolve().parents[1] / "chatbot" / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
_src_dir = Path(__file__).resolve().parents[1] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import main  # noqa: E402
from services import mcp_manager as mcp_mod  # noqa: E402
from services.mcp_manager import (  # noqa: E402
    McpManager,
    build_subprocess_env,
    expand_env_value,
    validate_server_command,
)
from ticket_to_code.runtime.state_store import DurableWorkflowStateStore  # noqa: E402


# ---------------------------------------------------------------------------
# Item 15: masked keys
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_settings_keys_response_has_no_raw_key(monkeypatch):
    secret = "sk-test-SECRET-1234567890abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    resp = await main.get_settings_keys()
    keys = resp.get("configured_keys") or resp.get("keys") or []
    assert keys, f"expected at least one configured key in response: {list(resp.keys())}"
    for entry in keys:
        assert "raw_key" not in entry, "raw_key must never be sent to the browser"
    openai_entries = [k for k in keys if k.get("id") == "openai"]
    assert openai_entries, "OPENAI_API_KEY from the environment should be discovered"
    assert openai_entries[0]["masked_key"] == "sk-t...cdef"
    assert secret not in json.dumps(resp), "full secret leaked somewhere in the response"


# ---------------------------------------------------------------------------
# Item 16 / 20: MCP command hardening
# ---------------------------------------------------------------------------

def test_mcp_allowlist_rejects_cmd_exe_and_absolute_paths():
    assert validate_server_command({"command": "cmd.exe", "args": ["/c", "npx"]}) is not None
    assert validate_server_command({"command": "cmd", "args": []}) is not None
    assert validate_server_command({"command": "bash", "args": ["-c", "id"]}) is not None
    assert validate_server_command({"command": r"C:\Windows\System32\cmd.exe", "args": []}) is not None
    assert validate_server_command({"command": "/usr/bin/python3", "args": []}) is not None
    assert validate_server_command({"command": "python.exe", "args": []}) is not None
    assert validate_server_command({"command": "", "args": []}) is not None
    # Allowed launchers with plain args are accepted
    for ok in ("npx", "node", "python", "python3", "uvx", "docker"):
        assert validate_server_command({"command": ok, "args": ["-y", "@scope/pkg@1.0.0", "--flag=value"]}) is None


@pytest.mark.parametrize("bad_arg", [
    "foo; rm -rf /",
    "a && b",
    "a | b",
    "`whoami`",
    "$HOME",
    "x > out.txt",
    "x < in.txt",
    "line1\nline2",
])
def test_mcp_allowlist_rejects_shell_metacharacters(bad_arg):
    reason = validate_server_command({"command": "npx", "args": ["-y", bad_arg]})
    assert reason is not None
    assert "metacharacter" in reason


def test_mcp_save_server_rejects_unsafe_definition(tmp_path):
    mgr = McpManager(config_path=tmp_path / "mcp_servers.json")
    with pytest.raises(ValueError):
        mgr.save_server("evil", {"command": "cmd.exe", "args": ["/c", "calc"]})
    with pytest.raises(ValueError):
        mgr.save_server("evil2", {"command": "npx", "args": ["-y", "pkg; echo pwned"]})
    saved = mgr.get_servers()
    assert "evil" not in saved and "evil2" not in saved
    # A safe definition still saves
    mgr.save_server("ok", {"command": "node", "args": ["server.js"]})
    assert "ok" in mgr.get_servers()


@pytest.mark.asyncio
async def test_mcp_test_connection_returns_error_dict_instead_of_running(tmp_path, monkeypatch):
    cfg = tmp_path / "mcp_servers.json"
    # Bypass save_server validation to simulate a hand-edited config file
    cfg.write_text(json.dumps({"mcpServers": {
        "shell": {"name": "shell", "command": "cmd.exe", "args": ["/c", "echo hi"], "env": {}, "enabled": True, "tools": []},
        "meta": {"name": "meta", "command": "npx", "args": ["-y", "pkg; echo pwned"], "env": {}, "enabled": True, "tools": [{"name": "t"}]},
    }}), encoding="utf-8")
    mgr = McpManager(config_path=cfg)

    popen_calls = []

    class _NeverPopen:
        def __init__(self, *a, **kw):
            popen_calls.append(a)
            raise AssertionError("subprocess.Popen must not be invoked for rejected commands")

    import subprocess
    monkeypatch.setattr(subprocess, "Popen", _NeverPopen)

    res = await mgr.test_connection("shell")
    assert res["status"] == "error" and res.get("code") == "command_rejected"
    res = await mgr.test_connection("meta")
    assert res["status"] == "error" and res.get("code") == "command_rejected"
    res = mgr.call_tool_sync("meta", "t", {})
    assert res.get("code") == "command_rejected"
    assert popen_calls == []


def test_mcp_no_cmd_exe_wrapper_in_source():
    src = Path(mcp_mod.__file__).read_text(encoding="utf-8")
    assert '"cmd.exe"' not in src, "npx must be resolved with shutil.which, not wrapped in cmd.exe /c"
    assert "shutil.which" in src


def test_mcp_env_var_expansion(monkeypatch):
    monkeypatch.setenv("FIGMA_PERSONAL_ACCESS_TOKEN", "figd_from_environment")
    monkeypatch.delenv("DEFINITELY_UNSET_NEMOFORGE_VAR", raising=False)
    assert expand_env_value("${FIGMA_PERSONAL_ACCESS_TOKEN}") == "figd_from_environment"
    assert expand_env_value("prefix-${FIGMA_PERSONAL_ACCESS_TOKEN}-suffix") == "prefix-figd_from_environment-suffix"
    assert expand_env_value("literal") == "literal"
    assert expand_env_value("${DEFINITELY_UNSET_NEMOFORGE_VAR}") is None

    env = build_subprocess_env({
        "name": "figma",
        "env": {
            "FIGMA_API_KEY": "${FIGMA_PERSONAL_ACCESS_TOKEN}",
            "PLAIN": "value",
            "MISSING": "${DEFINITELY_UNSET_NEMOFORGE_VAR}",
        },
    })
    assert env["FIGMA_API_KEY"] == "figd_from_environment"
    assert env["PLAIN"] == "value"
    assert "MISSING" not in env, "unset references must be omitted, not passed as empty/literal"


def test_committed_mcp_config_has_no_figma_token():
    cfg = _backend_dir / "mcp_servers.json"
    example = _backend_dir / "mcp_servers.example.json"
    for p in (cfg, example):
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            assert "figd_" not in txt, f"{p.name} must not contain a literal Figma token"
    assert example.exists()
    assert "${FIGMA_PERSONAL_ACCESS_TOKEN}" in example.read_text(encoding="utf-8")
    gitignore = (_backend_dir.parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "chatbot/backend/mcp_servers.json" in gitignore


# ---------------------------------------------------------------------------
# Item 17: file-tree containment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_tree_rejects_path_outside_registered_roots(tmp_path, monkeypatch):
    for k in ("AVIATOR_WORKSPACE_PATH", "TICKET_TO_CODE_WORKSPACE_PATH", "WORKSPACE_PATH"):
        monkeypatch.delenv(k, raising=False)
    registered = tmp_path / "project"
    (registered / "src").mkdir(parents=True)
    (registered / "src" / "a.py").write_text("x = 1")
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    monkeypatch.setattr(main, "projects", {
        "p1": main.Project(id="p1", name="p", type=main.ProjectType.LOCAL, path=str(registered)),
    })

    with pytest.raises(HTTPException) as exc:
        await main.get_file_tree(path=str(outside))
    assert exc.value.status_code == 403

    # Traversal out of a registered root is also refused
    with pytest.raises(HTTPException) as exc:
        await main.get_file_tree(path=str(registered / ".." / "elsewhere"))
    assert exc.value.status_code == 403

    # System paths are refused even when they exist
    with pytest.raises(HTTPException) as exc:
        await main.get_file_tree(path=str(Path.home()))
    assert exc.value.status_code == 403

    # Inside a registered root is fine
    tree = await main.get_file_tree(path=str(registered))
    assert tree["type"] == "folder"
    names = {c["name"] for c in tree["children"]}
    assert "src" in names


# ---------------------------------------------------------------------------
# Item 18: upload filename sanitisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "..", "../../etc/passwd", "..\\..\\windows\\win.ini", "a/../b", "foo..bar", None])
def test_safe_upload_filename_rejects_traversal(bad):
    with pytest.raises(HTTPException) as exc:
        main._safe_upload_filename(bad)
    assert exc.value.status_code == 400


def test_safe_upload_filename_reduces_to_basename():
    assert main._safe_upload_filename("report.pdf") == "report.pdf"
    assert main._safe_upload_filename("nested/dir/screenshot.png") == "screenshot.png"
    assert main._safe_upload_filename("C:\\Users\\x\\log.txt") == "log.txt"


# ---------------------------------------------------------------------------
# Item 19: Playwright host filter
# ---------------------------------------------------------------------------

def test_playwright_host_filter(monkeypatch):
    monkeypatch.delenv("NEMOFORGE_ALLOWED_VERIFY_HOSTS", raising=False)
    assert main._is_allowed_verify_url("http://localhost:4200") is True
    assert main._is_allowed_verify_url("http://127.0.0.1:3000/app") is True
    assert main._is_allowed_verify_url("http://[::1]:8080/") is True
    assert main._is_allowed_verify_url("https://LOCALHOST/") is True
    assert main._is_allowed_verify_url("http://169.254.169.254/latest/meta-data/") is False
    assert main._is_allowed_verify_url("http://evil.example.com/") is False
    assert main._is_allowed_verify_url("http://localhost.evil.com/") is False
    assert main._is_allowed_verify_url("file:///etc/passwd") is False
    assert main._is_allowed_verify_url("") is False
    assert main._is_allowed_verify_url(None) is False

    monkeypatch.setenv("NEMOFORGE_ALLOWED_VERIFY_HOSTS", "staging.internal, app.corp.local")
    assert main._is_allowed_verify_url("https://staging.internal/login") is True
    assert main._is_allowed_verify_url("https://APP.corp.local:8443/") is True
    assert main._is_allowed_verify_url("https://other.internal/") is False


@pytest.mark.asyncio
async def test_playwright_endpoint_returns_400_for_disallowed_host(monkeypatch):
    monkeypatch.delenv("NEMOFORGE_ALLOWED_VERIFY_HOSTS", raising=False)

    async def _must_not_run(**kwargs):
        raise AssertionError("verifier must not be invoked for a disallowed host")

    monkeypatch.setattr(main.playwright_verifier, "verify_dom_scenario", _must_not_run)
    with pytest.raises(HTTPException) as exc:
        await main.verify_with_playwright({"target_url": "http://169.254.169.254/"})
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Item 22: orphan reconciliation in the durable store
# ---------------------------------------------------------------------------

def test_mark_stale_running_on_temp_store(tmp_path):
    db = tmp_path / "wf.db"
    store = DurableWorkflowStateStore(db_path=db)
    store["wf-running-1"] = {"workflow_id": "wf-running-1", "status": "running", "current_phase": "generate"}
    store["wf-running-2"] = {"workflow_id": "wf-running-2", "status": "running"}
    store["wf-done"] = {"workflow_id": "wf-done", "status": "completed"}
    store["wf-failed"] = {"workflow_id": "wf-failed", "status": "failed", "error": "boom"}

    n = store.mark_stale_running("Workflow interrupted (backend restarted)")
    assert n == 2

    # In-memory cache updated
    for wid in ("wf-running-1", "wf-running-2"):
        assert store[wid]["status"] == "failed"
        assert store[wid]["current_phase"] == "failed"
        assert store[wid]["error"] == "Workflow interrupted (backend restarted)"
        assert store[wid].get("completed_at")
    assert store["wf-done"]["status"] == "completed"
    assert store["wf-failed"]["error"] == "boom"

    # Persisted: a fresh store instance sees the failed status
    reopened = DurableWorkflowStateStore(db_path=db)
    assert reopened["wf-running-1"]["status"] == "failed"
    assert reopened["wf-running-2"]["status"] == "failed"
    assert reopened["wf-done"]["status"] == "completed"

    # Idempotent
    assert reopened.mark_stale_running("again") == 0


def test_load_cache_is_bounded_to_recent_workflows(tmp_path):
    db = tmp_path / "wf.db"
    store = DurableWorkflowStateStore(db_path=db)
    limit = DurableWorkflowStateStore.CACHE_PRELOAD_LIMIT
    for i in range(limit + 5):
        store[f"wf-{i:04d}"] = {"workflow_id": f"wf-{i:04d}", "status": "completed"}

    reopened = DurableWorkflowStateStore(db_path=db)
    assert len(reopened) == limit
    # Old rows remain readable through the read-through path
    assert "wf-0000" in reopened
    assert reopened["wf-0000"]["status"] == "completed"
