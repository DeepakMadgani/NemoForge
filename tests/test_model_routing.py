"""Model routing regression tests (harness hardening items 15 & 16).

  * no process-wide os.environ["LLM_MODEL"] writes remain in chatbot/ or src/
  * build_stage_model_resolver maps graph node names onto the router's AUTO_STAGE tiers
  * register_stage_model_resolver_for_current_thread is thread-scoped and tolerates a missing hook
"""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_root = Path(__file__).resolve().parents[1]
_backend_dir = _root / "chatbot" / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
_src_dir = _root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import main  # noqa: E402
from ticket_to_code.intelligence.model_router import ModelMode  # noqa: E402


_ENV_WRITE_RE = re.compile(r'os\.environ\[\s*["\']LLM_MODEL["\']\s*\]\s*=')


def test_no_process_wide_llm_model_env_writes():
    offenders = []
    for base in (_root / "chatbot", _root / "src"):
        for py in base.rglob("*.py"):
            if "node_modules" in py.parts or ".venv" in py.parts:
                continue
            try:
                text = py.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), 1):
                if _ENV_WRITE_RE.search(line):
                    offenders.append(f"{py.relative_to(_root)}:{idx}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)


class _FakeRouter:
    @staticmethod
    def select_model_for_stage(stage_name, ticket, mode=None, active_task_model=None):
        return SimpleNamespace(model_id={"plan": "deep-x", "generate_code": "code-x", "investigate": "fast-x"}[stage_name])


def test_build_stage_model_resolver_auto_stage():
    fn = main.build_stage_model_resolver(
        ModelMode.AUTO_STAGE, {"title": "t"}, SimpleNamespace(model_id="base"), router=_FakeRouter
    )
    assert fn("plan") == "deep-x"
    assert fn("remediation_plan") == "deep-x"
    assert fn("edit_loop") == "code-x"
    assert fn("generate_tests") == "code-x"
    assert fn("fix_build") == "fast-x"
    assert fn("investigate") is None
    assert fn("") is None


def test_build_stage_model_resolver_default_mode_pins_selected_model():
    fn = main.build_stage_model_resolver(
        ModelMode.DEFAULT, {"title": "t"}, SimpleNamespace(model_id="base"), router=_FakeRouter
    )
    for node in ("plan", "edit_loop", "fix_build", "investigate", "anything"):
        assert fn(node) == "base"


def test_register_resolver_is_thread_scoped():
    fn = main.build_stage_model_resolver(
        ModelMode.AUTO_STAGE, {"title": "t"}, SimpleNamespace(model_id="base"), router=_FakeRouter
    )
    captured = []
    try:
        assert main.register_stage_model_resolver_for_current_thread(fn, setter=captured.append) is True
        assert len(captured) == 1
        dispatcher = captured[0]
        assert dispatcher("plan") == "deep-x"

        seen = {}

        def _other():
            seen["val"] = dispatcher("plan")

        t = threading.Thread(target=_other)
        t.start()
        t.join()
        assert seen["val"] is None
    finally:
        main._STAGE_RESOLVERS_BY_THREAD.pop(threading.get_ident(), None)


def test_register_resolver_without_hook_returns_false(monkeypatch):
    monkeypatch.setattr(main, "_set_stage_model_resolver", None)
    fn = main.build_stage_model_resolver(
        ModelMode.DEFAULT, {"title": "t"}, SimpleNamespace(model_id="base"), router=_FakeRouter
    )
    try:
        assert main.register_stage_model_resolver_for_current_thread(fn) is False
        assert main._thread_scoped_stage_resolver("plan") == "base"
    finally:
        main._STAGE_RESOLVERS_BY_THREAD.pop(threading.get_ident(), None)
    assert main._thread_scoped_stage_resolver("plan") is None


def test_workflow_hook_is_wired_when_available():
    """When the workflow exposes set_stage_model_resolver, main imports it directly."""
    try:
        from ticket_to_code.workflow import set_stage_model_resolver
    except ImportError:
        pytest.skip("workflow hook not present")
    assert main._set_stage_model_resolver is set_stage_model_resolver
