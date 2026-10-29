"""llm_invoke hardening: pricing, fallback chain, per-call trajectory, orphan calls.

Covers plan items 17 (LLM_FALLBACK_MODELS chain), 21 (llm_calls.jsonl trajectory),
22 (real cost accounting via runtime/pricing.py) and 23 (no "most recent ticket"
budget fallback — orphan calls are not charged and are warned once per call site).
"""

import json
import logging
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ticket_to_code import llm_utils
from ticket_to_code.llm_utils import llm_invoke
from ticket_to_code.runtime import pricing
from ticket_to_code.runtime.pricing import estimate_cost_usd, reset_pricing_cache
from ticket_to_code.runtime.run_context import (
    BudgetExceeded,
    RunBudget,
    RunCancelled,
    RunContext,
    current_run_context,
)


class FakeLLM:
    """Mock-free LLM double (llm_invoke rejects unittest.mock objects by design)."""

    def __init__(self, content="OK", exc=None, meta=None, model_name=None):
        self.content = content
        self.exc = exc
        self.meta = dict(meta or {})
        self.calls = 0
        self.kwargs = []
        if model_name is not None:
            self.model_name = model_name

    def invoke(self, messages, **kwargs):
        self.calls += 1
        self.kwargs.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return AIMessage(content=self.content, response_metadata=dict(self.meta))


@pytest.fixture
def messages():
    return [HumanMessage(content="hello there")]


@pytest.fixture
def run_ctx():
    tokens = []

    def _set(ctx):
        tokens.append(current_run_context.set(ctx))
        return ctx

    yield _set
    for tok in reversed(tokens):
        current_run_context.reset(tok)


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TICKET_TO_CODE_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("AVIATOR_TRACE_LLM_CONTENT", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("AVIATOR_DEFAULT_MAX_OUTPUT_TOKENS", raising=False)
    return tmp_path


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── pricing ─────────────────────────────────────────────────────────────────

def test_pricing_known_model_non_zero():
    reset_pricing_cache()
    assert estimate_cost_usd("gemini-2.5-pro", 1_000_000, 0) == pytest.approx(1.25)
    assert estimate_cost_usd("gemini-2.5-pro", 0, 1_000_000) == pytest.approx(10.0)
    # prefix / substring matching: dated snapshot ids and provider-prefixed ids
    assert estimate_cost_usd("claude-3-5-sonnet-20241022", 1_000_000, 0) == pytest.approx(3.0)
    assert estimate_cost_usd("nebius/nvidia/nemotron-3-super-120b-a12b", 1_000_000, 0) == pytest.approx(0.60)


def test_pricing_unknown_model_zero_and_single_warning(caplog):
    reset_pricing_cache()
    caplog.set_level(logging.WARNING, logger="ticket_to_code.runtime.pricing")
    assert estimate_cost_usd("totally-unknown-model", 1000, 1000) == 0.0
    assert estimate_cost_usd("totally-unknown-model", 5000, 5000) == 0.0
    assert estimate_cost_usd("", 10, 10) == 0.0
    warnings = [r for r in caplog.records if "No pricing entry" in r.getMessage() and "totally-unknown-model" in r.getMessage()]
    assert len(warnings) == 1, "unknown model must warn exactly once per process"


def test_pricing_env_json_override(tmp_path, monkeypatch):
    table = {"my-model": [1.0, 2.0], "other-model": {"input": 3.0, "output": 4.0}, "gemini-2.5-pro": [100.0, 0.0]}
    p = tmp_path / "prices.json"
    p.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setenv("AVIATOR_PRICING_JSON", str(p))
    reset_pricing_cache()
    try:
        assert estimate_cost_usd("my-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
        assert estimate_cost_usd("other-model", 1_000_000, 1_000_000) == pytest.approx(7.0)
        assert estimate_cost_usd("gemini-2.5-pro", 1_000_000, 0) == pytest.approx(100.0)  # override wins
    finally:
        monkeypatch.delenv("AVIATOR_PRICING_JSON", raising=False)
        reset_pricing_cache()


def test_pricing_unreadable_env_json_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIATOR_PRICING_JSON", str(tmp_path / "missing.json"))
    reset_pricing_cache()
    try:
        assert estimate_cost_usd("gemini-2.5-flash", 1_000_000, 0) == pytest.approx(0.30)
    finally:
        monkeypatch.delenv("AVIATOR_PRICING_JSON", raising=False)
        reset_pricing_cache()


# ── fallback chain (item 17) ────────────────────────────────────────────────

def test_fallback_model_answers_when_primary_fails_non_retryable(messages, run_ctx, trace_dir, monkeypatch):
    ctx = run_ctx(RunContext(ticket_id="T-FB"))
    primary = FakeLLM(exc=RuntimeError("400 bad request"))
    fb = FakeLLM(content="FB-OK")
    resolved = []

    def _resolver(name):
        resolved.append(name)
        return fb

    monkeypatch.setenv("LLM_FALLBACK_MODELS", "fb-model")
    monkeypatch.setattr(llm_utils, "_FALLBACK_RESOLVER", _resolver)

    resp = llm_invoke(primary, messages, max_retries=1, base_delay=0.0)

    assert resp.content == "FB-OK"
    assert primary.calls == 1 and fb.calls == 1
    assert resolved == ["fb-model"]
    assert resp.response_metadata.get("fallback_used") == "fb-model"
    assert ctx.budget.llm_calls == 1
    assert ctx.budget._call_log[-1].fallback_used == "fb-model"
    line = _read_jsonl(trace_dir / "T-FB" / "llm_calls.jsonl")[-1]
    assert line["fallback_used"] == "fb-model"


def test_fallback_skips_unavailable_models_and_raises_primary_when_all_fail(messages, run_ctx, trace_dir, monkeypatch):
    run_ctx(RunContext(ticket_id="T-FB2"))
    primary = FakeLLM(exc=RuntimeError("400 bad request"))
    fb_bad = FakeLLM(exc=RuntimeError("fallback also broken"))

    def _resolver(name):
        if name == "missing":
            raise RuntimeError("cannot build")
        return fb_bad

    monkeypatch.setenv("LLM_FALLBACK_MODELS", "missing, fb-bad ,")
    monkeypatch.setattr(llm_utils, "_FALLBACK_RESOLVER", _resolver)

    with pytest.raises(RuntimeError, match="400 bad request"):  # PRIMARY exception, not the fallback's
        llm_invoke(primary, messages, max_retries=1, base_delay=0.0)
    assert fb_bad.calls == 1
    line = _read_jsonl(trace_dir / "T-FB2" / "llm_calls.jsonl")[-1]
    assert line["finish_reason"] == "error"
    assert "400 bad request" in line["error"]


def test_fallback_resolver_returning_mock_is_discarded(messages, run_ctx, trace_dir, monkeypatch):
    from unittest.mock import MagicMock
    run_ctx(RunContext(ticket_id="T-FB3"))
    primary = FakeLLM(exc=RuntimeError("400 bad request"))
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "fb")
    monkeypatch.setattr(llm_utils, "_FALLBACK_RESOLVER", lambda name: MagicMock())
    with pytest.raises(RuntimeError, match="400 bad request"):
        llm_invoke(primary, messages, max_retries=1, base_delay=0.0)


def test_budget_exceeded_never_falls_back(messages, run_ctx, trace_dir, monkeypatch):
    budget = RunBudget(max_wall_s=0)
    budget._start_wall = time.monotonic() - 10
    run_ctx(RunContext(ticket_id="T-BUDGET", budget=budget))
    primary = FakeLLM()
    fb = FakeLLM(content="FB")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "fb")
    monkeypatch.setattr(llm_utils, "_FALLBACK_RESOLVER", lambda name: fb)

    with pytest.raises(BudgetExceeded):
        llm_invoke(primary, messages, max_retries=3, base_delay=0.0)
    assert primary.calls == 0 and fb.calls == 0


def test_run_cancelled_never_falls_back(messages, run_ctx, trace_dir, monkeypatch):
    ctx = run_ctx(RunContext(ticket_id="T-CANCEL"))
    ctx.request_cancel("stop")
    primary = FakeLLM()
    fb = FakeLLM(content="FB")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "fb")
    monkeypatch.setattr(llm_utils, "_FALLBACK_RESOLVER", lambda name: fb)

    with pytest.raises(RunCancelled):
        llm_invoke(primary, messages, max_retries=3, base_delay=0.0)
    assert primary.calls == 0 and fb.calls == 0


def test_retryable_error_exhausted_with_max_retries_one_raises_without_fallbacks(messages, run_ctx, trace_dir):
    run_ctx(RunContext(ticket_id="T-RETRY"))
    primary = FakeLLM(exc=ConnectionError("Connection reset by peer"))
    with pytest.raises(ConnectionError):
        llm_invoke(primary, messages, max_retries=1, base_delay=0.0)
    assert primary.calls == 1  # no second attempt with max_retries=1
    line = _read_jsonl(trace_dir / "T-RETRY" / "llm_calls.jsonl")[-1]
    assert line["finish_reason"] == "error"


# ── trajectory (item 21) ────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "ts", "phase", "model", "latency_ms", "tokens_in", "tokens_out", "finish_reason",
    "prompt_sha256", "response_sha256", "tool_call_names", "fallback_used", "cost_usd",
}


def test_trajectory_written_under_ticket_dir(messages, run_ctx, trace_dir):
    ctx = run_ctx(RunContext(ticket_id="T-TRAJ"))
    ctx.start_phase("planning")
    llm = FakeLLM(content="answer", meta={"finish_reason": "stop", "model_name": "gemini-2.5-flash"})
    llm_invoke(llm, messages, max_retries=1)

    path = trace_dir / "T-TRAJ" / "llm_calls.jsonl"
    assert path.is_file()
    line = _read_jsonl(path)[-1]
    assert _REQUIRED_KEYS <= set(line.keys())
    assert line["phase"] == "planning"
    assert line["model"] == "gemini-2.5-flash"
    assert line["finish_reason"] == "stop"
    assert line["tokens_in"] > 0 and line["tokens_out"] > 0
    assert len(line["prompt_sha256"]) == 64 and len(line["response_sha256"]) == 64
    assert "prompt" not in line and "response" not in line  # content off by default


def test_trajectory_includes_content_when_env_flag_set(messages, run_ctx, trace_dir, monkeypatch):
    run_ctx(RunContext(ticket_id="T-CONTENT"))
    monkeypatch.setenv("AVIATOR_TRACE_LLM_CONTENT", "1")
    llm_invoke(FakeLLM(content="the-answer"), messages, max_retries=1)
    line = _read_jsonl(trace_dir / "T-CONTENT" / "llm_calls.jsonl")[-1]
    assert "hello there" in line["prompt"]
    assert line["response"] == "the-answer"


def test_trajectory_accepts_plain_string_prompt(run_ctx, trace_dir):
    run_ctx(RunContext(ticket_id="T-STR"))
    llm = FakeLLM(content="ok")
    resp = llm_invoke(llm, "a plain string prompt of some length", max_retries=1)
    assert resp.content == "ok"
    line = _read_jsonl(trace_dir / "T-STR" / "llm_calls.jsonl")[-1]
    assert line["tokens_in"] >= 1


# ── orphan calls (item 23) ──────────────────────────────────────────────────

def test_no_run_context_not_charged_and_logged_to_orphan(messages, trace_dir, caplog, monkeypatch):
    assert current_run_context.get(None) is None
    monkeypatch.setattr(llm_utils, "_ORPHAN_WARNED", set())
    caplog.set_level(logging.WARNING, logger="ticket_to_code.llm_utils")

    llm = FakeLLM()
    for _ in range(2):  # same call site twice -> exactly one warning
        resp = llm_invoke(llm, messages, max_retries=1)
    assert resp.content == "OK"
    assert llm.calls == 2

    orphan = trace_dir / "orphan" / "llm_calls.jsonl"
    assert orphan.is_file()
    assert len(_read_jsonl(orphan)) == 2
    warned = [r for r in caplog.records if "without an active RunContext" in r.getMessage()]
    assert len(warned) == 1


# ── cost accounting (item 22) ───────────────────────────────────────────────

def test_cost_is_charged_from_provider_usage(messages, run_ctx, trace_dir):
    reset_pricing_cache()
    ctx = run_ctx(RunContext(ticket_id="T-COST"))
    llm = FakeLLM(
        content="x",
        meta={
            "model_name": "gemini-2.5-pro",
            "token_usage": {"prompt_tokens": 100_000, "completion_tokens": 10_000, "total_tokens": 110_000},
        },
    )
    llm_invoke(llm, messages, max_retries=1)
    expected = 0.1 * 1.25 + 0.01 * 10.0
    assert ctx.budget.cost_usd == pytest.approx(expected)
    assert ctx.budget.tokens_in == 100_000 and ctx.budget.tokens_out == 10_000
    line = _read_jsonl(trace_dir / "T-COST" / "llm_calls.jsonl")[-1]
    assert line["cost_usd"] == pytest.approx(expected)


def test_cost_budget_ceiling_trips(messages, run_ctx, trace_dir):
    reset_pricing_cache()
    run_ctx(RunContext(ticket_id="T-CEIL", budget=RunBudget(max_cost_usd=1e-9)))
    llm = FakeLLM(
        content="x",
        meta={"model_name": "gemini-2.5-pro", "token_usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}},
    )
    with pytest.raises(BudgetExceeded) as excinfo:
        llm_invoke(llm, messages, max_retries=1)
    assert excinfo.value.kind == "cost_usd"


# ── Vertex GenAI model (item 18): cached client + role-tagged Content ───────

def test_vertex_genai_model_caches_client_and_builds_role_tagged_contents(monkeypatch):
    import sys
    import types as _types
    from langchain_core.messages import SystemMessage

    created = []
    calls = []

    class _Part:
        def __init__(self, text):
            self.text = text

        @classmethod
        def from_text(cls, text):
            return cls(text)

    class _Content:
        def __init__(self, role, parts):
            self.role, self.parts = role, parts

    class _Cfg:
        def __init__(self, **kw):
            self.kw = kw

    class _Thinking:
        def __init__(self, **kw):
            self.kw = kw

    class _Models:
        def generate_content(self, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            return SimpleNamespace(text="hi", usage_metadata=SimpleNamespace(
                prompt_token_count=3, candidates_token_count=1, total_token_count=4,
                thoughts_token_count=0, cached_content_token_count=0,
            ))

    class _Client:
        def __init__(self, **kw):
            created.append(kw)
            self.models = _Models()

    genai_mod = _types.ModuleType("google.genai")
    genai_mod.Client = _Client
    types_mod = _types.ModuleType("google.genai.types")
    types_mod.Content, types_mod.Part = _Content, _Part
    types_mod.GenerateContentConfig, types_mod.ThinkingConfig = _Cfg, _Thinking
    genai_mod.types = types_mod
    google_mod = _types.ModuleType("google")
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)

    from ticket_to_code.runtime.llm_factory import VertexGenAIChatModel
    model = VertexGenAIChatModel(model_name="gemini-2.5-pro", project="proj", location="europe-west4")

    msgs = [SystemMessage(content="sys"), HumanMessage(content="q1"), AIMessage(content="a1"), HumanMessage(content="q2")]
    model._generate(msgs)
    model._generate(msgs, max_tokens=55)

    assert len(created) == 1, "genai.Client must be built once per model instance"
    assert created[0] == {"vertexai": True, "project": "proj", "location": "europe-west4"}
    assert len(calls) == 2
    contents = calls[0]["contents"]
    assert [(c.role, c.parts[0].text) for c in contents] == [("user", "q1"), ("model", "a1"), ("user", "q2")]
    assert calls[0]["config"].kw["system_instruction"] == "sys"
    assert calls[1]["config"].kw["max_output_tokens"] == 55


def test_token_call_record_has_fallback_used_field():
    from ticket_to_code.runtime.run_context import TokenCallRecord
    rec = TokenCallRecord()
    assert rec.fallback_used == ""
    budget = RunBudget()
    budget.charge(tokens_in=1, tokens_out=1, cost_usd=0.0, model="m", fallback_used="fb")
    assert budget._call_log[-1].to_dict()["fallback_used"] == "fb"
