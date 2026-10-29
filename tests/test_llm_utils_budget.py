"""llm_invoke must honour the active RunContext: wall-clock budget and cancellation.

Regression for the dead wall-clock check: `current_run_context` used to be
referenced before its (function-local) import, so the deadline check raised
UnboundLocalError which was swallowed and the budget never fired.
"""

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ticket_to_code import llm_utils
from ticket_to_code.llm_utils import llm_invoke
from ticket_to_code.runtime.run_context import (
    BudgetExceeded,
    RunBudget,
    RunCancelled,
    RunContext,
    current_run_context,
)


class _FakeLLM:
    """Minimal stand-in; llm_invoke rejects unittest.mock objects by design."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return AIMessage(content="OK")


@pytest.fixture
def messages():
    return [HumanMessage(content="hello")]


@pytest.fixture
def run_context_token():
    """Reset the contextvar after each test so state never leaks between tests."""
    tokens = []

    def _set(ctx):
        tokens.append(current_run_context.set(ctx))
        return ctx

    yield _set
    for tok in reversed(tokens):
        current_run_context.reset(tok)


def test_module_level_imports_resolve():
    """The names used by the deadline check must be bound at module scope."""
    assert llm_utils.current_run_context is current_run_context
    assert llm_utils.BudgetExceeded is BudgetExceeded
    assert llm_utils.RunCancelled is RunCancelled


def test_wall_clock_budget_exceeded_raises_before_invoke(messages, run_context_token):
    budget = RunBudget(max_wall_s=0)
    budget._start_wall = time.monotonic() - 10  # elapsed is already well past the ceiling
    ctx = run_context_token(RunContext(ticket_id="T-WALL", budget=budget))

    llm = _FakeLLM()
    with pytest.raises(BudgetExceeded):
        llm_invoke(llm, messages, max_retries=3, base_delay=0.0)

    assert llm.calls == 0, "deadline must be enforced before the provider is called"
    assert ctx.budget.elapsed_s > 0


def test_wall_clock_within_budget_invokes_normally(messages, run_context_token):
    run_context_token(RunContext(ticket_id="T-OK", budget=RunBudget(max_wall_s=3600)))

    llm = _FakeLLM()
    resp = llm_invoke(llm, messages, max_retries=1)

    assert resp.content == "OK"
    assert llm.calls == 1


def test_request_cancel_raises_run_cancelled(messages, run_context_token):
    ctx = run_context_token(RunContext(ticket_id="T-CANCEL"))
    ctx.request_cancel("user pressed stop")

    llm = _FakeLLM()
    with pytest.raises(RunCancelled) as excinfo:
        llm_invoke(llm, messages, max_retries=3, base_delay=0.0)

    assert llm.calls == 0
    assert excinfo.value.ticket_id == "T-CANCEL"
    assert "user pressed stop" in str(excinfo.value)
    assert ctx.cancelled is True


def test_run_cancelled_is_not_retried(messages, run_context_token):
    """Cancellation set between attempts must stop the retry loop immediately."""
    ctx = run_context_token(RunContext(ticket_id="T-MIDWAY"))

    class _FlakyThenCancelled:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages, **kwargs):
            self.calls += 1
            ctx.request_cancel()
            raise ConnectionError("Connection reset by peer")  # retryable

    llm = _FlakyThenCancelled()
    with pytest.raises(RunCancelled):
        llm_invoke(llm, messages, max_retries=5, base_delay=0.0)

    assert llm.calls == 1
