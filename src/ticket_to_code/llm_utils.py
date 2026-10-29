"""
LLM utility helpers shared across all agents.
"""
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# run_context has no dependency back on this module, so a plain import is safe;
# the guard only protects against a future circular import making these names
# unavailable at import time (the checks below then degrade to no-ops).
try:
    from ticket_to_code.runtime.run_context import BudgetExceeded, RunCancelled, current_run_context
except ImportError:  # pragma: no cover - defensive
    class BudgetExceeded(RuntimeError):  # type: ignore[no-redef]
        pass

    class RunCancelled(RuntimeError):  # type: ignore[no-redef]
        pass

    current_run_context = None  # type: ignore[assignment]

# Network / transient error types to retry on
_RETRYABLE_ERRORS = (
    "ReadError", "ConnectError", "RemoteProtocolError",
    "WinError 10054", "WinError 10053", "WinError 10061",
    "Connection reset", "forcibly closed", "Connection refused",
    "503", "429", "ResourceExhausted",
    # OAuth2 / Vertex AI token-refresh timeouts (ConnectTimeoutError wraps urllib3)
    "ConnectTimeout", "TimeoutError", "timed out",
    "Max retries exceeded",
)


def _find_run_context():
    """Resolve the active RunContext (contextvar first, then the *active* ticket's transient store).

    There is deliberately NO "most recent ticket" fallback: when the context var
    is lost and no ticket is active we charge nothing rather than billing an
    unrelated run.
    """
    run_ctx = current_run_context.get(None) if current_run_context is not None else None
    if run_ctx is not None:
        return run_ctx
    try:
        from ticket_to_code.workflow import _transient_store, _transient_lock, get_active_ticket_id
        active_tid = get_active_ticket_id()
        with _transient_lock:
            if active_tid and active_tid in _transient_store and "run_ctx" in _transient_store[active_tid]:
                return _transient_store[active_tid]["run_ctx"]
    except Exception:
        pass
    return None


# Call sites that invoked an LLM without any RunContext (warned once per process each).
_ORPHAN_WARNED: set = set()
# Test hook: callable(model_name) -> llm. When None, LLMRegistry.get_llm(model_name=...) is used.
_FALLBACK_RESOLVER = None


def _is_mock(obj) -> bool:
    return getattr(type(obj), "__module__", "").startswith("unittest.mock")


def _fallback_models() -> list:
    raw = os.getenv("LLM_FALLBACK_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _resolve_fallback_llm(model_name: str):
    """Build the fallback LLM for ``model_name``; returns None when unavailable."""
    try:
        if _FALLBACK_RESOLVER is not None:
            cand = _FALLBACK_RESOLVER(model_name)
        else:
            from ticket_to_code.runtime.llm_factory import LLMRegistry
            cand = LLMRegistry.get_llm(model_name=model_name)
    except Exception as exc:
        logger.warning("LLM fallback model %r could not be initialised: %s", model_name, exc)
        return None
    if cand is None or _is_mock(cand):
        return None
    return cand


def _warn_orphan_call_site() -> None:
    """Log once per process per call site when an LLM call has no RunContext."""
    try:
        import inspect
        frame = inspect.stack()[2]
        key = f"{frame.filename}:{frame.lineno}"
    except Exception:
        key = "unknown"
    if key in _ORPHAN_WARNED:
        return
    _ORPHAN_WARNED.add(key)
    logger.warning("llm_invoke called without an active RunContext (not budgeted) at %s", key)


def _prompt_text(messages) -> str:
    """Canonical text of a prompt (str or message list) for hashing/tracing."""
    if isinstance(messages, str):
        return messages
    try:
        parts = []
        for m in messages:
            if isinstance(m, str):
                parts.append(m)
            else:
                parts.append(f"{type(m).__name__}:{getattr(m, 'content', '')}")
        return "\n".join(parts)
    except Exception:
        return str(messages)


def _estimate_input_tokens(messages) -> int:
    try:
        if isinstance(messages, str):
            total_chars = len(messages)
        else:
            total_chars = sum(len(getattr(m, "content", m if isinstance(m, str) else "") or "") for m in messages)
        return max(1, total_chars // 4) if total_chars > 0 else 0
    except Exception:
        return 0


def _write_trajectory(ticket_id, record: dict, messages=None, response_text=None) -> None:
    """Append one JSON line to <trace_dir>/llm_calls.jsonl. Never raises."""
    try:
        import hashlib
        import json
        from datetime import datetime, timezone
        from ticket_to_code.tracing.paths import get_trace_dir

        prompt_text = _prompt_text(messages) if messages is not None else ""
        resp_text = response_text if response_text is not None else ""
        line = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8", errors="ignore")).hexdigest(),
            "response_sha256": hashlib.sha256(str(resp_text).encode("utf-8", errors="ignore")).hexdigest(),
        }
        line.update(record)
        if os.getenv("AVIATOR_TRACE_LLM_CONTENT", "").strip() == "1":
            line["prompt"] = prompt_text
            line["response"] = resp_text
        path = get_trace_dir(ticket_id or "orphan", create=True) / "llm_calls.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - logging must never break a call
        logger.debug("trajectory log skipped: %s", exc)


def _invoke_once(llm, messages, effective_max_tokens):
    if effective_max_tokens is not None:
        try:
            return llm.invoke(messages, max_tokens=effective_max_tokens)
        except TypeError:
            try:
                return llm.invoke(messages, max_output_tokens=effective_max_tokens)
            except TypeError:
                return llm.invoke(messages)
    return llm.invoke(messages)


def _check_deadline() -> None:
    """Cooperative cancellation + hard wall-clock deadline (checked before every attempt)."""
    try:
        _rc = _find_run_context()
        if _rc is not None and getattr(_rc, "cancelled", False):
            raise RunCancelled(
                getattr(_rc, "ticket_id", ""), getattr(_rc, "cancel_reason", "")
            )
        if _rc is not None and getattr(_rc, "budget", None) is not None:
            _rc.budget._refresh_elapsed()
            if _rc.budget.elapsed_s > _rc.budget.max_wall_s:
                raise BudgetExceeded(
                    "wall_s", _rc.budget.max_wall_s, round(_rc.budget.elapsed_s, 1)
                )
    except (BudgetExceeded, RunCancelled):
        raise
    except Exception:
        pass  # deadline check must never break normal invocation


def _attempt_with_retries(llm, messages, max_retries: int, base_delay: float, effective_max_tokens):
    """Run the provider call with the transient-error retry policy. Returns the response or raises."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            _check_deadline()
            return _invoke_once(llm, messages, effective_max_tokens)
        except (BudgetExceeded, RunCancelled):
            raise
        except Exception as exc:
            exc_str = f"{type(exc).__name__}: {str(exc)} {repr(exc)}"
            # DNS / endpoint-unreachable failures are NOT transient: retrying
            # an offline network burned 7+ hours in a 2026-09-21 benchmark run.
            is_fatal = any(
                e in exc_str.lower()
                for e in ("nameresolutionerror", "getaddrinfo failed", "failed to resolve")
            )
            if is_fatal:
                if attempt < 3 and max_retries > 1:
                    wait_s = 2.0 * (attempt + 1)
                    logger.warning(f"LLM DNS hiccup on attempt {attempt + 1}: {exc}. Retrying in {wait_s}s...")
                    time.sleep(wait_s)
                    last_exc = exc
                    continue
                logger.error(f"LLM endpoint unreachable (DNS failure) — aborting after retries: {exc}")
                raise
            is_retryable = any(e.lower() in exc_str.lower() for e in _RETRYABLE_ERRORS)
            # Also catch by exception class name
            is_retryable = is_retryable or any(
                e.lower() in type(exc).__name__.lower() for e in ("ReadError", "ConnectError", "RemoteProtocolError", "Timeout", "Connection")
            )
            if is_retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"LLM network error (attempt {attempt + 1}/{max_retries}): {exc}. "
                    f"Retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
                last_exc = exc
                continue
            raise  # non-retryable or last attempt
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM invocation failed (max_retries=0 or unknown error)")


def llm_invoke(
    llm,
    messages,
    max_retries: int = 5,
    base_delay: float = 5.0,
    max_tokens: Optional[int] = None,
):
    """
    Invoke an LLM with automatic retry on transient network/API errors.

    Retries up to `max_retries` times with exponential backoff.
    Raises the original exception if all retries fail.

    Args:
        llm: LangChain or custom chat model.
        messages: List of BaseMessage objects.
        max_retries: Maximum retry attempts on transient network errors.
        base_delay: Initial exponential backoff delay in seconds.
        max_tokens: Optional completion token limit. If None, respects
                    AVIATOR_DEFAULT_MAX_OUTPUT_TOKENS environment variable.
    """
    llm_type = type(llm)
    llm_module = getattr(llm_type, "__module__", "")
    if llm_module.startswith("unittest.mock"):
        raise RuntimeError(
            "Invalid mocked LLM instance detected in runtime path "
            f"({llm_type.__name__}). Check LLMRegistry wiring and remove test mocks."
        )

    # Determine effective max_tokens
    effective_max_tokens = max_tokens
    if effective_max_tokens is None:
        _env_max = os.getenv("AVIATOR_DEFAULT_MAX_OUTPUT_TOKENS", "").strip()
        if _env_max.isdigit():
            effective_max_tokens = int(_env_max)

    # Resolve the run context ONCE up front; it decides budgeting and the trajectory ticket id.
    run_ctx = _find_run_context()
    ticket_id = getattr(run_ctx, "ticket_id", "") if run_ctx is not None else ""
    phase = "unknown"
    try:
        if run_ctx is not None and getattr(run_ctx, "budget", None) is not None:
            phase = run_ctx.budget._current_phase or "unknown"
    except Exception:
        pass

    fallback_used = ""
    answering_llm = llm
    t0 = time.monotonic()
    try:
        resp = _attempt_with_retries(llm, messages, max_retries, base_delay, effective_max_tokens)
    except (BudgetExceeded, RunCancelled):
        raise  # never fall back on budget/cancellation
    except Exception as primary_exc:
        resp = None
        for fb_model in _fallback_models():
            fb_llm = _resolve_fallback_llm(fb_model)
            if fb_llm is None:
                continue
            logger.warning(
                "LLM primary call failed (%s: %s); trying fallback model %r",
                type(primary_exc).__name__, str(primary_exc)[:200], fb_model,
            )
            try:
                resp = _attempt_with_retries(fb_llm, messages, 1, base_delay, effective_max_tokens)
                fallback_used = fb_model
                answering_llm = fb_llm
                break
            except (BudgetExceeded, RunCancelled):
                raise
            except Exception as fb_exc:
                logger.warning("LLM fallback model %r failed too: %s", fb_model, str(fb_exc)[:200])
                continue
        if resp is None:
            _write_trajectory(
                ticket_id,
                {
                    "phase": phase,
                    "model": getattr(llm, "model_name", None) or getattr(llm, "model", "unknown"),
                    "latency_ms": round((time.monotonic() - t0) * 1000.0, 1),
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "finish_reason": "error",
                    "error": f"{type(primary_exc).__name__}: {str(primary_exc)[:500]}",
                    "fallback_used": fallback_used,
                    "cost_usd": 0.0,
                },
                messages=messages,
            )
            raise primary_exc

    latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
    if fallback_used:
        try:
            meta = getattr(resp, "response_metadata", None)
            if isinstance(meta, dict):
                meta["fallback_used"] = fallback_used
        except Exception:
            pass

    # ── Usage, cost, budget charge, trajectory ───────────────────────────
    usage = extract_token_usage(resp)
    input_tokens = usage.get("prompt_tokens", 0)
    input_source = usage.get("input_source", "none")
    if input_tokens == 0:
        est = _estimate_input_tokens(messages)
        if est:
            input_tokens, input_source = est, "estimated"
    output_tokens = usage.get("completion_tokens", 0)
    model_name = usage.get("model", "unknown")
    if not model_name or model_name == "unknown":
        model_name = (
            getattr(answering_llm, "model_name", None)
            or getattr(answering_llm, "model", None)
            or fallback_used
            or "unknown"
        )
    try:
        from ticket_to_code.runtime.pricing import estimate_cost_usd
        cost_usd = estimate_cost_usd(str(model_name), input_tokens, output_tokens)
    except Exception:
        cost_usd = 0.0

    finish_reason = None
    tool_call_names = None
    try:
        meta = getattr(resp, "response_metadata", None)
        if isinstance(meta, dict):
            finish_reason = meta.get("finish_reason")
        tcs = getattr(resp, "tool_calls", None)
        if tcs:
            tool_call_names = [
                (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)) for tc in tcs
            ]
    except Exception:
        pass

    _write_trajectory(
        ticket_id,
        {
            "phase": phase,
            "model": model_name,
            "latency_ms": latency_ms,
            "tokens_in": input_tokens,
            "tokens_out": output_tokens,
            "finish_reason": finish_reason,
            "tool_call_names": tool_call_names,
            "fallback_used": fallback_used,
            "cost_usd": cost_usd,
        },
        messages=messages,
        response_text=getattr(resp, "content", None) if resp is not None else None,
    )

    if run_ctx is None:
        _warn_orphan_call_site()
    elif getattr(run_ctx, "budget", None) is not None:
        try:
            run_ctx.budget.charge(
                tokens_in=input_tokens,
                tokens_out=output_tokens,
                cost_usd=cost_usd,
                model=str(model_name),
                input_source=input_source,
                output_source=usage.get("output_source", "none"),
                cached_tokens=usage.get("cached_tokens", 0),
                thinking_tokens=usage.get("thinking_tokens", 0),
                fallback_used=fallback_used,
            )
        except BudgetExceeded:
            raise
        except TypeError:
            # Older RunBudget without the fallback_used kwarg.
            run_ctx.budget.charge(
                tokens_in=input_tokens,
                tokens_out=output_tokens,
                cost_usd=cost_usd,
                model=str(model_name),
                input_source=input_source,
                output_source=usage.get("output_source", "none"),
                cached_tokens=usage.get("cached_tokens", 0),
                thinking_tokens=usage.get("thinking_tokens", 0),
            )
        except Exception as _charge_err:
            logger.debug(f"Budget charge non-fatal: {_charge_err}")

    return resp


def extract_token_usage(response) -> dict:
    """
    Extract token usage metrics from a LangChain AIMessage response.

    Uses a three-level fallback hierarchy with full KV cache and reasoning extraction:
      1. ``response_metadata.token_usage`` (provider-reported actual)
      2. ``usage_metadata`` (newer LangChain field, also provider-reported)
      3. Character-based estimation (``len(content) // 4``, marked as estimated)

    Returns:
        dict with keys: model, prompt_tokens, completion_tokens, total_tokens,
                        cached_tokens, thinking_tokens, input_source, output_source
    """
    usage = {
        "model": "unknown",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "thinking_tokens": 0,
        "input_source": "none",
        "output_source": "none",
    }
    
    try:
        # ── Level 1: response_metadata (standard LangChain / Google GenAI / OpenAI) ────
        if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
            meta = response.response_metadata
            usage["model"] = (
                meta.get("model_name")
                or meta.get("model", "unknown")
            )
            
            token_usage = meta.get("token_usage", {})
            if isinstance(token_usage, dict) and token_usage:
                pt = token_usage.get("prompt_tokens", 0)
                ct = token_usage.get("completion_tokens", 0)
                tt = token_usage.get("total_tokens", 0)
                if pt or ct or tt:
                    usage["prompt_tokens"] = pt
                    usage["completion_tokens"] = ct
                    usage["total_tokens"] = tt or (pt + ct)
                    if pt:
                        usage["input_source"] = "provider"
                    if ct:
                        usage["output_source"] = "provider"
                
                # OpenAI cached & reasoning tokens
                pt_details = token_usage.get("prompt_tokens_details", {})
                if isinstance(pt_details, dict) and pt_details.get("cached_tokens"):
                    usage["cached_tokens"] = max(usage["cached_tokens"], int(pt_details["cached_tokens"]))
                ct_details = token_usage.get("completion_tokens_details", {})
                if isinstance(ct_details, dict) and ct_details.get("reasoning_tokens"):
                    usage["thinking_tokens"] = max(usage["thinking_tokens"], int(ct_details["reasoning_tokens"]))

            # Google GenAI / Vertex AI metadata inside response_metadata
            um_meta = meta.get("usage_metadata", {})
            if isinstance(um_meta, dict):
                cached = um_meta.get("cached_content_token_count", 0)
                if cached:
                    usage["cached_tokens"] = max(usage["cached_tokens"], int(cached))
                thoughts = um_meta.get("thoughts_token_count", 0)
                if thoughts:
                    usage["thinking_tokens"] = max(usage["thinking_tokens"], int(thoughts))
                if usage["prompt_tokens"] == 0 and um_meta.get("prompt_token_count"):
                    usage["prompt_tokens"] = int(um_meta["prompt_token_count"])
                    usage["input_source"] = "response_metadata"
                if usage["completion_tokens"] == 0 and um_meta.get("candidates_token_count"):
                    usage["completion_tokens"] = int(um_meta["candidates_token_count"])
                    usage["output_source"] = "response_metadata"
                if usage["total_tokens"] == 0 and um_meta.get("total_token_count"):
                    usage["total_tokens"] = int(um_meta["total_token_count"])

            # Direct cached tokens or Anthropic fields in response_metadata
            if meta.get("cached_content_token_count"):
                usage["cached_tokens"] = max(usage["cached_tokens"], int(meta["cached_content_token_count"]))
            if meta.get("cache_read_input_tokens"):
                usage["cached_tokens"] = max(usage["cached_tokens"], int(meta["cache_read_input_tokens"]))
            if meta.get("thoughts_token_count"):
                usage["thinking_tokens"] = max(usage["thinking_tokens"], int(meta["thoughts_token_count"]))

        # ── Level 2: usage_metadata (newer LangChain) ──────────
        if hasattr(response, "usage_metadata"):
            um = response.usage_metadata
            if isinstance(um, dict):
                it = um.get("input_tokens", 0)
                ot = um.get("output_tokens", 0)
                tt = um.get("total_tokens", 0)
                if (it or ot or tt) and (usage["prompt_tokens"] == 0 and usage["completion_tokens"] == 0):
                    usage["prompt_tokens"] = it
                    usage["completion_tokens"] = ot
                    usage["total_tokens"] = tt or (it + ot)
                    if it:
                        usage["input_source"] = "usage_metadata"
                    if ot:
                        usage["output_source"] = "usage_metadata"
                
                # LangChain standard token details
                it_details = um.get("input_token_details", {})
                if isinstance(it_details, dict):
                    cached_val = it_details.get("cache_read", 0) or it_details.get("cached_tokens", 0)
                    if cached_val:
                        usage["cached_tokens"] = max(usage["cached_tokens"], int(cached_val))
                ot_details = um.get("output_token_details", {})
                if isinstance(ot_details, dict):
                    reasoning_val = ot_details.get("reasoning", 0) or ot_details.get("thinking", 0)
                    if reasoning_val:
                        usage["thinking_tokens"] = max(usage["thinking_tokens"], int(reasoning_val))

                if um.get("cached_content_token_count"):
                    usage["cached_tokens"] = max(usage["cached_tokens"], int(um["cached_content_token_count"]))
                if um.get("cached_tokens"):
                    usage["cached_tokens"] = max(usage["cached_tokens"], int(um["cached_tokens"]))
                if um.get("thinking_tokens"):
                    usage["thinking_tokens"] = max(usage["thinking_tokens"], int(um["thinking_tokens"]))

        # ── Level 3: Estimate output tokens from response content ──────────
        if usage["completion_tokens"] == 0:
            content = getattr(response, "content", "")
            if content:
                usage["completion_tokens"] = max(1, len(content) // 4)
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                usage["output_source"] = "estimated"

        if usage["total_tokens"] == 0:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    except Exception as e:
        logger.debug(f"Failed to extract token usage: {e}")
        
    return usage


def expand_ticket_keywords(ticket_text: str) -> dict:
    """
    Expand key concepts (versions, config keys, identifiers) from ticket text into
    common codebase permutations. The LLM determines the appropriate extraction level.
    Returns a dict with 'current_state_literals' and 'desired_state_literals'.
    """
    import json
    import re
    from ticket_to_code.runtime.llm_factory import LLMRegistry
    from langchain_core.messages import SystemMessage, HumanMessage
    
    llm = LLMRegistry.get_llm(assistant=False)
    
    prompt = f"""
You are an expert software engineer extracting literals from a ticket description.
Your task is to identify key terms (e.g. versions, configuration keys, specific names) and expand them into common codebase permutations.

Ticket Text:
{ticket_text}

First, determine the appropriate level of extraction based on the ticket context:
- low: For general bugs or UI changes. Only extract highly obvious, explicit values. Minimal permutations.
- medium: For general API changes or features. Moderate permutations (e.g., camelCase, snake_case).
- high: For version bumps or configuration updates. Aggressively extract all identifiers and extensively permute them (e.g., dots to underscores, flat numbers like 260300).

Then, perform the extraction and return ONLY a JSON object in this format:
{{
  "current_state_literals": ["list of strings that might represent the current state in the code"],
  "desired_state_literals": ["list of strings that might represent the desired/new state in the code"]
}}
"""
    try:
        resp = llm_invoke(llm, [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=prompt)])
        content = getattr(resp, "content", "")
        # extract JSON
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fenced:
            raw = fenced.group(1)
        else:
            brace = re.search(r"\{.*\}", content, re.DOTALL)
            raw = brace.group(0) if brace else "{}"
        
        data = json.loads(raw)
        return {
            "current_state_literals": data.get("current_state_literals", []),
            "desired_state_literals": data.get("desired_state_literals", [])
        }
    except Exception as exc:
        logger.warning(f"Failed to expand ticket keywords: {exc}")
        return {"current_state_literals": [], "desired_state_literals": []}
