"""
RunContext — per-run request-scoped telemetry, budget, and health.

Implements B11 (RunContext + GlobalRunBudget) and B12 (DegradationRegistry + RunHealth).

Design principles:
  - One RunContext per ticket run, never shared across concurrent runs.
  - Budget is checked at every LLM call boundary (charge before proceed).
  - DegradationRegistry is monotonic: HEALTHY → DEGRADED → FAILED only.
  - run_health is derived (never stored) and reflects the worst subsystem status.
  - RunRecord is emitted once at the terminal node for telemetry.

Author: Deepak Madgani
Date: July 2026
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class HealthLevel(str, Enum):
    """Monotonic health levels — transitions only go downward."""
    HEALTHY  = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED   = "FAILED"

    def __lt__(self, other: "HealthLevel") -> bool:
        _order = {HealthLevel.HEALTHY: 0, HealthLevel.DEGRADED: 1, HealthLevel.FAILED: 2}
        return _order[self] < _order[other]

    def __le__(self, other: "HealthLevel") -> bool:
        return self == other or self < other

    def is_worse_than(self, other: "HealthLevel") -> bool:
        return self > other  # type: ignore[operator]


# ============================================================================
# EXCEPTIONS
# ============================================================================

class RunCancelled(RuntimeError):
    """Raised at the next LLM boundary after RunContext.request_cancel() was called."""

    def __init__(self, ticket_id: str = "", reason: str = ""):
        self.ticket_id = ticket_id
        self.reason = reason
        detail = f" ticket={ticket_id}" if ticket_id else ""
        detail += f" reason={reason}" if reason else ""
        super().__init__(f"Run cancelled{detail}")


class BudgetExceeded(RuntimeError):
    """Raised when any RunBudget ceiling is breached."""
    def __init__(self, kind: str, limit, actual) -> None:
        self.kind = kind
        self.limit = limit
        self.actual = actual
        super().__init__(
            f"RunBudget exceeded: {kind} limit={limit} actual={actual}"
        )


# ============================================================================
# TOKEN TRACKING — per-call and per-phase telemetry (observational only)
# ============================================================================

@dataclass
class TokenCallRecord:
    """
    Record of a single LLM call's token usage.

    Every field is observational telemetry — never affects workflow decisions.
    The ``source`` fields distinguish provider-reported actuals from estimates.
    """
    call_id:        int   = 0
    phase:          str   = "unknown"
    model:          str   = "unknown"

    input_tokens:   int   = 0
    output_tokens:  int   = 0
    total_tokens:   int   = 0
    cached_tokens:  int   = 0
    thinking_tokens: int  = 0

    input_source:   str   = "none"     # "provider" | "usage_metadata" | "estimated" | "none"
    output_source:  str   = "none"     # "provider" | "usage_metadata" | "estimated" | "none"

    timestamp:      str   = ""
    # Model id answered via the LLM_FALLBACK_MODELS chain ("" when the primary model answered).
    fallback_used:  str   = ""

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class PhaseUsage:
    """Accumulated token usage for a single workflow phase — computed automatically."""
    tokens_in:       int = 0
    tokens_out:      int = 0
    cached_tokens:   int = 0
    thinking_tokens: int = 0
    llm_calls:       int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "thinking_tokens": self.thinking_tokens,
            "llm_calls": self.llm_calls,
        }


# ============================================================================
# BUDGET
# ============================================================================

@dataclass
class RunBudget:
    """
    Hard ceilings on per-run resource consumption (B11).

    All ceilings are configurable so tests can set them small.
    Budget is charged at the LLM wrapper boundary; nodes only *check*.
    """
    max_wall_s:   float = float('inf')   # Wall-clock enforcement disabled — sufficiency controls stopping
    max_llm_calls: int  = 150
    max_tokens:   int   = int(os.getenv("NEMOFORGE_MAX_TOKENS", os.getenv("AVIATOR_MAX_TOKENS", "1500000")))
    max_cost_usd: float = 15.0

    # Live counters (mutated by charge())
    elapsed_s:   float = field(default=0.0, init=False)
    llm_calls:   int   = field(default=0,   init=False)
    tokens_in:   int   = field(default=0,   init=False)
    tokens_out:  int   = field(default=0,   init=False)
    cost_usd:    float = field(default=0.0, init=False)
    tokens_cached: int = field(default=0,   init=False)  # Reused via warm KV prefix cache
    tokens_thinking: int = field(default=0, init=False)  # Reasoning tokens in think/reasoning models
    tokens_excluded: int = field(default=0, init=False)  # Saved via AST skeleton compression
    kv_trace:        List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    _start_wall: float = field(default_factory=time.monotonic, init=False, repr=False)

    # ── Token instrumentation (observational — never affects workflow) ─────
    phase_ledger:    Dict[str, PhaseUsage]   = field(default_factory=dict, init=False, repr=False)
    _call_log:       List[TokenCallRecord]   = field(default_factory=list, init=False, repr=False)
    _current_phase:  str                     = field(default="unknown", init=False, repr=False)
    _call_counter:   int                     = field(default=0, init=False, repr=False)
    on_charge:       Optional[Callable]      = field(default=None, init=False, repr=False)

    def record_kv_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """Record per-stage KV cache state, hit rate, and trace for live UI."""
        if not telemetry:
            return
        self.kv_trace.append(telemetry)
        cached = telemetry.get("cached_tokens", 0)
        if cached > 0:
            self.tokens_cached += cached
        if self.on_charge:
            try:
                self.on_charge(self._snapshot())
            except Exception:
                pass

    def record_savings(self, cached: int = 0, excluded: int = 0) -> None:
        """Record tokens saved/excluded via KV prefix caching or AST skeleton compression."""
        self.tokens_cached += cached
        self.tokens_excluded += excluded
        if self.on_charge:
            try:
                self.on_charge(self._snapshot())
            except Exception:
                pass

    def _refresh_elapsed(self) -> None:
        self.elapsed_s = time.monotonic() - self._start_wall

    def charge(
        self,
        tokens_in:  int   = 0,
        tokens_out: int   = 0,
        cost_usd:   float = 0.0,
        *,
        model:          str = "unknown",
        input_source:   str = "none",
        output_source:  str = "none",
        cached_tokens:  int = 0,
        thinking_tokens: int = 0,
        fallback_used:  str = "",
    ) -> None:
        """
        Record one LLM call's resource consumption and check all ceilings.

        Raises BudgetExceeded if any limit is breached.
        Must be called inside a finally block so partial charges are recorded
        even when the call raises.

        The ``model``, ``input_source``, and ``output_source`` kwargs are
        purely observational telemetry — they never affect budget checks.
        """
        self._refresh_elapsed()
        self.llm_calls  += 1
        self.tokens_in  += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd   += cost_usd
        if cached_tokens > 0:
            self.tokens_cached += cached_tokens
        if thinking_tokens > 0:
            self.tokens_thinking += thinking_tokens

        logger.debug(
            "RunBudget.charge: calls=%d tokens=%d cached=%d thinking=%d cost=%.4f elapsed=%.1fs",
            self.llm_calls, self.tokens_in + self.tokens_out, cached_tokens, thinking_tokens, self.cost_usd, self.elapsed_s,
        )

        # ── Token instrumentation (observational — wrapped in try/except) ──
        try:
            from datetime import datetime as _dt
            phase = self._current_phase or "unknown"

            # Per-phase ledger: auto-creates entry for any new phase
            if phase not in self.phase_ledger:
                self.phase_ledger[phase] = PhaseUsage()
            entry = self.phase_ledger[phase]
            entry.tokens_in  += tokens_in
            entry.tokens_out += tokens_out
            entry.cached_tokens += cached_tokens
            entry.thinking_tokens += thinking_tokens
            entry.llm_calls  += 1

            # Per-call record
            self._call_counter += 1
            call_rec = TokenCallRecord(
                call_id=self._call_counter,
                phase=phase,
                model=model,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                total_tokens=tokens_in + tokens_out,
                cached_tokens=cached_tokens,
                thinking_tokens=thinking_tokens,
                input_source=input_source,
                output_source=output_source,
                timestamp=_dt.now().isoformat(),
                fallback_used=fallback_used or "",
            )
            self._call_log.append(call_rec)

            # Record per-stage KV cache trace entry if cached tokens are present
            if cached_tokens > 0:
                hit_pct = round((cached_tokens / max(1, tokens_in)) * 100.0, 1) if tokens_in > 0 else 100.0
                self.kv_trace.append({
                    "stage": phase,
                    "model_id": model,
                    "cache_state": "WARM_HIT",
                    "cached_tokens": cached_tokens,
                    "new_tokens": max(0, tokens_in - cached_tokens),
                    "cost_savings_pct": hit_pct,
                    "thinking_tokens": thinking_tokens,
                    "timestamp": _dt.now().isoformat(),
                })

            # Fire UI callback (if wired)
            if self.on_charge:
                try:
                    self.on_charge(self._snapshot())
                except Exception:
                    pass  # Never crash the workflow for UI telemetry
        except Exception:
            pass  # Token instrumentation must never affect workflow

        # ── Budget enforcement ──────────────────────────────────────────────
        # Wall-clock enforcement removed — elapsed_s is tracked for telemetry
        # only.  Evidence sufficiency is the normal stopping mechanism.
        if self.llm_calls > self.max_llm_calls:
            raise BudgetExceeded("llm_calls", self.max_llm_calls, self.llm_calls)
        if (self.tokens_in + self.tokens_out) > self.max_tokens:
            raise BudgetExceeded("tokens", self.max_tokens, self.tokens_in + self.tokens_out)
        if self.cost_usd > self.max_cost_usd:
            raise BudgetExceeded("cost_usd", self.max_cost_usd, round(self.cost_usd, 4))

    def check(self) -> None:
        """Refresh elapsed time for telemetry. Wall-clock enforcement removed."""
        self._refresh_elapsed()

    def remaining(self) -> Dict[str, float]:
        self._refresh_elapsed()
        return {
            "wall_s":    max(0.0, self.max_wall_s - self.elapsed_s) if self.max_wall_s != float('inf') else float('inf'),
            "llm_calls": max(0,   self.max_llm_calls - self.llm_calls),
            "tokens":    max(0,   self.max_tokens    - (self.tokens_in + self.tokens_out)),
            "cost_usd":  max(0.0, self.max_cost_usd  - self.cost_usd),
        }

    def budget_exceeded(self) -> bool:
        self._refresh_elapsed()
        return (
            # Wall-clock removed — not a solver constraint
            self.llm_calls > self.max_llm_calls
            or (self.tokens_in + self.tokens_out) > self.max_tokens
            or self.cost_usd > self.max_cost_usd
        )

    # ── Token instrumentation helpers (observational) ─────────────────────

    def _snapshot(self) -> dict:
        """Current budget state as a plain dict — safe to serialize for UI."""
        self._refresh_elapsed()
        gross_total = self.tokens_in + self.tokens_out
        excluded = self.tokens_cached + self.tokens_excluded
        real_costed = max(0, gross_total - self.tokens_cached)
        savings_pct = round((excluded / (gross_total + excluded)) * 100.0, 1) if (gross_total + excluded) > 0 else 0.0
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_thinking": self.tokens_thinking,
            "thinking_tokens": self.tokens_thinking,
            "total_tokens": gross_total,
            "cached_tokens": self.tokens_cached,
            "excluded_tokens": excluded,
            "real_costed_tokens": real_costed,
            "savings_pct": savings_pct,
            "llm_calls": self.llm_calls,
            "elapsed_s": round(self.elapsed_s, 1),
            "current_phase": self._current_phase or "unknown",
            "latest_kv_state": self.kv_trace[-1] if self.kv_trace else None,
            "kv_trace": self.kv_trace[-15:],
            "phase_breakdown": {
                name: usage.to_dict()
                for name, usage in self.phase_ledger.items()
            },
        }

    def get_call_log(self) -> List[dict]:
        """Return all per-call records as plain dicts."""
        return [rec.to_dict() for rec in self._call_log]


# ============================================================================
# DEGRADATION REGISTRY (B12)
# ============================================================================

@dataclass
class DegradationRegistry:
    """
    Tracks the health of every subsystem for this run.

    Transitions are monotonic: HEALTHY → DEGRADED → FAILED.
    All subsystems start HEALTHY; callers call .degrade() on any failure.
    """
    _registry: Dict[str, HealthLevel] = field(default_factory=dict, init=False)
    _reasons:  Dict[str, List[str]]   = field(default_factory=dict, init=False)

    # Subsystems whose FAILED status must block the run from claiming success.
    EVIDENCE_CRITICAL = frozenset({
        "rag_engine",
        "pgvector",
        "neo4j",
        "sqlite_index",
    })

    def register(self, subsystem: str, status: HealthLevel = HealthLevel.HEALTHY) -> None:
        """Explicitly register a subsystem (usually called at init)."""
        if subsystem not in self._registry:
            self._registry[subsystem] = status
        logger.debug("DegradationRegistry.register: %s → %s", subsystem, status.value)

    def degrade(self, subsystem: str, level: HealthLevel, reason: str = "") -> None:
        """
        Downgrade subsystem to *level* (monotonic — cannot improve).
        Logs a warning so the degradation is visible in the trace.
        """
        current = self._registry.get(subsystem, HealthLevel.HEALTHY)
        if level.is_worse_than(current):  # type: ignore[arg-type]
            self._registry[subsystem] = level
            self._reasons.setdefault(subsystem, []).append(reason)
            logger.warning(
                "⚠️  RunHealth: %s → %s | reason: %s",
                subsystem, level.value, reason or "(unspecified)",
            )
        else:
            # Already at same or worse — still append reason for traceability
            self._reasons.setdefault(subsystem, []).append(reason)

    def health(self) -> HealthLevel:
        """Worst health level across all registered subsystems (derived, never stored)."""
        if not self._registry:
            return HealthLevel.HEALTHY
        worst = max(self._registry.values(), key=lambda h: {
            HealthLevel.HEALTHY: 0, HealthLevel.DEGRADED: 1, HealthLevel.FAILED: 2
        }[h])
        return worst

    def has_critical_failure(self) -> bool:
        """True if any evidence-critical subsystem is FAILED."""
        for sub in self.EVIDENCE_CRITICAL:
            if self._registry.get(sub, HealthLevel.HEALTHY) == HealthLevel.FAILED:
                return True
        return False

    def snapshot(self) -> Dict[str, str]:
        """Return a serialisable dict suitable for RunRecord."""
        return {sub: lvl.value for sub, lvl in self._registry.items()}

    def reasons_snapshot(self) -> Dict[str, List[str]]:
        return dict(self._reasons)


# ============================================================================
# PHASE TIMER
# ============================================================================

@dataclass
class _PhaseTimer:
    name: str
    start: float = field(default_factory=time.monotonic)
    duration_ms: Optional[float] = None

    def stop(self) -> float:
        self.duration_ms = (time.monotonic() - self.start) * 1000
        return self.duration_ms


# ============================================================================
# RUN CONTEXT (B11 + B12)
# ============================================================================

@dataclass
class RunContext:
    """
    Per-run request-scoped container (B11 + B12).

    Carries identity, phase timers, degradation registry, and budget.
    Passed by reference; never global.

    Usage in workflow nodes::

        ctx = state.get("run_ctx")
        ctx.start_phase("evidence_collection")
        ...
        ctx.end_phase("evidence_collection")

        # charge at LLM boundary (inside finally so partials are recorded):
        try:
            response = llm.invoke(...)
        finally:
            ctx.budget.charge(tokens_in=..., tokens_out=..., cost_usd=...)
    """
    run_id:    str   = field(default_factory=lambda: str(uuid.uuid4()))
    ticket_id: str   = ""
    repo:      str   = ""
    indexed_commit:  Optional[str] = None
    current_commit:  Optional[str] = None
    final_status:    str = "RUNNING"

    # Sub-objects
    budget:   RunBudget          = field(default_factory=RunBudget)
    registry: DegradationRegistry = field(default_factory=DegradationRegistry)

    # Phase timers
    _phases: Dict[str, _PhaseTimer] = field(default_factory=dict, init=False, repr=False)

    # Accumulated stats for RunRecord
    discovery_candidates: int  = 0
    discovery_truncated:  int  = 0
    discovery_near_miss:  int  = 0
    evidence_promoted:    int  = 0
    evidence_blocked:     int  = 0
    evidence_backed:      bool = False
    grounding_confidence: float = 0.0
    architecture_chunks:  int  = 0
    context_sources_used: List[str] = field(default_factory=list)
    generation_truncations: int = 0
    validation_passed:    Optional[bool] = None
    validation_level:     Optional[str]  = None
    retries:              int  = 0
    hard_stop:            bool = False

    # Cooperative cancellation: set via request_cancel(); checked at LLM boundaries.
    cancelled:            bool = False
    cancel_reason:        str  = ""

    # ── Cancellation API ──────────────────────────────────────────────────────

    def request_cancel(self, reason: str = "") -> None:
        """Flag this run as cancelled. llm_invoke raises RunCancelled on its next attempt."""
        if not self.cancelled:
            logger.info("RunContext.request_cancel: ticket=%s reason=%s", self.ticket_id, reason or "-")
        self.cancelled = True
        if reason:
            self.cancel_reason = reason

    def raise_if_cancelled(self) -> None:
        """Raise RunCancelled when request_cancel() has been called."""
        if self.cancelled:
            raise RunCancelled(self.ticket_id, self.cancel_reason)

    # ── Phase timer API ───────────────────────────────────────────────────────

    def start_phase(self, name: str) -> None:
        self._phases[name] = _PhaseTimer(name)
        # Token instrumentation: tell RunBudget which phase is active
        try:
            self.budget._current_phase = name
        except Exception:
            pass  # Never crash for instrumentation
        logger.debug("RunContext.start_phase: %s", name)

    def end_phase(self, name: str) -> float:
        """Stop timer for *name*. Returns duration in ms. Safe to call even if start missed."""
        timer = self._phases.get(name)
        if timer is None:
            logger.debug("RunContext.end_phase: '%s' had no start — skipped", name)
            return 0.0
        ms = timer.stop()
        # Token instrumentation: reset current phase to "between_phases"
        try:
            if self.budget._current_phase == name:
                self.budget._current_phase = "between_phases"
        except Exception:
            pass  # Never crash for instrumentation
        logger.debug("RunContext.end_phase: %s = %.1f ms", name, ms)
        return ms

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Context manager that auto-ends the phase in finally."""
        self.start_phase(name)
        try:
            yield
        finally:
            self.end_phase(name)

    # ── Convenience delegations ───────────────────────────────────────────────

    def degrade(self, subsystem: str, level: HealthLevel, reason: str = "") -> None:
        self.registry.degrade(subsystem, level, reason)

    def health(self) -> HealthLevel:
        return self.registry.health()

    def check_budget(self) -> None:
        """Raise BudgetExceeded if wall-clock limit exceeded. Call at node boundaries."""
        self.budget.check()

    def record(self, key: str, value) -> None:
        """Generic attribute setter for one-off telemetry updates."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            logger.debug("RunContext.record: unknown key '%s' (ignoring)", key)

    # ── RunRecord emission ────────────────────────────────────────────────────

    def to_record(self) -> "RunRecord":
        phase_timings = {
            name: round(timer.duration_ms or 0.0)
            for name, timer in self._phases.items()
        }
        self.budget._refresh_elapsed()
        # Token instrumentation: include per-phase and per-call breakdown
        try:
            phase_breakdown = {
                name: usage.to_dict()
                for name, usage in self.budget.phase_ledger.items()
            }
            call_log = self.budget.get_call_log()
        except Exception:
            phase_breakdown = {}
            call_log = []
        return RunRecord(
            run_id=self.run_id,
            ticket_id=self.ticket_id,
            repo=self.repo,
            indexed_commit=self.indexed_commit,
            current_commit=self.current_commit,
            final_status=self.final_status,
            run_health=self.health().value,
            degradations=self.registry.snapshot(),
            phase_timings_ms=phase_timings,
            llm_calls=self.budget.llm_calls,
            tokens_in=self.budget.tokens_in,
            tokens_out=self.budget.tokens_out,
            cost_usd=round(self.budget.cost_usd, 4),
            discovery_candidates=self.discovery_candidates,
            discovery_truncated=self.discovery_truncated,
            discovery_near_miss=self.discovery_near_miss,
            evidence_promoted=self.evidence_promoted,
            evidence_blocked=self.evidence_blocked,
            evidence_backed=self.evidence_backed,
            grounding_confidence=self.grounding_confidence,
            architecture_chunks=self.architecture_chunks,
            context_sources_used=self.context_sources_used,
            generation_truncations=self.generation_truncations,
            validation_passed=self.validation_passed,
            validation_level=self.validation_level,
            retries=self.retries,
            hard_stop=self.hard_stop,
            budget_exceeded=self.budget.budget_exceeded(),
            phase_token_breakdown=phase_breakdown,
            call_log=call_log,
        )


# ============================================================================
# RUN RECORD (emitted at terminal node)
# ============================================================================

@dataclass
class RunRecord:
    """
    Immutable audit record emitted at every terminal node.

    Required fields match the telemetry schema from the production readiness
    review (Part 6).  All alert conditions are documented inline.
    """
    run_id:        str
    ticket_id:     str
    repo:          str
    indexed_commit: Optional[str]
    current_commit: Optional[str]
    final_status:  str
    run_health:    str                 # HEALTHY | DEGRADED | FAILED

    degradations:  Dict[str, str]      # subsystem → HealthLevel.value
    phase_timings_ms: Dict[str, int]

    # LLM cost
    llm_calls:     int
    tokens_in:     int
    tokens_out:    int
    cost_usd:      float

    # Discovery
    discovery_candidates: int
    discovery_truncated:  int
    discovery_near_miss:  int

    # Evidence
    evidence_promoted: int
    evidence_blocked:  int
    evidence_backed:   bool           # CRITICAL ALERT: False + success → confident-wrong risk

    # Grounding
    grounding_confidence: float

    # Context
    architecture_chunks: int          # ALERT if 0
    context_sources_used: List[str]

    # Generation
    generation_truncations: int

    # Validation
    validation_passed: Optional[bool]
    validation_level:  Optional[str]  # "FULL" | "STATIC" | None

    # Retry / budget
    retries:        int
    hard_stop:      bool
    budget_exceeded: bool

    # Token instrumentation (per-phase and per-call breakdown)
    phase_token_breakdown: Dict[str, dict] = field(default_factory=dict)
    call_log:              List[dict]       = field(default_factory=list)

    # Derived alerts (computed on to_dict())
    def alerts(self) -> List[str]:
        """Return list of golden-signal alert strings that should page."""
        a: List[str] = []
        if self.run_health == HealthLevel.FAILED.value:
            a.append("run_health=FAILED")
        if self.architecture_chunks == 0:
            a.append("architecture_chunks=0")
        if not self.evidence_backed and self.final_status == "COMPLETE":
            a.append("CRITICAL: evidence_backed=False with status=COMPLETE (confident-wrong risk)")
        if self.budget_exceeded:
            a.append("budget_exceeded")
        return a

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d["alerts"] = self.alerts()
        return d


import contextvars
current_run_context: contextvars.ContextVar[Optional[RunContext]] = contextvars.ContextVar("current_run_context", default=None)

__all__ = [
    "HealthLevel",
    "RunCancelled",
    "BudgetExceeded",
    "TokenCallRecord",
    "PhaseUsage",
    "RunBudget",
    "DegradationRegistry",
    "RunContext",
    "RunRecord",
    "current_run_context",
]
