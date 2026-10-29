"""DORMANT: not used by any production code path (only by tests) as of 2026-10-29.
Kept because tests pin its behaviour; wire it in or delete it together with its tests.


Runtime Telemetry & Evaluation Observability Stream ("Evals for AI").

Collects and streams live performance, token economics, and decision metrics:
- Real-time token consumption vs cached tokens (KV-cache hit rate %).
- Per-tool execution latency, call frequencies, and success rates.
- Closed-loop compiler self-healing cycles.
- Decision ledger for human-in-the-loop interactions.
- Evaluates run convergence scores and saves durable reports to `.aviator/evals/`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolTelemetry:
    call_count: int = 0
    total_duration_ms: int = 0
    success_count: int = 0
    failure_count: int = 0

    @property
    def avg_duration_ms(self) -> float:
        return (self.total_duration_ms / self.call_count) if self.call_count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.call_count) if self.call_count > 0 else 1.0


class RuntimeTelemetryTracker:
    """Aggregates multi-dimensional telemetry and outputs live evals for developers and models."""

    def __init__(self, session_id: str, workspace_path: Path):
        self.session_id = session_id
        self.workspace_path = Path(workspace_path)
        self.start_time = time.time()
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_cached = 0
        self.tools: Dict[str, ToolTelemetry] = {}
        self.phase_timings_ms: Dict[str, int] = {}
        self.compiler_runs = 0
        self.compiler_passes = 0
        self.compiler_repairs = 0
        self.clarifications_requested = 0
        self.clarifications_answered = 0
        self.decision_ledger: List[Dict[str, Any]] = []

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> None:
        """Records token metrics from LLM responses."""
        self.tokens_in += prompt_tokens
        self.tokens_out += completion_tokens
        self.tokens_cached += cached_tokens

    def record_tool_call(self, tool_name: str, duration_ms: int, success: bool = True) -> None:
        """Records latency and outcome of a tool execution."""
        stat = self.tools.setdefault(tool_name, ToolTelemetry())
        stat.call_count += 1
        stat.total_duration_ms += duration_ms
        if success:
            stat.success_count += 1
        else:
            stat.failure_count += 1

    def record_compiler_check(self, passed: bool, repaired: bool = False) -> None:
        """Records compiler gate outcomes."""
        self.compiler_runs += 1
        if passed:
            self.compiler_passes += 1
        if repaired:
            self.compiler_repairs += 1

    def record_clarification(self, question: str, response: str, was_fallback: bool = False) -> None:
        """Logs interactive human-in-the-loop questions and answers."""
        self.clarifications_requested += 1
        if not was_fallback:
            self.clarifications_answered += 1
        self.decision_ledger.append({
            "question": question,
            "response": response,
            "was_fallback": was_fallback,
            "timestamp": datetime.now().isoformat(),
        })

    def record_phase_duration(self, phase_name: str, duration_ms: int) -> None:
        """Records elapsed time for a high-level phase."""
        self.phase_timings_ms[phase_name] = self.phase_timings_ms.get(phase_name, 0) + duration_ms

    def export_eval_report(self) -> Dict[str, Any]:
        """Generates a structured evaluation and observability report."""
        total_tokens = self.tokens_in + self.tokens_out
        cache_rate = (self.tokens_cached / (self.tokens_in + self.tokens_cached) * 100) if (self.tokens_in + self.tokens_cached) > 0 else 0.0
        elapsed_s = round(time.time() - self.start_time, 2)

        tool_stats = {
            tname: {
                "calls": t.call_count,
                "avg_ms": round(t.avg_duration_ms, 1),
                "success_rate": round(t.success_rate * 100, 1),
            }
            for tname, t in self.tools.items()
        }

        report = {
            "session_id": self.session_id,
            "elapsed_seconds": elapsed_s,
            "token_economics": {
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "tokens_cached": self.tokens_cached,
                "total_tokens": total_tokens,
                "cache_hit_rate_pct": round(cache_rate, 2),
            },
            "compiler_evals": {
                "total_checks": self.compiler_runs,
                "passes": self.compiler_passes,
                "repairs": self.compiler_repairs,
                "pass_rate_pct": round((self.compiler_passes / self.compiler_runs * 100), 1) if self.compiler_runs > 0 else 100.0,
            },
            "human_in_loop": {
                "clarifications_requested": self.clarifications_requested,
                "clarifications_answered": self.clarifications_answered,
                "decisions": self.decision_ledger,
            },
            "phase_timings_ms": self.phase_timings_ms,
            "tools": tool_stats,
            "timestamp": datetime.now().isoformat(),
        }

        # Persist report to .nemoforge/evals and .aviator/evals
        try:
            for dirname in (".nemoforge", ".aviator"):
                eval_dir = self.workspace_path / dirname / "evals"
                eval_dir.mkdir(parents=True, exist_ok=True)
                report_path = eval_dir / f"{self.session_id}.json"
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to persist eval report to disk: {e}")

        return report
