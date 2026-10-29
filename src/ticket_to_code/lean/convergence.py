"""Convergence guarantee: progress is tracked by failure signature, not by call count.

A run continues as long as each failure differs from the last (progress). It halts the
instant a failure signature repeats (zero progress) or oscillates (A→B→A). This is what
makes the 8-hour edit loop structurally impossible without an arbitrary cap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Literal, Optional


class Action(Enum):
    CONTINUE_REPAIR = "continue_repair"
    STOP_UNPROVEN = "stop_unproven"
    SUCCESS = "success"


@dataclass
class StructuredFailure:
    kind: Literal["compile", "test", "ui_order", "console", "network", "patch_reject", "infra"]
    file: Optional[str] = None
    expected: Any = None
    actual: Any = None
    hint: str = ""

    def signature(self) -> str:
        payload = json.dumps(
            {"kind": self.kind, "file": self.file, "expected": self.expected, "actual": self.actual},
            sort_keys=True, default=str,
        )
        return f"{self.kind}:{hashlib.sha1(payload.encode()).hexdigest()[:12]}"

    def repair_hint(self) -> str:
        if self.hint:
            return self.hint
        if self.kind == "ui_order":
            return f"grid column order wrong: expected {self.expected}, actual {self.actual}"
        if self.kind == "compile":
            return f"compile error in {self.file}: {self.actual}"
        return f"{self.kind} failure: {self.actual}"


@dataclass
class ConvergenceTracker:
    seen: List[str] = field(default_factory=list)
    reason: str = ""

    def evaluate(self, failure: StructuredFailure) -> Action:
        sig = failure.signature()
        if not self.seen:
            self.seen.append(sig)
            return Action.CONTINUE_REPAIR
        if sig == self.seen[-1]:
            self.reason = f"zero progress: identical failure '{sig}' repeated"
            return Action.STOP_UNPROVEN
        if sig in self.seen:
            self.reason = f"oscillation: failure '{sig}' seen before — patch is flip-flopping"
            return Action.STOP_UNPROVEN
        self.seen.append(sig)
        return Action.CONTINUE_REPAIR
