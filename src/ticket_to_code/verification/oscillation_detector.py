"""
Oscillation Detector & Infinite-Loop Prevention

Grounded in AI IDE Verification Architecture:
1. Normalized Diagnostic Hashing: Hashes the sorted set of compiler/linter error
   tuples {file, code, stripped_message}, explicitly stripping volatile line numbers
   and memory addresses.
2. Cycle Detection: Catches flip-flop behavior where fixing error A reintroduces error B
   (A -> B -> A).
3. Stagnation Detection: Catches zero-impact edits where the error set is identical before and after.
4. Hard-Stop Circuit Breaker: Enforces maximum recovery turns (default: 3) and commands
   immediate rollback to the last known good state upon detected cycle or stagnation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Set

logger = logging.getLogger(__name__)


class VerificationTransition(str, Enum):
    CLEAN = "clean"                     # 0 errors, build passed
    PROGRESS = "progress"               # Error set strictly shrinking
    STAGNATION = "stagnation"           # Error set unchanged after patch
    OSCILLATION = "oscillation"         # Error set cycled back to prior state (A -> B -> A)
    REGRESSION = "regression"           # Error set grew larger
    BUDGET_EXHAUSTED = "budget_exhausted" # Max retries exceeded


@dataclass(frozen=True)
class NormalizedDiagnostic:
    """Normalized, deterministic diagnostic error token."""
    file: str
    code: str
    message: str

    @classmethod
    def create(cls, file: str, code: str, raw_message: str) -> "NormalizedDiagnostic":
        """Strip volatile line numbers, column numbers, timestamps, and memory refs."""
        norm_file = file.replace("\\", "/").lower()
        norm_code = str(code).strip().upper()

        # Strip volatile fragments from message
        msg = raw_message.strip()
        # Remove line/col numbers like (12,34) or :12:34: or line 45:12
        msg = re.sub(r'\bline\s+\d+(?:[:,\s]+(?:col\s+)?\d+)?', ' ', msg, flags=re.IGNORECASE)
        msg = re.sub(r'[:\(]\s*\d+\s*[:,\s]+\s*\d+\s*[\)]?', ' ', msg)
        msg = re.sub(r'\b\d+:\d+\b', ' ', msg)
        # Remove standalone line references
        msg = re.sub(r'\bline\s+\d+\b', ' ', msg, flags=re.IGNORECASE)
        # Remove memory hex addresses like 0x7fa12b3
        msg = re.sub(r'0x[0-9a-fA-F]+', '0xADDR', msg)
        # Remove timestamps like 2026-09-28T...
        msg = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', 'TIMESTAMP', msg)
        # Normalize whitespace
        msg = " ".join(msg.split())

        return cls(file=norm_file, code=norm_code, message=msg)


@dataclass
class TransitionEvaluation:
    """Verdict of the error-set hash evaluation."""
    transition: VerificationTransition
    error_count: int
    error_hash: str
    should_continue: bool
    requires_rollback: bool
    diagnostic_summary: str


class OscillationDetector:
    """
    Tracks and hashes diagnostic error sets across repair iterations.
    
    Prevents infinite auto-heal loops, stagnation, and cycling.
    """

    def __init__(self, max_recovery_turns: int = 3):
        self.max_recovery_turns = max_recovery_turns
        self.history_hashes: List[str] = []
        self.history_counts: List[int] = []
        self.history_state_keys: List[str] = []
        self.turn_count: int = 0

    def compute_error_hash(self, diagnostics: Sequence[NormalizedDiagnostic]) -> str:
        """
        Compute deterministic SHA-256 hash of sorted, normalized diagnostics.
        """
        if not diagnostics:
            return "CLEAN_ZERO_ERRORS"

        # Sort deterministically
        sorted_diags = sorted(diagnostics, key=lambda d: (d.file, d.code, d.message))
        canonical_tokens = [f"{d.file}|{d.code}|{d.message}" for d in sorted_diags]
        payload = "\n".join(canonical_tokens).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def compute_state_key(
        self,
        error_hash: str,
        diff_hash: Optional[str] = None,
    ) -> str:
        """
        Compute compound state_key = SHA-256(diff_hash + error_hash).
        Grounded in OpenAI Codex harness: detects exact repeat of a patch attempt.
        """
        payload = f"{diff_hash or 'NO_DIFF'}|{error_hash}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def evaluate(
        self,
        diagnostics: Sequence[NormalizedDiagnostic],
        diff_hash: Optional[str] = None,
    ) -> TransitionEvaluation:
        """
        Evaluate current diagnostic set against past iterations.

        Returns TransitionEvaluation with next-step instructions.
        """
        self.turn_count += 1
        current_count = len(diagnostics)
        current_hash = self.compute_error_hash(diagnostics)
        current_state_key = self.compute_state_key(current_hash, diff_hash)

        # 1. Clean build: 0 errors
        if current_count == 0:
            logger.info(f"[OscillationDetector] Turn {self.turn_count}: Clean build (0 errors)!")
            self.history_hashes.append(current_hash)
            self.history_counts.append(0)
            self.history_state_keys.append(current_state_key)
            return TransitionEvaluation(
                transition=VerificationTransition.CLEAN,
                error_count=0,
                error_hash=current_hash,
                should_continue=False,  # Done!
                requires_rollback=False,
                diagnostic_summary="Build and tests passed cleanly with 0 errors.",
            )

        # 2. First evaluation baseline
        if not self.history_hashes:
            self.history_hashes.append(current_hash)
            self.history_counts.append(current_count)
            self.history_state_keys.append(current_state_key)
            logger.info(
                f"[OscillationDetector] Turn 1 baseline: {current_count} errors, hash={current_hash}"
            )
            return TransitionEvaluation(
                transition=VerificationTransition.PROGRESS,
                error_count=current_count,
                error_hash=current_hash,
                should_continue=True,
                requires_rollback=False,
                diagnostic_summary=f"Initial error set: {current_count} errors.",
            )

        prev_hash = self.history_hashes[-1]
        prev_count = self.history_counts[-1]

        # 3. Check Stagnation (exact same error set as previous turn)
        if current_hash == prev_hash:
            logger.warning(
                f"[OscillationDetector] Turn {self.turn_count}: Stagnation detected! "
                f"Error hash {current_hash} unchanged. Edit made 0 impact."
            )
            return TransitionEvaluation(
                transition=VerificationTransition.STAGNATION,
                error_count=current_count,
                error_hash=current_hash,
                should_continue=False,  # Abort auto-heal
                requires_rollback=True,  # Rollback ineffective edit
                diagnostic_summary=(
                    f"Stagnation: Error hash '{current_hash}' is identical to previous turn. "
                    "Patch failed to resolve target errors."
                ),
            )

        # 4. Check Cycling / Oscillation (hash or state_key seen in earlier turns: A -> B -> A)
        is_hash_cycle = current_hash in self.history_hashes[:-1]
        is_state_cycle = current_state_key in self.history_state_keys[:-1]
        if is_hash_cycle or is_state_cycle:
            prior_turn = (
                self.history_hashes.index(current_hash) + 1
                if is_hash_cycle
                else self.history_state_keys.index(current_state_key) + 1
            )
            logger.warning(
                f"[OscillationDetector] Turn {self.turn_count}: Oscillation cycle detected! "
                f"State matches Turn {prior_turn} (hash={current_hash}, state={current_state_key})."
            )
            return TransitionEvaluation(
                transition=VerificationTransition.OSCILLATION,
                error_count=current_count,
                error_hash=current_hash,
                should_continue=False,  # Abort auto-heal immediately
                requires_rollback=True,  # Revert to last good state
                diagnostic_summary=(
                    f"Oscillation: Current state matches Turn {prior_turn} (hash={current_hash}). "
                    "Auto-heal is cycling in an infinite repair loop."
                ),
            )

        # 5. Check Turn Budget
        if self.turn_count > self.max_recovery_turns:
            logger.warning(
                f"[OscillationDetector] Turn {self.turn_count}: Max recovery budget "
                f"({self.max_recovery_turns}) exhausted."
            )
            return TransitionEvaluation(
                transition=VerificationTransition.BUDGET_EXHAUSTED,
                error_count=current_count,
                error_hash=current_hash,
                should_continue=False,
                requires_rollback=True,
                diagnostic_summary=f"Exhausted maximum retry budget of {self.max_recovery_turns} turns.",
            )

        # 6. Check Progress vs Regression
        self.history_hashes.append(current_hash)
        self.history_counts.append(current_count)
        self.history_state_keys.append(current_state_key)

        if current_count < prev_count:
            logger.info(
                f"[OscillationDetector] Turn {self.turn_count}: Progress made! "
                f"Errors reduced from {prev_count} to {current_count} (hash={current_hash})."
            )
            return TransitionEvaluation(
                transition=VerificationTransition.PROGRESS,
                error_count=current_count,
                error_hash=current_hash,
                should_continue=True,
                requires_rollback=False,
                diagnostic_summary=f"Progress: Error count shrank from {prev_count} to {current_count}.",
            )
        else:
            logger.warning(
                f"[OscillationDetector] Turn {self.turn_count}: Regression! "
                f"Errors increased from {prev_count} to {current_count} (hash={current_hash})."
            )
            return TransitionEvaluation(
                transition=VerificationTransition.REGRESSION,
                error_count=current_count,
                error_hash=current_hash,
                should_continue=(self.turn_count < self.max_recovery_turns),
                requires_rollback=False,
                diagnostic_summary=f"Regression: Error count increased from {prev_count} to {current_count}.",
            )

    def reset(self) -> None:
        """Reset history for a new verification session."""
        self.history_hashes.clear()
        self.history_counts.clear()
        self.history_state_keys.clear()
        self.turn_count = 0
