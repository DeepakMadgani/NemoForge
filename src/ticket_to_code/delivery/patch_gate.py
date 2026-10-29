"""
PatchGate — delivery safety checks before any external patch emission (B9).

Three deterministic gates (in order):
  1. scope_enforce  — reject any diff that touches files outside writable_set (pre+post)
  2. apply_check    — verify diff applies cleanly to current HEAD via git
  3. secret_scan    — reject diff containing high-confidence secret patterns

All gates are fail-closed: when git is unavailable or a check raises unexpectedly,
the gate returns a failure result rather than silently passing.

Author: Deepak Madgani
Date: July 2026
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ApplyResult:
    ok:      bool
    rejects: List[str] = field(default_factory=list)
    error:   str       = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class ScopeResult:
    ok:         bool
    violations: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class SecretHit:
    rule:  str
    match: str
    line:  int = 0
    file:  str = ""


@dataclass
class DeliveryResult:
    """Final gate result after all three checks."""
    delivered:       bool
    apply_ok:        bool
    scope_clean:     bool
    secret_hits:     List[SecretHit] = field(default_factory=list)
    reject_reason:   str             = ""
    head_commit:     str             = ""
    idempotency_key: str             = ""


# ============================================================================
# SECRET PATTERNS (high-precision — start conservative, expand over time)
# ============================================================================

# Each entry: (rule_name, compiled_regex)
_SECRET_PATTERNS: List[tuple] = [
    ("aws_access_key",       re.compile(r"AKIA[0-9A-Z]{16}")),
    ("aws_secret_key",       re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"]([A-Za-z0-9/+=]{40})")),
    ("private_key_header",   re.compile(r"-----BEGIN (RSA|EC|OPENSSH|PGP) PRIVATE KEY-----")),
    ("github_token",         re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("generic_api_key",      re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([A-Za-z0-9]{32,})")),
    ("google_oauth",         re.compile(r"ya29\.[0-9A-Za-z_-]+")),
    ("slack_token",          re.compile(r"xox[baprs]-[0-9A-Za-z]{10,}")),
    ("stripe_key",           re.compile(r"sk_live_[0-9a-zA-Z]{24}")),
]


# ============================================================================
# PATCH GATE
# ============================================================================

class PatchGate:
    """
    Deterministic pre-delivery safety gate.

    All methods are pure (no LLM calls, no DB queries).
    Methods are intentionally fail-closed.

    Usage::

        gate = PatchGate(repo_path="/path/to/repo")
        result = gate.check_all(
            diff="...",
            writable_files={"src/foo.ts", "src/bar.ts"},
        )
        if not result.delivered:
            logger.error("Delivery blocked: %s", result.reject_reason)
    """

    def __init__(self, repo_path: str = ".") -> None:
        self.repo_path = Path(repo_path).resolve()
        self._git_available = shutil.which("git") is not None
        if not self._git_available:
            logger.warning(
                "⚠️  PatchGate: git not found in PATH — apply_check will fail-closed"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def check_all(
        self,
        diff: str,
        writable_files: Set[str],
        head_commit: str = "",
    ) -> DeliveryResult:
        """
        Run all three gates in order.  Returns on first failure.

        Args:
            diff:            Unified diff string.
            writable_files:  Set of repo-relative paths the generator was allowed
                             to modify.  Empty set disables scope check.
            head_commit:     Current HEAD SHA (informational only).
        """
        if not diff or not diff.strip():
            return DeliveryResult(
                delivered=False,
                apply_ok=False,
                scope_clean=False,
                reject_reason="empty diff",
            )

        # Gate 1: scope
        scope = self.scope_enforce(diff, writable_files)
        if not scope.ok:
            logger.warning(
                "PatchGate: scope violations → %s",
                ", ".join(scope.violations),
            )
            return DeliveryResult(
                delivered=False,
                apply_ok=False,
                scope_clean=False,
                reject_reason=f"scope_violation: {scope.violations}",
                head_commit=head_commit,
            )

        # Gate 2: apply check
        apply_res = self.apply_check(diff)
        if not apply_res.ok:
            logger.warning("PatchGate: apply_check failed → %s", apply_res.error)
            return DeliveryResult(
                delivered=False,
                apply_ok=False,
                scope_clean=True,
                reject_reason=f"apply_check_failed: {apply_res.error}",
                head_commit=head_commit,
            )

        # Gate 3: secret scan
        hits = self.secret_scan(diff)
        if hits:
            logger.error(
                "PatchGate: secret_scan found %d hit(s): %s",
                len(hits),
                [h.rule for h in hits],
            )
            return DeliveryResult(
                delivered=False,
                apply_ok=True,
                scope_clean=True,
                secret_hits=hits,
                reject_reason=f"secret_scan_hit: {[h.rule for h in hits]}",
                head_commit=head_commit,
            )

        # All clear
        idem_key = self._idempotency_key(diff)
        logger.info("✅ PatchGate: all checks passed (idempotency_key=%s)", idem_key)
        return DeliveryResult(
            delivered=True,
            apply_ok=True,
            scope_clean=True,
            head_commit=head_commit,
            idempotency_key=idem_key,
        )

    # ── Gate 1: Scope Enforcement ──────────────────────────────────────────────

    def scope_enforce(self, diff: str, writable_files: Set[str]) -> ScopeResult:
        """
        Parse the diff header lines and check every modified file is in writable_files.

        If writable_files is empty the check is skipped (returns ok=True).
        Normalises path separators so Windows / POSIX mismatches do not cause
        false positives.
        """
        if not writable_files:
            return ScopeResult(ok=True)

        # Normalise writable set
        norm_writable = {p.replace("\\", "/") for p in writable_files}

        # Extract modified file paths from unified diff headers
        touched: List[str] = []
        for line in diff.splitlines():
            if line.startswith("--- ") or line.startswith("+++ "):
                parts = line.split("\t")[0]  # git sometimes adds a tab+timestamp
                path  = parts[4:].strip()    # strip "--- " or "+++ "
                if path.startswith("a/") or path.startswith("b/"):
                    path = path[2:]
                if path and path != "/dev/null":
                    norm = path.replace("\\", "/")
                    if norm not in touched:
                        touched.append(norm)

        violations = [p for p in touched if p not in norm_writable]
        ok = len(violations) == 0
        if not ok:
            logger.debug(
                "scope_enforce violations: %s (writable: %s)",
                violations, norm_writable,
            )
        return ScopeResult(ok=ok, violations=violations)

    # ── Gate 2: Git Apply Check ────────────────────────────────────────────────

    def apply_check(self, diff: str) -> ApplyResult:
        """
        Run `git apply --check` against the current HEAD in repo_path.

        Fail-closed: if git is unavailable or the command errors unexpectedly,
        returns ApplyResult(ok=False).
        """
        if not self._git_available:
            return ApplyResult(
                ok=False,
                error="git not available in PATH — apply_check is fail-closed",
            )

        tmp_patch = None
        try:
            # Write diff to a temp file
            fd, tmp_patch = tempfile.mkstemp(suffix=".patch", prefix="aviator_")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(diff)

            result = subprocess.run(
                ["git", "apply", "--check", tmp_patch],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return ApplyResult(ok=True)

            error_msg = (result.stderr or result.stdout).strip()
            rejects   = [
                line for line in error_msg.splitlines()
                if "error:" in line.lower() or "patch failed" in line.lower()
            ]
            return ApplyResult(ok=False, rejects=rejects, error=error_msg[:400])

        except subprocess.TimeoutExpired:
            return ApplyResult(ok=False, error="git apply --check timed out after 30s")
        except FileNotFoundError:
            return ApplyResult(ok=False, error="git not found")
        except Exception as exc:
            logger.exception("apply_check unexpected error")
            return ApplyResult(ok=False, error=str(exc)[:200])
        finally:
            if tmp_patch and os.path.exists(tmp_patch):
                try:
                    os.unlink(tmp_patch)
                except OSError:
                    pass

    # ── Gate 3: Secret Scan ────────────────────────────────────────────────────

    def secret_scan(self, diff: str) -> List[SecretHit]:
        """
        Scan unified diff for high-confidence secret patterns.

        Only scans added lines (lines starting with '+' but not '+++').
        Returns a list of SecretHit objects; empty list means clean.
        """
        hits: List[SecretHit] = []
        current_file = ""
        lineno       = 0

        for line in diff.splitlines():
            # Track current file
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
                lineno = 0
                continue
            if line.startswith("+++ "):
                current_file = line[4:].strip()
                lineno = 0
                continue
            if line.startswith("@@"):
                # Parse hunk header: @@ -l,s +l,s @@
                m = re.search(r"\+(\d+)", line)
                lineno = int(m.group(1)) if m else lineno
                continue

            if line.startswith("+") and not line.startswith("+++"):
                lineno += 1
                content = line[1:]  # strip leading '+'
                for rule_name, pattern in _SECRET_PATTERNS:
                    if pattern.search(content):
                        hits.append(
                            SecretHit(
                                rule=rule_name,
                                match=content[:120],
                                line=lineno,
                                file=current_file,
                            )
                        )
                        logger.warning(
                            "🔒 secret_scan hit: rule=%s file=%s line=%d",
                            rule_name, current_file, lineno,
                        )
                        break  # one hit per added line is enough
            elif not line.startswith("-"):
                lineno += 1  # context line

        return hits

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _idempotency_key(diff: str) -> str:
        """SHA256-based idempotency key for the diff content."""
        import hashlib
        return hashlib.sha256(diff.encode("utf-8", errors="replace")).hexdigest()[:16]

    @staticmethod
    def current_head(repo_path: str = ".") -> str:
        """Return current HEAD commit SHA or empty string if unavailable."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(repo_path).resolve()),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""
