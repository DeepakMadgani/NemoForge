"""
Build Diagnostic Classifier — differential + infrastructure attribution.

Classifies raw build/compiler diagnostics so the workflow can:
  * skip source repair when a failure is an environment/dependency problem
    (Artifactory down, 401/403, network timeout, unresolved artifact …);
  * accept a build whose ONLY failures are pre-existing (baseline) errors the
    ticket did not introduce;
  * repair only TICKET_INTRODUCED / GENERATED_* diagnostics.

Attribution is file-ownership based (the same proven signal used by the error
resolver): a diagnostic whose source file our ticket never generated/modified
is PRE_EXISTING. Infrastructure detection uses generic build-tool patterns and
never hardcodes a specific artifact/company name.

Language-agnostic: file paths are parsed from the common compiler formats
(TS ``path(12,3): error``, Java ``/path.java:[12,4]`` / ``path.java:12:``,
generic ``path:line``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ticket_to_code.models import ErrorCategory


# ── Infrastructure / environment failure patterns (generic, lowercased) ─────────
# These indicate the build could not run to completion for reasons unrelated to
# source correctness. Deliberately conservative: e.g. "could not resolve
# dependency" (infra) is included, but bare "could not find symbol" (a real Java
# compile error) is NOT.
_INFRA_PATTERNS = (
    "could not resolve dependencies",
    "could not resolve dependency",
    "could not resolve all dependencies",
    "could not resolve all files",
    "could not resolve artifact",
    "could not download",
    "could not transfer artifact",
    "could not GET",
    "could not find artifact",
    "could not get resource",
    "failure to find",
    "was cached in the local repository",
    "repository ... unavailable",
    "repository unavailable",
    "unable to load maven",
    "artifactory",
    "nexus",
    "connection refused",
    "connection timed out",
    "connect timed out",
    "read timed out",
    "read timeout",
    "network is unreachable",
    "no route to host",
    "temporary failure in name resolution",
    "unknownhostexception",
    "unknown host",
    "name or service not known",
    "401 unauthorized",
    "403 forbidden",
    "407 proxy authentication",
    "status code 401",
    "status code 403",
    "peer not authenticated",
    "pkix path building failed",
    "unable to find valid certification path",
    "received fatal alert",
    "sslhandshakeexception",
    "handshake_failure",
    "gateway timeout",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway",
    "proxyerror",
    "econnrefused",
    "etimedout",
    "enotfound",
    "getaddrinfo",
    "socket hang up",
    "network error",
    "failed to fetch",
    "registry error",
    "npm err! network",
    "eai_again",
)

# path(line,col)  |  /path.ext:[line,col]  |  path.ext:line:col  |  path.ext:line
_PATH_RES = (
    re.compile(r'([^\s(){}]+\.[A-Za-z0-9]+)\((\d+)[,:]'),          # TS: file.ts(12,3)
    re.compile(r'([^\s:(){}]+\.[A-Za-z0-9]+):\[?(\d+)[,:\]]'),      # Java: file.java:[12,4] / file.java:12:
    re.compile(r'([^\s:(){}]+\.[A-Za-z0-9]+):(\d+)'),               # generic file.ext:12
)


@dataclass
class DiagnosticClass:
    raw: str
    file_path: str | None
    category: str          # ErrorCategory value


@dataclass
class BuildDiagnosticReport:
    diagnostics: list = field(default_factory=list)   # list[DiagnosticClass]

    def by_category(self, category: str) -> list:
        return [d for d in self.diagnostics if d.category == category]

    @property
    def infrastructure(self) -> list:
        return self.by_category(ErrorCategory.INFRASTRUCTURE.value)

    @property
    def pre_existing(self) -> list:
        return self.by_category(ErrorCategory.PRE_EXISTING.value)

    @property
    def blocking(self) -> list:
        """Diagnostics that must be repaired (ticket/generated/unknown-owned)."""
        blocking_cats = {
            ErrorCategory.TICKET_INTRODUCED.value,
            ErrorCategory.GENERATED_COMPANION.value,
            ErrorCategory.GENERATED_DEPENDENCY_FAILURE.value,
            ErrorCategory.UNKNOWN.value,
        }
        return [d for d in self.diagnostics if d.category in blocking_cats]

    @property
    def is_infrastructure_only(self) -> bool:
        return bool(self.infrastructure) and not self.blocking

    @property
    def is_differential_accept(self) -> bool:
        """There ARE errors, but none are attributable to this ticket."""
        return bool(self.diagnostics) and not self.blocking and not self.infrastructure

    def summary(self) -> str:
        counts: dict = {}
        for d in self.diagnostics:
            counts[d.category] = counts.get(d.category, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"


def _normalize_path(p: str) -> str:
    if not p:
        return ""
    p = str(p).strip().replace("\\", "/")
    if p.startswith("/") and len(p) > 2 and p[1].isalpha() and p[2] == ":":
        p = p[1:]
    return p


def _extract_file(line: str) -> str | None:
    for rex in _PATH_RES:
        m = rex.search(line)
        if m:
            return _normalize_path(m.group(1))
    return None


def _is_infrastructure(line: str) -> bool:
    low = line.lower()
    return any(p in low for p in _INFRA_PATTERNS)


def _owns(file_path: str, our_files: set[str]) -> bool:
    norm = _normalize_path(file_path).lower()
    for mod in our_files:
        if not mod:
            continue
        mod_norm = _normalize_path(mod).lower()
        if norm == mod_norm or norm.endswith(mod_norm) or mod_norm.endswith(norm):
            return True
    return False


def classify_build_diagnostics(
    errors: list[str],
    our_files: set[str],
    dependency_error_markers: tuple[str, ...] = (
        "cannot find symbol",
        "cannot resolve symbol",
        "is not a function",
        "has no exported member",
        "does not exist on type",
        "cannot find name",
    ),
) -> BuildDiagnosticReport:
    """Classify each raw diagnostic line.

    Args:
        errors: raw build/compiler error lines.
        our_files: normalized (lowercased, forward-slashed) paths our ticket
            generated or modified.
    """
    report = BuildDiagnosticReport()
    norm_our = {f.replace("\\", "/").lower() for f in our_files if f}

    for raw in errors:
        if not raw or not str(raw).strip():
            continue
        line = str(raw)

        if _is_infrastructure(line):
            report.diagnostics.append(
                DiagnosticClass(line, None, ErrorCategory.INFRASTRUCTURE.value)
            )
            continue

        fpath = _extract_file(line)
        if fpath is None:
            report.diagnostics.append(
                DiagnosticClass(line, None, ErrorCategory.UNKNOWN.value)
            )
            continue

        if _owns(fpath, norm_our):
            low = line.lower()
            if any(m in low for m in dependency_error_markers):
                cat = ErrorCategory.GENERATED_DEPENDENCY_FAILURE.value
            else:
                cat = ErrorCategory.TICKET_INTRODUCED.value
            report.diagnostics.append(DiagnosticClass(line, fpath, cat))
        else:
            report.diagnostics.append(
                DiagnosticClass(line, fpath, ErrorCategory.PRE_EXISTING.value)
            )

    return report
