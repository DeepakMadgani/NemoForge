"""
Change Authorization — evidence is NOT change authorization.

The central architectural distinction: a file may be discovered, inspected,
read, structurally related, part of a dependency chain, or verified evidence
WITHOUT being authorized for modification.

Roles:
    CHANGE_TARGET        — explicitly authorized for modification
    READ_ONLY_REFERENCE  — relevant/evidence, but must not be modified
    GENERATED_DEPENDENCY — a new artifact required by a complete change path
    UNRELATED_FILE       — forbidden / not relevant

A writable task is only promoted to CHANGE_TARGET when it is justified:
  1. the ticket explicitly authorizes it (declared scope), or a companion of an
     authorized file (e.g. component .html/.scss beside an authorized .ts), or
  2. no authorization scope was declared (fall back to task-level authorization).

Configuration/build/deployment files are protected: they may only be modified
when ticket-authorized or the ticket clearly targets configuration.

All checks are universal file-category checks — NOT project/framework/language
business rules. No hardcoded xchange-ui / Java / GraphQL assumptions.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from ticket_to_code.agents.canonical_path import (
    canonical_repo_path,
    canonical_component_base,
    COMPONENT_EXTENSIONS,
)


class ChangeRole(str, Enum):
    CHANGE_TARGET = "change_target"
    READ_ONLY_REFERENCE = "read_only_reference"
    GENERATED_DEPENDENCY = "generated_dependency"
    UNRELATED_FILE = "unrelated_file"


# ── Universal configuration / build / deployment file categories ────────────────
_CONFIG_BASENAMES = {
    "application.yml", "application.yaml", "application.properties",
    "bootstrap.yml", "bootstrap.yaml", "logback.xml", "log4j2.xml",
    ".env", ".env.local", ".env.production", ".npmrc", "web.config", "app.config",
}
_CONFIG_EXTS = {".env", ".properties", ".ini", ".cfg", ".conf"}
_CONFIG_DIR_MARKERS = ("/config/", "/configs/", "/environments/", "/env/", "/.github/", "/deploy/", "/deployment/", "/helm/", "/k8s/")
_BUILD_BASENAMES = {
    "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
    "package.json", "package-lock.json", "yarn.lock", "angular.json",
    "tsconfig.json", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "makefile", "cargo.toml", "go.mod", "requirements.txt", "pyproject.toml",
}
# .yml/.yaml are config ONLY when in a config-ish location or a known config basename,
# because .yml can also be legitimate ticket data — avoid over-protecting.
_YAML_EXTS = {".yml", ".yaml"}

# Structural component suffixes only
_COMPONENT_SUFFIXES = COMPONENT_EXTENSIONS


def _norm(p: str) -> str:
    return (p or "").replace("\\", "/").lower()


def _basename(p: str) -> str:
    return _norm(p).split("/")[-1]


def is_config_file(file_path: str) -> bool:
    n = _norm(file_path)
    base = _basename(n)
    if base in _CONFIG_BASENAMES:
        return True
    for ext in _CONFIG_EXTS:
        if base.endswith(ext):
            return True
    if base.endswith(tuple(_YAML_EXTS)) and any(m in n for m in _CONFIG_DIR_MARKERS):
        return True
    return False


def is_build_config(file_path: str) -> bool:
    return _basename(file_path) in _BUILD_BASENAMES


def _match(
    file_path: str,
    paths: set[str],
    workspace_root: Optional[Union[str, Path]] = None,
) -> bool:
    if not file_path or not paths:
        return False
    c_file = canonical_repo_path(file_path, workspace_root) if workspace_root else None
    norm_file = _norm(file_path)

    for p in paths:
        if not p:
            continue
        c_p = canonical_repo_path(p, workspace_root) if workspace_root else None
        if c_file and c_p and c_file == c_p:
            return True
        norm_p = _norm(p)
        if norm_file == norm_p or norm_file.endswith("/" + norm_p) or norm_p.endswith("/" + norm_file):
            return True
        if norm_file.endswith(norm_p) or norm_p.endswith(norm_file):
            return True
    return False


def _companion_base(
    file_path: str,
    workspace_root: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Derive component base (dir + stem) for structurally valid components only."""
    if workspace_root:
        cb = canonical_component_base(file_path, workspace_root)
        if cb:
            return cb

    n = _norm(file_path)

    # 1. Java companion base (src/main/java <-> src/test/java, and Impl)
    if n.endswith(".java"):
        clean_p = re.sub(r'(^|/)src/test/java/', r'\1src/main/java/', n)
        parts = clean_p.split("/")
        d = "/".join(parts[:-1])
        base = parts[-1][:-5]
        if base.endswith("tests"):
            base = base[:-5]
        elif base.endswith("test"):
            base = base[:-4]
        elif base.startswith("test"):
            base = base[4:]
        if base.endswith("impl"):
            base = base[:-4]
        return f"{d}/{base}" if d else base

    # 2. TypeScript / JavaScript companion base (handles .ts, .tsx, .js, .jsx and their specs/tests)
    if n.endswith((".ts", ".tsx", ".js", ".jsx")):
        d = "/".join(n.split("/")[:-1])
        base = _basename(n)
        for suf in _COMPONENT_SUFFIXES:
            if base.endswith(suf):
                base = base[: len(base) - len(suf)]
                return f"{d}/{base}" if d else base

        clean_p = re.sub(r'\.(spec|test)\.(ts|tsx|js|jsx)$', r'.\2', n)
        parts = clean_p.split("/")
        d = "/".join(parts[:-1])
        base = parts[-1].rsplit(".", 1)[0]
        return f"{d}/{base}" if d else base

    # 3. Python companion base
    if n.endswith(".py"):
        clean_p = re.sub(r'(^|/)tests?(/|$)', r'\1', n)
        clean_p = re.sub(r'(^|/)src(/|$)', r'\1', clean_p)
        parts = clean_p.split("/")
        d = "/".join(parts[:-1]).strip("/")
        base = parts[-1][:-3]
        if base.startswith("test_"):
            base = base[5:]
        elif base.endswith("_test"):
            base = base[:-5]
        return f"{d}/{base}" if d else base

    # 4. Other Angular component template/style companions (.html, .scss, .css)
    d = "/".join(n.split("/")[:-1])
    base = _basename(n)
    matched = False
    for suf in _COMPONENT_SUFFIXES:
        if base.endswith(suf):
            base = base[: len(base) - len(suf)]
            matched = True
            break
    if not matched:
        return None
    return f"{d}/{base}" if d else base


def is_companion(
    file_path: str,
    authorized_files: set[str],
    workspace_root: Optional[Union[str, Path]] = None,
) -> bool:
    """True if file_path is a structurally valid companion of any authorized file."""
    cb = _companion_base(file_path, workspace_root)
    if not cb:
        return False
    for a in authorized_files:
        acb = _companion_base(a, workspace_root)
        if acb and (acb == cb or acb.endswith("/" + cb) or cb.endswith("/" + acb)):
            return True
    return False


def classify_change_role(
    file_path: str,
    authorized_files: set[str] | None = None,
    forbidden_files: set[str] | None = None,
    scope_declared: bool = False,
    ticket_targets_config: bool = False,
    proven_targets: set[str] | None = None,
    workspace_root: Optional[Union[str, Path]] = None,
    structured_suppliers: dict[str, dict] | None = None,
    scope_proof: Optional[Any] = None,
) -> tuple[str, str]:
    """Return (ChangeRole value, reason) for a proposed writable file.

    Invariant: DISCOVERED != CHANGE_TARGET. Discovery / relationship / RAG /
    planner-suggestion grant permission to INVESTIGATE, never to modify. A file
    becomes CHANGE_TARGET only when the ticket declares it (scope_declared) OR
    evidence proves it (proven_targets).

    When a validated TicketScopeProof is present, write permission MUST be derived
    strictly from scope_proof.is_file_writable(file_path). If not writable, it is
    demoted to READ_ONLY_REFERENCE unconditionally.
    """
    _scope_status = getattr(scope_proof, "status", None)
    _is_proven = getattr(scope_proof, "is_scope_proven", False) or (_scope_status == "PROVEN")
    authorized_files = authorized_files or set()
    forbidden_files = forbidden_files or set()
    proven_targets = proven_targets or set()
    structured_suppliers = structured_suppliers or {}
    n = _norm(file_path)

    _in_authorized = bool(authorized_files and (_match(n, authorized_files, workspace_root) or is_companion(n, authorized_files, workspace_root)))

    if scope_proof is not None and _is_proven:
        if not scope_proof.is_file_writable(file_path) and not _in_authorized:
            return ChangeRole.READ_ONLY_REFERENCE.value, (
                f"file {file_path} is not in TicketScopeProof approved writable files"
            )
    elif scope_proof is not None and not _is_proven:
        if not _in_authorized:
            return ChangeRole.READ_ONLY_REFERENCE.value, (
                f"TicketScopeProof is {_scope_status or 'UNPROVEN'} — all files are read-only"
            )

    if forbidden_files and _match(n, forbidden_files, workspace_root):
        return ChangeRole.UNRELATED_FILE.value, "explicitly forbidden by ticket"

    # Configuration/build protection — evidence/compilation is NOT authorization.
    if is_config_file(n) or is_build_config(n):
        if scope_declared and _match(n, authorized_files, workspace_root):
            return ChangeRole.CHANGE_TARGET.value, "configuration explicitly ticket-authorized"
        if ticket_targets_config:
            return ChangeRole.CHANGE_TARGET.value, "ticket targets configuration"
        return ChangeRole.READ_ONLY_REFERENCE.value, (
            "configuration/build file protected — no ticket/evidence justification"
        )

    if scope_declared:
        # Strict whitelist: no dependency expansion or companion inference may bypass it
        if _match(n, authorized_files, workspace_root) or is_companion(n, authorized_files, workspace_root):
            return ChangeRole.CHANGE_TARGET.value, "in ticket-authorized change set"
        return ChangeRole.READ_ONLY_REFERENCE.value, (
            "outside ticket-authorized scope — strict whitelist"
        )

    # No declared scope: evidence-backed proof required.
    # 1. Scope proof proven writable files
    if scope_proof is not None and _is_proven and scope_proof.is_file_writable(file_path, workspace_root=workspace_root):
        return ChangeRole.CHANGE_TARGET.value, "proven writable in TicketScopeProof"

    # 2. Files in authorized change set (from preflight guidance, planning, companions)
    if _in_authorized:
        return ChangeRole.CHANGE_TARGET.value, "in authorized change set"

    # 3. Evidence-proven primary feature targets and their companions
    if _match(n, proven_targets, workspace_root):
        return ChangeRole.CHANGE_TARGET.value, "evidence-proven primary feature target"
    if is_companion(n, proven_targets, workspace_root):
        return ChangeRole.CHANGE_TARGET.value, "companion of evidence-proven primary feature target"

    # Check structured supplier authorization (GENERATED_DEPENDENCY)
    if structured_suppliers:
        supp_info = None
        c_fp = canonical_repo_path(file_path, workspace_root) if workspace_root else n
        for supp_path, info in structured_suppliers.items():
            c_sp = canonical_repo_path(supp_path, workspace_root) if workspace_root else _norm(supp_path)
            if (c_fp and c_sp and c_fp == c_sp) or _match(file_path, {supp_path}, workspace_root):
                supp_info = info
                break
        if supp_info:
            consumer = supp_info.get("consumer", "authorized target")
            capability = supp_info.get("capability", "required capability")
            return (
                ChangeRole.GENERATED_DEPENDENCY.value,
                f"structured supplier required by {consumer} for capability '{capability}'"
            )

    return ChangeRole.READ_ONLY_REFERENCE.value, (
        "discovered but not evidence-proven — read-only until modification is justified"
    )



# ── Cross-boundary completeness + invented-API prevention ───────────────────────

def verify_api_call_has_contract(endpoint: str, detected_contracts) -> bool:
    """True if a consumer's API call maps to a real detected provider contract.

    ``detected_contracts`` are APIContract-like objects with ``.endpoint`` (and
    optionally ``.response_type``). Prevents inventing an operation that no
    provider actually exposes.
    """
    if not endpoint:
        return False
    e = endpoint.strip().lower()
    for c in (detected_contracts or []):
        cep = str(getattr(c, "endpoint", getattr(c, "name", "")) or "").strip().lower()
        if cep and (cep == e or cep.endswith("/" + e) or e.endswith("/" + cep)):
            return True
    return False


def provider_change_is_reachable(provider_symbol: str, detected_contracts) -> bool:
    """True if a provider-side symbol is exposed through a real contract.

    A backend method that is not exposed via any detected API/endpoint is dead
    code for a cross-boundary consumer.
    """
    if not provider_symbol:
        return False
    s = provider_symbol.strip().lower()
    for c in (detected_contracts or []):
        for attr in ("endpoint", "name", "response_type", "request_type"):
            v = str(getattr(c, attr, "") or "").strip().lower()
            if v and (v == s or s in v or v in s):
                return True
    return False


def cross_boundary_change_is_complete(
    provider_changes: set[str],
    consumer_changes: set[str],
    detected_contracts,
) -> tuple[bool, str]:
    """Validate that provider-side changes are exposed to consumer-side changes.

    If there are provider changes AND consumer changes but NO detected contract
    links them, the plan is incomplete → should return to planning.
    """
    if not provider_changes or not consumer_changes:
        return True, "no cross-boundary pairing to validate"
    if detected_contracts:
        return True, "verified contract path exists between provider and consumer"
    return False, (
        "provider and consumer changes present but no verified consumer-facing "
        "contract links them — cross-boundary change is incomplete"
    )
