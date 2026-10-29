"""Production-readiness benchmark runner for autonomous ticket solving.

This runner is designed for empirical evaluation once architecture is stable.
It supports:
- single-pass benchmark execution
- A/B comparison for experience-enabled vs experience-disabled
- category-level and stage-level bottleneck analysis
- localization and owner-identification metrics when ground truth is provided

Usage:
  python -m ticket_to_code.benchmarks.production_readiness_benchmark \
    --workspace C:/repo \
    --tickets C:/path/to/benchmark_tickets.json \
    --report-dir C:/path/to/report \
    --execution-mode contract \
    --ab-experience
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import re
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

from dotenv import load_dotenv


def _load_benchmark_environment() -> None:
    current = Path(__file__).resolve()
    env_path = next((parent / ".env" for parent in current.parents if (parent / ".env").exists()), None)
    project_root = next((parent for parent in current.parents if parent.name == "aviator-plugin-sample"), None)

    if env_path is None:
        env_root = project_root
    else:
        load_dotenv(env_path, override=False)
        env_root = env_path.parent

    if env_root is not None:
        os.environ.setdefault("POSTGRES_CONNECTION", "postgresql://postgres:postgres@localhost:5433/postgres")
        os.environ.setdefault("NEO4J_PASSWORD", "aviator-dev")
        os.environ.setdefault("TRACE_MODE", "true")

    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials and env_root is not None:
        default_credentials = env_root / "otl-cs-csai.json"
        if default_credentials.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(default_credentials.resolve())
            credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    if credentials and env_root is not None and not Path(credentials).is_absolute():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str((env_root / credentials).resolve())
        credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    if not os.environ.get("GOOGLE_CLOUD_PROJECT") and credentials and Path(credentials).exists():
        try:
            data = json.loads(Path(credentials).read_text(encoding="utf-8"))
            project_id = str(data.get("project_id") or "").strip()
            if project_id:
                os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        except Exception:
            pass


_load_benchmark_environment()

from ticket_to_code import TicketPriority, ValueEdgeTicket, run_autonomous_workflow


STATUS_SOLVED_WITH_PATCH = "solved_with_patch"
STATUS_NO_ACTION_REQUIRED = "no_action_required"
STATUS_NEEDS_RETRY = "needs_retry"
STATUS_FAILED_CLOSED = "failed_closed"
STATUS_FAILED = "failed"
STATUS_OTHER = "other"

SUPPORTED_TECH_ALIASES: Dict[str, str] = {
    "java": "java",
    "java-maven": "java-maven",
    "java-gradle": "java-gradle",
    "dotnet": "dotnet",
    ".net": "dotnet",
    "csharp": "csharp",
    "c#": "csharp",
    "nodejs": "nodejs",
    "node": "nodejs",
    "angular": "angular",
}


@dataclass
class TicketMetrics:
    ticket_id: str
    category: str
    complexity: str
    workspace_path: str
    status: str
    terminal_bucket: str
    latency_seconds: float
    changed_files_count: int
    changed_files: List[str]
    localization_accuracy: Optional[float]
    owner_identified: Optional[bool]
    expected_effect_achieved: bool
    verification_passed: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    failure_stage: str
    failure_reason: str
    human_acceptance: Optional[str]
    generation_diagnostics: Optional[Dict[str, Any]] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@contextmanager
def _temporary_env(updates: Dict[str, str]):
    old_values: Dict[str, Optional[str]] = {}
    for key, value in updates.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _load_tickets(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("tickets"), list):
        return payload["tickets"]
    raise ValueError("Expected JSON list, JSONL, or {'tickets': [...]} payload")


def _priority(raw: Any) -> TicketPriority:
    v = str(raw or "medium").strip().lower()
    mapping = {
        "critical": TicketPriority.CRITICAL,
        "high": TicketPriority.HIGH,
        "medium": TicketPriority.MEDIUM,
        "low": TicketPriority.LOW,
    }
    return mapping.get(v, TicketPriority.MEDIUM)


def _to_ticket(row: Dict[str, Any], index: int) -> ValueEdgeTicket:
    ticket_id = str(row.get("ticket_id") or row.get("id") or f"BENCH-{index:04d}")
    title = str(row.get("title") or row.get("ticket_title") or f"Benchmark ticket {index}")
    description = str(row.get("description") or row.get("ticket_description") or "")
    labels = [str(x) for x in (row.get("labels") or []) if str(x).strip()]
    acceptance = [str(x) for x in (row.get("acceptance_criteria") or []) if str(x).strip()]

    _expected = [str(x).replace("\\", "/") for x in (row.get("expected_changed_files") or []) if str(x).strip()]
    _owner = [str(x).replace("\\", "/") for x in (row.get("expected_owner_files") or []) if str(x).strip()]
    _forbidden = [str(x).replace("\\", "/") for x in (row.get("forbidden_files") or []) if str(x).strip()]

    return ValueEdgeTicket(
        ticket_id=ticket_id,
        title=title[:180],
        description=description,
        priority=_priority(row.get("priority")),
        labels=labels,
        acceptance_criteria=acceptance,
        expected_changed_files=_expected,
        expected_owner_files=_owner,
        forbidden_files=_forbidden,
    )


def _infer_contract_type(ticket: ValueEdgeTicket, labels: List[str]) -> str:
    text = " ".join(labels + [ticket.title, ticket.description]).lower()
    if any(k in text for k in ("version", "upgrade", "downgrade", "bump")):
        return "version"
    if any(k in text for k in ("permission", "access denied", "forbidden", "unauthorized")):
        return "permission"
    if any(k in text for k in ("refactor", "cleanup", "restructure", "simplify")):
        return "refactor"
    if any(k in text for k in ("config", "configuration", "env", "yaml", "json", "properties")):
        return "config"
    return "bug"


def _to_contract_ticket(row: Dict[str, Any], ticket: ValueEdgeTicket) -> Dict[str, Any]:
    acceptance = [str(x) for x in (row.get("acceptance_criteria") or []) if str(x).strip()]
    if not acceptance:
        acceptance = ["Implementation satisfies ticket intent without unrelated changes."]

    expected = [str(x).replace("\\", "/") for x in (row.get("expected_changed_files") or []) if str(x).strip()]
    forbidden = [str(x).replace("\\", "/") for x in (row.get("forbidden_files") or []) if str(x).strip()]

    return {
        "ticket_id": ticket.ticket_id,
        "title": ticket.title,
        "description": ticket.description,
        "type": str(row.get("type") or _infer_contract_type(ticket, ticket.labels)),
        "expected_changed_files": expected,
        "forbidden_files": forbidden,
        "acceptance_criteria": acceptance,
        "priority": str(row.get("priority") or "medium").lower(),
        "risk_level": str(row.get("risk_level") or "medium").lower(),
    }


def _to_contract_policy(row: Dict[str, Any]) -> Dict[str, Any]:
    constraints = [str(x) for x in (row.get("constraints") or []) if str(x).strip()]
    return {
        "no_hardcoding": any("hardcoding" in c.lower() for c in constraints),
        "require_expected_scope": bool(row.get("require_expected_scope", False)),
        "max_retries": int(row.get("max_fix_attempts") or 3),
    }


def _extract_changed_files(state: Dict[str, Any], contract_report: Dict[str, Any]) -> List[str]:
    changed: List[str] = []

    for item in (contract_report.get("changed_files") or []):
        if isinstance(item, dict):
            p = str(item.get("path") or "").strip()
            if p:
                changed.append(p.replace("\\", "/"))

    if changed:
        return sorted(set(changed))

    for item in (state.get("generated_code") or []):
        p = ""
        if isinstance(item, dict):
            p = str(item.get("file_path") or item.get("path") or "").strip()
        else:
            p = str(getattr(item, "file_path", "") or getattr(item, "path", "")).strip()
        if p:
            changed.append(p.replace("\\", "/"))

    return sorted(set(changed))


def _extract_owner_identified(state: Dict[str, Any], expected_owner_files: List[str]) -> Optional[bool]:
    if not expected_owner_files:
        return None

    planner_decisions = state.get("planner_decisions") or {}
    if not isinstance(planner_decisions, dict):
        return None

    normalized_expected = {p.replace("\\", "/") for p in expected_owner_files}
    planned_required = set()

    for fp, meta in planner_decisions.items():
        path = str(fp).replace("\\", "/")
        if isinstance(meta, dict) and str(meta.get("decision", "")).upper() == "REQUIRED":
            planned_required.add(path)

    return bool(normalized_expected & planned_required)


def _extract_localization_accuracy(changed_files: List[str], expected_changed_files: List[str]) -> Optional[float]:
    if not expected_changed_files:
        return None

    changed = {p.replace("\\", "/") for p in changed_files}
    expected = {p.replace("\\", "/") for p in expected_changed_files}
    if not expected:
        return None

    return len(changed & expected) / len(expected)


def _extract_verification_passed(state: Dict[str, Any], contract_report: Dict[str, Any]) -> bool:
    validation = contract_report.get("validation_evidence") or []
    if validation:
        return any(
            isinstance(v, dict)
            and v.get("check") == "diagnostics_changed_files"
            and str(v.get("result", "")).lower() == "pass"
            for v in validation
        )

    errors = state.get("errors") or []
    return len(errors) == 0


_SUCCESS_STATUSES = {"solved", "completed", "memory_updated"}
_VERIFIED_LEVELS = {"test_verified", "compile_only"}


def _is_verified_success(status: str, state: Dict[str, Any]) -> bool:
    """The workflow never emits "solved"; it emits completed/memory_updated plus a verification_level.
    Success requires BOTH a terminal success status and a verification level that means something ran."""
    s = (status or "").lower().strip()
    if s not in _SUCCESS_STATUSES:
        return False
    level = str(state.get("verification_level") or "").lower()
    return level in _VERIFIED_LEVELS


def _restore_workspace_after_ticket(state: Dict[str, Any], workspace_path: str) -> None:
    """Put the repo back the way the run found it so the next ticket starts from the same baseline.
    Uses the run's own pre-edit snapshots; never runs git reset on a user repo."""
    try:
        ws = Path(workspace_path)
        originals = state.get("original_file_contents") or {}
        norm_originals = {str(k).replace("\\", "/") for k in originals}
        for rel, content in originals.items():
            p = Path(rel) if Path(rel).is_absolute() else ws / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(content)
        for g in (state.get("generated_code") or []):
            rel = getattr(g, "file_path", "") or (g.get("file_path", "") if isinstance(g, dict) else "")
            if rel and str(rel).replace("\\", "/") not in norm_originals:
                p = Path(rel) if Path(rel).is_absolute() else ws / rel
                if p.is_file():
                    p.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning("workspace restore after ticket failed: %s", exc)


def _terminal_bucket(status: str, changed_files_count: int, state: Optional[Dict[str, Any]] = None) -> str:
    s = status.lower().strip()
    if s in _SUCCESS_STATUSES:
        if state is not None and not _is_verified_success(s, state):
            return STATUS_OTHER
        return STATUS_SOLVED_WITH_PATCH if changed_files_count > 0 else STATUS_OTHER
    if s == "solved":
        return STATUS_SOLVED_WITH_PATCH if changed_files_count > 0 else STATUS_OTHER
    if s == "no_action_required":
        return STATUS_NO_ACTION_REQUIRED
    if s == "needs_retry":
        return STATUS_NEEDS_RETRY
    if s == "failed_closed":
        return STATUS_FAILED_CLOSED
    if s in {"failed", "error", "exception", "cancelled"}:
        return STATUS_FAILED
    return STATUS_OTHER


def _classify_failure_stage(state: Dict[str, Any], status: str) -> Tuple[str, str]:
    s = str(status or "").lower()
    reason = str(state.get("validation_failure_reason") or "")
    errors = " ".join(str(e) for e in (state.get("errors") or []))
    text = f"{s} {reason} {errors}".lower()

    if not text.strip() or s in {"solved", "no_action_required"}:
        return "none", ""

    if any(k in text for k in ("owner", "grounded", "localiz", "missing required owner")):
        return "localization", text[:280]
    if any(k in text for k in ("discover", "rag", "candidate", "context", "evidence")):
        return "retrieval", text[:280]
    if any(k in text for k in ("plan", "planner", "task", "strategy")):
        return "planning", text[:280]
    if any(k in text for k in ("generate", "patch", "diff", "no effective changes")):
        return "generation", text[:280]
    if any(k in text for k in ("build", "test", "validation", "semantic")):
        return "validation", text[:280]
    if any(k in text for k in ("timeout", "credential", "permission", "import", "module", "connection", "exception")):
        return "environment", text[:280]
    return "unknown", text[:280]


def _trace_dir(workspace_path: str, ticket_id: str) -> Path:
    from ticket_to_code.tracing.paths import get_trace_dir
    return get_trace_dir(ticket_id)


def _collect_token_metrics(workspace_path: str, ticket_id: str) -> Tuple[int, int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    tdir = _trace_dir(workspace_path, ticket_id)
    if not tdir.exists():
        return 0, 0, 0

    # Preferred source: run_record.json written by the workflow's terminal nodes.
    rr = tdir / "run_record.json"
    if rr.is_file():
        try:
            data = json.loads(rr.read_text(encoding="utf-8"))
            ti = int(data.get("tokens_in") or 0)
            to = int(data.get("tokens_out") or 0)
            if ti or to:
                return ti, to, ti + to
        except Exception:
            pass

    for fp in tdir.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        for obj in _walk_json(data):
            if isinstance(obj, dict) and "metrics" in obj and isinstance(obj["metrics"], dict):
                m = obj["metrics"]
                prompt_tokens += int(m.get("prompt_tokens") or 0)
                completion_tokens += int(m.get("completion_tokens") or 0)
                total_tokens += int(m.get("total_tokens") or 0)

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens


def _walk_json(node: Any) -> Iterable[Any]:
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_json(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_json(item)


def _sanitize_category(value: Any) -> str:
    v = str(value or "uncategorized").strip().lower()
    return re.sub(r"\s+", "_", v)


def _sanitize_complexity(value: Any) -> str:
    v = str(value or "unknown").strip().lower()
    return re.sub(r"\s+", "_", v)


def _normalize_technology(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return SUPPORTED_TECH_ALIASES.get(key)


def _infer_ticket_technology(row: Dict[str, Any], default_technology: Optional[str]) -> Optional[str]:
    explicit = _normalize_technology(row.get("technology"))
    if explicit:
        return explicit

    default_norm = _normalize_technology(default_technology)
    if default_norm:
        return default_norm

    category = _sanitize_category(row.get("category"))
    if category == "ui_responsive_or_layout":
        return "angular"

    if category == "backend_or_service":
        return "java"

    labels = [str(x).strip().lower() for x in (row.get("labels") or []) if str(x).strip()]
    label_text = " ".join(labels)
    if any(k in label_text for k in ("angular", "frontend", "ui", "css")):
        return "angular"
    if any(k in label_text for k in ("java", "spring", "backend", "service", "api")):
        return "java"

    scope_text = str(row.get("repo_scope") or row.get("workspace_path") or "").replace("\\", "/").lower()
    if "/frontend" in scope_text or scope_text.endswith("frontend"):
        return "angular"
    if "aviator-platform" in scope_text:
        return "java"

    return None


def _extract_generation_diagnostics(state: Dict[str, Any], changed_files: List[str]) -> Optional[Dict[str, Any]]:
    """Extract structured generation diagnostics from workflow state.

    Pulls write_patch_evidence, verification_evidence, and stage_timings
    from the reasoning engine's ToolContext (surfaced via state dict).
    """
    diag: Dict[str, Any] = {}

    # Write patch evidence (from reasoning engine ctx.write_patch_evidence)
    wp_evidence = state.get("write_patch_evidence") or []
    if wp_evidence:
        patches = []
        for ev in wp_evidence:
            if isinstance(ev, dict):
                patches.append(ev)
            elif hasattr(ev, "__dict__"):
                patches.append({k: v for k, v in ev.__dict__.items()})
        diag["write_patch_attempts"] = len(patches)
        diag["write_patch_evidence"] = patches[:10]
        diag["patches_accepted"] = sum(1 for p in patches if p.get("accepted"))
        diag["patches_rejected"] = sum(1 for p in patches if not p.get("accepted"))
        # Aggregate rejection reasons
        reasons: Dict[str, int] = {}
        for p in patches:
            r = str(p.get("rejection_reason", "unknown"))
            if r and r != "accepted":
                reasons[r] = reasons.get(r, 0) + 1
        diag["rejection_reasons"] = reasons
    else:
        diag["write_patch_attempts"] = 0

    # Verification evidence
    v_evidence = state.get("verification_evidence") or []
    if v_evidence:
        verifications = []
        for ev in v_evidence:
            if isinstance(ev, dict):
                verifications.append(ev)
            elif hasattr(ev, "__dict__"):
                verifications.append({k: v for k, v in ev.__dict__.items()})
        diag["verification_attempts"] = len(verifications)
        diag["verification_evidence"] = verifications[:5]
        diag["verifications_passed"] = sum(1 for v in verifications if v.get("passed"))
    else:
        diag["verification_attempts"] = 0

    # Stage timings
    stage_timings = state.get("stage_timings") or {}
    if stage_timings:
        diag["stage_timings_ms"] = {k: round(v * 1000, 1) for k, v in stage_timings.items()}

    # Target files and candidate file rank
    diag["changed_files"] = changed_files
    diag["candidate_file_count"] = len(state.get("candidate_files") or [])

    return diag if diag else None


def _run_single_ticket(
    row: Dict[str, Any],
    index: int,
    default_workspace: str,
    execution_mode: str,
    default_technology: Optional[str],
    max_fix_attempts_override: Optional[int] = None,
) -> TicketMetrics:
    ticket = _to_ticket(row, index)
    workspace_path = str(row.get("repo_scope") or row.get("workspace_path") or default_workspace)
    technology = _infer_ticket_technology(row, default_technology)

    if technology is None:
        missing_reason = (
            "missing or unsupported technology; provide ticket technology explicitly "
            "(supported: java, java-maven, java-gradle, dotnet, csharp, nodejs, angular)"
        )
        return TicketMetrics(
            ticket_id=ticket.ticket_id,
            category=_sanitize_category(row.get("category")),
            complexity=_sanitize_complexity(row.get("complexity")),
            workspace_path=workspace_path,
            status="failed",
            terminal_bucket=STATUS_FAILED,
            latency_seconds=0.0,
            changed_files_count=0,
            changed_files=[],
            localization_accuracy=None,
            owner_identified=None,
            expected_effect_achieved=False,
            verification_passed=False,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            failure_stage="environment",
            failure_reason=missing_reason,
            human_acceptance=(str(row.get("human_acceptance")).strip().lower() if row.get("human_acceptance") is not None else None),
        )

    contract_ticket = _to_contract_ticket(row, ticket)
    contract_policy = _to_contract_policy(row)

    started = time.perf_counter()
    state: Dict[str, Any] = {}
    status = "failed"

    effective_max_fix_attempts = int(max_fix_attempts_override) if max_fix_attempts_override is not None else int(row.get("max_fix_attempts") or 3)

    # Per-ticket Playwright scenario (path relative to this folder); only applies when UI verify is enabled.
    _scenario = str(row.get("ui_scenario") or "").strip()
    if _scenario:
        _sp = Path(_scenario)
        os.environ["TICKET_TO_CODE_UI_SCENARIO"] = str(_sp if _sp.is_absolute() else Path(__file__).parent / _sp)
    else:
        os.environ.pop("TICKET_TO_CODE_UI_SCENARIO", None)

    try:
        state = run_autonomous_workflow(
            ticket=ticket,
            workspace_path=workspace_path,
            technology=technology,
            max_fix_attempts=effective_max_fix_attempts,
        )
        status = str(state.get("status") or "unknown")
    except Exception as exc:  # noqa: BLE001
        state = {"status": "failed", "errors": [str(exc)], "validation_failure_reason": str(exc)}
        status = "failed"

    latency = time.perf_counter() - started
    contract_report = state.get("contract_report") if isinstance(state.get("contract_report"), dict) else {}
    if contract_report:
        status = str(contract_report.get("status") or status)

    changed_files = _extract_changed_files(state, contract_report)
    
    # Added Generation Diagnostics
    gen_diag = _extract_generation_diagnostics(state, changed_files)
    
    changed_count = len(changed_files)

    expected_files = [str(x) for x in (row.get("expected_changed_files") or []) if str(x).strip()]
    expected_owner = [str(x) for x in (row.get("expected_owner_files") or []) if str(x).strip()]

    localization_accuracy = _extract_localization_accuracy(changed_files, expected_files)
    owner_identified = _extract_owner_identified(state, expected_owner)
    verification_passed = _extract_verification_passed(state, contract_report)

    no_change_evidence = (
        status.lower() == "no_action_required"
        or str(state.get("code_status") or "").lower() == "code_already_satisfied"
        or bool((state.get("outcome_verification") or {}).get("verified"))
    )

    requires_code = True
    inv = state.get("investigation_result")
    if isinstance(inv, dict):
        requires_code = bool(inv.get("requires_code_changes", True))
    elif hasattr(inv, "requires_code_changes"):
        requires_code = bool(getattr(inv, "requires_code_changes", True))

    expected_effect_achieved = bool(
        (requires_code and changed_count > 0 and verification_passed and _is_verified_success(status, state))
        or ((not requires_code or no_change_evidence) and status.lower() == "no_action_required")
    )

    prompt_toks, completion_toks, total_toks = _collect_token_metrics(workspace_path, ticket.ticket_id)

    failure_stage, failure_reason = _classify_failure_stage(state, status)
    terminal_bucket = _terminal_bucket(status, changed_count, state)
    _restore_workspace_after_ticket(state, workspace_path)

    human_acceptance = row.get("human_acceptance")
    human_acceptance_s = str(human_acceptance).strip().lower() if human_acceptance is not None else None

    return TicketMetrics(
        ticket_id=ticket.ticket_id,
        category=_sanitize_category(row.get("category")),
        complexity=_sanitize_complexity(row.get("complexity")),
        workspace_path=workspace_path,
        status=status.lower(),
        terminal_bucket=terminal_bucket,
        latency_seconds=latency,
        changed_files_count=changed_count,
        changed_files=changed_files,
        localization_accuracy=localization_accuracy,
        owner_identified=owner_identified,
        expected_effect_achieved=expected_effect_achieved,
        verification_passed=verification_passed,
        prompt_tokens=prompt_toks,
        completion_tokens=completion_toks,
        total_tokens=total_toks,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        human_acceptance=human_acceptance_s,
        generation_diagnostics=gen_diag,
    )


def _aggregate(metrics: List[TicketMetrics]) -> Dict[str, Any]:
    total = len(metrics)

    buckets: Dict[str, int] = {
        STATUS_SOLVED_WITH_PATCH: 0,
        STATUS_NO_ACTION_REQUIRED: 0,
        STATUS_NEEDS_RETRY: 0,
        STATUS_FAILED_CLOSED: 0,
        STATUS_FAILED: 0,
        STATUS_OTHER: 0,
    }
    stage_counts: Dict[str, int] = {
        "none": 0,
        "localization": 0,
        "retrieval": 0,
        "planning": 0,
        "generation": 0,
        "validation": 0,
        "environment": 0,
        "unknown": 0,
    }

    for m in metrics:
        buckets[m.terminal_bucket] = buckets.get(m.terminal_bucket, 0) + 1
        stage_counts[m.failure_stage] = stage_counts.get(m.failure_stage, 0) + 1

    loc_vals = [m.localization_accuracy for m in metrics if m.localization_accuracy is not None]
    owner_vals = [1.0 if m.owner_identified else 0.0 for m in metrics if m.owner_identified is not None]

    accept_known = [m.human_acceptance for m in metrics if m.human_acceptance in {"accepted", "rejected"}]
    accepted = sum(1 for x in accept_known if x == "accepted")

    by_category: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        cat = by_category.setdefault(
            m.category,
            {
                "count": 0,
                "solved_with_patch": 0,
                "no_action_required": 0,
                "needs_retry": 0,
                "failed": 0,
                "avg_latency_seconds": 0.0,
            },
        )
        cat["count"] += 1
        if m.terminal_bucket == STATUS_SOLVED_WITH_PATCH:
            cat["solved_with_patch"] += 1
        elif m.terminal_bucket == STATUS_NO_ACTION_REQUIRED:
            cat["no_action_required"] += 1
        elif m.terminal_bucket == STATUS_NEEDS_RETRY:
            cat["needs_retry"] += 1
        elif m.terminal_bucket in {STATUS_FAILED, STATUS_FAILED_CLOSED}:
            cat["failed"] += 1

    for cat_name, cat in by_category.items():
        subset = [m for m in metrics if m.category == cat_name]
        cat["avg_latency_seconds"] = mean(m.latency_seconds for m in subset) if subset else 0.0

    top_bottlenecks = sorted(
        ((k, v) for k, v in stage_counts.items() if k != "none"),
        key=lambda item: item[1],
        reverse=True,
    )

    summary = {
        "generated_at": _now_iso(),
        "total_tickets": total,
        "terminal_outcomes": buckets,
        "success_rate_effect_achieved": (sum(1 for m in metrics if m.expected_effect_achieved) / total) if total else 0.0,
        "avg_latency_seconds": mean(m.latency_seconds for m in metrics) if metrics else 0.0,
        "p95_latency_seconds": _p95([m.latency_seconds for m in metrics]),
        "avg_prompt_tokens": mean(m.prompt_tokens for m in metrics) if metrics else 0.0,
        "avg_completion_tokens": mean(m.completion_tokens for m in metrics) if metrics else 0.0,
        "avg_total_tokens": mean(m.total_tokens for m in metrics) if metrics else 0.0,
        "localization_accuracy_mean": mean(loc_vals) if loc_vals else None,
        "owner_identification_rate": mean(owner_vals) if owner_vals else None,
        "verification_pass_rate": (sum(1 for m in metrics if m.verification_passed) / total) if total else 0.0,
        "human_acceptance_rate": (accepted / len(accept_known)) if accept_known else None,
        "human_acceptance_coverage": (len(accept_known) / total) if total else 0.0,
        "failure_taxonomy": stage_counts,
        "top_bottlenecks": [{"stage": k, "count": v} for k, v in top_bottlenecks[:5]],
        "by_category": by_category,
    }

    return {
        "summary": summary,
        "tickets": [
            {
                "ticket_id": m.ticket_id,
                "category": m.category,
                "complexity": m.complexity,
                "workspace_path": m.workspace_path,
                "status": m.status,
                "terminal_bucket": m.terminal_bucket,
                "latency_seconds": round(m.latency_seconds, 3),
                "changed_files_count": m.changed_files_count,
                "changed_files": m.changed_files,
                "localization_accuracy": m.localization_accuracy,
                "owner_identified": m.owner_identified,
                "expected_effect_achieved": m.expected_effect_achieved,
                "verification_passed": m.verification_passed,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "total_tokens": m.total_tokens,
                "failure_stage": m.failure_stage,
                "failure_reason": m.failure_reason,
                "human_acceptance": m.human_acceptance,
            }
            for m in metrics
        ],
    }


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(0.95 * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def run_benchmark(
    tickets: List[Dict[str, Any]],
    *,
    workspace: str,
    execution_mode: str,
    technology: Optional[str],
    experience_enabled: bool,
    max_fix_attempts_override: Optional[int] = None,
) -> Dict[str, Any]:
    rows: List[TicketMetrics] = []

    with _temporary_env({"AVIATOR_ENABLE_MEMORY_UPDATE": "1" if experience_enabled else "0"}):
        for idx, row in enumerate(tickets, start=1):
            rows.append(
                _run_single_ticket(
                    row=row,
                    index=idx,
                    default_workspace=workspace,
                    execution_mode=execution_mode,
                    default_technology=technology,
                    max_fix_attempts_override=max_fix_attempts_override,
                )
            )

    payload = _aggregate(rows)
    payload["run_config"] = {
        "execution_mode": execution_mode,
        "experience_enabled": experience_enabled,
        "technology": technology,
        "max_fix_attempts_override": max_fix_attempts_override,
    }
    return payload


def _ab_compare(enabled_report: Dict[str, Any], disabled_report: Dict[str, Any]) -> Dict[str, Any]:
    e = enabled_report["summary"]
    d = disabled_report["summary"]

    return {
        "effect_achieved_delta": _safe_num(e.get("success_rate_effect_achieved")) - _safe_num(d.get("success_rate_effect_achieved")),
        "localization_accuracy_delta": _safe_num(e.get("localization_accuracy_mean")) - _safe_num(d.get("localization_accuracy_mean")),
        "owner_identification_delta": _safe_num(e.get("owner_identification_rate")) - _safe_num(d.get("owner_identification_rate")),
        "verification_pass_delta": _safe_num(e.get("verification_pass_rate")) - _safe_num(d.get("verification_pass_rate")),
        "avg_total_tokens_delta": _safe_num(e.get("avg_total_tokens")) - _safe_num(d.get("avg_total_tokens")),
        "avg_latency_delta_seconds": _safe_num(e.get("avg_latency_seconds")) - _safe_num(d.get("avg_latency_seconds")),
    }


def _safe_num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    s = report["summary"]
    lines = [
        "# Production Readiness Benchmark",
        "",
        f"Generated: {s['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Total tickets: {s['total_tickets']}",
        f"- Success (expected effect achieved): {s['success_rate_effect_achieved']:.2%}",
        f"- Avg latency: {s['avg_latency_seconds']:.2f}s",
        f"- P95 latency: {s['p95_latency_seconds']:.2f}s",
        f"- Avg total tokens: {s['avg_total_tokens']:.1f}",
        f"- Localization accuracy (mean): {('n/a' if s['localization_accuracy_mean'] is None else f"{s['localization_accuracy_mean']:.2%}")}",
        f"- Owner identification rate: {('n/a' if s['owner_identification_rate'] is None else f"{s['owner_identification_rate']:.2%}")}",
        f"- Human acceptance rate: {('n/a' if s['human_acceptance_rate'] is None else f"{s['human_acceptance_rate']:.2%}")} (coverage: {s['human_acceptance_coverage']:.2%})",
        "",
        "## Terminal Outcomes",
        "",
        "| Outcome | Count |",
        "|---|---:|",
    ]

    for key, value in s["terminal_outcomes"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Failure Taxonomy",
        "",
        "| Stage | Count |",
        "|---|---:|",
    ])
    for key, value in s["failure_taxonomy"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Top Bottlenecks",
        "",
        "| Stage | Count |",
        "|---|---:|",
    ])
    for row in s["top_bottlenecks"]:
        lines.append(f"| {row['stage']} | {row['count']} |")

    lines.extend([
        "",
        "## Category Breakdown",
        "",
        "| Category | Count | Solved with Patch | No Action Required | Needs Retry | Failed | Avg Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cat, row in sorted(s["by_category"].items()):
        lines.append(
            f"| {cat} | {row['count']} | {row['solved_with_patch']} | {row['no_action_required']} | "
            f"{row['needs_retry']} | {row['failed']} | {row['avg_latency_seconds']:.2f} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ab_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    enabled = report["experience_enabled"]["summary"]
    disabled = report["experience_disabled"]["summary"]
    comp = report["comparison"]

    lines = [
        "# Production Readiness Benchmark (A/B Experience)",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Experiment",
        "",
        "- Variant A: experience enabled (AVIATOR_ENABLE_MEMORY_UPDATE=1)",
        "- Variant B: experience disabled (AVIATOR_ENABLE_MEMORY_UPDATE=0)",
        f"- Tickets: {enabled['total_tickets']}",
        "",
        "## Headline Comparison",
        "",
        "| Metric | Enabled | Disabled | Delta (Enabled - Disabled) |",
        "|---|---:|---:|---:|",
        f"| Success (expected effect) | {enabled['success_rate_effect_achieved']:.2%} | {disabled['success_rate_effect_achieved']:.2%} | {comp['effect_achieved_delta']:+.2%} |",
        f"| Localization accuracy | {('n/a' if enabled['localization_accuracy_mean'] is None else f"{enabled['localization_accuracy_mean']:.2%}")} | {('n/a' if disabled['localization_accuracy_mean'] is None else f"{disabled['localization_accuracy_mean']:.2%}")} | {comp['localization_accuracy_delta']:+.2%} |",
        f"| Owner identification rate | {('n/a' if enabled['owner_identification_rate'] is None else f"{enabled['owner_identification_rate']:.2%}")} | {('n/a' if disabled['owner_identification_rate'] is None else f"{disabled['owner_identification_rate']:.2%}")} | {comp['owner_identification_delta']:+.2%} |",
        f"| Verification pass rate | {enabled['verification_pass_rate']:.2%} | {disabled['verification_pass_rate']:.2%} | {comp['verification_pass_delta']:+.2%} |",
        f"| Avg total tokens | {enabled['avg_total_tokens']:.1f} | {disabled['avg_total_tokens']:.1f} | {comp['avg_total_tokens_delta']:+.1f} |",
        f"| Avg latency (s) | {enabled['avg_latency_seconds']:.2f} | {disabled['avg_latency_seconds']:.2f} | {comp['avg_latency_delta_seconds']:+.2f} |",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(report_dir: Path, report: Dict[str, Any], filename_prefix: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{filename_prefix}.json"
    md_path = report_dir / f"{filename_prefix}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if "summary" in report:
        _write_markdown_report(md_path, report)
    elif report.get("mode") == "ab_experience":
        _write_ab_markdown_report(md_path, report)
    else:
        md_path.write_text("# Benchmark Report\n\nUnsupported report shape. See JSON file for details.\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production-readiness benchmark suite.")
    parser.add_argument("--workspace", required=True, help="Default workspace path for tickets")
    parser.add_argument("--tickets", required=True, help="Path to benchmark ticket file (.json or .jsonl)")
    parser.add_argument("--report-dir", required=True, help="Directory for benchmark reports")
    parser.add_argument(
        "--execution-mode",
        default="contract",
        choices=["contract", "pipeline", "reasoning", "auto"],
        help="Execution mode to evaluate",
    )
    parser.add_argument("--technology", default=None, help="Optional explicit technology")
    parser.add_argument(
        "--ab-experience",
        action="store_true",
        help="Run A/B benchmark with experience enabled vs disabled",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight settings for pilot runs (forces max_fix_attempts=1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_fix_attempts_override = 1 if args.lightweight else None

    tickets = _load_tickets(Path(args.tickets))
    if not tickets:
        raise ValueError("No tickets found in benchmark input")

    report_dir = Path(args.report_dir)

    # Without this the run only shows warnings; keep the full step-by-step log next to the report.
    import logging
    report_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(report_dir / "run.log", encoding="utf-8"), logging.StreamHandler()],
    )
    os.environ.setdefault("TICKET_TO_CODE_TRACE_DIR", str(report_dir / "traces"))

    if args.ab_experience:
        enabled = run_benchmark(
            tickets,
            workspace=args.workspace,
            execution_mode=args.execution_mode,
            technology=args.technology,
            experience_enabled=True,
            max_fix_attempts_override=max_fix_attempts_override,
        )
        disabled = run_benchmark(
            tickets,
            workspace=args.workspace,
            execution_mode=args.execution_mode,
            technology=args.technology,
            experience_enabled=False,
            max_fix_attempts_override=max_fix_attempts_override,
        )

        combined = {
            "generated_at": _now_iso(),
            "mode": "ab_experience",
            "experience_enabled": enabled,
            "experience_disabled": disabled,
            "comparison": _ab_compare(enabled, disabled),
        }

        _write_outputs(report_dir, combined, "production_readiness_ab_report")
        print("A/B benchmark complete")
        print(json.dumps(combined["comparison"], indent=2))
        return

    single = run_benchmark(
        tickets,
        workspace=args.workspace,
        execution_mode=args.execution_mode,
        technology=args.technology,
        experience_enabled=True,
        max_fix_attempts_override=max_fix_attempts_override,
    )
    _write_outputs(report_dir, single, "production_readiness_report")

    summary = single["summary"]
    print("Benchmark complete")
    print(f"  Tickets: {summary['total_tickets']}")
    print(f"  Expected-effect success: {summary['success_rate_effect_achieved']:.2%}")
    print(f"  Avg latency: {summary['avg_latency_seconds']:.2f}s")
    print(f"  Top bottleneck: {summary['top_bottlenecks'][0] if summary['top_bottlenecks'] else 'none'}")


if __name__ == "__main__":
    main()
