"""
ExecutionTracer — forensic observability for the Ticket-to-Code workflow.

Activated by setting TRACE_MODE=true in the environment.
When disabled every public method is a no-op with near-zero overhead.

Trace files are written under:
    <workspace>/traces/<ticket_slug>__<timestamp>/
        01_ticket.json
        02_investigation.json
        03_discovery.json
        04_ranking.json
        05_planning.json
        06_candidate_validation.json
        07_localization.json
        08_generation.json
        09_scope_validation.json
        10_build.json
        11_summary.json
        report.md

Author: Deepak Madgani
Date: June 2026
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trace_enabled() -> bool:
    return os.environ.get("TRACE_MODE", "false").lower() in ("true", "1", "yes")


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len] if slug else "unknown"


def _infer_sources(signals: List[str]) -> List[str]:
    """Map signal names to canonical source labels."""
    sources: set = set()
    for s in signals:
        sl = str(s).lower()
        if any(k in sl for k in ("sqlite", "symbol", "fts")):
            sources.add("sqlite")
        if any(k in sl for k in ("fs", "filesystem", "path_segment", "extension", "keyword_path")):
            sources.add("filesystem")
        if any(k in sl for k in ("java", "stereotype")):
            sources.add("javaparser")
        if any(k in sl for k in ("ts_decorator", "typescript", "angular")):
            sources.add("typescript_ast")
        if "graph" in sl:
            sources.add("graph")
        if any(k in sl for k in ("vec", "vector")):
            sources.add("vector")
        if "ownership" in sl:
            sources.add("ownership")
        if "sibling" in sl:
            sources.add("sibling_cluster")
    return sorted(sources) if sources else ["unknown"]


def _signals_reason(signals: List[str], confidence: float) -> str:
    if not signals:
        return f"No strong signals; confidence={confidence:.3f}"
    top = ", ".join(str(s) for s in signals[:5])
    return f"{top}; confidence={confidence:.3f}"


def _safe(obj: Any) -> Any:
    """Make an object JSON-serialisable (fallback to str)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionTracer
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionTracer:
    """
    Write-only forensic tracer.  Pass via WorkflowAgents.tracer.

    All public methods are safe to call unconditionally:
    they become no-ops when TRACE_MODE is not enabled.
    """

    def __init__(self, workspace_path: str) -> None:
        self.enabled: bool = _trace_enabled()
        self._workspace: Path = Path(workspace_path)
        self._start: datetime = datetime.now()
        self._finalized: bool = False
        self.trace_dir: Optional[Path] = None
        self.ticket_id: str = ""
        self._acc: Dict[str, Any] = {}

    # ─── Internal ────────────────────────────────────────────────────────────

    def _write(self, filename: str, data: dict) -> None:
        if not self.enabled or self.trace_dir is None:
            return
        try:
            out = json.dumps(_safe(data), indent=2, ensure_ascii=False)
            (self.trace_dir / filename).write_text(out, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"[Tracer] write failed ({filename}): {exc}")

    @staticmethod
    def _task_dict(task: Any) -> dict:
        tt = task.task_type
        lang = task.language
        return {
            "id": getattr(task, "id", "?"),
            "title": getattr(task, "title", ""),
            "file_path": getattr(task, "file_path", ""),
            "task_type": getattr(tt, "value", str(tt)),
            "language": getattr(lang, "value", str(lang)),
            "target_method": getattr(task, "target_method", None),
            "target_class": getattr(task, "target_class", None),
            "localization_confidence": round(getattr(task, "localization_confidence", 0.0), 4),
            "localization_reason": getattr(task, "localization_reason", None),
            "new_file_creation_allowed": getattr(task, "new_file_creation_allowed", False),
        }

    # ─── Phase 1: Ticket ─────────────────────────────────────────────────────

    def record_ticket(self, ticket: Any, **kwargs) -> None:
        """Call at the start of investigate_node.  Creates the trace directory."""
        if not self.enabled:
            return
        ticket_id = getattr(ticket, "ticket_id", "unknown")
        title = getattr(ticket, "title", "")
        slug = _slugify(title or ticket_id)
        ts = self._start.strftime("%Y%m%d_%H%M%S")
        self.ticket_id = ticket_id
        from ticket_to_code.tracing.paths import get_trace_root
        self.trace_dir = get_trace_root() / "execution_traces" / f"{slug}__{ts}"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📝 ExecutionTracer → {self.trace_dir}")

        self._acc = {
            "ticket_id": ticket_id,
            "title": title,
            "start_time": self._start.isoformat(),
        }
        data = {
            "ticket_id": ticket_id,
            "title": title,
            "description": getattr(ticket, "description", ""),
            "priority": str(getattr(ticket, "priority", "")),
            "labels": list(getattr(ticket, "labels", [])),
            "acceptance_criteria": list(getattr(ticket, "acceptance_criteria", [])),
            "created_date": str(getattr(ticket, "created_date", "")),
        }
        self._write("01_ticket.json", data)

    # ─── Phase 2: Investigation / Analysis ───────────────────────────────────

    def record_analysis(
        self,
        requirements: Any = None,
        investigation: Any = None,
        solution_guidance: Any = None,
        **kwargs,
    ) -> None:
        """Call at the end of unified_analysis_node."""
        if not self.enabled:
            return
        path = "code_changes" if requirements else "solution_guidance"
        self._acc["path"] = path

        req_data = None
        if requirements:
            req_data = {
                "ticket_type": str(getattr(requirements, "ticket_type", "")),
                "affected_components": list(getattr(requirements, "affected_components", [])),
                "technical_requirements": list(getattr(requirements, "technical_requirements", [])),
                "business_requirements": list(getattr(requirements, "business_requirements", [])),
                "estimated_complexity": getattr(requirements, "estimated_complexity", 0),
                "requires_code_changes": getattr(requirements, "requires_code_changes", None),
            }

        inv_data = None
        if investigation:
            inv_data = {
                "ticket_type": str(getattr(investigation, "ticket_type", "")),
                "requires_code_changes": getattr(investigation, "requires_code_changes", False),
                "root_cause_hypothesis": getattr(investigation, "root_cause_hypothesis", None),
                "affected_systems": list(getattr(investigation, "affected_systems", [])),
                "recommended_action": getattr(investigation, "recommended_action", ""),
                "confidence": round(getattr(investigation, "confidence", 0.0), 4),
            }

        sg_data = None
        if solution_guidance:
            sg_data = {
                "summary": str(getattr(solution_guidance, "summary", "")),
                "steps": list(getattr(solution_guidance, "steps", [])),
            }

        self._write("02_investigation.json", {
            "path": path,
            "investigation": inv_data,
            "requirements": req_data,
            "solution_guidance": sg_data,
        })

    # ─── Phase 3+4: Discovery & Ranking ──────────────────────────────────────

    def record_discovery(self, candidates: List[dict], **kwargs) -> None:
        """
        Call at the end of discovery_node.
        candidates: list of dicts with keys path, confidence, signals, ...
        Stores EVERY candidate (not just winners).
        """
        if not self.enabled:
            return

        ranked = sorted(candidates, key=lambda c: c.get("confidence", 0.0), reverse=True)
        items = []
        for idx, c in enumerate(ranked, 1):
            signals = list(c.get("signals", []))
            sources = _infer_sources(signals)
            items.append({
                "rank": idx,
                "path": c.get("path", ""),
                "confidence": round(float(c.get("confidence", 0)), 4),
                "sources": sources,
                "signals": signals,
                "retrieval_reason": _signals_reason(signals, float(c.get("confidence", 0))),
            })

        # Explain why #1 beat #2
        rank_explanation = ""
        if len(items) >= 2:
            w, r = items[0], items[1]
            delta = round(w["confidence"] - r["confidence"], 4)
            rank_explanation = (
                f"'{w['path']}' scored {w['confidence']} vs "
                f"'{r['path']}' scored {r['confidence']} (Δ={delta}). "
                f"Winner signals: {w['signals']}. "
                f"Runner-up signals: {r['signals']}."
            )

        self._acc["top_candidates"] = items[:5]
        self._acc["total_candidates_discovered"] = len(items)

        self._write("03_discovery.json", {
            "total_candidates": len(items),
            "candidates": items,
        })
        self._write("04_ranking.json", {
            "method": (
                "unified_score  "
                "weights: fs=0.20, sql=0.20, ja=0.10, ts=0.10, gr=0.20, vec=0.20  "
                "× ownership_boost (1 + 0.5 × s_own)"
            ),
            "total_ranked": len(items),
            "ranking": items,
            "winner_vs_runner_up": rank_explanation,
            "top_3": items[:3],
        })

    # ─── Phase 5: Planning ───────────────────────────────────────────────────

    def record_planning(self, plan: Any, active_discovered: List[dict], **kwargs) -> None:
        """Call at the end of plan_node."""
        if not self.enabled:
            return
        tasks = [self._task_dict(t) for t in plan.tasks]
        self._acc["planned_tasks"] = tasks
        self._write("05_planning.json", {
            "architectural_pattern": str(getattr(plan, "pattern", "")),
            "affected_modules": list(getattr(plan, "affected_modules", [])),
            "total_tasks": len(tasks),
            "tasks": tasks,
            "api_changes_count": len(getattr(plan, "api_changes", [])),
            "candidates_passed_to_planner": [
                {
                    "path": c.get("path", ""),
                    "confidence": round(float(c.get("confidence", 0)), 4),
                    "signals": list(c.get("signals", [])),
                }
                for c in (active_discovered or [])
            ],
        })

    # ─── Phase 6: Candidate Validation ───────────────────────────────────────

    def record_validation(
        self,
        status: str,
        plan_tasks: List[Any],
        discovered_paths: set,
        invalid: List[str],
        weak_tasks: List[str],
        blacklisted: List[str],
        retry_count: int,
        reason: str = "",
     **kwargs,) -> None:
        """Call before every return in validate_candidates_node."""
        if not self.enabled:
            return

        approved = []
        rejected = []
        for task in plan_tasks:
            fp = getattr(task, "file_path", "")
            tt_obj = getattr(task, "task_type", None)
            tt = getattr(tt_obj, "value", str(tt_obj)) if tt_obj else "?"
            if tt == "read_only":
                continue
            entry = {
                "path": fp,
                "task_type": tt,
                "in_discovery": fp in discovered_paths,
            }
            if fp in invalid:
                entry["rejection_reason"] = "path_not_in_repo"
                rejected.append(entry)
            elif fp in weak_tasks:
                entry["rejection_reason"] = "low_confidence_or_ownership_failure"
                rejected.append(entry)
            else:
                approved.append(entry)

        self._acc["validation_status"] = status
        self._acc["validation_rejected"] = rejected
        self._write("06_candidate_validation.json", {
            "status": status,
            "retry_count": retry_count,
            "failure_reason": reason,
            "approved_candidates": approved,
            "rejected_candidates": rejected,
            "blacklisted_accumulation": list(blacklisted),
        })

    # ─── Phase 7: Localization ────────────────────────────────────────────────

    def record_localization(
        self,
        planned_paths: List[str] = None,
        localized_tasks: List[Any] = None,
     **kwargs,) -> None:
        """
        Call at the end of localize_node.
        planned_paths: file_path values BEFORE localization (snapshot taken before the call)
        localized_tasks: tasks AFTER localize_planned_tasks()
        """
        if not self.enabled:
            return

        rows = []
        planned_paths = planned_paths or []
        localized_tasks = localized_tasks or []
        for planned_fp, after in zip(planned_paths, localized_tasks):
            resolved_fp = getattr(after, "file_path", "")
            tt_obj = getattr(after, "task_type", None)
            tt = getattr(tt_obj, "value", str(tt_obj)) if tt_obj else "?"
            candidates = getattr(after, "file_candidates", [])
            alternates = []
            for c in (candidates[1:] if len(candidates) > 1 else []):
                alternates.append({
                    "path": getattr(c, "path", str(c)),
                    "confidence": round(float(getattr(c, "confidence", 0)), 4),
                    "signals": list(getattr(c, "signals", [])),
                })
            rows.append({
                "planned_path": planned_fp,
                "resolved_path": resolved_fp,
                "path_changed": planned_fp != resolved_fp,
                "resolved_task_type": tt,
                "localization_confidence": round(float(getattr(after, "localization_confidence", 0.0)), 4),
                "localization_reason": getattr(after, "localization_reason", ""),
                "alternate_candidates": alternates,
            })

        self._acc["localized_tasks"] = rows
        self._write("07_localization.json", {"tasks": rows})

    # ─── Phase 2F: Ownership Completeness ────────────────────────────────────

    def record_evidence_ranking(self, ranked_files: list, **kwargs) -> None:
        """
        Call at the end of evidence_ranking_node.
        Writes evidence_ranking.json with per-file scores, signals, penalties.
        """
        if not self.enabled:
            return
        rows = [rf.to_dict() if hasattr(rf, 'to_dict') else rf for rf in ranked_files]
        self._write("evidence_ranking.json", {
            "total_ranked": len(rows),
            "top_10": rows[:10],
            "all_files": rows,
        })

    def record_ownership_completeness(self, data: dict, **kwargs) -> None:
        """
        Call at the end of ownership_completeness_node.
        Writes 07b_ownership_completeness.json with the full analysis.
        """
        if not self.enabled:
            return
        self._acc["ownership_completeness"] = data
        self._write("07b_ownership_completeness.json", data)

    def record_artifact_classification(self, classifications: list[dict], **kwargs) -> None:
        """
        Writes artifact_classification.json
        """
        if not self.enabled:
            return
        self._acc["artifact_classification"] = classifications
        self._write("artifact_classification.json", {"classifications": classifications})

    def record_generation_routing(self, routing: list[dict], **kwargs) -> None:
        """
        Writes generation_routing.json
        """
        if not self.enabled:
            return
        self._acc["generation_routing"] = routing
        self._write("generation_routing.json", {"routing": routing})

    # ─── Phase 8+9: Generation & Scope Validation ────────────────────────────

    def record_generation(
        self,
        generated_code: List[Any],
        generated_tests: List[Any],
        allowed_files: List[str],
        readonly_files: List[str],
        scope_violations: List[str],
     **kwargs,) -> None:
        """Call at the end of generate_code_node (and optionally generate_tests_node)."""
        if not self.enabled:
            return

        def _gen(g: Any) -> dict:
            ct = getattr(g, "change_type", None)
            return {
                "file_path": getattr(g, "file_path", ""),
                "task_type": getattr(ct, "value", str(ct)) if ct else "?",
                "language": str(getattr(g, "language", "")),
                "char_count": len(getattr(g, "content", "")),
                "line_count": len(getattr(g, "content", "").splitlines()),
            }

        code_items = [_gen(g) for g in (generated_code or [])]
        test_items = [_gen(g) for g in (generated_tests or [])]

        self._write("08_generation.json", {
            "allowed_writable_files": list(allowed_files or []),
            "readonly_context_files": list(readonly_files or []),
            "generated_code_files": code_items,
            "generated_test_files": test_items,
            "total_files_written": len(code_items) + len(test_items),
        })
        self._write("09_scope_validation.json", {
            "allowed_writable_files": list(allowed_files or []),
            "actual_files_written": [g["file_path"] for g in code_items],
            "scope_violations": list(scope_violations or []),
            "violation_count": len(scope_violations or []),
        })

        modified = [g["file_path"] for g in code_items if g.get("task_type") != "create"]
        created = [g["file_path"] for g in code_items if g.get("task_type") == "create"]
        self._acc["modified_files"] = modified
        self._acc["created_files"] = created
        self._acc["scope_violations"] = list(scope_violations or [])

    # ─── Phase 10: Build ─────────────────────────────────────────────────────

    def record_build(self, build_result: Any, ts_errors: Optional[List[str]] = None, **kwargs) -> None:
        """Call at the end of build_node."""
        if not self.enabled:
            return
        if build_result is None:
            return
        st = getattr(build_result, "status", None)
        status_str = getattr(st, "value", str(st)) if st else "unknown"
        errors = list(getattr(build_result, "errors", []) or [])
        warnings = list(getattr(build_result, "warnings", []) or [])
        self._acc["build_status"] = status_str
        self._acc["build_errors"] = errors[:5]
        self._write("10_build.json", {
            "status": status_str,
            "exit_code": getattr(build_result, "exit_code", -1),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings[:20],
            "duration_seconds": round(float(getattr(build_result, "duration_seconds", 0)), 2),
            "ts_compile_check": {
                "ran": ts_errors is not None,
                "error_count": len(ts_errors or []),
                "errors": (ts_errors or [])[:20],
            },
        })

    # ─── Phase 11: Summary + Markdown ────────────────────────────────────────

    def finalize(self, state: dict) -> None:
        """
        Write 11_summary.json and report.md.
        Idempotent — safe to call from multiple terminal nodes.
        """
        if not self.enabled or self._finalized:
            return
        self._finalized = True

        end_time = datetime.now()
        duration = (end_time - self._start).total_seconds()

        # Pull build result from state if record_build() was not called yet
        build_result = state.get("build_result")
        if build_result and "build_status" not in self._acc:
            self.record_build(build_result)

        summary = {
            **self._acc,
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "workflow_status": self._determine_status(state),
            "selected_files": [
                t.get("resolved_path", t.get("planned_path", ""))
                for t in (self._acc.get("localized_tasks") or [])
            ],
            "modified_files": self._acc.get("modified_files", []),
            "created_files": self._acc.get("created_files", []),
            "scope_violations": self._acc.get("scope_violations", []),
            "build_status": self._acc.get("build_status", "not_run"),
            "top_candidates": self._acc.get("top_candidates", []),
            "total_candidates_discovered": self._acc.get("total_candidates_discovered", 0),
        }

        self._write("11_summary.json", summary)
        self._write_markdown_report(summary, state)
        logger.info(f"📝 Trace finalised → {self.trace_dir}")

    def _determine_status(self, state: dict) -> str:
        build = state.get("build_result")
        if build:
            st = getattr(getattr(build, "status", None), "value", "")
            if st == "success":
                test = state.get("test_result")
                if test:
                    tst = getattr(getattr(test, "status", None), "value", "")
                    return "success" if tst in ("all_passed", "success") else "tests_failed"
                return "build_ok_tests_pending"
            return "build_failed"
        if state.get("solution_guidance"):
            return "solution_only"
        return "partial"

    def _write_markdown_report(self, summary: dict, state: dict) -> None:  # noqa: C901
        if not self.enabled or self.trace_dir is None:
            return

        lines: List[str] = []
        add = lines.append  # shorthand

        add("# Ticket-to-Code Execution Trace Report")
        add("")
        add(f"| Field | Value |")
        add(f"|-------|-------|")
        add(f"| **Ticket** | {summary.get('title', '?')} |")
        add(f"| **ID** | `{summary.get('ticket_id', '?')}` |")
        add(f"| **Status** | `{summary.get('workflow_status', '?')}` |")
        add(f"| **Duration** | {summary.get('duration_seconds', 0):.1f}s |")
        add(f"| **Start** | {summary.get('start_time', '?')} |")
        add(f"| **End** | {summary.get('end_time', '?')} |")
        add(f"| **Trace dir** | `{self.trace_dir}` |")
        add("")

        # ── 1. Investigation ─────────────────────────────────────────────────
        add("## 1. Investigation")
        add("")
        add(f"**Execution path:** `{summary.get('path', 'unknown')}`")
        add("")

        # ── 2. Discovery ─────────────────────────────────────────────────────
        add("## 2. Discovery — All Candidates")
        add("")
        add(f"**Total candidates found:** {summary.get('total_candidates_discovered', 0)}")
        add("")
        top = summary.get("top_candidates", [])
        if top:
            add("### Top 5 Candidates (by unified score)")
            add("")
            add("| Rank | File | Score | Sources | Key Signals |")
            add("|------|------|-------|---------|-------------|")
            for c in top:
                srcs = ", ".join(c.get("sources", []))
                sigs = "; ".join(str(s) for s in c.get("signals", [])[:3])
                add(f"| {c['rank']} | `{c['path']}` | {c['confidence']} | {srcs} | {sigs} |")
            add("")

        # ── 3. Ranking ───────────────────────────────────────────────────────
        add("## 3. Ranking")
        add("")
        add("**Method:** unified_score = (fs×0.20 + sql×0.20 + ja×0.10 + ts×0.10 + gr×0.20 + vec×0.20) × (1 + 0.5×own)")
        add("")
        if len(top) >= 2:
            w, r = top[0], top[1]
            add(f"**Why `{w['path']}` beat `{r['path']}`:**")
            add(f"> Score {w['confidence']} vs {r['confidence']} (Δ={round(w['confidence'] - r['confidence'], 4)}).  ")
            add(f"> Winner signals: `{w['signals']}`  ")
            add(f"> Runner-up signals: `{r['signals']}`")
            add("")

        # ── 4. Planning ──────────────────────────────────────────────────────
        add("## 4. Planning")
        add("")
        planned = summary.get("planned_tasks", [])
        if planned:
            add("| # | Title | File | Type | Language |")
            add("|---|-------|------|------|----------|")
            for i, t in enumerate(planned, 1):
                title = (t.get("title", "") or "")[:50]
                add(f"| {i} | {title} | `{t.get('file_path', '')}` | `{t.get('task_type', '')}` | {t.get('language', '')} |")
        add("")

        # ── 5. Candidate Validation ──────────────────────────────────────────
        add("## 5. Candidate Validation")
        add("")
        vs = summary.get("validation_status", "?")
        icon = "✅" if vs == "candidates_validated" else "⚠️"
        add(f"**Result:** {icon} `{vs}`")
        rejected = summary.get("validation_rejected", [])
        if rejected:
            add("")
            add("### Rejected / Blacklisted")
            add("")
            for r in rejected:
                add(f"- `{r.get('path', '?')}` — **{r.get('rejection_reason', '?')}**")
        add("")

        # ── 6. Localization ──────────────────────────────────────────────────
        add("## 6. Localization")
        add("")
        localized = summary.get("localized_tasks", [])
        if localized:
            add("| Planned Path | → Resolved Path | Changed? | Confidence | Reason |")
            add("|-------------|-----------------|----------|-----------|--------|")
            for t in localized:
                changed = "**YES ✅**" if t.get("path_changed") else "no"
                reason = (t.get("localization_reason", "") or "")[:60]
                add(
                    f"| `{t.get('planned_path', '')}` "
                    f"| `{t.get('resolved_path', '')}` "
                    f"| {changed} "
                    f"| {t.get('localization_confidence', 0):.3f} "
                    f"| {reason} |"
                )
        add("")

        # ── 7. Generation ────────────────────────────────────────────────────
        add("## 7. Code Generation")
        add("")
        modified = summary.get("modified_files", [])
        created = summary.get("created_files", [])
        if not modified and not created:
            add("No files generated.")
        if modified:
            add("### Modified Files")
            for f in modified:
                add(f"- `{f}`")
        if created:
            add("")
            add("### Created Files (new)")
            for f in created:
                add(f"- `{f}` ⚠️")
        violations = summary.get("scope_violations", [])
        if violations:
            add("")
            add("### ⚠️ Scope Violations")
            for v in violations:
                add(f"- {v}")
        add("")

        # ── 8. Build ─────────────────────────────────────────────────────────
        add("## 8. Build")
        add("")
        build_status = summary.get("build_status", "not_run")
        icon = "✅" if build_status == "success" else ("❌" if build_status == "failure" else "⏭️")
        add(f"**Status:** {icon} `{build_status}`")
        build_errors = summary.get("build_errors", [])
        if build_errors:
            add("")
            add("### Build Errors (first 5)")
            add("")
            add("```")
            for e in build_errors[:5]:
                add(str(e))
            add("```")
        add("")

        # ── Footer ───────────────────────────────────────────────────────────
        add("---")
        add("")
        add("*Full trace files available in:*")
        add(f"`{self.trace_dir}`")

        try:
            (self.trace_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"[Tracer] report write failed: {exc}")
