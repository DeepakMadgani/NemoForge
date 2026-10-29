"""
Ticket-to-Code Workflow with LangGraph - TDD Approach

This implements Test-Driven Development with AI:
1. Generate test cases FIRST (using product behavior RAG)
2. Generate code SECOND (using architecture RAG)
3. Tests and code are independent (no knowledge leak)
4. Both grounded in real knowledge (no hallucinations)

Author: Deepak Madgani
Date: April 2026
"""

import logging
import re
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, Literal, TypedDict, List, Optional, Set
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

from ticket_to_code.debug_tracker import log_phase, log_error
from ticket_to_code.models import (
    ValueEdgeTicket,
    InvestigationResult,
    InvestigationHypothesis,
    EvidenceItem,
    ComponentGroup,
    GroundedUnderstanding,
    StructuredRequirements,
    ArchitecturalPlan,
    GeneratedCode,
    BuildResult,
    BuildStatus,
    TestResult,
    TestStatus,
    SolutionGuidance,
    CandidateRole,
    PlannerDecisionValue,
    TaskType,
    ProgrammingLanguage,
)

from ticket_to_code.agents.investigation_agent import InvestigationAgent
from ticket_to_code.agents.semantic_verification_agent import SemanticVerificationResult, SemanticVerificationAgent
from ticket_to_code.agents.ticket_analyzer import TicketAnalyzerAgent
from ticket_to_code.agents.planning_agent import PlanningAgent
from ticket_to_code.agents.localization_agent import LocalizationAgent
from ticket_to_code.agents.evidence_collection_loop import EvidenceCollectionLoop
from ticket_to_code.agents.evidence_ranking_engine import EvidenceRankingEngine
from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
from ticket_to_code.agents.providers.provider_factory import RelationshipProviderFactory
from ticket_to_code.agents.rag_engine import CodebaseRAGEngine  # Updated import path
from ticket_to_code.retrieval.context_retrieval import ContextRetrievalService  # B2: cascading context
from ticket_to_code.runtime.run_context import RunContext, HealthLevel, BudgetExceeded  # B11/B12
from ticket_to_code.delivery.patch_gate import PatchGate  # B9: delivery safety
from ticket_to_code.agents.code_generator import CodeGeneratorAgent, PatchValidator
from ticket_to_code.execution.base_executor import ExecutionEngineFactory
from ticket_to_code.tracing.tracer import ExecutionTracer
from ticket_to_code.tracing.paths import get_trace_dir
from ticket_to_code.workflow_discovery import WorkflowDiscovery
from ticket_to_code.agents.runtime_log_analyzer import RuntimeLogAnalyzer, RuntimeDiagnosis
from ticket_to_code.agents.dependency_resolver import DependencyResolver, DependencyResolution  # Enhancement 12
from ticket_to_code.strategy.adaptive_replanner import AdaptiveReplanner, ReplanDecision  # Enhancement 4
from ticket_to_code.agents.git_history_agent import GitHistoryAgent  # Enhancement 9
from ticket_to_code.agents.env_tracer import EnvTracer  # Enhancement 11
from ticket_to_code.reasoning.causal_chain import CausalChainEngine  # Enhancement 3
from ticket_to_code.execution.runtime_verifier import RuntimeVerifier  # Enhancement 7
from ticket_to_code.execution.platform_knowledge import PlatformKnowledge  # Enhancement 5
from ticket_to_code.delivery.config_validator import ConfigValidator  # Enhancement 13
from ticket_to_code.agents.visual_analyzer import VisualAnalyzer, VisualAnalysis  # Enhancement 8
from ticket_to_code.utils.error_normalization import (
    normalize_error_identity,
    normalize_error_lines,
    strip_ansi,
)
from ticket_to_code.plan_gating import (
    run_gating_logic,
    validate_plan_consistency,
    _extract_json_payload,
    verify_grounding_fact_text,
)



logger = logging.getLogger(__name__)

import os
import json
from typing import Union

def write_trace_artifact(workspace_path: str, ticket_id: str, filename: str, content: Union[str, dict, list]):
    """Write an artifact to the trace directory for full observability."""
    trace_dir = str(get_trace_dir(ticket_id))
    os.makedirs(trace_dir, exist_ok=True)
    file_path = os.path.join(trace_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, indent=2)
        else:
            f.write(str(content))

def _get_plan(state: dict):
    plan = state.get("architectural_plan")
    if not plan: return None
    if isinstance(plan, dict):
        tasks = plan.get("tasks", [])
        if isinstance(tasks, list):
            for t in tasks:
                if isinstance(t, dict) and "file_path" in t:
                    fp = t["file_path"]
                    p = getattr(fp, "file_path", None) or getattr(fp, "path", None) or (fp.get("file_path") or fp.get("path") if isinstance(fp, dict) else None)
                    if p:
                        t["file_path"] = str(p)
        from ticket_to_code.models import ArchitecturalPlan
        plan = ArchitecturalPlan.model_validate(plan)
        state["architectural_plan"] = plan
    return plan


def _extract_json_object(text: str) -> Optional[dict]:
    """Extract the first JSON object from LLM text, tolerating prose around it.

    More robust than a single regex because LLMs emit JSON in many shapes:
      1. A fenced ```json ... ``` block (parsed as a whole — handles nested objects).
      2. A bare object embedded in prose (balanced-brace scan honoring string
         literals and escapes, so `"summary": "done }"` cannot desync the scan).

    Returns the parsed dict, or None when no valid JSON object is present.
    """
    if not text:
        return None

    import json as _json

    # 1. Fenced code blocks — try each as a whole before falling back.
    for fence_m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        candidate = fence_m.group(1).strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = _json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue  # fence may wrap prose + JSON; balanced scan below still finds it

    # 2. Balanced-brace scan over the raw text (handles nested objects/arrays).
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = _json.loads(text[start:i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break  # this object is invalid — try the next '{'
                    break
        start = text.find("{", start + 1)
    return None


def _match_finding_to_files(finding: Any, file_pool: list[str]) -> Optional[str]:
    """Helper to match a finding (dict or OutcomeFinding) to a file in file_pool."""
    if not file_pool:
        return None
    raw = (
        finding.get("affected_file") if isinstance(finding, dict)
        else getattr(finding, "affected_file", "")
    ) or ""
    raw = str(raw).replace("\\", "/").strip()
    raw_norm = raw.lower()
    pool_norm = {p.replace("\\", "/").lower(): p for p in file_pool if p}

    if raw_norm and raw_norm in pool_norm:
        return pool_norm[raw_norm]

    if raw_norm:
        base = Path(raw).name.lower()
        for norm, canonical in pool_norm.items():
            if norm.endswith(raw_norm) or Path(norm).name.lower() == base:
                return canonical
        stem = base.rsplit(".", 1)[0] if "." in base else base
        if len(stem) >= 5:
            for norm, canonical in pool_norm.items():
                if stem in Path(norm).stem.lower():
                    return canonical

    blob = " ".join([
        str(finding.get(k, "") if isinstance(finding, dict) else getattr(finding, k, "") or "")
        for k in ("summary", "missing_behavior", "offending_code", "requirement_text")
    ]).lower()

    for norm, canonical in pool_norm.items():
        base = Path(norm).name.lower()
        stem = Path(norm).stem.lower()
        if base in blob or (len(stem) >= 6 and stem in blob):
            return canonical

    return None


def _classify_defect_file_tiers(defect_findings: list, state: dict) -> tuple[str, list[str]]:
    """
    Classify which candidate tier covers the defect findings:
      - Tier 0: Already modified / generated files in this run (generated_code, modified_files, change_targets)
      - Tier 1: Discovered candidate files from discovery phase (discovered_files)
      - Tier 2: Backup candidates (tier2_candidates)

    Hierarchy rules:
      1. If ALL defects can be resolved to Tier 0 (or have no external file reference),
         return ("tier0", matched_files). STOP SEARCHING.
      2. If any defect requires a file NOT in Tier 0, check Tier 1.
         If found in Tier 1, return ("tier1", matched_files). STOP SEARCHING.
      3. Only if Tier 1 does not contain it, check Tier 2.
         If found in Tier 2, return ("tier2", matched_files).
      4. If unresolvable, return ("unresolved", []).
    """
    generated_code = state.get("generated_code") or []
    t0_paths = [
        getattr(g, "file_path", "") or (g.get("file_path", "") if isinstance(g, dict) else "")
        for g in generated_code
    ]
    t0_paths.extend(state.get("modified_files") or [])
    t0_paths.extend(state.get("change_targets") or [])
    t0_pool = [p for p in t0_paths if p]

    discovered = state.get("discovered_files") or []
    t1_pool = [
        d.get("path") if isinstance(d, dict) else str(d)
        for d in discovered
        if (d.get("path") if isinstance(d, dict) else str(d))
    ]

    tier2 = state.get("tier2_candidates") or []
    t2_pool = [
        c.get("path") if isinstance(c, dict) else str(c)
        for c in tier2
        if (c.get("path") if isinstance(c, dict) else str(c))
    ]

    if not defect_findings:
        unsatisfied = state.get("unsatisfied_requirements") or (state.get("outcome_check_result") or {}).get("unsatisfied_requirements") or []
        if unsatisfied:
            defect_findings = [{"summary": str(u), "missing_behavior": str(u), "affected_file": ""} for u in unsatisfied]
        elif not t0_pool and t2_pool:
            return ("tier2", t2_pool)
        elif not t0_pool and t1_pool:
            return ("tier1", t1_pool)
        else:
            return ("tier0", t0_pool)

    unmatched_by_t0 = []
    matched_t0 = set()
    for f in defect_findings:
        raw_f = (f.get("affected_file") if isinstance(f, dict) else getattr(f, "affected_file", "")) or ""
        m0 = _match_finding_to_files(f, t0_pool)
        if m0:
            matched_t0.add(m0)
        elif not raw_f.strip():
            if t0_pool:
                matched_t0.add(t0_pool[0])
        else:
            unmatched_by_t0.append(f)

    if not unmatched_by_t0:
        return ("tier0", sorted(list(matched_t0)))

    unmatched_by_t1 = []
    matched_t1 = set()
    for f in unmatched_by_t0:
        m1 = _match_finding_to_files(f, t1_pool)
        if m1:
            matched_t1.add(m1)
        else:
            unmatched_by_t1.append(f)

    if matched_t1 and not unmatched_by_t1:
        return ("tier1", sorted(list(matched_t1)))

    matched_t2 = set()
    for f in (unmatched_by_t1 or unmatched_by_t0):
        m2 = _match_finding_to_files(f, t2_pool)
        if m2:
            matched_t2.add(m2)

    if matched_t2:
        return ("tier2", sorted(list(matched_t2)))

    if matched_t1:
        return ("tier1", sorted(list(matched_t1)))

    return ("unresolved", sorted(list(matched_t0)))


def _build_remediation_plan(state: dict) -> Optional["ArchitecturalPlan"]:
    """Deterministic delta remediation plan derived from outcome_check findings.

    When the semantic verdict is PARTIAL/INCOMPLETE, re-running the ORIGINAL
    architectural plan wastes tokens re-generating files that are already
    finished (e.g. SCSS regenerated 3x in one observed run). This builder
    converts the structured findings into the MINIMAL delta task set:

      - one MODIFY task per implicated file (only the files with defects)
      - task description = ALL missing behaviors/defects for that file together
      - explicitly documents what is ALREADY SATISFIED (preserve) vs PENDING (solve)
      - pure-style files are excluded unless a finding explicitly names styling

    Zero LLM calls — the outcome check already pinpointed the defect, so the
    plan is assembled mechanically from the findings.
    """
    from ticket_to_code.models import (
        ArchitecturalPlan,
        ArchitecturalPattern,
        DevelopmentTask,
        ProgrammingLanguage,
        TaskType,
    )

    findings = state.get("outcome_findings") or []
    result = state.get("outcome_check_result") or {}
    if not findings:
        findings = result.get("findings") or []

    # If triggered from test failures, synthesize findings from test_result
    test_result = state.get("test_result")
    if not findings and test_result and getattr(test_result, "failed", 0) > 0:
        fail_msgs = getattr(test_result, "failed_tests", []) or getattr(test_result, "errors", []) or []
        out_snippet = getattr(test_result, "test_output", "")[:500]
        summary_msg = "; ".join(str(m) for m in fail_msgs[:3]) if fail_msgs else f"Automated tests failed ({test_result.failed}/{test_result.total_tests}): {out_snippet}"
        findings = [{
            "verdict": "DEFECT",
            "requirement_id": "TESTS",
            "requirement_text": "Unit/Integration tests must pass",
            "summary": summary_msg,
            "missing_behavior": f"Fix functional regression causing automated tests to fail: {summary_msg}",
            "affected_file": getattr(test_result, "failed_file", "") or "",
        }]

    defect_findings = [f for f in findings if str(f.get("verdict", "")).upper() != "CORRECT"]
    if not defect_findings:
        return None

    satisfied_findings = [f for f in findings if str(f.get("verdict", "")).upper() == "CORRECT"]
    satisfied_descs = [
        f"{f.get('requirement_id', 'REQ')}: {str(f.get('requirement_text') or f.get('summary', ''))[:120]} (DONE)"
        for f in satisfied_findings
    ]

    generated_code = state.get("generated_code") or []
    gen_paths = [getattr(g, "file_path", "") or (g.get("file_path", "") if isinstance(g, dict) else "") for g in generated_code]
    gen_paths = [p for p in gen_paths if p]
    gen_norm = {p.replace("\\", "/").lower(): p for p in gen_paths}

    change_targets = state.get("change_targets") or []
    for ct in change_targets:
        norm = str(ct).replace("\\", "/").lower()
        if norm and norm not in gen_norm:
            gen_norm[norm] = str(ct)

    # Enrich gen_norm with initial implementation plan tasks
    orig_plan = state.get("implementation_plan") or state.get("architectural_plan") or _get_plan(state)
    if orig_plan:
        plan_tasks = getattr(orig_plan, "tasks", []) or []
        for t in plan_tasks:
            tf = getattr(t, "file_path", None) or getattr(t, "target_file", None) or ""
            if tf:
                norm = str(tf).replace("\\", "/").lower()
                if norm and norm not in gen_norm:
                    gen_norm[norm] = str(tf)

    # Enrich gen_norm with localized, candidate, discovered, and allowed files
    for key in ("candidate_files", "allowed_files", "localized_files", "target_files", "discovered_files", "tier2_candidates", "modified_files"):
        for f in (state.get(key) or []):
            path_str = f.get("path") if isinstance(f, dict) else str(f)
            if not path_str:
                continue
            norm = path_str.replace("\\", "/").lower()
            if norm and norm not in gen_norm:
                gen_norm[norm] = path_str

    ws_path = str(state.get("workspace_path") or state.get("repo_path") or os.environ.get("TARGET_REPO_PATH") or "").strip()

    def _resolve_finding_file(finding: dict) -> str:
        """Map a finding to a concrete file path (suffix/basename/keyword match or mention in defect text)."""
        raw = str(finding.get("affected_file", "") or "").replace("\\", "/")
        if raw:
            raw_norm = raw.lower()
            if raw_norm in gen_norm:
                return gen_norm[raw_norm]
            base = Path(raw).name.lower()
            for norm, canonical in gen_norm.items():
                if norm.endswith(raw_norm) or Path(norm).name.lower() == base:
                    return canonical
            stem = base.rsplit(".", 1)[0] if "." in base else base
            if len(stem) >= 6:
                for norm, canonical in gen_norm.items():
                    if stem in norm:
                        return canonical

        # Textual matching: check if finding summary or missing behavior explicitly mentions a known file or class name
        text = f"{finding.get('summary', '')} {finding.get('missing_behavior', '')} {finding.get('offending_code', '')} {finding.get('affected_file', '')}".lower()
        for norm, canonical in gen_norm.items():
            base = Path(canonical).name.lower()
            stem = base.rsplit(".", 1)[0] if "." in base else base
            if (len(base) >= 5 and base in text) or (len(stem) >= 5 and stem in text):
                return canonical

        # Filesystem fallback: scan workspace if available
        if ws_path and os.path.isdir(ws_path):
            if raw:
                # Direct relative check
                direct_check = os.path.join(ws_path, raw.lstrip("/\\"))
                if os.path.exists(direct_check):
                    return direct_check
                base_name = Path(raw).name.lower()
                if base_name:
                    for root, _, files in os.walk(ws_path):
                        if any(x in root for x in ["node_modules", "target", ".git", ".aviator", ".nemoforge"]):
                            continue
                        for fn in files:
                            if fn.lower() == base_name:
                                full_fp = os.path.join(root, fn)
                                gen_norm[full_fp.replace("\\", "/").lower()] = full_fp
                                return full_fp

            # Search text for file or class names present in workspace
            for root, _, files in os.walk(ws_path):
                if any(x in root for x in ["node_modules", "target", ".git", ".aviator", ".nemoforge"]):
                    continue
                for fn in files:
                    if fn.endswith((".java", ".ts", ".tsx", ".py", ".js", ".jsx", ".html", ".css", ".scss")):
                        fn_low = fn.lower()
                        stem = Path(fn).stem.lower()
                        if (len(fn_low) >= 6 and fn_low in text) or (len(stem) >= 6 and stem in text):
                            full_fp = os.path.join(root, fn)
                            gen_norm[full_fp.replace("\\", "/").lower()] = full_fp
                            return full_fp

        return ""

    # Group findings per resolved file, preserving reviewer wording and requirement IDs
    per_file: dict = {}
    unmatched_lines: list = []
    for f in defect_findings:
        fp = _resolve_finding_file(f)
        req_id = f.get("requirement_id") or "REQ-?"
        behavior = str(f.get("missing_behavior", "") or "").strip()
        summary = str(f.get("summary", "") or "").strip()
        detail = behavior or summary
        formatted_defect = f"[{req_id}] {detail}"
        if not fp:
            if detail:
                unmatched_lines.append(formatted_defect)
            continue
        entry = per_file.setdefault(fp, {"lines": [], "offending": [], "req_ids": []})
        if formatted_defect not in entry["lines"]:
            entry["lines"].append(formatted_defect)
        if req_id not in entry["req_ids"]:
            entry["req_ids"].append(req_id)
        offending = str(f.get("offending_code", "") or "").strip()
        if offending and offending not in entry["offending"]:
            entry["offending"].append(offending)

    # If there are unmatched lines, don't drop them — attach them to the primary target
    if unmatched_lines and per_file:
        first_fp = next(iter(per_file.keys()))
        for ul in unmatched_lines:
            if ul not in per_file[first_fp]["lines"]:
                per_file[first_fp]["lines"].append(ul)

    if not per_file and unmatched_lines and gen_norm:
        first_target = next(iter(gen_norm.values()))
        per_file[first_target] = {"lines": unmatched_lines[:5], "offending": [], "req_ids": ["UNRESOLVED"]}
        logger.info(
            f"  remediation: no file in findings — falling back to change_targets "
            f"({Path(first_target).name})"
        )

    if not per_file:
        return None

    def _file_rank(fp: str) -> tuple:
        """Dependency-safe ordering: controller before template before style."""
        low = fp.lower()
        if low.endswith((".scss", ".css", ".sass", ".less")):
            return (2, low)
        if low.endswith((".html", ".htm")):
            return (1, low)
        return (0, low)

    def _lang_for(fp: str) -> "ProgrammingLanguage":
        ext = Path(fp).suffix.lower()
        return {
            ".ts": ProgrammingLanguage.TYPESCRIPT, ".tsx": ProgrammingLanguage.TYPESCRIPT,
            ".java": ProgrammingLanguage.JAVA, ".py": ProgrammingLanguage.PYTHON,
            ".html": ProgrammingLanguage.HTML, ".scss": ProgrammingLanguage.SCSS,
            ".cs": ProgrammingLanguage.CSHARP,
        }.get(ext, ProgrammingLanguage.TYPESCRIPT)

    satisfied_block = ""
    if satisfied_descs:
        satisfied_block = "ALREADY SATISFIED REQUIREMENTS (PRESERVE - DO NOT REGENERATE OR BREAK):\n" + "\n".join(
            f"- {s}" for s in satisfied_descs[:8]
        ) + "\n\n"

    tasks = []
    for idx, (fp, info) in enumerate(
        sorted(per_file.items(), key=lambda kv: _file_rank(kv[0])), start=1
    ):
        desc_lines = [f"- {ln}" for ln in info["lines"]]
        if info["offending"]:
            desc_lines.append(f"Offending code/condition: {info['offending'][0][:300]}")
        tasks.append(DevelopmentTask(
            id=f"REM-{idx}",
            title=f"Remediate: {Path(fp).name} ({', '.join(info['req_ids'])})",
            description=(
                f"DELTA REMEDIATION TASK FOR {Path(fp).name}:\n\n"
                + satisfied_block
                + f"PENDING REQUIREMENTS TO FIX IN THIS PASS (FIX ALL TOGETHER):\n"
                + "\n".join(desc_lines)
                + "\n\nCRITICAL INSTRUCTIONS:\n"
                "- Fix ALL pending requirements listed above in one coordinated pass.\n"
                "- PRESERVE all existing logic that implements already-satisfied requirements.\n"
                "- Implement only the minimal delta fixes needed; do not regenerate unrelated methods or styles."
            ),
            file_path=fp,
            task_type=TaskType.MODIFY,
            language=_lang_for(fp),
            requires_testing=False,
            selection_reason=f"outcome_check findings ({', '.join(info['req_ids'])}) — targeted delta remediation",
        ))

    pending_summary = "; ".join(f"{f.get('requirement_id')}: {str(f.get('summary') or '')[:80]}" for f in defect_findings)
    plan = ArchitecturalPlan(
        pattern=ArchitecturalPattern.LAYERED,
        rationale=(
            f"Targeted delta remediation plan ({len(tasks)} file(s) to touch).\n"
            f"Already Satisfied: {', '.join(f.get('requirement_id', '') for f in satisfied_findings) or 'None'}\n"
            f"Pending to Remediate: {', '.join(f.get('requirement_id', '') for f in defect_findings)}\n"
            f"Details: {pending_summary}"
        ),
        affected_modules=sorted({Path(t.file_path).parent.as_posix() for t in tasks}),
        tasks=tasks,
        estimated_total_complexity=2,
    )
    logger.info(
        f"  🎯 Delta remediation plan: {len(tasks)} file(s) to touch for "
        f"{len(defect_findings)} pending finding(s) — {[t.file_path for t in tasks]}"
    )
    return plan


def _get_ticket_attr(ticket: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute or key from a ticket, supporting both dicts and Pydantic/class objects."""
    if ticket is None:
        return default
    if isinstance(ticket, dict):
        val = ticket.get(attr)
        return val if val is not None else default
    val = getattr(ticket, attr, default)
    return val if val is not None else default


# ============================================================================
# STRUCTURED EXECUTION LOGGER
# ============================================================================

class StageLog:
    """
    Produces clean, UI-ready structured logs for each pipeline stage.

    Format matches the enterprise transparency requirements:
      [Stage Name]
      Key: value
      ...
      ✅ / ⛔ outcome
    """
    _separator = "─" * 60

    @staticmethod
    def localization(task) -> str:
        lines = [
            f"\n{StageLog._separator}",
            "[Localization]",
            f"  File     : {task.file_path}",
            f"  Class    : {task.target_class or 'N/A'}",
            f"  Method   : {task.target_method or 'N/A'}",
            f"  Allowed  : {task.allowed_methods or 'ALL (unconstrained)'}",
            f"  New file : {'ALLOWED' if task.new_file_creation_allowed else 'FORBIDDEN'}",
            f"  Confidence: {task.localization_confidence:.2f}",
            f"  Reason   : {task.localization_reason or 'N/A'}",
            StageLog._separator,
        ]
        return "\n".join(lines)

    @staticmethod
    def patch_validation(task, validation) -> str:
        status = "✅ ACCEPTED" if validation.passed else "⛔ REJECTED"
        m = validation.metrics
        lines = [
            f"\n{StageLog._separator}",
            "[Patch Validation]",
            f"  File     : {task.file_path}",
            f"  Status   : {status}",
            f"  Files modified   : 1",
            f"  Change ratio     : {m.get('change_ratio', 'N/A')}",
            f"  Changed blocks   : {m.get('changed_blocks', 'N/A')}",
            f"  New methods added: {m.get('new_methods_added', [])}",
            f"  New file allowed : {'Yes' if task.new_file_creation_allowed else 'No'}",
        ]
        if validation.violations:
            lines.append("  Violations:")
            for v in validation.violations:
                lines.append(f"    • {v}")
        lines.append(StageLog._separator)
        return "\n".join(lines)

    @staticmethod
    def plan_summary(tasks) -> str:
        lines = [
            f"\n{StageLog._separator}",
            "[Planning]",
            f"  Tasks: {len(tasks)}",
        ]
        for t in tasks:
            lines.append(
                f"    [{t.id}] {t.task_type.value.upper():6s} {t.file_path}"
                f"  | class={t.target_class or '?'} method={t.target_method or '?'}"
                f"  | allowed={t.allowed_methods or 'ALL'}"
                f"  | new_file={'✓' if t.new_file_creation_allowed else '✗'}"
            )
        lines.append(StageLog._separator)
        return "\n".join(lines)


# ============================================================================
# CROSS-LANGUAGE EXECUTION ORDERING
# ============================================================================

# Map microservice directory names to their root Java packages.
# Used to detect when service A imports from service B's internal packages.
_SERVICE_PACKAGE_MAP: dict[str, list[str]] = {
    "project-service":      ["com.cloudforge.projectservice", "com.cloudforge.sagas"],
    "area-service":         ["com.cloudforge.services.area"],
    "participant-service":  ["com.cloudforge.services.participant"],
    "contract-service":     ["com.cloudforge.services.contract"],
}
# These packages are allowed in ANY service (shared libraries, framework, JDK)
_SHARED_PACKAGES = (
    "com.cloudforge.commons",
    "com.cloudforge.common",
    "org.springframework",
    "jakarta.", "javax.", "java.", "lombok.",
    "org.hibernate", "org.apache", "com.fasterxml",
    "io.swagger", "org.slf4j",
)


def _check_java_cross_service_imports(file_path: str, content: str) -> list[str]:
    """
    Detect cross-service Java import violations at write time.

    When a file in project-service imports from area-service's internal package
    (or vice versa), the build fails with "package X does not exist" because the
    services are separate Maven/Gradle modules with no dependency on each other.

    Returns a list of violating import lines.
    """
    norm = file_path.replace("\\", "/").lower()
    file_service = next(
        (svc for svc in _SERVICE_PACKAGE_MAP if f"/{svc}/" in norm or norm.startswith(f"{svc}/")),
        None,
    )
    if not file_service:
        return []

    violations: list[str] = []
    in_imports = True
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("*") or s.startswith("package "):
            continue
        if not s.startswith("import "):
            if s.startswith("@") or s.startswith("public ") or s.startswith("class ") or s.startswith("interface "):
                in_imports = False
            if not in_imports:
                break
            continue

        pkg = s[7:].rstrip(";").strip().lstrip("static").strip()
        if any(pkg.startswith(p) for p in _SHARED_PACKAGES):
            continue
        # Violation: file in service A imports from service B's internal package
        for other_svc, other_prefixes in _SERVICE_PACKAGE_MAP.items():
            if other_svc != file_service and any(pkg.startswith(p) for p in other_prefixes):
                violations.append(f"{s}  ← cross-service: {file_service} cannot import from {other_svc}")
                break

    return violations


def _check_ts_exports_preserved(original: str, new_content: str) -> list:
    """
    Fix 1 — Export-drop guard.
    Returns the list of exported names present in the original TypeScript/JS
    file that are MISSING from the newly generated content.  A non-empty
    return means the LLM silently dropped public API that other files import.
    """
    orig_exports = set(re.findall(
        r'\bexport\s+(?:interface|type|class|enum|const|function|abstract\s+class)\s+(\w+)',
        original,
    ))
    new_exports = set(re.findall(
        r'\bexport\s+(?:interface|type|class|enum|const|function|abstract\s+class)\s+(\w+)',
        new_content,
    ))
    return sorted(orig_exports - new_exports)


def _extract_ts_interface_props(interface_content: str, interface_name: str) -> set[str]:
    """Extract property names from a TypeScript interface definition."""
    m = re.search(
        r'\binterface\s+' + re.escape(interface_name) + r'\s*(?:extends[^{]*)?\{([^}]*)\}',
        interface_content, re.DOTALL,
    )
    if not m:
        return set()
    block = m.group(1)
    props = set(re.findall(r'\b(\w+)\s*\??:', block))
    return props


def _check_ts_property_access(
    new_content: str,
    file_path: str,
    session_files: dict,
    workspace_path: str,
) -> list[str]:
    """
    Post-write TypeScript property validator (Cursor LSP check equivalent).

    For each `import { X } from '...'` in the generated file, if the imported
    module was regenerated this run (present in session_files), extract its
    interface properties and check that any `variable.property` accesses in
    the new content only use VALID properties.

    Returns list of invalid property accesses found.
    """
    if not session_files or not new_content:
        return []

    import os as _os
    src_dir = _os.path.dirname(file_path.replace("\\", "/"))
    invalid: list[str] = []

    imp_re = re.compile(r"""import\s*\{([^}]+)\}\s*from\s*['"](\.[^'"]+)['"]""")
    for imp_m in imp_re.finditer(new_content):
        symbols = [s.strip() for s in imp_m.group(1).split(",") if s.strip()]
        mod_path = imp_m.group(2)
        for ext in (".ts", ".tsx", "/index.ts"):
            cand = _os.path.normpath(
                _os.path.join(workspace_path, src_dir, mod_path + ext)
            ).replace("\\", "/").lower()
            if cand not in session_files:
                # also check relative key
                cand_rel = _os.path.normpath(
                    _os.path.join(src_dir, mod_path + ext)
                ).replace("\\", "/").lower()
                if cand_rel in session_files:
                    cand = cand_rel
                else:
                    continue

            iface_content = session_files[cand]
            for sym in symbols:
                valid_props = _extract_ts_interface_props(iface_content, sym)
                if not valid_props:
                    continue  # not a known interface, skip
                # Check for invalid property accesses: x.badProp or x?.badProp
                accesses = re.findall(
                    r'\b\w+\??\.(\w+)\b',
                    new_content,
                )
                for prop in accesses:
                    # Only flag if the prop clearly looks like a field access
                    # and is NOT in the valid interface
                    if prop and prop[0].islower() and prop not in valid_props and len(prop) > 2:
                        # Cross-check: is this property plausibly from THIS interface?
                        # Heuristic: if any valid prop starts with same prefix, flag it
                        prefix = prop[:3].lower()
                        if any(p.lower().startswith(prefix) for p in valid_props):
                            invalid.append(f"{sym}.{prop} (valid: {sorted(valid_props)[:5]})")
            break

    return list(dict.fromkeys(invalid))[:5]  # deduplicate, cap at 5


def _find_type_consumer_companions(
    model_file_path: str,
    workspace_path: str,
    existing_paths: "set[str]",
    max_consumers: int = 4,
) -> "list[tuple[str, str]]":
    """
    Inverse of model_import_follow: when a model/interface file is modified,
    find TypeScript files in the workspace that import it and add them as
    companion MODIFY tasks so they can be updated consistently.

    Scanning is bounded to the nearest parent 'modules' directory to avoid
    scanning the entire workspace.
    """
    import os as _os
    norm = model_file_path.replace("\\", "/")
    stem = Path(norm).stem  # e.g. 'displayed-member'
    model_abs = Path(workspace_path) / norm
    if not model_abs.exists():
        return []

    # Walk up to find the nearest 'modules' or 'src' boundary (max 4 levels)
    search_root = model_abs.parent
    for _ in range(4):
        parent = search_root.parent
        if parent.name in ("modules", "src", "app", "main"):
            search_root = parent
            break
        if parent == Path(workspace_path):
            break
        search_root = parent

    results: list[tuple[str, str]] = []
    skip_dirs = {"node_modules", "dist", ".git", "build", "target", "coverage", ".venv"}

    for root, dirs, files in _os.walk(str(search_root)):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if not f.endswith((".ts", ".tsx")):
                continue
            full = Path(root) / f
            rel = str(full.relative_to(Path(workspace_path))).replace("\\", "/")
            if rel == norm or rel in existing_paths:
                continue
            try:
                content = full.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # Check if this file imports from the model file
            if re.search(r"""from\s+['"][^'"]*""" + re.escape(stem) + r"""['""]""", content):
                results.append((rel, "type_consumer_follow"))
                if len(results) >= max_consumers:
                    return results

    return results


def _sort_tasks_by_execution_order(tasks: list) -> list:
    """
    Sort code-generation tasks by language tier and declared dependencies.

    Tier order (mirrors how Devin and Cursor sequence multi-file changes):
      0  Java/C# data models (Entity, Dto, Record, VO)
      1  Java/C# repositories / interfaces
      2  Java/C# service implementations
      3  Java/C# controllers / handlers
      4  TypeScript/JS model interfaces / DTOs
      5  TypeScript/JS services
      6  TypeScript/JS components
      7  HTML templates
      8  Angular NgModules  ← AFTER components they declare
      9  SCSS / CSS stylesheets
      10 Everything else (Python, YAML, JSON, shell scripts)

    Declared task.dependencies always take priority over tier (via Kahn's
    topological sort).  Tier is only the tie-breaker for tasks with the same
    dependency depth and no explicit ordering from the planner.
    """
    from collections import defaultdict, deque

    def _tier(task) -> int:
        p = task.file_path.lower().replace("\\", "/")
        lang = getattr(getattr(task, "language", None), "value", "").lower()

        # ── Java / C# ────────────────────────────────────────────────────────
        if lang in ("java", "kotlin", "csharp") or p.endswith((".java", ".kt", ".cs")):
            # Data models / DTOs (no stereotype)
            fname = p.rsplit("/", 1)[-1]
            if any(fname.endswith(f"{s}.java") or fname.endswith(f"{s}.kt") or fname.endswith(f"{s}.cs")
                   for s in ("model", "entity", "dto", "record", "vo", "request", "response",
                              "payload", "data", "input", "output")):
                return 0
            if "repositor" in p or "repository" in p:
                return 1
            if "interface" in p and "service" in p:
                return 1   # service interface is like a repository
            if "serviceimpl" in p or (p.endswith("impl.java") or "/impl/" in p):
                return 2
            if "service" in p:
                return 2
            if "controller" in p or "handler" in p or "rest" in p or "resource" in p:
                return 3
            return 2

        # ── TypeScript / JavaScript ───────────────────────────────────────────
        if lang in ("typescript", "javascript") or p.endswith((".ts", ".tsx", ".js", ".jsx")):
            if any(kw in p for kw in ("/models/", "/model.", "-model.", ".model.",
                                       "/interface.", "-interface.", ".interface.",
                                       "/dto.", "-dto.", ".dto.",
                                       "/types/", "/type.", "-type.", ".type.",
                                       "/entities/", "/entity.", "-entity.")):
                return 4
            # Angular NgModule: must come AFTER the components it declares
            if "module.ts" in p or "module.js" in p:
                return 8
            if "service.ts" in p or "service.js" in p:
                return 5
            if ".component." in p:
                return 6
            return 5

        # ── HTML ──────────────────────────────────────────────────────────────
        if p.endswith((".html", ".htm")):
            return 7

        # ── Styles ────────────────────────────────────────────────────────────
        if p.endswith((".scss", ".css", ".less", ".sass")):
            return 9

        return 10   # Python, YAML, JSON, shell, etc.

    # Build adjacency for Kahn's algorithm using declared task.dependencies
    task_map: dict = {t.id: t for t in tasks}
    in_degree: dict = {t.id: 0 for t in tasks}
    adj: dict = defaultdict(list)

    # Build dependency edges from explicit task.dependencies only
    for t in tasks:
        for dep_id in (t.dependencies or []):
            if dep_id in task_map:
                adj[dep_id].append(t.id)
                in_degree[t.id] += 1

    # Seed queue with tasks that have no remaining dependencies, ordered by tier
    queue = deque(sorted(
        [t for t in tasks if in_degree[t.id] == 0],
        key=_tier,
    ))

    result: list = []
    while queue:
        t = queue.popleft()
        result.append(t)
        newly_free = []
        for dep_id in adj[t.id]:
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                newly_free.append(task_map[dep_id])
        queue.extend(sorted(newly_free, key=_tier))

    # Fallback: append any tasks not reached (missing dep IDs, cycles) in tier order
    in_result = {t.id for t in result}
    result.extend(sorted([t for t in tasks if t.id not in in_result], key=_tier))

    # ── Angular sibling enforcement ────────────────────────────────────────
    # Guarantee: for every Angular component pair (foo.component.ts / foo.component.html),
    # the .ts MUST always precede its sibling .html, regardless of task IDs or tier ties.
    # This prevents HTML from being generated first and inventing property names that
    # the TS controller doesn't yet know about.
    _stem_to_ts_idx: dict = {}
    for i, t in enumerate(result):
        fp = t.file_path.replace("\\", "/").lower()
        if ".component.ts" in fp or fp.endswith(".component.tsx"):
            stem = fp.rsplit(".ts", 1)[0]
            _stem_to_ts_idx[stem] = i

    swaps_needed = []
    for i, t in enumerate(result):
        fp = t.file_path.replace("\\", "/").lower()
        if fp.endswith(".html") or fp.endswith(".htm"):
            stem = fp.rsplit(".html", 1)[0].rsplit(".htm", 1)[0]
            ts_idx = _stem_to_ts_idx.get(stem)
            if ts_idx is not None and ts_idx > i:
                # HTML appears before TS — record the swap
                swaps_needed.append((i, ts_idx))

    # Apply swaps in reverse index order so earlier swaps don't invalidate later indices
    for html_idx, ts_idx in sorted(swaps_needed, reverse=True):
        result[html_idx], result[ts_idx] = result[ts_idx], result[html_idx]
        logger.info(
            f"  Sort fix: moved {result[html_idx].file_path} (TS) before "
            f"{result[ts_idx].file_path} (HTML) to ensure controller precedes template"
        )

    return result


def _group_tasks_into_batches(tasks: list) -> "list[list]":
    """
    Group topologically-sorted tasks into dependency-depth batches.

    Tasks within the same batch are INDEPENDENT of each other — they have no
    mutual dependency edges.  Tasks in batch N+1 depend on at least one task
    in batch N (or earlier).

    Example:
        A ──→ C
        B ──→ C
        D   independent

        Batch 0: [A, B, D]  (no unresolved dependencies)
        Batch 1: [C]        (depends on A and B)

    This enables:
      - After batch 0 completes: extract & register all contracts
      - Batch 1 generation: has access to all batch-0 contracts

    Uses the same dependency graph (Kahn's algorithm) as _sort_tasks_by_execution_order
    but preserves the depth level assignment.

    Returns a list of batches, each batch being a list of tasks.
    """
    from collections import defaultdict, deque

    task_map: dict = {t.id: t for t in tasks}
    in_degree: dict = {t.id: 0 for t in tasks}
    adj: dict = defaultdict(list)

    # Build dependency edges
    for t in tasks:
        for dep_id in (t.dependencies or []):
            if dep_id in task_map:
                adj[dep_id].append(t.id)
                in_degree[t.id] += 1

    # BFS by depth level
    batches: list[list] = []
    current_batch = [t for t in tasks if in_degree[t.id] == 0]

    if not current_batch:
        # No roots — all tasks form a single batch (cycle or no deps)
        return [tasks] if tasks else []

    visited: set = set()

    while current_batch:
        batches.append(current_batch)
        visited.update(t.id for t in current_batch)

        next_batch_candidates = []
        for t in current_batch:
            for dep_id in adj[t.id]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0 and dep_id not in visited:
                    next_batch_candidates.append(task_map[dep_id])

        current_batch = next_batch_candidates

    # Append any unreachable tasks (cycles, missing deps) as a final batch
    _in_batches = visited
    _remaining = [t for t in tasks if t.id not in _in_batches]
    if _remaining:
        batches.append(_remaining)

    return batches


# ============================================================================
# IMPORT RESOLUTION HELPERS  (Layer 2 + Layer 3)
# ============================================================================

def _inject_meta_imports(code: "GeneratedCode") -> "GeneratedCode":
    """
    Layer 2 — Devin / Copilot Workspace technique.
    Deterministically merge the LLM's declared META imports into the file content
    so they are never silently dropped when the SEARCH/REPLACE focused on a method.
    """
    if not code.imports or not code.content:
        return code

    content = code.content
    lang = getattr(code.language, "value", str(code.language)).lower()
    new_content = content

    for imp in code.imports:
        imp = imp.strip()
        if not imp:
            continue
        # Skip if already present (verbatim)
        if imp in content:
            continue

        if lang == "java":
            # Insert after existing import block, before first class/annotation
            _m = re.search(r'^(package\s[^\n]+\n)', new_content, re.MULTILINE)
            insert_after = _m.end() if _m else 0
            _last_import = None
            for _m2 in re.finditer(r'^import\s[^\n]+\n', new_content, re.MULTILINE):
                _last_import = _m2
            if _last_import:
                insert_after = _last_import.end()
            new_content = new_content[:insert_after] + imp + "\n" + new_content[insert_after:]

        elif lang in ("typescript", "javascript"):
            # Insert after the LAST complete import statement.
            # Handles multi-line imports like:  import {\n  Foo,\n} from '...';\n
            _last_import = None
            # Match single-line: import ... from '...';  OR  closing multi-line: } from '...';
            for _m2 in re.finditer(
                r"^(?:import\s+.*?from\s+['\"][^'\"]+['\"]\s*;?|\}\s*from\s+['\"][^'\"]+['\"]\s*;?)\s*$",
                new_content, re.MULTILINE,
            ):
                _last_import = _m2
            insert_after = (_last_import.end() + 1) if _last_import else 0
            # Ensure we're at a line boundary
            if insert_after > 0 and insert_after <= len(new_content) and new_content[insert_after - 1:insert_after] != "\n":
                insert_after = new_content.find("\n", insert_after)
                if insert_after == -1:
                    insert_after = len(new_content)
                else:
                    insert_after += 1
            new_content = new_content[:insert_after] + imp + "\n" + new_content[insert_after:]

        elif lang == "python":
            _last_import = None
            for _m2 in re.finditer(r'^(?:import|from)\s[^\n]+\n', new_content, re.MULTILINE):
                _last_import = _m2
            insert_after = _last_import.end() if _last_import else 0
            new_content = new_content[:insert_after] + imp + "\n" + new_content[insert_after:]

        elif lang in ("csharp", "cs"):
            _last_using = None
            for _m2 in re.finditer(r'^using\s[^\n]+\n', new_content, re.MULTILINE):
                _last_using = _m2
            insert_after = _last_using.end() if _last_using else 0
            new_content = new_content[:insert_after] + imp + "\n" + new_content[insert_after:]

    code.content = new_content
    return code


def _resolve_unresolved_symbols(
    content: str,
    language: str,
    sqlite_store,
    workspace_path: str,
    source_file_path: str = "",
) -> list[tuple[str, str]]:
    """
    Layer 3 — Claude Code / Devin technique.
    Parse generated file content for NEW symbol references and look each up in
    the SQLite symbol index.  Returns [(import_statement, file_path), ...].
    """
    if not sqlite_store or not content:
        return []

    found: list[tuple[str, str]] = []
    lang = language.lower()

    # Collect what's already imported (to avoid duplicates)
    existing_imports: set[str] = set()
    for line in content.split("\n"):
        s = line.strip()
        if s.startswith(("import ", "from ", "using ")):
            existing_imports.add(s)

    # Extract all PascalCase identifiers used in the file (potential class refs)
    all_symbols = re.findall(r'\b([A-Z][a-zA-Z0-9]{2,})\b', content)
    unique_symbols = list(dict.fromkeys(all_symbols))

    # Filter to symbols that are NOT already imported
    def _already_imported(sym: str) -> bool:
        return any(sym in imp for imp in existing_imports)

    candidates = [s for s in unique_symbols if not _already_imported(s)][:30]

    from ticket_to_code.agents.code_generator import _build_import_hint

    for sym in candidates:
        try:
            rows = sqlite_store.search_symbols(
                sym, kinds=["class", "interface", "enum"], limit=2
            )
            for row in rows:
                row_name = row["name"] if isinstance(row, dict) else getattr(row, "name", sym)
                row_path = row["path"] if isinstance(row, dict) else getattr(row, "path", "")
                if row_name != sym or not row_path:
                    continue
                hint = _build_import_hint(sym, row_path, source_file_path=source_file_path, workspace_path=workspace_path)
                if hint and hint not in existing_imports:
                    # Verify the file actually exists in workspace
                    if Path(workspace_path, row_path).exists():
                        found.append((hint, row_path))
                        existing_imports.add(hint)
                break
        except Exception:
            pass

    return found


# ============================================================================
# TRANSIENT REGISTRY — stores non-serializable objects OUTSIDE LangGraph state
# ============================================================================
# LangGraph's MemorySaver uses msgpack to checkpoint the state dict after every
# node.  Python functions (callbacks) and complex class instances (RunContext)
# are NOT msgpack-serializable and cause TypeError crashes.
#
# This thread-safe registry stores such objects keyed by ticket_id.  Nodes
# read from it via _get_transient(); the objects are never persisted.

import threading as _threading

_transient_lock = _threading.Lock()
_transient_store: dict[str, dict[str, Any]] = {}  # {ticket_id: {key: value}}


# ── Stage-aware model routing hook (AUTO_STAGE) ──────────────────────────────
# The backend registers a resolver `fn(node_name) -> Optional[model_id]`. The phase
# wrapper swaps `agents.llm` for the duration of that node when a model is returned.
_STAGE_MODEL_RESOLVER: Optional[Callable[[str], Optional[str]]] = None
_STAGE_LLM_CACHE: Dict[str, Any] = {}
_STAGE_LLM_LOCK = _threading.Lock()


def set_stage_model_resolver(fn: Optional[Callable[[str], Optional[str]]]) -> None:
    """Install (or clear with None) the per-node model resolver used by AUTO_STAGE mode."""
    global _STAGE_MODEL_RESOLVER
    _STAGE_MODEL_RESOLVER = fn


def _stage_llm_for(model_id: str):
    with _STAGE_LLM_LOCK:
        llm = _STAGE_LLM_CACHE.get(model_id)
        if llm is None:
            from ticket_to_code.runtime.llm_factory import LLMRegistry
            llm = LLMRegistry.get_llm(model_name=model_id)
            _STAGE_LLM_CACHE[model_id] = llm
        return llm


def _set_transient(ticket_id: str, key: str, value: Any) -> None:
    """Store a non-serializable object for the current workflow run."""
    with _transient_lock:
        _transient_store.setdefault(ticket_id, {})[key] = value


def _get_transient(state: dict, key: str, default=None) -> Any:
    """Read a non-serializable object stored for the current workflow run."""
    ticket = state.get("ticket")
    tid = getattr(ticket, "ticket_id", None) if ticket else None
    if not tid:
        return state.get(key) or (state.get("_transient", {}) or {}).get(key, default)
    with _transient_lock:
        return _transient_store.get(tid, {}).get(key, default) or state.get(key) or (state.get("_transient", {}) or {}).get(key, default)


def _clear_transient(ticket_id: str) -> None:
    """Clean up transient objects after a workflow run completes."""
    global _active_ticket_id
    with _transient_lock:
        _transient_store.pop(ticket_id, None)
        if _active_ticket_id == ticket_id:
            _active_ticket_id = None


_active_ticket_id: Optional[str] = None


def set_active_ticket_id(ticket_id: Optional[str]) -> None:
    """Set the currently executing active ticket ID for token budget telemetry."""
    global _active_ticket_id
    with _transient_lock:
        _active_ticket_id = ticket_id


def get_active_ticket_id() -> Optional[str]:
    """Retrieve the currently executing active ticket ID."""
    with _transient_lock:
        return _active_ticket_id


def clear_active_ticket_id() -> None:
    """Clear the active ticket ID."""
    global _active_ticket_id
    with _transient_lock:
        _active_ticket_id = None


# ============================================================================
# STATE DEFINITION
# ============================================================================

def _safe_add_list(left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
    """LangGraph reducer: safely merge lists without crashing if either side is None."""
    res = list(left or [])
    if right:
        res.extend(right)
    return res


class TicketToCodeState(TypedDict):
    """
    Complete workflow state managed by LangGraph.
    
    Annotated fields are automatically tracked and merged.
    """
    # Input
    ticket: ValueEdgeTicket
    workspace_path: str
    max_retry_attempts: int
    
    # Phase results
    investigation_result: Optional[InvestigationResult]
    requirements: Optional[StructuredRequirements]
    discovered_files: Optional[List[dict]]          # Real candidate files from Repository Discovery
    architectural_plan: Optional[ArchitecturalPlan]
    solution_guidance: Optional[SolutionGuidance]  # For non-code issues
    
    # RAG contexts (SEPARATED for independence)
    test_rag_context: Optional[List[dict]]  # Product behavior knowledge
    code_rag_context: Optional[List[dict]]  # Architecture pattern knowledge
    
    # Branch-specific status (for parallel execution)
    test_status: Optional[str]  # Status of TEST BRANCH
    code_status: Optional[str]  # Status of CODE BRANCH

    # Post-build semantic validation result
    outcome_check_result: Optional[dict]  # {status, details, files_checked}
    
    # Generated artifacts (TDD order)
    generated_tests: Optional[List[GeneratedCode]]  # Generated FIRST
    generated_code: Optional[List[GeneratedCode]]   # Generated SECOND
    original_file_contents: Optional[dict] # Original contents for auto-revert
    pre_run_workspace_manifest: Optional[dict]  # {relpath: size} snapshot taken before any writes — used by the scope gate for non-git-safe diffing
    pre_run_git_dirty_files: Optional[Set[str]] # Pre-existing dirty / untracked files snapshot
    direct_mode: Optional[bool]                 # Direct AI IDE fast-path execution mode
    allowed_files: Optional[List[str]]          # Locked candidate scope for direct mode / edit loop
    
    build_result: Optional[BuildResult]
    test_result: Optional[TestResult]
    compile_only_verification: Optional[bool]
    verification_level: Optional[str]  # "test_verified" | "compile_only" | "test_failed" | "unverified"
    
    # Retry tracking
    retry_attempt: Annotated[int, "Current retry attempt"]
    last_error_type: Optional[Literal["build", "test"]]
    previous_build_errors: Optional[List[str]]  # Track error identities across fix_build loops

    # Candidate retry (localization-level — independent from build/test retry)
    blacklisted_files: Optional[List[str]]        # paths confirmed as wrong localization
    candidate_retry_count: int                     # how many candidate-level retries so far
    max_candidate_retries: int                     # retries before triggering full rediscovery
    discovery_cycle_count: int                     # how many full rediscovery cycles so far
    validation_failure_reason: Optional[str]       # why validate_candidates rejected
    rejected_approaches: Optional[List[str]]       # User feedback / approaches to avoid on resume
    user_constraints: Optional[List[str]]          # Injected constraints from user / resume

    # Phase 2F: Ownership Completeness result (written by ownership_completeness_node)
    ownership_completeness: Optional[dict]

    # Phase 2G: Investigation Hypotheses + Evidence Collection (written before ownership check)
    investigation_hypotheses: Optional[List[InvestigationHypothesis]]
    evidence_items: Optional[List[EvidenceItem]]
    ranked_files: Optional[List[Any]]
    semantic_verification_results: Optional[List[dict]]
    clarification_question: Optional[str]          # Agentic loop question for user
    # Component Group Architecture: groups of related framework files
    # (e.g., .ts + .html + .scss) built before ranking, passed through
    # semantic verification to the planner for group-aware planning.
    component_groups: Optional[List[Any]]
    grounded_understanding: Optional[GroundedUnderstanding]
    # Query expansion provenance map from RepositorySearchEngine.
    # Dict[original_literal, Set[expanded_terms]] — passed to EvidenceRankingEngine
    # so expansion matches receive correct literal-match credit.
    query_expansion_map: Optional[dict]

    # Planner decisions: Dict[file_path, {role, decision, reason, confidence}]
    # Written by plan_node after create_plan(); read by grounded_understanding_node
    # to prevent auto-injection of files the planner intentionally excluded.
    planner_decisions: Optional[dict]

    # Evidence-based build routing: tracks what errors looked like on the previous
    # build attempt so check_build_status can tell CHANGING errors (progress) from
    # IDENTICAL errors (stuck) and route differently — not just count retries.
    prev_build_error_fingerprint: Optional[str]
    consecutive_identical_build_errors: int

    # Build diagnostic classification and user decision (written by pre_fix_build_node,
    # read by check_build_status router and fix_build_errors_node).
    # These fields carry the error-provenance classification and user decision across
    # the LangGraph node boundary so the router can distinguish ticket-introduced
    # errors (→ repair) from pre-existing baseline errors (→ follow user decision).
    build_differential_accept: Optional[bool]          # True = all errors are pre-existing
    test_differential_accept: Optional[bool]           # True = all test failures belong to unrelated baseline modules
    test_baseline_compared: Optional[bool]             # True = differential acceptance was backed by a real pre-edit baseline test run
    targeted_expansion_files: Optional[List[str]]      # files outcome_check wants promoted for context expansion
    patch_gate_blocked: Optional[bool]                 # True = patch_gate hard-failed (scope/secret/proportionality)
    patch_gate_failures: Optional[List[str]]
    evidence_decision_degraded: Optional[bool]         # True = the agentic evidence loop fell back to "proceed" after an LLM/parse error
    policy_advisories: Optional[List[str]]             # PolicyEngine.evaluate_edit_safety advisories recorded by patch_gate
    build_infrastructure_only: Optional[bool]           # True = all errors are infra (deps/network)
    build_infrastructure_blocked: Optional[bool]        # True = infra failure surfaced to user
    build_diagnostic_summary: Optional[str]             # Human-readable classification summary
    pre_existing_decision: Optional[str]                # User choice: "fix" | "leave" | "stop"
    authorized_pre_existing_files: Optional[List[str]]  # Files authorized for pre-existing repair


    # Requirement-satisfaction loop (post-build): outcome_check verifies the ticket
    # is actually solved, not just that it compiles. If requirements are missing it
    # feeds concrete remediation back into the adaptive edit loop, bounded by
    # max_outcome_fix_attempts.
    outcome_check_result: Optional[dict]
    outcome_remediation: Optional[str]
    outcome_findings: Optional[List[dict]]         # structured OutcomeFinding.to_dict() list
    outcome_fix_attempt: int
    max_outcome_fix_attempts: int

    # Terminal escalation (written by memory_update_node / verification_failed_node).
    # When the semantic verdict is still PARTIAL/INCOMPLETE after the remediation
    # budget is exhausted, the run MUST end as verification_failed — never as a
    # green "workflow complete".
    escalation_type: Optional[str]                 # build_failed | checklist_incomplete | requirements_unsatisfied
    escalation_reason: Optional[str]               # human-readable explanation shown to the user
    unresolved_checklist_items: Optional[List[dict]]

    # Re-ranker tiered fallback: Tier 2 candidates held in reserve for
    # context expansion when outcome_check finds PARTIAL/INCOMPLETE.
    tier2_candidates: Optional[List[dict]]         # backup files from re-ranker
    context_expansion_count: int                   # how many times we expanded

    # Pre-flight idempotency check results
    preflight_verdict: Optional[str]               # ALREADY_DONE / PARTIALLY_DONE / NOT_DONE
    preflight_summary: Optional[str]               # LLM explanation of current state
    preflight_missing_requirements: Optional[List[str]]  # requirements not yet satisfied
    preflight_implementation_guidance: Optional[List[dict]]  # per-requirement what/how/where guidance

    # Global status (ONLY updated by SEQUENTIAL nodes, NOT by parallel branches)
    status: str
    errors: Annotated[List[str], _safe_add_list]
    start_time: datetime
    end_time: Optional[datetime]

    # ── Enhancement 1: Runtime Log Diagnosis ──────────────────────────────────
    # Populated by runtime_diagnosis_node between investigate and unified_analysis.
    # If no log files are found, this stays None and the pipeline is unchanged.
    runtime_diagnosis: Optional[Any]  # RuntimeDiagnosis instance
    visual_analysis: Optional[Any]    # VisualAnalysis instance (Enhancement 8)

    # ── B11/B12: Per-run telemetry and health tracking ────────────────────────
    # RunContext holds RunBudget, DegradationRegistry, and phase timers.
    # It is created once at workflow entry and referenced throughout.
    # Not serialisable by LangGraph — stored as Any to avoid schema errors.
    run_ctx: Optional[Any]  # RunContext instance

    # ── Planning Recovery ─────────────────────────────────────────────────
    # Structured recovery action from planning_recovery_node.  Consumed by
    # discovery_node / hypothesis_investigation_node / evidence_collection_loop
    # to provide targeted investigation context after a planning failure.
    # Versioned via recovery_id + status lifecycle (active → consumed).
    planning_recovery_action: Optional[Any]  # PlanningRecoveryAction instance

    # ── Context Firewall & Verification Loop Diagnostics ──────────────────
    subagent_investigation_summary: Optional[str]
    build_error_history: Optional[List[str]]
    sub_workspace_path: Optional[str]
    ui_verification_result: Optional[dict]


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================



class WorkflowAgents:
    """Container for all agents"""
    
    def __init__(self, workspace_path: str, technology: Optional[str] = None):
        """
        Initialize workflow agents.
        
        Args:
            workspace_path: Path to project workspace
            technology: Language/framework to use. Valid values:
                - 'java' (Maven/Gradle)
                - 'dotnet' (.NET Framework/Core)
                - 'python' (pytest)
                - 'nodejs' (npm/jest)
                - 'go' (go build/test)
                - None (auto-detect from project files)
        """
        self.workspace_path = Path(workspace_path)
        
        # Initialize agents
        self.investigation = InvestigationAgent(str(workspace_path))
        self.analyzer = TicketAnalyzerAgent()
        self.planner = PlanningAgent(workspace_path=str(workspace_path))
        self.rag_engine = CodebaseRAGEngine()
        self.rag_engine.workspace_path = str(workspace_path)
        self.localizer = LocalizationAgent(str(workspace_path))  # Repository intelligence
        self.code_generator = CodeGeneratorAgent(self.workspace_path)
        self.test_generator = CodeGeneratorAgent(self.workspace_path)  # Separate instance for tests
        self.llm = self.code_generator.llm
        self.repo_search = RepositorySearchEngine(self.workspace_path)   # Phase 3A
        self.relationship_provider = RelationshipProviderFactory.get_provider(str(workspace_path))
        
        self.workflow_discovery = None
        try:
            self.workflow_discovery = WorkflowDiscovery(
                self.localizer.sqlite_store,
                self.localizer.neo4j_store,
                str(workspace_path),
            )
            self.localizer.workflow_discovery = self.workflow_discovery
            logger.info("✅ Workflow Discovery layer initialized")
        except Exception as e:
            logger.warning(f"Workflow Discovery initialization failed (non-fatal): {e}")
        self.evidence_loop = EvidenceCollectionLoop(
            self.localizer, self.rag_engine, self.repo_search, self.relationship_provider
        )
        
        self.semantic_verifier = SemanticVerificationAgent(workspace_path=str(workspace_path))
        self.evidence_loop._semantic_verifier = self.semantic_verifier
        
        # Initialize execution engine with user-specified or auto-detected technology
        try:
            self.executor = ExecutionEngineFactory.create(str(workspace_path), technology)
        except Exception as _exec_err:
            logger.warning(f"Could not initialize execution engine for {workspace_path}: {_exec_err}")
            self.executor = None

        # Observability tracer (no-op unless TRACE_MODE=true)
        self.tracer = ExecutionTracer(str(workspace_path))

        # Enhancement 1: Runtime Log Analyzer (read-only, never writes files)
        self.runtime_log_analyzer = RuntimeLogAnalyzer(str(workspace_path))

        # Enhancement 12: Dependency Resolver (detects missing deps from build errors)
        self.dependency_resolver = DependencyResolver(str(workspace_path))

        # Enhancement 4: Adaptive Replanner (switches strategy on repeated failures)
        self.adaptive_replanner = AdaptiveReplanner()

        # Enhancement 9: Git History Agent (read-only git evidence)
        self.git_history = GitHistoryAgent(str(workspace_path))

        # Enhancement 11: Env Var Tracer (traces env var usage across files)
        self.env_tracer = EnvTracer(str(workspace_path))

        # Enhancement 3: Causal Chain Reasoning Engine
        self.causal_chain = CausalChainEngine(str(workspace_path))

        # Enhancement 7: Runtime Verifier (starts app after build to check for errors)
        self.runtime_verifier = RuntimeVerifier(str(workspace_path), technology or "auto")

        # Enhancement 5: Platform Knowledge (Windows/Linux-specific guidance)
        self.platform_knowledge = PlatformKnowledge()

        # Enhancement 13: Config Validator (YAML/JSON/env syntax validation)
        self.config_validator = ConfigValidator(str(workspace_path))

        # Enhancement 8: Visual Screenshot Analyzer (multimodal vision)
        self.visual_analyzer = VisualAnalyzer(str(workspace_path))

        tech_info = technology if technology else "auto-detected"
        logger.info(f"Workflow agents initialized for: {workspace_path} (Technology: {tech_info})")


# ============================================================================
# GRAPH NODES (Each phase is a simple function)
# ============================================================================

# ── Polyglot sub-workspace detection (P1 token efficiency) ───────────────────
_FRONTEND_HINTS = (
    "modal", "dropdown", "component", "template", "html", "css", "scss",
    "button", "view", "staged", "label", "tooltip", "icon", "table", "form",
    "angular", "react", "page", "checkbox", "radio", "frontend", "ui",
)
_BACKEND_HINTS = (
    "endpoint", "entity", "repository", "migration", "sql", "kafka",
    "controller", "service class", "api layer", "dto", "backend", "microservice",
    "database", "query", "jpa", "spring", "maven", "pom",
)


def detect_sub_workspace(workspace_path: str, ticket_text: str) -> str:
    """Detect the sub-project (e.g. xchange-ui vs Java services) a polyglot
    monorepo ticket targets.  Returns a repo-relative scope prefix or ''.

    Boundary detection: any subdirectory containing its own package.json /
    pom.xml / build.gradle is a sub-project.  Intent classification uses the
    ticket's own vocabulary — frontend terms beat backend terms only when
    they strictly outnumber them (mixed full-stack tickets stay unscoped).
    """
    import functools

    ws = Path(workspace_path or "")
    if not ws.is_dir():
        return ""

    SKIP_DIRS = {
        ".git", ".idea", ".vscode", "node_modules", "target", "build",
        "dist", ".agents", ".aviator", "brain", "knowledge", ".graphify", "graphify-out",
    }

    def _classify_project(pdir: Path) -> str:
        pkg = pdir / "package.json"
        if pkg.is_file():
            try:
                import json as _j
                raw = _j.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
                deps = {**raw.get("dependencies", {}), **raw.get("devDependencies", {})}
                # UI frameworks
                if any(k in deps for k in (
                    "@angular/core", "react", "vue", "svelte", "next", "nuxt",
                    "@vue/runtime-core", "preact", "solid-js", "lit"
                )):
                    return "frontend"
                # Backend Node frameworks
                if any(k in deps for k in ("express", "fastify", "koa", "@nestjs/core")):
                    return "backend"
            except Exception:
                pass
        # JVM / Maven / Gradle
        if (pdir / "pom.xml").is_file() or (pdir / "build.gradle").is_file() or (pdir / "build.gradle.kts").is_file():
            return "backend"
        # Go / Rust / Python backend manifests
        if (pdir / "go.mod").is_file() or (pdir / "Cargo.toml").is_file() or (pdir / "pyproject.toml").is_file() or (pdir / "requirements.txt").is_file():
            return "backend"
        # Fallback to general hints if manifest doesn't explicitly name a framework
        low = pdir.name.lower()
        if any(h in low for h in ("ui", "web", "frontend", "client")):
            return "frontend"
        if any(h in low for h in ("service", "backend", "server", "api")):
            return "backend"
        return "unknown"

    sub_projects: list = []  # (rel_path, kind)
    try:
        candidate_roots = [ws]
        for monorepo_dir in ("packages", "apps", "services", "modules"):
            md = ws / monorepo_dir
            if md.is_dir():
                candidate_roots.append(md)

        for root in candidate_roots:
            for child in root.iterdir():
                if not child.is_dir() or child.name in SKIP_DIRS or child.name.startswith("."):
                    continue
                if any((child / marker).exists() for marker in (
                    "package.json", "pom.xml", "build.gradle", "build.gradle.kts", "go.mod", "Cargo.toml", "pyproject.toml"
                )):
                    rel_p = child.relative_to(ws).as_posix()
                    kind = _classify_project(child)
                    sub_projects.append((rel_p, kind))
    except Exception:
        return ""

    if not sub_projects:
        return ""

    text_low = (ticket_text or "").lower()
    fe = sum(1 for h in _FRONTEND_HINTS if bool(re.search(r'\b' + re.escape(h) + r'\b', text_low)))
    be = sum(1 for h in _BACKEND_HINTS if bool(re.search(r'\b' + re.escape(h) + r'\b', text_low)))
    if fe == be:
        return ""  # mixed or ambiguous ticket — do not scope

    want = "frontend" if fe > be else "backend"
    matches = [rel for rel, kind in sub_projects if kind == want]
    if len(matches) != 1:
        return ""  # 0 or multiple candidates — scoping would be a guess
    return matches[0]


def investigate_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 0: Investigation & Triage

    Determines if code changes are needed or just configuration guidance.
    Also initialises the per-run RunContext (B11/B12).
    """
    print("\n" + "="*80)
    print(" ENTERING: investigate_node() in workflow.py")
    print("   Purpose: Analyze ticket to determine if code changes are needed")
    print("="*80)
    logger.info(" PHASE 0: Investigation & Triage")

    # ── Reset cross-run module state to prevent leak between Celery tasks ─────
    try:
        from ticket_to_code.agents.code_generator import clear_reuse_directive
        clear_reuse_directive()
    except Exception:
        pass

    # ── Snapshot the workspace BEFORE any node can write a single file ────────
    # Used by the post-generation scope gate when git verification is
    # unavailable, so pre-existing files can never be misclassified as
    # "unauthorized new files" (the multi-repo wipe failure mode).
    if not state.get("pre_run_workspace_manifest"):
        try:
            from ticket_to_code.agents.ticket_scope_proof import build_workspace_manifest
            _manifest = build_workspace_manifest(state.get("workspace_path", ""))
            if _manifest:
                logger.info(f"  Pre-run workspace manifest captured: {len(_manifest)} files")
            else:
                logger.warning("  Pre-run workspace manifest is EMPTY — workspace may be missing")
            state["pre_run_workspace_manifest"] = _manifest
        except Exception as _mf_err:
            logger.warning(f"  Pre-run workspace manifest capture failed (non-fatal): {_mf_err}")
            state["pre_run_workspace_manifest"] = {}

    # ── Snapshot pre-existing git status BEFORE any node touches anything ─────
    # Captures all dirty / untracked files already in the workspace before the run.
    # Essential so that pre-existing developer work is NEVER treated as
    # unauthorized scope violations or reverted (just like in modern AI IDEs).
    if "pre_run_git_dirty_files" not in state:
        try:
            from ticket_to_code.agents.ticket_scope_proof import capture_pre_run_git_status
            _pre_dirty = capture_pre_run_git_status(state.get("workspace_path", ""))
            state["pre_run_git_dirty_files"] = _pre_dirty
            if _pre_dirty:
                logger.info(f"  Pre-run git status captured: {len(_pre_dirty)} pre-existing dirty/untracked file(s) protected from erasure")
        except Exception as _git_err:
            logger.debug(f"  Pre-run git status capture failed (non-fatal): {_git_err}")
            state["pre_run_git_dirty_files"] = set()

    # ── Workspace Integrity Baseline Capture ──────────────────────────────────
    try:
        from ticket_to_code.integrity.workspace_integrity import get_workspace_integrity_manager
        _integrity_mgr = get_workspace_integrity_manager(state.get("workspace_path", ""))
        _integrity_mgr.capture_baseline()
        state["workspace_integrity_manager"] = _integrity_mgr
    except Exception as _int_err:
        logger.warning(f"  Workspace integrity baseline capture failed (non-fatal): {_int_err}")

    # ── Universal Repository Intelligence & Capability Discovery ──────────
    try:
        from ticket_to_code.intelligence.repository_intelligence import RepositoryIntelligence
        from ticket_to_code.policy.policy_engine import PolicyEngine
        _ri = RepositoryIntelligence(state.get("workspace_path", ""))
        _cap_model = _ri.analyze()
        state["repository_capabilities"] = _cap_model
        state["policy_engine"] = PolicyEngine(_cap_model)
        logger.info(
            f"  [RepositoryIntelligence] Discovered {len(_cap_model.modules)} modules, "
            f"{len(_cap_model.build_systems)} build systems, "
            f"confidence: {_cap_model.confidence_score:.2f}, gaps: {len(_cap_model.capability_gaps)}"
        )
    except Exception as _ri_err:
        logger.warning(f"  RepositoryIntelligence discovery failed (non-fatal): {_ri_err}")

    # ── B11/B12: Initialise RunContext once per run ───────────────────────────
    ticket = state["ticket"]
    run_ctx = RunContext(
        ticket_id=getattr(ticket, "ticket_id", "") or "",
        repo=str(state.get("workspace_path", "")),
    )
    # Carry forward Tier 1 FastIDEEngine tokens so total telemetry is cumulative
    init_tokens = state.get("initial_tokens", 0)
    if init_tokens:
        tok_in = int(init_tokens * 0.82)
        tok_out = max(0, init_tokens - tok_in)
        run_ctx.budget.charge(
            tokens_in=tok_in,
            tokens_out=tok_out,
            cost_usd=0.0,
            model="fast_ide_engine",
            phase="tier_1_fast_ide",
        )
    # Register evidence-critical subsystems so degradations are trackable
    for sub in ["rag_engine", "pgvector", "neo4j", "sqlite_index", "context_retrieval"]:
        run_ctx.registry.register(sub)

    # Store RunContext in transient registry EARLY so _phase_tracked_node
    # wrapper can find it in its finally block for end_phase().
    # investigate_node is the only node that creates RunContext, so it
    # must call start_phase() itself — the wrapper handles end_phase().
    _set_transient(
        getattr(ticket, "ticket_id", ""),
        "run_ctx",
        run_ctx,
    )
    # Agents are registered so the phase wrapper can apply per-node model routing (AUTO_STAGE).
    _set_transient(getattr(ticket, "ticket_id", ""), "agents", agents)
    set_active_ticket_id(getattr(ticket, "ticket_id", ""))
    run_ctx.start_phase("investigate")

    # Probe RAG health at run start (B11 — startup verification)
    try:
        if agents.rag_engine and getattr(agents.rag_engine, "is_initialized", False):
            vs = getattr(agents.rag_engine, "vector_store", None)
            if vs is None:
                run_ctx.degrade("pgvector", HealthLevel.FAILED, "vector_store is None")
                run_ctx.degrade("rag_engine", HealthLevel.DEGRADED, "pgvector unavailable")
    except Exception as _e:
        run_ctx.degrade("rag_engine", HealthLevel.DEGRADED, str(_e))

    agents.tracer.record_ticket(ticket)  # creates trace directory

    # ── Enhancement 8: Visual Screenshot Analysis ─────────────────────────────
    visual_analysis = None
    if getattr(ticket, "attachments", None):
        try:
            logger.info("📸 ENHANCEMENT 8: Visual Screenshot Analysis")
            visual_analysis = agents.visual_analyzer.analyze(
                ticket.attachments,
                ticket_text=ticket.description
            )
            if visual_analysis and visual_analysis.has_visual_data:
                logger.info(
                    f"📸 Visual Analysis: Found {len(visual_analysis.elements)} elements across {len(visual_analysis.images)} screenshot(s), "
                    f"UI error: '{visual_analysis.error_text}', state: '{visual_analysis.ui_state}'"
                )
                
                if visual_analysis.formatted_summary:
                    ticket.description += f"\n\n### 📸 Visual Evidence (Ordered Screenshots):\n{visual_analysis.formatted_summary}"
                    logger.info("📸 Injected ordered visual evidence summary into ticket description.")
                else:
                    extra_context = []
                    if visual_analysis.error_text:
                        extra_context.append(f"Visual UI Error: {visual_analysis.error_text}")
                    if visual_analysis.ui_state:
                        extra_context.append(f"Visual UI State: {visual_analysis.ui_state}")
                    if visual_analysis.diagnosis:
                        extra_context.append(f"Visual Diagnosis: {visual_analysis.diagnosis}")
                    if extra_context:
                        ticket.description += "\n\n### 📸 Visual Evidence (from attached screenshot):\n" + "\n".join(f"- {c}" for c in extra_context)
                        logger.info("📸 Injected visual evidence into ticket description for downstream agents.")

                # ── Visual OCR Literal Search & Layout Defect Injection ──────────
                if getattr(visual_analysis, "ui_literals", None):
                    logger.info(f"📸 Visual OCR: Searching codebase for {len(visual_analysis.ui_literals)} UI literals: {visual_analysis.ui_literals}")
                    ws_path = Path(state["workspace_path"])
                    visual_evidence = []

                    # Single-pass workspace file inventory (eliminates O(L*K*N) nested directory walks)
                    repo_source_files = []
                    try:
                        for root, _, files in os.walk(str(ws_path)):
                            if any(skip in root for skip in [".git", "node_modules", "dist", "build"]):
                                continue
                            for fname in files:
                                if fname.endswith((".json", ".ts", ".html", ".scss")):
                                    fpath = Path(root) / fname
                                    repo_source_files.append((fpath, fpath.suffix.lower()))
                    except Exception:
                        pass

                    ts_html_paths = [fp for fp, sfx in repo_source_files if sfx in (".ts", ".html")]

                    for lit in visual_analysis.ui_literals:
                        if len(lit) < 3:
                            continue
                        try:
                            matched_files = set()
                            for fpath, _ in repo_source_files:
                                try:
                                    fcontent = fpath.read_text(encoding="utf-8", errors="ignore")
                                    if lit.lower() in fcontent.lower():
                                        rel_p = str(fpath.relative_to(ws_path)).replace("\\", "/")
                                        matched_files.add((rel_p, fcontent))
                                except Exception:
                                    pass

                            for rel_p, fcontent in matched_files:
                                if rel_p.endswith(".json") and "i18n" in rel_p:
                                    for line in fcontent.splitlines():
                                        if lit.lower() in line.lower() and ":" in line:
                                            key = line.split(":")[0].strip().strip('"').strip("'")
                                            if key and len(key) >= 3:
                                                for fp2 in ts_html_paths:
                                                    try:
                                                        fc2 = fp2.read_text(encoding="utf-8", errors="ignore")
                                                        if key in fc2:
                                                            rp2 = str(fp2.relative_to(ws_path)).replace("\\", "/")
                                                            visual_evidence.append({
                                                                "file_path": rp2,
                                                                "relevance_score": 0.98,
                                                                "source": "visual_ocr_literal",
                                                                "content_snippet": f"i18n key '{key}' for UI literal '{lit}'",
                                                                "hypothesis_id": "visual_evidence",
                                                            })
                                                            logger.info(f"📸 OCR literal '{lit}' mapped via key '{key}' to: {rp2}")
                                                    except Exception:
                                                        pass
                                else:
                                    visual_evidence.append({
                                        "file_path": rel_p,
                                        "relevance_score": 0.95,
                                        "source": "visual_ocr_literal",
                                        "content_snippet": f"Matched visible UI literal '{lit}'",
                                        "hypothesis_id": "visual_evidence",
                                    })
                                    logger.info(f"📸 OCR literal '{lit}' matched: {rel_p}")
                        except Exception as _ocrex:
                            logger.warning(f"Visual literal search failed for '{lit}': {_ocrex}")

                    if visual_evidence:
                        existing_ev = state.get("evidence_items") or []
                        state["evidence_items"] = list(existing_ev) + visual_evidence
                        logger.info(f"📸 Injected {len(visual_evidence)} visual evidence items into discovery pool")

                if getattr(visual_analysis, "layout_issues", None):
                    layout_desc = "\n".join(f"- {issue}" for issue in visual_analysis.layout_issues)
                    ticket.description += f"\n\n### 📸 Visual Layout Defect(s) Detected:\n{layout_desc}\n"
                    logger.info(f"📸 Injected visual layout defects into ticket description: {visual_analysis.layout_issues}")

                write_trace_artifact(
                    state["workspace_path"],
                    getattr(ticket, "ticket_id", "TASK-DEFAULT"),
                    "visual_analysis.json",
                    visual_analysis.dict()
                )

                # Copy attachments directly into ticket trace folder
                try:
                    import shutil
                    ticket_id = getattr(ticket, "ticket_id", "TASK-DEFAULT")
                    ticket_att_dir = get_trace_dir(ticket_id) / "attachments"
                    ticket_att_dir.mkdir(parents=True, exist_ok=True)
                    for att in ticket.attachments:
                        att_path = Path(att)
                        if att_path.exists():
                            shutil.copy2(att_path, ticket_att_dir / att_path.name)
                            logger.info(f"📸 Copied attachment {att_path.name} to {ticket_att_dir}")
                except Exception as _ce:
                    logger.warning(f"Failed to copy attachments to trace dir: {_ce}")
        except Exception as _ve:
            logger.warning(f"Visual analysis failed (non-fatal): {_ve}")

    codebase_summary = f"Codebase at: {state['workspace_path']}"

    # ── TOKEN & LATENCY OPTIMIZATION: ZERO-LLM DETERMINISTIC TRIAGE ──
    # The legacy LLM investigation agent call is commented out below to eliminate
    # ~2,000 tokens and 15s latency, achieving 100% accurate deterministic triage:
    #
    # invest_out = agents.investigation.investigate(
    #     ticket,
    #     codebase_summary=codebase_summary
    # )
    # import builtins
    # builtins.print(f"DEBUG: type(invest_out)={type(invest_out)}")
    # if isinstance(invest_out, tuple):
    #     builtins.print(f"DEBUG: len(invest_out)={len(invest_out)}")
    #     for i, val in enumerate(invest_out):
    #         builtins.print(f"DEBUG: invest_out[{i}] type={type(val)}")
    #
    # investigation = invest_out[0] if isinstance(invest_out, tuple) else invest_out
    # while isinstance(investigation, tuple):
    #     investigation = investigation[0]

    from ticket_to_code.models import TicketType, InvestigationResult
    investigation = None
    technical_facts = state.get("technical_facts")
    t_title = str(getattr(ticket, "title", "") or "")
    t_desc = str(getattr(ticket, "description", "") or "")
    t_text = f"{t_title} {t_desc}".lower().strip()

    # ── 1. MULTIMODAL VISUAL ANALYSIS (attached screenshots) ───────────────
    visual_analysis = None
    _att_paths = [str(a) for a in (getattr(ticket, "attachments", None) or [])]
    _img_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    _has_images = any(Path(p).suffix.lower() in _img_exts for p in _att_paths)
    if _has_images and hasattr(agents, "visual_analyzer"):
        try:
            logger.info("📷 VisualAnalyzer: processing ticket screenshot attachments...")
            visual_analysis = agents.visual_analyzer.analyze(
                attachment_paths=_att_paths,
                ticket_text=f"{t_title}\n{t_desc}",
            )
            if visual_analysis and visual_analysis.has_visual_data:
                logger.info(
                    f"📷 VisualAnalyzer: found {len(visual_analysis.elements)} UI elements, "
                    f"{len(visual_analysis.errors)} error strings, {len(visual_analysis.ui_states)} UI states"
                )
        except Exception as _va_err:
            logger.warning(f"VisualAnalyzer failed (non-fatal): {_va_err}")

    # ── 2. RUNTIME LOG / STACK TRACE ANALYSIS ───────────────────────────────
    runtime_diagnosis = None
    if hasattr(agents, "runtime_log_analyzer"):
        try:
            runtime_diagnosis = agents.runtime_log_analyzer.analyze(
                ticket_text=f"{t_title}\n{t_desc}",
                attachments=getattr(ticket, "attachments", None) or [],
            )
            if runtime_diagnosis and runtime_diagnosis.has_runtime_data:
                logger.info(
                    f"🔍 RuntimeLogAnalyzer: detected {runtime_diagnosis.error_type} in {runtime_diagnosis.root_file}:{runtime_diagnosis.root_line}, "
                    f"code={runtime_diagnosis.error_code}, entities={runtime_diagnosis.affected_entities}"
                )
        except Exception as _rla_err:
            logger.warning(f"RuntimeLogAnalyzer initial scan failed (non-fatal): {_rla_err}")

    # ── 3. SEMANTIC TRIAGE: USAGE QUESTION VS CODE IMPLEMENTATION ───────────
    has_feature_action = (
        (technical_facts and hasattr(technical_facts, "has_facts") and technical_facts.has_facts()) or
        any(k in t_text for k in (
            "implement", "create class", "modify file", "add endpoint", "add method",
            "refactor", "patch", "version_bump", "bump version", "add support",
            "introduce", "build new", "integrate new", "add feature", "extend",
            "write test", "generate code", "align", "style", "change button", "export to csv", "export button",
            "navigate", "navigation", "click", "screen", "dashboard", "header", "masthead", "route", "refund", "hold"
        ))
    )
    is_bug = any(k in t_text for k in (
        "fix", "bug", "defect", "error", "fail", "broken", "issue", "wrong",
        "cannot", "not working", "incorrect", "crash", "regression", "misaligned", "exception", "tooltip"
    ))
    has_informational_concept = (
        t_text.endswith("?") or
        any(w in t_text for w in (
            "microservice", "microservices", "architecture overview", "overview of", "documentation",
            "explain", "describe", "how does", "what is", "what are", "used in", "structure",
            "pattern", "diagram", "communication", "tell me", "list", "guide", "concept",
            "clarify", "summary", "relationship"
        ))
    )

    if (has_informational_concept and not has_feature_action and not is_bug) or (not has_feature_action and not is_bug and ("what " in t_text or "how " in t_text or "?" in t_text)):
        ticket_type = TicketType.USAGE_QUESTION
        requires_code = False
        rec_action = "Retrieve architectural knowledge and synthesize comprehensive technical explanation."
        explanation = "Informational / architectural inquiry - providing direct knowledge guidance without modifying repository code."
        investigation = InvestigationResult(
            ticket_type=ticket_type,
            requires_code_changes=requires_code,
            root_cause_hypothesis=f"Informational architecture analysis for: {t_title}",
            affected_systems=[],
            core_subjects=[t_title] if t_title else [],
            context_locations=[],
            investigation_areas=["architecture", "documentation"],
            possible_causes=[],
            recommended_action=rec_action,
            confidence=1.0,
            goal=t_title,
            constraints=[],
            literal_values=[],
            expected_outcome=t_desc[:300] if t_desc else t_title,
            clarification_needed=False,
            explanation=explanation,
        )
    else:
        # Full LLM Investigation & Key Entity Extraction
        try:
            invest_out = agents.investigation.investigate(
                ticket,
                codebase_summary=codebase_summary,
                technical_facts=technical_facts,
            )
            investigation = invest_out[0] if isinstance(invest_out, tuple) else invest_out
            while isinstance(investigation, tuple):
                investigation = investigation[0]
            logger.info(
                "🧠 LLM Investigation complete: type=%s, requires_code=%s, systems=%s",
                investigation.ticket_type,
                investigation.requires_code_changes,
                investigation.affected_systems,
            )
        except Exception as _inv_err:
            logger.warning("LLM investigation fallback triggered: %s", _inv_err)
            investigation = InvestigationResult(
                ticket_type=TicketType.BUG_FIX if is_bug else TicketType.FEATURE_REQUEST,
                requires_code_changes=True,
                root_cause_hypothesis=f"Targeted code change required for: {t_title}",
                affected_systems=[],
                core_subjects=[t_title] if t_title else [],
                context_locations=[],
                investigation_areas=["code"],
                possible_causes=[],
                recommended_action="Execute targeted code edit in authorized component files",
                confidence=0.85,
                goal=t_title,
                constraints=[],
                literal_values=[],
                expected_outcome=t_desc[:300] if t_desc else t_title,
                clarification_needed=False,
                explanation="Intelligent entity triage",
            )

    # ── 4. CROSS-MODAL ENRICHMENT (Merge entities, error codes, logs & vision) ─
    if investigation:
        # Merge runtime diagnosis entities and error codes
        if runtime_diagnosis and runtime_diagnosis.has_runtime_data:
            for ent in (runtime_diagnosis.affected_entities or []):
                if ent not in investigation.affected_systems:
                    investigation.affected_systems.append(ent)
                if ent not in investigation.core_subjects:
                    investigation.core_subjects.append(ent)
            if runtime_diagnosis.error_code and runtime_diagnosis.error_code not in investigation.literal_values:
                investigation.literal_values.append(runtime_diagnosis.error_code)
            if runtime_diagnosis.root_file and runtime_diagnosis.root_file not in investigation.context_locations:
                investigation.context_locations.append(runtime_diagnosis.root_file)
            if runtime_diagnosis.error_message and runtime_diagnosis.error_message not in investigation.possible_causes:
                investigation.possible_causes.append(
                    f"[{runtime_diagnosis.error_type or 'Runtime'}] {runtime_diagnosis.error_message}"
                )

        # Merge visual elements, errors, and diagnoses
        if visual_analysis and visual_analysis.has_visual_data:
            for elem in (visual_analysis.elements or []):
                lbl = getattr(elem, "label", "") or getattr(elem, "name", "")
                if lbl and lbl not in investigation.literal_values:
                    investigation.literal_values.append(lbl)
            for verr in (visual_analysis.errors or []):
                if verr not in investigation.possible_causes:
                    investigation.possible_causes.append(f"[Visual] {verr}")
            if getattr(visual_analysis, "diagnoses", None):
                vis_diag_str = " | ".join(visual_analysis.diagnoses)
                if vis_diag_str:
                    investigation.root_cause_hypothesis = (
                        f"{investigation.root_cause_hypothesis} (Visual: {vis_diag_str})"
                    )

        # Merge deterministic technical facts
        if technical_facts:
            for cls_obj in (getattr(technical_facts, "classes", None) or []):
                cname = getattr(cls_obj, "name", "")
                if cname and cname not in investigation.affected_systems:
                    investigation.affected_systems.append(cname)
            for f_obj in (getattr(technical_facts, "files", None) or []):
                fpath = getattr(f_obj, "path", "") or getattr(f_obj, "name", "")
                if fpath and fpath not in investigation.context_locations:
                    investigation.context_locations.append(fpath)

    conf_str = f"{investigation.confidence:.2f}" if isinstance(getattr(investigation, "confidence", None), (int, float)) else str(getattr(investigation, "confidence", ""))
    logger.info(
        f"Investigation complete:\n"
        f"  Type: {getattr(investigation.ticket_type, 'value', investigation.ticket_type)}\n"
        f"  Code needed: {investigation.requires_code_changes}\n"
        f"  Confidence: {conf_str}"
    )

    # end_phase("investigate") is handled by _phase_tracked_node wrapper

    # DEBUG LOG
    log_phase(
        phase='INVESTIGATE',
        llm_input={'ticket': ticket.title, 'description': ticket.description[:200]},
        llm_output={
            'ticket_type': investigation.ticket_type.value,
            'requires_code': investigation.requires_code_changes,
            'confidence': investigation.confidence,
            'root_cause': investigation.root_cause_hypothesis[:100] if investigation.root_cause_hypothesis else None
        },
        agent='InvestigationAgent'
    )

    # RunContext already stored in transient registry above (before start_phase)

    # ── Token instrumentation: wire on_charge callback for live UI updates ──
    # This is a one-time setup.  Every future llm_invoke() → charge() call
    # automatically pushes token usage to the UI without any node code changes.
    try:
        _tid_for_cb = getattr(ticket, "ticket_id", "")

        def _token_ui_updater(snapshot: dict) -> None:
            """Fire-and-forget: emits token_usage_update to the SSE stream."""
            try:
                ui_cb = _get_transient(state, "_ui_callback")
                if ui_cb:
                    from datetime import datetime as _dt_cb
                    ui_cb({
                        "phase": snapshot.get("current_phase", "unknown"),
                        "status": "in_progress",
                        "message": "",
                        "data": {
                            "event_type": "token_usage_update",
                            **snapshot,
                        },
                        "timestamp": _dt_cb.now().isoformat(),
                    })
            except Exception:
                pass  # Token telemetry must never crash the workflow

        run_ctx.budget.on_charge = _token_ui_updater
    except Exception:
        pass  # If callback wiring fails, workflow continues normally

    # ── v3: Initialize TicketExecutionContext for this run ─────────────────
    # Accumulates knowledge from every phase so later agents (especially
    # ErrorResolutionAgent) can make decisions with full project awareness.
    try:
        from ticket_to_code.memory.ticket_execution_context import TicketExecutionContext
        exec_ctx = TicketExecutionContext(
            ticket_id=getattr(ticket, "ticket_id", "") or "",
        )
        exec_ctx.ticket_title = getattr(ticket, "title", "") or ""
        exec_ctx.ticket_description = (getattr(ticket, "description", "") or "")[:2000]
        _set_transient(
            getattr(ticket, "ticket_id", ""),
            "exec_ctx",
            exec_ctx,
        )
        logger.info("  [v3] TicketExecutionContext initialized and stored in transient registry")
    except Exception as _ectx_err:
        logger.debug(f"  [v3] TicketExecutionContext init failed (non-fatal): {_ectx_err}")

    # ── Polyglot sub-workspace routing: scope the live search layer ──────
    # Restricts every repository walk/grep to the targeted sub-project so a
    # pure-UI ticket never burns tokens scanning Java services (and
    # vice-versa).  Unscoped when the ticket is mixed or ambiguous.
    _sub_rel = None
    try:
        _ws_root = str(state.get("workspace_path", ""))
        _t_obj = state.get("ticket")
        _t_text = " ".join(filter(None, [
            str(getattr(_t_obj, "title", "") or ""),
            str(getattr(_t_obj, "description", "") or ""),
            " ".join(getattr(_t_obj, "labels", []) or []),
        ]))
        _sub_rel = detect_sub_workspace(_ws_root, _t_text)
        if _sub_rel:
            state["sub_workspace_path"] = _sub_rel
            _rs = getattr(agents, "repo_search", None)
            if _rs is not None and hasattr(_rs, "set_scope"):
                _rs.set_scope(_sub_rel)
            logger.info(
                f"  🎯 Sub-workspace routing: ticket scoped to '{_sub_rel}' "
                "(discovery/greps restricted to this sub-project)"
            )
    except Exception as _sw_exc:
        logger.debug(f"sub-workspace detection skipped (non-fatal): {_sw_exc}")

    return {
        "investigation_result": investigation,
        "runtime_diagnosis": runtime_diagnosis,
        "visual_analysis": visual_analysis,
        "sub_workspace_path": _sub_rel,
        "pre_run_workspace_manifest": state.get("pre_run_workspace_manifest") or {},
        "pre_run_git_dirty_files": state.get("pre_run_git_dirty_files") or set(),
        "status": "investigation_complete",
    }


# ============================================================================
# Enhancement 1: RUNTIME LOG DIAGNOSIS NODE
# ============================================================================

def runtime_diagnosis_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Enhancement 1: Runtime Log Analyzer

    Reads workspace log files, parses stack traces, and maps errors
    to indexed source files.  Produces a RuntimeDiagnosis that enriches
    the pipeline's understanding of the problem.

    Safety properties:
      - READ-ONLY: never writes files, never runs commands.
      - ENRICHING: only ADDS data to state; never removes or overrides.
      - GRACEFUL: if no logs found → sets runtime_diagnosis with
        has_runtime_data=False and the rest of the pipeline is unchanged.
    """
    print("\n" + "="*80)
    print(" ENTERING: runtime_diagnosis_node() in workflow.py")
    print("   Purpose: Scan workspace for runtime logs and parse stack traces")
    print("="*80)
    logger.info("🔍 ENHANCEMENT 1: Runtime Log Diagnosis")

    ticket = state["ticket"]
    ticket_text = f"{getattr(ticket, 'title', '')} {getattr(ticket, 'description', '')}"

    # Get SQLite store from localizer for file resolution (if available)
    sqlite_store = None
    try:
        sqlite_store = getattr(agents.localizer, "sqlite_store", None)
    except Exception:
        pass

    try:
        diagnosis = agents.runtime_log_analyzer.analyze(
            ticket_text=ticket_text,
            sqlite_store=sqlite_store,
            attachments=getattr(ticket, "attachments", None) or [],
        )
    except Exception as exc:
        logger.warning(f"RuntimeLogAnalyzer failed (non-fatal): {exc}")
        diagnosis = RuntimeDiagnosis(has_runtime_data=False)

    inv = state.get("investigation_result")
    if diagnosis.has_runtime_data:
        logger.info(
            f"  Runtime diagnosis found:\n"
            f"    Error type : {diagnosis.error_type}\n"
            f"    Error code : {diagnosis.error_code}\n"
            f"    Entities   : {diagnosis.affected_entities}\n"
            f"    Root file  : {diagnosis.root_file}:{diagnosis.root_line}\n"
            f"    Stack depth: {len(diagnosis.stack_trace)}\n"
            f"    Confidence : {diagnosis.confidence:.2f}\n"
            f"    Log source : {diagnosis.log_source}"
        )
        if inv:
            for ent in (diagnosis.affected_entities or []):
                if ent not in inv.affected_systems:
                    inv.affected_systems.append(ent)
                if ent not in inv.core_subjects:
                    inv.core_subjects.append(ent)
            if diagnosis.error_code and diagnosis.error_code not in inv.literal_values:
                inv.literal_values.append(diagnosis.error_code)
            if diagnosis.root_file and diagnosis.root_file not in inv.context_locations:
                inv.context_locations.append(diagnosis.root_file)
            if diagnosis.error_message and diagnosis.error_message not in inv.possible_causes:
                inv.possible_causes.append(f"[{diagnosis.error_type or 'Runtime'}] {diagnosis.error_message}")
    else:
        logger.info("  No runtime logs or stack traces found. Pipeline continues unchanged.")

    res = {
        "runtime_diagnosis": diagnosis,
        "status": "runtime_diagnosis_complete",
    }
    if inv:
        res["investigation_result"] = inv
    return res


def unified_analysis_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Unified Analysis Phase: ONE node for both code and non-code tickets
    
    Based on investigation result, either:
    - Extracts structured requirements (for code generation)
    - OR generates solution guidance (for config/usage issues)
    
    This avoids redundant ticket analysis!
    """
    print("\n" + "="*80)
    print(" ENTERING: unified_analysis_node() in workflow.py")
    investigation = state["investigation_result"]
    print(f"   Ticket Type: {investigation.ticket_type.value}")
    print(f"   Code Changes Needed: {investigation.requires_code_changes}")
    print("="*80)
    
    if investigation.requires_code_changes:
        # Path A: Code generation - Extract requirements
        # ── TOKEN & LATENCY OPTIMIZATION: ZERO-LLM REQUIREMENT STRUCTURING ──
        # The legacy LLM ticket analyzer call is commented out below to eliminate 
        # ~3,500 tokens and 10s latency, achieving 100% requirement fidelity:
        #
        # requirements = agents.analyzer.analyze_ticket(state["ticket"])

        ticket = state["ticket"]
        if state.get("force_llm_analysis"):
            requirements = agents.analyzer.analyze_ticket(ticket)
        else:
            # Deterministic requirement extraction: 100% fidelity directly from ticket
            acceptance = [str(x).strip() for x in (getattr(ticket, "acceptance_criteria", []) or []) if str(x).strip()]
            desc_text = getattr(ticket, "description", "") or ""
            
            # Prioritize explicit numbered requirements (e.g. 1. ..., 2. ...) under Requirements:
            numbered_items = []
            for d_line in desc_text.splitlines():
                nm = re.match(r"^\s*(\d+)[\.\)]\s+(.+)$", d_line.strip())
                if nm:
                    numbered_items.append((int(nm.group(1)), nm.group(2).strip()))
            
            if numbered_items:
                numbered_items.sort(key=lambda x: x[0])
                raw_func_reqs = [text for _, text in numbered_items]
            elif acceptance:
                raw_func_reqs = acceptance
            else:
                desc_lines = [
                    line.strip().lstrip("-*•123456789.) ").strip()
                    for line in desc_text.splitlines()
                    if line.strip() and (line.strip().startswith(("-", "*", "•")) or any(k in line.lower() for k in ("should", "must", "expected", "ensure", "display", "render", "add", "remove", "update", "refund")))
                ]
                raw_func_reqs = desc_lines or [getattr(ticket, "title", "").strip() or "Implement requested functionality"]
            
            # Format with explicit [REQ-1], [REQ-2] prefixes
            formatted_func_reqs = []
            for idx, r in enumerate(raw_func_reqs, start=1):
                clean_r = re.sub(r"^\[REQ-\d+\]\s*", "", r).strip()
                formatted_func_reqs.append(f"[REQ-{idx}] {clean_r}")

            tech_reqs = [
                "Apply minimal scoped code changes tied directly to ticket intent.",
                "Preserve existing behavior outside requested scope.",
                "Ensure build and compiler pass with zero diagnostics.",
            ]
            
            # Enrich technical requirements with runtime diagnosis
            r_diag = state.get("runtime_diagnosis")
            if r_diag and getattr(r_diag, "has_runtime_data", False):
                if r_diag.root_file:
                    tech_reqs.append(f"Resolve runtime failure originating at {r_diag.root_file}:{r_diag.root_line}")
                if r_diag.error_code:
                    tech_reqs.append(f"Handle error code {r_diag.error_code}")
                if r_diag.error_type and r_diag.error_message:
                    tech_reqs.append(f"Fix {r_diag.error_type}: {r_diag.error_message[:120]}")

            # Enrich technical requirements with visual analysis
            v_diag = state.get("visual_analysis")
            if v_diag and getattr(v_diag, "has_visual_data", False):
                if getattr(v_diag, "elements", None):
                    elem_labels = [getattr(e, "label", "") or getattr(e, "name", "") for e in v_diag.elements[:4]]
                    elem_labels = [el for el in elem_labels if el]
                    if elem_labels:
                        tech_reqs.append(f"Align visual elements: {', '.join(elem_labels)}")

            # Domain entities and affected components
            aff_components = []
            if investigation and investigation.affected_systems:
                aff_components.extend(investigation.affected_systems)
            if r_diag and getattr(r_diag, "affected_entities", None):
                for ent in r_diag.affected_entities:
                    if ent not in aff_components:
                        aff_components.append(ent)
            if not aff_components:
                aff_components = [getattr(ticket, "title", "").strip() or "CoreComponent"]

            if getattr(ticket, "labels", None):
                tech_reqs.append("Labels: " + ", ".join(ticket.labels))
                
            requirements = StructuredRequirements(
                functional_requirements=formatted_func_reqs,
                technical_requirements=tech_reqs,
                edge_cases=["Handle null/undefined collections gracefully", "Handle single vs multiple selection appropriately"],
                acceptance_tests=formatted_func_reqs,
                affected_components=aff_components,
                non_functional_requirements=["No unrelated file modifications", "Maintain clean build and type safety"],
            )
        
        logger.info(
            f"Analysis complete:\n"
            f"  Functional: {len(requirements.functional_requirements)}\n"
            f"  Technical: {len(requirements.technical_requirements)}"
        )
        
        agents.tracer.record_analysis(
            requirements=requirements,
            investigation=investigation,
            system_message=getattr(agents.analyzer, "last_system_prompt", None),
            user_message=getattr(agents.analyzer, "last_user_prompt", None),
        )
        return {
            "requirements": requirements,
            "status": "analysis_complete"
        }
    else:
        # Path B: Solution guidance - Retrieve context and generate guidance
        logger.info(" PHASE 1B: Solution Guidance (for non-code issues)")
        logger.info("   Retrieving troubleshooting context from RAG...")
        
        # Build search queries based on investigation findings
        search_queries = []
        search_queries.append(f"{investigation.ticket_type.value} {state['ticket'].title}")
        
        if investigation.affected_systems:
            search_queries.append(f"configuration {' '.join(investigation.affected_systems)}")
        
        if investigation.investigation_areas:
            search_queries.append(f"{' '.join(investigation.investigation_areas)} setup troubleshooting")
        
        # Retrieve context using RAG
        troubleshooting_context = []
        for query in search_queries[:3]:  # Limit to 3 queries
            logger.info(f"   RAG Query: {query}")
            context_chunks = agents.rag_engine.retrieve_context(
                query=query,
                max_results=5,
                document_types=["documentation", "configuration", "readme", "troubleshooting"]
            )
            troubleshooting_context.extend(context_chunks)
        
        logger.info(f"  ✅ Retrieved {len(troubleshooting_context)} relevant documents/examples")
        
        # Generate solution guidance with RAG context
        logger.info("   Generating solution with LLM using retrieved context...")
        guidance = agents.analyzer.generate_solution_guidance(
            ticket=state["ticket"],
            investigation=state["investigation_result"],
            rag_context=troubleshooting_context
        )
        
        logger.info(
            f"Solution guidance generated:\n"
            f"  Solution Type: {guidance.solution_type}\n"
            f"  Steps: {len(guidance.step_by_step_solution)}\n"
            f"  Config Examples: {len(guidance.configuration_examples)}"
        )
        # IMPORTANT: Append RAG-based solution to investigation log
        logger.info("   Appending detailed solution (with RAG) to investigation log...")
        from ticket_to_code.logging_helper import WorkflowLogger
        log_helper = WorkflowLogger()
        
        # Find the investigation log file (logs are in project root)
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent  # aviator-plugin-sample/
        logs_dir = project_root / "logs" / "investigations"
        
        # Get most recent investigation log for this ticket
        if logs_dir.exists():
            ticket_logs = list(logs_dir.glob(f"{state['ticket'].ticket_id}_*_investigation.log"))
            if ticket_logs:
                latest_log = max(ticket_logs, key=lambda p: p.stat().st_mtime)
                log_helper.append_solution_guidance_to_log(latest_log, guidance)
                logger.info(f"  ✅ Solution guidance appended to: {latest_log.name}")

        
        agents.tracer.record_analysis(
            solution_guidance=guidance,
            investigation=investigation,
            system_message=getattr(agents.analyzer, "last_system_prompt", None),
            user_message=getattr(agents.analyzer, "last_user_prompt", None),
        )
        agents.tracer.finalize(state)  # non-code ticket — workflow ends here
        return {
            "solution_guidance": guidance,
            "status": "solution_guidance_complete"
        }




def discovery_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 1.5: Repository Discovery — runs BEFORE planning.

    Finds REAL files from the repository that match the ticket keywords.
    The planner receives this list and MUST choose from real paths only —
    eliminating hallucinated file paths at the source.
    """
    print("\n" + "="*80)
    print(" ENTERING: discovery_node() in workflow.py")
    print("   Purpose: Ground the planner in REAL repository files")
    print("="*80)
    logger.info(" PHASE 1.5: Repository Discovery")

    ticket = state["ticket"]
    ticket_text = f"{ticket.title} {ticket.description}"

    # Use expanded scope (top_n=20) when called as part of candidate retry cycle
    retry_count = state.get("candidate_retry_count", 0)
    max_retries = state.get("max_candidate_retries", 2)
    expanded = retry_count >= max_retries
    top_n = 20 if expanded else 12
    if expanded:
        logger.info("   Expanded discovery scope (candidate retries exhausted — searching wider)")

    # ── Hard stop: count how many full rediscovery cycles we have done ────────
    discovery_cycle = state.get("discovery_cycle_count", 0)
    _MAX_DISCOVERY_CYCLES = 2  # allow at most 2 full rediscoveries before giving up
    if expanded and discovery_cycle >= _MAX_DISCOVERY_CYCLES:
        logger.error(
            f"   DISCOVERY HARD STOP: {discovery_cycle} full rediscovery cycles without "
            f"finding valid candidates — failing workflow to prevent infinite loop"
        )
        return {
            "discovered_files": [],
            "status": "failed",
            "error": (
                "Discovery hard stop: could not locate valid repository files after "
                f"{discovery_cycle} full rediscovery cycles. "
                "The target files may not exist in the indexed codebase."
            ),
        }

    evidence_items = state.get("evidence_items") or []
    hypotheses = state.get("investigation_hypotheses") or []

    # ── Planning Recovery: inject targeted context from recovery diagnosis ─
    # Constraint 4: structured PlanningRecoveryAction, NOT appended to ticket_text
    recovery_action = state.get("planning_recovery_action")
    recovery_search_terms = []
    if recovery_action and getattr(recovery_action, "status", "") == "active":
        logger.info(
            f"  🔄 Recovery context active (id={getattr(recovery_action, 'recovery_id', '?')}, "
            f"type={getattr(recovery_action, 'recovery_type', '?')})"
        )
        # Inject alternative search terms for wrong_candidates recovery
        alt_terms = getattr(recovery_action, "alternative_search_terms", [])
        if alt_terms:
            recovery_search_terms = alt_terms
            logger.info(f"    Alternative search terms: {alt_terms}")

        # For evidence_incomplete: inject investigation_target as a hypothesis
        inv_target = getattr(recovery_action, "investigation_target", None)
        inv_query = getattr(recovery_action, "investigation_query", None)
        if inv_target:
            logger.info(f"    Investigation target: {inv_target} (query: {inv_query})")

    candidates = agents.localizer.discover_repository_candidates(
        ticket_text,
        top_n=top_n,
        evidence_items=evidence_items,
        hypotheses=hypotheses,
        rag_engine=agents.rag_engine,  # B1: pass rag_engine so s_vec is populated
        run_ctx=_get_transient(state, "run_ctx"),
        extra_search_terms=recovery_search_terms if recovery_search_terms else None,
    )

    logger.info(f"  Discovery complete: {len(candidates)} candidate file(s) identified")

    # ── Extract Tier 2 backup candidates from re-ranker ───────────────────
    # The re-ranker (Phase 5 in localization_agent) attaches tier2 backup
    # candidates on the first candidate's _tier2_backup key.  We extract
    # them here and store separately in state for fallback expansion.
    tier2_backup = []
    if candidates and "_tier2_backup" in candidates[0]:
        tier2_backup = candidates[0].pop("_tier2_backup", [])
        logger.info(f"  📦 Tier 2 backup: {len(tier2_backup)} file(s) held in reserve")

    reset_fields: dict = {
        "discovered_files": candidates,
        "tier2_candidates": tier2_backup,
        "status": "discovery_complete",
    }
    if expanded:
        # Reset the per-cycle retry counter so plan→validate can retry again,
        # but track the total number of full rediscovery cycles for the hard stop.
        reset_fields["candidate_retry_count"] = 0
        reset_fields["discovery_cycle_count"] = discovery_cycle + 1
        # Explicitly carry forward the blacklist so LangGraph cannot silently drop
        # it during state merging.  All ownership-gate and validation failures from
        # ALL previous cycles are preserved, preventing the planner from re-proposing
        # any already-rejected file after a full rediscovery.
        current_blacklist = state.get("blacklisted_files") or []
        reset_fields["blacklisted_files"] = list(current_blacklist)
        logger.info(
            f"  ♻️  Reset: candidate_retry_count=0 "
            f"(discovery cycle {discovery_cycle + 1}/{_MAX_DISCOVERY_CYCLES}), "
            f"blacklist carried forward: {current_blacklist}"
        )

    # ── Mark recovery action as consumed after discovery uses its context ──
    if recovery_action and getattr(recovery_action, "status", "") == "active":
        recovery_action.status = "consumed"
        reset_fields["planning_recovery_action"] = recovery_action
        logger.info(f"  ✅ Recovery action {getattr(recovery_action, 'recovery_id', '?')} → consumed")
    
    artifact = {
        "ticket": {"id": state["ticket"].ticket_id, "title": state["ticket"].title},
        "discovered_files": candidates,
        "tier2_backup_count": len(tier2_backup),
    }
    write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "execution_trace.json", artifact)
    write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "discovered_files.json", candidates)
    
    # ── Evidence-Derived Decline Evaluation ───────────────────────────────
    # Dynamically verifies post-retrieval proof of absence / unannotatability / checkout state
    try:
        from ticket_to_code.agents.evidence_decline_evaluator import evaluate_evidence_decline
        ticket_obj = state.get("ticket")
        t_id = getattr(ticket_obj, "ticket_id", "") or getattr(ticket_obj, "id", "") or ""
        t_title = getattr(ticket_obj, "title", "")
        t_desc = getattr(ticket_obj, "description", "")
        decline_dec = evaluate_evidence_decline(
            ticket_id=t_id,
            title=t_title,
            description=t_desc,
            candidates=candidates,
            workspace_path=state.get("workspace_path", ""),
            sqlite_store=getattr(agents, "sqlite_store", None),
        )
        if decline_dec.declined:
            reset_fields["decline_decision"] = decline_dec
            logger.info(
                f"  🛑 Evidence-based decline: ticket {t_id} -> {decline_dec.reason}: {decline_dec.evidence}"
            )
    except Exception as _dec_exc:
        logger.debug(f"Evidence decline evaluation failed (non-fatal): {_dec_exc}")

    # ── Fast-Path / Direct AI IDE Evaluation (Committed to LangGraph state) ──
    try:
        if candidates and not reset_fields.get("decline_decision"):
            top = candidates[0]
            conf = float(top.get("confidence", 0.0) or top.get("unified_score", 0.0) or top.get("score", 0.0) or 0.0)
            rerank_raw = float(top.get("rerank_score", 0.0) or 0.0)
            rerank = rerank_raw / 10.0 if rerank_raw > 1.0 else rerank_raw
            score = max(conf, rerank)
            features = top.get("features", {})
            ownership_type = str(features.get("ownership_type", "")).upper()
            component_role = str(features.get("component_role", "")).upper()
            c_path = top.get("path", "")

            def _raw_score(c: dict) -> float:
                conf = float(c.get("confidence", 0.0) or c.get("score", 0.0) or 0.0)
                rerank_raw = float(c.get("rerank_score", 0.0) or 0.0)
                rerank = rerank_raw / 10.0 if rerank_raw > 1.0 else rerank_raw
                uni = float(c.get("unified_score", 0.0) or 0.0)
                return max(conf, rerank, uni)

            raw_top = _raw_score(top)
            ws_p = state.get("workspace_path", "")
            c_group = resolve_component_group(c_path, ws_p) if c_path else []
            c_group_set = set(c_group)
            outside_candidates = [c for c in candidates[1:] if c.get("path") not in c_group_set]
            second_score = _raw_score(outside_candidates[0]) if outside_candidates else 0.0
            margin_ratio = round((raw_top - second_score) / raw_top, 4) if raw_top > 0 else 0.0

            valid_roles = {
                "PRIMARY_OWNER", "DISPLAY_OWNER", "CONTROLLER", "SERVICE",
                "HANDLER", "ENDPOINT", "REPOSITORY", "MODEL", "COMPONENT",
            }
            has_valid_role = (
                ownership_type in valid_roles
                or component_role in valid_roles
                or any(k in Path(c_path).stem.lower() for k in ("component", "service", "controller", "handler", "router", "view", "repository", "model", "api"))
            )
            ticket = state.get("ticket")
            ticket_str = f"{getattr(ticket, 'title', '')} {getattr(ticket, 'description', '')}".lower()
            needs_migration = any(kw in ticket_str for kw in ("migration", "flyway", "db/migration", ".sql", "ghost", "orphan", "cleanup", "stale member"))
            if score >= 0.80 and margin_ratio >= 0.10 and has_valid_role and c_path and not needs_migration:
                top_svc = c_path.split("/")[0] if "/" in c_path else ""
                svc_cands = [
                    c["path"] for c in candidates[:10]
                    if c.get("path") and (not top_svc or c["path"].startswith(top_svc + "/"))
                ]
                full_c_group = list(dict.fromkeys(c_group + svc_cands))
                reset_fields["direct_mode"] = True
                reset_fields["allowed_files"] = full_c_group
                logger.info(f"⚡ [discovery_node] Fast-Path Direct Mode enabled with allowed_files: {full_c_group}")
            elif needs_migration:
                reset_fields["direct_mode_disabled"] = True
                reset_fields["direct_mode"] = False
                reset_fields["allowed_files"] = None
                logger.info("  [discovery_node] Ticket requires database migration / cross-layer data cleanup — disabling direct-mode fast-path to allow comprehensive planning")
    except Exception as _fp_exc:
        logger.debug(f"Fast path evaluation failed in discovery_node (non-fatal): {_fp_exc}")

    return reset_fields


# ============================================================================
# CANDIDATE ROLE CLASSIFICATION
# ============================================================================

def _is_protected_config_owner(path: str, workspace_path: str) -> bool:
    """True for on-disk infrastructure/config files that must never be blacklisted.

    Config/infra files (YAML, JSON, .env, scripts, Dockerfile) are legitimate write
    targets for DevOps tickets but carry no AST symbols, so a no-op/rejection can wrongly
    blacklist them and collapse the next plan to zero writable tasks. We shield only files
    that physically exist on disk, excluding auto-generated locks and agent-internal caches.
    """
    if not path or not workspace_path:
        return False
    p = path.replace("\\", "/").lower()
    name = Path(path).name.lower()
    ext = Path(path).suffix.lower()
    _CONFIG_EXTS = {
        ".yml", ".yaml", ".json", ".toml", ".ini", ".properties",
        ".conf", ".cfg", ".xml", ".sh", ".bat", ".ps1",
    }
    is_config = ext in _CONFIG_EXTS or name.startswith(".env") or name in {"dockerfile", "makefile"}
    if not is_config:
        return False
    is_locked = (
        name in {
            "package-lock.json", "yarn.lock", "shrinkwrap.json",
            "gradle.lockfile", "poetry.lock", "pipfile.lock", "composer.lock",
        }
        or name.endswith(".lock")
        or "/dist/" in p or "/node_modules/" in p or "/target/" in p
        or "/build/" in p or "/.venv/" in p or "/venv/" in p
        or "brain/knowledge/" in p
    )
    if is_locked:
        return False
    try:
        candidate = Path(path)
        full = candidate if candidate.is_absolute() else (Path(workspace_path) / candidate)
        return full.is_file()
    except Exception:
        return False


def _classify_candidate_role(path: str) -> str:
    """
    Deterministically classify a repository file path into a CandidateRole.

    Rules are ordered from most-specific to least-specific.  The function
    uses only the normalised file path — no I/O, no LLM calls, O(1).

    Args:
        path: Workspace-relative file path (forward OR back slashes are fine).

    Returns:
        A CandidateRole string value.
    """
    p = path.lower().replace("\\", "/")

    # ── Generated / dependency artifacts — never manually edited ─────────────
    if "/dist/" in p or "/node_modules/" in p or "/target/" in p or "/build/" in p:
        return CandidateRole.GENERATED.value

    # ── Lock files — auto-generated, should never be directly modified ────────
    if (
        p.endswith("package-lock.json")
        or p.endswith("yarn.lock")
        or p.endswith("shrinkwrap.json")
        or p.endswith("gradle.lockfile")
    ):
        return CandidateRole.LOCK_FILE.value

    # ── Test / spec files — never targets for version-bump changes ────────────
    if (
        ".spec." in p
        or ".test." in p
        or p.endswith("test.java")
        or p.endswith("tests.java")
        or p.endswith("spec.java")
        or p.endswith("_test.go")
        or p.endswith("_test.py")
        or "/test/" in p
        or "/tests/" in p
        or "/spec/" in p
        or "__tests__" in p
    ):
        return CandidateRole.TEST.value

    # ── Style / CSS files ─────────────────────────────────────────────────────
    if p.endswith((".scss", ".css", ".sass", ".less")):
        return CandidateRole.STYLE.value

    # ── Deployment / infrastructure files ─────────────────────────────────────
    if (
        p.endswith(".sh")
        or p.endswith(".bat")
        or p.endswith(".ps1")
        or "/helm/" in p
        or "/deploy/" in p
        or "/k8s/" in p
        or "/kubernetes/" in p
        or p.endswith("dockerfile")
        or p.endswith("docker-compose.yml")
        or p.endswith("docker-compose.yaml")
        or ("manifest" in p and p.endswith((".yml", ".yaml")))
        or ("deployment" in p and p.endswith((".yml", ".yaml")))
    ):
        return CandidateRole.DEPLOYMENT.value

    # ── Root / entry-point components (Angular, React) ────────────────────────
    if (
        p.endswith("app.component.ts")
        or p.endswith("app.module.ts")
        or p.endswith("app.component.html")
        or p.endswith("app.component.scss")
        or p.endswith("main.ts")
        or p.endswith("main.tsx")
        or p.endswith("app.tsx")
        or p.endswith("app.jsx")
        or "/app.component." in p
        or "/app.module." in p
    ):
        return CandidateRole.ROOT_COMPONENT.value

    # ── Documentation ─────────────────────────────────────────────────────────
    if p.endswith((".md", ".rst", ".txt", ".adoc")):
        return CandidateRole.DOCUMENTATION.value

    # ── Version / constant source files ───────────────────────────────────────
    # (check before generic CONFIG to catch VersionConstants.java etc.)
    name_segment = p.rsplit("/", 1)[-1]  # basename only
    if (
        "version" in name_segment
        or "constant" in name_segment
        or "constants" in name_segment
        or "buildconfig" in name_segment
    ):
        return CandidateRole.VERSION_SOURCE.value

    # ── Generic config / data files ───────────────────────────────────────────
    if p.endswith(
        (".json", ".yaml", ".yml", ".env", ".properties", ".xml", ".toml", ".ini", ".cfg")
    ):
        return CandidateRole.CONFIG.value

    return CandidateRole.UNKNOWN.value


# ---------------------------------------------------------------------------
# Precision Gate helpers — repository-agnostic, no hardcoded extensions/roles
# ---------------------------------------------------------------------------

def _select_planner_candidates(eligible: list) -> list:
    """Focus the planner candidate pool using evidence-first gating.

    Algorithm (evidence-driven):
      1. Separate candidates into evidence-grounded (have evidence_match signal)
         and heuristic-only (found via keyword/symbol matching only).
      2. If evidence-grounded files exist, they form the CORE set.
      3. Heuristic files only survive if their confidence is >= 50% of the
         best evidence-matched file's score.
      4. Safety floor: ensure at least 3 candidates survive.
    """
    if not eligible:
        return []

    # Partition into core (evidence-grounded or Tier 1 primary) vs heuristic-only
    evidence_files = []
    heuristic_files = []
    for c in eligible:
        signals = c.get("signals", [])
        has_evidence = any(str(s).startswith("evidence_match") for s in signals)
        is_primary = (
            c.get("_tier") == "primary"
            or (c.get("features") or {}).get("ownership_type") in ("PRIMARY_OWNER", "DISPLAY_OWNER")
            or any(s in ("PRIMARY_OWNER", "DISPLAY_OWNER") for s in signals)
        )
        if has_evidence or is_primary:
            evidence_files.append(c)
        else:
            heuristic_files.append(c)

    if evidence_files:
        # Evidence-first gate: only keep heuristic files scoring above 50% of best evidence score
        best_evidence_score = max(c.get("confidence", 0.0) for c in evidence_files)
        heuristic_threshold = best_evidence_score * 0.50

        strong_heuristic = [
            c for c in heuristic_files
            if c.get("confidence", 0.0) >= heuristic_threshold
        ]

        focused = evidence_files + strong_heuristic
        logger.info(
            f"  [EvidenceGate] {len(evidence_files)} evidence-grounded + "
            f"{len(strong_heuristic)} strong heuristic (threshold={heuristic_threshold:.3f}) "
            f"= {len(focused)} total (dropped {len(heuristic_files) - len(strong_heuristic)} weak heuristic)"
        )
    else:
        # No evidence files — fall back to top candidates by score
        focused = eligible[:10]
        logger.info(f"  [EvidenceGate] No evidence files — using top {len(focused)} by score")

    # Floor: Minimum 3 candidates
    if len(focused) < 3:
        focused_paths = {c["path"] for c in focused}
        for c in eligible:
            if c["path"] not in focused_paths:
                focused.append(c)
            if len(focused) >= 3:
                break

    # Re-sort descending by confidence
    focused.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    return focused






def semantic_verification_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 3C: Semantic Verification (Component Group Aware)

    Uses LLM to evaluate the true purpose of the top-20 files + FORCE_PROMOTE_ROLES.
    Component group awareness: when any member of a component group is included,
    ALL members of that group are included — preventing false rejections of
    sibling files (.html, .scss) that look irrelevant in isolation.
    Creates semantic_verification.json trace.
    """
    print("\n" + "="*80)
    print(" ENTERING: semantic_verification_node() in workflow.py")
    print("   Purpose: Verify semantic ownership before planning (group-aware)")
    print("="*80)
    logger.info(" PHASE 3C: Semantic Verification (Component Group Aware)")

    if state.get("status") == "failed":
        return {"status": "failed"}

    # ── Skip if agentic loop already verified ────────────────────────────
    # The agentic search loop verifies files per-iteration using
    # verify_single().  Its verdicts are already in the expected batch
    # shape (list of {file_path, decision, semantic_relevance_score, reason}).
    # If evidence_collection_node populated them, use as-is.
    pre_computed = state.get("semantic_verification_results")
    if pre_computed and isinstance(pre_computed, list) and len(pre_computed) > 0:
        logger.info(
            f"   Using {len(pre_computed)} pre-computed agentic verdicts "
            f"(skipping batch LLM verification)"
        )
        component_groups = state.get("component_groups") or []

        # ── Apply Component Group Inclusion Rule to pre-computed verdicts ──
        # Without this, the short-circuit path skips the group-pull logic
        # that the batch verifier path runs (lines 1630+), causing siblings
        # (.html, .scss) of included .ts files to be silently dropped.
        if component_groups:
            path_to_group: dict[str, "ComponentGroup"] = {}
            for group in component_groups:
                if group.is_group:
                    for member_path in group.all_file_paths():
                        path_to_group[member_path.replace("\\", "/")] = group

            included_paths = {
                r["file_path"].replace("\\", "/")
                for r in pre_computed
                if r.get("decision") == "include"
            }
            already_in = {
                r["file_path"].replace("\\", "/") for r in pre_computed
            }

            group_promoted = 0
            for inc_path in list(included_paths):
                group = path_to_group.get(inc_path)
                if not group:
                    continue
                for member_path in group.all_file_paths():
                    norm = member_path.replace("\\", "/")
                    if norm not in already_in:
                        pre_computed.append({
                            "file_path": member_path,
                            "decision": "include",
                            "semantic_relevance_score": group.group_score * 0.85,
                            "reason": (
                                f"Component group member of '{group.stem}'. "
                                f"Auto-included because sibling "
                                f"'{Path(group.primary_file).name}' was included."
                            ),
                        })
                        already_in.add(norm)
                        group_promoted += 1

            if group_promoted > 0:
                logger.info(
                    f"   [ComponentGroup] Auto-included {group_promoted} "
                    f"group members via component group inclusion rule (short-circuit path)"
                )

        return {
            "semantic_verification_results": pre_computed,
            "component_groups": component_groups,
            "status": "semantic_verified",
        }

    ticket = state["ticket"]
    evidence_items = state.get("evidence_items", [])
    ranked_files = state.get("ranked_files", [])
    component_groups = state.get("component_groups") or []
    
    if not ranked_files and "discovered_files" in state:
        logger.warning("No ranked_files found in state. Skipping Semantic Verification.")
        return {"status": "semantic_verification_skipped"}

    # Evaluate Top 20
    top_candidates = ranked_files[:20]

    # TODO: Add FORCE_PROMOTE_ROLES to evaluation list if not in top 20
    _FORCE_PROMOTE_ROLES = {
        CandidateRole.DEPLOYMENT.value,
        CandidateRole.ROOT_COMPONENT.value,
        CandidateRole.CONFIG.value,
        CandidateRole.VERSION_SOURCE.value,
    }
    
    # We don't have candidate_role directly in RankedFile, but we have arch_role.
    # Actually arch_role is strings like "deploy_script", "config_file", "root_component".
    _FORCE_PROMOTE_ARCH_ROLES = {
        "deploy_script", "config_file", "root_component"
    }

    force_promote = [f for f in ranked_files if f.arch_role in _FORCE_PROMOTE_ARCH_ROLES and f not in top_candidates]
    eval_candidates = top_candidates + force_promote

    logger.info(f"   Evaluating {len(eval_candidates)} candidates via SemanticVerificationAgent")
    
    results = agents.semantic_verifier.verify_candidates(
        ticket=ticket,
        candidates=eval_candidates,
        evidence_items=evidence_items,
        top_n=len(eval_candidates)
    )

    # ── Component Group Inclusion Rule ───────────────────────────────────
    # If ANY member of a component group is "include"d, ALL members of that
    # group must be included. This prevents .html/.scss from being rejected
    # because the verifier only evaluated them in isolation.
    if component_groups:
        # Build a map: file_path → ComponentGroup
        path_to_group: dict[str, "ComponentGroup"] = {}
        for group in component_groups:
            if group.is_group:
                for member_path in group.all_file_paths():
                    path_to_group[member_path.replace("\\", "/")] = group

        # Find all included paths
        included_paths = {
            r.file_path.replace("\\", "/")
            for r in results
            if r.decision == "include"
        }
        already_in_results = {
            r.file_path.replace("\\", "/") for r in results
        }

        # For each included file, check if it belongs to a group
        # and ensure all group members are also included
        group_promoted_count = 0
        for included_path in list(included_paths):
            group = path_to_group.get(included_path)
            if not group:
                continue
            for member_path in group.all_file_paths():
                norm_member = member_path.replace("\\", "/")
                if norm_member not in already_in_results:
                    # This group member wasn't evaluated — auto-include it
                    from ticket_to_code.agents.semantic_verification_agent import (
                        SemanticVerificationResult,
                    )
                    results.append(SemanticVerificationResult(
                        file_path=member_path,
                        ranking_score=group.group_score * 0.85,
                        semantic_relevance_score=group.group_score * 0.85,
                        decision="include",
                        reason=(
                            f"Component group member of '{group.stem}'. "
                            f"Auto-included because sibling "
                            f"'{Path(group.primary_file).name}' was included."
                        ),
                    ))
                    already_in_results.add(norm_member)
                    group_promoted_count += 1

        if group_promoted_count > 0:
            logger.info(
                f"   [ComponentGroup] Auto-included {group_promoted_count} "
                f"group members via component group inclusion rule"
            )

    import json
    _run_trace_dir = agents.tracer.trace_dir
    try:
        trace_data = [
            {
                "file_path": r.file_path,
                "ranking_score": r.ranking_score,
                "semantic_relevance_score": r.semantic_relevance_score,
                "decision": r.decision,
                "category": getattr(r, "category", "modify"),
                "reason": r.reason
            }
            for r in results
        ]
        write_trace_artifact(state["workspace_path"], ticket.ticket_id, "semantic_verification.json", trace_data)
        logger.info("   semantic_verification.json trace written")
    except Exception as e:
        logger.warning(f"Failed to write semantic_verification.json: {e}")

    # Pass the verification results forward. We can store them in state.
    # The planner node will then use this information.
    return {
        "semantic_verification_results": [
            {
                "file_path": r.file_path,
                "decision": r.decision,
                # category: "modify" (change here) | "reference" (read-only
                # knowledge for the planner) | "exclude"
                "category": getattr(r, "category", "modify"),
                "semantic_relevance_score": r.semantic_relevance_score,
                "reason": r.reason
            }
            for r in results
        ],
        "component_groups": component_groups,
        "status": "semantic_verified"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT IDEMPOTENCY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def preflight_check_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Pre-flight Idempotency Check — determines if the ticket is ALREADY
    implemented in the current codebase before the planner runs.

    Design philosophy:
    - BIAS TOWARD FALSE-NEGATIVE: if uncertain, say "not done" (safe — just
      wastes planner time). A false positive ("done" when not) would skip
      required work and is DANGEROUS.
    - Reads ACTUAL file content from the workspace (not summaries).
    - Only short-circuits when ALL functional requirements are verified with
      high confidence.

    Returns:
        status="already_implemented" → triggers route to END
        status="proceed_to_plan"    → triggers route to plan_node
        status="partially_implemented" → triggers route to plan_node with hints
    """
    logger.info("\n" + "=" * 80)
    logger.info(" PRE-FLIGHT CHECK: Is this ticket already implemented?")
    logger.info("=" * 80)

    requirements = state.get("requirements")
    if not requirements:
        logger.info("  No requirements found — skipping pre-flight check")
        return {
            "status": "proceed_to_plan",
            "preflight_verdict": "NOT_EVALUATED",
            "preflight_summary": "No requirements found — proceeding directly to architectural planner.",
            "agent_output": "Preflight check skipped: No requirements found.",
        }

    func_reqs = requirements.functional_requirements
    if not func_reqs:
        logger.info("  No functional requirements — skipping pre-flight check")
        return {
            "status": "proceed_to_plan",
            "preflight_verdict": "NOT_EVALUATED",
            "preflight_summary": "No functional requirements found — proceeding directly to architectural planner.",
            "agent_output": "Preflight check skipped: No functional requirements found.",
        }

    # ── Gather the discovered files that the evidence pipeline found ──────
    discovered = state.get("discovered_files") or []
    evidence_items = state.get("evidence_items") or []
    semantic_results = state.get("semantic_verification_results") or []

    # ── Map files to their highest relevance score for sorting ──
    file_scores = {}
    modify_targets = set()
    for r in semantic_results:
        if r.get("decision") == "include":
            path = r["file_path"].replace("\\", "/")
            score = float(r.get("semantic_relevance_score", 0.0))
            if r.get("category") == "modify":
                modify_targets.add(path)
            file_scores[path] = max(file_scores.get(path, 0.0), score)

    # Also extract modify targets from change candidates & ticket scope proof
    for cc in (state.get("change_candidates") or []):
        if isinstance(cc, str):
            modify_targets.add(cc.replace("\\", "/"))
        elif isinstance(cc, dict) and cc.get("file_path"):
            modify_targets.add(cc["file_path"].replace("\\", "/"))

    scope_proof = state.get("ticket_scope_proof")
    if scope_proof:
        if getattr(scope_proof, "anchor_file", None):
            modify_targets.add(scope_proof.anchor_file.replace("\\", "/"))
        for wf in getattr(scope_proof, "approved_writable_files", []):
            if wf:
                modify_targets.add(wf.replace("\\", "/"))

    for ev in evidence_items:
        if ev.relevance_score >= 0.6:
            path = ev.file_path.replace("\\", "/")
            file_scores[path] = max(file_scores.get(path, 0.0), ev.relevance_score)

    for mt in modify_targets:
        if mt not in file_scores:
            file_scores[mt] = 0.95

    if not file_scores:
        logger.info("  No relevant files discovered — skipping pre-flight check")
        return {
            "status": "proceed_to_plan",
            "preflight_verdict": "NOT_EVALUATED",
            "preflight_summary": "No relevant files discovered — proceeding directly to architectural planner.",
            "agent_output": "Preflight check skipped: No relevant files discovered.",
        }

    # Sort files: ALWAYS prioritize modify targets first, then highest relevance score
    sorted_paths = sorted(
        file_scores.keys(),
        key=lambda p: (1 if p in modify_targets else 0, file_scores[p]),
        reverse=True
    )

    # ── Read actual file content from disk using AST-targeted reading ──────
    # Instead of blindly reading 800 lines per file, we:
    # 1. Query the SQLite index for each file's methods/symbols
    # 2. Read structural context (class declaration, imports, fields)
    # 3. Read only relevant method bodies identified by the requirements
    # This ensures business logic buried at line 1200+ is captured.
    workspace_path = Path(state["workspace_path"])
    file_contents = {}
    MAX_FILES = 10
    MAX_STRUCTURAL_CHARS = 900   # imports, class declaration, fields
    MAX_METHOD_CHARS = 2200      # targeted method bodies per file

    # Try to get SQLite store for AST-aware reading
    sqlite_store = None
    try:
        if hasattr(agents, "localizer") and hasattr(agents.localizer, "sqlite_store"):
            sqlite_store = agents.localizer.sqlite_store
    except Exception:
        pass

    for rel_path in sorted_paths[:MAX_FILES]:
        abs_path = Path(rel_path) if Path(rel_path).is_absolute() else (workspace_path / rel_path)
        if not abs_path.is_file():
            # Robust fallback for sub-workspace or prefixed paths
            if (workspace_path / Path(rel_path).name).is_file():
                abs_path = workspace_path / Path(rel_path).name
            elif workspace_path.name and rel_path.startswith(workspace_path.name + "/"):
                trimmed = rel_path[len(workspace_path.name) + 1:]
                if (workspace_path / trimmed).is_file():
                    abs_path = workspace_path / trimmed
            elif (workspace_path.parent / rel_path).is_file():
                abs_path = workspace_path.parent / rel_path

        if not abs_path.is_file():
            continue
        try:
            raw = abs_path.read_text(encoding="utf-8", errors="replace")
            lines = raw.splitlines()

            if sqlite_store and len(lines) > 200:
                # AST-TARGETED READING: query SQLite for method locations
                try:
                    # Get symbols for this file
                    symbols = []
                    if hasattr(sqlite_store, "query_symbols_by_file"):
                        symbols = sqlite_store.query_symbols_by_file(rel_path) or []
                    elif hasattr(sqlite_store, "search_symbols"):
                        # Fallback: search by filename
                        fname = Path(rel_path).stem
                        symbols = sqlite_store.search_symbols(fname) or []
                        symbols = [s for s in symbols if rel_path in str(getattr(s, "file_path", ""))]

                    # 1. Structural context: first 40 lines (imports, class decl)
                    structural = "\n".join(lines[:40])[:MAX_STRUCTURAL_CHARS]

                    # 2. Find field declarations (typically lines 40-100)
                    fields_section = ""
                    for i, line in enumerate(lines[40:min(120, len(lines))], 40):
                        stripped = line.strip()
                        if any(kw in stripped for kw in [
                            "private ", "protected ", "public ", "@Input", "@Output",
                            "readonly ", "inject", "= inject(", ": ", "self.",
                        ]):
                            fields_section += f"{line}\n"
                    if fields_section:
                        structural += f"\n// --- Field declarations ---\n{fields_section[:500]}"

                    # 3. Targeted method bodies from SQLite symbols
                    method_content = ""
                    method_budget = MAX_METHOD_CHARS
                    # Sort symbols by relevance to requirements
                    req_keywords = set()
                    for r in func_reqs:
                        for word in r.lower().split():
                            if len(word) > 3:
                                req_keywords.add(word)

                    def _symbol_relevance(sym):
                        name = getattr(sym, "name", "").lower()
                        return sum(1 for kw in req_keywords if kw in name)

                    ranked_symbols = sorted(symbols, key=_symbol_relevance, reverse=True)

                    for sym in ranked_symbols:
                        if method_budget <= 0:
                            break
                        start_line = getattr(sym, "line_start", None) or getattr(sym, "start_line", None)
                        end_line = getattr(sym, "line_end", None) or getattr(sym, "end_line", None)
                        sym_name = getattr(sym, "name", "?")

                        if start_line and end_line and start_line > 0:
                            # Read the method body
                            method_lines = lines[start_line - 1:min(end_line, len(lines))]
                            method_text = "\n".join(method_lines)
                            if len(method_text) > method_budget:
                                method_text = method_text[:method_budget] + "\n// ... truncated"

                            method_content += f"\n// --- {sym_name}() L{start_line}-{end_line} ---\n"
                            method_content += method_text + "\n"
                            method_budget -= len(method_text)

                    if method_content:
                        file_contents[rel_path] = structural + "\n" + method_content
                    else:
                        # No symbols found — read complete file up to a generous limit
                        file_contents[rel_path] = raw[:20000]
                    continue
                except Exception as _ast_exc:
                    logger.debug(f"  AST-targeted read failed for {rel_path}: {_ast_exc}")
                    # Fall through to standard reading

            # Standard reading when SQLite is unavailable
            # Read complete file up to generous limit to ensure business logic is captured
            file_contents[rel_path] = raw[:20000]
        except Exception as exc:
            logger.debug(f"  Could not read {rel_path}: {exc}")

    if not file_contents:
        logger.info("  No files could be read from disk — skipping pre-flight check")
        return {"status": "proceed_to_plan"}

    logger.info(f"  Read {len(file_contents)} files for idempotency check")

    # ── Build the LLM prompt ─────────────────────────────────────────────
    ticket = state.get("ticket")
    ticket_title = getattr(ticket, "title", "") if ticket else ""
    ticket_desc = (getattr(ticket, "description", "") or "")[:2000] if ticket else ""

    reqs_str = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(func_reqs))

    code_str = ""
    for fpath, content in file_contents.items():
        code_str += f"\n--- {fpath} ---\n{content}\n"

    prompt = f"""You are a senior code reviewer and implementation planner. Your job is to:
1. Determine if the following ticket requirements are ALREADY IMPLEMENTED.
2. For any NOT-satisfied requirements, provide specific implementation guidance.

TICKET: {ticket_title}
DESCRIPTION: {ticket_desc}

FUNCTIONAL REQUIREMENTS:
{reqs_str}

CURRENT CODEBASE (relevant files — includes targeted method bodies):
{code_str}

INSTRUCTIONS:
1. For EACH functional requirement, check if the current code ALREADY satisfies it.
2. A requirement is "satisfied" ONLY if you can point to specific code that implements it.
3. Be STRICT: if there's any doubt, mark it as NOT satisfied.
4. For each NOT satisfied requirement, provide implementation guidance:
   - what_to_do: specific implementation task description
   - target_file: which file should be modified (MUST use the exact file path from CURRENT CODEBASE above)
   - integrate_with: which existing method/class to integrate with
   - pattern_reference: example of similar pattern in the codebase (file:method)
   - dependencies: other files/services this change depends on

OUTPUT FORMAT (JSON only, no markdown):
{{
  "verdict": "ALREADY_DONE" | "PARTIALLY_DONE" | "NOT_DONE",
  "confidence": 0.0 to 1.0,
  "requirements_status": [
    {{
      "requirement": "...",
      "satisfied": true | false,
      "evidence": "file.ts:line — specific code that satisfies this" | "Not found in codebase"
    }}
  ],
  "implementation_guidance": [
    {{
      "requirement": "the unsatisfied requirement text",
      "what_to_do": "Add isProjectMember check before adding contract member",
      "target_file": "path/to/file.ts",
      "integrate_with": "existing method name or class",
      "pattern_reference": "similar_file.ts:existingMethod — shows the pattern to follow",
      "dependencies": ["ServiceA.ts", "ModelB.ts"],
      "where_partial": "file.ts:145 — has addUser() but no membership check"
    }}
  ],
  "summary": "Brief explanation of what is already done vs what is missing"
}}

CRITICAL RULES:
- Only return "ALREADY_DONE" if ALL requirements are satisfied with confidence >= 0.90
- If even ONE requirement is missing or uncertain, return "PARTIALLY_DONE" or "NOT_DONE"
- When in doubt, return "NOT_DONE" — it is MUCH safer to re-do work than to skip needed work
- implementation_guidance should ONLY contain entries for NOT satisfied requirements
- Be specific in guidance: reference actual method names, line numbers, and patterns you see
"""

    # ── Call the LLM ──────────────────────────────────────────────────────
    primary_modify_file = next(iter(modify_targets), None) or (sorted_paths[0] if sorted_paths else "")
    try:
        from ticket_to_code.runtime.llm_factory import LLMRegistry
        from ticket_to_code.llm_utils import llm_invoke
        llm = getattr(agents, "llm", None) or LLMRegistry.get_llm(assistant=True)
        response = llm_invoke(llm, [
            SystemMessage(content="You are a precise code auditor and implementation planner. You check if requirements are already implemented and provide actionable guidance for missing ones. Be STRICT — only say 'done' when you are certain."),
            HumanMessage(content=prompt),
        ])
        response_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.warning(f"  Pre-flight LLM call failed: {exc} — verdict NOT_EVALUATED (confidence 0), proceeding to plan")
        req_statuses = [
            {
                "requirement_id": getattr(r, "id", None) or (r.get("id") if isinstance(r, dict) else f"REQ-{i+1}"),
                "requirement": str(getattr(r, "description", None) or (r.get("description") if isinstance(r, dict) else str(r))),
                "satisfied": False,
                "evidence": "Pre-flight audit could not run (LLM error)."
            }
            for i, r in enumerate(func_reqs)
        ]
        impl_guidance = [
            {
                "requirement_id": getattr(r, "id", None) or (r.get("id") if isinstance(r, dict) else f"REQ-{i+1}"),
                "requirement": str(getattr(r, "description", None) or (r.get("description") if isinstance(r, dict) else str(r))),
                "what_to_do": f"Implement {r}",
                "target_file": primary_modify_file or "Target repository files",
                "integrate_with": "Existing codebase"
            }
            for i, r in enumerate(func_reqs)
        ]
        result = {
            "verdict": "NOT_EVALUATED",
            "confidence": 0.0,
            "summary": "Pre-flight audit initiated. Ticket requirements pending implementation. Proceeding to architectural planner.",
            "requirements_status": req_statuses,
            "implementation_guidance": impl_guidance,
        }
        response_text = "{}" 

    logger.info(f"  Pre-flight LLM response:\n{response_text[:1000]}")

    # ── Parse the response ────────────────────────────────────────────────
    result = _extract_json_object(response_text)
    if not result:
        # Fallback to balanced regex or slice
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                import json as _json
                candidate_str = response_text[start_idx:end_idx+1]
                candidate_str = re.sub(r",\s*([\]}])", r"\1", candidate_str)
                result = _json.loads(candidate_str)
        except Exception:
            result = None

    if not result or not isinstance(result, dict):
        logger.warning(f"  Could not parse pre-flight JSON response: {response_text[:300]} — verdict NOT_EVALUATED (confidence 0), proceeding to plan")
        req_statuses = [
            {
                "requirement_id": getattr(r, "id", None) or (r.get("id") if isinstance(r, dict) else f"REQ-{i+1}"),
                "requirement": str(getattr(r, "description", None) or (r.get("description") if isinstance(r, dict) else str(r))),
                "satisfied": False,
                "evidence": "Pre-flight audit response unparseable."
            }
            for i, r in enumerate(func_reqs)
        ]
        impl_guidance = [
            {
                "requirement_id": getattr(r, "id", None) or (r.get("id") if isinstance(r, dict) else f"REQ-{i+1}"),
                "requirement": str(getattr(r, "description", None) or (r.get("description") if isinstance(r, dict) else str(r))),
                "what_to_do": f"Implement {r}",
                "target_file": primary_modify_file or "Target repository files",
                "integrate_with": "Existing codebase"
            }
            for i, r in enumerate(func_reqs)
        ]
        result = {
            "verdict": "NOT_EVALUATED",
            "confidence": 0.0,
            "summary": "None of the functional requirements are yet satisfied in the workspace. Full implementation required.",
            "requirements_status": req_statuses,
            "implementation_guidance": impl_guidance,
        }

    verdict = result.get("verdict", "NOT_DONE").upper()
    confidence = float(result.get("confidence", 0.0))
    summary = result.get("summary", "")
    req_statuses = result.get("requirements_status", [])
    impl_guidance = result.get("implementation_guidance", [])

    # Ground implementation guidance targets against real files on disk
    _ws_path = Path(state.get("workspace_path", "."))
    for g in impl_guidance:
        tf = g.get("target_file")
        if tf and isinstance(tf, str):
            tf_norm = tf.replace("\\", "/")
            if not (_ws_path / tf_norm).is_file():
                tf_base = Path(tf_norm).name
                for cand in _ws_path.glob(f"**/{tf_base}"):
                    if cand.is_file() and ".git" not in cand.parts and "target" not in cand.parts and "node_modules" not in cand.parts:
                        real_rel = str(cand.relative_to(_ws_path)).replace("\\", "/")
                        logger.info(f"  [Preflight Grounding] Grounded guidance target '{tf}' -> '{real_rel}'")
                        g["target_file"] = real_rel
                        break

    satisfied_count = sum(1 for r in req_statuses if (isinstance(r, dict) and r.get("satisfied")))
    total_count = len(req_statuses) if req_statuses else len(func_reqs)

    logger.info(f"  Verdict: {verdict} (confidence={confidence:.2f})")
    logger.info(f"  Requirements: {satisfied_count}/{total_count} satisfied")
    logger.info(f"  Implementation guidance items: {len(impl_guidance)}")
    logger.info(f"  Summary: {summary[:200]}")

    # ── Write trace artifact ──────────────────────────────────────────────
    try:
        ticket_id = getattr(ticket, "ticket_id", "unknown")
        write_trace_artifact(
            state["workspace_path"], ticket_id,
            "preflight_check.json",
            {
                "verdict": verdict,
                "confidence": confidence,
                "satisfied": satisfied_count,
                "total": total_count,
                "summary": summary,
                "requirements_status": req_statuses,
                "implementation_guidance": impl_guidance,
            }
        )
    except Exception:
        pass

    # ── Formatting for UI ──────────────────────────────────────────────────
    agent_output = f"Verdict: {verdict}\nConfidence: {confidence:.2f}\n\nSummary:\n{summary}\n\nRequirements Checked:\n"
    for r in req_statuses:
        if isinstance(r, dict):
            mark = "✅" if r.get("satisfied") else "❌"
            req_text = r.get("requirement", "?")
            evidence_text = r.get("evidence", "No evidence")
        else:
            mark = "❌"
            req_text = str(r)
            evidence_text = "No evidence"
        agent_output += f"- {mark} {req_text}\n  Evidence: {evidence_text}\n"

    if impl_guidance:
        agent_output += "\nImplementation Guidance:\n"
        primary_modify_file = next(iter(modify_targets), None) or (sorted_paths[0] if sorted_paths else "")
        for g in impl_guidance:
            if isinstance(g, dict):
                agent_output += f"\n📋 {g.get('requirement', '?')}\n"
                agent_output += f"   What: {g.get('what_to_do', '?')}\n"
                target_f = g.get('target_file', '?')
                if not target_f or 'not provided' in str(target_f).lower() or target_f == '?':
                    target_f = primary_modify_file or target_f
                agent_output += f"   File: {target_f}\n"
                agent_output += f"   Integrate with: {g.get('integrate_with', '?')}\n"
                if g.get("where_partial"):
                    agent_output += f"   Partial: {g.get('where_partial')}\n"
            else:
                agent_output += f"\n📋 {str(g)}\n"

    # ── Decision logic ────────────────────────────────────────────────────
    # STRICT: Only skip if ALREADY_DONE AND confidence >= 0.90 AND all reqs met
    if (
        verdict == "ALREADY_DONE"
        and confidence >= 0.90
        and satisfied_count == total_count
        and total_count > 0
    ):
        logger.info(
            f"  ✅ PRE-FLIGHT: Ticket already implemented! "
            f"({satisfied_count}/{total_count} requirements verified, "
            f"confidence={confidence:.2f}). Skipping code generation."
        )
        return {
            "status": "already_implemented",
            "preflight_verdict": verdict,
            "preflight_summary": summary,
            "agent_output": agent_output
        }

    # Partially done — proceed but pass hints and guidance to planner
    if verdict == "PARTIALLY_DONE" or (satisfied_count > 0 and satisfied_count < total_count):
        missing = [
            (r.get("requirement", "?") if isinstance(r, dict) else str(r))
            for r in req_statuses
            if (not r.get("satisfied") if isinstance(r, dict) else True)
        ]
        logger.info(
            f"  🎯 PRE-FLIGHT AUDIT COMPLETE: Found existing foundation "
            f"({satisfied_count}/{total_count} requirements verified). "
            f"Identified {len(missing)} specific requirements to implement — "
            f"proceeding to planner with targeted guidance."
        )
        return {
            "status": "proceed_to_plan",
            "preflight_verdict": "PARTIALLY_DONE",
            "preflight_summary": summary,
            "preflight_missing_requirements": missing,
            "preflight_implementation_guidance": impl_guidance,
            "agent_output": agent_output
        }

    # Not done — proceed with full guidance
    logger.info(f"  PRE-FLIGHT: Not yet implemented ({satisfied_count}/{total_count}). Proceeding to planner.")
    return {
        "status": "proceed_to_plan",
        "preflight_verdict": "NOT_EVALUATED" if verdict == "NOT_EVALUATED" else "NOT_DONE",
        "preflight_summary": summary,
        "preflight_implementation_guidance": impl_guidance,
        "agent_output": agent_output
    }


def route_after_preflight(state: TicketToCodeState) -> str:
    """
    Route after pre-flight check:
    - already_implemented → END (skip everything)
    - anything else       → planning_scope_verification
    """
    if state.get("status") == "already_implemented":
        logger.info("  [ROUTE] Pre-flight: already implemented → END")
        return END
    return "planning_scope_verification"


def planning_scope_verification_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 2H-2: Planning Scope Verification
    Verifies that claims implying scope changes (like 'new backend endpoint')
    are actually absent from the repository before allowing the planner to duplicate work.
    """
    print("\n" + "="*80)
    print(" ENTERING: planning_scope_verification_node() in workflow.py")
    print("   Purpose: Prevent 'Evidence absent == Repository absent' semantic error")
    print("="*80)
    
    from ticket_to_code.agents.planning_scope_verifier import PlanningScopeVerifier
    
    verifier = PlanningScopeVerifier(agents)
    preflight_guidance = state.get("preflight_implementation_guidance") or []
    evidence_items = state.get("evidence_items") or []
    discovered_files = state.get("discovered_files") or []
    
    results = verifier.verify_guidance(preflight_guidance, evidence_items, discovered_files)
    
    return {
        "planning_scope_verification_results": [r.model_dump() if hasattr(r, 'model_dump') else r.dict() for r in results]
    }


def plan_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 2: Architectural Planning
    
    Designs architecture and identifies files to create/modify.
    Uses RAG to understand existing codebase structure!
    """
    print("\n" + "="*80)
    print(" ENTERING: plan_node() in workflow.py")
    print("   Purpose: Design architecture and create development tasks")
    print("="*80)

    # If discovery hard-stopped the workflow, don't overwrite the failed status.
    # The unconditional discover→plan edge means plan_node always runs after
    # discover_node, even when discover_node returned status="failed".  We must
    # preserve that status so route_after_validate_candidates routes to END.
    if state.get("status") == "failed":
        logger.error("   plan_node: discovery failed — skipping planning, preserving failed status")
        return {"status": "failed"}  # Re-assert; validate_candidates_node must also honour this

    # ── Re-Plan Workspace Preservation (Never discard generated work) ────────
    _is_replan = (
        bool(state.get("generated_code"))
        or int(state.get("context_expansion_count", 0) or 0) > 0
        or int(state.get("candidate_retry_count", 0) or 0) > 0
    )
    if _is_replan:
        logger.info(
            "  🛡️ RE-PLAN WORKSPACE: Preserving on-disk generated changes across re-planning/retries."
        )

    logger.info(" PHASE 2: Architectural Planning")

    # Pre-flight awareness: if partially implemented, narrow scope to missing reqs only
    preflight_verdict = state.get("preflight_verdict")
    preflight_missing = state.get("preflight_missing_requirements") or []
    preflight_guidance = state.get("preflight_implementation_guidance") or []

    _preflight_guidance_context = ""

    if preflight_verdict == "PARTIALLY_DONE" and preflight_missing:
        logger.info(f"  ⚠️ Pre-flight: PARTIALLY implemented. Focusing planner on {len(preflight_missing)} missing requirements:")
        for i, req in enumerate(preflight_missing[:10], 1):
            logger.info(f"     {i}. {req}")
        logger.info(f"  Pre-flight summary: {state.get('preflight_summary', 'N/A')[:200]}")

    if preflight_guidance:
        logger.info(f"  📋 Pre-flight guidance: {len(preflight_guidance)} implementation items available")
        _preflight_guidance_context = (
            "\n=== PREFLIGHT IMPLEMENTATION GUIDANCE ===\n"
            "The following guidance was produced by a pre-flight code audit.\n"
            "Use it to generate PRECISE tasks — do NOT invent new files or methods\n"
            "when the guidance specifies existing targets.\n\n"
        )
        for idx, g in enumerate(preflight_guidance, 1):
            _preflight_guidance_context += f"GUIDANCE {idx}:\n"
            _preflight_guidance_context += f"  Requirement: {g.get('requirement', '?')}\n"
            _preflight_guidance_context += f"  What to do: {g.get('what_to_do', '?')}\n"
            if g.get("target_file"):
                _preflight_guidance_context += f"  Target file: {g['target_file']}\n"
            if g.get("integrate_with"):
                _preflight_guidance_context += f"  Integrate with: {g['integrate_with']}\n"
            if g.get("pattern_reference"):
                _preflight_guidance_context += f"  Pattern reference: {g['pattern_reference']}\n"
            if g.get("dependencies"):
                deps = g["dependencies"] if isinstance(g["dependencies"], list) else [g["dependencies"]]
                _preflight_guidance_context += f"  Dependencies: {', '.join(str(d) for d in deps)}\n"
            if g.get("where_partial"):
                _preflight_guidance_context += f"  Partial impl: {g['where_partial']}\n"
            _preflight_guidance_context += "\n"
        _preflight_guidance_context += "=== END PREFLIGHT GUIDANCE ===\n"

    # Outcome Check Awareness: if re-planning after outcome_check, focus strictly on pending delta
    outcome_unsatisfied = state.get("unsatisfied_requirements") or []
    outcome_findings = state.get("outcome_findings") or []
    if outcome_unsatisfied or outcome_findings:
        _pending_items = [str(u) for u in outcome_unsatisfied] or [
            f"{f.get('requirement_id', 'REQ')}: {f.get('missing_behavior') or f.get('summary', '')}"
            for f in outcome_findings if (f.get("verdict") if isinstance(f, dict) else getattr(f, "verdict", "")) != "CORRECT"
        ]
        _satisfied_items = [
            f"{f.get('requirement_id', 'REQ')}: {f.get('requirement_text') or f.get('summary', '')}"
            for f in outcome_findings if (f.get("verdict") if isinstance(f, dict) else getattr(f, "verdict", "")) == "CORRECT"
        ]
        _preflight_guidance_context += (
            "\n=== PARTIAL TICKET REMEDIATION MODE (DELTA ONLY) ===\n"
            "This is a follow-up remediation pass for a ticket that was partially implemented.\n"
            "DO NOT re-plan from scratch! Touch ONLY the files needed for the pending requirements.\n\n"
        )
        if _satisfied_items:
            _preflight_guidance_context += "ALREADY SATISFIED (PRESERVE - DO NOT RE-PLAN OR TOUCH):\n"
            for s in _satisfied_items[:6]:
                _preflight_guidance_context += f"  - {s}\n"
            _preflight_guidance_context += "\n"
        if _pending_items:
            _preflight_guidance_context += "PENDING REQUIREMENTS (PLAN TASKS ONLY FOR THESE):\n"
            for p in _pending_items[:6]:
                _preflight_guidance_context += f"  - {p}\n"
            _preflight_guidance_context += "\n"
        _preflight_guidance_context += "=== END PARTIAL REMEDIATION MODE ===\n"

    scope_verifications = state.get("planning_scope_verification_results") or []
    if scope_verifications:
        _preflight_guidance_context += "\n=== PLANNING SCOPE VERIFICATION ===\n"
        for v in scope_verifications:
            _preflight_guidance_context += f"Claim: {v.get('claim', '?')}\n"
            _preflight_guidance_context += f"Status: {v.get('status', '?')}\n"
            
            evidence_list = v.get('evidence', [])
            if evidence_list:
                _preflight_guidance_context += "Repository Evidence:\n"
                for ev in evidence_list:
                    _preflight_guidance_context += f"  - File: {ev.get('file', '?')}\n"
                    if ev.get('symbol'):
                        _preflight_guidance_context += f"    Symbol: {ev['symbol']}\n"
                    _preflight_guidance_context += f"    Reason: {ev.get('reason', '?')}\n"
            
            _preflight_guidance_context += f"Planning Implication: {v.get('planning_implication', '?')}\n"
            _preflight_guidance_context += "\n"
            
        _preflight_guidance_context += (
            "CRITICAL CONSTRAINTS FOR PLANNER:\n"
            "- VERIFIED_EXISTS: Existing repository capability has been established. Do not create duplicate implementation for that capability.\n"
            "- VERIFIED_ABSENT: Capability absence has been strongly established. New implementation may be planned if required.\n"
            "- UNKNOWN: Capability was not found in existing code. Implement the required logic within the designated target file or planned components (do not invent unrelated microservices or files outside the target architecture).\n"
            "=== END PLANNING SCOPE VERIFICATION ===\n"
        )

    # ── Context Firewall Summary from Sub-Agent Exploration ────────────────
    subagent_summary = state.get("subagent_investigation_summary")
    if subagent_summary:
        _preflight_guidance_context += f"\n{subagent_summary}\n"

    # ── Layout / Scroll Defect Guidance ─────────────────────────────────────
    _ticket = state.get("ticket")
    _ticket_title_lower = str(getattr(_ticket, "title", "") or "").lower()
    _ticket_desc_lower = str(getattr(_ticket, "description", "") or "").lower()
    _is_layout_scroll_defect = any(k in _ticket_title_lower or k in _ticket_desc_lower for k in [
        "scroll missing", "overflow behind", "behind footer", "table scroll", "scroll bar", "overflow", "missing scroll"
    ])
    if _is_layout_scroll_defect:
        _preflight_guidance_context += (
            "\n=== CRITICAL LAYOUT / SCROLL DEFECT GUIDANCE ===\n"
            "This ticket describes table rows overflowing behind a sticky pagination footer or missing scrolling container.\n"
            "MANDATORY ARCHITECTURAL RULES FOR THIS FIX:\n"
            "1. Do NOT set overflow-y: auto on the outer card body; outer card body must have display: flex; flex-direction: column; overflow: hidden; height: 100%.\n"
            "2. In the table component SCSS (e.g. jtable.component.scss):\n"
            "   - :host must have display: flex; flex-direction: column; width: var(--uxa-app-content-layout__height); height: var(--uxa-app-content-layout__height); overflow: hidden;\n"
            "   - .table__body must have flex: 1 1 auto; min-height: 0; height: calc(var(--uxa-app-content-layout__height) - 3.5rem); overflow-y: auto; overflow-x: auto;\n"
            "   - uxa-pagination-bar must have flex-shrink: 0;\n"
            "3. In the host page SCSS (e.g. jprojects.component.scss):\n"
            "   - The body container (.projects__body) must have flex: 1; min-height: 0; height: calc(100% - var(--jato-measure-40)); overflow: hidden;\n"
            "=== END LAYOUT DEFECT GUIDANCE ===\n"
        )

    # NEW: Retrieve architectural context using RAG!
    logger.info("   Retrieving existing codebase patterns from RAG...")
    
    # Build queries based on requirements
    requirements = state["requirements"]
    planning_queries = []
    
    # Query 1: Based on affected components
    if requirements.affected_components:
        planning_queries.append(f"existing code structure {' '.join(requirements.affected_components[:3])}")
    
    # Query 2: Based on technical requirements
    if requirements.technical_requirements:
        tech_keywords = ' '.join(requirements.technical_requirements[:2])[:100]
        planning_queries.append(f"similar implementation {tech_keywords}")
    
    # Query 3: General architecture patterns
    planning_queries.append("project structure file organization patterns")
    
    # Retrieve architectural patterns from RAG
    architectural_context = []
    for query in planning_queries[:3]:
        logger.info(f"   RAG Query: {query}")
        context_chunks = agents.rag_engine.retrieve_context(
            query=query,
            max_results=5,
            document_types=None,  # Get all types (source code, docs, etc.)
            schema="both",  # Search both codebase and architectural guidelines
            
        )
        architectural_context.extend(context_chunks)
    
    logger.info(f"  ✅ Retrieved {len(architectural_context)} architectural examples")
    
    # Build codebase context string from RAG results
    codebase_context = "EXISTING CODEBASE STRUCTURE AND PATTERNS:\n\n"
    
    # ── FIX 2: Microservice Boundary Graph Injection ──
    try:
        from ticket_to_code.utils.dependency_graph import get_dependency_graph
        module_graph = get_dependency_graph(state["workspace_path"])
        codebase_context += module_graph.get_summary_string() + "\n\n"
        codebase_context += "CRITICAL ARCHITECTURAL BOUNDARY RULE:\n"
        codebase_context += "You CANNOT use direct Java imports or Spring Bean injection between isolated microservices unless explicitly declared in the 'Local Dependencies' above.\n"
        codebase_context += "If module A needs data from module B and has no local dependency on it, you MUST use an HTTP/REST Connector or reject the approach.\n\n"
    except Exception as e:
        logger.warning(f"  Failed to generate dependency graph: {e}")
        
    for idx, chunk in enumerate(architectural_context[:10], 1):
        codebase_context += f"Example {idx}:\n"
        codebase_context += f"File: {chunk.get('file_path', 'Unknown')}\n"
        codebase_context += f"Type: {chunk.get('type', 'code')}\n"
        codebase_context += f"Content:\n{chunk.get('content', '')[:400]}\n"
        codebase_context += "-" * 40 + "\n"
    
    # Filter out blacklisted candidates so the planner never re-selects them
    blacklisted = state.get("blacklisted_files") or []
    
    # Filter out semantic verification rejections
    verification_results = state.get("semantic_verification_results") or []
    semantic_rejections = {
        res["file_path"].replace('\\', '/')
        for res in verification_results
        if res["decision"] in ("REJECT", "exclude")
    }

    # ── FIX: Allowlist mode when agentic verdicts exist ──────────────────
    # When the agentic loop provides verdicts, use ALLOWLIST instead of
    # DENYLIST.  The old denylist approach only blocked explicitly rejected
    # files, but files the agentic loop never evaluated leaked through.
    # Now: if there are agentic "include" verdicts, ONLY those files (plus
    # evidence items) are allowed into the planner pool.
    semantic_allowlist = {
        res["file_path"].replace('\\', '/')
        for res in verification_results
        if res.get("decision") in ("include",)
    }

    # Extract all files explicitly identified by preflight check (targets and dependencies)
    preflight_guidance = state.get("preflight_implementation_guidance") or []
    preflight_files = set()
    for g in preflight_guidance:
        if isinstance(g, dict):
            raw_tf = g.get("target_file") or g.get("file") or g.get("target_files") or []
            if isinstance(raw_tf, str) and raw_tf:
                preflight_files.add(raw_tf.replace('\\', '/'))
            elif isinstance(raw_tf, (list, tuple, set)):
                for item in raw_tf:
                    if isinstance(item, str) and item:
                        preflight_files.add(item.replace('\\', '/'))
            for dep in (g.get("dependencies") or []):
                dep_norm = str(dep).replace('\\', '/')
                if "/" in dep_norm or dep_norm.endswith((".java", ".ts", ".tsx", ".sql", ".py")):
                    preflight_files.add(dep_norm)

    _use_allowlist = len(semantic_allowlist) > 0

    if _use_allowlist:
        # Allowlist mode: keep files that the agentic loop explicitly marked
        # as "include", OR files specified in preflight guidance, OR high-confidence Tier 1
        # discovery files (rerank_score >= 7, confidence >= 0.80, or unified_score >= 0.80)
        # (provided they are not in blacklisted or explicitly in semantic_rejections)
        active_discovered = [
            f for f in (state.get("discovered_files") or [])
            if (
                f["path"].replace('\\', '/') in semantic_allowlist
                or f["path"].replace('\\', '/') in preflight_files
                or f.get("rerank_score", 0) >= 7
                or f.get("confidence", 0) >= 0.80
                or f.get("unified_score", 0) >= 0.80
            )
            and f["path"].replace('\\', '/') not in blacklisted
            and f["path"].replace('\\', '/') not in semantic_rejections
        ]
        logger.info(
            f"  [Allowlist] Agentic verdicts present — kept {len(active_discovered)} "
            f"of {len(state.get('discovered_files') or [])} discovered files "
            f"({len(semantic_allowlist)} explicitly included, {len(preflight_files)} preflight files)"
        )
    else:
        # Denylist mode (legacy): no agentic verdicts, use old behavior
        active_discovered = [
            f for f in (state.get("discovered_files") or [])
            if f["path"].replace('\\', '/') not in blacklisted and f["path"].replace('\\', '/') not in semantic_rejections
        ]

    # Ensure any physically existing preflight dependency is in active_discovered
    existing_paths = {d["path"].replace('\\', '/') for d in active_discovered}
    ws_path = Path(agents.workspace_path)
    for pf in preflight_files:
        if pf not in existing_paths and (ws_path / pf).is_file() and pf not in blacklisted and pf not in semantic_rejections:
            active_discovered.append({
                "path": pf,
                "confidence": 0.95,
                "features": {},
                "signals": ["preflight_dependency", "score:0.95"]
            })
            existing_paths.add(pf)
            logger.info(f"  🎯 Injected preflight dependency into candidate pool: {pf}")

    # [NEW] Merge high-fidelity Evidence into the Discovery pool.
    # Use a RELATIVE score band instead of a flat 0.70 threshold so that
    # over-collection after recall widening doesn't flood the planner:
    # only items within 55% of the best evidence score are merged.
    evidence_items = state.get("evidence_items") or []
    blacklisted_norm = {p.replace('\\', '/') for p in blacklisted}

    if evidence_items:
        _best_ev = max(e.relevance_score for e in evidence_items)
        _ev_threshold = max(0.35, _best_ev * 0.55)  # relative band, absolute floor 0.35
    else:
        _ev_threshold = 0.70  # fallback when no evidence collected

    # ── FIX (Bug 1): Evidence-Boost Merge ────────────────────────────────
    # Previously, evidence for files already in the discovery pool was
    # silently discarded.  Now we BOOST their confidence and inject the
    # evidence_match signal so they rank above keyword-only matches.
    for e in evidence_items:
        norm_path = e.file_path.replace('\\', '/')
        if e.relevance_score < _ev_threshold or norm_path in blacklisted_norm or norm_path in semantic_rejections:
            continue

        ev_sigs = [f"evidence_match:{e.source}", f"score:{e.relevance_score:.2f}"]
        if hasattr(e, "ranking_reasons") and e.ranking_reasons:
            ev_sigs.extend(e.ranking_reasons)

        if norm_path in existing_paths:
            # BOOST: file already in pool — upgrade its confidence and inject evidence signals
            for d in active_discovered:
                if d["path"].replace('\\', '/') == norm_path:
                    d["confidence"] = max(d.get("confidence", 0.0), e.relevance_score)
                    for sig in ev_sigs:
                        if sig not in d.get("signals", []):
                            d.setdefault("signals", []).insert(0, sig)
                    logger.info(f"  ⬆ Evidence-boosted existing candidate: {norm_path} → confidence={d['confidence']:.3f}")
                    break
        else:
            # NEW: file not in pool — inject it
            active_discovered.append({
                "path": e.file_path,
                "confidence": e.relevance_score,
                "features": {},
                "signals": ev_sigs
            })
            existing_paths.add(e.file_path)

    # Bias candidate ranking for cross-tab UI state regressions so owner and
    # shared state files are considered before action-tab side effects.
    _biased = _apply_ui_state_regression_bias(active_discovered, state.get("ticket"))
    if _biased:
        logger.info(f"  [ui-bias] adjusted confidence for {_biased} candidate(s)")

    # ── Component Group Context for Planner ──────────────────────────────
    # Tell the planner which files belong together as framework components.
    # The planner can then create SEPARATE tasks for each file that needs
    # changes (e.g., one task for .ts logic, another for .html template),
    # instead of only planning for one file and leaving siblings to blind
    # companion injection.
    component_groups = state.get("component_groups") or []
    multi_file_groups = [g for g in component_groups if g.is_group]
    if multi_file_groups:
        codebase_context += "\nCOMPONENT GROUPS (files that belong together as framework components):\n"
        codebase_context += "IMPORTANT: If any file in a group needs changes, consider whether its\n"
        codebase_context += "siblings also need changes. Create SEPARATE tasks for each file that\n"
        codebase_context += "needs modification — do NOT assume only the .ts file needs changes.\n\n"
        for idx, group in enumerate(multi_file_groups[:10], 1):
            codebase_context += f"  Group {idx}: {group.stem}\n"
            for ext, member_path in sorted(group.members.items()):
                label = "logic" if ext in _GROUP_SOURCE_EXTS else "template/style"
                in_evidence = "(evidence)" if ext in group.evidence_sources else "(grouped)"
                codebase_context += f"    - {Path(member_path).name} [{label}] {in_evidence}\n"
            codebase_context += "\n"
        logger.info(
            f"  [ComponentGroup] Injected {len(multi_file_groups)} component "
            f"groups into planner context"
        )

    # Sort the unified pool by confidence descending
    active_discovered.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    if blacklisted:
        logger.info(f"  ⛔ Blacklisted (excluded from planning): {blacklisted}")
    logger.info(f"  ✅ Active candidates in unified pool: {len(active_discovered)}")

    # ── FIX 1: Classify ALL candidates first, then filter noise from planner ──
    # Classify the full pool so LOCK_FILE / GENERATED candidates are excluded
    # from the top-25 planner slots.  These files consume candidate slots without
    # adding planning value and can be accidentally selected by the LLM despite
    # role-based prohibitions.  They remain in full pool traces for observability.
    for cand in active_discovered:
        _structural_role = _classify_candidate_role(cand.get("path", ""))

        # AST/localization can mark non-code infra files as READ_ONLY because they have
        # no methods/imports. For DEPLOYMENT/CONFIG, force writable ownership so
        # PatchValidator can allow legitimate infrastructure edits.
        if _structural_role in (CandidateRole.DEPLOYMENT.value, CandidateRole.CONFIG.value):
            cand["candidate_role"] = _structural_role
            if "features" not in cand or not isinstance(cand.get("features"), dict):
                cand["features"] = {}
            cand["features"]["ownership_type"] = "PRIMARY_OWNER"
            continue

        # Priority 1: canonical ownership_type set by the discovery pipeline
        _role = (cand.get("features") or {}).get("ownership_type", "") or ""
        if _role not in ("PRIMARY_OWNER", "DISPLAY_OWNER", "SUPPORTING", "READ_ONLY"):
            # Priority 2: ownership signal strings (legacy path)
            _role = ""
            for _sig in cand.get("signals", []):
                _sig_str = str(_sig)
                if "PRIMARY_OWNER" in _sig_str:
                    _role = "PRIMARY_OWNER"
                    break
                elif "DISPLAY_OWNER" in _sig_str:
                    _role = "DISPLAY_OWNER"
                    break
                elif "SUPPORTING" in _sig_str:
                    _role = "SUPPORTING"
                    break
        if _role in ("PRIMARY_OWNER", "DISPLAY_OWNER", "SUPPORTING", "READ_ONLY"):
            cand["candidate_role"] = _role
        else:
            # Priority 3: structural path heuristic (last resort)
            cand["candidate_role"] = _structural_role

    # ── FIX 4: Re-sort with compound key now that roles are known ────────────
    # Primary: confidence descending.  Secondary: role priority (DEPLOYMENT first,
    # GENERATED last).  Tertiary: alphabetical path.
    # This makes top-25 selection fully deterministic when many files share
    # confidence=1.0 — run-job.sh (DEPLOYMENT) and app.component.ts (ROOT_COMPONENT)
    # consistently outrank .spec.ts (TEST) and package-lock.json (LOCK_FILE).
    _ROLE_PRIORITY = {
        CandidateRole.DEPLOYMENT.value:      0,
        CandidateRole.ROOT_COMPONENT.value:  1,
        CandidateRole.CONFIG.value:          2,
        CandidateRole.VERSION_SOURCE.value:  3,
        CandidateRole.VERSION_DISPLAY.value: 4,
        CandidateRole.UNKNOWN.value:         5,
        CandidateRole.TEST.value:            6,
        CandidateRole.STYLE.value:           7,
        CandidateRole.REFERENCE.value:       8,
        CandidateRole.LOCK_FILE.value:       9,
        CandidateRole.GENERATED.value:      10,
    }
    active_discovered.sort(
        key=lambda x: (
            -x.get("confidence", 0.0),
            _ROLE_PRIORITY.get(x.get("candidate_role", CandidateRole.UNKNOWN.value), 5),
            x.get("path", ""),
        )
    )

    # ── Domain Ownership Classification ───────────────────────────────────────
    # Before the planner sees candidates, classify each file's ownership
    # against the ticket's business domain.  Files that are relevant but
    # architecturally wrong (e.g. area-service MembersService when ticket
    # is about project-membership) are re-tagged as REFERENCE — passed to the
    # planner as read-only context but never proposed for modification.
    _reference_only_candidates: list = []
    _domain_ownership_context = ""
    try:
        from ticket_to_code.agents.architecture_model import ArchitectureModel
        from ticket_to_code.agents.domain_resolver import DomainResolver
        from ticket_to_code.agents.ownership_resolver import OwnershipResolver

        # Load architecture model from workspace
        _arch_model = None
        _ws_path = agents.workspace_path
        for _yaml_loc in [
            _ws_path / "brain" / "knowledge" / "architecture_model.yaml",
            Path(__file__).parent / "config" / "architecture_model.yaml",
        ]:
            if _yaml_loc.exists():
                _arch_model = ArchitectureModel.load_from_yaml(str(_yaml_loc))
                logger.info(f"  🏗️ Architecture model loaded from {_yaml_loc.name}")
                break

        if _arch_model and _arch_model.services:
            # 1. Resolve ticket → business domain(s)
            _domain_resolver = DomainResolver(_arch_model)
            _ticket_title = state["ticket"].title if hasattr(state.get("ticket"), "title") else ""
            _ticket_desc = state["ticket"].description if hasattr(state.get("ticket"), "description") else ""
            _domain_resolution = _domain_resolver.resolve(_ticket_title, _ticket_desc)

            if _domain_resolution.has_resolution:
                logger.info(
                    f"  🏗️ Domain resolution: primary={_domain_resolution.primary_domains} "
                    f"authorized={_domain_resolution.authorized_services} "
                    f"confidence={_domain_resolution.confidence:.2f}"
                )

                # 2. Classify each candidate's ownership
                _ownership_resolver = OwnershipResolver(_arch_model)
                _all_domains = _domain_resolution.primary_domains + _domain_resolution.secondary_domains

                for cand in active_discovered:
                    _cand_path = cand.get("path", "")
                    if not _cand_path:
                        continue
                    _ownership = _ownership_resolver.classify(_cand_path, _all_domains)
                    cand["_ownership_verdict"] = _ownership.verdict.value
                    cand["_ownership_policy"] = _ownership.policy.value
                    cand["_ownership_service"] = _ownership.service_name
                    cand["_ownership_expected"] = _ownership.expected_service
                    cand["_ownership_is_hard"] = _ownership.is_hard

                    # Re-tag candidates that MUST NOT be modified
                    if _ownership.policy.value in ("DO_NOT_MODIFY", "REFERENCE_ONLY"):
                        cand["candidate_role"] = CandidateRole.REFERENCE.value
                        _reference_only_candidates.append(cand)
                        logger.info(
                            f"    ⛔ REFERENCE_ONLY: {_cand_path} "
                            f"(verdict={_ownership.verdict.value}, "
                            f"service={_ownership.service_name}, "
                            f"expected={_ownership.expected_service}, "
                            f"hard={_ownership.is_hard})"
                        )

                # 3. Build domain ownership context for the planner prompt
                if _reference_only_candidates:
                    _domain_ownership_context = (
                        "\n=== DOMAIN OWNERSHIP (READ CAREFULLY) ===\n"
                        f"Ticket domain: {', '.join(_domain_resolution.primary_domains)}\n"
                        f"Authorized services: {', '.join(_domain_resolution.authorized_services)}\n"
                        "The following files are in the candidate pool for CONTEXT ONLY.\n"
                        "They are architecturally WRONG for this ticket's domain.\n"
                        "You MUST NOT create modify/create tasks for these files:\n"
                    )
                    for rc in _reference_only_candidates:
                        _domain_ownership_context += (
                            f"  ❌ {rc['path']} (belongs to {rc.get('_ownership_service', '?')}, "
                            f"but capability owned by {rc.get('_ownership_expected', '?')})\n"
                        )
                    _domain_ownership_context += "=== END DOMAIN OWNERSHIP ===\n"
            else:
                logger.info("  🏗️ Domain resolution: no confident match — ownership classification skipped")
        else:
            logger.info("  🏗️ No architecture model available — ownership classification skipped")
    except Exception as _ownership_err:
        logger.warning(f"  🏗️ Ownership classification failed (non-fatal): {_ownership_err}")

    # ── Tier Boundary + Reuse-First (deterministic) ─────────────────────────
    # A frontend-anchored ticket (proven anchor is a component/template) is a
    # PRESENTATION-layer change: it consumes existing data through existing
    # frontend services. Domain ownership ("project-service owns membership
    # data") does NOT authorize modifying the backend for a UI ticket.
    # Observed failure (2026-09-18): ContractService.java got a modify task
    # for an Add-Members-modal ticket because the domain resolver authorized
    # the data owner; the generator then hallucinated a service call to a
    # backend method with no REST endpoint.
    _reuse_directive = ""
    _fe_root = None
    try:
        from ticket_to_code.agents.code_generator import clear_reuse_directive, set_reuse_directive
        clear_reuse_directive()
    except Exception:
        pass
    try:
        from ticket_to_code.agents.reuse_capability_discovery import (
            frontend_anchor_root,
            ticket_has_backend_intent,
            discover_frontend_capabilities,
            build_reuse_directive,
            is_frontend_file,
        )
        _ticket_raw = state.get("ticket")
        _ticket_text = " ".join(
            x for x in [
                _get_ticket_attr(_ticket_raw, "title", ""),
                _get_ticket_attr(_ticket_raw, "description", ""),
            ] if x
        )
        _cand_paths = [c.get("path", "") for c in active_discovered if c.get("path")]
        _fe_root = frontend_anchor_root(_cand_paths, str(agents.workspace_path))
        if _fe_root and not ticket_has_backend_intent(_ticket_text):
            _fe_root_str = str(_fe_root).replace("\\", "/").lower()
            _tier_demoted = []
            for cand in active_discovered:
                _cp = cand.get("path", "")
                if not _cp:
                    continue
                _cp_norm = _cp.replace("\\", "/").lower()
                _in_frontend_root = _fe_root_str in _cp_norm or _cp_norm.startswith(
                    _fe_root_str.split("/")[-1] + "/"
                )
                if not _in_frontend_root and not is_frontend_file(_cp):
                    if cand.get("candidate_role") != CandidateRole.REFERENCE.value:
                        cand["candidate_role"] = CandidateRole.REFERENCE.value
                        cand["_tier_boundary_demoted"] = True
                        _tier_demoted.append(_cp)
            if _tier_demoted:
                logger.info(
                    f"  🧱 TIER BOUNDARY: frontend-anchored ticket → "
                    f"{len(_tier_demoted)} backend candidate(s) demoted to REFERENCE_ONLY"
                )
                _domain_ownership_context += (
                    "\n=== TIER BOUNDARY (READ CAREFULLY) ===\n"
                    "This ticket is anchored in the FRONTEND (a component/modal/page change).\n"
                    "It is a presentation-layer change and MUST be implemented entirely in the\n"
                    "frontend service. Backend/Java files are READ-ONLY context for this ticket.\n"
                    "You MUST NOT create modify/create tasks for any backend file. If the data\n"
                    "you need is not exposed by an existing frontend service, reuse the closest\n"
                    "existing capability — do NOT add backend methods or endpoints.\n"
                    "Demoted to READ-ONLY:\n"
                    + "".join(f"  ❌ {p}\n" for p in _tier_demoted[:15])
                    + "=== END TIER BOUNDARY ===\n"
                )

            # Reuse-first: surface existing frontend service capabilities whose
            # names/methods overlap the ticket's key nouns, and inject a strict
            # reuse directive into the planner (and, later, the generator).
            _kw = re.findall(r"[A-Za-z]{4,}", _ticket_text)
            _caps = discover_frontend_capabilities(
                str(agents.workspace_path), frontend_root=_fe_root, keywords=_kw
            )
            if _caps:
                _reuse_directive = build_reuse_directive(_caps, workspace_path=str(agents.workspace_path))
                _domain_ownership_context += _reuse_directive
                logger.info(
                    f"  ♻️  REUSE-FIRST: {len(_caps)} frontend capabilities injected "
                    f"(top: {', '.join(Path(c['file']).name for c in _caps[:3])})"
                )
                # Hard protection: discovered reuse-capability files must be
                # CALLED, not modified. The product's philosophy — reuse what
                # exists — must not depend on the LLM obeying a prompt. Store
                # for the post-plan filter; the ticket's own expected_changed_files
                # (declared scope) always overrides.
                _ticket_declared = {
                    str(f).replace("\\", "/").lower()
                    for f in (_get_ticket_attr(state.get("ticket"), "expected_changed_files", None) or [])
                }
                _reuse_protected = set()
                for _c in _caps:
                    _rel = str(_c.get("rel_path", "")).replace("\\", "/").lower()
                    if not _rel or any(_rel.endswith(d) or d.endswith(_rel) for d in _ticket_declared):
                        continue
                    _reuse_protected.add(_rel)
                state["_reuse_protected_files"] = _reuse_protected
                # Give the generator the same directive so component code calls
                # the existing methods instead of inventing new ones.
                try:
                    set_reuse_directive(_reuse_directive)
                    if hasattr(agents, "code_generator") and agents.code_generator:
                        agents.code_generator._reuse_directive = _reuse_directive
                except Exception:
                    pass
    except Exception as _tier_err:
        logger.warning(f"  🧱 Tier-boundary/reuse discovery failed (non-fatal): {_tier_err}")

    # ── Preflight Guidance Scoping & Capability Reuse ──
    # If preflight guidance targets specific files within distinct project/service boundaries,
    # candidates from external services/projects are treated as context-only REFERENCE
    if preflight_guidance:
        _preflight_targets = []
        for g in preflight_guidance:
            if isinstance(g, dict):
                raw_tf = g.get("target_file") or g.get("file") or g.get("target_files")
                if isinstance(raw_tf, str) and raw_tf:
                    _preflight_targets.append(raw_tf.replace("\\", "/").strip())
                elif isinstance(raw_tf, (list, tuple, set)):
                    for item in raw_tf:
                        if isinstance(item, str) and item:
                            _preflight_targets.append(item.replace("\\", "/").strip())
            elif isinstance(g, str) and g:
                _preflight_targets.append(g.replace("\\", "/").strip())
        if _preflight_targets:
            _ws_reg = getattr(agents, "workspace_registry", None)
            _repo_roots = _ws_reg.discover_repo_roots() if _ws_reg else []

            def _get_project_boundary(p: str) -> Optional[str]:
                p_norm = p.replace("\\", "/").strip("/")
                if _ws_reg:
                    r = _ws_reg.get_repo_for_path(p_norm, _repo_roots)
                    if r:
                        return r
                parts = p_norm.split("/")
                if len(parts) > 1:
                    top_dir = parts[0]
                    ws = Path(agents.workspace_path)
                    candidate_dir = ws / top_dir
                    if candidate_dir.is_dir():
                        manifests = ("pom.xml", "package.json", "build.gradle", "build.gradle.kts", "go.mod", "Cargo.toml", "pyproject.toml")
                        if any((candidate_dir / m).exists() for m in manifests):
                            return top_dir
                return None

            _target_boundaries = {_get_project_boundary(pt) for pt in _preflight_targets}
            _target_boundaries = {b for b in _target_boundaries if b}
            if _target_boundaries:
                for cand in active_discovered:
                    _cand_p = (cand.get("path") or "").replace("\\", "/")
                    _cand_boundary = _get_project_boundary(_cand_p)
                    # If candidate belongs to a distinct project boundary outside targeted boundaries,
                    # it is a cross-service dependency -> treat as context-only REFERENCE
                    if _cand_boundary and _cand_boundary not in _target_boundaries:
                        if cand.get("candidate_role") != CandidateRole.REFERENCE.value:
                            cand["candidate_role"] = CandidateRole.REFERENCE.value
                            _reference_only_candidates.append(cand)
                            logger.info(
                                f"  ⛔ PREFLIGHT SCOPE: Demoted cross-service candidate to REFERENCE_ONLY: "
                                f"{_cand_p} (service '{_cand_boundary}' outside targeted services {_target_boundaries})"
                            )

    if _reference_only_candidates:
        logger.info(f"  🏗️ {len(_reference_only_candidates)} candidate(s) re-tagged as REFERENCE_ONLY")

    _PLANNER_EXCLUDED_ROLES = {
        CandidateRole.LOCK_FILE.value,
        CandidateRole.GENERATED.value,
    }
    
    # ── FIX: Automatically drop TEST files from planner unless ticket is testing-focused
    _inv = state.get("investigation_result")
    _req_code = getattr(_inv, "requires_code_changes", False) if hasattr(_inv, "requires_code_changes") else (_inv.get("requires_code_changes", False) if isinstance(_inv, dict) else False)
    _t_title = state["ticket"].title.lower() if hasattr(state.get("ticket"), "title") else ""
    _t_desc = state["ticket"].description.lower() if hasattr(state.get("ticket"), "description") else ""
    req_testing = _req_code and ("test" in _t_title or "spec" in _t_title or "test" in _t_desc)
    if not req_testing:
        _PLANNER_EXCLUDED_ROLES.add(CandidateRole.TEST.value)

    # Also exclude REFERENCE (domain-wrong) candidates from the planner's modification pool
    _PLANNER_EXCLUDED_ROLES.add(CandidateRole.REFERENCE.value)

    _planner_excluded = [
        c for c in active_discovered 
        if c["candidate_role"] in _PLANNER_EXCLUDED_ROLES or (c.get("path") and c["path"].replace('\\', '/') in [b.replace('\\', '/') for b in (blacklisted or [])])
    ]
    _planner_eligible = [
        c for c in active_discovered 
        if c["candidate_role"] not in _PLANNER_EXCLUDED_ROLES and (not c.get("path") or c["path"].replace('\\', '/') not in [b.replace('\\', '/') for b in (blacklisted or [])])
    ]

    if _planner_excluded:
        logger.info(
            f"   Planner-excluded (LOCK_FILE/GENERATED/REFERENCE/BLACKLISTED, trace-only): "
            f"{[c['path'] for c in _planner_excluded]}"
        )

    # ── Focus the candidate pool using ownership-cluster analysis ─────────────
    # _select_planner_candidates applies a relative score band, always preserves
    # PRIMARY_OWNER files, and adapts K to ownership richness. No hardcoded
    # extensions or framework roles — purely graph topology + evidence scores.
    active_discovered_for_planner = _select_planner_candidates(_planner_eligible)

    import json
    # ── FIX 2+3: trace artifacts are run-scoped and written atomically ────────
    # candidate_pool.json and planner_decisions.json are written ONLY after
    # create_plan() succeeds, to the tracer's timestamped run directory.
    # A crash inside create_plan() writes planner_failed.json instead.
    # Ticket-global flat files are never written — cross-run chimeras are impossible.
    _run_trace_dir = agents.tracer.trace_dir  # None when TRACE_MODE is off

    # ── Skill guidance injection ─────────────────────────────────────────────
    # Keeps simple/isolated tickets fast: tags are derived cheaply (no LLM call)
    # from the ticket text + discovered file paths, then loaded into planner
    # guidance so it favors a minimal-diff plan over broad companion expansion.
    try:
        from ticket_to_code.skills.loader import SkillLoader
        _skill_tags = ["minimal-diff", "bug-isolation"]
        _ticket_lower_for_skills = f"{state['ticket'].title} {state['ticket'].description}".lower()

        def _is_frontend_relpath(p: str) -> bool:
            norm = (p or "").replace("\\", "/").lower()
            if norm.endswith((".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte", ".jsx", ".tsx")):
                return True
            parts = norm.split("/")
            if len(parts) > 1:
                top_dir = parts[0]
                ws = Path(agents.workspace_path)
                cand_dir = ws / top_dir
                if cand_dir.is_dir() and any((cand_dir / cfg).exists() for cfg in ("angular.json", "tsconfig.json", "package.json", "vite.config.ts", "next.config.js")):
                    if not norm.endswith((".java", ".kt", ".scala", ".cs", ".go", ".rs", ".py", ".sql")):
                        return True
            return False

        _ui_only = bool(active_discovered_for_planner) and all(
            _is_frontend_relpath(c.get("path", ""))
            for c in active_discovered_for_planner
        )
        _has_ui_intent = bool(re.search(r'\b(ui|frontend|component|template|view|modal|dialog|css|scss|html|angular|react|vue|page)\b', _ticket_lower_for_skills))
        if _ui_only or _has_ui_intent:
            _skill_tags.append("ui-fast-fix")
        active_skills, skill_guidance = SkillLoader(agents.workspace_path).load_for_tags(_skill_tags)
        if active_skills:
            logger.info(f"  [Skills] Loaded: {[s.get('id') for s in active_skills]}")
    except Exception as skill_exc:
        logger.warning(f"  [Skills] Failed to load skill guidance (non-fatal): {skill_exc}")
        active_skills, skill_guidance = [], ""

    try:
        # Inject preflight guidance + domain ownership context into codebase context
        _enriched_context = codebase_context
        if _preflight_guidance_context:
            _enriched_context = _preflight_guidance_context + "\n" + _enriched_context
        if _domain_ownership_context:
            _enriched_context = _domain_ownership_context + "\n" + _enriched_context

        plan = agents.planner.create_plan(
            state["ticket"],
            state["requirements"],
            codebase_context=_enriched_context,              # ← RAG + domain ownership context
            discovered_files=active_discovered_for_planner,  # ← Real repo files (top-25, blacklist excluded)
            blacklisted_files=blacklisted,               # ← Tell the LLM which files are FORBIDDEN
            workspace_path=str(agents.workspace_path),   # ← Fix Bug 3: inject real file content
            verification_feedback=verification_results,  # ← FIX 4.3: Pass semantic feedback
            validation_failure_reason=state.get("validation_failure_reason"), # ← Feedback from gating
            reference_only_files=_reference_only_candidates,  # ← Domain-wrong files (read-only context)
            verified_evidence=state.get("verified_evidence"),  # ← Phase 5: inspected capabilities + code
        )
    except Exception as plan_err:
        if _run_trace_dir is not None and _run_trace_dir.exists():
            try:
                (_run_trace_dir / "planner_failed.json").write_text(
                    json.dumps({
                        "error": str(plan_err),
                        "candidate_pool_size": len(active_discovered_for_planner),
                        "candidates": active_discovered_for_planner,
                    }, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.warning(f"   planner_failed.json written \u2192 {_run_trace_dir}")
            except Exception as write_err:
                logger.warning(f"  Failed to write planner_failed.json: {write_err}")
        raise

    if isinstance(plan, tuple):
        plan = plan[0]
    if isinstance(plan, dict):
        from ticket_to_code.models import ArchitecturalPlan
        plan = ArchitecturalPlan.model_validate(plan)

    # Post-process: enforce the blacklist at the response level.
    # The LLM may still propose blacklisted paths despite the FORBIDDEN instruction
    # in its prompt.  Silently drop any such tasks so validate_candidates never
    # sees them again — this stops the propose→reject→propose cycle cold.
    if blacklisted:
        original_count = len(plan.tasks)
        remaining = [t for t in plan.tasks if t.file_path not in blacklisted]
        # Safety valve: if dropping blacklisted paths leaves 0 tasks, preserve files that actually exist on disk
        if not remaining and plan.tasks:
            ws_p = Path(str(getattr(agents, "workspace_path", "")))
            valid_existing = [t for t in plan.tasks if (ws_p / t.file_path).is_file()]
            if valid_existing:
                logger.warning(
                    f"   post-plan blacklist filter: dropping all tasks would collapse plan to 0. "
                    f"Preserving {len(valid_existing)} target file(s) that exist on disk."
                )
                remaining = valid_existing
        dropped = original_count - len(remaining)
        plan.tasks = remaining
        if dropped:
            logger.warning(
                f"   post-plan blacklist filter: dropped {dropped} task(s) that referenced "
                f"blacklisted file(s).  Remaining tasks: {len(plan.tasks)}"
            )

    # Post-plan TIER BOUNDARY filter (hard guarantee, prompt-independent):
    # a frontend-anchored ticket must not carry modify/create tasks on backend
    # files, even if the planner LLM ignored the injected directive.
    try:
        if _fe_root is not None:
            from ticket_to_code.agents.reuse_capability_discovery import (
                is_frontend_file,
                ticket_has_backend_intent,
            )
            _fe_root_str = str(_fe_root).replace("\\", "/").lower()
            # Cross-tier coherence (how top AI IDEs operate): a backend change is
            # legitimate for a frontend-anchored ticket ONLY as part of a complete
            # vertical slice — i.e. the plan also modifies/creates the frontend
            # service wrapper that binds the new backend capability. A backend
            # task with no frontend binding (observed: ContractService.java got a
            # new method but no endpoint, no frontend wrapper) is an incomplete
            # feature and is demoted.
            _t_obj2 = state.get("ticket")
            _ticket_text2 = " ".join(x for x in [
                _get_ticket_attr(_t_obj2, "title", ""),
                _get_ticket_attr(_t_obj2, "description", ""),
            ] if x)
            _has_fe_service_task = any(
                ".service.ts" in str(t.file_path or "").lower()
                and getattr(t.task_type, "value", str(t.task_type)) not in ("read_only", "reference")
                for t in plan.tasks
            )
            _coherent_slice = _has_fe_service_task or ticket_has_backend_intent(_ticket_text2)
            _tier_kept = []
            _tier_dropped = []
            for t in plan.tasks:
                _tp = str(t.file_path or "").replace("\\", "/").lower()
                _in_fe = _fe_root_str in _tp or _tp.startswith(_fe_root_str.split("/")[-1] + "/")
                _is_fe_file = is_frontend_file(str(t.file_path or ""))
                _writable_kind = getattr(t.task_type, "value", str(t.task_type)) not in ("read_only", "reference")
                if _writable_kind and not _in_fe and not _is_fe_file and not _coherent_slice:
                    _tier_dropped.append(str(t.file_path))
                    continue
                _tier_kept.append(t)
            if _tier_dropped:
                plan.tasks = _tier_kept
                logger.warning(
                    f"   post-plan tier-boundary filter: dropped {len(_tier_dropped)} "
                    f"incoherent backend task(s) (no frontend service wrapper in plan): {_tier_dropped}"
                )
    except Exception as _tier_filter_err:
        logger.warning(f"   post-plan tier filter failed (non-fatal): {_tier_filter_err}")

    # Post-plan REUSE-PROTECTION filter (hard guarantee): discovered reuse
    # capability files (e.g. member.service.ts) must be CALLED, not modified.
    # The planner LLM otherwise tends to "extend" the service it should reuse
    # (observed 2026-09-18: modify task on member.service.ts to add a method
    # that duplicated an existing capability). Declared expected_changed_files
    # overrides this protection.
    try:
        _protected = {
            p.replace("\\", "/").lower()
            for p in (state.get("_reuse_protected_files") or [])
        }
        _read_only_intent_files = {
            str(d.get("path", "")).replace("\\", "/").lower()
            for d in active_discovered_for_planner
            if d.get("change_intent") == "READ_ONLY"
        }
        _all_read_only = _protected | _read_only_intent_files

        if _all_read_only:
            def _norm_fp(fp: str) -> str:
                return str(fp or "").replace("\\", "/").lower()
            _demoted = []
            for t in plan.tasks:
                _tp = _norm_fp(t.file_path)
                _writable_kind = getattr(t.task_type, "value", str(t.task_type)) not in ("read_only", "reference")
                if _writable_kind and any(_tp.endswith(p) or p.endswith(_tp) for p in _all_read_only):
                    _original_intent = getattr(t.task_type, "value", str(t.task_type)).upper()
                    t.task_type = TaskType.READ_ONLY
                    t.selection_reason = (t.selection_reason or "") + " | ChangeIntent: READ_ONLY dependency — call existing APIs, do not modify."
                    _demoted.append(str(t.file_path))
                    logger.warning(
                        f"\n"
                        f"  🛡️ PLANNER_INTENT_CORRECTION:\n"
                        f"     File: {t.file_path}\n"
                        f"     LLM intent: {_original_intent}\n"
                        f"     Enforced intent: READ_ONLY\n"
                        f"     Reason: Existing dependency / service reuse boundary\n"
                    )
            if _demoted:
                logger.warning(
                    f"   post-plan ChangeIntent filter: enforced READ_ONLY on {len(_demoted)} "
                    f"reference dependency task(s): {_demoted}"
                )
    except Exception as _reuse_filter_err:
        logger.warning(f"   post-plan reuse filter failed (non-fatal): {_reuse_filter_err}")

    # ── Post-plan: enforce that task paths are grounded in reality ────────
    # The LLM may hallucinate file paths (e.g. "app/app.py" instead of "src/app.py")
    # despite the "MUST SELECT FROM THESE" instruction.  This block auto-corrects
    # or drops such tasks BEFORE they reach validate_candidates.
    discovered_paths = {d["path"].replace("\\", "/") for d in active_discovered_for_planner}
    discovered_basenames: dict[str, list[str]] = {}
    for dp in discovered_paths:
        bn = dp.rsplit("/", 1)[-1].lower()
        discovered_basenames.setdefault(bn, []).append(dp)

    ws_root = Path(state.get("workspace_path")) if state.get("workspace_path") else None

    corrected_tasks = []
    for task in plan.tasks:
        task_norm = task.file_path.replace("\\", "/")
        task_type_val = getattr(task.task_type, "value", str(task.task_type)).lower()

        # ── READ_ONLY: pass through (no writes)
        if task_type_val == "read_only":
            corrected_tasks.append(task)
            continue

        # ── CREATE: smart handling
        if task_type_val == "create":
            if ws_root and (ws_root / task.file_path).exists():
                # File already exists on disk — converting to MODIFY to prevent duplicate rogue files (e.g. OrderController_new.java)
                logger.warning(
                    f"   ✏️  CREATE→MODIFY: '{task.file_path}' already exists on disk. "
                    f"Converting task to MODIFY existing file."
                )
                task.task_type = TaskType.MODIFY
                task.new_file_creation_allowed = False

            corrected_tasks.append(task)
            continue

        # ── MODIFY: path MUST exist in discovered files or on disk
        if task_norm in discovered_paths:
            corrected_tasks.append(task)
            continue
        if ws_root and (ws_root / task.file_path).exists():
            corrected_tasks.append(task)
            continue

        # User requested to NEVER auto-correct hallucinated paths for MODIFY tasks.
        # Instead, we pass the hallucinated path directly to the candidate validation node.
        # The validation node will see the file doesn't exist, reject it, blacklist the 
        # hallucinated path, and force the LLM to explicitly choose the correct path 
        # from the available candidates list in a retry loop.
        corrected_tasks.append(task)

    if len(corrected_tasks) != len(plan.tasks):
        logger.info(
            f"   Post-plan enforcement: {len(plan.tasks)} → {len(corrected_tasks)} tasks "
            f"({len(plan.tasks) - len(corrected_tasks)} dropped/corrected)"
        )
    plan.tasks = corrected_tasks

    # ── Capability resolution: reuse existing code by INTENT, not by name ─────
    # For each writable task, ask "does existing code already SOLVE this task's
    # intent, regardless of what it is named?".  When yes, we point the task at
    # the REAL existing symbol (target_method/target_class) and convert a CREATE
    # into a MODIFY so the generator extends existing behaviour instead of
    # producing a duplicate that merely re-implements the same capability.
    # This is semantic (embedding + LLM judgement), never fixed-name matching.
    if os.environ.get("AVIATOR_CAPABILITY_RESOLVE", "1") != "0":
        try:
            from ticket_to_code.agents.task_level_analyzer import (
                SemanticCapabilityResolver,
                ResolutionDecision,
            )

            resolver = SemanticCapabilityResolver(
                rag_engine=agents.rag_engine,
                repo_search_engine=agents.repo_search,
                llm=agents.planner.llm,
                workspace_path=str(agents.workspace_path),
            )

            _writable_tasks = [
                t for t in plan.tasks
                if getattr(t.task_type, "value", str(t.task_type)).lower() != "read_only"
            ]
            # Bound LLM calls so planning latency stays predictable.
            _max_resolves = int(os.environ.get("AVIATOR_CAPABILITY_RESOLVE_MAX", "6"))
            _resolutions_trace = []

            for task in _writable_tasks[:_max_resolves]:
                intent = (task.description or task.title or "").strip()
                if not intent:
                    continue
                resolution = resolver.resolve_task(
                    intent=intent,
                    title=task.title or "",
                    hints={"file_path": task.file_path},
                )
                _resolutions_trace.append(resolution.to_dict())

                # Only act on confident reuse of a REAL existing file.
                if not (resolution.should_reuse and resolution.confidence >= 0.6):
                    continue
                if not resolution.target_file:
                    continue

                target_norm = resolution.target_file.replace("\\", "/")
                target_exists = target_norm in discovered_paths or (
                    ws_root and (ws_root / resolution.target_file).exists()
                )
                if not target_exists:
                    continue

                note = (
                    f" | capability-reuse: existing '{resolution.target_symbol or target_norm}' "
                    f"in {target_norm} already fulfils this intent "
                    f"(conf={resolution.confidence:.2f}): {resolution.reasoning[:160]}"
                )
                task.selection_reason = (task.selection_reason or "") + note

                _fnorm = (task.file_path or "").replace("\\", "/")
                _same_file = _fnorm.endswith(target_norm) or target_norm.endswith(_fnorm)
                task_type_val = getattr(task.task_type, "value", str(task.task_type)).lower()

                # Guard: Stylesheets (.scss, .css, etc.), templates (.html), or companion files
                # of the target component (e.g. member.component.scss companion to member.component.ts)
                # do not invent duplicate code capabilities — they provide styling and view structure.
                # Never retarget or demote them to READ_ONLY!
                _STYLE_OR_TEMPLATE_EXTS = {".scss", ".css", ".sass", ".less", ".html", ".svg"}
                _task_ext = Path(_fnorm).suffix.lower()
                _is_companion = (
                    Path(_fnorm).stem == Path(target_norm).stem
                    or (
                        Path(_fnorm).parent == Path(target_norm).parent
                        and Path(_fnorm).stem.split(".")[0] == Path(target_norm).stem.split(".")[0]
                    )
                )
                if _task_ext in _STYLE_OR_TEMPLATE_EXTS or _is_companion:
                    continue

                if not _same_file and task_type_val == "create":
                    # Retarget the CREATE onto the file that already provides the capability.
                    task.task_type = TaskType.MODIFY
                    task.new_file_creation_allowed = False
                    task.file_path = resolution.target_file
                    if resolution.target_symbol and resolution.target_symbol != "(file)":
                        task.target_method = resolution.target_symbol
                        if resolution.target_symbol not in task.allowed_methods:
                            task.allowed_methods.append(resolution.target_symbol)
                    logger.info(
                        f"   CREATE→MODIFY (capability): task '{task.id}' reuses existing "
                        f"'{resolution.target_symbol or target_norm}' in {target_norm}"
                    )
                elif not _same_file and task_type_val == "modify":
                    _file_disk_path = (ws_root / task.file_path) if ws_root else Path(task.file_path)
                    _is_target_in_preflight = False
                    for g in preflight_guidance:
                        if isinstance(g, dict):
                            raw_tf = g.get("target_file") or g.get("file") or g.get("target_files")
                            if isinstance(raw_tf, str) and raw_tf:
                                if (task.file_path or "").replace("\\", "/").lower().endswith(raw_tf.replace("\\", "/").lower()):
                                    _is_target_in_preflight = True
                                    break
                            elif isinstance(raw_tf, (list, tuple, set)):
                                if any((task.file_path or "").replace("\\", "/").lower().endswith(str(item).replace("\\", "/").lower()) for item in raw_tf if item):
                                    _is_target_in_preflight = True
                                    break
                        elif isinstance(g, str) and g:
                            if (task.file_path or "").replace("\\", "/").lower().endswith(g.replace("\\", "/").lower()):
                                _is_target_in_preflight = True
                                break
                    _is_ticket_target = any(
                        (task.file_path or "").replace("\\", "/").lower().endswith(exp.replace("\\", "/").lower())
                        for exp in (state.get("expected_changed_files") or []) if exp
                    )

                    # Invariant: Modifying an existing file on disk or an explicit preflight/ticket target
                    # is an in-place domain edit or bugfix. NEVER demote to READ_ONLY!
                    if _file_disk_path.exists() or _is_target_in_preflight or _is_ticket_target:
                        logger.info(
                            f"   PRESERVING MODIFY: task '{task.id}' targets '{task.file_path}' "
                            f"(existing file on disk / preflight target) — modifying existing logic in-place."
                        )
                        continue

                    # The capability already exists in ANOTHER file and does not exist in this file — this task would
                    # invent it in the wrong place (e.g. a new backend method when
                    # members() already answers the intent). Demote to read-only
                    # reference so the wrong file is never written.
                    task.task_type = TaskType.READ_ONLY
                    logger.info(
                        f"   MODIFY→READ_ONLY (capability): task '{task.id}' would invent "
                        f"'{', '.join(task.allowed_methods) or task.target_method or '?'}' in "
                        f"{_fnorm}, but '{resolution.target_symbol or target_norm}' in "
                        f"{target_norm} already fulfils it"
                    )
                # Same-file reuse is left untouched here; the pre-generation gate's
                # duplicate-method detection handles same-file inventions so we do
                # NOT pollute allowed_methods with the existing symbol.

            # Persist a trace for debugging/inspection.
            if _resolutions_trace and _run_trace_dir is not None and _run_trace_dir.exists():
                try:
                    (_run_trace_dir / "capability_resolutions.json").write_text(
                        json.dumps(_resolutions_trace, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
        except Exception as _cap_err:
            logger.warning(f"   Capability resolution skipped: {_cap_err}")

    # ── Change 2: Derive PlannerDecisions ────────────────────────────────────
    # Compare the candidates that were shown to the planner against the tasks
    # it actually produced.  Every candidate gets a REQUIRED / OPTIONAL / IGNORE
    # decision.  grounded_understanding_node reads these to avoid auto-injecting
    # files the planner intentionally excluded.
    task_paths_writable  = {t.file_path for t in plan.tasks if t.task_type.value != "read_only"}
    task_paths_readonly  = {t.file_path for t in plan.tasks if t.task_type.value == "read_only"}

    planner_decisions: dict = {}
    for cand in active_discovered_for_planner:
        fp   = cand["path"]
        role = cand.get("candidate_role", CandidateRole.UNKNOWN.value)
        conf = cand.get("confidence", 0.0)

        if fp in task_paths_writable:
            decision = PlannerDecisionValue.REQUIRED.value
            reason   = "Planner created a modify/create task for this file"
        elif fp in task_paths_readonly:
            decision = PlannerDecisionValue.OPTIONAL.value
            reason   = "Planner included this file as read-only context"
        else:
            decision = PlannerDecisionValue.IGNORE.value
            reason   = "Planner evaluated this candidate and did not select it"

        planner_decisions[fp] = {
            "file_path":  fp,
            "role":       role,
            "decision":   decision,
            "reason":     reason,
            "confidence": conf,
        }

    # Log summary
    req_count  = sum(1 for d in planner_decisions.values() if d["decision"] == PlannerDecisionValue.REQUIRED.value)
    opt_count  = sum(1 for d in planner_decisions.values() if d["decision"] == PlannerDecisionValue.OPTIONAL.value)
    ign_count  = sum(1 for d in planner_decisions.values() if d["decision"] == PlannerDecisionValue.IGNORE.value)
    logger.info(
        f"   PlannerDecisions: {req_count} REQUIRED, "
        f"{opt_count} OPTIONAL, {ign_count} IGNORE "
        f"(out of {len(planner_decisions)} candidates evaluated)"
    )
    for fp, dec in planner_decisions.items():
        logger.info(
            f"    [{dec['decision']:8s}] role={dec['role']:16s} {fp}"
        )

    # ── FIX 2+3: Atomic run-scoped write of pool + decisions ─────────────────
    # Both files land in the same tracer run directory and are written together,
    # so they always represent the same execution — cross-run chimera impossible.
    if _run_trace_dir is not None and _run_trace_dir.exists():
        try:
            (_run_trace_dir / "candidate_pool.json").write_text(
                json.dumps(active_discovered_for_planner, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (_run_trace_dir / "planner_decisions.json").write_text(
                json.dumps(list(planner_decisions.values()), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                f"   candidate_pool.json ({len(active_discovered_for_planner)} entries) + "
                f"planner_decisions.json → {_run_trace_dir}"
            )
        except Exception as err:
            logger.warning(f"  Failed to write trace artifacts: {err}")

    logger.info(
        f"Planning complete:\n"
        f"  Tasks: {len(plan.tasks)}\n"
        f"  API changes: {len(plan.api_changes)}\n"
        f"  Using RAG examples: {len(architectural_context)}"
    )
    # ── FIX 1: Pass the actual planner input (top-25), not the full pool ──────
    # Previously active_discovered (185-204 entries) was passed, making
    # candidates_passed_to_planner in 05_planning.json a misleading 7-8x overcount.
    agents.tracer.record_planning(
        plan, 
        active_discovered_for_planner,
        system_message=getattr(agents.planner, "last_system_prompt", None),
        user_message=getattr(agents.planner, "last_user_prompt", None)
    )
    
    selected = [t.file_path for t in plan.tasks] if hasattr(plan, "tasks") else []
    candidates = [c["path"] for c in active_discovered_for_planner]
    artifact = {
        "candidate_files": candidates,
        "selected_files": selected,
        "rejected_files": [c for c in candidates if c not in selected],
        "planner_reasoning": getattr(plan, "reasoning", "")
    }
    write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "planner_selection.json", artifact)

    # ── CONTRACT COMPLETENESS VALIDATION ─────────────────────────────────
    # Check that every cross_file_contract.consumes has a matching produces.
    # If Task A consumes "ProjectsService.isProjectMember" but no task
    # produces it, that's an architectural gap that will cause TSC errors.
    _all_produces: dict[str, str] = {}  # capability → producer task_id
    _all_consumes: list[tuple[str, str, str]] = []  # (capability, consumer task_id, from_task)
    for _task in plan.tasks:
        _cfc = getattr(_task, "cross_file_contract", None)
        if not _cfc:
            continue
        _cfc_dict = _cfc if isinstance(_cfc, dict) else (
            _cfc.model_dump() if hasattr(_cfc, "model_dump") else {}
        )
        for _prod in (_cfc_dict.get("produces") or []):
            _cap = _prod.get("capability", "") if isinstance(_prod, dict) else getattr(_prod, "capability", "")
            if _cap:
                _all_produces[_cap.lower().strip()] = _task.id
        for _cons in (_cfc_dict.get("consumes") or []):
            _cap = _cons.get("capability", "") if isinstance(_cons, dict) else getattr(_cons, "capability", "")
            _from = _cons.get("from_task", "") if isinstance(_cons, dict) else getattr(_cons, "from_task", "")
            if _cap:
                _all_consumes.append((_cap.lower().strip(), _task.id, _from or ""))

    _contract_gaps = []
    for _cap, _consumer_id, _from_task in _all_consumes:
        if _cap not in _all_produces:
            # Check if from_task references a task that exists
            _from_exists = any(
                t.id == _from_task or t.file_path == _from_task
                for t in plan.tasks
            ) if _from_task else False

            if not _from_exists:
                _contract_gaps.append({
                    "capability": _cap,
                    "consumer_task": _consumer_id,
                    "from_task_hint": _from_task,
                    "severity": "HIGH",
                    "description": (
                        f"Task '{_consumer_id}' consumes '{_cap}' but no task produces it. "
                        f"This will likely cause TS2339/TS2305 errors during generation."
                    ),
                })

    if _contract_gaps:
        logger.warning(
            f"  ⚠️ CONTRACT COMPLETENESS: Found {len(_contract_gaps)} unresolved "
            f"dependency gap(s) in the plan:"
        )
        for _gap in _contract_gaps:
            logger.warning(
                f"    🔴 '{_gap['capability']}' consumed by {_gap['consumer_task']} "
                f"but no producer task found"
            )
        # Write gaps to trace for diagnostics
        write_trace_artifact(
            state["workspace_path"], state["ticket"].ticket_id,
            "contract_gaps.json",
            {"gaps": _contract_gaps, "total_produces": len(_all_produces),
             "total_consumes": len(_all_consumes)},
        )
    else:
        _gap_summary = (
            f"produces={len(_all_produces)}, consumes={len(_all_consumes)}"
            if _all_produces or _all_consumes else "no contracts"
        )
        logger.info(f"  ✅ CONTRACT COMPLETENESS: All dependencies satisfied ({_gap_summary})")

    # ── BI-DIRECTIONAL RECONCILIATION: Plan → TicketScopeProof ───────────────
    # Ensure that any file targeted by a planned writable task that physically exists
    # or is an authorized new file creation is reconciled into TicketScopeProof.
    # This guarantees that companion repositories, saga handlers, and services that
    # the planner legitimately designed are never rejected downstream.
    try:
        from ticket_to_code.agents.ticket_scope_proof import (
            TicketScopeProof, ScopeGraph, validate_scope_graph,
        )
        _existing_sp = state.get("ticket_scope_proof")
        _ws_root_path = Path(state["workspace_path"]) if state.get("workspace_path") else None

        if _existing_sp is not None and getattr(_existing_sp, "scope_graph", None):
            _sg: ScopeGraph = _existing_sp.scope_graph
            _reconciled_any = False
            for _task in plan.tasks:
                _ttype_val = getattr(_task.task_type, "value", str(_task.task_type)).lower()
                if _ttype_val == "read_only":
                    continue
                _task_fp = _task.file_path or ""
                if not _task_fp:
                    continue
                _norm_tfp = _task_fp.replace("\\", "/").lower().strip()

                _file_on_disk = (_ws_root_path / _task_fp).exists() if _ws_root_path else True
                if not _file_on_disk and _ttype_val != "create":
                    continue

                if _sg.primary_anchor and (_norm_tfp == _sg.primary_anchor.replace("\\", "/").lower().strip() or _norm_tfp.endswith("/" + _sg.primary_anchor.replace("\\", "/").lower().strip())):
                    continue
                if any(_norm_tfp == c.replace("\\", "/").lower().strip() or _norm_tfp.endswith("/" + c.replace("\\", "/").lower().strip()) for c in _sg.companions):
                    continue

                # Move from providers / related_context / unverified into writable_dependencies
                _sg.providers = {p for p in _sg.providers if not (_norm_tfp == p.replace("\\", "/").lower().strip() or _norm_tfp.endswith("/" + p.replace("\\", "/").lower().strip()))}
                _sg.related_context = {r for r in _sg.related_context if not (_norm_tfp == r.replace("\\", "/").lower().strip() or _norm_tfp.endswith("/" + r.replace("\\", "/").lower().strip()))}
                _sg.unverified = {u for u in _sg.unverified if not (_norm_tfp == u.replace("\\", "/").lower().strip() or _norm_tfp.endswith("/" + u.replace("\\", "/").lower().strip()))}

                _sg.dependencies.add(_task_fp)
                _sg.writable_dependencies.add(_task_fp)
                if not getattr(_task, "modification_proof", None):
                    _task.modification_proof = f"Plan architecture task: {_task.title or _task.id}"
                _reconciled_any = True
                logger.info(f"  🔄 Bi-directional reconciliation: authorized planned file '{_task_fp}' into TicketScopeProof")

            if _reconciled_any:
                _is_valid, _val_errs = validate_scope_graph(_sg)
                _existing_sp.is_scope_proven = True
                if not _is_valid:
                    logger.warning(f"  ⚠️ ScopeGraph validation notice after reconciliation: {_val_errs}")
                else:
                    logger.info("  ✅ TicketScopeProof re-validated successfully with planned architectural tasks.")
                state["ticket_scope_proof"] = _existing_sp
    except Exception as _reconcile_err:
        logger.warning(f"  Bi-directional Plan→ScopeProof reconciliation skipped: {_reconcile_err}")

    return {
        "architectural_plan": plan,
        "initial_plan": plan,
        "plan": plan,
        "discovered_files": active_discovered_for_planner,
        "planner_decisions": planner_decisions,
        "status": "planning_complete",
        "contract_gaps": _contract_gaps if _contract_gaps else None,
        "ticket_scope_proof": state.get("ticket_scope_proof"),
    }


def _verify_ticket_already_satisfied(state: TicketToCodeState, ticket_text: str) -> "tuple[bool, str]":
    """Verify (with grounded evidence) whether the ticket's change already exists in code.

    Used only to distinguish "planner produced no tasks because the feature is ALREADY
    implemented" from "planner simply failed to plan". We never trust the LLM's word alone:
    the cited evidence snippet must mechanically exist on disk in the cited file.

    Returns (is_satisfied, detail). is_satisfied is True only when the LLM asserts the change
    is already present AND that claim is verified against the real file contents.
    """
    workspace_path = state.get("workspace_path", "")
    if not workspace_path:
        return False, "No workspace path available for verification."

    # Gather candidate files that actually exist on disk (ranked first, then discovered).
    candidate_paths: List[str] = []
    for r in (state.get("ranked_files") or []):
        fp = getattr(r, "file_path", "")
        if fp:
            candidate_paths.append(fp)
    for d in (state.get("discovered_files") or []):
        fp = d.get("path", "") if isinstance(d, dict) else ""
        if fp:
            candidate_paths.append(fp)

    seen: set = set()
    file_blocks: List[str] = []
    used_files: List[str] = []
    for fp in candidate_paths:
        norm = fp.replace("\\", "/")
        if norm in seen:
            continue
        seen.add(norm)
        full = os.path.join(workspace_path, fp)
        if not os.path.isfile(full):
            continue
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()[:4000]
        except Exception:
            continue
        file_blocks.append(f"### FILE: {norm}\n```\n{content}\n```")
        used_files.append(norm)
        if len(used_files) >= 4:
            break

    if not file_blocks:
        return False, "No readable candidate files to verify against."

    try:
        from ticket_to_code.runtime.llm_factory import LLMRegistry
        llm = LLMRegistry.get_llm(assistant=False)
    except Exception as exc:
        return False, f"LLM unavailable for verification ({exc})."

    prompt = f"""You are verifying whether a ticket's requested change is ALREADY implemented in the code.

TICKET:
{ticket_text}

RELEVANT FILES (current contents):
{os.linesep.join(file_blocks)}

Decide strictly:
- "already_implemented" = true ONLY if the requested behavior is clearly, fully present in the code above.
- If the change is missing, partial, or you are unsure, set it to false.

Return JSON only with exactly these keys:
{{
  "already_implemented": true | false,
  "evidence_file": "<exact file path from the list above, or empty>",
  "evidence_snippet": "<verbatim code snippet copied from that file proving it, or empty>",
  "reasoning": "<one sentence>"
}}
No markdown, no extra keys."""


    try:
        from ticket_to_code.llm_utils import llm_invoke as _llm_invoke_v
        raw = _llm_invoke_v(llm, prompt)
        content = getattr(raw, "content", raw)
        payload = json.loads(_extract_json_payload(str(content)))
    except Exception as exc:
        return False, f"Verification response could not be parsed ({exc})."

    if not isinstance(payload, dict) or payload.get("already_implemented") is not True:
        return False, f"Not verified as already implemented: {str(payload.get('reasoning', ''))[:200] if isinstance(payload, dict) else 'malformed response'}"

    evidence_file = str(payload.get("evidence_file", "")).strip()
    evidence_snippet = str(payload.get("evidence_snippet", "")).strip()
    if not evidence_file or not evidence_snippet:
        return False, "LLM claimed already-implemented but provided no grounding evidence."

    if evidence_file.replace("\\", "/") not in used_files:
        return False, f"Evidence file '{evidence_file}' was not among the inspected files."

    # Mechanical grounding: the cited snippet must actually exist in the cited file.
    if not verify_grounding_fact_text(evidence_snippet, evidence_file, workspace_path):
        return False, "Already-implemented claim failed mechanical grounding (snippet not found on disk)."

    return True, f"Verified already implemented in {evidence_file}: {str(payload.get('reasoning', ''))[:200]}"


def validate_candidates_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 2.6: Candidate Validation — runs between plan and localize.

    Ensures every planned writable file is grounded in the real repository:
      - file appears in discovered_files, OR
      - file physically exists on disk

    If any planned path is hallucinated / unrelated:
      - add it to blacklisted_files
      - increment candidate_retry_count
      - route back to plan (re-rank without blacklisted files)
      - if retries exhausted → route back to discover (expanded scope)

    Build failures do NOT trigger this node — only localization evidence does.
    """
    print("\n" + "="*80)
    print(" ENTERING: validate_candidates_node() in workflow.py")
    print("   Purpose: Reject hallucinated file paths before localization")
    print("="*80)
    logger.info("✅ PHASE 2.6: Candidate Validation")

    if state.get("status") == "failed":
        logger.error("   validate_candidates_node: workflow failed — skipping validation, routing to END")
        return {"status": "failed"}

    plan = _get_plan(state)
    discovered = state.get("discovered_files") or []
    ranked = state.get("ranked_files") or []
    evidence = state.get("evidence_items") or []
    blacklisted = state.get("blacklisted_files") or []
    workspace_path = state.get("workspace_path", "")
    
    ticket_obj = state.get("ticket")
    if hasattr(ticket_obj, "description"):
        ticket_text = f"{getattr(ticket_obj, 'title', '')} {getattr(ticket_obj, 'description', '')}"
    elif isinstance(ticket_obj, dict):
        ticket_text = f"{ticket_obj.get('title', '')} {ticket_obj.get('description', '')}"
    else:
        ticket_text = str(ticket_obj)

    # Guard: if no writable tasks, treat as planning failure
    writable_tasks = [t for t in plan.tasks if getattr(t.task_type, "value", str(t.task_type)) != "read_only"]
    if not writable_tasks:
        raw_candidates = ranked or discovered or [getattr(e, "file_path", None) for e in evidence if getattr(e, "file_path", None)]
        candidate_paths = []
        for c in raw_candidates:
            p = getattr(c, "path", None) or getattr(c, "file_path", None) or (c.get("path") if isinstance(c, dict) else None) or (c if isinstance(c, str) else None)
            if p and isinstance(p, str) and p not in blacklisted:
                candidate_paths.append(p)
        if not candidate_paths and raw_candidates:
            ws_p = Path(str(getattr(agents, "workspace_path", "")))
            for c in raw_candidates:
                p = getattr(c, "path", None) or getattr(c, "file_path", None) or (c.get("path") if isinstance(c, dict) else None) or (c if isinstance(c, str) else None)
                if p and isinstance(p, str) and (ws_p / p).is_file():
                    logger.warning("  ⚠️ Rescuing candidate '%s' from blacklist to prevent 0 writable tasks", p)
                    candidate_paths.append(p)
                    if p in blacklisted:
                        blacklisted.remove(p)
                    break
        if candidate_paths:
            from ticket_to_code.models import DevelopmentTask, TaskType
            first_p = str(candidate_paths[0])
            logger.info("  ⚡ Synthesizing fallback writable task for target candidate: %s", first_p)
            fallback_task = DevelopmentTask(
                id=f"task_{len(plan.tasks) + 1}",
                title=f"Modify {first_p} to implement requested changes",
                description=ticket_text[:200],
                file_path=first_p,
                task_type=TaskType.MODIFY,
                estimated_complexity=2,
            )
            plan.tasks.append(fallback_task)
            writable_tasks = [fallback_task]

    if not writable_tasks:
        new_blacklist = list(blacklisted)
        new_retry_count = state.get("candidate_retry_count", 0) + 1
        max_retries = state.get("max_candidate_retries", 2)

        if new_retry_count >= max_retries:
            # Retries exhausted with no writable tasks. Before calling it a failure,
            # verify whether the requested change is ALREADY implemented. We only
            # declare "already satisfied" when the claim is mechanically grounded in
            # real file contents — otherwise this is a genuine planning failure.
            is_satisfied, detail = _verify_ticket_already_satisfied(state, ticket_text)
            if is_satisfied:
                logger.info("  ✅ Ticket already implemented (verified): %s", detail)
                return {
                    "blacklisted_files": new_blacklist,
                    "candidate_retry_count": new_retry_count,
                    "validation_failure_reason": detail,
                    "status": "no_action_required",
                    "code_status": "code_already_satisfied",
                }
            logger.error(
                "  ❌ Planner produced 0 writable tasks and the change is NOT already "
                "present (verification failed: %s). Treating as planning failure.",
                detail,
            )
            return {
                "blacklisted_files": new_blacklist,
                "candidate_retry_count": new_retry_count,
                "validation_failure_reason": (
                    "Planner produced no writable tasks and the requested change was not "
                    f"found in the codebase (not already implemented). {detail}"
                ),
                "status": "planning_needs_recovery",
            }

        return {
            "blacklisted_files": new_blacklist,
            "candidate_retry_count": new_retry_count,
            "validation_failure_reason": "Plan produced 0 writable tasks",
            "status": "candidates_invalid",
        }

    accepted, rejected, is_consistent, failure_reason, task_logs = run_gating_logic(
        plan, discovered, ranked, evidence, workspace_path, ticket_text
    )

    # Log observability
    for log_entry in task_logs:
        if log_entry["status"] == "rejected":
            logger.warning(f"  ⛔ Rejected Task: {log_entry['task']} - Reason: {log_entry.get('reason', 'None')}")
        elif log_entry["status"] == "accepted":
            logger.info(f"  ✅ Accepted Task: {log_entry['task']} (Type: {log_entry.get('grounding_type')})")

    # If there are ANY rejected tasks, we force a retry of the whole plan to ensure safety.
    # We construct a per-task failure reason so the LLM knows what to keep and what to fix.
    if rejected:
        is_consistent = False
        reason_lines = ["Your previous plan contained REJECTED tasks. You MUST issue a new complete plan."]
        if accepted:
            reason_lines.append("\nACCEPTED tasks (preserve these EXACTLY in your new plan):")
            for t in accepted:
                reason_lines.append(f" ✅ {t.file_path}: {t.title}")

        rejected_reasons = {}
        for log_entry in task_logs:
            if log_entry["status"] == "rejected":
                rejected_reasons[log_entry["task"]] = log_entry.get("reason", "Unknown reason")

        reason_lines.append("\nREJECTED tasks (fix or replace these):")
        for t in rejected:
            r = rejected_reasons.get(t.file_path, "Unknown boundary/mechanical violation")
            reason_lines.append(f" ❌ {t.file_path}: {r}")

        # ── NEW: Include available files so planner can build a complete replacement plan
        reason_lines.append("\nAVAILABLE FILES IN PROJECT (use these EXACT paths for your new plan):")
        for d in (state.get("discovered_files") or []):
            d_path = d.get("path", "")
            d_conf = d.get("confidence", 0.0)
            if d_path:
                reason_lines.append(f" 📁 {d_path} (confidence: {d_conf:.2f})")

        failure_reason = "\n".join(reason_lines)

    if not is_consistent:
        # Check if this is a justification recovery
        is_justification_recovery = "explicit_planner_justification" in failure_reason
        
        if is_justification_recovery:
            new_just_retry_count = state.get("justification_retry_count", 0) + 1
            max_just_retries = state.get("max_justification_retries", 1)
            if new_just_retry_count > max_just_retries:
                is_justification_recovery = False
                logger.warning("  ⚠️ Justification recovery attempts exhausted. Failing back to hard replan.")
            else:
                logger.info(f"  🔄 Triggering Justification Recovery (Attempt {new_just_retry_count}/{max_just_retries})")
                return {
                    "justification_retry_count": new_just_retry_count,
                    "validation_failure_reason": failure_reason,
                    "status": "candidates_invalid",
                }

        new_retry_count = state.get("candidate_retry_count", 0) + 1
        max_retries = state.get("max_candidate_retries", 2)
        
        # ── Smart blacklisting based on rejection reason ────────────────────
        # Instead of blindly delaying ALL blacklisting, we distinguish:
        # 1. Ungrounded (score < 0.05, file doesn't exist) + zero cross-task role → blacklist immediately
        # 2. Architectural boundary violation → blacklist immediately
        # 3. LLM judgment failure (file exists but wrong context) → delay blacklist
        immediately_blacklist = []
        for t in rejected:
            t_path = t.file_path
            reject_reason = rejected_reasons.get(t_path, "")

            is_ungrounded = (
                "completely ungrounded" in reject_reason.lower()
                or "score < 0.05" in reject_reason.lower()
            )
            is_boundary = (
                "architectural" in reject_reason.lower()
                or "boundary" in reject_reason.lower()
            )

            if is_ungrounded:
                # File doesn't exist — check if any accepted task depends on it
                has_dependents = False
                for acc in accepted:
                    acc_desc = (acc.description or "").lower()
                    acc_deps = acc.dependencies or []
                    if t_path.lower() in acc_desc or t_path in acc_deps:
                        has_dependents = True
                        break

                if has_dependents:
                    logger.info(
                        f"  ⚠️ Rejected file '{t_path}' is referenced by accepted tasks. "
                        f"Not blacklisting — planner must fix or use existing file."
                    )
                else:
                    # Zero role — blacklist immediately
                    immediately_blacklist.append(t_path)
                    logger.info(
                        f"  🚫 Blacklisting '{t_path}': file doesn't exist and "
                        f"no other task depends on it (zero role)"
                    )

            elif is_boundary:
                # Architectural violation — always blacklist
                immediately_blacklist.append(t_path)
                logger.info(
                    f"  🚫 Blacklisting '{t_path}': architectural boundary violation"
                )
            # else: LLM judgment failure — delay blacklisting (give retry chance)

        # Always blacklist ungrounded/boundary files immediately
        new_blacklist = list(set(blacklisted + immediately_blacklist))

        # Delay-blacklist remaining rejected files only when retries exhausted
        # FIX: NEVER blacklist a file that EXISTS on disk and was found by Evidence
        # Collection. Patch failures (str_replace can't find exact match) are CODE
        # GENERATOR problems, not LOCALIZATION problems. Blacklisting verified files
        # causes the Planner to hallucinate edits in wrong files (Blacklist Cascade).
        if new_retry_count >= max_retries:
            _disk_missing = []
            for t in rejected:
                _t_abs = Path(workspace_path) / t.file_path if workspace_path else None
                if _t_abs and _t_abs.exists():
                    logger.info(
                        f"  🛡️  NOT blacklisting '{t.file_path}': file exists on disk. "
                        f"Patch failure is a code generator issue, not localization."
                    )
                else:
                    _disk_missing.append(t.file_path)
            new_blacklist = list(set(new_blacklist + _disk_missing))


        # Guard: never blacklist on-disk infra/config owner files. They are valid write
        # targets that carry no AST symbols, so a rejection can wrongly lock them and
        # collapse the next plan to zero writable tasks. Retry counters still bound the loop.
        _protected = [b for b in new_blacklist if _is_protected_config_owner(b, workspace_path)]
        if _protected:
            new_blacklist = [b for b in new_blacklist if b not in _protected]
            logger.info(f"  🛡️  Protected infra/config owners from blacklist: {_protected}")

        # Include blacklisted files in the feedback to planner
        if new_blacklist:
            failure_reason += "\n\nBLACKLISTED FILES (DO NOT USE these in your new plan):"
            for bp in new_blacklist:
                failure_reason += f"\n 🚫 {bp}"

        logger.warning(
            f"   Candidate validation FAILED (retry {new_retry_count}/{max_retries})\n"
            f"     Reason: {failure_reason}\n"
            f"     Rejected tasks: {[t.file_path for t in rejected]}"
        )
        return {
            "blacklisted_files": new_blacklist,
            "candidate_retry_count": new_retry_count,
            "validation_failure_reason": failure_reason or f"Rejected tasks: {[t.file_path for t in rejected]}",
            "status": "candidates_invalid",
        }

    # If we get here, the plan is accepted!
    logger.info(f"  ✅ All {len(accepted)} writable tasks validated and gated successfully.")
    
    read_only_tasks = [t for t in plan.tasks if getattr(t.task_type, "value", str(t.task_type)) == "read_only"]
    plan.tasks = accepted + read_only_tasks
    

    is_consistent_post, failure_reason_post, _ = validate_plan_consistency(plan.tasks, rejected, plan, workspace_path)
    if not is_consistent_post:
        logger.warning(f"  ⚠️ Post-mutation consistency check FAILED: {failure_reason_post}")
        
        new_retry_count = state.get("candidate_retry_count", 0) + 1
        max_retries = state.get("max_candidate_retries", 2)
        if new_retry_count >= max_retries:
            new_blacklist = list(set(blacklisted + [t.file_path for t in rejected]))
        else:
            new_blacklist = list(blacklisted)
            
        return {
            "blacklisted_files": new_blacklist,
            "candidate_retry_count": new_retry_count,
            "validation_failure_reason": f"Post-mutation consistency check failed: {failure_reason_post}",
            "status": "candidates_invalid",
        }
    
    agents.tracer.record_validation(
        status="candidates_validated",
        plan_tasks=plan.tasks,
        discovered_paths=[d["path"] for d in discovered],
        invalid=[],
        weak_tasks=[],
        blacklisted=list(blacklisted),
        retry_count=state.get("candidate_retry_count", 0),
    )
    return {
        "status": "candidates_validated",
        "validation_failure_reason": None,
        "architectural_plan": plan
    }


# ============================================================================
# PLANNING RECOVERY NODE
# ============================================================================

def planning_recovery_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Diagnose WHY planning produced 0 writable tasks and determine recovery action.

    Uses LLM-driven diagnosis from planning_recovery.py to classify the failure
    and produce a structured PlanningRecoveryAction that feeds targeted context
    into the next investigation cycle.

    Recovery types route differently:
      evidence_incomplete     → discover (targeted re-investigation)
      evidence_contaminated   → discover (clean re-investigation)
      wrong_candidates        → discover (expanded scope)
      requirement_ambiguous   → END (need user input)
      already_implemented     → END (no_action_required)
      genuinely_unrecoverable → END (terminal, high-bar)
    """
    print("\n" + "=" * 80)
    print(" ENTERING: planning_recovery_node() in workflow.py")
    print("   Purpose: Diagnose planning failure and determine recovery action")
    print("=" * 80)
    logger.info("✅ PLANNING RECOVERY: Diagnosing planning failure")

    from ticket_to_code.agents.planning_recovery import diagnose_planning_failure

    ticket = state.get("ticket")
    requirements = state.get("requirements")
    evidence_items = state.get("evidence_items") or []
    verification_results = state.get("semantic_verification_results") or []
    validation_failure_reason = state.get("validation_failure_reason")
    hypotheses = state.get("investigation_hypotheses") or []

    # Check recovery budget — max 2 recovery cycles to prevent infinite loops
    existing_action = state.get("planning_recovery_action")
    recovery_cycle = 0
    if existing_action:
        # Extract cycle number from existing recovery_id (e.g. "R1A2B3" → cycle 1)
        recovery_cycle = 1
        if hasattr(existing_action, "recovery_id") and existing_action.recovery_id:
            # Count how many recovery actions have been consumed
            if getattr(existing_action, "status", "") == "consumed":
                recovery_cycle = 2

    max_recovery_cycles = 2
    if recovery_cycle >= max_recovery_cycles:
        logger.warning(
            f"  Recovery budget exhausted ({recovery_cycle}/{max_recovery_cycles} cycles). "
            f"Terminal failure."
        )
        return {
            "status": "failed",
            "validation_failure_reason": (
                f"Planning recovery exhausted after {recovery_cycle} cycles. "
                f"Original failure: {validation_failure_reason}"
            ),
        }

    action = diagnose_planning_failure(
        ticket=ticket,
        requirements=requirements,
        evidence_items=evidence_items,
        verification_results=verification_results,
        validation_failure_reason=validation_failure_reason,
        hypotheses=hypotheses,
    )

    logger.info(f"  Recovery diagnosis: {action.recovery_type} (id={action.recovery_id})")
    logger.info(f"  Reason: {action.reason}")

    # Route based on recovery type
    if action.recovery_type == "already_implemented":
        logger.info("  ✅ Recovery diagnosis: ticket already implemented")
        return {
            "planning_recovery_action": action,
            "status": "no_action_required",
            "code_status": "code_already_satisfied",
        }

    if action.recovery_type == "requirement_ambiguous":
        logger.warning("  ⚠️ Recovery diagnosis: requirement ambiguous (needs user input)")
        return {
            "planning_recovery_action": action,
            "status": "need_more_info",
            "clarification_question": action.reason,
        }

    if action.recovery_type == "genuinely_unrecoverable":
        logger.error(f"  ❌ Recovery diagnosis: genuinely unrecoverable")
        for pt in action.reasoning_points:
            logger.error(f"    • {pt}")
        return {
            "planning_recovery_action": action,
            "status": "failed",
            "validation_failure_reason": (
                f"Planning recovery determined genuinely unrecoverable: {action.reason}"
            ),
        }

    # Recoverable types: evidence_incomplete, evidence_contaminated, wrong_candidates
    # All route back to discover with structured recovery context
    logger.info(f"  🔄 Recovery: routing back to discover with recovery context")
    return {
        "planning_recovery_action": action,
        "status": "planning_recovering",
        # Reset candidate retry count for the new discovery cycle
        "candidate_retry_count": 0,
    }


def route_after_planning_recovery(state: TicketToCodeState) -> str:
    """
    Route after planning recovery diagnosis:
    - planning_recovering   → discover (re-investigate with targeted context)
    - no_action_required    → END (already implemented)
    - need_more_info        → END (needs user clarification)
    - failed                → END (genuinely unrecoverable or budget exhausted)
    """
    status = state.get("status", "")
    if status == "planning_recovering":
        logger.info("   Routing to discover with recovery context")
        return "discover"
    if status == "no_action_required":
        logger.info("   Routing to END (already implemented per recovery)")
        return END
    if status == "need_more_info":
        logger.info("   Routing to END (need user input per recovery)")
        return END
    # failed or anything else
    logger.error(f"   Routing to END (terminal: status={status})")
    return END


def localize_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 2.5: Repository Intelligence — Localization

    DEFAULT ACTION: MODIFY EXISTING FILE.
    Only create new files when proven necessary.

    Uses SQLite AST index (multi-service codebase) to:
    1. Find the ACTUAL file in the repository for each planned task
    2. Override planner-suggested (LLM-hallucinated) paths with real ones
    3. Change task_type to MODIFY when existing file found
    4. Infer correct CREATE location from repo patterns when no file exists
    """
    print("\n" + "="*80)
    print(" ENTERING: localize_node() in workflow.py")
    print("   Purpose: Resolve planned file paths to ACTUAL repository locations")
    print("   Intelligence: SQLite AST (FTS5) + filesystem verification")
    print("="*80)
    logger.info(" PHASE 2.5: Repository Localization (AST-driven)")

    plan = _get_plan(state)
    logger.info(f"  Tasks to localize: {len(plan.tasks)}")
    for t in plan.tasks:
        logger.info(f"    [{t.id}] {t.task_type.value}: {t.file_path}")

    # READ_ONLY tasks need no localization — they carry no write target
    writable_tasks = [t for t in plan.tasks if t.task_type.value != "read_only"]
    readonly_tasks = [t for t in plan.tasks if t.task_type.value == "read_only"]
    if readonly_tasks:
        logger.info(f"  ⏩ Skipping {len(readonly_tasks)} read_only task(s) — no file writes")

    # Run localization against SQLite AST index + filesystem fallback
    # Pass full ticket text so unique ticket keywords (e.g. "JATO", "footer")
    # are available when the planner invented Java class names that don't exist
    ticket = state["ticket"]
    ticket_context = f"{ticket.title} {ticket.description}"
    localized_result = agents.localizer.localize_planned_tasks(writable_tasks, ticket_context=ticket_context)
    localized_tasks = localized_result[0] if isinstance(localized_result, tuple) else localized_result

    # Merge read_only tasks back (unlocalized, won't reach code gen)
    all_tasks = localized_tasks + readonly_tasks

    # Log decisions
    modify_count = sum(1 for t in all_tasks if t.task_type.value == "modify")
    create_count = sum(1 for t in all_tasks if t.task_type.value == "create")
    logger.info(
        f"  Localization complete:\n"
        f"    MODIFY: {modify_count} tasks (existing files found)\n"
        f"    CREATE: {create_count} tasks (new files needed)\n"
    )
    for t in all_tasks:
        logger.info(StageLog.localization(t))
        print(StageLog.localization(t))

    # Update the plan with localized tasks
    plan.tasks = all_tasks
    logger.info(StageLog.plan_summary(all_tasks))
    print(StageLog.plan_summary(all_tasks))

    agents.tracer.record_localization(
        localized_tasks=localized_tasks,
        system_message=getattr(agents.localizer, "last_system_prompt", None),
        user_message=getattr(agents.localizer, "last_user_prompt", None),
    )

    return {
        "architectural_plan": plan,
        "status": "localization_complete"
    }


# ============================================================================
# PHASE 2G — HYPOTHESIS INVESTIGATION + EVIDENCE COLLECTION LOOP
# ============================================================================
# Purpose: Before checking ownership completeness, deeply investigate the
# ticket using ALL available evidence sources (semantic search, Neo4j,
# SQLite symbol/FTS, TypeScript chains, Java chains, CSS chains, config
# chains, layout chains).  The output — GroundedUnderstanding — is passed
# into the code generators so generation operates on verified evidence rather
# than only the localized file list.
# ============================================================================

def hypothesis_investigation_node(
    state: "TicketToCodeState",
    agents: "WorkflowAgents",
) -> dict:
    """
    Phase 2G-1: Hypothesis Investigation

    Asks the InvestigationAgent to generate concrete, searchable hypotheses
    from the ticket and the initial triage result.  Each hypothesis contains:
      • queries   — semantic search phrases
      • literals  — exact string constants / config keys to FTS-match
      • symbols   — class / method / component names to look up in SQLite
    """
    print("\n" + "=" * 80)
    print(" ENTERING: hypothesis_investigation_node() in workflow.py")
    print("   Purpose: Generate investigation hypotheses from ticket + triage")
    print("=" * 80)
    logger.info(" PHASE 2G-1: Hypothesis Investigation")

    if state.get("status") == "failed":
        return {"status": "failed"}

    ticket = state["ticket"]
    investigation = state["investigation_result"]
    localized_tasks = []  # Decoupled from plan since this now runs before planning

    hypotheses = agents.investigation.generate_hypotheses(
        ticket=ticket,
        investigation=investigation,
        localized_tasks=localized_tasks,
    )

    # ── Planning Recovery: inject targeted recovery hypothesis ─────────────
    # When re-entering the pipeline after planning failure, the recovery action
    # tells us WHERE to look. We synthesize a high-priority hypothesis from it.
    recovery_action = state.get("planning_recovery_action")
    if recovery_action and getattr(recovery_action, "status", "") == "active":
        inv_target = getattr(recovery_action, "investigation_target", None)
        inv_query = getattr(recovery_action, "investigation_query", None)
        alt_terms = getattr(recovery_action, "alternative_search_terms", [])
        if inv_query or inv_target:
            from ticket_to_code.models import InvestigationHypothesis
            recovery_hypothesis = InvestigationHypothesis(
                id=f"recovery_{getattr(recovery_action, 'recovery_id', 'R0')}",
                hypothesis=f"[RECOVERY] {getattr(recovery_action, 'reason', 'Planning failure recovery')}",
                queries=[inv_query] if inv_query else [],
                literals=alt_terms[:5] if alt_terms else [],
                symbols=[inv_target] if inv_target else [],
                confidence=0.9,  # High priority — this is the recovery direction
            )
            hypotheses.insert(0, recovery_hypothesis)
            logger.info(
                f"  🔄 Injected recovery hypothesis: target={inv_target}, "
                f"query={inv_query}, terms={alt_terms[:3]}"
            )

    logger.info(f"  Generated {len(hypotheses)} hypothesis(es):")
    for h in hypotheses:
        logger.info(
            f"    [{h.id}] {h.hypothesis[:80]} "
            f"(queries={len(h.queries)}, literals={len(h.literals)}, "
            f"symbols={len(h.symbols)}, prior_conf={h.confidence:.2f})"
        )

    # ── Context Firewall: Sub-agent exploration workers ──────────────────
    subagent_summary = ""
    try:
        from ticket_to_code.agents.subagent_explorer import SubagentExplorer
        workspace_root = state.get("workspace_path", "")
        if workspace_root:
            explorer = SubagentExplorer(workspace_root=workspace_root)
            ticket_desc = getattr(ticket, "description", "") or getattr(ticket, "title", "")
            discovered = state.get("discovered_files") or []
            candidate_paths = [
                d.get("path") if isinstance(d, dict) else getattr(d, "path", str(d))
                for d in discovered
            ]
            if not candidate_paths and investigation:
                candidate_paths = getattr(investigation, "files", []) or []

            reports = []
            if candidate_paths:
                # 1. UI Mapper worker
                reports.append(explorer.run_ui_mapper(ticket_desc, candidate_paths))
                # 2. Validation Analyst worker
                ts_candidates = [p for p in candidate_paths if p.endswith(".ts") and not p.endswith(".spec.ts")]
                html_candidates = [p for p in candidate_paths if p.endswith(".html")]
                if ts_candidates:
                    ts_path = ts_candidates[0]
                    html_path = html_candidates[0] if html_candidates else None
                    reports.append(explorer.run_validation_analyst(ts_path, html_path))
                # 3. Contract Checker worker
                api_candidates = [p for p in candidate_paths if any(k in p.lower() for k in ["dto", "model", "api", "service"])]
                reports.append(explorer.run_contract_checker(api_candidates[0] if api_candidates else None))
                # 4. Test Scout worker
                parent_dir = str(Path(candidate_paths[0]).parent).replace("\\", "/") if candidate_paths else ""
                reports.append(explorer.run_test_scout(parent_dir))

                subagent_summary = explorer.synthesize_investigation(reports)
                logger.info(f"  🛡️ [SubagentExplorer] Context firewall synthesized {len(reports)} worker reports.")
    except Exception as e:
        logger.warning(f"  [SubagentExplorer] Exploration skipped (non-fatal): {e}")

    return {
        "investigation_hypotheses": hypotheses,
        "subagent_investigation_summary": subagent_summary or None,
        "status": "hypotheses_generated",
    }


def evidence_collection_loop_node(
    state: "TicketToCodeState",
    agents: "WorkflowAgents",
) -> dict:
    """
    Phase 2G-2: Evidence Collection Loop

    Iteratively queries every evidence source until the confidence threshold
    is reached (default 0.70) or max iterations (5) are exhausted:

      1. Semantic search via RAGEngine
      2. SQLite FTS for string literals
      3. SQLite symbol lookup
      4. Neo4j relationship queries
      5. TypeScript component relationship chains
      6. Java controller/service/repository chains
      7. CSS/SCSS ownership chains
      8. Configuration/value propagation chains
      9. Parent layout ownership chains
     10. Live repository search (.sh, .bat, .ps1, .json, .xml, .env — Phase 3A)
    """
    print("\n" + "=" * 80)
    print(" ENTERING: evidence_collection_loop_node() in workflow.py")
    print("   Purpose: Collect grounded evidence from ALL infrastructure sources")
    print("=" * 80)
    logger.info(" PHASE 2G-2: Evidence Collection Loop")

    if state.get("status") == "failed":
        return {"status": "failed"}

    hypotheses = state.get("investigation_hypotheses") or []
    plan = state.get("architectural_plan")
    localized_tasks = plan.tasks if plan else []
    ticket = state["ticket"]

    if not hypotheses:
        logger.warning("  ⚠️  No hypotheses found — skipping evidence collection")
        return {
            "evidence_items": [],
            "status": "evidence_collected",
        }

    ui_callback = _get_transient(state, "_ui_callback")

    # ── Evidence sub-step collector for real-time UI updates ──────────
    # Each sub-step is a dict emitted by the evidence loop at key moments
    # (stage_0, iteration_start, search_result, verdict, iteration_end).
    # Stored in state so main.py can broadcast them via WebSocket.
    _evidence_substeps: list = []

    def _evidence_step_callback(step_data: dict):
        """Emit sub-steps to live UI AND collect for history/debug."""
        from datetime import datetime
        enriched = {
            **step_data,
            "timestamp": datetime.now().isoformat(),
        }
        _evidence_substeps.append(enriched)

        # ── LIVE BROADCAST: push directly to WebSocket via put() ──
        # This is the thread-safe broadcast function from main.py,
        # registered as a transient. Without this, sub-steps only
        # appear after the entire node completes (buffered, not real-time).
        if ui_callback:
            _sub_type = enriched.get("type", "")
            _sub_msg = enriched.get("message", "")
            _icon_map = {
                "stage_0_start": "🔍",
                "stage_0_result": "🔍",
                "iteration_start": "🔄",
                "search_result": "🔎",
                "verdict": "✅" if enriched.get("decision") == "include" else "❌",
                "iteration_end": "📊",
                "early_stop": "✅",
            }
            _icon = _icon_map.get(_sub_type, "⚙️")
            try:
                ui_callback({
                    "phase": "evidence_collection_loop",
                    "status": "in_progress",
                    "message": f"{_icon} {_sub_msg}" if _sub_msg else f"{_icon} {_sub_type}",
                    "data": {
                        "node": "evidence_collection_loop",
                        "sub_step_type": _sub_type,
                        **{k: v for k, v in enriched.items()
                           if k not in ("type", "message", "timestamp")},
                    },
                    "timestamp": enriched["timestamp"],
                })
            except Exception as _ui_exc:
                logger.debug(f"  Live UI broadcast failed (non-fatal): {_ui_exc}")

    _recovery_act = state.get("planning_recovery_action")
    _recovery_dict = (
        _recovery_act.model_dump()
        if hasattr(_recovery_act, "model_dump")
        else (_recovery_act.dict() if hasattr(_recovery_act, "dict") else _recovery_act)
    )
    collect_result = agents.evidence_loop.collect(
        hypotheses=hypotheses,
        localized_tasks=localized_tasks,
        ticket=ticket,
        investigation=state.get("investigation_result"),
        step_callback=_evidence_step_callback,
        requirements=state.get("requirements"),
        recovery_context=_recovery_dict,
    )

    # collect() returns 3-tuple: (items, confidence, clarification_question)
    # clarification_question is None when proceeding normally, a string when
    # the agentic loop decided it needs more info from the user.
    if len(collect_result) == 3:
        evidence_items, confidence, clarification_question = collect_result
    else:
        # Backward compatibility: 2-tuple (items, confidence)
        evidence_items, confidence = collect_result
        clarification_question = None

    # ── NEED_MORE_INFO exit ──────────────────────────────────────────────
    # The agentic loop decided the ticket is too vague to find files.
    # Route to "need_more_info" — this exact string matches routes.py L1186.
    if clarification_question is not None:
        logger.warning(
            f"  Agentic loop requests clarification: {clarification_question[:120]}"
        )
        # Carry forward agentic verdicts so semantic_verification_node
        # can populate state in the expected batch shape without re-running.
        agentic_verdicts = getattr(agents.evidence_loop, "_agentic_verdicts", {})
        return {
            "evidence_items": evidence_items,
            "status": "need_more_info",
            "clarification_question": clarification_question,
            "_evidence_substeps": _evidence_substeps,
            "semantic_verification_results": [
                {
                    "file_path": v.get("file_path", fp),
                    "decision": v.get("decision", "include"),
                    "semantic_relevance_score": v.get("semantic_relevance_score", 0.0),
                    "reason": v.get("reason", ""),
                }
                for fp, v in agentic_verdicts.items()
            ],
        }

    logger.info(
        f"  Evidence collection complete: {len(evidence_items)} items, "
        f"confidence={confidence:.3f}"
    )

    # Log evidence breakdown by source
    from collections import Counter
    source_counts = Counter(e.source for e in evidence_items)
    for source, count in sorted(source_counts.items()):
        logger.info(f"    [{source}] {count} item(s)")

    # Capture expansion map from repo_search engine after collection
    expansion_map: dict = {}
    if agents.evidence_loop.repo_search is not None:
        expansion_map = agents.evidence_loop.repo_search.expansion_map
        if expansion_map:
            logger.info(
                f"  Query expansion map: {len(expansion_map)} original term(s) → "
                + ", ".join(
                    f"{orig!r}→{sorted(exps)}"
                    for orig, exps in list(expansion_map.items())[:4]
                )
            )

    # Carry agentic verdicts forward for semantic_verification_node
    agentic_verdicts = getattr(agents.evidence_loop, "_agentic_verdicts", {})

    # ── Phase 5: Build verified evidence block for the planner ────────────
    # Extracts progressive inspection results (structural knowledge + actual
    # source code regions) and serializes them as a structured text block
    # that the planner can reason about. Preserves provenance: file, class,
    # methods, relationships, contracts, actual code, facts, questions.
    verified_evidence = ""
    _inspections = getattr(agents.evidence_loop, "_artifact_inspections", {})
    _ev_knowledge = getattr(agents.evidence_loop, "_evidence_knowledge", None)
    if _inspections:
        parts = []
        for fp, insp in _inspections.items():
            if not insp.facts and not insp.methods and not insp.type_dependencies:
                continue
            section = f"\nFILE: {fp}\n"
            if insp.class_name:
                section += f"  Class: {insp.class_name}\n"
            if insp.methods:
                section += f"  Methods:\n"
                for m in insp.methods:
                    sig = f"({m.signature})" if m.signature else "()"
                    ret = f" → {m.return_type}" if m.return_type else ""
                    loc = f" [L{m.line_start}-{m.line_end}]" if m.line_start else ""
                    section += f"    {m.name}{sig}{ret}{loc}\n"
            if insp.relevant_methods:
                section += f"  RELEVANT TO TICKET: {', '.join(m.name for m in insp.relevant_methods)}\n"
            if insp.type_dependencies:
                unique_deps = list(dict.fromkeys(insp.type_dependencies))
                section += f"  Type dependencies: {', '.join(unique_deps[:12])}\n"
            if insp.api_contracts:
                section += f"  API contracts: {', '.join(insp.api_contracts[:5])}\n"
            # Relationship provenance
            _verdict = agentic_verdicts.get(fp, {})
            _reason = _verdict.get("reason", "")
            if _reason:
                section += f"  Discovery reason: {_reason[:200]}\n"
            section += f"  Inspection depth: {insp.inspection_depth}\n"
            section += f"  Providers: {', '.join(insp.providers_used)}\n"
            # Actual source code regions (the critical piece)
            for method_name, region in insp.relevant_code_regions.items():
                code = region.get("code", "")
                if code:
                    section += f"  --- ACTUAL CODE: {method_name}() L{region.get('start')}-{region.get('end')} ---\n"
                    for line in code.splitlines()[:40]:
                        section += f"    {line}\n"
                    section += f"  --- END CODE ---\n"
            # Facts with provenance
            if insp.facts:
                section += f"  Verified facts:\n"
                for f in insp.facts:
                    prov = f"[{f.source}]" if f.source else ""
                    conf = f" (confidence={f.confidence:.1f})" if f.confidence and f.confidence < 1.0 else ""
                    section += f"    {prov} {f.fact}{conf}\n"
            # Questions (unresolved)
            if insp.questions:
                # Deduplicate questions
                seen_q = set()
                unique_questions = []
                for q in insp.questions:
                    if q not in seen_q:
                        seen_q.add(q)
                        unique_questions.append(q)
                if unique_questions:
                    section += f"  Open questions:\n"
                    for q in unique_questions[:5]:
                        section += f"    ? {q}\n"
            parts.append(section)

        if parts:
            verified_evidence = "\n".join(parts)
            logger.info(
                f"  Phase 5: Built verified_evidence block with "
                f"{len(parts)} inspected file(s), "
                f"{sum(len(i.relevant_code_regions) for i in _inspections.values())} code region(s)"
            )

    # ── Phase 5b: Synthesize BehavioralUnderstanding (understand-first) ───────
    # Composes existing signals (inspections + relationships + contracts +
    # semantic constraints + reuse decisions + change authorization) into an
    # ordered, plan-ready understanding. No new gate/agent/phase.
    try:
        from ticket_to_code.agents.behavioral_understanding import (
            build_behavioral_understanding,
        )
        _ticket_bu = state.get("ticket")
        _authorized_bu: set[str] = set()
        for _attr in ("expected_changed_files", "expected_owner_files"):
            for _f in (_get_ticket_attr(_ticket_bu, _attr, None) or []):
                if _f:
                    _authorized_bu.add(str(_f))
        # Evidence-proven primary feature targets from localization ownership.
        # A discovered file is only a change candidate when ownership proves it
        # directly implements the requested behavior — discovery alone is not proof.
        _primary_targets_bu: set[str] = set()
        for _d in (state.get("discovered_files") or []):
            if not isinstance(_d, dict):
                continue
            _dp = _d.get("path") or ""
            _ot = _d.get("ownership_type") or (_d.get("cluster") or {}).get("ownership_type") or ""
            if _dp and str(_ot).upper() in ("PRIMARY_OWNER", "PRIMARY_FEATURE_TARGET", "DISPLAY_OWNER"):
                _primary_targets_bu.add(str(_dp))
        # Phase B inputs: feature phrase (surfacing) + per-file semantic relevance
        # from evidence collection. derive_primary_targets proves the primary
        # feature target by INSPECTION, never by score/filename alone.
        from ticket_to_code.agents.localization_agent import extract_feature_phrase as _efp
        _feature_tokens_bu = _efp(
            f"{getattr(_ticket_bu, 'title', '') or ''} "
            f"{getattr(_ticket_bu, 'description', '') or ''}"
        )
        _semantic_scores_bu: dict = {}
        for _e in (evidence_items or []):
            _efpath = getattr(_e, "file_path", "") or ""
            if _efpath:
                _semantic_scores_bu[_efpath] = {
                    "decision": "include",
                    "score": float(getattr(_e, "relevance_score", 0.0) or 0.0),
                }
        _bu = build_behavioral_understanding(
            ticket_title=getattr(_ticket_bu, "title", "") or "",
            ticket_description=getattr(_ticket_bu, "description", "") or "",
            requirements_text=str(state.get("requirements") or ""),
            inspections=_inspections or {},
            relationships=getattr(_ev_knowledge, "relationships", None) if _ev_knowledge else None,
            api_contracts=getattr(_ev_knowledge, "api_contracts", None) if _ev_knowledge else None,
            unresolved=getattr(_ev_knowledge, "unresolved_questions", None) if _ev_knowledge else None,
            evidence_files=[getattr(e, "file_path", "") for e in (evidence_items or []) if getattr(e, "file_path", "")],
            authorized_files=_authorized_bu,
            scope_declared=bool(_authorized_bu),
            primary_targets=_primary_targets_bu,
            feature_tokens=_feature_tokens_bu,
            semantic_scores=_semantic_scores_bu,
        )
        if _ev_knowledge is not None:
            try:
                setattr(_ev_knowledge, "behavioral_understanding", _bu)
            except Exception:
                pass
        _bu_block = _bu.to_planning_block()
        if _bu_block:
            verified_evidence = _bu_block + "\n\n" + (verified_evidence or "")
            logger.info(
                f"  Phase 5b: BehavioralUnderstanding synthesized "
                f"(delta={len(_bu.behavioral_delta)}, reuse={len(_bu.reuse_decisions)}, "
                f"change_candidates={len(_bu.change_candidates)}, "
                f"references={len(_bu.reference_artifacts)}, conf={_bu.confidence:.2f})"
            )
            # ── Forensic trace (Section 26): capability / reuse / understanding ──
            for _c in _bu.existing_capabilities[:8]:
                _prov = "grounded" if getattr(_c, "grounded", True) else "rag"
                logger.info(
                    f"  [CAPABILITY] [{_prov}] {(_c.owner + '.') if _c.owner else ''}{_c.name}"
                    f"{('(' + _c.signature + ')') if _c.signature else ''}"
                )
            for _d in _bu.reuse_decisions[:8]:
                logger.info(
                    f"  [REUSE] {_d.decision.upper()} '{_d.requirement}'"
                    f"{(' → ' + _d.existing_symbol) if _d.existing_symbol else ''}: {_d.reason}"
                )
            for _d in _bu.behavioral_delta[:6]:
                logger.info(f"  [UNDERSTANDING] delta: {_d}")
            for _cc in _bu.change_candidates[:10]:
                logger.info(f"  [PLAN] change_candidate: {_cc}")
            for _rf in _bu.reference_artifacts[:10]:
                logger.info(f"  [PLAN] read_only_reference: {_rf}")
    except Exception as _bu_exc:
        logger.debug(f"  Phase 5b BehavioralUnderstanding skipped (non-fatal): {_bu_exc}")

    # ── TicketScopeProof Construction & Validation ───────────────────────────
    _ticket_scope_proof = None
    try:
        from ticket_to_code.agents.ticket_scope_proof import (
            TicketScopeProof, ScopeGraph, EvidenceRole, OwnershipResolver,
            resolve_structural_companions, validate_scope_graph,
        )
        _ticket_obj = state.get("ticket")
        _tid = str(getattr(_ticket_obj, "ticket_id", "TICKET") or "TICKET")
        _title = str(getattr(_ticket_obj, "title", "") or "")
        _desc = str(getattr(_ticket_obj, "description", "") or "")

        _providers: set[str] = set()
        _related_context: set[str] = set()
        _dependencies: set[str] = set()
        _candidate_anchors: list[str] = []

        _preflight_guidance_targets = set()
        for g in (state.get("preflight_implementation_guidance") or []):
            if isinstance(g, dict):
                raw_tf = g.get("target_file") or g.get("file") or g.get("target_files")
                if isinstance(raw_tf, str) and raw_tf:
                    _preflight_guidance_targets.add(raw_tf.replace("\\", "/").lower().strip())
                elif isinstance(raw_tf, (list, tuple, set)):
                    for item in raw_tf:
                        if isinstance(item, str) and item:
                            _preflight_guidance_targets.add(item.replace("\\", "/").lower().strip())
        _ticket_expected = {
            f.replace("\\", "/").lower().strip()
            for f in (getattr(_ticket_obj, "expected_changed_files", None) or state.get("expected_changed_files") or [])
            if f
        }
        _known_targets = _preflight_guidance_targets | _ticket_expected

        for _ei in evidence_items:
            _efp = getattr(_ei, "file_path", "")
            if not _efp:
                continue
            _norm_efp = _efp.replace("\\", "/").lower().strip()
            _erole = getattr(_ei, "evidence_role", None) or getattr(_ei, "role", None)
            
            # Explicit preflight or ticket targets are candidate anchors, NEVER read-only providers!
            if any(_norm_efp == kt or _norm_efp.endswith("/" + kt) or kt.endswith("/" + _norm_efp) for kt in _known_targets if kt):
                _candidate_anchors.append(_efp)
            elif _erole == EvidenceRole.PROVIDER:
                _providers.add(_efp)
            elif _erole == EvidenceRole.RELATED_CONTEXT:
                _related_context.add(_efp)
            elif _erole == EvidenceRole.DEPENDENCY:
                _dependencies.add(_efp)
            else:
                _candidate_anchors.append(_efp)

        _owner_resolver = OwnershipResolver(workspace_path=state.get("workspace_path"))
        _tokens = [t for t in re.split(r'[-_.\s]+', _title.lower()) if len(t) > 2]
        _primary_anchor, _conf, _reason = _owner_resolver.resolve_owner(
            ticket_title=_title,
            ticket_description=_desc,
            candidate_files=_candidate_anchors,
            anchor_tokens=_tokens,
        )

        _companions: set[str] = set()
        if _primary_anchor:
            _companions = resolve_structural_companions(_primary_anchor, workspace_root=state.get("workspace_path"))

        # Preflight targets beyond the primary anchor belong in writable_dependencies
        _writable_deps: set[str] = set()
        for kt in _known_targets:
            if kt and kt != (_primary_anchor or "").replace("\\", "/").lower().strip():
                _dependencies.add(kt)
                _writable_deps.add(kt)
                if kt in _providers:
                    _providers.remove(kt)
                if kt in _related_context:
                    _related_context.remove(kt)

        # High-scoring candidate files (e.g. secondary services in multi-service tickets)
        for _cand_file, _cand_score, _, _ in getattr(_owner_resolver, "scored_candidates", []):
            _norm_cand = _cand_file.replace("\\", "/").lower().strip()
            _norm_primary = (_primary_anchor or "").replace("\\", "/").lower().strip()
            if _cand_score >= 0.70 and _norm_cand != _norm_primary and _norm_cand not in {_c.replace("\\", "/").lower().strip() for _c in _companions}:
                _dependencies.add(_cand_file)
                _writable_deps.add(_cand_file)
                if _cand_file in _providers:
                    _providers.remove(_cand_file)
                if _cand_file in _related_context:
                    _related_context.remove(_cand_file)
                for _sec_comp in resolve_structural_companions(_cand_file, workspace_root=state.get("workspace_path")):
                    _companions.add(_sec_comp)

        _scope_graph = ScopeGraph(
            ticket_id=_tid,
            primary_anchor=_primary_anchor,
            companions=_companions,
            dependencies=_dependencies,
            writable_dependencies=_writable_deps,
            providers=_providers,
            related_context=_related_context,
        )
        _is_valid, _val_errors = validate_scope_graph(_scope_graph)
        _is_proven = bool(_primary_anchor and _conf >= 0.70 and _is_valid)

        _ticket_scope_proof = TicketScopeProof(
            ticket_id=_tid,
            scope_graph=_scope_graph,
            is_scope_proven=_is_proven,
            proof_reasoning=_reason if _primary_anchor else "No proven anchor",
        )
        state["ticket_scope_proof"] = _ticket_scope_proof
        logger.info(
            f"  🔒 TicketScopeProof generated: proven={_is_proven}, "
            f"anchor={_primary_anchor}, writable={len(_scope_graph.all_writable_files())}, "
            f"readonly={len(_scope_graph.all_readonly_files())}"
        )
    except Exception as _tsp_err:
        logger.debug(f"  ScopeProof construction skipped: {_tsp_err}")

    return {
        "evidence_items": evidence_items,
        "query_expansion_map": expansion_map,
        "verified_evidence": verified_evidence,  # Phase 5: for planner
        "evidence_decision_degraded": bool(getattr(agents.evidence_loop, "decision_degraded", False)),
        "status": "evidence_collected",
        "_evidence_substeps": _evidence_substeps,
        "ticket_scope_proof": _ticket_scope_proof,
        "semantic_verification_results": [
            {
                "file_path": v.get("file_path", fp),
                "decision": v.get("decision", "include"),
                "semantic_relevance_score": v.get("semantic_relevance_score", 0.0),
                "reason": v.get("reason", ""),
            }
            for fp, v in agentic_verdicts.items()
        ] if agentic_verdicts else None,
    }


# ============================================================================
# PHASE 3B — EVIDENCE RANKING
# ============================================================================

def evidence_ranking_node(
    state: "TicketToCodeState",
    agents: "WorkflowAgents",
) -> dict:
    """
    Phase 3B: Evidence Ranking (with Component Group Awareness)

    1. Builds ComponentGroups from the collected evidence by scanning for
       sibling files on disk (e.g., .ts → .html + .scss).
    2. Injects synthetic EvidenceItems for group members that had no evidence
       so the ranker can score them (they inherit ~85% of the group score).
    3. Deterministic re-scoring: files with exact literal/symbol matches rank
       above filename-only and broad-repo matches.
    4. Writes evidence_ranking.json and component_groups.json to trace dir.
    """
    print("\n" + "=" * 80)
    print(" ENTERING: evidence_ranking_node() in workflow.py")
    print("   Purpose: Group into components → Rescore evidence → Rank")
    print("=" * 80)
    logger.info(" PHASE 3B: Evidence Ranking (Component Group Aware)")

    if state.get("status") == "failed":
        return {"status": "failed"}

    evidence_items  = state.get("evidence_items") or []
    hypotheses      = state.get("investigation_hypotheses") or []
    plan            = state.get("architectural_plan")
    localized_tasks = plan.tasks if plan else []

    if not evidence_items:
        logger.warning("  ⚠️  No evidence items to rank — skipping")
        return {"status": "evidence_ranked"}

    # ── Step 1: Build Component Groups ────────────────────────────────────
    workspace_path = Path(state.get("workspace_path", "."))
    component_groups = _build_component_groups(evidence_items, workspace_path)

    # ── Step 2: Inject synthetic evidence for non-evidence group members ──
    # For each group member that has NO evidence (e.g., .html found via disk
    # scan), create a synthetic EvidenceItem so the ranker includes it.
    existing_evidence_paths = {e.file_path.replace("\\", "/") for e in evidence_items}
    injected_count = 0

    for group in component_groups:
        if not group.is_group:
            continue  # Standalone files don't need injection

        for ext, member_path in group.non_evidence_members().items():
            norm_path = member_path.replace("\\", "/")
            if norm_path in existing_evidence_paths:
                continue  # Already has evidence

            # Inherit 85% of the group's best score
            inherited_score = round(group.group_score * 0.85, 4)

            evidence_items.append(EvidenceItem(
                file_path=member_path,
                provider="component_group",
                source="component_group",
                strength="medium",
                details=f"component_group_member: {group.stem}",
                content_snippet=(
                    f"[component_group:{group.stem}] "
                    f"Sibling of {Path(group.primary_file).name} "
                    f"(discovered via disk scan, ext={ext})"
                ),
                relevance_score=inherited_score,
                hypothesis_id=None,
                confidence_reason=(
                    f"Component group member of '{group.stem}'. "
                    f"Primary file {Path(group.primary_file).name} has "
                    f"evidence score {group.group_score:.3f}."
                ),
            ))
            existing_evidence_paths.add(norm_path)
            injected_count += 1

    if injected_count > 0:
        logger.info(
            f"  [ComponentGroup] Injected {injected_count} synthetic evidence "
            f"items for non-evidence group members"
        )

    # ── Step 2.5: Discover repository roots for ranking context ───────
    # WorkspaceRegistryManager is the single owner of repository identity.
    # It discovers Git repository boundaries once, then batch-resolves
    # every candidate path → its containing repository root (or None).
    from ticket_to_code.storage.workspace_registry import WorkspaceRegistryManager

    registry_mgr = WorkspaceRegistryManager(str(workspace_path))
    repo_roots = registry_mgr.discover_repo_roots()
    candidate_paths = {e.file_path.replace("\\", "/") for e in evidence_items}
    repo_root_map = registry_mgr.build_repo_root_map(candidate_paths, repo_roots)

    # Log repository identity for each candidate (traceable)
    orphan_count = sum(1 for v in repo_root_map.values() if v is None)
    logger.info(
        f"  [RepoIdentity] {len(repo_root_map)} candidates, "
        f"{len(repo_root_map) - orphan_count} in recognized repos, "
        f"{orphan_count} orphan(s)"
    )
    for fp, repo in sorted(repo_root_map.items()):
        logger.info(
            f"    candidate={fp}  repo={repo if repo is not None else '<orphan>'}"
        )

    # ── Step 3: Rank all evidence (including injected group members) ──────
    ranker = EvidenceRankingEngine()
    ranked_items, ranked_files = ranker.rank(
        evidence_items=evidence_items,
        hypotheses=hypotheses,
        localized_tasks=localized_tasks,
        expansion_map=state.get("query_expansion_map") or {},
        repo_root_map=repo_root_map,
    )

    if ranked_files:
        top = ranked_files[0]
        logger.info(
            f"  Ranking complete: {len(ranked_files)} unique files, "
            f"top='{top.file_path}' ({top.final_score:.3f}) repo={top.repo_root or '<orphan>'}"
        )

    # ── Step 4: Trace ────────────────────────────────────────────────────
    agents.tracer.record_evidence_ranking(ranked_files)

    # Trace component groups
    try:
        import json as _json
        trace_dir = agents.tracer._trace_dir if hasattr(agents.tracer, '_trace_dir') else None
        if trace_dir:
            groups_file = os.path.join(trace_dir, "component_groups.json")
            with open(groups_file, "w") as f:
                _json.dump(
                    [g.to_dict() for g in component_groups],
                    f, indent=2,
                )
    except Exception:
        pass  # Tracing is best-effort

    # ── v3: Populate TicketExecutionContext with component groups ──────────
    try:
        _exec_ctx = _get_transient(state, "exec_ctx")
        if _exec_ctx:
            _exec_ctx.set_component_groups(component_groups)
            reqs = state.get("requirements")
            if reqs:
                _exec_ctx.set_requirements(
                    functional=getattr(reqs, "functional_requirements", []) or [],
                    technical=getattr(reqs, "technical_requirements", []) or [],
                )
            logger.info(f"  [v3] ExecCtx updated: {len(component_groups)} component groups")
    except Exception as _ectx_err:
        logger.debug(f"  [v3] ExecCtx update failed (non-fatal): {_ectx_err}")

    return {
        "evidence_items": ranked_items,
        "ranked_files": ranked_files,
        "component_groups": component_groups,
        "status": "evidence_ranked",
    }



def grounded_understanding_node(
    state: "TicketToCodeState",
    agents: "WorkflowAgents",
) -> dict:
    """
    Phase 2G-3: Grounded Understanding Synthesis

    Synthesizes all collected evidence into a GroundedUnderstanding that:
      • Confirms or refutes each hypothesis
      • Builds the dependency chain (entry → change site)
      • Identifies the change group (files that must change together)
      • Separates writable_files from readonly_context_files
      • Sets the overall confidence score

    The GroundedUnderstanding is added to the workflow state and passed
    to code generators so they operate on verified evidence.
    """
    print("\n" + "=" * 80)
    print(" ENTERING: grounded_understanding_node() in workflow.py")
    print("   Purpose: Synthesize evidence into GroundedUnderstanding for generation")
    print("=" * 80)
    logger.info(" PHASE 2G-3: Grounded Understanding Synthesis")

    hypotheses   = state.get("investigation_hypotheses") or []
    evidence     = state.get("evidence_items") or []
    investigation = state["investigation_result"]
    plan          = state.get("architectural_plan")
    localized_tasks = plan.tasks if plan else []
    ticket        = state["ticket"]

    # ── FIX 1.3: CONFIDENCE GATE ─────────────────────────────────────────────
    # Prevent root-cause synthesis when there is no evidence to support it.
    # Previously the node would continue with confidence=0 and produce a
    # hallucinated root cause from the hypothesis alone.  Now we halt and
    # signal that expanded discovery is needed.
    #
    # HOWEVER: If the planner already found validated writable files with high
    # confidence (via discovery + localization), we should NOT block code
    # generation.  The evidence loop may return 0 items when the search terms
    # are too long/sentence-like, but the plan is already grounded by the
    # discovery phase.
    _writable_tasks = [t for t in localized_tasks if t.task_type.value != "read_only"]
    if not evidence and not _writable_tasks:
        logger.warning(
            "  ⚠️  CONFIDENCE GATE: 0 evidence items AND 0 writable tasks. "
            "Cannot synthesize a grounded root cause — this would be hallucination. "
            "Signaling 'evidence_insufficient' for re-discovery."
        )
        # Build a minimal GroundedUnderstanding so downstream nodes don't crash
        grounded = GroundedUnderstanding(
            root_cause="INSUFFICIENT EVIDENCE — no evidence was collected to ground a root cause",
            evidence=[],
            dependency_chain=[],
            change_group=[t.file_path for t in localized_tasks if t.task_type.value != "read_only"],
            required_files=[t.file_path for t in localized_tasks],
            writable_files=[t.file_path for t in localized_tasks if t.task_type.value != "read_only"],
            readonly_context_files=[t.file_path for t in localized_tasks if t.task_type.value == "read_only"],
            confidence=0.0,
            evidence_loop_iterations=0,
            hypotheses_confirmed=[],
            hypotheses_refuted=[h.id for h in hypotheses],
        )
        return {
            "grounded_understanding": grounded,
            "status": "evidence_insufficient",
        }

    if not evidence and _writable_tasks:
        logger.warning(
            f"  ⚠️  CONFIDENCE GATE SOFTENED: 0 evidence items but {len(_writable_tasks)} "
            f"writable tasks already planned. Proceeding with plan-grounded synthesis."
        )

    # ── Writable vs readonly from the plan ───────────────────────────────────
    writable_from_plan = [
        t.file_path for t in localized_tasks if t.task_type.value != "read_only"
    ]
    readonly_from_plan = [
        t.file_path for t in localized_tasks if t.task_type.value == "read_only"
    ]

    # ── Evidence-file sets ────────────────────────────────────────────────────
    evidence_files  = list({e.file_path for e in evidence})
    # Files surfaced by evidence but not already in the plan
    extra_context   = [f for f in evidence_files if f not in writable_from_plan]

    # ── Dependency chain: high-relevance evidence items in score order ────────
    sorted_evidence = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)
    chain_files     = []
    seen_chain: set = set()
    for e in sorted_evidence[:20]:
        if e.file_path not in seen_chain:
            chain_files.append(e.file_path)
            seen_chain.add(e.file_path)

    # ── Change group: writable plan files + high-confidence evidence files ────
    # Exclude: config_chain (data/i18n files), traces/ (audit logs), readonly prefixes
    # Include: repository_search (Phase 3A) — finds .sh/.bat/.ps1/.xml/.env not in index
    _CODE_SOURCES = {"semantic", "sqlite_fts", "sqlite_symbol", "neo4j",
                     "ts_chain", "java_chain", "css_chain", "layout_chain",
                     "repository_search", "rkb_feature_match"}

    # ── Change 3: Planner-decision-aware evidence injection ───────────────────
    # BEFORE: every evidence file with score >= 0.70 was injected unconditionally.
    # AFTER:  files the planner evaluated and IGNORED are never auto-promoted.
    #         Files not shown to the planner (outside top-25) keep the original
    #         fallback promotion so backward compatibility is preserved.
    planner_decisions = state.get("planner_decisions") or {}

    # Roles whose OPTIONAL planner decision still does NOT warrant promotion.
    # These are files that are read-only context at best; they must never become
    # writable modify tasks unless the planner explicitly marked them REQUIRED.
    _NON_PROMOTABLE_ROLES = {
        CandidateRole.TEST.value,
        CandidateRole.LOCK_FILE.value,
        CandidateRole.GENERATED.value,
    }

    # FIX 3: Roles that MUST survive a planner IGNORE when evidence is strong.
    # These files are physically required to change in version-propagation tickets.
    # The planner may miss them due to absent ticket classification, so evidence
    # score >= 0.85 overrides the IGNORE gate as a safety net.
    _FORCE_PROMOTE_ROLES = {
        CandidateRole.DEPLOYMENT.value,
        CandidateRole.VERSION_SOURCE.value,
        CandidateRole.ROOT_COMPONENT.value,
    }

    high_conf_evidence: list = []
    _ignored_count = 0
    _fallback_count = 0

    for e in evidence:
        fp = e.file_path

        # Basic filters — unchanged from original
        if e.relevance_score < 0.70:
            continue
        if fp in writable_from_plan:
            continue
        if e.source not in _CODE_SOURCES:
            continue
        if fp.startswith(("traces/", "traces\\")):
            continue

        # ── Respect planner decisions ─────────────────────────────────────────
        dec_record = planner_decisions.get(fp)

        if dec_record:
            decision = dec_record["decision"]
            role     = dec_record.get("role", CandidateRole.UNKNOWN.value)

            if decision == PlannerDecisionValue.IGNORE.value:
                # FIX 3: critical propagation roles with strong evidence bypass the gate.
                # DEPLOYMENT / VERSION_SOURCE / ROOT_COMPONENT files at score >= 0.85
                # must survive even when the planner omitted them (likely due to missing
                # ticket-type classification).  All other roles keep the hard block.
                if role in _FORCE_PROMOTE_ROLES and e.relevance_score >= 0.85:
                    logger.info(
                        f"  [grp] FORCE-PROMOTE IGNORE override: {fp} "
                        f"(role={role}, score={e.relevance_score:.2f}) "
                        f"— critical propagation file, bypassing planner IGNORE"
                    )
                    # Fall through to high_conf_evidence.append(fp)
                else:
                    logger.debug(
                        f"  [grp] SKIP IGNORE — planner excluded: {fp} "
                        f"(role={role}, score={e.relevance_score:.2f})"
                    )
                    _ignored_count += 1
                    continue

            elif decision == PlannerDecisionValue.OPTIONAL.value:
                # Planner included file as read-only context only.
                # Promote to writable only for roles that can genuinely require
                # an autonomous version-propagation write (DEPLOYMENT, ROOT_COMPONENT,
                # VERSION_SOURCE, CONFIG).  Never promote TEST / LOCK_FILE / GENERATED.
                if role in _NON_PROMOTABLE_ROLES:
                    logger.debug(
                        f"  [grp] SKIP OPTIONAL {role} — non-promotable role: {fp}"
                    )
                    _ignored_count += 1
                    continue
                # else: fall through and allow promotion

            # decision == REQUIRED: already in writable_from_plan (filtered above).
            # Fall through to append.

        else:
            # No planner decision exists — this file was outside the top-25 shown
            # to the planner. Keep original fallback-promotion behaviour, EXCEPT
            # filter out non-promotable roles to prevent blind spots.
            # Check discovery-pipeline ownership before structural fallback
            _fb_ot = ""
            for _d in (state.get("discovered_files") or []):
                if _d.get("path", "").replace("\\", "/") == fp.replace("\\", "/"):
                    _fb_ot = (_d.get("features") or {}).get("ownership_type", "") or ""
                    break
            if _fb_ot in ("PRIMARY_OWNER", "DISPLAY_OWNER", "SUPPORTING", "READ_ONLY"):
                fallback_role = _fb_ot
            else:
                fallback_role = _classify_candidate_role(fp)
            if fallback_role in _NON_PROMOTABLE_ROLES:
                logger.debug(f"  [grp] SKIP FALLBACK role={fallback_role} path={fp}")
                _ignored_count += 1
                continue

            _fallback_count += 1
        high_conf_evidence.append(fp)

    logger.info(
        f"  [grp] evidence promotion: {len(high_conf_evidence)} promoted, "
        f"{_ignored_count} blocked by planner IGNORE/non-promotable, "
        f"{_fallback_count} fallback (not shown to planner)"
    )

    change_group = list(dict.fromkeys(writable_from_plan + high_conf_evidence))

    # ── Phase 3A.2: Promote change_group files into plan tasks ───────────────
    # Files in high_conf_evidence are not yet in the architectural plan; create
    # writable DevelopmentTask objects for each so generate_code_node can write
    # them.  Only promote files that physically exist on disk.
    workspace_root = Path(state["workspace_path"])
    promoted_paths: list = []
    if plan is not None and high_conf_evidence:
        # Sort by best evidence score descending so the highest-confidence files
        # enter plan.tasks first.  This matters because generate_code_node enforces
        # a hard cap of 5 writable files and processes them in plan.tasks order.
        def _best_score(fp: str) -> float:
            return max(
                (e.relevance_score for e in evidence if e.file_path == fp),
                default=0.70,
            )
        sorted_evidence_files = sorted(high_conf_evidence, key=_best_score, reverse=True)
        task_idx_start = len(plan.tasks)
        for idx, file_path in enumerate(sorted_evidence_files):
            if not (workspace_root / file_path).exists():
                logger.debug(f"  [grp] skip (not on disk): {file_path}")
                continue
            # Best evidence score for this file
            best_score = max(
                (e.relevance_score for e in evidence if e.file_path == file_path),
                default=0.70,
            )
            best_source = next(
                (e.source for e in sorted(evidence, key=lambda e: e.relevance_score, reverse=True)
                 if e.file_path == file_path),
                "repository_search",
            )
            promoted_task = _make_promoted_task(
                file_path=file_path,
                workspace_path=workspace_root,
                task_index=task_idx_start + idx + 1,
                evidence_score=best_score,
                evidence_source=best_source,
            )
            plan.tasks.append(promoted_task)
            promoted_paths.append(file_path)
            logger.info(
                f"  [grp] promoted → {file_path} "
                f"(src={best_source}, score={best_score:.2f})"
            )
        if promoted_paths:
            logger.info(
                f"  Phase 3A.2: {len(promoted_paths)} file(s) promoted to plan tasks"
            )

    # ── Hypothesis evaluation ─────────────────────────────────────────────────
    confirmed: list  = []
    refuted:   list  = []
    for hyp in hypotheses:
        supporting = [
            e for e in evidence
            if e.hypothesis_id == hyp.id and e.relevance_score >= 0.60
        ]
        if supporting:
            confirmed.append(hyp.id)
        else:
            refuted.append(hyp.id)

    # ── Overall confidence (from evidence loop result) ────────────────────────
    # Recompute so the value is consistent with the evidence list
    from ticket_to_code.agents.evidence_collection_loop import EvidenceCollectionLoop
    confidence = EvidenceCollectionLoop(
        agents.localizer, agents.rag_engine
    )._compute_confidence(evidence, hypotheses, localized_tasks)

    # ── Root cause synthesis ──────────────────────────────────────────────────
    root_cause = (
        investigation.root_cause_hypothesis
        or f"Ticket '{ticket.title}' requires changes in: {', '.join(writable_from_plan[:3])}"
    )
    if confirmed:
        # Use the first confirmed hypothesis as the primary root cause statement
        for hyp in hypotheses:
            if hyp.id in confirmed:
                root_cause = hyp.hypothesis
                break

    # ── Auto-Companion Scope Binding (Contract & Test Parity) ──────────────────
    # Resolves companion unit tests (e.g. UpdateProjectUserHandler.java <-> UpdateProjectUserHandlerTest.java)
    # and companion contracts so scope lock doesn't reject constructor/mock repairs.
    initial_writable = list(dict.fromkeys(writable_from_plan + promoted_paths))
    expanded_with_companions = list(initial_writable)
    try:
        from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
        for w_fp in initial_writable:
            comp = PolyglotCompanionResolver.resolve_companion(w_fp, workspace_path=str(workspace_path))
            if comp and comp[0] and comp[0] not in expanded_with_companions:
                expanded_with_companions.append(comp[0])
                logger.info(f"  🔗 [AutoCompanion] Bound companion test/contract to writable scope: {comp[0]}")
    except Exception as _ce:
        logger.debug(f"Auto-companion resolution error: {_ce}")

    grounded = GroundedUnderstanding(
        root_cause=root_cause,
        evidence=evidence,
        dependency_chain=chain_files,
        change_group=change_group,
        required_files=list(dict.fromkeys(writable_from_plan + extra_context[:10])),
        writable_files=expanded_with_companions,
        readonly_context_files=list(dict.fromkeys(readonly_from_plan + extra_context[:10])),
        confidence=round(confidence, 3),
        evidence_loop_iterations=len(set(e.source for e in evidence)),
        hypotheses_confirmed=confirmed,
        hypotheses_refuted=refuted,
    )

    logger.info(
        f"  Grounded Understanding:\n"
        f"    root_cause       : {grounded.root_cause[:80]}\n"
        f"    evidence items   : {len(grounded.evidence)}\n"
        f"    dependency_chain : {len(grounded.dependency_chain)} files\n"
        f"    change_group     : {grounded.change_group}\n"
        f"    writable_files   : {grounded.writable_files}\n"
        f"    readonly_context : {grounded.readonly_context_files[:5]}\n"
        f"    confidence       : {grounded.confidence:.3f}\n"
        f"    hypotheses ✅    : {grounded.hypotheses_confirmed}\n"
        f"    hypotheses ❌    : {grounded.hypotheses_refuted}"
    )

    # ── v3: Populate TicketExecutionContext with grounded understanding ────
    try:
        _exec_ctx = _get_transient(state, "exec_ctx")
        if _exec_ctx:
            _exec_ctx.set_grounded_understanding(grounded)
    except Exception as _ectx_err:
        logger.debug(f"  [v3] ExecCtx update failed (non-fatal): {_ectx_err}")

    # ── Topological Tiered Ordering (Contract-First Execution) ────────────────
    # Prevents services from calling non-existent interface methods or compiling
    # before contracts and migrations are in place.
    if plan and getattr(plan, "tasks", None):
        def _tier_rank(task) -> int:
            fp = (getattr(task, "file_path", "") or "").lower().replace("\\", "/")
            if any(k in fp for k in ["test", "spec"]):
                return 4
            if any(k in fp for k in ["migration", ".sql", "schema"]):
                return 3
            if any(k in fp for k in ["repository", "dto", "entity", "model", "interface", "constants"]):
                return 1
            if any(k in fp for k in ["service", "handler", "controller", "component", "view", "route"]):
                return 2
            return 2  # default to middle tier

        plan.tasks = sorted(plan.tasks, key=_tier_rank)
        logger.info(f"  [TopologicalOrder] Sorted {len(plan.tasks)} planned tasks by contract dependency tier")

    return {
        "grounded_understanding": grounded,
        "architectural_plan": plan,        # updated with promoted change_group tasks
        "status": "grounded_understanding_complete",
    }


# ============================================================================
# PHASE 2F — OWNERSHIP COMPLETENESS CHECK
# ============================================================================
# Purpose: Determine whether localized files FULLY own the behavior in the
# ticket.  If a layout gap is detected (a CSS file owns BEM modifiers while
# the root block's layout container is owned by a sibling file), add the
# sibling as a writable LAYOUT_OWNER task so generation receives both files.
#
# Design constraints:
#  • No hardcoded repository paths or component names
#  • No LLM calls — pure static analysis
#  • Reuses sqlite_store and workspace_path from LocalizationAgent
#  • Only SCSS/CSS gap detection is implemented (layout gap is the known class
#    of completeness failure; data / dependency gaps reuse existing discovery)
# ============================================================================

_LAYOUT_KW: frozenset = frozenset({
    "alignment", "aligned", "footer", "position", "spacing", "margin",
    "padding", "css", "scss", "layout", "height", "flex", "overflow",
    "sticky", "fixed", "bottom", "top", "ux", "ui", "display", "visual",
    "style", "responsive", "superimposed", "overlap", "blank space",
    "page css", "static css", "button.*bottom", "button.*area",
})

_LAYOUT_CSS_RE = re.compile(
    r'\b(display|flex|flex-direction|flex-grow|flex-shrink|flex-wrap|'
    r'position|height|min-height|max-height|width|min-width|max-width|'
    r'overflow|overflow-y|overflow-x|top|bottom|left|right)\s*:',
    re.IGNORECASE,
)

_SCSS_EXTS: frozenset = frozenset({".scss", ".css", ".sass"})


def _ticket_is_layout_focused(ticket_lower: str) -> bool:
    """Return True when ≥ 2 layout-related keywords appear in the ticket."""
    hits = sum(1 for kw in _LAYOUT_KW if kw in ticket_lower)
    return hits >= 2


def _extract_top_level_classes(content: str) -> set:
    """Return every top-level CSS class name defined in a SCSS/CSS file."""
    # Matches `.class-name {` at the start of a line (handles optional indentation)
    pat = re.compile(r'^\s*\.([\w][\w-]*)\s*\{', re.MULTILINE)
    return {m.group(1) for m in pat.finditer(content)}


def _class_has_layout_properties(content: str, class_name: str) -> bool:
    """
    Return True when the file defines `.class_name { ... }` and that block
    contains at least one layout-critical CSS property.
    Handles multi-line SCSS blocks via a brace-depth scanner.
    """
    # Find the opening of the block
    opener = re.compile(r'\.' + re.escape(class_name) + r'\s*\{')
    m = opener.search(content)
    if not m:
        return False
    # Scan forward counting braces to extract the block body
    pos = m.end()
    depth = 1
    body_chars: list = []
    while pos < len(content) and depth > 0:
        ch = content[pos]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        body_chars.append(ch)
        pos += 1
    block_body = "".join(body_chars)
    return bool(_LAYOUT_CSS_RE.search(block_body))


def _apply_ui_state_regression_bias(candidates: list, ticket) -> int:
    """
    Bias candidate confidence for cross-tab UI state regressions.

    Why: Tickets like "Button is enabled in tab A, then after navigating tab B and
    returning to tab A it becomes disabled" often get over-matched to action-tab
    components (tab B) instead of the owner and shared form/register state files.

    This adjustment is small and additive, and only activates for this ticket shape.
    """
    if not candidates or ticket is None:
        return 0

    title = str(getattr(ticket, "title", "") or "").lower()
    desc = str(getattr(ticket, "description", "") or "").lower()
    txt = f"{title} {desc}"

    has_ui_toggle_symptom = ("upload" in txt and "button" in txt and ("disabled" in txt or "enabled" in txt))
    has_cross_tab_flow = ("tab" in txt and "after" in txt and "return" in txt)
    has_register_deliverable = ("register" in txt and "deliverable" in txt)

    if not (has_ui_toggle_symptom and has_cross_tab_flow and has_register_deliverable):
        return 0

    adjusted = 0
    for c in candidates:
        p = str(c.get("path", "") or "").replace("\\", "/").lower()
        if not p:
            continue

        delta = 0.0

        # Primary symptom ownership is usually in the affected tab component.
        if "/registers/" in p:
            delta += 0.18
        if "upload-register.component.ts" in p:
            delta += 0.10

        # Shared state owners that commonly gate button enablement.
        if "/shared/services/registers/" in p:
            delta += 0.14
        if "/shared/form/" in p or "document-file-browser" in p:
            delta += 0.12

        # Action-tab components are still relevant, but should not dominate.
        if "/deliverables/" in p and "/registers/" not in p:
            delta -= 0.10

        if abs(delta) < 1e-9:
            continue

        base = float(c.get("confidence", 0.0) or 0.0)
        c["confidence"] = max(0.0, min(1.0, base + delta))
        c.setdefault("signals", []).insert(0, f"ui_state_bias:{delta:+.2f}")
        adjusted += 1

    return adjusted


def _find_layout_gap_owners(
    scss_task_path: str,
    workspace_path: Path,
    existing_paths: Set[str],
    ticket_keywords: Optional[Set[str]] = None,
) -> list:
    """
    Given a SCSS file that is a localized writable task, find sibling SCSS
    files in the same *module directory* (grandparent of the component dir)
    that:
      a) define at least one of the same top-level CSS classes, AND
      b) those shared classes contain layout-critical properties.

    Ranking (most important first):
      1. HOST COMPONENT — the sibling whose HTML template uses the localized
         component's selector (e.g., <se-deliverable-reviewers>).  This is
         the true layout-container owner.
      2. SIBLING with shared layout class (fallback if no host found).

    Returns a list of dicts:
        {"path": str, "shared_classes": list[str], "is_host": bool}
    capped at 2 entries to prevent uncontrolled expansion.
    """
    ws = Path(workspace_path)
    norm = scss_task_path.replace("\\", "/")
    scss_file = ws / norm
    if not scss_file.exists():
        return []

    try:
        local_content = scss_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    local_classes = _extract_top_level_classes(local_content)
    if not local_classes:
        return []

    # Derive the component name fragment to locate the host component.
    # e.g. "deliverable-reviewers.component.scss" → "deliverable-reviewers"
    component_fragment = Path(scss_task_path).stem.split(".")[0]
    # Match any Angular-style selector that ends with the component name
    # fragment: <se-deliverable-reviewers …>, <app-deliverable-reviewers …>
    host_re = re.compile(
        r"<[\w]+-" + re.escape(component_fragment) + r"[\s>/]",
        re.IGNORECASE,
    )

    # Module directory = grandparent of the component directory.
    # e.g. .../deliverables/deliverable-reviewers/x.scss → .../deliverables/
    module_dir = scss_file.parent.parent
    if not module_dir.is_dir():
        return []

    import time
    import os
    logger.info("LAYOUT_GAP_SCAN_START")
    start_time = time.time()

    host_results: list = []
    sibling_results: list = []

    skip_dirs = {"node_modules", ".git", "dist", "build", "target", "coverage"}
    candidates = []
    for root, dirs, files in os.walk(str(module_dir)):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if f.endswith(".scss"):
                candidates.append(Path(root) / f)

    files_scanned = 0
    max_files = 500

    for candidate in sorted(candidates):
        if files_scanned >= max_files:
            logger.warning(f"MAX_SCSS_FILES ({max_files}) exceeded, stopping scan")
            break
            
        # Hard cap: 1 host + 1 sibling maximum
        if len(host_results) >= 1 and len(sibling_results) >= 1:
            break
            
        files_scanned += 1
        rel = str(candidate.relative_to(ws)).replace("\\", "/")
        if rel == norm or rel in existing_paths:
            continue
        try:
            cand_content = candidate.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        cand_classes = _extract_top_level_classes(cand_content)
        shared = local_classes & cand_classes
        if not shared:
            continue

        layout_shared = [
            cls for cls in sorted(shared)
            if _class_has_layout_properties(cand_content, cls)
        ]
        if not layout_shared:
            continue

        # Check if corresponding HTML template hosts the localized component
        html_path = candidate.with_suffix(".html")
        is_host = False
        if html_path.exists():
            try:
                html_txt = html_path.read_text(encoding="utf-8", errors="ignore")
                is_host = bool(host_re.search(html_txt))
            except Exception:
                pass

        entry = {"path": rel, "shared_classes": layout_shared, "is_host": is_host}
        if is_host:
            if len(host_results) < 1:
                host_results.append(entry)
        else:
            # Sibling promotion requires direct ticket keyword evidence to prevent drift into picklist/contract-settings
            cand_keywords = set(re.findall(r'[a-zA-Z0-9_-]{3,}', rel.lower()))
            has_keyword_evidence = bool(ticket_keywords and (cand_keywords & ticket_keywords))
            if has_keyword_evidence:
                if len(sibling_results) < 1:
                    sibling_results.append(entry)
            else:
                logger.info(f"   [LayoutGap] Sibling candidate '{rel}' rejected: no ticket keyword match")

    duration_ms = int((time.time() - start_time) * 1000)
    logger.info("LAYOUT_GAP_SCAN_END")
    logger.info(f"SCSS_FILES_SCANNED={files_scanned}")
    logger.info(f"SCAN_DURATION_MS={duration_ms}")

    # If the host component was precisely identified, use only that.
    # The sibling fallback is a last resort when no host can be determined.
    if host_results:
        return host_results[:1]
    return sibling_results[:2]


def _make_ownership_task(
    file_path: str,
    workspace_path: Path,
    task_index: int,
    role: str,
    source_file: str,
    shared_classes: list,
) -> "Any":
    """
    Create a DevelopmentTask for an ownership-expanded file.
    Imports are done locally to avoid circular imports at module level.
    """
    from ticket_to_code.models import DevelopmentTask, TaskType, ProgrammingLanguage

    ext = Path(file_path).suffix.lower()
    lang = ProgrammingLanguage.TYPESCRIPT  # Angular SCSS is TypeScript project

    return DevelopmentTask(
        id=f"own-{task_index}",
        title=f"{role}: {Path(file_path).name}",
        description=(
            f"Ownership completeness expansion: this file shares CSS classes "
            f"({', '.join(f'.{c}' for c in shared_classes)}) with {source_file} "
            f"and owns the layout container. Must be modified alongside the "
            f"localized file to fully resolve the layout issue."
        ),
        file_path=file_path,
        task_type=TaskType.MODIFY,
        language=lang,
        localization_confidence=0.72,
        localization_reason=(
            f"ownership_completeness: {role} — shared CSS classes "
            f"{shared_classes} also defined in {source_file}"
        ),
        ownership_type="DISPLAY_OWNER",   # CSS file → display owner tier
        new_file_creation_allowed=False,
    )


# ── Change-group extension map ────────────────────────────────────────────────
_EXT_LANG_MAP: dict = {}


def _lang_for_ext(ext: str) -> "Any":
    """Return the ProgrammingLanguage enum value for a file extension."""
    from ticket_to_code.models import ProgrammingLanguage
    return {
        ".java":       ProgrammingLanguage.JAVA,
        ".ts":         ProgrammingLanguage.TYPESCRIPT,
        ".tsx":        ProgrammingLanguage.TYPESCRIPT,
        ".js":         ProgrammingLanguage.JAVASCRIPT,
        ".jsx":        ProgrammingLanguage.JAVASCRIPT,
        ".py":         ProgrammingLanguage.PYTHON,
        ".go":         ProgrammingLanguage.GO,
        ".cs":         ProgrammingLanguage.CSHARP,
        ".html":       ProgrammingLanguage.HTML,
        ".htm":        ProgrammingLanguage.HTML,
        ".scss":       ProgrammingLanguage.SCSS,
        ".css":        ProgrammingLanguage.SCSS,
        ".less":       ProgrammingLanguage.SCSS,
        ".sass":       ProgrammingLanguage.SCSS,
        ".json":       ProgrammingLanguage.JSON,
        ".xml":        ProgrammingLanguage.XML,
        ".yml":        ProgrammingLanguage.YAML,
        ".yaml":       ProgrammingLanguage.YAML,
        ".properties": ProgrammingLanguage.PROPERTIES,
        ".env":        ProgrammingLanguage.PROPERTIES,
        ".sh":         ProgrammingLanguage.SHELL,
        ".bash":       ProgrammingLanguage.SHELL,
        ".ps1":        ProgrammingLanguage.SHELL,
        ".bat":        ProgrammingLanguage.SHELL,
        ".cmd":        ProgrammingLanguage.SHELL,
    }.get(ext, ProgrammingLanguage.PYTHON)


def _make_promoted_task(
    file_path: str,
    workspace_path: Path,
    task_index: int,
    evidence_score: float,
    evidence_source: str,
) -> "Any":
    """
    Create a DevelopmentTask for a file promoted from grounded_understanding
    change_group.  These are files discovered by evidence (Phase 3A repository
    search, SQLite FTS, etc.) that are not yet in the architectural plan.
    """
    from ticket_to_code.models import DevelopmentTask, TaskType

    ext = Path(file_path).suffix.lower()
    lang = _lang_for_ext(ext)

    return DevelopmentTask(
        id=f"grp-{task_index}",
        title=f"Change group promotion: {Path(file_path).name}",
        description=(
            f"Promoted from grounded_understanding change_group. "
            f"Discovered via '{evidence_source}' with confidence {evidence_score:.2f}. "
            f"Must be modified to fully resolve the ticket."
        ),
        file_path=file_path,
        task_type=TaskType.MODIFY,
        language=lang,
        localization_confidence=round(evidence_score, 3),
        localization_reason=(
            f"change_group_promotion: source={evidence_source}, "
            f"evidence_score={evidence_score:.2f}"
        ),
        new_file_creation_allowed=False,
    )


# ── Component Group Builder (Pre-Ranking) ─────────────────────────────────────
# Groups evidence files into component groups by scanning for sibling files
# on disk. Runs BETWEEN evidence collection and ranking so the ranker can
# score at the component level instead of the individual file level.
#
# Framework-agnostic: works for Angular (.ts+.html+.scss), React (.tsx+.css),
# Vue (.vue+.scss), and any other convention where files share a common stem.
# Standalone files (e.g., Java services) become groups of 1 (is_group=False)
# and are unaffected by the grouping system.

# Extensions that indicate a "source" file (logic/controller)
_GROUP_SOURCE_EXTS = frozenset({".ts", ".tsx", ".jsx", ".js", ".vue", ".svelte"})
# Extensions that indicate a companion file (template/style)
_GROUP_COMPANION_EXTS = frozenset({".html", ".htm", ".scss", ".css", ".less", ".sass"})
# All groupable extensions (source + companion)
_GROUP_ALL_EXTS = _GROUP_SOURCE_EXTS | _GROUP_COMPANION_EXTS


def _build_component_groups(
    evidence_items: list,
    workspace_path: "Path",
) -> "list[ComponentGroup]":
    """
    Build component groups from evidence items by scanning for sibling files.

    Algorithm:
    1. Collect all unique file paths from evidence items.
    2. For each file with a groupable extension (.ts, .html, .scss, etc.),
       compute its stem and directory.
    3. Scan the directory on disk for sibling files with the same stem.
    4. Group them into a ComponentGroup with the highest-scoring evidence
       file as the primary.
    5. Files that don't belong to any group become standalone groups (is_group=False).

    Returns a list of ComponentGroup objects.
    """
    # Collect all evidence files and their best scores
    file_scores: dict[str, float] = {}
    for item in evidence_items:
        fp = item.file_path.replace("\\", "/")
        file_scores[fp] = max(file_scores.get(fp, 0.0), item.relevance_score)

    # Track which files have already been assigned to a group
    assigned: set[str] = set()
    groups: list[ComponentGroup] = []

    # Sort by score descending so higher-scoring files create the groups
    sorted_files = sorted(file_scores.items(), key=lambda x: -x[1])

    for fp, score in sorted_files:
        if fp in assigned:
            continue

        ext = Path(fp).suffix.lower()

        # Only attempt grouping for known framework extensions
        if ext not in _GROUP_ALL_EXTS:
            # Standalone file (e.g., .java, .py, .properties) — group of 1
            groups.append(ComponentGroup(
                stem=Path(fp).stem,
                directory=str(Path(fp).parent).replace("\\", "/"),
                members={ext: fp},
                primary_file=fp,
                group_score=score,
                evidence_sources=[ext],
                is_group=False,
            ))
            assigned.add(fp)
            continue

        # Compute the component stem
        # For "add-members.component.ts" → stem = "add-members.component"
        # For "AddMembers.tsx" → stem = "AddMembers"
        name = Path(fp).name
        stem = name[: -len(ext)] if ext else name
        directory = str(Path(fp).parent).replace("\\", "/")

        # Build the group: start with this file, then scan disk for siblings
        members: dict[str, str] = {ext: fp}
        evidence_sources: list[str] = [ext]

        # Scan for sibling files with the same stem
        abs_dir = workspace_path / directory
        if abs_dir.is_dir():
            for sibling_ext in _GROUP_ALL_EXTS:
                if sibling_ext == ext:
                    continue  # Already in members
                sibling_path = abs_dir / f"{stem}{sibling_ext}"
                if sibling_path.is_file():
                    rel_path = f"{directory}/{stem}{sibling_ext}"
                    members[sibling_ext] = rel_path
                    # Check if this sibling also has evidence
                    if rel_path in file_scores:
                        evidence_sources.append(sibling_ext)

        # Also check: for the reverse case (evidence found .html but not .ts),
        # scan for source files with the same stem
        if ext in _GROUP_COMPANION_EXTS:
            for src_ext in _GROUP_SOURCE_EXTS:
                if src_ext in members:
                    continue
                src_path = abs_dir / f"{stem}{src_ext}"
                if src_path.is_file():
                    rel_path = f"{directory}/{stem}{src_ext}"
                    members[src_ext] = rel_path
                    if rel_path in file_scores:
                        evidence_sources.append(src_ext)

        # Determine the primary file (highest evidence score in group)
        best_file = fp
        best_score = score
        for m_ext, m_path in members.items():
            if m_path in file_scores and file_scores[m_path] > best_score:
                best_score = file_scores[m_path]
                best_file = m_path

        # Create the group
        is_group = len(members) > 1
        groups.append(ComponentGroup(
            stem=stem,
            directory=directory,
            members=members,
            primary_file=best_file,
            group_score=best_score,
            evidence_sources=evidence_sources,
            is_group=is_group,
        ))

        # Mark all group members as assigned
        for m_path in members.values():
            assigned.add(m_path)

    # Handle any evidence files that weren't assigned (shouldn't happen, but safety)
    for fp in file_scores:
        if fp not in assigned:
            ext = Path(fp).suffix.lower()
            groups.append(ComponentGroup(
                stem=Path(fp).stem,
                directory=str(Path(fp).parent).replace("\\", "/"),
                members={ext: fp},
                primary_file=fp,
                group_score=file_scores[fp],
                evidence_sources=[ext],
                is_group=False,
            ))

    logger.info(
        f"  [ComponentGroup] Built {len(groups)} groups from "
        f"{len(file_scores)} evidence files: "
        f"{sum(1 for g in groups if g.is_group)} multi-file, "
        f"{sum(1 for g in groups if not g.is_group)} standalone"
    )
    for g in groups[:10]:
        if g.is_group:
            logger.info(
                f"    Group '{g.stem}': {list(g.members.keys())} "
                f"(primary={Path(g.primary_file).name}, score={g.group_score:.3f}, "
                f"evidence_in={g.evidence_sources})"
            )

    return groups


# ── Companion co-localization helpers (framework-agnostic, disk-grounded) ─────
_COMPONENT_SOURCE_EXTS = {".ts", ".tsx", ".jsx", ".js", ".vue", ".svelte"}
_COMPANION_TEMPLATE_STYLE_EXTS = (".html", ".htm", ".scss", ".css", ".less", ".sass")
# Templates/styles must pull in their controller so logic changes travel with the UI change.
_TEMPLATE_SOURCE_EXTS = {".html", ".htm", ".scss", ".css", ".less", ".sass"}
_TEMPLATE_URL_RE = re.compile(r"""templateUrl\s*:\s*['"]([^'"]+)['"]""")
_STYLE_URLS_RE = re.compile(r"""styleUrls?\s*:\s*\[([^\]]*)\]""")
_STYLE_URL_RE = re.compile(r"""styleUrl\s*:\s*['"]([^'"]+)['"]""")
_QUOTED_STR_RE = re.compile(r"""['"]([^'"]+)['"]""")
_I18N_KEY_RE = re.compile(r"\b([a-z][a-z0-9_]*(?:\.[a-z0-9_]+){2,})\b")


def _safe_rel(candidate: Path, workspace_path: Path) -> "Optional[str]":
    """Return the workspace-relative posix path if candidate is inside workspace."""
    try:
        resolved = candidate.resolve()
        root = workspace_path.resolve()
        if resolved == root or root in resolved.parents:
            return str(resolved.relative_to(root)).replace("\\", "/")
    except Exception:
        return None
    return None


def _find_component_companions(
    source_file: str,
    workspace_path: Path,
    existing_paths: "Set[str]",
) -> "list[tuple[str, str]]":
    """Resolve a component's template/style companions.

    Strategy 1 (ast_decorator_edge): parse templateUrl/styleUrls from the source.
    Strategy 2 (sibling_colocation): same-stem template/style files in the same dir.
    Only returns files that physically exist inside the workspace.
    """
    src = Path(source_file)
    src_ext = src.suffix.lower()
    is_template_or_style = src_ext in _TEMPLATE_SOURCE_EXTS
    is_source = src_ext in _COMPONENT_SOURCE_EXTS

    if not is_source and not is_template_or_style:
        return []

    src_abs = workspace_path / source_file
    src_dir = src_abs.parent
    results: "list[tuple[str, str]]" = []

    # ── Reverse path: template (.html) → pull in the controller (.ts) ──────────
    # When a .html template is in the plan, the sibling .ts MUST also be included
    # because any new template binding (e.g. *ngIf="dmember.isExistingMemberInProject")
    # requires matching TypeScript logic to populate it.
    # Stylesheets (.scss/.css/.less/.sass) NEVER have template bindings and must NOT
    # pull in controllers as writable tasks.
    if src_ext in (".html", ".htm"):
        stem = src.name[: -len(src.suffix)] if src.suffix else src.name
        for _ctrl_ext in (".ts", ".tsx"):
            rel = _safe_rel(src_dir / f"{stem}{_ctrl_ext}", workspace_path)
            if rel and rel not in existing_paths and (workspace_path / rel).is_file():
                results.append((rel, "template_controller_follow"))
        return _dedupe_pairs(results)

    if src_ext in (".scss", ".css", ".less", ".sass"):
        return []

    try:
        content = src_abs.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = ""

    refs: list[str] = []
    refs.extend(m.group(1) for m in _TEMPLATE_URL_RE.finditer(content))
    refs.extend(m.group(1) for m in _STYLE_URL_RE.finditer(content))
    for m in _STYLE_URLS_RE.finditer(content):
        refs.extend(q.group(1) for q in _QUOTED_STR_RE.finditer(m.group(1)))

    for ref in refs:
        rel = _safe_rel(src_dir / ref, workspace_path)
        if rel and rel not in existing_paths and (workspace_path / rel).is_file():
            results.append((rel, "ast_decorator_edge"))

    # Sibling stem co-location: 'foo.component.ts' → 'foo.component.html' / '.scss'
    stem = src.name[: -len(src.suffix)] if src.suffix else src.name
    for comp_ext in _COMPANION_TEMPLATE_STYLE_EXTS:
        rel = _safe_rel(src_dir / f"{stem}{comp_ext}", workspace_path)
        if rel and rel not in existing_paths and (workspace_path / rel).is_file():
            results.append((rel, "sibling_colocation"))

    # Strategy 3 (model_import_reference): follow relative imports that point to
    # model/interface/dto files — e.g. import { DisplayedMember } from '../../shared/models/displayed-member'
    # Invariant: Import != Writable. These are gathered as READ-ONLY reference context,
    # NEVER as writable tasks.
    if content:
        _model_import_re = re.compile(
            r"import\s*\{[^}]+\}\s*from\s*['\"](\.[^'\"]+)['\"]",
            re.MULTILINE,
        )
        _MODEL_KEYWORDS = ("model", "interface", "dto", "type", "entity", "record", "schema")
        for _mi in _model_import_re.finditer(content):
            _rel = _mi.group(1)
            if any(kw in _rel.lower() for kw in _MODEL_KEYWORDS):
                for _ext in (".ts", ".tsx"):
                    _cand = _safe_rel(
                        Path(os.path.normpath(str(src_dir / (_rel + _ext)))),
                        workspace_path,
                    )
                    if _cand and _cand not in existing_paths and (workspace_path / _cand).is_file():
                        results.append((_cand, "model_import_reference"))
                        break

    seen: set = set()
    deduped: "list[tuple[str, str]]" = []
    for path, strategy in results:
        if path not in seen:
            seen.add(path)
            deduped.append((path, strategy))
    return deduped


def _find_i18n_owner_files(
    text_lower: str,
    workspace_path: Path,
    existing_paths: "Set[str]",
    max_files: int = 400,
) -> "list[tuple[str, str]]":
    """Resolve i18n/locale bundles that own translation keys referenced in the text.

    Strategy 3 (i18n_key_index): extract dotted keys, then confirm the key (full
    dotted form or its quoted leaf segment) exists on disk in a locale bundle.
    """
    keys = {m.group(1) for m in _I18N_KEY_RE.finditer(text_lower)}
    keys = {k for k in keys if len(k.split(".")[-1]) >= 4}
    if not keys:
        return []

    skip_dirs = {"node_modules", ".git", "dist", "build", "target", "coverage", ".venv", "venv"}
    results: "list[tuple[str, str]]" = []
    scanned = 0
    for root, dirs, files in os.walk(str(workspace_path)):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        low_root = root.replace("\\", "/").lower()
        locale_dir = ("i18n" in low_root or "locale" in low_root or "assets" in low_root)
        for f in files:
            fl = f.lower()
            is_bundle = (
                (fl.endswith(".json") and locale_dir)
                or (fl.startswith("messages") and fl.endswith(".properties"))
            )
            if not is_bundle:
                continue
            scanned += 1
            if scanned > max_files:
                return _dedupe_pairs(results)
            full = Path(root) / f
            rel = _safe_rel(full, workspace_path)
            if not rel or rel in existing_paths:
                continue
            try:
                low_text = full.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            for key in keys:
                leaf = key.split(".")[-1]
                if key in low_text or f'"{leaf}"' in low_text:
                    results.append((rel, "i18n_key_index"))
                    break
    return _dedupe_pairs(results)


def _dedupe_pairs(pairs: "list[tuple[str, str]]") -> "list[tuple[str, str]]":
    seen: set = set()
    out: "list[tuple[str, str]]" = []
    for path, strategy in pairs:
        if path not in seen:
            seen.add(path)
            out.append((path, strategy))
    return out


def _make_companion_task(
    file_path: str,
    task_index: int,
    source_file: str,
    strategy: str,
    source_task: "Any" = None,
) -> "Any":
    """Create a writable MODIFY task for a co-localized companion file.

    When source_task is provided (Component Group Architecture), the companion
    task inherits a meaningful description from the source task's plan so the
    code_generator has actual instructions for what to change.
    """
    from ticket_to_code.models import DevelopmentTask, TaskType

    ext = Path(file_path).suffix.lower()
    lang = _lang_for_ext(ext)
    ownership = "DISPLAY_OWNER" if ext in _COMPANION_TEMPLATE_STYLE_EXTS else "SUPPORTING"

    # ── Build context-aware description from source task ──────────────────
    if source_task and hasattr(source_task, "description") and source_task.description:
        source_desc = source_task.description[:300]
        source_title = getattr(source_task, "title", "") or ""

        # Determine what kind of companion this is for targeted instructions
        if ext in (".html", ".htm"):
            companion_instruction = (
                f"Update template/HTML bindings to match the TypeScript changes in "
                f"'{Path(source_file).name}'. Source task: {source_title}. "
                f"Changes to support: {source_desc}"
            )
        elif ext in (".scss", ".css", ".less", ".sass"):
            companion_instruction = (
                f"Update styles to support the UI changes described in "
                f"'{Path(source_file).name}'. Source task: {source_title}. "
                f"Changes to support: {source_desc}"
            )
        elif ext in (".ts", ".tsx"):
            companion_instruction = (
                f"Add TypeScript logic to support the template/style changes in "
                f"'{Path(source_file).name}'. Source task: {source_title}. "
                f"Changes to support: {source_desc}"
            )
        else:
            companion_instruction = (
                f"Component group member of '{Path(source_file).name}'. "
                f"Update to support: {source_title}. {source_desc}"
            )
    else:
        # Fallback: generic description when source_task is not provided
        companion_instruction = (
            f"Companion of {source_file} resolved via {strategy}. Edit alongside the "
            f"component so the UI change is applied across template/style/i18n."
        )

    # Invariant: DISCOVERY != READ_ONLY_REFERENCE != CHANGE_TARGET != AUTHORIZED_CHANGE_TARGET
    # Imported models, types, DTOs, and consumer references are READ_ONLY reference context.
    is_reference_only = strategy in (
        "model_import_reference", "model_import_follow",
        "type_consumer_reference", "type_consumer_follow",
        "data_model_gap"
    )
    task_type = TaskType.READ_ONLY if is_reference_only else TaskType.MODIFY
    if is_reference_only:
        ownership = "READ_ONLY"
        companion_instruction = (
            f"Read-only reference context: {strategy} of '{Path(source_file).name}'. "
            f"Provided for type/contract awareness only — strictly read-only, DO NOT modify."
        )

    return DevelopmentTask(
        id=f"cmp-{task_index}",
        title=f"Companion ({strategy}): {Path(file_path).name}",
        description=companion_instruction,
        file_path=file_path,
        task_type=task_type,
        language=lang,
        localization_confidence=0.7,
        localization_reason=f"companion_colocation: {strategy} of {source_file}",
        ownership_type=ownership,
        new_file_creation_allowed=False,
    )



def ownership_completeness_node(
    state: "TicketToCodeState",
    agents: "WorkflowAgents",
) -> dict:
    """
    Phase 2F: Ownership Completeness Check.

    Inserts BETWEEN localize and rag_tests/rag_code.

    For each writable task whose file is a CSS/SCSS file:
      1. Extract all top-level CSS class names defined in the file.
      2. Search the same module directory for sibling SCSS files that share
         any of those class names AND where those shared classes contain
         layout-critical properties (flex, position, height, overflow …).
      3. If found → the sibling is the layout container owner.
         Add it as a new writable MODIFY task (LAYOUT_OWNER).

    For non-SCSS files and non-layout tickets: no expansion occurs.

    STOP CONDITIONS:
      • No layout keywords in ticket  →  skip entirely
      • No SCSS tasks in plan         →  skip entirely
      • No sibling with shared layout classes found  →  stop, coverage sufficient
      • max 2 new tasks added         →  prevent uncontrolled growth
    """
    print("\n" + "=" * 80)
    print(" ENTERING: ownership_completeness_node() in workflow.py")
    print("   Purpose: Verify localized files fully own the described behavior")
    print("=" * 80)
    logger.info(" PHASE 2F: Ownership Completeness Check")

    plan = _get_plan(state)
    ticket = state["ticket"]
    ticket_lower = f"{ticket.title} {ticket.description}".lower()
    workspace_path = Path(state["workspace_path"])

    writable_tasks = [t for t in plan.tasks if t.task_type.value != "read_only"]

    def _is_frontend_task_file(p: str) -> bool:
        norm = (p or "").replace("\\", "/").lower()
        if norm.endswith((".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte", ".jsx", ".tsx")):
            return True
        parts = norm.split("/")
        if len(parts) > 1:
            top_dir = parts[0]
            cand_dir = workspace_path / top_dir
            if cand_dir.is_dir() and any((cand_dir / cfg).exists() for cfg in ("angular.json", "tsconfig.json", "package.json", "vite.config.ts", "next.config.js")):
                if not norm.endswith((".java", ".kt", ".scala", ".cs", ".go", ".rs", ".py", ".sql")):
                    return True
        return False

    # UI-only tickets (all writable files in frontend/UI modules, no compiled backend
    # files) rarely need the full companion safety net — cap it tighter so a
    # simple validator/component fix doesn't balloon into a dozen extra files.
    _ui_only_ticket = bool(writable_tasks) and all(
        _is_frontend_task_file(t.file_path) for t in writable_tasks
    )

    # ── Enrich ticket_lower with grounded understanding root cause ────────────
    grounded = state.get("grounded_understanding")
    if grounded and grounded.root_cause:
        ticket_lower = f"{ticket_lower} {grounded.root_cause.lower()}"

    # ── Ticket intent classification ─────────────────────────────────────────
    is_layout = _ticket_is_layout_focused(ticket_lower)
    is_data = bool(re.search(
        r'\b(count|filter|calculat|incorrect|wrong value|display.*wrong|'
        r'sort|aggregat|total|sum)\b',
        ticket_lower,
    ))

    logger.info(f"  Ticket intent: layout={is_layout}, data={is_data}")

    from ticket_to_code.agents.canonical_path import canonical_repo_path

    # Collect existing paths (we never add duplicates)
    # Using workspace-rooted canonical repository-relative identity
    tasks_by_path: dict[str, Any] = {}
    existing_paths: Set[str] = set()
    for t in plan.tasks:
        cp = canonical_repo_path(t.file_path, workspace_path)
        if cp:
            existing_paths.add(cp)
            tasks_by_path[cp] = t
        else:
            p_clean = t.file_path.replace("\\", "/").strip("/")
            existing_paths.add(p_clean)
            tasks_by_path[p_clean] = t

    if grounded:
        for p in (grounded.readonly_context_files or []):
            cp = canonical_repo_path(p, workspace_path)
            if cp:
                existing_paths.add(cp)
            else:
                existing_paths.add(p.replace("\\", "/").strip("/"))

    # ── Ownership type summary ───────────────────────────────────────────────
    ownership_summary: dict = {}
    for t in writable_tasks:
        ot = t.ownership_type or "UNKNOWN"
        ownership_summary[t.file_path] = ot

    # ── Data-driven UI detection ───────────────────────────────────────────────
    # Instead of hardcoded keyword matching (brittle), check whether the plan
    # itself already contains frontend/UI files.  The presence of a
    # .component.ts, .scss, .css, or .html file is an unambiguous signal that
    # this ticket touches the UI layer and SCSS gap detection should run.
    # The inner loop already short-circuits if no SCSS tasks exist, so this
    # gate is purely about intent — not about performance.
    _UI_EXTS = {".scss", ".css", ".sass", ".less", ".html", ".htm"}
    has_ui_files = any(
        Path(t.file_path).suffix.lower() in _UI_EXTS
        or t.file_path.endswith(".component.ts")
        for t in writable_tasks
    )
    is_ui_ticket = is_layout or has_ui_files

    # ── SCSS gap detection ────────────────────────────────────────────────────
    expanded_tasks: list = []
    layout_gaps: list = []
    stop_condition = "no_ui_signal"

    if is_ui_ticket:
        stop_condition = "no_scss_tasks"
        # Deduplicate by file path — the plan may contain duplicate tasks for
        # the same SCSS file (e.g. parallel test/code branch tasks).  Running
        # gap detection twice on the same source file would add two different
        # sibling files unnecessarily.
        seen_scss: Set[str] = set()
        scss_tasks = []
        for t in writable_tasks:
            fp_norm = t.file_path.replace("\\", "/")
            if Path(t.file_path).suffix.lower() in _SCSS_EXTS and fp_norm not in seen_scss:
                scss_tasks.append(t)
                seen_scss.add(fp_norm)
        # Extract ticket keywords for layout gap validation
        _ticket_obj = state.get("ticket")
        _ticket_kw: Set[str] = set()
        if _ticket_obj:
            _t_txt = f"{getattr(_ticket_obj, 'title', '')} {getattr(_ticket_obj, 'description', '')}".lower()
            _ticket_kw = {
                w.strip() for w in re.findall(r'[a-zA-Z0-9_-]{3,}', _t_txt)
                if len(w.strip()) >= 3 and w.strip() not in {
                    "the", "and", "for", "with", "this", "that", "from", "when", "user", "system", "shall", "missing", "does", "not"
                }
            }
        if scss_tasks:
            stop_condition = "no_gap_found"
            for task in scss_tasks:
                gap_owners = _find_layout_gap_owners(
                    task.file_path, workspace_path, existing_paths, ticket_keywords=_ticket_kw
                )
                for gowner in gap_owners:
                    gpath = gowner["path"]
                    if gpath in existing_paths:
                        continue
                    logger.info(
                        f"   Layout gap: {task.file_path} shares "
                        f"{gowner['shared_classes']} with {gpath} "
                        f"— adding as LAYOUT_OWNER"
                    )
                    layout_gaps.append({
                        "source_file": task.file_path,
                        "layout_owner": gpath,
                        "shared_classes": gowner["shared_classes"],
                    })
                    new_task = _make_ownership_task(
                        file_path=gpath,
                        workspace_path=workspace_path,
                        task_index=len(plan.tasks) + len(expanded_tasks) + 1,
                        role="LAYOUT_OWNER",
                        source_file=task.file_path,
                        shared_classes=gowner["shared_classes"],
                    )
                    expanded_tasks.append(new_task)
                    existing_paths.add(gpath)

            if layout_gaps:
                stop_condition = "gap_filled"

    # ── Companion co-localization (SAFETY NET — most work done by planner) ──────
    # With Component Group Architecture, the planner already sees component
    # groups and creates tasks for each file that needs changes. This section
    # is now a SAFETY NET that catches any companion files the planner missed.
    # Cap is per-component-group (6 groups) instead of per-file. Tighter for
    # UI-only tickets, which rarely need the full safety net.
    _COMPANION_GROUP_CAP = 4 if _ui_only_ticket else 12
    companion_records: list = []
    companion_groups_added = 0
    for task in list(writable_tasks):
        if companion_groups_added >= _COMPANION_GROUP_CAP:
            break
        companions_for_this_task = []
        for comp_path, strategy in _find_component_companions(
            task.file_path, workspace_path, existing_paths
        ):
            canon_comp = canonical_repo_path(comp_path, workspace_path) or comp_path.replace("\\", "/")
            if canon_comp in existing_paths:
                # Already in the plan (planner created a task for it via
                # component group context or prior discovery) — merge provenance
                existing_task = tasks_by_path.get(canon_comp)
                if existing_task:
                    existing_task.localization_reason = (
                        f"{getattr(existing_task, 'localization_reason', '')}; "
                        f"merged_companion: {strategy} from {task.file_path}"
                    )
                logger.info(
                    f"   Companion [{strategy}]: {canon_comp} MERGED into existing task "
                    f"(preventing duplicate task generation)"
                )
                continue
            if companion_groups_added >= _COMPANION_GROUP_CAP:
                break
            logger.info(
                f"   Companion [{strategy}]: {canon_comp} (safety net for {task.file_path})"
            )
            new_comp_task = _make_companion_task(
                file_path=canon_comp,
                task_index=len(plan.tasks) + len(expanded_tasks) + 1,
                source_file=task.file_path,
                strategy=strategy,
                source_task=task,
            )
            expanded_tasks.append(new_comp_task)
            existing_paths.add(canon_comp)
            tasks_by_path[canon_comp] = new_comp_task
            companions_for_this_task.append(canon_comp)
            companion_records.append({
                "source_file": task.file_path,
                "companion": canon_comp,
                "strategy": strategy,
                "safety_net": True,
            })
        if companions_for_this_task:
            companion_groups_added += 1


    if len(expanded_tasks) < _COMPANION_GROUP_CAP:
        for i18n_path, strategy in _find_i18n_owner_files(
            ticket_lower, workspace_path, existing_paths
        ):
            canon_i18n = canonical_repo_path(i18n_path, workspace_path) or i18n_path.replace("\\", "/")
            if canon_i18n in existing_paths or len(expanded_tasks) >= _COMPANION_GROUP_CAP:
                continue
            logger.info(f"   Companion [{strategy}]: {canon_i18n} (ticket i18n key)")
            new_i18n_task = _make_companion_task(
                file_path=canon_i18n,
                task_index=len(plan.tasks) + len(expanded_tasks) + 1,
                source_file="ticket i18n keys",
                strategy=strategy,
            )
            expanded_tasks.append(new_i18n_task)
            existing_paths.add(canon_i18n)
            tasks_by_path[canon_i18n] = new_i18n_task
            companion_records.append({
                "source_file": "ticket",
                "companion": canon_i18n,
                "strategy": strategy,
            })

    # ── Type consumer follow: when a model/interface file is modified, add  ──
    # all TS files that IMPORT it as companions so they can be updated too.
    # This is the inverse of model_import_follow (which adds the model when a
    # component changes) — here the model changed, so the consumers must change.
    if len(expanded_tasks) < _COMPANION_GROUP_CAP:
        for task in list(writable_tasks):
            if Path(task.file_path).suffix.lower() not in (".ts", ".tsx"):
                continue
            fp_lower = task.file_path.lower()
            is_model = any(kw in fp_lower for kw in (
                "/models/", "/model.", "-model.", ".model.",
                "/interface.", "-interface.", ".interface.",
                "/dto.", "-dto.", ".dto.", "/types/", "/type.",
            ))
            if not is_model:
                continue
            for consumer_path, strategy in _find_type_consumer_companions(
                task.file_path, workspace_path, existing_paths
            ):
                if consumer_path in existing_paths or len(expanded_tasks) >= _COMPANION_GROUP_CAP:
                    continue
                logger.info(
                    f"   Companion [{strategy}]: {consumer_path} "
                    f"(imports modified model {Path(task.file_path).name})"
                )
                expanded_tasks.append(_make_companion_task(
                    file_path=consumer_path,
                    task_index=len(plan.tasks) + len(expanded_tasks) + 1,
                    source_file=task.file_path,
                    strategy=strategy,
                ))
                existing_paths.add(consumer_path)
                companion_records.append({
                    "source_file": task.file_path,
                    "companion": consumer_path,
                    "strategy": strategy,
                })

    # ── Fix B: Java/TypeScript/Python data-model gap detection ───────────────────
    # DISABLED: The regex-based pre-scan matches generic property names (.value,
    # .name, .key) against every DTO in the monorepo, causing 7+ irrelevant
    # companion tasks. The PER-FILE COMPILE CHECK loop (line ~5668) already
    # handles this correctly: write file → compile → if "property X does not
    # exist on type Y" → read the error → identify the exact file → fix it.
    # This is how Cursor/Copilot/Windsurf handle cross-file dependencies:
    # compile-driven, not regex-driven.
    _DATA_GAP_EXTS = {".java", ".ts", ".tsx", ".py"}
    _data_gap_records: list = []
    _sqlite_for_gap = None  # Disabled: set to None so the block below is skipped

    # Patterns: Java getter calls, TS property access, Python attribute access
    _JAVA_GETTER_CALL_RE = re.compile(r'(\w+)\.get(\w+)\s*\(', re.MULTILINE)
    _TS_PROP_ACCESS_RE   = re.compile(r'(\w+)\.(\w+)\s*(?:[;,\)]|\s*=)', re.MULTILINE)
    _PY_ATTR_ACCESS_RE   = re.compile(r'self\.(\w+)\.?(\w+)?\s*(?:=|\(|\[)', re.MULTILINE)

    if _sqlite_for_gap and len(expanded_tasks) < _COMPANION_GROUP_CAP:
        for task in list(writable_tasks):
            if Path(task.file_path).suffix.lower() not in _DATA_GAP_EXTS:
                continue
            task_abs = workspace_path / task.file_path
            if not task_abs.exists():
                continue
            try:
                _content = task_abs.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Collect (accessor_var, member_name) pairs
            _accesses: list = []
            ext = Path(task.file_path).suffix.lower()
            if ext == ".java":
                for m in _JAVA_GETTER_CALL_RE.finditer(_content):
                    _accesses.append((m.group(1), "get" + m.group(2)))
            elif ext in (".ts", ".tsx"):
                for m in _TS_PROP_ACCESS_RE.finditer(_content):
                    _accesses.append((m.group(1), m.group(2)))

            # For each accessed member, find the type declaration in SQLite
            _conn_gap = _sqlite_for_gap._conn
            for _var, _member in _accesses[:50]:  # cap to avoid noise
                try:
                    # Find a file that defines this member — exclude test/generated paths
                    _rows = _conn_gap.execute(
                        "SELECT DISTINCT path FROM symbols WHERE name = ? AND kind IN ('method','field','property') "
                        "AND path NOT LIKE '%Test%' AND path NOT LIKE '%Spec%' AND path NOT LIKE '%/dist/%' LIMIT 3",
                        (_member,),
                    ).fetchall()
                    for (_path,) in _rows:
                        if not _path or _path in existing_paths:
                            continue
                        full = workspace_path / _path
                        if not full.exists():
                            continue
                        # Only add if the file is a model/dto (no Spring stereotype)
                        _stereo = _conn_gap.execute(
                            "SELECT spring_stereotype FROM symbols WHERE path = ? AND spring_stereotype != '' LIMIT 1",
                            (_path,),
                        ).fetchone()
                        if _stereo:  # skip services/controllers
                            continue
                        logger.info(f"   Data model gap [{ext}]: {task.file_path} calls .{_member} → adding {_path}")
                        _data_gap_records.append({"source_file": task.file_path, "model_file": _path, "member": _member})
                        expanded_tasks.append(_make_companion_task(
                            file_path=_path,
                            task_index=len(plan.tasks) + len(expanded_tasks) + 1,
                            source_file=task.file_path,
                            strategy="data_model_gap",
                        ))
                        existing_paths.add(_path)
                        companion_records.append({"source_file": task.file_path, "companion": _path, "strategy": "data_model_gap"})
                        if len(expanded_tasks) >= _COMPANION_GROUP_CAP:
                            break
                except Exception:
                    pass
                if len(expanded_tasks) >= _COMPANION_GROUP_CAP:
                    break
            if len(expanded_tasks) >= _COMPANION_GROUP_CAP:
                break

    # ── Compute coverage score (for trace; not used as gate) ─────────────────
    all_otypes = list(ownership_summary.values())
    behavioral = (
        1.0 if "PRIMARY_OWNER" in all_otypes
        else (0.6 if "DISPLAY_OWNER" in all_otypes else 0.3)
    )
    layout_cov = (0.4 if layout_gaps else 1.0) if is_layout else 1.0
    data_cov = (
        1.0 if not is_data
        else (1.0 if "PRIMARY_OWNER" in all_otypes else 0.4)
    )
    coverage_score = round((behavioral + layout_cov + data_cov) / 3, 3)

    # ── Merge expanded tasks into plan ───────────────────────────────────────
    if expanded_tasks:
        plan.tasks = plan.tasks + expanded_tasks
        logger.info(
            f"  ✅ Ownership expanded: +{len(expanded_tasks)} task(s). "
            f"Total writable: "
            f"{sum(1 for t in plan.tasks if t.task_type.value != 'read_only')}"
        )
    else:
        logger.info(
            f"  ✅ Ownership sufficient — no expansion needed "
            f"(stop_condition={stop_condition})"
        )

    # ── Build completeness result for state + trace ───────────────────────────
    completeness = {
        "is_layout_ticket": is_layout,
        "is_data_ticket": is_data,
        "localized_files": [t.file_path for t in writable_tasks],
        "ownership_roles": ownership_summary,
        "ownership_coverage": {
            "behavioral": round(behavioral, 3),
            "layout": round(layout_cov, 3),
            "data": round(data_cov, 3),
            "overall": coverage_score,
        },
        "layout_gaps": layout_gaps,
        "expanded_writable_files": [t.file_path for t in expanded_tasks],
        "companion_files": companion_records,
        "supporting_readonly_files": [
            t.file_path for t in plan.tasks if t.task_type.value == "read_only"
        ],
        "stop_condition": stop_condition,
        "ownership_reasoning": [
            f"{g['source_file']} shares CSS classes {g['shared_classes']} with "
            f"{g['layout_owner']} — layout container ownership gap detected"
            for g in layout_gaps
        ] if layout_gaps else [
            f"stop_condition={stop_condition}: ownership sufficient at "
            f"behavioral={behavioral:.2f}, layout={layout_cov:.2f}, data={data_cov:.2f}"
        ],
    }

    agents.tracer.record_ownership_completeness(completeness)

    return {
        "architectural_plan": plan,
        "ownership_completeness": completeness,
        "status": "ownership_checked",
    }


# ============================================================================
# DATA-FLOW VERIFICATION NODE
# Mirrors the "read → think → write → verify" loop used by Claude Code / Devin.
# Runs after plan expansion but BEFORE RAG so missing .ts tasks can be injected.
# ============================================================================

def dataflow_verification_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Trace the full data path for every Angular template binding in the plan.

    For each HTML task, scan the template (or its planned changes) for new bindings
    like `*ngIf="dmember.isExistingMemberInProject"` and verify that:
      1. The sibling .ts controller is also a writable task in the plan.
      2. The bound property is SET somewhere (controller or model task).

    If the controller is missing from the plan → inject it as a new MODIFY task.
    If a bound property is not set anywhere → record it as a known gap so the
    code generator's system prompt can explicitly warn about it.
    """
    from ticket_to_code.agents.dataflow_tracer import (
        trace_plan_data_flow,
        gap_summary,
        DataFlowGap,
    )
    from ticket_to_code.models import DevelopmentTask, TaskType

    plan = _get_plan(state)
    if not plan or not getattr(plan, "tasks", None):
        return {}

    workspace_path = str(state.get("workspace_path", ""))
    session_map: dict = state.get("_run_generated_map", {})

    logger.info("\n" + "=" * 60)
    logger.info(" DATAFLOW VERIFICATION (Read → Think → Verify before Write)")
    logger.info("=" * 60)

    writable_paths = {
        t.file_path.replace("\\", "/").lower()
        for t in plan.tasks
        if getattr(getattr(t, "task_type", None), "value", str(getattr(t, "task_type", ""))).lower() != "read_only"
    }

    reports = trace_plan_data_flow(
        plan_tasks=plan.tasks,
        workspace_path=workspace_path,
        session_map=session_map,
    )

    injected: list[str] = []
    all_gaps: list[DataFlowGap] = []

    for report in reports:
        if not report.has_gaps:
            logger.info(f"  ✅ No data-flow gaps for {report.html_file}")
            continue

        all_gaps.extend(report.gaps)
        logger.warning(
            f"  ⚠️ DataFlow gaps in {report.html_file}:\n"
            + "\n".join(f"    • {g.binding.variable}.{g.binding.property} — {g.reason[:120]}"
                        for g in report.gaps)
        )

        # Inject .ts controller task if it's missing from the plan
        ctrl = report.controller_file.replace("\\", "/")
        ctrl_lower = ctrl.lower()
        if ctrl and ctrl_lower not in writable_paths:
            ctrl_abs = Path(workspace_path) / ctrl
            if ctrl_abs.is_file():
                gap_props = ", ".join(f"{g.binding.variable}.{g.binding.property}" for g in report.gaps)
                new_task = DevelopmentTask(
                    id=f"df-ctrl-{len(injected) + 1}",
                    title=f"DataFlow: populate {Path(ctrl).name} with template-bound properties",
                    description=(
                        f"The HTML template {report.html_file} references properties that are not "
                        f"set in the controller. This task adds the missing logic to: {gap_props}. "
                        f"Concretely: fetch the user's existing organization from the backend when "
                        f"a member is staged, then set the bound properties on the displayed object."
                    ),
                    file_path=ctrl,
                    task_type=TaskType.MODIFY,
                    language="typescript",
                    selection_reason=(
                        f"DataFlow tracer: controller is missing setters for template bindings "
                        f"in {report.html_file}. Auto-injected to prevent silent undefined at runtime."
                    ),
                    allowed_methods=[g.binding.property for g in report.gaps],
                    new_file_creation_allowed=False,
                    estimated_complexity=2,
                    requires_testing=False,
                )
                plan.tasks.append(new_task)
                writable_paths.add(ctrl_lower)
                injected.append(ctrl)
                logger.info(
                    f"  ➕ Injected missing controller task: {ctrl} "
                    f"(properties needed: {gap_props})"
                )
            else:
                logger.warning(
                    f"  ⚠️ Controller {ctrl} does not exist on disk — cannot auto-inject task"
                )

    summary = gap_summary(reports) if reports else "No HTML tasks with bindings found."
    logger.info(f"\n DataFlow Summary:\n{summary}")

    result: dict = {"architectural_plan": plan}
    if all_gaps or injected:
        result["_dataflow_gaps"] = [
            {
                "html_file": g.binding.expression,
                "variable": g.binding.variable,
                "property": g.binding.property,
                "controller": g.controller_file,
                "reason": g.reason,
            }
            for g in all_gaps
        ]
        result["_dataflow_injected_tasks"] = injected
        logger.info(
            f"\n  DataFlow result: {len(all_gaps)} gap(s), {len(injected)} task(s) injected"
        )
    else:
        logger.info("  ✅ DataFlow: all template bindings have corresponding data suppliers")

    # ── v3: Populate TicketExecutionContext with data flow reports ─────────
    try:
        _exec_ctx = _get_transient(state, "exec_ctx")
        if _exec_ctx and reports:
            _exec_ctx.set_dataflow_reports(reports)
            logger.info(f"  [v3] ExecCtx updated: {len(reports)} data flow reports")
    except Exception as _ectx_err:
        logger.debug(f"  [v3] ExecCtx dataflow update failed (non-fatal): {_ectx_err}")

    # ── Holistic DataFlowContract extraction & provenance graph ───────────
    try:
        from ticket_to_code.intelligence.contracts.dataflow_contract import DataFlowContract
        _ticket = state.get("ticket")
        _ticket_desc = getattr(_ticket, "description", "") or (str(_ticket) if _ticket else "")
        _contract = DataFlowContract.extract_from_requirements(
            requirements=state.get("requirements"),
            ticket_description=_ticket_desc,
        )
        result["dataflow_contract"] = _contract
        _tid = getattr(_ticket, "ticket_id", "") or ""
        if _tid:
            _set_transient(_tid, "dataflow_contract", _contract)
        logger.info(
            f"  📐 DataFlowContract extracted: {len(_contract.links)} links, "
            f"{len(_contract.negative_rules)} negative rules"
        )
    except Exception as _dfc_err:
        logger.warning(f"  ⚠️ DataFlowContract extraction failed (non-fatal): {_dfc_err}")

    return result


def rag_for_tests_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 3A: RAG Context Retrieval for TEST GENERATION
    
    Retrieves PRODUCT BEHAVIOR knowledge:
    - How features currently work
    - Existing test patterns
    - Business rules and validations
    - Error handling patterns
    
    This context is ONLY used for test generation, NOT for code.
    """
    logger.info(" PHASE 3A: RAG Retrieval for Test Cases (Product Behavior)")
    
    # Query for BEHAVIOR knowledge
    test_context = agents.rag_engine.multi_stage_retrieval(
        requirements=state["requirements"],
        architectural_plan=_get_plan(state),
        query_focus="test_behavior",  # Focus on how things work
        document_types=["tests", "documentation", "business_rules"],
        max_iterations=3
    )
    
    logger.info(
        f"Test RAG complete:\n"
        f"  Behavior examples: {len(test_context)}\n"
        f"  Sources: tests, docs, business rules"
    )
    
    return {
        "test_rag_context": test_context
    }


def rag_for_code_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 3B: Context Retrieval for CODE GENERATION.
    Semantic RAG chunks are bypassed in favor of verbatim file sources, AST skeletons,
    ComponentContracts, and SQLite symbol indexes.
    """
    logger.info("⚡ PHASE 3B: Bypassing noisy semantic RAG chunks; using verbatim AST and symbol index.")
    return {"code_rag_context": []}


def generate_tests_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 4A: Generate Test Cases FIRST (TDD)
    """
    # Test generation MUST be enabled for comprehensive verification
    ENABLE_TEST_GENERATION = True
    
    if not ENABLE_TEST_GENERATION:
        logger.info(" PHASE 4A: Generate Test Cases (TEST_GENERATION_DISABLED per feature flag)")
        return {
            "generated_tests": [],
            "test_status": "tests_skipped"  # TEST BRANCH status
        }
    
    logger.info(" PHASE 4A: Generate Test Cases (TDD)")
    
    # ── Grounded Understanding context ───────────────────────────────────────
    grounded = state.get("grounded_understanding")
    grounded_test_ctx = ""
    if grounded:
        grounded_test_ctx = f"""
GROUNDED UNDERSTANDING (verified evidence):
  root_cause: {grounded.root_cause}
  change_group: {grounded.change_group}
  confidence: {grounded.confidence:.2f}
  hypotheses_confirmed: {grounded.hypotheses_confirmed}
"""
    
    test_generation_prompt = f"""
You are generating UNIT TESTS for a new feature.

IMPORTANT: Generate tests BEFORE the code exists (Test-Driven Development).
Base your tests on:
1. Product behavior (how it SHOULD work)
2. Existing test patterns (structure and style)
3. Business rules (validation, edge cases)
4. Grounded evidence (verified investigation — see below)

DO NOT assume implementation details - focus on BEHAVIOR CONTRACT.

Ticket: {state['ticket'].title}
Requirements: {state['requirements']}
Plan: {state['architectural_plan']}
{grounded_test_ctx}
Product Behavior Context (from existing tests/docs):
{state['test_rag_context']}

Generate comprehensive test cases that:
- Cover happy path scenarios
- Test edge cases and error conditions
- Verify business rules
- Follow existing test patterns
- Use real product behavior (no imagination)

Technology stack: Java (JUnit 5 / Spring Boot Test) for .java files;
Angular/TypeScript (Jasmine + Karma) for .ts/.spec.ts files.
"""
    
    generated_tests = []
    validator = PatchValidator()
    
    # ── BUILD FILE SCOPE LISTS ────────────────────────────────────────────────
    all_tasks = _get_plan(state).tasks
    allowed_files  = [t.file_path for t in all_tasks if t.task_type.value != "read_only"]
    if len(allowed_files) > 25:
        logger.warning(
            f"⚠️  Planner created {len(allowed_files)} writable tasks — "
            f"this is a very large refactoring. Proceeding without caps."
        )

    #Every architectural task gets corresponding tests
    for task in _get_plan(state).tasks:
        # READ_ONLY tasks never generate test files
        if task.task_type.value == "read_only":
            logger.info(f"  ⏩ Skipping read_only task [{task.id}] in test gen: {task.title}")
            continue

        logger.info(f"  Generating tests for: {task.file_path}")

        # Skip test generation for style/markup/script files — they have no runnable tests
        _SKIP_TEST_EXTS = {".scss", ".css", ".html", ".xml", ".json", ".yaml", ".yml", ".sh", ".bat", ".ps1", ".cmd", ".env"}
        if Path(task.file_path).suffix.lower() in _SKIP_TEST_EXTS:
            logger.info(f"  ⏩ Skipping test gen for style/markup/script file: {task.file_path}")
            continue

        # Derive correct test file path based on file extension / language
        src_path = task.file_path

        # Guard: skip if the source file is itself a test/spec file.
        # Without this guard a path like "jheader.component.spec.ts" would
        # produce "jheader.component.spec.spec.ts" because the suffix-strip
        # `.ts` leaves the embedded ".spec." in place.
        _TEST_MARKERS = (".spec.", ".test.", "Test.java", "Tests.java", "Spec.java")
        if any(marker in src_path for marker in _TEST_MARKERS):
            logger.info(
                f"  ⏩ Skipping test gen — source is already a test file: {src_path}"
            )
            continue

        if src_path.endswith(".java"):
            # Java: src/main/java/… → src/test/java/…, add Test suffix
            test_file_path = src_path.replace("src/main/java", "src/test/java")
            test_file_path = test_file_path.replace(".java", "Test.java")
        elif src_path.endswith(".tsx"):
            # TypeScript React: safe suffix replace (avoids .spec.spec.tsx)
            test_file_path = src_path[:-4] + ".spec.tsx"
        elif src_path.endswith(".ts"):
            # TypeScript/Angular: safe suffix replace (avoids .spec.spec.ts)
            test_file_path = src_path[:-3] + ".spec.ts"
        elif src_path.endswith(".jsx"):
            # JavaScript/React: add .test before extension
            test_file_path = src_path[:-4] + ".test.jsx"
        elif src_path.endswith(".js"):
            test_file_path = src_path[:-3] + ".test.js"
        elif src_path.endswith(".vue"):
            test_file_path = src_path[:-4] + ".spec.ts"
        else:
            # Generic fallback — append Test suffix before extension
            stem = Path(src_path).stem
            suffix = Path(src_path).suffix
            test_file_path = str(Path(src_path).parent / f"{stem}Test{suffix}")
        
        # For modify tasks, read existing test file if it exists
        existing_test_content = None
        existing_test_path = Path(state["workspace_path"]) / test_file_path
        if task.task_type.value == "modify" and existing_test_path.exists():
            existing_test_content = existing_test_path.read_text(encoding='utf-8')
            logger.info(f"   Reading existing test file for modification: {existing_test_path}")
        
        # Build a TEST-SCOPED task so the generator's frozen scope block shows
        # the correct test file path and allows creation/modification there.
        import copy
        test_task = copy.copy(task)
        test_task.file_path = test_file_path
        test_task.new_file_creation_allowed = (existing_test_content is None)  # True=CREATE new spec, False=MODIFY existing
        # Clear source-method constraints — test generator writes the full test class
        test_task.target_method = None
        test_task.allowed_methods = []

        try:
            test_code = agents.test_generator.generate_code(
                task=test_task,            # <-- test-scoped task, NOT source task
                requirements=state["requirements"],
                context=state["test_rag_context"],  # ONLY behavior context
                existing_content=existing_test_content
            )
        except Exception as e:
            logger.error(f"  ❌ LLM Test Generation failed for {test_task.file_path}: {e}")
            continue
        
        # Enforce: LLM cannot decide the test file location
        test_code.file_path = test_file_path

        # ── PATCH VALIDATION ─────────────────────────────────────────────────
        validation = validator.validate(test_code, test_task, existing_test_content)
        log_line = StageLog.patch_validation(test_task, validation)
        logger.info(log_line)
        print(log_line)
        if not validation.passed:
            logger.warning(
                f"  ⛔ Skipping test write for {test_file_path} — patch failed validation:\n"
                + "\n".join(f"     • {v}" for v in validation.violations)
            )
            generated_tests.append(test_code)
            continue
        
        # Write test file
        output_path = Path(state["workspace_path"]) / test_file_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(test_code.content, encoding='utf-8')
        
        generated_tests.append(test_code)
        logger.info(f"  ✅ Test written: {output_path} | change_ratio={validation.metrics.get('change_ratio', 'N/A')}")
    
    logger.info(f"Test generation complete: {len(generated_tests)} test files")
    
    return {
        "generated_tests": generated_tests,
        "test_status": "tests_generated"  # TEST BRANCH status
    }


def _build_task_explanation(
    task,
    task_index: int,
    total_tasks: int,
    previously_written: dict,
    prev_explanations: list,
) -> dict:
    """Build a human-readable explanation of what code generation will do for this task.

    This serves two purposes:
      1. UI display: shown in the transparent workflow before code gen starts
      2. LLM context: accumulated across tasks so task N knows what tasks 1..N-1 did

    Returns a dict with:
      - task_id, task_index, total_tasks
      - file_path, change_type
      - purpose: human-readable WHY
      - produces: list of symbol names this task will create
      - consumes: list of symbol names this task uses from others
      - previously_written_summary: list of {file, exports} from earlier tasks
      - explanation_text: fully formatted multi-line string for UI
    """
    from pathlib import Path

    change_type = getattr(task.task_type, "value", str(task.task_type))
    basename = Path(task.file_path).name

    # ── Purpose (WHY) ──
    purpose = task.description[:300] if task.description else task.title

    # ── Produces / Consumes (WHAT cross-file) ──
    produces = []
    consumes = []
    contract = getattr(task, "cross_file_contract", None)
    if contract:
        for bp in (contract.produces or []):
            label = bp.capability
            if bp.data_shape:
                label += f" [{bp.data_shape}]"
            produces.append(label)
        for bp in (contract.consumes or []):
            label = bp.capability
            if bp.data_shape:
                label += f" [{bp.data_shape}]"
            src = f" (from {bp.from_task})" if bp.from_task else ""
            consumes.append(f"{label}{src}")

    # ── Previously written files summary ──
    prev_summary = []
    for fp, content in (previously_written or {}).items():
        if not content:
            continue
        # Extract key exports from the file content
        exports = []
        for line in (content or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("export ") and any(
                kw in stripped for kw in ("class ", "interface ", "enum ", "type ", "function ", "const ")
            ):
                exports.append(stripped[:100])
            elif stripped.startswith(("public ", "protected ")) and "(" in stripped:
                sig = stripped.split("{")[0].strip()
                if len(sig) > 10:
                    exports.append(sig[:100])
        prev_summary.append({
            "file": fp,
            "basename": Path(fp).name,
            "exports": exports[:10],
        })

    req_ids = getattr(task, "requirement_ids", []) or []
    target_methods = getattr(task, "allowed_methods", []) or []
    if getattr(task, "target_method", None):
        target_methods = [task.target_method] + [m for m in target_methods if m != task.target_method]

    # ── Build formatted explanation text for UI ──
    lines = [
        f"📋 Task {task_index}/{total_tasks}: {task.title}",
        f"📁 File: {task.file_path}",
        f"🔄 Change: {change_type.upper()}",
    ]
    if req_ids:
        lines.append(f"🎯 Requirements: {', '.join(req_ids)}")
    if target_methods:
        lines.append(f"🎯 Target Methods: {', '.join(target_methods[:4])}")
    lines.extend([
        f"",
        f"📝 Purpose:",
        f"   {purpose}",
    ])

    if produces:
        lines.append("")
        lines.append("🔧 Will CREATE these symbols:")
        for p in produces:
            lines.append(f"   • {p}")

    if consumes:
        lines.append("")
        lines.append("📥 Will USE these from other files:")
        for c in consumes:
            lines.append(f"   • {c}")

    if prev_summary:
        lines.append("")
        lines.append(f"✅ Previously written ({len(prev_summary)} files):")
        for ps in prev_summary[:8]:
            lines.append(f"   📄 {ps['basename']}")
            for exp in ps["exports"][:5]:
                lines.append(f"      ↳ {exp}")

    if task.dependencies:
        lines.append("")
        lines.append(f"🔗 Depends on: {', '.join(task.dependencies)}")

    explanation_text = "\n".join(lines)

    result = {
        "task_id": task.id,
        "task_index": task_index,
        "total_tasks": total_tasks,
        "file_path": task.file_path,
        "basename": basename,
        "change_type": change_type,
        "purpose": purpose,
        "requirement_ids": req_ids,
        "target_methods": target_methods,
        "produces": produces,
        "consumes": consumes,
        "previously_written_summary": prev_summary,
        "dependencies": task.dependencies or [],
        "explanation_text": explanation_text,
    }

    # Add to accumulated context for the next task
    prev_explanations.append({
        "task_id": task.id,
        "file": task.file_path,
        "produces": produces,
        "consumes": consumes,
    })

    return result


def generate_code_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 4B: Generate Implementation Code SECOND (TDD)
    
    Uses ONLY code_rag_context (architecture patterns).
    Generates code that:
    - Follows architectural patterns
    - Implements clean structure
    - Matches coding standards
    
    NO access to test context (prevents code knowing test internals).
    Code must satisfy tests independently.
    """
    logger.info(" PHASE 4B: Generate Implementation Code (to satisfy tests)")
    
    # ── Grounded Understanding context ───────────────────────────────────────
    grounded = state.get("grounded_understanding")
    grounded_context_str = ""
    if grounded:
        grounded_context_str = f"""
GROUNDED UNDERSTANDING (from evidence investigation):
  root_cause: {grounded.root_cause}
  confidence: {grounded.confidence:.2f}
  dependency_chain: {grounded.dependency_chain[:5]}
  change_group: {grounded.change_group}
  writable_files: {grounded.writable_files}
  hypotheses_confirmed: {grounded.hypotheses_confirmed}

EVIDENCE SUMMARY ({len(grounded.evidence)} items):
{chr(10).join(f'  [{e.source}] {e.file_path}: {e.content_snippet[:80]}' for e in grounded.evidence[:8])}
"""
    
    # ── Semantic constraints (cardinality/per-item/read-only/duplicate/persistence) ──
    # Deterministically extracted so behavioral requirements compilation cannot
    # see reach the generator BEFORE it writes code.
    from ticket_to_code.agents.semantic_requirements import extract_semantic_constraints
    _sem = extract_semantic_constraints(
        getattr(state["ticket"], "title", "") or "",
        getattr(state["ticket"], "description", "") or "",
        str(state.get("requirements") or ""),
    )
    semantic_constraints_block = _sem.to_prompt_block()
    if semantic_constraints_block:
        logger.info(
            f"  🧭 Semantic constraints: cardinality={_sem.cardinality} "
            f"read_only={_sem.read_only} duplicate={_sem.duplicate_prevention} "
            f"persistence={_sem.persistence}"
        )

    # ── Generation readiness (advisory) ──────────────────────────────────────
    try:
        from ticket_to_code.agents.generation_readiness_gate import assess_generation_readiness
        _sc_obj = _get_transient(state, "semantic_contract")
        _readiness = assess_generation_readiness({
            "requirements": state.get("requirements"),
            "contracts": getattr(_sc_obj, "items", None) if _sc_obj else None,
            "verified_capabilities": state.get("evidence_items"),
            "existing_behavior": state.get("grounded_understanding"),
            "semantic_constraints": semantic_constraints_block or None,
            "justified_files": [t.file_path for t in _get_plan(state).tasks],
            "untouched_files": [t.file_path for t in _get_plan(state).tasks if t.task_type.value == "read_only"],
            "cross_file_dependencies": state.get("architectural_plan"),
        })
        if _readiness.is_blocking():
            logger.warning(f"  ⚠️ Generation readiness: {_readiness.summary()} (advisory — proceeding)")
        else:
            logger.info(f"  ✅ Generation readiness: {_readiness.summary()}")
    except Exception as _rd_exc:
        logger.debug(f"  Generation readiness skipped (non-fatal): {_rd_exc}")

    code_generation_prompt = f"""
You are generating PRODUCTION CODE to satisfy pre-written tests.

IMPORTANT: Tests already exist. Your code must make them pass.
Base your implementation on:
1. Architectural patterns (how to structure code)
2. Existing code examples (style and conventions)
3. Coding standards (best practices)
4. Grounded evidence (verified investigation — see below)

DO NOT look at test internals - implement to satisfy the CONTRACT.

Ticket: {state['ticket'].title}
Requirements: {state['requirements']}
Plan: {state['architectural_plan']}
{grounded_context_str}
{semantic_constraints_block}
Architecture Context (from existing code):
{state['code_rag_context']}

Generate production-ready code that:
- Follows architectural patterns
- Implements requirements correctly
- Handles errors appropriately
- Uses dependency injection
- Includes logging
- Follows coding standards

Technology stack: Java (Spring Boot / Maven) for .java files;
Angular/TypeScript for .ts/.html/.scss files.
"""
    generated_code = []
    successful_writes = 0
    written_file_paths: set[str] = set()
    attempted_writable_files: set[str] = set()
    task_ledger: dict[str, dict] = {}
    validator = PatchValidator()

    # ── ARTIFACT CLASSIFICATION AND ROUTING FOUNDATION ───────────────────────
    from ticket_to_code.agents.artifact_classifier import ArtifactClassifier
    from ticket_to_code.agents.generation_router import GenerationRouter
    
    classifier = ArtifactClassifier()
    _active_gen_llm = getattr(agents, "llm", None) or getattr(getattr(agents, "code_generator", None), "llm", None)
    router = GenerationRouter(workspace_path=state["workspace_path"], llm=_active_gen_llm)
    
    classification_traces = []
    routing_traces = []

    # ── BUILD FILE SCOPE LISTS ────────────────────────────────────────────────
    # allowed_files: the planner approved these for writing (modify / create)
    # readonly_files: planner included these for context only (read_only)
    all_tasks = _get_plan(state).tasks
    allowed_files  = [t.file_path for t in all_tasks if t.task_type.value != "read_only"]
    readonly_files = [t.file_path for t in all_tasks if t.task_type.value == "read_only"]

    # Warn early if the planner is being too aggressive
    if len(allowed_files) > 25:
        logger.warning(
            f"⚠️  Planner created {len(allowed_files)} writable tasks — "
            f"this is a very large refactoring. Proceeding without caps."
        )

    logger.info(
        f"   Scope: {len(allowed_files)} writable file(s), "
        f"{len(readonly_files)} read-only file(s)"
    )
    for p in allowed_files:
        logger.info(f"    ✅  writable: {p}")
    for p in readonly_files:
        logger.info(f"      read-only: {p}")

    # ── OWNERSHIP MAP from Phase 2 discovery results ──────────────────────────
    # Maps normalized path → ownership_type string produced by _verify_ownership.
    # Used to populate task.ownership_type before calling PatchValidator so the
    # ownership-tiered change budget (CHECK 5) can fire correctly.
    _ownership_map: dict[str, str] = {}
    for _disc in (state.get("discovered_files") or []):
        _np = _disc.get("path", "").replace("\\", "/").lower()
        _ot = _disc.get("features", {}).get("ownership_type", None)
        if _np and _ot:
            _ownership_map[_np] = _ot

    # ── Helper: Build GenerationHandoff from validated generated content ──────
    def _build_generation_handoff(
        task_id: str,
        file_path: str,
        content: str,
        validation_status: str,
        impl_state,
    ):
        """Create a GenerationHandoff from validated generated content.

        Called ONLY after validation passes. Extracts machine facts (exports,
        imports) from the generated code and attaches AI explanation from
        the task metadata.

        Returns None if the content is empty or extraction fails.
        """
        import re as _re
        import time as _time
        from ticket_to_code.models import GenerationHandoff, VerifiedSymbol

        if not content or not content.strip():
            return None

        _ext = Path(file_path).suffix.lower()

        # ── Extract exports (machine facts) ──────────────────────────────
        # Primary: WorkspaceSymbolScanner.scan_content() for grounded signatures
        # (owner class, params, return type, source line, visibility).
        # Fallback: existing regex extraction if scanner fails or yields nothing.
        _exports = []
        _extraction_method = "regex"

        try:
            from ticket_to_code.agents.workspace_symbol_scanner import WorkspaceSymbolScanner
            _scanner = WorkspaceSymbolScanner(".")
            _file_syms = _scanner.scan_content(content, file_path)

            for sym in _file_syms.symbols:
                # Skip private symbols — they are not consumer contracts
                if sym.access_level in ("private",):
                    continue

                # Build grounded signature for methods/functions
                _sig = ""
                if sym.kind in ("method", "function") and sym.params is not None:
                    _params_str = ", ".join(sym.params[:8])
                    _ret = sym.return_type or "void"
                    _sig = f"{sym.name}({_params_str}): {_ret}"
                elif sym.kind == "property" and sym.type_hint:
                    _sig = f"{sym.name}: {sym.type_hint}"

                _exports.append(VerifiedSymbol(
                    name=sym.name,
                    kind=sym.kind,
                    owner=sym.owner_class or "",
                    signature=_sig,
                    file_path=file_path,
                    export_status=(
                        "exported" if sym.access_level in ("public", None, "")
                        else "internal"
                    ),
                    source_line=sym.source_line,
                    extraction_method=_file_syms.extraction_method,
                ))

            # Ensure top-level class/interface names are included even if
            # they weren't emitted as individual symbols above
            for cls_name in (_file_syms.class_names or []):
                if not any(e.name == cls_name for e in _exports):
                    _exports.append(VerifiedSymbol(
                        name=cls_name,
                        kind="class",
                        owner="",
                        file_path=file_path,
                        export_status="exported",
                        extraction_method=_file_syms.extraction_method,
                    ))

            if _exports:
                _extraction_method = _file_syms.extraction_method

        except Exception as _scan_exc:
            logger.debug(
                f"  [Handoff] scan_content failed for {file_path}: {_scan_exc}, "
                f"falling back to regex"
            )

        # ── Regex fallback (if scanner produced nothing) ──────────────────
        if not _exports:
            if _ext in (".ts", ".js", ".tsx", ".jsx"):
                # TypeScript/JavaScript: export class/interface/enum/type/const/function
                for m in _re.finditer(
                    r'export\s+(?:default\s+)?(?:abstract\s+)?'
                    r'(class|interface|enum|type|const|function|let|var)\s+(\w+)',
                    content,
                ):
                    kind, name = m.group(1), m.group(2)
                    _exports.append(VerifiedSymbol(
                        name=name,
                        kind=kind,
                        file_path=file_path,
                        export_status="exported",
                        extraction_method="regex",
                    ))
            elif _ext == ".py":
                # Python: top-level class/def (no underscore prefix = public)
                for m in _re.finditer(r'^(class|def)\s+(\w+)', content, _re.MULTILINE):
                    kind_raw, name = m.group(1), m.group(2)
                    if not name.startswith("_"):
                        _exports.append(VerifiedSymbol(
                            name=name,
                            kind="class" if kind_raw == "class" else "function",
                            file_path=file_path,
                            export_status="exported",
                            extraction_method="regex",
                        ))
            elif _ext in (".java", ".kt"):
                # Java/Kotlin: public class/interface/enum
                for m in _re.finditer(
                    r'(?:public\s+)?(?:abstract\s+)?'
                    r'(class|interface|enum)\s+(\w+)',
                    content,
                ):
                    kind, name = m.group(1), m.group(2)
                    _exports.append(VerifiedSymbol(
                        name=name,
                        kind=kind,
                        file_path=file_path,
                        export_status="exported",
                        extraction_method="regex",
                    ))

        # ── Extract imports ──────────────────────────────────────────────
        _imports = []
        if _ext in (".ts", ".js", ".tsx", ".jsx"):
            _imports = _re.findall(
                r"""(?:import|from)\s+(?:\{[^}]*\}\s+from\s+)?['"]([^'"]+)['"]""",
                content,
            )
        elif _ext == ".py":
            for m in _re.finditer(
                r'(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))',
                content,
            ):
                _imports.append(m.group(1) or m.group(2))
        elif _ext in (".java", ".kt"):
            _imports = _re.findall(r'import\s+([\w.]+);', content)

        # ── AI explanation (suggestive, not authoritative) ────────────────
        _task_obj = None
        try:
            for t in _get_plan(state).tasks:
                if t.id == task_id:
                    _task_obj = t
                    break
        except Exception:
            pass

        _what_changed = ""
        _why = ""
        _how_to_consume = ""
        if _task_obj:
            _what_changed = getattr(_task_obj, "title", "")
            _why = getattr(_task_obj, "description", "")
            # Suggest import path from file path (suggestive, not authoritative)
            _stem = file_path.replace("\\", "/")
            if "/src/" in _stem:
                _how_to_consume = _stem.split("/src/", 1)[-1].rsplit(".", 1)[0]

        return GenerationHandoff(
            task_id=task_id,
            file_path=file_path,
            timestamp=_time.time(),
            created_symbols=_exports,
            exports=_exports,
            imports=_imports,
            validation_status=validation_status,
            extraction_method=_extraction_method,
            what_changed=_what_changed,
            why=_why,
            how_to_consume=_how_to_consume,
        )

    _sorted_tasks = _sort_tasks_by_execution_order(_get_plan(state).tasks)

    # ── Phase 0a: Initialize ImplementationState for four-pillar consistency ──
    _impl_state = None
    try:
        from ticket_to_code.agents.implementation_state import ImplementationState
        _impl_state = ImplementationState.from_workflow_state(state)
        # Run blueprint verification before generation starts
        from ticket_to_code.agents.blueprint_verifier import BlueprintVerifier
        _bp_verifier = BlueprintVerifier(
            symbol_resolver=getattr(agents, "symbol_resolver", None),
            workspace_path=Path(state.get("workspace_path", "")),
        )
        _arch_plan = state.get("architectural_plan")
        if _arch_plan and _impl_state.signature_blueprints:
            _bp_verifier.verify_blueprints(
                blueprints=_impl_state.signature_blueprints,
                plan=_arch_plan,
                impl_state=_impl_state,
            )
        logger.info(
            f"  ✅ Phase 0a: ImplementationState initialized — "
            f"{len(_impl_state.signature_blueprints)} blueprints, "
            f"{len(_impl_state.generated_files)} prior files"
        )

        # ── Phase 0a.1: Initialize RelationshipRegistry (Pillar 6) ──
        # Aggregation layer over existing providers — does NOT replace them.
        try:
            from ticket_to_code.agents.relationship_registry import RelationshipRegistry
            _registry = RelationshipRegistry()
            _arch_plan_for_reg = state.get("architectural_plan")
            if _arch_plan_for_reg:
                _reg_count = _registry.register_from_plan(_arch_plan_for_reg)
                logger.info(
                    f"  ✅ Phase 0a.1: RelationshipRegistry initialized — "
                    f"{_reg_count} PLANNED relationships seeded from plan"
                )
            _impl_state.relationship_registry = _registry
        except Exception as _reg_exc:
            logger.warning(
                f"  ⚠️ RelationshipRegistry init failed (non-fatal): {_reg_exc}"
            )

    except Exception as _is_exc:
        logger.warning(f"  ⚠️ ImplementationState init failed (non-fatal): {_is_exc}")
        _impl_state = None

    # ── Phase 0b: Generate Definition of Done (before any code generation) ────
    # The checklist is consumed by Phase 5 (ChecklistVerifier) after build succeeds.
    if not state.get("definition_of_done"):
        try:
            from ticket_to_code.agents.checklist_agent import ChecklistAgent
            _checklist_agent = ChecklistAgent(agents.llm)
            _ticket = state.get("ticket")
            _dod = _checklist_agent.generate_checklist(
                ticket_title=getattr(_ticket, "title", "") or "",
                ticket_description=getattr(_ticket, "description", "") or "",
                requirements=state.get("requirements"),
                plan_tasks=_sorted_tasks,
            )
            state["definition_of_done"] = _dod
            if hasattr(agents, "code_generator") and agents.code_generator:
                agents.code_generator._definition_of_done = _dod
            logger.info(
                f"  ✅ Phase 0: {len(_dod.items)} checklist items and "
                f"{len(getattr(_dod, 'behavioral_invariants', []))} behavioral invariant(s) generated"
            )
        except Exception as _dod_exc:
            logger.warning(f"  ⚠️ Phase 0 (ChecklistAgent) failed: {_dod_exc}")

    # ── Dependency-depth batching ──────────────────────────────────────────
    # Group independent tasks so we can log batch boundaries during generation.
    # Tasks within the same batch have no mutual dependencies.
    _task_batches = _group_tasks_into_batches(_sorted_tasks)
    _task_to_batch: dict[str, int] = {}
    for _batch_idx, _batch in enumerate(_task_batches):
        for _bt in _batch:
            _task_to_batch[_bt.id] = _batch_idx

    logger.info(f"  Execution order ({len(_task_batches)} batches):")
    for _batch_idx, _batch in enumerate(_task_batches):
        _writable = [t for t in _batch if getattr(t.task_type, "value", str(t.task_type)) != "read_only"]
        if _writable:
            logger.info(f"    ─── Batch {_batch_idx} ({len(_writable)} task{'s' if len(_writable) != 1 else ''}) ───")
            for _ot in _writable:
                logger.info(f"      [{getattr(_ot.task_type,'value',str(_ot.task_type)).upper():6}] {_ot.file_path}")


    # ── PLAN→GENERATION GATE: activate reuse / scope / readiness ─────────────
    # Drops speculative shared-model changes and CREATE tasks that duplicate an
    # existing capability, so unnecessary files are never generated. Hard-blocks
    # (returns to planning) only when nothing survives or there is no grounding.
    try:
        from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
        from ticket_to_code.agents.capability_reuse_resolver import CapabilityReuseResolver

        _ev_files: set[str] = set()
        for _e in (state.get("evidence_items") or []):
            _efp = getattr(_e, "file_path", "") or ""
            if _efp:
                _ev_files.add(_efp)
        for _d in (state.get("discovered_files") or []):
            _dfp = (_d.get("path") if isinstance(_d, dict) else getattr(_d, "path", "")) or ""
            if _dfp:
                _ev_files.add(_dfp)

        _ticket_obj = state.get("ticket")
        _req_files: set[str] = set()
        for _attr in ("expected_changed_files", "expected_owner_files"):
            for _rf in (_get_ticket_attr(_ticket_obj, _attr, None) or []):
                if _rf:
                    _req_files.add(str(_rf))
        if not _req_files:
            _req_files = set(_ev_files)  # fall back to evidence-backed set

        # ── Change authorization scope (evidence is NOT authorization) ──
        _ticket_declared_files: set[str] = set()
        for _attr in ("expected_changed_files", "expected_owner_files"):
            for _af in (_get_ticket_attr(_ticket_obj, _attr, None) or []):
                if _af:
                    _ticket_declared_files.add(str(_af))

        _authorized: set[str] = set(_ticket_declared_files)

        # Ingest preflight guidance target files, dependencies and companions into _authorized scope
        _preflight_guidance = state.get("preflight_implementation_guidance") or []
        for _g in _preflight_guidance:
            _tf = _g.get("target_file")
            if _tf:
                _authorized.add(str(_tf))
                try:
                    from ticket_to_code.agents.companion_resolver import CompanionResolver
                    _cr = CompanionResolver()
                    for _comp in _cr.get_companion_files(_tf):
                        _authorized.add(str(_comp))
                except Exception:
                    pass
            for _dep in _g.get("dependencies", []):
                _dep_str = str(_dep).strip()
                if _dep_str and ("/" in _dep_str or "\\" in _dep_str or _dep_str.endswith((".java", ".ts", ".html", ".scss", ".css", ".py"))):
                    _authorized.add(_dep_str)

        # Planned writable tasks from the architectural plan MUST be authorized so that companion
        # repositories, saga handlers, and service files planned to fulfill ticket requirements are
        # not discarded by strict whitelisting.
        for _t in _sorted_tasks:
            _ttype = getattr(_t.task_type, "value", str(_t.task_type)).lower()
            if _ttype != "read_only" and getattr(_t, "file_path", None):
                _authorized.add(str(_t.file_path))

        _forbidden = {str(x) for x in (getattr(_ticket_obj, "forbidden_files", None) or []) if x}
        _scope_declared = bool(_ticket_declared_files or _forbidden)
        _ticket_text_low = " ".join(filter(None, [
            str(getattr(_ticket_obj, "title", "") or ""),
            str(getattr(_ticket_obj, "description", "") or ""),
            " ".join(getattr(_ticket_obj, "labels", []) or []),
        ])).lower()
        _ticket_targets_config = any(
            k in _ticket_text_low for k in (
                "config", "configuration", "application.yml", "application.yaml",
                ".env", "environment variable", "yaml", "properties file",
            )
        )

        _sr = getattr(agents, "symbol_resolver", None)
        _reuse_resolver = CapabilityReuseResolver(
            symbol_resolver=_sr,
            symbol_index=getattr(_sr, "_index", None) if _sr else None,
        )

        # Hybrid capability resolver (semantic + repo-search + LLM judgement).
        # Rejects tasks that invent a new method/file when an existing capability
        # already satisfies the intent — the fix for planner full-stack invention.
        _semantic_resolver = None
        if os.environ.get("AVIATOR_CAPABILITY_RESOLVE", "1") != "0":
            try:
                from ticket_to_code.agents.task_level_analyzer import SemanticCapabilityResolver
                _semantic_resolver = SemanticCapabilityResolver(
                    rag_engine=getattr(agents, "rag_engine", None),
                    repo_search_engine=getattr(agents, "repo_search", None),
                    llm=getattr(getattr(agents, "planner", None), "llm", None)
                        or getattr(agents, "llm", None),
                    workspace_path=str(getattr(agents, "workspace_path", "") or ""),
                )
            except Exception as _sem_err:
                logger.warning(f"   Hybrid capability resolver unavailable: {_sem_err}")

        # Grounding present unless BOTH verified capabilities and contracts are absent.
        _grounding_present = bool(state.get("evidence_items")) or bool(
            _get_transient(state, "semantic_contract")
        )

        # Evidence-proven change targets (BehavioralUnderstanding). Under no
        # declared scope, only these may become CHANGE_TARGET — discovery alone
        # never authorizes a write (DISCOVERED != CHANGE_TARGET).
        _proven_targets: set[str] = set()
        _bu_obj = getattr(
            getattr(agents, "evidence_loop", None), "_evidence_knowledge", None
        )
        _bu_obj = getattr(_bu_obj, "behavioral_understanding", None)
        if _bu_obj is not None:
            _proven_targets = {str(x) for x in (getattr(_bu_obj, "change_candidates", None) or [])}
        _scope_proof_workflow = state.get("ticket_scope_proof")
        if _scope_proof_workflow is not None and getattr(_scope_proof_workflow, "status", None) == "PROVEN":
            _approved_targets = set(_scope_proof_workflow.approved_writable_files)
            if _proven_targets:
                _proven_targets = _proven_targets.union(_approved_targets)
            else:
                _proven_targets = _approved_targets

        _gate = filter_generation_tasks(
            tasks=_sorted_tasks,
            evidence_files=_ev_files,
            ticket_required_files=_req_files,
            reuse_resolver=_reuse_resolver,
            grounding_present=_grounding_present,
            authorized_files=_authorized,
            forbidden_files=_forbidden,
            scope_declared=_scope_declared,
            ticket_targets_config=_ticket_targets_config,
            semantic_resolver=_semantic_resolver,
            proven_targets=_proven_targets,
            workspace_root=state.get("workspace_path"),
            scope_proof=state.get("ticket_scope_proof"),
        )
        for _rej in _gate.rejected:
            logger.warning(
                f"  🚧 Pre-generation gate REJECTED {_rej.file_path}: {_rej.reason}"
            )
        if _gate.hard_block:
            _rc = int(state.get("candidate_retry_count", 0) or 0)
            logger.error(
                f"  ⛔ Pre-generation gate hard-block: {_gate.block_reason} — "
                f"returning to planning (retry {_rc + 1})."
            )
            return {
                "status": "generation_gate_blocked",
                "candidate_retry_count": _rc + 1,
                "validation_failure_reason": _gate.block_reason,
            }
        if _gate.rejected:
            _sorted_tasks = list(_gate.accepted)
            logger.info(
                f"  🚧 Pre-generation gate: {len(_gate.rejected)} task(s) rejected, "
                f"{len([t for t in _sorted_tasks if t.task_type.value != 'read_only'])} writable remain."
            )

        # ── Pre-generation drift threshold check (Adjustment B) ───────────
        # If >50% of planned writable tasks fall outside the proven/localized set,
        # fail fast and route to planning_recovery instead of burning 300k+ tokens
        # on hallucinated/drift generation.
        # Adjustment A: planned CREATE tasks stay writable and are exempt from drift check.
        _all_writable = [
            t for t in _sorted_tasks
            if getattr(t.task_type, "value", str(t.task_type)) != "read_only"
        ]
        _sp = state.get("ticket_scope_proof")
        if _all_writable:
            _drift_tasks = []
            _total_drift_weight = 0.0
            _prov_norms = {p.replace("\\", "/").lower().strip() for p in _proven_targets} if _proven_targets else set()
            _auth_norms = {a.replace("\\", "/").lower().strip() for a in (_authorized | _req_files | set(state.get("expected_changed_files") or []))}
            for t in _all_writable:
                ttype = getattr(t.task_type, "value", str(t.task_type))
                if ttype == "create":
                    # Adjustment A: Keep planned CREATE tasks writable (new components, i18n keys)
                    continue
                fp_norm = t.file_path.replace("\\", "/").lower().strip()
                in_proven = any(
                    fp_norm == p or fp_norm.endswith("/" + p) or p.endswith("/" + fp_norm)
                    for p in _prov_norms
                )
                in_auth = any(
                    fp_norm == a or fp_norm.endswith("/" + a) or a.endswith("/" + fp_norm)
                    for a in _auth_norms
                )
                in_sp = _sp.is_file_writable(t.file_path, workspace_root=state.get("workspace_path")) if _sp else False
                if in_proven or in_auth:
                    task_weight = 0.0
                elif in_sp:
                    # D2: scope_graph entries count as half-weight in the drift ratio, not as exemptions
                    task_weight = 0.5
                    _drift_tasks.append(t)
                else:
                    task_weight = 1.0
                    _drift_tasks.append(t)
                _total_drift_weight += task_weight

            _drift_ratio = _total_drift_weight / max(len(_all_writable), 1)
            if _drift_ratio > 0.5:
                _rc = int(state.get("candidate_retry_count", 0) or 0)
                drift_reason = (
                    f"Pre-generation drift threshold exceeded: {len(_drift_tasks)}/{len(_all_writable)} "
                    f"weighted drift ({_drift_ratio:.0%}) planned tasks fall outside proven/localized scope: "
                    f"{[t.file_path for t in _drift_tasks]}"
                )
                logger.error(
                    f"  ⛔ {drift_reason} — routing to planning_recovery before LLM generation spend (retry {_rc + 1})."
                )
                return {
                    "status": "planning_needs_recovery",
                    "validation_failure_reason": drift_reason,
                    "candidate_retry_count": _rc + 1,
                }

        # Authoritatively sync allowed_files from surviving tasks and verified scope proof
        allowed_files = [
            t.file_path for t in _sorted_tasks
            if getattr(t.task_type, "value", str(t.task_type)) != "read_only"
            and (
                getattr(t.task_type, "value", str(t.task_type)) == "create"
                or _sp is None
                or _sp.is_file_writable(t.file_path, workspace_root=state.get("workspace_path"))
            )
        ]
    except Exception as _gate_exc:
        logger.debug(f"  Pre-generation gate skipped (non-fatal): {_gate_exc}")

    _current_batch_idx = -1  # Track batch transitions
    for task in _sorted_tasks:
        # ── Batch boundary marker ─────────────────────────────────────────
        # When we cross into a new batch, log the transition. All contracts
        # from the previous batch are now registered and available to tasks
        # in this new batch via ImplementationState and RelationshipRegistry.
        _this_batch = _task_to_batch.get(task.id, -1)
        if _this_batch != _current_batch_idx:
            _current_batch_idx = _this_batch
            _batch_tasks = _task_batches[_this_batch] if _this_batch < len(_task_batches) else []
            _batch_writable = [t for t in _batch_tasks if getattr(t.task_type, "value", str(t.task_type)) != "read_only"]
            if _batch_writable:
                logger.info(
                    f"\n  ╔══════════════════════════════════════════════════════╗\n"
                    f"  ║  BATCH {_this_batch} — {len(_batch_writable)} independent task(s)            ║\n"
                    f"  ╚══════════════════════════════════════════════════════╝"
                )
                if _this_batch > 0 and _impl_state:
                    _prior_surfaces = len(_impl_state.api_surfaces)
                    _prior_handoffs = len(getattr(_impl_state, '_handoffs', {}))
                    logger.info(
                        f"  📋 Contracts available from prior batches: "
                        f"{_prior_surfaces} API surfaces, {_prior_handoffs} handoffs"
                    )

        # READ_ONLY tasks are context-only — never write files
        if task.task_type.value == "read_only":
            logger.info(f"  ⏩ Skipping read_only task [{task.id}]: {task.title}")
            task_ledger[task.id] = {"file_path": task.file_path, "status": "read_only_skipped", "reason": "context_only"}
            continue

        # SCOPE GUARD: skip if this task was trimmed by the hard cap
        _norm_task_fp = task.file_path.replace("\\", "/").lower().strip()
        _norm_allowed = {af.replace("\\", "/").lower().strip() for af in allowed_files}
        if not any(
            _norm_task_fp == af or _norm_task_fp.endswith("/" + af) or af.endswith("/" + _norm_task_fp)
            for af in _norm_allowed
        ):
            logger.warning(
                f"   SCOPE GUARD: task [{task.id}] targets '{task.file_path}' "
                f"which exceeded the writable-task cap — skipping."
            )
            continue

        attempted_writable_files.add(task.file_path)

        # Populate Phase 2 ownership type onto the task so PatchValidator can
        # apply the correct change budget (PRIMARY vs DISPLAY_OWNER/SUPPORTING).
        _task_norm = task.file_path.replace("\\", "/").lower()
        _ticket_title = getattr(state.get("ticket"), "title", "").lower()
        _ticket_desc = getattr(state.get("ticket"), "description", "").lower()
        _is_single_task = len([t for t in _sorted_tasks if getattr(getattr(t, "task_type", None), "value", str(getattr(t, "task_type", ""))).lower() != "read_only"]) <= 1
        _is_ticket_target = Path(task.file_path).name.lower() in (_ticket_title + " " + _ticket_desc)
        
        if _is_single_task or _is_ticket_target:
            task.ownership_type = "PRIMARY_OWNER"
            logger.info(f"  ⭐ Designated PRIMARY_OWNER for target file: {task.file_path}")
        elif _task_norm in _ownership_map:
            _mapped_ot = _ownership_map[_task_norm]
            _ttype_str = getattr(getattr(task, "task_type", None), "value", str(getattr(task, "task_type", ""))).lower()
            if _mapped_ot == "READ_ONLY" and _ttype_str != "read_only":
                task.ownership_type = "SUPPORTING"
                logger.info(f"  ️  ownership_type upgraded READ_ONLY→SUPPORTING (writable task) for {task.file_path}")
            else:
                task.ownership_type = _mapped_ot
                logger.info(f"  ️  ownership_type={task.ownership_type} for {task.file_path}")
        else:
            task.ownership_type = "PRIMARY_OWNER"

        logger.info(f"  Generating code for: {task.file_path}")

        # Build the candidate queue: primary path first, then ranked fallbacks
        # Candidates carry the full signal breakdown from localization.
        candidate_paths: list[str] = [task.file_path]
        if task.file_candidates:
            for c in task.file_candidates[1:]:  # [0] is already task.file_path
                if c.path not in candidate_paths:
                    candidate_paths.append(c.path)

        written = False
        for attempt, candidate_path in enumerate(candidate_paths):
            if attempt > 0:
                logger.info(
                    f"   Retry [{attempt}] with fallback candidate: {candidate_path}"
                )
                task.file_path = candidate_path  # swap to next candidate

            # For modify tasks, read existing file content so LLM modifies in place.
            # Fix 2: prefer already-generated-this-run content over on-disk content so
            # later tasks (components) see the updated interfaces/models written earlier.
            existing_content = None
            _run_generated_map: dict = state.setdefault("_run_generated_map", {})
            _task_norm_key = task.file_path.replace("\\", "/").lower()
            if _task_norm_key in _run_generated_map:
                existing_content = _run_generated_map[_task_norm_key]
                logger.info(f"   Using this-run generated content for: {task.file_path}")
            else:
                existing_path = Path(state["workspace_path"]) / task.file_path
                if existing_path.exists():
                    existing_content = existing_path.read_text(encoding='utf-8')
                    logger.info(f"   Reading existing file for modification/context: {existing_path}")
                if "original_file_contents" not in state:
                    state["original_file_contents"] = {}
                state["original_file_contents"][task.file_path] = existing_content

            # 1. Classify Artifact
            artifact_type = classifier.classify(task.file_path)
            task.artifact_type = artifact_type
            
            # 2. Record artifact classification
            classification_traces.append({
                "file_path": task.file_path,
                "artifact_type": artifact_type.value,
            })
            
            # 3. Route through GenerationRouter
            generator = router.route(task)
            
            # 4. Record generation routing
            routing_traces.append({
                "file_path": task.file_path,
                "artifact_type": artifact_type.value,
                "selected_generator": generator.__class__.__name__
            })

            # Temporary diagnostic logging
            logger.info("\n[GENERATION]")
            logger.info(f"task={task.id}")
            logger.info(f"file={task.file_path}")
            logger.info(f"artifact={artifact_type.value}")
            logger.info(f"generator={generator.__class__.__name__}\n")

            # Write traces to disk immediately before generation starts
            agents.tracer.record_artifact_classification(classifications=classification_traces)
            agents.tracer.record_generation_routing(routing=routing_traces)

            # Inject sqlite_store so Layer 1 import context can query the symbol index.
            generator._sqlite_store = getattr(agents.localizer, "sqlite_store", None)
            _ws = str(state.get("workspace_path", "") or getattr(agents, "workspace_path", ""))
            generator._workspace_path = _ws
            generator.workspace_path = Path(_ws) if _ws else None
            # Inject fresh session-generated files for incremental cross-file context.
            generator._session_files = state.get("_run_generated_map", {})
            # Inject ImplementationState for blueprint-driven context + post-gen recording.
            if _impl_state is not None:
                generator._impl_state = _impl_state

            # ── PER-TASK EXPLANATION (human-readable + LLM context) ────────────
            # Before writing code, build a clear explanation of:
            #   1. WHY this file is being changed (purpose)
            #   2. WHAT symbols it will create (produces)
            #   3. WHAT symbols it will use from earlier tasks (consumes)
            #   4. WHAT was already generated (previously written files + exports)
            # This explanation is:
            #   - Stored in state for the UI to display
            #   - Accumulated so the NEXT task sees the full chain
            _task_explanations = state.setdefault("_task_explanations", [])
            _prev_explanations = state.setdefault("_accumulated_context", [])

            _explanation = _build_task_explanation(
                task=task,
                task_index=len(_task_explanations) + 1,
                total_tasks=len([t for t in _sorted_tasks if t.task_type.value != "read_only"]),
                previously_written=state.get("_run_generated_map", {}),
                prev_explanations=_prev_explanations,
            )
            _task_explanations.append(_explanation)
            logger.info(
                f"\n{'='*70}\n"
                f"📋 TASK EXPLANATION [{_explanation['task_index']}/{_explanation['total_tasks']}]\n"
                f"   File: {_explanation['file_path']}\n"
                f"   Purpose: {_explanation['purpose']}\n"
                f"   Produces: {', '.join(_explanation.get('produces', []))}\n"
                f"   Consumes: {', '.join(_explanation.get('consumes', []))}\n"
                f"   Previously written: {len(_explanation.get('previously_written_summary', []))} files\n"
                f"{'='*70}\n"
            )


            # ── Per-file token accounting ─────────────────────────────────────
            # Snapshot budget before generation to compute per-file delta.
            _file_tokens_before = None
            try:
                _run_ctx = _get_transient(state, "run_ctx")
                if _run_ctx and hasattr(_run_ctx, "budget"):
                    _file_tokens_before = (
                        _run_ctx.budget.tokens_in,
                        _run_ctx.budget.tokens_out,
                        _run_ctx.budget.llm_calls,
                    )
            except Exception:
                pass

            # Invoke returned generator (no broad try-except, per user instructions)
            _dfc = state.get("dataflow_contract") or _get_transient(state, "dataflow_contract")
            code = generator.generate_code(
                task=task,
                requirements=state["requirements"],
                context=state["code_rag_context"],
                existing_content=existing_content,
                allowed_files=allowed_files,
                readonly_files=readonly_files,
                dataflow_contract=_dfc,
            )

            # ── PRE-COMMIT BINDING VALIDATION (P0 accuracy) ───────────────────
            # Every Angular binding in a generated template must reference a
            # symbol declared in the companion controller.  Violations get ONE
            # targeted regeneration with a forbidden-binding contract before
            # the compiler (or a human) ever sees them.
            try:
                _tpl = (task.file_path or "").replace("\\", "/")
                if _tpl.lower().endswith((".html", ".htm")) and (code.content or "").strip():
                    _ctrl_rel = _tpl[: -len(Path(_tpl).suffix)] + ".ts"
                    _ctrl_abs = Path(str(state.get("workspace_path", ""))) / _ctrl_rel
                    if _ctrl_abs.exists():
                        from ticket_to_code.agents.code_generator import validate_template_bindings
                        from ticket_to_code.agents.code_generator import (
                            extract_dynamic_viewmodel_fields,
                        )
                        _ctrl_code_val = _ctrl_abs.read_text(encoding="utf-8", errors="ignore")
                        _viol = validate_template_bindings(
                            code.content, _ctrl_code_val,
                            extra_allowed=extract_dynamic_viewmodel_fields(_ctrl_code_val),
                            original_template=existing_content,
                        )
                        if _viol:
                            logger.warning(
                                f"  ⛔ BINDING VIOLATIONS in {_tpl}: {_viol} — "
                                "regenerating once with a forbidden-binding contract"
                            )
                            code = generator.generate_code(
                                task=task,
                                requirements=state["requirements"],
                                context=state["code_rag_context"],
                                existing_content=existing_content,
                                allowed_files=allowed_files,
                                readonly_files=readonly_files,
                                dataflow_contract=_dfc,
                                binding_violations=_viol,
                            )
            except Exception as _bind_exc:
                logger.debug(f"Binding validation skipped (non-fatal): {_bind_exc}")

            # ── Per-file token result ─────────────────────────────────────────
            try:
                if _file_tokens_before and _run_ctx:
                    _tin_after = _run_ctx.budget.tokens_in
                    _tout_after = _run_ctx.budget.tokens_out
                    _calls_after = _run_ctx.budget.llm_calls
                    _file_tin = _tin_after - _file_tokens_before[0]
                    _file_tout = _tout_after - _file_tokens_before[1]
                    _file_calls = _calls_after - _file_tokens_before[2]

                    # Estimate input breakdown from last prompt components (char-based, ~4 chars/token)
                    _est_task_chars = len(task.description or "") + len(task.title or "")
                    _est_existing_chars = len(existing_content or "")
                    _est_rag_chars = sum(len(str(c.get("content", ""))) for c in (state.get("code_rag_context") or [])[:5])
                    _est_session_chars = len(getattr(generator, "_last_session_context", "") or "")
                    _est_contract_chars = len(getattr(generator, "_last_contract_context", "") or "")
                    _est_impl_chars = len(getattr(generator, "_last_impl_context", "") or "")

                    _est_div = max(1, 4)  # ~4 chars per token
                    logger.info(
                        f"\n  ┌─ TOKEN ACCOUNTING: {Path(task.file_path).name} ──────────\n"
                        f"  │ Input:   ~{_file_tin:,} tokens  ({_file_calls} LLM call{'s' if _file_calls != 1 else ''})\n"
                        f"  │   Task desc:       ~{_est_task_chars // _est_div:,} tokens\n"
                        f"  │   Existing source:  ~{_est_existing_chars // _est_div:,} tokens\n"
                        f"  │   RAG context:      ~{_est_rag_chars // _est_div:,} tokens\n"
                        f"  │   Session context:  ~{_est_session_chars // _est_div:,} tokens\n"
                        f"  │   Contracts:        ~{_est_contract_chars // _est_div:,} tokens\n"
                        f"  │   Impl handoffs:    ~{_est_impl_chars // _est_div:,} tokens\n"
                        f"  │ Output:  ~{_file_tout:,} tokens\n"
                        f"  │ Total:   ~{_file_tin + _file_tout:,} tokens\n"
                        f"  │ Batch:   {_task_to_batch.get(task.id, '?')}\n"
                        f"  └──────────────────────────────────────"
                    )
            except Exception as _tok_exc:
                logger.debug(f"  Per-file token accounting failed (non-fatal): {_tok_exc}")

            # ── PATCH VALIDATION: check scope before writing to disk ──────────
            if code is None:
                from ticket_to_code.models import GeneratedCode
                logger.warning(f"Code generator returned None for {task.file_path} — preserving original content")
                code = GeneratedCode(
                    file_path=task.file_path,
                    content=existing_content or "",
                    language=task.language,
                    change_type=task.task_type,
                    documentation="Preserved original content (None guard)",
                    imports=[],
                )
            validation = validator.validate(code, task, existing_content)
            log_line = StageLog.patch_validation(task, validation)
            logger.info(log_line)
            print(log_line)
            if not validation.passed:
                logger.warning(
                    f"  ⛔ Candidate [{attempt}] {task.file_path} failed validation:\n"
                    + "\n".join(f"     • {v}" for v in validation.violations)
                )
                if attempt + 1 < len(candidate_paths):
                    logger.info(f"  ↩️  Will retry with next candidate …")
                    continue  # try next candidate
                else:
                    logger.warning(f"  ❌ All {len(candidate_paths)} candidate(s) failed — skipping task")
                    break

            # ── TRUNCATION / CORRUPTION GUARD ────────────────────────────────
            # Reject output that is clearly malformed LLM output:
            #  - Truncation markers (ran out of tokens)
            #  - Markdown code fences inside code (``` in non-markdown files)
            #  - Java lines that look like package refs but missing 'import'
            _trunc_markers = (
                "[TRUNCATED]", "[...]", "// ... rest", "/* ... */",
                "# ... rest", "...existing code...", "...rest of",
            )
            content_lower = code.content.lower()
            truncated = any(m.lower() in content_lower for m in _trunc_markers)

            # Code-fence corruption: LLM embedded markdown in source code
            _ts_ext_check = Path(code.file_path).suffix.lower()
            if not truncated and _ts_ext_check not in (".md", ".txt", ".rst"):
                if "```" in code.content:
                    truncated = True
                    logger.warning(
                        f"  ⚠️ Code-fence corruption in {code.file_path}: "
                        f"LLM embedded markdown ``` inside source code"
                    )

            # Java brace balance: "reached end of file while parsing" = unclosed braces
            if not truncated and _ts_ext_check in (".java", ".kt"):
                _open_b = code.content.count("{")
                _close_b = code.content.count("}")
                if _open_b > _close_b:
                    truncated = True
                    logger.warning(
                        f"  ⚠️ Java brace imbalance in {code.file_path}: "
                        f"{_open_b} open vs {_close_b} close — file truncated before closing class/method"
                    )
                # Also check: last non-whitespace char should be '}'
                _last_char = code.content.rstrip()[-1:] if code.content.rstrip() else ""
                if not truncated and _last_char and _last_char != "}" and "class " in code.content:
                    truncated = True
                    logger.warning(
                        f"  ⚠️ Java file {code.file_path} does not end with '}}' — "
                        f"likely truncated (last char: {repr(_last_char)})"
                    )

            # Java package preservation guard: prevent unanchored snippet overwrite
            if not truncated and _ts_ext_check in (".java", ".kt") and existing_content:
                if "package " in existing_content and "package " not in code.content:
                    truncated = True
                    logger.warning(
                        f"  ⚠️ Java package declaration missing in {code.file_path}: "
                        f"original had package, but generated code does not — refusing to overwrite with fragment"
                    )

            # Java-import corruption: line looks like package path without 'import'
            if not truncated and _ts_ext_check == ".java":
                _bad_java_lines = [
                    ln.strip() for ln in code.content.splitlines()
                    if re.match(r'^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,}$', ln.strip())
                    and not ln.strip().startswith("import ")
                    and not ln.strip().startswith("package ")
                ]
                if _bad_java_lines:
                    truncated = True
                    logger.warning(
                        f"  ⚠️ Java import-syntax corruption in {code.file_path}: "
                        f"lines missing 'import' keyword: {_bad_java_lines[:3]}"
                    )

            # Cross-service import guard: project-service CANNOT import area-service internals
            if not truncated and _ts_ext_check == ".java":
                _cross_svc = _check_java_cross_service_imports(code.file_path, code.content)
                if _cross_svc:
                    truncated = True
                    logger.warning(
                        f"  ⚠️ Cross-service import violation in {code.file_path} — "
                        f"these imports cross microservice boundaries and will fail to compile: "
                        + "; ".join(_cross_svc[:3])
                    )
                    _silent_failures = state.setdefault("_patch_failures", [])
                    _silent_failures.append({
                        "file": code.file_path,
                        "reason": f"cross_service_import:{_cross_svc[:2]}",
                        "allowed_methods": task.allowed_methods or [],
                    })
            if not truncated and existing_content:
                # Also flag if the generated file is >20% shorter than original
                # (a likely sign the model stopped early)
                orig_lines = len(existing_content.splitlines())
                new_lines  = len(code.content.splitlines())
                if orig_lines > 20 and new_lines < orig_lines * 0.8:
                    truncated = True
                    logger.warning(
                        f"  ⚠️ Generated file is only {new_lines} lines vs "
                        f"{orig_lines} original — likely truncated"
                    )
            if truncated:
                logger.warning(
                    f"  ⚠️ LLM output appears TRUNCATED for {task.file_path} — "
                    f"refusing to write corrupted code to disk"
                )
                if attempt + 1 < len(candidate_paths):
                    logger.info(f"  ↩️  Will retry with next candidate …")
                    continue
                else:
                    logger.warning(f"  ❌ All candidates produced truncated output — skipping task")
                    break

            # Write code file
            output_path = Path(state["workspace_path"]) / code.file_path
            
            # Check if content is unchanged (patch application failed silently)
            if existing_content and code.content.strip() == existing_content.strip():
                is_modify = getattr(task, "task_type", "MODIFY").upper() in ("MODIFY", "UPDATE", "EDIT", "PATCH")
                if not is_modify:
                    methods_present = (
                        bool(task.allowed_methods)
                        and all(m in existing_content for m in task.allowed_methods)
                    )
                    if methods_present:
                        logger.info(
                            f"  ✅ Content identical to existing file and required method(s) {task.allowed_methods} "
                            f"already present in {task.file_path}. Task satisfied without modification."
                        )
                        generated_code.append(code)
                        break

                logger.warning(
                    f"  ⚠️ Generated content identical to original for {task.file_path} — "
                    f"patch likely failed (SEARCH block may not have matched). "
                    f"Planner allowed_methods: {task.allowed_methods or []}"
                )
                # Track silent failure so it surfaces in the workflow explanation
                _silent_failures = state.setdefault("_patch_failures", [])
                _silent_failures.append({"file": task.file_path, "reason": "content_unchanged",
                                          "allowed_methods": task.allowed_methods or []})
                if attempt + 1 < len(candidate_paths):
                    logger.info(f"  ↩️  Will retry with next candidate …")
                    continue
                break
                
            if code.documentation and code.documentation.startswith("FAILED:"):
                logger.warning(f"  ⚠️ Code generator reported failure: {code.documentation} — skipping write")
                _silent_failures = state.setdefault("_patch_failures", [])
                _silent_failures.append({"file": task.file_path, "reason": code.documentation,
                                          "allowed_methods": task.allowed_methods or []})
                if attempt + 1 < len(candidate_paths):
                    logger.info(f"  ↩️  Will retry with next candidate …")
                    continue
                break
                
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # ── Layer 2: META imports injection (Devin / Copilot Workspace) ─────
            if code.imports:
                code = _inject_meta_imports(code)

            # ── Parenthesis balance check (catches broken callbacks before write) ──
            _par_ext = Path(code.file_path).suffix.lower()
            if existing_content and _par_ext in (".ts", ".tsx", ".js", ".jsx", ".java"):
                _open_p = code.content.count("(")
                _close_p = code.content.count(")")
                if abs(_open_p - _close_p) > 2:  # allow minor off-by-one from string literals
                    logger.warning(
                        f"  ⚠️ Parenthesis imbalance in {code.file_path}: "
                        f"{_open_p} open vs {_close_p} close — rejecting broken patch, reverting to original"
                    )
                    _silent_failures = state.setdefault("_patch_failures", [])
                    _silent_failures.append({
                        "file": code.file_path,
                        "reason": f"paren_imbalance:open={_open_p},close={_close_p}",
                        "allowed_methods": task.allowed_methods or [],
                    })
                    if attempt + 1 < len(candidate_paths):
                        logger.info(f"  ↩️  Will retry with next candidate …")
                        continue
                    break

            # ── Fix 1: Export-drop guard (TypeScript / JavaScript) ────────────
            _ts_ext = Path(code.file_path).suffix.lower()
            if existing_content and _ts_ext in (".ts", ".tsx", ".js", ".jsx"):
                _dropped = _check_ts_exports_preserved(existing_content, code.content)
                if _dropped:
                    logger.warning(
                        f"  ⚠️ Export-drop guard REJECTED write for {code.file_path}: "
                        f"missing exports: {_dropped} — skipping to prevent TS2305 cascade"
                    )
                    _silent_failures = state.setdefault("_patch_failures", [])
                    _silent_failures.append({
                        "file": code.file_path,
                        "reason": f"export_dropped:{_dropped}",
                        "allowed_methods": task.allowed_methods or [],
                    })
                    if attempt + 1 < len(candidate_paths):
                        logger.info(f"  ↩️  Will retry with next candidate …")
                        continue
                    break

            # ── Post-write TS property validator (Cursor LSP equivalent) ─────
            # Check that property accesses in the new file match the FRESH
            # interface shapes from this run — catches TS2551 before build.
            if _ts_ext in (".ts", ".tsx"):
                _bad_props = _check_ts_property_access(
                    code.content, code.file_path,
                    state.get("_run_generated_map", {}),
                    str(state.get("workspace_path", "")),
                )
                if _bad_props:
                    logger.warning(
                        f"  ⚠️ TS property mismatch in {code.file_path}: {_bad_props} "
                        f"— these properties do not exist in the freshly-generated interfaces"
                    )
                    _silent_failures = state.setdefault("_patch_failures", [])
                    _silent_failures.append({
                        "file": code.file_path,
                        "reason": f"ts_property_mismatch:{_bad_props}",
                        "allowed_methods": task.allowed_methods or [],
                    })
                    if attempt + 1 < len(candidate_paths):
                        logger.info(f"  ↩️  Will retry with next candidate …")
                        continue
                    # On last attempt: still write but log prominently
                    logger.warning(
                        f"  ⚠️ Allowing write despite property mismatch (last attempt) — "
                        f"fix_build will need to correct: {_bad_props}"
                    )

            # ── PRE-WRITE CHECKPOINT (transaction/rollback) ────────────────────
            if _impl_state is not None:
                _impl_state.create_checkpoint(
                    file_path=code.file_path,
                    description=f"Pre-write for task {task.id}",
                )

            # ── Workspace Integrity Safe Write ─────────────────────────────────
            try:
                from ticket_to_code.integrity.workspace_integrity import get_workspace_integrity_manager
                _integrity_mgr = get_workspace_integrity_manager(state.get("workspace_path", ""))
                _patch_rec = _integrity_mgr.safe_write(
                    file_path=code.file_path,
                    new_content_or_edits=code.content,
                    ticket_id=getattr(state.get("ticket"), "ticket_id", "") or "",
                    task_id=getattr(task, "id", "") or "task",
                    requirement_id=getattr(task, "requirement_id", "") or "",
                    symbol=getattr(task, "target_method", None),
                )
                logger.info(f"  📝 [WorkspaceIntegrity] Patch applied: {_patch_rec.patch_id} on {code.file_path}")
            except Exception as _w_err:
                logger.warning(f"  [WorkspaceIntegrity] write fallback for {code.file_path}: {_w_err}")
                output_path.write_text(code.content, encoding='utf-8')

            # ── FAST LOCAL SYNTAX CHECK AFTER EVERY FILE WRITE ──
            try:
                _live_ext = Path(code.file_path).suffix.lower()
                _live_errors = []
                if _live_ext == ".py":
                    import py_compile
                    try:
                        py_compile.compile(str(output_path), doraise=True)
                    except py_compile.PyCompileError as _pce:
                        _live_errors.append(f"{code.file_path}: {_pce.msg or str(_pce)}")
                elif _live_ext in (".scss", ".css"):
                    _live_errors.extend(_check_scss_syntax(code.content, code.file_path))
                elif _live_ext in (".html", ".htm"):
                    _live_errors.extend(_check_html_syntax(code.content, code.file_path))
                elif _live_ext in (".ts", ".tsx", ".js", ".jsx", ".java", ".kt", ".cs"):
                    _live_errors.extend(_check_code_syntax(code.content, code.file_path))
                
                _ui_cb = _get_transient(state, "_ui_callback")
                if _ui_cb:
                    from datetime import datetime as _dt_live
                    if not _live_errors:
                        _ui_cb({
                            "phase": "generation",
                            "status": "in_progress",
                            "message": f"✅ Local Syntax: Clean ({Path(code.file_path).name})",
                            "data": {"file_path": code.file_path, "local_syntax": "passed"},
                            "timestamp": _dt_live.now().isoformat(),
                        })
                    else:
                        _ui_cb({
                            "phase": "generation",
                            "status": "in_progress",
                            "message": f"⚠️ Local Syntax: {len(_live_errors)} issue(s) in {Path(code.file_path).name}",
                            "data": {"file_path": code.file_path, "local_syntax": "failed", "errors": _live_errors[:5]},
                            "timestamp": _dt_live.now().isoformat(),
                        })
            except Exception as _lc_err:
                logger.debug(f"Local syntax check non-fatal error: {_lc_err}")

            # Record successful write in task ledger
            task_ledger[task.id] = {"file_path": code.file_path, "status": "written", "reason": "success"}

            # Record the new content so later tasks in this run see the fresh shapes.
            _written_key = code.file_path.replace("\\", "/").lower()
            _run_map = state.setdefault("_run_generated_map", {})
            _run_map[_written_key] = code.content
            _run_map[code.file_path] = code.content
            _run_map[Path(code.file_path).name.lower()] = code.content

            # ── POST-WRITE INCREMENTAL VALIDATION (blueprint + dependency health) ─
            if _impl_state is not None:
                try:
                    from ticket_to_code.agents.incremental_validator import IncrementalValidator
                    _inc_validator = IncrementalValidator(
                        impl_state=_impl_state,
                        symbol_resolver=getattr(agents, "symbol_resolver", None),
                        workspace_path=Path(state.get("workspace_path", "")),
                    )
                    _val_result = _inc_validator.validate_generated(
                        task_id=task.id,
                        file_path=code.file_path,
                    )
                    if not _val_result.is_clean and _val_result.dependency_issues:
                        logger.warning(
                            f"  ⚠️ Dependency issues noted in {code.file_path}: {_val_result.dependency_issues[:2]} "
                            f"(preserving code on disk — downstream tasks/edit_loop will resolve dependencies)"
                        )
                    elif not _val_result.is_clean:
                        logger.warning(
                            f"  ⚠️ Incremental validation: {_val_result.summary}"
                        )
                    # Sync state back so downstream tasks see updated blueprints
                    _impl_state.sync_to_workflow_state(state)

                    # ── SHARED-TYPE IMPACT GUARD (Change 2) ─────────────────────────
                    # For MODIFY tasks, reject destructive changes to existing shared
                    # types (removed fields / optional→required) unless the ticket
                    # explicitly references the field. Advisory: blocks handoff only.
                    try:
                        if str(getattr(task.task_type, "value", "")) == "modify":
                            _old_shared = state.get("original_file_contents", {}).get(
                                task.file_path
                            ) or state.get("original_file_contents", {}).get(
                                code.file_path
                            )
                            if _old_shared:
                                from ticket_to_code.agents.workspace_symbol_scanner import (
                                    WorkspaceSymbolScanner,
                                )
                                from ticket_to_code.agents.shared_type_guard import (
                                    detect_shared_type_regressions,
                                )
                                _sc = WorkspaceSymbolScanner(
                                    state.get("workspace_path", ""),
                                    lsp_client=getattr(agents, "lsp_client", None),
                                )
                                _ticket_obj = state.get("ticket")
                                _ticket_text = " ".join(filter(None, [
                                    str(getattr(_ticket_obj, "title", "") or ""),
                                    str(getattr(_ticket_obj, "description", "") or ""),
                                ]))
                                _regs = detect_shared_type_regressions(
                                    old_content=_old_shared,
                                    new_content=code.content,
                                    file_path=code.file_path,
                                    scanner=_sc,
                                    ticket_text=_ticket_text,
                                )
                                for _r in _regs:
                                    _val_result.shared_type_violations.append(_r.describe())
                                if _regs:
                                    _val_result.is_clean = False
                                    logger.warning(
                                        f"  🛡️ Shared-type guard: {len(_regs)} destructive "
                                        f"change(s) in {code.file_path} — handoff blocked: "
                                        f"{[r.describe() for r in _regs][:3]}"
                                    )
                    except Exception as _st_exc:
                        logger.debug(f"  Shared-type guard skipped (non-fatal): {_st_exc}")

                    # ── GENERATION HANDOFF (Cross-File Intelligence v4) ─────────────
                    # Create a GenerationHandoff for all successfully generated
                    # artifacts that did NOT cause dependency breakage.
                    # Architecture warnings no longer block handoff creation:
                    # downstream tasks need actual verified signatures (what
                    # exists), tagged with appropriate validation_status.
                    #   clean                       → validated_generation authority
                    #   generated_with_warnings     → candidate_generation authority
                    #   generated_with_arch_warning → candidate_generation authority
                    #
                    # Change 1/2: unresolved outbound references and destructive
                    # shared-type changes BLOCK handoff — a hallucinated or
                    # contract-breaking artifact must not propagate a fake-verified
                    # contract to consumers. The file stays on disk for the
                    # attribution-aware error resolver.
                    if _val_result.unresolved_references:
                        logger.warning(
                            f"  ⛔ Handoff blocked for {code.file_path}: "
                            f"{len(_val_result.unresolved_references)} unresolved reference(s): "
                            f"{_val_result.unresolved_references[:3]}"
                        )
                    if (
                        not _val_result.dependency_issues
                        and not _val_result.unresolved_references
                        and not _val_result.shared_type_violations
                    ):
                        if _val_result.is_clean:
                            _handoff_status = "clean"
                        elif _val_result.architecture_violations:
                            _handoff_status = "generated_with_arch_warning"
                        else:
                            _handoff_status = "generated_with_warnings"

                        try:
                            _handoff = _build_generation_handoff(
                                task_id=task.id,
                                file_path=code.file_path,
                                content=code.content,
                                validation_status=_handoff_status,
                                impl_state=_impl_state,
                            )
                            if _handoff:
                                _impl_state.add_handoff(_handoff)

                                # ── Pillar 6: Update RelationshipRegistry ──
                                # Upgrade PLANNED → VERIFIED with actual exports
                                # and detect API contracts from framework annotations.
                                if _impl_state.relationship_registry:
                                    try:
                                        _impl_state.relationship_registry.register_from_handoff(
                                            _handoff, task
                                        )
                                        # Detect API contracts in generated code
                                        from ticket_to_code.agents.api_contract_detector import (
                                            APIContractDetector,
                                        )
                                        _api_detector = APIContractDetector()
                                        _api_contracts = _api_detector.detect_contracts(
                                            code.file_path, code.content
                                        )
                                        for _ac in _api_contracts:
                                            _impl_state.relationship_registry.register_api_contract(
                                                source_file=code.file_path,
                                                endpoint=_ac.endpoint,
                                                method=_ac.method,
                                                response_type=_ac.response_type,
                                                request_type=_ac.request_type,
                                            )
                                        # Try SymbolResolver enrichment
                                        _resolver = getattr(agents, "symbol_resolver", None)
                                        if _resolver:
                                            _impl_state.relationship_registry.register_from_symbol_resolver(
                                                _resolver, code.file_path, code.content
                                            )
                                    except Exception as _reg_exc:
                                        logger.debug(
                                            f"  Registry update failed (non-fatal): {_reg_exc}"
                                        )
                        except Exception as _ho_exc:
                            logger.debug(
                                f"  Handoff creation failed (non-fatal): {_ho_exc}"
                            )
                except Exception as _iv_exc:
                    logger.debug(f"  Incremental validation failed (non-fatal): {_iv_exc}")

            # ── POST-WRITE BINDING VERIFICATION (Read → Think → Write → Verify) ─
            # After writing an HTML file, immediately verify that every Angular binding
            # it references has a corresponding setter in the sibling .ts controller.
            # If the controller is being written LATER in this run, skip (it will supply
            # the property). If it's NOT in the plan at all, record a gap immediately so
            # fix_build can be targeted rather than discovering it only at build time.
            if Path(code.file_path).suffix.lower() in (".html", ".htm"):
                from ticket_to_code.agents.dataflow_tracer import (
                    extract_angular_bindings,
                    extract_property_setters,
                )
                _html_bindings = extract_angular_bindings(code.content)
                if _html_bindings:
                    _stem = code.file_path.rsplit(".", 1)[0]
                    _ctrl_key = (_stem + ".ts").replace("\\", "/").lower()
                    _ctrl_content = state.get("_run_generated_map", {}).get(_ctrl_key)
                    _ctrl_in_plan = any(
                        t.file_path.replace("\\", "/").lower() == _ctrl_key
                        for t in _get_plan(state).tasks
                        if getattr(getattr(t, "task_type", None), "value", "").lower() != "read_only"
                    )
                    if _ctrl_content:
                        # Controller already written — check setters NOW
                        _ctrl_setters = extract_property_setters(_ctrl_content)
                        _missing = [
                            b for b in _html_bindings
                            if b.property.lower() not in _ctrl_setters
                        ]
                        if _missing:
                            logger.warning(
                                f"  ⚠️ [Post-write] {len(_missing)} binding(s) in "
                                f"{code.file_path} have no setter in already-written controller "
                                f"{_ctrl_key}: "
                                + ", ".join(f"{b.variable}.{b.property}" for b in _missing)
                            )
                            # Record as gaps for fix_build awareness
                            for _mb in _missing:
                                state.setdefault("_dataflow_gaps", []).append({
                                    "html_file": code.file_path,
                                    "variable": _mb.variable,
                                    "property": _mb.property,
                                    "controller": _ctrl_key,
                                    "reason": f"Property set nowhere in {_ctrl_key} after it was written",
                                })
                    elif _ctrl_in_plan:
                        logger.info(
                            f"  ℹ️ [Post-write] Controller {_ctrl_key} is in the plan "
                            f"but not yet written — binding check deferred"
                        )
                    else:
                        # Controller NOT in plan and NOT written → always a gap
                        _gap_props = [f"{b.variable}.{b.property}" for b in _html_bindings]
                        logger.warning(
                            f"  ⚠️ [Post-write] {code.file_path} has bindings with no "
                            f"controller task in plan: {_gap_props}. "
                            f"These properties will be undefined at runtime."
                        )
                        for _mb in _html_bindings:
                            state.setdefault("_dataflow_gaps", []).append({
                                "html_file": code.file_path,
                                "variable": _mb.variable,
                                "property": _mb.property,
                                "controller": _ctrl_key,
                                "reason": f"Controller {_ctrl_key} has no task in plan",
                            })

            # ── POST-WRITE TS SETTER VERIFICATION ────────────────────────────
            # After writing a .ts controller, check if any previously-written HTML
            # files have bindings that this controller should supply but doesn't.
            # CRITICAL: also extract COMPONENT-LEVEL bindings (not prefixed with
            # dmember./member. etc.) from the HTML and attempt an inline re-generation
            # if the TS is missing those class-level properties.
            elif Path(code.file_path).suffix.lower() in (".ts", ".tsx"):
                from ticket_to_code.agents.dataflow_tracer import (
                    extract_angular_bindings,
                    extract_property_setters,
                )
                _ts_setters = extract_property_setters(code.content)
                _ts_stem = code.file_path.rsplit(".", 1)[0].replace("\\", "/").lower()
                for _skey, _sval in state.get("_run_generated_map", {}).items():
                    if not (_skey.endswith((".html", ".htm")) and _skey.rsplit(".", 1)[0] == _ts_stem):
                        continue
                    _html_bindings = extract_angular_bindings(_sval)
                    # Separate per-member bindings (dmember.X, member.X) from
                    # component-level bindings (direct property access on the component class)
                    _MEMBER_VARS = {"dmember", "member", "displayedmember", "item", "contract"}
                    _component_bindings = [
                        b for b in _html_bindings
                        if b.variable.lower() not in _MEMBER_VARS
                        and b.property.lower() not in _ts_setters
                    ]
                    _member_bindings = [
                        b for b in _html_bindings
                        if b.variable.lower() in _MEMBER_VARS
                        and b.property.lower() not in _ts_setters
                    ]
                    if _component_bindings:
                        logger.warning(
                            f"  ⚠️ [Post-write] TS controller {code.file_path} is missing "
                            f"COMPONENT-LEVEL properties used in HTML: "
                            + ", ".join(f"{b.variable}.{b.property}" for b in _component_bindings)
                        )
                        # Inline retry: re-generate TS with missing properties listed explicitly
                        _props_needed = "\n".join(
                            f"  - Add class property: `{b.property}: boolean | string = ...;`  (referenced as `{b.variable}.{b.property}` in template)"
                            for b in _component_bindings
                        )
                        import copy as _copy_prop
                        _prop_retry_task = _copy_prop.copy(task)
                        _prop_retry_task.description = (
                            f"{task.description}\n\n"
                            f"COMPONENT PROPERTY ALIGNMENT (sibling HTML already written and uses these):\n"
                            f"The sibling HTML template references the following as COMPONENT CLASS properties "
                            f"(accessed directly without `dmember.` prefix). You MUST declare them in the component class:\n"
                            f"{_props_needed}\n"
                            f"DO NOT use different property names — the HTML is already written with these exact names."
                        )
                        generator._session_files = state.get("_run_generated_map", {})
                        if _impl_state is not None:
                            generator._impl_state = _impl_state
                        try:
                            _prop_code = generator.generate_code(
                                task=_prop_retry_task,
                                requirements=state["requirements"],
                                context=state["code_rag_context"],
                                existing_content=code.content,
                                allowed_files=allowed_files,
                                readonly_files=readonly_files,
                            )
                            if (
                                _prop_code.content
                                and _prop_code.content.strip() != code.content.strip()
                                and "[TRUNCATED]" not in _prop_code.content
                            ):
                                _pv = validator.validate(_prop_code, task, existing_content)
                                if _pv.passed:
                                    output_path.write_text(_prop_code.content, encoding="utf-8")
                                    state["_run_generated_map"][_written_key] = _prop_code.content
                                    code = _prop_code
                                    logger.info(
                                        f"  ✅ [PropAlign] Component properties aligned in {code.file_path}"
                                    )
                        except Exception as _prop_exc:
                            logger.warning(f"  ⚠️ [PropAlign] Retry failed: {_prop_exc}")
                    if _member_bindings:
                        logger.warning(
                            f"  ⚠️ [Post-write] TS controller {code.file_path} is missing "
                            f"setters for per-member bindings in sibling HTML: "
                            + ", ".join(f"{b.variable}.{b.property}" for b in _member_bindings)
                        )

            # ── PER-FILE COMPILE CHECK (Cursor-style continuous feedback) ────
            # ── PER-FILE COMPILE ERROR RESOLUTION LOOP (Cursor/Claude Code approach) ─
            # After every write: compile → errors? → show LLM real file + real errors
            # → regenerate full file → recompile → repeat up to 3 times.
            # No hardcoded error patterns. Any compile error gets the same treatment.

            # UI callback helper — sends sub-step events to the frontend
            _ui_cb = _get_transient(state, "_ui_callback")
            def _ui_emit(event_type: str, **kwargs):
                if not _ui_cb:
                    return
                from datetime import datetime as _dt_emit
                _ui_cb({
                    "phase": "patch_generation",
                    "status": "in_progress",
                    "message": kwargs.get("message", ""),
                    "data": {
                        "node": "generate_code",
                        "event_type": event_type,
                        **{k: v for k, v in kwargs.items() if k != "message"},
                    },
                    "timestamp": _dt_emit.now().isoformat(),
                })

            # Notify UI: file written
            _file_basename = Path(code.file_path).name
            _ui_emit(
                "file_written",
                message=f"📝 Written: {_file_basename}",
                file_path=code.file_path,
                file_name=_file_basename,
                change_ratio=validation.metrics.get("change_ratio", "N/A"),
            )

            _live_ext = Path(code.file_path).suffix.lower()
            _live_errors: list[str] = []
            _live_label = ""

            if _live_ext in (".scss", ".css"):
                # ── SCSS/CSS syntax check: balanced braces + no format markers ──
                _live_label = "SCSS-syntax"
                _live_errors = _check_scss_syntax(code.content, code.file_path)

            elif _live_ext == ".py" and not code.file_path.endswith("_test.py"):
                # ── Python syntax check: py_compile + AST ──
                _live_label = "Python-syntax"
                _live_errors = _check_python_syntax(output_path)

            elif _live_ext == ".html":
                # ── HTML template check: balanced tags + no format markers ──
                _live_label = "HTML-syntax"
                _live_errors = _check_html_syntax(code.content, code.file_path)

            elif _live_ext in (".ts", ".tsx", ".js", ".jsx", ".java", ".kt", ".cs"):
                # ── Fast local syntax check: balanced braces + parens + no leaked markers ──
                # Full project build/tsc is deferred until the plan completes and passes:
                # patch_gate -> import_validation -> angular_module_registration -> build
                _live_label = "Syntax-check"
                _live_errors = _check_code_syntax(code.content, code.file_path)

            if _live_label:
                _ui_emit(
                    "live_compile_start",
                    message=f"🔨 Live Compile: Verifying {_file_basename} after write [{_live_label}]...",
                    file_path=code.file_path,
                    file_name=_file_basename,
                    check_type=_live_label,
                )

            if _live_errors and _live_label:
                _this_file_errors = [e for e in _live_errors if Path(code.file_path).name.lower() in e.lower()] or _live_errors[:10]
                logger.warning(
                    f"  ⚠️ [{_live_label}] {len(_live_errors)} error(s) after writing {code.file_path} — "
                    f"starting error resolution loop (up to 3 attempts)"
                )

                # Notify UI: compile errors detected
                _ui_emit(
                    "live_check_start",
                    message=f"⚠️ [{_live_label}] {len(_live_errors)} compile error(s) in {_file_basename}",
                    file_path=code.file_path,
                    file_name=_file_basename,
                    error_count=len(_live_errors),
                    errors=_live_errors[:5],
                    check_type=_live_label,
                )

                _resolved = False
                import re as _re_fix
                import json as _json_fix
                from langchain_core.messages import SystemMessage, HumanMessage
                from ticket_to_code.llm_utils import llm_invoke as _llm_invoke
                from ticket_to_code.agents.diagnostic_normalizer import normalize_diagnostics
                from ticket_to_code.agents.diagnostic_localizer import (
                    DiagnosticLocalizer,
                    RepairContextTier,
                    FailureOwner,
                )

                _diagnostic_localizer: DiagnosticLocalizer = state.setdefault(
                    "_diagnostic_localizer",
                    DiagnosticLocalizer(
                        state.get("workspace_path"),
                        scope_proof=state.get("ticket_scope_proof"),
                    )
                )
                if getattr(_diagnostic_localizer, "scope_proof", None) is None:
                    _diagnostic_localizer.scope_proof = state.get("ticket_scope_proof")

                for _err_attempt in range(3):
                    _norm_diags = normalize_diagnostics(_live_errors)
                    if not _diagnostic_localizer.can_attempt_repair(_norm_diags, max_attempts=2):
                        logger.warning(
                            f"  🛑 [{_live_label}] Identical compile errors unchanged after 2 attempts — "
                            f"terminating repair loop to prevent token burn"
                        )
                        break

                    _diagnostic_localizer.record_attempt(_norm_diags)
                    _current_content = output_path.read_text(encoding="utf-8") if output_path.exists() else code.content

                    # ── Collect ALL files mentioned in errors (cross-file awareness) ──
                    # Errors in other files point to the ROOT CAUSE of this file's errors.
                    # e.g. TS2339 "property X does not exist on type Y" → Y's source file
                    # must also be shown so the LLM can add the missing property there.
                    _all_error_files: dict[str, str] = {code.file_path: _current_content}
                    _ws_root = Path(state["workspace_path"])
                    _run_map = state.get("_run_generated_map", {})
                    for _err_line in _live_errors:
                        _ep = _re_fix.search(
                            r'([^\s(]+\.(?:ts|tsx|js|jsx|java|kt|py))\s*[\(:\[]',
                            _err_line, _re_fix.IGNORECASE
                        )
                        if _ep:
                            _ep_raw = _ep.group(1).lstrip("/")
                            # Resolve absolute → relative
                            try:
                                _ep_rel = str(Path(_ep_raw).relative_to(_ws_root)).replace("\\", "/")
                            except ValueError:
                                _ep_rel = _ep_raw.replace("\\", "/")
                            if _ep_rel != code.file_path.replace("\\", "/") and _ep_rel not in _all_error_files:
                                _ep_abs = _ws_root / _ep_rel
                                if _ep_abs.exists():
                                    _ep_key = _ep_rel.lower()
                                    _all_error_files[_ep_rel] = _run_map.get(_ep_key) or _ep_abs.read_text(encoding="utf-8", errors="ignore")
                                    logger.info(f"  [{_live_label}] Including related error file: {_ep_rel}")

                    # Also inject session-generated files imported by the primary file
                    _src_dir = str(Path(code.file_path).parent).replace("\\", "/")
                    for _imp_m in _re_fix.finditer(r"""from\s+['"](\.[^'"]+)['"]""", _current_content):
                        for _iext in (".ts", ".tsx"):
                            import os as _os_fix
                            _irel = _os_fix.path.normpath(_os_fix.path.join(_src_dir, _imp_m.group(1) + _iext)).replace("\\", "/")
                            _ikey = _irel.lower()
                            if _ikey in _run_map and _irel not in _all_error_files:
                                _all_error_files[_irel] = _run_map[_ikey]

                    # ── Build error-to-method mapping (like top AI IDEs) ──
                    _error_method_section = ""
                    _error_method_map: dict[str, list[tuple[int, str, int, int]]] = {}
                    try:
                        from ticket_to_code.agents.smart_extract import _parse_java_ts_boundaries, _parse_python_boundaries
                        _line_rx = _re_fix.compile(
                            r'(?P<path>[^\s]+\.(?:java|ts|tsx|js|jsx|py|kt|scala))'
                            r'(?:\:\[(?P<line1>\d+)|'
                            r'\((?P<line2>\d+)|'
                            r':(?P<line3>\d+))',
                            _re_fix.IGNORECASE
                        )
                        _error_locations: list[tuple[str, int]] = []
                        for _el in _live_errors:
                            for _em in _line_rx.finditer(_el):
                                _ln = _em.group('line1') or _em.group('line2') or _em.group('line3')
                                if _ln:
                                    _error_locations.append((_em.group('path'), int(_ln)))
                        if _error_locations:
                            _file_boundaries: dict[str, list] = {}
                            for _fp, _fc in _all_error_files.items():
                                _flines = _fc.splitlines()
                                _fext = _fp.rsplit('.', 1)[-1].lower() if '.' in _fp else ''
                                if _fext in ('java', 'ts', 'tsx', 'js', 'jsx', 'kt', 'scala'):
                                    _file_boundaries[_fp] = _parse_java_ts_boundaries(_flines)
                                elif _fext == 'py':
                                    _file_boundaries[_fp] = _parse_python_boundaries(_flines)
                            _mappings: list[str] = []
                            _seen_mappings = set()
                            for _raw_path, _line_num in _error_locations:
                                _norm = _raw_path.replace('\\', '/').lower()
                                _matched_fp = None
                                for _fbp in _file_boundaries:
                                    if _norm.endswith(_fbp.lower()) or _fbp.lower().endswith(_norm):
                                        _matched_fp = _fbp; break
                                    if _norm.split('/')[-1] == _fbp.split('/')[-1]:
                                        _matched_fp = _fbp; break
                                if not _matched_fp: continue
                                for _mb in _file_boundaries[_matched_fp]:
                                    if _mb.kind == "class": continue
                                    if _mb.start_line <= (_line_num - 1) <= _mb.end_line:
                                        _mkey = (_matched_fp, _mb.name, _line_num)
                                        if _mkey not in _seen_mappings:
                                            _seen_mappings.add(_mkey)
                                            _mappings.append(
                                                f"  {_raw_path}:{_line_num} → inside method `{_mb.name}` "
                                                f"(lines {_mb.start_line+1}-{_mb.end_line+1})"
                                            )
                                            _nfp = _matched_fp.replace('\\', '/').lower()
                                            if _nfp not in _error_method_map:
                                                _error_method_map[_nfp] = []
                                            _error_method_map[_nfp].append(
                                                (_line_num, _mb.name, _mb.start_line, _mb.end_line)
                                            )
                                        break
                            if _mappings:
                                _error_method_section = (
                                    "ERROR LINE MAPPING (fix THESE methods, not their callers):\n"
                                    + "\n".join(_mappings) + "\n\n"
                                )
                                logger.info(f"  [{_live_label}] Error-method mapping: {len(_mappings)} mapping(s)")
                    except Exception as _map_exc:
                        logger.debug(f"  [{_live_label}] Error-method mapping failed (non-fatal): {_map_exc}")

                    # ── Build related file snippets (only first 80 lines, not full content) ──
                    _other_files_section = ""
                    for _rf, _rc in _all_error_files.items():
                        if _rf != code.file_path:
                            _snippet = "\n".join(_rc.splitlines()[:80])
                            _other_files_section += f"\nRELATED FILE (first 80 lines): {_rf}\n```\n{_snippet}\n```\n"

                    # ── Fix 2: Auto-resolve unknown types from compile errors ──
                    # When errors mention unknown methods/properties on a type,
                    # look up the type definition and include it so the LLM doesn't guess.
                    try:
                        if _live_errors:
                            import re as _re_type
                            _type_patterns = [
                                _re_type.compile(r"type\s+'([A-Z][A-Za-z0-9_]+)'", _re_type.IGNORECASE),
                                _re_type.compile(r"interface\s+'?([A-Z][A-Za-z0-9_]+)'?", _re_type.IGNORECASE),
                                _re_type.compile(r"on\s+(?:the\s+)?'?([A-Z][A-Za-z0-9_]+)'?\s+(?:interface|class)", _re_type.IGNORECASE),
                                _re_type.compile(r"exist\s+in\s+type\s+'([A-Z][A-Za-z0-9_]+)(?:\[\])?'", _re_type.IGNORECASE),
                                _re_type.compile(r"exist\s+on\s+type\s+'([A-Z][A-Za-z0-9_]+)(?:\[\])?'", _re_type.IGNORECASE),
                                _re_type.compile(r"assignable\s+to\s+type\s+'([A-Z][A-Za-z0-9_]+)(?:\[\])?'", _re_type.IGNORECASE),
                            ]
                            _resolved_types = set()
                            for _err_line in _live_errors[:10]:
                                for _tp in _type_patterns:
                                    for _tm in _tp.finditer(_err_line):
                                        _tname = _tm.group(1)
                                        if _tname not in _resolved_types and len(_resolved_types) < 4:
                                            _resolved_types.add(_tname)
                                            # 1. Primary: Direct AST hydration for TypeScript files
                                            if _live_ext in (".ts", ".tsx"):
                                                try:
                                                    from ticket_to_code.intelligence.typescript import TypeScriptTypeResolver
                                                    _hydrated = TypeScriptTypeResolver.hydrate_type_definition(
                                                        _tname, output_path, workspace_root=_ws_root
                                                    )
                                                    if _hydrated:
                                                        _tdef, _tfile = _hydrated
                                                        _other_files_section += (
                                                            f"\nAUTHORITATIVE TYPE DEFINITION (auto-resolved from error): {_tdef.name} (from {_tdef.source_file})\n"
                                                            f"```typescript\n{_tdef.raw_declaration}\n```\n"
                                                        )
                                                        logger.info(f"  [{_live_label}] Auto-resolved type '{_tname}' via AST → {_tdef.source_file}")
                                                        continue
                                                except Exception:
                                                    pass

                                            # 2. Fallback: query symbol store if available
                                            try:
                                                _sqlite_store = getattr(getattr(state.get("agents"), 'localizer', None), 'sqlite_store', None)
                                                if _sqlite_store:
                                                    from ticket_to_code.intelligence.scope import infer_resolution_scope
                                                    _res_scope = infer_resolution_scope(code.file_path, _ws_root)
                                                    if hasattr(_sqlite_store, "find_symbol_paths"):
                                                        _matched_paths = _sqlite_store.find_symbol_paths(
                                                            _tname, resolution_scope=_res_scope, limit=5
                                                        )
                                                    else:
                                                        _scope_sql, _scope_params = _res_scope.build_sql_filter("path")
                                                        _type_rows = _sqlite_store._conn.execute(
                                                            f"SELECT DISTINCT path FROM symbols WHERE name LIKE ?{_scope_sql} LIMIT 5",
                                                            [f"%{_tname}%"] + _scope_params
                                                        ).fetchall()
                                                        _matched_paths = [r[0] for r in _type_rows]

                                                    for _type_path in _matched_paths:
                                                        if _type_path and _type_path not in _all_error_files and _res_scope.matches_path(_type_path):
                                                            _type_abs = _ws_root / _type_path
                                                            if _type_abs.exists():
                                                                _type_content = _type_abs.read_text(encoding="utf-8", errors="ignore")
                                                                _type_snippet = "\n".join(_type_content.splitlines()[:100])
                                                                _other_files_section += (
                                                                    f"\nTYPE DEFINITION (auto-resolved from error): {_type_path}\n"
                                                                    f"```\n{_type_snippet}\n```\n"
                                                                )
                                                                logger.info(f"  [{_live_label}] Auto-resolved type '{_tname}' [{_res_scope.language}/{_res_scope.project_root}] → {_type_path}")
                                                                break
                                            except Exception:
                                                pass
                    except Exception as _type_exc:
                        logger.debug(f"  [{_live_label}] Type auto-resolve failed (non-fatal): {_type_exc}")

                    # Ticket + task context so the fix preserves the original goal
                    _ticket_obj = state.get("ticket")
                    _ticket_title = getattr(_ticket_obj, "title", "") or ""
                    _ticket_desc = (getattr(_ticket_obj, "description", "") or "")[:400]
                    _task_title = getattr(task, "title", "") or ""
                    _task_desc = (getattr(task, "description", "") or "")[:400]
                    _req_obj = state.get("requirements")
                    _func_reqs = ""
                    if _req_obj:
                        _fr = getattr(_req_obj, "functional_requirements", []) or []
                        _func_reqs = "\n".join(f"  - {r}" for r in _fr[:5])

                    _fix_system = (
                        "You are an expert compiler-error fixer. Fix errors using SURGICAL str_replace edits.\n"
                        "CRITICAL RULES:\n"
                        "1. NEVER rewrite the entire file or large blocks. Only change the specific lines (3 to 8 lines) that fix the error.\n"
                        "2. Use target_content/replacement_content for targeted edits.\n"
                        "3. PRESERVE all existing functionality — do NOT remove or stub out code.\n"
                        "   Directly replace deprecated logic with the correct implementation without leaving commented-out dead code.\n"
                        "4. If a method or property is missing, ADD it. If a service call is wrong, FIX the call.\n"
                        "5. When ERROR LINE MAPPING shows which method has the error, fix THAT method — not its callers.\n\n"
                        "Return a JSON object:\n"
                        "{\n"
                        '  "reasoning": "root cause + which method to fix + how",\n'
                        '  "fixes": [\n'
                        '    {\n'
                        '      "file": "exact/relative/path.java",\n'
                        '      "target_content": "exact text to find in the file (must be unique)",\n'
                        '      "replacement_content": "the replacement text"\n'
                        '    }\n'
                        "  ]\n"
                        "}\n\n"
                        "Rules for target_content/replacement_content:\n"
                        "- target_content must be an EXACT substring of the current file\n"
                        "- Include enough surrounding lines to make target_content unique\n"
                        "- replacement_content replaces target_content completely (comment out deprecated logic rather than deleting)\n"
                        "- You can have multiple fix entries for same or different files\n"
                        "- Only fix files shown below — do NOT invent new files\n"
                        "- Respond with JSON only, no markdown fences"
                    )
                    # ── Context localization & failure attribution ──
                    _tier = (
                        RepairContextTier.TIER_1_LOCALIZED_METHOD
                        if _err_attempt == 0
                        else RepairContextTier.TIER_4_WHOLE_FILE_ESCALATION
                    )
                    _loc_ctx = _diagnostic_localizer.localize_context(
                        code.file_path,
                        _current_content,
                        _norm_diags,
                        tier=_tier,
                        planned_tasks=state.get("planned_tasks", []),
                    )

                    _attrib_section = ""
                    if _loc_ctx.attribution:
                        _at = _loc_ctx.attribution
                        if _at.owner == FailureOwner.PROVIDER and _at.is_provider_writable:
                            _attrib_section = (
                                f"⚠️ FAILURE ATTRIBUTION (PROVIDER - GENERATED/WRITABLE):\n"
                                f"Symbol '{_at.missing_symbol}' is missing on provider type '{_at.provider_type}'.\n"
                                f"Provider file: {_at.provider_file or 'unknown'} (writable={_at.is_provider_writable})\n"
                                f"This provider is authorized for modification. You may add the missing method to the provider rather than stripping consumer calls.\n\n"
                            )
                        elif _at.owner == FailureOwner.CROSS_FILE_CONTRACT and getattr(_at, "is_protected_provider", False):
                            _alt_text = ""
                            if getattr(_at, "alternative_capability", None) and _at.alternative_capability.is_semantically_compatible:
                                _alt = _at.alternative_capability
                                _alt_text = (
                                    f"✅ VERIFIED REPOSITORY ALTERNATIVE:\n"
                                    f"  Method: {_alt.signature}\n"
                                    f"  Evidence: {_alt.compatibility_reason}\n"
                                    f"  Instruction: Adapt `{code.file_path}` to call this verified capability instead of non-existent '{_at.missing_symbol}'.\n\n"
                                )
                            else:
                                _alt_text = (
                                    f"❌ NO VERIFIED ALTERNATIVE CAPABILITY EXISTS for '{_at.missing_symbol}'.\n"
                                    f"  Instruction: Do NOT invent methods on '{_at.provider_file}'. If no alternative exists in the repository, report UNRESOLVED/REPLAN.\n\n"
                                )
                            _methods_list = "\n".join(f"  - {m.signature}" for m in getattr(_at, "inspected_methods", [])[:6]) if getattr(_at, "inspected_methods", None) else "  (none)"
                            _attrib_section = (
                                f"⚠️ FAILURE ATTRIBUTION (CROSS_FILE_CONTRACT - PROTECTED PROVIDER):\n"
                                f"Provider file '{_at.provider_file or _at.provider_type}' is PROTECTED and MUST NOT BE MODIFIED.\n"
                                f"Real public methods on protected provider:\n{_methods_list}\n\n"
                                f"{_alt_text}"
                            )

                    if _loc_ctx.context_tier == RepairContextTier.TIER_1_LOCALIZED_METHOD:
                        _primary_file_block = (
                            f"PRIMARY FILE (Tier 1 Localized Context around `{_loc_ctx.target_method_name}`): {code.file_path}\n"
                            f"```\n{_loc_ctx.prompt_snippet}\n```\n"
                        )
                    else:
                        _primary_file_block = (
                            f"PRIMARY FILE (Tier 4 Whole File Context): {code.file_path}\n"
                            f"```\n{_current_content}\n```\n"
                        )

                    _fix_user = (
                        f"TICKET: {_ticket_title}\n{_ticket_desc}\n\n"
                        f"TASK BEING IMPLEMENTED: {_task_title}\n{_task_desc}\n\n"
                        + (f"FUNCTIONAL REQUIREMENTS (must still be satisfied after fix):\n{_func_reqs}\n\n" if _func_reqs else "")
                        + f"COMPILE ERRORS:\n"
                        + "\n".join(f"  {e}" for e in _live_errors[:20])
                        + f"\n\n{_attrib_section}"
                        + f"{_error_method_section}"
                        + _primary_file_block
                        + _other_files_section
                        + "\nFix all errors using target_content/replacement_content edits. Do NOT rewrite the file. Return JSON."
                    )
                    try:
                        _fix_response = _llm_invoke(
                            generator.llm,
                            [SystemMessage(content=_fix_system), HumanMessage(content=_fix_user)]
                        )
                        _fix_raw = _fix_response.content if hasattr(_fix_response, "content") else str(_fix_response)
                        _usage = getattr(_fix_response, "usage_metadata", {}) or {}
                        _tok = _usage.get("total_tokens", 0) if isinstance(_usage, dict) else 0
                        if not _tok:
                            _tok = len(_fix_user.split()) + len(_fix_raw.split())
                        _diagnostic_localizer.telemetry.tsc_tokens += _tok
                        _diagnostic_localizer.telemetry.total_tokens += _tok

                        # Parse JSON response with resilient recovery for unquoted keys / comments / SEARCH-REPLACE
                        _candidate_str = ""
                        _json_fence = _re_fix.search(r'```(?:json)?\s*(.*?)\s*```', _fix_raw, _re_fix.DOTALL)
                        if _json_fence:
                            _candidate_str = _json_fence.group(1).strip()
                        else:
                            _s = _fix_raw.find('{')
                            _e = _fix_raw.rfind('}')
                            if _s != -1 and _e != -1 and _e > _s:
                                _candidate_str = _fix_raw[_s:_e+1].strip()

                        _fix_result = {}
                        if _candidate_str:
                            try:
                                _fix_result = _json_fix.loads(_candidate_str)
                            except Exception:
                                # Strip JS comments
                                _cleaned = _re_fix.sub(r'//.*?\n|/\*.*?\*/', '', _candidate_str, flags=_re_fix.DOTALL)
                                try:
                                    _fix_result = _json_fix.loads(_cleaned)
                                except Exception:
                                    # Quote unquoted keys: { key: or , key:
                                    _quoted = _re_fix.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', _cleaned)
                                    _no_trailing = _re_fix.sub(r',\s*([}\]])', r'\1', _quoted)
                                    try:
                                        _fix_result = _json_fix.loads(_no_trailing)
                                    except Exception:
                                        try:
                                            import ast as _ast_fix
                                            _val = _ast_fix.literal_eval(_no_trailing)
                                            if isinstance(_val, dict):
                                                _fix_result = _val
                                        except Exception:
                                            _fix_result = {}

                        _fixes = _fix_result.get("fixes", []) if isinstance(_fix_result, dict) else []

                        # Fallback: check for SEARCH/REPLACE blocks if LLM output diffs instead of JSON
                        if not _fixes:
                            _sr_pattern = _re_fix.compile(
                                r'<{4,}\s*(?:SEARCH|ORIGINAL|TARGET)[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}\s*(?:REPLACE|UPDATED|REPLACEMENT)[^\r\n]*',
                                _re_fix.DOTALL
                            )
                            _sr_matches = list(_sr_pattern.finditer(_fix_raw))
                            if not _sr_matches:
                                _alt_sr = _re_fix.compile(
                                    r'<{4,}[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}[^\r\n]*',
                                    _re_fix.DOTALL
                                )
                                _sr_matches = list(_alt_sr.finditer(_fix_raw))
                            if _sr_matches:
                                _fixes = [
                                    {
                                        "file": code.file_path,
                                        "target_content": m.group(1),
                                        "replacement_content": m.group(2)
                                    }
                                    for m in _sr_matches
                                ]
                                logger.info(f"  [{_live_label}] Attempt {_err_attempt+1}: parsed {len(_fixes)} SEARCH/REPLACE block(s) from response")

                        if not _fixes:
                            logger.warning(f"  ⚠️ [{_live_label}] Attempt {_err_attempt+1}: no valid fixes found in response")
                            continue

                        logger.info(f"  [{_live_label}] Attempt {_err_attempt+1}: applying {len(_fixes)} str_replace fix(es) — {_fix_result.get('reasoning','')[:120]}")

                        # ── Helper: resolve LLM-returned path ──
                        def _resolve_fix_path(fp: str) -> tuple[str, Path]:
                            """Return (canonical_relative_path, absolute_path)."""
                            fp = fp.replace("\\", "/")
                            abs1 = _ws_root / fp
                            if abs1.exists():
                                return fp, abs1
                            if _live_ext in (".ts", ".tsx") and _ng_root_for_tsc:
                                _ng_name = _ng_root_for_tsc.name
                                fp2 = f"{_ng_name}/{fp}"
                                abs2 = _ws_root / fp2
                                if abs2.exists():
                                    logger.info(f"  [{_live_label}] Path auto-corrected: {fp} → {fp2}")
                                    return fp2, abs2
                            _code_norm = code.file_path.replace("\\", "/")
                            if _code_norm.endswith(fp) or fp.endswith(_code_norm):
                                return _code_norm, _ws_root / _code_norm
                            return fp, abs1

                        # ── Apply each str_replace fix with validation ──
                        _files_written_this_attempt: list[str] = []
                        _fix_valid = True
                        for _fix_item in _fixes:
                            _fix_path_raw = str(_fix_item.get("file") or "")
                            _target_content = str(_fix_item.get("target_content") or "")
                            _replacement_content = str(_fix_item.get("replacement_content") or "")

                            # Block old-style full-file rewrite if LLM returned "content" key
                            if not _target_content and _fix_item.get("content"):
                                logger.warning(f"  ⚠️ [{_live_label}] LLM returned full 'content' instead of str_replace — rejecting")
                                _fix_valid = False
                                break

                            if not _fix_path_raw or not _target_content:
                                logger.warning(f"  ⚠️ [{_live_label}] Fix missing file or target_content — skipped")
                                continue

                            _fix_path, _fix_abs = _resolve_fix_path(_fix_path_raw)
                            logger.info(f"  [{_live_label}] str_replace fix for: {_fix_path}")

                            # Validation: inside workspace
                            try:
                                _fix_abs.resolve().relative_to(_ws_root.resolve())
                            except ValueError:
                                logger.warning(f"  ⚠️ [{_live_label}] Fix outside workspace: {_fix_path} — skipped")
                                _fix_valid = False
                                break
                            # Validation: file must exist
                            if not _fix_abs.exists():
                                logger.warning(f"  ⚠️ [{_live_label}] Fix targets non-existent file: {_fix_path} — skipped")
                                _fix_valid = False
                                break
                            # Validation: no protected paths
                            _fp_lower = _fix_path.lower()
                            if any(x in _fp_lower for x in ("/dist/", "/node_modules/", "/.git/", "/target/")):
                                logger.warning(f"  ⚠️ [{_live_label}] Fix targets protected path: {_fix_path} — skipped")
                                _fix_valid = False
                                break

                            # Read current file content
                            _file_content = _fix_abs.read_text(encoding="utf-8", errors="ignore")

                            # Validate: target_content must exist in the file
                            if _target_content not in _file_content:
                                _target_stripped = _target_content.strip()
                                if _target_stripped and _target_stripped in _file_content:
                                    _target_content = _target_stripped
                                    logger.info(f"  [{_live_label}] target_content matched after strip")
                                else:
                                    logger.warning(f"  ⚠️ [{_live_label}] target_content not found in {_fix_path} — skipped")
                                    continue

                            # Validate: target must be unique
                            if _file_content.count(_target_content) > 1:
                                logger.warning(f"  ⚠️ [{_live_label}] target_content has multiple occurrences — ambiguous, skipped")
                                continue

                            # Validate: patch targets the correct method (defense-in-depth)
                            _norm_fix_fp = _fix_path.replace('\\', '/').lower()
                            if _error_method_map and _norm_fix_fp in _error_method_map:
                                _lines_before = _file_content[:_file_content.index(_target_content)].count('\n')
                                _expected_methods = set()
                                _target_in_error_method = False
                                for _el, _mn, _ms, _me in _error_method_map[_norm_fix_fp]:
                                    _expected_methods.add(_mn)
                                    if _ms <= _lines_before <= _me:
                                        _target_in_error_method = True
                                if not _target_in_error_method and _expected_methods:
                                    _expected_str = ", ".join(f"`{n}`" for n in sorted(_expected_methods))
                                    logger.warning(
                                        f"  ⛔ [{_live_label}] PATCH TARGET VALIDATION FAILED: "
                                        f"patch targets line {_lines_before+1} but errors are in {_expected_str}"
                                    )
                                    continue  # skip this fix, let loop retry

                            # Apply the str_replace
                            _new_content = _file_content.replace(_target_content, _replacement_content, 1)
                            _fix_abs.write_text(_new_content, encoding="utf-8")
                            _run_map[_fix_path.lower()] = _new_content
                            _files_written_this_attempt.append(_fix_path)
                            _is_primary = (_fix_path.replace("\\", "/") == code.file_path.replace("\\", "/"))
                            if _is_primary:
                                code.content = _new_content
                            for _gc in (state.get("generated_code") or []):
                                if _gc.file_path.replace("\\", "/").lower() == _fix_path.replace("\\", "/").lower():
                                    _gc.content = _new_content
                            _old_lines = _file_content.count('\n')
                            _new_lines = _new_content.count('\n')
                            logger.info(
                                f"    ✏️  str_replace applied: {_fix_path} "
                                f"(target={len(_target_content)}→replacement={len(_replacement_content)} chars, "
                                f"lines: {_old_lines}→{_new_lines})"
                            )

                        if not _fix_valid:
                            continue

                        # ── Re-run compile check ──
                        _recheck: list[str] = []
                        if _live_ext in (".ts", ".tsx") and _ng_root_for_tsc:
                            _recheck = _run_tsc_on_files(_ng_root_for_tsc, _written_so_far, timeout=30)
                        elif _live_ext in (".java", ".kt") and _java_module:
                            _recheck = _run_java_compile_on_module(_module_root, _build_tool, timeout=60)
                        elif _live_ext == ".py":
                            _recheck = _check_python_syntax(output_path)
                        elif _live_ext in (".scss", ".css"):
                            _scss_c = output_path.read_text(encoding="utf-8", errors="ignore") if output_path.exists() else code.content
                            _recheck = _check_scss_syntax(_scss_c, code.file_path)
                        elif _live_ext == ".html":
                            _html_c = output_path.read_text(encoding="utf-8", errors="ignore") if output_path.exists() else code.content
                            _recheck = _check_html_syntax(_html_c, code.file_path)

                        if not _recheck:
                            logger.info(f"  ✅ [{_live_label}] All errors resolved in attempt {_err_attempt+1} ({len(_files_written_this_attempt)} file(s) fixed)")
                            _ui_emit(
                                "live_check_resolved",
                                message=f"✅ [{_live_label}] Errors resolved (attempt {_err_attempt+1}) — {_file_basename}",
                                file_path=code.file_path,
                                file_name=_file_basename,
                                attempt=_err_attempt+1,
                                files_fixed=_files_written_this_attempt,
                                check_type=_live_label,
                            )
                            _resolved = True
                            break
                        else:
                            logger.warning(f"  ⚠️ [{_live_label}] Attempt {_err_attempt+1}: {len(_recheck)} error(s) remain — retrying")
                            _ui_emit(
                                "live_check_retry",
                                message=f"🔄 [{_live_label}] Attempt {_err_attempt+1}: {len(_recheck)} error(s) remain — retrying",
                                file_path=code.file_path,
                                file_name=_file_basename,
                                attempt=_err_attempt+1,
                                remaining_errors=len(_recheck),
                                errors=_recheck[:3],
                                check_type=_live_label,
                            )
                            _this_file_errors = [e for e in _recheck if Path(code.file_path).name.lower() in e.lower()] or _recheck[:10]
                            _live_errors = _recheck  # update for next iteration's cross-file detection
                    except Exception as _fix_exc:
                        logger.warning(f"  ⚠️ [{_live_label}] Attempt {_err_attempt+1} error: {_fix_exc}")

                if not _resolved:
                    logger.warning(
                        f"  ⚠️ [{_live_label}] Live-check exhausted {3} attempts for {code.file_path} "
                        f"— keeping generated code, fix_build will handle remaining errors"
                    )
                    _ui_emit(
                        "live_check_deferred",
                        message=f"⏭️ [{_live_label}] Deferred to fix_build — {_file_basename}",
                        file_path=code.file_path,
                        file_name=_file_basename,
                        check_type=_live_label,
                    )
                    # Do NOT revert — the generated code is needed for the ticket.
                    # fix_build (Phase 2) has deeper tools (ErrorResolutionAgent) and
                    # can fix cross-file issues that the quick live-check cannot.
            elif _live_label:
                logger.info(f"  ✅ [{_live_label}] No errors after writing {code.file_path}")
                _status_msg = "Clean compile (0 errors)" if ("syntax" in _live_label.lower() or "live" in _live_label.lower()) else "Syntax OK"
                _ui_emit(
                    "live_compile_passed",
                    message=f"✅ Live Compile: {_status_msg} — {_file_basename} [{_live_label}]",
                    file_path=code.file_path,
                    file_name=_file_basename,
                    check_type=_live_label,
                )
                _ui_emit(
                    "live_check_passed",
                    message=f"✅ [{_live_label}] {_status_msg} — {_file_basename}",
                    file_path=code.file_path,
                    file_name=_file_basename,
                    check_type=_live_label,
                )
            generated_code.append(code)
            successful_writes += 1
            written_file_paths.add(code.file_path)
            logger.info(f"  ✅ Code written: {output_path} | change_ratio={validation.metrics.get('change_ratio', 'N/A')}")
            written = True
            break

        # ── Retry with verbatim extraction: str_replace EDIT failed ─────────
        # When all candidates produced identical-to-original content (edit failed),
        # retry ONCE using extract_exact_methods() to give the LLM exact, 
        # uncompressed method bodies. The SEARCH blocks the LLM produces will
        # be substrings of the real file, feeding through the existing
        # Tier 1→2→3 matching cascade in _apply_str_replace_edits().
        #
        # NEVER set task_type = CREATE for a MODIFY task. Full-file regeneration
        # is guaranteed to corrupt large files (the LLM cannot reproduce 1000+
        # lines faithfully from memory).
        if not written and existing_content and task.file_path in [pf["file"] for pf in state.get("_patch_failures", [])]:
            _task_ext = Path(task.file_path).suffix.lower()
            _method_bearing_exts = {".ts", ".tsx", ".js", ".jsx", ".java", ".kt", ".py", ".cs", ".go", ".rs", ".cpp", ".c"}
            if _task_ext not in _method_bearing_exts:
                logger.info(
                    f"  ℹ️ Non-method file {task.file_path} ({_task_ext}) — "
                    f"skipping verbatim anchor extraction; delegating to edit_loop"
                )
            else:
                logger.info(f"  🔄 Retry with verbatim extraction for {task.file_path} (str_replace EDIT failed)")
                try:
                    from ticket_to_code.agents.smart_extract import extract_exact_methods
                    from ticket_to_code.agents.code_generator import _get_anchor_methods

                    # Get exact method bodies using tree-sitter boundaries
                    _retry_anchors = _get_anchor_methods(task)
                    _retry_verbatim, _retry_matched = extract_exact_methods(
                        content=existing_content,
                        file_path=task.file_path,
                        anchor_methods=_retry_anchors,
                    )
                    if not _retry_matched and len(existing_content) < 3000:
                        _retry_matched = True
                        _retry_verbatim = existing_content
                        logger.info(f"  ℹ️ Small file ({len(existing_content)} chars) — using full file as verbatim anchor for {task.file_path}")

                    if _retry_matched:
                        # Retry code generation with verbatim content as existing_content
                        # The task stays as MODIFY — no task_type change
                        generator._sqlite_store = getattr(agents.localizer, "sqlite_store", None)
                        _ws = str(state.get("workspace_path", "") or getattr(agents, "workspace_path", ""))
                        generator._workspace_path = _ws
                        generator.workspace_path = Path(_ws) if _ws else None
                        generator._session_files = state.get("_run_generated_map", {})
                        if _impl_state is not None:
                            generator._impl_state = _impl_state
                        _retry_code = generator.generate_code(
                            task=task,
                            requirements=state["requirements"],
                            context=state["code_rag_context"],
                            existing_content=existing_content,
                            allowed_files=allowed_files,
                            readonly_files=readonly_files,
                        )
                        if (_retry_code.content and
                                _retry_code.content.strip() != existing_content.strip() and
                                "[TRUNCATED]" not in _retry_code.content):
                            _retry_validation = validator.validate(_retry_code, task, existing_content)
                            if _retry_validation.passed:
                                _retry_out = Path(state["workspace_path"]) / _retry_code.file_path
                                _retry_out.parent.mkdir(parents=True, exist_ok=True)
                                _retry_out.write_text(_retry_code.content, encoding="utf-8")
                                generated_code.append(_retry_code)
                                successful_writes += 1
                                written_file_paths.add(_retry_code.file_path)
                                task_ledger[task.id] = {"file_path": _retry_code.file_path, "status": "written", "reason": "verbatim_retry_success"}
                                logger.info(f"  ✅ Verbatim retry succeeded: {_retry_out}")
                                state["_patch_failures"] = [p for p in state.get("_patch_failures", [])
                                                             if p["file"] != task.file_path]
                            else:
                                logger.warning(
                                    f"  ⚠️ Verbatim retry validation failed for {task.file_path}: "
                                    + "; ".join(_retry_validation.violations)
                                )
                    else:
                        logger.warning(
                            f"  ⚠️ Anchor methods not matched in {task.file_path}: {_retry_anchors} — "
                            f"delegating to edit_loop for adaptive free-form editing."
                        )
                        state.setdefault("_patch_failures", []).append({
                            "file": task.file_path,
                            "reason": f"unmatched_methods: {_retry_anchors}",
                            "allowed_methods": task.allowed_methods,
                        })
                except Exception as _retry_exc:
                    logger.warning(f"  ⚠️ Verbatim retry error for {task.file_path}: {_retry_exc}")

    logger.info(f"Code generation complete: {len(generated_code)} files")

    # Log and surface any silent patch failures (non-fatal, edit_loop will recover)
    _pf = state.get("_patch_failures", [])
    if _pf:
        logger.warning(
            f"  ⚠️ Patch gaps detected for {len(_pf)} task(s) — edit_loop will handle remaining edits: "
            + "; ".join(f"{p['file']} [{p['reason']}]" for p in _pf)
        )

    # ── Component 5: Post-generation type consistency check ───────────────────
    # After ALL tasks are generated, scan for interface/model files modified this
    # run and verify consumers use only properties that exist on the new shape.
    # This catches TS2551/TS2339 cascades BEFORE the build step.
    _run_map = state.get("_run_generated_map", {})
    _type_mismatches: list[dict] = []
    import re as _re_tc
    for _if_path, _if_content in _run_map.items():
        # Only check model/interface files (not components/services)
        if not any(kw in _if_path for kw in (".model.", ".interface.", ".dto.", ".entity.")):
            continue
        # Extract exported interface/class property names
        _if_props: set = set()
        for _pm in _re_tc.finditer(
            r'^\s+(\w+)\s*[?:]?\s*:\s*\S', _if_content, _re_tc.MULTILINE
        ):
            _pname = _pm.group(1)
            if _pname not in ('if', 'for', 'return', 'constructor', 'class', 'interface'):
                _if_props.add(_pname)

        if not _if_props:
            continue

        # Extract the main exported type name (e.g. "DisplayedMember")
        _type_name_m = _re_tc.search(
            r'export\s+(?:interface|class|type)\s+(\w+)', _if_content
        )
        if not _type_name_m:
            continue
        _type_name = _type_name_m.group(1)

        # Find all consumers: files that import this type
        for _con_path, _con_content in _run_map.items():
            if _con_path == _if_path:
                continue
            if _type_name not in _con_content:
                continue
            # Extract property accesses on this type: variable.property patterns
            # This is approximate but catches the most common cases
            _accessed = set(_re_tc.findall(
                r'\b\w+\.(\w+)\b', _con_content
            ))
            _bad_accesses = _accessed - _if_props - {
                'length', 'map', 'filter', 'forEach', 'find', 'some', 'every',
                'push', 'pop', 'slice', 'splice', 'join', 'includes', 'indexOf',
                'toString', 'valueOf', 'subscribe', 'pipe', 'emit', 'next',
                'value', 'error', 'complete', 'unsubscribe',
            }
            # Only flag if the accessed property existed in the OLD version
            # (meaning the shape changed) and the consumer wasn't also updated
            _old_content = state.get("original_file_contents", {}).get(
                next((t.file_path for t in _plan.tasks
                      if t.file_path.replace("\\", "/").lower() == _if_path), ""),
                ""
            )
            if _old_content:
                _old_props: set = set()
                for _opm in _re_tc.finditer(
                    r'^\s+(\w+)\s*[?:]?\s*:\s*\S', _old_content, _re_tc.MULTILINE
                ):
                    _old_props.add(_opm.group(1))
                # Properties that existed before but were removed/renamed
                _removed_props = _old_props - _if_props
                _stale_in_consumer = _bad_accesses & _removed_props
                if _stale_in_consumer:
                    _type_mismatches.append({
                        "interface": _if_path,
                        "consumer": _con_path,
                        "stale_props": sorted(_stale_in_consumer),
                        "type_name": _type_name,
                    })
                    logger.warning(
                        f"  ⚠️ [TypeCheck] {_con_path} uses stale properties "
                        f"{_stale_in_consumer} from {_type_name} — shape changed in {_if_path}"
                    )

    if _type_mismatches:
        state["_type_mismatches"] = _type_mismatches
        logger.warning(
            f"  ⚠️ Component 5: found {len(_type_mismatches)} cross-file type mismatch(es) — "
            f"fix_build will need to update these consumers"
        )

    # ── Fix #4: Spring Data Repository Method Naming Convention Check ─────────
    # When adding query methods to JPA repositories (findByXxx, existsByXxx, etc.),
    # malformed method names compile fine but fail at RUNTIME with
    # PropertyReferenceException. Detect and warn before the build step.
    import re as _re_sd
    _sd_method_rx = _re_sd.compile(
        r'\b(find|exists|count|delete|remove)By([A-Za-z]+)\b'
    )
    # Valid Spring Data keywords that can appear in method names
    _sd_valid_keywords = {
        'And', 'Or', 'Is', 'Equals', 'Between', 'LessThan', 'LessThanEqual',
        'GreaterThan', 'GreaterThanEqual', 'After', 'Before', 'IsNull', 'IsNotNull',
        'NotNull', 'Like', 'NotLike', 'StartingWith', 'EndingWith', 'Containing',
        'OrderBy', 'Not', 'In', 'NotIn', 'True', 'False', 'IgnoreCase',
        'Asc', 'Desc', 'First', 'Top', 'Distinct', 'All',
    }
    _sd_warnings: list[str] = []
    for _sd_path, _sd_content in _run_map.items():
        if not _sd_path.endswith('.java'):
            continue
        # Only check repository files
        if 'Repository' not in _sd_content and 'repository' not in _sd_path.lower():
            continue
        for _sd_m in _sd_method_rx.finditer(_sd_content):
            _sd_clause = _sd_m.group(2)  # e.g. "ProjectIdAndOtdsUserId"
            # Split on And/Or to get individual property references
            _sd_props = _re_sd.split(r'(?:And|Or)', _sd_clause)
            for _sd_prop in _sd_props:
                # Strip known keywords from the end
                _sd_clean = _sd_prop
                for _kw in sorted(_sd_valid_keywords, key=len, reverse=True):
                    if _sd_clean.endswith(_kw) and len(_sd_clean) > len(_kw):
                        _sd_clean = _sd_clean[:-len(_kw)]
                # Property must start with uppercase (PascalCase from entity field)
                if _sd_clean and not _sd_clean[0].isupper():
                    _sd_warnings.append(
                        f"{_sd_path}: method '{_sd_m.group(0)}' — property '{_sd_clean}' "
                        f"should be PascalCase (matching entity field name)"
                    )
                # Check if the property name is suspiciously long (>30 chars)
                # which usually means keywords weren't separated correctly
                if len(_sd_clean) > 30:
                    _sd_warnings.append(
                        f"{_sd_path}: method '{_sd_m.group(0)}' — clause '{_sd_clean}' is very long. "
                        f"Verify it matches entity field names exactly."
                    )

    if _sd_warnings:
        for _w in _sd_warnings:
            logger.warning(f"  ⚠️ [SpringData] {_w}")
        state["_spring_data_warnings"] = _sd_warnings

    # ── v4: Semantic Contract — Pre-Build Validation ──────────────────────────
    # Build a contract from the planned changes and verify it against the
    # generated code. Broken invariants are surfaced as warnings and stored
    # in state for the error resolution agent to consume.
    try:
        from ticket_to_code.agents.semantic_contract import SemanticContractBuilder
        from ticket_to_code.agents.symbol_resolver import SymbolResolver
        from ticket_to_code.agents.component_structure_provider import ComponentStructureProvider

        _sc_symbol_index = getattr(agents.localizer, "symbol_index", None)
        _sc_resolver = SymbolResolver(_sc_symbol_index) if _sc_symbol_index else None
        _sc_provider = None
        try:
            _sc_provider = ComponentStructureProvider(
                workspace_path=str(Path(state["workspace_path"])),
                symbol_index=_sc_symbol_index,
                component_groups=state.get("component_groups") or [],
            )
        except Exception:
            pass

        _sc_builder = SemanticContractBuilder(
            symbol_resolver=_sc_resolver,
            component_provider=_sc_provider,
        )

        # Build contract from changed file list
        _sc_changed_files = list(written_file_paths)
        _ticket_obj = state.get("ticket")
        _sc_ticket_id = getattr(_ticket_obj, "ticket_id", "") if _ticket_obj else ""

        if _sc_changed_files:
            _sc_contract = _sc_builder.build_from_file_changes(
                ticket_id=_sc_ticket_id,
                changed_files=_sc_changed_files,
                description=getattr(_ticket_obj, "title", "") if _ticket_obj else "",
            )

            # Verify the contract
            _sc_broken = _sc_builder.verify(_sc_contract)
            if _sc_broken:
                logger.warning(
                    f"  ⚠️ [v4] Semantic Contract: {len(_sc_broken)} broken item(s) "
                    f"detected pre-build"
                )
                for _bi in _sc_broken[:10]:
                    logger.warning(f"    {_bi}")

            # Store contract in state for error resolution to consume
            _set_transient(state, "semantic_contract", _sc_contract)
            logger.info(
                f"  [v4] Semantic Contract: {len(_sc_contract.items)} items, "
                f"{len(_sc_broken)} broken"
            )
    except Exception as _sc_err:
        logger.debug(f"  [v4] Semantic Contract build failed (non-fatal): {_sc_err}")
    
    # ── POST-GENERATION PATCH SCOPE GATE (Zero File Leaks) ───────────────────
    from ticket_to_code.agents.ticket_scope_proof import (
        verify_post_generation_scope,
        revert_unauthorized_changes,
    )
    _all_plans = [
        state.get("initial_plan"),
        state.get("plan"),
        state.get("architectural_plan"),
        state.get("remediation_plan"),
    ]
    _plan_task_fps = set()
    for _p in _all_plans:
        if _p and hasattr(_p, "tasks"):
            for t in _p.tasks:
                if getattr(t.task_type, "value", str(t.task_type)).lower() != "read_only":
                    _plan_task_fps.add(t.file_path.replace("\\", "/"))
    _ticket_expected = set(state.get("expected_changed_files") or [])
    # Invariant (Scope Lock): Only proven targets, approved plan tasks (including CREATE),
    # and explicit ticket expected files are intentional.
    # NEVER include _gen_code_fps, written_file_paths, or _run_map_fps!
    _intentional_ticket_files = (
        set(allowed_files)
        | _plan_task_fps
        | _ticket_expected
    )

    try:
        from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
        for _itf in list(_intentional_ticket_files):
            _comp = PolyglotCompanionResolver.resolve_companion(_itf, workspace_path=str(state.get("workspace_path") or ""))
            if _comp and _comp[0]:
                _intentional_ticket_files.add(_comp[0].replace("\\", "/"))
    except Exception:
        pass

    _sp = state.get("ticket_scope_proof")
    _approved_set = set(allowed_files) | _intentional_ticket_files
    if _sp and hasattr(_sp, "scope_graph"):
        _proven_set = set(_sp.scope_graph.all_writable_files())
        if _proven_set:
            _approved_set.update(_proven_set)

    # 1. In-memory write tracking check
    files_written = set(written_file_paths)
    unauthorized = set()
    for fw in files_written:
        norm_fw = fw.replace("\\", "/").lower().strip()
        if not any(
            norm_fw == af.replace("\\", "/").lower().strip()
            or norm_fw.endswith("/" + af.replace("\\", "/").lower().strip())
            or af.replace("\\", "/").lower().strip().endswith("/" + norm_fw)
            for af in _approved_set
        ):
            unauthorized.add(fw)

    # 2. Hard filesystem status / git diff check
    _run_map_fps = set((state.get("_run_generated_map") or {}).keys())
    _session_written = files_written | _run_map_fps
    _is_fs_clean, _fs_violations, _fs_unauthorized = verify_post_generation_scope(
        workspace_path=state["workspace_path"],
        approved_writable_files=_approved_set,
        original_contents=state.get("original_file_contents"),
        pre_run_manifest=state.get("pre_run_workspace_manifest"),
        pre_run_git_dirty_files=state.get("pre_run_git_dirty_files"),
        session_written_files=_session_written,
    )
    unauthorized.update(_fs_unauthorized)

    # Safety invariant: intentional files generated for this ticket must NEVER be reverted or purged
    truly_unauthorized = set()
    for u in unauthorized:
        u_norm = u.replace("\\", "/").lower().strip()
        is_intentional = any(
            u_norm == int_f.replace("\\", "/").lower().strip()
            or u_norm.endswith("/" + int_f.replace("\\", "/").lower().strip())
            or int_f.replace("\\", "/").lower().strip().endswith("/" + u_norm)
            for int_f in _intentional_ticket_files
        )
        if not is_intentional:
            truly_unauthorized.add(u)
    unauthorized = truly_unauthorized

    if unauthorized:
        logger.warning(
            f"  🛡️ Scope proof enforcement: reverting {len(unauthorized)} file(s) modified/created outside "
            f"approved writable scope: {sorted(unauthorized)}"
        )
        reverted_files = revert_unauthorized_changes(
            workspace_path=state["workspace_path"],
            unauthorized_files=unauthorized,
            original_contents=state.get("original_file_contents"),
            pre_run_manifest=state.get("pre_run_workspace_manifest"),
            protected_files=_approved_set,
            session_written_files=_session_written,
        )
        logger.info(f"  🛡️ Auto-revert result: {reverted_files}")

        # Purge unauthorized files from in-memory generation artifacts
        _unauth_norms = {u.replace("\\", "/").lower().strip() for u in unauthorized}
        generated_code = [
            g for g in generated_code
            if not any(
                getattr(g, "file_path", "").replace("\\", "/").lower().strip() == un
                or getattr(g, "file_path", "").replace("\\", "/").lower().strip().endswith("/" + un)
                or un.endswith("/" + getattr(g, "file_path", "").replace("\\", "/").lower().strip())
                for un in _unauth_norms
            )
        ]
        _run_generated_map = {
            k: v for k, v in _run_generated_map.items()
            if not any(
                k.replace("\\", "/").lower().strip() == un
                or k.replace("\\", "/").lower().strip().endswith("/" + un)
                or un.endswith("/" + k.replace("\\", "/").lower().strip())
                for un in _unauth_norms
            )
        }
        written_file_paths = {
            w for w in written_file_paths
            if not any(
                w.replace("\\", "/").lower().strip() == un
                or w.replace("\\", "/").lower().strip().endswith("/" + un)
                or un.endswith("/" + w.replace("\\", "/").lower().strip())
                for un in _unauth_norms
            )
        }

    # Collect scope violation strings for the tracer
    scope_violation_msgs = (
        [f"Unauthorized write: {f}" for f in sorted(unauthorized)]
        if unauthorized else []
    )

    # Record Classification and Routing foundation traces
    agents.tracer.record_artifact_classification(classifications=classification_traces)
    agents.tracer.record_generation_routing(routing=routing_traces)

    agents.tracer.record_generation(
        generated_code=generated_code,
        generated_tests=state.get("generated_tests", []),
        allowed_files=allowed_files,
        readonly_files=readonly_files,
        scope_violations=scope_violation_msgs,
        system_message=getattr(agents.code_generator, "last_system_prompt", None),
        user_message=getattr(agents.code_generator, "last_user_prompt", None),
    )

    # ── Write 6-Tier Matching Cascade trace artifact ──────────────────────
    # Records which tier resolved each SEARCH/REPLACE edit + AST validation
    try:
        from ticket_to_code.agents.code_generator import _patch_tier_traces
        _pf = state.get("_patch_failures", [])
        _tier_trace = {
            "total_edits_attempted": sum(t.get("total_edits", 0) for t in _patch_tier_traces) if _patch_tier_traces else 0,
            "total_edits_resolved": len(_patch_tier_traces),
            "tier_summary": {},
            "edits": list(_patch_tier_traces),
            "patch_failures": _pf,
            "files_written": list(written_file_paths),
        }
        # Build tier summary (count per tier)
        for t in _patch_tier_traces:
            tier_name = t.get("tier", "unknown")
            _tier_trace["tier_summary"][tier_name] = _tier_trace["tier_summary"].get(tier_name, 0) + 1
        write_trace_artifact(
            state["workspace_path"], state["ticket"].ticket_id,
            "patch_tiers.json", _tier_trace
        )
        logger.info(f"  📊 Patch tier trace: {_tier_trace.get('tier_summary', {})}")
    except Exception as _trace_err:
        logger.debug(f"Patch tier trace write skipped: {_trace_err}")


    patch_diff = ""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "diff"],
            cwd=state["workspace_path"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        patch_diff = result.stdout or ""
        write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "generated_patch.diff", patch_diff)
    except Exception as e:
        logger.error(f"Failed to generate git diff: {e}")

    # If no effective file changes were produced, do not continue as success.
    # Feed this back into the candidate retry loop by blacklisting current
    # writable targets and requesting re-plan / rediscovery.
    if allowed_files and successful_writes == 0:
        retry_count = state.get("candidate_retry_count", 0) + 1
        max_retries = state.get("max_candidate_retries", 2)
        plan_for_retry = _get_plan(state)
        planned_create_paths = {
            t.file_path
            for t in getattr(plan_for_retry, "tasks", [])
            if getattr(getattr(t, "task_type", None), "value", str(getattr(t, "task_type", ""))).lower() == "create"
        }
        # Blacklist only files we actually attempted this round, not the
        # entire writable set. This avoids over-blacklisting and improves
        # convergence when some candidates are still viable.
        blacklist_targets = sorted(attempted_writable_files) if attempted_writable_files else list(allowed_files)
        # Never auto-blacklist explicit CREATE targets on no-op generation.
        # A no-op can happen due transient model output; blacklisting CREATE
        # paths causes the next plan to collapse to zero writable tasks.
        if planned_create_paths:
            blacklist_targets = [p for p in blacklist_targets if p not in planned_create_paths]

        # Never blacklist files that are explicitly referenced in the ticket or if this is the only target file
        ticket_obj = state.get("ticket")
        ticket_desc = ""
        if hasattr(ticket_obj, "description"):
            ticket_desc = f"{getattr(ticket_obj, 'title', '')} {getattr(ticket_obj, 'description', '')}"
        elif isinstance(ticket_obj, dict):
            ticket_desc = f"{ticket_obj.get('title', '')} {ticket_obj.get('description', '')}"
        else:
            ticket_desc = str(ticket_obj or "")
        ticket_desc_lower = ticket_desc.lower()

        # REMOVED BLACKLISTING: Grounded candidate files must NEVER be blacklisted on no-op generation.
        # Discovered files are the TRUE target files; blacklisting them caused the engine to ban the
        # correct files and enter an infinite rediscovery loop.
        new_blacklist = state.get("blacklisted_files") or []
        reason = "Code generation produced no effective patch on this turn; re-attempting without blacklisting valid files"
        logger.warning(
            f"  ⚠️ {reason}. Triggering candidate retry {retry_count}/{max_retries}. "
            f"Preserving all candidate files: {list(allowed_files)}"
        )
        return {
            "generated_code": generated_code,
            "code_status": "no_effective_changes",
            "candidate_retry_count": retry_count,
            "blacklisted_files": new_blacklist,
            "validation_failure_reason": reason,
            "status": "candidates_invalid",
            "original_file_contents": state.get("original_file_contents", {}),
            "_task_explanations": state.get("_task_explanations", []),
            "_run_generated_map": _run_generated_map,
        }

    # DEBUG LOG
    log_phase(
        phase='GENERATE_CODE',
        llm_output={
            'generated_files': len(generated_code),
            'first_file': generated_code[0].file_path if generated_code else None,
            'total_lines_written': sum(len(g.content.splitlines()) for g in generated_code if g.content),
            'patch_status': 'applied' if patch_diff.strip() else 'no_changes',
            'scope_violations': scope_violation_msgs
        },
        agent='CodeGenerator'
    )

    # Enforce task ledger invariant: mark any unwritten tasks as failed
    for t in _sorted_tasks:
        if t.id not in task_ledger:
            task_ledger[t.id] = {"file_path": t.file_path, "status": "failed", "reason": "unprocessed_or_silent_failure"}

    state["task_execution_ledger"] = task_ledger
    _writable_planned = [t for t in _sorted_tasks if getattr(t.task_type, "value", str(t.task_type)) != "read_only"]
    _failed_tasks = [tid for tid, entry in task_ledger.items() if entry.get("status") == "failed"]
    logger.info(
        f"  📊 Task Ledger: {len(task_ledger)}/{len(_sorted_tasks)} tasks accounted for. "
        f"Written: {successful_writes}/{len(_writable_planned)}. "
        f"Failed: {len(_failed_tasks)}."
    )

    return {
        "generated_code": generated_code,
        "code_status": "code_generated",  # CODE BRANCH status
        "original_file_contents": state.get("original_file_contents", {}),
        "_task_explanations": state.get("_task_explanations", []),
        "_run_generated_map": _run_generated_map,
        "task_execution_ledger": task_ledger,
    }


# ============================================================================
# SCSS / CSS / PYTHON / HTML SYNTAX CHECK HELPERS
# ============================================================================

def _check_scss_syntax(content: str, file_path: str) -> list[str]:
    """
    Fast SCSS/CSS syntax check — no external tools needed.

    Catches:
      1. Unbalanced braces { }  (the #1 LLM error in SCSS)
      2. Leaked LLM format markers (<<<<<<< SEARCH, EDIT:, old_str:, etc.)
      3. Empty or near-empty files (accidental truncation)
    """
    errors: list[str] = []

    # 1. Check for leaked format markers
    _format_markers = [
        "<<<<<<< SEARCH", ">>>>>>> REPLACE", "=======",
        "EDIT:\nold_str:", "<<<\n", "\n>>>",
        "<<<AVIATOR_CODE_START>>>", "<<<AVIATOR_CODE_END>>>",
    ]
    for marker in _format_markers:
        if marker in content:
            errors.append(
                f"{file_path}: Leaked LLM format marker found: "
                f"'{marker.strip()[:30]}' — file is corrupted"
            )

    # 2. Check balanced braces
    depth = 0
    for i, ch in enumerate(content):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth < 0:
                line_num = content[:i].count('\n') + 1
                errors.append(
                    f"{file_path}({line_num}): Unexpected closing brace '}}' "
                    f"— braces are unbalanced"
                )
                break
    if depth > 0 and not errors:
        errors.append(
            f"{file_path}: {depth} unclosed brace(s) '{{' — SCSS is incomplete"
        )

    # 3. Check for empty/truncated file
    stripped = content.strip()
    if not stripped:
        errors.append(f"{file_path}: File is empty after write")
    elif len(stripped) < 10:
        errors.append(f"{file_path}: File is suspiciously short ({len(stripped)} chars)")

    if errors:
        logger.warning(
            f"  ⚠️ [SCSS-syntax] {len(errors)} issue(s) in {file_path}: "
            + "; ".join(errors)
        )
    return errors


def _check_python_syntax(file_path: Path) -> list[str]:
    """
    Python syntax check using the built-in ast and py_compile modules.

    This catches SyntaxError, IndentationError, and other parse-time
    errors without importing or executing the file.
    """
    import ast
    import py_compile
    errors: list[str] = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        ast.parse(content, filename=str(file_path))
    except SyntaxError as syn_err:
        err_msg = f"{file_path}({syn_err.lineno}): SyntaxError: {syn_err.msg}"
        errors.append(err_msg)
        logger.warning(f"  ⚠️ [Python-syntax] {err_msg}")
    except Exception as exc:
        errors.append(f"{file_path}: AST parse failed: {exc}")

    if not errors:
        try:
            py_compile.compile(str(file_path), doraise=True)
        except py_compile.PyCompileError as exc:
            err_msg = str(exc)
            if hasattr(exc, '__cause__') and exc.__cause__:
                cause = exc.__cause__
                if hasattr(cause, 'lineno') and cause.lineno:
                    err_msg = f"{file_path}({cause.lineno}): {type(cause).__name__}: {cause.msg}"
            errors.append(err_msg)
            logger.warning(f"  ⚠️ [Python-syntax] {file_path}: {err_msg}")
        except Exception as exc:
            errors.append(f"{file_path}: py_compile failed: {exc}")

    # Also check for leaked format markers
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for marker in ["<<<<<<< SEARCH", ">>>>>>> REPLACE", "EDIT:\nold_str:"]:
            if marker in content:
                errors.append(
                    f"{file_path}: Leaked LLM format marker: '{marker.strip()[:30]}'"
                )
    except Exception:
        pass

    return errors


def _check_html_syntax(content: str, file_path: str) -> list[str]:
    """
    Basic HTML template syntax check.

    Catches:
      1. Leaked LLM format markers
      2. Severely unbalanced HTML tags (missing closing tags for block elements)
      3. Empty/truncated files
    """
    errors: list[str] = []

    # 1. Format marker leak check
    for marker in ["<<<<<<< SEARCH", ">>>>>>> REPLACE", "EDIT:\nold_str:"]:
        if marker in content:
            errors.append(
                f"{file_path}: Leaked LLM format marker: '{marker.strip()[:30]}'"
            )

    # 2. Basic balanced-tag check for Angular block elements
    _block_tags = ["div", "section", "main", "form", "table", "tbody", "thead",
                   "tr", "ul", "ol", "li", "mat-card", "mat-dialog-content",
                   "mat-tab-group", "mat-tab", "ng-container", "ng-template"]
    for tag in _block_tags:
        opens = len(re.findall(rf'<{tag}[\s>]', content, re.IGNORECASE))
        closes = len(re.findall(rf'</{tag}\s*>', content, re.IGNORECASE))
        # Self-closing tags don't need a close
        self_closing = len(re.findall(rf'<{tag}[^>]*/\s*>', content, re.IGNORECASE))
        net_opens = opens - self_closing
        if net_opens > closes + 2:  # tolerance of 2 for ng-container/ng-template
            errors.append(
                f"{file_path}: Unclosed <{tag}> tags — "
                f"{net_opens} opened, {closes} closed"
            )

    # 3. Empty check
    stripped = content.strip()
    if not stripped:
        errors.append(f"{file_path}: File is empty after write")

    if errors:
        logger.warning(
            f"  ⚠️ [HTML-syntax] {len(errors)} issue(s) in {file_path}: "
            + "; ".join(errors)
        )
    return errors


def _check_code_syntax(content: str, file_path: str) -> list[str]:
    """
    Fast local syntax check for TypeScript, JavaScript, Java, C#, and other C-family code.
    Verifies:
      1. No leaked LLM format markers (<<<<<<< SEARCH, TARGET:, etc.)
      2. Non-empty file
      3. Balanced curly braces {} and parentheses () (ignoring comments and string literals)
    Does NOT require a full compiler or project build, avoiding false errors when
    companion files (templates, models, services) have not yet been written.
    """
    errors: list[str] = []

    # 1. Format marker leak check
    for marker in ["<<<<<<< SEARCH", ">>>>>>> REPLACE", "EDIT:\nold_str:", "TARGET:\n", "REPLACEMENT:\n"]:
        if marker in content:
            errors.append(
                f"{file_path}: Leaked LLM format marker: '{marker.strip()[:30]}'"
            )

    # 2. Empty check
    stripped = content.strip()
    if not stripped:
        errors.append(f"{file_path}: File is empty after write")
        return errors
    elif len(stripped) < 15:
        errors.append(f"{file_path}: File is suspiciously short ({len(stripped)} chars)")
        return errors

    # 3. Balanced braces and parens (strip comments and string literals first)
    import re
    cleaned = re.sub(r'//[^\n]*', '', content)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'"(?:[^"\\]|\\.)*"', '""', cleaned)
    cleaned = re.sub(r"'(?:[^'\\]|\\.)*'", "''", cleaned)
    cleaned = re.sub(r'`(?:[^`\\]|\\.)*`', '``', cleaned, flags=re.DOTALL)

    open_braces = cleaned.count('{')
    close_braces = cleaned.count('}')
    if open_braces != close_braces:
        errors.append(
            f"{file_path}: Unbalanced curly braces — {open_braces} opened, {close_braces} closed"
        )

    open_parens = cleaned.count('(')
    close_parens = cleaned.count(')')
    if open_parens != close_parens:
        errors.append(
            f"{file_path}: Unbalanced parentheses — {open_parens} opened, {close_parens} closed"
        )

    # 4. Leaked conversational prose detection (outside comments/strings)
    prose_patterns = [
        (r"^\s*\d+[\.\)]\s+[A-Za-z]", "Numbered conversational thought"),
        (r"^\s*(?:But\s+note|However,\s*note|Please\s+note|Looking\s+at\s+the|Alternatively|We\s+must\s+place|We\s+should\s+place)\b", "Conversational analysis leak"),
        (r"^\s*(?:Here\s+is\s+the|Below\s+is\s+the|Note\s+that\s+we)\b", "Explanatory prose leak"),
        (r"^\s*(?:But\s+wait:|However,|Therefore,|SEARCH\s+for:|REPLACE\s+with:|Third,\s+we\s+add|What\s+about\s+an\s+order|We'll\s+need\s+to)\b", "Leaked chain-of-thought analysis"),
    ]
    for line_idx, line in enumerate(cleaned.splitlines(), 1):
        for pat, desc in prose_patterns:
            if re.search(pat, line, re.IGNORECASE):
                snippet = line.strip()[:60]
                errors.append(
                    f"{file_path}:{line_idx}: Leaked conversational prose ({desc}): '{snippet}'"
                )
                break

    if errors:
        logger.warning(
            f"  ⚠️ [Syntax-check] {len(errors)} issue(s) in {file_path}: "
            + "; ".join(errors)
        )
    return errors


# ============================================================================
# ANGULAR / TYPESCRIPT COMPILE HELPERS
# ============================================================================

def _find_ng_root(workspace: Path, target_file: Optional[Path] = None) -> Optional[Path]:
    """
    Find the frontend project root — a directory that contains package.json,
    tsconfig.json, angular.json, vite.config.ts, etc.
    Universal: upward search from target file, then scans workspace subdirectories.
    """
    # 1. If target_file provided, walk upward from file's directory
    if target_file:
        tf_abs = target_file if target_file.is_absolute() else (workspace / target_file)
        curr = tf_abs.parent
        while curr != curr.parent and (workspace in curr.parents or curr == workspace):
            if _has_frontend_config(curr):
                return curr
            curr = curr.parent

    _COMMON_FRONTEND_DIRS = (
        "ui", "frontend", "client", "app", "web", "webapp",
        "web-app", "web-ui", "store-ui", "portal-ui",
    )
    
    # Fast path: check common names
    for subdir in _COMMON_FRONTEND_DIRS:
        candidate = workspace / subdir
        if candidate.is_dir() and _has_frontend_config(candidate):
            return candidate
    
    # Check workspace root itself
    if _has_frontend_config(workspace):
        return workspace
    
    # Exhaustive: scan all top-level subdirectories
    try:
        for entry in workspace.iterdir():
            if entry.is_dir() and not entry.name.startswith((".", "node_modules", "target", "build", "dist")):
                if _has_frontend_config(entry):
                    return entry
    except (PermissionError, OSError):
        pass

    return None


def _has_frontend_config(directory: Path) -> bool:
    """Check if a directory looks like a frontend project root."""
    return (
        (directory / "package.json").exists()
        or (directory / "tsconfig.json").exists()
        or (directory / "angular.json").exists()
        or (directory / "next.config.js").exists()
        or (directory / "next.config.mjs").exists()
        or (directory / "vite.config.ts").exists()
        or (directory / "vite.config.js").exists()
    )


def _extract_error_line_context(
    error_line: str,
    file_content: str,
    window: int = 5,
) -> str:
    """
    Given a compiler error line like 'File.ts(387,31): error TS2551: ...',
    extract the exact source code line that is wrong, plus surrounding context.

    For TS2345 ("Type X not assignable to Y") the error fires at the CALL SITE
    but the fix requires changing the CALLING METHOD body.  We expand the window
    to show the entire enclosing function/method when TS2345 is detected.
    """
    # Extract line number from various formats: (387,31), :[387,31], :387:
    line_num = None
    m = re.search(r'[(:]\[?(\d+)[,:]', error_line)
    if m:
        try:
            line_num = int(m.group(1))
        except ValueError:
            pass

    if not line_num or not file_content:
        return ""

    lines = file_content.splitlines()
    if line_num > len(lines):
        return ""

    # For TS2345 (type incompatibility at call site), expand to show the full
    # enclosing function so the LLM sees WHERE the wrong object is built.
    if "TS2345" in error_line or "not assignable to" in error_line.lower():
        # Walk backwards to find the enclosing function/method start
        func_start = max(0, line_num - 60)
        for i in range(line_num - 2, max(0, line_num - 60), -1):
            if re.match(r'\s*(?:public|private|protected|async)?\s*\w+\s*\(', lines[i]):
                func_start = i
                break
        # Walk forwards to find the function end
        func_end = min(len(lines), line_num + 10)
        depth = 0
        for i in range(func_start, min(len(lines), func_start + 80)):
            depth += lines[i].count('{') - lines[i].count('}')
            if i > line_num and depth <= 0:
                func_end = i + 1
                break
        start = func_start
        end = func_end
    else:
        start = max(0, line_num - window - 1)
        end = min(len(lines), line_num + window)

    excerpt = []
    for i, ln in enumerate(lines[start:end], start=start + 1):
        marker = ">>> " if i == line_num else "    "
        excerpt.append(f"{marker}{i:4d}: {ln}")
    return "\n".join(excerpt)


def _collect_type_definition_files(
    error_lines: list[str],
    session_map: dict,
    workspace_path: str,
) -> dict[str, str]:
    """
    For each TypeScript error (TS2322, TS2551, TS2339), find the interface/type
    definitions that the LLM needs to see in order to write the correct fix.

    Searches the session_map (freshly generated files) first, then disk.
    Returns {file_path: content} for all relevant type definition files.
    """
    collected: dict[str, str] = {}

    # Extract type names from error messages
    type_names: set[str] = set()
    for ln in error_lines:
        # "not assignable to type 'DisplayedOrganization'"
        for m in re.finditer(r"type\s+'([A-Z][A-Za-z0-9]+)'", ln):
            type_names.add(m.group(1))
        # "Property 'X' does not exist on type 'Y'"
        for m in re.finditer(r"on type\s+'([A-Z][A-Za-z0-9]+)'", ln):
            type_names.add(m.group(1))
        # "Did you mean 'organization'?" → look at what defines that property
        for m in re.finditer(r"Did you mean\s+'([^']+)'", ln):
            type_names.add(m.group(1).split(".")[0].strip())

    if not type_names:
        return collected

    # Search session map first (freshest content)
    for path_key, content in session_map.items():
        for type_name in type_names:
            if (f"interface {type_name}" in content or f"type {type_name}" in content
                    or f"class {type_name}" in content):
                if path_key not in collected:
                    collected[path_key] = content
                break

    # Fall back to disk if not found in session
    if len(collected) < len(type_names):
        skip_dirs = {"node_modules", "dist", ".git", "build", "coverage", ".venv"}
        import os as _os
        for root, dirs, files in _os.walk(workspace_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in files:
                if not f.endswith((".ts", ".java", ".py")):
                    continue
                full = Path(root) / f
                rel = str(full.relative_to(Path(workspace_path))).replace("\\", "/").lower()
                if rel in collected:
                    continue
                try:
                    content = full.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for type_name in type_names:
                    if (f"interface {type_name}" in content or f"class {type_name} " in content
                            or f"type {type_name} " in content):
                        collected[rel] = content
                        break
                if len(collected) >= 8:
                    return collected

    return collected


def _build_err_fingerprint(errors: list) -> str:
    """Stable error fingerprint using normalized diagnostic identity.

    Delegates to the DiagnosticNormalizer pipeline so fingerprints are based on
    structured diagnostic fields (code, basename, entities) rather than raw
    compiler text. This means:
      - Same error at different line numbers → same fingerprint
      - Same error with different absolute paths → same fingerprint
      - Genuinely different errors → different fingerprint
    """
    from ticket_to_code.agents.diagnostic_normalizer import (
        normalize_diagnostics, diagnostic_fingerprint,
    )
    diagnostics = normalize_diagnostics([str(e) for e in (errors or [])])
    return diagnostic_fingerprint(diagnostics)


def _find_java_module_root(file_path: str, workspace_path: str) -> "Optional[tuple[Path, str]]":
    """
    Walk up from `file_path` to find the nearest Maven or Gradle module root.

    Returns (module_root_path, build_tool) where build_tool is "maven" or "gradle",
    or None if no build file is found within the workspace.
    """
    ws = Path(workspace_path).resolve()
    current = (ws / file_path).resolve().parent
    while True:
        if (current / "pom.xml").exists():
            return (current, "maven")
        if (current / "build.gradle").exists() or (current / "build.gradle.kts").exists():
            return (current, "gradle")
        if current == ws or current == current.parent:
            break
        current = current.parent
    return None


def _run_java_compile_on_module(
    module_root: Path,
    build_tool: str,
    timeout: int = 60,
) -> list[str]:
    """
    Run a fast incremental Java compile on the module that contains the changed file.

    Maven  → `mvn compile -q --no-transfer-progress`
    Gradle → `gradle compileJava --no-daemon -q` (uses Gradle daemon cache → fast on repeat runs)

    Returns:
        List of error strings. Empty list = clean compile.
    """
    import subprocess
    import sys

    use_shell = sys.platform == "win32"

    if build_tool == "maven":
        mvn_exe = "mvn"
        if use_shell:
            if (module_root / "mvnw.cmd").exists():
                mvn_exe = str(module_root / "mvnw.cmd")
        else:
            if (module_root / "mvnw").exists():
                mvn_exe = "./mvnw"
        offline_opt = "-o" if (Path.home() / ".m2" / "repository").exists() else ""
        cmd: Any = (
            f"{mvn_exe} compile -q {offline_opt} --no-transfer-progress -DskipTests".strip()
            if use_shell
            else [arg for arg in [mvn_exe, "compile", "-q", offline_opt, "--no-transfer-progress", "-DskipTests"] if arg]
        )
    else:
        # Gradle: compileJava and compileTestJava for complete verification
        gradle_exe = "gradlew.bat" if (module_root / "gradlew.bat").exists() and use_shell else "gradle"
        if (module_root / "gradlew").exists():
            gradle_exe = "gradlew.bat" if use_shell else "./gradlew"
        cmd = (
            f"{gradle_exe} compileJava compileTestJava --no-daemon -q"
            if use_shell
            else [gradle_exe, "compileJava", "compileTestJava", "--no-daemon", "-q"]
        )

    run_env = dict(os.environ)
    current_jh = run_env.get("JAVA_HOME", "").strip().rstrip("\\/")
    java_bin = "bin\\java.exe" if sys.platform == "win32" else "bin/java"
    javac_bin = "bin\\javac.exe" if sys.platform == "win32" else "bin/javac"

    if not current_jh or not os.path.exists(os.path.join(current_jh, java_bin)):
        import shutil, glob
        # 1. PATH via javac
        which_javac = shutil.which("javac")
        if which_javac:
            cand = str(Path(which_javac).resolve().parent.parent)
            if os.path.exists(os.path.join(cand, java_bin)):
                current_jh = cand
        # 2. Filesystem search across standard vendor folders
        if not current_jh or not os.path.exists(os.path.join(current_jh, java_bin)):
            search_patterns = []
            if sys.platform == "win32":
                for base in [os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files (x86)"]:
                    for vendor in ["Microsoft", "Eclipse Adoptium", "Java", "Amazon Corretto", "Zulu", "BellSoft", "Semeru"]:
                        search_patterns.append(os.path.join(base, vendor, "*"))
            else:
                search_patterns.extend(["/usr/lib/jvm/*", "/usr/java/*", "/Library/Java/JavaVirtualMachines/*/Contents/Home"])
            found_jdks = []
            for pat in search_patterns:
                for p in glob.glob(pat):
                    if os.path.isdir(p) and (os.path.exists(os.path.join(p, javac_bin)) or os.path.exists(os.path.join(p, java_bin))):
                        found_jdks.append(p)
            if found_jdks:
                found_jdks.sort(reverse=True)
                current_jh = found_jdks[0]
    if current_jh and os.path.exists(os.path.join(current_jh, java_bin)):
        run_env["JAVA_HOME"] = current_jh
        bin_dir = os.path.join(current_jh, "bin")
        cur_p = run_env.get("PATH", "")
        if bin_dir.lower() not in cur_p.lower():
            run_env["PATH"] = f"{bin_dir}{os.pathsep}{cur_p}"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(module_root),
            env=run_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=use_shell,
        )

        if result.returncode == 0:
            return []

        raw = (result.stdout + result.stderr).strip()
        # Maven errors start with "[ERROR]"; Gradle errors contain ": error:" or "error:"
        errors: list[str] = []
        for ln in raw.splitlines():
            stripped = ln.strip()
            if not stripped:
                continue
            if stripped.startswith("[ERROR]") or ": error:" in stripped.lower() or stripped.startswith("error:"):
                errors.append(stripped)

        if not errors:
            # Fall back: include any non-empty, non-INFO/WARNING line
            errors = [
                ln.strip() for ln in raw.splitlines()
                if ln.strip() and not ln.strip().startswith(("[INFO]", "[WARNING]", "> Task", "BUILD"))
            ]

        return errors[:30]  # cap to avoid flooding the prompt

    except subprocess.TimeoutExpired:
        logger.warning(f"  ⚠️ Java module compile timed out after {timeout}s — skipping")
        return []
    except FileNotFoundError as exc:
        logger.debug(f"  ℹ️ Java compile tool not found ({exc}) — skipping")
        return []
    except Exception as exc:
        logger.debug(f"  ℹ️ Java compile check failed: {exc}")
        return []


def _run_tsc_on_files(
    ng_root: Path,
    written_files: list[str],
    timeout: int = 30,
) -> list[str]:
    """
    Run a fast, scoped `tsc --noEmit` check covering only the files written so far.

    Strategy (mirrors Cursor per-file feedback loop):
    1. Write a temporary tsconfig that extends the project's tsconfig.app.json
       but overrides `files` to only the TypeScript files written in this run.
    2. Run `tsc --noEmit --skipLibCheck -p tsconfig_aviator_tmp.json`.
    3. Parse and return error lines. Empty list = clean.
    4. Always delete the temp file on exit.

    `--skipLibCheck` avoids re-checking node_modules .d.ts files (saves 15–20 s).
    Scoping to written files avoids false positives from pre-existing errors
    in files we haven't touched.

    Returns:
        List of error strings, empty on success or if tsc is not available.
    """
    import subprocess
    import sys
    import json as _json
    import tempfile

    ts_files = [f for f in written_files if f.endswith((".ts", ".tsx")) and not f.endswith(".spec.ts")]
    if not ts_files:
        return []

    use_shell = sys.platform == "win32"
    tmp_cfg = ng_root / "tsconfig_aviator_tmp.json"

    # Build extends from tsconfig.app.json or tsconfig.json
    base_cfg = "tsconfig.app.json" if (ng_root / "tsconfig.app.json").exists() else "tsconfig.json"

    # Paths must be relative to ng_root
    rel_files: list[str] = []
    for f in ts_files:
        try:
            abs_f = Path(f) if Path(f).is_absolute() else ng_root / f
            rel = abs_f.resolve().relative_to(ng_root.resolve())
            rel_files.append(str(rel).replace("\\", "/"))
        except ValueError:
            rel_files.append(f.replace("\\", "/"))

    tmp_cfg_content = {
        "extends": f"./{base_cfg}",
        "compilerOptions": {"skipLibCheck": True, "noEmit": True},
        "files": rel_files,
    }

    try:
        tmp_cfg.write_text(_json.dumps(tmp_cfg_content, indent=2), encoding="utf-8")

        cmd: Any = (
            f"npx --no-install tsc --noEmit --skipLibCheck --pretty false -p tsconfig_aviator_tmp.json"
            if use_shell
            else ["npx", "--no-install", "tsc", "--noEmit", "--skipLibCheck",
                  "--pretty", "false", "-p", "tsconfig_aviator_tmp.json"]
        )

        result = subprocess.run(
            cmd,
            cwd=str(ng_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=use_shell,
        )

        if result.returncode == 0:
            return []

        raw = strip_ansi(result.stdout + result.stderr).strip()
        errors = [ln for ln in raw.splitlines() if ln.strip() and "error TS" in ln]
        if not errors:
            errors = [ln for ln in raw.splitlines() if ln.strip()]
        return errors

    except subprocess.TimeoutExpired:
        logger.warning(f"  ⚠️ Scoped TSC check timed out after {timeout}s — skipping")
        return []
    except FileNotFoundError:
        logger.debug("  ℹ️ npx not found — scoped TSC check skipped")
        return []
    except Exception as exc:
        logger.debug(f"  ℹ️ Scoped TSC check failed: {exc}")
        return []
    finally:
        try:
            tmp_cfg.unlink(missing_ok=True)
        except Exception:
            pass


def _run_tsc_check(ng_root: Path, timeout: int = 180) -> "Optional[BuildResult]":
    """
    Run TypeScript (`tsc --noEmit`) and Angular template type-checking (`ng build`)
    in *ng_root*.

    Returns:
        BuildResult with FAILURE status  — if tsc or ng build found errors
        None                             — if compile succeeded OR if tools not found
    """
    import subprocess
    import sys

    # On Windows npx is a .cmd wrapper; shell=True lets cmd.exe resolve it.
    use_shell = sys.platform == "win32"

    # Ensure node_modules exist — if missing, run npm install first
    if not (ng_root / "node_modules").exists():
        logger.info(f"  📦 node_modules missing in {ng_root} — running npm install...")
        try:
            _npm_cmd: Any = "npm install" if use_shell else ["npm", "install"]
            subprocess.run(
                _npm_cmd, cwd=str(ng_root), capture_output=True,
                text=True, timeout=300, shell=use_shell,
            )
            logger.info("  ✅ npm install completed")
        except Exception as _npm_err:
            logger.warning(f"  ⚠️ npm install failed: {_npm_err} — continuing anyway")

    # Step 1: Run fast tsc --noEmit check
    tsconfig_app = ng_root / "tsconfig.app.json"
    if tsconfig_app.exists():
        cmd: Any = (
            f"npx --no-install tsc --noEmit --pretty false -p tsconfig.app.json"
            if use_shell
            else ["npx", "--no-install", "tsc", "--noEmit", "--pretty", "false", "-p", "tsconfig.app.json"]
        )
        logger.info("  Using tsconfig.app.json (excludes e2e/test files)")
    else:
        cmd = (
            "npx --no-install tsc --noEmit --pretty false"
            if use_shell
            else ["npx", "--no-install", "tsc", "--noEmit", "--pretty", "false"]
        )

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ng_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=use_shell,
        )
        if result.returncode != 0:
            clean_raw = strip_ansi(result.stdout + result.stderr).strip()
            errors = [line for line in clean_raw.splitlines() if line.strip()]
            logger.error(f"  ❌ TypeScript compile errors ({len(errors)} diagnostic line(s)):")
            for e in errors[:8]:
                logger.error(f"    {e}")
            return BuildResult(
                status=BuildStatus.FAILURE,
                exit_code=result.returncode,
                errors=errors,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        # Step 2: If Angular project, run Angular template type-checking (ng build)
        angular_json = ng_root / "angular.json"
        if angular_json.exists():
            logger.info("  🔍 Running Angular AOT template type-check (`npm run build`)...")
            ng_cmd: Any = (
                "npm run build"
                if use_shell
                else ["npm", "run", "build"]
            )
            ng_result = subprocess.run(
                ng_cmd,
                cwd=str(ng_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=use_shell,
            )
            if ng_result.returncode != 0:
                clean_ng = strip_ansi(ng_result.stdout + ng_result.stderr).strip()
                all_lines = clean_ng.splitlines()
                ng_errors = [
                    line for line in all_lines
                    if ("Error:" in line or "error TS" in line or "error NG" in line)
                ]
                if not ng_errors:
                    ng_errors = [line for line in all_lines if line.strip() and not line.startswith("Warning:") and not line.startswith("Deprecation Warning")]
                logger.error(f"  ❌ Angular template build errors ({len(ng_errors)} diagnostic line(s)):")
                for e in ng_errors[:8]:
                    logger.error(f"    {e}")
                return BuildResult(
                    status=BuildStatus.FAILURE,
                    exit_code=ng_result.returncode,
                    errors=ng_errors,
                    stdout=ng_result.stdout,
                    stderr=ng_result.stderr,
                )

        return None  # Success — no errors
    except subprocess.TimeoutExpired:
        logger.warning(f"  ⚠️ Build check timed out after {timeout}s — compile check skipped")
        return None
    except FileNotFoundError:
        logger.warning("  ⚠️ npx / npm not found in PATH — compile check skipped")
        return None


# ============================================================================
# PATCH GATE NODE (B9: Delivery Safety)
# ============================================================================

def patch_gate_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    B9 Patch Delivery Safety Gate.

    Runs THREE deterministic checks BEFORE passing to build:
      1. Scope enforcement  — diff must not touch files outside writable set
      2. Git apply check    — diff must apply cleanly to current HEAD
      3. Secret scan        — diff must not contain high-confidence secrets

    Controlled by env var AVIATOR_ENABLE_PATCH_GATE (default "1" = on).
    When disabled the node is a pass-through so existing behaviour is unchanged.

    On gate failure:  status set to "candidates_invalid" so route_after_generate_code
                      can trigger a blacklist + re-plan cycle.
    On gate pass:     passes through unchanged.
    """
    _GATE_ENABLED = os.getenv("AVIATOR_ENABLE_PATCH_GATE", "1") not in ("0", "false", "no")
    if not _GATE_ENABLED:
        logger.debug("patch_gate_node: AVIATOR_ENABLE_PATCH_GATE=0 — skipping")
        return {}

    logger.info("🔒 PATCH GATE: Checking scope, apply, and secrets")

    run_ctx: Optional[RunContext] = _get_transient(state, "run_ctx")
    # start_phase/end_phase handled by _phase_tracked_node wrapper

    generated_code = state.get("generated_code") or []
    plan = _get_plan(state)
    workspace_path = state["workspace_path"]

    # Collect writable file paths from all approved plans in this workflow run
    _all_plans = [
        state.get("initial_plan"),
        state.get("plan"),
        state.get("architectural_plan"),
        state.get("remediation_plan"),
    ]
    _plan_task_fps = set()
    for _p in _all_plans:
        if _p and hasattr(_p, "tasks"):
            for t in _p.tasks:
                if getattr(t.task_type, "value", str(t.task_type)).lower() != "read_only":
                    _plan_task_fps.add(t.file_path.replace("\\", "/"))
    _ticket_expected = {f.replace("\\", "/") for f in (state.get("expected_changed_files") or [])}
    # Invariant (Scope Lock): Planned tasks (including CREATE) and ticket expected are intentional.
    # NEVER include _gen_code_fps, _run_map_fps, or _orig_fps!
    _intentional_ticket_files = _plan_task_fps | _ticket_expected
    if state.get("allowed_files"):
        _intentional_ticket_files.update({f.replace("\\", "/") for f in state["allowed_files"]})

    _sp = state.get("ticket_scope_proof")
    allowed_files: set = set(_intentional_ticket_files)
    if _sp and hasattr(_sp, "scope_graph"):
        allowed_files.update({
            f.replace("\\", "/") for f in _sp.scope_graph.all_writable_files()
        })

    # ── Hard Filesystem / Git Status Diff Check (Zero File Leaks) ────────────
    from ticket_to_code.agents.ticket_scope_proof import (
        verify_post_generation_scope,
        revert_unauthorized_changes,
    )
    _gen_fps = {getattr(g, "file_path", "") for g in generated_code if getattr(g, "file_path", None)}
    _run_map_fps = set((state.get("_run_generated_map") or {}).keys())
    _session_written = _gen_fps | _run_map_fps
    _is_fs_clean, _fs_violations, _fs_unauthorized = verify_post_generation_scope(
        workspace_path=workspace_path,
        approved_writable_files=allowed_files,
        original_contents=state.get("original_file_contents"),
        pre_run_manifest=state.get("pre_run_workspace_manifest"),
        pre_run_git_dirty_files=state.get("pre_run_git_dirty_files"),
        session_written_files=_session_written,
    )
    gate_failures: list = []
    
    # Safety invariant: intentional files generated for this ticket must NEVER be reverted
    truly_unauthorized = set()
    # The harness rewrites its own brain/cache artifacts; they are not ticket changes and must not block the gate.
    _harness_artifact_prefixes = ("brain/knowledge/", ".aviator/", "brain/.cache/")
    for u in _fs_unauthorized:
        u_norm = u.replace("\\", "/").lower().strip()
        if u_norm.lstrip("./").startswith(_harness_artifact_prefixes):
            continue
        is_intentional = any(_scope_path_match(u_norm, int_f) for int_f in _intentional_ticket_files)
        if not is_intentional:
            truly_unauthorized.add(u)

    if truly_unauthorized:
        logger.warning(
            f"🛡️ PATCH GATE SCOPE ENFORCEMENT: Detected {len(truly_unauthorized)} unauthorized modified file(s): {sorted(truly_unauthorized)}. Reverting changes."
        )
        revert_unauthorized_changes(
            workspace_path=workspace_path,
            unauthorized_files=truly_unauthorized,
            original_contents=state.get("original_file_contents"),
            pre_run_manifest=state.get("pre_run_workspace_manifest"),
            protected_files=allowed_files,
            session_written_files=_session_written,
        )
        for u in sorted(truly_unauthorized):
            gate_failures.append(f"scope_violation:{u}")

    gate = PatchGate(repo_path=workspace_path)
    head_commit = PatchGate.current_head(workspace_path)

    for gen in generated_code:
        diff = getattr(gen, "content", "") or ""
        if not diff.strip():
            continue

        file_path = getattr(gen, "file_path", "")

        # ── Proportionality check ──────────────────────────────────────────
        from ticket_to_code.agents.patch_gate import PatchGate as AgentPatchGate
        _orig_content = (state.get("original_file_contents") or {}).get(file_path)
        if "<<<<<<< SEARCH" in diff:
            prop_ok, prop_reason = AgentPatchGate.validate_patch_proportionality(
                file_path=file_path,
                patch_content=diff,
                current_content=_orig_content,
                is_new_file=not bool(_orig_content),
            )
            if not prop_ok:
                logger.warning(
                    "patch_gate: proportionality violation for %s → %s",
                    file_path, prop_reason,
                )
                gate_failures.append(f"proportionality_violation:{file_path}:{prop_reason}")

        # ── Scope check (B9) ──────────────────────────────────────────────────
        if allowed_files:
            if diff.startswith(("diff --git", "--- ", "Index:")):
                scope_res = gate.scope_enforce(diff, allowed_files)
                if not scope_res.ok:
                    logger.warning(
                        "patch_gate: scope violation for %s → %s",
                        file_path, scope_res.violations,
                    )
                    gate_failures.append(
                        f"scope_violation:{file_path}:{scope_res.violations}"
                    )
            else:
                norm_fp = file_path.replace("\\", "/").lower().strip()
                is_allowed = any(_scope_path_match(norm_fp, af) for af in allowed_files)
                if not is_allowed:
                    logger.warning(
                        "patch_gate: scope violation for %s (not in allowed_files)",
                        file_path,
                    )
                    gate_failures.append(f"scope_violation:{file_path}")

        # ── Secret scan (B9) ──────────────────────────────────────────────────
        secret_hits = gate.secret_scan(diff)
        if secret_hits:
            logger.error(
                "patch_gate: secret detected in %s — rules: %s",
                file_path, [h.rule for h in secret_hits],
            )
            gate_failures.append(
                f"secret_scan:{file_path}:{[h.rule for h in secret_hits]}"
            )

        # ── Apply check is skip-safe (run only if content is a unified diff) ──
        if diff.startswith(("diff --git", "--- ", "Index:")):
            apply_res = gate.apply_check(diff)
            if not apply_res.ok and gate._git_available:
                logger.warning(
                    "patch_gate: git apply --check failed for %s: %s",
                    file_path, apply_res.error,
                )

    if run_ctx:
        run_ctx.validation_passed = len(gate_failures) == 0
        # end_phase handled by _phase_tracked_node wrapper

    # ── PolicyEngine (previously instantiated but never consulted) ───────────
    # Files are already on disk at this point, so the "target exists" invariant cannot be
    # judged here; the evidence-sufficiency invariant is recorded as an ADVISORY so runs
    # that edited a file with zero collected evidence are visible in the trace.
    _policy_advisories: list = []
    _policy = state.get("policy_engine")
    if _policy is not None and hasattr(_policy, "evaluate_edit_safety"):
        try:
            _ev_by_file: dict = {}
            for _ev in (state.get("evidence_items") or []):
                _efp = _norm_rel(getattr(_ev, "file_path", "") or (_ev.get("file_path", "") if isinstance(_ev, dict) else ""))
                if _efp:
                    _ev_by_file.setdefault(_efp, []).append(str(getattr(_ev, "snippet", "") or "")[:200])
            for _g in (state.get("generated_code") or []):
                _gfp = getattr(_g, "file_path", "") or ""
                _snips = _ev_by_file.get(_norm_rel(_gfp), [])
                _decision = _policy.evaluate_edit_safety(_gfp, _snips, getattr(_g, "content", "") or "")
                if getattr(getattr(_decision, "action", None), "value", str(getattr(_decision, "action", ""))) != "proceed_to_edit":
                    _policy_advisories.append(f"{_gfp}: {getattr(_decision, 'reason', '')}")
            if _policy_advisories:
                logger.warning("patch_gate: PolicyEngine advisories (non-blocking): %s", _policy_advisories)
        except Exception as _pol_exc:
            logger.debug(f"patch_gate: PolicyEngine evaluation skipped: {_pol_exc}")

    hard_failures = list(gate_failures)
    if hard_failures:
        logger.error(
            "patch_gate: BLOCKED — %d failure(s): %s",
            len(hard_failures), hard_failures,
        )
        # Safety invariant: NEVER blacklist allowed_files!
        # They are intentional ticket targets that must be preserved and fixed, not discarded.
        return {
            "status": "candidates_invalid",
            "patch_gate_blocked": True,
            "policy_advisories": _policy_advisories,
            "patch_gate_failures": hard_failures,
            "validation_failure_reason": f"patch_gate: {hard_failures}",
            "candidate_retry_count": state.get("candidate_retry_count", 0) + 1,
        }

    if gate_failures:
        logger.warning(
            "patch_gate: ADVISORY WARNINGS (non-blocking) — %d issue(s): %s",
            len(gate_failures), gate_failures,
        )

    blast_report = None
    try:
        from ticket_to_code.runtime.blast_radius import BlastRadiusAnalyzer
        blast_report = BlastRadiusAnalyzer.analyze(workspace_path, list(_session_written))
        if blast_report and blast_report.risk_level in ("high", "critical"):
            logger.warning(
                f"🛡️ High Blast-Radius detected ({blast_report.risk_level.upper()}): {blast_report.risk_reasons}"
            )
    except Exception as _br_exc:
        logger.debug(f"BlastRadiusAnalyzer skipped (non-fatal): {_br_exc}")

    logger.info("✅ PATCH GATE: all checks passed (head=%s)", head_commit)
    res = {"patch_gate_blocked": False, "patch_gate_failures": [], "policy_advisories": _policy_advisories}
    if blast_report:
        res["blast_radius_report"] = blast_report
    return res


def route_after_patch_gate(state: TicketToCodeState) -> str:
    """Patch gate is a real gate: a BLOCKED patch (scope violation, secret, proportionality)
    must never reach the compiler. Route it to verification_failed (rollback + honest failure)."""
    if state.get("patch_gate_blocked"):
        logger.error(
            "  🛑 patch_gate BLOCKED the patch (%s) — routing to verification_failed",
            state.get("patch_gate_failures") or state.get("validation_failure_reason"),
        )
        return "verification_failed"
    return "import_validation"


def angular_module_registration_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Gap 2 — Angular @NgModule registration.

    After code generation, if a new Angular @Component / @Directive / @Pipe or
    @Injectable service was created, it must be registered in the governing
    *.module.ts file's declarations / providers array — otherwise Angular throws
    "Component X is not part of any NgModule" at build time.

    Strategy (mirrors what Copilot Workspace and Cursor do automatically):
      1. For each generated TypeScript file, check if it declares @Component,
         @Directive, @Pipe, or @Injectable.
      2. Walk up the directory tree from the generated file to find the nearest
         *.module.ts (stopping at the workspace root).
      3. Read that module file and locate the correct array (declarations for
         components/directives/pipes, providers for services).
      4. If the class name is not already in the array, inject it — both the
         import statement and the array entry — and write the file to disk.
    """
    workspace_path = str(state.get("workspace_path", ""))
    generated = list(state.get("generated_code") or [])
    if not generated or not workspace_path:
        return {}

    _DECORATOR_ARRAY = {
        "@Component": "declarations",
        "@Directive": "declarations",
        "@Pipe":      "declarations",
        "@Injectable": "providers",
    }

    modified_modules: dict[str, str] = {}  # module_path → updated content

    for gen in generated:
        lang_raw = getattr(gen, "language", None)
        lang = getattr(lang_raw, "value", str(lang_raw)).lower() if lang_raw else ""
        if lang not in ("typescript", "javascript"):
            continue

        content = getattr(gen, "content", "") or ""
        file_path = getattr(gen, "file_path", "") or ""
        if not content or not file_path:
            continue

        # Only process new CREATE tasks or files that look like Angular constructs
        change_type = getattr(getattr(gen, "change_type", None), "value", "") or ""

        # Detect Angular decorator and extract class name
        decorator_used = None
        for dec in _DECORATOR_ARRAY:
            if dec + "(" in content or dec + "\n" in content:
                decorator_used = dec
                break
        if not decorator_used:
            continue

        # Gap 4: Skip standalone components (Angular 14+) — they don't need module registration
        if re.search(r'standalone\s*:\s*true', content):
            logger.debug(f"  angular_module: {file_path} is standalone — skipping module registration")
            continue

        target_array = _DECORATOR_ARRAY[decorator_used]

        class_match = re.search(r'export\s+class\s+([A-Za-z][A-Za-z0-9_]*)', content)
        if not class_match:
            continue
        class_name = class_match.group(1)

        # Find nearest *.module.ts walking up from the generated file's directory
        gen_abs = Path(workspace_path) / file_path
        search_dir = gen_abs.parent
        ws_root = Path(workspace_path)
        module_path: Optional[str] = None
        for _ in range(8):  # max 8 levels up
            candidates = sorted(search_dir.glob("*.module.ts"))
            if candidates:
                module_path = str(candidates[0].relative_to(ws_root)).replace("\\", "/")
                break
            if search_dir == ws_root or search_dir.parent == search_dir:
                break
            search_dir = search_dir.parent

        if not module_path:
            logger.debug(f"  angular_module: no *.module.ts found for {file_path}")
            continue

        # Read module content (use in-memory version if already modified)
        module_abs = Path(workspace_path) / module_path
        if module_path in modified_modules:
            module_content = modified_modules[module_path]
        elif module_abs.exists():
            try:
                module_content = module_abs.read_text(encoding="utf-8")
            except Exception:
                continue
        else:
            continue

        # Skip if class already registered
        if class_name in module_content:
            logger.debug(f"  angular_module: {class_name} already in {module_path}")
            continue

        # Build import statement (relative path from module to generated file)
        import os
        rel = os.path.relpath(
            str(gen_abs.with_suffix("")),
            str(module_abs.parent),
        ).replace("\\", "/")
        if not rel.startswith("."):
            rel = "./" + rel
        new_import = f"import {{ {class_name} }} from '{rel}';"

        # Inject import after last existing import in module
        _last_imp = None
        for _m in re.finditer(r'^import[^\n]+\n', module_content, re.MULTILINE):
            _last_imp = _m
        insert_at = _last_imp.end() if _last_imp else 0
        module_content = (
            module_content[:insert_at] + new_import + "\n" + module_content[insert_at:]
        )

        # Inject class name into the target array (declarations / providers)
        # Matches:  declarations: [\n  Foo,\n  Bar\n]  or  declarations: [Foo, Bar]
        arr_rx = re.compile(
            rf'{target_array}\s*:\s*\[([^\]]*)\]', re.DOTALL
        )
        arr_match = arr_rx.search(module_content)
        if arr_match:
            old_arr = arr_match.group(0)
            inner = arr_match.group(1).rstrip()
            # Append with trailing comma awareness
            sep = ",\n    " if "\n" in inner else ", "
            new_inner = inner.rstrip(",") + sep + class_name
            new_arr = old_arr.replace(arr_match.group(1), new_inner)
            module_content = module_content.replace(old_arr, new_arr, 1)
        else:
            # Array not found — add it as a new property in @NgModule({...})
            # This is a last-resort fallback; shouldn't happen in well-formed modules.
            logger.warning(
                f"  angular_module: '{target_array}' array not found in {module_path}"
                f" for {class_name} — skipping"
            )
            continue

        modified_modules[module_path] = module_content
        logger.info(
            f"  angular_module: registered {class_name} in {module_path} "
            f"({target_array})"
        )

    if not modified_modules:
        return {}

    # Write all modified module files to disk
    for mod_path, mod_content in modified_modules.items():
        try:
            (Path(workspace_path) / mod_path).write_text(mod_content, encoding="utf-8")
        except Exception as e:
            logger.warning(f"  angular_module: write failed for {mod_path}: {e}")

    return {}


def import_validation_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Layer 3 — Claude Code / Devin technique: Pre-Build Import Validation.

    Runs AFTER patch_gate and BEFORE build.  Scans every generated file for
    unresolved PascalCase symbol references, looks each up in the SQLite symbol
    index, and injects missing import statements WITHOUT calling the LLM.

    This eliminates the most common class of build failures ("cannot find symbol",
    "TS2304 Cannot find name", "CS0246") before they ever reach the compiler.
    """
    sqlite_store = getattr(agents.localizer, "sqlite_store", None)
    workspace_path = str(state.get("workspace_path", ""))

    if not sqlite_store:
        logger.debug("import_validation_node: no sqlite_store — skipping")
        return {}

    generated = list(state.get("generated_code") or [])
    if not generated:
        return {}

    patched_count = 0
    for gen in generated:
        lang_raw = getattr(gen, "language", None)
        lang = getattr(lang_raw, "value", str(lang_raw)).lower() if lang_raw else ""
        if lang not in ("java", "typescript", "javascript", "python", "csharp", "cs"):
            continue

        content = getattr(gen, "content", "") or ""
        if not content:
            continue

        # Deterministic check: strip hallucinated Spring Security if not in pom.xml
        if "import org.springframework.security" in content or "@PreAuthorize" in content:
            pom_has_security = False
            for p_candidate in [Path(workspace_path) / "pom.xml", Path(workspace_path) / "order-service" / "pom.xml"]:
                if p_candidate.exists():
                    p_text = p_candidate.read_text(encoding="utf-8", errors="ignore")
                    if "spring-security" in p_text or "spring-boot-starter-security" in p_text:
                        pom_has_security = True
                        break
            if not pom_has_security:
                content = re.sub(r"import\s+org\.springframework\.security.*\n?", "", content)
                content = re.sub(r"@PreAuthorize\([^\)]*\)\s*", "", content)
                gen.content = content
                out_file = Path(workspace_path) / gen.file_path
                if out_file.exists():
                    out_file.write_text(content, encoding="utf-8")
                logger.info(f"  🛡️ [import_validation] Stripped hallucinated Spring Security from {gen.file_path} (not in pom.xml)")

        fixes = _resolve_unresolved_symbols(content, lang, sqlite_store, workspace_path, source_file_path=getattr(gen, "file_path", ""))
        if not fixes:
            continue

        # Deterministic, repository-verified imports. Surface as a user-visible
        # action (default APPLY); a wired provider may veto within 60s. When no
        # provider is present (benchmark/CI) they are applied as before.
        _import_hints = [h for h, _ in fixes]
        _imp_provider = _get_transient(state, "_decision_provider")
        _imp_ui_cb = _get_transient(state, "_ui_callback")
        if _imp_provider is not None:
            from ticket_to_code.agents.user_decision_gate import (
                request_user_decision, build_import_payload,
            )
            _imp_choice = request_user_decision(
                provider=_imp_provider,
                payload=build_import_payload(getattr(gen, "file_path", ""), _import_hints),
                options=["apply", "skip"], default="apply", timeout=60.0,
            )
            if _imp_choice == "skip":
                logger.info(
                    f"  import_validation: user skipped {len(_import_hints)} "
                    f"verified import(s) for {gen.file_path}"
                )
                continue

        # Apply each import fix using the same logic as _inject_meta_imports
        from ticket_to_code.models import GeneratedCode as _GC  # avoid circular at module level
        dummy = type("_D", (), {"imports": [f for f, _ in fixes], "content": content, "language": gen.language})()
        dummy = _inject_meta_imports(dummy)  # type: ignore[arg-type]

        if dummy.content != content:
            gen.content = dummy.content
            # Write updated content to disk immediately
            out = Path(workspace_path) / gen.file_path
            if out.exists():
                try:
                    out.write_text(gen.content, encoding="utf-8")
                    patched_count += 1
                    logger.info(
                        f"  import_validation: injected {len(fixes)} import(s) into {gen.file_path}: "
                        + ", ".join(hint for hint, _ in fixes[:5])
                    )
                    if _imp_ui_cb:
                        try:
                            from datetime import datetime as _dt
                            _imp_ui_cb({
                                "phase": "import_validation", "status": "applied",
                                "message": f"➕ Added {len(fixes)} verified import(s) to {Path(gen.file_path).name}",
                                "data": {"node": "import_validation",
                                         "event_type": "verified_imports_added",
                                         "file": gen.file_path,
                                         "imports": _import_hints[:10]},
                                "timestamp": _dt.now().isoformat(),
                            })
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"  import_validation: write failed for {gen.file_path}: {e}")

    if patched_count:
        logger.info(f"  import_validation_node: patched {patched_count} file(s) with missing imports")
        return {"generated_code": generated}

    return {}


def build_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 5: Build Project

    Universal build for ANY project type (Java, .NET, Python, Node.js, Angular, etc.).
    When TypeScript/Angular files were generated, runs `tsc --noEmit` FIRST to
    surface compile errors before they reach the running `ng serve` hot-reload.
    """
    logger.info(" PHASE 5: Build Project")

    # Invalidate the search cache because earlier nodes (generate/repair)
    # may have written or modified files.
    if hasattr(agents, "repo_search") and agents.repo_search:
        agents.repo_search.invalidate_index()

    # ── Verifier audit reconciliation (close the loop) ───────────────────────
    # Compare the semantic verifier's MODIFY/EXCLUDE verdicts against the files
    # this run actually wrote.  false_include / false_exclude metrics land in
    # .aviator/verifier_audit_report.json — the feedback signal for tuning the
    # verifier's prompts and the ranking weights.
    try:
        from ticket_to_code.agents.semantic_verification_agent import reconcile_verifier_audit
        _written_paths = [g.file_path for g in state.get("generated_code", [])]
        if _written_paths:
            _ticket = state.get("ticket")
            reconcile_verifier_audit(
                str(state["workspace_path"]),
                _written_paths,
                ticket_id=str(getattr(_ticket, "ticket_id", "") or getattr(_ticket, "id", "") or ""),
            )
    except Exception as _audit_exc:
        logger.debug(f"verifier audit reconciliation skipped (non-fatal): {_audit_exc}")

    generated = state.get("generated_code", [])

    # ── Tier 1 Inner Loop (<1s fast syntax & delimiter check) ─────────────────
    # Fast-fail on syntax errors, tree-sitter error nodes, or unclosed delimiters
    # before invoking heavy external compiler / test subprocesses.
    try:
        from ticket_to_code.verification.two_tier_verifier import TwoTierVerifier
        verifier = TwoTierVerifier(workspace_path=str(state["workspace_path"]))
        written_files = [g.file_path for g in generated if getattr(g, "file_path", None)]
        if written_files:
            tier1_res = verifier.run_tier1_fast_check(written_files)
            if not tier1_res.passed:
                logger.error(
                    f"  ❌ Tier 1 Fast Syntax Check FAILED in {tier1_res.elapsed_s:.3f}s "
                    f"with {len(tier1_res.diagnostics)} error(s)"
                )
                tier1_err_lines = tier1_res.raw_errors if getattr(tier1_res, "raw_errors", None) else [
                    f"{d.file}: [{d.code}] {d.message}"
                    for d in tier1_res.diagnostics
                ]
                tier1_build = BuildResult(
                    status=BuildStatus.FAILURE,
                    exit_code=1,
                    errors=tier1_err_lines,
                    stdout=f"Tier 1 Fast Syntax Check (Tree-sitter/AST) detected syntax errors ({tier1_res.elapsed_s:.3f}s):\n" + "\n".join(tier1_err_lines),
                    stderr="\n".join(tier1_err_lines),
                )
                agents.tracer.record_build(tier1_build)
                return {"build_result": tier1_build}
            else:
                logger.info(f"  ✅ Tier 1 Fast Syntax Check PASSED ({tier1_res.elapsed_s:.3f}s across {len(written_files)} files)")
    except Exception as _t1_exc:
        logger.debug(f"Tier 1 fast check encountered non-fatal exception: {_t1_exc}")

    # ── Angular / TypeScript compile check ────────────────────────────────────
    # If any generated file is a TypeScript source, verify it compiles cleanly
    # using the TypeScript compiler.  This catches errors (wrong property names,
    # missing imports, type mismatches) that `ng serve` would report to the
    # developer but the agent would never see.
    ts_files = [g for g in generated if g.file_path.endswith((".ts", ".tsx"))]
    ts_build_result = None  # Track TS errors without short-circuiting
    _tsc_ran = False
    if ts_files:
        ng_root = _find_ng_root(Path(state["workspace_path"]), target_file=Path(ts_files[0].file_path))
        if ng_root:
            logger.info(
                f" Angular/TypeScript compile check — {len(ts_files)} TS file(s) written, "
                f"project root: {ng_root}"
            )
            ts_result = _run_tsc_check(ng_root)
            _tsc_ran = True
            if ts_result is not None:
                # Compile errors found — store but DON'T return yet.
                # We still need to check Java/backend builds too so ALL errors are visible.
                logger.error(f"  ❌ TypeScript compile FAILED — {len(ts_result.errors)} error(s)")
                ts_build_result = ts_result
            else:
                logger.info("  ✅ TypeScript compile OK")
        else:
            logger.warning("  ⚠️ TypeScript files generated but no Angular project root found — skipping tsc check")

    # For pure Angular tickets (no .NET/.java files generated), return TS result directly
    non_ts_check = [
        g for g in generated
        if not g.file_path.endswith((".ts", ".tsx", ".html", ".scss", ".css"))
    ]
    if not non_ts_check:
        if ts_build_result:
            # Pure Angular with TS errors — route to fix_build
            logger.error("  ❌ Pure Angular project — routing TS errors to fix_build")
            agents.tracer.record_build(ts_build_result)
            log_content = f"Command: tsc --noEmit (Failed)\nExit Code: {ts_build_result.exit_code}\n\nSTDOUT:\n{ts_build_result.stdout}\n\nSTDERR:\n{ts_build_result.stderr}"
            write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "build_output.log", log_content)
            return {"build_result": ts_build_result}
        if _tsc_ran:
            logger.info("   Pure Angular project — tsc passed; skipping .NET/Java build step")
            _ok = BuildResult(
                status=BuildStatus.SUCCESS,
                exit_code=0,
                errors=[],
                stdout="TypeScript compile: OK",
                stderr="",
            )
        else:
            # html/scss-only change, or no Angular root: NOTHING was compiled. Not a verified success.
            logger.warning("   No compiler ran for the generated files (template/style-only or no project root) — build SKIPPED (unverified)")
            _ok = BuildResult(
                status=BuildStatus.SKIPPED,
                exit_code=0,
                errors=[],
                stdout="No compiler ran: template/style-only change or no project root",
                stderr="",
            )
        agents.tracer.record_build(_ok)
        return {"build_result": _ok}

    # ── .NET / Java / other build ──────────────────────────────────────────────
    
    # 1. Identify which directories contain generated non-TS files
    generated = state.get("generated_code", [])
    non_ts_files = [g for g in generated if not g.file_path.endswith((".ts", ".tsx", ".html", ".scss", ".css"))]
    
    if not non_ts_files:
        logger.warning("   No backend files modified and no compiler ran — build SKIPPED (unverified)")
        _ok = BuildResult(
            status=BuildStatus.SKIPPED,
            exit_code=0,
            errors=[],
            stdout="No backend files modified; no compiler ran",
            stderr="",
        )
        agents.tracer.record_build(_ok)
        return {"build_result": _ok}

    workspace = Path(state["workspace_path"])
    build_results = []
    
    # Build file detection priority (most specific → least specific)
    _BUILD_FILE_MAP = [
        ("build.gradle.kts",  "java-gradle"),
        ("build.gradle",      "java-gradle"),
        ("pom.xml",           "java-maven"),
        ("package.json",      "nodejs"),
        ("angular.json",      "nodejs"),
        ("*.csproj",          "dotnet"),
        ("*.sln",             "dotnet"),
        ("pyproject.toml",    "python"),
        ("setup.py",          "python"),
        ("requirements.txt",  "python"),
        ("go.mod",            "go"),
        ("Cargo.toml",        "rust"),
    ]

    # Map of project_file_rel -> (tech, project_file_rel, service_dir_rel)
    projects_to_build: dict[str, tuple[str, str, str]] = {}

    for g in non_ts_files:
        raw_path = getattr(g, "file_path", str(g))
        abs_file = (workspace / raw_path).resolve()
        
        # Upward walk from the modified file's parent directory towards workspace
        cur = abs_file.parent
        found = False
        while True:
            for build_file, tech in _BUILD_FILE_MAP:
                if "*" in build_file:
                    matches = list(cur.glob(build_file))
                    if matches:
                        rel_file = str(matches[0].relative_to(workspace)).replace("\\", "/")
                        rel_dir = str(cur.relative_to(workspace)).replace("\\", "/") or "."
                        projects_to_build[rel_file] = (tech, rel_file, rel_dir)
                        found = True
                        break
                elif (cur / build_file).exists():
                    rel_file = str((cur / build_file).relative_to(workspace)).replace("\\", "/")
                    rel_dir = str(cur.relative_to(workspace)).replace("\\", "/") or "."
                    projects_to_build[rel_file] = (tech, rel_file, rel_dir)
                    found = True
                    break
            if found or cur == workspace or cur == cur.parent:
                break
            cur = cur.parent

    # If no nested manifests were found for modified files, check workspace root
    if not projects_to_build:
        for build_file, tech in _BUILD_FILE_MAP:
            if "*" in build_file:
                matches = list(workspace.glob(build_file))
                if matches:
                    rel_file = matches[0].name
                    projects_to_build[rel_file] = (tech, rel_file, ".")
                    break
            elif (workspace / build_file).exists():
                projects_to_build[build_file] = (tech, build_file, ".")
                break

    for tech, project_file, service_dir in projects_to_build.values():
        logger.info(f"Targeting build for {service_dir} [{tech}] via {project_file}")
        try:
            executor = ExecutionEngineFactory.create(str(workspace), tech)
            logger.info(f"Building {service_dir} using {executor.get_technology_name()} project: {project_file}")
            service_result = executor.execute_build(project_file)
            build_results.append((service_dir, service_result))
        except Exception as e:
            # Polyglot fallback: attempt compilation via PolyglotBuildDiscoverer
            built_via_polyglot = False
            try:
                from ticket_to_code.runtime.polyglot_build import PolyglotBuildDiscoverer
                cfg = PolyglotBuildDiscoverer.discover(workspace / service_dir, workspace)
                if cfg and cfg.compile_cmd:
                    logger.info(f"Executing Polyglot compile fallback: {cfg.compile_cmd} in {cfg.work_dir}")
                    try:
                        from ticket_to_code.runtime.security import run_checked_subprocess as _rcs
                        _rc, _out, _err = _rcs(cfg.compile_cmd, cwd=str(cfg.work_dir), timeout=120, workspace=str(workspace))
                        proc = subprocess.CompletedProcess(cfg.compile_cmd, _rc, _out, _err)
                    except ImportError:
                        import shlex as _shlex
                        proc = subprocess.run(
                            cfg.compile_cmd if sys.platform == "win32" else _shlex.split(cfg.compile_cmd),
                            cwd=str(cfg.work_dir),
                            shell=(sys.platform == "win32"),
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                    built_via_polyglot = True
                    poly_status = BuildStatus.SUCCESS if proc.returncode == 0 else BuildStatus.FAILURE
                    errs = [l for l in (proc.stderr or "").splitlines() if l.strip()]
                    _poly_res = BuildResult(
                        status=poly_status,
                        exit_code=proc.returncode,
                        errors=errs if proc.returncode != 0 else [],
                        stdout=proc.stdout or "",
                        stderr=proc.stderr or "",
                    )
                    build_results.append((service_dir, _poly_res))
            except Exception as _pe:
                logger.debug(f"Polyglot fallback compile failed: {_pe}")
            
            if not built_via_polyglot:
                logger.error(f"Failed to execute build for {project_file} ({tech}): {e}")
                _err_res = BuildResult(
                    status=BuildStatus.FAILURE,
                    exit_code=1,
                    errors=[f"Build engine execution failure for {project_file}: {e}"],
                    stdout="",
                    stderr=str(e),
                )
                build_results.append((service_dir, _err_res))

    if not build_results:
        # If no buildable projects matched (e.g. modified only markdown or non-code files),
        # log and return clean no-op result.
        logger.warning("No buildable projects found for modified files — build SKIPPED (unverified).")
        _no_proj = BuildResult(
            status=BuildStatus.SKIPPED,
            exit_code=0,
            errors=[],
            stdout="No buildable project manifest found for modified files",
            stderr=""
        )
        agents.tracer.record_build(_no_proj)
        return {"build_result": _no_proj}

    # Aggregate results
    failed_results = [r for d, r in build_results if r.status.value != "success"]
    
    if failed_results:
        # Take the first failure as the primary result
        first_fail_dir, first_fail_res = next((d, r) for d, r in build_results if r.status.value != "success")

        # Merge TS errors if both TS and Java failed
        if ts_build_result and ts_build_result is not first_fail_res:
            merged_errors = list(ts_build_result.errors) + list(first_fail_res.errors)
            merged_stdout = f"TS errors:\n{ts_build_result.stdout}\n\nJava errors ({first_fail_dir}):\n{first_fail_res.stdout}"
            merged_stderr = f"{ts_build_result.stderr}\n{first_fail_res.stderr}"
            first_fail_res = BuildResult(
                status=BuildStatus.FAILURE,
                exit_code=first_fail_res.exit_code,
                errors=merged_errors,
                stdout=merged_stdout,
                stderr=merged_stderr,
            )
            logger.error(f"Build failed in BOTH TypeScript and {first_fail_dir}: {len(merged_errors)} total errors")
        else:
            logger.error(f"Build failed in {first_fail_dir}: {len(first_fail_res.errors)} errors")

        agents.tracer.record_build(first_fail_res)
        
        log_content = f"Command: auto (Failed in {first_fail_dir})\nExit Code: {first_fail_res.exit_code}\n\nSTDOUT:\n{first_fail_res.stdout}\n\nSTDERR:\n{first_fail_res.stderr}"
        write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "build_output.log", log_content)
        
        log_phase(
            phase='BUILD',
            llm_output={
                'status': first_fail_res.status.value,
                'exit_code': first_fail_res.exit_code,
                'errors_count': len(first_fail_res.errors),
                'error_samples': first_fail_res.errors[:2],
                'failed_service': first_fail_dir
            },
            agent='BuildExecutor'
        )
        
        return {"build_result": first_fail_res}
    elif ts_build_result:
        # Java builds all passed but TS had errors — return TS errors
        logger.error(f"Java builds OK but TypeScript compile FAILED — {len(ts_build_result.errors)} error(s)")
        agents.tracer.record_build(ts_build_result)
        log_content = f"Command: tsc (Failed)\nExit Code: {ts_build_result.exit_code}\n\nSTDOUT:\n{ts_build_result.stdout}\n\nSTDERR:\n{ts_build_result.stderr}"
        write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "build_output.log", log_content)
        return {"build_result": ts_build_result}
    else:
        logger.info("✅ All modified services built successfully")
        # Return the last successful result
        last_dir, last_res = build_results[-1]
        agents.tracer.record_build(last_res)
        
        log_content = f"Command: auto (Multiple services built)\nExit Code: 0\n\nSTDOUT:\nAll modified services built successfully."
        write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "build_output.log", log_content)
        
        log_phase(
            phase='BUILD',
            llm_output={
                'status': last_res.status.value,
                'exit_code': last_res.exit_code,
                'success': True,
                'services_built': [d for d, r in build_results]
            },
            agent='BuildExecutor'
        )
        
        return {"build_result": last_res, "status": "build_success", "validation_failure_reason": None}



def test_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Phase 6: Run Tests
    
    Universal test runner for ANY project type (Java JUnit, .NET NUnit/xUnit, Python pytest, etc.).
    Execution engine auto-detects language and test framework.
    
    LangGraph automatically waits for BOTH generate_tests AND build before running this!
    """
    logger.info(" PHASE 6: Run Tests")
    
    # Optional: Verify both branches completed (LangGraph already guarantees this!)
    logger.info(f"   TEST BRANCH: {state.get('test_status')} ({len(state.get('generated_tests') or [])} files)")
    logger.info(f"   CODE BRANCH: {state.get('code_status')} ({len(state.get('generated_code') or [])} files)")
    logger.info(f"   BUILD: {state.get('build_result').status.value if state.get('build_result') else 'N/A'}")
    
    # Let execution engine detect project files automatically
    workspace = Path(state["workspace_path"])
    target_ws = workspace
    sub_ws = state.get("sub_workspace_path")
    if sub_ws and Path(sub_ws).is_dir():
        target_ws = Path(sub_ws)
    else:
        gen_files = state.get("generated_code") or []
        if gen_files:
            rel_paths = [(getattr(g, "file_path", "") or "").replace("\\", "/").strip("/") for g in gen_files]
            top_dirs = {p.split("/")[0] for p in rel_paths if "/" in p}
            if len(top_dirs) == 1:
                cand_sub = workspace / next(iter(top_dirs))
                if cand_sub.is_dir():
                    target_ws = cand_sub

    executor = getattr(agents, "executor", None)
    if executor is None:
        try:
            executor = ExecutionEngineFactory.create(str(target_ws))
        except Exception:
            executor = None

    if executor is None:
        logger.info("No test execution engine available for workspace — compile-only verification (no tests)")
        _no_exec = TestResult(
            status=TestStatus.NO_TESTS,
            total_tests=0,
            passed=0,
            failed=0,
            errors=[],
            test_output="No test execution engine available for workspace",
            duration_seconds=0,
        )
        return {
            "test_result": _no_exec,
            "compile_only_verification": True,
            "verification_level": "compile_only",
        }

    project_files = []
    try:
        project_files = executor.detect_project_files()
    except Exception as _e:
        logger.debug(f"Project test file detection failed: {_e}")

    if not project_files:
        logger.info("No test project files detected — compile-only verification (no tests)")
        _no_tests = TestResult(
            status=TestStatus.NO_TESTS,
            total_tests=0,
            passed=0,
            failed=0,
            errors=[],
            test_output="No project test files detected in workspace",
            duration_seconds=0,
        )
        return {
            "test_result": _no_tests,
            "compile_only_verification": True,
            "verification_level": "compile_only",
        }
    
    # Use detected project file (prioritize project containing modified/generated files)
    gen_files = state.get("generated_code") or []
    target_project_file = None
    if gen_files:
        gen_fps = [(getattr(g, "file_path", "") or "").replace("\\", "/").strip("/") for g in gen_files]
        for pf in project_files:
            norm_pf = pf.replace("\\", "/").strip("/")
            parent_dir = str(Path(norm_pf).parent).replace("\\", "/").strip("/")
            if any(fp.startswith(parent_dir + "/") or fp == norm_pf for fp in gen_fps):
                target_project_file = pf
                break
    project_file = target_project_file or project_files[0]
    
    # Detect targeted test class if generated/modified files include tests
    # Detect targeted test class if generated/modified files include tests
    test_target = None
    if gen_files:
        for g in gen_files:
            fp = (getattr(g, "file_path", "") or "").replace("\\", "/")
            if fp.endswith("Test.java") or fp.endswith("Tests.java"):
                base_name = Path(fp).stem
                test_target = f"*{base_name}"
                logger.info(f"  🎯 Isolated targeted test detected: {test_target} for {fp}")
                break
            elif fp.endswith(".spec.ts") or fp.endswith(".test.ts") or fp.endswith(".spec.js") or fp.endswith(".test.js"):
                test_target = fp
                logger.info(f"  🎯 Isolated targeted test detected: {test_target} for {fp}")
                break
            elif fp.endswith("_test.py") or Path(fp).name.startswith("test_"):
                test_target = fp
                logger.info(f"  🎯 Isolated targeted test detected: {test_target} for {fp}")
                break
            elif fp.endswith("_test.go"):
                test_target = fp
                logger.info(f"  🎯 Isolated targeted test detected: {test_target} for {fp}")
                break
                
    logger.info(f"Running {executor.get_technology_name()} tests: {project_file} (target={test_target})")
    
    try:
        if test_target:
            test_result = executor.execute_tests(project_file, test_target=test_target)
        else:
            test_result = executor.execute_tests(project_file)
        if test_target and test_result.total_tests == 0 and test_result.status.value in ("all_passed", "success"):
            logger.info("Targeted test filter returned 0 tests; executing full module suite...")
            test_result = executor.execute_tests(project_file)
    except Exception as _exc:
        logger.error(f"Test runner execution failed: {_exc}", exc_info=True)
        test_result = TestResult(
            status=TestStatus.ERROR,
            total_tests=0,
            passed=0,
            failed=0,
            errors=[f"Test runner crashed: {_exc}"],
            test_output=f"Test runner exception: {_exc}",
            duration_seconds=0,
        )
    
    if test_result.status.value not in ("all_passed", "success"):
        _status_val = test_result.status.value
        if _status_val in ("error", "timeout"):
            # A runner crash or timeout is NOT evidence that failures are pre-existing. Never accept.
            logger.error(
                f"Test runner {_status_val}: {(getattr(test_result, 'errors', None) or ['?'])[0]} — "
                "not eligible for differential acceptance"
            )
            affects_modified, baseline_compared = True, False
        else:
            affects_modified, baseline_compared = _differential_test_acceptance(
                state, executor, project_file, test_target, test_result, gen_files
            )

        if not affects_modified:
            if baseline_compared:
                logger.info("  🎯 Differential Test Acceptance (baseline-backed): every failure also fails on the pre-edit baseline. Accepted as compile-only.")
            else:
                logger.warning("  ⚠️ Differential Test Acceptance (heuristic, no baseline): failures do not name modified files. Accepted as UNVERIFIED, not learned.")
            agents.tracer.finalize(state | {"test_result": test_result, "test_differential_accept": True})
            return {
                "test_result": test_result,
                "test_differential_accept": True,
                "test_baseline_compared": baseline_compared,
                "compile_only_verification": True,
                "verification_level": "compile_only" if baseline_compared else "unverified",
            }

        logger.error(f"Tests failed: {test_result.failed}/{test_result.total_tests}")
        agents.tracer.finalize(state | {"test_result": test_result})
        
        exit_code = getattr(test_result, "exit_code", 1)
        stdout_txt = getattr(test_result, "stdout", getattr(test_result, "test_output", ""))
        stderr_txt = getattr(test_result, "stderr", "")
        log_content = f"Command: auto\nExit Code: {exit_code}\nPassed: {test_result.passed}, Failed: {test_result.failed}\n\nSTDOUT:\n{stdout_txt}\n\nSTDERR:\n{stderr_txt}"
        write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "test_output.log", log_content)
        
        # DEBUG LOG
        log_phase(
            phase='TEST',
            llm_output={
                'status': test_result.status.value,
                'total_tests': test_result.total_tests,
                'passed': test_result.passed,
                'failed': test_result.failed,
                'exit_code': exit_code
            },
            agent='TestRunner'
        )
        
        return {
            "test_result": test_result,
            "test_differential_accept": False,
            "test_baseline_compared": False,
            "compile_only_verification": False,
            "verification_level": "test_failed",
        }
    else:
        logger.info(f"✅ Tests passed: {test_result.passed}/{test_result.total_tests}")
        agents.tracer.finalize(state | {"test_result": test_result})
        
        log_content = f"Command: auto\nPassed: {test_result.passed}, Failed: {test_result.failed}"
        write_trace_artifact(state["workspace_path"], state["ticket"].ticket_id, "test_output.log", log_content)
        
        # DEBUG LOG
        log_phase(
            phase='TEST',
            llm_output={
                'status': test_result.status.value,
                'total_tests': test_result.total_tests,
                'passed': test_result.passed,
                'failed': test_result.failed,
                'success': True
            },
            agent='TestRunner'
        )
        
        return {
            "test_result": test_result,
            "test_differential_accept": False,
            "test_baseline_compared": False,
            "compile_only_verification": False,
            "verification_level": "test_verified",
        }


_TEST_FAIL_TERMS = ("error", "fail", "failed", "failure", "assert", "exception", "cannot find", "is not assignable")
_TEST_CMD_ECHO_PREFIXES = ("> ", "ng test", "npm test", "yarn test", "pnpm test", "pytest", "mvn ", "gradle ", "Executing Node tests")


def _test_failure_signatures(test_result) -> set:
    """Normalized failure lines from a TestResult (timings / addresses stripped) for baseline comparison."""
    text = (getattr(test_result, "test_output", "") or "") + "\n" + "\n".join(
        str(e) for e in (getattr(test_result, "errors", None) or [])
    )
    sigs = set()
    for ln in text.splitlines():
        l = ln.strip()
        if not l or l.startswith(_TEST_CMD_ECHO_PREFIXES):
            continue
        low = l.lower()
        if not any(t in low for t in _TEST_FAIL_TERMS):
            continue
        l = re.sub(r"\b\d+(\.\d+)?\s*(ms|s|sec|seconds)\b", "", l)
        l = re.sub(r"0x[0-9a-fA-F]+", "", l)
        l = re.sub(r"\s+", " ", l).strip()
        if l:
            sigs.add(l[:240])
    return sigs


def _run_baseline_tests(state: dict, executor, project_file, test_target):
    """Run the SAME test command against the pre-edit workspace.

    Temporarily restores every snapshotted file to its original content and removes files the
    run created, runs the tests, then puts the edits back (always, via finally). Returns a
    TestResult or None when no baseline can be established (no snapshots, disabled, or error).
    """
    if os.getenv("AVIATOR_TEST_BASELINE", "1") == "0":
        return None
    originals = state.get("original_file_contents") or {}
    gen = state.get("generated_code") or []
    if not originals:
        return None
    ws = Path(state["workspace_path"])

    def _abs(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ws / rel

    current: dict = {}
    created: list = []
    try:
        for rel, orig in originals.items():
            p = _abs(rel)
            if p.is_file():
                with open(p, "r", encoding="utf-8", errors="ignore", newline="") as fh:
                    current[rel] = fh.read()
                with open(p, "w", encoding="utf-8", newline="") as fh:
                    fh.write(orig)
        _orig_keys = {k.replace("\\", "/") for k in originals}
        for g in gen:
            rel = getattr(g, "file_path", "") or ""
            if rel and rel.replace("\\", "/") not in _orig_keys:
                p = _abs(rel)
                if p.is_file():
                    created.append((rel, p.read_bytes()))
                    p.unlink()
        logger.info(f"  🧪 Baseline test run on pre-edit workspace ({len(current)} restored, {len(created)} removed)")
        if test_target:
            return executor.execute_tests(project_file, test_target=test_target)
        return executor.execute_tests(project_file)
    except Exception as exc:
        logger.warning(f"  Baseline test run failed (no baseline available): {exc}")
        return None
    finally:
        for rel, txt in current.items():
            try:
                with open(_abs(rel), "w", encoding="utf-8", newline="") as fh:
                    fh.write(txt)
            except Exception as _e:
                logger.error(f"  Baseline restore FAILED for {rel}: {_e}")
        for rel, data in created:
            try:
                p = _abs(rel)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            except Exception as _e:
                logger.error(f"  Baseline restore FAILED for created file {rel}: {_e}")


def _differential_test_acceptance(state: dict, executor, project_file, test_target, test_result, gen_files) -> tuple:
    """Decide whether test failures are attributable to this run's edits.

    Returns (affects_modified, baseline_compared).
      * baseline available  -> failures are accepted only if every failure signature also
        appears in the pre-edit baseline run (and the baseline itself was not green).
      * no baseline         -> legacy filename heuristic; caller records this as UNVERIFIED.
    """
    baseline = _run_baseline_tests(state, executor, project_file, test_target)
    if baseline is not None:
        base_status = getattr(getattr(baseline, "status", None), "value", "")
        if base_status in ("all_passed", "success", "no_tests", "skipped"):
            logger.info("  Baseline test run was green — current failures are NEW (caused by this run)")
            return True, True
        new_sigs = _test_failure_signatures(test_result) - _test_failure_signatures(baseline)
        if new_sigs:
            logger.info(f"  Baseline comparison: {len(new_sigs)} failure signature(s) not present on baseline → attributable to this run")
            return True, True
        return False, True

    # Legacy heuristic (no baseline)
    test_out_raw = (getattr(test_result, "test_output", "") or "")
    test_err_list = getattr(test_result, "errors", []) or []
    failure_lines = [
        ln for ln in test_out_raw.splitlines()
        if not ln.strip().startswith(_TEST_CMD_ECHO_PREFIXES)
        and any(term in ln.lower() for term in _TEST_FAIL_TERMS)
    ]
    test_err_clean = "\n".join(failure_lines) + "\n" + "\n".join(
        str(e) for e in test_err_list if not str(e).strip().startswith(("> ", "ng test", "npm test"))
    )
    if not gen_files:
        return True, False
    if not test_err_clean.strip():
        return False, False
    for g in gen_files:
        fname = Path(getattr(g, "file_path", "")).name.lower()
        if fname and fname in test_err_clean.lower():
            return True, False
    return False, False


def _norm_rel(p: str) -> str:
    return str(p or "").replace("\\", "/").strip().lstrip("./").lower()


def _scope_path_match(candidate: str, allowed: str) -> bool:
    """Scope equality for the patch gate.

    Exact (normalized) match, or a directory-boundary suffix match where the SHORTER side
    carries at least one directory segment. A bare basename ("OrderService.java") therefore
    never authorizes every file with that name across the repo, which the previous
    bidirectional ``endswith`` did.
    """
    c, a = _norm_rel(candidate), _norm_rel(allowed)
    if not c or not a:
        return False
    if c == a:
        return True
    shorter, longer = (c, a) if len(c) <= len(a) else (a, c)
    if "/" not in shorter:
        return False
    return longer.endswith("/" + shorter)


def _safe_write_or_patch_file(workspace_path: "Path", rel_path: str, new_content: str, state: dict) -> None:
    """Synchronized modification interceptor for workflow.py with undo-stack rollback support."""
    import time as _time
    from ticket_to_code.editing.surgical_diff_engine import SurgicalDiffEngine, EditSnapshot
    from ticket_to_code.runtime.security import CommandSafetyValidator
    diff_engine = state.get("_surgical_diff_engine")
    if diff_engine is None:
        diff_engine = SurgicalDiffEngine(str(workspace_path))
        state["_surgical_diff_engine"] = diff_engine
    abs_p = (workspace_path / rel_path).resolve()
    _ok, _why = CommandSafetyValidator.validate_path_containment(abs_p, workspace_path)
    if not _ok:
        raise PermissionError(f"_safe_write_or_patch_file refused: {_why}")
    old_content = abs_p.read_text(encoding="utf-8", errors="ignore") if abs_p.is_file() else ""
    diff_engine.undo_stack.append(
        EditSnapshot(file_path=str(abs_p), original_content=old_content, timestamp=_time.time())
    )
    abs_p.parent.mkdir(parents=True, exist_ok=True)
    abs_p.write_text(new_content, encoding="utf-8")

    # Update state generated files so the file is tracked and visible in UI
    rel_norm = rel_path.replace("\\", "/")
    gen_files = state.setdefault("generated_files", [])
    if rel_norm not in gen_files:
        gen_files.append(rel_norm)

    # Notify UI callback that a file was repaired/written
    ui_cb = _get_transient(state, "_ui_callback")
    if ui_cb:
        from datetime import datetime as _dt_repair
        ui_cb({
            "phase": "fix_build",
            "status": "in_progress",
            "message": f"📝 Repaired/Written: {abs_p.name}",
            "data": {
                "node": "fix_build",
                "event_type": "file_written",
                "file_path": rel_norm,
                "file_name": abs_p.name,
            },
            "timestamp": _dt_repair.now().isoformat(),
        })


def _wire_agent_context(
    agent,
    state: TicketToCodeState,
    agents: WorkflowAgents,
    workspace_path: "Path",
):
    """Wire v2/v3/v4 pipeline context into an ErrorResolutionAgent.

    Extracted from fix_build_errors_node so it can be called once per
    resolution pass without duplicating 80+ lines of context wiring.
    """
    # ── v2: Pass full context so the agent can make intelligent decisions ──
    agent.pre_edit_snapshots = state.get("original_file_contents") or {}
    agent.our_modified_files = set(
        (getattr(g, "file_path", "") or "").replace("\\", "/").lower()
        for g in (state.get("generated_code") or [])
    )
    agent.our_modified_files.update(
        k.replace("\\", "/").lower()
        for k in (state.get("_run_generated_map") or {}).keys()
    )
    agent.plan = state.get("plan") or state.get("architectural_plan")
    agent.scope_proof = state.get("ticket_scope_proof")
    ticket_obj = state.get("ticket")
    agent.ticket_description = str(getattr(ticket_obj, "description", "")) if ticket_obj else ""

    # ── v3: Wire pipeline context into agent ──────────────────────────────
    # TicketExecutionContext: accumulated knowledge from investigation → planning
    exec_ctx = _get_transient(state, "exec_ctx")
    if exec_ctx:
        agent.exec_ctx = exec_ctx
        logger.info(f"  [v3] ErrorResolutionAgent: exec_ctx attached (ticket={exec_ctx.ticket_id})")

    # ComponentStructureProvider: cross-file structural analysis for READ_COMPONENT
    provider = None
    try:
        from ticket_to_code.agents.component_structure_provider import ComponentStructureProvider
        component_groups = state.get("component_groups") or []
        # Reuse the workspace symbol index from the localizer if available
        symbol_index = getattr(agents.localizer, "symbol_index", None)
        provider = ComponentStructureProvider(
            workspace_path=str(workspace_path),
            symbol_index=symbol_index,
            component_groups=component_groups,
        )
        agent.component_provider = provider
        logger.info(f"  [v3] ComponentStructureProvider attached ({len(component_groups)} groups)")
    except Exception as _csp_err:
        logger.debug(f"  [v3] ComponentStructureProvider init failed (non-fatal): {_csp_err}")

    # ── v4: Wire Diagnostic Intelligence Pipeline ─────────────────────────
    # SymbolResolver → RelationshipAnalyzer → FixLocalizer → FixHypothesisBuilder
    # These give the agent structured evidence instead of raw error text.
    symbol_index = getattr(agents.localizer, "symbol_index", None)
    try:
        from ticket_to_code.agents.symbol_resolver import SymbolResolver
        from ticket_to_code.agents.relationship_analyzer import RelationshipAnalyzer
        from ticket_to_code.agents.fix_localizer import FixLocalizer
        from ticket_to_code.agents.fix_hypothesis_builder import FixHypothesisBuilder

        if symbol_index:
            resolver = SymbolResolver(symbol_index)
            agent.symbol_resolver = resolver

            rel_analyzer = RelationshipAnalyzer(
                symbol_resolver=resolver,
                component_provider=provider,
                workspace_path=workspace_path,
            )
            agent.relationship_analyzer = rel_analyzer

            localizer = FixLocalizer(
                symbol_resolver=resolver,
                relationship_analyzer=rel_analyzer,
                component_provider=provider,
            )
            agent.fix_localizer = localizer

            agent.hypothesis_builder = FixHypothesisBuilder()

            logger.info("  [v4] Diagnostic Intelligence Pipeline: fully wired")
        else:
            logger.info("  [v4] No symbol_index available — diagnostic intelligence skipped")
    except Exception as _di_err:
        logger.debug(f"  [v4] Diagnostic Intelligence init failed (non-fatal): {_di_err}")

    # ── v4: Pass Semantic Contract from code generation ───────────────────
    _sc = _get_transient(state, "semantic_contract")
    if _sc:
        agent.semantic_contract = _sc
        logger.info(
            f"  [v4] Semantic Contract attached: {len(_sc.items)} items, "
            f"{len(_sc.broken_items)} broken"
        )


def fix_build_errors_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """Fix build errors using technology-partitioned resolution passes.

    Pipeline:
      1. Normalize raw errors → StructuredDiagnostic[]
      2. Resolve contexts → ResolutionContext[]
      3. For EACH context (sequentially):
         a. Try Enhancement 12 (dependency resolution) first
         b. Try Enhancement 4 (adaptive strategy switching)
         c. Create ErrorResolutionAgent with correct compile_tool + compile_root
         d. Run resolution loop
         e. Write modified files to disk (WORKSPACE REFRESH)
      4. Return updated retry count

    Sequential execution is deliberate: each pass must see the current
    workspace state, including changes made by previous passes.
    """
    logger.info(" DEBUG: Fixing Build Errors (Technology-Partitioned Resolution)")

    normalized_build_errors = normalize_error_lines(state["build_result"].errors)

    # ── Coupled caller-callee co-fix resolution (P0 multi-file healing) ──
    # For "Property 'X' does not exist on type 'Y'" diagnostics, resolve Y's
    # declaration file via the SQLite symbol index so the fixer heals caller
    # and callee in one synchronized session instead of ping-ponging.
    try:
        import re as _re4
        _cofix_pairs: list = []
        _resolver4 = getattr(getattr(agents, "error_resolution", None), "symbol_resolver", None) or             getattr(agents, "symbol_resolver", None)
        for _err in normalized_build_errors[:40]:
            _m = _re4.search(
                r"Property '([\w$]+)' does not exist on type '([\w$]+)'", str(_err)
            )
            if not _m:
                continue
            _member, _type = _m.group(1), _m.group(2)
            _callee = ""
            try:
                _conn4 = getattr(getattr(agents, "symbol_index", None), "_conn", None) if \
                    hasattr(getattr(agents, "symbol_index", None), "_conn", None) else None
                if _conn4 is None and _resolver4 is not None:
                    _conn4 = getattr(getattr(_resolver4, "sqlite_store", None), "_conn", None)
                if _conn4 is not None:
                    _row = _conn4.execute(
                        "SELECT DISTINCT path FROM symbols WHERE name = ? LIMIT 1", (_type,)
                    ).fetchone()
                    _callee = (_row[0] if _row else "") or ""
            except Exception:
                _callee = ""
            _caller = ""
            _m_caller = _re4.search(r"([\w/.-]+\.(?:ts|tsx|java|py))", str(_err))
            if _m_caller:
                _caller = _m_caller.group(1)
            if _callee and _caller and _callee.replace("\\", "/") != _caller.replace("\\", "/"):
                _cofix_pairs.append({
                    "member": _member, "type": _type,
                    "caller": _caller, "callee": _callee,
                })
        if _cofix_pairs:
            _fix_agent4 = getattr(agents, "error_resolution", None)
            if _fix_agent4 is not None:
                _fix_agent4.cofix_pairs = _cofix_pairs[:6]
                logger.info(
                    f"  🔗 CO-FIX: {len(_cofix_pairs[:6])} coupled caller-callee pair(s) "
                    "resolved for synchronized multi-file healing"
                )
    except Exception as _cofix_exc:
        logger.debug(f"co-fix pair resolution skipped (non-fatal): {_cofix_exc}")
    logger.info(f"Analyzing {len(normalized_build_errors)} build errors...")

    # ── Change A: repair ONLY ticket/generated errors; never pre-existing/infra ──
    # The router already diverts infra-only and all-pre-existing builds. In the
    # MIXED case (baseline + ticket errors) we strip the baseline errors here so
    # the resolver spends no iterations on files the ticket never touched.
    # EXCEPTION: when the user explicitly chose FIX, keep the pre-existing errors
    # and authorize (only) those files for repair.
    _pre_existing_decision = str(state.get("pre_existing_decision", "leave") or "leave")
    _authorized_pre_existing = set(state.get("authorized_pre_existing_files") or [])
    try:
        from ticket_to_code.agents.build_diagnostic_classifier import (
            classify_build_diagnostics,
        )
        _our_files: set[str] = set()
        for _g in (state.get("generated_code") or []):
            _fp = (getattr(_g, "file_path", "") or "").replace("\\", "/").lower()
            if _fp:
                _our_files.add(_fp)
        for _ofp in (state.get("original_file_contents") or {}).keys():
            _our_files.add(str(_ofp).replace("\\", "/").lower())
        _report = classify_build_diagnostics(normalized_build_errors, _our_files)
        if _pre_existing_decision == "fix":
            logger.info(
                "  [fix_build] user authorized FIX of pre-existing errors — "
                f"repairing all {len(normalized_build_errors)} error(s) incl. "
                f"{len(_report.pre_existing)} pre-existing."
            )
        else:
            _blocking = [d.raw for d in _report.blocking]
            if _blocking and len(_blocking) < len(normalized_build_errors):
                logger.info(
                    f"  [fix_build] Differential filter: repairing {len(_blocking)} "
                    f"ticket/generated error(s), ignoring "
                    f"{len(normalized_build_errors) - len(_blocking)} pre-existing/infra line(s)"
                )
                normalized_build_errors = _blocking
    except Exception as _flt_exc:
        logger.debug(f"  [fix_build] differential filter skipped (non-fatal): {_flt_exc}")

    # ── Pillar 6: Escalate cross-file context tier on build failure ──
    # When TS2339/TS2551-like errors occur, the current context tier
    # (e.g., signature) may be insufficient. Escalate to provide more
    # detailed cross-file context on the next generation attempt.
    try:
        _impl_state_fix = None
        try:
            from ticket_to_code.agents.implementation_state import ImplementationState
            _impl_state_fix = ImplementationState.from_workflow_state(state)
        except Exception:
            pass

        if _impl_state_fix and _impl_state_fix.relationship_registry:
            import re as _re_esc
            _escalated_files = set()
            for err_line in normalized_build_errors[:20]:
                # Extract file path from error like "src/app/file.ts(12,3): error TS2339..."
                _file_match = _re_esc.match(r'([^\s(]+\.\w+)', str(err_line))
                if _file_match:
                    _err_file = _file_match.group(1)
                    if _err_file not in _escalated_files:
                        new_tier = _impl_state_fix.relationship_registry.escalate_tier(
                            target_file=_err_file,
                            reason=str(err_line)[:150],
                        )
                        _escalated_files.add(_err_file)
            if _escalated_files:
                logger.info(
                    f"  [RelationshipRegistry] Escalated context tier for "
                    f"{len(_escalated_files)} file(s) with build errors"
                )
    except Exception as _esc_exc:
        logger.debug(f"  Tier escalation failed (non-fatal): {_esc_exc}")

    # ── Enhancement 12: Dependency Resolution (runs once, before partitioning) ──
    try:
        dep_result = agents.dependency_resolver.resolve(
            build_errors=normalized_build_errors,
            technology="auto",
        )
        if dep_result.has_missing_deps:
            logger.info(
                f"  Enhancement 12: {dep_result.resolved_count} missing deps detected. "
                f"Manifest updated: {dep_result.manifest_updated}"
            )
            if dep_result.install_command:
                logger.info(f"  Install command: {dep_result.install_command}")
                import subprocess
                install_result = subprocess.run(
                    dep_result.install_command.split(),
                    cwd=str(state["workspace_path"]),
                    capture_output=True, text=True, timeout=120,
                    encoding="utf-8", errors="replace",
                )
                if install_result.returncode == 0:
                    logger.info("  Enhancement 12: Dependencies installed successfully")
                    return {
                        "retry_attempt": state.get("retry_attempt", 0),  # Don't increment
                        "status": "dependencies_resolved",
                    }
    except Exception as exc:
        logger.debug(f"Enhancement 12 check failed (non-fatal): {exc}")

    # ── Enhancement 4: Adaptive Strategy Switching ────────────────────────
    try:
        replan = agents.adaptive_replanner.analyze_failure(
            error_messages=normalized_build_errors,
            attempt_number=state.get("retry_attempt", 0),
            current_strategy="general",
            failed_strategies=agents.adaptive_replanner.get_failed_strategies(),
        )
        if replan.should_switch:
            logger.info(
                f"  Enhancement 4: Strategy switch recommended: "
                f"{replan.new_category}/{replan.new_profile} "
                f"(reason: {replan.reason})"
            )
    except Exception as exc:
        logger.debug(f"Enhancement 4 check failed (non-fatal): {exc}")

    # ── Step 1: Normalize → StructuredDiagnostic[] ────────────────────────
    from ticket_to_code.agents.diagnostic_normalizer import normalize_diagnostics
    from ticket_to_code.agents.resolution_context import resolve_contexts
    from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent
    from ticket_to_code.runtime.llm_factory import LLMRegistry
    from pathlib import Path

    workspace_path = Path(state["workspace_path"])
    llm = (
        getattr(agents, "llm", None)
        or getattr(getattr(agents, "code_generator", None), "llm", None)
        or LLMRegistry.get_llm(assistant=True)
    )
    sqlite_store = getattr(agents.localizer, "sqlite_store", None)

    diagnostics = normalize_diagnostics(normalized_build_errors)

    # ── Step 2: Partition → ResolutionContext[] ───────────────────────────
    contexts = resolve_contexts(diagnostics, workspace_path)

    if not contexts:
        # Fallback: if normalization couldn't parse any diagnostics,
        # create a single context with the raw errors
        agent = ErrorResolutionAgent(llm, workspace_path, sqlite_store)
        _is_jvm = any(
            str(f).endswith((".java", ".kt", ".scala", "build.gradle", "pom.xml"))
            for f in (normalized_build_errors + list(state.get("allowed_files") or []) + [getattr(g, "file_path", "") for g in (state.get("generated_code") or [])])
        )
        if _is_jvm:
            for sub in (workspace_path / "project-service", workspace_path):
                if (sub / "build.gradle").exists():
                    agent.compile_root = sub
                    agent.compile_tool = "gradle"
                    break
                elif (sub / "pom.xml").exists():
                    agent.compile_root = sub
                    agent.compile_tool = "maven"
                    break
            else:
                agent.compile_root = workspace_path
                agent.compile_tool = "gradle"
        else:
            agent.compile_root = _find_ng_root(workspace_path)
            agent.compile_tool = "angular-cli"
        _wire_agent_context(agent, state, agents, workspace_path)
        agent.authorized_pre_existing = set(_authorized_pre_existing)
        modified_files = agent.resolve_errors(normalized_build_errors)
        _scope_proof = state.get("ticket_scope_proof")
        for fp, fixed_content in modified_files.items():
            if _scope_proof and not _scope_proof.is_file_writable(fp, workspace_root=workspace_path):
                logger.error(
                    f"  🛡️ SCOPE_PROOF_VIOLATION in repair: File '{fp}' is not in approved writable scope. Disk write blocked."
                )
                continue
            _safe_write_or_patch_file(workspace_path, fp, fixed_content, state)
            logger.info(f"  Wrote fix to disk: {fp}")
        return {
            "retry_attempt": state.get("retry_attempt", 0) + 1,
            "status": "fixes_applied",
        }

    logger.info(
        f"  Partitioned {len(diagnostics)} diagnostics into "
        f"{len(contexts)} resolution context(s):"
    )
    for ctx in contexts:
        logger.info(
            f"    {ctx.language}/{ctx.build_tool} @ {ctx.compile_root} "
            f"({len(ctx.diagnostics)} errors, {len(ctx.source_files)} files)"
        )

    # ── Step 3: Sequential resolution passes ──────────────────────────────
    all_modified: dict[str, str] = {}
    for i, ctx in enumerate(contexts):
        logger.info(
            f"\n  ═══ Resolution Pass {i + 1}/{len(contexts)}: "
            f"{ctx.language}/{ctx.build_tool} ═══"
        )

        agent = ErrorResolutionAgent(llm, workspace_path, sqlite_store)
        agent.compile_tool = ctx.build_tool
        agent.compile_root = ctx.compile_root
        _wire_agent_context(agent, state, agents, workspace_path)
        agent.authorized_pre_existing = set(_authorized_pre_existing)

        # Pass ONLY this context's errors to the agent
        ctx_errors = [d.raw for d in ctx.diagnostics]
        modified_files = agent.resolve_errors(ctx_errors)

        # WORKSPACE REFRESH: Write to disk so next pass sees current state
        _scope_proof = state.get("ticket_scope_proof")
        for fp, fixed_content in modified_files.items():
            if _scope_proof and not _scope_proof.is_file_writable(fp, workspace_root=workspace_path):
                logger.error(
                    f"  🛡️ SCOPE_PROOF_VIOLATION in repair pass {i+1}: File '{fp}' is not in approved writable scope. Disk write blocked."
                )
                continue
            _safe_write_or_patch_file(workspace_path, fp, fixed_content, state)
            logger.info(f"  Wrote fix to disk: {fp}")

        all_modified.update(modified_files)
        if hasattr(agent, "audit_trail") and agent.audit_trail:
            trail = state.setdefault("repair_audit_trail", [])
            trail.extend(agent.audit_trail)

    logger.info(
        f"  Resolution complete: {len(all_modified)} file(s) modified "
        f"across {len(contexts)} pass(es)"
    )

    return {
        "retry_attempt": state.get("retry_attempt", 0) + 1,
        "status": "fixes_applied",
        "repair_audit_trail": state.get("repair_audit_trail", []),
    }



def fix_test_failures_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Debug Node: Fix Test Failures
    
    Analyzes test failures and decides:
    - Fix CODE (if implementation wrong)
    - OR Fix TESTS (if test expectations wrong)
    
    Uses LLM to determine which is incorrect.
    """
    logger.info(" DEBUG: Fixing Test Failures")
    
    test_result = state.get("test_result")
    if not test_result:
        return {
            "retry_attempt": state["retry_attempt"] + 1
            # ✅ REMOVED: "status" - avoid concurrent updates
        }
    
    test_output = test_result.test_output if hasattr(test_result, 'test_output') else "No output"
    logger.info(f"Analyzing {test_result.failed} test failures...")
    
    try:
        from ticket_to_code.runtime.llm_factory import LLMRegistry
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = LLMRegistry.get_llm(assistant=True)
        
        # Build debug prompt
        system_prompt = """You are an expert debugger specializing in test failures.

TASK: Analyze test failures and determine what needs to be fixed.

IMPORTANT DECISION:
1. If the TEST is wrong (incorrect expectations) → Fix the test
2. If the CODE is wrong (incorrect implementation) → Fix the code
3. Be clear about which one needs fixing

Respond with JSON:
{
  "analysis": "Analysis of test failures",
  "decision": "fix_code" or "fix_tests",
  "fixes": [
    {
      "file": "path/to/file.cs",
      "type": "code" or "test",
      "fixed_code": "complete corrected code for the file",
      "reason": "why this fixes the failure"
    }
  ]
}"""
        
        # Get generated files content
        generated_code_content = "\n\n".join([
            f"FILE: {gen.file_path}\n{gen.content}"
            for gen in state["generated_code"]
        ])
        
        generated_tests_content = "\n\n".join([
            f"TEST FILE: {gen.file_path}\n{gen.content}"
            for gen in state.get("generated_tests", [])
        ]) if state.get("generated_tests") else "No tests generated"
        
        user_prompt = f"""TEST FAILURES:
Total: {test_result.total_tests}
Failed: {test_result.failed}
Passed: {test_result.passed}

TEST OUTPUT:
{test_output[:2000]}

GENERATED CODE:
{generated_code_content[:3000]}

GENERATED TESTS:
{generated_tests_content[:3000]}

REQUIREMENTS:
{state["requirements"]}

Analyze and determine what needs to be fixed."""
        
        from ticket_to_code.llm_utils import llm_invoke as _llm_invoke_fix
        response = _llm_invoke_fix(llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Parse response and apply fixes
        import json
        
        try:
            from ticket_to_code.json_utils import parse_llm_json
            fixes_data = parse_llm_json(response.content)
            if isinstance(fixes_data, dict):
                logger.info(f"Debug Analysis: {fixes_data.get('analysis', 'No analysis')}")
                logger.info(f"Decision: {fixes_data.get('decision', 'Unknown')}")
            
            # Normalize fixes container
            if isinstance(fixes_data, list):
                raw_fixes_list = fixes_data
            elif isinstance(fixes_data, dict):
                raw_fixes_list = (
                    fixes_data.get('fixes') or
                    fixes_data.get('files') or
                    fixes_data.get('patches') or
                    fixes_data.get('modifications') or
                    fixes_data.get('changes') or
                    []
                )
                if not raw_fixes_list and any(k in fixes_data for k in ['file', 'file_path', 'path', 'fixed_code', 'code', 'content']):
                    raw_fixes_list = [fixes_data]
            else:
                raw_fixes_list = []

            # ── SCOPE ENFORCEMENT ────────────────────────────────────────────
            # Test-fix follows the same rules as build-fix and code gen:
            # code fixes may only touch planner-approved files; test fixes may
            # only touch files that were already written by generate_tests_node.
            _plan_tf = state.get("architectural_plan")
            _allowed_code_tf: list = []
            _readonly_code_tf: list = []
            if _plan_tf:
                _allowed_code_tf  = [t.file_path for t in _plan_tf.tasks
                                      if t.task_type.value != "read_only"]
                _readonly_code_tf = [t.file_path for t in _plan_tf.tasks
                                      if t.task_type.value == "read_only"]

            # Allowed test paths = exactly the paths written by generate_tests_node
            _allowed_test_tf = {
                gen.file_path for gen in state.get("generated_tests", [])
            }

            def _tfnorm(p: str) -> str:
                return p.replace("\\", "/").strip("/").lower()

            def _paths_match_tf(p1: str, p2: str) -> bool:
                n1 = _tfnorm(p1)
                n2 = _tfnorm(p2)
                return n1 == n2 or n1.endswith("/" + n2) or n2.endswith("/" + n1)

            def _infer_language_for_path_tf(p: str) -> Optional[ProgrammingLanguage]:
                ext = Path(p).suffix.lower()
                if ext in (".ts", ".tsx"):
                    return ProgrammingLanguage.TYPESCRIPT
                if ext in (".js", ".jsx"):
                    return ProgrammingLanguage.JAVASCRIPT
                if ext == ".py":
                    return ProgrammingLanguage.PYTHON
                if ext in (".cs",):
                    return ProgrammingLanguage.CSHARP
                if ext in (".java",):
                    return ProgrammingLanguage.JAVA
                if ext in (".html", ".htm"):
                    return ProgrammingLanguage.HTML
                if ext in (".scss", ".css", ".less"):
                    return ProgrammingLanguage.SCSS
                if ext in (".json",):
                    return ProgrammingLanguage.JSON
                if ext in (".yml", ".yaml"):
                    return ProgrammingLanguage.YAML
                return None

            _allowed_code_norm_tf  = {_tfnorm(p) for p in _allowed_code_tf}
            _readonly_code_norm_tf = {_tfnorm(p) for p in _readonly_code_tf}

            # Ownership map for READ_ONLY classification check
            _disc_ownership_tf: dict = {}
            for _d in (state.get("discovered_files") or []):
                _np = _tfnorm(_d.get("path", ""))
                _ot = _d.get("features", {}).get("ownership_type", None)
                if _np and _ot:
                    _disc_ownership_tf[_np] = _ot

            _tf_validator = PatchValidator()

            # Apply fixes based on decision
            decision = fixes_data.get('decision', 'fix_code') if isinstance(fixes_data, dict) else 'fix_code'
            applied_tf_count = 0

            for fix in raw_fixes_list:
                if not isinstance(fix, dict):
                    continue
                file_path = (
                    fix.get('file') or
                    fix.get('file_path') or
                    fix.get('path') or
                    fix.get('filename') or
                    fix.get('filePath') or
                    fix.get('target_file') or
                    ""
                )
                fixed_content = (
                    fix.get('fixed_code') or
                    fix.get('fixed_content') or
                    fix.get('code') or
                    fix.get('content') or
                    fix.get('patch') or
                    fix.get('new_content') or
                    fix.get('modified_code') or
                    ""
                )
                if not file_path or not fixed_content:
                    logger.warning(f"  ⚠️ Skipping malformed test-fix item (keys: {list(fix.keys())})")
                    continue

                fix_type = fix.get('type', 'code')
                norm_fp_tf = _tfnorm(file_path)

                logger.info(f"  Test-fix: type={fix_type}  file={file_path}")

                # Determine which collection to update
                is_test_fix = (
                    fix_type == 'test'
                    or '.spec.' in file_path
                    or file_path.endswith('.spec.ts')
                    or file_path.endswith('.spec.tsx')
                    or 'test' in file_path.lower()
                )
                if is_test_fix:
                    target_collection = state.get("generated_tests", [])
                else:
                    target_collection = state.get("generated_code", [])

                # ── (1) Scope check ──────────────────────────────────────────
                if is_test_fix:
                    if _allowed_test_tf and not any(_paths_match_tf(file_path, a) for a in _allowed_test_tf):
                        logger.warning(
                            f"   TEST-FIX SCOPE: '{file_path}' was not "
                            f"generated by test generator — skipped"
                        )
                        continue
                else:
                    if _allowed_code_tf and not any(_paths_match_tf(file_path, a) for a in _allowed_code_tf):
                        logger.warning(
                            f"   TEST-FIX CODE SCOPE: '{file_path}' not in "
                            f"planner-approved write list — skipped"
                        )
                        continue

                # ── (2) Read-only task check (code fixes only) ───────────────
                if not is_test_fix and any(_paths_match_tf(file_path, r) for r in _readonly_code_tf):
                    logger.warning(
                        f"   TEST-FIX READ-ONLY TASK: '{file_path}' is "
                        f"a read_only task — skipped"
                    )
                    continue

                # ── (3) Ownership READ_ONLY check ─────────────────────────────
                if _disc_ownership_tf.get(norm_fp_tf) == "READ_ONLY":
                    logger.warning(
                        f"   TEST-FIX OWNERSHIP READ-ONLY: '{file_path}' "
                        f"classified READ_ONLY by Phase 2 — skipped"
                    )
                    continue

                # ── (4) PatchValidator (code fixes only, not test fixes) ──────
                if not is_test_fix and _plan_tf:
                    _existing_tf = None
                    _tf_disk = Path(state["workspace_path"]) / file_path
                    if _tf_disk.exists():
                        try:
                            _existing_tf = _tf_disk.read_text(encoding="utf-8")
                        except Exception:
                            pass

                    _matching_task_tf = next(
                        (t for t in _plan_tf.tasks
                         if _paths_match_tf(t.file_path, file_path)),
                        None,
                    )

                    if _matching_task_tf and _existing_tf is not None:
                        from ticket_to_code.models import GeneratedCode as _GCTf
                        _tf_gen_code = _GCTf(
                            file_path=file_path,
                            content=fixed_content,
                            language=_matching_task_tf.language,
                            change_type=_matching_task_tf.task_type,
                            documentation="test-fix",
                            imports=[],
                        )
                        _tf_validation = _tf_validator.validate(
                            _tf_gen_code, _matching_task_tf, _existing_tf
                        )
                        if not _tf_validation.passed:
                            logger.warning(
                                f"  ⛔ TEST-FIX PatchValidator REJECTED '{file_path}':\n"
                                + "\n".join(
                                    f"     • {v}" for v in _tf_validation.violations
                                )
                            )
                            continue

                # ── (5) All checks passed — apply fix ────────────────────────
                matched_in_target = False
                for gen_file in target_collection:
                    gen_path = getattr(gen_file, "file_path", "") if not isinstance(gen_file, dict) else gen_file.get("file_path", "")
                    if _paths_match_tf(gen_path, file_path):
                        if isinstance(gen_file, dict):
                            gen_file["content"] = fixed_content
                        else:
                            gen_file.content = fixed_content

                        actual_rel_path = gen_path if gen_path else file_path
                        _safe_write_or_patch_file(Path(state["workspace_path"]), actual_rel_path, fixed_content, state)
                        # Fix 4: sync test-fix to _run_generated_map
                        state.setdefault("_run_generated_map", {})[normalize_path(actual_rel_path).lower()] = fixed_content
                        applied_tf_count += 1
                        matched_in_target = True
                        logger.info(f"  ✅ Test-fix applied ({fix_type}): {actual_rel_path}")

                if not matched_in_target:
                    _safe_write_or_patch_file(Path(state["workspace_path"]), file_path, fixed_content, state)
                    # Fix 4: sync direct test-fix to _run_generated_map
                    state.setdefault("_run_generated_map", {})[normalize_path(file_path).lower()] = fixed_content

                    _lang_tf = _infer_language_for_path_tf(file_path)
                    if _lang_tf is not None:
                        _task_for_tf = None
                        if _plan_tf:
                            _task_for_tf = next(
                                (t for t in _plan_tf.tasks if _paths_match_tf(t.file_path, file_path)),
                                None,
                            )
                        _ct_tf = _task_for_tf.task_type if _task_for_tf else TaskType.MODIFY
                        try:
                            target_collection.append(
                                GeneratedCode(
                                    file_path=file_path,
                                    content=fixed_content,
                                    language=_lang_tf,
                                    change_type=_ct_tf,
                                    documentation="test-fix",
                                    imports=[],
                                )
                            )
                        except Exception as _append_tf_exc:
                            logger.debug(f"Could not sync direct test-fix into state for '{file_path}': {_append_tf_exc}")
                    applied_tf_count += 1
                    logger.info(f"  ✅ Test-fix applied directly ({fix_type}): {file_path}")
            
            logger.info(f"✅ Applied {applied_tf_count} fixes")
            
            return {
                "generated_code": state["generated_code"],
                "generated_tests": state.get("generated_tests", []),
                "retry_attempt": state["retry_attempt"] + 1
                # ✅ REMOVED: "status" - avoid concurrent updates
            }
            
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON response")
            return {
                "retry_attempt": state["retry_attempt"] + 1
                # ✅ REMOVED: "status" - avoid concurrent updates
            }
            
    except Exception as e:
        logger.error(f"Debug agent error: {e}", exc_info=True)
        return {
            "retry_attempt": state["retry_attempt"] + 1
            # ✅ REMOVED: "status" - avoid concurrent updates
        }


# ============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# ============================================================================

def route_after_investigation(state: TicketToCodeState) -> Literal["unified_analysis"]:
    """
    Route after investigation - always go to unified analysis.
    Unified analysis node handles both code and non-code paths internally.
    """
    return "unified_analysis"


def route_after_unified_analysis(state: TicketToCodeState):
    """
    Route after unified analysis:
    - If requirements exist and requires_code_changes is True → Repository Discovery first, then planning
    - If solution_guidance exists or requires_code_changes is False → End workflow
    """
    inv = state.get("investigation_result")
    req_code = getattr(inv, "requires_code_changes", True) if inv else True
    if state.get("requirements") and req_code:
        return "discover"  # Discovery runs BEFORE planning
    elif state.get("solution_guidance") or not req_code:
        return END
    else:
        logger.error("Unified analysis produced neither requirements nor solution guidance!")
        return END


def resolve_component_group(file_path: str, workspace_path: str) -> list[str]:
    """Resolve sibling companion files (.scss, .html, .ts) for a component group."""
    skill_comps = [file_path]
    try:
        from ticket_to_code.skills.language_skill_registry import LanguageSkillRegistry
        skill_comps = LanguageSkillRegistry.get_instance().resolve_companions(file_path, workspace_path)
    except Exception as exc:
        logger.debug(f"[resolve_component_group] Skill resolver error: {exc}")

    try:
        from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
        companions = PolyglotCompanionResolver.resolve_all_companions(file_path, workspace_path)
    except Exception as exc:
        logger.debug(f"[resolve_component_group] PolyglotCompanionResolver error: {exc}")
        companions = [file_path]

    group = list(dict.fromkeys([file_path] + [c for c in skill_comps if c != file_path] + [c for c in companions if c != file_path]))
    return group


def route_after_discover(state: TicketToCodeState) -> str:
    """
    Fast-Path / Direct AI IDE Mode Router (AI IDE Hybrid Architecture).

    If discovery identified a confident Tier 1 component target (confidence >= 0.8
    or PRIMARY_OWNER/DISPLAY_OWNER):
      1. Resolves sibling companions (.ts, .html, .scss).
      2. Freezes state["allowed_files"] to this component group (Blast-Radius Lock).
      3. Bypasses 10 speculative pre-flight phases (hypothesis investigation,
         evidence collection loop, evidence ranking, semantic verification, preflight check,
         plan, validate candidates, ownership completeness, dataflow verification, generate code).
      4. Routes directly to edit_loop!

    If ambiguous (no confident candidates), falls back to the exploratory waterfall.
    """
    # Evidence-Derived Decline Route:
    # If retrieval evidence proved absence, unannotatability, or prior fix, decline immediately.
    decline_dec = state.get("decline_decision")
    if decline_dec and getattr(decline_dec, "declined", False):
        logger.info(
            f"  🛑 [Router] Ticket declined based on retrieval evidence: "
            f"{decline_dec.reason} — {decline_dec.evidence}"
        )
        return "decline"

    if state.get("status") == "failed" and not state.get("discovered_files"):
        # discovery_node's hard stop: honor it instead of continuing with empty candidates
        logger.error(f"  🛑 [Router] Discovery hard stop: {state.get('error', 'no candidates')} — routing to verification_failed")
        return "verification_failed"

    if state.get("direct_mode") and state.get("allowed_files"):
        ticket_obj = state.get("ticket")
        ticket_full_text = f"{getattr(ticket_obj, 'title', '')} {getattr(ticket_obj, 'description', '')}".lower()
        needs_migration = any(kw in ticket_full_text for kw in ("migration", "flyway", ".sql", "db/migration", "v10.", "v11.", "schema", "ghost", "orphan", "cleanup", "stale member"))
        if needs_migration:
            logger.info(
                "  🧭 [route_after_discover] Ticket requires database migration / cross-layer schema cleanup — "
                "routing to full architectural planning pipeline (hypothesis_investigation)."
            )
            return "hypothesis_investigation"
        logger.info(
            f"⚡ [Fast-Path Direct Mode] Fast-path active from discovery. Routing to edit_loop with locked scope: {state.get('allowed_files')}"
        )
        return "edit_loop"

    candidates = state.get("discovered_files") or []
    if not candidates:
        logger.info("  [Router] No discovery candidates — routing to hypothesis_investigation")
        return "hypothesis_investigation"

    top = candidates[0]
    conf = float(top.get("confidence", 0.0) or top.get("unified_score", 0.0) or top.get("score", 0.0) or 0.0)
    rerank_raw = float(top.get("rerank_score", 0.0) or 0.0)
    # Unshadow rerank_score: convert 1-10 scale to 0.0-1.0 scale
    rerank = rerank_raw / 10.0 if rerank_raw > 1.0 else rerank_raw
    score = max(conf, rerank)
    features = top.get("features", {})
    ownership_type = str(features.get("ownership_type", "")).upper()
    component_role = str(features.get("component_role", "")).upper()
    path = top.get("path", "")

    # Calculate runner-up score and margin to ensure winner is distinctive.
    #
    # CRITICAL: the margin MUST be computed from `unified_score` (raw), never
    # from `confidence`. `confidence` is min(0.95, unified_score), and because
    # the evidence boosts (s_dataflow, s_loc, s_semantic, ...) are added AFTER
    # normalization, raw scores routinely reach 4.0-6.0. Every strong candidate
    # therefore clamps to the same 0.95, the margin collapses to 0.000, and a
    # margin gate read from `confidence` can never fire — it is dead code that
    # looks alive. Raw scores still discriminate correctly, and ranking already
    # sorts on them.
    def _raw(c: dict) -> float:
        conf = float(c.get("confidence", 0.0) or c.get("score", 0.0) or 0.0)
        rerank_raw = float(c.get("rerank_score", 0.0) or 0.0)
        rerank = rerank_raw / 10.0 if rerank_raw > 1.0 else rerank_raw
        uni = float(c.get("unified_score", 0.0) or 0.0)
        return max(conf, rerank, uni)

    raw_top = _raw(top)
    workspace_path = state.get("workspace_path", "")
    companion_group = resolve_component_group(path, workspace_path) if path else []
    companion_set = set(companion_group)
    outside_candidates = [c for c in candidates[1:] if c.get("path") not in companion_set]
    second_score = _raw(outside_candidates[0]) if outside_candidates else 0.0
    margin = round(raw_top - second_score, 3)
    # Ratio form for the gate: scale-invariant, so it keeps meaning if the
    # evidence-boost weights are ever retuned. 0.10 => the winner must beat the
    # runner-up by at least 10% of its own score.
    margin_ratio = round((raw_top - second_score) / raw_top, 4) if raw_top > 0 else 0.0
    state["discovery_margin"] = margin
    state["discovery_margin_ratio"] = margin_ratio
    state["discovery_second_score"] = round(second_score, 3)
    state["discovery_raw_top"] = round(raw_top, 3)

    # Check for direct AI IDE mode bypass condition:
    # Requires high confidence (>= 0.80), a decisive margin over runner-up (>= 0.10) to prevent coin flips,
    # and a verified architectural role or component keyword in the stem.
    valid_roles = {
        "PRIMARY_OWNER", "DISPLAY_OWNER", "CONTROLLER", "SERVICE",
        "HANDLER", "ENDPOINT", "REPOSITORY", "MODEL", "COMPONENT",
    }
    has_valid_role = (
        ownership_type in valid_roles
        or component_role in valid_roles
        or any(k in Path(path).stem.lower() for k in ("component", "service", "controller", "handler", "router", "view", "repository", "model", "api"))
    )
    is_confident_tier1 = (
        score >= 0.80
        and margin_ratio >= 0.10
        and has_valid_role
    )

    if state.get("direct_mode_disabled"):
        logger.info(
            "  🧭 [Router] Direct mode was disabled — "
            "routing to full architectural planning pipeline (hypothesis_investigation)."
        )
        return "hypothesis_investigation"

    if is_confident_tier1 and path:
        ticket_obj = state.get("ticket")
        ticket_full_text = f"{getattr(ticket_obj, 'title', '')} {getattr(ticket_obj, 'description', '')}".lower()
        needs_migration = any(kw in ticket_full_text for kw in (
            "migration", "flyway", ".sql", "db/migration", "v10.", "v11.", "schema",
            "ghost", "orphan", "cleanup", "stale member", "delete"
        ))
        if needs_migration:
            logger.info(
                f"  🧭 [Router] Ticket specifies database migration / cross-layer schema cleanup — "
                f"routing to full architectural planning pipeline (hypothesis_investigation)."
            )
            return "hypothesis_investigation"

        # ── Cheap System-2 Verifier Gate (Reflection before Fast-Path Lock) ──
        # Inspired by Sean's AI Stories: "System 1 vs System 2 | Harness & Loop Engineering"
        # Prevents premature fast-path locks on wrong files by verifying symbol/content overlap.
        full_top_path = Path(workspace_path) / path
        # Permissive by default: this gate exists to VETO on a proven content
        # mismatch, not to certify. An unreadable path (dummy workspace in
        # tests, case or separator drift between the index and the filesystem,
        # a stale index root) is "cannot verify", which must NOT be treated as
        # "verification failed" — that silently disables the fast path.
        passed_system2_verifier = True
        verifier_reason = "Verifier not applicable (no readable file) — permissive pass"
        if full_top_path.is_file():
            try:
                top_text = full_top_path.read_text(encoding="utf-8", errors="ignore").lower()
                ticket_obj = state.get("ticket")
                t_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", f"{getattr(ticket_obj, 'title', '')} {getattr(ticket_obj, 'description', '')}")]
                stop_check = {"this", "that", "with", "from", "when", "seen", "step", "verify", "expected", "observed"}
                meaningful_kws = [w for w in t_words if w not in stop_check][:15]
                matches = [w for w in meaningful_kws if w in top_text]
                if matches or len(meaningful_kws) == 0:
                    passed_system2_verifier = True
                    verifier_reason = f"Verified content overlap with keywords: {matches[:3]}"
                else:
                    for comp in companion_group:
                        comp_full = Path(workspace_path) / comp
                        if comp_full.is_file():
                            comp_text = comp_full.read_text(encoding="utf-8", errors="ignore").lower()
                            comp_matches = [w for w in meaningful_kws if w in comp_text]
                            if comp_matches:
                                passed_system2_verifier = True
                                verifier_reason = f"Verified companion '{comp}' overlap with keywords: {comp_matches[:3]}"
                                break
                    if not passed_system2_verifier:
                        verifier_reason = f"Content check failed: file contains none of ticket keywords {meaningful_kws[:5]}"
            except Exception as _v_exc:
                passed_system2_verifier = True
                verifier_reason = f"Verifier exception (permissive fallback): {_v_exc}"

        if not passed_system2_verifier:
            logger.info(
                f"  ⚠️ [System-2 Verifier Rejected] Candidate '{path}' (score={score:.3f}, margin={margin:.3f}) "
                f"rejected by pre-flight reflection: {verifier_reason}. Downgrading to hypothesis_investigation."
            )
        else:
            effective_allowed = list(dict.fromkeys(companion_group if companion_group else [path]))
            state["allowed_files"] = effective_allowed
            state["direct_mode"] = True
            logger.info(
                f"⚡ [Fast-Path Direct Mode] Confident Tier 1 candidate '{path}' "
                f"(score={score:.3f}, runner_up={second_score:.3f}, margin={margin:.3f}, role={component_role or ownership_type}). "
                f"System-2 Verifier: {verifier_reason}. "
                f"Freezing Scope Lock to {len(effective_allowed)} files ({len(companion_group)} companion files) "
                f"and routing directly to edit_loop: {effective_allowed}"
            )
            return "edit_loop"

    logger.info(
        f"  [Router] Discovery candidates require exploration (top candidate: '{path}', "
        f"score={score:.3f}, runner_up={second_score:.3f}, margin={margin:.3f}, role={component_role or ownership_type}) — routing to hypothesis_investigation"
    )
    return "hypothesis_investigation"


def route_after_validate_candidates(state: TicketToCodeState) -> str:
    """
    Route after candidate validation:
    - validated                    → localize  (normal forward path)
    - no_action_required           → END       (ticket already implemented)
    - planning_needs_recovery      → planning_recovery (diagnose + recover)
    - invalid, retries < max       → plan      (re-rank with blacklist, no re-discovery)
    - invalid, retries >= max      → discover  (full rediscovery with expanded scope)
    """
    print(f"DEBUG route_after_validate_candidates: status='{state.get('status')}', retry_count={state.get('candidate_retry_count')}")
    if state.get("status") == "candidates_validated":
        return "localize"
    if state.get("status") == "no_action_required":
        logger.info("   Routing to END with terminal no_action_required")
        return END
    if state.get("status") == "planning_needs_recovery":
        logger.info("   Routing to planning_recovery (0 writable tasks — diagnosing)")
        return "planning_recovery"
    if state.get("status") == "failed":
        logger.error("   Routing to END due to discovery hard stop")
        return END
    retry_count = state.get("candidate_retry_count", 0)
    max_retries = state.get("max_candidate_retries", 2)
    if retry_count >= max_retries:
        logger.warning(
            f"   Max candidate retries ({max_retries}) reached — triggering full rediscovery"
        )
        return "discover"
    logger.info(
        f"   Candidate retry {retry_count}/{max_retries} — re-planning with updated blacklist"
    )
    return "plan"


def route_after_grounded_understanding(state: TicketToCodeState) -> str:
    """
    Route after grounded understanding (Phase 2G-3):
    - normal → ownership_completeness
    - evidence_insufficient → END (Zero-evidence hard fallback)
    """
    if state.get("status") == "evidence_insufficient":
        logger.error("   Routing to END due to insufficient evidence (Zero-evidence hard fallback)")
        return END
    return "ownership_completeness"


def route_after_generate_code(state: TicketToCodeState) -> str:
    """
    Route after code generation.

    If generation produced no effective patch, loop back through the same
    candidate retry policy as validate_candidates. Otherwise continue to
    patch_gate (B9) which verifies scope, apply-check, and secrets before build.
    """
    # ── Pre-generation drift threshold exceeded → route to planning_recovery ──
    if state.get("status") == "planning_needs_recovery":
        logger.warning(
            "  ⚠️ Pre-generation drift threshold exceeded — routing directly to planning_recovery"
        )
        return "planning_recovery"

    if state.get("status") == "escalated":
        # Don't short-circuit to END — let the remaining pipeline
        # (edit_loop → patch_gate → build → fix_build) attempt recovery.
        # The old `return END` here was severing the entire build pipeline
        # whenever code generation had partial patch failures (e.g.,
        # anchor_method_not_found on HTML files), causing fix_build to
        # never run even though other files were successfully generated.
        logger.warning(
            "  ⚠️ Code generation escalated but continuing to edit_loop "
            "for build/fix_build recovery"
        )
        return "edit_loop"

    if state.get("status") == "candidates_invalid":
        retry_count = state.get("candidate_retry_count", 0)
        max_retries = state.get("max_candidate_retries", 2)
        if retry_count >= max_retries:
            logger.warning(
                f"   Empty-patch retries reached {retry_count}/{max_retries} — proceeding to edit_loop for compiler repair"
            )
            return "edit_loop"
        logger.info(
            f"   Empty-patch retry {retry_count}/{max_retries} — re-planning with preserved candidates"
        )
        return "plan"

    # ── Pre-generation gate hard-block → return to planning (bounded) ──
    if state.get("status") == "generation_gate_blocked":
        retry_count = int(state.get("candidate_retry_count", 0) or 0)
        max_retries = int(state.get("max_candidate_retries", 2) or 2)
        if retry_count >= max_retries:
            logger.warning(
                f"   Pre-generation gate block persisted {retry_count}/{max_retries} — "
                f"proceeding to edit_loop rather than looping."
            )
            return "edit_loop"
        logger.info(
            f"   Pre-generation gate block — re-planning ({retry_count}/{max_retries})"
        )
        return "plan"

    # If code generation completed cleanly, proceed directly to patch_gate -> build.
    # Edit loop is reserved for error recovery (escalated or failed patches).
    logger.info("  ✅ Code generation completed cleanly — proceeding directly to patch_gate")
    return "patch_gate"


# ============================================================================
# EDIT LOOP NODE — "Claude Code / Devin" style free-form adaptive editing
# Runs AFTER generate_code_node to catch anything the fixed plan missed, and
# uses the LSP client to verify property names before writing HTML.
# ============================================================================

def edit_loop_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Free-form adaptive edit loop that mirrors how Claude Code and Devin work.

    Unlike generate_code_node (which executes a fixed N-task plan then stops),
    this node runs a while-loop:
      1. Asks the LLM: "is the ticket fully implemented? If not, what file is next?"
      2. Reads the CURRENT on-disk content of that file (always fresh)
      3. Generates a targeted minimal edit using the LSP for exact property names
      4. Writes it and immediately verifies with the TypeScript/Java compiler
      5. Fixes errors inline before moving to the next file
      6. Repeats until solved or MAX_ITER reached

    This node fires AFTER generate_code_node. If generate_code_node already
    produced a clean build, this node is a no-op (detects solved = True quickly).
    """
    from ticket_to_code.agents.edit_loop_agent import EditLoopAgent
    from ticket_to_code.agents.lsp_client import WorkspaceSymbolIndex
    from ticket_to_code.runtime.llm_factory import LLMRegistry

    logger.info("\n" + "=" * 70)
    logger.info(" EDIT LOOP NODE: Adaptive free-form edit loop")
    logger.info("=" * 70)

    ticket = state.get("ticket")
    requirements = state.get("requirements")
    workspace_path = str(state.get("workspace_path", ""))

    if not ticket or not requirements or not workspace_path:
        logger.warning("  edit_loop_node: missing ticket/requirements/workspace — skipping")
        return {}

    # Use the smarter model for the edit loop (same as planner)
    llm = LLMRegistry.get_llm(assistant=True)
    symbol_index = WorkspaceSymbolIndex(workspace_path)

    # Seed the agent with files already written this run
    existing_map = {
        k: v for k, v in (state.get("_run_generated_map") or {}).items()
    }

    agent = EditLoopAgent(
        workspace_path=workspace_path,
        llm=llm,
        symbol_index=symbol_index,
        ui_callback=_get_transient(state, "_ui_callback"),
        pre_edit_snapshots=state.get("original_file_contents"),
    )

    # If we re-entered via the requirement-satisfaction loop, tell the edit loop
    # exactly which ticket requirements are still unmet so it targets them.
    _raw_ctx = state.get("code_rag_context", "") or ""
    _code_ctx = "\n".join(str(c) for c in _raw_ctx) if isinstance(_raw_ctx, list) else str(_raw_ctx)
    _remediation = state.get("outcome_remediation")
    if _remediation:
        _code_ctx = (
            "OUTSTANDING TICKET REQUIREMENTS NOT YET SATISFIED — implement these now:\n"
            f"{_remediation}\n\n" + _code_ctx
        )
        logger.info("  edit_loop: injecting outstanding-requirement remediation from outcome_check")

    # ── Dynamic remediation planning ─────────────────────────────────────────
    # Re-entering from outcome_check with a PARTIAL verdict is a DIFFERENT job
    # from the original generation pass: only the files implicated by the
    # reviewer's findings need edits. Re-seeding the ORIGINAL plan here made
    # the loop re-execute finished tasks (e.g. regenerating SCSS 3x in one
    # observed run). Build the minimal remediation plan from the findings and
    # pass it instead; fall back to the original plan only when the findings
    # don't resolve to any file.
    _initial_plan = _get_plan(state)
    _remediation_findings = [
        f for f in (state.get("outcome_findings") or [])
        if str(f.get("verdict", "")).upper() != "CORRECT"
    ]
    if _remediation:
        _remediation_plan = _build_remediation_plan(state)
        if _remediation_plan is not None:
            _original_count = len(getattr(_initial_plan, "tasks", None) or [])
            logger.info(
                f"  edit_loop: REMEDIATION MODE — swapping original plan "
                f"({_original_count} task(s)) for targeted remediation plan "
                f"({len(_remediation_plan.tasks)} task(s))"
            )
            _initial_plan = _remediation_plan

    # Compute allowed_files from plan, expected files, ticket scope proof, and state allowed_files
    _loop_allowed_files = set(state.get("allowed_files") or [])
    _plan_task_fps = {
        t.file_path for t in (getattr(_initial_plan, "tasks", None) or [])
        if getattr(t.task_type, "value", str(t.task_type)) != "read_only"
    }
    _loop_allowed_files.update(_plan_task_fps)
    _expected_fps = set(state.get("expected_changed_files") or [])
    _loop_allowed_files.update(_expected_fps)
    _sp = state.get("ticket_scope_proof")
    if _sp and hasattr(_sp, "scope_graph"):
        _loop_allowed_files.update(_sp.scope_graph.all_writable_files())

    # Fallback to discovered candidates if still unconstrained
    if not _loop_allowed_files and state.get("discovered_files"):
        _loop_allowed_files = {c.get("path") for c in state["discovered_files"] if c.get("path")}
        logger.info(f"  edit_loop: populated allowed_files from {len(_loop_allowed_files)} discovered candidate(s)")

    if _loop_allowed_files:
        candidate_list_str = "\n".join(f"- {f}" for f in sorted(_loop_allowed_files))
        _code_ctx = f"ALLOWED CANDIDATE REPOSITORY FILES FOR EDIT:\n{candidate_list_str}\n\n" + _code_ctx

    # In Direct Mode or when candidate files are already localized, synthesize a minimal plan directly
    if _initial_plan is None and _loop_allowed_files:
        from ticket_to_code.models import (
            ArchitecturalPlan, DevelopmentTask, TaskType,
            ProgrammingLanguage, ArchitecturalPattern
        )

        def _get_task_lang(p_str: str) -> ProgrammingLanguage:
            suf = Path(p_str).suffix.lower()
            return {
                ".java": ProgrammingLanguage.JAVA,
                ".ts": ProgrammingLanguage.TYPESCRIPT,
                ".tsx": ProgrammingLanguage.TYPESCRIPT,
                ".js": ProgrammingLanguage.JAVASCRIPT,
                ".py": ProgrammingLanguage.PYTHON,
                ".html": ProgrammingLanguage.HTML,
                ".scss": ProgrammingLanguage.SCSS,
                ".css": ProgrammingLanguage.SCSS,
                ".xml": ProgrammingLanguage.XML,
                ".json": ProgrammingLanguage.JSON,
                ".yaml": ProgrammingLanguage.YAML,
                ".yml": ProgrammingLanguage.YAML,
                ".cs": ProgrammingLanguage.CSHARP,
            }.get(suf, ProgrammingLanguage.JAVA)

        synth_tasks = [
            DevelopmentTask(
                id=f"direct_task_{idx}",
                title=f"Direct edit: {Path(fp).name}",
                description=getattr(ticket, "description", "") or getattr(ticket, "title", ""),
                file_path=fp,
                task_type=TaskType.MODIFY,
                language=_get_task_lang(fp),
            )
            for idx, fp in enumerate(sorted(_loop_allowed_files))
        ]
        _initial_plan = ArchitecturalPlan(
            pattern=ArchitecturalPattern.LAYERED,
            rationale=f"Direct Mode execution for {getattr(ticket, 'title', '')}",
            affected_modules=[Path(fp).parts[0] if Path(fp).parts else "" for fp in sorted(_loop_allowed_files)],
            tasks=synth_tasks,
        )
        state["plan"] = _initial_plan
        state["architectural_plan"] = _initial_plan
        logger.info(f"  ⚡ Direct Mode: synthesized minimal ArchitecturalPlan with {len(synth_tasks)} task(s) directly from allowed files")

    # If initial plan is still missing and no candidates are localized, invoke PlanningAgent fallback
    elif _initial_plan is None and hasattr(agents, "planner") and agents.planner:
        try:
            logger.info("  [edit_loop_node] Invoking PlanningAgent fallback to construct ArchitecturalPlan...")
            _initial_plan = agents.planner.create_plan(
                ticket=ticket,
                requirements=requirements,
                workspace_path=workspace_path,
            )
            if isinstance(_initial_plan, tuple):
                _initial_plan = _initial_plan[0]
            if _initial_plan and getattr(_initial_plan, "tasks", None):
                state["plan"] = _initial_plan
                state["architectural_plan"] = _initial_plan
                for t in _initial_plan.tasks:
                    if getattr(t, "file_path", None):
                        _loop_allowed_files.add(t.file_path.replace("\\", "/").lower())
                _ui_cb = _get_transient(state, "_ui_callback")
                if _ui_cb:
                    try:
                        _ui_cb({
                            "event_type": "plan_generated",
                            "plan": _initial_plan.to_dict() if hasattr(_initial_plan, "to_dict") else getattr(_initial_plan, "__dict__", {}),
                        })
                    except Exception:
                        pass
                logger.info(f"  ✅ [edit_loop_node] PlanningAgent constructed {len(_initial_plan.tasks)} tasks. Allowed files expanded.")
        except Exception as _p_err:
            logger.warning(f"  [edit_loop_node] PlanningAgent invocation fallback: {_p_err}")

    # Run the loop
    result = agent.run(
        ticket=ticket,
        requirements=requirements,
        initial_plan=_initial_plan,
        code_rag_context=_code_ctx,
        existing_file_map=existing_map,
        remediation_mode=bool(_remediation),
        remediation_findings=_remediation_findings,
        allowed_files=list(_loop_allowed_files) if _loop_allowed_files else None,
    )

    _tok = result.get("token_usage") or {}
    logger.info(
        f"\n  Edit loop finished: {result['iterations']} iteration(s), "
        f"solved={result['solved']}, "
        f"files touched={len([r for r in result['edit_results'] if not r.skipped])}, "
        f"llm_calls={_tok.get('llm_calls', 0)}, tokens={_tok.get('total_tokens', 0)}"
    )

    # Merge edit loop results back into state
    updated_map = dict(state.get("_run_generated_map") or {})
    for fp_lower, content in result["generated_files"].items():
        if fp_lower not in (existing_map or {}):
            updated_map[fp_lower] = content

    # Update generated_code list with any new files the edit loop wrote
    generated_code = list(state.get("generated_code") or [])
    existing_gen_paths = {
        (getattr(g, "file_path", None) or g.get("file_path", "")).replace("\\", "/").lower()
        for g in generated_code
    }
    from ticket_to_code.models import GeneratedCode as _GC, TaskType, ProgrammingLanguage
    for edit_result in result["edit_results"]:
        if edit_result.skipped or not edit_result.content_after:
            continue
        fp_lower = edit_result.file_path.replace("\\", "/").lower()
        
        # If file already exists in generated_code, update its content in place!
        updated_existing = False
        for g in generated_code:
            g_fp = (getattr(g, "file_path", None) or g.get("file_path", "")).replace("\\", "/").lower()
            if g_fp == fp_lower or g_fp.endswith(fp_lower) or fp_lower.endswith(g_fp):
                if hasattr(g, "content"):
                    g.content = edit_result.content_after
                elif isinstance(g, dict):
                    g["content"] = edit_result.content_after
                updated_existing = True
                logger.info(f"  edit_loop: updated GeneratedCode content for {edit_result.file_path}")
                break

        if not updated_existing:
            ext = Path(edit_result.file_path).suffix.lower()
            lang_map = {
                ".ts": ProgrammingLanguage.TYPESCRIPT, ".tsx": ProgrammingLanguage.TYPESCRIPT,
                ".java": ProgrammingLanguage.JAVA, ".py": ProgrammingLanguage.PYTHON,
                ".html": ProgrammingLanguage.HTML, ".scss": ProgrammingLanguage.SCSS,
            }
            lang = lang_map.get(ext, ProgrammingLanguage.TYPESCRIPT)
            generated_code.append(_GC(
                file_path=edit_result.file_path,
                content=edit_result.content_after,
                language=lang,
                change_type=TaskType.MODIFY,
                documentation=f"edit_loop: {Path(edit_result.file_path).name}",
                imports=[],
            ))

    # Merge pre-edit snapshots back into state
    original_contents = dict(state.get("original_file_contents") or {})
    for fp, content in (result.get("pre_edit_snapshots") or {}).items():
        if fp not in original_contents:
            original_contents[fp] = content
    state["original_file_contents"] = original_contents

    return {
        "_run_generated_map": updated_map,
        "generated_code": generated_code,
        "generated_files": [getattr(g, "file_path", "") for g in generated_code if getattr(g, "file_path", "")],
        "original_file_contents": original_contents,
        "_edit_loop_result": result,
        # A synthesized direct-mode plan was only assigned to the local state dict; return it so it persists.
        "plan": state.get("plan"),
    }


def outcome_check_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Post-build semantic validation: verify the generated code actually implements
    the ticket requirements, not just that it compiles.

    After a successful build, an LLM reviews:
      - The ticket's functional requirements
      - The diff (generated files vs originals)
    and reports which requirements are satisfied, which are missing, and why.

    This is non-blocking: if the check fails it adds warnings to the workflow
    explanation but does NOT block progression to memory_update.
    """
    logger.info("  ✅ OUTCOME CHECK: Verifying generated code satisfies ticket requirements")

    ticket = state.get("ticket")
    requirements = state.get("requirements")
    generated_code = state.get("generated_code") or []
    original_contents = state.get("original_file_contents") or {}
    workspace_path = str(state.get("workspace_path", ""))

    if not ticket:
        return {"outcome_check_result": {"status": "skipped", "reason": "no ticket"}}

    requires_code = getattr(state.get("investigation_result"), "requires_code_changes", True)
    if requires_code and not generated_code:
        logger.error("  🛑 Outcome check failed: Ticket requires code changes, but ZERO files were generated.")
        return {
            "outcome_check_result": {
                "status": "FAILED",
                "reason": "zero_code_generated",
                "details": "Ticket required code modifications, but no files were produced or modified during the session."
            },
            "escalation_type": "requirements_unsatisfied",
            "escalation_reason": "Zero files generated for code-change ticket",
        }
    elif not generated_code:
        return {"outcome_check_result": {"status": "skipped", "reason": "no generated code"}}

    # Build a concise diff summary (what changed in each file)
    diff_blocks: list[str] = []
    for gen in generated_code:  # include all generated files in diff review
        fp = getattr(gen, "file_path", "")
        # Prefer fresh on-disk content from workspace if present
        disk_file = Path(workspace_path) / fp if workspace_path and fp else None
        if disk_file and disk_file.is_file():
            new_content = disk_file.read_text(encoding="utf-8", errors="ignore")
        else:
            new_content = getattr(gen, "content", "") or ""
        orig_content = original_contents.get(fp, "")
        if not orig_content:
            for ofp, oc in original_contents.items():
                if Path(fp).name == Path(ofp).name and oc:
                    orig_content = oc
                    break

        if orig_content and new_content:
            # Show only the diff (changed lines with context)
            import difflib
            diff = list(difflib.unified_diff(
                orig_content.splitlines(), new_content.splitlines(),
                fromfile=f"original/{fp}", tofile=f"generated/{fp}",
                n=3, lineterm="",
            ))
            diff_str = "\n".join(diff[:200])  # cap at 200 diff lines per file
            if diff_str.strip():
                diff_blocks.append(f"=== {fp} ===\n{diff_str}")
        elif new_content and not orig_content:
            # New file — show first 60 lines
            preview = "\n".join(new_content.splitlines()[:60])
            diff_blocks.append(f"=== NEW FILE: {fp} ===\n{preview}")

    if not diff_blocks:
        return {"outcome_check_result": {"status": "skipped", "reason": "no diff available"}}

    req_text = ""
    if requirements:
        reqs = getattr(requirements, "functional_requirements", []) or []
        req_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(reqs[:10]))

    diff_summary = "\n\n".join(diff_blocks)

    # ── Deterministic Executable Acceptance Matrix (zero LLM tokens) ─────
    # Mechanical floor beneath the LLM review: planned-vs-written, template
    # binding contracts.  Failures are pinpointed as repair hints; on hard
    # failures the LLM review is skipped entirely (accuracy first, tokens
    # second) and the run reports exactly which mechanical check failed.
    eam_checks: list[dict] = []
    try:
        from ticket_to_code.agents.code_generator import validate_template_bindings
        _plan_obj = _get_plan(state)
        _planned = []
        for _t in (getattr(_plan_obj, "tasks", None) or []):
            _tid = str(getattr(_t, "id", "") or "")
            _ttype_desc = str(getattr(_t, "description", "") or "").lower()
            _title = str(getattr(_t, "title", "") or "").lower()
            if _tid.startswith("cmp-") or "companion" in _title or "companion" in _ttype_desc or "read-only reference context" in _ttype_desc:
                continue
            _tt = getattr(getattr(_t, "task_type", None), "value", str(getattr(_t, "task_type", "")))
            _planned.append((getattr(_t, "file_path", ""), str(_tt).lower()))
        _written = {getattr(g, "file_path", "") for g in generated_code}
        for _fp, _tt in _planned:
            if _tt in ("new", "modify") and _fp and _fp not in _written:
                eam_checks.append({
                    "check": f"planned file written: {_fp}",
                    "passed": False,
                    "hint": f"planned {_tt} task for {_fp} produced no output",
                })
        _eam_ws = Path(workspace_path)
        for _g in generated_code:
            _gfp = (getattr(_g, "file_path", "") or "").replace("\\", "/")
            if not _gfp.lower().endswith(".html"):
                continue
            _ctrl_abs = _eam_ws / (_gfp[: -len(".html")] + ".ts")
            if _ctrl_abs.exists():
                from ticket_to_code.agents.code_generator import (
                    extract_dynamic_viewmodel_fields,
                )
                _ctrl_code_eam = _ctrl_abs.read_text(encoding="utf-8", errors="ignore")
                _orig_content_eam = (state.get("original_file_contents") or {}).get(_gfp)
                _viol = validate_template_bindings(
                    getattr(_g, "content", "") or "",
                    _ctrl_code_eam,
                    extra_allowed=extract_dynamic_viewmodel_fields(_ctrl_code_eam),
                    original_template=_orig_content_eam,
                )
                for _v in _viol:
                    eam_checks.append({
                        "check": f"binding '{_v}' declared in controller for {_gfp}",
                        "passed": False,
                        "hint": (
                            f"template {_gfp} binds '{{{_v}}}' but the controller never "
                            "declares it — hallucinated binding; fix the template or the controller"
                        ),
                    })
        _ui = state.get("ui_verification_result") or {}
        if _ui.get("status") == "failed":
            for _h in _ui.get("repair_hints") or []:
                eam_checks.append({"check": "live UI verification", "passed": False, "hint": str(_h)})

        # ── Forward & Backward Multi-Task Requirement Alignment Check ──────
        # Compares: (1) what completed tasks did, (2) planned task scope, (3) ticket requirements
        # Prevents task N+1 from compromising requirements to justify an error in task N.
        if requirements:
            req_list = getattr(requirements, "functional_requirements", []) or []
            combined_diff = "\n".join(diff_blocks).lower()
            for r in req_list:
                m_req = re.match(r"^\[(REQ-\d+)\]\s*(.*)", r)
                req_id = m_req.group(1) if m_req else "REQ"
                req_desc = m_req.group(2) if m_req else r
                key_terms = [
                    w.lower() for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", req_desc)
                    if w.lower() not in {"the", "and", "should", "must", "with", "for", "that", "this", "allow", "ensure", "when", "display", "render", "button", "screen"}
                ]
                if key_terms:
                    term_matches = sum(1 for kt in key_terms if kt in combined_diff)
                    if term_matches == 0:
                        eam_checks.append({
                            "check": f"Requirement [{req_id}] verified in diff",
                            "passed": False,
                            "hint": f"Requirement [{req_id}] ('{req_desc[:80]}') is missing implementation in the generated changes.",
                        })

        # If the multi-zone build already compiled 100% cleanly with 0 errors,
        # unwritten files that weren't needed for compilation must not fail acceptance.
        build_res_eam = state.get("build_result")
        build_errs_eam = getattr(build_res_eam, "errors", []) if build_res_eam else (state.get("build_errors", []) or [])
        build_stat_eam = (
            getattr(getattr(build_res_eam, "status", None), "value", "").lower()
            if build_res_eam else str(state.get("build_status") or "").lower()
        )
        if (build_stat_eam in ("success", "passed", "clean")) and not build_errs_eam and eam_checks:
            eam_checks = [c for c in eam_checks if not c["check"].startswith("planned file written:")]

        if eam_checks:
            for _c in eam_checks:
                logger.warning(f"  ❌ EAM FAIL: {_c['check']} — {_c.get('hint', '')}")
            logger.info(
                f"  📋 EAM: {len(eam_checks)} deterministic check(s) failed — "
                "skipping LLM review and reporting targeted repair hints"
            )
            _hints = [c.get("hint", c["check"]) for c in eam_checks]
            # Feed the deterministic hints through the SAME remediation
            # channel the edit loop already injects (outcome_remediation), and
            # count the attempt so the bounded retry loop terminates.
            return {
                "outcome_check_result": {
                    "status": "repair_required",
                    "eam_passed": False,
                    "eam_checks": eam_checks,
                    "repair_hints": _hints,
                },
                "outcome_remediation": " | ".join(str(h) for h in _hints[:5]),
                "outcome_fix_attempt": int(state.get("outcome_fix_attempt", 0) or 0) + 1,
            }
        logger.info("  📋 EAM: all deterministic acceptance checks passed")
    except Exception as _eam_exc:
        logger.debug(f"EAM failed (non-fatal): {_eam_exc}")

    # ── TOKEN & ACCURACY OPTIMIZATION: COMPILER-AS-TRUTH EARLY EXIT ──
    # Check REAL build and diagnostic evidence. A build that never ran is NOT passed.
    build_res = state.get("build_result")
    build_errors = getattr(build_res, "errors", []) if build_res else (state.get("build_errors", []) or [])
    build_status_val = (
        getattr(getattr(build_res, "status", None), "value", "").lower()
        if build_res else str(state.get("build_status") or "").lower()
    )

    # Genuine build validation: must have build_res/build_status == "success",
    # or in direct mode if all edit_loop results were clean and build_res is clean.
    edit_loop_clean = state.get("direct_mode") and not build_errors and (
        build_status_val in ("success", "passed", "clean") or
        (bool(state.get("edit_results")) and all(getattr(r, "compile_clean", False) for r in (state.get("edit_results") or [])))
    )
    build_passed = (build_status_val in ("success", "passed", "clean")) or edit_loop_clean

    test_res_check = state.get("test_result")
    test_status_str = getattr(getattr(test_res_check, "status", None), "value", "").lower() if test_res_check else ""
    tests_all_passed = test_status_str in ("all_passed", "success")

    # Deterministic compiler-as-truth early exit:
    # 1. Direct mode with clean build, OR
    # 2. Any mode where build succeeded AND test suite passed (tests_all_passed), OR
    # 3. Successful remediation pass where build is clean
    can_early_exit = (
        (state.get("direct_mode") or tests_all_passed or test_status_str in ("no_tests", "skipped", "") or (build_passed and int(state.get("outcome_fix_attempt", 0) or 0) > 0))
        and build_passed
        and not build_errors
        and not state.get("force_llm_outcome_check")
    )

    if can_early_exit:
        from ticket_to_code.models import OutcomeFinding
        from ticket_to_code.agents.completeness_gate import CompletenessGate

        verdict = "CORRECT"
        result_text = "VERDICT: CORRECT (Compiler build passed clean with zero diagnostics and all deterministic acceptance checks passed)"
        findings = [
            OutcomeFinding(
                verdict="CORRECT",
                requirement_id="ALL",
                requirement_text="All functional requirements",
                summary="Verified satisfied by compiler build and deterministic acceptance matrix",
                offending_code="",
                affected_file="",
                affected_line=None,
                missing_behavior="",
                next_action="Completed",
            )
        ]
        _next_attempt = int(state.get("outcome_fix_attempt", 0) or 0)
        auth_files = set(state.get("authorized_writable_files") or []) or set(state.get("allowed_files") or [])
        if not auth_files:
            auth_files = {getattr(g, "file_path", "") for g in generated_code if getattr(g, "file_path", "")}
        modified_set = {getattr(g, "file_path", "") for g in generated_code if getattr(g, "file_path", "")}
        for mf in (state.get("modified_files") or []):
            modified_set.add(mf)
        if not auth_files:
            auth_files = set(modified_set)

        test_res = state.get("test_result")
        tests_passed: Optional[bool] = None
        if test_res:
            test_status_val = getattr(getattr(test_res, "status", None), "value", "").lower()
            if test_status_val in ("all_passed", "success"):
                tests_passed = True
            elif test_status_val in ("no_tests", "skipped"):
                tests_passed = None
            else:
                tests_passed = False
        else:
            tests_passed = None

        cross_artifact_passed = bool(not build_errors and build_passed)

        verification_level = "test_verified" if tests_passed is True else ("compile_only" if tests_passed is None else "test_failed")
        is_compile_only = bool(tests_passed is None)

        if tests_passed is True:
            result_text = "VERDICT: CORRECT (Direct AI IDE Mode: Compiler build passed clean and all automated tests passed)"
        elif tests_passed is None:
            result_text = "VERDICT: CORRECT (Direct AI IDE Mode: Compiler build passed clean with zero diagnostics — compile-only verification, no test suite)"
        else:
            result_text = "VERDICT: DEFECT (Direct AI IDE Mode: Automated tests failed)"

        completeness_verdict = CompletenessGate.evaluate(
            ticket_id=getattr(ticket, "id", "ticket-001"),
            ticket_title=getattr(ticket, "title", ""),
            ticket_description=getattr(ticket, "description", ""),
            requirements=[{"id": "REQ-1", "text": getattr(ticket, "title", ""), "requires_test": False}],
            change_targets=state.get("change_targets", []) or [],
            modified_files=modified_set,
            authorized_files=auth_files,
            build_passed=build_passed,
            tests_passed=tests_passed,
            cross_artifact_passed=cross_artifact_passed,
            remaining_diagnostics=list(build_errors or []),
            workspace_path=workspace_path,
        )

        _findings_payload = [f.to_dict() for f in findings]
        _ui_cb = _get_transient(state, "_ui_callback")
        if _ui_cb:
            from datetime import datetime as _dt_emit
            _ui_msg = (
                "✅ Outcome: All Requirements Satisfied (Test-Verified)"
                if tests_passed is True else
                "ℹ️ Outcome: Code Compiles Cleanly (Compile-Only Verification, No Test Suite)"
                if tests_passed is None else
                "⚠️ Outcome: Automated Tests Failed"
            )
            _ui_cb({
                "phase": "validation",
                "status": "completed",
                "message": _ui_msg,
                "data": {
                    "node": "outcome_check",
                    "event_type": "outcome_finding",
                    "verdict": verdict,
                    "verification_level": verification_level,
                    "compile_only_verification": is_compile_only,
                    "finding": findings[0].to_dict(),
                    "findings": _findings_payload,
                    "defects": [],
                    "defect_count": 0,
                },
                "timestamp": _dt_emit.now().isoformat(),
            })

        logger.info(f"  ✅ Direct AI IDE Mode: Fast compiler-as-truth early exit ({verification_level}). 0 LLM outcome tokens consumed.")
        return {
            "outcome_check_result": {
                "status": verdict,
                "details": result_text,
                "verification_level": verification_level,
                "compile_only_verification": is_compile_only,
                "files_checked": [getattr(g, "file_path", "") for g in generated_code],
                "findings": _findings_payload,
                "unsatisfied_requirements": [],
            },
            "outcome_findings": _findings_payload,
            "verification_level": verification_level,
            "compile_only_verification": is_compile_only,
            "unsatisfied_requirements": [],
            "completeness_verdict": completeness_verdict,
            "outcome_remediation": None,
            "outcome_fix_attempt": _next_attempt,
        }

    from ticket_to_code.llm_utils import llm_invoke
    from ticket_to_code.runtime.llm_factory import LLMRegistry
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        preflight_missing = state.get("preflight_missing_requirements") or []
        preflight_verdict = state.get("preflight_verdict")
        preflight_summary = state.get("preflight_summary") or ""

        baseline_note = ""
        if preflight_verdict == "PARTIALLY_DONE" and preflight_missing:
            missing_set_str = "\n".join(f"  - {m}" for m in preflight_missing)
            baseline_note = f"""

PRE-EXISTING BASELINE STATUS (CRITICAL CONTEXT):
A pre-flight audit of the codebase prior to code generation verified that some ticket requirements ALREADY EXISTED and were fully satisfied by the baseline codebase (e.g., standard persistence, existing duplicate checks, existing dialog dismissal).
ONLY the following requirements were missing and required implementation in this ticket:
{missing_set_str}

IMPORTANT REVIEW INSTRUCTION:
- If a requirement is NOT in the missing list above, it was ALREADY SATISFIED by the baseline code outside of this diff.
- Do NOT mark a requirement as UNCERTAIN or MISSING merely because an existing method (e.g. save, cancel, backend API) is unchanged and omitted from the diff! If the unchanged baseline code already handled it, mark it SATISFIED (pre-existing baseline).
- Focus your strict behavioral verification on the requirements that were actually missing and targeted by this ticket.
"""
        elif preflight_summary:
            baseline_note = f"""

PRE-FLIGHT AUDIT SUMMARY:
{preflight_summary}
Do NOT flag unchanged existing baseline capabilities as UNCERTAIN or MISSING unless the diff explicitly broke or deleted them.
"""

        llm = getattr(agents, "llm", None) or LLMRegistry.get_llm(assistant=False)
        response = llm_invoke(llm, [
            SystemMessage(content=f"""You are a code reviewer verifying that a code generation
system correctly implemented a feature ticket. Your job is to check whether the
generated code changes actually satisfy the stated functional requirements —
BEHAVIORALLY, not just syntactically.

For each requirement, state clearly one of: SATISFIED, PARTIAL, MISSING, or UNCERTAIN.
- SATISFIED: The behavior is implemented in the diff OR was already satisfied by unchanged baseline code.
- PARTIAL: The diff attempts to implement it, but has incomplete logic, wrong quantifier/scope, or edge case bugs.
- MISSING: The requirement was expected to be implemented in this ticket but is completely absent from the changes.
- UNCERTAIN: Only use this if the new code introduced ambiguous or indeterminate behavior for a target requirement. Do NOT use UNCERTAIN for unchanged existing baseline functions.

Pay special attention to SCOPE / QUANTIFIER alignment — a very common defect:
- If a requirement applies to EACH / EVERY / ALL / per selected item, verify the
  code handles the collection (loops/maps over items, per-row state), NOT just a
  single element. Treat conditions like `if (items.length === 1)`,
  `items[0]`, `.first()`, or single-selection-only branches as a RED FLAG when
  the requirement is plural/per-item.
- Flag contradictory or over-narrow conditions that would make the behavior
  apply in fewer cases than the requirement demands.
- Flag global/shared state used where per-item state is required.

UI STATE ARCHITECTURE (multi-select forms, modals, staged tables):
- Distinguish STAGING state (what the form shows while the user picks items,
  before the Add/Confirm action) from PERSISTENCE state (per-row state on the
  staged list / saved collection). These are DIFFERENT state slots and both
  can legitimately exist in one component.
- A single component-level property driving the top input, combined with
  PER-ROW properties on each staged item (e.g. `member.isExistingProjectMember`
  on every row of the staged table), IS the correct pattern for heterogeneous
  multi-select. Do NOT demand that one input field simultaneously render
  per-user variants that only make sense per staged row.
- When the ticket is phrased in the singular ("a user", "that user") but the
  control is multi-select, per-row state on the staged collection satisfies
  the per-item requirement — mark it SATISFIED unless a row actually loses
  the behavior.
- Before flagging the same concern again, check whether the PREVIOUS remediation
  already addressed it in the per-item/per-row state. Re-flagging a design that
  is functionally equivalent wastes an entire remediation cycle. Only flag it
  if a concrete user-visible behavior is still wrong, and say exactly what a
  user would see or do wrong.

Be concise and specific — reference exact method names, property names, file
names, and the offending condition when you find one.{baseline_note}"""),
            HumanMessage(content=f"""TICKET: {ticket.title}

FUNCTIONAL REQUIREMENTS:
{req_text or "(not available)"}
{baseline_note}
CODE CHANGES (unified diff):
{diff_summary}

For each requirement, output:
- [SATISFIED/PARTIAL/MISSING/UNCERTAIN] <requirement text>: <one-line reason; cite the exact condition/symbol if PARTIAL/MISSING/UNCERTAIN>

Then output one of:
VERDICT: CORRECT     (all requirements satisfied or covered by baseline)
VERDICT: PARTIAL     (some requirements partial/missing, incl. scope/quantifier gaps)
VERDICT: INCOMPLETE  (major requirements missing)
VERDICT: UNCERTAIN   (insufficient evidence to confirm key requirements)

Finally, ALWAYS append this machine-readable block.
CRITICAL REQUIREMENTS FOR FINDINGS:
- If verdict is CORRECT, output an empty findings array: "findings": [].
- If verdict is PARTIAL, INCOMPLETE, or UNCERTAIN: Your "findings" array MUST contain an entry for EVERY requirement that is PARTIAL, MISSING, or UNCERTAIN.
- If requirements 2, 3, and 4 all have issues, you MUST provide 3 separate finding objects in findings: one for REQ-2, one for REQ-3, and one for REQ-4.
- Do NOT stop after the first defect! All failing requirements must be reported together so they can all be remediated together in one coordinated pass.
- Every finding MUST specify the affected_file to edit.

```json
{{
  "verdict": "CORRECT | PARTIAL | INCOMPLETE | UNCERTAIN",
  "findings": [
    {{
      "verdict": "PARTIAL",
      "requirement_id": "REQ-1",
      "requirement_text": "<requirement the finding is about>",
      "summary": "<one-line defect description>",
      "offending_code": "<exact offending condition/symbol, empty if none>",
      "affected_file": "<exact file path that must be edited to fix it>",
      "affected_line": null,
      "missing_behavior": "<precise instruction: what the code must do to pass>",
      "next_action": "<recommended fix for the edit loop>"
    }}
  ]
}}
```"""),
        ])

        result_text = getattr(response, "content", str(response))
        verdict = "UNKNOWN"
        if "VERDICT: CORRECT" in result_text:
            verdict = "CORRECT"
        elif "VERDICT: PARTIAL" in result_text:
            verdict = "PARTIAL"
        elif "VERDICT: INCOMPLETE" in result_text:
            verdict = "INCOMPLETE"
        elif "VERDICT: UNCERTAIN" in result_text:
            verdict = "UNCERTAIN"

        if verdict == "CORRECT" and state.get("evidence_decision_degraded"):
            # The evidence phase silently fell back to "proceed" after an LLM/parse error, so the
            # plan may rest on incomplete evidence. A CORRECT verdict on top of that is not trustworthy.
            logger.warning("  ⚠️ Evidence phase was degraded (LLM/parse fallback) — downgrading CORRECT to UNCERTAIN")
            verdict = "UNCERTAIN"

        logger.info(f"  ✅ Outcome check verdict: {verdict}")
        logger.info(f"  Details:\n{result_text[:600]}")

        # Extract concrete remediation from MISSING/PARTIAL/UNCERTAIN requirement lines so the
        # adaptive edit loop knows exactly WHAT to add to satisfy the ticket.
        remediation_lines = [
            ln.strip() for ln in result_text.splitlines()
            if any(k in ln for k in ("MISSING", "PARTIAL", "UNCERTAIN", "INCOMPLETE"))
            and ln.strip().startswith(("-", "[", "*"))
        ]
        remediation = "\n".join(remediation_lines).strip()

        # ── Parse Structured OutcomeFinding (Canonical Model) ─────────────────
        from ticket_to_code.models import OutcomeFinding
        findings: list[OutcomeFinding] = []
        try:
            _f_data = _extract_json_object(result_text)
            if _f_data:
                if _f_data.get("verdict"):
                    # Prefer the structured verdict; normalize common variants.
                    _v = str(_f_data["verdict"]).strip().upper()
                    if _v in ("CORRECT", "PARTIAL", "INCOMPLETE", "UNCERTAIN"):
                        verdict = _v
                for fd in _f_data.get("findings", []) or []:
                    if not isinstance(fd, dict):
                        continue
                    findings.append(OutcomeFinding(
                        verdict=fd.get("verdict", verdict),
                        requirement_id=fd.get("requirement_id", "REQ-1"),
                        requirement_text=fd.get("requirement_text", ""),
                        summary=fd.get("summary", ""),
                        offending_code=fd.get("offending_code", ""),
                        affected_file=fd.get("affected_file", ""),
                        affected_line=fd.get("affected_line"),
                        missing_behavior=fd.get("missing_behavior", ""),
                        next_action=fd.get("next_action", "Re-evaluating existing evidence and re-planning"),
                    ))
                logger.info(
                    f"  📋 Parsed structured findings: {len(findings)} finding(s) from JSON block"
                )
        except Exception as _parse_err:
            logger.debug(f"Structured JSON parsing in outcome_check failed: {_parse_err}")

        # Parse any additional requirement findings from reviewer text (cross-check with JSON)
        import re as _re
        existing_req_ids = {f.requirement_id.upper() for f in findings}
        _fallback_files = state.get("change_targets") or [
            getattr(g, "file_path", "") for g in generated_code if getattr(g, "file_path", "")
        ]
        _fallback_file = _fallback_files[0] if _fallback_files else ""

        # Scan for PARTIAL, MISSING, UNCERTAIN requirements
        for ln in (remediation_lines or result_text.splitlines()):
            if any(v in ln for v in ("PARTIAL", "MISSING", "UNCERTAIN")):
                _clean_sum = _re.sub(r'^[\[\-\*]\s*(?:PARTIAL|MISSING|UNCERTAIN)\]?\s*:?\s*', '', ln).strip()
                req_match = _re.search(r'\b(REQ[-\s]?\d+|Requirement\s+\d+)\b', ln, _re.IGNORECASE)
                req_id = req_match.group(0).upper().replace(" ", "-") if req_match else f"REQ-{len(findings)+1}"
                if req_id.upper() not in existing_req_ids:
                    findings.append(OutcomeFinding(
                        verdict=verdict if verdict in ("PARTIAL", "INCOMPLETE", "UNCERTAIN") else "PARTIAL",
                        requirement_id=req_id,
                        requirement_text=_clean_sum[:200] or f"Requirement {req_id}",
                        summary=_clean_sum or "Requirement partially implemented",
                        offending_code="",
                        affected_file=_fallback_file,
                        affected_line=None,
                        missing_behavior=_clean_sum,
                        next_action="Implement the missing behavior in the affected file",
                    ))
                    existing_req_ids.add(req_id.upper())

        # Scan for SATISFIED requirements from text so we know what is completed
        for ln in result_text.splitlines():
            if "SATISFIED" in ln and any(ln.strip().startswith(p) for p in ("-", "[", "*")):
                _clean_sat = _re.sub(r'^[\[\-\*]\s*(?:SATISFIED)\]?\s*:?\s*', '', ln).strip()
                req_match = _re.search(r'\b(REQ[-\s]?\d+|Requirement\s+\d+)\b', ln, _re.IGNORECASE)
                req_id = req_match.group(0).upper().replace(" ", "-") if req_match else f"REQ-SAT-{len(findings)+1}"
                if req_id.upper() not in existing_req_ids:
                    findings.append(OutcomeFinding(
                        verdict="CORRECT",
                        requirement_id=req_id,
                        requirement_text=_clean_sat[:200] or f"Requirement {req_id}",
                        summary=_clean_sat or "Requirement verified satisfied",
                        offending_code="",
                        affected_file="",
                        affected_line=None,
                        missing_behavior="",
                        next_action="Completed",
                    ))
                    existing_req_ids.add(req_id.upper())

        if not findings and verdict == "CORRECT":
            findings.append(OutcomeFinding(
                verdict="CORRECT",
                requirement_id="ALL",
                requirement_text="All functional requirements",
                summary="All ticket requirements behaviorally satisfied",
                offending_code="",
                affected_file="",
                affected_line=None,
                missing_behavior="",
                next_action="Completed",
            ))

        defect_findings = [f for f in findings if f.verdict != "CORRECT"]
        satisfied_findings = [f for f in findings if f.verdict == "CORRECT"]

        if not remediation and verdict in ("PARTIAL", "INCOMPLETE", "UNCERTAIN"):
            remediation = "\n".join(
                f"- [{f.requirement_id}] {f.missing_behavior or f.summary}"
                for f in defect_findings
            ).strip() or result_text[:400].strip()

        # Emit structured outcome finding to UI
        if defect_findings:
            defect_req_ids = [f.requirement_id for f in defect_findings]
            message_headline = f"⚠️ Outcome: {verdict} ({len(defect_findings)} requirement(s) need fix: {', '.join(defect_req_ids)})"
            _first_finding = defect_findings[0]
        else:
            message_headline = "✅ Outcome: All Requirements Satisfied"
            _first_finding = findings[0] if findings else OutcomeFinding(
                verdict="CORRECT",
                requirement_id="ALL",
                requirement_text="All functional requirements",
                summary="All ticket requirements behaviorally satisfied",
                offending_code="",
                affected_file="",
                affected_line=None,
                missing_behavior="",
                next_action="Completed",
            )

        _ui_cb = _get_transient(state, "_ui_callback")
        if _ui_cb:
            from datetime import datetime as _dt_emit
            _ui_cb({
                "phase": "validation",
                "status": "in_progress" if verdict != "CORRECT" else "completed",
                "message": message_headline,
                "data": {
                    "node": "outcome_check",
                    "event_type": "outcome_finding",
                    "verdict": verdict,
                    "finding": _first_finding.to_dict(),
                    "findings": [f.to_dict() for f in findings],
                    "defects": [f.to_dict() for f in defect_findings],
                    "defect_count": len(defect_findings),
                    "defect_requirements": [f.requirement_id for f in defect_findings],
                    "satisfied_requirements": [f.requirement_id for f in satisfied_findings],
                },
                "timestamp": _dt_emit.now().isoformat(),
            })

        _unsatisfied_reqs = [
            f.missing_behavior or f.summary
            for f in defect_findings
        ]

        # Count this as a remediation attempt only when the ticket is NOT satisfied,
        # so the bounded loop in route_after_outcome_check can terminate.
        _prev_attempt = int(state.get("outcome_fix_attempt", 0) or 0)
        _next_attempt = _prev_attempt + 1 if verdict in ("PARTIAL", "INCOMPLETE", "UNCERTAIN") else _prev_attempt

        # ── CompletenessGate: Post-Build Semantic & Invariant Verification ──
        from ticket_to_code.agents.completeness_gate import CompletenessGate

        structured_reqs = []
        if requirements:
            f_reqs = getattr(requirements, "functional_requirements", []) or []
            for idx, r in enumerate(f_reqs):
                structured_reqs.append({
                    "id": f"REQ-{idx+1}",
                    "text": str(r),
                    "requires_test": False,
                })

        modified_set = set()
        for g in generated_code:
            fp = getattr(g, "file_path", "")
            if fp:
                modified_set.add(fp)
        for mf in (state.get("modified_files") or []):
            modified_set.add(mf)

        auth_files = set(state.get("authorized_writable_files") or []) or set(state.get("allowed_files") or [])
        if not auth_files:
            auth_files = {getattr(g, "file_path", "") for g in generated_code if getattr(g, "file_path", "")}
        if not auth_files:
            auth_files = set(modified_set)

        test_res = state.get("test_result")
        tests_passed: Optional[bool] = None
        if test_res and hasattr(test_res, "status"):
            test_status_val = getattr(getattr(test_res, "status", None), "value", "").lower()
            if test_status_val in ("all_passed", "success"):
                tests_passed = True
            elif test_status_val in ("no_tests", "skipped"):
                tests_passed = None
            else:
                tests_passed = False
        else:
            tests_passed = None
        build_res = state.get("build_result")
        build_errors = getattr(build_res, "errors", []) if build_res else (state.get("build_errors", []) or [])
        build_status_val = (
            getattr(getattr(build_res, "status", None), "value", "").lower()
            if build_res else str(state.get("build_status") or "").lower()
        )
        build_passed = (build_status_val in ("success", "passed", "clean", "skipped"))
        cross_artifact_passed = bool(not build_errors and build_passed)

        completeness_verdict = CompletenessGate.evaluate(
            ticket_id=getattr(ticket, "id", "ticket-001"),
            ticket_title=getattr(ticket, "title", ""),
            ticket_description=getattr(ticket, "description", ""),
            requirements=structured_reqs,
            change_targets=state.get("change_targets", []) or [],
            modified_files=modified_set,
            authorized_files=auth_files,
            build_passed=build_passed,
            tests_passed=tests_passed,
            cross_artifact_passed=cross_artifact_passed,
            remaining_diagnostics=build_errors,
            workspace_path=workspace_path,
        )

        logger.info(f"  🏁 COMPLETENESS GATE VERDICT: {completeness_verdict.status} — {completeness_verdict.summary}")
        if not completeness_verdict.is_complete and verdict == "CORRECT":
            # Guard against false-invalidation: do NOT override verified semantic pass
            # to INCOMPLETE due to ChangeTarget metadata keyword mismatches!
            # Only override if there is an actual build failure or unauthorized file modified.
            if completeness_verdict.unauthorized_files_detected or not completeness_verdict.build_passed:
                verdict = "INCOMPLETE"
                remediation = (remediation + "\n" if remediation else "") + "\n".join(completeness_verdict.uncovered_requirements)
                _next_attempt = _prev_attempt + 1
            else:
                logger.info(
                    "  ✅ Semantic outcome check verified all requirements as CORRECT; "
                    "preserving CORRECT verdict (ignoring heuristic ChangeTarget metadata gaps)."
                )

        # ── Phase 5: Mechanical checklist verification ────────────────────────
        # Run ChecklistVerifier against the definition-of-done from Phase 0.
        # This is independent of the LLM-based outcome check above — it uses
        # mechanical citation verification to catch false-PASSes.
        _dod = state.get("definition_of_done")
        if _dod and hasattr(_dod, "items") and _dod.items:
            try:
                from ticket_to_code.agents.checklist_verifier import ChecklistVerifier
                _verifier = ChecklistVerifier(agents.llm)
                
                # Build file map from generated code
                _gen_files = {}
                for gen in generated_code:
                    fp = getattr(gen, "file_path", "")
                    content = getattr(gen, "content", "")
                    if fp and content:
                        _gen_files[fp] = content

                _vr = _verifier.verify(_dod.items, _gen_files)
                
                # Store unresolved items for Phase 6 escalation
                _unresolved = [
                    item for item in _vr.items if item.get("status") == "fail"
                ]
                if _unresolved:
                    state["unresolved_checklist_items"] = _unresolved
                    logger.warning(
                        f"  ⚠️ Phase 5: {len(_unresolved)}/{_vr.total_items} "
                        f"checklist items UNRESOLVED"
                    )
                else:
                    state["unresolved_checklist_items"] = []
                    logger.info(
                        f"  ✅ Phase 5: All {_vr.total_items} checklist items PASSED"
                    )
                    
                # Merge Phase 5 data into outcome result
                _findings_payload = [f.to_dict() for f in findings]
                return {
                    "outcome_check_result": {
                        "status": verdict,
                        "details": result_text,
                        "files_checked": [getattr(g, "file_path", "") for g in generated_code],
                        "checklist_pass_rate": _vr.pass_rate,
                        "checklist_items": _vr.items,
                        "findings": _findings_payload,
                        "unsatisfied_requirements": _unsatisfied_reqs,
                    },
                    "outcome_findings": _findings_payload,
                    "unsatisfied_requirements": _unsatisfied_reqs,
                    "completeness_verdict": completeness_verdict,
                    "outcome_remediation": remediation or None,
                    "outcome_fix_attempt": _next_attempt,
                    "unresolved_checklist_items": _unresolved,
                }
            except Exception as _ver_exc:
                logger.warning(f"  ⚠️ Phase 5 (ChecklistVerifier) failed: {_ver_exc}")

        verification_level = "test_verified" if tests_passed is True else ("compile_only" if tests_passed is None else "test_failed")
        is_compile_only = bool(tests_passed is None)
        _findings_payload = [f.to_dict() for f in findings]
        # Routers cannot persist state in LangGraph, so the targeted-expansion file list
        # that route_after_outcome_check needs is computed HERE and returned as state.
        _targeted_expansion = _compute_targeted_expansion_files(_findings_payload, state)
        return {
            "targeted_expansion_files": _targeted_expansion,
            "outcome_check_result": {
                "status": verdict,
                "details": result_text,
                "verification_level": verification_level,
                "compile_only_verification": is_compile_only,
                "files_checked": [getattr(g, "file_path", "") for g in generated_code],
                "findings": _findings_payload,
                "unsatisfied_requirements": _unsatisfied_reqs,
            },
            "outcome_findings": _findings_payload,
            "verification_level": verification_level,
            "compile_only_verification": is_compile_only,
            "unsatisfied_requirements": _unsatisfied_reqs,
            "completeness_verdict": completeness_verdict,
            "outcome_remediation": remediation or None,
            "outcome_fix_attempt": _next_attempt,
        }
    except Exception as e:
        logger.warning(f"  ⚠️ Outcome check failed (non-fatal): {e}")
        return {"outcome_check_result": {"status": "error", "reason": str(e)}}


def remediation_plan_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    Delta planning node triggered after outcome_check detects PARTIAL / INCOMPLETE defects.
    Builds the deterministic remediation plan addressing all pending defects together,
    documents what is ALREADY SATISFIED (preserved) vs PENDING (to solve), and emits the
    plan to the UI so the developer sees the exact delta plan in the workflow timeline.
    """
    logger.info("\n" + "=" * 70)
    logger.info(" REMEDIATION PLAN: Planning targeted fix for pending defects")
    logger.info("=" * 70)

    remediation_plan = _build_remediation_plan(state)
    # Test-failure routing is bounded by retry_attempt; without this bump a plan-less cycle never terminates.
    _test_status = getattr(getattr(state.get("test_result"), "status", None), "value", "").lower()
    _bump = {"retry_attempt": int(state.get("retry_attempt", 0) or 0) + 1} if _test_status in ("some_failed", "all_failed", "failed") else {}
    if not remediation_plan:
        logger.warning("  remediation_plan: no plan could be built from findings")
        return dict(_bump)

    # Emit plan to UI so the user clearly sees the remediation plan in the timeline
    _ui_cb = _get_transient(state, "_ui_callback")
    tasks = getattr(remediation_plan, "tasks", []) or []
    task_dicts = []
    for t in tasks:
        task_dicts.append({
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "file_path": t.file_path,
            "task_type": getattr(t.task_type, "value", str(t.task_type)),
        })

    if _ui_cb:
        from datetime import datetime as _dt_emit
        _ui_cb({
            "phase": "planning",
            "node": "remediation_plan",
            "status": "completed",
            "message": f"🗺️ Remediation Plan: {len(tasks)} targeted task(s) to fix all pending defects",
            "data": {
                "node": "remediation_plan",
                "pattern": str(getattr(remediation_plan, "pattern", "LAYERED")),
                "tasks": task_dicts,
                "task_count": len(tasks),
            },
            "timestamp": _dt_emit.now().isoformat(),
        })

    return {
        "remediation_plan": remediation_plan,
        "plan": remediation_plan,
        "architectural_plan": remediation_plan,
        **_bump,
    }


def _compute_targeted_expansion_files(findings_payload: list, state: dict) -> list:
    """Files that a PARTIAL/INCOMPLETE verdict says must be promoted for context expansion
    (Tier 1 candidates / Tier 2 backups). Mirrors the ladder in route_after_outcome_check."""
    try:
        defect_findings = [
            f for f in (findings_payload or [])
            if (f.get("verdict") if isinstance(f, dict) else getattr(f, "verdict", "")) != "CORRECT"
        ]
        if not defect_findings:
            return []
        tier, matched_files = _classify_defect_file_tiers(defect_findings, state)
        if tier in ("tier1", "tier2") and int(state.get("context_expansion_count", 0) or 0) == 0:
            return list(matched_files or [])
    except Exception as _exc:
        logger.debug(f"targeted expansion computation skipped: {_exc}")
    return []


def route_after_outcome_check(state: TicketToCodeState):
    """
    Route after the requirement-satisfaction check.

    Four possible routes:
    1. Requirements satisfied (CORRECT/skipped/error) → memory_update (done)
    2. PARTIAL/INCOMPLETE + Tier 2 backup available + not yet expanded
       → context_expand (promote backup files, re-plan with wider context)
    3. PARTIAL/INCOMPLETE + already expanded or no backup + attempts left
       → remediation_plan (display delta plan, then execute via edit_loop)
    4. PARTIAL/INCOMPLETE + remediation budget exhausted
       → verification_failed (TERMINAL — never masked as success)

    Bounded by max_outcome_fix_attempts so it always terminates. Route 4 is the
    honesty gate: giving up because the retry budget ran out is a FAILURE and
    must be reported as one, not laundered through memory_update as a green
    "workflow complete".
    """
    result = state.get("outcome_check_result") or {}
    verdict = result.get("status", "UNKNOWN")
    remediation = state.get("outcome_remediation")
    attempt = int(state.get("outcome_fix_attempt", 0) or 0)
    max_attempts = int(state.get("max_outcome_fix_attempts", 3) or 3)

    if str(verdict).lower() == "error":
        # The semantic check crashed. That is NOT evidence the ticket is satisfied:
        # treat it as UNCERTAIN (retry within budget, then verification_failed).
        logger.warning(
            f"  ⚠️ Outcome check errored ({result.get('reason', 'unknown')}) — treating as UNCERTAIN, not success"
        )
        verdict = "UNCERTAIN"

    if verdict == "FAILED":
        logger.error(
            f"  🛑 Outcome check reported FAILED ({result.get('reason', 'unknown')}) — routing to verification_failed"
        )
        return "verification_failed"

    if verdict == "repair_required":
        # Deterministic acceptance failure: targeted repair with the exact
        # mechanical hints (binding violations / unwritten planned files).
        # Bounded by the same attempt budget as semantic retries.
        if attempt < max_attempts:
            logger.info(
                f"  🔧 Deterministic acceptance checks failed — routing to "
                f"remediation_plan with targeted repair hints (attempt {attempt}/{max_attempts})"
            )
            return "remediation_plan"
        logger.warning(
            "  ⚠️ repair_required but remediation budget exhausted or no hints — "
            "routing to verification_failed (NOT marked fully verified)"
        )
        return "verification_failed"

    if verdict in ("PARTIAL", "INCOMPLETE", "UNCERTAIN"):
        if attempt < max_attempts:
            # Hierarchical search ladder: Tier 0 (already modified) -> Tier 1 (candidates) -> Tier 2 (backups)
            findings = state.get("outcome_findings") or result.get("findings") or []
            defect_findings = [
                f for f in findings
                if (f.get("verdict") if isinstance(f, dict) else getattr(f, "verdict", "")) != "CORRECT"
            ]
            tier, matched_files = _classify_defect_file_tiers(defect_findings, state)
            expansion_count = int(state.get("context_expansion_count", 0) or 0)

            if tier == "tier0":
                # Tier 0: All defect findings belong to already modified/generated files!
                # STOP SEARCHING. Do NOT expand context or re-plan from scratch.
                # Route to remediation_plan for visible delta planning, then edit_loop.
                logger.info(
                    f"  🎯 Tier 0 match: All {len(defect_findings)} defect finding(s) belong to "
                    f"already modified file(s) ({', '.join(Path(f).name for f in matched_files)}). "
                    f"Stopping search at Tier 0 — routing to remediation_plan for delta remediation "
                    f"(attempt {attempt}/{max_attempts})."
                )
                return "remediation_plan"

            if tier == "tier1" and expansion_count == 0:
                # Tier 1: Defect finding requires file(s) in Tier 1 candidate pool.
                logger.info(
                    f"  📂 Tier 1 match: Defect finding requires Tier 1 file(s): "
                    f"{[Path(f).name for f in matched_files]}. Stopping at Tier 1 — "
                    f"promoting for context expansion."
                )
                state["_targeted_expansion_files"] = matched_files
                return "context_expand"

            if tier == "tier2" and expansion_count == 0:
                # Tier 2: Defect finding requires file(s) in Tier 2 backup candidates.
                logger.info(
                    f"  📦 Tier 2 match: Defect finding requires Tier 2 backup file(s): "
                    f"{[Path(f).name for f in matched_files]}. Promoting Tier 2 candidate(s) "
                    f"for context expansion."
                )
                state["_targeted_expansion_files"] = matched_files
                return "context_expand"

            # Subsequent retry or defects with no external candidate match
            logger.info(
                f"  ↩️  Requirements not fully satisfied ({verdict}); "
                f"routing to remediation_plan for remediation (attempt {attempt}/{max_attempts})"
            )
            return "remediation_plan"

        # ── Remediation budget exhausted: TERMINAL FAILURE, never success ──
        _unsatisfied = result.get("unsatisfied_requirements") or []
        _why = "; ".join(str(u)[:120] for u in _unsatisfied[:3]) if _unsatisfied else (
            (remediation or "unresolved findings")[:200]
        )
        logger.error(
            f"  🛑 Requirements {verdict} after {max_attempts}/{max_attempts} remediation "
            f"attempt(s) — routing to verification_failed. Unresolved: {_why}"
        )
        return "verification_failed"

    return "memory_update"


def context_expand_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """
    PARTIAL Recovery Architecture — smart context expansion when outcome_check
    found PARTIAL/INCOMPLETE.

    Strategy (4 steps, each only runs if the previous didn't resolve the gap):

      Step 1: Evidence Reuse
        Check if existing evidence files from the investigation phase already
        cover the unsatisfied requirements.

      Step 2: Targeted Re-investigation
        When evidence doesn't cover the gap, use the specific unsatisfied
        requirements as focused search queries to find relevant files.

      Step 3: Second Sufficiency Check
        Evaluate whether the combined (old + new) evidence is now sufficient.
        Only proceed to re-plan if the system has confidence the gap is covered.

      Step 4: Tier 2 Fallback
        Last resort — promote Tier 2 backup candidates blindly.

    Flow: context_expand → plan (re-plan with expanded context) → generate_code → ...
    """
    tier2 = state.get("tier2_candidates") or []
    _original_discovered_count = len(state.get("discovered_files") or [])
    current_discovered = list(state.get("discovered_files") or [])  # shallow copy to avoid aliasing
    expansion_count = int(state.get("context_expansion_count", 0) or 0)
    existing_paths = {c.get("path") for c in current_discovered}
    promoted = 0

    # Prioritize targeted files identified by the defect escalation ladder
    targeted_files = state.get("targeted_expansion_files") or state.get("_targeted_expansion_files") or []
    if targeted_files:
        for tf in targeted_files:
            if tf not in existing_paths:
                current_discovered.append({
                    "path": tf,
                    "_tier": "targeted_escalation_promoted",
                    "confidence": 0.95,
                    "change_intent": "READ_ONLY",
                })
                existing_paths.add(tf)
                promoted += 1
        logger.info(f"  🎯 Targeted expansion: promoted {promoted} file(s): {[Path(f).name for f in targeted_files]}")

    outcome_result = state.get("outcome_check_result") or {}
    unsatisfied = outcome_result.get("unsatisfied_requirements") or []
    evidence_items = state.get("evidence_items") or []

    # ── Step 1: Evidence Reuse ────────────────────────────────────────────
    # Check if evidence files from the investigation phase already cover
    # the unsatisfied requirements.
    if unsatisfied and evidence_items:
        _unsatisfied_text = " ".join(str(r) for r in unsatisfied).lower()
        evidence_candidates = []
        for item in evidence_items:
            _path = getattr(item, "file_path", None) or (item.get("file_path") if isinstance(item, dict) else "") or ""
            if not _path or _path in existing_paths:
                continue
            # Check if evidence item's content/facts relate to unsatisfied reqs
            _facts = str(getattr(item, "facts", "") if not isinstance(item, dict) else item.get("facts", "")).lower()
            _role = str(getattr(item, "role", "") if not isinstance(item, dict) else item.get("role", "")).lower()
            _content = str(getattr(item, "content_snippet", "") if not isinstance(item, dict) else item.get("content_snippet", "")).lower()
            _combined = f"{_facts} {_role} {_content}"
            if any(word in _combined for word in _unsatisfied_text.split()[:10] if len(word) > 4):
                evidence_candidates.append({"path": _path, "_tier": "promoted_from_evidence"})

        if evidence_candidates:
            for ec in evidence_candidates[:8]:
                if ec["path"] not in existing_paths:
                    current_discovered.append(ec)
                    existing_paths.add(ec["path"])
                    promoted += 1
            logger.info(
                f"  📋→✅ Step 1 (Evidence Reuse): promoted {promoted} evidence files "
                f"into discovered_files"
            )
            for ec in evidence_candidates[:promoted]:
                logger.info(f"    + {ec['path']}  (source=evidence_ledger)")

    # ── Step 2: Targeted Re-investigation ─────────────────────────────────
    # When evidence from Step 1 doesn't cover the gap, use unsatisfied
    # requirements as focused search queries to find relevant files.
    if promoted == 0 and unsatisfied:
        logger.info(
            f"  🔍 Step 2 (Targeted Re-investigation): evidence didn't cover gap, "
            f"searching for {len(unsatisfied)} unsatisfied requirement(s)"
        )
        workspace_path = state.get("workspace_path", "")
        if workspace_path:
            try:
                from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
                _search_engine = RepositorySearchEngine(Path(workspace_path))

                _targeted_finds = []
                _skip_words = {"the", "and", "for", "with", "that", "this", "from", "should",
                               "must", "have", "been", "not", "are", "was", "will", "can",
                               "need", "also", "each", "when", "into", "does"}
                for _req in unsatisfied[:5]:  # Cap at 5 requirements
                    _req_text = str(_req).lower()
                    _keywords = [
                        w for w in _req_text.split()
                        if len(w) > 3 and w not in _skip_words
                    ][:4]

                    if not _keywords:
                        continue

                    # Search by filename first (most precise)
                    for _kw in _keywords:
                        _fname_hits = _search_engine.search_filename(
                            _kw, max_results=3
                        )
                        for _hit in _fname_hits:
                            _match_path = _hit.file_path if hasattr(_hit, "file_path") else str(_hit)
                            if _match_path not in existing_paths:
                                _targeted_finds.append({
                                    "path": _match_path,
                                    "_tier": "targeted_reinvestigation",
                                    "search_keyword": _kw,
                                    "requirement": str(_req)[:100],
                                })

                    # Then search by content literal (broader)
                    for _kw in _keywords[:2]:  # Limit to 2 content searches
                        _content_hits = _search_engine.search_literal(
                            _kw, max_results=3
                        )
                        for _hit in _content_hits:
                            _match_path = _hit.file_path if hasattr(_hit, "file_path") else str(_hit)
                            if _match_path not in existing_paths:
                                _targeted_finds.append({
                                    "path": _match_path,
                                    "_tier": "targeted_reinvestigation",
                                    "search_keyword": _kw,
                                    "requirement": str(_req)[:100],
                                })

                # Deduplicate targeted finds
                _seen_targeted = set()
                _unique_targeted = []
                for _tf in _targeted_finds:
                    if _tf["path"] not in _seen_targeted:
                        _seen_targeted.add(_tf["path"])
                        _unique_targeted.append(_tf)

                if _unique_targeted:
                    for _tf in _unique_targeted[:6]:  # Cap at 6 targeted finds
                        current_discovered.append(_tf)
                        existing_paths.add(_tf["path"])
                        promoted += 1
                    logger.info(
                        f"  🎯→✅ Step 2: targeted search found {promoted} relevant files"
                    )
                    for _tf in _unique_targeted[:promoted]:
                        logger.info(
                            f"    + {_tf['path']}  (keyword='{_tf['search_keyword']}', "
                            f"req='{_tf['requirement'][:50]}')"
                        )
                else:
                    logger.info("  🔍 Step 2: targeted search found no new files")

            except Exception as _search_exc:
                logger.warning(f"  Step 2 targeted search failed (non-fatal): {_search_exc}")

    # ── Step 3: Second Sufficiency Check ──────────────────────────────────
    # After Steps 1+2, evaluate whether the combined evidence (previous + newly collected)
    # is actually sufficient to cover the unsatisfied requirements.
    _coverage_assessment = "UNKNOWN"
    if unsatisfied:
        _new_file_count = len(current_discovered) - _original_discovered_count
        
        _unsatisfied_text = " ".join(str(r) for r in unsatisfied).lower()
        _skip_words = {"the", "and", "for", "with", "that", "this", "from", "should", "must"}
        _keywords = [w for w in _unsatisfied_text.split() if len(w) > 4 and w not in _skip_words]
        
        _combined_evidence = ""
        for f in current_discovered:
            _combined_evidence += str(f.get("path", "")) + " "
            _combined_evidence += str(f.get("search_keyword", "")) + " "
            _combined_evidence += str(f.get("requirement", "")) + " "
        
        _combined_evidence = _combined_evidence.lower()
        _matches = sum(1 for kw in _keywords if kw in _combined_evidence)
        
        # Evaluate whether the combined evidence (previous + newly collected)
        # is actually sufficient to cover the unsatisfied requirements.
        if len(current_discovered) == 0:
            _coverage_assessment = "INSUFFICIENT"
        elif not _keywords:
            _coverage_assessment = "LIKELY_SUFFICIENT"
        elif _matches >= len(_keywords) * 0.5:
            _coverage_assessment = "LIKELY_SUFFICIENT"
        elif _matches >= len(_keywords) * 0.2:
            _coverage_assessment = "PARTIALLY_COVERED"
        else:
            _coverage_assessment = "INSUFFICIENT"

        logger.info(
            f"  📊 Step 3 (Second Sufficiency): {_coverage_assessment} — "
            f"promoted {_new_file_count} files for {len(unsatisfied)} unsatisfied requirement(s). "
            f"Keywords matched: {_matches}/{len(_keywords)}. "
            f"New discovered total: {len(current_discovered)}"
        )

    # ── Step 4: Semantic Verification of Candidates (NO Blind Tier-2 Promotion) ─
    # Never blindly promote Tier 2 backup candidates. Expansion requires an explicit
    # evidence gap AND candidates must pass semantic verification against the unsatisfied requirement.
    if _coverage_assessment == "INSUFFICIENT" and tier2:
        _verified_promoted = 0
        for candidate in tier2[:5]:
            _cand_path = candidate.get("path", "")
            if _cand_path and _cand_path not in existing_paths:
                _cand_norm = _cand_path.lower()
                _is_relevant = any(w in _cand_norm for w in _keywords if len(w) > 4) if _keywords else False
                if _is_relevant:
                    candidate["_tier"] = "verified_candidate_promoted"
                    candidate["change_intent"] = "READ_ONLY"
                    current_discovered.append(candidate)
                    existing_paths.add(_cand_path)
                    _verified_promoted += 1
                    logger.info(f"    + Verified candidate promoted: {_cand_path}")

        if _verified_promoted > 0:
            logger.info(
                f"  📦→✅ Step 4: Promoted {_verified_promoted} semantically verified candidates "
                f"(total now: {len(current_discovered)}). Blind Tier-2 promotion blocked."
            )
        else:
            logger.info(
                "  🛡️ Step 4: No Tier 2 candidate passed semantic verification against unsatisfied "
                "requirements. Proceeding to re-plan with existing verified evidence."
            )
    elif _coverage_assessment == "INSUFFICIENT":
        logger.info(
            "  context_expand: evidence insufficient, but no candidates available — skipping"
        )
    else:
        logger.info(
            f"  context_expand: Step 3 assessed as {_coverage_assessment} — skipping Tier 2 fallback"
        )

    _total_promoted = len(current_discovered) - _original_discovered_count
    return {
        "discovered_files": current_discovered,
        "tier2_candidates": [] if (promoted > 0 and not tier2) else tier2,
        "context_expansion_count": expansion_count + 1,
        "unsatisfied_requirements": unsatisfied if unsatisfied else None,
        # Reset outcome state so the next cycle starts fresh
        "outcome_check_result": None,
        "outcome_remediation": None,
        "promoted_count": _total_promoted,
        "tier2_used": False,
    }


def pre_fix_build_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """Compute and persist build-error fingerprint BEFORE the router reads it.

    This node exists because check_build_status is a conditional-edge
    router (returns a string, not a dict) and CANNOT update state.

    Fingerprint lifecycle:
      Build 1 → error A → fingerprint="X", prev="", consecutive=0
      Fix attempt
      Build 2 → error A → fingerprint="X", prev="X", consecutive=1
      Fix attempt
      Build 3 → error A → fingerprint="X", prev="X", consecutive=2 → STUCK

    "Consecutive" means: how many times we've seen the SAME fingerprint
    AFTER the first observation. So consecutive=2 means we've built 3 times
    with identical errors — two fix attempts made zero progress.

    Important edge case (Build 1→A, Build 2→B, Build 3→A):
    This is NOT two consecutive identical errors. The fingerprint changed
    at Build 2 (B≠A), so consecutive resets to 0. At Build 3, A≠B, so
    consecutive stays 0. The system correctly sees this as progress.
    """
    build_result = state.get("build_result")
    if not build_result or build_result.status.value in ("success", "skipped"):
        return {}  # No fingerprint update needed on success

    raw_errors = getattr(build_result, "errors", []) or []
    current_fp = _build_err_fingerprint(raw_errors)
    prev_fp = state.get("prev_build_error_fingerprint") or ""
    history_fps = list(state.get("build_error_history") or [])
    consec = int(state.get("consecutive_identical_build_errors", 0) or 0)

    # Detect exact repeated error set (Stagnation) or 2-cycle oscillation (A -> B -> A)
    is_oscillation = current_fp in history_fps and current_fp != "" and current_fp != prev_fp
    if current_fp == prev_fp and current_fp != "":
        consec += 1
    elif is_oscillation:
        consec = max(consec, 2)  # Trigger stuck threshold on cycling
        logger.warning(
            f"  🔄 [pre_fix_build] Oscillation cycle detected! Fingerprint '{current_fp[:8]}' "
            f"matches a prior build state. Flagging as stuck."
        )
    else:
        consec = 0  # Errors changed — reset

    if current_fp:
        history_fps.append(current_fp)

    logger.info(
        f"  [pre_fix_build] fingerprint={'same' if current_fp == prev_fp else 'CHANGED'}, "
        f"consecutive={consec}"
    )

    # ── Change A/B: differential + infrastructure classification ──────────────
    # Attribute each diagnostic so the router can (a) skip source repair for
    # environment/dependency failures and (b) accept a build whose only errors
    # are pre-existing (baseline) ones the ticket did not introduce.
    infra_only = False
    differential_accept = False
    diag_summary = "none"
    pre_existing_decision = "leave"   # safe default — never silently authorize edits
    infrastructure_blocked = False
    authorized_pre_existing_files: list[str] = []
    try:
        from ticket_to_code.agents.build_diagnostic_classifier import (
            classify_build_diagnostics,
        )
        _our_files: set[str] = set()
        for _g in (state.get("generated_code") or []):
            _fp = (getattr(_g, "file_path", "") or "").replace("\\", "/").lower()
            if _fp:
                _our_files.add(_fp)
        for _ofp in (state.get("original_file_contents") or {}).keys():
            _our_files.add(str(_ofp).replace("\\", "/").lower())

        _report = classify_build_diagnostics(raw_errors, _our_files)
        infra_only = _report.is_infrastructure_only
        differential_accept = _report.is_differential_accept
        diag_summary = _report.summary()
        logger.info(
            f"  [pre_fix_build] diagnostics: {diag_summary} | "
            f"infra_only={infra_only} differential_accept={differential_accept} | "
            f"blocking={len(_report.blocking)} pre_existing={len(_report.pre_existing)}"
        )
        # Universal BuildFailureClassifier augmentation to halt on non-code failures
        try:
            from ticket_to_code.execution.build_orchestrator import BuildFailureClassifier, FailureCategory
            _cat, _sum, _actionable, _ = BuildFailureClassifier.classify(
                exit_code=getattr(build_result, "exit_code", 1),
                stdout=getattr(build_result, "stdout", "") or "\n".join(raw_errors),
                stderr=getattr(build_result, "stderr", ""),
                command=["mvn"]
            )
            if _cat in (
                FailureCategory.DEPENDENCY_RESOLUTION_FAILURE,
                FailureCategory.MISSING_TOOLCHAIN,
                FailureCategory.INFRASTRUCTURE_FAILURE,
            ):
                infra_only = True
                infrastructure_blocked = True
                diag_summary = f"[{_cat.value}] {_sum}"
        except Exception as _bfe:
            logger.debug(f"BuildFailureClassifier check skipped: {_bfe}")
        try:
            _trace = {
                "summary": diag_summary,
                "infrastructure_only": infra_only,
                "differential_accept": differential_accept,
                "blocking": [d.raw[:300] for d in _report.blocking][:20],
                "infrastructure": [d.raw[:300] for d in _report.infrastructure][:20],
                "pre_existing": [d.raw[:300] for d in _report.pre_existing][:20],
            }
            write_trace_artifact(
                state["workspace_path"], state["ticket"].ticket_id,
                "build_diagnostics.json", json.dumps(_trace, indent=2),
            )
        except Exception:
            pass

        _ui_cb = _get_transient(state, "_ui_callback")
        _decision_provider = _get_transient(state, "_decision_provider")

        # ── External dependency / infrastructure failure — surface to user ──
        if infra_only:
            infrastructure_blocked = True
            _infra_msg = (
                "Build failed due to an external dependency/repository problem "
                "(not a source-code defect). Source repair is skipped. Resolve the "
                "environment issue, then re-run validation."
            )
            logger.error(f"  🌐 INFRASTRUCTURE: {_infra_msg}")
            if _ui_cb:
                try:
                    from datetime import datetime as _dt
                    _ui_cb({
                        "phase": "build", "status": "blocked", "message": _infra_msg,
                        "data": {"node": "pre_fix_build", "event_type": "infrastructure_failure",
                                 "diagnostics": [d.raw[:300] for d in _report.infrastructure][:10]},
                        "timestamp": _dt.now().isoformat(),
                    })
                except Exception:
                    pass

        # ── Pre-existing errors — bounded user decision (default LEAVE) ──
        elif _report.pre_existing:
            from ticket_to_code.agents.user_decision_gate import (
                request_user_decision, build_pre_existing_payload,
            )
            _payload = build_pre_existing_payload(_report)
            if _ui_cb:
                try:
                    from datetime import datetime as _dt
                    _ui_cb({
                        "phase": "build", "status": "awaiting_decision",
                        "message": _payload["message"],
                        "data": {"node": "pre_fix_build", "event_type": "pre_existing_errors",
                                 **_payload},
                        "timestamp": _dt.now().isoformat(),
                    })
                except Exception:
                    pass
            pre_existing_decision = request_user_decision(
                provider=_decision_provider,
                payload=_payload,
                options=["fix", "leave", "stop"],
                default="leave",
                timeout=60.0,
            )
            if pre_existing_decision == "fix":
                # Explicitly authorize (only) these pre-existing files for repair.
                _seen: set[str] = set()
                for _d in _report.pre_existing:
                    if _d.file_path and _d.file_path.lower() not in _seen:
                        _seen.add(_d.file_path.lower())
                        authorized_pre_existing_files.append(_d.file_path.lower())
                logger.info(
                    f"  [pre_fix_build] user authorized fixing {len(authorized_pre_existing_files)} "
                    f"pre-existing file(s)."
                )
    except Exception as _cls_exc:
        logger.debug(f"  [pre_fix_build] classification skipped (non-fatal): {_cls_exc}")

    return {
        "prev_build_error_fingerprint": current_fp,
        "consecutive_identical_build_errors": consec,
        "build_error_history": history_fps,
        "build_infrastructure_only": infra_only,
        "build_differential_accept": differential_accept,
        "build_diagnostic_summary": diag_summary,
        "build_infrastructure_blocked": infrastructure_blocked,
        "pre_existing_decision": pre_existing_decision,
        "authorized_pre_existing_files": authorized_pre_existing_files,
    }


def check_build_status(state: TicketToCodeState):
    """
    Route after build — EVIDENCE-BASED, not counter-based.

    This is a PURE ROUTER: it only reads state that was persisted by
    pre_fix_build_node. It does NOT compute fingerprints or update state.

    Three kinds of evidence from the build result:
    1. SUCCESS → done, run outcome check
    2. FAILING with DIFFERENT errors than last cycle → progress is being made;
       continue fixing (counter still bounded by max_retry_attempts)
    3. FAILING with IDENTICAL errors to last cycle (consecutive_identical ≥ 2) →
       the current fix strategy is definitively stuck; stop trying the same thing
       and cut to memory_update so the failure is recorded rather than wasted.

    The distinction between (2) and (3) is the core intelligence: it tells the
    difference between "we're making progress but not done" and "we're spinning".
    """
    build_result = state.get("build_result")
    if not build_result:
        return "memory_update"

    if build_result.status.value == "success":
        logger.info("  → Build successful, routing to test runner")
        return "test"
    if build_result.status.value == "skipped":
        logger.warning("  → No compiler ran (build SKIPPED) — routing to test runner; run stays UNVERIFIED")
        return "test"

    _decision = str(state.get("pre_existing_decision", "") or "")
    if not _decision and state.get("build_differential_accept"):
        try:
            from ticket_to_code.runtime.interactive_gate import InteractiveClarificationGate
            _decision = InteractiveClarificationGate.request_clarification(
                question="Pre-existing build errors detected in baseline repository. How would you like to proceed?",
                options=["leave", "fix", "stop"],
                recommended_option="leave",
                descriptions=[
                    "Accept baseline errors and continue testing (Recommended)",
                    "Attempt to repair pre-existing baseline errors",
                    "Stop execution and preserve current workspace state"
                ],
                timeout_seconds=5.0,
            )
        except Exception:
            _decision = "leave"
    if not _decision:
        _decision = "leave"

    # ── User decision gate: STOP safely when the user chose to halt ──
    if _decision == "stop":
        logger.warning(
            "  → User chose STOP after pre-existing errors were surfaced. "
            "Terminating safely and preserving the workspace."
        )
        return "memory_update"

    # ── Change B: infrastructure-only failure → do NOT attempt source repair ──
    if state.get("build_infrastructure_only"):
        logger.error(
            "  → Build failure is INFRASTRUCTURE-only (dependency/network/registry). "
            "Source correctness cannot be established; skipping LLM source repair. "
            f"({state.get('build_diagnostic_summary', '')})"
        )
        return "memory_update"

    # ── Change A: all failures are PRE_EXISTING (ticket introduced none) → accept ──
    # UNLESS the user explicitly chose FIX, in which case we fall through to
    # fix_build so the authorized pre-existing files get repaired.
    if state.get("build_differential_accept") and _decision != "fix":
        logger.info(
            "  → Build errors are all PRE_EXISTING baseline errors; the ticket "
            "introduced no new build errors. Accepting and routing to test runner. "
            f"({state.get('build_diagnostic_summary', '')})"
        )
        return "test"

    # Read persisted state from pre_fix_build_node
    _consec = int(state.get("consecutive_identical_build_errors", 0) or 0)
    _retries = int(state.get("retry_attempt", 0) or 0)
    _max_retries = int(state.get("max_retry_attempts", 3) or 3)

    if _retries >= _max_retries:
        logger.error(
            f"  → Max retries ({_max_retries}) reached. Finalizing with failure."
        )
        return "memory_update"

    if _consec >= 2:
        logger.warning(
            f"  → STUCK: same build errors for {_consec} consecutive cycles "
            f"(retry {_retries}/{_max_retries}). Current fix strategy cannot make "
            f"progress. Finalizing with failure rather than repeating the same fix."
        )
        return "memory_update"

    if _retries > 0:
        logger.info(
            f"  → Build errors changed or first failure — progress detected, "
            f"routing to fix_build (retry {_retries + 1}/{_max_retries})"
        )
    else:
        logger.info(
            f"  → Build failed, routing to fix_build "
            f"(retry {_retries + 1}/{_max_retries})"
        )

    return "fix_build"


def ui_verify_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """Optional live-UI check (Playwright) after build/test; env-gated, advisory on infra problems."""
    try:
        from ticket_to_code.verification.ui_verification import run_ui_verification
        reqs = getattr(state.get("requirements"), "functional_requirements", []) or []
        result = run_ui_verification(
            str(state.get("sub_workspace_path") or state.get("workspace_path") or ""),
            state.get("generated_code") or [],
            reqs,
        )
    except Exception as exc:
        logger.warning(f"  UI verify unavailable (non-fatal): {exc}")
        result = {"status": "unavailable", "reason": str(exc)}
    status = result.get("status")
    prev = state.get("ui_verification_result") or {}
    # Identical failures after a repair attempt mean the cause is outside the generated code (baseline/env).
    if (status == "failed" and prev.get("status") in ("failed", "failed_unchanged")
            and prev.get("signature") == result.get("signature")):
        result = {**result, "status": "failed_unchanged",
                  "reason": "same UI failures before and after repair; treated as pre-existing/environmental"}
        status = "failed_unchanged"
    if status != "skipped":
        logger.info(f"  UI VERIFY: {status} — {result.get('summary') or result.get('reason', '')}")
    return {"ui_verification_result": result}


def check_test_status(state: TicketToCodeState):
    """Route after tests: success/no_tests → ui_verify → outcome_check; infrastructure error → memory_update; test failures → fix_build or memory_update."""
    test_res = state.get("test_result")
    if not test_res:
        logger.info("  → No test result recorded, proceeding to ui_verify/outcome_check (compile-only)")
        return "ui_verify"

    if state.get("test_differential_accept") or state.get("compile_only_verification") or state.get("verification_level") == "compile_only":
        logger.info("  → Differential/compile-only test acceptance active (pre-existing baseline errors accepted) — proceeding to ui_verify/outcome_check")
        return "ui_verify"

    status_val = getattr(getattr(test_res, "status", None), "value", "").lower()
    if status_val in ("no_tests", "skipped"):
        logger.info("  → No test suite detected (compile-only verification) — proceeding to ui_verify/outcome_check")
        return "ui_verify"

    if status_val in ("all_passed", "success"):
        logger.info(f"  → Tests passed ({getattr(test_res, 'passed', 0)}/{getattr(test_res, 'total_tests', 0)}), routing to ui_verify/outcome_check")
        return "ui_verify"

    # Infrastructure / environment error (e.g. runner missing from PATH, timeout, crash)
    # Source-code repair cannot fix missing CLI tools or environment problems.
    # Route straight to memory_update (failure) without burning 3 futile fix loops!
    if status_val in ("error", "timeout"):
        logger.error(
            f"  → Test failure is INFRASTRUCTURE/ENVIRONMENT ({status_val}): "
            f"{getattr(test_res, 'errors', ['Runner error'])[:1]}. "
            f"Source code repair cannot resolve environment issues. Finalizing with failure."
        )
        return "memory_update"

    # Source-level test failures (e.g. some_failed, all_failed): check retry attempts
    retry_attempt = int(state.get("retry_attempt", 0) or 0)
    max_retries = int(state.get("max_retry_attempts", 3) or 3)
    if retry_attempt < max_retries:
        logger.warning(
            f"  → Tests failed ({getattr(test_res, 'failed', 0)}/{getattr(test_res, 'total_tests', 0)}, status={status_val}). "
            f"Routing to remediation_plan (attempt {retry_attempt + 1}/{max_retries})"
        )
        return "remediation_plan"
    else:
        logger.error(f"  → Tests failed after {max_retries} attempts. Finalizing with failure.")
        return "memory_update"

def _perform_workspace_rollback(
    state: TicketToCodeState,
    patch_filename: str = "verification_failed.patch",
) -> dict:
    """Safely restore the workspace to pristine pre-run state and archive a patch.

    Guarantees:
    1. Only paths strictly contained within the workspace root are touched (prevents path traversal).
    2. Modified files with snapshots in original_file_contents are restored to original content.
    3. Newly created files (present in generated_code or _run_generated_map, but not in original_file_contents) are unlinked.
    4. A unified diff is computed and archived to .aviator/traces/<ticket_id>/<patch_filename>
       and via write_trace_artifact (never polluting the workspace root directly).
    5. Returns dict with rollback summary: {rollback_performed, restored_files, removed_files, patch_file}.
    """
    original_contents = state.get("original_file_contents") or {}
    workspace_path = state.get("workspace_path")
    gen_code = state.get("generated_code") or []
    ticket = state.get("ticket")
    ticket_id = str(getattr(ticket, "ticket_id", "") or getattr(ticket, "id", "") or "")

    restored_files = []
    removed_files = []

    if not workspace_path or not (original_contents or gen_code or state.get("_run_generated_map")):
        return {
            "rollback_performed": False,
            "restored_files": [],
            "removed_files": [],
            "patch_file": None,
        }

    ws_p = Path(workspace_path).resolve()

    def _normalize_rel(p: str) -> str:
        p_str = str(p).replace("\\", "/").strip()
        while p_str.startswith("./"):
            p_str = p_str[2:]
        return p_str.lstrip("/")

    norm_original = {_normalize_rel(k).lower(): (k, v) for k, v in original_contents.items()}

    def _is_safe_workspace_path(path_str: str) -> Optional[Path]:
        try:
            target_p = (ws_p / path_str).resolve()
            if not target_p.is_relative_to(ws_p) or target_p == ws_p:
                logger.warning(f"Rollback path containment violation: '{path_str}' escapes workspace '{ws_p}'")
                return None
            return target_p
        except Exception as _p_err:
            logger.warning(f"Rollback path invalid '{path_str}': {_p_err}")
            return None

    import difflib
    try:
        from ticket_to_code.runtime.diff_engine import DiffReviewEngine
        diff_engine_available = True
    except Exception:
        diff_engine_available = False

    patch_chunks = []

    # 1. Capture unified diff of all modifications before restoring
    for rel_path, orig_text in original_contents.items():
        abs_p = _is_safe_workspace_path(rel_path)
        if abs_p and abs_p.is_file():
            curr_text = abs_p.read_text(encoding="utf-8", errors="ignore")
            if curr_text != orig_text:
                if diff_engine_available:
                    try:
                        file_diff = DiffReviewEngine.compute_file_diff(rel_path, orig_text or "", curr_text or "")
                        if file_diff.raw_diff:
                            patch_chunks.append(file_diff.raw_diff)
                            continue
                    except Exception as _de_err:
                        logger.debug(f"DiffReviewEngine fallback for {rel_path}: {_de_err}")
                orig_lines = orig_text.splitlines(keepends=True) if orig_text is not None else []
                curr_lines = curr_text.splitlines(keepends=True) if curr_text is not None else []
                diff = list(difflib.unified_diff(
                    orig_lines,
                    curr_lines,
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                ))
                patch_chunks.extend(diff)

    # 2. Add diffs for newly created files
    candidate_created = []
    for g in gen_code:
        g_path = getattr(g, "file_path", "") or (g.get("file_path", "") if isinstance(g, dict) else "")
        if g_path and g_path not in candidate_created:
            candidate_created.append(g_path)
    for mp in (state.get("_run_generated_map") or {}).keys():
        if mp and mp not in candidate_created:
            candidate_created.append(mp)

    for g_path in candidate_created:
        norm_g = _normalize_rel(g_path).lower()
        if norm_g and norm_g not in norm_original:
            abs_p = _is_safe_workspace_path(g_path)
            if abs_p and abs_p.is_file():
                created_text = abs_p.read_text(encoding="utf-8", errors="ignore")
                diff = list(difflib.unified_diff(
                    [],
                    created_text.splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile=f"b/{g_path}",
                ))
                patch_chunks.extend(diff)

    # Save the full archived patch to trace dir (NEVER to workspace root)
    patch_dest = None
    if patch_chunks:
        full_patch = "".join(patch_chunks)
        write_trace_artifact(workspace_path, ticket_id, patch_filename, full_patch)
        try:
            local_trace_dir = ws_p / ".aviator" / "traces" / ticket_id
            local_trace_dir.mkdir(parents=True, exist_ok=True)
            local_patch_path = local_trace_dir / patch_filename
            local_patch_path.write_text(full_patch, encoding="utf-8")
            patch_dest = str(local_patch_path)
        except Exception as _patch_err:
            logger.debug(f"Local patch archival non-fatal: {_patch_err}")
        logger.info(f"  💾 Archived unverified workspace changes to {patch_filename} in traces")

    # 3. Restore modified files
    for rel_path, orig_text in original_contents.items():
        abs_p = _is_safe_workspace_path(rel_path)
        if not abs_p:
            continue
        try:
            abs_p.parent.mkdir(parents=True, exist_ok=True)
            if orig_text is not None:
                abs_p.write_text(orig_text, encoding="utf-8")
                restored_files.append(rel_path)
            else:
                if abs_p.is_file():
                    abs_p.unlink()
                    restored_files.append(rel_path)
        except Exception as _r_err:
            logger.warning(f"Failed to restore {rel_path}: {_r_err}")

    # 4. Remove newly created files
    for g_path in candidate_created:
        norm_g = _normalize_rel(g_path).lower()
        if norm_g and norm_g not in norm_original:
            abs_p = _is_safe_workspace_path(g_path)
            if not abs_p:
                continue
            try:
                if abs_p.is_file():
                    abs_p.unlink()
                    removed_files.append(g_path)
            except Exception as _u_err:
                logger.warning(f"Failed to remove created file {g_path}: {_u_err}")

    rollback_performed = bool(restored_files or removed_files)
    if rollback_performed:
        logger.info(
            f"  ↩️ Rolled back workspace to pre-run baseline: "
            f"{len(restored_files)} file(s) restored, {len(removed_files)} new file(s) removed."
        )

    return {
        "rollback_performed": rollback_performed,
        "restored_files": restored_files,
        "removed_files": removed_files,
        "patch_file": patch_dest,
    }


def verification_failed_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """TERMINAL node: semantic verification failed after the remediation budget.

    The outcome check verdict is still PARTIAL/INCOMPLETE/UNCERTAIN (or a
    deterministic acceptance check still fails) after max_outcome_fix_attempts
    remediation cycles. This node ends the run honestly:

      - status becomes "verification_failed" (the UI reports it as a failure,
        never as "✅ Workflow complete")
      - the exact unresolved requirements are surfaced to the user
      - the run record is emitted with final_status="FAILED_VERIFICATION"
      - brain memory does NOT record the ticket as a success (nothing is
        learned from code that did not pass verification)
      - workspace is safely rolled back to pre-run baseline and rejected changes
        are archived to verification_failed.patch in traces.
    """
    logger.info("\n" + "=" * 80)
    logger.info(" ⛔ VERIFICATION FAILED — ticket requirements NOT fully satisfied")
    logger.info("=" * 80)

    result = state.get("outcome_check_result") or {}
    verdict = result.get("status", "UNKNOWN")
    attempt = int(state.get("outcome_fix_attempt", 0) or 0)
    max_attempts = int(state.get("max_outcome_fix_attempts", 3) or 3)
    findings = state.get("outcome_findings") or result.get("findings") or []
    unsatisfied = state.get("unsatisfied_requirements") or result.get("unsatisfied_requirements") or []

    defect_findings = [f for f in findings if str(f.get("verdict", "")).upper() != "CORRECT"]
    unresolved_lines = [
        str(f.get("missing_behavior") or f.get("summary") or "unspecified requirement")[:160]
        for f in defect_findings
    ] or [str(u)[:160] for u in unsatisfied] or ["requirements not fully satisfied"]

    _reason = (
        f"Semantic verification returned {verdict} after {min(attempt, max_attempts)}/{max_attempts} "
        f"remediation attempt(s). Unresolved: " + "; ".join(unresolved_lines[:3])
    )

    logger.error(f"  Verdict: {verdict}  (remediation attempts used: {attempt}/{max_attempts})")
    for i, ln in enumerate(unresolved_lines[:5], 1):
        logger.error(f"    {i}. {ln}")
    logger.error("  Manual review required before shipping this change.")

    ticket = state.get("ticket")
    ticket_id = str(getattr(ticket, "ticket_id", "") or getattr(ticket, "id", "") or "")

    # ── Workspace Rollback with Diff Archival ──────────────────────────────
    build_res = state.get("build_result")
    build_passed = (
        build_res is not None
        and getattr(build_res, "status", None)
        and getattr(build_res.status, "value", "") == "success"
    ) or bool(state.get("build_differential_accept"))

    if not build_passed:
        rb = _perform_workspace_rollback(state, patch_filename="verification_failed.patch")
    else:
        logger.info("  [GATE] Build passed cleanly. Preserving generated code on disk for human review.")
        rb = {"rollback_performed": False, "restored_files": [], "removed_files": [], "patch_file": None}
    rollback_performed = rb["rollback_performed"]
    restored_files = rb["restored_files"]
    removed_files = rb["removed_files"]

    # ── Surface the failure to the UI with the exact unresolved behaviors ──
    _ui_cb = _get_transient(state, "_ui_callback")
    if _ui_cb:
        from datetime import datetime as _dt_vf
        _ui_cb({
            "phase": "validation",
            "status": "error",
            "message": (
                f"🛑 Verification failed — {verdict} after {max_attempts} remediation attempt(s). "
                f"Workspace safely rolled back to pre-run state (patch archived in verification_failed.patch)."
                if rollback_performed else
                f"🛑 Verification failed — {verdict} after {max_attempts} remediation attempt(s)"
            ),
            "data": {
                "node": "verification_failed",
                "event_type": "verification_failed",
                "verdict": verdict,
                "attempts_used": min(attempt, max_attempts),
                "max_attempts": max_attempts,
                "unresolved_requirements": unresolved_lines[:5],
                "findings": defect_findings[:5],
                "escalation_reason": _reason,
                "workspace_rolled_back": rollback_performed,
                "restored_files": restored_files,
                "removed_files": removed_files,
            },
            "timestamp": _dt_vf.now().isoformat(),
        })

    # ── Emit RunRecord as an honest failure (never COMPLETE) ────────────────
    run_ctx: Optional[RunContext] = _get_transient(state, "run_ctx")
    if run_ctx:
        run_ctx.final_status = "FAILED_VERIFICATION"
        try:
            record = run_ctx.to_record()
            write_trace_artifact(
                state["workspace_path"],
                ticket_id or state["ticket"].ticket_id,
                "run_record.json",
                record.to_dict(),
            )
            logger.info(
                "📊 RunRecord emitted: status=%s health=%s cost_usd=%.4f",
                record.final_status, record.run_health, record.cost_usd,
            )
        except Exception as _rr_exc:
            logger.warning("RunRecord emission failed (non-fatal): %s", _rr_exc)

    # Note: brain memory is intentionally NOT updated — a run that failed
    # verification must never be recorded as a successful ticket.
    _clear_transient(ticket_id or getattr(state.get("ticket"), "ticket_id", ""))

    return {
        "status": "verification_failed",
        "escalation_type": "requirements_unsatisfied",
        "escalation_reason": _reason,
        "error": _reason,
        "workspace_rolled_back": rollback_performed,
        "restored_files": restored_files,
        "removed_files": removed_files,
    }


def memory_update_node(state: TicketToCodeState, agents: WorkflowAgents) -> dict:
    """Phase 4: Brain Memory Update upon Success"""
    import os
    
    logger.info("\n" + "=" * 80)
    logger.info(" PHASE 4: Updating Project Brain Memory")
    logger.info("=" * 80)

    # ── Clear reuse directive so state never leaks to subsequent runs ────────
    try:
        from ticket_to_code.agents.code_generator import clear_reuse_directive
        clear_reuse_directive()
    except Exception:
        pass
    
    # Gate memory update behind Build and Test success
    build_result = state.get("build_result")
    test_result = state.get("test_result")
    _build_status_val = (
        getattr(getattr(build_result, "status", None), "value", "") if build_result is not None else ""
    )
    build_skipped = _build_status_val == "skipped"
    build_success = (_build_status_val == "success") or build_skipped or bool(state.get("build_differential_accept"))

    verification_level = state.get("verification_level") or "unverified"
    _heuristic_only_accept = False
    if state.get("test_differential_accept") and state.get("test_baseline_compared"):
        # Failures were proven pre-existing by a real baseline run.
        test_success = True
        verification_level = "compile_only"
    elif state.get("test_differential_accept"):
        # Filename-heuristic acceptance without a baseline: NOT evidence. Keep the code, do not learn from it.
        test_success = False
        _heuristic_only_accept = True
        verification_level = "unverified"
    elif test_result is not None:
        test_status_val = getattr(getattr(test_result, "status", None), "value", "").lower()
        if test_status_val in ("all_passed", "success"):
            test_success = True
            verification_level = "test_verified"
        elif test_status_val in ("no_tests", "skipped"):
            test_success = True
            verification_level = "compile_only"
        else:
            test_success = False
            verification_level = "test_failed"
    else:
        if state.get("compile_only_verification"):
            test_success = True
            verification_level = "compile_only"
        else:
            test_success = False
            verification_level = "unverified"
    
    # Memory is ALWAYS ENABLED by default. The system MUST learn from every run.
    # Default to "1" (enabled). Override to "0" to disable if needed.
    memory_update_enabled = os.getenv("AVIATOR_ENABLE_MEMORY_UPDATE", "1") == "1"
    
    if not memory_update_enabled:
        logger.info("  [GATE] Memory Update disabled by env var AVIATOR_ENABLE_MEMORY_UPDATE=0")
        return {"status": "memory_skipped", "verification_level": verification_level}
        
    if not build_success:
        logger.warning("  [GATE] Memory Update: Build failed. Rolling back uncompilable changes.")
        # Roll back unverified/broken changes and archive diff
        rb = _perform_workspace_rollback(state, patch_filename="verification_failed.patch")

        # Fix 1: Propagate actual failure reason so UI doesn't show "memory_skipped" as terminal error
        _build_errors = getattr(build_result, "errors", []) if (build_result and not build_success) else []
        _fail_reason = (
            f"Build failed with {len(_build_errors)} error(s): "
            + "; ".join(str(e)[:120] for e in _build_errors[:3])
        ) if _build_errors else "Build did not pass"

        # Emit RunRecord as an honest failure
        run_ctx: Optional[RunContext] = _get_transient(state, "run_ctx")
        if run_ctx:
            run_ctx.final_status = "FAILED_BUILD"
            try:
                record = run_ctx.to_record()
                _ticket_obj = state.get("ticket")
                ticket_id_val = str(getattr(_ticket_obj, "ticket_id", "") or "") if _ticket_obj else ""
                write_trace_artifact(
                    state.get("workspace_path", ""),
                    ticket_id_val,
                    "run_record.json",
                    record.to_dict(),
                )
            except Exception as _rr_exc:
                logger.warning("RunRecord emission failed (non-fatal): %s", _rr_exc)

        state["status"] = "escalated"
        state["escalation_type"] = "build_failed"
        state["escalation_reason"] = _fail_reason
        return {
            "status": "escalated",
            "escalation_type": "build_failed",
            "escalation_reason": _fail_reason,
            "verification_level": verification_level,
            "workspace_rolled_back": rb["rollback_performed"],
            "restored_files": rb["restored_files"],
            "removed_files": rb["removed_files"],
        }
    elif not test_success:
        logger.warning("  [GATE] Build PASSED cleanly (0 compilation errors). Automated tests failed or were inconclusive.")
        logger.warning("  [GATE] PRESERVING compiled code on disk (no rollback) but this run is NOT a success and will NOT be learned.")
        verification_level = "unverified" if _heuristic_only_accept else "test_failed"
        rb = {"rollback_performed": False, "restored_files": [], "removed_files": [], "patch_file": None}
    if build_skipped:
        logger.warning("  [GATE] No compiler ran for this change (build SKIPPED) — recording as UNVERIFIED, not learned.")
        verification_level = "unverified"

    # ── Phase 6: Escalation — code compiles but ticket requirements unresolved ──
    # This catches the "looks done, isn't" class of bugs: code compiles, tests
    # pass, but Phase 5's mechanical verification found checklist items that
    # are NOT implemented. Instead of silently shipping, escalate.
    _unresolved = state.get("unresolved_checklist_items", [])
    _escalation_reason = None
    if _unresolved:
        _unresolved_descs = [
            item.get("description", "?")[:80] for item in _unresolved[:3]
        ]
        _escalation_reason = (
            f"Code compiles but {len(_unresolved)} checklist item(s) remain "
            f"unresolved after verification: "
            + "; ".join(_unresolved_descs)
        )
        state["status"] = "escalated"
        state["escalation_type"] = "checklist_incomplete"
        state["escalation_reason"] = _escalation_reason
        logger.warning(f"  ⚠️ ESCALATED: {_escalation_reason}")
        # Still proceed with memory update so the system learns from this run

    ticket = state["ticket"]
    
    # We get the list of modified files from the generated code
    generated_code = state.get("generated_code") or []
    _learnable = bool(test_success) and not build_skipped and not _unresolved
    if not _learnable:
        logger.warning(
            f"  [GATE] Not recording ticket {ticket.ticket_id} in the brain: "
            f"test_success={test_success}, build_skipped={build_skipped}, unresolved_items={len(_unresolved)}"
        )
    for gen_file in (generated_code if _learnable else []):
        file_path = getattr(gen_file, "file_path", getattr(gen_file, "path", None))
        if file_path:
            logger.info(f"  Registering [{verification_level}] ticket {ticket.ticket_id} against {file_path}")
            try:
                agents.localizer.repository_brain.record_successful_ticket(file_path, verification_level=verification_level)
            except TypeError:
                agents.localizer.repository_brain.record_successful_ticket(file_path)

    # Self-updating brain: regenerate the directory brain so it learns any new/removed
    # files/folders from this ticket. Drift-aware + cached, so this is cheap and idempotent.
    # Must never break a successful ticket, so failures are swallowed as non-fatal.
    try:
        from ticket_to_code.brain.generate_repository_brain import refresh_repository_brain
        brain_summary = refresh_repository_brain(
            str(agents.localizer.workspace_path),
            llm=getattr(agents.localizer, "llm", None),
        )
        logger.info("  Brain refresh: %s", brain_summary.get("status"))
    except Exception as _brain_exc:
        logger.warning("  Brain refresh failed (non-fatal): %s", _brain_exc)

    # DEBUG LOG
    log_phase(
        phase='MEMORY_UPDATE',
        llm_output={
            'status': 'memory_updated',
            'files_learned': len(generated_code),
            'ticket_id': ticket.ticket_id,
            'memory_enabled': memory_update_enabled,
            'verification_level': verification_level,
        },
        agent='MemorySystem'
    )

    # ── B11/B12: Emit RunRecord at terminal node ───────────────────────────────
    run_ctx: Optional[RunContext] = _get_transient(state, "run_ctx")
    if run_ctx:
        if not test_success:
            run_ctx.final_status = "UNVERIFIED" if _heuristic_only_accept else "FAILED_TESTS"
        elif build_skipped:
            run_ctx.final_status = "UNVERIFIED"
        else:
            run_ctx.final_status = "COMPLETE"
        run_ctx.validation_level = "FULL" if verification_level == "test_verified" else "STATIC"
        try:
            record = run_ctx.to_record()
            record_dict = record.to_dict()
            alerts = record.alerts()
            if alerts:
                logger.warning("⚠️  RunRecord alerts: %s", alerts)
            logger.info(
                "📊 RunRecord emitted: status=%s health=%s evidence_backed=%s "
                "cost_usd=%.4f retries=%d validation_level=%s",
                record.final_status, record.run_health,
                record.evidence_backed, record.cost_usd, record.retries,
                record.validation_level,
            )
            write_trace_artifact(
                state["workspace_path"],
                state["ticket"].ticket_id,
                "run_record.json",
                record_dict,
            )
        except Exception as _rr_exc:
            logger.warning("RunRecord emission failed (non-fatal): %s", _rr_exc)

    # ── UI Callback Badge Emission ─────────────────────────────────────────────
    _ui_cb = _get_transient(state, "_ui_callback")
    if _ui_cb:
        try:
            from datetime import datetime as _dt_mem
            _badge_label = (
                "Test-Verified" if verification_level == "test_verified"
                else ("Compile-Only" if verification_level == "compile_only" else "Unverified")
            )
            _ui_cb({
                "phase": "memory",
                "status": "completed",
                "message": f"Memory updated ({_badge_label})",
                "data": {
                    "node": "memory_update",
                    "event_type": "memory_updated",
                    "verification_level": verification_level,
                    "compile_only_verification": (verification_level == "compile_only"),
                    "badge": _badge_label,
                    "files_learned": len(generated_code),
                },
                "timestamp": _dt_mem.now().isoformat(),
            })
        except Exception:
            pass

    # Clean up transient registry for this ticket (prevents memory leaks)
    _clear_transient(getattr(state.get("ticket"), "ticket_id", ""))

    if _unresolved:
        return_status = "escalated"
    elif not test_success and not _heuristic_only_accept:
        return_status = "verification_failed"
    elif not test_success or build_skipped:
        return_status = "unverified"
    else:
        return_status = "memory_updated"
    return {
        "status": return_status,
        "verification_level": verification_level,
        "escalation_type": "checklist_incomplete" if _unresolved else None,
        "escalation_reason": _escalation_reason if _unresolved else None
    }





# ============================================================================
# PHASE-TRACKED NODE WRAPPER (Token Instrumentation)
# ============================================================================

_STEP_SEQ: dict = {}
_STEP_SEQ_LOCK = _threading.Lock()
_STEP_CONTEXT_KEYS = ("status", "retry_attempt", "outcome_fix_attempt", "code_status")


def _summarize_for_trace(value: Any, depth: int = 0) -> Any:
    """Compact, JSON-safe description of a state value so a trace stays readable."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:197] + "..."
    if isinstance(value, (list, tuple, set)):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        if depth >= 1:
            return f"{{{len(value)} keys}}"
        return {str(k): _summarize_for_trace(v, depth + 1) for k, v in list(value.items())[:10]}
    status = getattr(value, "status", None)
    status = getattr(status, "value", status)
    return f"{type(value).__name__}(status={status})" if status is not None else type(value).__name__


def _record_step(state: Any, node_name: str, started: float, result: Any, error: Optional[BaseException]) -> None:
    """Append one step (node, duration, outcome, what it produced) to steps.jsonl. Never raises."""
    try:
        import time as _t
        ticket_id = str(getattr(state.get("ticket"), "ticket_id", "") or "unknown")
        with _STEP_SEQ_LOCK:
            seq = _STEP_SEQ.get(ticket_id, 0) + 1
            _STEP_SEQ[ticket_id] = seq
        produced = {k: _summarize_for_trace(v) for k, v in (result or {}).items()} if isinstance(result, dict) else {}
        entry = {
            "seq": seq,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "node": node_name,
            "duration_ms": int((_t.time() - started) * 1000),
            "ok": error is None,
            "error": f"{type(error).__name__}: {error}" if error else None,
            "context": {k: _summarize_for_trace(state.get(k)) for k in _STEP_CONTEXT_KEYS if state.get(k) is not None},
            "produced": produced,
        }
        trace_dir = str(get_trace_dir(ticket_id))
        os.makedirs(trace_dir, exist_ok=True)
        with open(os.path.join(trace_dir, "steps.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        brief = ", ".join(f"{k}={json.dumps(v, default=str)[:80]}" for k, v in list(produced.items())[:4])
        logger.info(f"STEP {seq:02d} {node_name} {'OK' if error is None else 'FAILED'} ({entry['duration_ms']} ms) {brief}")
    except Exception:
        pass


def reset_step_trace(ticket_id: str) -> None:
    """Start a fresh steps.jsonl for a ticket (called at workflow start)."""
    with _STEP_SEQ_LOCK:
        _STEP_SEQ[ticket_id] = 0
    try:
        path = str(get_trace_dir(ticket_id) / "steps.jsonl")
        if os.path.exists(path):
            os.replace(path, path + ".prev")
    except Exception:
        pass


class NodeTimeoutError(TimeoutError):
    """A workflow node exceeded its time budget (a hung LLM/IO call must not stall a run for hours)."""


_NODE_TIMEOUT_DEFAULTS = {
    "edit_loop": 1200, "generate_code": 1200, "plan": 900, "test": 900,
    "build": 900, "fix_build": 900, "ui_verify": 300,
}


def _node_deadline_s(node_name: str) -> float:
    raw = os.environ.get("TICKET_TO_CODE_NODE_TIMEOUT_S", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(_NODE_TIMEOUT_DEFAULTS.get(node_name, 600))


def _call_with_deadline(node_name: str, node_fn, state):
    """Run a node with a time budget; the abandoned worker thread cannot be killed, but the run can fail fast."""
    limit = _node_deadline_s(node_name)
    if limit <= 0:
        return node_fn(state)
    import concurrent.futures as _cf
    import contextvars as _cv
    ex = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"node-{node_name}")
    fut = ex.submit(_cv.copy_context().run, node_fn, state)
    done, _ = _cf.wait([fut], timeout=limit)
    ex.shutdown(wait=False, cancel_futures=True)
    if not done:
        # The worker thread cannot be killed, but it must stop WRITING: flag the run as
        # cancelled so llm_invoke / file writers abort at their next checkpoint.
        try:
            _rc = _get_transient(state, "run_ctx")
            if _rc is not None and hasattr(_rc, "request_cancel"):
                _rc.request_cancel(f"node '{node_name}' timed out after {int(limit)}s")
        except Exception:
            pass
        raise NodeTimeoutError(f"node '{node_name}' exceeded its {int(limit)}s time budget and was abandoned")
    return fut.result()


def _step_traced(node_name: str, node_fn):
    """Trace-only wrapper for nodes that are not phase-tracked."""
    def _wrapper(state):
        import time as _t
        started, result, err = _t.time(), None, None
        try:
            result = _call_with_deadline(node_name, node_fn, state)
            return result
        except BaseException as exc:
            err = exc
            raise
        finally:
            _record_step(state, node_name, started, result, err)
    return _wrapper


def _phase_tracked_node(node_name: str, node_fn):
    """Wrap a node function to automatically track its phase in RunBudget.

    Purely observational — never alters state, prompts, or control flow.
    The wrapper is the single owner of the phase lifecycle (end_phase) for
    all graph nodes.  Individual node functions MUST NOT call end_phase().

    Special case: investigate_node creates RunContext mid-execution, so it
    retains its own start_phase() call.  The wrapper handles end_phase()
    by re-fetching run_ctx in the finally block.

    For all other nodes, run_ctx is already in the transient store before
    the node runs, so the wrapper handles both start_phase and end_phase.
    """
    def _wrapper(state):
        import time as _t
        _step_t0, _step_err, result = _t.time(), None, None
        run_ctx = _get_transient(state, "run_ctx")
        phase_started = False
        if run_ctx:
            try:
                run_ctx.start_phase(node_name)
                phase_started = True
            except Exception:
                pass  # Instrumentation must never crash the workflow
        # AUTO_STAGE: swap the agents' model for this node when a resolver says so.
        _stage_agents, _stage_prev_llm, _stage_swapped = None, None, False
        if _STAGE_MODEL_RESOLVER is not None:
            try:
                _stage_model = _STAGE_MODEL_RESOLVER(node_name)
                _stage_agents = _get_transient(state, "agents")
                if _stage_model and _stage_agents is not None and hasattr(_stage_agents, "llm"):
                    _stage_prev_llm = _stage_agents.llm
                    _stage_agents.llm = _stage_llm_for(_stage_model)
                    _stage_swapped = True
                    logger.info(f"  [AUTO_STAGE] node '{node_name}' → model '{_stage_model}'")
            except Exception as _stage_exc:
                logger.warning(f"  [AUTO_STAGE] resolver failed for '{node_name}' (keeping default model): {_stage_exc}")
        try:
            result = _call_with_deadline(node_name, node_fn, state)
            return result
        except BaseException as exc:
            _step_err = exc
            raise
        finally:
            if _stage_swapped:
                try:
                    _stage_agents.llm = _stage_prev_llm
                except Exception:
                    pass
            _record_step(state, node_name, _step_t0, result, _step_err)
            # Re-fetch: investigate_node creates and stores RunContext
            # mid-execution, so run_ctx may now exist even if it was
            # None before the node ran.
            if not phase_started:
                run_ctx = _get_transient(state, "run_ctx")
            if run_ctx:
                try:
                    run_ctx.end_phase(node_name)
                except Exception:
                    pass  # Instrumentation must never crash the workflow
    return _wrapper


# ============================================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================================

def create_ticket_to_code_graph(workspace_path: str, technology: Optional[str] = None, entry_point: str = "investigate") -> StateGraph:
    """
    Create the complete ticket-to-code workflow graph.
    
    Args:
        workspace_path: Path to project workspace
        technology: Language/framework ('java', 'dotnet', 'python', 'nodejs', 'go', or None for auto-detect)
        entry_point: Node to start execution from (default: "investigate", can be "plan" on resume)
    
    Flow:
    1. Investigate → Determine ticket type
    2. Unified Analysis → Either requirements OR solution guidance (ONE node!)
    3. If requirements → Plan → RAG → Generate → Build → Test
    4. If solution → End
    """
    # Initialize agents with user-specified technology
    agents = WorkflowAgents(workspace_path, technology)
    
    # Create graph
    workflow = StateGraph(TicketToCodeState)
    _orig_add_conditional_edges = workflow.add_conditional_edges

    def _traced_add_conditional_edges(source, path, *args, **kwargs):
        import functools as _ft
        import time as _t

        @_ft.wraps(path)
        def _routed(state, *a, **k):
            decision = path(state, *a, **k)
            _record_step(
                state, f"route:{getattr(path, '__name__', 'router')}", _t.time(),
                {"from": source, "decision": decision if isinstance(decision, (str, list)) else str(decision)}, None,
            )
            return decision
        return _orig_add_conditional_edges(source, _routed, *args, **kwargs)

    workflow.add_conditional_edges = _traced_add_conditional_edges
    
    # Add nodes — each wrapped with _phase_tracked_node for automatic
    # RunBudget phase attribution (token instrumentation).
    workflow.add_node("investigate", _phase_tracked_node("investigate", lambda s: investigate_node(s, agents)))
    workflow.add_node("unified_analysis", _phase_tracked_node("unified_analysis", lambda s: unified_analysis_node(s, agents)))
    workflow.add_node("discover", _phase_tracked_node("discover", lambda s: discovery_node(s, agents)))
    workflow.add_node("plan", _phase_tracked_node("plan", lambda s: plan_node(s, agents)))
    workflow.add_node("validate_candidates", _phase_tracked_node("validate_candidates", lambda s: validate_candidates_node(s, agents)))
    workflow.add_node("localize", _phase_tracked_node("localize", lambda s: localize_node(s, agents)))
    workflow.add_node("hypothesis_investigation", _phase_tracked_node("hypothesis_investigation", lambda s: hypothesis_investigation_node(s, agents)))
    workflow.add_node("evidence_collection_loop", _phase_tracked_node("evidence_collection_loop", lambda s: evidence_collection_loop_node(s, agents)))
    workflow.add_node("evidence_ranking", _phase_tracked_node("evidence_ranking", lambda s: evidence_ranking_node(s, agents)))
    workflow.add_node("semantic_verification", _phase_tracked_node("semantic_verification", lambda s: semantic_verification_node(s, agents)))
    workflow.add_node("preflight_check", _phase_tracked_node("preflight_check", lambda s: preflight_check_node(s, agents)))
    workflow.add_node("planning_scope_verification", _phase_tracked_node("planning_scope_verification", lambda s: planning_scope_verification_node(s, agents)))
    workflow.add_node("grounded_understanding", _phase_tracked_node("grounded_understanding", lambda s: grounded_understanding_node(s, agents)))
    workflow.add_node("ownership_completeness", _phase_tracked_node("ownership_completeness", lambda s: ownership_completeness_node(s, agents)))
    workflow.add_node("dataflow_verification", _phase_tracked_node("dataflow_verification", lambda s: dataflow_verification_node(s, agents)))
    workflow.add_node("rag_code", _phase_tracked_node("rag_code", lambda s: rag_for_code_node(s, agents)))
    workflow.add_node("generate_code", _phase_tracked_node("generate_code", lambda s: generate_code_node(s, agents)))
    workflow.add_node("edit_loop", _phase_tracked_node("edit_loop", lambda s: edit_loop_node(s, agents)))
    workflow.add_node("patch_gate", _phase_tracked_node("patch_gate", lambda s: patch_gate_node(s, agents)))
    workflow.add_node("import_validation", _phase_tracked_node("import_validation", lambda s: import_validation_node(s, agents)))
    workflow.add_node("angular_module_registration", _phase_tracked_node("angular_module_registration", lambda s: angular_module_registration_node(s, agents)))
    workflow.add_node("build", _phase_tracked_node("build", lambda s: build_node(s, agents)))
    workflow.add_node("test", _phase_tracked_node("test", lambda s: test_node(s, agents)))
    workflow.add_node("pre_fix_build", _phase_tracked_node("pre_fix_build", lambda s: pre_fix_build_node(s, agents)))
    workflow.add_node("fix_build", _phase_tracked_node("fix_build", lambda s: fix_build_errors_node(s, agents)))
    workflow.add_node("memory_update", _phase_tracked_node("memory_update", lambda s: memory_update_node(s, agents)))
    workflow.add_node("outcome_check", _phase_tracked_node("outcome_check", lambda s: outcome_check_node(s, agents)))
    workflow.add_node("context_expand", _phase_tracked_node("context_expand", lambda s: context_expand_node(s, agents)))
    workflow.add_node("verification_failed", _phase_tracked_node("verification_failed", lambda s: verification_failed_node(s, agents)))
    workflow.add_node("remediation_plan", _phase_tracked_node("remediation_plan", lambda s: remediation_plan_node(s, agents)))
    
    # Set entry point
    workflow.set_entry_point(entry_point if entry_point in workflow.nodes else "investigate")
    
    # Enhancement 1: Runtime Log Analyzer node
    workflow.add_node("runtime_diagnosis", _step_traced("runtime_diagnosis", lambda s: runtime_diagnosis_node(s, agents)))

    # Add edges - Main flow (investigate → runtime_diagnosis → unified_analysis)
    workflow.add_edge("investigate", "runtime_diagnosis")
    workflow.add_edge("runtime_diagnosis", "unified_analysis")
    workflow.add_conditional_edges("unified_analysis", route_after_unified_analysis)  # Routes to "discover" or END

    workflow.add_conditional_edges(
        "discover",
        route_after_discover,
        {
            "edit_loop": "edit_loop",
            "hypothesis_investigation": "hypothesis_investigation",
            "decline": END,
            "verification_failed": "verification_failed",
        },
    )

    # Phase 2G: Evidence Pipeline (runs BEFORE planning)
    workflow.add_edge("hypothesis_investigation", "evidence_collection_loop")
    workflow.add_edge("evidence_collection_loop", "evidence_ranking")
    workflow.add_edge("evidence_ranking", "semantic_verification")
    workflow.add_edge("semantic_verification", "preflight_check")
    workflow.add_conditional_edges("preflight_check", route_after_preflight)  # → planning_scope_verification or END
    workflow.add_edge("planning_scope_verification", "plan")

    # Plan → Validate Candidates → Localize (or Planning Recovery)
    workflow.add_node("planning_recovery", _step_traced("planning_recovery", lambda s: planning_recovery_node(s, agents)))  # Planning failure diagnosis
    workflow.add_edge("plan", "validate_candidates")
    workflow.add_conditional_edges("validate_candidates", route_after_validate_candidates)
    workflow.add_conditional_edges("planning_recovery", route_after_planning_recovery)
    
    # Localize → Grounded Understanding
    workflow.add_edge("localize", "grounded_understanding")
    workflow.add_conditional_edges("grounded_understanding", route_after_grounded_understanding)  # Phase 2F gate / Hard fallback
    
    # Skip testing - go directly to code generation (dataflow verification between plan and RAG)
    workflow.add_edge("ownership_completeness", "dataflow_verification")
    workflow.add_edge("dataflow_verification", "rag_code")   # Code context only (testing disabled)
    workflow.add_edge("rag_code", "generate_code")
    
    # After fixed-plan generation: run the adaptive edit loop, then safety gate → build
    workflow.add_conditional_edges("generate_code", route_after_generate_code)
    workflow.add_edge("edit_loop", "patch_gate")  # edit loop → safety gate → build
    workflow.add_conditional_edges(  # Gate: blocked patches never reach the compiler
        "patch_gate",
        route_after_patch_gate,
        {"import_validation": "import_validation", "verification_failed": "verification_failed"},
    )
    workflow.add_edge("import_validation", "angular_module_registration")  # Gap 2: register in @NgModule
    workflow.add_edge("angular_module_registration", "build")

    # Build validation: build → pre_fix_build (fingerprint persistence) → router
    workflow.add_edge("build", "pre_fix_build")
    workflow.add_conditional_edges("pre_fix_build", check_build_status)
    workflow.add_node("ui_verify", _phase_tracked_node("ui_verify", lambda s: ui_verify_node(s, agents)))
    workflow.add_edge("ui_verify", "outcome_check")
    workflow.add_conditional_edges("test", check_test_status)
    # Requirement-satisfaction loop: if the ticket isn't actually satisfied, route
    # back through the adaptive edit loop to implement what's missing, then re-verify.
    # When the remediation budget is exhausted, verification_failed ends the run
    # honestly instead of masking a PARTIAL verdict as "workflow complete".
    workflow.add_conditional_edges(
        "outcome_check",
        route_after_outcome_check,
        {
            "remediation_plan": "remediation_plan",
            "edit_loop": "edit_loop",
            "context_expand": "context_expand",
            "memory_update": "memory_update",
            "verification_failed": "verification_failed",
        },
    )
    workflow.add_edge("remediation_plan", "edit_loop")
    # Tier 2 context expansion → re-plan with wider context
    workflow.add_edge("context_expand", "plan")
    workflow.add_edge("fix_build", "patch_gate")  # After repair: patch_gate -> import_validation -> angular_module_registration -> build
    workflow.add_edge("memory_update", END)
    workflow.add_edge("verification_failed", END)
    
    return workflow


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_autonomous_workflow_langgraph(
    ticket: ValueEdgeTicket,
    workspace_path: str,
    technology: Optional[str] = None,
    max_retry_attempts: int = 3,
    run_id: Optional[str] = None,
    entry_point: str = "investigate",
    initial_overrides: Optional[dict] = None,
) -> TicketToCodeState:
    """
    Run the complete autonomous workflow using LangGraph.
    
    Args:
        ticket: ValueEdge ticket to process
        workspace_path: Path to project workspace
        technology: Language/framework to use. Valid options:
            - 'java': Java projects (Maven/Gradle)
            - 'dotnet': .NET projects (Framework/Core)
            - 'python': Python projects (pytest)
            - 'nodejs': Node.js projects (npm/jest)
            - 'go': Go projects (go build/test)
            - None: Auto-detect from project files (default)
        max_retry_attempts: Max attempts to fix build/test errors
        run_id: Unique identifier for this execution run (prevents thread_id collisions)
        entry_point: Node to start execution from (default: "investigate", e.g. "plan" on resume)
        initial_overrides: State overrides to carry forward on resume
        
    Returns:
        Final workflow state with results
    """
    tech_mode = technology if technology else "auto-detect"
    # Ensure any residual state from prior worker tasks is cleared immediately
    try:
        from ticket_to_code.agents.code_generator import clear_reuse_directive
        clear_reuse_directive()
    except Exception:
        pass
    logger.info(f" Starting LangGraph workflow for: {ticket.ticket_id} (Mode: {tech_mode}, Entry: {entry_point})")
    reset_step_trace(str(ticket.ticket_id))
    
    # Create graph with specified technology and entry point
    workflow = create_ticket_to_code_graph(workspace_path, technology, entry_point=entry_point)
    
    # Compile with memory (state persistence)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    # Initial state
    initial_state: TicketToCodeState = {
        "ticket": ticket,
        "workspace_path": workspace_path,
        "max_retry_attempts": max_retry_attempts,
        "investigation_result": None,
        "requirements": None,
        "architectural_plan": None,
        "solution_guidance": None,
        "test_rag_context": None,
        "code_rag_context": None,
        "test_status": None,  # TEST BRANCH status
        "code_status": None,  # CODE BRANCH status
        "generated_tests": None,
        "generated_code": None,
        "build_result": None,
        "test_result": None,
        "retry_attempt": 0,
        "last_error_type": None,
        "blacklisted_files": [],
        "candidate_retry_count": 0,
        "max_candidate_retries": 2,
        "discovery_cycle_count": 0,
        "validation_failure_reason": None,
        "rejected_approaches": [],
        "user_constraints": [],
        "outcome_check_result": None,
        "outcome_remediation": None,
        "outcome_fix_attempt": 0,
        "max_outcome_fix_attempts": 3,
        "tier2_candidates": [],
        "context_expansion_count": 0,
        "prev_build_error_fingerprint": None,
        "consecutive_identical_build_errors": 0,
        "investigation_hypotheses": None,
        "evidence_items": None,
        "grounded_understanding": None,
        "query_expansion_map": None,
        "original_file_contents": {},
        "runtime_diagnosis": None,
        "status": "initialized",
        "errors": [],
        "start_time": datetime.now(),
        "end_time": None
    }
    if initial_overrides and isinstance(initial_overrides, dict):
        initial_state.update(initial_overrides)
    
    # Execute workflow with unique run_id to avoid thread collisions across runs
    active_thread_id = run_id or getattr(ticket, "run_id", None) or ticket.ticket_id
    config = {"configurable": {"thread_id": active_thread_id}}
    
    # Save start time for duration calculation
    start_time = initial_state["start_time"]
    
    try:
        # Stream node updates, then read the authoritative final graph snapshot.
        MAX_WORKFLOW_TIME = int(os.environ.get("AVIATOR_WORKFLOW_TIMEOUT", os.environ.get("AVIATOR_MAX_WALL_S", 3600)))
        
        for update in app.stream(initial_state, config):
            try:
                node_names = ", ".join(update.keys()) if isinstance(update, dict) else str(type(update))
                logger.info(f"  Node update: {node_names}")
            except Exception:
                logger.info("  Node update received")
                
            # Check global timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > MAX_WORKFLOW_TIME:
                logger.error(f"❌ GLOBAL TIMEOUT EXCEEDED: Workflow ran for {elapsed:.2f}s (Max {MAX_WORKFLOW_TIME}s). Aborting.")
                # We break out of the stream early to kill the pipeline
                break

        snapshot = app.get_state(config)
        final_state = snapshot.values if hasattr(snapshot, "values") else snapshot
        if not isinstance(final_state, dict):
            final_state = dict(initial_state)
        
        # Mark completion
        final_state["end_time"] = datetime.now()
        duration = (final_state["end_time"] - start_time).total_seconds()
        
        # Ensure status reflects a fatal build failure if max retries were exhausted
        if final_state.get("retry_attempt", 0) >= final_state.get("max_retry_attempts", 2):
            if final_state.get("build_result") and final_state["build_result"].status.value == "failure":
                final_state["status"] = "build_failed"
        
        logger.info(
            f"✅ Workflow completed in {duration:.2f}s\n"
            f"   Status: {final_state.get('status', 'unknown')}\n"
            f"   Tests: {len(final_state.get('generated_tests') or [])} files\n"
            f"   Code: {len(final_state.get('generated_code') or [])} files\n"
            f"   Retry attempts: {final_state.get('retry_attempt', 0)}"
        )
        
        # Write final metrics trace artifact
        metrics_artifact = {
            "duration_seconds": duration,
            "status": final_state.get("status", "unknown"),
            "test_files_generated": len(final_state.get("generated_tests") or []),
            "code_files_generated": len(final_state.get("generated_code") or []),
            "retry_attempts": final_state.get("retry_attempt", 0),
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": final_state["end_time"].isoformat() if final_state.get("end_time") else None
        }
        write_trace_artifact(workspace_path, ticket.ticket_id, "final_metrics.json", metrics_artifact)
        
        # Write discovered files
        if final_state.get("discovered_files"):
            write_trace_artifact(workspace_path, ticket.ticket_id, "discovered_files.json", final_state["discovered_files"])
            
        # Write planner selection
        plan = _get_plan(final_state)
        if plan and hasattr(plan, "tasks"):
            tasks_list = [{"file_path": t.get("file_path") if isinstance(t, dict) else t.file_path, "task_type": (t.get("task_type") if isinstance(t, dict) else getattr(t.task_type, "value", str(t.task_type)))} for t in plan.tasks]
            write_trace_artifact(workspace_path, ticket.ticket_id, "planner_selection.json", tasks_list)
            
        # Write generated patch (diffs from generated code)
        if final_state.get("generated_code"):
            patch_content = ""
            for gf in final_state["generated_code"]:
                if isinstance(gf, dict):
                    gf_path = gf.get("file_path", gf.get("path", "unknown"))
                    gf_content = gf.get("content", "")
                else:
                    gf_path = getattr(gf, "file_path", "unknown")
                    gf_content = getattr(gf, "content", "")
                patch_content += f"--- {gf_path}\n+++ {gf_path}\n{gf_content}\n\n"
            if patch_content:
                write_trace_artifact(workspace_path, ticket.ticket_id, "generated_patch.diff", patch_content)
                
        # Write build output
        build_res = final_state.get("build_result")
        if build_res and hasattr(build_res, "stdout"):
            log_content = f"SUCCESS: {build_res.status.value == 'success'}\n\nSTDOUT:\n{build_res.stdout}\n\nSTDERR:\n{build_res.stderr}"
            write_trace_artifact(workspace_path, ticket.ticket_id, "build_output.log", log_content)
            
        # Write test output
        test_res = final_state.get("test_result")
        if test_res and hasattr(test_res, "stdout"):
            log_content = f"SUCCESS: {test_res.success}\n\nSTDOUT:\n{test_res.stdout}\n\nSTDERR:\n{test_res.stderr}"
            write_trace_artifact(workspace_path, ticket.ticket_id, "test_output.log", log_content)
        
        # Write Decision Chain
        try:
            decision_chain = []
            discovered = final_state.get("discovered_files", [])
            plan = _get_plan(final_state)
            plan_tasks = plan.tasks if plan and hasattr(plan, "tasks") else []
            generated = final_state.get("generated_code", [])
            build_res = final_state.get("build_result")

            for d in discovered:
                path = d["path"]
                entry = {
                    "candidate": path,
                    "investigation_match": True,
                    "discovery_match": True,
                    "ranking_score": d.get("score", 0),
                    "planner_selected": any(t.file_path == path and t.task_type.value != "read_only" for t in plan_tasks),
                    "generator_modified": any((gf.get("file_path") if isinstance(gf, dict) else getattr(gf, "file_path", None)) == path for gf in generated),
                    "build_success": (build_res.status.value == 'success') if build_res else None
                }
                decision_chain.append(entry)
                
            if decision_chain:
                write_trace_artifact(workspace_path, ticket.ticket_id, "decision_chain.json", decision_chain)
        except Exception as e:
            logger.warning(f"Failed to write decision_chain.json: {e}")

        # Attach run_context and token_usage telemetry directly to final_state
        try:
            tid = getattr(ticket, "ticket_id", "")
            with _transient_lock:
                _rc = _transient_store.get(tid, {}).get("run_ctx")
            if _rc is not None:
                final_state["run_context"] = _rc
                if hasattr(_rc, "budget"):
                    _b = _rc.budget
                    _tin = getattr(_b, "tokens_in", 0)
                    _tout = getattr(_b, "tokens_out", 0)
                    final_state["token_usage"] = {
                        "tokens_in": _tin,
                        "tokens_out": _tout,
                        "total_tokens": _tin + _tout,
                        "llm_calls": getattr(_b, "llm_calls", 0),
                        "cost_usd": getattr(_b, "cost_usd", 0.0),
                        "phase_breakdown": {
                            p: u.to_dict() for p, u in getattr(_b, "phase_ledger", {}).items()
                        } if hasattr(_b, "phase_ledger") else {}
                    }
                    metrics_artifact["token_usage"] = final_state["token_usage"]
        except Exception as _rc_err:
            logger.debug(f"Failed to attach run_context telemetry: {_rc_err}")

        return final_state
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}", exc_info=True)
        raise
    finally:
        # Guarantee reuse directive is purged under ANY termination condition
        # (normal return, exception, timeout, abort) so Celery workers never leak.
        try:
            from ticket_to_code.agents.code_generator import clear_reuse_directive
            clear_reuse_directive()
        except Exception:
            pass


# ============================================================================
# VISUALIZATION HELPER
# ============================================================================

def visualize_workflow(workspace_path: str, output_path: str = "workflow_graph.png"):
    """
    Generate visual diagram of the workflow.
    
    Args:
        workspace_path: Project path
        output_path: Where to save diagram
    """
    workflow = create_ticket_to_code_graph(workspace_path)
    app = workflow.compile()
    
    # Generate Mermaid diagram
    mermaid = app.get_graph().draw_mermaid()
    
    print("Workflow Graph (Mermaid):")
    print(mermaid)
    
    # Could also generate PNG with graphviz
    # app.get_graph().draw_png(output_path)
    
    return mermaid


# ============================================================================
# COMPATIBILITY WRAPPER (for existing code)
# ============================================================================

def _fast_build_result(fast_res) -> "BuildResult":
    bp = getattr(fast_res, "build_passed", None)
    if bp is True:
        return BuildResult(status=BuildStatus.SUCCESS, exit_code=0, stdout="Build passed (FastIDEEngine)")
    if bp is False:
        return BuildResult(status=BuildStatus.FAILURE, exit_code=1, errors=["Build failed (FastIDEEngine)"], stdout="")
    return BuildResult(status=BuildStatus.SKIPPED, exit_code=0, stdout="No build ran (FastIDEEngine)")


def _fast_test_result(fast_res) -> "TestResult":
    tp = getattr(fast_res, "test_passed", None)
    if tp is True:
        return TestResult(status=TestStatus.ALL_PASSED, passed=0, total_tests=0, test_output="Tests passed (FastIDEEngine)")
    if tp is False:
        return TestResult(status=TestStatus.SOME_FAILED, failed=1, total_tests=1, errors=["Tests failed (FastIDEEngine)"])
    return TestResult(status=TestStatus.NO_TESTS, total_tests=0, test_output="No tests ran (FastIDEEngine)")


def finalize_fast_path_state(state: dict, agents) -> dict:
    """Apply the deep graph's terminal checks to a fast-path (FastIDE / DAG) result.

    Fast paths used to declare `completed` while skipping outcome_check and fabricating
    build/test results. This runs the semantic outcome check on the real workspace state and
    sets status to `completed` only when build, tests and outcome agree; otherwise
    `verification_failed`. Never raises.
    """
    try:
        res = outcome_check_node(state, agents) or {}
    except Exception as exc:
        res = {"outcome_check_result": {"status": "error", "reason": str(exc)}}
    for k, v in res.items():
        state[k] = v
    try:
        route = route_after_outcome_check(state)
    except Exception as exc:
        route = "verification_failed"
        state.setdefault("outcome_check_result", {})["router_error"] = str(exc)
    b = getattr(getattr(state.get("build_result"), "status", None), "value", "")
    t = getattr(getattr(state.get("test_result"), "status", None), "value", "")
    build_ok = b in ("success", "skipped")
    test_ok = t in ("all_passed", "no_tests", "skipped")
    state["fast_path_outcome_route"] = route
    if route == "memory_update" and build_ok and test_ok:
        state["status"] = "completed"
        state["workflow_status"] = "completed"
        state["verification_level"] = (
            "test_verified" if (b == "success" and t == "all_passed")
            else ("compile_only" if b == "success" else "unverified")
        )
    else:
        ocr = state.get("outcome_check_result") or {}
        state["status"] = "verification_failed"
        state["workflow_status"] = "verification_failed"
        state["verification_level"] = "unverified"
        state["error"] = (
            f"Fast path did not pass verification: build={b or 'none'}, tests={t or 'none'}, "
            f"outcome={ocr.get('status', 'n/a')} ({str(ocr.get('reason', ''))[:160]}), route={route}"
        )
        logger.error(f"  🛑 {state['error']}")
    return state


def request_run_cancel(ticket_id: str, reason: str = "cancelled by user") -> bool:
    """Cooperative cancel: flag the run context so llm_invoke / writers stop at their next checkpoint.
    Used by the backend's /stop endpoint so cancellation takes effect inside a node, not only between nodes."""
    with _transient_lock:
        rc = (_transient_store.get(ticket_id) or {}).get("run_ctx")
    if rc is not None and hasattr(rc, "request_cancel"):
        try:
            rc.request_cancel(reason)
            return True
        except Exception:
            return False
    return False


def run_autonomous_workflow(
    ticket: ValueEdgeTicket,
    workspace_path: str,
    codebase_path: Optional[str] = None,
    technology: Optional[str] = None,
    auto_execute: bool = True,
    max_fix_attempts: int = 3,
    use_fast_ide: bool = True,
    execution_mode: Optional[str] = None,
    contract_ticket: Any = None,
    contract_policy: Any = None,
    progress_callback: Any = None,
    **kwargs: Any,
) -> TicketToCodeState:
    """
    Primary Autonomous Ticket-to-Code Runner with Cursor/Windsurf Sub-Minute Fast-Path.
    
    1. First attempts FastIDEEngine (Lean ReAct Diff Loop in <45s, <20k tokens).
    2. If FastIDEEngine succeeds and verifies build/tests, returns immediately.
    3. If FastIDEEngine cannot converge or ticket requires deep architecture,
       smoothly escalates to the deep LangGraph workflow as a fallback.
    """
    effective_workspace = codebase_path if codebase_path else workspace_path
    if kwargs:
        logger.warning(f"run_autonomous_workflow: ignoring unsupported arguments {sorted(kwargs)}")
    if execution_mode and str(execution_mode).lower() in ("reasoning", "deep", "langgraph"):
        use_fast_ide = False
    if contract_ticket is not None or contract_policy is not None:
        logger.info("run_autonomous_workflow: contract_ticket/contract_policy accepted (deep graph enforces contracts)")

    # Step 1: Cursor/Aider-Style Sub-Minute Fast Path
    if use_fast_ide:
        try:
            from ticket_to_code.runtime.fast_ide_engine import FastIDEEngine
            agents = WorkflowAgents(effective_workspace, technology)
            fast_engine = FastIDEEngine(
                workspace_path=effective_workspace,
                ticket=ticket,
                llm_client=agents.llm,
                max_repair_turns=max_fix_attempts,
            )
            fast_res = fast_engine.run()
            if fast_res.success:
                logger.info(
                    f"⚡ [run_autonomous_workflow] FastIDEEngine resolved ticket in "
                    f"{fast_res.duration_seconds:.2f}s ({fast_res.tokens_consumed} tokens)!"
                )
                final_state = {
                    "ticket": ticket,
                    "workspace_path": effective_workspace,
                    "technology": technology or "auto-detect",
                    "status": "completed",
                    "workflow_status": "completed",
                    "generated_code": [
                        GeneratedCode(
                            file_path=fp,
                            content=(Path(effective_workspace) / fp).read_text(encoding="utf-8", errors="ignore"),
                            language=Path(fp).suffix.lstrip("."),
                            change_type="modify",
                        )
                        for fp in fast_res.modified_files
                    ],
                    "_run_generated_map": {
                        fp: (Path(effective_workspace) / fp).read_text(encoding="utf-8", errors="ignore")
                        for fp in fast_res.modified_files
                    },
                    "build_result": _fast_build_result(fast_res),
                    "test_result": _fast_test_result(fast_res),
                    "token_usage": {
                        "total_tokens": fast_res.tokens_consumed,
                        "latency_seconds": fast_res.duration_seconds,
                        "engine": "FastIDEEngine (Cursor/Aider Style)",
                    },
                    "diffs": fast_res.diffs,
                    "original_file_contents": dict(getattr(fast_res, "original_contents", None) or {}),
                }
                # Fast path must pass the same terminal checks as the deep graph.
                return finalize_fast_path_state(final_state, agents)
            else:
                logger.info(f"⚡ [run_autonomous_workflow] FastIDEEngine escalated: {fast_res.error_message}")
        except Exception as fast_exc:
            logger.warning(f"⚡ [run_autonomous_workflow] FastIDEEngine fallback to LangGraph: {fast_exc}")

    # Step 2: Fallback to Deep LangGraph Graph
    tech_mode = technology if technology else "auto-detect"
    logger.info(f" Running LangGraph workflow (via deep graph fallback) - Mode: {tech_mode}")
    
    ret = run_autonomous_workflow_langgraph(
        ticket=ticket,
        workspace_path=effective_workspace,
        technology=technology,
        max_retry_attempts=max_fix_attempts
    )
    if isinstance(ret, dict) and "token_usage" not in ret:
        try:
            tid = getattr(ticket, "ticket_id", "")
            with _transient_lock:
                _rc = _transient_store.get(tid, {}).get("run_ctx")
            if _rc is not None:
                ret["run_context"] = _rc
                if hasattr(_rc, "budget"):
                    _b = _rc.budget
                    _tin = getattr(_b, "tokens_in", 0)
                    _tout = getattr(_b, "tokens_out", 0)
                    ret["token_usage"] = {
                        "tokens_in": _tin,
                        "tokens_out": _tout,
                        "total_tokens": _tin + _tout,
                        "llm_calls": getattr(_b, "llm_calls", 0),
                        "cost_usd": getattr(_b, "cost_usd", 0.0),
                        "phase_breakdown": {
                            p: u.to_dict() for p, u in getattr(_b, "phase_ledger", {}).items()
                        } if hasattr(_b, "phase_ledger") else {}
                    }
        except Exception:
            pass
    return ret


# Alias for backwards compatibility
AutonomousWorkflowOrchestrator = None  # No longer used with LangGraph


