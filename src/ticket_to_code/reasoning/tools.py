"""
Tool layer for the Enterprise dynamic reasoning engine.

Each Tool wraps an EXISTING capability (RAG, repo search, localization,
generation, build, outcome verification) or the Enterprise Brain. The reasoning
engine decides *when* to call each tool — the tools do not dictate control flow.

Design goals:
  - Real functionality: tools call the already-built agents/engines.
  - Safe: every tool catches its own errors and returns a string observation.
  - Compact observations: results are trimmed so the reasoner context stays lean.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ticket_to_code.reasoning.belief_state import (
    BeliefState,
    Contradiction,
    EvidenceItem,
    Objective,
    Observation,
)

from ticket_to_code.runtime.limits import TOOL_OUTPUT_MAX_CHARS, truncate

logger = logging.getLogger(__name__)

# Kept as an alias for callers/tests that still reference the old name.
_MAX_OBS_CHARS = TOOL_OUTPUT_MAX_CHARS


# ── Structured patch rejection reasons ───────────────────────────────────────
PATCH_REJECTION_EMPTY_CONTENT = "empty_content"
PATCH_REJECTION_HALLUCINATED_PATH = "hallucinated_path"
PATCH_REJECTION_OUTSIDE_WRITABLE = "outside_writable_scope"
PATCH_REJECTION_FILE_NOT_FOUND = "file_not_found"
PATCH_REJECTION_INVALID_PATCH = "invalid_patch"
PATCH_REJECTION_VERIFICATION_FAILED = "verification_failed"
PATCH_REJECTION_PATCH_APPLY_FAILED = "patch_apply_failed"
PATCH_ACCEPTED = "accepted"


@dataclass
class WritePatchEvidence:
    """Structured evidence captured for every write_patch invocation."""
    # Input context
    ticket_id: str = ""
    attempt_number: int = 0
    candidate_files: List[str] = field(default_factory=list)
    selected_target_file: str = ""
    target_file_exists: bool = False
    in_candidate_set: bool = False
    patch_type: str = ""  # create | modify | delete
    content_length: int = 0
    content_snippet: str = ""
    # Validation outcome
    accepted: bool = False
    rejection_reason: str = ""
    # Patch effectiveness
    original_file_hash: str = ""
    patched_file_hash: str = ""
    effective_diff_lines: int = 0
    bytes_changed: int = 0


@dataclass
class VerificationEvidence:
    """Structured evidence captured for every verify_outcome invocation."""
    verification_executed: bool = False
    verifier_used: str = ""
    passed: bool = False
    rejection_reason: str = ""
    mode: str = ""
    detail: str = ""


@dataclass
class ToolContext:
    """Shared state passed to every tool invocation."""
    ticket: Any
    workspace_path: str
    agents: Any                      # WorkflowAgents instance (existing capabilities)
    brain: Any                       # RepositoryBrain instance
    brain_manager: Any = None        # BrainManager (knowledge indexes)
    repository_brain: Any = None     # RepositoryBrainStorage (repo-local owner hints)
    ticket_type: str = ""
    feature_graph: Any = None        # FeatureGraph instance (feature -> owner/consumer files)
    experience_store: Any = None     # ExperienceStore (dynamic policy memory)
    files_read: Dict[str, str] = field(default_factory=dict)   # path -> content cache
    candidate_files: Dict[str, float] = field(default_factory=dict)  # path -> confidence
    written_files: Dict[str, str] = field(default_factory=dict)      # path -> content
    # Build state (used by the engine to enforce a build-fix loop before finishing)
    last_build_status: str = ""            # "" | success | failure | error | timeout | skipped
    last_build_errors: List[str] = field(default_factory=list)
    build_attempts: int = 0
    # Verification state (used by selector/engine to avoid premature finalization)
    last_verify_passed: bool = False
    last_verify_mode: str = ""
    last_verify_reason: str = ""
    # Capability execution trail (P4-ready telemetry)
    capability_history: List[dict] = field(default_factory=list)
    # New runtime state (evidence before belief)
    observations: List[Observation] = field(default_factory=list)
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    working_theories: Dict[str, Any] = field(default_factory=dict)  # key -> WorkingTheory-like dict
    objectives: List[Objective] = field(default_factory=list)
    belief_state: BeliefState = field(default_factory=BeliefState)
    # Structured evidence telemetry
    write_patch_evidence: List[WritePatchEvidence] = field(default_factory=list)
    verification_evidence: List[VerificationEvidence] = field(default_factory=list)
    # Stage timing (stage_name -> cumulative_seconds)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    current_stage: str = "initialization"


@dataclass(frozen=True)
class CapabilityMetadata:
    """Cost/benefit model for capability selection.

    The selector uses these fields to pick the cheapest capability that can
    satisfy the current missing need.
    """
    cost: int
    confidence_gain: float
    risk: float = 0.2
    prerequisites: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[ToolContext, dict], str]
    metadata: CapabilityMetadata

    def run(self, ctx: ToolContext, args: dict) -> str:
        start = time.perf_counter()
        try:
            out = self.func(ctx, args or {})
            text = str(out) if out is not None else "(no output)"
            result = truncate(text, TOOL_OUTPUT_MAX_CHARS)
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            ctx.capability_history.append(
                {
                    "capability": self.name,
                    "status": "ok",
                    "elapsed_ms": elapsed_ms,
                    "cost": self.metadata.cost,
                    "confidence_gain": self.metadata.confidence_gain,
                }
            )
            return result
        except Exception as exc:  # tools must never crash the reasoner
            logger.warning("Tool '%s' error: %s", self.name, exc)
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            ctx.capability_history.append(
                {
                    "capability": self.name,
                    "status": "error",
                    "elapsed_ms": elapsed_ms,
                    "cost": self.metadata.cost,
                    "confidence_gain": self.metadata.confidence_gain,
                    "error": str(exc),
                }
            )
            return f"ERROR in tool '{self.name}': {exc}"


# ── individual tool implementations ─────────────────────────────────────────

def _t_recall_memory(ctx: ToolContext, args: dict) -> str:
    """Recall similar past tickets and suggested owner files from the Repository Brain."""
    text = f"{getattr(ctx.ticket, 'title', '')} {getattr(ctx.ticket, 'description', '')}"
    similar = ctx.brain.recall_similar_tickets(text, ctx.ticket_type, limit=5)
    owners = ctx.brain.suggest_owner_files(text, ctx.ticket_type, limit=8)
    for o in owners:
        ctx.candidate_files.setdefault(o["file"], o["confidence"])
    lines = [f"Repository Brain: {ctx.brain.summary().get('tickets_solved', 0)} tickets solved previously."]
    if owners:
        lines.append("Likely owner files (learned):")
        lines += [f"  - {o['file']} (conf {o['confidence']})" for o in owners]
    if similar:
        lines.append("Similar past tickets:")
        for s in similar:
            lines.append(
                f"  - [{s.get('type')}] {s.get('title','')[:60]} → "
                f"{', '.join(s.get('files_modified', [])[:3])} ({s.get('outcome')})"
            )
    if not owners and not similar:
        lines.append("No prior repository memory for this ticket yet.")
    return "\n".join(lines)


def _t_search_code(ctx: ToolContext, args: dict) -> str:
    """Search repository for a literal string. args: {query}"""
    query = str(args.get("query", "")).strip()
    if not query:
        return "search_code needs a 'query'."
    results = ctx.agents.repo_search.search_literal(query, max_results=15) or []
    if not results:
        return f"No matches for '{query}'."
    lines = [f"Found {len(results)} file(s) containing '{query}':"]
    for r in results[:15]:
        fp = getattr(r, "file_path", getattr(r, "path", str(r)))
        ctx.candidate_files.setdefault(fp, ctx.candidate_files.get(fp, 0.4))
        lines.append(f"  - {fp}")
    return "\n".join(lines)


def _t_grep(ctx: ToolContext, args: dict) -> str:
    """Regex search across the repository. args: {pattern, max_results?}"""
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        return "grep needs a 'pattern'."
    max_results = int(args.get("max_results", 15) or 15)
    results = ctx.agents.repo_search.search_regex(pattern, max_results=max_results) or []
    if not results:
        return f"No regex matches for '{pattern}'."
    lines = [f"Regex '{pattern}' matched {len(results)} file(s):"]
    for r in results[:max_results]:
        fp = getattr(r, "file_path", getattr(r, "path", str(r)))
        line_no = getattr(r, "line_number", None)
        matched_text = getattr(r, "matched_text", "").strip()
        ctx.candidate_files.setdefault(fp, ctx.candidate_files.get(fp, 0.4))
        if line_no and matched_text:
            lines.append(f"  - {fp}:{line_no}: {matched_text[:120]}")
        else:
            lines.append(f"  - {fp}")
    return "\n".join(lines)


def _t_grep_search(ctx: ToolContext, args: dict) -> str:
    """Live regex or literal pattern search across repository with line numbers and matched content snippets.
    args: {query, is_regex?, case_insensitive?, max_results?}
    """
    query = str(args.get("query", args.get("pattern", ""))).strip()
    if not query:
        return "grep_search needs a 'query' or 'pattern'."
    is_regex = bool(args.get("is_regex", False))
    max_results = int(args.get("max_results", 20) or 20)

    if is_regex:
        results = ctx.agents.repo_search.search_regex(query, max_results=max_results) or []
    else:
        results = ctx.agents.repo_search.search_literal(query, max_results=max_results) or []

    if not results:
        return f"No matches found for '{query}'."

    lines = [f"Found {len(results)} match(es) for '{query}':"]
    for r in results[:max_results]:
        fp = getattr(r, "file_path", getattr(r, "path", str(r)))
        line_no = getattr(r, "line_number", getattr(r, "line", "?"))
        matched_text = getattr(r, "matched_text", getattr(r, "text", "")).strip()
        ctx.candidate_files.setdefault(fp, ctx.candidate_files.get(fp, 0.45))
        if matched_text:
            lines.append(f"  {fp}:{line_no}: {matched_text[:160]}")
        else:
            lines.append(f"  {fp}")
    return "\n".join(lines)


def _t_rag(ctx: ToolContext, args: dict) -> str:
    """Semantic RAG retrieval. args: {query}"""
    query = str(args.get("query", "")).strip()
    if not query:
        return "rag needs a 'query'."
    try:
        chunks = ctx.agents.rag_engine.retrieve_context(query, max_results=8) or []
    except Exception as exc:
        return f"RAG unavailable: {exc}"
    if not chunks:
        return f"RAG returned nothing for '{query}'."
    lines = [f"RAG retrieved {len(chunks)} chunk(s) for '{query}':"]
    for c in chunks[:8]:
        fp = c.get("file_path", "unknown") if isinstance(c, dict) else getattr(c, "file_path", "unknown")
        ctx.candidate_files.setdefault(fp, ctx.candidate_files.get(fp, 0.5))
        preview = (c.get("content", "") if isinstance(c, dict) else getattr(c, "content", ""))[:120]
        lines.append(f"  - {fp}: {preview.strip()}")
    return "\n".join(lines)


def _t_view_file(ctx: ToolContext, args: dict) -> str:
    """View file contents with line numbers and optional line slice range.
    args: {path, start_line?, end_line?, max_lines?}
    """
    rel = str(args.get("path", "")).strip()
    if not rel:
        return "view_file needs a 'path'."
    disk = Path(ctx.workspace_path) / rel
    if not disk.exists():
        for cand in list(ctx.candidate_files):
            if cand.replace("\\", "/").endswith(rel.replace("\\", "/")):
                disk = Path(ctx.workspace_path) / cand
                rel = cand
                break
    if not disk.exists() or not disk.is_file():
        return f"File not found: {rel}"
    text = disk.read_text(encoding="utf-8", errors="ignore")
    ctx.files_read[rel] = text
    lines = text.splitlines()
    total_lines = len(lines)

    start_line = max(1, int(args.get("start_line", 1) or 1))
    max_lines = int(args.get("max_lines", 300) or 300)
    end_line_arg = args.get("end_line")
    if end_line_arg:
        end_line = min(total_lines, int(end_line_arg))
    else:
        end_line = min(total_lines, start_line + max_lines - 1)

    start_idx = max(0, start_line - 1)
    end_idx = min(total_lines, end_line)

    logger.info(f"Analyzed {rel} #L{start_line}-{end_line}")
    formatted_lines = []
    for idx in range(start_idx, end_idx):
        line_num = idx + 1
        formatted_lines.append(f"{line_num}: {lines[idx]}")

    header = f"File: {rel} #L{start_line}-{end_line} (Showing {start_line} to {end_line}, Total lines: {total_lines}):\n"
    return header + "\n".join(formatted_lines)


def _t_read_file(ctx: ToolContext, args: dict) -> str:
    """Read a repository file on demand with optional line slicing. args: {path, start_line?, end_line?, max_lines?}"""
    rel = str(args.get("path", "")).strip()
    if not rel:
        return "read_file needs a 'path'."
    disk = Path(ctx.workspace_path) / rel
    if not disk.exists():
        # try suffix match against candidates
        for cand in list(ctx.candidate_files):
            if cand.replace("\\", "/").endswith(rel.replace("\\", "/")):
                disk = Path(ctx.workspace_path) / cand
                rel = cand
                break
    if not disk.exists() or not disk.is_file():
        return f"File not found: {rel}"
    text = disk.read_text(encoding="utf-8", errors="ignore")
    ctx.files_read[rel] = text
    lines = text.splitlines()
    total_lines = len(lines)

    start_line = max(1, int(args.get("start_line", 1) or 1))
    max_lines = int(args.get("max_lines", 200) or 200)
    end_line_arg = args.get("end_line")
    if end_line_arg:
        end_line = min(total_lines, int(end_line_arg))
    else:
        end_line = min(total_lines, start_line + max_lines - 1)

    start_idx = max(0, start_line - 1)
    end_idx = min(total_lines, end_line)

    logger.info(f"Analyzed {rel} #L{start_line}-{end_line}")
    shown = "\n".join(lines[start_idx:end_idx])
    suffix = "" if end_idx >= total_lines else f"\n... ({total_lines - end_idx} more lines)"
    prefix = "" if start_idx == 0 else f"... (skipped first {start_idx} lines)\n"
    return f"File: {rel} #L{start_line}-{end_line} (Total lines: {total_lines}):\n{prefix}{shown}{suffix}"


def _t_find_owner(ctx: ToolContext, args: dict) -> str:
    """Find the most likely owner file(s) for the ticket via discovery + brain."""
    ticket = ctx.ticket
    ticket_text = f"{getattr(ticket, 'title', '')} {getattr(ticket, 'description', '')}"
    # Brain first (learned owners) — repository intelligence advantage.
    brain_owners = ctx.brain.suggest_owner_files(ticket_text, ctx.ticket_type, limit=6)
    lines: List[str] = []
    if brain_owners:
        lines.append("Learned owners (Repository Brain):")
        for o in brain_owners:
            ctx.candidate_files[o["file"]] = max(ctx.candidate_files.get(o["file"], 0), o["confidence"])
            lines.append(f"  - {o['file']} (conf {o['confidence']})")
    # Live discovery fallback / confirmation.
    try:
        candidates = ctx.agents.localizer.discover_repository_candidates(
            ticket_text, top_n=10, rag_engine=ctx.agents.rag_engine,
        ) or []
        if candidates:
            lines.append("Live discovery candidates:")
            for c in candidates[:10]:
                fp = c.get("path") if isinstance(c, dict) else getattr(c, "path", str(c))
                conf = c.get("confidence", 0.0) if isinstance(c, dict) else getattr(c, "confidence", 0.0)
                if fp:
                    ctx.candidate_files[fp] = max(ctx.candidate_files.get(fp, 0), float(conf))
                    lines.append(f"  - {fp} (conf {round(float(conf), 3)})")
    except Exception as exc:
        lines.append(f"(discovery unavailable: {exc})")
    return "\n".join(lines) if lines else "No owner candidates found."


def _file_hash(path: Path) -> str:
    """SHA-256 of a file's content, or empty string if missing."""
    if not path.exists() or not path.is_file():
        return ""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


def _t_write_patch(ctx: ToolContext, args: dict) -> str:
    """Write new content to a repository file. args: {path, content}

    Only files present in candidate_files (grounded) OR explicitly confirmed
    may be written, to keep the agent from touching unrelated files.
    """
    rel = str(args.get("path", "")).strip()
    content = args.get("content", "")
    ticket_id = getattr(ctx.ticket, "ticket_id", "UNKNOWN")
    attempt = len(ctx.write_patch_evidence) + 1

    ev = WritePatchEvidence(
        ticket_id=ticket_id,
        attempt_number=attempt,
        candidate_files=sorted(ctx.candidate_files.keys())[:20],
        selected_target_file=rel,
        content_length=len(str(content)),
        content_snippet=str(content)[:200],
    )

    # Validate inputs
    if not rel or content == "":
        ev.rejection_reason = PATCH_REJECTION_EMPTY_CONTENT
        ctx.write_patch_evidence.append(ev)
        logger.info("write_patch_result: ticket=%s attempt=%d accepted=false reason=%s",
                    ticket_id, attempt, ev.rejection_reason)
        return "write_patch needs 'path' and 'content'."

    disk = Path(ctx.workspace_path) / rel
    ev.target_file_exists = disk.exists() and disk.is_file()
    ev.in_candidate_set = any(
        rel.replace("\\", "/") == c.replace("\\", "/") or
        rel.replace("\\", "/").endswith(c.replace("\\", "/")) or
        c.replace("\\", "/").endswith(rel.replace("\\", "/"))
        for c in ctx.candidate_files
    )
    ev.patch_type = "modify" if ev.target_file_exists else "create"

    if not ev.in_candidate_set:
        logger.warning(
            "write_patch: target '%s' NOT in candidate set [%s]. Allowing but flagging.",
            rel, ", ".join(sorted(ctx.candidate_files.keys())[:8]),
        )

    # Capture original hash before write
    ev.original_file_hash = _file_hash(disk)
    original_size = disk.stat().st_size if disk.exists() and disk.is_file() else 0
    original_lines = []
    if disk.exists() and disk.is_file():
        try:
            original_lines = disk.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            pass

    try:
        disk.parent.mkdir(parents=True, exist_ok=True)
        disk.write_text(str(content), encoding="utf-8")
        ctx.written_files[rel] = str(content)

        # Measure effectiveness
        ev.patched_file_hash = _file_hash(disk)
        new_size = len(str(content).encode("utf-8"))
        ev.bytes_changed = abs(new_size - original_size)
        new_lines = str(content).splitlines()
        # Simple line-level diff count
        max_len = max(len(original_lines), len(new_lines))
        diff_lines = 0
        for i in range(max_len):
            orig = original_lines[i] if i < len(original_lines) else None
            new = new_lines[i] if i < len(new_lines) else None
            if orig != new:
                diff_lines += 1
        ev.effective_diff_lines = diff_lines

        if ev.original_file_hash == ev.patched_file_hash and ev.original_file_hash:
            ev.accepted = False
            ev.rejection_reason = PATCH_REJECTION_INVALID_PATCH
            ctx.write_patch_evidence.append(ev)
            logger.info(
                "write_patch_result: ticket=%s attempt=%d accepted=false reason=%s "
                "effective_diff=0 (identical content)",
                ticket_id, attempt, ev.rejection_reason,
            )
            return f"Wrote {rel} but content is IDENTICAL to original — no effective change."

        ev.accepted = True
        ev.rejection_reason = PATCH_ACCEPTED
        ctx.write_patch_evidence.append(ev)
        logger.info(
            "write_patch_result: ticket=%s attempt=%d accepted=true path=%s "
            "type=%s in_candidates=%s diff_lines=%d bytes_changed=%d",
            ticket_id, attempt, rel, ev.patch_type, ev.in_candidate_set,
            ev.effective_diff_lines, ev.bytes_changed,
        )
        return f"Wrote {len(str(content))} chars to {rel}. Effective diff: {ev.effective_diff_lines} lines changed."

    except Exception as exc:
        ev.rejection_reason = PATCH_REJECTION_PATCH_APPLY_FAILED
        ctx.write_patch_evidence.append(ev)
        logger.info(
            "write_patch_result: ticket=%s attempt=%d accepted=false reason=%s error=%s",
            ticket_id, attempt, ev.rejection_reason, exc,
        )
        return f"Failed to write {rel}: {exc}"


def _t_run_build(ctx: ToolContext, args: dict) -> str:
    """Build the project using the existing execution engine."""
    try:
        project_files = ctx.agents.executor.detect_project_files()
        if not project_files:
            ctx.last_build_status = "skipped"
            ctx.last_build_errors = []
            return "No project file detected — build skipped."
        ctx.build_attempts += 1
        result = ctx.agents.executor.execute_build(project_files[0])
        status = getattr(getattr(result, "status", None), "value", "unknown")
        errors = [str(e) for e in (getattr(result, "errors", []) or [])]
        ctx.last_build_status = status
        ctx.last_build_errors = errors
        if status == "success":
            return "Build: SUCCESS"
        return (
            f"Build: {status} with {len(errors)} error(s). "
            f"Read the errors, open the failing file(s), fix them, then run_build again:\n"
            + "\n".join(f"  - {e}" for e in errors[:8])
        )
    except Exception as exc:
        ctx.last_build_status = "error"
        ctx.last_build_errors = [str(exc)]
        return f"Build failed to run: {exc}"


def _t_verify_outcome(ctx: ToolContext, args: dict) -> str:
    """Verify the requested change is actually true now (ticket-aware)."""
    from ticket_to_code.reasoning.verifiers import verify_outcome
    vev = VerificationEvidence()
    try:
        result = verify_outcome(ctx)
        vev.verification_executed = True
        vev.verifier_used = str(result.get("verifier", "default"))
        vev.mode = str(result.get("mode", ""))
        verified = result.get("verified")
        vev.passed = bool(verified)
        vev.rejection_reason = "" if verified else str(result.get("reason", "unknown"))
        vev.detail = str(result.get("reason", ""))[:300]
    except Exception as exc:
        vev.verification_executed = False
        vev.rejection_reason = f"verifier_error: {exc}"
        verified = False
        result = {"verified": False, "mode": "error", "reason": str(exc)}

    ctx.last_verify_passed = bool(verified)
    ctx.last_verify_mode = str(result.get("mode", ""))
    ctx.last_verify_reason = str(result.get("reason", ""))
    ctx.verification_evidence.append(vev)

    logger.info(
        "verify_outcome_result: executed=%s verifier=%s passed=%s reason=%s",
        vev.verification_executed, vev.verifier_used, vev.passed, vev.rejection_reason,
    )
    return f"Outcome verification: {'PASS' if verified else 'FAIL'} ({result.get('mode')}). {result.get('reason','')}"


def _t_recall_feature(ctx: ToolContext, args: dict) -> str:
    """Query the Enterprise Feature Graph for this ticket's feature (owners/validators). No args."""
    fg = getattr(ctx, "feature_graph", None)
    if fg is None:
        return "No feature graph available."
    text = f"{getattr(ctx.ticket, 'title', '')} {getattr(ctx.ticket, 'description', '')}"
    feature = fg.match_feature(text, ctx.ticket_type)
    if feature is None:
        return "No known feature matches this ticket yet. Discover the owner dynamically (find_owner)."
    owners = fg.resolve_owner_candidates(feature, ctx.workspace_path)
    for o in owners:
        ctx.candidate_files[o["file"]] = max(ctx.candidate_files.get(o["file"], 0.0), float(o["confidence"]))
    summ = fg.feature_summary(feature)
    lines = [f"Feature: {summ['feature']} (confidence {summ['confidence']})"]
    if owners:
        lines.append("Owner candidates (learned first, then hints):")
        lines += [f"  - {o['file']} ({o['source']}, conf {o['confidence']})" for o in owners[:8]]
    else:
        lines.append("No owners known yet — use find_owner to discover, learning will remember it.")
    if summ.get("validators"):
        lines.append("Validators: " + "; ".join(summ["validators"][:4]))
    if summ.get("common_mistakes"):
        lines.append("Avoid: " + "; ".join(summ["common_mistakes"][:3]))
    return "\n".join(lines)


def _t_brain_query(ctx: ToolContext, args: dict) -> str:
    """Ticket-scoped Brain lookup (features/patterns/playbooks/owners). No args."""
    bm = getattr(ctx, "brain_manager", None)
    if bm is None:
        return "No Brain manager available."
    text = f"{getattr(ctx.ticket, 'title', '')} {getattr(ctx.ticket, 'description', '')}"
    payload = bm.query_ticket(text, ctx.ticket_type)
    for fp in payload.get("owner_files", []):
        ctx.candidate_files[fp] = max(ctx.candidate_files.get(fp, 0.0), 0.8)
    lines = []
    feats = payload.get("features", [])
    pats = payload.get("ticket_patterns", [])
    pbs = payload.get("playbooks", [])
    if feats:
        lines.append("Brain features: " + ", ".join(str(f.get("name") or f.get("id")) for f in feats[:4]))
    if pats:
        lines.append("Brain patterns: " + ", ".join(str(p.get("pattern_id")) for p in pats[:3]))
    if pbs:
        lines.append("Brain playbooks: " + ", ".join(str(p.get("playbook")) for p in pbs[:3]))
    if payload.get("owner_files"):
        lines.append("Brain owner files: " + ", ".join(payload["owner_files"][:8]))
    if payload.get("skills"):
        lines.append("Suggested skills: " + ", ".join(payload["skills"][:6]))
    return "\n".join(lines) if lines else "Brain query returned no direct match."


def _t_brain_search_feature(ctx: ToolContext, args: dict) -> str:
    """Search Brain feature index. args: {query}"""
    bm = getattr(ctx, "brain_manager", None)
    if bm is None:
        return "No Brain manager available."
    query = str(args.get("query", "")).strip()
    rows = bm.search_feature(query, limit=8)
    if not rows:
        return f"No Brain feature match for '{query}'."
    lines = []
    for r in rows:
        fid = str(r.get("id", ""))
        name = str(r.get("name", fid))
        owners = bm.get_owner_files(fid or name)
        if owners:
            for fp in owners[:6]:
                ctx.candidate_files[fp] = max(ctx.candidate_files.get(fp, 0.0), 0.75)
        lines.append(f"- {name} ({fid}) owners={owners[:4]}")
    return "\n".join(lines)


def _t_brain_get_owner_files(ctx: ToolContext, args: dict) -> str:
    """Get owner files from Brain feature map. args: {feature}"""
    bm = getattr(ctx, "brain_manager", None)
    if bm is None:
        return "No Brain manager available."
    feature = str(args.get("feature", "")).strip()
    if not feature:
        return "brain_get_owner_files needs 'feature'."
    owners = bm.get_owner_files(feature)
    if not owners:
        return f"No Brain owner files for '{feature}'."
    for fp in owners:
        ctx.candidate_files[fp] = max(ctx.candidate_files.get(fp, 0.0), 0.85)
    return "Owner files: " + ", ".join(owners[:10])


def _t_brain_get_playbook(ctx: ToolContext, args: dict) -> str:
    """Get Brain investigation playbook. args: {name}"""
    bm = getattr(ctx, "brain_manager", None)
    if bm is None:
        return "No Brain manager available."
    name = str(args.get("name", "")).strip()
    if not name:
        return "brain_get_playbook needs 'name'."
    pb = bm.get_playbook(name)
    if not pb:
        return f"No playbook found for '{name}'."
    steps = pb.get("steps") or []
    evidence = pb.get("expected_evidence") or []
    stops = pb.get("stop_conditions") or []
    return (
        f"Playbook: {pb.get('playbook')}\n"
        f"Steps: {' | '.join(str(s) for s in steps[:6])}\n"
        f"Expected evidence: {' | '.join(str(s) for s in evidence[:4])}\n"
        f"Stop conditions: {' | '.join(str(s) for s in stops[:3])}"
    )


def _t_brain_get_root_causes(ctx: ToolContext, args: dict) -> str:
    """Get likely root causes from Brain patterns. args: {pattern}"""
    bm = getattr(ctx, "brain_manager", None)
    if bm is None:
        return "No Brain manager available."
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        return "brain_get_root_causes needs 'pattern'."
    causes = bm.get_root_causes(pattern)
    if not causes:
        return f"No root-cause profile found for '{pattern}'."
    return "Root causes: " + ", ".join(f"{c['cause']}({c['probability']})" for c in causes[:8])


def build_default_registry() -> Dict[str, Tool]:
    """Return the standard repository tool registry."""
    tools = [
        Tool(
            "recall_memory",
            "Recall similar past tickets and learned owner files from the Repository Brain. No args.",
            _t_recall_memory,
            CapabilityMetadata(cost=1, confidence_gain=0.3, risk=0.05, prerequisites=[], provides=["owner_candidates", "history_context"]),
        ),
        Tool(
            "brain_query",
            "Query Brain indexes for ticket-aligned features, patterns, playbooks, and owner files. No args.",
            _t_brain_query,
            CapabilityMetadata(cost=1, confidence_gain=0.55, risk=0.05, prerequisites=[], provides=["feature_context", "owner_candidates", "history_context"]),
        ),
        Tool(
            "brain_search_feature",
            "Search Brain feature index. args: {query}",
            _t_brain_search_feature,
            CapabilityMetadata(cost=1, confidence_gain=0.45, prerequisites=[], provides=["feature_context", "owner_candidates"]),
        ),
        Tool(
            "brain_get_owner_files",
            "Get owner files from Brain feature map. args: {feature}",
            _t_brain_get_owner_files,
            CapabilityMetadata(cost=1, confidence_gain=0.5, prerequisites=[], provides=["owner_candidates"]),
        ),
        Tool(
            "brain_get_playbook",
            "Get investigation playbook from Brain. args: {name}",
            _t_brain_get_playbook,
            CapabilityMetadata(cost=1, confidence_gain=0.4, prerequisites=[], provides=["feature_context", "history_context"]),
        ),
        Tool(
            "brain_get_root_causes",
            "Get probable root causes from Brain patterns. args: {pattern}",
            _t_brain_get_root_causes,
            CapabilityMetadata(cost=1, confidence_gain=0.35, prerequisites=[], provides=["history_context"]),
        ),
        Tool(
            "recall_feature",
            "Query the Feature Graph for this ticket's feature: owners, validators, common mistakes. No args.",
            _t_recall_feature,
            CapabilityMetadata(cost=1, confidence_gain=0.45, prerequisites=[], provides=["feature_context", "owner_candidates", "validators"]),
        ),
        Tool(
            "find_owner",
            "Find the most likely owner file(s) for this ticket (Brain + live discovery). No args.",
            _t_find_owner,
            CapabilityMetadata(cost=1, confidence_gain=0.5, risk=0.1, prerequisites=[], provides=["owner_candidates"]),
        ),
        Tool(
            "search_code",
            "Search the repo for an exact literal string. args: {query}",
            _t_search_code,
            CapabilityMetadata(cost=3, confidence_gain=0.25, risk=0.1, prerequisites=[], provides=["owner_candidates", "file_context"]),
        ),
        Tool(
            "grep",
            "Regex search across the repo. args: {pattern}",
            _t_grep,
            CapabilityMetadata(cost=3, confidence_gain=0.25, prerequisites=[], provides=["owner_candidates", "file_context"]),
        ),
        Tool(
            "grep_search",
            "Live pattern/regex search with line numbers and matched snippets. args: {query, is_regex?, max_results?}",
            _t_grep_search,
            CapabilityMetadata(cost=2, confidence_gain=0.4, prerequisites=[], provides=["owner_candidates", "file_context"]),
        ),
        Tool(
            "rag",
            "Semantic retrieval of relevant code chunks. args: {query}",
            _t_rag,
            CapabilityMetadata(cost=10, confidence_gain=0.1, risk=0.1, prerequisites=[], provides=["file_context", "architecture_context"]),
        ),
        Tool(
            "read_file",
            "Read a file on demand. args: {path, start_line?, end_line?, max_lines?}",
            _t_read_file,
            CapabilityMetadata(cost=1, confidence_gain=0.35, risk=0.1, prerequisites=["owner_candidates"], provides=["file_context"]),
        ),
        Tool(
            "view_file",
            "View file contents with line numbers and optional line slice range. args: {path, start_line?, end_line?, max_lines?}",
            _t_view_file,
            CapabilityMetadata(cost=1, confidence_gain=0.45, risk=0.05, prerequisites=[], provides=["file_context"]),
        ),
        Tool(
            "write_patch",
            "Write new content to a file. args: {path, content}",
            _t_write_patch,
            CapabilityMetadata(cost=1, confidence_gain=0.6, risk=0.75, prerequisites=["owner_candidates", "file_context"], provides=["patch"]),
        ),
        Tool(
            "run_build",
            "Compile/build the project. No args.",
            _t_run_build,
            CapabilityMetadata(cost=2, confidence_gain=0.7, risk=0.35, prerequisites=["patch"], provides=["build_status", "build_success"]),
        ),
        Tool(
            "verify_outcome",
            "Verify the ticket's requested change is actually true now. No args.",
            _t_verify_outcome,
            CapabilityMetadata(cost=1, confidence_gain=0.9, risk=0.2, prerequisites=["build_success"], provides=["verification"]),
        ),
    ]
    return {t.name: t for t in tools}
