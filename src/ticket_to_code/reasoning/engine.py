"""
Dynamic Repository Reasoning Engine (Ticket-to-Code v2).

A ReAct-style loop that reasons like a senior principal engineer instead of running a
fixed pipeline:

    understand → what do I know? → what's missing? → pick a tool →
    read/search/act → observe → update understanding → repeat → verify → done

Key properties:
  - DYNAMIC control: the LLM chooses the next tool each step (no fixed graph).
  - ON-DEMAND context: files are read only when the reasoner asks.
  - REASONING MEMORY: a running scratchpad carries what was learned so far.
  - REPOSITORY BRAIN: prior tickets seed and, on success, are learned back.
  - GLOBAL BUDGET: hard cap on steps to guarantee termination.

The reasoner ORCHESTRATES existing capabilities (RAG, discovery, generation,
build, verification) exposed as tools. It does not replace them.
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ticket_to_code.reasoning.belief_state import (
    Contradiction,
    EvidenceItem,
    Objective,
    Observation,
    objective_coverage,
    source_reliability,
)
from ticket_to_code.reasoning.capability_selector import (
    select_capability,
    should_override_action,
)
from ticket_to_code.reasoning.repository_brain import RepositoryBrain, CC4EBrain
from ticket_to_code.reasoning.experience import ExperienceStore
from ticket_to_code.reasoning.feature_graph import FeatureGraph
from ticket_to_code.reasoning.brain_manager import BrainManager
from ticket_to_code.reasoning.skills import select_skill
from ticket_to_code.reasoning.tools import ToolContext, build_default_registry

logger = logging.getLogger(__name__)

_MAX_STEPS = 18                 # global safety budget (guarantees termination)
_MAX_CONSECUTIVE_ERRORS = 4     # stop if the reasoner keeps failing


@dataclass
class ReasoningResult:
    ticket_id: str
    status: str                       # "solved" | "unverified" | "failed"
    verified: bool
    written_files: List[str] = field(default_factory=list)
    steps: List[dict] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""
    lessons: List[str] = field(default_factory=list)


_SYSTEM_PROMPT = """You are the principal AI software engineer for this repository.
You solve a ticket by REASONING step by step and calling tools — exactly like a
senior engineer, not a fixed pipeline.

Each step, respond with a SINGLE JSON object and nothing else:
{{
  "thought": "what you now understand and what you still need",
  "action": "<tool_name>",
  "action_input": {{ ... }}
}}

When the requested change is fully implemented AND you have verified it, respond:
{{
  "thought": "why the ticket is complete",
  "final": {{ "summary": "...", "confidence": 0.0, "lessons": ["..."] }}
}}

Rules:
- Read files ON DEMAND. Do not guess file contents — use read_file.
- Prefer learned repository knowledge (recall_memory / find_owner) before broad search.
- Make the MINIMAL correct change. Do not touch unrelated files.
- After editing, ALWAYS run_build. If the build fails, read the errors, open the
  failing file(s), fix them, and run_build again — keep looping until it builds.
- Always call verify_outcome before declaring final success.
- You may ONLY finish when the build succeeds (or no project build exists) AND
  verify_outcome passes.
- Output ONLY the JSON object. No markdown, no prose outside JSON.

Available tools:
{tool_docs}

Capability metadata (cost, confidence gain, prerequisites, provides):
{capability_metadata}

Deterministic selector recommendation for NEXT step:
{selector_recommendation}

CC4E strategy for this ticket ({strategy_name}):
{strategy_guidance}

Preferred capabilities for this ticket type (use these first when relevant):
{skill_capabilities}

Validators that MUST hold before you finish:
{skill_validators}

Common mistakes to avoid:
{strategy_mistakes}
"""


def _extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from an LLM response."""
    if not text:
        return None
    # Strip code fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else None
    if raw is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        raw = brace.group(0) if brace else None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        # Best-effort cleanup of trailing commas.
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            return json.loads(cleaned)
        except Exception:
            return None


def _classify_ticket(ticket: Any) -> str:
    """Lightweight ticket-type inference (deterministic, no LLM)."""
    text = f"{getattr(ticket, 'title', '')} {getattr(ticket, 'description', '')}".lower()
    labels = [str(l).lower() for l in (getattr(ticket, "labels", []) or [])]
    joined = text + " " + " ".join(labels)
    if any(k in joined for k in ("organization", "organisation", "permission", "access", "rbac", "coordinator", "co-ordinator", "private issues", "role")):
        return "access_control"
    if any(k in joined for k in ("version", "release", "bump")):
        return "version_bump"
    if any(k in joined for k in ("endpoint", "api", "swagger", "controller")):
        return "api_change"
    if any(k in joined for k in ("config", "yaml", "properties", "timeout", "setting")):
        return "configuration"
    if any(k in joined for k in ("css", "style", "layout", "footer", "header", "angular", "react", "jsx", "tsx", "frontend", "ui")):
        return "ui"
    if any(k in joined for k in ("bug", "fix", "error", "crash", "race", "defect")):
        return "bug_fix"
    return "feature_request"


def _init_objectives(ticket: Any) -> List[Objective]:
    """Create objective checklist from acceptance criteria or fallback ticket goal."""
    rows: List[Objective] = []
    criteria = list(getattr(ticket, "acceptance_criteria", []) or [])
    for idx, item in enumerate(criteria, start=1):
        text = str(item or "").strip()
        if not text:
            continue
        rows.append(Objective(key=f"AC-{idx}", text=text))
    if not rows:
        rows.append(Objective(key="AC-1", text=str(getattr(ticket, "title", "Implement requested change"))))
    return rows


def _state_signature(ctx: ToolContext) -> str:
    return "|".join(
        [
            f"owner:{1 if ctx.candidate_files else 0}",
            f"ctx:{1 if ctx.files_read else 0}",
            f"patch:{1 if ctx.written_files else 0}",
            f"build:{1 if ctx.last_build_status in ('success', 'skipped') else 0}",
            f"verify:{1 if ctx.last_verify_passed else 0}",
            f"contr:{len(ctx.contradictions or [])}",
        ]
    )


def _enter_stage(ctx: ToolContext, stage: str) -> None:
    """Track cumulative time per stage and transition to a new stage."""
    now = _time.perf_counter()
    prev = getattr(ctx, '_stage_start', None)
    if prev is not None and ctx.current_stage:
        elapsed = now - prev
        ctx.stage_timings[ctx.current_stage] = ctx.stage_timings.get(ctx.current_stage, 0.0) + elapsed
    ctx.current_stage = stage
    ctx._stage_start = now  # type: ignore[attr-defined]


def _contradiction_resolution_hint(ctx: ToolContext) -> Dict[str, Any]:
    """Pick a focused contradiction-resolution action from claim keys."""
    if not ctx.contradictions:
        return {"action": "find_owner", "action_input": {}}
    c = ctx.contradictions[0]
    key = str(c.claim_key or "")
    if key.startswith("owner:"):
        fp = key.split("owner:", 1)[1]
        return {"action": "read_file", "action_input": {"path": fp, "max_lines": 220}}
    if key.startswith("build:"):
        return {"action": "run_build", "action_input": {}}
    if key.startswith("verify:"):
        return {"action": "verify_outcome", "action_input": {}}
    return {"action": "find_owner", "action_input": {}}


def _record_observation(ctx: ToolContext, step: int, action: str, action_input: dict, observation: str) -> None:
    ctx.observations.append(
        Observation(step=step, action=action, args=dict(action_input or {}), output=str(observation))
    )


def _extract_evidence(ctx: ToolContext, step: int, action: str, observation: str) -> List[EvidenceItem]:
    """Derive interpreted evidence from raw observation output.

    This is intentionally conservative: it records support/contradiction hints
    and keeps claims revisable.
    """
    out = str(observation or "")
    rel = source_reliability(action)
    refs: List[str] = []
    ev: List[EvidenceItem] = []

    # Candidate owner evidence from owner-discovery actions.
    if action in {"find_owner", "brain_query", "brain_get_owner_files", "recall_feature", "recall_memory"}:
        for fp in list(ctx.candidate_files.keys())[:8]:
            refs = [fp]
            key = f"owner:{fp}"
            ev.append(EvidenceItem(
                step=step,
                source_action=action,
                claim_key=key,
                claim_text=f"{fp} is a candidate owner",
                polarity=1,
                source_reliability=rel,
                references=refs,
            ))

    if action == "run_build":
        if "Build: SUCCESS" in out:
            ev.append(EvidenceItem(
                step=step,
                source_action=action,
                claim_key="build:success",
                claim_text="Project build succeeded",
                polarity=1,
                source_reliability=rel,
            ))
        else:
            ev.append(EvidenceItem(
                step=step,
                source_action=action,
                claim_key="build:success",
                claim_text="Project build failed",
                polarity=-1,
                source_reliability=rel,
            ))

    if action == "verify_outcome":
        if "PASS" in out:
            ev.append(EvidenceItem(
                step=step,
                source_action=action,
                claim_key="verify:pass",
                claim_text="Outcome verified",
                polarity=1,
                source_reliability=rel,
            ))
        else:
            ev.append(EvidenceItem(
                step=step,
                source_action=action,
                claim_key="verify:pass",
                claim_text="Outcome not verified",
                polarity=-1,
                source_reliability=rel,
            ))

    if action == "read_file":
        # Reading file adds high-quality context evidence for selected file.
        path = ""
        if "(" in out:
            path = out.split("(", 1)[0].strip()
        if path:
            ev.append(EvidenceItem(
                step=step,
                source_action=action,
                claim_key=f"context:{path}",
                claim_text=f"Read file context from {path}",
                polarity=1,
                source_reliability=rel,
                references=[path],
            ))

    return ev


def _detect_contradictions(ctx: ToolContext) -> List[Contradiction]:
    by_key: Dict[str, Dict[str, List[str]]] = {}
    for item in (ctx.evidence_items or []):
        bucket = by_key.setdefault(item.claim_key, {"support": [], "contradict": []})
        ref = f"{item.step}:{item.source_action}"
        if item.polarity > 0:
            bucket["support"].append(ref)
        elif item.polarity < 0:
            bucket["contradict"].append(ref)

    out: List[Contradiction] = []
    for key, bucket in by_key.items():
        if bucket["support"] and bucket["contradict"]:
            out.append(
                Contradiction(
                    claim_key=key,
                    support_refs=bucket["support"][:8],
                    contradict_refs=bucket["contradict"][:8],
                    status="open",
                )
            )
    return out


def _update_objectives(ctx: ToolContext) -> None:
    coverage_build = any(e.claim_key == "build:success" and e.polarity > 0 for e in (ctx.evidence_items or []))
    coverage_verify = any(e.claim_key == "verify:pass" and e.polarity > 0 for e in (ctx.evidence_items or []))
    for o in ctx.objectives:
        # Conservative completion gate: objective can be satisfied only when
        # both build and verification evidence are positive.
        if coverage_build and coverage_verify:
            o.status = "satisfied"


def _reflect_before_learning(ctx: ToolContext, verified: bool, current_lessons: List[str]) -> List[str]:
    """Distill concise lessons from execution history before persistence."""
    lessons = list(current_lessons or [])
    hist = ctx.capability_history or []
    if not hist:
        return lessons

    # Keep effective and ineffective moves separate.
    counts: Dict[str, int] = {}
    for h in hist:
        cap = str(h.get("capability", ""))
        counts[cap] = counts.get(cap, 0) + 1

    repeated = [k for k, v in counts.items() if v >= 3]
    if repeated:
        lessons.append("Avoid repeated low-yield loops: " + ", ".join(sorted(repeated)[:4]))

    if verified:
        if ctx.last_build_status == "success":
            lessons.append("Verified runs should preserve build-success + verify-outcome as hard completion gates.")
    else:
        if ctx.contradictions:
            lessons.append("Resolve contradictory evidence before committing to owner edits.")

    return list(dict.fromkeys([x.strip() for x in lessons if str(x).strip()]))[:8]


def run_reasoning_agent(
    ticket: Any,
    workspace_path: str,
    technology: Optional[str] = None,
    investigation: Any = None,
) -> ReasoningResult:
    """Run the dynamic reasoning engine for a single ticket across any repository."""
    from ticket_to_code.workflow import WorkflowAgents  # reuse existing capabilities
    from ticket_to_code.runtime.run_context import RunContext, current_run_context

    ticket_id = getattr(ticket, "ticket_id", "UNKNOWN")
    ticket_type = _classify_ticket(ticket)
    ticket_text = f"{getattr(ticket, 'title', '')} {getattr(ticket, 'description', '')}"
    skill = select_skill(ticket_type, ticket_text)

    logger.info("🧠 Repository Reasoner start: %s (type=%s, skill=%s)", ticket_id, ticket_type, skill.name)

    agents = WorkflowAgents(workspace_path, technology)
    brain = RepositoryBrain(workspace_path)
    experience = ExperienceStore(workspace_path)
    brain_manager = BrainManager()
    try:
        from ticket_to_code.brain.brain_storage import RepositoryBrainStorage
        repository_brain = RepositoryBrainStorage()
        repository_brain.load_default(workspace_path)
    except Exception as _repo_brain_exc:
        logger.debug("repository brain load skipped: %s", _repo_brain_exc)
        repository_brain = None
    feature_graph = FeatureGraph(workspace_path)
    registry = build_default_registry()

    ctx = ToolContext(
        ticket=ticket,
        workspace_path=workspace_path,
        agents=agents,
        brain=brain,
        experience_store=experience,
        brain_manager=brain_manager,
        repository_brain=repository_brain,
        ticket_type=ticket_type,
        feature_graph=feature_graph,
    )
    ctx.objectives = _init_objectives(ticket)
    # Attach investigation (for outcome verifier literals) if provided.
    setattr(ctx, "investigation", investigation)
    # Initialize stage timing
    ctx._stage_start = _time.perf_counter()  # type: ignore[attr-defined]
    ctx.current_stage = "initialization"

    run_ctx = RunContext(ticket_id=ticket_id, repo=str(workspace_path))
    token = current_run_context.set(run_ctx)

    tool_docs = "\n".join(f"- {t.name}: {t.description}" for t in registry.values())
    capability_metadata = "\n".join(
        (
            f"- {t.name}: cost={t.metadata.cost}, gain={t.metadata.confidence_gain}, risk={getattr(t.metadata, 'risk', 0.2)}, "
            f"requires={t.metadata.prerequisites or ['nothing']}, provides={t.metadata.provides or ['none']}"
        )
        for t in registry.values()
    )

    # Prefer FEATURE GRAPH validators/mistakes (learned, non-stale) over static skill fields.
    matched_feature = None
    try:
        matched_feature = feature_graph.match_feature(ticket_text, ticket_type)
    except Exception as _fg_exc:
        logger.debug("feature match skipped: %s", _fg_exc)
    effective_validators = list(skill.validators)
    effective_mistakes = list(skill.common_mistakes)
    if matched_feature is not None:
        fv = matched_feature.get("validators") or []
        fm = matched_feature.get("common_mistakes") or []
        if fv:
            effective_validators = fv
        if fm:
            effective_mistakes = fm

    scratchpad: List[str] = [
        f"TICKET {ticket_id}: {getattr(ticket, 'title', '')}",
        f"DESCRIPTION: {getattr(ticket, 'description', '')}",
        f"TICKET TYPE: {ticket_type}",
        f"SKILL LOADED: {skill.name}",
        "OBJECTIVES: " + " | ".join(f"{o.key}:{o.text}" for o in ctx.objectives[:8]),
    ]

    try:
        bsum = brain_manager.summary()
        scratchpad.append(
            "BRAIN INDEX READY: "
            f"features={bsum.get('feature_nodes', 0)}, "
            f"patterns={bsum.get('ticket_patterns', 0)}, "
            f"playbooks={bsum.get('playbooks', 0)}"
        )
    except Exception as _bm_exc:
        logger.debug("brain manager summary skipped: %s", _bm_exc)

    # Deterministic skill seeding (no LLM): known owners, literals, etc.
    try:
        seed_summary = skill.prepare(ctx)
        if seed_summary:
            scratchpad.append(f"SKILL SEED: {seed_summary}")
    except Exception as _seed_exc:
        logger.warning("Skill prepare() failed (non-fatal): %s", _seed_exc)

    # Memory-first: reuse learned owners before any reasoning (no LLM call).
    try:
        pre = brain_manager.query_ticket(ticket_text, ticket_type)
        pre_owners = pre.get("owner_files", [])
        for fp in pre_owners:
            ctx.candidate_files[fp] = max(ctx.candidate_files.get(fp, 0.0), 0.8)
        if pre_owners:
            scratchpad.append("BRAIN OWNERS: " + ", ".join(pre_owners[:8]))

        learned_owners = brain.suggest_owner_files(ticket_text, ticket_type, limit=6)
        for o in learned_owners:
            ctx.candidate_files[o["file"]] = max(ctx.candidate_files.get(o["file"], 0.0), float(o["confidence"]))
        if learned_owners:
            scratchpad.append(
                "MEMORY OWNERS (reuse before rediscovering): "
                + ", ".join(f"{o['file']}({o['confidence']})" for o in learned_owners[:6])
            )

        repo_brain_matches = []
        if repository_brain is not None:
            repo_brain_matches = repository_brain.find_nodes_by_semantic_intent(ticket_text)
            repo_seeded: list[str] = []
            for match in repo_brain_matches[:6]:
                node = match.get("node")
                score = float(match.get("score") or 0.0)
                confidence = min(0.92, 0.55 + min(score, 20.0) / 40.0)
                for participant in list(getattr(node, "participating_files", []) or [])[:8]:
                    fp = str(getattr(participant, "file_path", "") or "").strip()
                    if not fp:
                        continue
                    ctx.candidate_files[fp] = max(ctx.candidate_files.get(fp, 0.0), confidence)
                    if len(repo_seeded) < 10:
                        repo_seeded.append(fp)
            if repo_seeded:
                scratchpad.append("REPOSITORY BRAIN OWNERS: " + ", ".join(repo_seeded[:10]))

        # For small UI/component tickets, preload the highest-signal grounded
        # component file so the selector can advance directly to patching.
        if ticket_type == "ui" and ctx.candidate_files:
            component_names = set(re.findall(r"\b[A-Z][A-Za-z0-9_]+\b", ticket_text))
            ranked_candidates = sorted(ctx.candidate_files.items(), key=lambda kv: kv[1], reverse=True)
            preload_path = None
            for fp, _score in ranked_candidates:
                lowered = fp.lower()
                if any(name.lower() in lowered for name in component_names):
                    preload_path = fp
                    break
                if "/src/components/" in f"/{lowered}" and lowered.endswith((".jsx", ".tsx", ".js", ".ts", ".css", ".scss")):
                    preload_path = fp
                    break
            if preload_path:
                disk = Path(workspace_path) / preload_path
                if disk.exists() and disk.is_file():
                    try:
                        preload_text = disk.read_text(encoding="utf-8", errors="ignore")[:12000]
                        ctx.files_read[preload_path] = preload_text
                        scratchpad.append(f"PRELOADED FILE CONTEXT: {preload_path}")
                    except Exception as _preload_exc:
                        logger.debug("file preload skipped for %s: %s", preload_path, _preload_exc)
    except Exception as _mem_exc:
        logger.debug("memory owner seeding skipped: %s", _mem_exc)

    _enter_stage(ctx, "reasoning")
    steps: List[dict] = []
    verified = False
    final_summary = ""
    final_conf = 0.0
    lessons: List[str] = []
    consecutive_errors = 0

    try:
        from ticket_to_code.runtime.llm_factory import LLMRegistry
        from ticket_to_code.llm_utils import llm_invoke
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = LLMRegistry.get_llm(assistant=False)
    except Exception as exc:
        logger.error("CC4E Reasoner cannot load LLM: %s", exc)
        current_run_context.reset(token)
        return ReasoningResult(ticket_id=ticket_id, status="failed", verified=False,
                               summary=f"LLM unavailable: {exc}")

    for step_idx in range(1, _MAX_STEPS + 1):
        _update_objectives(ctx)
        cov = objective_coverage(ctx.objectives)
        selector = select_capability(ctx, registry, preferred_capabilities=skill.capabilities)
        selector_line = (
            f"need={selector.need}; recommend={selector.recommended_action or 'none'}; "
            f"reason={selector.reason}; alternatives={selector.alternatives}; "
            f"objective_coverage={round(cov, 3)}; contradictions_open={len(ctx.contradictions or [])}"
        )
        system = _SYSTEM_PROMPT.format(
            tool_docs=tool_docs,
            capability_metadata=capability_metadata,
            selector_recommendation=selector_line,
            strategy_name=skill.name,
            strategy_guidance=skill.guidance,
            skill_capabilities="\n".join(f"- {c}" for c in skill.capabilities) or "- (kernel default order)",
            skill_validators="\n".join(f"- {v}" for v in effective_validators) or "- Build succeeds and outcome verified.",
            strategy_mistakes="\n".join(f"- {m}" for m in effective_mistakes),
        )

        try:
            run_ctx.budget.check()
        except Exception as exc:
            lessons.append("Stopped by global budget.")
            final_summary = f"Global budget reached: {exc}"
            break

        user = "\n".join(scratchpad[-40:]) + "\n\nDecide the next single action as JSON."
        try:
            resp = llm_invoke(llm, [SystemMessage(content=system), HumanMessage(content=user)])
            content = str(getattr(resp, "content", "") or "")
        except Exception as exc:
            consecutive_errors += 1
            scratchpad.append(f"[step {step_idx}] LLM error: {exc}")
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                final_summary = "Repeated LLM errors."
                break
            continue

        decision = _extract_json(content)
        if not decision:
            consecutive_errors += 1
            scratchpad.append(f"[step {step_idx}] Could not parse a JSON action. Respond with one JSON object only.")
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                final_summary = "Reasoner produced unparseable output repeatedly."
                break
            continue
        consecutive_errors = 0

        thought = str(decision.get("thought", "")).strip()

        # Final answer?
        if "final" in decision:
            fin = decision.get("final") or {}
            final_summary = str(fin.get("summary", thought or "Ticket completed."))
            try:
                final_conf = float(fin.get("confidence", 0.0))
            except Exception:
                final_conf = 0.0
            lessons = [str(x) for x in (fin.get("lessons") or [])][:6]
            if ctx.contradictions:
                scratchpad.append(
                    f"[step {step_idx}] You declared done but {len(ctx.contradictions)} contradiction(s) remain open. "
                    "Resolve evidence conflicts before finishing."
                )
                steps.append({"step": step_idx, "thought": thought, "action": "final",
                              "rejected": "contradictions_open", "contradictions_open": len(ctx.contradictions)})
                if step_idx < _MAX_STEPS:
                    continue
            # Completion must satisfy objective coverage.
            cov = objective_coverage(ctx.objectives)
            if cov < 0.999:
                scratchpad.append(
                    f"[step {step_idx}] You declared done but objective coverage is only {round(cov, 3)}. "
                    "Keep working until all objectives are satisfied."
                )
                steps.append({"step": step_idx, "thought": thought, "action": "final",
                              "rejected": "objective_coverage_incomplete", "objective_coverage": cov})
                if step_idx < _MAX_STEPS:
                    continue
            # Enforce a real build before trusting success: solve → build → fix loop.
            if ctx.written_files and ctx.last_build_status not in ("success", "skipped"):
                scratchpad.append(
                    f"[step {step_idx}] You declared done but the project has not built "
                    f"successfully yet (last build status: "
                    f"'{ctx.last_build_status or 'never run'}'). Run run_build, read any "
                    f"errors, fix the failing file(s), and build again before finishing."
                )
                steps.append({"step": step_idx, "thought": thought, "action": "final",
                              "rejected": "build_not_successful",
                              "build_status": ctx.last_build_status or "never_run"})
                if step_idx < _MAX_STEPS:
                    continue
            # Enforce verification before trusting success.
            from ticket_to_code.reasoning.verifiers import verify_outcome
            vres = verify_outcome(ctx)
            verified = bool(vres.get("verified"))
            _update_objectives(ctx)
            steps.append({"step": step_idx, "thought": thought, "action": "final",
                          "verified": verified, "verify_mode": vres.get("mode"),
                          "build_status": ctx.last_build_status or "never_run"})
            if not verified:
                scratchpad.append(
                    f"[step {step_idx}] You declared done but verify_outcome FAILED "
                    f"({vres.get('mode')}: {vres.get('reason')}). Keep working."
                )
                # Give the reasoner a chance to fix unless out of budget.
                if step_idx < _MAX_STEPS:
                    continue
            break

        action = str(decision.get("action", "")).strip()
        action_input = decision.get("action_input") or {}
        if not isinstance(action_input, dict):
            action_input = {"value": action_input}

        # Contradiction-specific focused override (claim-aware).
        if selector.need == "resolve_contradictions" and ctx.contradictions:
            hint = _contradiction_resolution_hint(ctx)
            hinted_action = str(hint.get("action", "")).strip()
            if hinted_action and action != hinted_action:
                scratchpad.append(
                    f"[step {step_idx}] CONTRADICTION-FOCUSED OVERRIDE: '{action}' -> '{hinted_action}' "
                    f"for claim '{ctx.contradictions[0].claim_key}'."
                )
                action = hinted_action
                action_input = hint.get("action_input") or {}

        if should_override_action(action, selector, registry):
            original_action = action
            action = str(selector.recommended_action)
            action_input = {}
            scratchpad.append(
                f"[step {step_idx}] SELECTOR OVERRIDE: '{original_action}' was replaced with "
                f"'{action}' because current need is '{selector.need}' and '{original_action}' "
                "was a costlier detour."
            )

        tool = registry.get(action)
        if not tool:
            scratchpad.append(f"[step {step_idx}] Unknown tool '{action}'. Available: {', '.join(registry)}")
            steps.append({"step": step_idx, "thought": thought, "action": action, "error": "unknown_tool"})
            continue

        # Track stage transitions based on action
        if action in {"recall_memory", "find_owner", "brain_query", "brain_search_feature",
                      "brain_get_owner_files", "recall_feature", "search_code", "grep", "rag", "read_file"}:
            _enter_stage(ctx, "investigation")
        elif action == "write_patch":
            _enter_stage(ctx, "generation")
        elif action == "run_build":
            _enter_stage(ctx, "building")
        elif action == "verify_outcome":
            _enter_stage(ctx, "verification")

        observation = tool.run(ctx, action_input)
        _record_observation(ctx, step_idx, action, action_input, observation)
        new_evidence = _extract_evidence(ctx, step_idx, action, observation)
        if new_evidence:
            ctx.evidence_items.extend(new_evidence)
            for ev in new_evidence:
                if ev.claim_key.startswith("owner:") and ev.polarity > 0 and ev.references:
                    for fp in ev.references:
                        ctx.belief_state.upsert_candidate("owner", fp, f"{ev.step}:{ev.source_action}")
                        th_key = f"owner:{fp}"
                        existing = ctx.working_theories.get(th_key)
                        if existing is None:
                            ctx.working_theories[th_key] = {
                                "key": th_key,
                                "statement": f"{fp} is canonical owner",
                                "candidates": [fp],
                                "evidence_refs": [f"{ev.step}:{ev.source_action}"],
                                "status": "open",
                            }
                        else:
                            refs = list(existing.get("evidence_refs") or [])
                            ref = f"{ev.step}:{ev.source_action}"
                            if ref not in refs:
                                refs.append(ref)
                            existing["evidence_refs"] = refs
                            if len(refs) >= 2:
                                existing["status"] = "corroborated"
                            ctx.working_theories[th_key] = existing
        ctx.contradictions = _detect_contradictions(ctx)

        if ctx.contradictions:
            scratchpad.append(
                f"[step {step_idx}] CONTRADICTION: {len(ctx.contradictions)} open claim contradiction(s). "
                "Prioritize read_file/find_owner/run_build/verify_outcome to resolve."
            )

        _update_objectives(ctx)
        scratchpad.append(f"[step {step_idx}] THOUGHT: {thought}")
        scratchpad.append(f"[step {step_idx}] ACTION: {action}({json.dumps(action_input)[:200]})")
        scratchpad.append(f"[step {step_idx}] OBSERVATION: {observation}")
        steps.append({"step": step_idx, "thought": thought, "action": action,
                      "action_input": action_input, "observation": observation[:400],
                      "need": selector.need,
                      "state_signature": _state_signature(ctx),
                      "evidence_count": len(ctx.evidence_items),
                      "contradictions_open": len(ctx.contradictions),
                      "objective_coverage": round(objective_coverage(ctx.objectives), 3)})

    _enter_stage(ctx, "complete")
    current_run_context.reset(token)

    written = list(ctx.written_files.keys())
    if verified:
        status = "solved"
    elif written:
        status = "unverified"
    else:
        status = "failed"

    if not final_conf:
        final_conf = 0.85 if verified else (0.4 if written else 0.1)

    lessons = _reflect_before_learning(ctx, verified, lessons)

    try:
        experience.record_run(
            intent=ticket_type,
            steps=steps,
            success=bool(verified),
            final_need="verification" if verified else "objective_coverage",
            final_state=None,
        )
    except Exception as exc:
        logger.warning("Experience learning failed: %s", exc)

    # Learn back into the CC4E Brain (only real successes reinforce owner maps).
    try:
        brain.record_ticket(
            ticket_id=ticket_id,
            title=getattr(ticket, "title", ""),
            description=getattr(ticket, "description", ""),
            ticket_type=ticket_type,
            files_modified=written,
            outcome="verified" if verified else ("changed" if written else "failed"),
            confidence=final_conf,
            lessons=lessons,
            validation=skill.name,
        )
    except Exception as exc:
        logger.warning("Repository Brain learning failed: %s", exc)

    # Learn owners back into the Feature Graph (only on verified success).
    if verified and written:
        try:
            feature_id = matched_feature.get("feature") if matched_feature else ticket_type
            feature_graph.record_success(
                feature_id_or_text=feature_id,
                ticket_id=ticket_id,
                owner_files=written,
                confidence=final_conf,
            )
        except Exception as exc:
            logger.warning("Feature Graph learning failed: %s", exc)

    logger.info("🧠 Repository Reasoner done: %s status=%s verified=%s files=%d",
                ticket_id, status, verified, len(written))

    return ReasoningResult(
        ticket_id=ticket_id,
        status=status,
        verified=verified,
        written_files=written,
        steps=steps,
        confidence=round(final_conf, 3),
        summary=final_summary or ("Solved and verified." if verified else "Did not fully verify."),
        lessons=lessons,
    )


# Backward compatibility alias
run_cc4e_reasoning_agent = run_reasoning_agent

