"""
Pre-Generation Gate — activate reuse/scope/readiness at the plan→generation edge.

Composes the already-implemented gates into a single, testable filter that runs
BEFORE the generation loop writes any file:

  * ScopeExpansionGuard   → drop speculative shared-model changes / unjustified files
  * CapabilityReuseResolver → drop CREATE tasks that duplicate an existing capability
                              (CREATE_NEW is only kept when no existing equivalent is
                               found — i.e. evidence of insufficiency)
  * GenerationReadinessGate → hard-block (return to planning) only when there is NO
                              grounding at all (contracts AND verified capabilities
                              both missing)

Deterministic; language- and project-agnostic. Rejected tasks are simply not
generated, so unnecessary files cannot be produced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pathlib import Path

from ticket_to_code.agents.scope_expansion_guard import evaluate_scope, is_shared_model_file
from ticket_to_code.agents.capability_reuse_resolver import ReuseDecision
from ticket_to_code.agents.change_authorization import (
    classify_change_role, ChangeRole, _match, is_companion, is_config_file, is_build_config,
)
from ticket_to_code.agents.canonical_path import canonical_repo_path


_STRIP_SUFFIXES = (
    ".component.ts", ".service.ts", ".model.ts", ".module.ts", ".spec.ts",
    ".ts", ".tsx", ".java", ".kt", ".py", ".cs", ".html", ".scss",
)


def _stem_type(file_path: str) -> str:
    """Best-effort type name from a file path (e.g. displayed-member.model.ts → DisplayedMember)."""
    name = (file_path or "").replace("\\", "/").split("/")[-1]
    low = name.lower()
    for suf in _STRIP_SUFFIXES:
        if low.endswith(suf):
            name = name[: len(name) - len(suf)]
            break
    parts = re.split(r"[-_.\s]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _ttype(task) -> str:
    return str(getattr(getattr(task, "task_type", None), "value", "") or "").lower()


def _norm_path(p: str) -> str:
    return (p or "").replace("\\", "/").lower().rstrip("/")


def _basename(p: str) -> str:
    return _norm_path(p).split("/")[-1]


def _same_file(a: str, b: str) -> bool:
    """True when two paths point at the same file (tolerant of abs vs repo-relative)."""
    na, nb = _norm_path(a), _norm_path(b)
    if not na or not nb:
        return False
    return na == nb or na.endswith(nb) or nb.endswith(na) or _basename(na) == _basename(nb)


def _introduced_methods(task) -> set[str]:
    """Symbol names this task would ADD (new capability), lowercased."""
    out: set[str] = set()
    for m in (getattr(task, "allowed_methods", None) or []):
        if m:
            out.add(str(m).strip().lower())
    tm = getattr(task, "target_method", None)
    if tm:
        out.add(str(tm).strip().lower())
    return out


def _task_intent(task) -> str:
    for attr in ("description", "title", "selection_reason"):
        v = getattr(task, attr, None)
        if v:
            return str(v)
    return _basename(getattr(task, "file_path", ""))


def _capability_reuse_violation(task, semantic_resolver, min_confidence: float = 0.6, workspace_root: Optional[str] = None):
    """Return a rejection reason when an existing capability already satisfies this
    task's intent, so the planner should REUSE it instead of inventing a new
    method/file. Capability-based (intent, not name); provider != change target.

    Only fires for tasks that INTRODUCE a capability (create, or modify that adds a
    new method). Returns None when the task is a legitimate change or no existing
    capability is found. Never raises — degrades to None.
    """
    if semantic_resolver is None:
        return None
    fp = getattr(task, "file_path", "")
    ttype = _ttype(task)
    introduced = _introduced_methods(task)
    # A modify task that only edits existing methods is a legitimate behaviour change.
    if ttype != "create":
        if introduced:
            try:
                p = Path(fp)
                if not p.is_absolute() and workspace_root:
                    p = Path(workspace_root) / fp
                if p.exists():
                    disk_content = p.read_text(encoding="utf-8", errors="ignore").lower()
                    introduced = {m for m in introduced if m.lower() not in disk_content}
            except Exception:
                pass
        if not introduced:
            return None
    try:
        resolution = semantic_resolver.resolve_task(
            intent=_task_intent(task),
            title=str(getattr(task, "title", "") or ""),
            hints={
                "file_path": fp,
                "language": str(getattr(task, "language", "") or ""),
                "keywords": sorted(introduced),
            },
        )
    except Exception:
        return None
    if resolution is None or not getattr(resolution, "should_reuse", False):
        return None
    if float(getattr(resolution, "confidence", 0.0) or 0.0) < min_confidence:
        return None

    # Invariant: If the decision is to EXTEND the existing code (ResolutionDecision.EXTEND_EXISTING),
    # adding new methods/symbols to this file is EXPECTED and INTENDED — never reject an extend_existing task!
    decision = getattr(resolution, "decision", None)
    dec_val = getattr(decision, "value", str(decision) if decision else "").lower()
    if dec_val == "extend_existing":
        return None

    target_file = getattr(resolution, "target_file", None)
    target_symbol = str(getattr(resolution, "target_symbol", "") or "")

    # If target_symbol matches the class/file name (e.g. ParticipatingMembersRepository),
    # it is a class/interface container, NOT an existing method that satisfies the intent.
    class_stem = Path(fp).stem.lower()
    if target_symbol and target_symbol.lower() == class_stem:
        return None

    # Reusing a capability that lives in ANOTHER file → this task should not create it here.
    if target_file and not _same_file(target_file, fp):
        return (
            f"existing capability '{target_symbol or 'method'}' in {target_file} already "
            f"satisfies this intent — reuse it instead of modifying {_basename(fp)}"
        )
    # Same file, but the task introduces a DIFFERENT new symbol → duplicate of an
    # existing method in the same provider (e.g. checkProjectMembership vs members()).
    if target_symbol and introduced and target_symbol.lower() not in introduced:
        return (
            f"existing '{target_symbol}' already satisfies this intent — reuse it "
            f"instead of adding {', '.join(sorted(introduced))}"
        )
    return None


@dataclass
class TaskGateDecision:
    file_path: str
    accepted: bool
    reason: str


@dataclass
class PreGenerationResult:
    accepted: list = field(default_factory=list)          # kept tasks
    rejected: list = field(default_factory=list)          # TaskGateDecision[]
    hard_block: bool = False
    block_reason: str = ""


def _discover_structured_suppliers(
    tasks: list,
    proven_targets: set[str],
    authorized_files: set[str],
    scope_declared: bool,
    evidence_files: set[str],
    workspace_root: Optional[Union[str, Path]] = None,
) -> dict[str, dict]:
    """Discover tasks that qualify as GENERATED_DEPENDENCY via structured capability contracts.

    Chain:
      authorized change target
      ↓
      consumer task
      ↓
      structured consumes capability
      ↓
      from_task / dependency
      ↓
      producer task
      ↓
      matching produces capability
      ↓
      compatible relationship
      ↓
      repository evidence
      ↓
      safety checks
      ↓
      GENERATED_DEPENDENCY

    Strict invariant: when scope_declared=True, strict whitelist forbids any expansion.
    """
    if scope_declared:
        return {}

    # Identify primary targets
    targets = [
        t for t in tasks
        if _match(getattr(t, "file_path", ""), proven_targets, workspace_root)
        or is_companion(getattr(t, "file_path", ""), proven_targets, workspace_root)
    ]
    if not targets:
        return {}

    structured_suppliers: dict[str, dict] = {}

    for target in targets:
        target_fp = getattr(target, "file_path", "")
        deps = [str(d) for d in (getattr(target, "dependencies", None) or [])]
        contract = getattr(target, "cross_file_contract", None)
        consumes = getattr(contract, "consumes", None) or []

        consumed_symbols = {
            str(getattr(c, "symbol_name", "")).lower()
            for c in consumes if getattr(c, "symbol_name", None)
        }
        consumed_task_ids = {
            str(getattr(c, "created_by_task", ""))
            for c in consumes if getattr(c, "created_by_task", None)
        }

        for supp in tasks:
            if supp == target:
                continue
            supp_fp = getattr(supp, "file_path", "")
            if not supp_fp:
                continue
            supp_id = str(getattr(supp, "id", ""))

            # 1. Dependency or Contract Edge
            dep_edge = (
                (supp_id and (supp_id in deps or supp_id in consumed_task_ids)) or
                any(_match(supp_fp, {d}, workspace_root) for d in deps)
            )
            if not dep_edge:
                continue

            # 2. Producer Capability Match
            supp_contract = getattr(supp, "cross_file_contract", None)
            supp_produces = getattr(supp_contract, "produces", None) or []
            produced_symbols = {
                str(getattr(p, "symbol_name", "")).lower()
                for p in supp_produces if getattr(p, "symbol_name", None)
            }
            supp_methods = {
                str(m).lower() for m in (getattr(supp, "allowed_methods", None) or [])
            }
            tm = getattr(supp, "target_method", None)
            if tm:
                supp_methods.add(str(tm).lower())

            matching_cap = None
            if consumed_symbols and produced_symbols:
                inter = consumed_symbols.intersection(produced_symbols)
                if inter:
                    matching_cap = sorted(inter)[0]
            elif consumed_symbols and supp_methods:
                inter = consumed_symbols.intersection(supp_methods)
                if inter:
                    matching_cap = sorted(inter)[0]

            if not matching_cap:
                # Invariant: A dependency edge alone is INSUFFICIENT without a matching capability!
                continue

            # 3. Safety: Not config / build
            if is_config_file(supp_fp) or is_build_config(supp_fp):
                continue

            # 4. Safety: Not speculative shared model
            if is_shared_model_file(supp_fp):
                continue

            # 5. Cross-boundary check:
            # Direct frontend -> backend implementation dependency without verified API contract is forbidden
            target_is_fe = any(target_fp.lower().endswith(x) for x in (".ts", ".tsx", ".html", ".scss", ".css"))
            supp_is_be = any(supp_fp.lower().endswith(x) for x in (".java", ".kt", ".py", ".cs"))
            if target_is_fe and supp_is_be:
                continue

            # 6. Repository existence check
            exists = False
            if evidence_files and _match(supp_fp, evidence_files, workspace_root):
                exists = True
            elif workspace_root:
                try:
                    exists = (Path(workspace_root) / supp_fp).exists()
                except Exception:
                    pass
            else:
                exists = Path(supp_fp).exists()

            if not exists:
                continue

            structured_suppliers[supp_fp] = {
                "consumer": target_fp,
                "capability": matching_cap,
                "task_id": supp_id,
            }

    return structured_suppliers


def filter_generation_tasks(
    tasks: list,
    evidence_files: set[str] | None = None,
    ticket_required_files: set[str] | None = None,
    reuse_resolver=None,
    grounding_present: bool = True,
    authorized_files: set[str] | None = None,
    forbidden_files: set[str] | None = None,
    scope_declared: bool = False,
    ticket_targets_config: bool = False,
    semantic_resolver=None,
    proven_targets: set[str] | None = None,
    workspace_root: Optional[Union[str, Path]] = None,
    scope_proof: Any | None = None,
) -> PreGenerationResult:
    """Filter plan tasks through change-authorization + reuse/scope gates.

    Args:
        tasks: plan tasks (need .file_path, .task_type.value, optional .target_class).
        evidence_files: repository-evidence-backed file paths.
        ticket_required_files: files the ticket explicitly requires.
        reuse_resolver: CapabilityReuseResolver (or None to skip duplicate checks).
        grounding_present: False only when there is NO grounding at all
            (contracts AND verified capabilities both missing) → hard-block.
        authorized_files: ticket-declared change set (expected_changed/owner).
        forbidden_files: ticket-declared forbidden files.
        scope_declared: True when the ticket declared an authorization scope.
        ticket_targets_config: True when the ticket clearly targets configuration.
        semantic_resolver: hybrid capability resolver (intent-based) used to reject
            tasks that invent a new method/file when an existing capability already
            satisfies the intent. Capability-based, not name-based.
        proven_targets: files evidence proved directly implement the requested
            behavior. Under no declared scope, ONLY these (and their companions)
            may become CHANGE_TARGET — discovery alone never authorizes a write.
        workspace_root: repository root for canonical path containment resolution.
        scope_proof: TicketScopeProof instance. When provided, any writable task
            whose file is not in scope_proof.is_file_writable() is strictly rejected.
    """
    evidence_files = {e.replace("\\", "/").lower() for e in (evidence_files or set())}
    ticket_required_files = {t.replace("\\", "/").lower() for t in (ticket_required_files or set())}
    authorized_files = {a.replace("\\", "/").lower() for a in (authorized_files or set())}
    forbidden_files = {f.replace("\\", "/").lower() for f in (forbidden_files or set())}
    proven_targets = {p.replace("\\", "/").lower() for p in (proven_targets or set())}
    planned = {getattr(t, "file_path", "") for t in tasks}

    structured_suppliers = _discover_structured_suppliers(
        tasks=tasks,
        proven_targets=proven_targets,
        authorized_files=authorized_files,
        scope_declared=scope_declared,
        evidence_files=evidence_files,
        workspace_root=workspace_root,
    )

    result = PreGenerationResult()
    for t in tasks:
        fp = getattr(t, "file_path", "")
        ttype = _ttype(t)
        if ttype == "read_only":
            result.accepted.append(t)
            continue

        # 0a. Hard TicketScopeProof & System-Computed ChangeIntent Gate:
        # Invariant: Only files proven as TICKET_ANCHOR, STRUCTURAL_COMPANION,
        # or explicitly verified writable DEPENDENCY may ever be modified.
        # NEVER trust an LLM boolean or ungrounded claim.
        if scope_proof is not None and hasattr(scope_proof, "is_file_writable"):
            from ticket_to_code.agents.ticket_scope_proof import (
                compute_change_intent_authorization, ChangeIntent, EvidenceRole, WriteAuthorization,
            )
            task_intent = getattr(t, "change_intent", None)
            if not task_intent or not isinstance(task_intent, ChangeIntent):
                role_cand = EvidenceRole.UNVERIFIED
                if hasattr(scope_proof, "get_file_role"):
                    try:
                        role_cand = scope_proof.get_file_role(fp, workspace_root=workspace_root)
                    except Exception:
                        role_cand = EvidenceRole.UNVERIFIED
                task_intent = ChangeIntent(
                    file_path=fp,
                    action=ttype,
                    reason=getattr(t, "selection_reason", "") or getattr(t, "description", ""),
                    role=role_cand,
                    write_authorization=WriteAuthorization.WRITE_ALLOWED if ttype != "read_only" else WriteAuthorization.READ_ONLY,
                    modification_proof=getattr(t, "modification_proof", None),
                )
            computed_intent = compute_change_intent_authorization(task_intent, scope_proof, workspace_root=workspace_root)
            setattr(t, "change_intent", computed_intent)
            if not computed_intent.write_allowed:
                _is_auth = bool(authorized_files and (_match(fp, authorized_files, workspace_root) or is_companion(fp, authorized_files, workspace_root)))
                if not _is_auth:
                    result.rejected.append(TaskGateDecision(
                        fp, False,
                        f"SCOPE_PROOF_VIOLATION: {computed_intent.reason} (role: {computed_intent.role.value})"
                    ))
                    continue
                else:
                    computed_intent.write_authorization = WriteAuthorization.WRITE_ALLOWED
                    setattr(t, "change_intent", computed_intent)

        # 0. Change authorization: evidence is NOT authorization. Reject
        #    forbidden, protected config, and out-of-authorized-scope files.
        role, role_reason = classify_change_role(
            fp,
            authorized_files=authorized_files,
            forbidden_files=forbidden_files,
            scope_declared=scope_declared,
            ticket_targets_config=ticket_targets_config,
            proven_targets=proven_targets,
            workspace_root=workspace_root,
            structured_suppliers=structured_suppliers,
            scope_proof=scope_proof,
        )
        try:
            setattr(t, "change_role", role)
        except Exception:
            pass
        if role in (ChangeRole.READ_ONLY_REFERENCE.value, ChangeRole.UNRELATED_FILE.value):
            result.rejected.append(TaskGateDecision(fp, False, f"{role}: {role_reason}"))
            continue

        # 1. Scope: reject speculative shared-model changes.
        spec = [
            v for v in evaluate_scope(planned, {fp}, evidence_files, ticket_required_files)
            if v.kind == "speculative_shared_model"
        ]
        if spec:
            result.rejected.append(TaskGateDecision(fp, False, spec[0].reason))
            continue

        # 2. CREATE that duplicates an existing capability → reuse instead.
        if ttype == "create" and reuse_resolver is not None:
            cap = getattr(t, "target_class", None) or _stem_type(fp)
            if cap:
                try:
                    res = reuse_resolver.resolve(cap, owner_type=getattr(t, "target_class", None))
                except Exception:
                    res = None
                if res is not None and res.decision in (
                    ReuseDecision.REUSE_EXISTING.value, ReuseDecision.ADAPT_EXISTING.value
                ):
                    result.rejected.append(TaskGateDecision(
                        fp, False,
                        f"reuse existing '{res.existing_symbol}' instead of creating duplicate — {res.evidence}",
                    ))
                    continue

        # 3. Hybrid capability reuse (intent-based): reject tasks that invent a new
        #    method/file when an existing capability already satisfies the intent.
        #    This is what stops the planner inventing a full-stack slice (e.g. a new
        #    backend isMemberOfProject) when members() already provides the answer.
        reuse_reason = _capability_reuse_violation(t, semantic_resolver, workspace_root=workspace_root)
        if reuse_reason:
            result.rejected.append(TaskGateDecision(fp, False, reuse_reason))
            continue

        result.accepted.append(t)

    writable_total = [t for t in tasks if _ttype(t) != "read_only"]
    writable_accepted = [t for t in result.accepted if _ttype(t) != "read_only"]
    if writable_total and not writable_accepted:
        result.hard_block = True
        result.block_reason = "all writable tasks rejected by pre-generation gate"
    elif not grounding_present:
        result.hard_block = True
        result.block_reason = "no grounding: verified contracts and capabilities both missing"

    return result
