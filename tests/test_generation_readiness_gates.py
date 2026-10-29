"""Focused tests for the "understand and generate correctly first" gates.

Covers A–J from the specification. Pure-logic, no LLM, no workspace required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.capability_reuse_resolver import (
    CapabilityReuseResolver, ReuseDecision,
)
from ticket_to_code.agents.scope_expansion_guard import (
    evaluate_scope, is_expansion_allowed,
)
from ticket_to_code.agents.semantic_requirements import extract_semantic_constraints
from ticket_to_code.agents.generation_readiness_gate import (
    assess_generation_readiness, check_contract_adherence,
)
from ticket_to_code.agents.edit_loop_policy import (
    decide_edit_action, EditAction, error_signature,
)
from ticket_to_code.agents.build_diagnostic_classifier import classify_build_diagnostics
from ticket_to_code.models import ErrorCategory


# ── Fakes ───────────────────────────────────────────────────────────────────
class _Defn:
    def __init__(self, methods, props=None):
        self.method_names = methods
        self.property_names = props or []

    def has_member(self, n):
        return n in self.method_names or n in self.property_names


class _Resolver:
    def __init__(self, types):
        self._types = types

    def resolve_type(self, owner):
        return self._types.get(owner)


# ── A. existing capability reused instead of creating a duplicate ────────────
def test_A_reuse_existing_capability():
    resolver = _Resolver({
        "ParticipatingMembersRepository": _Defn(
            ["findByOtdsUserIdAndProjectId", "findByEmailAndProjectId"]
        )
    })
    r = CapabilityReuseResolver(symbol_resolver=resolver)
    res = r.resolve("findByUserIdAndProjectId", owner_type="ParticipatingMembersRepository")
    assert res.decision != ReuseDecision.CREATE_NEW.value
    assert res.existing_symbol in ("findByOtdsUserIdAndProjectId", "findByEmailAndProjectId")
    assert not res.requires_evidence

    # exact match → REUSE
    res2 = r.resolve("findByEmailAndProjectId", owner_type="ParticipatingMembersRepository")
    assert res2.decision == ReuseDecision.REUSE_EXISTING.value


# ── B. unnecessary shared model changes are rejected ─────────────────────────
def test_B_reject_unnecessary_shared_model_change():
    planned = {"src/app/members/add-members.component.ts"}
    proposed = {"src/app/shared/models/displayed-member.model.ts"}
    violations = evaluate_scope(planned, proposed)  # no evidence, not ticket-required
    assert any(v.kind == "speculative_shared_model" for v in violations)


# ── C. scope expansion requires evidence ─────────────────────────────────────
def test_C_scope_expansion_requires_evidence():
    planned = {"a.ts"}
    # Not in plan, no evidence → violation
    assert not is_expansion_allowed("b.ts", planned)
    # Backed by evidence → allowed
    assert is_expansion_allowed("b.ts", planned, evidence_files={"b.ts"})


# ── D. wrong API property/argument guesses prevented (verified contract) ─────
def test_D_contract_first_prevents_guessing():
    verified = ["otdsUserId", "projectId"]
    referenced = ["userId", "projectId"]  # 'userId' is a guess for 'otdsUserId'
    violations = check_contract_adherence(referenced, verified)
    assert any(v.referenced == "userId" and v.verified_closest == "otdsUserId" for v in violations)
    # A fully-correct reference set produces no violations
    assert check_contract_adherence(["otdsUserId", "projectId"], verified) == []


# ── E. cardinality requirements reach generation ─────────────────────────────
def test_E_cardinality_extracted():
    sc = extract_semantic_constraints(
        "Add members",
        "The organization must be shown for every selected member on save.",
    )
    assert sc.cardinality and ("every" in sc.cardinality or sc.is_per_item)
    assert sc.persistence  # 'on save'
    assert sc.to_prompt_block()  # non-empty → reaches the prompt


# ── F. per-user / per-row state requirements reach generation ────────────────
def test_F_per_item_state_extracted():
    sc = extract_semantic_constraints(
        "Add members modal",
        "For each staged user, show their organization as read-only static text per row.",
    )
    assert sc.is_per_item
    assert sc.read_only
    block = sc.to_prompt_block()
    assert "EACH item" in block or "per-row" in block or "per-item" in block


# ── G. clean generation exits without Edit Loop ──────────────────────────────
def test_G_clean_exits():
    action = decide_edit_action(
        compile_errors=[], target_file="a.ts",
        owned_files={"a.ts"}, planned_files={"a.ts"}, prev_signatures=set(),
    )
    assert action == EditAction.EXIT_CLEAN


# ── H. repeated compiler error stops (not 8 iterations) ──────────────────────
def test_H_repeated_error_stops():
    errs = ["a.ts(3,5): error TS2339: Property 'x' does not exist."]
    sig = error_signature(errs)
    action = decide_edit_action(
        compile_errors=errs, target_file="a.ts",
        owned_files={"a.ts"}, planned_files={"a.ts"}, prev_signatures={sig},
    )
    assert action == EditAction.STOP_REPEATED
    # First occurrence → a single targeted repair
    action_first = decide_edit_action(
        compile_errors=errs, target_file="a.ts",
        owned_files={"a.ts"}, planned_files={"a.ts"}, prev_signatures=set(),
    )
    assert action_first == EditAction.TARGETED_REPAIR


# ── I. fundamental dependency problems return to planning ────────────────────
def test_I_new_dependency_returns_to_planning():
    action = decide_edit_action(
        compile_errors=["x.ts(1,1): error TS2307: Cannot find module './new-service'"],
        target_file="x.ts", owned_files={"x.ts"}, planned_files={"x.ts"},
        prev_signatures=set(), new_dependency=True,
    )
    assert action == EditAction.RETURN_TO_PLANNING

    # Unrelated file (not ours/planned) → reject
    action2 = decide_edit_action(
        compile_errors=["issue.ts(1,1): error"], target_file="issue.ts",
        owned_files={"x.ts"}, planned_files={"x.ts"}, prev_signatures=set(),
    )
    assert action2 == EditAction.REJECT_UNRELATED


# ── J. baseline + infrastructure protections unchanged ───────────────────────
def test_J_baseline_and_infra_unchanged():
    our = {"src/app/members/add-members.component.ts"}
    base = [
        "src/app/issues/issue.ts(10,5): error TS2339: Property 'actions' does not exist.",
    ]
    assert classify_build_diagnostics(base, our).is_differential_accept
    infra = ['> Could not download x.jar', '> Could not GET "https://nexus/..": 403 Forbidden']
    assert classify_build_diagnostics(infra, our).is_infrastructure_only
    java = ['/svc/Member.java:[42,30] cannot find symbol: method foo']
    cats = {d.category for d in classify_build_diagnostics(java, {"/svc/member.java"}).diagnostics}
    assert ErrorCategory.GENERATED_DEPENDENCY_FAILURE.value in cats


# ── K. plan→generation gate: unnecessary files cannot be generated ───────────
class _Task:
    def __init__(self, file_path, ttype, target_class=None):
        self.file_path = file_path
        self.task_type = type("_T", (), {"value": ttype})()
        self.target_class = target_class
        self.id = file_path


def test_K_duplicate_create_task_rejected():
    from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
    # Repo already has DisplayedMember → creating it again must be rejected (reuse).
    resolver = _Resolver({"DisplayedMember": _Defn(["organization"], ["id", "name"])})
    reuse = CapabilityReuseResolver(symbol_resolver=resolver)
    tasks = [
        _Task("src/app/members/add-members.component.ts", "modify"),
        _Task("src/app/shared/models/displayed-member.model.ts", "create", target_class="DisplayedMember"),
    ]
    res = filter_generation_tasks(tasks, reuse_resolver=reuse,
                                  proven_targets={"src/app/members/add-members.component.ts"})
    rejected_files = {r.file_path for r in res.rejected}
    assert "src/app/shared/models/displayed-member.model.ts" in rejected_files
    assert any(t.file_path.endswith("add-members.component.ts") for t in res.accepted)
    assert not res.hard_block  # at least one writable task survives


def test_K_speculative_shared_model_rejected():
    from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
    tasks = [
        _Task("src/app/members/add-members.component.ts", "modify"),
        _Task("src/app/shared/models/organization.model.ts", "modify"),  # shared, not required
    ]
    res = filter_generation_tasks(tasks, evidence_files=set(), ticket_required_files=set())
    assert any(r.file_path.endswith("organization.model.ts") for r in res.rejected)


def test_K_evidence_backed_shared_model_allowed():
    from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
    shared = "src/app/shared/models/organization.model.ts"
    tasks = [
        _Task("src/app/members/add-members.component.ts", "modify"),
        _Task(shared, "modify"),
    ]
    # Evidence backs the shared-model change → allowed.
    res = filter_generation_tasks(tasks, evidence_files={shared}, ticket_required_files=set(),
                                  proven_targets={shared})
    assert all(not r.file_path.endswith("organization.model.ts") for r in res.rejected)


def test_K_all_rejected_hard_blocks_to_planning():
    from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
    resolver = _Resolver({"DisplayedMember": _Defn(["organization"])})
    reuse = CapabilityReuseResolver(symbol_resolver=resolver)
    tasks = [
        _Task("src/app/shared/models/displayed-member.model.ts", "create", target_class="DisplayedMember"),
    ]
    res = filter_generation_tasks(tasks, reuse_resolver=reuse)
    assert res.hard_block and "rejected" in res.block_reason


def test_K_no_grounding_hard_blocks():
    from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
    tasks = [_Task("a.ts", "modify")]
    res = filter_generation_tasks(tasks, grounding_present=False, proven_targets={"a.ts"})
    assert res.hard_block and "grounding" in res.block_reason


def test_K_new_capability_without_existing_is_allowed():
    from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
    # No existing equivalent → CREATE_NEW is legitimate (evidence of insufficiency).
    reuse = CapabilityReuseResolver(symbol_resolver=_Resolver({}))
    tasks = [_Task("src/app/members/new-widget.component.ts", "create", target_class="NewWidget")]
    res = filter_generation_tasks(tasks, reuse_resolver=reuse,
                                  proven_targets={"src/app/members/new-widget.component.ts"})
    assert not res.rejected and not res.hard_block


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
