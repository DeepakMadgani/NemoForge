"""Architecture tests: evidence vs change-authorization, config protection,
reuse, contract-first, cross-boundary completeness, semantic readiness, and the
Add Members regression.

Project- and language-agnostic. Pure logic; no LLM, no workspace.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.change_authorization import (
    classify_change_role, ChangeRole, is_config_file, is_build_config,
    verify_api_call_has_contract, provider_change_is_reachable,
    cross_boundary_change_is_complete,
)
from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks
from ticket_to_code.agents.capability_reuse_resolver import CapabilityReuseResolver, ReuseDecision
from ticket_to_code.agents.generation_readiness_gate import (
    check_contract_adherence, evaluate_semantic_readiness, RequirementStatus,
)
from ticket_to_code.agents.edit_loop_policy import decide_edit_action, EditAction, error_signature


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


class _Task:
    def __init__(self, file_path, ttype, target_class=None):
        self.file_path = file_path
        self.task_type = type("_T", (), {"value": ttype})()
        self.target_class = target_class
        self.id = file_path


class _Contract:
    def __init__(self, endpoint, response_type="", name=""):
        self.endpoint = endpoint
        self.response_type = response_type
        self.name = name


AUTH = {
    "src/app/modules/members/add-members/add-members.component.ts",
    "src/app/modules/members/add-members/add-members.component.html",
}
FORBIDDEN = {"project-service/src/main/resources/application.yml"}


# 1. Evidence-only backend file cannot become writable.
def test_01_evidence_only_backend_not_writable():
    role, _ = classify_change_role(
        "project-service/src/main/java/.../ContractMemberService.java",
        authorized_files=AUTH, scope_declared=True,
    )
    assert role == ChangeRole.READ_ONLY_REFERENCE.value


# 2. Dependency-chain evidence cannot become writable automatically.
def test_02_dependency_chain_not_auto_writable():
    tasks = [
        _Task("src/app/modules/members/add-members/add-members.component.ts", "modify"),
        _Task("src/app/modules/shared/services/members/member.service.ts", "modify"),
    ]
    res = filter_generation_tasks(tasks, authorized_files=AUTH, scope_declared=True)
    rej = {r.file_path for r in res.rejected}
    assert any("member.service.ts" in r for r in rej)


# 3. Existing capability causes REUSE instead of CREATE_NEW.
def test_03_reuse_over_create():
    resolver = _Resolver({"MemberService": _Defn(["members"])})
    r = CapabilityReuseResolver(symbol_resolver=resolver).resolve("members", owner_type="MemberService")
    assert r.decision == ReuseDecision.REUSE_EXISTING.value and not r.requires_evidence


# 4. Existing API contract prevents invented API.
def test_04_existing_contract_prevents_invention():
    contracts = [_Contract("members")]
    assert verify_api_call_has_contract("members", contracts)
    assert not verify_api_call_has_contract("isProjectMember", contracts)


# 5. Backend implementation without an exposed consumer contract cannot satisfy a frontend task.
def test_05_backend_without_contract_is_dead():
    contracts = [_Contract("members", response_type="ParticipatingMember")]
    assert not provider_change_is_reachable("findByUserIdAndProjectId", contracts)
    assert provider_change_is_reachable("members", contracts)


# 6. A genuinely required new cross-boundary capability must require a complete contract path.
def test_06_cross_boundary_completeness():
    ok, _ = cross_boundary_change_is_complete({"Svc.java"}, {"comp.ts"}, detected_contracts=[])
    assert ok is False
    ok2, _ = cross_boundary_change_is_complete({"Svc.java"}, {"comp.ts"}, detected_contracts=[_Contract("isMember")])
    assert ok2 is True


# 7. Speculative shared model modification is rejected (no scope declared).
def test_07_speculative_shared_model_rejected():
    tasks = [
        _Task("src/app/members/add-members.component.ts", "modify"),
        _Task("src/app/shared/models/organization.model.ts", "modify"),
    ]
    res = filter_generation_tasks(tasks)  # no scope → shared-model guard applies
    assert any("organization.model.ts" in r.file_path for r in res.rejected)


# 8. Configuration modification without ticket/evidence justification is rejected.
def test_08_config_protected():
    role, _ = classify_change_role(
        "project-service/src/main/resources/application.yml",
        authorized_files=AUTH, scope_declared=True, ticket_targets_config=False,
    )
    assert role == ChangeRole.READ_ONLY_REFERENCE.value
    assert is_config_file("x/application.yml") and is_build_config("pom.xml")


# 9. Evidence-backed reference file remains read-only.
def test_09_evidence_reference_read_only():
    role, _ = classify_change_role(
        "src/app/modules/shared/models/participating-members.ts",
        authorized_files=AUTH, scope_declared=True,
    )
    assert role == ChangeRole.READ_ONLY_REFERENCE.value


# 10. Critical unresolved contract blocks generation.
def test_10_critical_unresolved_blocks():
    rep = evaluate_semantic_readiness(
        critical_requirements=["how to query project membership via other contract"],
        grounded_facts=[],  # nothing grounded
    )
    assert rep.blocks()
    assert rep.per_requirement[list(rep.per_requirement)[0]] == RequirementStatus.UNRESOLVED.value


# 11. Optional uncertainty does not block generation.
def test_11_optional_uncertainty_no_block():
    rep = evaluate_semantic_readiness(
        critical_requirements=["organization field returned by members query"],
        optional_requirements=["cosmetic spacing preference undocumented"],
        grounded_facts=["members query returns company organization field name"],
    )
    assert not rep.blocks()


# 12. Authorized frontend-only change remains allowed.
def test_12_authorized_frontend_allowed():
    role, _ = classify_change_role(
        "src/app/modules/members/add-members/add-members.component.ts",
        authorized_files=AUTH, scope_declared=True,
    )
    assert role == ChangeRole.CHANGE_TARGET.value
    # companion .scss allowed
    role2, _ = classify_change_role(
        "src/app/modules/members/add-members/add-members.component.scss",
        authorized_files=AUTH, scope_declared=True,
    )
    assert role2 == ChangeRole.CHANGE_TARGET.value


# 13. Genuine backend-required ticket remains allowed (backend in authorized scope).
def test_13_backend_ticket_allowed():
    auth = {"project-service/src/main/java/com/x/FooService.java"}
    role, _ = classify_change_role(
        "project-service/src/main/java/com/x/FooService.java",
        authorized_files=auth, scope_declared=True,
    )
    assert role == ChangeRole.CHANGE_TARGET.value


# 14. Genuine full-stack ticket remains allowed.
def test_14_fullstack_ticket_allowed():
    auth = {"be/Foo.java", "fe/foo.component.ts"}
    tasks = [_Task("be/Foo.java", "modify"), _Task("fe/foo.component.ts", "modify")]
    res = filter_generation_tasks(tasks, authorized_files=auth, scope_declared=True)
    assert not res.rejected and not res.hard_block


# 15. Reuse does not create modification tasks in the capability owner unless adaptation required.
def test_15_reuse_no_owner_modification():
    # A CREATE task duplicating existing capability is rejected (reuse instead).
    resolver = _Resolver({"DisplayedMember": _Defn(["organization"])})
    reuse = CapabilityReuseResolver(symbol_resolver=resolver)
    tasks = [
        _Task("fe/add-members.component.ts", "modify"),
        _Task("fe/shared/models/displayed-member.model.ts", "create", target_class="DisplayedMember"),
    ]
    res = filter_generation_tasks(tasks, reuse_resolver=reuse)
    assert any("displayed-member" in r.file_path for r in res.rejected)


# 16. Generator cannot introduce a new API call that has no verified contract.
def test_16_no_invented_api_call():
    contracts = [_Contract("members")]
    assert not verify_api_call_has_contract("isProjectMember", contracts)


# 17. Repeated error stops the edit loop.
def test_17_repeated_error_stops():
    errs = ["a.ts(3,5): error TS2339"]
    assert decide_edit_action(errs, "a.ts", {"a.ts"}, {"a.ts"}, {error_signature(errs)}) == EditAction.STOP_REPEATED


# 18. Clean generation exits without entering repair.
def test_18_clean_exit():
    assert decide_edit_action([], "a.ts", {"a.ts"}, {"a.ts"}, set()) == EditAction.EXIT_CLEAN


# 19. Baseline errors do not authorize unrelated source modifications.
def test_19_baseline_unrelated_rejected():
    assert decide_edit_action(
        ["issue.ts(1,1): error"], "issue.ts", {"x.ts"}, {"x.ts"}, set()
    ) == EditAction.REJECT_UNRELATED


# 20. Infrastructure failures do not authorize source modifications.
def test_20_infra_no_source_repair():
    from ticket_to_code.agents.build_diagnostic_classifier import classify_build_diagnostics
    infra = ['> Could not download x.jar', '> Could not GET "https://nexus/..": 403 Forbidden']
    rep = classify_build_diagnostics(infra, {"x.ts"})
    assert rep.is_infrastructure_only and not rep.blocking


# ── Add Members regression: only the 3 component files are authorized ────────
def test_add_members_regression():
    authorized = {
        "src/app/modules/members/add-members/add-members.component.ts",
        "src/app/modules/members/add-members/add-members.component.html",
        # displayed-member.ts is ticket-declared in the dataset:
        "src/app/modules/shared/models/displayed-member.ts",
    }
    tasks = [
        _Task("src/app/modules/members/add-members/add-members.component.ts", "modify"),
        _Task("src/app/modules/members/add-members/add-members.component.html", "modify"),
        _Task("src/app/modules/members/add-members/add-members.component.scss", "modify"),  # companion
        # The files that WRONGLY changed in the failing run — must be read-only:
        _Task("src/app/modules/shared/services/members/member.service.ts", "modify"),
        _Task("src/app/modules/shared/models/participating-members.ts", "modify"),
        _Task("src/app/modules/shared/models/project.ts", "modify"),
        _Task("src/app/modules/shared/models/saga.types.ts", "modify"),
        _Task("project-service/src/main/java/com/x/ContractMemberService.java", "modify"),
        _Task("project-service/src/main/java/com/x/ContractMemberRepository.java", "modify"),
        _Task("project-service/src/main/resources/application.yml", "modify"),
    ]
    res = filter_generation_tasks(
        tasks, authorized_files=authorized, scope_declared=True, ticket_targets_config=False,
    )
    accepted = {t.file_path for t in res.accepted}
    rejected = {r.file_path for r in res.rejected}

    # The 3 component files must be the ONLY writable targets.
    assert "src/app/modules/members/add-members/add-members.component.ts" in accepted
    assert "src/app/modules/members/add-members/add-members.component.html" in accepted
    assert "src/app/modules/members/add-members/add-members.component.scss" in accepted

    for wrong in [
        "member.service.ts", "participating-members.ts", "project.ts", "saga.types.ts",
        "ContractMemberService.java", "ContractMemberRepository.java", "application.yml",
    ]:
        assert any(wrong in r for r in rejected), f"{wrong} should be rejected (read-only reference)"

    assert not res.hard_block  # component targets survive


# ── DISCOVERED != CHANGE_TARGET invariant ───────────────────────────────────
# 21. No declared scope + not evidence-proven → READ_ONLY (discovery is not authorization).
def test_21_discovered_not_proven_is_read_only():
    role, reason = classify_change_role(
        "project-service/src/main/java/com/x/ContractMemberService.java",
        scope_declared=False,  # interactive ticket: no explicit scope
    )
    assert role == ChangeRole.READ_ONLY_REFERENCE.value
    assert "not evidence-proven" in reason


# 22. Evidence-proven primary target → CHANGE_TARGET even without declared scope.
def test_22_evidence_proven_is_change_target():
    fp = "src/app/modules/members/add-members/add-members.component.ts"
    role, _ = classify_change_role(fp, scope_declared=False, proven_targets={fp})
    assert role == ChangeRole.CHANGE_TARGET.value
    # companion follows the proven target
    role2, _ = classify_change_role(
        "src/app/modules/members/add-members/add-members.component.scss",
        scope_declared=False, proven_targets={fp},
    )
    assert role2 == ChangeRole.CHANGE_TARGET.value


# 23. Relationship/RAG-discovered, unproven backend/generic infra is NOT writable.
def test_23_relationship_discovered_not_writable():
    backend = "project-service/src/main/java/com/x/ContractMemberService.java"
    generic = "xchange-ui/src/shared/form/services/ot-control.service.ts"
    proven = "src/app/modules/members/add-members/add-members.component.ts"
    tasks = [
        _Task(proven, "modify"),
        _Task(backend, "modify"),
        _Task(generic, "modify"),
    ]
    # No declared scope; only the component is evidence-proven.
    res = filter_generation_tasks(tasks, proven_targets={proven})
    accepted = {t.file_path for t in res.accepted}
    rejected = {r.file_path for r in res.rejected}
    assert proven in accepted
    assert backend in rejected and generic in rejected
    assert not res.hard_block


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
