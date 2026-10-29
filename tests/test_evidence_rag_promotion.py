"""Change B — RAG/semantic is part of Evidence Collection, and PRIMARY_FEATURE_TARGET
is proven by INSPECTION before planning.

Invariants proved here:
    RAG_RELEVANCE != OWNERSHIP
    FILENAME_MATCH != OWNERSHIP   (filename only lowers the behavioral-confidence bar)
    RELATIONSHIP  != WRITE_PERMISSION
    SURFACE -> INSPECT -> PROVE -> (primary) -> AUTHORIZE

Pure logic; no LLM/workspace. The ordering test builds the real LangGraph with a
stubbed WorkflowAgents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.behavioral_understanding import (
    derive_primary_targets, build_behavioral_understanding,
)


class _M:
    def __init__(self, name, signature="", return_type=""):
        self.name, self.signature, self.return_type = name, signature, return_type


class _Insp:
    def __init__(self, class_name, methods, relevant=None, facts=None, api=None):
        self.class_name = class_name
        self.methods = [_M(*m) if isinstance(m, tuple) else _M(m) for m in methods]
        rel = methods if relevant is None else relevant
        self.relevant_methods = [_M(*m) if isinstance(m, tuple) else _M(m) for m in rel]
        self.api_contracts = api or []
        self.relevant_code_regions = {}
        self.facts = facts or []


class _Fact:
    def __init__(self, fact):
        self.fact, self.source = fact, "sqlite"


COMP = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
SVC = "xchange-ui/src/app/modules/shared/services/members/member.service.ts"
BACK = "project-service/src/main/java/com/cloudforge/projectservice/service/ContractMemberService.java"
GEN = "xchange-ui/src/shared/form/services/ot-control.service.ts"


def _one(names):
    return _Insp("X", names)


# 1. RAG-surfaced candidate promoted after behavioral inspection.
def test_rag_candidate_promoted_after_inspection():
    prim = derive_primary_targets({COMP: _one(["onUserSelect", "saveMember"])},
                                  feature_tokens=["members"],
                                  semantic_scores={COMP: {"decision": "include", "score": 0.8}})
    assert COMP in prim


# 2. A RAG hit that was NOT inspected (no relevant methods) is not promoted.
def test_uninspected_rag_hit_not_promoted():
    prim = derive_primary_targets({COMP: _Insp("X", ["a"], relevant=[])},
                                  feature_tokens=["members"],
                                  semantic_scores={COMP: {"decision": "include", "score": 0.95}})
    assert COMP not in prim


# 3. RAG score ALONE never promotes (inspection floor).
def test_rag_score_alone_never_promotes():
    prim = derive_primary_targets({COMP: _Insp("X", ["a"], relevant=[])},
                                  feature_tokens=["members"],
                                  semantic_scores={COMP: {"decision": "include", "score": 0.99}})
    assert COMP not in prim


# 4. A reused capability PROVIDER (backend) is never primary.
def test_reused_provider_not_primary():
    prim = derive_primary_targets({BACK: _one(["members"])}, feature_tokens=["members"],
                                  reuse_provider_files={BACK},
                                  semantic_scores={BACK: {"decision": "include", "score": 0.9}})
    assert BACK not in prim


# 5. A related backend (inspected, moderately relevant, no filename match) stays
#    read-only — it must clear the STRONG behavioral bar, which it does not.
def test_related_backend_not_primary_without_strong_proof():
    prim = derive_primary_targets({BACK: _one(["addContractMember"])},
                                  feature_tokens=["members"],
                                  semantic_scores={BACK: {"decision": "include", "score": 0.55}})
    assert BACK not in prim


# 6. Generic infra explicitly excluded by the verifier → not primary.
def test_generic_infra_excluded_by_semantic():
    prim = derive_primary_targets({GEN: _one(["setControl"])}, feature_tokens=["members"],
                                  semantic_scores={GEN: {"decision": "exclude", "score": 0.9}})
    assert GEN not in prim


# 7. No filename match: semantic relevance alone NEVER promotes to primary target.
def test_no_filename_match_promoted_via_semantic():
    comp = "src/components/participant-assignment-panel.tsx"
    prim = derive_primary_targets({comp: _one(["assignParticipant"])},
                                  feature_tokens=["manage", "participants"],
                                  semantic_scores={comp: {"decision": "include", "score": 0.72}})
    assert comp not in prim


# 8. Fallback (no verifier verdict): filename-feature match + inspection promotes.
def test_fallback_feature_name_plus_inspection():
    prim = derive_primary_targets({COMP: _one(["onUserSelect"])},
                                  feature_tokens=["add", "members"], semantic_scores=None)
    assert COMP in prim


# 9. Fallback does NOT promote a non-feature-named inspected file.
def test_fallback_no_name_no_promote():
    prim = derive_primary_targets({GEN: _one(["setControl"])},
                                  feature_tokens=["add", "members"], semantic_scores=None)
    assert GEN not in prim


# 10. Ordering invariant: evidence (where promotion happens) precedes plan.
def test_rag_promotion_before_planning():
    import inspect
    import ticket_to_code.workflow as wf

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, n):
            return None

    wf.WorkflowAgents = _Stub
    g = wf.create_ticket_to_code_graph(".")
    static = {e for e in g.edges if isinstance(e, tuple)}
    branches = set(getattr(g, "branches", {}).keys())

    # The evidence chain is static and terminates at preflight_check, which
    # conditionally routes to plan → evidence/RAG promotion happens before planning.
    assert ("hypothesis_investigation", "evidence_collection_loop") in static
    assert ("evidence_collection_loop", "evidence_ranking") in static
    assert ("evidence_ranking", "semantic_verification") in static
    assert ("semantic_verification", "preflight_check") in static
    assert "preflight_check" in branches  # conditional route to plan
    # plan is never entered directly before the evidence chain.
    assert not any(dst == "plan" for (src, dst) in static
                   if src in ("unified_analysis", "discover", "hypothesis_investigation"))

    # The promotion (build_behavioral_understanding) runs in the EVIDENCE node,
    # not in the planner — the test fails if promotion moves into/after planning.
    ev_src = inspect.getsource(wf.evidence_collection_loop_node)
    plan_src = inspect.getsource(wf.plan_node)
    assert "build_behavioral_understanding" in ev_src
    assert "build_behavioral_understanding" not in plan_src


# 11. Full Add Members regression at the behavioral-understanding level.
def test_add_members_bu_regression():
    desc = ("In the Add Members modal, check whether that user is already a member of "
            "the current project via any other contract; if so show organization read-only, "
            "else editable dropdown; prevent duplicate contract members; on Save persist "
            "members with organizations.")
    inspections = {
        COMP: _Insp("AddMembersComponent",
                    ["onUserSelect", "onSave", "getOrganizationData"]),
        SVC: _Insp("MemberService",
                   [("members", "filter: FilterParticipantMemberInput", "ParticipatingMember[]")],
                   facts=[_Fact("MemberService.members(filter) returns ParticipatingMember[] with company")],
                   api=["members"]),
        BACK: _Insp("ContractMemberService", ["addContract", "removeContract"]),
        GEN: _Insp("OtControlService", ["setControl", "getControl"]),
    }
    sem = {
        COMP: {"decision": "include", "score": 0.85},   # implements the modal behavior
        SVC: {"decision": "include", "score": 0.60},    # provider (reused)
        BACK: {"decision": "include", "score": 0.55},   # related, not implementing
        GEN: {"decision": "exclude", "score": 0.90},    # generic infra
    }
    bu = build_behavioral_understanding(
        ticket_title="Add Members modal organization display",
        ticket_description=desc,
        inspections=inspections,
        evidence_files=list(inspections.keys()),
        feature_tokens=["members"],
        semantic_scores=sem,
    )
    cc = " ".join(bu.change_candidates)
    refs = " ".join(bu.reference_artifacts)
    # add-members.component.ts → PRIMARY_FEATURE_TARGET (change candidate)
    assert "add-members.component.ts" in cc
    assert "add-members.component.ts" not in refs
    # members() → REUSE_EXISTING
    assert any(r.existing_symbol == "members" and r.decision in ("reuse", "adapt")
               for r in bu.reuse_decisions if "member" in r.requirement)
    # provider + related backend + generic infra → READ_ONLY references
    assert "member.service.ts" in refs
    assert "ContractMemberService.java" in refs
    assert "ContractMemberService.java" not in cc
    assert "ot-control.service.ts" in refs


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
