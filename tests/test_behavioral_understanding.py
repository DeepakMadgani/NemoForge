"""Tests for BehavioralUnderstanding synthesis (understand-first).

Proves the understanding composes existing signals into current/requested/delta,
reuses existing capabilities instead of inventing, keeps reference files out of
change candidates, carries per-item semantics, and reaches the planner block.
Project/language agnostic; pure logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.behavioral_understanding import (
    build_behavioral_understanding, BehavioralUnderstanding,
)


class _Method:
    def __init__(self, name, signature="", return_type=""):
        self.name = name
        self.signature = signature
        self.return_type = return_type


class _Inspection:
    def __init__(self, class_name, methods, relevant=None, api_contracts=None, facts=None):
        self.class_name = class_name
        self.methods = methods
        self.relevant_methods = relevant or methods
        self.api_contracts = api_contracts or []
        self.facts = facts or []


class _Fact:
    def __init__(self, fact, source="sqlite"):
        self.fact = fact
        self.source = source


ADD_MEMBERS_DESC = (
    "In the Add Members modal, when a user is staged to be added to a contract, "
    "check whether that user is already a member of the current project via any "
    "other contract in the same project. If they are, display their organization "
    "as read-only static text showing their existing organization name, not an "
    "editable dropdown. If the user is new to the project, show the editable "
    "organization dropdown. Prevent duplicate contract members and show the "
    "existing warning. On Save, persist members with their assigned organizations."
)

INSPECTIONS = {
    "src/app/modules/shared/services/members/member.service.ts": _Inspection(
        "MemberService",
        [_Method("members", "filter: FilterParticipantMemberInput", "ParticipatingMember[]")],
        api_contracts=["members"],
        facts=[_Fact("MemberService.members(filter) returns ParticipatingMember[] with company")],
    ),
    "src/app/modules/members/add-members/add-members.component.ts": _Inspection(
        "AddMembersComponent",
        [_Method("stageMember"), _Method("save")],
        facts=[_Fact("component calls memberService.members()")],
    ),
}
RELATIONSHIPS = [
    {"source": "add-members.component.ts", "target": "MemberService.members", "type": "calls"},
    {"source": "MemberService.members", "target": "ParticipatingMember.company", "type": "has_type"},
]
AUTHORIZED = {
    "src/app/modules/members/add-members/add-members.component.ts",
    "src/app/modules/members/add-members/add-members.component.html",
}


def _bu():
    return build_behavioral_understanding(
        ticket_title="Add Members modal organization display",
        ticket_description=ADD_MEMBERS_DESC,
        requirements_text="Display existing organization read-only for existing members.",
        inspections=INSPECTIONS,
        relationships=RELATIONSHIPS,
        api_contracts=["members"],
        unresolved=[],
        evidence_files=[
            "src/app/modules/members/add-members/add-members.component.ts",
            "src/app/modules/members/add-members/add-members.component.html",
            "src/app/modules/members/add-members/add-members.component.scss",
            "src/app/modules/shared/services/members/member.service.ts",
            "src/app/modules/shared/models/participating-members.ts",
        ],
        authorized_files=AUTHORIZED,
        scope_declared=True,
    )


# E/F/G. existing capability reuse (members) instead of inventing checkProjectMembership
def test_reuse_existing_membership_capability():
    bu = _bu()
    caps = {c.name for c in bu.existing_capabilities}
    assert "members" in caps
    # a "check ... member" requirement resolves to REUSE of members(), not create_new
    reuse = {r.requirement: r for r in bu.reuse_decisions}
    member_reqs = [r for r in bu.reuse_decisions if "member" in r.requirement]
    assert member_reqs, "should derive a membership requirement"
    assert any(r.decision in ("reuse", "adapt") and r.existing_symbol == "members"
               for r in member_reqs), "membership requirement must reuse existing members()"
    # never invents a checkProjectMembership capability
    assert "checkprojectmembership" not in " ".join(caps).lower()


# K. reference file is NOT a change candidate
def test_reference_not_change_candidate():
    bu = _bu()
    assert any("member.service.ts" in r for r in bu.reference_artifacts)
    assert all("member.service.ts" not in c for c in bu.change_candidates)
    assert any("add-members.component.ts" in c for c in bu.change_candidates)
    # companion scss authorized too
    assert any("add-members.component.scss" in c for c in bu.change_candidates)


# L. per-item / multi-user semantics captured
def test_per_item_semantics_captured():
    bu = _bu()
    assert bu.semantic.get("per_item") or bu.semantic.get("cardinality")
    assert bu.semantic.get("read_only")
    assert any("per-item" in d.lower() or "per selected" in d.lower() for d in bu.behavioral_delta)


# M. behavioral delta present
def test_behavioral_delta_present():
    bu = _bu()
    assert bu.behavioral_delta
    assert bu.requested_behavior


# N/current behavior + relationships derived
def test_current_behavior_and_relationships():
    bu = _bu()
    assert bu.current_behavior
    assert any("calls" in r for r in bu.relevant_relationships)


# O. planner block contains ordered understanding
def test_planning_block_ordered_and_reaches_planner():
    bu = _bu()
    block = bu.to_planning_block()
    for header in [
        "CURRENT BEHAVIOR", "REQUESTED BEHAVIOR", "BEHAVIORAL DELTA",
        "EXISTING CAPABILITIES", "REUSE DECISIONS", "CHANGE CANDIDATES",
        "READ-ONLY REFERENCES",
    ]:
        assert header in block, f"planner block missing {header}"


# R. no invented API — CREATE_NEW only when no capability matches
def test_create_new_only_without_capability():
    bu = build_behavioral_understanding(
        ticket_title="Export widget",
        ticket_description="Generate a brand new export widget capability.",
        inspections={},  # no existing capabilities
        evidence_files=["fe/export.ts"],
    )
    # With no existing capabilities, a required capability may be create_new,
    # but nothing is falsely reported as reusable.
    assert all(r.decision == "create_new" for r in bu.reuse_decisions) or not bu.reuse_decisions


# Fix 2. Evidence-first reuse: existing capability under a DIFFERENT name is reused.
def test_reuse_by_semantics_different_name():
    # Capability name has no "member" token; behavior evidence proves it returns members.
    class _M:
        def __init__(self, name, signature="", return_type=""):
            self.name, self.signature, self.return_type = name, signature, return_type

    class _I:
        def __init__(self):
            self.class_name = "RosterService"
            self.methods = [_M("fetchRoster", "filter: RosterFilter", "ParticipatingMember[]")]
            self.relevant_methods = self.methods
            self.api_contracts = ["roster"]
            self.relevant_code_regions = {}
            self.facts = [type("_F", (), {"fact": "returns project members with company", "source": "sqlite"})()]

    bu = build_behavioral_understanding(
        ticket_title="Membership",
        ticket_description="Check whether the user is already a member of the project.",
        inspections={"fe/roster.service.ts": _I()},
        evidence_files=["fe/roster.service.ts"],
    )
    member_reqs = [r for r in bu.reuse_decisions if "member" in r.requirement]
    assert member_reqs, "should derive a membership requirement"
    assert any(r.decision in ("reuse", "adapt") and r.existing_symbol == "fetchRoster"
               for r in member_reqs), "evidence must reuse fetchRoster despite the name"


# Fix 3. No declared ticket scope still yields minimal, delta-derived candidates.
def test_no_scope_delta_driven_change_candidates():
    bu = build_behavioral_understanding(
        ticket_title="Add Members modal organization display",
        ticket_description=ADD_MEMBERS_DESC,
        inspections=INSPECTIONS,
        relationships=RELATIONSHIPS,
        evidence_files=[
            "src/app/modules/members/add-members/add-members.component.ts",
            "src/app/modules/members/add-members/add-members.component.scss",
            "src/app/modules/shared/services/members/member.service.ts",
            "src/app/modules/shared/models/participating-members.ts",
        ],
        # No authorized_files; evidence proved the component is the primary target.
        primary_targets={"src/app/modules/members/add-members/add-members.component.ts"},
    )
    # Consumer implements the delta → change candidate; reused provider → reference.
    assert any("add-members.component.ts" in c for c in bu.change_candidates)
    assert any("add-members.component.scss" in c for c in bu.change_candidates)  # companion
    assert any("member.service.ts" in r for r in bu.reference_artifacts)
    assert all("member.service.ts" not in c for c in bu.change_candidates)
    assert any("participating-members.ts" in r for r in bu.reference_artifacts)


# RAG cannot override verified evidence: unrelated RAG must NOT force CREATE_NEW,
# and a grounded capability is always preferred over RAG.
def test_rag_cannot_override_grounded_reuse():
    bu = build_behavioral_understanding(
        ticket_title="Membership",
        ticket_description="Check whether the user is already a member of the project.",
        inspections=INSPECTIONS,          # grounded members() exists
        relationships=RELATIONSHIPS,
        evidence_files=list(INSPECTIONS.keys()),
        rag_capabilities=[                 # unrelated RAG (area-service/transmittals)
            {"name": "getTransmittals", "owner": "TransmittalService", "file": "area-service/x.ts"},
        ],
    )
    member = [r for r in bu.reuse_decisions if "member" in r.requirement]
    assert member and any(r.decision == "reuse" and r.existing_symbol == "members" for r in member)
    # No membership requirement resolved to create_new because grounding exists.
    assert all(not (r.decision == "create_new") for r in member)
    # Provenance preserved: grounded capability present, RAG marked supplementary.
    provs = {c.name: c.grounded for c in bu.existing_capabilities}
    assert provs.get("members") is True and provs.get("getTransmittals") is False


# RAG-only (no grounding) never yields blind REUSE — at most ADAPT (verify first).
def test_rag_only_is_supplementary_not_authoritative():
    bu = build_behavioral_understanding(
        ticket_title="Roster",
        ticket_description="Get the project roster list.",
        inspections={},  # no grounded capabilities
        rag_capabilities=[{"name": "getProjectRoster", "owner": "RagSvc"}],
        evidence_files=["fe/x.ts"],
    )
    roster = [r for r in bu.reuse_decisions if "roster" in r.requirement]
    # RAG may suggest ADAPT (verify) but must never be authoritative REUSE.
    assert all(r.decision != "reuse" for r in roster)


# Genuine cross-boundary: a backend file that IS the changing consumer stays a candidate.
def test_backend_change_target_when_evidence_requires():
    class _M:
        def __init__(self, n): self.name=n; self.signature=""; self.return_type=""
    class _I:
        def __init__(self, cls):
            self.class_name=cls; self.methods=[_M("handle")]; self.relevant_methods=self.methods
            self.api_contracts=[]; self.relevant_code_regions={}; self.facts=[]
    # Backend service is the consumer that must change (relationship source), not a reused provider.
    rels = [{"source": "be/OrderService.java", "target": "be/OrderRepo.java", "type": "calls"}]
    bu = build_behavioral_understanding(
        ticket_title="Add order status field",
        ticket_description="Persist a new order status when saving.",
        inspections={"be/OrderService.java": _I("OrderService")},
        relationships=rels,
        evidence_files=["be/OrderService.java", "be/OrderRepo.java"],
        primary_targets={"be/OrderService.java"},  # evidence proved it must change
    )
    # No frontend/backend rule — backend is a change target when evidence proves it.
    assert any("OrderService.java" in c for c in bu.change_candidates)


# TASK 1/2: members() is elevated as a STRUCTURED capability (signature + return
# type + provider + reuse) in the planner block — not a bare name in a comma list.
def test_members_structured_in_planner_block():
    bu = _bu()
    block = bu.to_planning_block()
    assert "members(filter: FilterParticipantMemberInput)" in block
    assert "-> ParticipatingMember[]" in block, "return type must be visible to the planner"
    assert "provider: src/app/modules/shared/services/members/member.service.ts" in block
    assert "REUSE" in block
    cap = next(c for c in bu.existing_capabilities if c.name == "members")
    assert cap.return_type == "ParticipatingMember[]"
    assert cap.reuse_decision in ("reuse", "adapt")
    assert cap.relevant is True
    assert cap.file.endswith("member.service.ts")


# TASK 1/2: a reuse-linked capability is NEVER dropped by the display cap even when
# many other capabilities exist (must-show ordering).
def test_reuse_linked_capability_not_dropped_by_cap():
    class _M:
        def __init__(self, n, s="", r=""): self.name, self.signature, self.return_type = n, s, r

    class _I:
        def __init__(self, cls, methods):
            self.class_name = cls
            self.methods = methods
            self.relevant_methods = methods
            self.api_contracts = []
            self.relevant_code_regions = {}
            self.facts = []

    # 20 noise methods + the real members() capability, members declared LAST.
    noise = [_M(f"noiseMethod{i}") for i in range(20)]
    insp = {
        "fe/noise.service.ts": _I("NoiseService", noise),
        "fe/member.service.ts": _I("MemberService",
                                    [_M("members", "filter: F", "ParticipatingMember[]")]),
    }
    bu = build_behavioral_understanding(
        ticket_title="Membership",
        ticket_description="Check whether the user is already a member of the project.",
        inspections=insp,
        evidence_files=list(insp.keys()),
    )
    block = bu.to_planning_block()
    assert "members(filter: F) -> ParticipatingMember[]" in block, \
        "reuse-linked members() must survive the display cap"


# TASK 7: a relevant backend PROVIDER whose capability is reused stays READ_ONLY_REFERENCE.
def test_relevant_backend_provider_stays_read_only():
    class _M:
        def __init__(self, n, s="", r=""): self.name, self.signature, self.return_type = n, s, r

    class _I:
        def __init__(self):
            self.class_name = "MembershipService"
            self.methods = [_M("findMembers", "projectId: UUID", "List<Member>")]
            self.relevant_methods = self.methods
            self.api_contracts = ["members"]
            self.relevant_code_regions = {}
            self.facts = [type("_F", (), {"fact": "returns project members", "source": "sqlite"})()]

    bu = build_behavioral_understanding(
        ticket_title="Membership check",
        ticket_description="Check whether the user is already a member of the project.",
        inspections={"be/MembershipService.java": _I()},
        relationships=[{"source": "fe/x.component.ts",
                        "target": "MembershipService.findMembers", "type": "calls"}],
        evidence_files=["be/MembershipService.java", "fe/x.component.ts"],
    )
    assert any("MembershipService.java" in r for r in bu.reference_artifacts)
    assert all("MembershipService.java" not in c for c in bu.change_candidates)


# TASK 9: the block the planner receives (prepended to verified_evidence handed to
# create_plan) actually contains the reusable capability + its reuse decision.
def test_planner_input_contains_capability():
    bu = _bu()
    block = bu.to_planning_block()
    assert block.startswith("=== BEHAVIORAL UNDERSTANDING")
    assert "EXISTING CAPABILITIES" in block and "members" in block
    assert "REUSE DECISIONS" in block


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")