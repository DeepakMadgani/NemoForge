"""Hybrid capability-reuse enforcement at the pre-generation gate.

Reproduces the Add Members forensic case (TASK-AD7D74EF): the planner invented a
backend method (isMemberOfProject) and a frontend service method
(checkProjectMembership) when MemberService.members() already satisfied the intent.
These tests prove the gate now rejects the invented tasks while keeping the
legitimate component change — capability (intent) based, not name based.

Pure logic; no LLM, no workspace. The hybrid resolver is faked.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks


# ── Fakes ───────────────────────────────────────────────────────────────────
class _Task:
    def __init__(self, file_path, ttype, allowed_methods=None, target_method=None,
                 description="", title=""):
        self.file_path = file_path
        self.task_type = type("_T", (), {"value": ttype})()
        self.allowed_methods = allowed_methods or []
        self.target_method = target_method
        self.description = description
        self.title = title
        self.language = ""
        self.id = file_path


class _Resolution:
    def __init__(self, should_reuse, target_file=None, target_symbol=None, confidence=0.9):
        self.should_reuse = should_reuse
        self.target_file = target_file
        self.target_symbol = target_symbol
        self.confidence = confidence


class _FakeSemanticResolver:
    """Maps a task intent to a canned CapabilityResolution by keyword."""
    def __init__(self, mapping):
        self._mapping = mapping  # keyword -> _Resolution

    def resolve_task(self, intent, title="", hints=None):
        text = f"{title} {intent}".lower()
        for kw, res in self._mapping.items():
            if kw in text:
                return res
        return _Resolution(should_reuse=False)


MEMBER_SVC = "workspace/CloudForge-Commerce/store-ui/src/app/modules/shared/services/members/member.service.ts"
BACKEND = "project-service/src/main/java/com/cloudforge/projectservice/service/ContractMemberService.java"
COMPONENT = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"


def _add_members_resolver():
    reuse_members = _Resolution(
        should_reuse=True, target_file=MEMBER_SVC, target_symbol="members", confidence=0.9
    )
    return _FakeSemanticResolver({
        "membership": reuse_members,
        "member of the project": reuse_members,
        "is a member": reuse_members,
    })


# 1. Invented backend method is rejected (capability lives in member.service.ts).
def test_backend_invention_rejected():
    tasks = [_Task(
        BACKEND, "modify",
        allowed_methods=["isMemberOfProject"],
        title="Add backend logic to check for existing project membership",
        description="check whether a user is already a member of the project",
    )]
    res = filter_generation_tasks(tasks, semantic_resolver=_add_members_resolver(),
                                  proven_targets={BACKEND})
    assert any("ContractMemberService" in r.file_path for r in res.rejected)
    assert res.hard_block  # all writable tasks rejected


# 2. Invented same-file duplicate (checkProjectMembership vs members) is rejected.
def test_frontend_duplicate_method_rejected():
    tasks = [_Task(
        MEMBER_SVC, "modify",
        allowed_methods=["checkProjectMembership"],
        title="Create frontend service method to check project membership",
        description="check whether the user is already a member of the project",
    )]
    # Evidence-backed so the shared/ scope guard does not mask the reuse rejection.
    res = filter_generation_tasks(
        tasks, evidence_files={MEMBER_SVC}, proven_targets={MEMBER_SVC},
        semantic_resolver=_add_members_resolver()
    )
    rej = res.rejected
    assert any("member.service.ts" in r.file_path for r in rej)
    assert any("reuse" in r.reason.lower() or "members" in r.reason.lower() for r in rej)


# 3. Legitimate component change (new UI behaviour) is accepted.
def test_component_behaviour_change_accepted():
    tasks = [_Task(
        COMPONENT, "modify",
        allowed_methods=["onUserSelect"],
        title="Implement conditional organization rendering",
        description="render organization as read-only text for existing members",
    )]
    # Component intent does not map to an existing capability → CREATE_NEW → accepted.
    res = filter_generation_tasks(tasks, semantic_resolver=_add_members_resolver(),
                                  proven_targets={COMPONENT})
    assert not res.rejected and not res.hard_block


# 4. Full Add Members plan: backend + service invented → rejected; component kept.
def test_full_add_members_plan_split():
    tasks = [
        _Task(BACKEND, "modify", allowed_methods=["isMemberOfProject"],
              description="check whether a user is already a member of the project"),
        _Task(MEMBER_SVC, "modify", allowed_methods=["checkProjectMembership"],
              description="check whether the user is already a member of the project"),
        _Task(COMPONENT, "modify", allowed_methods=["onUserSelect"],
              description="render organization read-only for existing members"),
    ]
    res = filter_generation_tasks(
        tasks,
        evidence_files={MEMBER_SVC, BACKEND, COMPONENT},
        proven_targets={COMPONENT},
        semantic_resolver=_add_members_resolver(),
    )
    rejected = {r.file_path for r in res.rejected}
    accepted = {t.file_path for t in res.accepted}
    assert BACKEND in rejected
    assert MEMBER_SVC in rejected
    assert COMPONENT in accepted


# 5. No semantic resolver → enforcement is inert (backwards compatible).
def test_no_resolver_is_inert():
    tasks = [_Task(BACKEND, "modify", allowed_methods=["isMemberOfProject"])]
    # Authorization allows a proven target; with no resolver, reuse enforcement is inert.
    res = filter_generation_tasks(tasks, proven_targets={BACKEND})
    assert not res.rejected


# 6. Modify that edits an existing method (no new symbol) is never reuse-rejected.
def test_modify_existing_method_not_rejected():
    tasks = [_Task(
        MEMBER_SVC, "modify",
        allowed_methods=["members"],  # editing the existing method, not inventing
        description="check whether a user is already a member of the project",
    )]
    res = filter_generation_tasks(
        tasks, evidence_files={MEMBER_SVC}, proven_targets={MEMBER_SVC},
        semantic_resolver=_add_members_resolver()
    )
    # target_symbol 'members' IS the introduced method → not a duplicate invention.
    assert not any("member.service.ts" in r.file_path for r in res.rejected)


# 7. Low-confidence reuse signal does not reject (avoids false positives).
def test_low_confidence_not_rejected():
    weak = _FakeSemanticResolver({
        "membership": _Resolution(should_reuse=True, target_file=MEMBER_SVC,
                                  target_symbol="members", confidence=0.4),
    })
    tasks = [_Task(BACKEND, "modify", allowed_methods=["isMemberOfProject"],
                   description="check project membership")]
    res = filter_generation_tasks(tasks, semantic_resolver=weak, proven_targets={BACKEND})
    assert not res.rejected


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
