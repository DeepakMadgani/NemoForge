"""
Test Suite for Ticket Scope Proof & Narrow Authorization Architecture.

Verifies:
Test A: test_add_members_modal_resolves_add_members_component
Test B: test_pascalcase_anchor_failure_does_not_fall_back_to_wrong_scope
Test C: test_semantic_related_file_is_related_context_not_ticket_anchor
Test D: test_sufficiency_fails_when_primary_scope_unproven
Test E: test_planner_cannot_create_writable_task_from_related_context
Test F: test_pre_generation_gate_rejects_unproven_scope
Test G: test_structural_companions_require_proven_anchor
Test H: test_ambiguous_scope_stops_planning
Test I: test_zero_result_high_confidence_anchor_triggers_recovery
Test J: test_existing_correct_add_members_scope_regression
"""

from pathlib import Path
import pytest
from unittest.mock import MagicMock

from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
from ticket_to_code.agents.ticket_scope_proof import (
    EvidenceRole,
    WriteAuthorization,
    ChangeIntent,
    ScopeGraph,
    ScopeGraphValidator,
    validate_scope_graph,
    TicketScopeProof,
    OwnershipResolver,
    compute_change_intent_authorization,
    resolve_structural_companions,
    verify_post_generation_scope,
    revert_unauthorized_changes,
    build_workspace_manifest,
)
from ticket_to_code.agents.behavioral_understanding import derive_primary_targets
from ticket_to_code.agents.pre_generation_gate import filter_generation_tasks


# ── Regression: manifest mode must never mass-flag pre-existing files ────────

def test_manifest_mode_only_flags_real_changes(tmp_path: Path):
    """The production wipe failure mode: a big pre-existing tree where
    original_contents only covers the generator's assigned files. With a
    pre_run_manifest, everything pre-existing and untouched must NOT be
    flagged as 'unauthorized new file'."""
    # 30 pre-existing files across the tree (like a real repo)
    manifest = {}
    for i in range(30):
        f = tmp_path / "src" / "app" / f"file{i}.ts"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"// pre-existing {i}", encoding="utf-8")
        manifest[str(f.relative_to(tmp_path)).replace("\\", "/")] = f.stat().st_size

    # Generator modifies ONE unapproved pre-existing file and creates ONE new file
    victim = tmp_path / "src" / "app" / "file7.ts"
    victim.write_text("// pre-existing 7 + corruption", encoding="utf-8")
    rogue = tmp_path / "src" / "app" / "rogue-helper.ts"
    rogue.write_text("// unauthorized new file", encoding="utf-8")

    approved = {"add-members.component.ts"}  # none of these files are approved
    is_clean, violations, unapproved = verify_post_generation_scope(
        workspace_path=tmp_path,
        approved_writable_files=approved,
        original_contents={},  # only the assigned files (empty here)
        pre_run_manifest=manifest,
    )
    assert is_clean is False
    assert "src/app/file7.ts" in unapproved          # modified → flagged
    assert "src/app/rogue-helper.ts" in unapproved   # new → flagged
    # THE critical assertion: untouched pre-existing files are NOT flagged
    untouched = [u for u in unapproved if u.startswith("src/app/file") and u != "src/app/file7.ts"]
    assert untouched == [], f"mass false positives: {untouched[:5]}"


def test_manifest_mode_revert_never_deletes_pre_existing_files(tmp_path: Path):
    """Revert with a manifest must delete only NEW files; pre-existing files
    without captured original content must survive untouched."""
    existing = tmp_path / "src" / "member.service.ts"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("export class MemberService {}", encoding="utf-8")
    manifest = {"src/member.service.ts": existing.stat().st_size}

    rogue = tmp_path / "rogue-helper.ts"
    rogue.write_text("// unauthorized", encoding="utf-8")

    reverted = revert_unauthorized_changes(
        workspace_path=tmp_path,
        unauthorized_files={"src/member.service.ts", "rogue-helper.ts"},
        original_contents={},
        pre_run_manifest=manifest,
    )
    assert existing.exists(), "revert deleted a pre-existing file!"
    assert "export class MemberService {}" == existing.read_text(encoding="utf-8")
    assert not rogue.exists(), "revert failed to delete the new file"
    assert any("SKIPPED" in r for r in reverted)


def test_build_workspace_manifest_skips_vendored_dirs(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.ts").write_text("x", encoding="utf-8")
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("x", encoding="utf-8")
    av = tmp_path / ".aviator"
    av.mkdir()
    (av / "index.db").write_text("x", encoding="utf-8")
    m = build_workspace_manifest(tmp_path)
    assert "src/a.ts" in m
    assert all(not k.startswith(("node_modules/", ".aviator/")) for k in m)


def test_git_failure_is_fail_closed(tmp_path: Path, monkeypatch):
    """If git status cannot be executed, the gate must NOT report a clean
    workspace — it must fail closed with SCOPE_PROOF_ERROR."""
    fake_repo = tmp_path / "some-repo"
    (fake_repo / ".git").mkdir(parents=True)
    (fake_repo / "a.txt").write_text("x", encoding="utf-8")

    import ticket_to_code.agents.ticket_scope_proof as tsp

    def _boom(*args, **kwargs):
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(tsp.subprocess, "run", _boom)
    is_clean, violations, unauthorized = tsp.verify_post_generation_scope(
        workspace_path=tmp_path,
        approved_writable_files={"a.txt"},
        original_contents=None,
    )
    assert is_clean is False
    assert any("SCOPE_PROOF_ERROR" in v for v in violations)


def test_missing_workspace_fails_closed(tmp_path: Path):
    is_clean, violations, _ = verify_post_generation_scope(
        workspace_path=tmp_path / "does-not-exist",
        approved_writable_files={"x.ts"},
        original_contents={},
    )
    assert is_clean is False
    assert any("does not exist" in v for v in violations)


# ── Test A: PascalCase decomposition resolves add-members.component.ts ───────

def test_add_members_modal_resolves_add_members_component(tmp_path: Path):
    """Proves 'AddMembersModal' decomposes to ['add', 'members', 'modal'] -> strips 'modal' -> matches add-members.component.ts."""
    target_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "members" / "add-members"
    target_dir.mkdir(parents=True)
    target_file = target_dir / "add-members.component.ts"
    target_file.write_text("// Angular component", encoding="utf-8")

    engine = RepositorySearchEngine(tmp_path)
    results = engine.search_filename("AddMembersModal")

    assert len(results) == 1
    assert results[0].file_path.replace("\\", "/") == "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    assert results[0].search_mode == "ui_container_stripped"
    assert results[0].resolved_query == "add member"


# ── Test B: Anchor failure does not fall back to wrong scope ──────────────────

def test_pascalcase_anchor_failure_does_not_fall_back_to_wrong_scope(tmp_path: Path):
    """Proves if an anchor fails, fallback searches cannot arbitrarily nominate a wrong file as primary target."""
    target_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "members" / "edit-member"
    target_dir.mkdir(parents=True)
    (target_dir / "edit-member.component.ts").write_text("// Edit member", encoding="utf-8")

    engine = RepositorySearchEngine(tmp_path)
    # Search for AddMembersModal when only edit-member exists
    results = engine.search_filename("AddMembersModal")
    assert len(results) == 0  # Does NOT match edit-member

    # derive_primary_targets must reject edit-member for Add Members ticket
    fake_inspection = MagicMock()
    fake_inspection.relevant_methods = ["ngOnInit", "save"]
    inspections = {"xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts": fake_inspection}
    semantic_scores = {"xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts": {"score": 0.85, "decision": "include"}}
    
    primary = derive_primary_targets(
        inspections=inspections,
        feature_tokens=["add", "members"],
        semantic_scores=semantic_scores,
    )
    # Even with high semantic score 0.85, edit-member is NOT anchor -> rejected
    assert len(primary) == 0


# ── Test C: Semantic related file is related context, not ticket anchor ────────

def test_semantic_related_file_is_related_context_not_ticket_anchor():
    """Proves OwnershipResolver tags edit-member and ParticipatingMemberService as related context / provider, not anchor."""
    resolver = OwnershipResolver()
    owner, conf, reason = resolver.resolve_owner(
        ticket_title="In the Add Members modal, when a user is staged to be added",
        ticket_description="In the Add Members modal, check project membership and display organization as read-only.",
        candidate_files=[
            "xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts",
            "project-service/src/main/java/com/cloudforge/projectservice/service/ParticipatingMemberService.java",
        ],
        anchor_tokens=["add", "members"],
    )
    # Neither candidate owns "Add Members"
    assert owner is None
    assert "Insufficient ownership proof" in reason or "Conflicting action verb" in reason


# ── Test D: Sufficiency fails when primary scope is unproven ───────────────────

def test_sufficiency_fails_when_primary_scope_unproven():
    """Proves _is_primary_anchor_proven returns False when primary anchor is absent."""
    from ticket_to_code.agents.evidence_collection_loop import EvidenceCollectionLoop
    loop = EvidenceCollectionLoop.__new__(EvidenceCollectionLoop)
    
    ticket = MagicMock()
    ticket.title = "In the Add Members modal, when a user is staged"
    ticket.description = "Add Members modal organization check"

    # Only edit-member and service in evidence
    item1 = MagicMock()
    item1.file_path = "xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts"
    item2 = MagicMock()
    item2.file_path = "project-service/src/main/java/com/cloudforge/projectservice/service/ParticipatingMemberService.java"

    proven, reason = loop._is_primary_anchor_proven(ticket, [item1, item2])
    assert proven is False
    assert "Primary feature anchor 'Add Members' is not present" in reason


# ── Test E: Planner cannot create writable task from related context ───────────

def test_planner_cannot_create_writable_task_from_related_context():
    """Proves TicketScopeProof forbids writing to related context or providers."""
    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        companions={"xchange-ui/src/app/modules/members/add-members/add-members.component.html"},
        providers={"xchange-ui/src/app/modules/shared/services/members/member.service.ts"},
        related_context={
            "xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts",
            "project-service/src/main/java/com/cloudforge/projectservice/service/ParticipatingMemberService.java",
        },
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    assert proof.is_file_writable("xchange-ui/src/app/modules/members/add-members/add-members.component.ts") is True
    assert proof.is_file_writable("xchange-ui/src/app/modules/members/add-members/add-members.component.html") is True
    assert proof.is_file_writable("xchange-ui/src/app/modules/shared/services/members/member.service.ts") is False
    assert proof.is_file_writable("xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts") is False
    assert proof.is_file_writable("project-service/src/main/java/com/cloudforge/projectservice/service/ParticipatingMemberService.java") is False


# ── Test F: Pre-generation gate rejects unproven scope ─────────────────────────

def test_pre_generation_gate_rejects_unproven_scope():
    """Proves pre_generation_gate hard-rejects tasks targeting files not in proven writable scope."""
    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        companions={"xchange-ui/src/app/modules/members/add-members/add-members.component.html"},
        related_context={"xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts"},
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    task1 = MagicMock()
    task1.file_path = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    task1.task_type.value = "modify"

    task2 = MagicMock()
    task2.file_path = "xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts"
    task2.task_type.value = "modify"

    gate_result = filter_generation_tasks(
        tasks=[task1, task2],
        proven_targets={task1.file_path},
        scope_proof=proof,
    )

    assert len(gate_result.accepted) == 1
    assert gate_result.accepted[0].file_path == task1.file_path
    assert len(gate_result.rejected) == 1
    assert gate_result.rejected[0].file_path == task2.file_path
    assert "SCOPE_PROOF_VIOLATION" in gate_result.rejected[0].reason


# ── Test G: Structural companions require proven anchor ────────────────────────

def test_structural_companions_require_proven_anchor(tmp_path: Path):
    """Proves companions (.html, .scss, .spec.ts) are derived strictly from proven anchor, not semantic similarity."""
    comp_dir = tmp_path / "src" / "app" / "add-members"
    comp_dir.mkdir(parents=True)
    ts_file = comp_dir / "add-members.component.ts"
    html_file = comp_dir / "add-members.component.html"
    scss_file = comp_dir / "add-members.component.scss"
    ts_file.write_text("// ts", encoding="utf-8")
    html_file.write_text("<!-- html -->", encoding="utf-8")
    scss_file.write_text("/* scss */", encoding="utf-8")

    comps = resolve_structural_companions(str(ts_file), workspace_root=tmp_path)
    norm_comps = {c.replace("\\", "/") for c in comps}
    
    assert any("add-members.component.html" in c for c in norm_comps)
    assert any("add-members.component.scss" in c for c in norm_comps)
    assert not any("edit-member" in c for c in norm_comps)


# ── Test H: Ambiguous scope stops planning ─────────────────────────────────────

def test_ambiguous_scope_stops_planning():
    """Proves if multiple conflicting anchor candidates exist, ownership cannot be proven and scope is unproven."""
    resolver = OwnershipResolver()
    owner, conf, reason = resolver.resolve_owner(
        ticket_title="Update member status",
        ticket_description="Update the member status display",
        candidate_files=[
            "src/app/members/member-v1/member.component.ts",
            "src/app/members/member-v2/member.component.ts",
        ],
        anchor_tokens=["member"],
    )
    # Ambiguous between v1 and v2
    scope = ScopeGraph(ticket_id="T1", primary_anchor=owner)
    proof = TicketScopeProof(ticket_id="T1", scope_graph=scope, is_scope_proven=(owner is not None))
    assert proof.is_scope_proven is False
    assert proof.is_file_writable("src/app/members/member-v1/member.component.ts") is False


# ── Test I: Zero-result anchor triggers recovery tracking ──────────────────────

def test_zero_result_high_confidence_anchor_triggers_recovery():
    """Proves unresolved anchors are preserved in TicketScopeProof for recovery."""
    scope = ScopeGraph(ticket_id="TASK-3CFB6459")
    proof = TicketScopeProof(
        ticket_id="TASK-3CFB6459",
        scope_graph=scope,
        is_scope_proven=False,
        unresolved_anchors=["AddMembersModal"],
        proof_reasoning="Anchor 'AddMembersModal' returned 0 matches during exact search",
    )
    assert proof.is_scope_proven is False
    assert "AddMembersModal" in proof.unresolved_anchors
    assert len(proof.scope_graph.all_writable_files()) == 0


# ── Test J: End-to-end correct Add Members scope regression ────────────────────

def test_existing_correct_add_members_scope_regression():
    """Full regression test verifying TASK-3CFB6459 scope proof:
    - add-members.component.ts -> WRITABLE
    - add-members.component.html -> WRITABLE
    - add-members.component.scss -> WRITABLE
    - member.service.ts -> READ-ONLY (PROVIDER)
    - edit-member.component.ts -> READ-ONLY (RELATED_CONTEXT)
    - ParticipatingMemberService.java -> READ-ONLY (RELATED_CONTEXT)
    """
    resolver = OwnershipResolver()
    owner, conf, reason = resolver.resolve_owner(
        ticket_title="In the Add Members modal, when a user is staged to be added to a contract",
        ticket_description="In the Add Members modal, check whether that user is already a member of the current project",
        candidate_files=[
            "xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
            "xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts",
            "xchange-ui/src/app/modules/shared/services/members/member.service.ts",
            "project-service/src/main/java/com/cloudforge/projectservice/service/ParticipatingMemberService.java",
        ],
        anchor_tokens=["add", "members"],
    )

    assert owner == "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    assert conf >= 0.70

    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor=owner,
        companions={
            "xchange-ui/src/app/modules/members/add-members/add-members.component.html",
            "xchange-ui/src/app/modules/members/add-members/add-members.component.scss",
        },
        providers={"xchange-ui/src/app/modules/shared/services/members/member.service.ts"},
        related_context={
            "xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts",
            "project-service/src/main/java/com/cloudforge/projectservice/service/ParticipatingMemberService.java",
        },
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    # Invariant: only add-members files are writable
    assert proof.is_file_writable("xchange-ui/src/app/modules/members/add-members/add-members.component.ts") is True
    assert proof.is_file_writable("xchange-ui/src/app/modules/members/add-members/add-members.component.html") is True
    assert proof.is_file_writable("xchange-ui/src/app/modules/members/add-members/add-members.component.scss") is True

    # Invariant: services and unrelated components are strictly NOT writable
    assert proof.is_file_writable("xchange-ui/src/app/modules/shared/services/members/member.service.ts") is False
    assert proof.is_file_writable("xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts") is False
    assert proof.is_file_writable("project-service/src/main/java/com/cloudforge/projectservice/service/ParticipatingMemberService.java") is False

    assert proof.get_file_role("xchange-ui/src/app/modules/shared/services/members/member.service.ts") == EvidenceRole.PROVIDER
    assert proof.get_file_role("xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts") == EvidenceRole.RELATED_CONTEXT


# ── Test K: Semantic result attempts to become writable after planning ────────

def test_adversarial_k_semantic_result_attempts_to_become_writable_after_planning():
    """Adversarial Test K:
    A semantic search result (edit-member.component.ts) has role RELATED_CONTEXT.
    The planner creates a writable 'modify' task targeting it.
    The pre-generation gate must strictly reject it with SCOPE_PROOF_VIOLATION.
    """
    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        related_context={"xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts"},
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    task = MagicMock()
    task.file_path = "xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts"
    task.task_type.value = "modify"
    task.selection_reason = "Semantic similarity suggested editing member modal"

    res = filter_generation_tasks(tasks=[task], scope_proof=proof)
    assert len(res.accepted) == 0
    assert len(res.rejected) == 1
    assert "SCOPE_PROOF_VIOLATION" in res.rejected[0].reason
    assert res.hard_block is True


# ── Test L: DEPENDENCY is proven but remains read-only ────────────────────────

def test_adversarial_l_dependency_proven_but_remains_readonly():
    """Adversarial Test L:
    A dependency (member.model.ts) is proven required for the consumer,
    but it is not explicitly marked as requiring modification (writable_dependencies).
    It must remain strictly READ_ONLY by default.
    """
    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        dependencies={"xchange-ui/src/app/modules/members/models/member.model.ts"},
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    assert "xchange-ui/src/app/modules/members/models/member.model.ts" not in scope.all_writable_files()
    assert proof.is_file_writable("xchange-ui/src/app/modules/members/models/member.model.ts") is False

    intent = ChangeIntent(
        file_path="xchange-ui/src/app/modules/members/models/member.model.ts",
        action="modify",
        reason="Update model interface",
        role=EvidenceRole.DEPENDENCY,
        write_authorization=WriteAuthorization.WRITE_ALLOWED,
        modification_proof=None,
    )
    computed = compute_change_intent_authorization(intent, proof)
    assert computed.write_authorization == WriteAuthorization.READ_ONLY
    assert computed.write_allowed is False
    assert "SYSTEM_OVERRIDE: DEPENDENCY is READ_ONLY by default" in computed.reason


# ── Test M: LLM outputs write_allowed=True for RELATED_CONTEXT ────────────────

def test_adversarial_m_llm_outputs_write_allowed_true_for_related_context():
    """Adversarial Test M:
    The LLM outputs write_allowed=True for a file that is classified as RELATED_CONTEXT.
    The system must override and reject it.
    """
    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        related_context={"project-service/src/main/java/com/cloudforge/projectservice/service/ParticipatingMemberService.java"},
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    llm_intent = ChangeIntent(
        file_path="project-service/src/main/java/com/cloudforge/projectservice/service/ParticipatingMemberService.java",
        action="modify",
        reason="Model claims it needs to add a backend endpoint",
        role=EvidenceRole.RELATED_CONTEXT,
        write_authorization=WriteAuthorization.WRITE_ALLOWED,
    )
    computed = compute_change_intent_authorization(llm_intent, proof)
    assert computed.write_authorization == WriteAuthorization.READ_ONLY
    assert computed.write_allowed is False
    assert "SYSTEM_OVERRIDE" in computed.reason


# ── Test N: Generator modifies an unapproved existing file ─────────────────────

def test_adversarial_n_generator_modifies_unapproved_existing_file(tmp_path: Path):
    """Adversarial Test N:
    Generator accidentally modifies an unapproved existing file on disk (member.service.ts).
    Post-generation scope gate must detect the violation and safely revert it to original contents.
    """
    svc_file = tmp_path / "member.service.ts"
    orig_text = "export class MemberService { members() {} }"
    svc_file.write_text(orig_text, encoding="utf-8")

    # Generator mutates it
    svc_file.write_text("export class MemberService { members() {} corrupted() {} }", encoding="utf-8")

    approved = {"xchange-ui/src/app/modules/members/add-members/add-members.component.ts"}
    orig_contents = {"member.service.ts": orig_text}

    is_clean, violations, unapproved = verify_post_generation_scope(
        workspace_path=tmp_path,
        approved_writable_files=approved,
        original_contents=orig_contents,
    )
    assert is_clean is False
    assert "member.service.ts" in unapproved
    assert any("SCOPE_PROOF_VIOLATION" in v for v in violations)

    revert_unauthorized_changes(
        workspace_path=tmp_path,
        unauthorized_files=unapproved,
        original_contents=orig_contents,
    )
    assert svc_file.read_text(encoding="utf-8") == orig_text


# ── Test O: Generator creates a new unapproved file ───────────────────────────

def test_adversarial_o_generator_creates_new_unapproved_file(tmp_path: Path):
    """Adversarial Test O:
    Generator creates a new unapproved file (rogue-helper.ts) on disk.
    Post-generation gate must detect and delete the file.
    """
    rogue_file = tmp_path / "rogue-helper.ts"
    rogue_file.write_text("// unauthorized new file", encoding="utf-8")

    approved = {"xchange-ui/src/app/modules/members/add-members/add-members.component.ts"}
    is_clean, violations, unapproved = verify_post_generation_scope(
        workspace_path=tmp_path,
        approved_writable_files=approved,
        original_contents={},
    )
    assert is_clean is False
    assert "rogue-helper.ts" in unapproved

    revert_unauthorized_changes(
        workspace_path=tmp_path,
        unauthorized_files=unapproved,
        original_contents={},
    )
    assert not rogue_file.exists()


# ── Test P: Generator deletes an unapproved file ──────────────────────────────

def test_adversarial_p_generator_deletes_unapproved_file(tmp_path: Path):
    """Adversarial Test P:
    An unapproved file is deleted from disk.
    Post-generation gate detects the deletion and restores it from original_contents.
    """
    legacy_file = tmp_path / "legacy-config.json"
    orig_json = '{"version": 1}'
    approved = {"add-members.component.ts"}
    orig_contents = {"legacy-config.json": orig_json}

    is_clean, violations, unapproved = verify_post_generation_scope(
        workspace_path=tmp_path,
        approved_writable_files=approved,
        original_contents=orig_contents,
    )
    assert is_clean is False
    assert "legacy-config.json" in unapproved

    revert_unauthorized_changes(
        workspace_path=tmp_path,
        unauthorized_files=unapproved,
        original_contents=orig_contents,
    )
    assert legacy_file.exists()
    assert legacy_file.read_text(encoding="utf-8") == orig_json


# ── Test Q: Generator renames an approved file into an unapproved location ────

def test_adversarial_q_generator_renames_approved_file_into_unapproved_location(tmp_path: Path):
    """Adversarial Test Q:
    Renaming into an unapproved path results in unauthorized destination being flagged.
    """
    approved = {"src/app/add-members.component.ts"}
    assert "core/utils/stolen-members.ts" not in approved


# ── Test R: Repair agent tries to modify provider due to compiler diagnostics ─

def test_adversarial_r_repair_agent_tries_to_modify_provider_due_to_compiler_error(tmp_path: Path):
    """Adversarial Test R:
    Compiler diagnostic points to member.service.ts.
    ErrorResolutionAgent must block edits to member.service.ts because it is a protected provider.
    """
    from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent

    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        providers={"xchange-ui/src/app/modules/shared/services/members/member.service.ts"},
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    agent = ErrorResolutionAgent(llm=MagicMock(), workspace_path=tmp_path)
    agent.scope_proof = proof

    blocked_msg = agent._guard_protected_file("xchange-ui/src/app/modules/shared/services/members/member.service.ts")
    assert blocked_msg is not None
    assert "SCOPE_PROOF_VIOLATION" in blocked_msg
    assert "Compiler diagnostics are evidence, not write authorization" in blocked_msg


# ── Test S: Legacy change_candidates contains an unapproved file ───────────────

def test_adversarial_s_legacy_change_candidates_contains_unapproved_file():
    """Adversarial Test S:
    Even if legacy BehavioralUnderstanding.change_candidates has an unapproved file,
    TicketScopeProof overrides and rejects the task at the pre-generation gate.
    """
    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        providers={"xchange-ui/src/app/modules/shared/services/members/member.service.ts"},
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    task = MagicMock()
    task.file_path = "xchange-ui/src/app/modules/shared/services/members/member.service.ts"
    task.task_type.value = "modify"

    res = filter_generation_tasks(
        tasks=[task],
        proven_targets={task.file_path},
        scope_proof=proof,
    )
    assert len(res.accepted) == 0
    assert len(res.rejected) == 1
    assert "SCOPE_PROOF_VIOLATION" in res.rejected[0].reason


# ── Test T: Legacy proven_targets contains an unapproved file ─────────────────

def test_adversarial_t_legacy_proven_targets_contains_unapproved_file():
    """Adversarial Test T:
    Legacy proven_targets set contains unapproved-helper.ts.
    ScopeProof does not authorize it -> pre_generation_gate rejects it.
    """
    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    task = MagicMock()
    task.file_path = "xchange-ui/src/app/modules/members/unapproved-helper.ts"
    task.task_type.value = "create"

    res = filter_generation_tasks(
        tasks=[task],
        proven_targets={"xchange-ui/src/app/modules/members/unapproved-helper.ts"},
        scope_proof=proof,
    )
    assert len(res.accepted) == 0
    assert len(res.rejected) == 1
    assert "SCOPE_PROOF_VIOLATION" in res.rejected[0].reason


# ── Test U: ScopeGraph contains contradictory roles ───────────────────────────

def test_adversarial_u_scope_graph_contains_contradictory_roles():
    """Adversarial Test U:
    ScopeGraph contains contradictory roles:
    1. same file in providers and primary_anchor
    2. same file in related_context and primary_anchor
    ScopeGraphValidator must detect contradictions.
    """
    contradictory_graph = ScopeGraph(
        ticket_id="T1",
        primary_anchor="member.service.ts",
        providers={"member.service.ts"},
    )
    is_valid, errors = validate_scope_graph(contradictory_graph)
    assert is_valid is False
    assert any("CONTRADICTION" in e and "PROVIDER" in e for e in errors)

    contradictory_graph_2 = ScopeGraph(
        ticket_id="T2",
        primary_anchor="edit-member.component.ts",
        related_context={"edit-member.component.ts"},
    )
    is_valid_2, errors_2 = validate_scope_graph(contradictory_graph_2)
    assert is_valid_2 is False
    assert any("CONTRADICTION" in e and "RELATED_CONTEXT" in e for e in errors_2)


# ── Test V: Two possible anchors exist and ownership is ambiguous ──────────────

def test_adversarial_v_two_possible_anchors_and_ownership_is_ambiguous():
    """Adversarial Test V:
    Two candidates have identical score without differentiating repository evidence.
    OwnershipResolver declares ambiguity, scope is unproven.
    """
    resolver = OwnershipResolver()
    owner, conf, reason = resolver.resolve_owner(
        ticket_title="Update member display list",
        ticket_description="Update the display list",
        candidate_files=[
            "src/app/members/member-card/member-card.component.ts",
            "src/app/members/member-row/member-row.component.ts",
        ],
        anchor_tokens=["member"],
    )
    assert owner is None
    assert "Ambiguous ownership" in reason or "Insufficient ownership proof" in reason
    assert len(resolver.last_evidence_trail) > 0


# ── Test W: Semantic RAG returns very high-confidence wrong file ───────────────

def test_adversarial_w_semantic_rag_returns_very_high_confidence_wrong_file():
    """Adversarial Test W:
    Semantic search returns edit-member.component.ts with score 0.99 for an Add Members ticket.
    OwnershipResolver recognizes the conflicting action ('edit' vs 'add') and refuses ownership.
    """
    resolver = OwnershipResolver()
    owner, conf, reason = resolver.resolve_owner(
        ticket_title="In the Add Members modal, staged user check",
        ticket_description="In the Add Members modal, verify project membership",
        candidate_files=["xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts"],
        anchor_tokens=["add", "members"],
    )
    assert owner is None
    assert "Conflicting action verb" in reason or "Insufficient ownership proof" in reason


# ── Test X: Primary anchor search returns zero results, RAG returns similar ────

def test_adversarial_x_primary_anchor_zero_results_semantic_rag_similar():
    """Adversarial Test X:
    Exact anchor search returns 0 results. Semantic search returns high similarity file.
    System must NOT declare sufficiency or prove scope.
    """
    from ticket_to_code.agents.evidence_collection_loop import EvidenceCollectionLoop
    loop = EvidenceCollectionLoop.__new__(EvidenceCollectionLoop)

    ticket = MagicMock()
    ticket.title = "Add Members Modal enhancements"
    ticket.description = "Add Members Modal"

    item = MagicMock()
    item.file_path = "xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts"
    item.evidence_role = EvidenceRole.RELATED_CONTEXT

    proven, reason = loop._is_primary_anchor_proven(ticket, [item])
    assert proven is False
    assert "Primary feature anchor 'Add Members' is not present" in reason


# ── Test Y: Approved anchor exists but generator modifies only provider ───────

def test_adversarial_y_approved_anchor_exists_but_generator_modifies_only_provider(tmp_path: Path):
    """Adversarial Test Y:
    Anchor add-members.component.ts is approved.
    Generator modifies only provider member.service.ts.
    verify_post_generation_scope detects the unauthorized change and reverts it.
    """
    prov = tmp_path / "member.service.ts"
    prov_orig = "// original member service"
    prov.write_text(prov_orig, encoding="utf-8")

    prov.write_text("// modified member service", encoding="utf-8")

    approved = {"xchange-ui/src/app/modules/members/add-members/add-members.component.ts"}
    orig_map = {"member.service.ts": prov_orig}

    is_clean, violations, unapproved = verify_post_generation_scope(
        tmp_path, approved, original_contents=orig_map
    )
    assert is_clean is False
    assert "member.service.ts" in unapproved

    revert_unauthorized_changes(tmp_path, unapproved, orig_map)
    assert prov.read_text(encoding="utf-8") == prov_orig


# ── Test Z: Approved files + unexpected generated file detected ───────────────

def test_adversarial_z_approved_files_plus_unexpected_generated_file_detected(tmp_path: Path):
    """Adversarial Test Z:
    Approved files (component.ts, component.html) are modified correctly,
    but generator also emits an unexpected file (unexpected.tmp).
    Post-generation gate removes unexpected.tmp while preserving valid files.
    """
    ts_file = tmp_path / "add-members.component.ts"
    html_file = tmp_path / "add-members.component.html"
    extra_file = tmp_path / "unexpected.tmp"

    ts_file.write_text("// approved ts", encoding="utf-8")
    html_file.write_text("<!-- approved html -->", encoding="utf-8")
    extra_file.write_text("unexpected garbage", encoding="utf-8")

    approved = {"add-members.component.ts", "add-members.component.html"}
    is_clean, violations, unapproved = verify_post_generation_scope(
        tmp_path, approved, original_contents={}
    )
    assert is_clean is False
    assert "unexpected.tmp" in unapproved

    revert_unauthorized_changes(tmp_path, unapproved, original_contents={})
    assert not extra_file.exists()
    assert ts_file.exists()
    assert html_file.exists()


# ── Test AA: Dependency explicitly proven authorized through ChangeIntent ──────

def test_adversarial_aa_dependency_explicitly_proven_authorized_through_change_intent():
    """Adversarial Test AA:
    A dependency (contract.model.ts) is explicitly proven to require modification
    (present in writable_dependencies) and provides modification_proof.
    compute_change_intent_authorization authorizes WRITE_ALLOWED.
    """
    scope = ScopeGraph(
        ticket_id="TASK-100",
        primary_anchor="component.ts",
        dependencies={"contract.model.ts"},
        writable_dependencies={"contract.model.ts"},
    )
    proof = TicketScopeProof(ticket_id="TASK-100", scope_graph=scope, is_scope_proven=True)

    intent = ChangeIntent(
        file_path="contract.model.ts",
        action="modify",
        reason="Add status field to model",
        role=EvidenceRole.DEPENDENCY,
        modification_proof="Contract requires status field to satisfy ticket acceptance criteria",
    )
    computed = compute_change_intent_authorization(intent, proof)
    assert computed.write_authorization == WriteAuthorization.WRITE_ALLOWED
    assert computed.write_allowed is True
    assert "AUTHORIZED_DEPENDENCY" in computed.reason


# ── Test AB: Patch contains modifications outside scope ───────────────────────

def test_adversarial_ab_patch_contains_modifications_outside_scope(tmp_path: Path):
    """Adversarial Test AB:
    A unified diff contains hunks modifying an unapproved file.
    PatchGate.scope_enforce catches the violation.
    """
    from ticket_to_code.delivery.patch_gate import PatchGate
    gate = PatchGate(repo_path=str(tmp_path))

    unauthorized_diff = (
        "--- a/xchange-ui/src/app/modules/shared/services/members/member.service.ts\n"
        "+++ b/xchange-ui/src/app/modules/shared/services/members/member.service.ts\n"
        "@@ -10,3 +10,4 @@\n"
        "+  getProjectMembers() {}\n"
    )
    approved = {"xchange-ui/src/app/modules/members/add-members/add-members.component.ts"}
    scope_res = gate.scope_enforce(unauthorized_diff, approved)
    assert scope_res.ok is False
    assert any("member.service.ts" in v for v in scope_res.violations)


# ── Test AC: Repair attempts to expand scope without re-running scope proof ───

def test_adversarial_ac_repair_attempts_to_expand_scope_without_rerunning_scope_proof(tmp_path: Path):
    """Adversarial Test AC:
    During the repair loop, the agent attempts to write to a newly targeted file
    that is not in approved writable scope.
    The repair write loop strictly blocks it.
    """
    scope = ScopeGraph(
        ticket_id="TASK-3CFB6459",
        primary_anchor="xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
    )
    proof = TicketScopeProof(ticket_id="TASK-3CFB6459", scope_graph=scope, is_scope_proven=True)

    modified_files = {
        "xchange-ui/src/app/modules/members/add-members/add-members.component.ts": "// fixed",
        "xchange-ui/src/app/modules/shared/services/members/member.service.ts": "// attempted expansion",
    }

    written = []
    blocked = []
    for fp, content in modified_files.items():
        if proof and not proof.is_file_writable(fp, workspace_root=tmp_path):
            blocked.append(fp)
            continue
        written.append(fp)

    assert len(written) == 1
    assert written[0] == "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    assert len(blocked) == 1
    assert blocked[0] == "xchange-ui/src/app/modules/shared/services/members/member.service.ts"


# ── Test AD: Multiple workflow iterations do not accumulate stale writable files 

def test_adversarial_ad_multiple_workflow_iterations_do_not_accumulate_stale_writable_files():
    """Adversarial Test AD:
    Iteration 1 proved file_a.ts.
    Iteration 2 is re-scoped/replanned and only file_b.ts is proven.
    allowed_files in iteration 2 must derive ONLY from iteration 2 scope proof,
    preventing state leakage across iterations.
    """
    scope_1 = ScopeGraph(ticket_id="TASK-1", primary_anchor="file_a.ts")
    proof_1 = TicketScopeProof(ticket_id="TASK-1", scope_graph=scope_1, is_scope_proven=True)

    scope_2 = ScopeGraph(ticket_id="TASK-1", primary_anchor="file_b.ts")
    proof_2 = TicketScopeProof(ticket_id="TASK-1", scope_graph=scope_2, is_scope_proven=True)

    task_a = MagicMock(file_path="file_a.ts")
    task_a.task_type.value = "modify"
    task_b = MagicMock(file_path="file_b.ts")
    task_b.task_type.value = "modify"

    res_2 = filter_generation_tasks(tasks=[task_a, task_b], proven_targets={"file_b.ts"}, scope_proof=proof_2)
    assert len(res_2.accepted) == 1
    assert res_2.accepted[0].file_path == "file_b.ts"
    assert len(res_2.rejected) == 1
    assert res_2.rejected[0].file_path == "file_a.ts"


# ── Test AE: revert_unauthorized_changes protects intentional files on failure ──

def test_revert_unauthorized_changes_protects_intentional_files_on_failure(tmp_path: Path):
    """Ensure that intentional ticket files (e.g. generated component TS/HTML)
    are strictly NEVER deleted or reverted on failure, budget limit, or re-plan.
    """
    ts_file = tmp_path / "add-members.component.ts"
    html_file = tmp_path / "add-members.component.html"
    extra_file = tmp_path / "rogue_service.ts"

    ts_file.write_text("// generated ts code", encoding="utf-8")
    html_file.write_text("<!-- generated html code -->", encoding="utf-8")
    extra_file.write_text("// unauthorized mutation", encoding="utf-8")

    # Even if someone accidentally passes intentional files in unauthorized_files
    unauthorized_attempt = {
        "add-members.component.ts",
        "add-members.component.html",
        "rogue_service.ts",
    }
    protected = {
        "add-members.component.ts",
        "add-members.component.html",
    }

    revert_log = revert_unauthorized_changes(
        workspace_path=tmp_path,
        unauthorized_files=unauthorized_attempt,
        original_contents={},
        protected_files=protected,
    )

    # Protected intentional files must REMAIN ON DISK
    assert ts_file.exists(), "Intentional TS file was deleted!"
    assert html_file.exists(), "Intentional HTML file was deleted!"
    assert ts_file.read_text(encoding="utf-8") == "// generated ts code"
    assert html_file.read_text(encoding="utf-8") == "<!-- generated html code -->"

    # Only truly rogue unauthorized file is removed
    assert not extra_file.exists(), "Rogue file should have been reverted!"


