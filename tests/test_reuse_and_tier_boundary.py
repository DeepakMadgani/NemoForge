"""Tests for cross-tier coherence: reuse-first discovery, tier boundary, and
pre-write service-call validation.

Background (2026-09-18 live run): for an Add-Members-modal (frontend) ticket,
the planner created a modify task on backend ContractService.java, and the
generator emitted `this.contractService.getProjectMembership()` — a method
that existed NOWHERE — while the existing reusable `member.service.ts ->
members(filter)` capability was ignored. These tests pin the fixes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.reuse_capability_discovery import (
    build_reuse_directive,
    discover_frontend_capabilities,
    frontend_anchor_root,
    is_frontend_file,
    ticket_has_backend_intent,
)


# ── Reuse capability discovery ──────────────────────────────────────────────

def test_discovers_member_service_capability(tmp_path: Path):
    svc_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "shared" / "services" / "members"
    svc_dir.mkdir(parents=True)
    (svc_dir / "member.service.ts").write_text(
        "export class MemberService {\n"
        "  constructor(private http: HttpClient) {}\n"
        "  members(filter: string) { return this.http.get('/api/members'); }\n"
        "  saveMember(m: any) { return this.http.post('/api/members', m); }\n"
        "}\n",
        encoding="utf-8",
    )
    caps = discover_frontend_capabilities(
        str(tmp_path),
        frontend_root=tmp_path / "xchange-ui",
        keywords=["member", "members", "organization"],
    )
    assert caps, "member.service.ts must be discovered"
    top = caps[0]
    assert top["class"] == "MemberService"
    assert top["rel_path"].endswith("shared/services/members/member.service.ts")
    method_names = {m["name"] for m in top["methods"]}
    assert "members" in method_names and "saveMember" in method_names


def test_discovery_skips_node_modules_and_specs(tmp_path: Path):
    nm = tmp_path / "xchange-ui" / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "member.service.ts").write_text("export class MemberService { members() {} }", encoding="utf-8")
    real = tmp_path / "xchange-ui" / "src"
    real.mkdir(parents=True)
    (real / "member.service.spec.ts").write_text("export class MemberService { members() {} }", encoding="utf-8")
    caps = discover_frontend_capabilities(str(tmp_path), frontend_root=tmp_path / "xchange-ui", keywords=["member"])
    assert caps == []


def test_reuse_directive_lists_methods(tmp_path: Path):
    svc = tmp_path / "member.service.ts"
    svc.write_text("export class MemberService { members(filter: string) {} }", encoding="utf-8")
    caps = discover_frontend_capabilities(str(tmp_path), frontend_root=tmp_path, keywords=["member"])
    text = build_reuse_directive(caps)
    assert "REUSE-FIRST" in text
    assert "members(filter: string)" in text
    assert "MUST NOT" in text


# ── Frontend anchoring / tier detection ─────────────────────────────────────

def test_frontend_anchor_root_detected(tmp_path: Path):
    fe = tmp_path / "xchange-ui"
    fe.mkdir()
    (fe / "package.json").write_text("{}", encoding="utf-8")
    anchor = str(fe / "src/app/modules/members/add-members/add-members.component.ts")
    backend = str(tmp_path / "project-service/src/main/java/X.java")
    root = frontend_anchor_root([backend, anchor], str(tmp_path))
    assert root is not None and root.name == "xchange-ui"


def test_no_anchor_for_backend_only(tmp_path: Path):
    backend = str(tmp_path / "project-service/src/main/java/X.java")
    assert frontend_anchor_root([backend], str(tmp_path)) is None


def test_backend_intent_detection():
    assert ticket_has_backend_intent("Add a REST endpoint for member search")
    assert ticket_has_backend_intent("The backend should validate duplicates")
    assert not ticket_has_backend_intent(
        "In the Add Members modal show organization as read-only text when the user is already a member"
    )


def test_is_frontend_file():
    assert is_frontend_file("src/app/x/add-members.component.ts")
    assert is_frontend_file("src/app/x/add-members.component.html")
    assert is_frontend_file("src/app/shared/services/members/member.service.ts")
    assert not is_frontend_file("project-service/src/main/java/com/x/ContractService.java")


# ── Pre-write service-call validation (code_generator) ──────────────────────

def _make_generator():
    from ticket_to_code.agents.code_generator import CodeGeneratorAgent
    gen = CodeGeneratorAgent()
    return gen


def test_hallucinated_service_call_detected():
    gen = _make_generator()
    gen._session_files = {
        "contract.service.ts": (
            "export class ContractService {\n"
            "  getContracts() { return []; }\n"
            "}\n"
        )
    }
    existing = (
        "export class AddMembersComponent {\n"
        "  constructor(private contractService: ContractService) {}\n"
        "}\n"
    )
    generated = (
        "check() {\n"
        "  const sub = this.contractService.getProjectMembership('p1', 'u1').subscribe(x => x);\n"
        "}\n"
    )
    unknown = gen._validate_service_calls(generated, existing)
    assert len(unknown) == 1
    assert "getProjectMembership" in unknown[0]
    assert "ContractService has no method" in unknown[0] or "has no method" in unknown[0]


def test_existing_service_call_not_flagged():
    gen = _make_generator()
    gen._session_files = {
        "contract.service.ts": (
            "export class ContractService {\n"
            "  getProjectMembership(projectId: string, userId: string) { return null; }\n"
            "}\n"
        )
    }
    existing = (
        "export class AddMembersComponent {\n"
        "  constructor(private contractService: ContractService) {}\n"
        "}\n"
    )
    generated = "check() { this.contractService.getProjectMembership('p1', 'u1'); }"
    assert gen._validate_service_calls(generated, existing) == []


def test_unresolvable_service_not_flagged():
    """When the service class cannot be found, do NOT false-flag (TSC will catch)."""
    gen = _make_generator()
    gen._session_files = {}
    gen.workspace_path = None
    existing = "export class C { constructor(private mystery: MysteryService) {} }"
    generated = "x() { this.mystery.something(); }"
    assert gen._validate_service_calls(generated, existing) == []


def test_method_defined_in_generated_content_not_flagged():
    gen = _make_generator()
    gen._session_files = {}
    gen.workspace_path = None
    existing = "export class AddMembersComponent { constructor(private self: AddMembersComponent) {} }"
    generated = (
        "helper() { return 1; }\n"
        "use() { this.self.helper(); }"
    )
    assert gen._validate_service_calls(generated, existing) == []


# ── Reuse directive lifecycle & dict/object safety ──────────────────────────

def test_reuse_directive_lifecycle_reset():
    from ticket_to_code.agents.code_generator import (
        set_reuse_directive,
        clear_reuse_directive,
    )
    import ticket_to_code.agents.code_generator as cg_mod

    set_reuse_directive("REUSE-FIRST: Use MemberService.members")
    assert getattr(cg_mod, "_reuse_directive", "") == "REUSE-FIRST: Use MemberService.members"

    clear_reuse_directive()
    assert getattr(cg_mod, "_reuse_directive", "") == ""


def test_get_ticket_attr_supports_both_dict_and_object():
    from ticket_to_code.workflow import _get_ticket_attr

    # 1. Plain dictionary (from Redis / Celery / JSON payload)
    dict_ticket = {
        "title": "Add Members Modal",
        "description": "Show organization read-only",
        "expected_changed_files": ["add-members.component.ts"],
    }
    assert _get_ticket_attr(dict_ticket, "title") == "Add Members Modal"
    assert _get_ticket_attr(dict_ticket, "description") == "Show organization read-only"
    assert _get_ticket_attr(dict_ticket, "expected_changed_files") == ["add-members.component.ts"]
    assert _get_ticket_attr(dict_ticket, "non_existent", "default_val") == "default_val"

    # 2. Pydantic / Class instance
    class DummyTicket:
        def __init__(self):
            self.title = "Class Ticket"
            self.description = "Class Description"
            self.expected_changed_files = ["class_file.ts"]

    obj_ticket = DummyTicket()
    assert _get_ticket_attr(obj_ticket, "title") == "Class Ticket"
    assert _get_ticket_attr(obj_ticket, "description") == "Class Description"
    assert _get_ticket_attr(obj_ticket, "expected_changed_files") == ["class_file.ts"]
    assert _get_ticket_attr(obj_ticket, "non_existent", "default_val") == "default_val"

    # 3. None ticket
    assert _get_ticket_attr(None, "title", "fallback") == "fallback"
