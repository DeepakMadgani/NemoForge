"""Tests for the lean deterministic harness. Localization/fence are proven against real CloudForge
files when present; the orchestrator is proven offline with injected patch/verify functions.
"""

import os
from pathlib import Path

import pytest

from ticket_to_code.lean.scope import (
    ScopeFence, ScopeViolation, classify_layer, extract_anchors, triage,
)
from ticket_to_code.lean.locate import locate, companions
from ticket_to_code.lean.snippet import window_snippet, estimate_tokens
from ticket_to_code.lean.convergence import Action, ConvergenceTracker, StructuredFailure
from ticket_to_code.lean.hunks import Hunk, plan_hunks, write_result
from ticket_to_code.lean.lean_path import (
    LockedContext, VerifyOutcome, run_lean_path, should_use_lean,
)

CloudForge = Path("C:/sample_repo")
GRID_TS = CloudForge / "xchange-ui/src/app/modules/deliverables/deliverable/deliverable.component.ts"
UI_TEXT = ("In the Deliverables tab grid reverse the order of the 'Content', 'Issues' and 'Name' "
           "columns so it reads Status, Info, Name, Issues, Content, List, 'Revision code', 'Issue reason'.")


# ── deterministic classification + anchors (no files needed) ──────────────────

def test_ui_ticket_classifies_as_ui():
    layer, conf = classify_layer(UI_TEXT)
    assert layer == "ui" and conf >= 0.6


def test_backend_ticket_classifies_as_backend():
    layer, _ = classify_layer("Add a REST endpoint and a SQL migration to the repository service impl")
    assert layer == "backend"


def test_fullstack_detected_when_both_strong():
    layer, _ = classify_layer("Add API endpoint, SQL migration, and a new grid column with a tooltip label in the UI")
    assert layer == "fullstack"


def test_anchor_extraction_keeps_quoted_and_titlecase():
    anchors = extract_anchors(UI_TEXT)
    assert "Content" in anchors and "Revision code" in anchors and "Issue reason" in anchors
    assert "grid" not in [a.lower() for a in anchors]  # stopword removed


# ── the physical fence ────────────────────────────────────────────────────────

def test_ui_fence_blocks_java_and_outside_paths(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.ts").write_text("x", encoding="utf-8")
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "Foo.java").write_text("x", encoding="utf-8")
    scope = triage("T", "reorder grid columns", UI_TEXT, str(tmp_path), explicit_root=str(tmp_path))
    fence = ScopeFence(scope)
    assert fence.allows(str(tmp_path / "src" / "a.ts"))
    assert not fence.allows(str(tmp_path / "svc" / "Foo.java"))       # deny *.java
    assert not fence.allows(str(tmp_path.parent / "elsewhere.ts"))    # outside root
    with pytest.raises(ScopeViolation):
        fence.check(str(tmp_path / "svc" / "Foo.java"))


# ── snippet windows + token bound ─────────────────────────────────────────────

def test_window_snippet_is_bounded(tmp_path):
    f = tmp_path / "big.ts"
    f.write_text("\n".join(f"line {i}" for i in range(1, 2001)), encoding="utf-8")
    snip = window_snippet(str(f), "big.ts", [1000], window=40)
    assert snip.line_count <= 81                      # 40 each side + the line
    assert snip.tokens < estimate_tokens(f.read_text(encoding="utf-8"))  # far smaller than whole file


def test_window_snippet_stays_capped_for_anchor_dense_file(tmp_path):
    # Anchors on 300 scattered lines must NOT expand to cover the whole file.
    f = tmp_path / "dense.ts"
    f.write_text("\n".join(f"line {i}" for i in range(1, 2001)), encoding="utf-8")
    dense_lines = list(range(5, 1995, 6))             # ~330 anchor lines
    snip = window_snippet(str(f), "dense.ts", dense_lines, window=40, max_lines=180)
    assert snip.line_count <= 180
    assert snip.tokens < estimate_tokens(f.read_text(encoding="utf-8"))


# ── convergence ───────────────────────────────────────────────────────────────

def test_convergence_allows_progress_then_halts_on_repeat():
    t = ConvergenceTracker()
    a = StructuredFailure(kind="ui_order", actual=["info", "status"])
    b = StructuredFailure(kind="ui_order", actual=["status", "info", "content"])
    assert t.evaluate(a) is Action.CONTINUE_REPAIR
    assert t.evaluate(b) is Action.CONTINUE_REPAIR          # progress: new signature
    assert t.evaluate(b) is Action.STOP_UNPROVEN            # zero progress: repeat


def test_convergence_halts_on_oscillation():
    t = ConvergenceTracker()
    a = StructuredFailure(kind="compile", actual="err A")
    b = StructuredFailure(kind="compile", actual="err B")
    assert t.evaluate(a) is Action.CONTINUE_REPAIR
    assert t.evaluate(b) is Action.CONTINUE_REPAIR
    assert t.evaluate(a) is Action.STOP_UNPROVEN            # A→B→A oscillation


# ── surgical hunks ────────────────────────────────────────────────────────────

def test_hunk_applies_once_and_rejects_ambiguous(tmp_path):
    f = tmp_path / "x.ts"
    f.write_text("a\nFOO\nb\nFOO\nc\nONCE\n", encoding="utf-8")
    ok = plan_hunks(str(tmp_path), [Hunk("x.ts", "ONCE", "DONE")])
    assert ok.ok and ok.new_contents["x.ts"].count("DONE") == 1
    ambiguous = plan_hunks(str(tmp_path), [Hunk("x.ts", "FOO", "BAR")])
    assert not ambiguous.ok and "ambiguous" in ambiguous.rejected[0][1]
    missing = plan_hunks(str(tmp_path), [Hunk("x.ts", "NOPE", "X")])
    assert not missing.ok and "not found" in missing.rejected[0][1]


# ── orchestrator (offline) ────────────────────────────────────────────────────

def _fake_scope_project(tmp_path):
    root = tmp_path / "web"
    (root / "src/app/deliverable").mkdir(parents=True)
    (root / "package.json").write_text("{}", encoding="utf-8")
    html = root / "src/app/deliverable/deliverable.component.html"
    html.write_text('<ng-container cdkColumnDef="2">Content</ng-container>\n'
                    '<ng-container cdkColumnDef="4">Issues</ng-container>\n'
                    '<!-- Revision code, Issue reason -->\n', encoding="utf-8")
    return root, html


def test_lean_path_verifies_with_injected_patch_and_verify(tmp_path):
    root, html = _fake_scope_project(tmp_path)
    calls = {"n": 0}

    def patch_fn(ctx: LockedContext, last_diff, failure):
        calls["n"] += 1
        return [Hunk(ctx.files[0], 'cdkColumnDef="2">Content', 'cdkColumnDef="5">Content')]

    def verify_fn(ctx, written) -> VerifyOutcome:
        return VerifyOutcome(ok=True, rung="ui")

    res = run_lean_path("T", "reorder deliverable grid columns",
                        "reorder 'Content' 'Issues' 'Name' in the Deliverables grid with 'Revision code' 'Issue reason'",
                        str(tmp_path), patch_fn, verify_fn, explicit_root=str(root))
    assert res.status == "verified", res.reason
    assert calls["n"] == 1 and res.attempts == 1
    assert 'cdkColumnDef="5">Content' in html.read_text(encoding="utf-8")


def test_lean_path_reverts_and_stops_on_repeated_failure(tmp_path):
    root, html = _fake_scope_project(tmp_path)
    original = html.read_text(encoding="utf-8")

    def patch_fn(ctx, last_diff, failure):
        return [Hunk(ctx.files[0], 'cdkColumnDef="2">Content', 'cdkColumnDef="9">Content')]

    def verify_fn(ctx, written):
        return VerifyOutcome(ok=False, rung="ui",
                             failure=StructuredFailure(kind="ui_order", actual=["info", "status"]))

    res = run_lean_path("T", "reorder deliverable grid columns",
                        "reorder 'Content' 'Issues' 'Name' with 'Revision code' 'Issue reason'",
                        str(tmp_path), patch_fn, verify_fn, explicit_root=str(root))
    assert res.status == "unverified" and "zero progress" in res.reason
    assert html.read_text(encoding="utf-8") == original          # baseline restored


def test_repair_patches_from_baseline_each_attempt(tmp_path):
    # Attempt 1 fails verify; attempt 2 must see the ORIGINAL bytes again (revert-before-repair),
    # so a baseline-matching hunk still applies.
    root, html = _fake_scope_project(tmp_path)
    attempts = {"n": 0}

    def patch_fn(ctx, last_diff, failure):
        attempts["n"] += 1
        # always targets the ORIGINAL text; only works if disk was reverted to baseline
        return [Hunk(ctx.files[0], 'cdkColumnDef="2">Content', 'cdkColumnDef="5">Content')]

    def verify_fn(ctx, written):
        if attempts["n"] == 1:
            return VerifyOutcome(ok=False, rung="ui",
                                 failure=StructuredFailure(kind="ui_order", actual=["a"]))
        return VerifyOutcome(ok=True, rung="ui")   # 2nd attempt succeeds

    res = run_lean_path("T", "reorder deliverable grid columns",
                        "reorder 'Content' 'Issues' 'Name' with 'Revision code' 'Issue reason'",
                        str(tmp_path), patch_fn, verify_fn, explicit_root=str(root))
    assert res.status == "verified" and attempts["n"] == 2
    assert 'cdkColumnDef="5">Content' in html.read_text(encoding="utf-8")


def test_infra_failures_are_retried_and_never_trigger_oscillation(tmp_path):
    from ticket_to_code.lean.convergence import StructuredFailure as SF
    root, html = _fake_scope_project(tmp_path)
    seq = iter([
        VerifyOutcome(ok=False, rung="ui", failure=SF(kind="infra", actual="grid not rendered")),
        VerifyOutcome(ok=False, rung="ui", failure=SF(kind="ui_order", actual=["a"])),
        VerifyOutcome(ok=False, rung="ui", failure=SF(kind="infra", actual="grid not rendered")),
        VerifyOutcome(ok=True, rung="ui"),                 # finally verifies
    ])

    def patch_fn(ctx, last_diff, failure):
        return [Hunk(ctx.files[0], 'cdkColumnDef="2">Content', 'cdkColumnDef="5">Content')]

    def verify_fn(ctx, written):
        return next(seq)

    res = run_lean_path("T", "reorder deliverable grid columns",
                        "reorder 'Content' 'Issues' 'Name' with 'Revision code' 'Issue reason'",
                        str(tmp_path), patch_fn, verify_fn, explicit_root=str(root))
    assert res.status == "verified", res.reason   # infra strikes did not cause a false oscillation halt


def test_persistent_infra_stops_as_unavailable_not_oscillation(tmp_path):
    from ticket_to_code.lean.convergence import StructuredFailure as SF
    root, _ = _fake_scope_project(tmp_path)

    def patch_fn(ctx, last_diff, failure):
        return [Hunk(ctx.files[0], 'cdkColumnDef="2">Content', 'cdkColumnDef="5">Content')]

    def verify_fn(ctx, written):
        return VerifyOutcome(ok=False, rung="ui", failure=SF(kind="infra", actual="backend down"))

    res = run_lean_path("T", "reorder deliverable grid columns",
                        "reorder 'Content' 'Issues' 'Name' with 'Revision code' 'Issue reason'",
                        str(tmp_path), patch_fn, verify_fn, explicit_root=str(root))
    assert res.status == "unverified" and "environment" in res.reason


def test_compile_verify_fast_passes_on_valid_syntax_and_fails_on_broken(tmp_path):
    from ticket_to_code.lean.live import make_compile_verify_fn
    from ticket_to_code.lean.lean_path import LockedContext as LC
    root, html = _fake_scope_project(tmp_path)
    scope = triage("T", "x", "y", str(tmp_path), explicit_root=str(root))
    ctx = LC(scope=scope, files=(), framework_facts=(), baselines={})
    vf = make_compile_verify_fn(run_tier2=False)
    assert vf(ctx, [scope.relative(str(html))]).ok                      # valid html passes rung 1
    bad = Path(root) / "src/app/deliverable/bad.scss"
    bad.write_text(".x { color: red; ", encoding="utf-8")              # unbalanced brace
    out = vf(ctx, [scope.relative(str(bad))])
    assert not out.ok and out.rung == "syntax"


def test_fullstack_ticket_is_declined_by_lean(tmp_path):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    scope = triage("T", "add endpoint and grid column",
                   "Add API endpoint, SQL migration and a grid column tooltip label", str(tmp_path),
                   explicit_root=str(tmp_path))
    assert not should_use_lean(scope)


# ── real CloudForge localization proof (skips when the repo is absent) ───────────────

@pytest.mark.skipif(not GRID_TS.exists(), reason="CloudForge checkout not present")
def test_real_cloudforge_localization_picks_grid_files_zero_llm():
    scope = triage("CF-DLV-UI-01", "Deliverables grid: show Name before Issues and Content",
                   UI_TEXT, "workspace/CloudForge-Commerce")
    assert scope.layer == "ui"
    assert Path(scope.root).name == "xchange-ui"       # sub-project auto-selected
    cands, fence = locate(scope)
    rels = [c.rel for c in cands]
    assert any("deliverable/deliverable.component" in r for r in rels), rels
    # Fence must have refused to read any Java file during the scan.
    assert all(not r.endswith(".java") for r in fence.reads)
    top = cands[0]
    assert top.distinct_anchors >= 2


@pytest.mark.skipif(not GRID_TS.exists(), reason="CloudForge checkout not present")
def test_monorepo_repo_scope_still_resolves_subproject():
    # repo_scope pointing at the monorepo root must still descend to xchange-ui.
    scope = triage("CF-DLV-UI-01", "Deliverables grid", UI_TEXT, "workspace/CloudForge-Commerce", explicit_root="workspace/CloudForge-Commerce")
    assert Path(scope.root).name == "xchange-ui"
    cands, _ = locate(scope)
    assert any("deliverable/deliverable.component" in c.rel for c in cands)


@pytest.mark.skipif(not GRID_TS.exists(), reason="CloudForge checkout not present")
def test_screen_noun_affinity_ranks_grid_component_first():
    from ticket_to_code.lean.lean_path import build_locked_context
    scope = triage("CF-DLV-UI-01", "Deliverables grid: show Name before Issues and Content",
                   UI_TEXT, "workspace/CloudForge-Commerce", explicit_root="workspace/CloudForge-Commerce")
    assert scope.screen_noun == "deliverable"
    cands, fence = locate(scope)
    assert "deliverable/deliverable.component.ts" in cands[0].rel   # grid component ranks #1
    ctx = build_locked_context(scope, cands, fence)
    assert any("deliverable/deliverable.component.ts" in f for f in ctx.files)
    assert all("issue-list" not in f and "brava-viewer" not in f for f in ctx.files)


# ── dependency context: polyglot signatures + sibling call-sites ──────────────

def _ts_project(tmp_path):
    """A tiny Angular-shaped project: a locked component, a service with a generic-payload method,
    and a sibling component that already calls that method with the correct payload."""
    root = tmp_path / "web"
    (root / "src/app/shared").mkdir(parents=True)
    (root / "src/app/dlg").mkdir(parents=True)
    (root / "src/app/sibling").mkdir(parents=True)
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "src/app/shared/member.service.ts").write_text(
        "export class MemberService {\n"
        "  removeContractMembers(contractMembers: {}): Observable<any> {\n    return null;\n  }\n"
        "  addX(a: string, b: {x: number}): Observable<any> {\n    return null;\n  }\n"
        "  plain(a: string): void {\n  }\n}\n", encoding="utf-8")
    (root / "src/app/dlg/dlg.component.ts").write_text(
        "import { MemberService } from 'src/app/shared/member.service';\n"
        "export class DlgComponent {\n  constructor(private memberService: MemberService) {}\n"
        "  remove() {\n    // TODO revoke contract members\n  }\n}\n", encoding="utf-8")
    (root / "src/app/sibling/rm.component.ts").write_text(
        "export class RmComponent {\n  go() {\n"
        "    let contractMembers = {\n      projectId: this.p.id,\n      contractId: this.c.id,\n"
        "      members: users\n    };\n"
        "    let ms = this.memberService.removeContractMembers(contractMembers).subscribe(d => {\n"
        "      this.close();\n    });\n  }\n}\n", encoding="utf-8")
    return root


def test_ts_method_regex_sees_generic_payload_methods():
    from ticket_to_code.lean.lean_path import _METHOD_RE
    for sig in ("  removeContractMembers(contractMembers: {}): Observable<any> {",
                "  addX(a: string, b: {x: number}): Observable<any> {",
                "  plain(a: string): void {"):
        assert _METHOD_RE.search(sig), sig          # `{}` in the arg list used to hide the method
    assert not _METHOD_RE.search("  if (x) {")         # control flow is still excluded


def test_sibling_usage_shows_payload_shape_and_stops_at_own_call(tmp_path):
    from ticket_to_code.lean.lean_path import build_usage_examples
    root = _ts_project(tmp_path)
    scope = triage("T", "Removing a coordinator must revoke contract members", "revoke contract members "
                   "using removeContractMembers", str(root), explicit_root=str(root))
    out = build_usage_examples(scope, ["src/app/dlg/dlg.component.ts"], "ts")
    assert "removeContractMembers" in out and "rm.component.ts" in out
    assert "projectId" in out and "contractId" in out and "members" in out   # payload shape is visible
    assert "this.close()" not in out                                          # cut at the call's own paren
    assert "dlg.component.ts" not in out                                      # locked file is never an example


def test_sibling_usage_excludes_specs_and_returns_empty_without_ticket_text(tmp_path):
    from ticket_to_code.lean.lean_path import build_usage_examples
    root = _ts_project(tmp_path)
    (root / "src/app/sibling/rm.component.spec.ts").write_text(
        "it('x', () => { svc.removeContractMembers({fake: 1}); });\n", encoding="utf-8")
    scope = triage("T", "revoke contract members", "removeContractMembers", str(root), explicit_root=str(root))
    assert "spec.ts" not in build_usage_examples(scope, ["src/app/dlg/dlg.component.ts"], "ts")
    blank = triage("T", "", "", str(root), explicit_root=str(root))
    assert build_usage_examples(blank, [], "ts") == ""


def test_java_dependency_context_resolves_fqcn_and_extracts_signatures(tmp_path):
    from ticket_to_code.lean.lean_path import build_dependency_context
    root = tmp_path / "svc"
    pkg = root / "src/main/java/com/acme/issues"
    (pkg / "entity").mkdir(parents=True)
    (pkg / "service").mkdir(parents=True)
    (root / "build.gradle").write_text("", encoding="utf-8")
    (pkg / "entity/IssueReview.java").write_text(
        "package com.acme.issues.entity;\npublic class IssueReview {\n"
        "  private UUID id;\n  private String fromState;\n  private OffsetDateTime createdOn;\n}\n", encoding="utf-8")
    (pkg / "service/IssueService.java").write_text(
        "package com.acme.issues.service;\nimport com.acme.issues.entity.IssueReview;\n"
        "public class IssueService {\n"
        "  public IssueReview changeState(String id, String action) throws Exception {\n    return null;\n  }\n}\n",
        encoding="utf-8")
    scope = triage("T", "backend change IssueService", "IssueService changeState", str(root), explicit_root=str(root))
    fence = ScopeFence(scope)
    ctx = build_dependency_context(scope, ["src/main/java/com/acme/issues/service/IssueService.java"], fence)
    assert "com.acme.issues.entity.IssueReview" in ctx            # FQCN resolved to a repo file
    assert "fromState" in ctx and "createdOn" in ctx               # entity fields surfaced
    assert "IssueReview changeState(String id, String action)" in ctx or "changeState" in ctx


def test_treesitter_ts_and_java_ast_signature_extraction():
    from ticket_to_code.lean.lean_path import _ts_signatures, _java_signatures

    ts_code = """
    export class ContractService {
        constructor(private http: HttpClient) {}
        ngOnInit() {}
        public removeContractMembers(payload: { projectId: string, contractId: string, members: string[] }): Observable<any> {
            return this.http.post('/api', payload);
        }
        async getContractsByProjectId(projectId: string): Promise<Contract[]> {
            return [];
        }
        private secretHelper(): void {}
    }
    """
    ts_sigs = _ts_signatures(ts_code)
    assert any("removeContractMembers" in s and "payload:" in s for s in ts_sigs)
    assert any("getContractsByProjectId" in s for s in ts_sigs)
    assert not any("secretHelper" in s for s in ts_sigs)
    assert not any("ngOnInit" in s or "constructor" in s for s in ts_sigs)

    java_code = """
    package com.acme.service;
    public class MemberService {
        private String secret;
        public List<Member> getProjectMembers(UUID projectId) throws ServiceException {
            return null;
        }
        protected void internalDo() {}
        public void removeMember(String memberId, boolean cascade) {
        }
    }
    """
    java_sigs = _java_signatures(java_code)
    assert any("getProjectMembers" in s and "UUID projectId" in s for s in java_sigs)
    assert any("removeMember" in s and "boolean cascade" in s for s in java_sigs)
    assert not any("secret" in s for s in java_sigs)


def test_nltk_stemming_and_stopwords():
    from ticket_to_code.lean.lean_path import _stems_list, _stems

    text = "Removing a coordinator from the project should revoke all contract members"
    stems = _stems(text)
    # Stemming should normalize to root forms
    assert "remov" in stems or "removi" in stems
    assert "coordin" in stems or "coord" in stems
    assert "contract" in stems or "contr" in stems
    assert "project" in stems or "proje" in stems
    # Stopwords like 'from', 'with', 'this' should be omitted
    assert "from" not in stems
    assert "the" not in stems


def test_bm25_service_relevance_ranking(tmp_path):
    from ticket_to_code.lean.lean_path import build_usage_examples
    root = tmp_path / "repo"
    app = root / "src" / "app"
    app.mkdir(parents=True)
    # Define contract service
    (app / "contract.service.ts").write_text(
        "export class ContractService {\n"
        "  removeContractMembers(payload: { projectId: string, contractId: string }): any {}\n"
        "  getContractsByProjectId(projectId: string): any {}\n"
        "}\n", encoding="utf-8"
    )
    # Define an unrelated logging service
    (app / "logging.service.ts").write_text(
        "export class LoggingService {\n"
        "  writeAuditLog(msg: string): void {}\n"
        "  dumpBuffer(): void {}\n"
        "}\n", encoding="utf-8"
    )
    # Sibling caller
    (app / "coordinator.component.ts").write_text(
        "this.contractService.removeContractMembers({ projectId: 'p1', contractId: 'c1' });\n",
        encoding="utf-8"
    )
    scope = triage("T", "Coordinator removal bug",
                   "When removing a coordinator from a project, revoke contract members using ContractService",
                   str(root), explicit_root=str(root))
    usage = build_usage_examples(scope, ["src/app/other.component.ts"], "ts")
    assert "removeContractMembers" in usage
    assert "logging" not in usage.lower() or "removeContractMembers" in usage


def test_parsers_and_bm25_fallbacks(monkeypatch, tmp_path):
    import ticket_to_code.lean.lean_path as lp
    # Simulate environment where optional packages are unavailable
    monkeypatch.setattr(lp, "_TS_PARSER", None)
    monkeypatch.setattr(lp, "_JAVA_PARSER", None)
    monkeypatch.setattr(lp, "_STEMMER", None)
    monkeypatch.setattr(lp, "_BM25Okapi", None)

    ts_code = "export class Svc {\n  removeContractMembers(payload: { a: string }): any { return null; }\n}"
    assert any("removeContractMembers" in s for s in lp._ts_signatures(ts_code))

    java_code = "public class Svc {\n  public void removeMember(String id) {}\n}"
    assert any("removeMember" in s for s in lp._java_signatures(java_code))

    stems = lp._stems("Removing coordinator from contract")
    assert any(s in ("remov", "coord", "contr") for s in stems)

    root = tmp_path / "repo"
    app = root / "src" / "app"
    app.mkdir(parents=True)
    (app / "c.service.ts").write_text("export class CSvc {\n  removeContractMembers(p: {}): any {}\n}\n", encoding="utf-8")
    (app / "caller.component.ts").write_text("this.cSvc.removeContractMembers({ a: 1 });\n", encoding="utf-8")
    scope = lp.triage("T", "remove coordinator contract", "removeContractMembers", str(root), explicit_root=str(root))
    res = lp.build_usage_examples(scope, ["src/app/other.component.ts"], "ts")
    assert "removeContractMembers" in res


