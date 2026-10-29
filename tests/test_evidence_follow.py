"""Fix 1 test: relationship-following deterministically inspects the provider.

Exercises EvidenceCollectionLoop._auto_follow_and_inspect with the existing
primitives stubbed, proving a followable edge (component → service) causes the
provider to be inspected (not merely recorded).
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.evidence_collection_loop import (
    EvidenceCollectionLoop, EvidenceKnowledge, ArtifactInspection,
    MethodLocation, InspectionFact, _top_root,
)


class _Ctx:
    def __init__(self):
        self.visited_files = set()


def test_auto_follow_inspects_provider():
    loop = object.__new__(EvidenceCollectionLoop)  # bypass heavy __init__
    loop._workspace = None

    ek = EvidenceKnowledge()
    evidence = {
        "fe/add-members.component.ts": {
            "relevant": True,
            "inspection": ArtifactInspection(file_path="fe/add-members.component.ts"),
            "followable": [{
                "kind": "calls",
                "source": "fe/add-members.component.ts",
                "dst_name": "MemberService",
                "target_file": "fe/member.service.ts",
                "visited": False,
            }],
        }
    }

    # Stub only the existing primitives the auto-follow REUSES.
    loop._should_inspect = lambda fp, verdict, ev, tc, kn: True

    def _inspect(fp, tc, ev, ht=""):
        insp = ArtifactInspection(file_path=fp, class_name="MemberService")
        insp.methods = [MethodLocation(
            name="members", signature="filter", return_type="ParticipatingMember[]",
            line_start=10, line_end=20,
        )]
        insp.relevant_methods = insp.methods
        insp.api_contracts = ["members"]
        insp.facts = [InspectionFact(fact="members() returns ParticipatingMember[]", source="sqlite")]
        return insp

    loop._inspect_artifact = _inspect
    loop._read_relevant_region = lambda fp, s, e, max_lines=80: "return this.gql.query(members)"
    loop._get_followable_relationships = lambda fp, visited, ev: []

    ticket = types.SimpleNamespace(title="Add Members", description="check member organization")
    loop._auto_follow_and_inspect(ticket, [], evidence, ek, _Ctx())

    # The provider service was FOLLOWED and inspected (not just recorded).
    assert "fe/member.service.ts" in ek.inspection_knowledge, "provider must be inspected"
    insp = ek.inspection_knowledge["fe/member.service.ts"]
    assert any(m.name == "members" for m in insp.methods)
    assert "members" in ek.api_contracts
    assert insp.relevant_code_regions, "should escalate to actual provider code"


def test_auto_follow_i18n_relationship():
    # HTML → translation key → en.json must be followed and inspected (non-code rel).
    loop = object.__new__(EvidenceCollectionLoop)
    loop._workspace = None
    ek = EvidenceKnowledge()
    evidence = {
        "fe/members.component.html": {
            "relevant": True,
            "inspection": ArtifactInspection(file_path="fe/members.component.html"),
            "followable": [{
                "kind": "i18n", "source": "fe/members.component.html",
                "dst_name": "members.load.error", "target_file": "fe/i18n/en.json",
                "visited": False,
            }],
        }
    }
    loop._should_inspect = lambda *a, **k: True

    def _inspect(fp, tc, ev, ht=""):
        insp = ArtifactInspection(file_path=fp)
        insp.facts = [InspectionFact(fact="en.json defines members.load.error wording", source="sqlite")]
        return insp

    loop._inspect_artifact = _inspect
    loop._read_relevant_region = lambda *a, **k: ""
    loop._get_followable_relationships = lambda *a, **k: []
    ticket = types.SimpleNamespace(title="Change members load error", description="reword error")
    loop._auto_follow_and_inspect(ticket, [], evidence, ek, _Ctx())
    assert "fe/i18n/en.json" in ek.inspection_knowledge, "i18n target must be followed + inspected"


def test_auto_follow_bounded_and_skips_visited():
    loop = object.__new__(EvidenceCollectionLoop)
    loop._workspace = None
    ek = EvidenceKnowledge()
    # target already inspected → must be skipped
    evidence = {
        "a.ts": {
            "relevant": True,
            "inspection": ArtifactInspection(file_path="a.ts"),
            "followable": [{"kind": "calls", "source": "a.ts", "dst_name": "B",
                            "target_file": "b.ts", "visited": True}],
        },
        "b.ts": {"inspection": ArtifactInspection(file_path="b.ts")},
    }
    calls = {"n": 0}

    def _inspect(fp, tc, ev, ht=""):
        calls["n"] += 1
        return ArtifactInspection(file_path=fp)

    loop._should_inspect = lambda *a, **k: True
    loop._inspect_artifact = _inspect
    loop._read_relevant_region = lambda *a, **k: ""
    loop._get_followable_relationships = lambda *a, **k: []
    ticket = types.SimpleNamespace(title="t", description="d")
    loop._auto_follow_and_inspect(ticket, [], evidence, ek, _Ctx())
    assert calls["n"] == 0, "already-inspected/visited target must not be re-inspected"


def test_top_root_helper():
    # repo_search yields workspace-relative paths (as seen in the real trace),
    # so the app/service segment is the top root.
    assert _top_root("xchange-ui/src/app/x.ts") == "xchange-ui"
    assert _top_root("area-service/src/main/Foo.java") == "area-service"
    assert _top_root("project-service/src/main/Bar.java") == "project-service"
    assert _top_root("single.ts") == ""  # no distinct top segment → no filtering


class _Hit:
    def __init__(self, fp, mt, ln=1):
        self.file_path = fp
        self.matched_text = mt
        self.line_number = ln


class _RepoSearch:
    """Stub: en.json (xchange-ui) defines a key; the key appears in a xchange-ui
    component AND an unrelated area-service Java constant."""
    def search_literal(self, query, extensions=None, max_results=10):
        exts = extensions or set()
        if ".json" in exts:
            return [_Hit("xchange-ui/src/app/i18n/en.json", '"members.add.title": "Add Member"')]
        if ".properties" in exts:
            return []
        return [
            _Hit("xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
                 "translate members.add.title"),
            _Hit("area-service/src/main/java/com/x/TransmittalConstants.java",
                 "members.add.title"),
        ]

    def flush_trace(self):
        pass


def test_i18n_key_consumers_constrained_to_same_app_root():
    # The failure the E2E exposed: an Angular UI key pulling in area-service Java.
    loop = object.__new__(EvidenceCollectionLoop)
    loop._workspace = None
    loop.repo_search = _RepoSearch()
    items = loop._trace_i18n_chain("Add Member", set())
    paths = {i.file_path for i in items}
    assert any("add-members.component.ts" in p for p in paths), "same-app consumer kept"
    assert not any("area-service" in p for p in paths), "cross-service noise must be excluded"


class _DirectTextRepo:
    """No i18n key found → direct-text fallback. The label appears verbatim in
    two xchange-ui files (real localization) and one unrelated area-service file."""
    def __init__(self, hits):
        self._hits = hits

    def search_literal(self, query, extensions=None, max_results=10):
        exts = extensions or set()
        if ".json" in exts or ".properties" in exts:
            return []  # force the direct-text fallback path
        return list(self._hits)


def test_direct_text_cross_root_outlier_excluded():
    # Majority root (xchange-ui) localizes; lone area-service verbatim match dropped.
    hits = [
        _Hit("xchange-ui/src/app/members/add-members.component.html", "Name has already been used"),
        _Hit("xchange-ui/src/app/members/add-members.component.ts", "Name has already been used"),
        _Hit("area-service/src/main/java/com/x/Msg.java", "Name has already been used"),
    ]
    loop = object.__new__(EvidenceCollectionLoop)
    loop._workspace = None
    loop.repo_search = _DirectTextRepo(hits)
    items = loop._trace_i18n_chain("Name has already been used", set())
    paths = {i.file_path for i in items}
    assert any("add-members.component.html" in p for p in paths)
    assert not any("area-service" in p for p in paths), "cross-root outlier must be dropped"
    # Provenance preserved: direct-text is a CANDIDATE, not relationship-grounded.
    for it in items:
        assert it.strength == "medium" and "direct_text_candidate" in it.details


def test_direct_text_same_root_localization_preserved():
    hits = [
        _Hit("xchange-ui/src/app/members/add-members.component.html", "Some unique message here"),
        _Hit("xchange-ui/src/app/members/add-members.component.ts", "Some unique message here"),
    ]
    loop = object.__new__(EvidenceCollectionLoop)
    loop._workspace = None
    loop.repo_search = _DirectTextRepo(hits)
    items = loop._trace_i18n_chain("Some unique message here", set())
    paths = {i.file_path for i in items}
    assert any("add-members.component.html" in p for p in paths)
    assert any("add-members.component.ts" in p for p in paths)


def test_direct_text_multi_root_tie_preserves_cross_boundary():
    # Evenly split roots (no strict majority) → keep all (legitimate cross-boundary).
    hits = [
        _Hit("xchange-ui/src/app/x.ts", "Shared cross boundary label"),
        _Hit("project-service/src/main/Y.java", "Shared cross boundary label"),
    ]
    loop = object.__new__(EvidenceCollectionLoop)
    loop._workspace = None
    loop.repo_search = _DirectTextRepo(hits)
    items = loop._trace_i18n_chain("Shared cross boundary label", set())
    roots = {p.split("/")[0] for p in {i.file_path for i in items}}
    assert "xchange-ui" in roots and "project-service" in roots, "tie preserves cross-boundary"


def test_stage05_no_arbitrary_top3_dependency():
    # Regression guard: Stage 0.5 must not reintroduce a fixed top-3 slice.
    # Breadth is relevance/information-gain driven (bounded only by a safety cap).
    import ticket_to_code.agents.evidence_collection_loop as _ecl
    src = Path(_ecl.__file__).read_text(encoding="utf-8")
    assert "all_evidence[:3]" not in src, "arbitrary top-3 Stage-0 limit must be gone"
    # The runaway cap is a generous safety net derived from the agentic budget.
    assert "_BOOTSTRAP_INSPECT_CAP" in src


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
