"""Regression tests for the 10/10 evidence-collection hardening pass.

Covers:
  • RepositorySearchEngine: fuzzy filename tier, Python file support,
    regex catastrophic-backtracking deadline
  • SemanticVerificationAgent: three-way MODIFY/REFERENCE/EXCLUDE mapping,
    mechanical citation verification, audit reconciliation
  • EvidenceCollectionLoop: new git_history / call_sites tools registered,
    unified cache invalidation, compound-anchor primary proof
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from ticket_to_code.agents.evidence_collection_loop import (
    EvidenceCollectionLoop,
    TOOL_CATALOG,
)
from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
from ticket_to_code.agents.semantic_verification_agent import (
    SemanticVerificationAgent,
    reconcile_verifier_audit,
)


# ── RepositorySearchEngine ───────────────────────────────────────────────────

def _make_ws(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "user_acount_service.py").write_text(          # deliberate typo
        "def process():\n    return 'hello world'\n", encoding="utf-8"
    )
    (src / "member.component.ts").write_text(
        "export class MemberComponent {}\n", encoding="utf-8"
    )
    return tmp_path


def test_fuzzy_filename_tier_recovers_typos(tmp_path):
    eng = RepositorySearchEngine(_make_ws(tmp_path))
    hits = eng.search_filename("user_account_service")
    assert any(h.file_path.endswith("user_acount_service.py") for h in hits)
    assert hits[0].search_mode == "fuzzy_typo"
    assert hits[0].is_ambiguous is False


def test_python_files_are_searchable(tmp_path):
    eng = RepositorySearchEngine(_make_ws(tmp_path))
    hits = eng.search_literal("hello world")
    assert hits and hits[0].file_path.endswith(".py")


def test_regex_search_cannot_hang_on_backtracking(tmp_path):
    eng = RepositorySearchEngine(_make_ws(tmp_path))
    t0 = time.monotonic()
    # Classic catastrophic-backtracking bait: would burn CPU for minutes on
    # the pure-python path without the per-file timeout + total deadline.
    eng.search_regex(r"(a+)+$", extensions={".ts"})
    assert time.monotonic() - t0 < 30


# ── SemanticVerificationAgent ────────────────────────────────────────────────

SNIPPET = (
    "[FILE SIGNATURE lines 1-30]\nexport class Foo {\n  validate() {}\n}\n"
    "[TARGET SYMBOL: validate lines 2-4]\nvalidate() {}"
)


def test_citation_verification_is_mechanical():
    verified = SemanticVerificationAgent._verify_citations(
        SNIPPET, ["L3", "validate", "nonexistentSymbol", "line 25"]
    )
    assert verified == 3          # bad symbol rejected, in-range lines accepted
    assert SemanticVerificationAgent._verify_citations(SNIPPET, []) == 0


def test_three_way_category_mapping():
    m = SemanticVerificationAgent
    # MODIFY and REFERENCE are both retained downstream…
    assert m._decision_for_category("modify") == "include"
    assert m._decision_for_category("reference") == "include"
    # …but reference preserves 85% of the score instead of being demoted
    assert m._score_for_category("reference", 1.0) == 0.85
    assert m._score_for_category("exclude", 1.0) == 0.3
    assert m._normalize_category("MODIFY") == "modify"
    assert m._normalize_category("nonsense") == "exclude"


def test_reconcile_audit_flags_misjudgments(tmp_path):
    aud = tmp_path / ".aviator" / "verifier_audit.json"
    aud.parent.mkdir(parents=True, exist_ok=True)
    aud.write_text(json.dumps([
        {"ticket_id": "T1", "file_path": "src/a.ts", "category": "modify",
         "verification_status": "completed"},
        {"ticket_id": "T1", "file_path": "src/b.ts", "category": "exclude",
         "verification_status": "completed"},
        # fail-closed entries must NOT count as verifier false-includes
        {"ticket_id": "T1", "file_path": "src/c.ts", "category": "modify",
         "verification_status": "not_completed_fail_closed"},
    ]), encoding="utf-8")

    report = reconcile_verifier_audit(str(tmp_path), ["src/a.ts", "src/b.ts"], ticket_id="T1")

    assert report["correct_modify"] == 1
    # b.ts was excluded but had to be modified → false_exclude (worst class)
    assert len(report["false_exclude"]) == 1
    assert report["false_exclude"][0]["file"] == "src/b.ts"
    # c.ts was fail-closed, not an LLM judgment → not counted as false_include
    assert len(report["false_include"]) == 0
    assert (tmp_path / ".aviator" / "verifier_audit_report.json").exists()


# ── EvidenceCollectionLoop ───────────────────────────────────────────────────

def test_new_tools_registered_in_catalog_and_loop():
    assert hasattr(EvidenceCollectionLoop, "_query_git_history")
    assert hasattr(EvidenceCollectionLoop, "_query_call_sites")
    assert hasattr(EvidenceCollectionLoop, "invalidate_caches")
    for name in ("git_history", "call_sites"):
        assert name in TOOL_CATALOG


def test_invalidate_caches_clears_repo_search_index(tmp_path):
    loop = EvidenceCollectionLoop.__new__(EvidenceCollectionLoop)
    eng = RepositorySearchEngine(_make_ws(tmp_path))
    eng._ensure_file_index()
    assert eng._file_index_built
    loop.repo_search = eng
    loop.localizer = None
    loop.invalidate_caches()
    assert eng._file_index_built is False


def test_primary_anchor_compound_positive(tmp_path):
    loop = EvidenceCollectionLoop.__new__(EvidenceCollectionLoop)
    ticket = MagicMock()
    ticket.title = "Add Members Modal enhancements"
    ticket.description = "Add Members Modal"
    item = MagicMock()
    item.file_path = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    item.content_snippet = ""
    proven, reason = loop._is_primary_anchor_proven(ticket, [item])
    assert proven, reason


def test_primary_anchor_compound_negative_generic_word_not_enough(tmp_path):
    """'members' matching a module DIRECTORY must not prove an 'Add Members' ticket."""
    loop = EvidenceCollectionLoop.__new__(EvidenceCollectionLoop)
    ticket = MagicMock()
    ticket.title = "Add Members Modal enhancements"
    ticket.description = "Add Members Modal"
    item = MagicMock()
    item.file_path = "xchange-ui/src/app/modules/members/edit-member/edit-member.component.ts"
    item.content_snippet = ""
    proven, reason = loop._is_primary_anchor_proven(ticket, [item])
    assert proven is False
    assert "Primary feature anchor 'Add Members' is not present" in reason


# ── 10/10 final pass: per-tool last-mile enhancements ────────────────────────

import sqlite3
from types import SimpleNamespace

import pytest


def _loop_stub(**attrs):
    loop = EvidenceCollectionLoop.__new__(EvidenceCollectionLoop)
    for k, v in attrs.items():
        setattr(loop, k, v)
    return loop


def _sqlite_conn(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE symbols (id INTEGER PRIMARY KEY, path TEXT, name TEXT, "
        "kind TEXT, spring_stereotype TEXT)"
    )
    conn.executemany(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?,?,?,?)",
        rows,
    )
    return conn


class _SQLiteStore:
    def __init__(self, conn):
        self._conn = conn


class _Localizer:
    def __init__(self, conn=None, graph_neighbors=None):
        self.sqlite_store = _SQLiteStore(conn) if conn is not None else None
        self.neo4j_store = None
        if graph_neighbors is not None:
            self._graph_neighbors = graph_neighbors


def test_filename_boundary_beats_dir_segment(tmp_path):
    """Query 'member' must match member-list.component.ts, never /memberships/."""
    src = tmp_path / "src"
    (src / "memberships").mkdir(parents=True)
    (src / "members").mkdir()
    (src / "memberships" / "list.component.ts").write_text("export class A {}\n", encoding="utf-8")
    (src / "members" / "member-list.component.ts").write_text("export class B {}\n", encoding="utf-8")
    eng = RepositorySearchEngine(tmp_path)
    hits = eng.search_filename("member")
    paths = [h.file_path for h in hits]
    assert any(p.endswith("member-list.component.ts") for p in paths)
    assert not any("memberships" in p for p in paths), paths


def test_rag_scores_calibrated_onto_evidence_scale():
    loop = _loop_stub()
    assert loop._calibrate_rag_score(1.0) == 0.78     # best semantic hit < exact matches
    assert loop._calibrate_rag_score(85.0) == 0.716   # 0-100 backend scale normalized
    assert loop._calibrate_rag_score(100.0) == 0.78   # top of the 0-100 scale
    assert loop._calibrate_rag_score(0.5) == 0.565
    assert loop._calibrate_rag_score("garbage") == 0.50
    assert 0.35 <= loop._calibrate_rag_score(0.0) <= 0.78


def test_symbol_exact_orders_type_definitions_first():
    conn = _sqlite_conn([
        ("src/a.py", "Foo", "method", ""),
        ("src/b.py", "Foo", "class", ""),
    ])
    loop = _loop_stub(localizer=_Localizer(conn=conn))
    items = loop._query_sqlite_symbols(["Foo"], "hyp", set())
    assert items, "no symbol items returned"
    assert items[0].file_path == "src/b.py"          # class outranks method
    assert items[0].relevance_score == 0.95
    assert items[1].file_path == "src/a.py"
    assert items[1].relevance_score == 0.88


def test_ts_chain_follows_second_hop():
    calls = {"n": 0}

    def graph_neighbors(paths, depth=1):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"src/hero.component.html": 0.9, "src/child.component.ts": 0.8}
        return {"src/lazy.module.ts": 0.6}

    loop = _loop_stub(localizer=_Localizer(graph_neighbors=graph_neighbors))
    loop._resolve_symbol_seed_to_path = lambda s, valid_extensions: (s, False, [])
    loop._read_snippet = lambda p, max_chars=300: ""
    items = loop._follow_ts_chains(["src/hero.component.ts"], set())
    deep = [i for i in items if i.details == "ts_chain_depth2"]
    assert any(i.file_path == "src/lazy.module.ts" for i in deep)
    assert deep[0].relevance_score == pytest.approx(0.6 * 0.85)  # decayed below hop-1


def test_java_import_pins_package(tmp_path):
    conn = _sqlite_conn([
        ("other/pkg/MemberDto.java", "MemberDto", "class", ""),
        ("com/x/MemberDto.java", "MemberDto", "class", ""),
    ])
    loop = _loop_stub(localizer=_Localizer(conn=conn), _workspace=tmp_path)
    loop._resolve_symbol_seed_to_path = lambda s, valid_extensions: (s, False, [])
    loop.localizer._graph_neighbors = lambda paths, depth=1: {}
    seed_dir = tmp_path / "com" / "x"
    seed_dir.mkdir(parents=True)
    (seed_dir / "MemberService.java").write_text(
        "import com.x.MemberDto;\npublic class MemberService {\n"
        "    public void save(MemberDto dto) {}\n}\n",
        encoding="utf-8",
    )
    items = loop._follow_java_chains(["com/x/MemberService.java"], set())
    dto_items = [i for i in items if i.symbol_name == "MemberDto"]
    assert [i.file_path for i in dto_items] == ["com/x/MemberDto.java"]


def test_java_ambiguous_bare_name_not_conflated(tmp_path):
    conn = _sqlite_conn([
        ("pkgA/MemberDto.java", "MemberDto", "class", ""),
        ("pkgB/MemberDto.java", "MemberDto", "class", ""),
    ])
    loop = _loop_stub(localizer=_Localizer(conn=conn), _workspace=tmp_path)
    loop._resolve_symbol_seed_to_path = lambda s, valid_extensions: (s, False, [])
    loop.localizer._graph_neighbors = lambda paths, depth=1: {}
    svc = tmp_path / "svc"
    svc.mkdir(parents=True)
    (svc / "MemberService.java").write_text(
        "public class MemberService {\n    public void save(MemberDto dto) {}\n}\n",
        encoding="utf-8",
    )
    items = loop._follow_java_chains(["svc/MemberService.java"], set())
    assert not [i for i in items if i.symbol_name == "MemberDto"], (
        "same-named DTOs across packages must be skipped, not conflated"
    )


def test_python_chain_ast_resolves_quoted_forward_refs(tmp_path):
    """Regex extraction cannot see quoted forward refs; the AST pass must."""
    conn = _sqlite_conn([("src/types.py", "MemberDto", "class", "")])
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "types.py").write_text("class MemberDto:\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "service.py").write_text(
        "def save(items: list['MemberDto']) -> 'MemberDto':\n    return items\n",
        encoding="utf-8",
    )
    loop = _loop_stub(localizer=_Localizer(conn=conn), _workspace=tmp_path)
    items = loop._follow_python_chains(["src/service.py"], set())
    assert {i.file_path for i in items} == {"src/types.py"}


def test_layout_chain_finds_conventional_shared_library(tmp_path):
    (tmp_path / "src" / "app" / "pages" / "home").mkdir(parents=True)
    (tmp_path / "src" / "app" / "shared").mkdir(parents=True)
    (tmp_path / "src" / "app" / "shared" / "lib.scss").write_text(
        ".panel {\n  display: flex;\n  gap: 8px;\n}\n", encoding="utf-8"
    )
    (tmp_path / "src" / "app" / "pages" / "home" / "home.component.scss").write_text(
        ".panel {\n  color: red;\n}\n", encoding="utf-8"
    )
    loop = _loop_stub(_workspace=tmp_path)
    items = loop._follow_layout_chains(
        ["src/app/pages/home/home.component.scss"], set()
    )
    assert any(i.file_path.endswith("src/app/shared/lib.scss") for i in items), (
        [i.file_path for i in items]
    )


def test_i18n_dotted_key_path_reconstructed_and_preferred(tmp_path):
    i18n_dir = tmp_path / "src" / "app" / "i18n"
    i18n_dir.mkdir(parents=True)
    (i18n_dir / "en.json").write_text(
        json.dumps({"validation": {"nameUsed": "Name has already been used"}}),
        encoding="utf-8",
    )

    class _Hit:
        def __init__(self, fp, text):
            self.file_path, self.matched_text, self.line_number = fp, text, 1

    queries = []

    def search_literal(q, extensions=None, max_results=10):
        queries.append(q)
        if ".json" in (extensions or set()):
            return [_Hit("src/app/i18n/en.json",
                         '"nameUsed": "Name has already been used"')]
        return [_Hit("src/app/comp.ts", "validation.nameUsed")]

    loop = _loop_stub(
        repo_search=SimpleNamespace(search_literal=search_literal),
        _workspace=tmp_path,
    )
    loop._read_snippet = lambda p, max_chars=300: ""
    items = loop._trace_i18n_chain("Name has already been used", set())
    assert "validation.nameUsed" in queries, queries
    # The reconstructed dotted path is searched BEFORE the bare leaf key
    assert queries.index("validation.nameUsed") < queries.index("nameUsed")
    assert any(i.file_path == "src/app/comp.ts" for i in items)


# ── Tool #18 (LSP) + tree-sitter AST upgrades for TS/Java chains ─────────────

def test_ts_chain_ast_resolves_imported_types(tmp_path):
    """Import specifiers + annotations from a real TS parse resolve to the
    defining file via the SQLite symbol index (regex can't do this)."""
    conn = _sqlite_conn([("src/member.service.ts", "MemberService", "class", "")])
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "member.service.ts").write_text(
        "export class MemberService {\n  load(): void {}\n}\n", encoding="utf-8"
    )
    (tmp_path / "src" / "member.component.ts").write_text(
        "import { MemberService } from './member.service';\n"
        "export class MemberComponent {\n"
        "  constructor(private svc: MemberService) {}\n"
        "}\n",
        encoding="utf-8",
    )
    loop = _loop_stub(
        localizer=_Localizer(conn=conn, graph_neighbors=lambda p, depth=1: {}),
        _workspace=tmp_path,
    )
    loop._resolve_symbol_seed_to_path = lambda s, valid_extensions: (s, False, [])
    items = loop._follow_ts_chains(["src/member.component.ts"], set())
    ast_items = [i for i in items if i.details == "ts_chain_ast"]
    assert any(
        i.file_path == "src/member.service.ts" and i.symbol_name == "MemberService"
        for i in ast_items
    ), [(i.file_path, i.symbol_name) for i in ast_items]


def test_java_tree_sitter_captures_field_types_regex_misses(tmp_path):
    """A private field declaration is invisible to the method-param regex —
    the tree-sitter pass must still resolve it."""
    conn = _sqlite_conn([("com/x/MemberDto.java", "MemberDto", "class", "")])
    (tmp_path / "com" / "x").mkdir(parents=True)
    (tmp_path / "com" / "x" / "MemberDto.java").write_text(
        "public class MemberDto {}\n", encoding="utf-8"
    )
    (tmp_path / "com" / "x" / "MemberService.java").write_text(
        "public class MemberService {\n"
        "    private MemberDto lastDto;\n"
        "}\n",
        encoding="utf-8",
    )
    loop = _loop_stub(
        localizer=_Localizer(conn=conn, graph_neighbors=lambda p, depth=1: {}),
        _workspace=tmp_path,
    )
    loop._resolve_symbol_seed_to_path = lambda s, valid_extensions: (s, False, [])
    items = loop._follow_java_chains(["com/x/MemberService.java"], set())
    assert any(
        i.symbol_name == "MemberDto" and i.file_path == "com/x/MemberDto.java"
        for i in items
    ), [(i.symbol_name, i.file_path) for i in items]


def test_lsp_tool_registered_and_finds_definition_and_references(tmp_path):
    """Tool #18: definition via lsp_client extractors + cross-file references."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "member.service.ts").write_text(
        "export class MemberService {\n"
        "  private cache: Map<string, string> = new Map();\n"
        "  load(id: string): void {\n"
        "    this.cache.set(id, id);\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "member.component.ts").write_text(
        "import { MemberService } from './member.service';\n"
        "export class MemberComponent {\n"
        "  constructor(private svc: MemberService) {\n"
        "    this.svc.load('1');\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    eng = RepositorySearchEngine(tmp_path)
    conn = _sqlite_conn([("src/member.service.ts", "MemberService", "class", "")])
    loop = _loop_stub(localizer=_Localizer(conn=conn), repo_search=eng, _workspace=tmp_path)

    assert "lsp" in TOOL_CATALOG
    assert hasattr(EvidenceCollectionLoop, "_query_lsp")

    items = loop._query_lsp("MemberService", set())
    defs = [i for i in items if i.details == "lsp_definition"]
    refs = [i for i in items if i.details == "lsp_reference"]
    assert defs, "no lsp_definition item"
    assert defs[0].file_path == "src/member.service.ts"
    assert defs[0].symbol_name == "MemberService"
    assert any(r.file_path == "src/member.component.ts" for r in refs), refs

    # File mode: query names a file directly → members block for that file
    file_items = loop._query_lsp("src/member.service.ts", set())
    assert file_items and file_items[0].details == "lsp_definition"


# ── Search evolution: cooccurrence, classification, git 3-pass, fuzzy, docs ──

import subprocess


def test_cooccurrence_and_narrows_candidates(tmp_path):
    """Two terms AND together must eliminate files matching only one."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "member.service.ts").write_text(
        "export class MemberService {\n"
        "  org = 'organization';\n"
        "  validate() {}\n"
        "}\n",
        encoding="utf-8",
    )
    (src / "other.service.ts").write_text(
        "export class OtherService {\n  org = 'organization';\n}\n", encoding="utf-8"
    )
    (src / "solo.ts").write_text(
        "export class Solo { name = 'MemberService'; }\n", encoding="utf-8"
    )
    eng = RepositorySearchEngine(tmp_path)
    both = eng.search_cooccurrence(["MemberService", "organization"])
    assert [h.file_path for h in both] == ["src/member.service.ts"], both
    assert both[0].search_mode == "cooccurrence"
    assert both[0].classification == "code"
    # Fewer than 2 terms is not a co-occurrence query
    assert eng.search_cooccurrence(["MemberService"]) == []


def test_hit_classification_tags_comment_string_test_doc(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "notes.ts").write_text("// uses paymentGateway internally\n", encoding="utf-8")
    (src / "cfg.ts").write_text('export const tag = "paymentGateway";\n', encoding="utf-8")
    (src / "widget.spec.ts").write_text("export const paymentGateway = 'x';\n", encoding="utf-8")
    (src / "README.md").write_text("# paymentGateway\n\nSee docs.\n", encoding="utf-8")
    (src / "boot.ts").write_text(
        "import { init } from './core';\n"
        "export function boot() { init('paymentGateway'); }\n",
        encoding="utf-8",
    )
    eng = RepositorySearchEngine(tmp_path)
    hits = eng.search_literal("paymentGateway")
    by_file = {}
    for h in hits:
        by_file[h.file_path] = h
    assert by_file["src/notes.ts"].classification == "comment"
    assert by_file["src/cfg.ts"].classification == "string"
    assert by_file["src/widget.spec.ts"].classification == "test_file"
    assert by_file["src/README.md"].classification == "doc"
    assert by_file["src/boot.ts"].classification == "code"


def test_ripgrep_branch_tags_and_demotes_classification(tmp_path):
    """The loop's consumer must tag the snippet and demote non-code hits."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "boot.ts").write_text("init('paymentGateway');\n", encoding="utf-8")
    (src / "README.md").write_text("# paymentGateway\n", encoding="utf-8")
    eng = RepositorySearchEngine(tmp_path)
    loop = _loop_stub(repo_search=eng, _workspace=tmp_path)
    items = loop._execute_tool("ripgrep", "paymentGateway", SimpleNamespace(visited_files=set()))
    doc_items = [i for i in items if i.file_path.endswith("README.md")]
    code_items = [i for i in items if i.file_path.endswith("boot.ts")]
    assert doc_items and "[doc]" in doc_items[0].content_snippet
    assert doc_items[0].relevance_score < 0.55          # 0.55 * 0.65
    assert code_items and "[code]" in code_items[0].content_snippet
    assert code_items[0].relevance_score > doc_items[0].relevance_score
    assert code_items[0].strength == "strong" and doc_items[0].strength == "medium"


def test_cooccurrence_tool_dispatch(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "svc.ts").write_text(
        "export class MemberService { org = 'organization'; }\n", encoding="utf-8"
    )
    (tmp_path / "src" / "other.ts").write_text("export class X { org = 'organization'; }\n", encoding="utf-8")
    eng = RepositorySearchEngine(tmp_path)
    loop = _loop_stub(repo_search=eng, _workspace=tmp_path)
    items = loop._execute_tool(
        "cooccurrence", "MemberService organization", SimpleNamespace(visited_files=set())
    )
    assert [i.file_path for i in items] == ["src/svc.ts"] or all(
        i.file_path.endswith("svc.ts") for i in items
    ), items
    # Single term degrades to plain ripgrep instead of failing
    single = loop._execute_tool("cooccurrence", "MemberService", SimpleNamespace(visited_files=set()))
    assert single  # ripgrep fallback returned the svc.ts hit


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def test_git_history_commit_message_and_cochange(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "src" / "a.ts").write_text("export const A = 'alpha-feature';\n", encoding="utf-8")
    (repo / "src" / "b.ts").write_text("export const B = 1;\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "PROJ-1234 add alpha-feature")
    (repo / "src" / "a.ts").write_text("export const A = 'alpha-feature v2';\n", encoding="utf-8")
    (repo / "src" / "d.ts").write_text("export const D = 4;\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "refactor internals")
    loop = _loop_stub(_workspace=repo)

    # Pass 2: message-only hits (b.ts never contains the query string)
    msg_items = loop._query_git_history("PROJ-1234", set())
    by_file = {i.file_path: i for i in msg_items}
    assert "src/a.ts" in by_file and "src/b.ts" in by_file, by_file
    assert by_file["src/b.ts"].details.startswith("git_grep_msg")

    # Pass 3: d.ts committed together with a.ts, found via co-change only
    co = [i for i in msg_items if i.details == "git_cochange"]
    assert any(i.file_path == "src/d.ts" for i in co), co

    # Pass 1 still works: pickaxe on content
    pick = loop._query_git_history("alpha-feature", set())
    assert any(i.file_path == "src/a.ts" for i in pick)


def test_symbol_lookup_fuzzy_tier(tmp_path):
    conn = _sqlite_conn([("src/member.service.ts", "MemberService", "class", "")])
    loop = _loop_stub(localizer=_Localizer(conn=conn))
    items = loop._query_sqlite_symbols(["MembrService"], "hyp", set())
    assert items and items[0].details == "sqlite_symbol_fuzzy"
    assert items[0].file_path == "src/member.service.ts"
    assert "FUZZY" in items[0].content_snippet
    # Exact matches must never go through the fuzzy tier
    exact = loop._query_sqlite_symbols(["MemberService"], "hyp", set())
    assert exact and exact[0].details == "sqlite_symbol_exact"


# ── Pipeline 10/10: short-circuit, binding contract, acceptance matrix ──────

from types import SimpleNamespace as _NS

from ticket_to_code.agents.code_generator import (
    extract_template_bindings,
    validate_template_bindings,
)


def test_template_binding_validation_detects_hallucinated_bindings():
    controller = (
        "import { Component } from '@angular/core';\n"
        "export class AddMembersModalComponent {\n"
        "  displayedMembers: DisplayedMember[] = [];\n"
        "  isExistingProjectMember = false;\n"
        "  selectUser(u: User) {}\n"
        "  saveMembers() {}\n"
        "}\n"
    )
    good_template = (
        "<span *ngIf=\"isExistingProjectMember\">{{ displayedMembers.length }}</span>\n"
        "<button (click)=\"saveMembers()\">Save</button>\n"
    )
    bad_template = good_template + "<ot-dropdown [(value)]=\"existingOrg\">{{ wrongVar }}</ot-dropdown>\n"

    assert validate_template_bindings(good_template, controller) == []
    assert extract_template_bindings(good_template)  # extractor returns real bindings

    violations = validate_template_bindings(bad_template, controller)
    assert "existingOrg" in violations and "wrongVar" in violations, violations
    # Declared members must never be flagged
    assert "isExistingProjectMember" not in violations
    assert "displayedMembers" not in violations


def test_companion_pair_short_circuit_positive_and_negative():
    loop = _loop_stub()
    ticket = _NS(title="Add Members Modal", description="Add Members Modal")
    pair_evidence = {
        "src/app/add-members-modal.component.ts": {"relevant": True},
        "src/app/add-members-modal.component.html": {"relevant": True},
    }
    proven, why = loop._companion_pair_proven(ticket, pair_evidence)
    assert proven, why
    assert "add-members-modal" in why.lower()

    # Unrelated component pair does NOT prove the ticket's anchor
    other_evidence = {
        "src/app/edit-project.component.ts": {"relevant": True},
        "src/app/edit-project.component.html": {"relevant": True},
    }
    proven2, _why2 = loop._companion_pair_proven(ticket, other_evidence)
    assert proven2 is False

    # Single file without its template half is not a pair
    proven3, _ = loop._companion_pair_proven(
        ticket, {"src/app/add-members-modal.component.ts": {"relevant": True}}
    )
    assert proven3 is False

    # Multi-component ticket: targets both add-members and project-overview
    multi_ticket = _NS(
        title="Add Members Modal and Project Overview",
        description="Update both add-members.component.ts and project-overview.component.ts",
    )
    # Only add-members is in evidence: must NOT short-circuit
    proven4, why4 = loop._companion_pair_proven(
        multi_ticket,
        {
            "src/app/add-members.component.ts": {"relevant": True},
            "src/app/add-members.component.html": {"relevant": True},
        },
    )
    assert proven4 is False
    assert "awaiting remaining: project-overview" in why4

    # When both add-members and project-overview are in evidence: short-circuit allowed
    proven5, why5 = loop._companion_pair_proven(
        multi_ticket,
        {
            "src/app/add-members.component.ts": {"relevant": True},
            "src/app/add-members.component.html": {"relevant": True},
            "src/app/project-overview.component.ts": {"relevant": True},
            "src/app/project-overview.component.html": {"relevant": True},
        },
    )
    assert proven5 is True
    assert "all 2 companion pairs proven" in why5


def test_outcome_check_eam_fails_fast_without_llm_on_planned_gap():
    """A planned task that produced no output must fail the EAM and return
    repair_required WITHOUT spending a single LLM token (agents=None)."""
    from ticket_to_code.workflow import outcome_check_node

    ticket = _NS(title="Add Members Modal", ticket_id="T1")
    plan = _NS(tasks=[
        _NS(id="t1", file_path="src/app/never-written.component.ts",
            task_type=_NS(value="modify")),
    ])
    state = {
        "ticket": ticket,
        "requirements": None,
        "generated_code": [_NS(file_path="src/app/other.component.ts", content="export class X {}")],
        "original_file_contents": {},
        "workspace_path": "Z:/definitely/not/a/real/workspace",
        "architectural_plan": plan,
    }
    result = outcome_check_node(state, None)  # agents=None: LLM path must not run
    oc = result["outcome_check_result"]
    assert oc["status"] == "repair_required"
    assert oc["eam_passed"] is False
    assert any("never-written.component.ts" in h for h in oc["repair_hints"]), oc


def test_outcome_check_eam_passes_mechanical_floor(tmp_path):
    """With every planned file written and no templates, the EAM passes and
    the node proceeds (here: gracefully skipped on empty diff)."""
    from ticket_to_code.workflow import outcome_check_node

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "widget.component.ts").write_text("export class W {}", encoding="utf-8")
    ticket = _NS(title="T", ticket_id="T1")
    plan = _NS(tasks=[
        _NS(id="t1", file_path="src/widget.component.ts", task_type=_NS(value="modify")),
    ])
    state = {
        "ticket": ticket,
        "requirements": None,
        "generated_code": [_NS(file_path="src/widget.component.ts", content="export class W2 {}")],
        "original_file_contents": {"src/widget.component.ts": "export class W {}"},
        "workspace_path": str(tmp_path),
        "architectural_plan": plan,
    }
    result = outcome_check_node(state, None)
    oc = result["outcome_check_result"]
    # EAM passed; node proceeds past the matrix (LLM-less here -> skipped)
    assert oc.get("eam_passed") is not False


# ── Final 4 gaps: sub-workspace routing, engine scope, co-fix, guard ─────────

from ticket_to_code.workflow import detect_sub_workspace


def test_sub_workspace_routing_frontend_vs_backend(tmp_path):
    (tmp_path / "xchange-ui").mkdir()
    (tmp_path / "xchange-ui" / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "contracts-service").mkdir()
    (tmp_path / "contracts-service" / "pom.xml").write_text("<project/>", encoding="utf-8")

    fe = detect_sub_workspace(str(tmp_path), "Add Members modal dropdown to the members page")
    assert fe == "xchange-ui", fe

    be = detect_sub_workspace(
        str(tmp_path), "Add repository endpoint entity migration for member database query"
    )
    assert be == "contracts-service", be

    # Mixed full-stack ticket must NOT be scoped (ambiguity is never guessed)
    mixed = detect_sub_workspace(
        str(tmp_path), "Add modal component and repository endpoint"
    )
    assert mixed == ""


def test_engine_scope_restricts_all_walks(tmp_path):
    (tmp_path / "xchange-ui" / "src").mkdir(parents=True)
    (tmp_path / "contracts-service").mkdir(parents=True)
    (tmp_path / "xchange-ui" / "src" / "modal.component.ts").write_text(
        "export class Modal { render = 'member-org'; }\n", encoding="utf-8"
    )
    (tmp_path / "contracts-service" / "MemberRepository.java").write_text(
        "class MemberRepository { String memberOrg; }\n", encoding="utf-8"
    )
    eng = RepositorySearchEngine(tmp_path)

    # Unscoped: both hits
    assert len(eng.search_literal("member")) == 2

    # Scoped to the UI sub-project: only the modal hit
    eng.set_scope("xchange-ui")
    scoped = eng.search_literal("member")
    assert [h.file_path for h in scoped] == ["xchange-ui/src/modal.component.ts"]

    # Scope is cleared for the next ticket
    eng.set_scope(None)
    assert len(eng.search_literal("member")) == 2


def test_dynamic_viewmodel_fields_close_the_contract_blindspot():
    """The live-ticket blindspot: isExistingMember is created inside a method
    body — invisible to the top-level AST contract, legal to bind in HTML."""
    from ticket_to_code.agents.code_generator import extract_dynamic_viewmodel_fields

    companion = (
        "import { Component } from '@angular/core';\n"
        "export class AddMembersComponent {\n"
        "  displayedMembers: DisplayedMember[] = [];\n"
        "  onMemberAdd() {\n"
        "    const displayedMember: any = {\n"
        "      user: selectedMember,\n"
        "      organization: isExistingMember ? existingOrganizationName : this.selectedOrg,\n"
        "      isExistingMember: isExistingMember\n"
        "    };\n"
        "  }\n"
        "  checkProjectMembership(user: ItemSelectItem) {\n"
        "    user['isExistingMember'] = true;\n"
        "    user['existingOrganizationName'] = existingMember.company?.name;\n"
        "  }\n"
        "}\n"
    )
    fields = extract_dynamic_viewmodel_fields(companion)
    assert "isExistingMember" in fields, fields
    assert "existingOrganizationName" in fields
    assert "organization" in fields


def test_deep_binding_validation_catches_hallucinated_members():
    """Root binding is declared, but a deep member invented by the LLM
    (dmember.isProjectMember vs the real isExistingMember) must be flagged —
    this is the silent-runtime-false failure AOT cannot catch."""
    from ticket_to_code.agents.code_generator import validate_template_bindings

    controller = (
        "export class AddMembersComponent {\n"
        "  displayedMembers: any[] = [];\n"
        "  onMemberAdd() {\n"
        "    const displayedMember: any = {\n"
        "      isExistingMember: isExistingMember,\n"
        "      organization: org\n"
        "    };\n"
        "  }\n"
        "}\n"
    )
    good = ('<div *ngFor="let dmember of displayedMembers">'
            '<span *ngIf="dmember.isExistingMember">{{ dmember.organization }}</span></div>')
    bad = ('<div *ngFor="let dmember of displayedMembers">'
           '<span *ngIf="dmember.isProjectMember">{{ dmember.organizaton }}</span></div>')

    assert validate_template_bindings(good, controller) == []
    violations = validate_template_bindings(bad, controller)
    assert "isProjectMember" in violations, violations
    assert "organizaton" in violations

    # Dynamic fields passed as extra_allowed legalize the real deep member
    from ticket_to_code.agents.code_generator import extract_dynamic_viewmodel_fields
    dyn = extract_dynamic_viewmodel_fields(controller)
    assert validate_template_bindings(good, controller, extra_allowed=dyn) == []


def test_sub_workspace_immune_to_node_modules_pollution(tmp_path):
    """The TASK-C2322C17 regression: 1,495 package.json files under
    node_modules must not abort scoping — top-level dirs only, vendor dirs
    skipped, classification by manifest content not directory names."""
    for i in range(50):  # simulate the bundle swarm without 1,495 real dirs
        vendor = tmp_path / "xchange-ui" / "node_modules" / f"pkg-{i}"
        vendor.mkdir(parents=True)
        (vendor / "package.json").write_text('{"name": "pkg"}', encoding="utf-8")
    (tmp_path / "xchange-ui" / "package.json").write_text(
        '{"dependencies": {"@angular/core": "^17.0.0"}}', encoding="utf-8"
    )
    (tmp_path / "project-service").mkdir()
    (tmp_path / "project-service" / "pom.xml").write_text("<project/>", encoding="utf-8")

    fe = detect_sub_workspace(
        str(tmp_path), "Add Members modal dropdown to the members page"
    )
    assert fe == "xchange-ui", fe
    be = detect_sub_workspace(str(tmp_path), "Add repository endpoint entity migration")
    assert be == "project-service", be


def test_sub_workspace_manifest_classification_beats_name_guessing(tmp_path):
    """A UI sub-project with NO ui-ish name ('portal') must still classify as
    frontend via its @angular/core dependency — never by folder-name hints."""
    (tmp_path / "portal").mkdir()
    (tmp_path / "portal" / "package.json").write_text(
        '{"dependencies": {"@angular/core": "^17.0.0", "rxjs": "^7.0.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "member-svc").mkdir()
    (tmp_path / "member-svc" / "package.json").write_text(
        '{"dependencies": {"express": "^4.0.0"}}', encoding="utf-8"
    )
    assert detect_sub_workspace(
        str(tmp_path), "Add modal dropdown component with html template styling"
    ) == "portal"


def test_sufficiency_check_coerces_structured_reasoning_points():
    """Gemini may output reasoning_points as structured dicts instead of strings.
    SufficiencyCheck must coerce them to strings without throwing Pydantic validation errors."""
    from ticket_to_code.models import SufficiencyCheck

    payload = {
        "is_sufficient": True,
        "confidence_score": 0.85,
        "evidence_score": 0.80,
        "coverage_score": 0.80,
        "reasoning_points": [
            {
                "aspect_covered": "The AddMembersComponent covers modal UI changes",
                "aspect_not_covered": "N/A",
                "recommendation": "Proceed to planning",
            },
            "Simple string point already fine",
        ],
        "what_is_known": [{"item": "add-members.component.ts found"}],
        "what_is_missing": [],
        "new_search_directions": [],
    }

    check = SufficiencyCheck.model_validate(payload)
    assert check.is_sufficient is True
    assert len(check.reasoning_points) == 2
    assert "The AddMembersComponent covers modal UI changes" in check.reasoning_points[0]
    assert check.reasoning_points[1] == "Simple string point already fine"
    assert "add-members.component.ts found" in check.what_is_known[0]


# ── Diminishing-returns deep-path bounding ───────────────────────────────────

def test_diminishing_returns_never_stops_without_relevant_evidence():
    """With zero relevant files the loop must keep searching (and ultimately
    ask_user) — stall counters alone must not end discovery empty-handed."""
    stop = EvidenceCollectionLoop._diminishing_returns_stop_reason
    assert stop(plateau_count=5, no_change_streak=5, n_relevant_files=0) is None
    assert stop(plateau_count=0, no_change_streak=0, n_relevant_files=3) is None


def test_diminishing_returns_stops_on_coverage_plateau():
    """The 24-call wander regression: once sufficiency coverage stops
    improving for 2 consecutive checks and relevant files exist, discovery
    must stop in favor of planning instead of burning the full budget."""
    stop = EvidenceCollectionLoop._diminishing_returns_stop_reason
    reason = stop(plateau_count=2, no_change_streak=0, n_relevant_files=4)
    assert reason is not None and "plateaued" in reason
    # One flat check is not enough — the controller gets its chance to hop
    assert stop(plateau_count=1, no_change_streak=0, n_relevant_files=4) is None


def test_diminishing_returns_stops_on_no_change_streak():
    """Two consecutive iterations that add no new artifacts/facts/methods/
    relationships end the loop when evidence is already in hand."""
    stop = EvidenceCollectionLoop._diminishing_returns_stop_reason
    reason = stop(plateau_count=0, no_change_streak=2, n_relevant_files=2)
    assert reason is not None and "no new information" in reason
    assert stop(plateau_count=0, no_change_streak=1, n_relevant_files=2) is None
