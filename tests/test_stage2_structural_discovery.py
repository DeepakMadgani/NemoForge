"""
Unit and behavioral test suite for Stage 2 Structural Discovery Enhancements.

Verifies:
1. ast_companion_bundle:
   - Resolves from a canonical component file path or symbol.
   - Returns existing companions on disk ONLY.
   - Distinguishes existing vs missing companions in provenance.
   - Preserves canonical repository-relative paths.
   - Surfaces ambiguity and stops if a symbol matches multiple component families.
   - Discovery only (no implication of write authorization).
2. symbol_lookup:
   - Exact match first (strong confidence, no fallback when exact match succeeds).
   - Prefix/substring fallback ONLY after zero exact results.
   - If multiple candidates match on fallback, surfaces ambiguity with weak strength.
3. ts_chain:
   - Accepts either a file path or a symbol/class name.
   - Resolves symbol to unique canonical path before traversing.
   - Ambiguous seed resolution stops rather than guessing.
4. java_chain:
   - Accepts either a file path or a symbol/class name.
   - Resolves symbol to unique canonical path before traversing.
   - Ambiguous seed resolution stops rather than guessing.

All tests invoke actual tool outputs via loop._execute_tool(...).
"""

import os
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import Mock

from ticket_to_code.agents.evidence_collection_loop import EvidenceCollectionLoop, SearchContext
from ticket_to_code.models import EvidenceItem


class _MockLocalizer:
    def __init__(self, db_conn):
        self.sqlite_store = Mock()
        self.sqlite_store._conn = db_conn
        self.neo4j_store = None

    def _graph_neighbors(self, seeds, depth=1):
        # Default mock: return connected neighbors if defined
        neighbors = {}
        for s in seeds:
            s_low = s.lower()
            if "add-members.component.ts" in s_low:
                neighbors["xchange-ui/src/app/modules/members/add-members/add-members.component.html"] = 0.95
            elif "projectservice.java" in s_low:
                neighbors["backend/src/main/java/com/service/ProjectRepository.java"] = 0.90
        return neighbors


def _create_sqlite_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE symbols (
            path TEXT,
            name TEXT,
            kind TEXT,
            spring_stereotype TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE edges (
            kind TEXT,
            path TEXT,
            dst_name TEXT,
            dst_id TEXT
        );
    """)
    return conn


def _make_stage2_loop(workspace_path: Path, conn: sqlite3.Connection) -> EvidenceCollectionLoop:
    loop = object.__new__(EvidenceCollectionLoop)
    loop._workspace = workspace_path
    loop.localizer = _MockLocalizer(conn)
    loop.repo_search = None
    loop.rag_engine = None
    loop.relationship_provider = None
    loop._current_followable_relationships = {}
    loop._semantic_verifier = None
    loop._get_dynamic_threshold = lambda term: 100
    loop._read_snippet = lambda p, *args, **kwargs: f"content of {p}"
    return loop


# ── 1. ast_companion_bundle Tests ───────────────────────────────────────────

def test_ast_companion_bundle_from_component_path(tmp_path: Path):
    """ast_companion_bundle from a component path returns existing companions only and records missing ones."""
    ui_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "members" / "add-members"
    ui_dir.mkdir(parents=True)

    # Create .ts, .html, .scss (intentionally leave .spec.ts missing)
    (ui_dir / "add-members.component.ts").write_text("// ts", encoding="utf-8")
    (ui_dir / "add-members.component.html").write_text("<!-- html -->", encoding="utf-8")
    (ui_dir / "add-members.component.scss").write_text("/* scss */", encoding="utf-8")

    conn = _create_sqlite_db()
    loop = _make_stage2_loop(tmp_path, conn)
    context = SearchContext()

    query = "xchange-ui/src/app/modules/members/add-members/add-members.component.html"
    items = loop._execute_tool("ast_companion_bundle", query, context)

    # 3 existing companions returned
    paths = [item.file_path for item in items]
    assert len(items) == 3
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.ts" in paths
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.html" in paths
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.scss" in paths

    # Missing .spec.ts must NOT be returned as an existing file
    assert not any("add-members.component.spec.ts" in p for p in paths)

    # Provenance distinguishing existing vs missing
    assert any(".component.spec.ts" in item.content_snippet for item in items)
    assert items[0].provider == "structural_companion"
    assert items[0].details == "agentic_ast_companion_bundle"


def test_ast_companion_bundle_from_symbol(tmp_path: Path):
    """ast_companion_bundle resolves a component class symbol to its canonical bundle."""
    ui_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "members" / "add-members"
    ui_dir.mkdir(parents=True)
    (ui_dir / "add-members.component.ts").write_text("// ts", encoding="utf-8")
    (ui_dir / "add-members.component.html").write_text("<!-- html -->", encoding="utf-8")

    conn = _create_sqlite_db()
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("xchange-ui/src/app/modules/members/add-members/add-members.component.ts", "AddMembersComponent", "class", "")
    )

    loop = _make_stage2_loop(tmp_path, conn)
    context = SearchContext()

    items = loop._execute_tool("ast_companion_bundle", "AddMembersComponent", context)

    paths = [item.file_path for item in items]
    assert len(items) == 2
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.ts" in paths
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.html" in paths


def test_ast_companion_bundle_ambiguous_symbol_surfaced(tmp_path: Path):
    """When a component symbol matches multiple distinct component families, surface ambiguity and do not guess."""
    conn = _create_sqlite_db()
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("v1/add-members.component.ts", "AddMembersComponent", "class", "")
    )
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("v2/add-members.component.ts", "AddMembersComponent", "class", "")
    )

    loop = _make_stage2_loop(tmp_path, conn)
    context = SearchContext()

    items = loop._execute_tool("ast_companion_bundle", "AddMembersComponent", context)

    assert len(items) == 1
    assert items[0].details == "agentic_ast_companion_ambiguous"
    assert items[0].strength == "weak"
    assert "[ast_companion_bundle:AMBIGUOUS]" in items[0].content_snippet
    assert items[0].file_path == ""


# ── 2. symbol_lookup Tests ──────────────────────────────────────────────────

def test_symbol_lookup_exact_first(tmp_path: Path):
    """Exact symbol match must take precedence with strong confidence, without prefix fallback."""
    conn = _create_sqlite_db()
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("src/app/member.service.ts", "MemberService", "class", "")
    )
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("src/app/member.service.mock.ts", "MemberServiceMock", "class", "")
    )

    loop = _make_stage2_loop(tmp_path, conn)
    context = SearchContext()

    items = loop._execute_tool("symbol_lookup", "MemberService", context)

    # Only the exact match is returned, not the mock
    assert len(items) == 1
    assert items[0].file_path == "src/app/member.service.ts"
    assert items[0].strength == "strong"
    assert items[0].details == "sqlite_symbol_exact"
    assert items[0].relevance_score == 0.95


def test_symbol_lookup_prefix_fallback_and_ambiguity(tmp_path: Path):
    """When zero exact matches exist, prefix fallback runs and surfaces ambiguity if multiple candidates match."""
    conn = _create_sqlite_db()
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("src/app/contract-a.ts", "ContractServiceAlpha", "class", "")
    )
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("src/app/contract-b.ts", "ContractServiceBeta", "class", "")
    )

    loop = _make_stage2_loop(tmp_path, conn)
    context = SearchContext()

    # "ContractService" does not exist exactly
    items = loop._execute_tool("symbol_lookup", "ContractService", context)

    assert len(items) == 2
    for it in items:
        assert it.strength == "weak"
        assert it.details == "sqlite_symbol_fallback_ambiguous"
        assert "(AMBIGUOUS)" in it.content_snippet
        assert it.relevance_score == 0.60


# ── 3. ts_chain Symbol Resolution Tests ──────────────────────────────────────

def test_ts_chain_symbol_seed_resolution(tmp_path: Path):
    """ts_chain resolves a symbol seed to its canonical path and traverses structural edges."""
    ui_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "members" / "add-members"
    ui_dir.mkdir(parents=True)
    comp_ts = ui_dir / "add-members.component.ts"
    comp_html = ui_dir / "add-members.component.html"
    comp_ts.write_text("// ts", encoding="utf-8")
    comp_html.write_text("<!-- html -->", encoding="utf-8")

    conn = _create_sqlite_db()
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("xchange-ui/src/app/modules/members/add-members/add-members.component.ts", "AddMembersComponent", "class", "")
    )

    loop = _make_stage2_loop(tmp_path, conn)
    context = SearchContext()

    # Query is a symbol name, not a path
    items = loop._execute_tool("ts_chain", "AddMembersComponent", context)

    assert len(items) >= 1
    assert any("add-members.component.html" in item.file_path for item in items)


def test_ts_chain_ambiguous_seed_stops_traversal(tmp_path: Path):
    """ts_chain must stop and return an error EvidenceItem when the seed symbol is ambiguous."""
    conn = _create_sqlite_db()
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("v1/add-members.component.ts", "AddMembersComponent", "class", "")
    )
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("v2/add-members.component.ts", "AddMembersComponent", "class", "")
    )

    loop = _make_stage2_loop(tmp_path, conn)
    context = SearchContext()

    items = loop._execute_tool("ts_chain", "AddMembersComponent", context)

    assert len(items) == 1
    assert items[0].details == "ts_chain_seed_ambiguous"
    assert items[0].strength == "weak"
    assert "[ts_chain:AMBIGUOUS_SEED]" in items[0].content_snippet


# ── 4. java_chain Symbol Resolution Tests ────────────────────────────────────

def test_java_chain_symbol_seed_resolution(tmp_path: Path):
    """java_chain resolves a symbol seed to its canonical path and traverses Spring layers."""
    conn = _create_sqlite_db()
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("backend/src/main/java/com/service/ProjectService.java", "ProjectService", "class", "Service")
    )
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("backend/src/main/java/com/service/ProjectRepository.java", "ProjectRepository", "class", "Repository")
    )

    loop = _make_stage2_loop(tmp_path, conn)
    context = SearchContext()

    items = loop._execute_tool("java_chain", "ProjectService", context)

    assert len(items) == 1
    assert items[0].file_path == "backend/src/main/java/com/service/ProjectRepository.java"
    assert items[0].details == "java_chain"


def test_java_chain_ambiguous_seed_stops_traversal(tmp_path: Path):
    """java_chain must stop and return an error EvidenceItem when the seed symbol is ambiguous."""
    conn = _create_sqlite_db()
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("v1/com/service/ProjectService.java", "ProjectService", "class", "Service")
    )
    conn.execute(
        "INSERT INTO symbols (path, name, kind, spring_stereotype) VALUES (?, ?, ?, ?)",
        ("v2/com/service/ProjectService.java", "ProjectService", "class", "Service")
    )

    loop = _make_stage2_loop(tmp_path, conn)
    context = SearchContext()

    items = loop._execute_tool("java_chain", "ProjectService", context)

    assert len(items) == 1
    assert items[0].details == "java_chain_seed_ambiguous"
    assert items[0].strength == "weak"
    assert "[java_chain:AMBIGUOUS_SEED]" in items[0].content_snippet
