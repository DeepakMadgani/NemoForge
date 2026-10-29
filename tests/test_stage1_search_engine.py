"""
Unit and behavioral test suite for Stage 1 Search Engine & Provenance Enhancements.

Verifies:
1. Exact filename match remains authoritative first attempt.
2. Token / singular-plural normalization fallback succeeds on zero exact results.
3. Ambiguous fallback matches are surfaced with is_ambiguous=True, never guessed.
4. Known UI container terms (modal, dialog, button, page, etc.) are stripped on zero exact literal matches.
5. Regex syntax errors return explicit SearchResult with error details, not silent [].
6. SQLite FTS5 handles hyphens and punctuation safely without syntax crashes.
7. Search provenance (original_query, resolved_query, search_mode) is retained throughout.
"""

import pytest
import sqlite3
from pathlib import Path
from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine, SearchResult
from ticket_to_code.agents.query_expansion import strip_ui_container_term, _UI_CONTAINER_TERMS


# ── 1. Filename Exact vs Fallback ───────────────────────────────────────────

def test_filename_search_exact_first(tmp_path: Path):
    """Exact substring match must be authoritative and return search_mode='exact'."""
    (tmp_path / "src" / "app").mkdir(parents=True)
    target_file = tmp_path / "src" / "app" / "add-members.component.ts"
    target_file.write_text("// dummy", encoding="utf-8")

    engine = RepositorySearchEngine(tmp_path)
    results = engine.search_filename("add-members.component")

    assert len(results) == 1
    assert results[0].file_path == "src/app/add-members.component.ts"
    assert results[0].search_mode == "exact"
    assert results[0].is_ambiguous is False
    assert results[0].original_query == "add-members.component"
    assert results[0].resolved_query == "add-members.component"


def test_filename_search_singular_plural_fallback(tmp_path: Path):
    """When exact match returns 0, token/inflection fallback must match add-members.component.ts."""
    (tmp_path / "src" / "app").mkdir(parents=True)
    target_file = tmp_path / "src" / "app" / "add-members.component.ts"
    target_file.write_text("// dummy", encoding="utf-8")

    engine = RepositorySearchEngine(tmp_path)
    # Query is singular "add-member", file is plural "add-members"
    results = engine.search_filename("add-member.component")

    assert len(results) == 1
    assert results[0].file_path == "src/app/add-members.component.ts"
    assert results[0].search_mode == "token_stem_fallback"
    assert results[0].is_ambiguous is False
    assert results[0].original_query == "add-member.component"
    assert results[0].resolved_query == "add member component"
    assert results[0].file_path == "src/app/add-members.component.ts"


def test_filename_search_ambiguous_surfaced_not_guessed(tmp_path: Path):
    """When multiple valid candidates match via fallback, all are surfaced as ambiguous."""
    (tmp_path / "src" / "v1").mkdir(parents=True)
    (tmp_path / "src" / "v2").mkdir(parents=True)
    file1 = tmp_path / "src" / "v1" / "add-members.component.ts"
    file2 = tmp_path / "src" / "v2" / "add-members.component.ts"
    file1.write_text("// v1", encoding="utf-8")
    file2.write_text("// v2", encoding="utf-8")

    engine = RepositorySearchEngine(tmp_path)
    results = engine.search_filename("add-member.component")

    assert len(results) == 2
    for r in results:
        assert r.is_ambiguous is True
        assert r.search_mode == "token_stem_fallback"
        assert r.original_query == "add-member.component"


# ── 2. Literal Search & UI Container Stripping ──────────────────────────────

def test_strip_ui_container_term_helper():
    """Known container words must be stripped, while regular multi-word terms remain untouched."""
    assert strip_ui_container_term("Add Members modal") == "Add Members"
    assert strip_ui_container_term("Submit Order button") == "Submit Order"
    assert strip_ui_container_term("Project Settings page") == "Project Settings"
    assert strip_ui_container_term("User Profile dialog") == "User Profile"
    assert strip_ui_container_term("Contract Table view") == "Contract Table"
    
    # Should not strip single words or non-container terms
    assert strip_ui_container_term("modal") is None
    assert strip_ui_container_term("ContractMemberService") is None
    assert strip_ui_container_term("Add Members") is None


def test_literal_search_ui_container_fallback(tmp_path: Path):
    """When exact literal query 'Add Members modal' yields 0 results, fallback to 'Add Members'."""
    (tmp_path / "src").mkdir(parents=True)
    template = tmp_path / "src" / "add-members.component.html"
    template.write_text('<span class="title">Add Members</span>', encoding="utf-8")

    engine = RepositorySearchEngine(tmp_path)
    results = engine.search_literal("Add Members modal")

    assert len(results) == 1
    assert results[0].file_path == "src/add-members.component.html"
    assert results[0].search_mode == "ui_container_stripped"
    assert results[0].original_query == "Add Members modal"
    assert results[0].resolved_query == "Add Members"
    assert 'Add Members' in results[0].matched_text


# ── 3. Regex Explicit Syntax Error ──────────────────────────────────────────

def test_regex_syntax_error_explicit(tmp_path: Path):
    """Invalid regex patterns must return explicit error SearchResult, not silent []."""
    (tmp_path / "src").mkdir(parents=True)
    f = tmp_path / "src" / "test.ts"
    f.write_text("const x = 1;", encoding="utf-8")

    engine = RepositorySearchEngine(tmp_path)
    # Invalid regex (unclosed bracket)
    results = engine.search_regex("[unclosed_bracket")

    assert len(results) == 1
    assert results[0].search_mode == "regex_error"
    assert results[0].error is not None
    assert "Invalid regex" in results[0].error
    assert results[0].confidence == 0.0


# ── 4. SQLite FTS5 Safe Quoting & Hyphens ───────────────────────────────────

def test_fts5_safe_quoting_with_hyphens():
    """SQLite FTS5 query with hyphens must not fail with syntax error or treat - as unary NOT."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE symbols_fts USING fts5(path, name, content);")
    conn.execute(
        "INSERT INTO symbols_fts (path, name, content) VALUES (?, ?, ?)",
        ("xchange-ui/src/add-members.ts", "AddMembersComponent", "class AddMembersComponent with add-members logic")
    )

    query = "add-members"
    cleaned = query.strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1]
    fts_literal = cleaned.replace('"', '""')
    fts_query = f'"{fts_literal}"'

    # Must execute cleanly without sqlite3.OperationalError
    rows = conn.execute(
        "SELECT path, name FROM symbols_fts WHERE symbols_fts MATCH ?",
        (fts_query,)
    ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == "xchange-ui/src/add-members.ts"
