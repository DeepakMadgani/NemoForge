"""Tests for Step 1: ResolutionScope and Query-Time Scoping.

Verifies:
1. TypeScript files (e.g. xchange-ui/.../foo.ts) can resolve ONLY .ts/.tsx/.d.ts in xchange-ui.
2. Java files (e.g. issues-service/.../Foo.java) can resolve ONLY .java in issues-service.
3. Identical symbol names across services/languages do NOT collide.
4. Scoping is enforced at SQL QUERY-TIME (query results physically exclude out-of-scope rows).
5. TypeScript diagnostic -> Java symbol = IMPOSSIBLE.
6. Java diagnostic -> TypeScript symbol = IMPOSSIBLE.
"""

import sys
import sqlite3
import pytest
from pathlib import Path


from ticket_to_code.intelligence.scope.language_scope import (
    ResolutionScope,
    infer_resolution_scope,
)

from ticket_to_code.runtime.storage import SqliteStore


# ── Scope Inference Tests ───────────────────────────────────────────────────

def test_infer_resolution_scope_typescript_frontend():
    path = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    scope = infer_resolution_scope(path)
    assert scope.language == "typescript"
    assert scope.project_root == "xchange-ui"
    assert ".ts" in scope.source_extensions
    assert ".tsx" in scope.source_extensions
    assert ".d.ts" in scope.source_extensions
    assert ".java" not in scope.source_extensions


def test_infer_resolution_scope_java_backend():
    path = "issues-service/src/main/java/com/cloudforge/solutions/services/issues/Foo.java"
    scope = infer_resolution_scope(path)
    assert scope.language == "java"
    assert scope.project_root == "issues-service"
    assert scope.source_extensions == (".java",)
    assert ".ts" not in scope.source_extensions


def test_infer_resolution_scope_with_workspace_root():
    ws = Path("C:/CC4E")
    target = ws / "xchange-ui/src/app/foo.ts"
    scope = infer_resolution_scope(target, workspace_root=ws)
    assert scope.language == "typescript"
    assert scope.project_root == "xchange-ui"


# ── Scope Matching Invariants ───────────────────────────────────────────────

def test_resolution_scope_matches_path_rules():
    ts_scope = ResolutionScope(
        language="typescript",
        project_root="xchange-ui",
        source_extensions=(".ts", ".tsx", ".d.ts"),
    )

    # Valid candidates in scope
    assert ts_scope.matches_path("xchange-ui/src/app/models/participating-members.ts")
    assert ts_scope.matches_path("xchange-ui/src/app/components/add-members.component.ts")
    assert ts_scope.matches_path("xchange-ui/src/typings/index.d.ts")

    # Invalid: wrong extension in same project
    assert not ts_scope.matches_path("xchange-ui/package.json")
    assert not ts_scope.matches_path("xchange-ui/src/styles.scss")

    # Invalid: wrong project root (even if TypeScript)
    assert not ts_scope.matches_path("admin-ui/src/app/foo.ts")

    # Invalid: Java backend files (CROSS-LANGUAGE POLLUTION MUST BE IMPOSSIBLE)
    assert not ts_scope.matches_path("issues-service/src/main/java/com/cloudforge/GqlBasedProjectService.java")
    assert not ts_scope.matches_path("project-service/src/main/java/com/cloudforge/MemberService.java")


# ── Query-Time SQL Scoping Tests ─────────────────────────────────────────────

@pytest.fixture
def populated_sqlite_db(tmp_path: Path) -> SqliteStore:
    """Create a temporary SQLite store with identical symbol names in different services/languages."""
    db_path = tmp_path / "test_symbols.db"
    store = SqliteStore(str(db_path))

    # Insert test symbols into SQLite store
    # 1. Java class in issues-service
    # 2. TypeScript interface in xchange-ui
    # 3. Java class in project-service
    # 4. Another TypeScript helper in xchange-ui
    cur = store._conn.cursor()
    # Insert files first to satisfy FOREIGN KEY constraint
    cur.execute("""
        INSERT INTO files (path, language, sha256, size_bytes, parse_ok)
        VALUES
        ('issues-service/src/main/java/com/cloudforge/GqlBasedProjectService.java', 'java', 'sha1', 100, 1),
        ('xchange-ui/src/app/modules/shared/models/participating-members.ts', 'typescript', 'sha2', 100, 1),
        ('project-service/src/main/java/com/cloudforge/project/MemberService.java', 'java', 'sha3', 100, 1),
        ('xchange-ui/src/app/modules/shared/services/members/member.service.ts', 'typescript', 'sha4', 100, 1)
    """)
    cur.execute("""
        INSERT INTO symbols (id, name, qualified_name, kind, path, start_line, end_line, start_col, end_col)
        VALUES 
        ('sym_java_issues', 'FilterParticipantMemberInput', 'com.cloudforge.issues.FilterParticipantMemberInput', 'class', 'issues-service/src/main/java/com/cloudforge/GqlBasedProjectService.java', 10, 50, 0, 0),
        ('sym_ts_xchange', 'FilterParticipantMemberInput', 'FilterParticipantMemberInput', 'interface', 'xchange-ui/src/app/modules/shared/models/participating-members.ts', 37, 51, 0, 0),
        ('sym_java_project', 'MemberService', 'com.cloudforge.project.MemberService', 'class', 'project-service/src/main/java/com/cloudforge/project/MemberService.java', 20, 100, 0, 0),
        ('sym_ts_xchange_svc', 'MemberService', 'MemberService', 'class', 'xchange-ui/src/app/modules/shared/services/members/member.service.ts', 20, 120, 0, 0)
    """)
    store._conn.commit()
    yield store
    store.close()


def test_typescript_diagnostic_java_symbol_impossible(populated_sqlite_db: SqliteStore):
    """CRITICAL ACCEPTANCE TEST: A TypeScript diagnostic must NEVER resolve to a Java symbol."""
    ts_diagnostic_file = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    scope = infer_resolution_scope(ts_diagnostic_file)

    # Search for symbol 'FilterParticipantMemberInput' which exists in both Java and TS
    results = populated_sqlite_db.search_symbols(
        "FilterParticipantMemberInput",
        resolution_scope=scope,
        limit=5,
    )

    # Must find EXACTLY the TypeScript file in xchange-ui
    assert len(results) == 1
    matched_path = results[0]["path"]
    assert matched_path == "xchange-ui/src/app/modules/shared/models/participating-members.ts"
    assert not matched_path.endswith(".java")
    assert "issues-service" not in matched_path


def test_java_diagnostic_typescript_symbol_impossible(populated_sqlite_db: SqliteStore):
    """CRITICAL ACCEPTANCE TEST: A Java diagnostic must NEVER resolve to a TypeScript symbol."""
    java_diagnostic_file = "issues-service/src/main/java/com/cloudforge/issues/service/Foo.java"
    scope = infer_resolution_scope(java_diagnostic_file)

    results = populated_sqlite_db.search_symbols(
        "FilterParticipantMemberInput",
        resolution_scope=scope,
        limit=5,
    )

    # Must find EXACTLY the Java file in issues-service
    assert len(results) == 1
    matched_path = results[0]["path"]
    assert matched_path == "issues-service/src/main/java/com/cloudforge/GqlBasedProjectService.java"
    assert matched_path.endswith(".java")
    assert "xchange-ui" not in matched_path


def test_sql_query_time_enforcement(populated_sqlite_db: SqliteStore):
    """Verify that the SQL query itself contains the scoping filters (query-time enforcement)."""
    ts_scope = ResolutionScope(
        language="typescript",
        project_root="xchange-ui",
        source_extensions=(".ts", ".tsx", ".d.ts"),
    )
    sql_frag, params = ts_scope.build_sql_filter("path")
    assert "path LIKE ?" in sql_frag
    assert any("%.ts" in p for p in params)
    assert any("xchange-ui" in p for p in params)

    # Execute direct query using the filter fragment
    query = f"SELECT path FROM symbols WHERE name LIKE ?{sql_frag}"
    rows = populated_sqlite_db._conn.execute(query, ["%FilterParticipantMemberInput%"] + params).fetchall()

    paths = [r[0] for r in rows]
    assert len(paths) == 1
    assert paths[0] == "xchange-ui/src/app/modules/shared/models/participating-members.ts"


def test_unscoped_search_preserves_existing_behavior(populated_sqlite_db: SqliteStore):
    """Verify that passing resolution_scope=None maintains original multi-result behavior."""
    results = populated_sqlite_db.search_symbols("FilterParticipantMemberInput", resolution_scope=None)
    assert len(results) == 2  # Both Java and TS returned when unscoped
    paths = {r["path"] for r in results}
    assert "issues-service/src/main/java/com/cloudforge/GqlBasedProjectService.java" in paths
    assert "xchange-ui/src/app/modules/shared/models/participating-members.ts" in paths


def test_find_symbol_paths_storage_abstraction(populated_sqlite_db: SqliteStore):
    """Verify SqliteStore.find_symbol_paths cleanly enforces ResolutionScope without raw SQL in workflow."""
    ts_scope = ResolutionScope(
        language="typescript",
        project_root="xchange-ui",
        source_extensions=(".ts", ".tsx", ".d.ts"),
    )
    # With TS scope: returns only the TS file path
    ts_paths = populated_sqlite_db.find_symbol_paths(
        "FilterParticipantMemberInput",
        resolution_scope=ts_scope,
        limit=5,
    )
    assert ts_paths == ["xchange-ui/src/app/modules/shared/models/participating-members.ts"]

    # With Java scope: returns only the Java file path
    java_scope = ResolutionScope(
        language="java",
        project_root="issues-service",
        source_extensions=(".java",),
    )
    java_paths = populated_sqlite_db.find_symbol_paths(
        "FilterParticipantMemberInput",
        resolution_scope=java_scope,
        limit=5,
    )
    assert java_paths == ["issues-service/src/main/java/com/cloudforge/GqlBasedProjectService.java"]
