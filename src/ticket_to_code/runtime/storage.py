"""
Standalone Storage Implementations for Ticket-to-Code.

Provides SqliteStore and Neo4jStore wrappers that replicate storage interfaces
without any dependency on external aviator platforms.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Protocol, Tuple, Union, runtime_checkable

from ticket_to_code.runtime.compat_models import Edge, EdgeKind, FileRecord, Symbol, SymbolKind

logger = logging.getLogger(__name__)


@runtime_checkable
class ResolutionScopeProtocol(Protocol):
    """Protocol decoupling storage layer from specific intelligence scope implementations."""

    def build_sql_filter(self, path_column: str = "path") -> Tuple[str, list]:
        ...

    def matches_path(self, path: Union[str, Path]) -> bool:
        ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path           TEXT PRIMARY KEY,
    language       TEXT NOT NULL,
    package        TEXT,
    sha256         TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    parse_ok       INTEGER NOT NULL,
    parse_error    TEXT,
    indexed_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS symbols (
    id             TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    type           TEXT,
    name           TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    package        TEXT,
    parent_id      TEXT,
    path           TEXT NOT NULL,
    start_line     INTEGER NOT NULL,
    end_line       INTEGER NOT NULL,
    start_col      INTEGER NOT NULL,
    end_col        INTEGER NOT NULL,
    signature      TEXT,
    return_type    TEXT,
    modifiers      TEXT,        -- JSON array
    annotations    TEXT,        -- JSON array
    parameter_types TEXT,       -- JSON array
    spring_stereotype TEXT,     -- Spring stereotype (Controller, Service, etc.)
    spring_endpoints TEXT,      -- JSON array of REST endpoints
    spring_dependencies TEXT,   -- JSON array of @Autowired dependencies
    is_feign_client INTEGER DEFAULT 0,
    feign_service_name TEXT,
    FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_symbols_name           ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_qualified_name ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_symbols_kind           ON symbols(kind);
CREATE INDEX IF NOT EXISTS idx_symbols_path           ON symbols(path);
CREATE INDEX IF NOT EXISTS idx_symbols_parent         ON symbols(parent_id);
CREATE INDEX IF NOT EXISTS idx_symbols_spring_stereotype ON symbols(spring_stereotype);
CREATE INDEX IF NOT EXISTS idx_symbols_is_feign_client ON symbols(is_feign_client);

CREATE TABLE IF NOT EXISTS edges (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT NOT NULL,
    src_id         TEXT NOT NULL,
    dst_id         TEXT,
    dst_name       TEXT NOT NULL,
    path           TEXT,
    start_line     INTEGER,
    FOREIGN KEY (src_id) REFERENCES symbols(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_src      ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst      ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst_name ON edges(dst_name);
CREATE INDEX IF NOT EXISTS idx_edges_kind     ON edges(kind);

-- Full-text search over symbol identifiers.
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    qualified_name,
    signature,
    content='symbols',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, name, qualified_name, signature)
    VALUES (new.rowid, new.name, new.qualified_name, COALESCE(new.signature, ''));
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name, signature)
    VALUES ('delete', old.rowid, old.name, old.qualified_name, COALESCE(old.signature, ''));
END;

CREATE TABLE IF NOT EXISTS project_config (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    source     TEXT,   -- which file it came from
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class SqliteStore:
    """A small wrapper that exposes idiomatic batch upserts and querying."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA busy_timeout = 5000;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the active SQLite connection."""
        yield self._conn

    def commit(self) -> None:
        """Commit any pending transactions."""
        if hasattr(self, "_conn") and self._conn:
            try:
                self._conn.commit()
            except Exception:
                pass

    def close(self) -> None:
        """Close the SQLite connection."""
        if hasattr(self, "_conn") and self._conn:
            try:
                self._conn.commit()
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def upsert_project_config(self, key: str, value: str, source: str) -> None:
        self._conn.execute(
            """
            INSERT INTO project_config(key, value, source, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                source=excluded.source, updated_at=excluded.updated_at
            """,
            (key, value, source),
        )
        self._conn.commit()

    def get_project_config(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM project_config ORDER BY key"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------ writes

    def upsert_file(self, file: Any) -> None:
        path = file["path"] if isinstance(file, dict) else file.path
        language = file.get("language", "") if isinstance(file, dict) else getattr(file, "language", "")
        package = file.get("package", "") if isinstance(file, dict) else getattr(file, "package", "")
        sha256 = file.get("sha256", "") if isinstance(file, dict) else getattr(file, "sha256", "")
        size_bytes = file.get("size_bytes", 0) if isinstance(file, dict) else getattr(file, "size_bytes", 0)
        parse_ok = file.get("parse_ok", True) if isinstance(file, dict) else getattr(file, "parse_ok", True)
        parse_error = file.get("parse_error") if isinstance(file, dict) else getattr(file, "parse_error", None)

        self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._conn.execute(
            """
            INSERT INTO files(path, language, package, sha256, size_bytes,
                              parse_ok, parse_error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                path,
                language,
                package,
                sha256,
                size_bytes,
                1 if parse_ok else 0,
                parse_error,
            ),
        )
        self._conn.commit()

    def upsert_symbol(self, symbol: Any) -> None:
        self.insert_symbols([symbol])

    def insert_symbols(self, symbols: Iterable[Any]) -> None:
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        rows = []
        for s in symbols:
            s_id = _get(s, "id") or ""
            kind_obj = _get(s, "kind")
            kind_val = kind_obj.value if hasattr(kind_obj, "value") else str(kind_obj or "")
            type_val = _get(s, "type", kind_val)
            name = _get(s, "name", "")
            qn = _get(s, "qualified_name", name)
            pkg = _get(s, "package", "")
            parent_id = _get(s, "parent_id") or _get(s, "parent")
            loc = _get(s, "location")
            if loc:
                loc_path = _get(loc, "path", "")
                s_line = _get(loc, "start_line", 0)
                e_line = _get(loc, "end_line", 0)
                s_col = _get(loc, "start_col", 0)
                e_col = _get(loc, "end_col", 0)
            else:
                loc_path = _get(s, "path") or _get(s, "file_path", "")
                s_line = _get(s, "start_line", 0)
                e_line = _get(s, "end_line", 0)
                s_col = _get(s, "start_col", 0)
                e_col = _get(s, "end_col", 0)
            sig = _get(s, "signature")
            ret_type = _get(s, "return_type")
            modifiers = json.dumps(_get(s, "modifiers", []))
            annotations = json.dumps(_get(s, "annotations", []))
            param_types = json.dumps(_get(s, "parameter_types", []))
            spring_stereo = _get(s, "spring_stereotype")
            spring_endpoints = json.dumps(_get(s, "spring_endpoints", []))
            spring_deps = json.dumps(_get(s, "spring_dependencies", []))
            is_feign = 1 if _get(s, "is_feign_client") else 0
            feign_name = _get(s, "feign_service_name")
            rows.append((
                s_id, kind_val, type_val, name, qn, pkg, parent_id,
                loc_path, s_line, e_line, s_col, e_col,
                sig, ret_type, modifiers, annotations, param_types,
                spring_stereo, spring_endpoints, spring_deps,
                is_feign, feign_name
            ))
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO symbols(
                id, kind, type, name, qualified_name, package, parent_id,
                path, start_line, end_line, start_col, end_col,
                signature, return_type, modifiers, annotations, parameter_types,
                spring_stereotype, spring_endpoints, spring_dependencies,
                is_feign_client, feign_service_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def insert_edges(self, edges: Iterable[Edge]) -> None:
        rows = [
            (
                e.kind.value if hasattr(e.kind, "value") else str(e.kind),
                e.src_id,
                e.dst_id,
                e.dst_name,
                e.location.path if e.location else None,
                e.location.start_line if e.location else None,
            )
            for e in edges
        ]
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO edges(kind, src_id, dst_id, dst_name, path, start_line)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------- reads

    def file_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def symbol_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]

    def find_symbol_paths(
        self,
        name: str,
        *,
        resolution_scope: Optional[ResolutionScopeProtocol] = None,
        limit: int = 5,
    ) -> list[str]:
        """Find distinct file paths defining a symbol by name with optional query-time scope filtering."""
        scope_sql = ""
        scope_params: list[object] = []
        if resolution_scope is not None and hasattr(resolution_scope, "build_sql_filter"):
            scope_sql, scope_params = resolution_scope.build_sql_filter("path")

        cur = self._conn.cursor()
        query = f"SELECT DISTINCT path FROM symbols WHERE name LIKE ?{scope_sql} LIMIT ?"
        rows = cur.execute(query, [f"%{name}%"] + scope_params + [limit]).fetchall()
        paths: list[str] = []
        for r in rows:
            p = r[0] if isinstance(r, (tuple, list)) else r["path"]
            if resolution_scope is None or resolution_scope.matches_path(p):
                paths.append(str(p))
        return paths

    def search_symbols(
        self,
        text: str,
        *,
        kinds: Optional[list[str]] = None,
        resolution_scope: Optional[ResolutionScopeProtocol] = None,
        limit: int = 25,
    ) -> list[sqlite3.Row]:
        """Hybrid keyword + FTS search over symbols."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.cursor()
        kind_filter = ""
        params: list[object] = []
        if kinds:
            placeholders = ",".join("?" * len(kinds))
            kind_filter = f" AND s.kind IN ({placeholders})"
            params.extend(kinds)

        scope_sql = ""
        scope_params: list[object] = []
        if resolution_scope is not None and hasattr(resolution_scope, "build_sql_filter"):
            scope_sql, scope_params = resolution_scope.build_sql_filter("s.path")

        # FTS attempt
        try:
            fts_params = [text] + params + scope_params + [limit]
            rows = cur.execute(
                f"""
                SELECT s.* FROM symbols_fts f
                JOIN symbols s ON s.rowid = f.rowid
                WHERE symbols_fts MATCH ?{kind_filter}{scope_sql}
                LIMIT ?
                """,
                fts_params,
            ).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass

        like_params = [f"%{text}%", f"%{text}%"] + params + scope_params + [limit]
        return cur.execute(
            f"""
            SELECT s.* FROM symbols s
            WHERE (s.name LIKE ? OR s.qualified_name LIKE ?){kind_filter}{scope_sql}
            LIMIT ?
            """,
            like_params,
        ).fetchall()

    def neighbors(self, symbol_id: str, *, direction: str = "out") -> list[sqlite3.Row]:
        """Return adjacent edges for a symbol."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.cursor()
        if direction == "out":
            return cur.execute(
                "SELECT * FROM edges WHERE src_id = ?", (symbol_id,)
            ).fetchall()
        if direction == "in":
            return cur.execute(
                "SELECT * FROM edges WHERE dst_id = ?", (symbol_id,)
            ).fetchall()
        return cur.execute(
            "SELECT * FROM edges WHERE src_id = ? OR dst_id = ?",
            (symbol_id, symbol_id),
        ).fetchall()

    def get_symbol(self, symbol_id: str) -> Optional[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(
            "SELECT * FROM symbols WHERE id = ?", (symbol_id,)
        ).fetchone()

    def close(self) -> None:
        self._conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Neo4jStore — drop-in replacement for aviator_core.storage.neo4j_store
# ═══════════════════════════════════════════════════════════════════════════════

class Neo4jStore:
    """Graph database backend using Neo4j with graceful degradation."""

    def __init__(
        self,
        workspace_path: str = "",
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "Neo4j support requires the neo4j driver. "
                "Install with: pip install neo4j"
            )

        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")

        if not self.password:
            raise ValueError(
                "Neo4j password required. Set NEO4J_PASSWORD environment variable "
                "or pass password parameter."
            )

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.workspace_path = workspace_path
        self._create_constraints()

    def _create_constraints(self) -> None:
        with self.driver.session() as session:
            old_constraints = [
                "DROP CONSTRAINT symbol_id_unique IF EXISTS",
                "DROP CONSTRAINT file_path_unique IF EXISTS",
            ]
            for old in old_constraints:
                try:
                    session.run(old)
                except Exception:
                    pass

            constraints = [
                "CREATE CONSTRAINT symbol_ws_id_unique IF NOT EXISTS FOR (s:Symbol) REQUIRE (s.workspace, s.id) IS UNIQUE",
                "CREATE CONSTRAINT file_ws_path_unique IF NOT EXISTS FOR (f:File) REQUIRE (f.workspace, f.path) IS UNIQUE",
            ]
            indexes = [
                "CREATE INDEX symbol_name IF NOT EXISTS FOR (s:Symbol) ON (s.name)",
                "CREATE INDEX symbol_qname IF NOT EXISTS FOR (s:Symbol) ON (s.qualified_name)",
                "CREATE INDEX symbol_kind IF NOT EXISTS FOR (s:Symbol) ON (s.kind)",
                "CREATE INDEX symbol_stereotype IF NOT EXISTS FOR (s:Symbol) ON (s.spring_stereotype)",
                "CREATE INDEX file_package IF NOT EXISTS FOR (f:File) ON (f.package)",
                "CREATE INDEX symbol_workspace IF NOT EXISTS FOR (s:Symbol) ON (s.workspace)",
                "CREATE INDEX file_workspace IF NOT EXISTS FOR (f:File) ON (f.workspace)",
            ]
            for c in constraints:
                try:
                    session.run(c)
                except Exception:
                    pass
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception:
                    pass

    def upsert_file(self, file: FileRecord) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (f:File {workspace: $workspace, path: $path})
                SET f.language = $language,
                    f.package = $package,
                    f.sha256 = $sha256,
                    f.size_bytes = $size_bytes,
                    f.parse_ok = $parse_ok,
                    f.parse_error = $parse_error
                """,
                path=file.path,
                workspace=str(self.workspace_path),
                language=file.language,
                package=file.package,
                sha256=file.sha256,
                size_bytes=file.size_bytes,
                parse_ok=file.parse_ok,
                parse_error=file.parse_error,
            )

    _BATCH_SIZE = 500

    def insert_symbols(self, symbols: Iterable[Symbol]) -> None:
        batch: list[dict] = []
        parent_batch: list[dict] = []

        def _flush(sess) -> None:
            if batch:
                sess.run(
                    """
                    UNWIND $rows AS r
                    MERGE (s:Symbol {workspace: $workspace, id: r.id})
                    SET s.kind = r.kind,
                        s.name = r.name,
                        s.qualified_name = r.qualified_name,
                        s.package = r.package,
                        s.path = r.path,
                        s.start_line = r.start_line,
                        s.end_line = r.end_line,
                        s.signature = r.signature,
                        s.return_type = r.return_type,
                        s.modifiers = r.modifiers,
                        s.annotations = r.annotations,
                        s.spring_stereotype = r.spring_stereotype,
                        s.spring_endpoints = r.spring_endpoints,
                        s.is_feign_client = r.is_feign_client,
                        s.feign_service_name = r.feign_service_name
                    WITH s, r
                    MERGE (f:File {workspace: $workspace, path: r.path})
                    MERGE (f)-[:CONTAINS]->(s)
                    """,
                    rows=batch,
                    workspace=str(self.workspace_path),
                )
                batch.clear()
            if parent_batch:
                sess.run(
                    """
                    UNWIND $rows AS r
                    MATCH (parent:Symbol {workspace: $workspace, id: r.parent_id})
                    MATCH (child:Symbol {workspace: $workspace, id: r.child_id})
                    MERGE (parent)-[:CONTAINS]->(child)
                    """,
                    rows=parent_batch,
                    workspace=str(self.workspace_path),
                )
                parent_batch.clear()

        with self.driver.session() as session:
            for symbol in symbols:
                batch.append({
                    "id": symbol.id,
                    "kind": symbol.kind.value if hasattr(symbol.kind, "value") else str(symbol.kind),
                    "name": symbol.name,
                    "qualified_name": symbol.qualified_name,
                    "package": symbol.package,
                    "path": symbol.location.path if symbol.location else "",
                    "start_line": symbol.location.start_line if symbol.location else 0,
                    "end_line": symbol.location.end_line if symbol.location else 0,
                    "signature": symbol.signature,
                    "return_type": symbol.return_type,
                    "modifiers": symbol.modifiers,
                    "annotations": symbol.annotations,
                    "spring_stereotype": symbol.spring_stereotype,
                    "spring_endpoints": symbol.spring_endpoints,
                    "is_feign_client": symbol.is_feign_client,
                    "feign_service_name": symbol.feign_service_name,
                })
                if symbol.parent_id:
                    parent_batch.append({"parent_id": symbol.parent_id, "child_id": symbol.id})
                if len(batch) >= self._BATCH_SIZE:
                    _flush(session)
            _flush(session)

    def insert_edges(self, edges: Iterable[Edge]) -> None:
        resolved: list[dict] = []
        unresolved: list[dict] = []

        for edge in edges:
            kind_val = edge.kind.value if hasattr(edge.kind, "value") else str(edge.kind)
            if edge.dst_id:
                resolved.append({
                    "src_id": edge.src_id,
                    "dst_id": edge.dst_id,
                    "dst_name": edge.dst_name,
                    "rel_type": kind_val.upper(),
                })
            else:
                unresolved.append({
                    "src_id": edge.src_id,
                    "dst_name": edge.dst_name,
                    "kind": kind_val,
                })

        with self.driver.session() as session:
            from itertools import groupby
            resolved_sorted = sorted(resolved, key=lambda e: e["rel_type"])
            for rel_type, group in groupby(resolved_sorted, key=lambda e: e["rel_type"]):
                rows = list(group)
                for i in range(0, len(rows), self._BATCH_SIZE):
                    chunk = rows[i:i + self._BATCH_SIZE]
                    session.run(
                        f"""
                        UNWIND $rows AS r
                        MATCH (src:Symbol {{workspace: $workspace, id: r.src_id}})
                        MATCH (dst:Symbol {{workspace: $workspace, id: r.dst_id}})
                        MERGE (src)-[:{rel_type} {{dst_name: r.dst_name}}]->(dst)
                        """,
                        rows=chunk,
                        workspace=str(self.workspace_path),
                    )

            for i in range(0, len(unresolved), self._BATCH_SIZE):
                chunk = unresolved[i:i + self._BATCH_SIZE]
                session.run(
                    """
                    UNWIND $rows AS r
                    MERGE (ext:External {name: r.dst_name})
                    WITH ext, r
                    MATCH (src:Symbol {workspace: $workspace, id: r.src_id})
                    MERGE (src)-[:REFERENCES {kind: r.kind}]->(ext)
                    """,
                    rows=chunk,
                    workspace=str(self.workspace_path),
                )

    def find_call_chain(self, start_symbol_id: str, max_depth: int = 3) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start:Symbol {workspace: $workspace, id: $start_id})-[:CALLS*1..%d]->(end:Symbol {workspace: $workspace})
                RETURN [node in nodes(path) | {
                    id: node.id,
                    name: node.name,
                    path: node.path,
                    kind: node.kind
                }] as chain
                LIMIT 50
                """ % max_depth,
                start_id=start_symbol_id,
                workspace=str(self.workspace_path),
            )
            return [record["chain"] for record in result]

    def find_spring_layer_traversal(self, entry_point: str, max_depth: int = 3) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start:Symbol {workspace: $workspace, id: $entry})-[:CALLS*1..%d]->(end:Symbol {workspace: $workspace})
                WHERE start.spring_stereotype IN ['Controller', 'RestController']
                AND end.spring_stereotype IN ['Service', 'Repository']
                RETURN [node in nodes(path) | {
                    id: node.id,
                    name: node.name,
                    stereotype: node.spring_stereotype,
                    path: node.path
                }] as layers
                LIMIT 30
                """ % max_depth,
                entry=entry_point,
                workspace=str(self.workspace_path),
            )
            return [record["layers"] for record in result]

    def find_microservice_dependencies(self, service_name: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (f:File {workspace: $workspace, package: $service})-[:CONTAINS]->(s:Symbol {workspace: $workspace})
                WHERE s.is_feign_client = true
                RETURN s.name as client_name,
                       s.feign_service_name as target_service,
                       s.path as path
                LIMIT 50
                """,
                service=service_name,
                workspace=str(self.workspace_path),
            )
            return [dict(record) for record in result]

    def clear_repository(self, repo_path: str) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MATCH (n {workspace: $workspace})
                WHERE n.path STARTS WITH $prefix
                DETACH DELETE n
                """,
                prefix=repo_path,
                workspace=str(self.workspace_path),
            )

    def query(self, query_string: str, parameters: Optional[dict] = None) -> list[dict]:
        if parameters is None:
            parameters = {}
        if hasattr(self, 'workspace_path') and self.workspace_path:
            parameters['workspace'] = str(self.workspace_path)
            if '$workspace' not in query_string:
                raise ValueError('SECURITY REJECTION: All Neo4j queries must be scoped with {workspace: $workspace}')
        with self.driver.session() as session:
            result = session.run(query_string, parameters)
            return [record.data() for record in result]

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
