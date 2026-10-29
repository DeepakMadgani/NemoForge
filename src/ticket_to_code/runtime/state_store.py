"""
Durable Workflow State Store.

Provides persistent, crash-proof storage for workflow states, steps, telemetry,
and human-in-the-loop decisions. Replaces volatile in-memory Python dictionaries
with atomic SQLite persistence so active tasks survive process restarts or multi-worker deployments.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "chatbot" / "backend" / "workflow_state.db"


class DurableWorkflowStateStore:
    """Thread-safe SQLite-backed workflow state store with memory cache."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path or _DEFAULT_DB_PATH).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._init_db()
        self._load_cache()

    @contextlib.contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workflows (
                        workflow_id TEXT PRIMARY KEY,
                        ticket_id TEXT,
                        project_id TEXT,
                        repo_path TEXT,
                        status TEXT NOT NULL,
                        current_phase TEXT,
                        active_model TEXT,
                        execution_mode TEXT,
                        model_mode TEXT,
                        token_usage TEXT,         -- JSON
                        workflow_output TEXT,     -- JSON
                        workflow_explanation TEXT,-- JSON
                        state TEXT,               -- JSON
                        selective_approvals TEXT, -- JSON
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_steps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_id TEXT NOT NULL,
                        phase TEXT,
                        status TEXT,
                        message TEXT,
                        data TEXT,                -- JSON
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_steps_wid ON workflow_steps(workflow_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status)")
                conn.commit()

    # Only the most recent N workflows are hydrated eagerly at startup. Older
    # rows stay on disk and are read through on demand by get()/__contains__.
    CACHE_PRELOAD_LIMIT = 200

    def _load_cache(self) -> None:
        with self._lock:
            try:
                with self._get_connection() as conn:
                    rows = conn.execute(
                        "SELECT * FROM workflows ORDER BY updated_at DESC, created_at DESC LIMIT ?",
                        (self.CACHE_PRELOAD_LIMIT,),
                    ).fetchall()
                    for r in rows:
                        wid = r["workflow_id"]
                        self._cache[wid] = self._hydrate_workflow(r, conn)
            except Exception as exc:
                logger.error(f"Failed to load workflow state cache: {exc}")

    def mark_stale_running(self, reason: str) -> int:
        """
        Mark every workflow persisted with status 'running' as 'failed'.

        Intended for process startup: nothing can be live in-memory at that
        point, so any 'running' row is an orphan left behind by a crash or
        restart. Updates both the SQLite table and the in-memory cache.
        Returns the number of rows transitioned.
        """
        now = datetime.now().isoformat()
        with self._lock:
            updated = 0
            try:
                with self._get_connection() as conn:
                    cur = conn.execute(
                        "UPDATE workflows SET status = 'failed', error = ?, updated_at = ? WHERE status = 'running'",
                        (reason, now),
                    )
                    updated = cur.rowcount if cur.rowcount is not None else 0
                    conn.commit()
            except Exception as exc:
                logger.warning(f"mark_stale_running: DB update failed: {exc}")
            for wf in self._cache.values():
                if wf.get("status") == "running":
                    wf["status"] = "failed"
                    wf["current_phase"] = "failed"
                    wf["error"] = reason
                    wf["updated_at"] = now
                    wf.setdefault("completed_at", now)
            if updated:
                logger.info(f"mark_stale_running: {updated} orphaned workflow(s) marked failed ({reason})")
            return updated

    def _hydrate_workflow(self, row: sqlite3.Row, conn: sqlite3.Connection) -> Dict[str, Any]:
        wid = row["workflow_id"]
        steps_rows = conn.execute(
            "SELECT phase, status, message, data, timestamp FROM workflow_steps WHERE workflow_id = ? ORDER BY id ASC",
            (wid,)
        ).fetchall()
        steps = []
        for s in steps_rows:
            data_parsed = None
            if s["data"]:
                try:
                    data_parsed = json.loads(s["data"])
                except Exception:
                    data_parsed = {}
            steps.append({
                "phase": s["phase"],
                "status": s["status"],
                "message": s["message"],
                "data": data_parsed,
                "timestamp": s["timestamp"],
            })

        def _loads_or_none(col: str):
            val = row[col]
            if not val:
                return None
            try:
                return json.loads(val)
            except Exception:
                return None

        return {
            "workflow_id": wid,
            "ticket_id": row["ticket_id"],
            "project_id": row["project_id"],
            "repo_path": row["repo_path"],
            "status": row["status"],
            "current_phase": row["current_phase"],
            "active_model": row["active_model"],
            "execution_mode": row["execution_mode"],
            "model_mode": row["model_mode"],
            "token_usage": _loads_or_none("token_usage"),
            "workflow_output": _loads_or_none("workflow_output") or {},
            "workflow_explanation": _loads_or_none("workflow_explanation"),
            "state": _loads_or_none("state") or {},
            "selective_approvals": _loads_or_none("selective_approvals"),
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "steps": steps,
        }

    def get(self, workflow_id: str, default: Any = None) -> Any:
        with self._lock:
            if workflow_id in self._cache:
                return self._cache[workflow_id]
            # Read-through cache from SQLite on cache miss
            try:
                with self._get_connection() as conn:
                    row = conn.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)).fetchone()
                    if row:
                        wf = self._hydrate_workflow(row, conn)
                        self._cache[workflow_id] = wf
                        return wf
            except Exception as exc:
                logger.debug(f"DB read-through failed for {workflow_id}: {exc}")
            return default

    def __getitem__(self, workflow_id: str) -> Dict[str, Any]:
        val = self.get(workflow_id)
        if val is None:
            raise KeyError(workflow_id)
        return val

    def __contains__(self, workflow_id: str) -> bool:
        with self._lock:
            if workflow_id in self._cache:
                return True
            try:
                with self._get_connection() as conn:
                    row = conn.execute("SELECT 1 FROM workflows WHERE workflow_id = ?", (workflow_id,)).fetchone()
                    return row is not None
            except Exception:
                return False

    def refresh(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Forces an atomic reload of workflow state from SQLite DB."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    row = conn.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)).fetchone()
                    if row:
                        wf = self._hydrate_workflow(row, conn)
                        self._cache[workflow_id] = wf
                        return wf
                    self._cache.pop(workflow_id, None)
            except Exception as exc:
                logger.warning(f"Failed to refresh workflow {workflow_id} from DB: {exc}")
            return None

    def set(self, workflow_id: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[workflow_id] = data
            self._flush_workflow_to_db(workflow_id, data)

    def __setitem__(self, workflow_id: str, data: Dict[str, Any]) -> None:
        self.set(workflow_id, data)

    def append_step(self, workflow_id: str, step: Dict[str, Any]) -> None:
        with self._lock:
            wf = self._cache.setdefault(workflow_id, {"workflow_id": workflow_id, "steps": []})
            wf.setdefault("steps", []).append(step)
            now = datetime.now().isoformat()
            wf["updated_at"] = now
            if "phase" in step:
                wf["current_phase"] = step["phase"]

            # Persist single step to DB
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT INTO workflow_steps (workflow_id, phase, status, message, data, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        workflow_id,
                        step.get("phase"),
                        step.get("status"),
                        step.get("message"),
                        json.dumps(step.get("data")) if step.get("data") is not None else None,
                        step.get("timestamp") or now,
                    ))
                    conn.execute("UPDATE workflows SET updated_at = ? WHERE workflow_id = ?", (now, workflow_id))
                    conn.commit()
            except Exception as exc:
                logger.warning(f"Could not append step to DB for {workflow_id}: {exc}")

    def update_status(self, workflow_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            if workflow_id in self._cache:
                self._cache[workflow_id]["status"] = status
                if error:
                    self._cache[workflow_id]["error"] = error
                self._cache[workflow_id]["updated_at"] = datetime.now().isoformat()
                self._flush_workflow_to_db(workflow_id, self._cache[workflow_id])

    def _flush_workflow_to_db(self, workflow_id: str, data: Dict[str, Any]) -> None:
        now = datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO workflows (
                        workflow_id, ticket_id, project_id, repo_path, status,
                        current_phase, active_model, execution_mode, model_mode,
                        token_usage, workflow_output, workflow_explanation,
                        state, selective_approvals, error, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workflow_id) DO UPDATE SET
                        status = excluded.status,
                        current_phase = excluded.current_phase,
                        active_model = excluded.active_model,
                        token_usage = excluded.token_usage,
                        workflow_output = excluded.workflow_output,
                        workflow_explanation = excluded.workflow_explanation,
                        state = excluded.state,
                        selective_approvals = excluded.selective_approvals,
                        error = excluded.error,
                        updated_at = excluded.updated_at
                """, (
                    workflow_id,
                    data.get("ticket_id"),
                    data.get("project_id"),
                    data.get("repo_path"),
                    data.get("status", "running"),
                    data.get("current_phase"),
                    data.get("active_model"),
                    data.get("execution_mode"),
                    data.get("model_mode"),
                    json.dumps(data.get("token_usage")) if data.get("token_usage") is not None else None,
                    json.dumps(data.get("workflow_output")) if data.get("workflow_output") is not None else None,
                    json.dumps(data.get("workflow_explanation")) if data.get("workflow_explanation") is not None else None,
                    json.dumps(data.get("state")) if data.get("state") is not None else None,
                    json.dumps(data.get("selective_approvals")) if data.get("selective_approvals") is not None else None,
                    data.get("error"),
                    data.get("created_at") or now,
                    now,
                ))
                conn.commit()
        except Exception as exc:
            logger.warning(f"Could not flush workflow {workflow_id} to DB: {exc}")

    def keys(self):
        with self._lock:
            return list(self._cache.keys())

    def __iter__(self):
        with self._lock:
            return iter(list(self._cache.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def values(self):
        with self._lock:
            return list(self._cache.values())

    def items(self):
        with self._lock:
            return list(self._cache.items())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            try:
                with self._get_connection() as conn:
                    conn.execute("DELETE FROM workflow_steps")
                    conn.execute("DELETE FROM workflows")
                    conn.commit()
            except Exception as exc:
                logger.warning(f"Failed to clear workflow state DB: {exc}")

    def pop(self, workflow_id: str, default: Any = None) -> Any:
        with self._lock:
            val = self._cache.pop(workflow_id, default)
            try:
                with self._get_connection() as conn:
                    conn.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
                    conn.commit()
            except Exception:
                pass
            return val

