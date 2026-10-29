"""
Storage Health & Distributed Store Synchronization Coordinator.

Tracks synchronization and operational health across the 3 storage backends:
1. SQLite (AST symbols, files, edges, FTS5)
2. pgvector (Vector embeddings for semantic RAG)
3. Neo4j (Graph database for cross-service call-chains and relationships)

Prevents the "Split-Brain" problem where the agent queries a stale or degraded store
without knowing that an earlier indexing phase failed or was desynced.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class StoreStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    DESYNCED = "desynced"


@dataclass
class StorageSyncManifest:
    project_id: str
    workspace_path: str
    sqlite_status: StoreStatus = StoreStatus.HEALTHY
    pgvector_status: StoreStatus = StoreStatus.HEALTHY
    neo4j_status: StoreStatus = StoreStatus.HEALTHY
    last_indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    files_indexed: int = 0
    symbols_indexed: int = 0
    vector_chunks: int = 0
    neo4j_nodes: int = 0
    neo4j_edges: int = 0
    content_hash: str = ""
    sync_errors: Dict[str, str] = field(default_factory=dict)

    @property
    def is_fully_synced(self) -> bool:
        return (
            self.sqlite_status == StoreStatus.HEALTHY
            and self.pgvector_status == StoreStatus.HEALTHY
            and self.neo4j_status == StoreStatus.HEALTHY
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "workspace_path": self.workspace_path,
            "is_fully_synced": self.is_fully_synced,
            "sqlite_status": self.sqlite_status.value,
            "pgvector_status": self.pgvector_status.value,
            "neo4j_status": self.neo4j_status.value,
            "last_indexed_at": self.last_indexed_at,
            "files_indexed": self.files_indexed,
            "symbols_indexed": self.symbols_indexed,
            "vector_chunks": self.vector_chunks,
            "neo4j_nodes": self.neo4j_nodes,
            "neo4j_edges": self.neo4j_edges,
            "content_hash": self.content_hash,
            "sync_errors": self.sync_errors,
        }


class StorageHealthCoordinator:
    """Monitors live store reachability and records sync manifests."""

    @staticmethod
    def check_port_reachable(host: str, port: int, timeout_s: float = 1.5) -> bool:
        """Fast non-blocking port check."""
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return True
        except Exception:
            return False

    @classmethod
    def get_pgvector_health(cls, host: str = "127.0.0.1", port: int = 5433) -> Tuple[StoreStatus, Optional[str]]:
        if not cls.check_port_reachable(host, port):
            return StoreStatus.OFFLINE, f"Port {port} unreachable on {host}"
        try:
            import psycopg2
            pw = os.getenv("POSTGRES_PASSWORD", "postgres")
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=pw,
                dbname=os.getenv("POSTGRES_DB", "postgres"),
                connect_timeout=2
            )
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            cur.close()
            conn.close()
            return StoreStatus.HEALTHY, None
        except Exception as exc:
            return StoreStatus.DEGRADED, str(exc)

    @classmethod
    def get_neo4j_health(cls, uri: Optional[str] = None) -> Tuple[StoreStatus, Optional[str]]:
        host = "127.0.0.1"
        port = 7687
        if not cls.check_port_reachable(host, port):
            return StoreStatus.OFFLINE, f"Neo4j Bolt port {port} unreachable on {host}"
        try:
            from neo4j import GraphDatabase
            target_uri = uri or os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            pw = os.getenv("NEO4J_PASSWORD", "aviator-dev")
            driver = GraphDatabase.driver(target_uri, auth=(user, pw))
            with driver.session() as s:
                s.run("RETURN 1").single()
            driver.close()
            return StoreStatus.HEALTHY, None
        except Exception as exc:
            return StoreStatus.DEGRADED, str(exc)

    @classmethod
    def save_manifest(cls, manifest: StorageSyncManifest, workspace_path: Path | str) -> None:
        """Persists the sync manifest into .ticket_to_code/storage_sync.json."""
        meta_dir = Path(workspace_path) / ".ticket_to_code"
        meta_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = meta_dir / "storage_sync.json"
        manifest_file.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_manifest(cls, workspace_path: Path | str) -> Optional[StorageSyncManifest]:
        """Loads the storage sync manifest from disk if it exists."""
        manifest_file = Path(workspace_path) / ".ticket_to_code" / "storage_sync.json"
        if not manifest_file.exists():
            return None
        try:
            d = json.loads(manifest_file.read_text(encoding="utf-8"))
            return StorageSyncManifest(
                project_id=d.get("project_id", ""),
                workspace_path=d.get("workspace_path", ""),
                sqlite_status=StoreStatus(d.get("sqlite_status", "healthy")),
                pgvector_status=StoreStatus(d.get("pgvector_status", "healthy")),
                neo4j_status=StoreStatus(d.get("neo4j_status", "healthy")),
                last_indexed_at=d.get("last_indexed_at", ""),
                files_indexed=d.get("files_indexed", 0),
                symbols_indexed=d.get("symbols_indexed", 0),
                vector_chunks=d.get("vector_chunks", 0),
                neo4j_nodes=d.get("neo4j_nodes", 0),
                neo4j_edges=d.get("neo4j_edges", 0),
                content_hash=d.get("content_hash", ""),
                sync_errors=d.get("sync_errors", {}),
            )
        except Exception as exc:
            logger.warning(f"Failed to read storage sync manifest: {exc}")
            return None

    @classmethod
    def check_git_drift(cls, workspace_path: Path | str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Verifies whether the current git HEAD commit matches the last indexed commit in the manifest.
        Returns (has_drift, current_head_commit, indexed_commit).
        """
        ws = Path(workspace_path)
        manifest = cls.load_manifest(ws)
        if not manifest:
            return False, None, None

        current_head = None
        if (ws / ".git").exists():
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(ws),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=(sys.platform == "win32"),
                )
                if res.returncode == 0:
                    current_head = res.stdout.strip()
            except Exception:
                pass

        if not current_head:
            return False, None, manifest.content_hash or None

        indexed = manifest.content_hash or None
        has_drift = bool(indexed and current_head and indexed != current_head)
        return has_drift, current_head, indexed

    @classmethod
    def reconcile_storage_if_drifted(cls, workspace_path: Path | str) -> Dict[str, Any]:
        """
        Automatically reconciles storage if git drift is detected by triggering IncrementalIndexer.
        """
        has_drift, current_head, indexed = cls.check_git_drift(workspace_path)
        if not has_drift or not current_head:
            return {"reconciled": False, "reason": "No drift detected or git not available", "active_commit": current_head}

        try:
            from ticket_to_code.indexing.incremental_indexer import IncrementalIndexer
            indexer = IncrementalIndexer(str(workspace_path))
            update_res = indexer.update_index()

            # Update manifest with new commit
            manifest = cls.load_manifest(workspace_path)
            if manifest:
                manifest.content_hash = current_head
                manifest.last_indexed_at = datetime.now().isoformat()
                cls.save_manifest(manifest, workspace_path)

            return {
                "reconciled": True,
                "previous_commit": indexed,
                "current_commit": current_head,
                "files_modified": update_res.files_modified,
                "files_added": update_res.files_added,
                "files_deleted": update_res.files_deleted,
            }
        except Exception as exc:
            logger.warning(f"Storage reconciliation failed: {exc}")
            return {"reconciled": False, "error": str(exc), "current_commit": current_head}
