"""
Standalone Vector Store Manager for Ticket-to-Code.

Replaces aviator.vector_store.VectorStoreManager with a direct pgvector
implementation using langchain_postgres (or a graceful no-op fallback).

Zero dependency on Aviator platform.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class _NoOpVectorStore:
    """
    Fallback vector store that does nothing — used when pgvector is
    unreachable or langchain_postgres is not installed.
    """

    def __init__(self, schema_name: str = "default"):
        self.schema_name = schema_name
        logger.warning(
            "VectorStore(%s) operating in no-op mode (pgvector unavailable)",
            schema_name,
        )

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> list:
        return []

    def add_texts(self, texts: list, metadatas: Optional[list] = None, **kwargs) -> list:
        logger.debug("No-op add_texts: %d texts skipped", len(texts))
        return [str(i) for i in range(len(texts))]

    def add_documents(self, documents: list, **kwargs) -> list:
        logger.debug("No-op add_documents: %d docs skipped", len(documents))
        return []

    def delete(self, ids: Optional[list] = None, filter: Optional[Dict] = None, **kwargs):
        pass

    def as_retriever(self, **kwargs):
        """Return self as a retriever-compatible object."""
        return self


class VectorStoreManager:
    """
    Standalone VectorStoreManager — provides the same .get(schema_name=...)
    interface as aviator.vector_store.VectorStoreManager.

    Connects to a PostgreSQL/pgvector instance configured via the
    POSTGRES_CONNECTION environment variable.
    """

    def __init__(self):
        self._stores: Dict[str, Any] = {}
        self._pg_online: Optional[bool] = None

    def _check_pg_online(self) -> bool:
        """Fast-fail check for PG connectivity (1.5s timeout)."""
        if self._pg_online is not None:
            return self._pg_online

        conn_str = os.getenv("POSTGRES_CONNECTION", "")
        if not conn_str:
            self._pg_online = False
            return False

        # Parse host:port from connection string
        host = "127.0.0.1"
        port = 5433
        try:
            # postgresql://user:pass@host:port/db
            if "@" in conn_str:
                after_at = conn_str.split("@")[1]
                host_port = after_at.split("/")[0]
                if ":" in host_port:
                    host, port_str = host_port.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = host_port
        except Exception:
            pass

        try:
            with socket.create_connection((host, port), timeout=1.5):
                self._pg_online = True
        except Exception:
            self._pg_online = False

        return self._pg_online

    def get(self, schema_name: str = "default", **kwargs) -> Any:
        """
        Get or create a vector store for the given schema.

        Returns a langchain_postgres PGVector store if available,
        otherwise a no-op fallback.
        """
        if schema_name in self._stores:
            return self._stores[schema_name]

        if not self._check_pg_online():
            store = _NoOpVectorStore(schema_name)
            self._stores[schema_name] = store
            return store

        try:
            from langchain_postgres import PGVector
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings_model = os.getenv("EMBEDDINGS_MODEL", "models/text-embedding-004")
            embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)

            conn_str = os.getenv("POSTGRES_CONNECTION", "")
            store = PGVector(
                collection_name=schema_name,
                connection=conn_str,
                embeddings=embeddings,
            )
            self._stores[schema_name] = store
            logger.info("VectorStore(%s) initialized with pgvector", schema_name)
            return store

        except Exception as exc:
            logger.warning(
                "VectorStore(%s) falling back to no-op: %s", schema_name, exc
            )
            store = _NoOpVectorStore(schema_name)
            self._stores[schema_name] = store
            return store

    _default_instance: Optional["VectorStoreManager"] = None

    @classmethod
    def get_vector_store(cls, collection_name: str = "default", schema_name: Optional[str] = None, **kwargs) -> Any:
        """Classmethod returning a vector store for the given collection/schema."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        target_schema = schema_name or collection_name
        return cls._default_instance.get(schema_name=target_schema, **kwargs)


# Compatibility aliases for aviator_core.storage.vector_store
VectorStore = _NoOpVectorStore


class VectorLocalizer:
    """Fallback vector-based code localizer."""

    def __init__(self, vector_store: Any = None, **kwargs):
        self.vector_store = vector_store or _NoOpVectorStore("codebase_knowledge")

    def localize(self, query: str, top_k: int = 5, **kwargs) -> list:
        try:
            return self.vector_store.similarity_search(query, k=top_k)
        except Exception:
            return []
