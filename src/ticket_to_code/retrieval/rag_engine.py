"""
Re-export shim: CodebaseRAGEngine lives in ticket_to_code.agents.rag_engine.
Kept so imports of ticket_to_code.retrieval.rag_engine keep working without duplicating implementation.
"""

from ticket_to_code.agents.rag_engine import CodebaseRAGEngine

__all__ = ["CodebaseRAGEngine"]
