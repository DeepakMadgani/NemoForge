"""
Chatbot service wrapper importing the canonical CallChainTracer.
"""

from ticket_to_code.lean.call_chain_tracer import CallChainTracer, _PRUNE_DIRS, _UI_PLUMBING

__all__ = ["CallChainTracer", "_PRUNE_DIRS", "_UI_PLUMBING"]
