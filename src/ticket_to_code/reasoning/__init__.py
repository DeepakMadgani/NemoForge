"""
Dynamic Repository Reasoning Engine (Ticket-to-Code v2).

A ReAct-style reasoner designed for any codebase. It orchestrates repository
capabilities (RAG, discovery, generation, build, verification) as tools and
compounds knowledge across tickets via the Repository Brain.

Public API:
    from ticket_to_code.reasoning import run_reasoning_agent, RepositoryBrain, BrainManager
"""

from .repository_brain import RepositoryBrain, CC4EBrain
from .brain_manager import BrainManager
from .engine import run_reasoning_agent, run_cc4e_reasoning_agent, ReasoningResult

__all__ = [
    "run_reasoning_agent",
    "run_cc4e_reasoning_agent",
    "ReasoningResult",
    "RepositoryBrain",
    "CC4EBrain",
    "BrainManager",
]
