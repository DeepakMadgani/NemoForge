"""
Agents package for Autonomous Ticket-to-Code System

Contains all AI agents:
- Investigation Agent
- Ticket Analyzer Agent
- Localization Agent (NEW - most critical for autonomous engineering)
- Planning Agent
- Code Generator Agent

Note: RAG Engine moved to ticket_to_code.retrieval.rag_engine
Note: Codebase Indexer moved to ticket_to_code.indexing.codebase_indexer
"""

from .investigation_agent import InvestigationAgent
from .ticket_analyzer import TicketAnalyzerAgent, analyze_ticket
from .localization_agent import LocalizationAgent, localize_targets
from .planning_agent import PlanningAgent, create_plan
from .code_generator import CodeGeneratorAgent, generate_code
from .artifact_generator import ArtifactGeneratorAgent
from .evidence_collection_loop import EvidenceCollectionLoop
from .repository_search_engine import RepositorySearchEngine, SearchResult

# Import from agents.rag_engine
from .rag_engine import CodebaseRAGEngine

__all__ = [
    "InvestigationAgent",
    "TicketAnalyzerAgent",
    "analyze_ticket",
    "LocalizationAgent",
    "localize_targets",
    "PlanningAgent",
    "create_plan",
    "CodebaseRAGEngine",
    "CodeGeneratorAgent",
    "generate_code",
    "ArtifactGeneratorAgent",
]
