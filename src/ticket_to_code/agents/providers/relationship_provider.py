"""
RelationshipProvider - type-only interface for code relationship lookups
(e.g. call graph, imports) used by EvidenceCollectionLoop.

This is currently a minimal placeholder: no concrete provider is wired up
in this checkout, so RelationshipProviderFactory.get_provider() returns None
and callers treat the relationship provider as optional.
"""

from typing import Protocol


class RelationshipProvider(Protocol):
    """Interface for a relationship/call-graph lookup provider."""
    ...
