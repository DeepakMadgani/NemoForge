"""
RelationshipProviderFactory - resolves a RelationshipProvider for a workspace.

Placeholder: no concrete relationship provider is implemented in this
checkout yet, so this always returns None. Callers (workflow.py,
EvidenceCollectionLoop) already treat the relationship provider as optional.
"""

from typing import Optional

from ticket_to_code.agents.providers.relationship_provider import RelationshipProvider


class RelationshipProviderFactory:
    @staticmethod
    def get_provider(workspace_path: str) -> Optional[RelationshipProvider]:
        return None
