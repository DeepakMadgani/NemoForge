"""
Artifact Generator Agent

Handles non-source-code artifacts (CONFIG, SCRIPT, GENERIC_TEXT) routed by
GenerationRouter. Per the Phase 3A.4A routing design, this delegates
to the same robust generation logic as CodeGeneratorAgent, ensuring full
implementation state, cross-file context, and validation safety nets.
"""

from ticket_to_code.agents.code_generator import CodeGeneratorAgent


class ArtifactGeneratorAgent(CodeGeneratorAgent):
    """Generates non-source-code artifacts with full contextual awareness and safety checks."""
    pass
