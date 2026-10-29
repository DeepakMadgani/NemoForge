import logging
from typing import Any, Optional
from ticket_to_code.models import ArtifactType, DevelopmentTask
from ticket_to_code.agents.code_generator import CodeGeneratorAgent
from ticket_to_code.agents.artifact_generator import ArtifactGeneratorAgent

logger = logging.getLogger(__name__)

class GenerationRouter:
    """
    Routes tasks to specialized generators based on their ArtifactType.
    For Phase 3A.4A, this is a foundational routing layer that directs all
    traffic to the existing CodeGeneratorAgent. Future phases will implement
    and map specialized generators.
    """
    
    def __init__(self, workspace_path: str, llm: Optional[Any] = None):
        # Initialize the existing code generator for SOURCE_CODE
        self.code_generator = CodeGeneratorAgent(workspace_path=workspace_path, llm=llm)
        # Initialize the new artifact generator for unstructured files
        self.artifact_generator = ArtifactGeneratorAgent()
        
    def route(self, task: DevelopmentTask):
        """
        Select the appropriate generator for the given task.
        """
        # Templates (.html, .htm) must always route to CodeGeneratorAgent
        # for ComponentContract and companion controller binding validation
        if hasattr(task, "file_path") and task.file_path:
            fp_lower = str(task.file_path).lower()
            if fp_lower.endswith((".html", ".htm", ".component.html")):
                return self.code_generator

        artifact_type = task.artifact_type
        
        if artifact_type == ArtifactType.SOURCE_CODE:
            return self.code_generator
        elif artifact_type == ArtifactType.STYLE:
            return self.code_generator
        elif artifact_type == ArtifactType.CONFIG:
            return self.artifact_generator
        elif artifact_type == ArtifactType.SCRIPT:
            return self.artifact_generator
        elif artifact_type == ArtifactType.SQL:
            return self.code_generator
        elif artifact_type == ArtifactType.GENERIC_TEXT:
            return self.artifact_generator
        else:
            return self.code_generator

