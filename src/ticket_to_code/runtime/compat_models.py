"""
Standalone Compatibility Models for Ticket-to-Code.

Replaces aviator.models (StateModel, Chunk, EmbeddingRequest) and
aviator_core.models (FileRecord, Symbol, SymbolKind, Edge, EdgeKind,
SourceLocation, IndexStats) with clean Pydantic definitions.

Zero external dependency on Aviator platform.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Replacements for aviator.models
# ═══════════════════════════════════════════════════════════════════════════════

class _Message(BaseModel):
    """Minimal message model matching the interface used by adt_tools.py."""
    type: str = "human"  # "human", "ai", "system"
    content: str = ""


class StateModel(BaseModel):
    """
    Standalone replacement for aviator.models.StateModel.

    Our code accesses `state.messages` (list of message objects with
    `.type` and `.content` attributes). This replica provides exactly that.
    """
    messages: List[_Message] = []

    class Config:
        arbitrary_types_allowed = True


class Chunk(BaseModel):
    """
    Standalone replacement for aviator.models.Chunk.

    Used by aviator_plugin_sample/auth.py for permission filtering.
    """
    content: str = ""
    metadata: Dict[str, Any] = {}
    id: Optional[str] = None


class EmbeddingRequest(BaseModel):
    """
    Standalone replacement for aviator.models.EmbeddingRequest.

    Used by aviator_plugin_sample/tasks.py and extensions.py.
    """
    text: str = ""
    metadata: Dict[str, Any] = {}
    schema_name: str = "default"


# ═══════════════════════════════════════════════════════════════════════════════
# Replacements for aviator_core.models
# ═══════════════════════════════════════════════════════════════════════════════

class SymbolKind(str, Enum):
    """Kinds of structural elements we extract from source files."""
    FILE = "file"
    PACKAGE = "package"
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"
    RECORD = "record"
    ANNOTATION_TYPE = "annotation_type"
    METHOD = "method"
    FUNCTION = "function"
    CONSTRUCTOR = "constructor"
    FIELD = "field"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    IMPORT = "import"
    UNKNOWN = "unknown"


class EdgeKind(str, Enum):
    """Kinds of directed relationships between symbols."""
    CONTAINS = "contains"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    REFERENCES = "references"
    IMPORTS = "imports"
    ANNOTATED_BY = "annotated_by"
    HAS_TYPE = "has_type"
    USES = "uses"
    TEMPLATE_OF = "template_of"
    STYLE_OF = "style_of"


class SourceLocation(BaseModel):
    """1-based location range inside a source file."""
    path: str = Field(..., description="Repo-relative POSIX path.")
    start_line: int = 1
    start_col: int = 1
    end_line: int = 1
    end_col: int = 1


class Symbol(BaseModel):
    """A single structural element extracted from a source file."""
    id: str
    kind: SymbolKind
    name: str
    qualified_name: str = Field(..., description="Fully-qualified name")
    package: Optional[str] = None
    parent_id: Optional[str] = None
    location: SourceLocation
    signature: Optional[str] = None
    modifiers: list[str] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    return_type: Optional[str] = None
    parameter_types: list[str] = Field(default_factory=list)
    doc: Optional[str] = None
    
    # Spring Framework intelligence
    spring_stereotype: Optional[str] = None
    spring_endpoints: list[str] = Field(default_factory=list)
    spring_dependencies: list[str] = Field(default_factory=list)
    is_feign_client: bool = False
    feign_service_name: Optional[str] = None

    model_config = {"frozen": False}


class Edge(BaseModel):
    """A directed relationship between two symbols."""
    kind: EdgeKind
    src_id: str
    dst_id: Optional[str] = None
    dst_name: str
    location: Optional[SourceLocation] = None


class FileRecord(BaseModel):
    """Metadata about a single indexed source file."""
    path: str
    language: str = "java"
    package: Optional[str] = None
    sha256: str
    size_bytes: int
    parse_ok: bool = True
    parse_error: Optional[str] = None


class IndexStats(BaseModel):
    """Summary returned by the indexer."""
    files_scanned: int = 0
    files_parsed: int = 0
    files_failed: int = 0
    symbols: int = 0
    edges: int = 0
    resolved_calls: int = 0
    duration_seconds: float = 0.0


class AgentState(BaseModel):
    """General agent state model."""
    active_file: Optional[str] = None
    iteration: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class WorkflowState(BaseModel):
    """General workflow state model."""
    status: str = "initialized"
    current_node: Optional[str] = None
    step_history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
