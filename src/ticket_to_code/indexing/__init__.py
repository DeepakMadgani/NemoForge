"""
Indexing Module - AST Parsing & Code Indexing

Handles parsing source code, extracting symbols/edges, and building indexes.
"""

from ticket_to_code.indexing.codebase_indexer import (
    CodebaseIndexer,
    index_codebase
)
from ticket_to_code.indexing.incremental_indexer import (
    IncrementalIndexer,
    FileChange,
    FileChangeType,
    IndexUpdate,
    update_index_incremental
)

__all__ = [
    'CodebaseIndexer',
    'index_codebase',
    'IncrementalIndexer',
    'FileChange',
    'FileChangeType',
    'IndexUpdate',
    'update_index_incremental',
]
