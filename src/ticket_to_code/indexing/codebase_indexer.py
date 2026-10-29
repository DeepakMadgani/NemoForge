"""
Codebase Indexer for RAG

Scans your C# project, chunks code, generates embeddings, and stores in vector database.

Run this ONCE before using the autonomous system to build the RAG index.

Author: Deepak Madgani
Date: April 2026
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import hashlib

from ticket_to_code.runtime.vector_store import VectorStoreManager
from ticket_to_code.runtime.llm_factory import LLMRegistry

from ticket_to_code.models import CodeChunk

# NOTE: This is the simple text-based chunker
# For AST-based parsing with symbols/edges, see aviator-platform/aviator_core/indexer.py

logger = logging.getLogger(__name__)


class CodebaseIndexer:
    """
    Indexes codebase into vector store for RAG retrieval.
    
    Scans project files, chunks code intelligently, generates embeddings,
    and stores in pgvector via Aviator VectorStoreManager.
    """
    
    # File extensions to index
    CODE_EXTENSIONS = {
        '.cs',      # C#
        '.java',    # Java
        '.kt',      # Kotlin
        '.py',      # Python
        '.ts',      # TypeScript
        '.js',      # JavaScript
        '.tsx',     # TypeScript React
        '.jsx',     # JavaScript React
        '.vue',     # Vue SFC
        '.svelte',  # Svelte
        '.go',      # Go
        '.rb',      # Ruby
        '.php',     # PHP
        '.sql',     # SQL
        # Presentation layers. These were absent, which made stylesheets and
        # templates invisible to content retrieval: measured SCSS recall was
        # 0/15 because no .scss file could ever enter the candidate pool, not
        # because they ranked poorly. Styling/markup tickets were unanswerable.
        '.scss',    # Sass (component)
        '.sass',    # Sass (indented)
        '.css',     # CSS
        '.less',    # Less
        '.html',    # Angular / Vue templates
        '.yaml',    # YAML
        '.yml',     # YAML
        '.json',    # JSON (configs)
    }
    
    # Directories to skip
    SKIP_DIRS = {
        'bin', 'obj', 'node_modules', '__pycache__', '.git', 
        '.vs', 'dist', 'build', 'target', 'out', 'Debug', 'Release',
        'packages', '.idea', '.vscode'
    }
    
    def __init__(
        self,
        workspace_path: str,
        collection_name: str = "codebase_chunks"
    ):
        """
        Initialize indexer.
        
        Args:
            workspace_path: Path to codebase to index
            collection_name: Vector store collection name
        """
        self.workspace_path = Path(workspace_path)
        self.collection_name = collection_name
        
        # Initialize Aviator services
        self.vector_store = VectorStoreManager.get_vector_store(
            collection_name=collection_name
        )
        self.llm = LLMRegistry.get_llm()
        
        logger.info(f"Indexer initialized for: {self.workspace_path}")
    
    def index_codebase(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Index entire codebase into vector store.
        
        Args:
            force_reindex: If True, clears existing index and rebuilds
            
        Returns:
            Statistics: files indexed, chunks created, etc.
        """
        logger.info("🔍 Starting codebase indexing...")
        
        if force_reindex:
            logger.info("Clearing existing index...")
            self._clear_index()
        
        # Scan for code files
        code_files = self._scan_codebase()
        logger.info(f"Found {len(code_files)} code files to index")
        
        # Process each file
        total_chunks = 0
        indexed_files = 0
        
        for file_path in code_files:
            try:
                chunks = self._process_file(file_path)
                if chunks:
                    self._store_chunks(chunks)
                    total_chunks += len(chunks)
                    indexed_files += 1
                    
                    if indexed_files % 10 == 0:
                        logger.info(f"Progress: {indexed_files}/{len(code_files)} files")
                        
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        stats = {
            "total_files": len(code_files),
            "indexed_files": indexed_files,
            "total_chunks": total_chunks,
            "collection": self.collection_name,
            "workspace": str(self.workspace_path)
        }
        
        logger.info(
            f"✅ Indexing complete!\n"
            f"   Files: {indexed_files}/{len(code_files)}\n"
            f"   Chunks: {total_chunks}\n"
            f"   Collection: {self.collection_name}"
        )
        
        return stats
    
    def _scan_codebase(self) -> List[Path]:
        """Scan workspace for code files"""
        code_files = []
        
        for file_path in self.workspace_path.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue
            
            # Skip files in excluded directories
            if any(skip_dir in file_path.parts for skip_dir in self.SKIP_DIRS):
                continue
            
            # Check if file extension matches
            if file_path.suffix.lower() in self.CODE_EXTENSIONS:
                code_files.append(file_path)
        
        return code_files
    
    def _process_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Process a single file into chunks.
        
        Args:
            file_path: File to process
            
        Returns:
            List of code chunks
        """
        try:
            # Read file
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Skip empty files
            if not content.strip():
                return []
            
            # Detect language
            language = self._detect_language(file_path)
            
            # Chunk the code
            chunks = self._chunk_code(content, file_path, language)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def _chunk_code(
        self, 
        content: str, 
        file_path: Path, 
        language: str
    ) -> List[CodeChunk]:
        """
        Intelligently chunk code into smaller pieces.
        
        Strategy:
        - For C#: Split by class/method boundaries
        - For other languages: Fixed-size chunks with overlap
        """
        chunks = []
        relative_path = str(file_path.relative_to(self.workspace_path))
        
        if language == "csharp":
            # C#-specific chunking (classes, methods)
            chunks = self._chunk_csharp(content, relative_path)
        else:
            # Generic chunking (fixed size with overlap)
            chunks = self._chunk_generic(content, relative_path, language)
        
        return chunks
    
    def _chunk_csharp(self, content: str, file_path: str) -> List[CodeChunk]:
        """
        Chunk C# code by classes and methods.
        
        Simple approach: Split by class/method keywords
        """
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_type = "code"
        chunk_id = 0
        
        for line in lines:
            current_chunk.append(line)
            
            # Check for class/method boundaries
            if 'class ' in line or 'interface ' in line or 'struct ' in line:
                current_type = "class"
            
            # If chunk gets too large (>100 lines), split
            if len(current_chunk) >= 100:
                chunk_content = '\n'.join(current_chunk)
                
                chunks.append(CodeChunk(
                    chunk_id=f"{file_path}:chunk{chunk_id}",
                    code=chunk_content,
                    file_path=file_path,
                    language="csharp",
                    chunk_type=current_type,
                    start_line=max(0, len(chunks) * 100),
                    end_line=min(len(lines), (len(chunks) + 1) * 100),
                    relevance_score=0.0  # Will be calculated during retrieval
                ))
                
                chunk_id += 1
                # Keep last 10 lines for context overlap
                current_chunk = current_chunk[-10:]
        
        # Add remaining chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(CodeChunk(
                chunk_id=f"{file_path}:chunk{chunk_id}",
                code=chunk_content,
                file_path=file_path,
                language="csharp",
                chunk_type=current_type,
                start_line=chunk_id * 100,
                end_line=len(lines),
                relevance_score=0.0
            ))
        
        return chunks
    
    def _chunk_generic(
        self, 
        content: str, 
        file_path: str, 
        language: str
    ) -> List[CodeChunk]:
        """
        Generic chunking: Fixed size with overlap.
        
        Chunk size: 50 lines
        Overlap: 10 lines
        """
        chunks = []
        lines = content.split('\n')
        
        chunk_size = 50
        overlap = 10
        chunk_id = 0
        
        i = 0
        while i < len(lines):
            end = min(i + chunk_size, len(lines))
            chunk_lines = lines[i:end]
            chunk_content = '\n'.join(chunk_lines)
            
            chunks.append(CodeChunk(
                chunk_id=f"{file_path}:chunk{chunk_id}",
                code=chunk_content,
                file_path=file_path,
                language=language,
                chunk_type="code",
                start_line=i,
                end_line=end,
                relevance_score=0.0
            ))
            
            chunk_id += 1
            i += (chunk_size - overlap)
        
        return chunks
    
    def _store_chunks(self, chunks: List[CodeChunk]):
        """
        Store chunks in vector database.
        
        Uses Aviator VectorStoreManager to generate embeddings and store.
        """
        for chunk in chunks:
            # Prepare metadata
            metadata = {
                "file_path": chunk.file_path,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
            
            # Store in vector database (Aviator handles embedding generation)
            self.vector_store.add_document(
                document_id=chunk.chunk_id,
                text=chunk.code,
                metadata=metadata
            )
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        ext = file_path.suffix.lower()
        
        lang_map = {
            '.cs': 'csharp',
            '.java': 'java',
            '.py': 'python',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.go': 'go',
            '.sql': 'sql',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
        }
        
        return lang_map.get(ext, 'unknown')
    
    def _clear_index(self):
        """Clear existing vector store collection"""
        try:
            self.vector_store.clear_collection()
            logger.info("Cleared existing index")
        except Exception as e:
            logger.warning(f"Could not clear index: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def index_codebase(
    workspace_path: str,
    collection_name: str = "codebase_chunks",
    force_reindex: bool = False
) -> Dict[str, Any]:
    """
    Index codebase for RAG retrieval.
    
    Run this ONCE before using the autonomous workflow.
    
    Args:
        workspace_path: Path to your C# project
        collection_name: Vector store collection name
        force_reindex: Rebuild index from scratch
        
    Returns:
        Indexing statistics
        
    Example:
        >>> stats = index_codebase(r"C:\\MyProject\\src")
        >>> print(f"Indexed {stats['total_chunks']} chunks")
    """
    indexer = CodebaseIndexer(workspace_path, collection_name)
    return indexer.index_codebase(force_reindex=force_reindex)
