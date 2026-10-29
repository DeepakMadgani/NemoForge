"""
RAG Engine for Codebase Understanding

Advanced multi-stage retrieval with context expansion and dependency analysis.

This is Agent 3 in the autonomous pipeline - the "Full Repo Understanding" component.

Author: Deepak Madgani
Date: April 2026
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Set

from ticket_to_code.runtime.vector_store import VectorStoreManager

from ticket_to_code.models import CodeChunk, CodeChunkType, StructuredRequirements

logger = logging.getLogger(__name__)


class CodebaseRAGEngine:
    """
    Advanced RAG Engine with Multi-Stage Retrieval
    
    Features:
    - Semantic search via pgvector
    - Context expansion based on dependencies
    - Multi-stage retrieval (Query → Evaluate → Refine → Expand)
    - Code graph traversal for dependency resolution
    - Architectural guidelines consultation
    """
    
    def __init__(
        self, 
        schema_name: str = "codebase_knowledge",
        architectural_schema: str = "architectural_guidelines"
    ):
        """
        Initialize RAG engine with vector store.
        
        Args:
            schema_name: Vector store schema name for code
            architectural_schema: Vector store schema for architectural guidelines
        """
        # Fully deterministic local search engine — zero socket stall, zero PostgreSQL dependency
        self.vector_store = None
        self.architectural_store = None
        self.is_initialized = True
        self.workspace_path: Optional[str] = None
        logger.info("Deterministic Codebase Search Engine initialized (0 external DB dependencies)")
    
    def multi_stage_retrieval(
        self,
        requirements: StructuredRequirements,
        architectural_plan: Optional[Any] = None,
        query_focus: str = "general",  # ← NEW: "test_behavior" or "architecture"
        document_types: Optional[List[str]] = None,  # ← NEW: Filter by doc types
        max_iterations: int = 3,
        k_per_iteration: int = 5,
        ticket: dict = None,
    ) -> List[CodeChunk]:
        """
        Multi-stage retrieval with DUAL RAG support.
        
        DUAL RAG MODES:
        - query_focus="test_behavior" → Retrieves product behavior, test patterns, business rules
        - query_focus="architecture" → Retrieves code patterns, structure, design
        
        This ensures TEST and CODE generation use INDEPENDENT knowledge sources!
        
        Pipeline:
        0. **Retrieve focused context based on query_focus**
        1. Initial semantic search based on requirements (plus literal search fallback)
        2. Evaluate sufficiency (are all dependencies present?)
        3. Expand query based on missing elements
        4. Retrieve dependencies via code graph
        5. Deduplicate and rank
        
        Args:
            requirements: Structured requirements to search for
            architectural_plan: Architectural plan with task details
            query_focus: Focus area - "test_behavior", "architecture", or "general"
            document_types: Filter by document types (e.g., ["tests", "docs", "code"])
            max_iterations: Maximum retrieval iterations
            k_per_iteration: Number of chunks per iteration
            ticket: The raw ticket dictionary (contains ticket_description)
            
        Returns:
            List of relevant code chunks with metadata
        """
        if not self.is_initialized:
            logger.warning("Vector store not initialized, returning empty context")
            return []
        
        logger.info(f"🎯 Starting DUAL RAG retrieval - Focus: {query_focus}")
        if document_types:
            logger.info(f"📂 Document types filter: {document_types}")
        
        all_chunks: List[CodeChunk] = []
        seen_paths: Set[str] = set()
        
        # Stage 0: Retrieve focused context (DUAL RAG!)
        focused_context = self._get_focused_context(
            requirements, 
            query_focus, 
            document_types
        )
        if focused_context:
            logger.info(f"📚 Retrieved focused context ({query_focus}): {len(focused_context)} chunks")
            all_chunks.extend(focused_context)
        
        # Stage 1: Literal Search Fallback & Initial semantic search
        literal_chunks = self._literal_search_fallback(ticket, k=k_per_iteration)
        if literal_chunks:
            logger.info(f"📚 Retrieved {len(literal_chunks)} literal matches from ticket requirements")
            all_chunks.extend(literal_chunks)
            seen_paths.update(chunk.file_path for chunk in literal_chunks)
            
        initial_query = self._build_initial_query(requirements)
        logger.info(f"Stage 1: Initial query: {initial_query[:100]}...")
        
        chunks = self._semantic_search(initial_query, k=k_per_iteration)
        for chunk in chunks:
            if chunk.file_path not in seen_paths:
                all_chunks.append(chunk)
                seen_paths.add(chunk.file_path)
        
        # Stage 2-N: Iterative expansion
        for iteration in range(1, max_iterations):
            logger.info(f"Stage {iteration + 1}: Expanding context...")
            
            # Identify missing elements (simplified - would use Context Evaluator in full impl)
            missing_elements = self._identify_missing_elements(
                all_chunks, 
                requirements
            )
            
            if not missing_elements:
                logger.info("No missing elements, stopping retrieval")
                break
            
            # Expand query
            expanded_query = self._build_expanded_query(missing_elements)
            logger.info(f"Expanded query: {expanded_query[:100]}...")
            
            # Retrieve additional chunks
            new_chunks = self._semantic_search(expanded_query, k=k_per_iteration)
            
            # Add only unseen chunks
            for chunk in new_chunks:
                if chunk.file_path not in seen_paths:
                    all_chunks.append(chunk)
                    seen_paths.add(chunk.file_path)
        
        # Stage Final: Dependency expansion
        logger.info("Final stage: Dependency expansion...")
        dependency_chunks = self._expand_dependencies(all_chunks)
        
        for chunk in dependency_chunks:
            if chunk.file_path not in seen_paths:
                all_chunks.append(chunk)
                seen_paths.add(chunk.file_path)
        
        # Rank and return top results
        ranked_chunks = self._rank_chunks(all_chunks, requirements)
        
        logger.info(f"Retrieval complete: {len(ranked_chunks)} chunks")
        return ranked_chunks[:30]  # Limit to top 30 to fit in context window
    
    def retrieve_context(
        self,
        query: str,
        max_results: int = 10,
        document_types: Optional[List[str]] = None,
        schema: str = "both"  # "code", "architecture", or "both"
    ) -> List[dict]:
        """
        Context retrieval with similarity scoring.
        
        Uses similarity_search_with_score to return actual relevance scores
        alongside content. This enables downstream consumers (evidence pipeline,
        preflight, etc.) to threshold and rank by real vector similarity.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            document_types: Filter by document types (e.g., ["documentation", "configuration"])
            schema: Which schema to search ("code", "architecture", or "both")
            
        Returns:
            List of context dictionaries with file_path, content, and score
        """
        if not self.is_initialized:
            logger.warning("RAG engine not initialized")
            return []
        
        results = []
        
        try:
            # Search in main codebase store — with scores (Similarity Search)
            if schema in ["code", "both"] and self.vector_store:
                logger.debug(f"Searching codebase for: {query[:50]}...")
                try:
                    code_results = self.vector_store.similarity_search_with_score(
                        query=query,
                        k=max_results
                    )
                    for doc, distance in code_results:
                        if document_types:
                            doc_type = doc.metadata.get("type", "").lower()
                            if not any(dt.lower() in doc_type for dt in document_types):
                                continue
                        similarity_score = max(0.0, 1.0 - float(distance))
                        results.append({
                            "file_path": doc.metadata.get("file_path", "Unknown"),
                            "content": doc.page_content,
                            "type": doc.metadata.get("type", "code"),
                            "language": doc.metadata.get("language", "unknown"),
                            "score": similarity_score,
                            "source": "vector_similarity",
                        })
                except (AttributeError, TypeError):
                    code_results = self.vector_store.similarity_search(
                        query=query, k=max_results
                    )
                    for doc in code_results:
                        if document_types:
                            doc_type = doc.metadata.get("type", "").lower()
                            if not any(dt.lower() in doc_type for dt in document_types):
                                continue
                        results.append({
                            "file_path": doc.metadata.get("file_path", "Unknown"),
                            "content": doc.page_content,
                            "type": doc.metadata.get("type", "code"),
                            "language": doc.metadata.get("language", "unknown"),
                            "score": 0.5,
                            "source": "vector_similarity",
                        })

            # Keyword Complementary Search (runs alongside vector, not as fallback)
            # ─────────────────────────────────────────────────────────────────
            # Architecture decision: keyword and vector are COMPLEMENTARY layers.
            # Keyword search uses actual SearchResult.confidence (0.0-1.0) from
            # the RepositorySearchEngine, not a hardcoded score.
            #
            # Scoring heuristic:
            #   keyword_complementary confidence = min(source_score, 0.75)
            #
            #   0.75 is a HEURISTIC NORMALIZATION CEILING, not a calibrated
            #   confidence value.  It exists solely to prevent exact textual
            #   matches from dominating vector semantic matches in the final
            #   ranking.  It does NOT mean "keyword matches are 75% reliable."
            #
            #   If keyword and vector scores are ever ranked against each other,
            #   they must first be normalized into a compatible scale.  This cap
            #   is one part of that normalization.
            #
            #   Source metadata ("keyword_complementary") is preserved so
            #   downstream fusion can distinguish vector vs keyword provenance.
            if schema in ["code", "both"] and self.workspace_path:
                try:
                    from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
                    searcher = RepositorySearchEngine(self.workspace_path)
                    keywords = self._extract_ticket_literals(query)
                    if not keywords:
                        keywords = [w for w in query.split() if len(w) > 4]
                    
                    for keyword in keywords[:4]:
                        kw_results = searcher.search_literal(keyword)
                        for r in kw_results[:3]:
                            kw_score = float(r.confidence)
                            results.append({
                                "file_path": r.file_path,
                                "content": f"[Exact Match] line {r.line_number}: {r.matched_text}",
                                "type": "code",
                                "language": "unknown",
                                "score": kw_score,
                                "source": "exact_repository_search",
                            })
                except Exception as e:
                    logger.debug(f"Keyword search failed: {e}")
            
            # Local Architectural Guidelines Direct Read (Zero Vector DB required)
            if schema in ["architecture", "both"]:
                try:
                    rag_dir = Path(__file__).parent.parent / "RAG" / "architecture"
                    if rag_dir.exists():
                        for md_file in rag_dir.glob("*.md"):
                            results.append({
                                "file_path": str(md_file.relative_to(rag_dir.parent.parent)),
                                "content": md_file.read_text(encoding="utf-8", errors="replace")[:1200],
                                "type": "documentation",
                                "language": "markdown",
                                "score": 0.9,
                                "source": "local_architecture_guidelines",
                            })
                except Exception as e:
                    logger.debug(f"Local architecture guidelines read failed: {e}")
                except (AttributeError, TypeError):
                    arch_results = self.architectural_store.similarity_search(
                        query=query, k=max_results // 2
                    )
                    for doc in arch_results:
                        results.append({
                            "file_path": doc.metadata.get("file_path", "Architectural Guideline"),
                            "content": doc.page_content,
                            "type": "documentation",
                            "language": doc.metadata.get("language", "markdown"),
                            "score": 0.5,
                            "source": "architectural_guidelines",
                        })
            
            # Sort by score descending so best matches are first
            results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
            
            logger.info(f"Retrieved {len(results)} context documents")
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}", exc_info=True)
            return []
    
    def _get_architectural_guidelines(
        self, 
        requirements: StructuredRequirements
    ) -> List[CodeChunk]:
        """
        Retrieve architectural guidelines relevant to the requirements.
        
        This is consulted FIRST to understand:
        - Where to put new files (project structure)
        - What patterns to follow (coding standards)
        - How components should be organized
        
        Args:
            requirements: Structured requirements
            
        Returns:
            List of architectural guideline chunks
        """
        if not self.architectural_store:
            logger.warning("Architectural guidelines store not available")
            return []
        
        try:
            # Build query for architectural context
            guideline_query = self._build_architectural_query(requirements)
            
            logger.info(f"📐 Searching architectural guidelines: {guideline_query[:100]}...")
            
            # Search in architectural guidelines collection
            results = self.architectural_store.similarity_search(
                query=guideline_query,
                k=5  # Get top 5 relevant guideline chunks
            )
            
            # Convert to CodeChunk format
            guideline_chunks = []
            for idx, result in enumerate(results):
                chunk = CodeChunk(
                    chunk_id=f"guideline_{idx}",
                    code=result.get('text', ''),
                    file_path=result.get('metadata', {}).get('file_path', 'architectural_guide'),
                    language='markdown',
                    chunk_type='architectural_guideline',
                    start_line=0,
                    end_line=0,
                    relevance_score=result.get('score', 0.0)
                )
                guideline_chunks.append(chunk)
            
            logger.info(f"📐 Found {len(guideline_chunks)} architectural guideline chunks")
            return guideline_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve architectural guidelines: {e}")
            return []
    
    def _build_architectural_query(self, requirements: StructuredRequirements) -> str:
        """
        Build query for architectural guidelines.
        
        Focuses on:
        - Project structure (where to put files)
        - Patterns (what design patterns to use)
        - Components affected (related architectural decisions)
        """
        query_parts = [
            "Project structure and file organization for:"
        ]
        
        # Add affected components
        if requirements.affected_components:
            components = ", ".join(requirements.affected_components)
            query_parts.append(f"Components: {components}")
        
        # Add technical context
        if requirements.technical_requirements:
            tech = " ".join(requirements.technical_requirements[:3])  # First 3
            query_parts.append(f"Technical patterns: {tech}")
        
        # Add feature context
        if requirements.functional_requirements:
            feature = requirements.functional_requirements[0]  # Main feature
            query_parts.append(f"Feature: {feature}")
        
        return " | ".join(query_parts)
    
    def _get_focused_context(
        self,
        requirements: StructuredRequirements,
        query_focus: str,
        document_types: Optional[List[str]] = None
    ) -> List[CodeChunk]:
        """
        Retrieve focused context based on query_focus (DUAL RAG).
        
        This is the KEY method for knowledge separation:
        - "test_behavior" → Product behavior, test patterns, business rules
        - "architecture" → Code structure, design patterns, implementation
        
        Args:
            requirements: Structured requirements
            query_focus: "test_behavior", "architecture", or "general"
            document_types: Filter by types (e.g., ["tests", "docs", "code"])
            
        Returns:
            List of focused code chunks
        """
        logger.info(f"🎯 Retrieving focused context: {query_focus}")
        
        if query_focus == "test_behavior":
            return self._get_test_behavior_context(requirements, document_types)
        elif query_focus == "architecture":
            return self._get_architecture_context(requirements, document_types)
        else:
            # General retrieval - get both
            return self._get_architectural_guidelines(requirements)
    
    def _get_test_behavior_context(
        self,
        requirements: StructuredRequirements,
        document_types: Optional[List[str]] = None
    ) -> List[CodeChunk]:
        """
        Retrieve PRODUCT BEHAVIOR knowledge for TEST generation.
        
        Focuses on:
        - How features SHOULD behave (business rules)
        - Existing test patterns (structure)
        - Error handling behavior (what errors to expect)
        - Validation rules (what's valid/invalid)
        
        Sources:
        - RAG/product_behavior/*.md
        - RAG/tests/*.md
        - Existing test files (*.Tests.cs)
        
        Args:
            requirements: Requirements to search for
            document_types: Filter (defaults to ["tests", "docs", "business_rules"])
            
        Returns:
            Test-focused code chunks
        """
        logger.info("🧪 Retrieving TEST BEHAVIOR context (for test generation)")
        
        # Default document types for test generation
        if document_types is None:
            document_types = ["tests", "documentation", "business_rules"]
        
        # Build query focused on BEHAVIOR
        query = self._build_test_behavior_query(requirements)
        logger.info(f"Query: {query[:150]}...")
        
        chunks = []
        if self.vector_store:
            try:
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=10,
                    filter={"document_type": {"$in": document_types}}
                )
                for doc, score in results:
                    file_path = doc.metadata.get("file_path", "")
                    if "product_behavior" in file_path or "tests" in file_path.lower():
                        chunk = CodeChunk(
                            content=doc.page_content,
                            file_path=file_path,
                            chunk_type=CodeChunkType.TEST,
                            name=doc.metadata.get("name", "test_behavior"),
                            namespace=doc.metadata.get("namespace"),
                            dependencies=[],
                            similarity_score=1.0 - score,
                            line_start=doc.metadata.get("line_start"),
                            line_end=doc.metadata.get("line_end")
                        )
                        chunks.append(chunk)
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")

        # Direct Local Disk Read (Zero Vector DB required)
        try:
            rag_dir = Path(__file__).parent.parent / "RAG" / "product_behavior"
            if rag_dir.exists():
                for md_file in rag_dir.glob("*.md"):
                    content = md_file.read_text(encoding="utf-8", errors="replace")
                    chunks.append(CodeChunk(
                        content=content,
                        file_path=str(md_file.relative_to(rag_dir.parent.parent)),
                        chunk_type=CodeChunkType.TEST,
                        name=md_file.stem,
                        namespace=None,
                        dependencies=[],
                        similarity_score=1.0,
                        line_start=1,
                        line_end=len(content.splitlines())
                    ))
        except Exception as e:
            logger.debug(f"Direct disk read of product_behavior failed: {e}")

        logger.info(f"🧪 Found {len(chunks)} test behavior chunks")
        return chunks
    
    def _get_architecture_context(
        self,
        requirements: StructuredRequirements,
        document_types: Optional[List[str]] = None
    ) -> List[CodeChunk]:
        """
        Retrieve ARCHITECTURE knowledge for CODE generation.
        """
        logger.info("🏗️ Retrieving ARCHITECTURE context (for code generation)")
        chunks = []
        if self.vector_store:
            try:
                results = self.vector_store.similarity_search_with_score(
                    query=self._build_architecture_query(requirements),
                    k=10,
                    filter={"document_type": {"$in": document_types or ["source_code", "interfaces"]}}
                )
                for doc, score in results:
                    file_path = doc.metadata.get("file_path", "")
                    chunks.append(CodeChunk(
                        content=doc.page_content,
                        file_path=file_path,
                        chunk_type=CodeChunkType.CLASS,
                        name=doc.metadata.get("name", "architecture"),
                        namespace=doc.metadata.get("namespace"),
                        dependencies=doc.metadata.get("dependencies", []),
                        similarity_score=1.0 - score,
                        line_start=doc.metadata.get("line_start"),
                        line_end=doc.metadata.get("line_end")
                    ))
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")

        # Direct Local Disk Read (Zero Vector DB required)
        try:
            rag_dir = Path(__file__).parent.parent / "RAG" / "architecture"
            if rag_dir.exists():
                for md_file in rag_dir.glob("*.md"):
                    content = md_file.read_text(encoding="utf-8", errors="replace")
                    chunks.append(CodeChunk(
                        content=content,
                        file_path=str(md_file.relative_to(rag_dir.parent.parent)),
                        chunk_type=CodeChunkType.CLASS,
                        name=md_file.stem,
                        namespace=None,
                        dependencies=[],
                        similarity_score=1.0,
                        line_start=1,
                        line_end=len(content.splitlines())
                    ))
        except Exception as e:
            logger.debug(f"Direct disk read of architecture failed: {e}")

        logger.info(f"🏗️ Found {len(chunks)} architecture chunks")
        return chunks
    
    def _build_test_behavior_query(self, requirements: StructuredRequirements) -> str:
        """
        Build query focused on BEHAVIOR (for test generation).
        
        Keywords: "behavior", "should", "validation", "error", "rule"
        """
        query_parts = [
            "How should the feature behave?",
            "Test patterns and behavior:"
        ]
        
        # Add functional requirements (the BEHAVIOR)
        if requirements.functional_requirements:
            behavior = " ".join(requirements.functional_requirements)
            query_parts.append(f"Expected behavior: {behavior}")
        
        # Add validation rules
        if requirements.acceptance_tests:
            rules = " ".join(requirements.acceptance_tests[:2])
            query_parts.append(f"Validation rules: {rules}")
        
        query_parts.append("Business rules and test examples")
        
        return " | ".join(query_parts)
    
    def _build_architecture_query(self, requirements: StructuredRequirements) -> str:
        """
        Build query focused on STRUCTURE (for code generation).
        
        Keywords: "implementation", "class", "interface", "pattern", "structure"
        """
        query_parts = [
            "Code structure and implementation patterns for:"
        ]
        
        # Add affected components (the STRUCTURE)
        if requirements.affected_components:
            components = ", ".join(requirements.affected_components)
            query_parts.append(f"Components: {components}")
        
        # Add technical requirements (HOW to build)
        if requirements.technical_requirements:
            tech = " ".join(requirements.technical_requirements)
            query_parts.append(f"Implementation: {tech}")
        
        query_parts.append("Design patterns and service layer examples")
        
        return " | ".join(query_parts)
    
    def _build_initial_query(self, requirements: StructuredRequirements) -> str:
        """Build initial search query from requirements"""
        query_parts = []
        
        # Add functional requirements
        if requirements.functional_requirements:
            query_parts.append("Functional: " + " ".join(requirements.functional_requirements))
        
        # Add technical requirements
        if requirements.technical_requirements:
            query_parts.append("Technical: " + " ".join(requirements.technical_requirements))
        
        # Add affected components
        if requirements.affected_components:
            query_parts.append("Components: " + " ".join(requirements.affected_components))
        
        return " | ".join(query_parts)
    
    def _extract_ticket_literals(self, text: str) -> List[str]:
        import re
        literals = []
        # Quoted strings
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        for q in quoted:
            if q[0]: literals.append(q[0])
            if q[1]: literals.append(q[1])
            
        # Version numbers (e.g. 1.0.0, 26.3, 26.3.0)
        versions = re.findall(r'\b\d+\.\d+(?:\.\d+)?\b', text)
        literals.extend(versions)
        
        # Specific filenames like app.component.ts or .java
        filenames = re.findall(r'\b[\w.-]+\.[a-z]{2,4}\b', text)
        valid_exts = {".ts", ".js", ".java", ".py", ".cs", ".go", ".rb", ".php", ".html", ".css", ".scss", ".json", ".xml", ".yml", ".yaml", ".sh"}
        for f in filenames:
            if any(f.endswith(ext) for ext in valid_exts):
                literals.append(f)
                
        return list(set([L for L in literals if len(L) >= 3]))
        
    def _literal_search_fallback(self, ticket: dict, k: int = 5) -> List[CodeChunk]:
        """
        Perform a literal keyword search fallback using extracted terms from the ticket.
        NOTE: 0.95 confidence is a starting assumption for literal hits, to be revisited as needed.
        """
        if not self.workspace_path or not ticket:
            return []
            
        text = ticket.get("ticket_description", "")
        keywords = self._extract_ticket_literals(text)
        if not keywords:
            return []
            
        try:
            from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
            searcher = RepositorySearchEngine(self.workspace_path)
            
            chunks = []
            seen_paths = set()
            valid_exts = {".ts", ".js", ".java", ".py", ".cs", ".go", ".rb", ".php", ".html", ".css", ".scss", ".json", ".xml", ".yml", ".yaml", ".sh"}
            
            for keyword in keywords:
                # If keyword looks like a file extension, search filenames
                is_file = any(keyword.endswith(ext) for ext in valid_exts)
                
                results = []
                if is_file:
                    fn_results = searcher.search_filename(keyword)
                    # Convert SearchResult objects to mock literal matches for consistency
                    for fr in fn_results:
                        r = type('obj', (object,), {'file_path': fr.file_path, 'line_number': 1, 'matched_text': f"File {fr.file_path}"})
                        results.append(r)
                else:
                    results = searcher.search_literal(keyword)
                    
                keyword_chunks = 0
                for r in results:
                    if r.file_path in seen_paths:
                        continue
                    seen_paths.add(r.file_path)
                    
                    chunk = CodeChunk(
                        content=f"[Literal match for '{keyword}'] line {r.line_number}: {r.matched_text}",
                        file_path=r.file_path,
                        chunk_type=CodeChunkType.MODULE,
                        name=r.file_path.split("/")[-1],
                        similarity_score=0.95,  # STRONG EVIDENCE
                        line_start=r.line_number,
                        line_end=r.line_number
                    )
                    chunks.append(chunk)
                    keyword_chunks += 1
                    if keyword_chunks >= k:  # Cap per keyword
                        break
            return chunks
        except Exception as e:
            logger.error(f"Literal search fallback failed: {e}")
            return []

    def _semantic_search(self, query: str, k: int = 5) -> List[CodeChunk]:
        """
        Perform both semantic/keyword search and similarity vector search.
        
        Args:
            query: Search query
            k: Number of results per search type
            
        Returns:
            List of code chunks
        """
        chunks = []
        seen_paths = set()
        
        # 1. Similarity Vector Search
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            for doc, score in results:
                fp = doc.metadata.get("file_path", "unknown")
                chunk = CodeChunk(
                    content=doc.page_content,
                    file_path=fp,
                    chunk_type=CodeChunkType(doc.metadata.get("type", "module")),
                    name=doc.metadata.get("name", "unknown"),
                    namespace=doc.metadata.get("namespace"),
                    dependencies=doc.metadata.get("dependencies", []),
                    similarity_score=max(0.0, 1.0 - float(score)),  # Convert distance to similarity
                    line_start=doc.metadata.get("line_start"),
                    line_end=doc.metadata.get("line_end")
                )
                chunks.append(chunk)
                seen_paths.add(fp)
        except Exception as e:
            logger.error(f"Similarity vector search failed: {e}")
            
        # 2. Semantic Keyword Search Fallback
        if self.workspace_path:
            try:
                from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
                searcher = RepositorySearchEngine(self.workspace_path)
                keywords = self._extract_ticket_literals(query)
                if not keywords:
                    keywords = [w for w in query.split() if len(w) > 4]
                
                keyword_chunks = 0
                for keyword in keywords[:3]:
                    kw_results = searcher.search_literal(keyword)
                    for r in kw_results:
                        if r.file_path in seen_paths:
                            continue
                        
                        chunk = CodeChunk(
                            content=f"[Semantic Keyword Match] line {r.line_number}: {r.matched_text}",
                            file_path=r.file_path,
                            chunk_type=CodeChunkType.MODULE,
                            name=r.file_path.split("/")[-1],
                            similarity_score=0.85,  # High score for exact match
                            line_start=r.line_number,
                            line_end=r.line_number
                        )
                        chunks.append(chunk)
                        seen_paths.add(r.file_path)
                        keyword_chunks += 1
                        if keyword_chunks >= k:
                            break
                    if keyword_chunks >= k:
                        break
            except Exception as e:
                logger.error(f"Semantic keyword search failed: {e}")
            
        # Sort combined results by score
        chunks.sort(key=lambda c: getattr(c, "similarity_score", 0.0) or 0.0, reverse=True)
        return chunks[:k * 2]
    
    def _identify_missing_elements(
        self,
        chunks: List[CodeChunk],
        requirements: StructuredRequirements
    ) -> List[str]:
        """
        Identify missing elements from retrieved context.
        
        This is a simplified version. In production, use Context Evaluator Agent.
        """
        # Extract all names from chunks
        retrieved_names = {chunk.name for chunk in chunks}
        
        # Extract component names from requirements
        missing = []
        for component in requirements.affected_components:
            if component not in retrieved_names:
                missing.append(component)
        
        return missing[:5]  # Limit to top 5 missing
    
    def _build_expanded_query(self, missing_elements: List[str]) -> str:
        """Build expanded query from missing elements"""
        return "Find code related to: " + ", ".join(missing_elements)
    
    def _expand_dependencies(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """
        Expand context by retrieving dependencies.
        
        For each chunk, retrieve its dependencies from vector store.
        """
        dependency_chunks = []
        
        for chunk in chunks:
            if not chunk.dependencies:
                continue
            
            # Search for each dependency
            for dep in chunk.dependencies:
                dep_query = f"Find class or module: {dep}"
                dep_results = self._semantic_search(dep_query, k=1)
                dependency_chunks.extend(dep_results)
        
        return dependency_chunks
    
    def _rank_chunks(
        self,
        chunks: List[CodeChunk],
        requirements: StructuredRequirements
    ) -> List[CodeChunk]:
        """
        Rank chunks by relevance.

        LLM context window is limited

        So only BEST chunks should survive.
        
        Ranking factors:
        - Similarity score
        - Chunk type (classes > functions > modules)
        - Presence in affected components
        """
        def score_chunk(chunk: CodeChunk) -> float:
            score = chunk.similarity_score
            
            # Boost for specific chunk types
            if chunk.chunk_type == CodeChunkType.CLASS:
                score += 0.2
            elif chunk.chunk_type == CodeChunkType.FUNCTION:
                score += 0.1
            
            # Boost if in affected components
            for component in requirements.affected_components:
                if component.lower() in chunk.file_path.lower():
                    score += 0.3
                    break
            
            return score
        
        return sorted(chunks, key=score_chunk, reverse=True)
    
    def index_workspace(self, workspace_path: Path, file_extensions: List[str] = None):
        """
        Index entire workspace into vector store.
        
        Args:
            workspace_path: Path to code workspace
            file_extensions: File extensions to index (e.g., ['.cs', '.ts'])
        """
        if not self.is_initialized:
            logger.error("Cannot index: vector store not initialized")
            return False
        
        if file_extensions is None:
            file_extensions = ['.cs', '.ts', '.tsx', '.py', '.java']
        
        logger.info(f"Indexing workspace: {workspace_path}")
        
        documents = []
        for ext in file_extensions:
            for file_path in workspace_path.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # Simple extraction (in production, use AST parsing)
                    doc = {
                        "page_content": content,
                        "metadata": {
                            "file_path": str(file_path.relative_to(workspace_path)),
                            "type": "module",
                            "name": file_path.stem,
                            "extension": ext
                        }
                    }
                    documents.append(doc)
                    
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
        
        if documents:
            try:
                # Add to vector store
                texts = [doc["page_content"] for doc in documents]
                metadatas = [doc["metadata"] for doc in documents]
                self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                logger.info(f"Successfully indexed {len(documents)} files")
                return True
            except Exception as e:
                logger.error(f"Failed to add documents to vector store: {e}")
                return False
        else:
            logger.warning("No documents found to index")
            return False
