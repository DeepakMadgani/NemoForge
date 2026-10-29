"""
Repository Memory System - Long-Term Engineering Intelligence

System remembers:
- Previous successful fixes
- Previous failures and their causes
- Architectural decisions
- Team coding styles
- Common patterns
- Historical ticket resolutions

This becomes institutional knowledge that improves over time.

Author: Deepak Madgani
Date: May 27, 2026
"""

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of repository memory"""
    SUCCESSFUL_FIX = "successful_fix"
    FAILED_ATTEMPT = "failed_attempt"
    ARCHITECTURAL_DECISION = "architectural_decision"
    CODING_PATTERN = "coding_pattern"
    TEAM_STYLE = "team_style"
    TICKET_RESOLUTION = "ticket_resolution"
    # Enhancement 4: Adaptive Strategy Switching
    FAILED_STRATEGY = "failed_strategy"
    STRATEGY_SWITCH = "strategy_switch"
    # Enhancement 12: Dependency Resolution
    DEPENDENCY_RESOLUTION = "dependency_resolution"


class MemoryEntry(BaseModel):
    """Single memory entry"""
    id: str = Field(..., description="Unique memory ID")
    type: MemoryType = Field(..., description="Type of memory")
    timestamp: datetime = Field(default_factory=datetime.now, description="When recorded")
    
    # Context
    ticket_id: Optional[str] = Field(None, description="Related ticket ID")
    files_involved: List[str] = Field(default_factory=list, description="Files modified")
    
    # Content
    description: str = Field(..., description="What happened")
    approach: str = Field(..., description="Approach taken")
    outcome: str = Field(..., description="What was the result")
    
    # Learnings
    lessons_learned: List[str] = Field(default_factory=list, description="Key learnings")
    do_this: List[str] = Field(default_factory=list, description="What worked well")
    avoid_this: List[str] = Field(default_factory=list, description="What to avoid")
    
    # Metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this memory")
    tags: List[str] = Field(default_factory=list, description="Tags for searching")
    related_memories: List[str] = Field(default_factory=list, description="Related memory IDs")


class RepositoryMemory:
    """
    Repository memory system.
    
    Stores and retrieves institutional knowledge about:
    - What works (successful patterns)
    - What doesn't work (failed attempts)
    - Team preferences (coding style)
    - Architectural decisions (why things are done this way)
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize repository memory.
        
        Args:
            workspace_path: Project root
        """
        self.workspace_path = Path(workspace_path)
        self.memory_dir = self.workspace_path / ".aviator" / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory index
        self.index_file = self.memory_dir / "index.json"
        self.memories: Dict[str, MemoryEntry] = {}
        
        self._load_memories()
        
        logger.info(f"Repository Memory initialized with {len(self.memories)} entries")
    
    def record_successful_fix(
        self,
        ticket_id: str,
        description: str,
        approach: str,
        files_modified: List[str],
        lessons_learned: List[str]
    ) -> MemoryEntry:
        """
        Record a successful fix for future reference.
        
        Args:
            ticket_id: Ticket that was fixed
            description: What was the problem
            approach: How it was fixed
            files_modified: Which files were changed
            lessons_learned: Key learnings
            
        Returns:
            Created memory entry
        """
        memory = MemoryEntry(
            id=self._generate_id(),
            type=MemoryType.SUCCESSFUL_FIX,
            ticket_id=ticket_id,
            files_involved=files_modified,
            description=description,
            approach=approach,
            outcome="Successfully resolved",
            lessons_learned=lessons_learned,
            do_this=[approach],
            tags=["success", "fix", ticket_id]
        )
        
        self._store_memory(memory)
        
        logger.info(f"Recorded successful fix: {memory.id}")
        return memory
    
    def record_failed_attempt(
        self,
        ticket_id: str,
        description: str,
        approach: str,
        error_message: str,
        avoid_list: List[str]
    ) -> MemoryEntry:
        """
        Record a failed attempt to learn from mistakes.
        
        Args:
            ticket_id: Ticket that failed
            description: What was attempted
            approach: Approach that failed
            error_message: Error that occurred
            avoid_list: What to avoid
            
        Returns:
            Created memory entry
        """
        memory = MemoryEntry(
            id=self._generate_id(),
            type=MemoryType.FAILED_ATTEMPT,
            ticket_id=ticket_id,
            description=description,
            approach=approach,
            outcome=f"Failed: {error_message}",
            avoid_this=avoid_list + [approach],
            tags=["failure", "learn", ticket_id]
        )
        
        self._store_memory(memory)
        
        logger.info(f"Recorded failed attempt: {memory.id}")
        return memory
    
    def record_architectural_decision(
        self,
        decision: str,
        rationale: str,
        files_affected: List[str],
        alternatives_considered: List[str]
    ) -> MemoryEntry:
        """
        Record architectural decision for future reference.
        
        Args:
            decision: What was decided
            rationale: Why this decision was made
            files_affected: Which files follow this pattern
            alternatives_considered: Other options that were rejected
            
        Returns:
            Created memory entry
        """
        memory = MemoryEntry(
            id=self._generate_id(),
            type=MemoryType.ARCHITECTURAL_DECISION,
            files_involved=files_affected,
            description=decision,
            approach=rationale,
            outcome="Decision documented",
            lessons_learned=[
                f"Decided: {decision}",
                f"Because: {rationale}"
            ],
            do_this=[decision],
            avoid_this=alternatives_considered,
            tags=["architecture", "decision"]
        )
        
        self._store_memory(memory)
        
        logger.info(f"Recorded architectural decision: {memory.id}")
        return memory
    
    def record_coding_pattern(
        self,
        pattern_name: str,
        description: str,
        example_code: str,
        when_to_use: str,
        files_using_pattern: List[str]
    ) -> MemoryEntry:
        """
        Record coding pattern for consistency.
        
        Args:
            pattern_name: Name of pattern
            description: What this pattern does
            example_code: Example implementation
            when_to_use: When to apply this pattern
            files_using_pattern: Files that use this
            
        Returns:
            Created memory entry
        """
        memory = MemoryEntry(
            id=self._generate_id(),
            type=MemoryType.CODING_PATTERN,
            files_involved=files_using_pattern,
            description=description,
            approach=f"Use {pattern_name} pattern: {example_code}",
            outcome="Pattern established",
            lessons_learned=[when_to_use],
            do_this=[f"Follow {pattern_name} pattern in similar contexts"],
            tags=["pattern", pattern_name]
        )
        
        self._store_memory(memory)
        
        logger.info(f"Recorded coding pattern: {memory.id}")
        return memory
    
    def recall_similar_fixes(
        self,
        description: str,
        files: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[MemoryEntry]:
        """
        Recall similar successful fixes from memory.
        
        Args:
            description: Problem description
            files: Files involved
            limit: Maximum memories to return
            
        Returns:
            List of relevant memories
        """
        logger.info(f"Recalling similar fixes for: {description[:50]}...")
        
        relevant_memories = []
        
        for memory in self.memories.values():
            if memory.type != MemoryType.SUCCESSFUL_FIX:
                continue
            
            # Simple relevance scoring (in production, use embeddings)
            relevance = 0.0
            
            # Check description similarity
            desc_words = set(description.lower().split())
            memory_words = set(memory.description.lower().split())
            common_words = desc_words & memory_words
            if common_words:
                relevance += len(common_words) / max(len(desc_words), 1)
            
            # Check file overlap
            if files and memory.files_involved:
                file_overlap = set(files) & set(memory.files_involved)
                if file_overlap:
                    relevance += len(file_overlap) / max(len(files), 1)
            
            if relevance > 0.1:
                relevant_memories.append((relevance, memory))
        
        # Sort by relevance
        relevant_memories.sort(key=lambda x: x[0], reverse=True)
        
        result = [mem for _, mem in relevant_memories[:limit]]
        
        logger.info(f"Found {len(result)} relevant memories")
        return result
    
    def recall_failures_to_avoid(
        self,
        description: str,
        approach: str
    ) -> List[MemoryEntry]:
        """
        Recall similar failures to avoid repeating mistakes.
        
        Args:
            description: What is being attempted
            approach: Planned approach
            
        Returns:
            List of relevant failure memories
        """
        logger.info(f"Checking for known failures...")
        
        relevant_failures = []
        
        for memory in self.memories.values():
            if memory.type != MemoryType.FAILED_ATTEMPT:
                continue
            
            # Check if similar approach
            if approach.lower() in memory.approach.lower():
                relevant_failures.append(memory)
            
            # Check if similar description
            desc_words = set(description.lower().split())
            memory_words = set(memory.description.lower().split())
            if len(desc_words & memory_words) > 2:
                relevant_failures.append(memory)
        
        logger.info(f"Found {len(relevant_failures)} relevant failures")
        return relevant_failures
    
    def recall_architectural_decisions(
        self,
        files: List[str]
    ) -> List[MemoryEntry]:
        """
        Recall architectural decisions affecting these files.
        
        Args:
            files: Files being modified
            
        Returns:
            List of relevant architectural decisions
        """
        logger.info("Recalling architectural decisions...")
        
        decisions = []
        
        for memory in self.memories.values():
            if memory.type != MemoryType.ARCHITECTURAL_DECISION:
                continue
            
            # Check if any affected files match
            if any(f in memory.files_involved for f in files):
                decisions.append(memory)
        
        logger.info(f"Found {len(decisions)} relevant decisions")
        return decisions
    
    def recall_coding_patterns(
        self,
        files: List[str]
    ) -> List[MemoryEntry]:
        """
        Recall coding patterns used in these files.
        
        Args:
            files: Files being modified
            
        Returns:
            List of relevant coding patterns
        """
        logger.info("Recalling coding patterns...")
        
        patterns = []
        
        for memory in self.memories.values():
            if memory.type != MemoryType.CODING_PATTERN:
                continue
            
            # Check if any files using pattern match
            if any(f in memory.files_involved for f in files):
                patterns.append(memory)
        
        logger.info(f"Found {len(patterns)} relevant patterns")
        return patterns
    
    def get_memory_summary(self) -> Dict[str, int]:
        """Get summary of memories by type"""
        summary = {}
        
        for memory_type in MemoryType:
            count = sum(
                1 for m in self.memories.values() 
                if m.type == memory_type
            )
            summary[memory_type.value] = count
        
        return summary
    
    def _store_memory(self, memory: MemoryEntry):
        """Store memory to disk"""
        self.memories[memory.id] = memory
        
        # Write individual memory file
        memory_file = self.memory_dir / f"{memory.id}.json"
        with open(memory_file, 'w') as f:
            json.dump(memory.dict(), f, indent=2, default=str)
        
        # Update index
        self._update_index()
    
    def _load_memories(self):
        """Load all memories from disk"""
        if not self.memory_dir.exists():
            return
        
        for memory_file in self.memory_dir.glob("*.json"):
            if memory_file.name == "index.json":
                continue
            
            try:
                with open(memory_file) as f:
                    data = json.load(f)
                    memory = MemoryEntry(**data)
                    self.memories[memory.id] = memory
            except Exception as e:
                logger.error(f"Failed to load memory {memory_file}: {e}")
    
    def _update_index(self):
        """Update memory index"""
        index = {
            "total_memories": len(self.memories),
            "by_type": self.get_memory_summary(),
            "last_updated": datetime.now().isoformat(),
            "memory_ids": list(self.memories.keys())
        }
        
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _generate_id(self) -> str:
        """Generate unique memory ID"""
        import uuid
        return f"mem-{uuid.uuid4().hex[:12]}"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def recall_knowledge(
    workspace_path: str,
    description: str,
    files: List[str]
) -> Dict[str, List[MemoryEntry]]:
    """
    Recall all relevant knowledge for a task.
    
    Args:
        workspace_path: Project root
        description: Task description
        files: Files being modified
        
    Returns:
        Dict with successful_fixes, failures, decisions, patterns
    """
    memory = RepositoryMemory(workspace_path)
    
    return {
        "successful_fixes": memory.recall_similar_fixes(description, files),
        "failures_to_avoid": memory.recall_failures_to_avoid(description, ""),
        "architectural_decisions": memory.recall_architectural_decisions(files),
        "coding_patterns": memory.recall_coding_patterns(files)
    }
