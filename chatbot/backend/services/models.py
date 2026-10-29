"""
Data models for ValueEdge tickets
Shared between scraper and API response formats
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ValueEdgeComment(BaseModel):
    """Comment/discussion on a ticket"""
    author: str
    text: str
    created_at: Optional[str] = None
    attachments: List[Dict[str, str]] = Field(default_factory=list)


class TicketContext(BaseModel):
    """
    Complete ticket context with all information
    Used for feeding into the autonomous workflow
    """
    # Core fields
    ticket_id: str
    title: str
    description: str
    priority: str
    status: str
    
    # Optional fields
    assignee: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    
    # Rich content
    comments: List[ValueEdgeComment] = Field(default_factory=list)
    attachments: List[Dict[str, str]] = Field(default_factory=list)
    
    # Metadata
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    
    # Source tracking
    source: str = "unknown"  # "api" or "playwright"
    raw_data: Optional[Dict] = None
    
    def to_prompt(self) -> str:
        """
        Convert ticket to a formatted prompt for LLM
        """
        prompt = f"""
# Ticket: {self.ticket_id}

## Title
{self.title}

## Description
{self.description}

## Priority & Status
- Priority: {self.priority}
- Status: {self.status}
- Assignee: {self.assignee or "Unassigned"}

## Labels
{', '.join(self.labels) if self.labels else "None"}

## Acceptance Criteria
"""
        if self.acceptance_criteria:
            for i, criterion in enumerate(self.acceptance_criteria, 1):
                prompt += f"{i}. {criterion}\n"
        else:
            prompt += "None specified\n"
        
        if self.comments:
            prompt += "\n## Comments & Discussions\n"
            for comment in self.comments:
                prompt += f"**{comment.author}** ({comment.created_at or 'date unknown'}):\n{comment.text}\n\n"
        
        if self.attachments:
            prompt += "\n## Attachments\n"
            for attachment in self.attachments:
                prompt += f"- {attachment.get('filename', 'unknown')} ({attachment.get('url', 'no URL')})\n"
        
        return prompt
