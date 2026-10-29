"""
ValueEdge API Client
Primary method for fetching ticket information using official APIs
"""

import httpx
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ValueEdgeComment(BaseModel):
    """Comment on a ticket"""
    author: str
    text: str
    created_at: datetime
    attachments: List[str] = []


class ValueEdgeAttachment(BaseModel):
    """File attachment"""
    filename: str
    url: str
    size: Optional[int] = None
    mime_type: Optional[str] = None


class LinkedTicket(BaseModel):
    """Linked/related ticket"""
    ticket_id: str
    title: str
    relationship: str  # "blocks", "blocked_by", "relates_to", etc.


class TicketContext(BaseModel):
    """
    Complete ticket context for autonomous workflow
    All information needed to understand and solve the ticket
    """
    # Basic Info
    ticket_id: str
    title: str
    description: str
    
    # Status & Priority
    priority: str
    status: str
    type: str  # bug, feature, task, etc.
    
    # Assignment
    assignee: Optional[str] = None
    reporter: Optional[str] = None
    team: Optional[str] = None
    
    # Requirements
    acceptance_criteria: List[str] = []
    labels: List[str] = []
    
    # Rich Context
    comments: List[ValueEdgeComment] = []
    attachments: List[ValueEdgeAttachment] = []
    linked_tickets: List[LinkedTicket] = []
    
    # Metadata
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    
    # Effort Tracking
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    story_points: Optional[int] = None
    
    # Screenshots/Logs (for debugging tickets)
    screenshots: List[str] = []
    logs: List[str] = []
    
    # Technical Context
    affected_components: List[str] = []
    affected_versions: List[str] = []
    environment: Optional[str] = None  # dev, staging, prod
    
    # Raw Data
    raw_data: Optional[Dict[str, Any]] = None
    
    # Source tracking
    source: Optional[str] = None  # "api" or "playwright"


class ValueEdgeAPIClient:
    """
    Official ValueEdge API Client
    
    Supports:
    - REST API
    - GraphQL API (if available)
    - OAuth/API Key authentication
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize ValueEdge API client
        
        Args:
            base_url: ValueEdge API base URL
            api_key: API key for authentication (preferred)
            username: Username for basic auth (fallback)
            password: Password for basic auth (fallback)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.username = username
        self.password = password
        self.timeout = timeout
        
        # Setup HTTP client
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if api_key:
            # API Key authentication (adjust header name based on your ValueEdge setup)
            headers["Authorization"] = f"Bearer {api_key}"
            # Alternative: headers["X-API-Key"] = api_key
        
        auth = None
        if username and password and not api_key:
            auth = (username, password)
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            auth=auth,
            timeout=timeout,
            follow_redirects=True
        )
        
    async def __aenter__(self):
        """Context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.client.aclose()
    
    async def test_connection(self) -> bool:
        """
        Test API connection and authentication
        
        Returns:
            True if connection successful
        """
        try:
            logger.info("🔌 Testing ValueEdge API connection...")
            
            # Try to get current user or health endpoint
            endpoints_to_try = [
                "/api/v1/user/me",
                "/api/user",
                "/api/health",
                "/health",
                "/"
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    response = await self.client.get(endpoint)
                    if response.status_code in [200, 401]:  # 401 means auth works, just not logged in
                        logger.info(f"✅ Connection successful via {endpoint}")
                        return True
                except:
                    continue
            
            logger.error("❌ Could not connect to any known endpoint")
            return False
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    async def get_ticket(self, ticket_id: str) -> Optional[TicketContext]:
        """
        Fetch complete ticket context from ValueEdge API
        
        Args:
            ticket_id: Ticket ID (e.g., "VE-CB-1234")
        
        Returns:
            TicketContext with all extracted information
        """
        try:
            logger.info(f"📥 Fetching ticket {ticket_id} via API")
            
            # Try different API endpoint patterns
            endpoints = [
                f"/api/v1/tickets/{ticket_id}",
                f"/api/tickets/{ticket_id}",
                f"/api/v1/issues/{ticket_id}",
                f"/api/issues/{ticket_id}",
                f"/rest/api/2/issue/{ticket_id}",  # Jira-like
                f"/api/v1/workitems/{ticket_id}",
            ]
            
            ticket_data = None
            for endpoint in endpoints:
                try:
                    response = await self.client.get(endpoint)
                    if response.status_code == 200:
                        ticket_data = response.json()
                        logger.info(f"✅ Ticket fetched via {endpoint}")
                        break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        continue
                    raise
            
            if not ticket_data:
                logger.error(f"❌ Ticket {ticket_id} not found in any endpoint")
                return None
            
            # Parse response into TicketContext
            context = await self._parse_ticket_response(ticket_id, ticket_data)
            
            # Fetch additional data
            await self._fetch_comments(ticket_id, context)
            await self._fetch_attachments(ticket_id, context)
            await self._fetch_linked_tickets(ticket_id, context)
            
            logger.info(f"✅ Complete context fetched for {ticket_id}")
            return context
            
        except Exception as e:
            logger.error(f"❌ Error fetching ticket {ticket_id}: {e}")
            return None
    
    async def _parse_ticket_response(self, ticket_id: str, data: Dict[str, Any]) -> TicketContext:
        """
        Parse API response into TicketContext
        Handles different API response formats
        """
        # Helper to safely extract nested values
        def get_field(obj, *keys, default=None):
            for key in keys:
                if isinstance(obj, dict):
                    obj = obj.get(key, default)
                else:
                    return default
            return obj
        
        # Extract fields (adjust based on your ValueEdge API response format)
        context = TicketContext(
            ticket_id=ticket_id,
            title=get_field(data, "title") or get_field(data, "summary") or "No title",
            description=get_field(data, "description") or get_field(data, "body") or "",
            priority=get_field(data, "priority", "name") or get_field(data, "priority") or "MEDIUM",
            status=get_field(data, "status", "name") or get_field(data, "status") or "OPEN",
            type=get_field(data, "type", "name") or get_field(data, "type") or "task",
            assignee=get_field(data, "assignee", "name") or get_field(data, "assignee"),
            reporter=get_field(data, "reporter", "name") or get_field(data, "reporter"),
            team=get_field(data, "team", "name") or get_field(data, "team"),
            
            # Extract labels
            labels=data.get("labels", []) or data.get("tags", []),
            
            # Dates
            created_date=self._parse_date(get_field(data, "created_at") or get_field(data, "createdDate")),
            updated_date=self._parse_date(get_field(data, "updated_at") or get_field(data, "updatedDate")),
            due_date=self._parse_date(get_field(data, "due_date") or get_field(data, "dueDate")),
            
            # Effort
            estimated_hours=get_field(data, "estimated_hours") or get_field(data, "timeEstimate"),
            actual_hours=get_field(data, "actual_hours") or get_field(data, "timeSpent"),
            story_points=get_field(data, "story_points") or get_field(data, "storyPoints"),
            
            # Technical
            environment=get_field(data, "environment"),
            affected_components=data.get("components", []) or [],
            affected_versions=data.get("affected_versions", []) or [],
            
            # Store raw data for debugging
            raw_data=data
        )
        
        # Extract acceptance criteria from description or custom field
        context.acceptance_criteria = self._extract_acceptance_criteria(
            data.get("acceptance_criteria") or data.get("acceptanceCriteria") or context.description
        )
        
        return context
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        
        try:
            # Try common formats
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return None
    
    def _extract_acceptance_criteria(self, text: Optional[str]) -> List[str]:
        """
        Extract acceptance criteria from text
        Looks for patterns like:
        - Acceptance Criteria:
        - AC:
        - Bullet lists after these headers
        """
        if not text:
            return []
        
        criteria = []
        lines = text.split('\n')
        in_ac_section = False
        
        for line in lines:
            line = line.strip()
            
            # Detect AC section start
            if any(header in line.lower() for header in ['acceptance criteria', 'ac:', 'acceptance:']):
                in_ac_section = True
                continue
            
            # Extract criteria
            if in_ac_section:
                # Stop at next section
                if line and not line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                    if line.isupper() or line.endswith(':'):
                        break
                
                # Extract bullet point
                if line.startswith(('-', '*', '•')):
                    criteria.append(line[1:].strip())
                elif line and line[0].isdigit() and line[1] == '.':
                    criteria.append(line[2:].strip())
        
        return criteria
    
    async def _fetch_comments(self, ticket_id: str, context: TicketContext):
        """Fetch ticket comments"""
        try:
            endpoints = [
                f"/api/v1/tickets/{ticket_id}/comments",
                f"/api/tickets/{ticket_id}/comments",
                f"/api/v1/issues/{ticket_id}/comments",
            ]
            
            for endpoint in endpoints:
                try:
                    response = await self.client.get(endpoint)
                    if response.status_code == 200:
                        comments_data = response.json()
                        
                        # Parse comments
                        if isinstance(comments_data, list):
                            for comment in comments_data:
                                context.comments.append(ValueEdgeComment(
                                    author=comment.get("author", {}).get("name", "Unknown"),
                                    text=comment.get("body", ""),
                                    created_at=self._parse_date(comment.get("created_at")) or datetime.now(),
                                    attachments=comment.get("attachments", [])
                                ))
                        
                        logger.info(f"✅ Fetched {len(context.comments)} comments")
                        break
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch comments: {e}")
    
    async def _fetch_attachments(self, ticket_id: str, context: TicketContext):
        """Fetch ticket attachments"""
        try:
            endpoints = [
                f"/api/v1/tickets/{ticket_id}/attachments",
                f"/api/tickets/{ticket_id}/attachments",
            ]
            
            for endpoint in endpoints:
                try:
                    response = await self.client.get(endpoint)
                    if response.status_code == 200:
                        attachments_data = response.json()
                        
                        if isinstance(attachments_data, list):
                            for att in attachments_data:
                                attachment = ValueEdgeAttachment(
                                    filename=att.get("filename", "unknown"),
                                    url=att.get("url", ""),
                                    size=att.get("size"),
                                    mime_type=att.get("mime_type")
                                )
                                context.attachments.append(attachment)
                                
                                # Categorize screenshots and logs
                                if any(ext in attachment.filename.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif']):
                                    context.screenshots.append(attachment.url)
                                elif any(ext in attachment.filename.lower() for ext in ['.log', '.txt']):
                                    context.logs.append(attachment.url)
                        
                        logger.info(f"✅ Fetched {len(context.attachments)} attachments")
                        break
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch attachments: {e}")
    
    async def _fetch_linked_tickets(self, ticket_id: str, context: TicketContext):
        """Fetch linked/related tickets"""
        try:
            endpoints = [
                f"/api/v1/tickets/{ticket_id}/links",
                f"/api/tickets/{ticket_id}/relationships",
            ]
            
            for endpoint in endpoints:
                try:
                    response = await self.client.get(endpoint)
                    if response.status_code == 200:
                        links_data = response.json()
                        
                        if isinstance(links_data, list):
                            for link in links_data:
                                context.linked_tickets.append(LinkedTicket(
                                    ticket_id=link.get("ticket_id", ""),
                                    title=link.get("title", ""),
                                    relationship=link.get("relationship", "relates_to")
                                ))
                        
                        logger.info(f"✅ Fetched {len(context.linked_tickets)} linked tickets")
                        break
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch linked tickets: {e}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_fetch_ticket():
    """Example: Fetch ticket using API"""
    
    async with ValueEdgeAPIClient(
        base_url="https://valueedge.your-company.com",
        api_key="your-api-key",  # Preferred
        # OR: username="user", password="pass"
    ) as client:
        
        # Test connection
        if not await client.test_connection():
            print("❌ Connection failed")
            return
        
        # Fetch ticket
        context = await client.get_ticket("VE-CB-1234")
        
        if context:
            print(f"✅ Ticket: {context.title}")
            print(f"Description: {context.description[:200]}...")
            print(f"Priority: {context.priority}")
            print(f"Status: {context.status}")
            print(f"Comments: {len(context.comments)}")
            print(f"Attachments: {len(context.attachments)}")
            print(f"Linked Tickets: {len(context.linked_tickets)}")
            print(f"Acceptance Criteria: {len(context.acceptance_criteria)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_fetch_ticket())
