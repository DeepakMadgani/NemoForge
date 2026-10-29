"""
ValueEdge Playwright Service
Fetches tickets using browser automation (for OTP/MFA support)
"""

from typing import Optional
import logging
from .valueedge_scraper import ValueEdgeScraper, ValueEdgeTicket
from .models import TicketContext, ValueEdgeComment

logger = logging.getLogger(__name__)


class ValueEdgeService:
    """
    ValueEdge integration using Playwright only
    
    Features:
    - Visual browser (for OTP/MFA)
    - Automatic login with credentials
    - Ticket extraction via web scraping
    """
    
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str
    ):
        """
        Initialize ValueEdge service
        
        Args:
            base_url: ValueEdge base URL
            username: Username for login
            password: Password for login
        """
        self.base_url = base_url
        self.username = username
        self.password = password
    
    async def get_ticket(self, ticket_id: str) -> Optional[TicketContext]:
        """
        Fetch ticket using Playwright scraper
        
        Args:
            ticket_id: Ticket ID (e.g., "5377391")
        
        Returns:
            TicketContext with complete ticket information
        """
        logger.info(f"🎫 Fetching ticket {ticket_id} using Playwright...")
        
        if not (self.username and self.password):
            logger.error("❌ Username and password required")
            return None
        
        try:
            logger.info("🎭 Launching browser...")
            
            async with ValueEdgeScraper(
                valueedge_url=self.base_url,
                username=self.username,
                password=self.password,
                headless=False,  # Show browser for OTP/MFA
                screenshot_on_error=True,
                slow_mo=500  # Slow down actions by 500ms for visibility
            ) as scraper:
                
                # Fetch ticket
                scraped_ticket = await scraper.get_ticket(ticket_id)
                if not scraped_ticket:
                    logger.error(f"❌ Failed to fetch ticket {ticket_id}")
                    return None
                
                # Convert to TicketContext
                context = self._convert_scraped_to_context(scraped_ticket)
                context.source = "playwright"
                
                logger.info("✅ Successfully fetched ticket via Playwright")
                return context
                
        except Exception as e:
            logger.error(f"❌ Playwright fetch error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _convert_scraped_to_context(self, scraped: ValueEdgeTicket) -> TicketContext:
        """
        Convert ValueEdgeTicket to TicketContext
        """
        # Convert comments
        comments = []
        for comment_data in scraped.comments:
            comments.append(ValueEdgeComment(
                author=comment_data.get("author", "Unknown"),
                text=comment_data.get("text", ""),
                created_at=comment_data.get("created_at"),
                attachments=comment_data.get("attachments", [])
            ))
        
        # Build context
        context = TicketContext(
            ticket_id=scraped.ticket_id,
            title=scraped.title,
            description=scraped.description,
            priority=scraped.priority,
            status=scraped.status,
            assignee=scraped.assignee,
            labels=scraped.labels,
            acceptance_criteria=scraped.acceptance_criteria,
            comments=comments,
            attachments=scraped.attachments,
            created_date=scraped.created_date,
            updated_date=scraped.updated_date,
            estimated_hours=scraped.estimated_hours,
            actual_hours=scraped.actual_hours,
            raw_data={"html": scraped.raw_html}
        )
        
        return context


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def fetch_ticket(
    ticket_id: str,
    valueedge_url: str,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> Optional[TicketContext]:
    """
    Convenience function to fetch a ticket
    
    Usage:
        context = await fetch_ticket(
            "VE-CB-1234",
            "https://valueedge.company.com",
            api_key="your-key"  # Or username/password
        )
    """
    service = ValueEdgeService(
        base_url=valueedge_url,
        api_key=api_key,
        username=username,
        password=password
    )
    
    return await service.get_ticket(ticket_id)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_usage():
    """Example: Fetch ticket with automatic fallback"""
    
    # Option 1: With API key (tries API first)
    service = ValueEdgeService(
        base_url="https://valueedge.your-company.com",
        api_key="your-api-key",
        username="backup-user",  # For Playwright fallback
        password="backup-pass"
    )
    
    context = await service.get_ticket("VE-CB-1234")
    
    if context:
        print(f"✅ Fetched via: {context.source}")
        print(f"Title: {context.title}")
        print(f"Priority: {context.priority}")
        print(f"Acceptance Criteria: {len(context.acceptance_criteria)}")
        print(f"Comments: {len(context.comments)}")
        print(f"Attachments: {len(context.attachments)}")
    else:
        print("❌ Failed to fetch ticket")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
