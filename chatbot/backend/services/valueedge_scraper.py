"""
ValueEdge Ticket Scraper using Playwright
Automatically extracts ticket information from ValueEdge web interface
"""

from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeout
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import asyncio
import logging
import re
import urllib.parse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ValueEdgeTicket(BaseModel):
    """Structured ticket data extracted from ValueEdge"""
    ticket_id: str
    title: str
    description: str
    priority: str
    status: str
    assignee: Optional[str] = None
    labels: List[str] = []
    acceptance_criteria: List[str] = []
    attachments: List[Dict[str, str]] = []  # List of {filename, url}
    comments: List[Dict[str, str]] = []  # List of {author, text, created_at}
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    raw_html: Optional[str] = None  # For debugging


class ValueEdgeScraper:
    """
    Playwright-based scraper for ValueEdge
    
    Features:
    - Automatic login
    - Ticket detail extraction
    - Screenshot capture
    - Error handling
    """
    
    def __init__(
        self,
        valueedge_url: str,
        username: str,
        password: str,
        headless: bool = True,
        screenshot_on_error: bool = True,
        slow_mo: int = 0
    ):
        self.valueedge_url = valueedge_url.rstrip('/')
        self.username = username
        self.password = password
        self.headless = headless
        self.screenshot_on_error = screenshot_on_error
        self.slow_mo = slow_mo
        self.browser: Optional[Browser] = None
        self.playwright = None
        
    async def __aenter__(self):
        """Context manager entry"""
        self.playwright = await async_playwright().start()
        # Use Microsoft Edge instead of Chromium
        self.browser = await self.playwright.chromium.launch(
            channel='msedge',  # Use installed Microsoft Edge
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def login(self, page: Page) -> bool:
        """
        Login to ValueEdge
        
        Returns:
            True if login successful, False otherwise
        """
        try:
            logger.info(f"🔐 Opening ValueEdge: {self.valueedge_url}")
            
            # Just navigate - don't wait for anything specific
            await page.goto(self.valueedge_url, wait_until="domcontentloaded")
            logger.info("📄 Page opened successfully")
            
            # Take screenshot to see initial state
            await page.screenshot(path="login_page_initial.png")
            logger.info("📸 Screenshot saved: login_page_initial.png")
            
            # Now just WAIT - let user do everything manually
            logger.info("")
            logger.info("=" * 80)
            logger.info("🪟 EDGE BROWSER IS NOW OPEN!")
            logger.info("=" * 80)
            logger.info("")
            logger.info("👉 YOU have 10 MINUTES to:")
            logger.info("   1. Complete any redirects")
            logger.info("   2. Enter Microsoft credentials")
            logger.info("   3. Approve OTP/MFA")
            logger.info("   4. Wait for redirect back to ValueEdge")
            logger.info("   5. Navigate to your ticket if needed")
            logger.info("")
            logger.info("⏰ Browser will stay open for 10 minutes!")
            logger.info("   No rush - take your time!")
            logger.info("")
            
            # Just wait 10 minutes - don't check anything
            await asyncio.sleep(600)  # 10 minutes
            
            logger.info("✅ 10 minutes complete - assuming login succeeded")
            logger.info(f"📍 Final URL: {page.url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return False
    
    async def get_ticket(self, ticket_id: str) -> Optional[ValueEdgeTicket]:
        """
        Fetch ticket details from ValueEdge
        
        Args:
            ticket_id: ValueEdge ticket ID (e.g., "VE-1234")
        
        Returns:
            ValueEdgeTicket object or None if failed
        """
        if not self.browser:
            raise RuntimeError("Browser not initialized. Use async with context manager.")
        
        try:
            logger.info(f"🎫 Fetching ticket: {ticket_id}")
            
            # Create new page
            page = await self.browser.new_page()
            
            # Login first
            login_success = await self.login(page)
            if not login_success:
                logger.error("Login failed, cannot fetch ticket")
                await page.close()
                return None
            
            # Navigate to ticket page
            # ValueEdge MicroFocus format: /ui/entity-navigation?p=workspace&entityType=work_item&id=ID
            
            # Extract workspace/project params from configured URL
            workspace_id = "4001/31003"  # Default workspace
            if "?p=" in self.valueedge_url:
                # Extract workspace ID from configured URL
                parsed = urllib.parse.urlparse(self.valueedge_url)
                params = urllib.parse.parse_qs(parsed.query)
                if 'p' in params:
                    workspace_id = params['p'][0]
            
            # Build base URL (remove query params if present)
            base_url = self.valueedge_url.split('?')[0]
            if not base_url.endswith('/ui'):
                # Ensure we have the right base
                if '/ui/' in base_url:
                    base_url = base_url.split('/ui/')[0] + '/ui'
                else:
                    base_url = base_url.rstrip('/') + '/ui'
            
            # Try different entity types
            ticket_url_patterns = [
                f"{base_url}/entity-navigation?p={workspace_id}&entityType=work_item&id={ticket_id}",
                f"{base_url}/entity-navigation?p={workspace_id}&entityType=story&id={ticket_id}",
                f"{base_url}/entity-navigation?p={workspace_id}&entityType=defect&id={ticket_id}",
                f"{base_url}/entity-navigation?p={workspace_id}&entityType=task&id={ticket_id}",
            ]
            
            ticket_loaded = False
            for ticket_url in ticket_url_patterns:
                try:
                    logger.info(f"🔍 Trying URL: {ticket_url}")
                    await page.goto(ticket_url, wait_until="domcontentloaded", timeout=15000)
                    
                    # For SPAs, wait for content to render
                    await asyncio.sleep(3)
                    
                    # Check if ticket content is visible (look for common elements)
                    # Try to find ticket-specific content that indicates successful load
                    content_indicators = [
                        "h1",  # Title
                        ".story-title",
                        ".work-item-title", 
                        "[data-aid='story-title']",
                        ".entity-title",
                        ".item-title",
                        "[class*='title']",
                        "span.name",
                        "div.name"
                    ]
                    
                    for indicator in content_indicators:
                        try:
                            element = await page.query_selector(indicator)
                            if element:
                                text = await element.text_content()
                                if text and len(text.strip()) > 0:
                                    ticket_loaded = True
                                    logger.info(f"✅ Ticket page loaded: {ticket_url}")
                                    logger.info(f"   Found content with selector '{indicator}': {text.strip()[:100]}")
                                    break
                        except:
                            continue
                    
                    if ticket_loaded:
                        break
                        
                except Exception as e:
                    logger.warning(f"⚠️ URL pattern failed: {ticket_url} - {e}")
                    continue
            
            if not ticket_loaded:
                logger.error(f"❌ Could not load ticket page for {ticket_id}")
                logger.error(f"   Tried {len(ticket_url_patterns)} URL patterns")
                logger.error(f"   Current URL: {page.url}")
                
                # Save screenshot for debugging
                try:
                    await page.screenshot(path=f"failed_ticket_{ticket_id}.png", full_page=True)
                    logger.info(f"📸 Saved debug screenshot: failed_ticket_{ticket_id}.png")
                except:
                    pass
                    
                await page.close()
                return None
            
            # Wait for content to load
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(1)  # Give dynamic content time to render
            
            # Extract ticket information
            logger.info(f"📊 Extracting ticket data from page...")
            ticket_data = await self._extract_ticket_data(page, ticket_id)
            
            # Take screenshot for verification
            screenshot_path = f"ticket_{ticket_id}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            logger.info(f"📸 Screenshot saved: {screenshot_path}")
            logger.info(f"✅ Ticket extracted: {ticket_data.title}")
            
            # Keep browser open so you can see the result!
            logger.info("⏸️ Keeping browser open for 30 seconds so you can see the ticket...")
            logger.info("   Check the Edge browser to verify the ticket details!")
            await asyncio.sleep(30)
            
            await page.close()
            logger.info("🎭 Browser closed - ticket fetch complete!")
            
            return ticket_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching ticket {ticket_id}: {e}")
            if self.screenshot_on_error:
                try:
                    await page.screenshot(path=f"error_{ticket_id}.png")
                    logger.info(f"📸 Error screenshot saved: error_{ticket_id}.png")
                except:
                    pass
            return None
    
    async def _extract_ticket_data(self, page: Page, ticket_id: str) -> ValueEdgeTicket:
        """
        Extract ticket data from the page
        This method uses multiple strategies to find data
        """
        logger.info(f"📊 Extracting data for {ticket_id}")
        
        # Get page HTML
        html = await page.content()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = await self._extract_title(page, soup)
        
        # Extract description
        description = await self._extract_description(page, soup)
        
        # Extract priority
        priority = await self._extract_priority(page, soup)
        
        # Extract status
        status = await self._extract_status(page, soup)
        
        # Extract assignee
        assignee = await self._extract_assignee(page, soup)
        
        # Extract labels/tags
        labels = await self._extract_labels(page, soup)
        
        # Extract acceptance criteria
        acceptance_criteria = await self._extract_acceptance_criteria(page, soup)
        
        # Extract dates
        created_date = await self._extract_date(page, soup, "created")
        updated_date = await self._extract_date(page, soup, "updated")
        
        # Extract hours
        estimated_hours = await self._extract_hours(page, soup, "estimated")
        actual_hours = await self._extract_hours(page, soup, "actual")
        
        ticket = ValueEdgeTicket(
            ticket_id=ticket_id,
            title=title,
            description=description,
            priority=priority,
            status=status,
            assignee=assignee,
            labels=labels,
            acceptance_criteria=acceptance_criteria,
            created_date=created_date,
            updated_date=updated_date,
            estimated_hours=estimated_hours,
            actual_hours=actual_hours,
            raw_html=html[:5000]  # First 5KB for debugging
        )
        
        logger.info(f"✅ Extracted ticket: {ticket.title}")
        return ticket
    
    async def _extract_title(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract ticket title"""
        selectors = [
            "h1.ticket-title",
            "h1[data-test-id='title']",
            ".ticket-header h1",
            "h1",
            ".title",
            "[data-field='title']"
        ]
        
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    title = await element.text_content()
                    if title and len(title.strip()) > 0:
                        return title.strip()
            except:
                continue
        
        # Fallback to soup
        title_elem = soup.find(['h1', 'h2'], class_=re.compile(r'title|header'))
        if title_elem:
            return title_elem.get_text(strip=True)
        
        return "Title not found"
    
    async def _extract_description(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract ticket description"""
        selectors = [
            ".ticket-description",
            "[data-test-id='description']",
            ".description",
            "[data-field='description']",
            ".markdown-body",
            ".ticket-content"
        ]
        
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    desc = await element.text_content()
                    if desc and len(desc.strip()) > 10:
                        return desc.strip()
            except:
                continue
        
        return "Description not found"
    
    async def _extract_priority(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract priority"""
        selectors = [
            "[data-field='priority']",
            ".priority",
            "[aria-label*='Priority']"
        ]
        
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    priority = await element.text_content()
                    if priority:
                        return priority.strip()
            except:
                continue
        
        return "MEDIUM"  # Default
    
    async def _extract_status(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract status"""
        selectors = [
            "[data-field='status']",
            ".status",
            "[aria-label*='Status']",
            ".ticket-status"
        ]
        
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    status = await element.text_content()
                    if status:
                        return status.strip()
            except:
                continue
        
        return "OPEN"  # Default
    
    async def _extract_assignee(self, page: Page, soup: BeautifulSoup) -> Optional[str]:
        """Extract assignee"""
        selectors = [
            "[data-field='assignee']",
            ".assignee",
            "[aria-label*='Assignee']"
        ]
        
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    assignee = await element.text_content()
                    if assignee:
                        return assignee.strip()
            except:
                continue
        
        return None
    
    async def _extract_labels(self, page: Page, soup: BeautifulSoup) -> List[str]:
        """Extract labels/tags"""
        labels = []
        
        selectors = [
            ".label",
            ".tag",
            "[data-field='labels']",
            ".ticket-label"
        ]
        
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                for elem in elements:
                    label = await elem.text_content()
                    if label:
                        labels.append(label.strip())
            except:
                continue
        
        return list(set(labels))  # Remove duplicates
    
    async def _extract_acceptance_criteria(self, page: Page, soup: BeautifulSoup) -> List[str]:
        """Extract acceptance criteria"""
        criteria = []
        
        # Look for acceptance criteria section
        ac_section_selectors = [
            "[data-field='acceptance-criteria']",
            ".acceptance-criteria",
            "section:has-text('Acceptance Criteria')",
            "div:has-text('Acceptance Criteria')"
        ]
        
        for selector in ac_section_selectors:
            try:
                section = await page.query_selector(selector)
                if section:
                    # Look for list items
                    items = await section.query_selector_all("li, p")
                    for item in items:
                        text = await item.text_content()
                        if text and len(text.strip()) > 5:
                            criteria.append(text.strip())
                    
                    if criteria:
                        break
            except:
                continue
        
        return criteria
    
    async def _extract_date(self, page: Page, soup: BeautifulSoup, date_type: str) -> Optional[str]:
        """Extract created or updated date"""
        selectors = [
            f"[data-field='{date_type}-date']",
            f".{date_type}-date",
            f"[aria-label*='{date_type}' i]"
        ]
        
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    date = await element.text_content()
                    if date:
                        return date.strip()
            except:
                continue
        
        return None
    
    async def _extract_hours(self, page: Page, soup: BeautifulSoup, hour_type: str) -> Optional[float]:
        """Extract estimated or actual hours"""
        selectors = [
            f"[data-field='{hour_type}-hours']",
            f".{hour_type}-hours"
        ]
        
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    hours_text = await element.text_content()
                    if hours_text:
                        # Extract number from text
                        match = re.search(r'(\d+(?:\.\d+)?)', hours_text)
                        if match:
                            return float(match.group(1))
            except:
                continue
        
        return None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def fetch_valueedge_ticket_example():
    """Example usage"""
    
    # Configuration
    VALUEEDGE_URL = "https://valueedge.your-company.com"
    USERNAME = "your-username"
    PASSWORD = "your-password"
    
    async with ValueEdgeScraper(
        valueedge_url=VALUEEDGE_URL,
        username=USERNAME,
        password=PASSWORD,
        headless=False  # Set to True in production
    ) as scraper:
        
        # Fetch ticket
        ticket = await scraper.get_ticket("VE-CB-1234")
        
        if ticket:
            print(f"✅ Ticket: {ticket.ticket_id}")
            print(f"Title: {ticket.title}")
            print(f"Description: {ticket.description[:200]}...")
            print(f"Priority: {ticket.priority}")
            print(f"Status: {ticket.status}")
            print(f"Labels: {ticket.labels}")
            print(f"Acceptance Criteria: {len(ticket.acceptance_criteria)} items")
        else:
            print("❌ Failed to fetch ticket")


if __name__ == "__main__":
    asyncio.run(fetch_valueedge_ticket_example())
