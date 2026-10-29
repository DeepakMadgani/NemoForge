# ValueEdge Integration - Complete Guide

## 🎯 Overview

The system automatically fetches tickets from ValueEdge using a **dual-strategy approach**:

### Strategy Priority:
1. **🚀 Official API** (Primary) - Fast, reliable, structured data
2. **🎭 Playwright Scraper** (Fallback) - Works when API unavailable

---

## 📊 What Data is Fetched

### Complete TicketContext includes:

```python
TicketContext(
    # Basic Information
    ticket_id="VE-CB-1234",
    title="Fix login bug",
    description="Detailed description...",
    
    # Status & Priority
    priority="HIGH",
    status="IN_PROGRESS",
    type="bug",
    
    # Assignment
    assignee="John Doe",
    reporter="Jane Smith",
    team="Engineering",
    
    # Requirements
    acceptance_criteria=[...],
    labels=["backend", "security"],
    
    # Rich Context
    comments=[...],              # All comments with attachments
    attachments=[...],           # Files, screenshots, logs
    linked_tickets=[...],        # Related/blocking tickets
    
    # Metadata
    created_date=datetime(...),
    updated_date=datetime(...),
    due_date=datetime(...),
    
    # Effort
    estimated_hours=8.0,
    actual_hours=6.5,
    story_points=5,
    
    # Technical Context
    screenshots=[...],           # PNG/JPG attachments
    logs=[...],                  # Log file attachments
    affected_components=[...],
    affected_versions=[...],
    environment="production",
    
    # Tracking
    source="api"  # or "playwright"
)
```

---

## 🚀 Quick Start

### 1. Configure ValueEdge Credentials

#### Option A: API Key (Recommended)
```python
config = {
    "url": "https://valueedge.your-company.com",
    "api_key": "your-api-key-here"
}
```

#### Option B: Username/Password
```python
config = {
    "url": "https://valueedge.your-company.com",
    "username": "your-username",
    "password": "your-password"
}
```

### 2. Test Connection

```powershell
# Test both methods
curl "http://localhost:8000/api/valueedge/test?url=https://valueedge.company.com&api_key=YOUR_KEY&username=USER&password=PASS"
```

Response:
```json
{
  "success": true,
  "recommended_method": "api",
  "methods": {
    "api": {
      "available": true,
      "message": "API connection successful"
    },
    "playwright": {
      "available": true,
      "message": "Playwright login successful"
    }
  }
}
```

### 3. Fetch Ticket

```powershell
curl -X POST "http://localhost:8000/api/valueedge/fetch-ticket?ticket_id=VE-CB-1234" `
  -H "Content-Type: application/json" `
  -d '{
    "url": "https://valueedge.your-company.com",
    "api_key": "your-key"
  }'
```

Response:
```json
{
  "success": true,
  "method": "api",
  "ticket": {
    "ticket_id": "VE-CB-1234",
    "title": "Fix authentication bug",
    "description": "...",
    "priority": "HIGH",
    "status": "OPEN",
    "comments": [...],
    "attachments": [...],
    "acceptance_criteria": [...]
  }
}
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```env
# Primary Method - API
VALUEEDGE_URL=https://valueedge.your-company.com
VALUEEDGE_API_KEY=your-api-key-here

# Fallback Method - Playwright
VALUEEDGE_USERNAME=your-username
VALUEEDGE_PASSWORD=your-password
```

### API Endpoint Customization

If your ValueEdge uses different API endpoints, edit `services/valueedge_api.py`:

```python
# In ValueEdgeAPIClient.get_ticket()
endpoints = [
    f"/api/v1/tickets/{ticket_id}",      # Default
    f"/api/v2/issues/{ticket_id}",       # Your custom endpoint
    f"/rest/api/issue/{ticket_id}",      # Jira-like
]
```

### Playwright Selector Customization

If your ValueEdge HTML structure is different, edit `services/valueedge_scraper.py`:

```python
# In ValueEdgeScraper._extract_title()
selectors = [
    "h1.ticket-title",        # Default
    "h1.your-custom-class",   # Your selector
    "[data-ticket-title]",    # Your attribute
]
```

---

## 📡 API Endpoints

### Test Connection
```
GET /api/valueedge/test
Query params: url, api_key?, username?, password?
```

### Fetch Ticket
```
POST /api/valueedge/fetch-ticket
Query params: ticket_id
Body: {url, api_key?, username?, password?}
```

### WebSocket Updates
```
WS /ws
Listen for real-time fetch progress
```

---

## 🎭 How Fallback Works

```
┌─────────────────┐
│  Fetch Request  │
└────────┬────────┘
         │
         ▼
   ┌──────────┐
   │ Try API  │
   └─────┬────┘
         │
    ┌────┴────┐
    │ Success?│
    └────┬────┘
         │
    ┌────┴────────┐
    │YES          │NO
    ▼             ▼
┌────────┐   ┌─────────────┐
│ Return │   │ Try Playwright│
└────────┘   └──────┬──────┘
                    │
               ┌────┴────┐
               │ Success?│
               └────┬────┘
                    │
               ┌────┴────────┐
               │YES          │NO
               ▼             ▼
          ┌────────┐    ┌───────┐
          │ Return │    │ Error │
          └────────┘    └───────┘
```

---

## 🔍 Troubleshooting

### Problem: API Returns 401 Unauthorized

**Solution:**
- Verify API key is correct
- Check if API key has expired
- Ensure user has permission to access tickets
- System will automatically fall back to Playwright

### Problem: Playwright Login Fails

**Solution:**
```powershell
# Test with visible browser
python -c "
from services.valueedge_scraper import ValueEdgeScraper
import asyncio

async def test():
    async with ValueEdgeScraper(
        valueedge_url='https://valueedge.company.com',
        username='user',
        password='pass',
        headless=False  # See browser
    ) as scraper:
        await scraper.get_ticket('VE-CB-1234')

asyncio.run(test())
"
```

Check `login_error.png` screenshot for details.

### Problem: Wrong Data Extracted

**Cause:** HTML selectors don't match your ValueEdge version

**Solution:** Update selectors in `valueedge_scraper.py`

1. Run with `headless=False`
2. Inspect HTML in browser DevTools
3. Update selectors in `_extract_*` methods
4. Test again

### Problem: Missing Comments/Attachments

**Cause:** API endpoint not found

**Solution:**
```python
# Add your endpoints in valueedge_api.py
async def _fetch_comments():
    endpoints = [
        f"/api/v1/tickets/{ticket_id}/comments",
        f"/your/custom/endpoint/{ticket_id}/comments",  # Add this
    ]
```

---

## 🚀 Performance

### API Method:
- **Speed:** ~1-2 seconds
- **Reliability:** High (if API available)
- **Data Quality:** Structured, complete

### Playwright Method:
- **Speed:** ~5-10 seconds (login + scrape)
- **Reliability:** Medium (depends on HTML stability)
- **Data Quality:** Good (extracted from HTML)

### Best Practices:
1. Always provide API key if available
2. Cache ticket data to reduce API calls
3. Use webhook for real-time updates (if ValueEdge supports it)

---

## 🔐 Security

### API Key Storage:
```env
# .env file (never commit!)
VALUEEDGE_API_KEY=secret-key-here
```

### Password Handling:
- Passwords sent via HTTPS only
- Never logged or stored permanently
- Used only for Playwright session

### Screenshots:
- Saved locally for debugging
- May contain sensitive data
- Delete after debugging: `rm ticket_*.png error_*.png`

---

## 📦 Dependencies

### For API Method:
- httpx (async HTTP client)
- python-dateutil (date parsing)

### For Playwright Method:
- playwright (browser automation)
- beautifulsoup4 (HTML parsing)

Install all:
```powershell
pip install -r requirements.txt
pip install -r requirements-playwright.txt
playwright install chromium
```

---

## 🧪 Testing

### Test API Method:
```python
from services.valueedge_api import ValueEdgeAPIClient
import asyncio

async def test():
    async with ValueEdgeAPIClient(
        base_url="https://valueedge.company.com",
        api_key="your-key"
    ) as client:
        context = await client.get_ticket("VE-CB-1234")
        print(context.title if context else "Failed")

asyncio.run(test())
```

### Test Playwright Method:
```powershell
cd backend
python test_scraper.py
```

### Test Unified Service:
```python
from services.valueedge_service import fetch_ticket
import asyncio

async def test():
    context = await fetch_ticket(
        "VE-CB-1234",
        "https://valueedge.company.com",
        api_key="your-key"
    )
    print(f"Fetched via: {context.source}")

asyncio.run(test())
```

---

## 📚 Next Steps

After fetching ticket:
1. ✅ Display ticket info in UI
2. ✅ Ask user for confirmation
3. ✅ Start autonomous workflow
4. ✅ Generate code/tests
5. ✅ Run build/tests
6. ✅ Show results to user

See `main.py` endpoint `/api/workflow/start` for integration.
