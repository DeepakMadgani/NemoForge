# Ticket-to-Code Chatbot Backend

FastAPI backend with **Playwright** for ValueEdge ticket scraping (visual browser with OTP support).

## 🎯 What This Does

- **Fetch ValueEdge Tickets** using Playwright (opens visible browser for OTP)
- **Extract ticket data** (title, description, comments, attachments, etc.)
- **WebSocket updates** to frontend in real-time
- **Project & Knowledge Base management** for autonomous workflows

## 📦 Where Playwright is Installed

Playwright is installed in your virtual environment:
```
C:\Users\dmadgani\Desktop\My_Aviator\aviator-plugin-sample\.venv\
```

**Playwright browsers** (Chromium) are stored at:
```
%USERPROFILE%\AppData\Local\ms-playwright\chromium-<version>\
```

## 🚀 Quick Start

### 1. Install Dependencies (Already Done!)

```powershell
# From aviator-plugin-sample directory
cd C:\Users\dmadgani\Desktop\My_Aviator\aviator-plugin-sample

# Playwright + dependencies already installed via uv:
uv pip install playwright beautifulsoup4 lxml fastapi uvicorn websockets pydantic gitpython

# Install Chromium browser (if not done):
.venv\Scripts\playwright.exe install chromium
```

### 2. Run Server

```powershell
# Navigate to backend
cd chatbot\backend

# Run with uvicorn (auto-reload enabled)
..\..\.venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8002 --reload
```

Server runs at: **http://localhost:8002**

## 🌐 Frontend

React frontend is in `chatbot/frontend/`:
```powershell
cd chatbot\frontend
npm run dev  # Runs on http://localhost:3000
```
```
POST /api/projects - Create project (local or git)
GET /api/projects - List all projects
GET /api/projects/{id} - Get project details
DELETE /api/projects/{id} - Delete project
```

### Knowledge Bases
```
POST /api/projects/{id}/knowledge-bases - Create KB
GET /api/projects/{id}/knowledge-bases - List KBs
POST /api/knowledge-bases/{id}/upload - Upload files
POST /api/knowledge-bases/{id}/index - Index files
```

### ValueEdge Integration
```
POST /api/valueedge/fetch-ticket - Fetch ticket using Playwright
GET /api/valueedge/test - Test connection
```

### Workflow
```
POST /api/workflow/start - Start autonomous workflow
GET /api/workflow/{id}/events - Get workflow events
```

### WebSocket
```
WS /ws - Real-time updates
```

## 🎭 Playwright Usage

The backend uses Playwright to automatically scrape ValueEdge tickets:

```python
# Example: Fetch ticket
response = await client.post("/api/valueedge/fetch-ticket", json={
    "ticket_id": "VE-CB-1234",
    "config": {
        "url": "https://valueedge.company.com",
        "username": "user",
        "password": "pass"
    }
})

ticket_data = response.json()["ticket"]
```

## 📊 WebSocket Events

Subscribe to real-time updates:

```javascript
const ws = new WebSocket("ws://localhost:8002/ws");

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.type, data.message);
};
```

Event types:
- `project_status` - Project creation/cloning updates
- `kb_status` - Knowledge base indexing updates
- `valueedge_scraping` - Ticket scraping progress
- `workflow_event` - Autonomous workflow events

## 🛠️ Development

### Run Tests
```powershell
pytest tests/
```

### Enable Debug Mode
```python
# In main.py
app = FastAPI(debug=True)
```

### View API Docs
- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

## 📝 Notes

- Playwright runs in **headless mode** by default for production
- Screenshots are saved for debugging: `ticket_{id}.png`, `error_{id}.png`
- WebSocket broadcasts all events to connected clients
- Projects and KBs are stored in memory (add DB later)

## 🔧 Troubleshooting

### Playwright Issues
```powershell
# Reinstall browsers
playwright install --force chromium

# Run in non-headless mode for debugging
# Edit valueedge_scraper.py: headless=False
```

### Port Already in Use
```powershell
# Find process using port 8002
netstat -ano | findstr :8002

# Kill process
taskkill /PID <pid> /F
```
