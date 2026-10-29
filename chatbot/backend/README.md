# NemoForge Backend Engine

FastAPI-powered orchestration service for NemoForge, integrating NVIDIA Nemotron on Nebius Token Factory, Tree-sitter AST symbol graphs, and multi-zone compiler verification.

---

## Core Capabilities

- **Nebius Token Factory Integration**: Native client targeting `https://api.tokenfactory.nebius.com/v1/` for NVIDIA Nemotron 3 Super (120B MoE) and Nemotron 70B Instruct.
- **Tree-sitter AST Slicing**: Instant SQLite WAL symbol indexer querying class hierarchies and method call graphs in <1s.
- **Polyglot Domain Disambiguation**: Language-aware intent classifier separating backend Java Spring Boot exceptions from frontend React TypeScript components.
- **Surgical Patch Engine**: Fail-closed 3-line search/replace anchors for precise code synthesis.
- **Multi-Zone Compiler Gate**: Live sandboxed execution of `mvn compile`, `gradle build`, and `npm test` with automated diagnostic repair loops.
- **Live WebSocket Dispatch**: Real-time event streaming powering the frontend ReAct Flow DAG visualizer.

---

## Key API Endpoints

### Project & Indexing
- `GET /api/projects`: List registered projects and indexing status.
- `POST /api/projects`: Register a new local or git workspace.
- `POST /api/projects/{id}/index`: Execute instant Tree-sitter AST indexing.
- `DELETE /api/projects/{id}`: Unregister project and purge `.nemoforge/index.db`.

### Intent & Workflows
- `POST /api/intent/classify`: Classify user prompt intent (`CODE_CHANGE` vs `EXPLAIN`).
- `POST /api/workflow/start`: Launch autonomous SWE workflow on target ticket.
- `GET /api/workflow/{id}/events`: Retrieve event stream for ReAct workflow.
- `GET /api/workflow/transparent/{id}/changes`: Retrieve native Git diff of staged modifications.
- `POST /api/workflow/transparent/{id}/changes/approve`: Selectively approve or reject staged file hunks before disk flush.

### Real-Time Updates
- `WS /ws`: WebSocket endpoint broadcasting live ReAct agent lifecycle events.

---

## Running Standalone

From the repository root:
```bash
# Windows
.venv\Scripts\uvicorn.exe main:app --app-dir chatbot\backend --host 0.0.0.0 --port 8002 --reload

# Linux / macOS
source .venv/bin/activate
python -m uvicorn main:app --app-dir chatbot/backend --host 0.0.0.0 --port 8002 --reload
```

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`
