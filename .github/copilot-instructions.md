# Aviator Plugin Development Guide

## Project Architecture

This is an **Aviator plugin** that extends the Aviator ADT (Agent Development Toolkit) using a **base image + plugin pattern**. The Aviator base image contains the ADT, LangGraph runtime, and infrastructure. Your plugin installs on top via entry points declared in `pyproject.toml`.

### Key Components

- **Entry Points**: All plugin capabilities are registered via `pyproject.toml` entry points (see sections below)
- **Tools**: Decorated with `@tool` from `langchain_core.tools` - simple functions the AI can call
- **Agents**: Complex tools that use sub-tools internally, created with `create_agent()` from LangGraph
- **Graph Nodes**: Custom LangGraph nodes that modify conversation flow
- **Auth/Permissions**: Content-system-specific authentication and RAG permission filtering

## Development Workflow

### Running the Stack

```bash
# Requires otl-cs-csai.json credentials file in project root
docker compose up --build

# Test health
curl http://localhost:3000/health

# Test chat
curl -X POST http://localhost:3000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"author": "user", "content": "your message"}]}'
```

The stack includes:
- `adt` service: Main Aviator API (port 3000)
- `aviator-worker`: Celery worker for async tasks
- `db`: PostgreSQL with pgvector
- `rabbitmq`: Message broker (management UI on port 15672)

### Multi-Stage Build Pattern

The Dockerfile uses two stages:
1. **Build stage**: Runs `uv build` to create distribution
2. **Runtime stage**: Installs plugin tarball into base image

The Aviator ADT is already in the base image - don't copy or install it again.

## Entry Point System

All plugin capabilities are auto-discovered via entry points in `pyproject.toml`. The entry point name must match your Content System name for auth/permission features.

### Tool Registration (`aviator.tool_modifiers`)

Register tools/agents in `extensions.py:tool_extension()`:

```python
def tool_extension(
    tools: list,
    state: StateModel | None = None,
    config: RunnableConfig | None = None,
    **kwargs: dict[str, Any],
) -> None:
    from .basic_tool import greeting_tool
    from .utility import call_utility_agent
    tools.append(greeting_tool)
    tools.append(call_utility_agent)  # Agents are tools too
```

### Startup Hooks (`aviator.startup_extensions`)

Execute initialization logic in `extensions.py:startup_extension()`:
- Load configuration
- Initialize connections
- Start background threads
- Set up resources

### Graph Extensions (`aviator.graph_extensions`)

Modify LangGraph conversation flow in `extensions.py:graph_extension()`:

```python
def graph_extension(graph: StateGraph, **kwargs: dict[str, Any]) -> None:
    graph.add_node("my_validation_node", my_validation_node)
    graph.add_edge("assistant", "my_validation_node")
```

Nodes receive `StateModel` and return modified state dict.

### Prompt Modifiers (`aviator.prompt_modifiers`)

Extend system prompts in `extensions.py:prompt_extension()`:

```python
def prompt_extension(
    prompt_name: str,
    template: ChatPromptTemplate,
    state: StateModel | None = None,
    config: RunnableConfig | None = None,
    **kwargs: dict[str, Any],
) -> None:
    match prompt_name:
        case "primary_assistant_prompt":
            system_message = next((msg for msg in template.messages 
                                  if isinstance(msg, SystemMessagePromptTemplate)), None)
            system_message.prompt.template += "Additional instructions..."
```

### Embedding Extensions (`aviator.embedding_extensions`)

Process documents during embedding in `extensions.py:embedding_extension()`:

```python
def embedding_extension(
    request: EmbeddingRequest,
    is_metadata: bool,
    **kwargs: dict[str, Any],
) -> None:
    """Process embedding requests to extract knowledge.
    
    Args:
        request: Embedding request with document data
        is_metadata: Whether embedding is metadata-only
        **kwargs: Additional arguments
    """
    metadata = request.metadata or {}
    doc_id = metadata.get("document_id", "unknown")
    doc_name = metadata.get("name", "Unknown Document")
    
    # Process document - extract entities, update knowledge graph, etc.
    kg = get_knowledge_graph()
    kg.add_entity(f"doc_{doc_id}", doc_name, "Document")
```

### Celery Tasks (`aviator.celery_imports`)

Register background tasks in `tasks.py`:

```python
from celery import Celery

app = Celery('aviator-worker')

@app.task(name='sample.process_embedding')
def process_embedding_task(embedding_data: Dict[str, Any]) -> None:
    """Background task for processing embeddings."""
    request = EmbeddingRequest(**embedding_data)
    # Process in worker...
```

Tasks run in the worker process and are registered via entry point:
```toml
[project.entry-points."aviator.celery_imports"]
"sample_tasks" = "aviator_plugin_sample.tasks"
```

### Authentication (`aviator.auth_handlers`)

Define in `auth.py` using FastAPI security schemes:

```python
from fastapi.security import APIKeyHeader

otcsticket = APIKeyHeader(name="otcsticket", auto_error=False)

def auth_otcsticket(otcsticket: Annotated[str, Security(otcsticket)]) -> dict:
    if not otcsticket:
        raise HTTPException(status_code=401, detail="Missing otcsticket header")
    return {"otcsticket": otcsticket}
```

### RAG Permission Filters (`aviator.rag_permission_filters`)

Filter RAG chunks based on user permissions in `auth.py`:

```python
def permission_filter(chunks: list[Chunk], user: dict) -> list[Chunk]:
    # Extract document IDs, check permissions, return filtered chunks
    # Must return empty list if user lacks credentials
```

### Custom REST Endpoints (`aviator.routers`)

Add FastAPI routes in `extensions.py`:

```python
router = APIRouter()

@router.get("/test", tags=["test"])
async def test_endpoint() -> JSONResponse:
    return {"status": "operational"}
```

## Tool Development Patterns

### Simple Tools

Decorated functions in their own files (e.g., `basic_tool.py`, `calculator_tool.py`):

```python
@tool
def tool_name(param: str) -> str:
    """Clear docstring describing when to use this tool.
    
    Args:
        param: Description
        
    Returns:
        Description
    """
    logger.info(f"Tool called with {param}")
    return result
```

- **Sync tools**: Regular functions
- **Async tools**: Use `async def` for API calls (see `weather_tool.py`)
- **Error handling**: Raise `ToolException` for user-facing errors

### Agent Tools

Agents are tools that use sub-tools. Pattern in `simple_agent.py`:

1. Define sub-tools with `@tool`
2. Create agent: `agent = create_agent(model=LLMRegistry.get_model(with_provider=True), tools=[...], state_schema=StateModel, middleware=[after_agent])`
3. Create caller function with `@tool(response_format="content_and_artifact")`
4. Use `InjectedState` and `InjectedToolCallId` annotations
5. Return `Command(update={"messages": [ToolMessage(...)]})`

The `after_agent` middleware extracts artifacts from tool messages.

## Code Conventions

- **Logging**: Use module-level `logger = logging.getLogger(__name__)` and log tool invocations
- **Type hints**: All function parameters and returns are typed
- **Docstrings**: Tools require comprehensive docstrings - the AI reads these to decide when to use the tool
- **Error handling**: Use `ToolException` for tool errors, `HTTPException` for API errors
- **Models**: Define Pydantic models in `models.py` for input validation (see `CalculationInput`)
- **Shared utilities**: Put reusable functions in `functions.py`
- **Node functions**: Go in `nodes.py`, receive `Annotated[StateModel, InjectedState]`, return `dict`

## Linting

Project uses Ruff with strict rules (see `pyproject.toml`). Many common rules are disabled (e.g., PTH for pathlib, E501 for line length). Run: `ruff check src/`

## Critical Implementation Details

- **StateModel**: The conversation state object passed to agents/nodes with `.messages`, `.query`, etc.
- **LLMRegistry**: Access LLM via `LLMRegistry.get_model(with_provider=True)` 
- **Command pattern**: Agents return `Command(update={...})` to modify state
- **Tool response formats**: Use `response_format="content_and_artifact"` for agents
- **Middleware**: `after_agent` middleware extracts artifacts from tool messages for agent responses
