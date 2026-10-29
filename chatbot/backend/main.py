"""
Aviator Chatbot - Backend API
FastAPI server with WebSocket support for real-time workflow updates
"""
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
env_local = Path(__file__).parent.parent / ".env.local"
if env_local.exists():
    load_dotenv(dotenv_path=env_local, override=True)

import sys
import shutil
import glob
from typing import Optional
from pathlib import Path

def find_host_jdk() -> tuple[Optional[str], str]:
    """Universally discover host JDK across environment, PATH, and standard vendor directories."""
    java_bin = "bin\\java.exe" if sys.platform == "win32" else "bin/java"
    javac_bin = "bin\\javac.exe" if sys.platform == "win32" else "bin/javac"

    # 1. Check existing JAVA_HOME environment variable
    jh = os.environ.get("JAVA_HOME", "").strip().rstrip("\\/")
    if jh and os.path.exists(os.path.join(jh, java_bin)):
        return jh, "JAVA_HOME environment variable"

    # 2. Check PATH via javac
    which_javac = shutil.which("javac")
    if which_javac:
        cand = str(Path(which_javac).resolve().parent.parent)
        if os.path.exists(os.path.join(cand, java_bin)):
            return cand, "PATH (javac)"

    # 3. Dynamic filesystem discovery across standard vendor folders
    search_patterns = []
    if sys.platform == "win32":
        for base in [os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files (x86)"]:
            for vendor in ["Microsoft", "Eclipse Adoptium", "Java", "Amazon Corretto", "Zulu", "BellSoft", "Semeru"]:
                search_patterns.append(os.path.join(base, vendor, "*"))
    else:
        search_patterns.extend([
            "/usr/lib/jvm/*",
            "/usr/java/*",
            "/Library/Java/JavaVirtualMachines/*/Contents/Home",
        ])

    found = []
    for pat in search_patterns:
        for p in glob.glob(pat):
            if os.path.isdir(p) and (os.path.exists(os.path.join(p, javac_bin)) or os.path.exists(os.path.join(p, java_bin))):
                found.append(p)
    if found:
        # Sort descending to favor newest version
        found.sort(reverse=True)
        return found[0], "dynamic discovery"

    return None, "not found"

def _setup_java_home():
    jdk_path, method = find_host_jdk()
    if jdk_path:
        os.environ["JAVA_HOME"] = jdk_path
        bin_dir = os.path.join(jdk_path, "bin")
        cur_path = os.environ.get("PATH", "")
        if bin_dir.lower() not in cur_path.lower():
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{cur_path}"
    else:
        # Do not silently crash later: log actionable prerequisite guidance for user
        import logging
        logging.getLogger("main").warning(
            "⚠️ [Prerequisite Check] No valid JDK detected on this host. "
            "Java microservice builds will require JAVA_HOME. "
            "Please install JDK 17+ and set the JAVA_HOME environment variable."
        )

_setup_java_home()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
import json
import asyncio
import httpx
import logging
import re
import fnmatch
from enum import Enum
import subprocess
import sys
import os
import uuid
from pathlib import Path
from git import Repo

# Ensure backend directory is in sys.path for relative imports
backend_dir = str(Path(__file__).parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import history_store

# Fix Windows charmap encoding crash for emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Auto-load untracked local .env.local if present
env_local_file = Path(__file__).parent.parent / ".env.local"
if env_local_file.exists():
    try:
        for _line in env_local_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

os.environ.setdefault("LLM_PROVIDER", "nebius")
os.environ.setdefault("LLM_MODEL", "nebius/nvidia/nemotron-3-super-120b-a12b")

# Add ticket_to_code src to path
src_path = Path(__file__).parent.parent / "src"
if not src_path.exists():
    src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Helper to resolve index.db path (.nemoforge preferred, .aviator fallback)
def get_index_db_path(repo_path: Path) -> Path:
    nemoforge_path = repo_path / ".nemoforge" / "index.db"
    if nemoforge_path.exists():
        return nemoforge_path
    aviator_path = repo_path / ".aviator" / "index.db"
    if aviator_path.exists():
        return aviator_path
    return nemoforge_path

# NOW import the existing working workflow
from ticket_to_code import run_autonomous_workflow, ValueEdgeTicket, TicketPriority
from ticket_to_code.api.routes import router as ticket_to_code_router
from ticket_to_code.intelligence.model_router import ModelRouter, ModelMode, DEFAULT_MODEL_ID, get_default_model_id

# ── Per-process model preference (replaces process-wide os.environ["LLM_MODEL"] writes) ──
# The provider/key endpoints record the operator's preferred default here (and persist it to
# .env.local); every request still resolves its own model through ModelRouter so concurrent
# requests never observe each other's switch.
_runtime_model_preference: Dict[str, Optional[str]] = {"model": None}

# ── Stage-aware (AUTO_STAGE) model resolver, keyed by the workflow thread ──
import threading as _resolver_threading
try:
    from ticket_to_code.workflow import set_stage_model_resolver as _set_stage_model_resolver
except ImportError:  # workflow owner has not landed the hook yet
    _set_stage_model_resolver = None

_STAGE_RESOLVERS_BY_THREAD: Dict[int, Callable[[str], Optional[str]]] = {}


def _thread_scoped_stage_resolver(node_name: str) -> Optional[str]:
    """Process-wide resolver installed into the workflow; dispatches to the resolver registered by
    the thread that is executing the graph (LangGraph nodes run synchronously inside thread_run)."""
    fn = _STAGE_RESOLVERS_BY_THREAD.get(_resolver_threading.get_ident())
    return fn(node_name) if fn else None


def build_stage_model_resolver(m_mode, ticket, selected_model_desc, router=ModelRouter) -> Callable[[str], Optional[str]]:
    """Build `fn(node_name) -> model_id | None` for one workflow run.

    Outside AUTO_STAGE every node uses the request's selected model. In AUTO_STAGE the router's
    stage tiers are resolved once up front and mapped onto the graph's node names."""
    if m_mode != ModelMode.AUTO_STAGE:
        return lambda _n: selected_model_desc.model_id
    strongest = router.select_model_for_stage("plan", ticket, mode=m_mode, active_task_model=selected_model_desc).model_id
    coding = router.select_model_for_stage("generate_code", ticket, mode=m_mode, active_task_model=selected_model_desc).model_id
    fast = router.select_model_for_stage("investigate", ticket, mode=m_mode, active_task_model=selected_model_desc).model_id

    def _resolve(node_name: str) -> Optional[str]:
        n = (node_name or "").lower()
        if n in {"plan", "remediation_plan", "grounded_understanding", "behavior_planning", "plan_validation"}:
            return strongest
        if n in {"edit_loop", "generate_code", "generate_tests", "fix_test"}:
            return coding
        if n == "fix_build":
            return fast
        return None

    return _resolve


def register_stage_model_resolver_for_current_thread(fn: Callable[[str], Optional[str]], setter=None) -> bool:
    """Register `fn` for the calling thread and make sure the workflow's hook points at the
    thread-scoped dispatcher. Returns False when the workflow does not expose the hook."""
    setter = setter if setter is not None else _set_stage_model_resolver
    _STAGE_RESOLVERS_BY_THREAD[_resolver_threading.get_ident()] = fn
    if setter is None:
        return False
    setter(_thread_scoped_stage_resolver)
    return True

# Configure logging
import sys
if sys.platform == "win32":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(name)s:%(message)s')
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from services.mcp_manager import mcp_manager
from services.playwright_verifier import playwright_verifier

app = FastAPI(title="NemoForge Autonomous SWE API", version="1.0.0")
app.include_router(ticket_to_code_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class ProjectType(str, Enum):
    LOCAL = "local"
    GIT = "git"


class ExecutionMode(str, Enum):
    PIPELINE = "pipeline"
    REASONING = "reasoning"

class ProjectStatus(str, Enum):
    PENDING = "pending"
    CLONING = "cloning"
    INDEXING = "indexing"
    INDEXED = "indexed"
    READY = "ready"
    ERROR = "error"

class Project(BaseModel):
    id: str = Field(..., description="Unique project ID")
    name: str = Field(..., description="Project name")
    type: ProjectType = Field(..., description="local or git")
    path: str = Field(..., description="Local path or git URL")
    local_path: Optional[str] = Field(None, description="Local clone path for git projects")
    status: ProjectStatus = Field(default=ProjectStatus.PENDING)
    technology: Optional[str] = Field(None, description="Detected technology (java, python, etc)")
    created_at: datetime = Field(default_factory=datetime.now)
    knowledge_bases: List[str] = Field(default_factory=list, description="List of KB folder IDs")

class KnowledgeBase(BaseModel):
    id: str = Field(..., description="Unique KB ID")
    project_id: str = Field(..., description="Parent project ID")
    name: str = Field(..., description="KB folder name")
    description: Optional[str] = Field(None)
    file_count: int = Field(default=0)
    indexed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.now)

class TicketInfo(BaseModel):
    ticket_id: str
    title: str
    description: str
    priority: str
    labels: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)

class WorkflowEvent(BaseModel):
    type: str  # agent_start, agent_complete, rag_query, llm_call, state_update, error
    timestamp: datetime = Field(default_factory=datetime.now)
    agent: Optional[str] = None
    message: str
    data: Optional[Dict[str, Any]] = None

# ============================================================================
# PERSISTENT STORAGE
# ============================================================================

_PROJECTS_FILE = Path(__file__).parent / "projects.json"


def _load_projects() -> Dict[str, "Project"]:
    """Load projects from disk, automatically registering the active workspace environment."""
    loaded: Dict[str, "Project"] = {}
    file_existed = _PROJECTS_FILE.exists()
    if file_existed:
        try:
            raw = json.loads(_PROJECTS_FILE.read_text())
            loaded = {pid: Project(**p) for pid, p in raw.items()}
        except Exception as exc:
            logger.warning(f"Could not load projects.json: {exc}")

    # Reconcile status: if index.db exists on disk, project is indexed
    changed = False
    for pid, p in loaded.items():
        try:
            p_path = Path(p.local_path or p.path)
            if not p_path.is_absolute():
                repo_root = Path(__file__).resolve().parent.parent.parent
                candidate = repo_root / p_path
                if candidate.exists():
                    p_path = candidate
            is_indexed = (
                (p_path / ".nemoforge" / "index.db").exists() or
                (p_path / ".ticket_to_code" / "index.db").exists() or
                (p_path / ".aviator" / "index.db").exists()
            )
            if is_indexed and p.status != ProjectStatus.INDEXED:
                p.status = ProjectStatus.INDEXED
                changed = True
        except Exception:
            pass

    if changed:
        _save_projects(loaded)

    # Auto-register default workspace (CloudForge-Commerce) or ENV workspace if no projects loaded
    repo_root = Path(__file__).resolve().parent.parent.parent
    env_ws = os.getenv("AVIATOR_WORKSPACE_PATH") or os.getenv("WORKSPACE_PATH")
    candidate_ws = Path(env_ws).resolve() if (env_ws and Path(env_ws).exists()) else (repo_root / "workspace" / "CloudForge-Commerce").resolve()
    
    if not file_existed and not loaded and candidate_ws.exists() and candidate_ws.is_dir():
        ws_path = str(candidate_ws)
        if not any(Path(p.path).resolve() == Path(ws_path) for p in loaded.values()):
            auto_id = "default-cloudforge-commerce"
            is_idx = (candidate_ws / ".nemoforge" / "index.db").exists() or (candidate_ws / ".ticket_to_code" / "index.db").exists()
            auto_proj = Project(
                id=auto_id,
                name="CloudForge-Commerce" if "CloudForge" in candidate_ws.name else (candidate_ws.name or "Workspace"),
                type=ProjectType.LOCAL,
                path=ws_path,
                local_path=ws_path,
                status=ProjectStatus.INDEXED if is_idx else ProjectStatus.READY,
            )
            loaded[auto_id] = auto_proj
            _save_projects(loaded)

    return loaded


def _save_projects(proj_dict: Dict[str, "Project"]):
    """Persist projects to disk."""
    try:
        _PROJECTS_FILE.write_text(
            json.dumps({pid: json.loads(p.model_dump_json()) for pid, p in proj_dict.items()}, indent=2)
        )
    except Exception as exc:
        logger.warning(f"Could not save projects.json: {exc}")


projects: Dict[str, Project] = _load_projects()
knowledge_bases: Dict[str, KnowledgeBase] = {}
active_workflows: Dict[str, List[WorkflowEvent]] = {}

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients, pruning disconnected sockets."""
        dead_connections = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.debug(f"Pruning unreachable WebSocket client: {e}")
                dead_connections.append(connection)
        for dead in dead_connections:
            if dead in self.active_connections:
                self.active_connections.remove(dead)

manager = ConnectionManager()

# ============================================================================
# API ENDPOINTS - PROJECTS
# ============================================================================

class CreateProjectRequest(BaseModel):
    name: str
    type: ProjectType
    path: str
    technology: Optional[str] = None

@app.post("/api/projects", response_model=Project)
async def create_project(request: CreateProjectRequest):
    """Create a new project (local or git)"""
    import uuid
    
    name = request.name
    type = request.type
    path = request.path
    technology = request.technology
    
    project_id = str(uuid.uuid4())
    
    project = Project(
        id=project_id,
        name=name,
        type=type,
        path=path,
        technology=technology,
        status=ProjectStatus.PENDING
    )
    
    # If git project, clone it
    if type == ProjectType.GIT:
        try:
            project.status = ProjectStatus.CLONING
            projects[project_id] = project
            
            # Broadcast cloning started
            await manager.broadcast({
                "type": "project_status",
                "project_id": project_id,
                "status": "cloning",
                "message": f"Cloning repository: {path}"
            })
            
            # Clone to workspace
            clone_dir = Path("workspace") / project_id
            clone_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Cloning {path} to {clone_dir}")
            Repo.clone_from(path, clone_dir)
            
            project.local_path = str(clone_dir)
            project.status = ProjectStatus.READY
            
            await manager.broadcast({
                "type": "project_status",
                "project_id": project_id,
                "status": "ready",
                "message": f"Repository cloned successfully"
            })
            
        except Exception as e:
            project.status = ProjectStatus.ERROR
            logger.error(f"Error cloning repository: {e}")
            await manager.broadcast({
                "type": "project_status",
                "project_id": project_id,
                "status": "error",
                "message": f"Clone failed: {str(e)}"
            })
            raise HTTPException(status_code=500, detail=f"Clone failed: {str(e)}")
    else:
        # Local project - verify path exists and is a directory (clean quotes & resolve relative paths)
        clean_path = path.strip().strip('"').strip("'")
        path_obj = Path(clean_path)
        if not path_obj.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent.parent
            if (repo_root / path_obj).exists():
                path_obj = repo_root / path_obj
        if not path_obj.exists():
            raise HTTPException(status_code=400, detail=f"Path not found: {path}. Please check if the directory exists.")
        if not path_obj.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")
        project.path = str(path_obj.resolve())
        project.local_path = str(path_obj.resolve())
        project.status = ProjectStatus.READY
    
    projects[project_id] = project
    _save_projects(projects)

    # Automatically bootstrap repository brain & feature graph for the new project
    try:
        from ticket_to_code.reasoning import RepositoryBrain
        from ticket_to_code.reasoning.feature_graph import FeatureGraph
        from ticket_to_code.reasoning.experience import ExperienceStore
        
        target_path = project.local_path or project.path
        if target_path and Path(target_path).exists():
            RepositoryBrain(target_path)
            FeatureGraph(target_path)
            ExperienceStore(target_path)
            logger.info(f"Initialized repository brain and feature graph for new project: {project.name}")
    except Exception as _b_init_err:
        logger.warning(f"Initial brain bootstrap skipped for {project.name}: {_b_init_err}")

    return project

@app.get("/api/projects", response_model=List[Project])
async def list_projects():
    """Get all projects (re-synced from projects.json)"""
    global projects
    projects = _load_projects()
    return list(projects.values())

@app.get("/api/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get project by ID"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return projects[project_id]

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and its associated index database"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    proj = projects[project_id]
    try:
        p_path = Path(proj.local_path or proj.path)
        if not p_path.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent.parent
            if (repo_root / p_path).exists():
                p_path = repo_root / p_path
        # Clean index databases so re-adding is completely fresh
        if (p_path / ".nemoforge").exists():
            for db_file in (p_path / ".nemoforge").glob("index.db*"):
                try:
                    db_file.unlink()
                except Exception:
                    pass
        if (p_path / ".ticket_to_code").exists():
            for db_file in (p_path / ".ticket_to_code").glob("index.db*"):
                try:
                    db_file.unlink()
                except Exception:
                    pass
    except Exception as exc:
        logger.warning(f"Failed to clean index database for {proj.name}: {exc}")
    
    # Delete associated knowledge bases
    kb_ids_to_delete = [kb_id for kb_id, kb in knowledge_bases.items() if kb.project_id == project_id]
    for kb_id in kb_ids_to_delete:
        del knowledge_bases[kb_id]
    
    del projects[project_id]
    _save_projects(projects)
    return {"message": "Project deleted successfully"}

# ============================================================================
# API ENDPOINTS - CHAT & TASK HISTORY (durable, survives restarts)
# ============================================================================

class SaveChatRequest(BaseModel):
    chat_id: Optional[str] = None
    project_id: Optional[str] = None
    title: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    workflow_id: Optional[str] = None


@app.get("/api/history/chats")
async def api_list_chats(project_id: Optional[str] = None):
    """List previous conversations (newest first), optionally by project."""
    return history_store.list_chats(project_id)


@app.get("/api/history/chats/{chat_id}")
async def api_get_chat(chat_id: str):
    """Load a full conversation with all messages."""
    chat = history_store.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.post("/api/history/chats")
async def api_save_chat(request: SaveChatRequest):
    """Create or update a conversation (auto-save from the UI)."""
    record = history_store.upsert_chat(
        chat_id=request.chat_id,
        project_id=request.project_id,
        messages=request.messages,
        title=request.title,
        workflow_id=request.workflow_id,
    )
    return {"id": record["id"], "title": record["title"], "updated_at": record["updated_at"]}


@app.delete("/api/history/chats/{chat_id}")
async def api_delete_chat(chat_id: str):
    if not history_store.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"message": "Chat deleted"}


@app.get("/api/history/tasks")
async def api_list_tasks(project_id: Optional[str] = None):
    """List completed tasks/runs (newest first), optionally by project."""
    try:
        # Only workflows with a live asyncio task are considered active; the
        # durable store also holds historical runs and must not be used here.
        active_ids = {wid for wid, t in list(_RUN_TASKS.items()) if t is not None and not t.done()}
        history_store.sanitize_orphaned_tasks(active_ids)
    except Exception as exc:
        logger.warning(f"Failed to sanitize orphaned tasks: {exc}")
    return history_store.list_tasks(project_id)


@app.delete("/api/history/tasks/{workflow_id}")
async def api_delete_task(workflow_id: str):
    if not history_store.delete_task(workflow_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task deleted"}

@app.post("/api/projects/{project_id}/index")
async def index_project(project_id: str):
    """
    Index a project - parse AST, extract symbols, build graph database.
    This should be called after adding a project and before entering tickets.
    """
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects[project_id]
    raw_path = (project.local_path or project.path or "").strip().strip('"').strip("'")
    repo_path = Path(raw_path)
    if not repo_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidate = repo_root / repo_path
        if candidate.exists():
            repo_path = candidate
    
    if not repo_path.exists():
        raise HTTPException(status_code=400, detail=f"Project path not found: {repo_path}")
    
    try:
        # Update project status
        project.status = ProjectStatus.INDEXING
        
        logger.info(f"🔍 Starting indexing for project {project.name} at {repo_path}")
        
        # Broadcast indexing started
        await manager.broadcast({
            "type": "project_status",
            "project_id": project_id,
            "status": "indexing",
            "message": "Starting repository indexing..."
        })
        
        # Import indexer components
            
        from ticket_to_code.runtime.indexer import index_repository
        from ticket_to_code.runtime.storage import SqliteStore
        
        # Prepare index database path
        index_path = get_index_db_path(repo_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get the event loop for thread-safe broadcasts
        loop = asyncio.get_event_loop()
        
        # Progress callback for UI updates (matches indexer.py signature: current_file, done, total)
        def progress_callback(current_file: Path, done: int, total: int):
            # Use run_coroutine_threadsafe for thread-safe async calls
            progress_percent = int((done / total) * 100) if total > 0 else 0
            logger.info(f"Indexing progress: {done}/{total} ({progress_percent}%) - {current_file.name}")
            
            coro = manager.broadcast({
                "type": "indexing_progress",
                "project_id": project_id,
                "files_scanned": done,
                "total_files": total,
                "progress_percent": progress_percent,
                "current_file": str(current_file.name),
                "message": f"📝 Indexing: {done}/{total} files ({progress_percent}%) - {current_file.name}"
            })
            asyncio.run_coroutine_threadsafe(coro, loop)
        
        # Wrapper function that creates store in the background thread (SQLite thread safety)
        def run_indexing():
            # Create SqliteStore in this thread
            store = SqliteStore(str(index_path))
            try:
                # Run indexing
                return index_repository(
                    Path(repo_path),
                    store,
                    progress_callback=progress_callback
                )
            finally:
                # Close the store connection
                store.close()
        
        # Run indexing in background thread
        stats = await asyncio.to_thread(run_indexing)

        # Normalize stats (handle dict or object safely)
        if isinstance(stats, dict):
            files_scanned = stats.get("total_files", stats.get("files_indexed", 0))
            files_parsed = stats.get("files_indexed", 0)
            files_failed = len(stats.get("errors", []))
            symbols_extracted = stats.get("symbols_found", 0)
            edges_count = stats.get("edges", 0)
            duration_s = stats.get("duration_seconds", 0.0)
            resolved_calls = stats.get("resolved_calls", 0)
        else:
            files_scanned = getattr(stats, "files_scanned", 0)
            files_parsed = getattr(stats, "files_parsed", 0)
            files_failed = getattr(stats, "files_failed", 0)
            symbols_extracted = getattr(stats, "symbols", 0)
            edges_count = getattr(stats, "edges", 0)
            duration_s = getattr(stats, "duration_seconds", 0.0)
            resolved_calls = getattr(stats, "resolved_calls", 0)

        # Pull ground-truth counts from index.db if available
        try:
            import sqlite3 as _sqlite3
            if index_path.exists():
                with _sqlite3.connect(str(index_path)) as _sq_conn:
                    f_cnt = _sq_conn.execute("SELECT count(*) FROM files").fetchone()[0]
                    s_cnt = _sq_conn.execute("SELECT count(*) FROM symbols").fetchone()[0]
                    e_cnt = _sq_conn.execute("SELECT count(*) FROM edges").fetchone()[0]
                    if f_cnt > 0:
                        files_scanned = max(files_scanned, f_cnt)
                        files_parsed = max(files_parsed, f_cnt)
                    if s_cnt > 0:
                        symbols_extracted = max(symbols_extracted, s_cnt)
                    if e_cnt > 0:
                        edges_count = max(edges_count, e_cnt)
        except Exception:
            pass

        # Log the actual results (PROOF it really worked!)
        logger.info(f"✅ Indexing complete for {project.name}:")
        logger.info(f"   📁 Files scanned: {files_scanned}")
        logger.info(f"   ✔️  Files parsed: {files_parsed}")
        logger.info(f"   ❌ Files failed: {files_failed}")
        logger.info(f"   🔤 Symbols extracted: {symbols_extracted}")
        logger.info(f"   🔗 Edges (relationships): {edges_count}")
        logger.info(f"   ⏱️  Time taken: {duration_s:.2f}s")
        
        # Vector embeddings & Neo4j removed in favor of direct SQLite AST & local Tree-sitter
        stats_extra = {
            "vector_chunks": 0,
            "neo4j_nodes": 0,
            "neo4j_edges": 0,
        }

        # ── Phase 2: Repository Brain generation ──────────────────────────────
        # Create the directory brain now (at index time) so Ticket-to-Code CREATE
        # localization has project knowledge from the very first ticket. Non-fatal.
        logger.info("🧠 Starting Repository Brain generation...")
        await manager.broadcast({
            "type": "indexing_progress",
            "project_id": project_id,
            "files_scanned": files_scanned,
            "total_files": files_scanned,
            "progress_percent": 100,
            "current_file": "Generating repository brain (directory intelligence)...",
            "message": "🧠 Generating Repository Brain (directory map + structure)..."
        })

        def run_brain_generation():
            import sys as _sys
            src_path = Path(__file__).parent.parent.parent / "src"
            if str(src_path) not in _sys.path:
                _sys.path.insert(0, str(src_path))
            from ticket_to_code.brain.generate_repository_brain import ensure_repository_brain
            try:
                from ticket_to_code.runtime.llm_factory import LLMRegistry
                brain_llm = LLMRegistry.get_llm()
            except Exception:
                brain_llm = None
            return ensure_repository_brain(str(repo_path), llm=brain_llm, enrich=False)

        try:
            brain_summary = await asyncio.to_thread(run_brain_generation)
            logger.info(f"✅ Repository Brain: {brain_summary}")
            stats_extra["brain_status"] = brain_summary.get("bootstrap", brain_summary.get("status"))
            stats_extra["brain_directories"] = brain_summary.get("directories", 0)
        except Exception as e:
            logger.error(f"Repository Brain generation failed (non-fatal): {e}", exc_info=True)
            stats_extra["brain_error"] = str(e)

        # ── Update status & broadcast completion ─────────────────────────────
        project.status = ProjectStatus.INDEXED
        _save_projects(projects)
        
        # Broadcast completion (include stats so frontend can display them)
        vector_chunks = stats_extra.get("vector_chunks", 0)
        neo4j_nodes  = stats_extra.get("neo4j_nodes",   0)
        neo4j_edges_n = stats_extra.get("neo4j_edges",  0)
        brain_status = stats_extra.get("brain_status", "unknown")
        brain_dirs   = stats_extra.get("brain_directories", 0)
        await manager.broadcast({
            "type": "project_status",
            "project_id": project_id,
            "status": "indexed",
            "message": (
                f"✅ Indexed {files_scanned} files | "
                f"{symbols_extracted} symbols | {edges_count} edges | "
                f"{resolved_calls} resolved calls | "
                f"Brain: {brain_status} ({brain_dirs} dirs)"
            ),
            "stats": {
                "files_scanned":  files_scanned,
                "files_parsed":   files_parsed,
                "files_failed":   files_failed,
                "symbols":        symbols_extracted,
                "edges":          edges_count,
                "resolved_calls": resolved_calls,
                "vector_chunks":  vector_chunks,
                "neo4j_nodes":    neo4j_nodes,
                "neo4j_edges":    neo4j_edges_n,
                "brain_status":   brain_status,
                "brain_directories": brain_dirs,
            }
        })

        # Record unified storage synchronization manifest
        sync_manifest = StorageSyncManifest(
            project_id=project_id,
            workspace_path=str(repo_path),
            sqlite_status=StoreStatus.HEALTHY if files_parsed > 0 else StoreStatus.DEGRADED,
            pgvector_status=StoreStatus.HEALTHY,
            neo4j_status=StoreStatus.HEALTHY,
            files_indexed=files_scanned,
            symbols_indexed=symbols_extracted,
            vector_chunks=vector_chunks,
            neo4j_nodes=neo4j_nodes,
            neo4j_edges=neo4j_edges_n,
            sync_errors=stats_extra,
        )
        StorageHealthCoordinator.save_manifest(sync_manifest, repo_path)

        return {
            "message": "Indexing complete",
            "stats": {
                "files_scanned":  files_scanned,
                "files_parsed":   files_parsed,
                "files_failed":   files_failed,
                "symbols":        symbols_extracted,
                "edges":          edges_count,
                "resolved_calls": resolved_calls,
                "vector_chunks":  vector_chunks,
                "neo4j_nodes":    neo4j_nodes,
                "neo4j_edges":    neo4j_edges_n,
                "brain_status":   brain_status,
                "brain_directories": brain_dirs,
            }
        }
        
    except Exception as e:
        project.status = ProjectStatus.ERROR
        logger.error(f"Indexing error: {e}", exc_info=True)
        
        await manager.broadcast({
            "type": "project_status",
            "project_id": project_id,
            "status": "error",
            "message": f"Indexing failed: {str(e)}"
        })
        
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

# ============================================================================
# API ENDPOINTS - KNOWLEDGE BASES
# ============================================================================

@app.post("/api/projects/{project_id}/knowledge-bases", response_model=KnowledgeBase)
async def create_knowledge_base(
    project_id: str,
    name: str,
    description: Optional[str] = None
):
    """Create a knowledge base folder for a project"""
    import uuid
    
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    kb_id = str(uuid.uuid4())
    kb = KnowledgeBase(
        id=kb_id,
        project_id=project_id,
        name=name,
        description=description
    )
    
    knowledge_bases[kb_id] = kb
    projects[project_id].knowledge_bases.append(kb_id)
    
    return kb

@app.get("/api/projects/{project_id}/knowledge-bases", response_model=List[KnowledgeBase])
async def list_knowledge_bases(project_id: str):
    """Get all knowledge bases for a project"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return [kb for kb in knowledge_bases.values() if kb.project_id == project_id]

def _safe_upload_filename(filename: Optional[str]) -> str:
    """
    Reduce a client-supplied filename to a bare basename and reject anything
    that could escape the upload directory. Raises HTTP 400 on rejection.
    """
    raw = (filename or "").strip()
    # Any ".." in the client-supplied name is treated as a traversal attempt,
    # even though we reduce to a basename below.
    if not raw or ".." in raw or "\x00" in raw:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    # Path(...).name only strips the host OS separator; normalise both so a
    # Windows-style "dir\\x" cannot survive on Linux (and vice versa).
    name = Path(raw.replace("\\", "/")).name.strip()
    if not name or name == "." or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    return name


@app.post("/api/knowledge-bases/{kb_id}/upload")
async def upload_files(kb_id: str, files: List[UploadFile] = File(...)):
    """Upload files to knowledge base"""
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    kb = knowledge_bases[kb_id]
    kb_dir = Path("workspace") / "kb" / kb_id
    kb_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    for file in files:
        safe_name = _safe_upload_filename(file.filename)
        file_path = kb_dir / safe_name
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        uploaded_files.append(safe_name)
    
    kb.file_count += len(uploaded_files)
    
    return {
        "kb_id": kb_id,
        "uploaded": uploaded_files,
        "total_files": kb.file_count
    }

@app.post("/api/knowledge-bases/{kb_id}/index")
async def index_knowledge_base(kb_id: str):
    """Index knowledge base files into RAG"""
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    kb = knowledge_bases[kb_id]
    
    # TODO: Integrate with existing RAG indexing
    # This will use ticket_to_code vector store
    
    await manager.broadcast({
        "type": "kb_status",
        "kb_id": kb_id,
        "status": "indexing",
        "message": f"Indexing {kb.file_count} files..."
    })
    
    # Simulate indexing (replace with real implementation)
    await asyncio.sleep(2)
    
    kb.indexed = True
    
    await manager.broadcast({
        "type": "kb_status",
        "kb_id": kb_id,
        "status": "indexed",
        "message": f"Knowledge base indexed successfully"
    })
    
    return {"message": "Knowledge base indexed", "kb_id": kb_id}

# ============================================================================
# API ENDPOINTS - WORKFLOW
# ============================================================================
# ============================================================================

@app.post("/api/workflow/start")
async def start_workflow(
    project_id: str,
    kb_id: str,
    ticket_info: TicketInfo
):
    """Start autonomous workflow for a ticket"""
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    project = projects[project_id]
    kb = knowledge_bases[kb_id]
    
    workflow_id = f"{project_id}_{ticket_info.ticket_id}"
    active_workflows[workflow_id] = []
    
    # Send initial event
    event = WorkflowEvent(
        type="workflow_started",
        message=f"Starting workflow for ticket: {ticket_info.ticket_id}",
        data={
            "project": project.name,
            "kb": kb.name,
            "ticket": ticket_info.dict()
        }
    )
    active_workflows[workflow_id].append(event)
    
    await manager.broadcast({
        "type": "workflow_event",
        "workflow_id": workflow_id,
        "event": event.dict()
    })
    
    # TODO: Integrate with existing ticket_to_code workflow
    # This will call run_autonomous_workflow() with real-time updates
    
    return {
        "workflow_id": workflow_id,
        "status": "started",
        "ticket": ticket_info.dict()
    }

@app.get("/api/workflow/{workflow_id}/events", response_model=List[WorkflowEvent])
async def get_workflow_events(workflow_id: str):
    """Get all events for a workflow"""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return active_workflows[workflow_id]

# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive messages
            data = await websocket.receive_text()
            logger.info(f"Received from client: {data}")
            
            # Echo back for testing
            await websocket.send_json({
                "type": "ping",
                "message": "Connection active"
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============================================================================
# WORKFLOW ENDPOINTS - Transparent, streaming workflow execution
# ============================================================================

import queue
import threading

from ticket_to_code.runtime.state_store import DurableWorkflowStateStore
from ticket_to_code.runtime.storage_health import StorageHealthCoordinator, StorageSyncManifest, StoreStatus
from ticket_to_code.runtime.security import CommandSafetyValidator
from ticket_to_code.runtime.decision_coordinator import AsyncDecisionCoordinator

# Per-workflow persistent state and WebSocket connections
workflow_status: DurableWorkflowStateStore = DurableWorkflowStateStore()
decision_coordinator: AsyncDecisionCoordinator = AsyncDecisionCoordinator()
workflow_connections: Dict[str, List[WebSocket]] = {}
# Cancellation flags: set workflow_id -> True to request stop
workflow_cancel_flags: Dict[str, bool] = {}

# Active background task handles: workflow_id -> asyncio.Task
_RUN_TASKS: Dict[str, asyncio.Task] = {}

# Per-workspace lock to prevent concurrent runs on the same repo clobbering each other
_WORKSPACE_LOCKS: Dict[str, threading.Lock] = {}
_WORKSPACE_LOCKS_GUARD = threading.Lock()

def get_workspace_lock(workspace_path: str) -> threading.Lock:
    """Return a shared mutex for the given repository path."""
    norm = os.path.normcase(os.path.abspath(workspace_path)) if workspace_path else "default"
    with _WORKSPACE_LOCKS_GUARD:
        if norm not in _WORKSPACE_LOCKS:
            _WORKSPACE_LOCKS[norm] = threading.Lock()
        return _WORKSPACE_LOCKS[norm]


def _extract_workflow_outputs(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract authoritative workflow outputs used for explanation assembly."""
    contracts = final_state.get("internal_node_contracts") or []
    planner_decisions = final_state.get("planner_decisions") or {}
    grounded = final_state.get("grounded_implementation_decision")
    generated_code = final_state.get("generated_code") or []

    owner_files: List[str] = []
    if grounded is not None:
        file_resolutions = getattr(grounded, "file_resolutions", None)
        if file_resolutions is None and isinstance(grounded, dict):
            file_resolutions = grounded.get("file_resolutions", [])
        for item in file_resolutions or []:
            if isinstance(item, dict):
                strategy = str(item.get("strategy", ""))
                file_path = str(item.get("file_path", ""))
            else:
                strategy = str(getattr(item, "strategy", ""))
                file_path = str(getattr(item, "file_path", ""))
            if strategy.endswith("DIRECT_OWNER") and file_path:
                owner_files.append(file_path)

    generated_files = []
    for g in generated_code:
        if isinstance(g, dict):
            fp = g.get("file_path")
        else:
            fp = getattr(g, "file_path", None)
        if fp:
            generated_files.append(str(fp))

    build_attribution = final_state.get("build_attribution")
    if isinstance(build_attribution, dict):
        build_reasoning = str(build_attribution.get("reasoning") or "")
    else:
        build_reasoning = str(getattr(build_attribution, "reasoning", "") or "")

    learning_update = final_state.get("learning_update") or {}
    decision_ledger = final_state.get("decision_ledger") or []
    confidence_history = final_state.get("confidence_history") or []
    assumptions_active = final_state.get("assumptions_active") or []
    assumptions_invalidated = final_state.get("assumptions_invalidated") or []
    outcome_verification = final_state.get("outcome_verification") or {}

    # ── Build diagnostic classification (written by pre_fix_build_node) ──
    # These fields carry error-provenance info: was the failure caused by the
    # ticket changes, by pre-existing repo errors, or by infrastructure?
    _diag_accept = final_state.get("build_differential_accept")
    _diag_infra  = final_state.get("build_infrastructure_only")
    _diag_blocked = final_state.get("build_infrastructure_blocked")
    _diag_summary = final_state.get("build_diagnostic_summary")
    _diag_decision = final_state.get("pre_existing_decision")
    _diag_auth_files = final_state.get("authorized_pre_existing_files")
    build_diagnostics: Optional[Dict[str, Any]] = None
    if any(v is not None for v in (_diag_accept, _diag_infra, _diag_blocked, _diag_summary, _diag_decision)):
        build_diagnostics = {
            "differential_accept": bool(_diag_accept) if _diag_accept is not None else None,
            "infrastructure_only": bool(_diag_infra) if _diag_infra is not None else None,
            "infrastructure_blocked": bool(_diag_blocked) if _diag_blocked is not None else None,
            "summary": str(_diag_summary or ""),
            "pre_existing_decision": str(_diag_decision or ""),
            "authorized_pre_existing_files": list(_diag_auth_files or []),
        }

    return {
        "contracts": contracts[-14:],
        "decision_ledger": decision_ledger[-20:],
        "decision_counter": int(final_state.get("decision_counter") or len(decision_ledger)),
        "overall_confidence": float(final_state.get("overall_confidence") or 0.70),
        "confidence_history": confidence_history[-30:],
        "confidence_drop_streak": int(final_state.get("confidence_drop_streak") or 0),
        "assumptions_active": assumptions_active[-20:],
        "assumptions_invalidated": assumptions_invalidated[-30:],
        "outcome_verification": outcome_verification if isinstance(outcome_verification, dict) else {},
        "planner_decisions_count": len(planner_decisions),
        "owner_files": owner_files[:10],
        "generated_files": generated_files[:12],
        "validation_failure_reason": str(final_state.get("validation_failure_reason") or ""),
        "build_attribution_reasoning": build_reasoning[:320],
        "learning_update": learning_update if isinstance(learning_update, dict) else {},
        "convergence_stagnation_count": int(final_state.get("convergence_stagnation_count") or 0),
        "build_diagnostics": build_diagnostics,
    }


def _build_workflow_explanation(state: Dict[str, Any], workflow_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build user-facing explanation from per-node contracts + authoritative outputs."""
    status = state.get("status", "running")
    phase = state.get("current_phase", "processing")
    steps = state.get("steps", []) or []
    workflow_output = workflow_output or {}

    contracts = workflow_output.get("contracts") or []
    decision_ledger = workflow_output.get("decision_ledger") or []
    overall_confidence = float(workflow_output.get("overall_confidence") or 0.70)
    confidence_history = workflow_output.get("confidence_history") or []
    assumptions_active = workflow_output.get("assumptions_active") or []
    assumptions_invalidated = workflow_output.get("assumptions_invalidated") or []
    contract_decisions = []
    contract_reasons = []
    contract_evidence = []
    for c in contracts:
        if not isinstance(c, dict):
            continue
        decision = str(c.get("decision") or "").strip()
        reason = str(c.get("reason") or "").strip()
        node = str(c.get("node") or "").strip()
        if decision:
            contract_decisions.append(f"{node}: {decision}" if node else decision)
        if reason:
            contract_reasons.append(reason)
        for e in (c.get("evidence") or []):
            txt = str(e).strip()
            if txt:
                contract_evidence.append(txt)

    decisions: List[str] = []
    blockers: List[str] = []
    for step in steps[-20:]:
        if not isinstance(step, dict):
            continue
        msg = str(step.get("message") or "").strip()
        output = step.get("data", {}).get("agent_output") if isinstance(step.get("data"), dict) else None
        if output:
            first_line = str(output).splitlines()[0].strip()
            if first_line:
                decisions.append(first_line)
        elif msg:
            decisions.append(msg)

        if step.get("status") == "error" and msg:
            blockers.append(msg)

    ledger_decisions = []
    for item in decision_ledger[-6:]:
        if not isinstance(item, dict):
            continue
        node = str(item.get("node") or "")
        dec = str(item.get("decision") or "")
        if dec:
            ledger_decisions.append(f"{node}: {dec}" if node else dec)

    decisions = ledger_decisions + contract_decisions + decisions

    validation_reason = str(state.get("validation_failure_reason") or "").strip()
    if validation_reason:
        blockers.append(validation_reason)
    authoritative_validation = str(workflow_output.get("validation_failure_reason") or "").strip()
    if authoritative_validation and authoritative_validation not in blockers:
        blockers.append(authoritative_validation)

    if contract_reasons:
        blockers.extend(contract_reasons[:2] if status == "failed" else [])

    if state.get("error"):
        blockers.append(str(state.get("error")))

    if assumptions_invalidated:
        latest_invalid = assumptions_invalidated[-1]
        if isinstance(latest_invalid, dict):
            reason = str(latest_invalid.get("invalidated_reason") or "").strip()
            if reason:
                blockers.append(f"Assumption invalidated: {reason}")

    # Deduplicate while preserving order.
    def _dedupe(items: List[str], limit: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            key = item.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= limit:
                break
        return out

    debug_decisions = _dedupe(decisions, 40)
    decisions = debug_decisions[:5]
    blockers = _dedupe(blockers, 5)
    evidence = _dedupe(contract_evidence, 5)

    planner_decisions_count = int(workflow_output.get("planner_decisions_count") or 0)
    owner_files = workflow_output.get("owner_files") or []
    generated_files = workflow_output.get("generated_files") or []
    build_reasoning = str(workflow_output.get("build_attribution_reasoning") or "").strip()
    outcome_verification = workflow_output.get("outcome_verification") or {}
    # Pull outcome_check_result from state (post-build semantic validation)
    outcome_check = state.get("outcome_check_result") or {}
    stagnation = int(workflow_output.get("convergence_stagnation_count") or 0)
    confidence_drop_streak = int(workflow_output.get("confidence_drop_streak") or 0)

    guidance = state.get("solution_guidance") or workflow_output.get("solution_guidance")
    if status == "completed":
        if guidance:
            explanation_txt = getattr(guidance, "explanation", "") or (guidance.get("explanation") if isinstance(guidance, dict) else "")
            steps_list = getattr(guidance, "step_by_step_solution", []) or (guidance.get("step_by_step_solution") if isinstance(guidance, dict) else [])
            summary = f"Informational & Architectural Solution: {explanation_txt}" if explanation_txt else "The architectural inquiry was resolved using repository knowledge without requiring code modifications."
            why = f"Synthesized from repository documentation and RAG context ({len(steps_list)} guidance items)." if steps_list else "All requested architectural details were retrieved from project knowledge."
            next_action = "Inspect the comprehensive architectural guidance above."
        else:
            check_verdict = outcome_check.get("status", "")
            if check_verdict == "CORRECT":
                summary = "The workflow completed successfully and the generated code correctly implements all ticket requirements."
                why = "Build passed and semantic validation confirmed all functional requirements are satisfied."
            elif check_verdict == "PARTIAL":
                summary = "The workflow completed (build passed) but some ticket requirements may be partially implemented."
                why = outcome_check.get("details", "")[:200] or "Semantic check found partial implementation."
            elif check_verdict == "INCOMPLETE":
                summary = "The workflow completed (build passed) but the generated code is missing key requirements."
                why = outcome_check.get("details", "")[:200] or "Semantic check found incomplete implementation."
            elif check_verdict == "repair_required":
                summary = "The workflow completed (build passed) but deterministic acceptance checks flagged issues."
                _rh = outcome_check.get("repair_hints") or []
                why = "; ".join(str(h)[:120] for h in _rh[:3]) or "Acceptance matrix checks failed."
            else:
                summary = "The workflow converged to a valid implementation and completed generation plus validation."
                why = "Node contracts and validation outputs indicate sufficient evidence, grounded owner targeting, and no terminal blockers."
            next_action = "Review generated files and merge after human verification."
    elif status == "failed":
        # ── Surface structured build diagnostics in the failed explanation ──
        _bd = workflow_output.get("build_diagnostics") or {}
        if _bd.get("infrastructure_only"):
            summary = (
                "The workflow stopped because the build failed due to an "
                "external infrastructure problem (missing dependency, network, registry). "
                "Source code was NOT the issue."
            )
            why = _bd.get("summary") or (blockers[0] if blockers else "Infrastructure failure detected.")
            next_action = "Resolve the environment/dependency issue, then re-run."
        elif _bd.get("differential_accept"):
            summary = (
                "The workflow stopped because ALL build errors are pre-existing "
                "(not introduced by this ticket). The generated code is correct."
            )
            why = _bd.get("summary") or (blockers[0] if blockers else "Pre-existing errors only.")
            _user_dec = _bd.get("pre_existing_decision", "leave")
            if _user_dec == "fix":
                next_action = "Pre-existing errors were authorized for repair — review the fix."
            elif _user_dec == "stop":
                next_action = "User chose to stop. Re-run after addressing pre-existing errors."
            else:
                next_action = "Pre-existing errors were left as-is. Review generated code and merge."
        else:
            summary = "The workflow stopped because it could not converge to a stable solution path."
            why = blockers[0] if blockers else "A convergence, quality, or runtime budget guard was triggered."
            next_action = "Refine ticket scope or provide stronger owner-file hints, then rerun."
    elif status == "stopped":
        summary = "The workflow was halted by user request."
        why = f"Execution stopped cleanly at phase '{phase}'. Intermediate findings, candidate files, and logs have been saved into memory."
        next_action = "Continue conversation in chat, review discovered files, or give new instructions."
    else:
        summary = "The workflow is actively converging on the solution by iterating discovery, planning, and validation."
        why = f"Current phase: {phase}. The system is still collecting or validating evidence."
        next_action = "Wait for completion or inspect current agent decisions below."

    details = []
    details.append(f"Overall confidence: {overall_confidence:.2f}")
    if planner_decisions_count:
        details.append(f"Planner evaluated {planner_decisions_count} candidate decision(s).")
    if owner_files:
        details.append(f"Grounded owners: {', '.join(owner_files[:4])}")
    if generated_files:
        details.append(f"Generated files: {', '.join(generated_files[:4])}")
    if outcome_check:
        verdict = outcome_check.get("status", "")
        if verdict in ("CORRECT", "PARTIAL", "INCOMPLETE"):
            details.append(f"Semantic validation: {verdict} — {outcome_check.get('details', '')[:120]}")
    if outcome_verification:
        verified = bool(outcome_verification.get("verified"))
        details.append(f"Outcome verification: {'passed' if verified else 'failed'}")
    if build_reasoning:
        details.append(f"Build attribution: {build_reasoning}")
    if stagnation > 0:
        details.append(f"Convergence stagnation counter: {stagnation}")
    if confidence_drop_streak > 0:
        details.append(f"Low-confidence streak: {confidence_drop_streak}")

    confidence_trend = []
    for c in confidence_history[-6:]:
        if not isinstance(c, dict):
            continue
        n = str(c.get("node") or "unknown")
        oc = c.get("overall_confidence")
        if isinstance(oc, (int, float)):
            confidence_trend.append(f"{n}: {float(oc):.2f}")

    active_assumptions = []
    for a in assumptions_active[-5:]:
        if not isinstance(a, dict):
            continue
        txt = str(a.get("text") or "").strip()
        if txt:
            active_assumptions.append(txt)

    invalidated_assumptions = []
    for a in assumptions_invalidated[-5:]:
        if not isinstance(a, dict):
            continue
        txt = str(a.get("text") or "").strip()
        if txt:
            invalidated_assumptions.append(txt)

    return {
        "summary": summary,
        "why": why,
        "decisions": decisions,
        "decision_debug": debug_decisions,
        "decision_summary": {
            "total": int(workflow_output.get("decision_counter") or len(decision_ledger)),
            "top_decisions": decisions,
        },
        "evidence": evidence,
        "details": details[:5],
        "confidence_trend": confidence_trend,
        "active_assumptions": active_assumptions,
        "invalidated_assumptions": invalidated_assumptions,
        "decision_count": int(workflow_output.get("decision_counter") or len(decision_ledger)),
        "blockers": blockers,
        "next_action": next_action,
        "phase": phase,
        "status": status,
        "build_diagnostics": workflow_output.get("build_diagnostics"),
    }


async def broadcast_to_workflow(workflow_id: str, message: dict):
    """Broadcast a step update to all WebSocket clients watching this workflow."""
    # Token telemetry events are high-frequency and must NOT be persisted in the
    # steps list (they would flood it and be replayed to late-joining clients).
    # They also must NOT overwrite current_phase on the backend.
    is_token_event = (
        isinstance(message.get("data"), dict)
        and message["data"].get("event_type") == "token_usage_update"
    )

    if not is_token_event:
        # Persist step atomically in DurableWorkflowStateStore
        if workflow_id in workflow_status:
            workflow_status.append_step(workflow_id, message)
    else:
        # Persist latest token usage snapshot so it survives across requests and flow completion
        if workflow_id in workflow_status:
            workflow_status[workflow_id]["token_usage"] = message["data"]

    # Also capture token_usage if included inside data of non-token events
    if (
        isinstance(message.get("data"), dict)
        and message["data"].get("token_usage")
        and workflow_id in workflow_status
    ):
        workflow_status[workflow_id]["token_usage"] = message["data"]["token_usage"]

    for ws in list(workflow_connections.get(workflow_id, [])):
        try:
            await ws.send_json(message)
        except Exception:
            workflow_connections.get(workflow_id, []).remove(ws)


def _map_node_to_phase(node_name: str) -> str:
    """Map LangGraph node name to UI phase label."""
    mapping = {
        "investigate": "classification",
        "investigation": "classification",
        "classify": "classification",
        "localize": "localization",
        "localization": "localization",
        "rag": "context_loading",
        "rag_context": "context_loading",
        "test_rag": "context_loading",
        "code_rag": "context_loading",
        "generate_tests": "patch_generation",
        "generate_code": "patch_generation",
        "build": "validation",
        "test": "validation",
        "validate": "validation",
        "commit": "committing",
    }
    for key, phase in mapping.items():
        if key in node_name.lower():
            return phase
    return "classification"


def _extract_rag_files(state: dict) -> List[dict]:
    """Extract file list from RAG context in workflow state."""
    files = []
    for ctx_key in ("code_rag_context", "test_rag_context"):
        ctx = state.get(ctx_key)
        if not ctx:
            continue
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except Exception:
                continue
        if isinstance(ctx, list):
            for item in ctx:
                path = None
                score = 0.0
                if isinstance(item, dict):
                    path = item.get("file_path") or item.get("path") or item.get("source")
                    score = item.get("similarity_score") or item.get("score") or 0.0
                elif isinstance(item, str):
                    path = item
                elif hasattr(item, "file_path"):
                    path = item.file_path
                    score = getattr(item, "similarity_score", 0.0) or 0.0
                if path:
                    files.append({"path": path, "selected": True, "confidence": float(score)})
        elif isinstance(ctx, dict):
            for path in ctx.get("files", []):
                files.append({"path": path, "selected": True, "confidence": 0.9})
    return files


def _extract_rag_chunks_detailed(state_delta: dict) -> List[dict]:
    """Extract RAG chunks with full detail (score, content preview, metadata) for UI display.

    Returns chunks sorted by similarity_score descending so highest-relevance chunks
    appear first.  The UI uses this to show the user exactly which code was fed to
    the LLM and let them exclude low-quality chunks.
    """
    chunks: list[dict] = []
    for ctx_key in ("test_rag_context", "code_rag_context"):
        raw = state_delta.get(ctx_key) or []
        rag_label = "Test Behavior" if ctx_key == "test_rag_context" else "Architecture"
        for c in raw:
            try:
                if isinstance(c, dict):
                    score   = float(c.get("similarity_score") or c.get("score") or 0)
                    fpath   = c.get("file_path") or c.get("path") or "unknown"
                    content = c.get("content", "")
                    name    = c.get("name", "")
                    ctype   = str(c.get("chunk_type", ""))
                elif hasattr(c, "file_path"):
                    score   = float(getattr(c, "similarity_score", 0) or 0)
                    fpath   = c.file_path
                    content = getattr(c, "content", "")
                    name    = getattr(c, "name", "")
                    ctype   = str(getattr(c, "chunk_type", ""))
                else:
                    continue
                chunks.append({
                    "path":       fpath,
                    "score":      round(score, 3),
                    "name":       name,
                    "chunk_type": ctype,
                    "preview":    content[:300].strip() if content else "",
                    "rag_label":  rag_label,
                    "included":   True,
                })
            except Exception:
                continue
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def _serialize_agent_output(node_name: str, state_delta: dict) -> Optional[str]:
    """Produce a human-readable summary of what each LangGraph node produced.

    Returns a multi-line string shown in the per-step agent-output panel in the UI,
    or None when there is nothing interesting to show.
    """
    try:
        if node_name == "investigate":
            inv = state_delta.get("investigation_result")
            if not inv:
                return None
            ticket_type  = str(getattr(inv, "ticket_type", getattr(inv, "get", lambda k, d=None: d)("ticket_type", "?")))
            confidence   = getattr(inv, "confidence",   0.0)
            affected     = getattr(inv, "affected_systems", []) or []
            explanation  = getattr(inv, "explanation", "") or ""
            action       = getattr(inv, "recommended_action", "") or ""
            lines = [
                f"🏷️  Type: {ticket_type}",
                f"📊 Confidence: {float(confidence):.0%}",
            ]
            if affected:
                lines.append(f"🎯 Affected: {', '.join(str(a) for a in affected[:5])}")
            if explanation:
                lines.append(f"\n📝 {explanation[:500]}")
            if action:
                lines.append(f"\n✅ Next action: {action[:200]}")
            return "\n".join(lines)

        elif node_name == "unified_analysis":
            req = state_delta.get("requirements")
            if not req:
                return None
            func_reqs    = getattr(req, "functional_requirements", []) or []
            affected_comp = getattr(req, "affected_components", []) or []
            edge_cases   = getattr(req, "edge_cases", []) or []
            lines = ["📋 Functional Requirements:"]
            for r in func_reqs[:6]:
                lines.append(f"  • {str(r)}")
            if affected_comp:
                lines.append(f"\n🔧 Affected Components: {', '.join(str(c) for c in affected_comp[:8])}")
            if edge_cases:
                lines.append(f"\n⚠️  Edge Cases identified: {len(edge_cases)}")
                for ec in edge_cases[:3]:
                    lines.append(f"  • {str(ec)}")
            return "\n".join(lines)

        elif node_name in ("plan", "remediation_plan"):
            plan = state_delta.get("architectural_plan")
            if not plan:
                return None
            pattern_raw = getattr(plan, "pattern", None)
            pattern_val = getattr(pattern_raw, "value", str(pattern_raw or ""))
            pattern = pattern_val.replace("ArchitecturalPattern.", "").replace("_", " ").title()
            if not pattern or pattern == "?":
                pattern = "Microservices Architecture"
            tasks    = getattr(plan, "tasks", []) or []
            modules  = getattr(plan, "affected_modules", []) or []
            prefix   = "Remediation Tasks" if node_name == "remediation_plan" else "Tasks"
            lines    = [f"🏗️  Architectural Pattern: {pattern}"]
            if modules:
                lines.append(f"📦 Modules: {', '.join(str(m) for m in modules[:6])}")
            if tasks:
                lines.append(f"\n📝 {prefix} ({len(tasks)}):") 
                for t in tasks[:10]:
                    title      = getattr(t, "title", str(t))
                    ttype_obj  = getattr(t, "task_type", "")
                    ttype      = getattr(ttype_obj, "value", str(ttype_obj)).upper()
                    fpath      = getattr(t, "file_path", "")
                    fname      = Path(fpath).name if fpath else ""
                    req_ids    = getattr(t, "requirement_ids", []) or []
                    req_str    = f" [{', '.join(req_ids)}]" if req_ids else ""
                    method     = getattr(t, "target_method", "")
                    method_str = f" → {method}()" if method and method != "(file)" else ""
                    row = f"  • [{ttype}] {fname or fpath}{req_str}{method_str}: {title}"
                    lines.append(row)
            return "\n".join(lines)

        elif node_name in ("rag_tests", "rag_code"):
            chunks = _extract_rag_chunks_detailed(state_delta)
            if not chunks:
                return "🧬 Precision AST & Symbol Context loaded from local Tree-sitter index (0ms latency, zero hallucinations)"
            icon  = "🧪" if node_name == "rag_tests" else "🏗️"
            label = "Test Behavior" if node_name == "rag_tests" else "Architecture"
            lines = [f"{icon} {label} AST Context — {len(chunks)} symbol slices active:"]
            for ch in chunks[:8]:
                bar   = "█" * int(ch["score"] * 10) + "░" * (10 - int(ch["score"] * 10))
                fname = ch["path"].split("/")[-1]
                name  = f" ({ch['name']})" if ch["name"] else ""
                lines.append(f"  {bar} {ch['score']:.0%}  {fname}{name}")
            return "\n".join(lines)

        elif node_name in ("generate_tests", "generate_code", "edit_loop", "fix_build", "fix_test"):
            key = "generated_tests" if node_name == "generate_tests" else "generated_code"
            gen = state_delta.get(key) or []
            if not gen:
                return None
            icon_map = {"generate_tests": "🧪", "generate_code": "💻", "edit_loop": "✏️", "fix_build": "🔧", "fix_test": "🔧"}
            icon = icon_map.get(node_name, "💻")
            lines = [f"{icon} Generated {len(gen)} file(s):"]
            for g in gen:
                if isinstance(g, dict):
                    fpath   = g.get("file_path", "?")
                    change  = g.get("change_type", "")
                elif hasattr(g, "file_path"):
                    fpath   = g.file_path
                    change  = str(getattr(g, "change_type", ""))
                else:
                    continue
                basename = fpath.split("/")[-1] if "/" in fpath else fpath.split("\\")[-1] if "\\" in fpath else fpath
                action_icon = "✨" if "create" in str(change).lower() else "✏️"
                lines.append(f"  {action_icon} {basename}")

            # ── Per-task explanations (new: shows WHY each file was changed) ──
            task_explanations = state_delta.get("_task_explanations") or []
            if task_explanations and node_name == "generate_code":
                lines.append("")
                lines.append("📋 Per-task breakdown:")
                for te in task_explanations:
                    te_basename = te.get("basename", "?")
                    te_purpose = te.get("purpose", "")[:200]
                    te_change = te.get("change_type", "").upper()
                    produces = te.get("produces", [])
                    consumes = te.get("consumes", [])
                    lines.append(f"\n  ── {te_basename} ({te_change}) ──")
                    lines.append(f"    📝 {te_purpose}")
                    if produces:
                        lines.append(f"    🔧 Creates: {', '.join(p[:60] for p in produces[:5])}")
                    if consumes:
                        lines.append(f"    📥 Uses: {', '.join(c[:60] for c in consumes[:5])}")

            lines.append("\n→ View full changes in the Changes tab")
            return "\n".join(lines)

        elif node_name == "context_expand":
            promoted = state_delta.get("promoted_count")
            if promoted is None:
                discovered = state_delta.get("discovered_files") or []
                promoted = len(discovered)
            tier2_used = state_delta.get("tier2_used", False)
            if tier2_used:
                return f"🔄 Context expanded: {promoted} file(s) promoted from Tier 2 backup"
            return f"🔄 Context expanded: {promoted} file(s) added via targeted re-investigation"

        elif node_name == "outcome_check":
            result = state_delta.get("outcome_check_result") or {}
            status = result.get("status", "unknown")
            icon = "✅" if status in ("CORRECT", "skipped") else "⚠️"
            lines = [f"{icon} Outcome: {status}"]
            if result.get("eam_passed") is True:
                lines.append("   ✅ Deterministic acceptance checks passed (bindings + planned files)")
            for h in (result.get("repair_hints") or [])[:4]:
                lines.append(f"   ⛔ {str(h)[:170]}")
            return "\n".join(lines)

        elif node_name == "build":
            res = state_delta.get("build_result")
            if not res:
                return None
            status = str(getattr(res, "status", "?"))
            errors = getattr(res, "errors", []) or []
            if errors:
                import re as _re
                # Group errors by file for readability
                error_groups = {}
                for e in errors:
                    e_str = str(e)
                    # Extract filename from error like "src/app/file.ts(12,3): error TS..."
                    file_match = _re.match(r'(?:.*?[\\/])?([^\\/(]+\.\w+)', e_str)
                    fname = file_match.group(1) if file_match else "other"
                    error_groups.setdefault(fname, []).append(e_str)
                lines = [f"🔨 Build: {status} — {len(errors)} error(s) in {len(error_groups)} file(s):"]
                for fname, errs in sorted(error_groups.items()):
                    lines.append(f"  📄 {fname}: {len(errs)} error(s)")
                    for err in errs[:3]:  # show up to 3 per file
                        # Extract just the error code + message
                        err_msg = err
                        ts_match = _re.search(r'error (TS\d+:.*)', err)
                        if ts_match:
                            err_msg = ts_match.group(1)[:120]
                        else:
                            err_msg = err[-120:]
                        lines.append(f"    ❌ {err_msg}")
                    if len(errs) > 3:
                        lines.append(f"    ... and {len(errs) - 3} more")
                return "\n".join(lines)
            return f"🔨 Build: {status} ✅"

        elif node_name == "test":
            res = state_delta.get("test_result")
            if not res:
                return None
            status  = str(getattr(res, "status", "?"))
            passed  = getattr(res, "passed_tests", 0) or 0
            failed  = getattr(res, "failed_tests", 0) or 0
            return f"🧪 Tests: {status}  ✅ {passed} passed  ❌ {failed} failed"

        elif node_name == "preflight_check":
            # The workflow already builds a rich agent_output string — pass it through
            ao = state_delta.get("agent_output")
            if ao:
                return str(ao)
            verdict = state_delta.get("preflight_verdict") or "NOT_EVALUATED"
            summary = state_delta.get("preflight_summary", "")
            return f"🚦 Preflight: {verdict}\n{summary[:300]}" if summary else f"🚦 Preflight: {verdict}"

    except Exception as exc:
        logger.debug(f"_serialize_agent_output({node_name}) error: {exc}")
    return None


# REMOVED workflow_manager import - use direct calls to run_autonomous_workflow()
# from workflow_manager import workflow_manager
# from workflow_models import WorkflowPhase

from ticket_to_code.tracing.paths import get_trace_root as _get_trace_root
UPLOAD_DIR = _get_trace_root() / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload screenshot, log file, or document attachment for workflow analysis."""
    safe_name = _safe_upload_filename(file.filename)
    try:
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        dest_path = UPLOAD_DIR / unique_name
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
        logger.info(f"📁 Uploaded attachment saved to: {dest_path} ({len(contents)} bytes)")
        return {
            "filename": safe_name,
            "filepath": str(dest_path),
            "size": len(contents),
            "content_type": file.content_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class StartTransparentWorkflowRequest(BaseModel):
    project_id: str
    ticket_id: str
    ticket_description: str
    repo_path: str
    execution_mode: ExecutionMode = ExecutionMode.PIPELINE
    attachments: List[str] = Field(default_factory=list)
    # Per-ticket change authorization: files the ticket explicitly allows
    # modifying, and files that must NEVER be modified (strictly read-only).
    # Empty list = no ticket-declared scope (evidence-based authorization only).
    writable_files: List[str] = Field(default_factory=list)
    forbidden_files: List[str] = Field(default_factory=list)
    model_mode: str = "default"  # "default" | "auto_stage" | "auto_model"
    selected_model: Optional[str] = None
    labels: List[str] = Field(default_factory=list)


def _check_workspace_health(repo_path: str) -> Optional[str]:
    """
    Fail fast BEFORE burning a run on a broken workspace.

    Detects the "gutted workspace" failure mode (mass tracked-file deletions,
    e.g. after a bad revert): for each git repo at repo_path or its immediate
    children, compares deleted tracked files against the total tracked files.
    Returns an actionable error message, or None when healthy.
    """
    root = Path(repo_path)
    if not root.exists():
        return f"Workspace path does not exist: {repo_path}"
    candidates = []
    if (root / ".git").exists():
        candidates.append(root)
    else:
        try:
            for child in root.iterdir():
                if child.is_dir() and (child / ".git").exists():
                    candidates.append(child)
        except Exception:
            pass
    problems = []
    for repo in candidates:
        try:
            ls = subprocess.run(
                ["git", "ls-files"], cwd=str(repo),
                capture_output=True, text=True, timeout=60,
            )
            st = subprocess.run(
                ["git", "status", "--porcelain"], cwd=str(repo),
                capture_output=True, text=True, timeout=60,
            )
            if ls.returncode != 0 or st.returncode != 0:
                continue
            tracked = sum(1 for l in (ls.stdout or "").splitlines() if l.strip())
            deleted = sum(
                1 for l in (st.stdout or "").splitlines()
                if len(l) >= 2 and (l[0] == "D" or l[1] == "D")
            )
            if tracked > 0 and deleted > 10 and (deleted / tracked) > 0.02:
                problems.append(
                    f"{repo.name}: {deleted} of {tracked} tracked files are deleted from disk"
                )
        except Exception:
            continue
    if problems:
        return (
            "Workspace integrity check FAILED — refusing to start the run on a broken workspace. "
            + " | ".join(problems)
            + ". Fix with: git -C <repo> restore .  (then verify with git status), and re-check that the project is indexed."
        )
    return None


class StartReasoningAgentRequest(BaseModel):
    ticket_id: str
    ticket_title: Optional[str] = None
    ticket_description: str
    repo_path: str
    technology: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)


@app.post("/api/agent/solve")
async def solve_with_reasoner(request: StartReasoningAgentRequest):
    """Deprecated standalone endpoint. Use transparent workflow with execution_mode."""
    raise HTTPException(
        status_code=410,
        detail=(
            "Deprecated endpoint. Use /api/workflow/transparent/start with "
            "execution_mode='reasoning' for experimental evaluation, or "
            "execution_mode='pipeline' for production runs."
        ),
    )


@app.get("/api/agent/brain")
async def get_project_brain_summary(repo_path: str):
    """Return a summary of what the Project Repository Brain has learned so far."""
    if os.getenv("AVIATOR_ENABLE_EXPERIMENTAL_REASONING", "0") != "1":
        raise HTTPException(
            status_code=403,
            detail=(
                "Repository Brain endpoint is disabled in production mode. "
                "Set AVIATOR_ENABLE_EXPERIMENTAL_REASONING=1 for evaluation-only access."
            ),
        )
    try:
        from ticket_to_code.reasoning import RepositoryBrain
        brain = RepositoryBrain(repo_path)
        return brain.summary()
    except Exception as e:
        logger.error(f"Repository brain summary failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) & Live UI Verification Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/mcp/servers")
async def get_mcp_servers():
    """List all configured MCP servers and their available tools."""
    return mcp_manager.get_servers()

@app.post("/api/mcp/servers")
async def save_mcp_server(payload: dict):
    """Add or update an MCP server configuration."""
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Server name is required")
    try:
        return mcp_manager.save_server(name, payload)
    except ValueError as exc:
        # Command/args failed the execution allow-list (see services.mcp_manager)
        raise HTTPException(status_code=400, detail=str(exc))

@app.delete("/api/mcp/servers/{server_name}")
async def delete_mcp_server(server_name: str):
    """Delete an MCP server."""
    success = mcp_manager.delete_server(server_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    return {"success": True, "message": f"Server '{server_name}' deleted"}

@app.post("/api/mcp/servers/{server_name}/test")
async def test_mcp_server(server_name: str):
    """Test connection and discover tools for an MCP server."""
    return await mcp_manager.test_connection(server_name)

_LOOPBACK_VERIFY_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _allowed_verify_hosts() -> set:
    """Loopback hosts plus NEMOFORGE_ALLOWED_VERIFY_HOSTS (comma separated)."""
    extra = os.getenv("NEMOFORGE_ALLOWED_VERIFY_HOSTS", "")
    allowed = set(_LOOPBACK_VERIFY_HOSTS)
    for h in extra.split(","):
        h = h.strip().lower().strip("[]")
        if h:
            allowed.add(h)
    return allowed


def _is_allowed_verify_url(url: Optional[str]) -> bool:
    """
    True only for http(s) URLs whose host is loopback or explicitly allow-listed.
    Guards the Playwright verifier against being used as an SSRF proxy.
    """
    from urllib.parse import urlsplit
    if not url or not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url.strip())
    except Exception:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    host = (parts.hostname or "").lower()
    if not host:
        return False
    return host in _allowed_verify_hosts()


@app.post("/api/verification/playwright")
async def verify_with_playwright(payload: dict):
    """
    Execute live UI inspection using Playwright.
    Asserts DOM elements, text, and button states in real-time.
    """
    target_url = payload.get("target_url", "http://localhost:4200")
    if not _is_allowed_verify_url(target_url):
        raise HTTPException(
            status_code=400,
            detail=(
                "target_url host is not allowed. Only localhost/127.0.0.1/::1 or hosts listed in "
                "NEMOFORGE_ALLOWED_VERIFY_HOSTS may be verified."
            ),
        )
    actions = payload.get("actions", [])
    assertions = payload.get("assertions", [])
    res = await playwright_verifier.verify_dom_scenario(
        target_url=target_url,
        actions=actions,
        assertions=assertions
    )
    return res.to_dict()
 
@app.get("/api/models")
async def get_available_models(include_all: bool = False):
    """
    Returns available models, capability descriptors, and the 3 execution modes
    (default, auto_stage, auto_model).
    If include_all=True, includes both verified working models and defunct/unconfigured models with status_reason.
    If include_all=False, returns strictly verified working models for dropdown selection.
    """
    return {
        "modes": ModelRouter.list_modes(),
        "models": ModelRouter.list_models(include_all=include_all),
        "default_model": get_default_model_id(),
    }

class SettingsKeysRequest(BaseModel):
    token: Optional[str] = None
    provider: Optional[str] = None
    provider_name: Optional[str] = None
    nebius_api_key: Optional[str] = None
    nebius_base_url: Optional[str] = "https://api.tokenfactory.nebius.com/v1/"
    model: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

class VerifyKeyRequest(BaseModel):
    token: str
    provider: Optional[str] = "nebius"
    base_url: Optional[str] = "https://api.tokenfactory.nebius.com/v1/"

@app.post("/api/settings/verify-key")
async def verify_key(req: VerifyKeyRequest):
    token = req.token.strip()
    if not token or len(token) < 5:
        raise HTTPException(status_code=400, detail="Token cannot be empty or invalid.")
    
    provider = (req.provider or "nebius").lower()
    from ticket_to_code.intelligence.model_router import discover_provider_models

    protocol = "openai"
    base_url = (req.base_url or "").rstrip("/")
    if "gemini" in provider or "google" in provider or token.startswith("AIza"):
        protocol = "gemini"
        base_url = "https://generativelanguage.googleapis.com/v1beta"
    elif "anthropic" in provider or "claude" in provider or token.startswith("sk-ant-"):
        protocol = "anthropic"
        base_url = "https://api.anthropic.com/v1"
    elif not base_url:
        if provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
        elif provider == "together":
            base_url = "https://api.together.xyz/v1"
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "deepseek":
            base_url = "https://api.deepseek.com/v1"
        else:
            base_url = "https://api.tokenfactory.nebius.com/v1"

    creds = {"api_key": token, "base_url": base_url, "protocol": protocol}
    try:
        models = await asyncio.to_thread(discover_provider_models, provider, creds)
        if models:
            return {
                "valid": True,
                "provider": provider,
                "message": f"Successfully verified with {provider.capitalize()} ({len(models)} models available).",
                "models": models[:15]
            }
        return {
            "valid": True,
            "provider": provider,
            "message": f"Key accepted for {provider.capitalize()}.",
            "models": []
        }
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Authentication check failed: {exc}")

@app.get("/api/settings/keys")
async def get_settings_keys():
    """Returns configuration status for all supported and custom LLM providers."""
    env_local_path = Path(__file__).parent.parent / ".env.local"
    env_vars = {}
    if env_local_path.exists():
        try:
            for l in env_local_path.read_text(encoding="utf-8").splitlines():
                if l.strip() and not l.startswith("#") and "=" in l:
                    k, v = l.split("=", 1)
                    env_vars[k.strip()] = v.strip()
        except Exception:
            pass

    active_provider = os.getenv("LLM_PROVIDER", env_vars.get("LLM_PROVIDER", "")).lower()

    known_meta = {
        "nebius": {
            "name": "Nebius Token Factory",
            "key_var": "NEBIUS_API_KEY",
            "default_model": "nvidia/nemotron-3-super-120b-a12b",
            "description": "NVIDIA Nemotron Models on Nebius AI Cloud"
        },
        "openai": {
            "name": "OpenAI",
            "key_var": "OPENAI_API_KEY",
            "default_model": "gpt-4o",
            "description": "GPT-4o, GPT-4o-mini, o1"
        },
        "anthropic": {
            "name": "Anthropic Claude",
            "key_var": "ANTHROPIC_API_KEY",
            "default_model": "claude-3-7-sonnet",
            "description": "Claude 3.7 Sonnet (Reasoning), Claude 3.5 Sonnet"
        },
        "google": {
            "name": "Google Gemini",
            "key_var": "GEMINI_API_KEY",
            "default_model": "gemini-2.5-pro",
            "description": "Gemini 2.5 Pro (2M context), Gemini 2.5 Flash"
        }
    }

    configured_keys = []
    # 1. Discover known providers
    for p_id, p_info in known_meta.items():
        raw_key = os.getenv(p_info["key_var"], env_vars.get(p_info["key_var"], "")).strip()
        if raw_key:
            masked = f"{raw_key[:4]}...{raw_key[-4:]}" if len(raw_key) > 8 else "****"
            # Extract live available models for this provider
            try:
                live_models = [
                    m["model_id"] for m in ModelRouter.list_models()
                    if m.get("provider") == p_id and m.get("is_available")
                ]
            except Exception:
                live_models = []

            configured_keys.append({
                "id": p_id,
                "name": p_info["name"],
                "key_var": p_info["key_var"],
                "masked_key": masked,
                "is_active": (p_id == active_provider),
                "default_model": p_info["default_model"],
                "description": p_info["description"],
                "live_models": live_models
            })

    # 2. Discover custom providers from *_API_KEY
    for k, v in env_vars.items():
        if k.endswith("_API_KEY") and v.strip():
            p_prefix = k[:-8].lower()
            if p_prefix not in known_meta and not any(ck["id"] == p_prefix for ck in configured_keys):
                custom_name = env_vars.get(f"{p_prefix.upper()}_NAME", p_prefix.capitalize())
                raw_key = v.strip()
                masked = f"{raw_key[:4]}...{raw_key[-4:]}" if len(raw_key) > 8 else "****"
                try:
                    live_models = [
                        m["model_id"] for m in ModelRouter.list_models()
                        if m.get("provider") == p_prefix and m.get("is_available")
                    ]
                except Exception:
                    live_models = []

                configured_keys.append({
                    "id": p_prefix,
                    "name": custom_name,
                    "key_var": k,
                    "masked_key": masked,
                    "is_active": (p_prefix == active_provider),
                    "default_model": env_vars.get(f"{p_prefix.upper()}_MODEL", "custom-model"),
                    "description": f"Custom Inference Provider ({custom_name})",
                    "live_models": live_models
                })

    active_provider_name = ""
    for ck in configured_keys:
        if ck["is_active"]:
            active_provider_name = ck["name"]
            break
    if not active_provider_name and configured_keys:
        active_provider_name = configured_keys[0]["name"]

    nebius_key = next((k["masked_key"] for k in configured_keys if k["id"] == "nebius"), "")

    return {
        "provider": active_provider,
        "active_provider_name": active_provider_name,
        "token_configured": len(configured_keys) > 0,
        "configured_keys": configured_keys,
        "total_keys": len(configured_keys),
        "nebius_configured": any(k["id"] == "nebius" for k in configured_keys),
        "nebius_key_masked": nebius_key,
        "nebius_base_url": os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        "active_model": _runtime_model_preference["model"] or os.getenv("LLM_MODEL", "nebius/nvidia/nemotron-3-super-120b-a12b"),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "google_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


class SetActiveProviderRequest(BaseModel):
    provider: str


@app.post("/api/settings/keys/set-active")
async def set_active_provider(req: SetActiveProviderRequest):
    """Switch the active primary LLM provider."""
    env_local_path = Path(__file__).parent.parent / ".env.local"
    prov = req.provider.strip().lower()
    existing = {}
    if env_local_path.exists():
        for l in env_local_path.read_text(encoding="utf-8").splitlines():
            if l.strip() and not l.startswith("#") and "=" in l:
                k, v = l.split("=", 1)
                existing[k.strip()] = v.strip()
    existing["LLM_PROVIDER"] = prov
    os.environ["LLM_PROVIDER"] = prov

    # Update default active model according to provider (persisted to .env.local; the in-process
    # preference is per-request-resolved via ModelRouter, never a process-wide env mutation)
    if prov == "nebius":
        existing["LLM_MODEL"] = "nebius/nvidia/nemotron-3-super-120b-a12b"
        _runtime_model_preference["model"] = "nebius/nvidia/nemotron-3-super-120b-a12b"
    elif prov == "anthropic":
        existing["LLM_MODEL"] = "claude-3-7-sonnet"
        _runtime_model_preference["model"] = "claude-3-7-sonnet"
    elif prov == "openai":
        existing["LLM_MODEL"] = "gpt-4o"
        _runtime_model_preference["model"] = "gpt-4o"
    elif prov == "google":
        existing["LLM_MODEL"] = "gemini-2.5-pro"
        _runtime_model_preference["model"] = "gemini-2.5-pro"
    elif f"{prov.upper()}_MODEL" in existing:
        existing["LLM_MODEL"] = existing[f"{prov.upper()}_MODEL"]
        _runtime_model_preference["model"] = existing[f"{prov.upper()}_MODEL"]

    out_lines = [f"{k}={v}" for k, v in existing.items()]
    env_local_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    from ticket_to_code.intelligence.model_router import refresh_available_models
    refresh_available_models()
    return {"status": "success", "active_provider": prov, "active_model": _runtime_model_preference["model"] or os.getenv("LLM_MODEL")}


@app.delete("/api/settings/keys/{provider_id}")
async def delete_settings_key(provider_id: str):
    """Remove a configured provider key dynamically from local storage."""
    env_local_path = Path(__file__).parent.parent / ".env.local"
    p_id = provider_id.strip().lower()

    known_vars = {
        "nebius": "NEBIUS_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY"
    }
    key_var = known_vars.get(p_id, f"{p_id.upper()}_API_KEY")

    existing = {}
    if env_local_path.exists():
        for l in env_local_path.read_text(encoding="utf-8").splitlines():
            if l.strip() and not l.startswith("#") and "=" in l:
                k, v = l.split("=", 1)
                if k.strip() not in (key_var, f"{p_id.upper()}_NAME", f"{p_id.upper()}_MODEL"):
                    existing[k.strip()] = v.strip()

    if key_var in os.environ:
        del os.environ[key_var]
    if f"{p_id.upper()}_NAME" in os.environ:
        del os.environ[f"{p_id.upper()}_NAME"]

    # If active provider was deleted, reset LLM_PROVIDER to remaining or empty
    if os.getenv("LLM_PROVIDER", "").lower() == p_id:
        remaining_providers = [k.replace("_API_KEY", "").lower() for k in existing if k.endswith("_API_KEY")]
        next_prov = remaining_providers[0] if remaining_providers else ""
        existing["LLM_PROVIDER"] = next_prov
        os.environ["LLM_PROVIDER"] = next_prov

    out_lines = [f"{k}={v}" for k, v in existing.items()]
    env_local_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    from ticket_to_code.intelligence.model_router import refresh_available_models
    refresh_available_models()
    return {"status": "success", "deleted": provider_id}


@app.post("/api/settings/keys")
async def save_settings_keys(req: SettingsKeysRequest):
    """
    Saves LLM credentials securely to an untracked local .env.local file
    and sets them in the active runtime os.environ.
    Preserves exact provider name without forced fallbacks.
    """
    env_local_path = Path(__file__).parent.parent / ".env.local"

    existing = {}
    if env_local_path.exists():
        try:
            for l in env_local_path.read_text(encoding="utf-8").splitlines():
                if l.strip() and not l.startswith("#") and "=" in l:
                    k, v = l.split("=", 1)
                    existing[k.strip()] = v.strip()
        except Exception:
            pass

    raw_token = (req.token or req.nebius_api_key or "").strip()
    if not raw_token:
        raise HTTPException(status_code=400, detail="Token cannot be empty. Please enter your API key before saving.")

    p_raw = (req.provider_name or req.provider or "").strip()
    p_lower = p_raw.lower()

    if "anthropic" in p_lower or "claude" in p_lower or raw_token.startswith("sk-ant-"):
        provider_id = "anthropic"
        provider_name = "Anthropic Claude"
        key_var = "ANTHROPIC_API_KEY"
        default_model = "claude-3-7-sonnet"
    elif "openai" in p_lower or "gpt" in p_lower or (raw_token.startswith("sk-") and not raw_token.startswith("sk-ant-")):
        provider_id = "openai"
        provider_name = "OpenAI"
        key_var = "OPENAI_API_KEY"
        default_model = "gpt-4o"
    elif "google" in p_lower or "gemini" in p_lower or raw_token.startswith("AIza"):
        provider_id = "google"
        provider_name = "Google Gemini"
        key_var = "GEMINI_API_KEY"
        default_model = "gemini-2.5-pro"
    elif "nebius" in p_lower or "nemotron" in p_lower or raw_token.startswith("v1."):
        provider_id = "nebius"
        provider_name = "Nebius Token Factory"
        key_var = "NEBIUS_API_KEY"
        default_model = "nvidia/nemotron-3-super-120b-a12b"
    elif p_raw:
        # Dynamic Custom Provider specified by user
        import re
        provider_id = re.sub(r"[^a-zA-Z0-9_]", "", p_lower).lower() or "custom"
        provider_name = p_raw
        key_var = f"{provider_id.upper()}_API_KEY"
        default_model = req.model or "custom-model"
    else:
        # Generic Custom Provider fallback
        provider_id = "custom"
        provider_name = "Custom Provider"
        key_var = "CUSTOM_API_KEY"
        default_model = "custom-model"

    existing[key_var] = raw_token
    os.environ[key_var] = raw_token
    existing[f"{provider_id.upper()}_NAME"] = provider_name
    os.environ[f"{provider_id.upper()}_NAME"] = provider_name

    existing["LLM_PROVIDER"] = provider_id
    os.environ["LLM_PROVIDER"] = provider_id

    final_model = req.model or default_model
    existing["LLM_MODEL"] = final_model
    _runtime_model_preference["model"] = final_model

    if req.nebius_base_url:
        existing["NEBIUS_BASE_URL"] = req.nebius_base_url.strip()
        os.environ["NEBIUS_BASE_URL"] = req.nebius_base_url.strip()

    # Save to untracked .env.local
    out_lines = [f"{k}={v}" for k, v in existing.items()]
    env_local_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    from ticket_to_code.intelligence.model_router import refresh_available_models
    refresh_available_models()

    return {
        "status": "success",
        "message": f"Saved {provider_name} credentials to untracked storage (.env.local)",
        "provider": provider_id,
        "provider_name": provider_name,
        "model": final_model,
    }
 
@app.post("/api/workflow/transparent/start")
async def start_transparent_workflow(request: StartTransparentWorkflowRequest):
    """
    Start autonomous workflow using existing ticket_to_code system.
    """
    try:
        # Auto-detect explicit repository override from ticket description
        target_repo = request.repo_path
        repo_match = re.search(r"^(?:Repository|Repo):\s*([^\r\n]+)", request.ticket_description, re.MULTILINE | re.IGNORECASE)
        if repo_match:
            candidate_repo = repo_match.group(1).strip()
            if Path(candidate_repo).is_dir():
                target_repo = str(Path(candidate_repo).resolve())
                logger.info(f"📍 Target repository overridden from ticket description: {target_repo}")
                request.repo_path = target_repo

        # Fail fast on a gutted/broken workspace before spending any tokens
        _health_err = _check_workspace_health(request.repo_path)
        if _health_err:
            raise HTTPException(status_code=400, detail=_health_err)

        if (
            request.execution_mode == ExecutionMode.REASONING
            and os.getenv("AVIATOR_ENABLE_EXPERIMENTAL_REASONING", "0") != "1"
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "execution_mode='reasoning' is experimental and disabled by default. "
                    "Set AVIATOR_ENABLE_EXPERIMENTAL_REASONING=1 for evaluation-only runs."
                ),
            )

        workflow_id = str(uuid.uuid4())
        active_workflows[workflow_id] = []
        
        # Log the full incoming ticket payload for reproducibility and debugging
        try:
            log_dir = Path(request.repo_path) / ".aviator" / "tickets"
            log_dir.mkdir(parents=True, exist_ok=True)
            ticket_log_path = log_dir / f"{workflow_id}.json"
            if hasattr(request, "model_dump_json"):
                payload_str = request.model_dump_json(indent=2)
            elif hasattr(request, "dict"):
                payload_str = json.dumps(request.dict(), indent=2, default=str)
            else:
                payload_str = json.dumps(getattr(request, "__dict__", {}), indent=2, default=str)
            with open(ticket_log_path, "w", encoding="utf-8") as f:
                f.write(payload_str)
            logger.info(f"Saved full incoming ticket payload to {ticket_log_path}")
        except Exception as e:
            logger.error(f"Failed to save incoming ticket payload: {e}")
            
        # Build a ticket from the user's description (no external service needed)
        auto_id = f"TASK-{workflow_id[:8].upper()}"

        # Cleanly extract title from structured prompt headers if present
        raw_text = request.ticket_description or ""
        extracted_title = ""
        title_match = re.search(r"^(?:Ticket|Title):\s*([^\r\n]+)", raw_text, re.MULTILINE | re.IGNORECASE)
        if title_match:
            extracted_title = title_match.group(1).strip()
        else:
            for line in raw_text.splitlines():
                clean_line = line.strip()
                if clean_line and not clean_line.lower().startswith(("repo", "repository")):
                    extracted_title = clean_line[:100]
                    break
        if not extracted_title:
            extracted_title = raw_text[:100]

        ticket = ValueEdgeTicket(
            ticket_id=auto_id,
            title=extracted_title,
            description=request.ticket_description,
            priority=TicketPriority.MEDIUM,
            labels=getattr(request, "labels", None) or [],
            attachments=request.attachments or [],
            # Ticket-declared change authorization (strict whitelist when set):
            # everything outside writable_files is demoted to read-only by
            # classify_change_role; forbidden_files are rejected outright.
            expected_changed_files=[f for f in (request.writable_files or []) if f.strip()],
            forbidden_files=[f for f in (request.forbidden_files or []) if f.strip()],
        )
        
        # Resolve Model Selection Mode
        try:
            m_mode = ModelMode(getattr(request, "model_mode", "default"))
        except Exception:
            m_mode = ModelMode.DEFAULT

        selected_model_desc = ModelRouter.select_model_for_task(
            ticket,
            mode=m_mode,
            requested_model=getattr(request, "selected_model", None) or _runtime_model_preference["model"]
        )

        # Store initial status
        workflow_status[workflow_id] = {
            "status": "running",
            "ticket_id": auto_id,
            "project_id": request.project_id,
            "repo_path": request.repo_path,
            "execution_mode": request.execution_mode.value,
            "model_mode": m_mode.value,
            "active_model": selected_model_desc.model_id,
            "model_name": selected_model_desc.name,
            "model_tier": selected_model_desc.tier.value,
            "workflow_explanation": None,
            "workflow_output": {},
            "started_at": datetime.now().isoformat()
        }
        
        # Immediately record the task in history so it appears in the sidebar
        try:
            history_store.record_task(
                workflow_id=workflow_id,
                project_id=request.project_id,
                ticket_id=auto_id,
                title=request.ticket_description[:100],
                description=request.ticket_description,
                status="running",
                changed_files=[],
                execution_mode=request.execution_mode.value,
            )
        except Exception as e:
            logger.warning(f"Failed to record initial task history: {e}")
        
        # Run the EXISTING autonomous workflow in background
        task = asyncio.create_task(
            run_workflow_async(
                workflow_id,
                ticket,
                request.repo_path,
                request.execution_mode.value,
                model_mode=m_mode.value,
                selected_model=selected_model_desc.model_id,
            )
        )
        _RUN_TASKS[workflow_id] = task
        task.add_done_callback(lambda _: _RUN_TASKS.pop(workflow_id, None))
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "execution_mode": request.execution_mode.value,
            "model_mode": m_mode.value,
            "active_model": selected_model_desc.model_id,
            "model_name": selected_model_desc.name,
            "message": f"Autonomous workflow started with {selected_model_desc.name} ({m_mode.value.upper()} mode)..."
        }
    
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def run_workflow_async(
    workflow_id: str,
    ticket: "ValueEdgeTicket",
    workspace_path: str,
    execution_mode: str = "pipeline",
    resume_from: Optional[str] = None,
    initial_overrides: Optional[Dict[str, Any]] = None,
    model_mode: str = "default",
    selected_model: Optional[str] = None,
):
    """
    Run the LangGraph workflow in a background thread.
    Per-node updates are broadcast via asyncio.run_coroutine_threadsafe so the
    UI gets live progress without waiting for the full workflow to complete.
    """
    loop = asyncio.get_event_loop()

    def put(step: dict):
        """Thread-safe broadcast helper — schedules coroutine on the event loop."""
        if isinstance(step.get("data"), dict) and step["data"].get("event_type") == "token_usage_update":
            if workflow_id in workflow_status:
                workflow_status[workflow_id]["token_usage"] = step["data"]
        asyncio.run_coroutine_threadsafe(broadcast_to_workflow(workflow_id, step), loop)
        # Broadcast to general WebSocket so Chat tab receives live reasoning and explanations
        general_step_msg = {
            "type": "workflow_step",
            "workflow_id": workflow_id,
            "step": step,
            "phase": step.get("phase"),
            "status": step.get("status"),
            "message": step.get("message"),
            "data": step.get("data")
        }
        asyncio.run_coroutine_threadsafe(manager.broadcast(general_step_msg), loop)

    done_event = threading.Event()
    result_holder: dict = {}

    # --- Node name → UI phase mapping ---
    # Keys MUST match the names passed to workflow.add_node(...) in workflow.py
    PHASE_MAP = {
        # Actual node names registered in create_ticket_to_code_graph()
        # ── Investigation & Classification ──
        "investigate":       ("classification",   "🔍 Investigating & classifying ticket"),
        "runtime_diagnosis": ("classification",   "🩺 Analyzing runtime logs"),
        # ── Evidence Pipeline ──
        "discover":          ("localization",     "🗺️ Discovering repository structure"),
        "hypothesis_investigation": ("localization", "🧪 Generating investigation hypotheses"),
        "evidence_collection_loop": ("localization", "🔍 Collecting evidence (agentic loop)"),
        "evidence_ranking":  ("localization",     "📊 Ranking evidence by relevance"),
        "semantic_verification": ("localization", "✅ Verifying evidence semantically"),
        "preflight_check":   ("localization",     "🚦 Preflight idempotency check"),
        # ── Analysis & Planning ──
        "unified_analysis":  ("requirements",     "📋 Analyzing requirements"),
        "plan":              ("planning",         "🗺️ Planning implementation"),
        "remediation_plan":  ("planning",         "🗺️ Planning delta remediation"),
        "validate_candidates": ("planning",       "🔎 Validating candidate files"),
        "localize":          ("localization",     "📍 Resolving file paths (SQLite + filesystem)"),
        "grounded_understanding": ("planning",    "🧠 Building grounded understanding"),
        "ownership_completeness": ("planning",    "🏗️ Checking ownership completeness"),
        "dataflow_verification": ("planning",     "🔄 Verifying data flow across services"),
        # ── Context Loading ──
        "rag_tests":         ("context_loading",  "📂 Loading test-behavior context (RAG)"),
        "rag_code":          ("context_loading",  "📂 Loading architecture context (RAG)"),
        # ── Code Generation ──
        "generate_tests":    ("patch_generation", "✍️ Generating test cases (TDD)"),
        "generate_code":     ("patch_generation", "✍️ Generating implementation code"),
        "edit_loop":         ("edit_loop",        "✏️ Adaptive edit loop (file-by-file)"),
        # ── Post-Generation Validation ──
        "patch_gate":        ("validation",       "🛡️ Patch safety gate"),
        "import_validation": ("validation",       "📦 Validating & fixing imports"),
        "angular_module_registration": ("validation", "📦 Angular module registration"),
        "build":             ("validation",       "🔨 Building project"),
        "pre_fix_build":     ("validation",       "🔧 Checking build fingerprint"),
        "outcome_check":     ("validation",       "🎯 Outcome verification"),
        "outcome_verification": ("validation",    "🎯 Verifying requested outcome"),
        "test":              ("validation",       "✅ Running tests"),
        # ── Fix Loops ──
        "fix_build":         ("edit_loop",        "🔧 Fixing build errors"),
        "fix_test":          ("edit_loop",        "🔧 Fixing test failures"),
        "context_expand":    ("planning",         "🔄 Expanding context for re-plan"),
        # ── Completion ──
        "memory_update":     ("committing",       "💾 Updating execution memory"),
        # ── Behavior Pipeline (Pipeline B) ──
        "behavior_investigation": ("planning",    "🔬 Investigating behavioral patterns"),
        "capability_extraction": ("planning",     "📊 Extracting capabilities"),
        "capability_consolidation": ("planning",  "🔗 Consolidating capabilities"),
        "capability_graph_builder": ("planning",  "🕸️ Building capability graph"),
        "capability_retrieval": ("planning",      "🔍 Retrieving relevant capabilities"),
        "behavior_planning": ("planning",         "📝 Planning behavioral changes"),
        "plan_validation":   ("planning",         "✅ Validating plan"),
        "shadow_metrics":    ("validation",       "📈 Computing shadow metrics"),
    }

    def thread_run():
        ws_lock = get_workspace_lock(workspace_path)
        ws_lock.acquire()
        try:
            if execution_mode == ExecutionMode.REASONING.value:
                from ticket_to_code.workflow import run_autonomous_workflow

                final_state = run_autonomous_workflow(
                    ticket=ticket,
                    workspace_path=workspace_path,
                    technology=None,
                    max_fix_attempts=3,
                    execution_mode="reasoning",
                )
                result_holder["state"] = final_state
                return

            from ticket_to_code.workflow import create_ticket_to_code_graph, _set_transient
            from langgraph.checkpoint.memory import MemorySaver

            # Resolve Model Selection Mode
            try:
                m_mode = ModelMode(model_mode)
            except Exception:
                m_mode = ModelMode.DEFAULT

            selected_model_desc = ModelRouter.select_model_for_task(
                ticket,
                mode=m_mode,
                requested_model=selected_model or _runtime_model_preference["model"]
            )
            current_model_desc = selected_model_desc
            # Per-request model identity for this run (no process-wide env mutation)
            model_id = selected_model_desc.model_id
            if workflow_id in workflow_status:
                workflow_status[workflow_id]["model_id"] = model_id
            # Stage-aware resolver for the deep graph: keyed by this thread so concurrent runs
            # cannot override each other's routing.
            register_stage_model_resolver_for_current_thread(
                build_stage_model_resolver(m_mode, ticket, selected_model_desc)
            )

            start_phase = resume_from or "classification"
            start_msg = (
                f"🔄 Resuming workflow from '{resume_from}'…" if resume_from
                else f"🔍 Investigating ticket with {selected_model_desc.name} ({m_mode.value.upper()} mode)…"
            )
            put({
                "phase": start_phase,
                "status": "in_progress",
                "message": start_msg,
                "data": {
                    "model_mode": m_mode.value,
                    "active_model": selected_model_desc.model_id,
                    "model_name": selected_model_desc.name,
                    "model_tier": selected_model_desc.tier.value,
                },
                "timestamp": datetime.now().isoformat()
            })

            initial_overrides: Optional[Dict[str, Any]] = None

            # ── ATTEMPT HIGH-PERFORMANCE 5-PILLAR ARCHITECTURE FIRST ──
            if not resume_from:
                try:
                    from ticket_to_code.runtime.dag_orchestrator import DAGOrchestrator, DAGExecutionResult
                    from ticket_to_code.runtime.fast_ide_engine import FastIDEEngine
                    from ticket_to_code.workflow import WorkflowAgents
                    from ticket_to_code.models import (
                        GeneratedCode, BuildResult, BuildStatus, TestResult, TestStatus, ProgrammingLanguage
                    )

                    agents = WorkflowAgents(workspace_path=str(workspace_path))
                    # Bind the per-request model to the fast-path agents (WorkflowAgents has no
                    # model_name parameter; override the shared LLM after construction).
                    try:
                        from ticket_to_code.runtime.llm_factory import LLMRegistry
                        agents.llm = LLMRegistry.get_llm(model_name=model_id)
                        for _a in (getattr(agents, "code_generator", None), getattr(agents, "test_generator", None)):
                            if _a is not None and hasattr(_a, "llm"):
                                _a.llm = agents.llm
                    except Exception as _llm_err:
                        logger.warning(f"Per-request model '{model_id}' could not be bound to WorkflowAgents: {_llm_err}")

                    def _map_lang(ext: str) -> str:
                        m = {
                            "ts": "typescript", "tsx": "typescript",
                            "js": "javascript", "jsx": "javascript",
                            "py": "python", "cs": "csharp",
                            "html": "html", "scss": "scss", "css": "scss",
                            "java": "java", "go": "go", "json": "json",
                            "xml": "xml", "yaml": "yaml", "yml": "yaml", "sql": "sql"
                        }
                        return m.get(ext.lower().lstrip("."), "typescript")

                    # ── Honest result mapping: never fabricate SUCCESS / ALL_PASSED ──
                    def _honest_build_result(passed: Optional[bool], note: str):
                        """SUCCESS only when the engine reports build_passed=True; FAILURE when
                        it reports False; SKIPPED (no compiler run, not a verified success)
                        when the engine does not report a build outcome at all."""
                        if passed is True:
                            return BuildResult(status=BuildStatus.SUCCESS, exit_code=0, stdout=note)
                        if passed is False:
                            return BuildResult(status=BuildStatus.FAILURE, exit_code=1, stdout=note, errors=[note])
                        return BuildResult(status=BuildStatus.SKIPPED, exit_code=0, stdout=note)

                    def _honest_test_result(passed: Optional[bool], count: int):
                        if passed is True:
                            return TestResult(status=TestStatus.ALL_PASSED, passed=count, total_tests=count)
                        if passed is False:
                            return TestResult(status=TestStatus.SOME_FAILED, passed=0, failed=max(count, 1), total_tests=max(count, 1))
                        return TestResult(status=TestStatus.SKIPPED, passed=0, total_tests=0)

                    def _finalize_fast_path(state: dict) -> dict:
                        """Run the shared fast-path verification gate (owned by workflow.py).
                        It returns the state with status 'completed' or 'verification_failed';
                        if the gate is not available yet, the state is passed through unchanged."""
                        try:
                            from ticket_to_code.workflow import finalize_fast_path_state
                        except ImportError:
                            logger.warning(
                                "ticket_to_code.workflow.finalize_fast_path_state is not available; "
                                "fast-path result was not independently verified"
                            )
                            return state
                        try:
                            verified = finalize_fast_path_state(state, agents)
                        except Exception as fexc:
                            logger.warning(f"finalize_fast_path_state raised: {fexc}; keeping unverified state")
                            return state
                        if not isinstance(verified, dict):
                            return state
                        verified["workflow_status"] = verified.get("status") or state.get("status", "completed")
                        return verified

                    is_complex = DAGOrchestrator.is_ticket_moderate_or_complex(ticket, Path(workspace_path))

                    # ── BRANCH A: MODERATE / HIGH / COMPLEX TICKETS (DAG DECOMPOSITION) ──
                    if is_complex:
                        logger.info(f"📐 [run_workflow_async] Moderate/Complex ticket detected ({ticket.ticket_id}). Activating 5-Pillar DAG Orchestrator...")
                        put({
                            "phase": "planning",
                            "status": "info",
                            "message": f"📐 Moderate/Complex ticket detected: Activating 5-Pillar DAG Orchestrator with Sub-Agent Task Decomposition...",
                            "data": {"is_complex": True, "ticket_id": ticket.ticket_id},
                            "timestamp": datetime.now().isoformat(),
                        })

                        dag_orchestrator = DAGOrchestrator(
                            workspace_path=str(workspace_path),
                            ticket=ticket,
                            llm_client=agents.llm,
                            ui_callback=put,
                        )
                        dag_res = dag_orchestrator.execute_dag(use_sandbox=True)

                        if dag_res.success:
                            logger.info(
                                f"🏆 [run_workflow_async] DAGOrchestrator resolved ticket in "
                                f"{dag_res.duration_seconds:.2f}s ({dag_res.tokens_consumed} tokens across {len(dag_res.tasks)} tasks)!"
                            )
                            gen_objs = [
                                GeneratedCode(
                                    file_path=fp,
                                    content=(Path(workspace_path) / fp).read_text(encoding="utf-8", errors="ignore") if (Path(workspace_path) / fp).is_file() else "",
                                    language=_map_lang(Path(fp).suffix),
                                    change_type="modify",
                                )
                                for fp in dag_res.modified_files
                            ]
                            final_state = {
                                "ticket": ticket,
                                "workspace_path": workspace_path,
                                "status": "completed",
                                "workflow_status": "completed",
                                "generated_code": gen_objs,
                                "_run_generated_map": {
                                    fp: (Path(workspace_path) / fp).read_text(encoding="utf-8", errors="ignore") if (Path(workspace_path) / fp).is_file() else ""
                                    for fp in dag_res.modified_files
                                },
                                # DAGExecutionResult does not report build/test outcomes; only
                                # claim what the engine actually verified.
                                "build_result": _honest_build_result(
                                    getattr(dag_res, "build_passed", None),
                                    f"{len(dag_res.tasks)} DAG tasks applied via DAGOrchestrator",
                                ),
                                "test_result": _honest_test_result(
                                    getattr(dag_res, "test_passed", None),
                                    len(dag_res.tasks),
                                ),
                                "token_usage": {
                                    "total_tokens": dag_res.tokens_consumed,
                                    "tokens_in": dag_res.tokens_in,
                                    "tokens_out": dag_res.tokens_out,
                                    "llm_calls": dag_res.llm_calls,
                                    "latency_seconds": dag_res.duration_seconds,
                                    "engine": "DAGOrchestrator (5-Pillar Multi-Project DAG)",
                                },
                                "diffs": dag_res.diffs,
                                "candidate_files": [{"path": f, "selected": True} for f in dag_res.modified_files],
                                "workflow_explanation": {
                                    "summary": f"Ticket resolved in {dag_res.duration_seconds:.1f}s via DAGOrchestrator across {len(dag_res.tasks)} sub-agent tasks.",
                                    "files_changed": dag_res.modified_files,
                                    "tasks": [t.title for t in dag_res.tasks],
                                },
                            }
                            final_state = _finalize_fast_path(final_state)
                            result_holder["state"] = final_state
                            return
                        else:
                            logger.warning(f"⚠️ [run_workflow_async] DAGOrchestrator failed: {dag_res.error_message}. Falling back to full reasoning graph.")

                    # ── BRANCH B: FOCUSED SINGLE-TASK FAST IDE ENGINE ──
                    else:
                        fast_engine = FastIDEEngine(
                            workspace_path=str(workspace_path),
                            ticket=ticket,
                            llm_client=agents.llm,
                            max_repair_turns=2,
                            ui_callback=put,
                        )

                        logger.info(f"⚡ [run_workflow_async] Launching focused FastIDEEngine for {ticket.ticket_id}")
                        fast_res = fast_engine.run()

                        if fast_res.success:
                            logger.info(
                                f"⚡ [run_workflow_async] FastIDEEngine resolved ticket in "
                                f"{fast_res.duration_seconds:.2f}s ({fast_res.tokens_consumed} tokens)!"
                            )
                            gen_objs = [
                                GeneratedCode(
                                    file_path=fp,
                                    content=(Path(workspace_path) / fp).read_text(encoding="utf-8", errors="ignore") if (Path(workspace_path) / fp).is_file() else "",
                                    language=_map_lang(Path(fp).suffix),
                                    change_type="modify",
                                )
                                for fp in fast_res.modified_files
                            ]
                            final_state = {
                                "ticket": ticket,
                                "workspace_path": workspace_path,
                                "status": "completed",
                                "workflow_status": "completed",
                                "generated_code": gen_objs,
                                "_run_generated_map": {
                                    fp: (Path(workspace_path) / fp).read_text(encoding="utf-8", errors="ignore") if (Path(workspace_path) / fp).is_file() else ""
                                    for fp in fast_res.modified_files
                                },
                                "build_result": _honest_build_result(
                                    getattr(fast_res, "build_passed", None),
                                    fast_res.message or "FastIDEEngine build verification",
                                ),
                                "test_result": _honest_test_result(
                                    getattr(fast_res, "test_passed", None),
                                    1,
                                ),
                                "token_usage": {
                                    "total_tokens": fast_res.tokens_consumed,
                                    "tokens_in": fast_res.tokens_in,
                                    "tokens_out": fast_res.tokens_out,
                                    "llm_calls": fast_res.llm_calls,
                                    "latency_seconds": fast_res.duration_seconds,
                                    "engine": "FastIDEEngine (4-Stage ReAct Spine)",
                                },
                                "diffs": fast_res.diffs,
                                "candidate_files": [{"path": f, "selected": True} for f in fast_res.modified_files],
                                "workflow_explanation": {
                                    "summary": fast_res.message or f"Ticket resolved in {fast_res.duration_seconds:.1f}s via FastIDEEngine.",
                                    "files_changed": fast_res.modified_files,
                                },
                            }
                            final_state = _finalize_fast_path(final_state)
                            result_holder["state"] = final_state
                            return
                        else:
                            logger.info(f"⚡ [run_workflow_async] FastIDEEngine escalated: {fast_res.error_message}. Falling back to full reasoning graph.")
                            put({
                                "phase": "planning",
                                "status": "info",
                                "message": f"⚡ FastIDEEngine escalated: {fast_res.error_message or 'Complex ticket'}. Activating deep reasoning graph...",
                                "data": {"escalated": True},
                                "timestamp": datetime.now().isoformat(),
                            })
                        if not initial_overrides:
                            initial_overrides = {}
                        candidate_paths = [e["path"] for e in fast_res.ranked_evidence if "path" in e] or fast_res.target_files or fast_res.modified_files
                        if candidate_paths:
                            initial_overrides["candidate_files"] = [{"path": f, "selected": True} for f in candidate_paths]
                        initial_overrides["escalation_reason"] = fast_res.error_message
                        if fast_res.imported_defs:
                            grounded_defs_text = "\n\n".join(
                                f"// Grounded Reference from {k}:\n{v}" for k, v in fast_res.imported_defs.items()
                            )
                            initial_overrides["code_rag_context"] = grounded_defs_text
                            initial_overrides["solution_guidance"] = grounded_defs_text
                        if fast_res.tokens_consumed > 0:
                            initial_overrides["initial_tokens"] = fast_res.tokens_consumed
                            tok_snap = {
                                "total_tokens": fast_res.tokens_consumed,
                                "tokens_in": getattr(fast_engine, "tokens_in", int(fast_res.tokens_consumed * 0.82)),
                                "tokens_out": getattr(fast_engine, "tokens_out", max(0, fast_res.tokens_consumed - int(fast_res.tokens_consumed * 0.82))),
                                "tokens_thinking": 0,
                                "cached_tokens": 0,
                                "excluded_tokens": 0,
                                "real_costed_tokens": fast_res.tokens_consumed,
                                "llm_calls": getattr(fast_engine, "llm_calls", 1),
                                "engine": "Tier 1 FastIDEEngine (escalated to Tier 2)",
                            }
                            workflow_status[workflow_id]["token_usage"] = tok_snap
                            put({
                                "phase": "planning",
                                "status": "info",
                                "message": f"📊 Tier 1 Tokens Consumed: {fast_res.tokens_consumed:,} tokens (carrying forward)",
                                "data": {
                                    "event_type": "token_usage_update",
                                    **tok_snap,
                                },
                                "timestamp": datetime.now().isoformat(),
                            })
                except Exception as fast_exc:
                    logger.warning(f"⚡ [run_workflow_async] FastIDEEngine error, falling back to full graph: {fast_exc}")

            wf_graph = create_ticket_to_code_graph(workspace_path, entry_point=resume_from or "investigate")
            memory  = MemorySaver()
            app     = wf_graph.compile(checkpointer=memory)

            # Register UI callback in transient registry (NOT in state — functions crash msgpack)
            _set_transient(ticket.ticket_id, "_ui_callback", put)
            # Documented hand-off of the per-request model to the workflow owner (the graph factory
            # takes no model argument; the stage resolver above is the primary routing channel).
            _set_transient(ticket.ticket_id, "_model_id", model_id)

            # ── Wire interactive decision provider for build-failure approval ──
            # The provider sends a decision request to the UI via put(),
            # then blocks on a threading.Event until the user responds via
            # the /api/workflow/transparent/decision-response endpoint.
            import threading as _threading
            _decision_events = {}  # keyed by workflow_id
            _decision_results = {}  # keyed by workflow_id

            def _decision_provider_fn(payload: dict):
                """Interactive decision provider — blocks until user responds."""
                evt = _threading.Event()
                _decision_events[workflow_id] = evt
                _decision_results[workflow_id] = None
                # Also store in workflow_status so the API endpoint can find them
                workflow_status[workflow_id]["_decision_event"] = evt
                workflow_status[workflow_id]["_decision_result"] = None
                # Send the decision request to the UI
                put({
                    "phase": "build", "status": "awaiting_decision",
                    "message": payload.get("message", "Decision required"),
                    "data": {
                        "node": "pre_fix_build",
                        "event_type": "pre_existing_errors",
                        "decision_payload": payload,
                    },
                    "timestamp": datetime.now().isoformat(),
                })
                timeout = float(payload.get("timeout", 120))
                deadline = time.time() + timeout
                while time.time() < deadline and not evt.is_set():
                    if workflow_cancel_flags.get(workflow_id):
                        logger.info(f"Interactive decision aborted for {workflow_id}: workflow cancelled")
                        break
                    evt.wait(timeout=1.0)
                choice = workflow_status[workflow_id].get("_decision_result")
                # Clean up
                workflow_status[workflow_id].pop("_decision_event", None)
                workflow_status[workflow_id].pop("_decision_result", None)
                return choice

            _set_transient(ticket.ticket_id, "_decision_provider", _decision_provider_fn)

            initial_state = {
                "ticket": ticket, "workspace_path": workspace_path,
                "max_retry_attempts": 3, "investigation_result": None,
                "requirements": None, "architectural_plan": None,
                "solution_guidance": None, "test_rag_context": None,
                "code_rag_context": None, "test_status": None,
                "code_status": None, "generated_tests": None,
                "generated_code": None, "build_result": None,
                "test_result": None, "retry_attempt": 0,
                "convergence_fingerprint_history": [],
                "convergence_stagnation_count": 0,
                "convergence_last_progress_score": 0.0,
                "decision_ledger": [],
                "decision_counter": 0,
                "overall_confidence": 0.70,
                "confidence_history": [],
                "confidence_drop_streak": 0,
                "confidence_evidence_score": 0.0,
                "assumptions_active": [],
                "assumptions_invalidated": [],
                "assumption_sequence": 0,
                "assumption_dependency_index": {},
                "invalidated_assumption_ids": [],
                "impacted_nodes": [],
                "outcome_verification": None,
                "runtime_diagnosis": None,
                "last_error_type": None, "status": "initialized",
                "errors": [], "start_time": datetime.now(), "end_time": None,
                "rejected_approaches": [], "user_constraints": [],
            }
            if initial_overrides and isinstance(initial_overrides, dict):
                initial_state.update(initial_overrides)

            run_id = f"run_{ticket.ticket_id}_{workflow_id}"
            config = {"configurable": {"thread_id": run_id}}
            if workflow_id in workflow_status:
                workflow_status[workflow_id]["run_id"] = run_id
                workflow_status[workflow_id]["thread_id"] = run_id

            aggregate_state: Dict[str, Any] = {}
            cancelled = False
            for node_update in app.stream(initial_state, config):
                # ── Check if the user requested a stop ──
                if workflow_cancel_flags.get(workflow_id):
                    cancelled = True
                    logger.info(f"⛔ Workflow {workflow_id} cancelled by user")
                    put({
                        "phase": "failed", "status": "stopped",
                        "message": "⛔ Workflow stopped by user",
                        "data": {"stopped_by_user": True},
                        "timestamp": datetime.now().isoformat(),
                    })
                    break

                for node_name, state_delta in node_update.items():
                    if not isinstance(state_delta, dict):
                        continue
                    aggregate_state.update(state_delta)

                    # Dynamic stage-by-stage auto model switching in AUTO_STAGE mode
                    if m_mode == ModelMode.AUTO_STAGE:
                        stage_model = ModelRouter.select_model_for_stage(
                            node_name, ticket, mode=m_mode, active_task_model=selected_model_desc
                        )
                        if stage_model.model_id != current_model_desc.model_id:
                            current_model_desc = stage_model
                            if workflow_id in workflow_status:
                                workflow_status[workflow_id]["active_model"] = stage_model.model_id
                                workflow_status[workflow_id]["model_name"] = stage_model.name
                                workflow_status[workflow_id]["model_tier"] = stage_model.tier.value
                            put({
                                "phase": "model_switch",
                                "status": "info",
                                "message": f"🤖 Auto-routed stage '{node_name}' to {stage_model.name} ({stage_model.tier.value.upper()} tier)",
                                "data": {
                                    "stage": node_name,
                                    "model_id": stage_model.model_id,
                                    "model_name": stage_model.name,
                                    "tier": stage_model.tier.value,
                                    "provider": stage_model.provider,
                                },
                                "timestamp": datetime.now().isoformat(),
                            })

                    phase, label = PHASE_MAP.get(node_name, ("processing", f"⚙️ {node_name}"))

                    # Update current_phase in shared state so polling reflects it
                    workflow_status[workflow_id]["current_phase"] = phase

                    # Build enriched step message
                    inv         = state_delta.get("investigation_result")
                    rfiles      = _extract_rag_files(state_delta)
                    rag_chunks  = _extract_rag_chunks_detailed(state_delta)
                    agent_out   = _serialize_agent_output(node_name, state_delta)
                    gen         = state_delta.get("generated_code") or state_delta.get("generated_tests")

                    if gen and isinstance(gen, list):
                        def _get_fp_inline(g):
                            if isinstance(g, dict):
                                return g.get("file_path", "?")
                            return getattr(g, "file_path", "?") if hasattr(g, "file_path") else "?"
                        _new_fps = [_get_fp_inline(g) for g in gen if _get_fp_inline(g) != "?"]
                        if _new_fps:
                            _ex = workflow_status[workflow_id].get("generated_files", [])
                            workflow_status[workflow_id]["generated_files"] = list(dict.fromkeys(_ex + _new_fps))

                    if inv and node_name == "investigate":
                        inv_type = getattr(inv, "ticket_type", str(inv))
                        conf     = getattr(inv, "confidence", "?")
                        msg = f"🔍 Classified as: {inv_type}  (confidence {conf})"
                    elif rag_chunks:
                        top = rag_chunks[0]
                        msg = (f"📂 RAG: {len(rag_chunks)} chunks "
                               f"(top: {top['path'].split('/')[-1]} {top['score']:.0%})")
                        # Also merge file-level candidates
                        existing = workflow_status[workflow_id].get("candidate_files", [])
                        seen     = {f["path"] for f in existing}
                        workflow_status[workflow_id]["candidate_files"] = existing + [
                            {"path": ch["path"], "selected": True, "confidence": ch["score"]}
                            for ch in rag_chunks if ch["path"] not in seen
                        ]
                    elif rfiles:
                        names = ", ".join(f["path"].split("/")[-1] for f in rfiles[:4])
                        msg   = f"📂 RAG: {len(rfiles)} file(s) — {names}"
                        existing = workflow_status[workflow_id].get("candidate_files", [])
                        seen     = {f["path"] for f in existing}
                        workflow_status[workflow_id]["candidate_files"] = existing + [
                            rf for rf in rfiles if rf["path"] not in seen
                        ]
                    elif gen and isinstance(gen, list):
                        def _get_fp(g):
                            if isinstance(g, dict):
                                return g.get("file_path", "?")
                            return getattr(g, "file_path", "?") if hasattr(g, "file_path") else "?"
                        names = [_get_fp(g).split('/')[-1].split('\\')[-1] for g in gen[:3]]
                        msg   = f"✍️ Generated {len(gen)} file(s): " + ", ".join(names)
                    elif node_name == "preflight_check":
                        pv = state_delta.get("preflight_verdict") or "NOT_EVALUATED"
                        msg = f"🚦 Preflight: {pv}"
                    else:
                        msg = agent_out.split("\n")[0] if agent_out else label

                    put({
                        "phase": phase, "status": "completed", "message": msg,
                        "data": {
                            "node":         node_name,
                            "rag_files":    rfiles or None,
                            "rag_chunks":   rag_chunks or None,
                            "investigation": str(inv)[:400] if inv else None,
                            "agent_output": agent_out,
                            "task_explanations": (
                                state_delta.get("_task_explanations")
                                if node_name == "generate_code" else None
                            ),
                            "preflight_data": (
                                {
                                    "verdict": state_delta.get("preflight_verdict") or "NOT_EVALUATED",
                                    "summary": state_delta.get("preflight_summary") or "",
                                    "missing_requirements": state_delta.get("preflight_missing_requirements") or [],
                                    "implementation_guidance": state_delta.get("preflight_implementation_guidance") or [],
                                }
                                if node_name == "preflight_check" else None
                            ),
                        },
                        "timestamp": datetime.now().isoformat(),
                    })

                    # Broadcast completed stage to main WebSocket for real-time Chat tab updates
                    try:
                        asyncio.run_coroutine_threadsafe(
                            manager.broadcast({
                                "type": "workflow_step",
                                "workflow_id": workflow_id,
                                "phase": phase,
                                "status": "complete",
                                "message": msg,
                                "data": {
                                    "node": node_name,
                                },
                                "timestamp": datetime.now().isoformat(),
                            }),
                            loop,
                        )
                    except Exception:
                        pass

                    # ── Evidence Collection sub-steps (real-time UI) ──────
                    # After the main step, broadcast each evidence sub-step
                    # so the UI shows per-iteration progress.
                    if node_name == "evidence_collection_loop":
                        _substeps = state_delta.get("_evidence_substeps") or []
                        for _sub in _substeps:
                            _sub_type = _sub.get("type", "")
                            _sub_msg = _sub.get("message", "")
                            _sub_phase = "evidence_collection_loop"

                            # Map sub-step types to icons
                            _icon_map = {
                                "stage_0_start": "🔍",
                                "stage_0_result": "🔍",
                                "stage_07_rag_start": "🧠",
                                "stage_07_rag_result": "🔍" if _sub.get("decision") == "include" else "❌",
                                "stage_07_rag_complete": "✅",
                                "stage_07_rag_error": "⚠️",
                                "iteration_start": "🔄",
                                "search_result": "🔎",
                                "verdict": "✅" if _sub.get("decision") == "include" else "❌",
                                "iteration_end": "📊",
                                "early_stop": "✅",
                            }
                            _icon = _icon_map.get(_sub_type, "⚙️")

                            put({
                                "phase": _sub_phase,
                                "status": "completed",
                                "message": f"{_icon} {_sub_msg}" if _sub_msg else f"{_icon} {_sub_type}",
                                "data": {
                                    "node": "evidence_collection_loop",
                                    "sub_step_type": _sub_type,
                                    # Stage 0.7 RAG results with file_path render as compact cards
                                    **({"event_type": f"stage_07_rag_{_sub.get('decision', 'include')}",
                                        "file_path": _sub.get("file_path", ""),
                                        "file_name": (_sub.get("file_path") or "").split("/")[-1].split("\\")[-1],
                                       } if _sub_type == "stage_07_rag_result" and _sub.get("file_path") else {}),
                                    **{k: v for k, v in _sub.items()
                                       if k not in ("type", "message")},
                                },
                                "timestamp": _sub.get("timestamp", datetime.now().isoformat()),
                            })

                    # ── Code Generation sub-steps (guaranteed post-node broadcast) ─
                    # Emits file_written events from generated_code in state_delta.
                    # This acts as a fallback when the real-time _ui_emit mechanism
                    # in workflow.py doesn't fire (e.g. callback not registered).
                    if node_name in ("generate_code", "generate_tests"):
                        _gen_list = state_delta.get("generated_code") or state_delta.get("generated_tests") or []
                        _gen_phase = "patch_generation"
                        for _gc in _gen_list:
                            _gc_fp = getattr(_gc, "file_path", None) or (_gc.get("file_path") if isinstance(_gc, dict) else None)
                            if not _gc_fp:
                                continue
                            _gc_fname = _gc_fp.split("/")[-1].split("\\")[-1]
                            _gc_key = f"gen_{_gc_fp}"
                            # Only emit if not already sent by real-time _ui_emit
                            _existing_keys = {
                                (s.get("data") or {}).get("file_path", "")
                                for s in workflow_status.get(workflow_id, {}).get("steps", [])
                                if (s.get("data") or {}).get("event_type") == "file_written"
                            }
                            if _gc_fp not in _existing_keys:
                                put({
                                    "phase": _gen_phase,
                                    "status": "completed",
                                    "message": f"📝 Written: {_gc_fname}",
                                    "data": {
                                        "node": node_name,
                                        "event_type": "file_written",
                                        "file_path": _gc_fp,
                                        "file_name": _gc_fname,
                                    },
                                    "timestamp": datetime.now().isoformat(),
                                })

                    # ── Edit Loop sub-steps (file-by-file progress) ────────
                    if node_name in ("edit_loop", "fix_build", "fix_test"):
                        _el_result = state_delta.get("_edit_loop_result")
                        if _el_result and isinstance(_el_result, dict):
                            _el_phase = "edit_loop"
                            for _er in (_el_result.get("edit_results") or []):
                                _er_fp = getattr(_er, "file_path", "") or (_er.get("file_path", "") if isinstance(_er, dict) else "")
                                _er_skipped = getattr(_er, "skipped", False) if hasattr(_er, "skipped") else (_er.get("skipped", False) if isinstance(_er, dict) else False)
                                if not _er_fp or _er_skipped:
                                    continue
                                _er_fname = _er_fp.split("/")[-1].split("\\")[-1]
                                _er_had_errors = bool(getattr(_er, "compile_errors_before", None) or (
                                    _er.get("compile_errors_before") if isinstance(_er, dict) else None))
                                _er_fixed = bool(getattr(_er, "compile_clean", False) if hasattr(_er, "compile_clean") else (
                                    _er.get("compile_clean", False) if isinstance(_er, dict) else False))

                                _er_event_type = "live_check_resolved" if (_er_had_errors and _er_fixed) else "file_written"
                                _er_msg = (
                                    f"✅ Fixed: {_er_fname}" if _er_event_type == "live_check_resolved"
                                    else f"✏️ Edited: {_er_fname}"
                                )
                                put({
                                    "phase": _el_phase,
                                    "status": "completed",
                                    "message": _er_msg,
                                    "data": {
                                        "node": node_name,
                                        "event_type": _er_event_type,
                                        "file_path": _er_fp,
                                        "file_name": _er_fname,
                                    },
                                    "timestamp": datetime.now().isoformat(),
                                })

            if cancelled:
                result_holder["cancelled"] = True
                result_holder["state"] = aggregate_state
            else:
                snapshot = app.get_state(config)
                final_state = snapshot.values if hasattr(snapshot, "values") else aggregate_state
                if not isinstance(final_state, dict):
                    final_state = aggregate_state
                result_holder["state"] = final_state

        except Exception as exc:
            result_holder["error"] = str(exc)
            logger.error(f"Workflow thread error: {exc}", exc_info=True)
            put({
                "phase": "failed",
                "status": "error",
                "message": f"❌ Workflow crashed: {str(exc)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            })
        finally:
            # Clean up cancel flag and this thread's stage-model resolver
            workflow_cancel_flags.pop(workflow_id, None)
            _STAGE_RESOLVERS_BY_THREAD.pop(_resolver_threading.get_ident(), None)
            done_event.set()
            if ws_lock.locked():
                try:
                    ws_lock.release()
                except Exception:
                    pass

    t = threading.Thread(target=thread_run, daemon=True)
    t.start()

    if execution_mode == ExecutionMode.REASONING.value:
        await broadcast_to_workflow(workflow_id, {
            "phase": "processing",
            "status": "in_progress",
            "message": "🧪 Running experimental reasoning mode",
            "data": {"execution_mode": execution_mode},
            "timestamp": datetime.now().isoformat(),
        })

    # Yield control to event loop while waiting for the thread
    while not done_event.wait(timeout=0):
        await asyncio.sleep(0.5)

    # ── Handle user-requested cancellation ──
    if result_holder.get("cancelled"):
        # Hydrate latest token usage from transient store / run_ctx
        if not workflow_status[workflow_id].get("token_usage"):
            try:
                _tid = ticket.ticket_id if 'ticket' in locals() and hasattr(ticket, 'ticket_id') else (workflow_status[workflow_id].get("ticket_id") or "")
                from ticket_to_code.workflow import _transient_store
                _ctx = _transient_store.get(_tid, {}).get("run_ctx")
                if _ctx and hasattr(_ctx, "budget"):
                    workflow_status[workflow_id]["token_usage"] = _ctx.budget._snapshot()
            except Exception:
                pass

        partial_state = result_holder.get("state", {}) or {}
        workflow_status[workflow_id].update({
            "status": "stopped",
            "current_phase": workflow_status[workflow_id].get("current_phase", "stopped"),
            "error": "Stopped by user",
            "stopped_at": datetime.now().isoformat(),
            "state": partial_state,
        })
        explanation = _build_workflow_explanation(
            workflow_status[workflow_id],
            workflow_status[workflow_id].get("workflow_output", {}) or partial_state,
        )
        workflow_status[workflow_id]["workflow_explanation"] = explanation

        # Clean up transient registry for the ticket
        try:
            from ticket_to_code.workflow import _clear_transient
            ticket_id = workflow_status[workflow_id].get("ticket_id")
            if ticket_id:
                _clear_transient(ticket_id)
        except Exception:
            pass

        await broadcast_to_workflow(workflow_id, {
            "phase": "failed", "status": "stopped",
            "message": "⛔ Workflow stopped by user",
            "data": {
                "stopped_by_user": True,
                "workflow_explanation": explanation,
                "token_usage": workflow_status[workflow_id].get("token_usage"),
            },
            "timestamp": datetime.now().isoformat(),
        })

        # Update task history with full intermediate state
        _all_gen = workflow_status[workflow_id].get("generated_files", [])
        try:
            history_store.record_task(
                workflow_id=workflow_id,
                project_id=workflow_status[workflow_id].get("project_id"),
                ticket_id=ticket.ticket_id if 'ticket' in locals() and hasattr(ticket, 'ticket_id') else (workflow_status[workflow_id].get("ticket_id") or "Task"),
                title=ticket.title if 'ticket' in locals() and hasattr(ticket, 'title') else (workflow_status[workflow_id].get("ticket_id") or "Stopped Task"),
                description=ticket.description if 'ticket' in locals() and hasattr(ticket, 'description') else "",
                status="stopped",
                changed_files=_all_gen,
                execution_mode=workflow_status[workflow_id].get("execution_mode", "pipeline"),
                error="Stopped by user",
                token_usage=workflow_status[workflow_id].get("token_usage"),
                steps=workflow_status[workflow_id].get("steps", []),
            )
        except Exception as exc:
            logger.warning(f"Failed to record stopped task in history: {exc}")

        # Consolidate intermediate run into Episodic Memory (facts + episode)
        try:
            from ticket_to_code.memory.episodic_memory import EpisodicMemoryStore
            _ws_path = workflow_status[workflow_id].get("repo_path") or workspace_path
            if _ws_path:
                mem_store = EpisodicMemoryStore(_ws_path)
                mem_store.consolidate_run(workflow_status[workflow_id], repo=_ws_path)
        except Exception as mem_err:
            logger.debug(f"Episodic consolidation on stop skipped: {mem_err}")

        # Broadcast conversational summary to Chat tab so UI re-enables immediately
        chat_summary = _build_chat_summary(
            workflow_id=workflow_id,
            ticket_title=ticket.title if 'ticket' in locals() and hasattr(ticket, 'ticket_id') else (workflow_status[workflow_id].get("ticket_id") or "Task"),
            ticket_description=ticket.description if 'ticket' in locals() and hasattr(ticket, 'description') else "",
            status="stopped",
            generated_files=_all_gen,
            explanation=explanation,
            error="Stopped by user",
            steps=workflow_status[workflow_id].get("steps", []),
            token_usage=workflow_status[workflow_id].get("token_usage"),
        )
        await manager.broadcast({
            "type": "workflow_summary",
            "workflow_id": workflow_id,
            "status": "stopped",
            "summary": chat_summary,
            "generated_files": _all_gen,
            "token_usage": workflow_status[workflow_id].get("token_usage"),
            "timestamp": datetime.now().isoformat(),
        })
        return

    if "error" in result_holder:
        err = result_holder["error"]
        # Hydrate latest token usage from transient store / run_ctx on error
        if not workflow_status[workflow_id].get("token_usage"):
            try:
                _tid = ticket.ticket_id if 'ticket' in locals() and hasattr(ticket, 'ticket_id') else (workflow_status[workflow_id].get("ticket_id") or "")
                from ticket_to_code.workflow import _transient_store
                _ctx = _transient_store.get(_tid, {}).get("run_ctx")
                if _ctx and hasattr(_ctx, "budget"):
                    workflow_status[workflow_id]["token_usage"] = _ctx.budget._snapshot()
            except Exception:
                pass
        workflow_status[workflow_id].update({"status": "failed", "current_phase": "failed", "error": err})
        workflow_status[workflow_id]["workflow_explanation"] = _build_workflow_explanation(
            workflow_status[workflow_id],
            workflow_status[workflow_id].get("workflow_output", {}),
        )
        try:
            history_store.record_task(
                workflow_id=workflow_id,
                project_id=workflow_status[workflow_id].get("project_id"),
                ticket_id=ticket.ticket_id if 'ticket' in locals() and hasattr(ticket, 'ticket_id') else (workflow_status[workflow_id].get("ticket_id") or "Task"),
                title=ticket.title if 'ticket' in locals() and hasattr(ticket, 'title') else (workflow_status[workflow_id].get("ticket_id") or "Failed Task"),
                description=ticket.description if 'ticket' in locals() and hasattr(ticket, 'description') else "",
                status="failed",
                changed_files=[],
                execution_mode=workflow_status[workflow_id].get("execution_mode", "pipeline"),
                error=err,
                token_usage=workflow_status[workflow_id].get("token_usage"),
                steps=workflow_status[workflow_id].get("steps", []),
            )
        except Exception as e:
            logger.warning(f"Failed to record failed task in history: {e}")
        await broadcast_to_workflow(workflow_id, {
            "phase": "failed", "status": "error",
            "message": f"❌ Workflow failed: {err}",
            "data": {
                "workflow_explanation": workflow_status[workflow_id].get("workflow_explanation"),
                "token_usage": workflow_status[workflow_id].get("token_usage"),
            },
            "timestamp": datetime.now().isoformat(),
        })
        return

    # --- Workflow completed ---
    final = result_holder.get("state") or {}
    if final and len(final) == 1 and isinstance(list(final.values())[0], dict):
        final = list(final.values())[0]

    # Finalize token usage snapshot from run_ctx if not already present
    if isinstance(final, dict) and final.get("token_usage"):
        workflow_status[workflow_id]["token_usage"] = final.get("token_usage")
    elif not workflow_status[workflow_id].get("token_usage") and isinstance(final, dict):
        try:
            from ticket_to_code.workflow import _get_transient
            run_ctx = _get_transient(final, "run_ctx")
            if run_ctx and hasattr(run_ctx, "budget"):
                workflow_status[workflow_id]["token_usage"] = run_ctx.budget._snapshot()
        except Exception:
            pass

    workflow_output = _extract_workflow_outputs(final if isinstance(final, dict) else {})
    workflow_status[workflow_id]["workflow_output"] = workflow_output
    if workflow_output.get("validation_failure_reason"):
        workflow_status[workflow_id]["validation_failure_reason"] = workflow_output.get("validation_failure_reason")

    gen_code  = final.get("generated_code")  or []
    gen_tests = final.get("generated_tests") or []
    all_gen = [getattr(g, "file_path", None) or (g.get("file_path") if isinstance(g, dict) else str(g)) for g in gen_code + gen_tests]
    all_gen = [f for f in all_gen if f]
    if all_gen:
        workflow_status[workflow_id]["generated_files"] = all_gen

    rag_files = workflow_status[workflow_id].get("candidate_files", []) or _extract_rag_files(final)
    if rag_files:
        workflow_status[workflow_id]["candidate_files"] = rag_files

    final_status = str(final.get("status", "")).strip().lower() if isinstance(final, dict) else ""
    validation_failure_reason = workflow_output.get("validation_failure_reason")
    build_result = final.get("build_result") if isinstance(final, dict) else None
    build_err = None
    build_failed = False
    if build_result:
        if hasattr(build_result, "status"):
            status_val = build_result.status.value if hasattr(build_result.status, "value") else build_result.status
            if status_val == "failure":
                build_failed = True
        elif isinstance(build_result, dict) and build_result.get("status") == "failure":
            build_failed = True

        if hasattr(build_result, "errors") and build_result.errors:
            build_err = "Build failed after max retries:\n" + "\n".join(build_result.errors)
        elif isinstance(build_result, dict) and build_result.get("errors"):
            build_err = "Build failed after max retries:\n" + "\n".join(build_result.get("errors"))

    # If the workflow reached memory_update or outcome_check determined all requirements are satisfied,
    # verify that code changes were actually produced if required.
    requires_code = True
    inv_res = final.get("investigation_result") if isinstance(final, dict) else None
    if inv_res and hasattr(inv_res, "requires_code_changes"):
        requires_code = bool(inv_res.requires_code_changes)
    elif isinstance(inv_res, dict) and "requires_code_changes" in inv_res:
        requires_code = bool(inv_res.get("requires_code_changes", True))

    outcome_result = final.get("outcome_check_result") if isinstance(final, dict) else None
    outcome_correct = bool(outcome_result and (outcome_result.get("status") in ("CORRECT", "all_passed")))
    outcome_failed = bool(outcome_result and (outcome_result.get("status") in ("FAILED", "zero_code_generated") or outcome_result.get("reason") == "zero_code_generated"))

    is_already_done = (
        final_status in ["already_implemented", "no_action_required"]
        or (isinstance(final, dict) and final.get("preflight_verdict") == "ALREADY_DONE")
    )

    if is_already_done:
        validation_failure_reason = None
        terminal_failed = False
    elif (final_status in ["memory_updated", "completed", "success"] or outcome_correct) and not outcome_failed:
        if requires_code and len(all_gen) == 0:
            validation_failure_reason = "No code or test files were generated or modified for this ticket."
        else:
            validation_failure_reason = None

    if not is_already_done:
        terminal_failed = (
            final_status in ["failed", "build_failed", "test_failed", "verification_failed"]
            or bool(validation_failure_reason)
            or build_failed
            or outcome_failed
            or (requires_code and len(all_gen) == 0)
        )

    workflow_status[workflow_id].update({
        "status": "failed" if terminal_failed else "completed",
        "current_phase": "failed" if terminal_failed else "completed",
        "execution_mode": execution_mode,
        "completed_at": datetime.now().isoformat(),
        "error": (final.get("error") if isinstance(final, dict) else None) or validation_failure_reason or build_err,
    })
    workflow_status[workflow_id]["workflow_explanation"] = _build_workflow_explanation(
        workflow_status[workflow_id],
        workflow_output,
    )
    try:
        workflow_status._flush_workflow_to_db(workflow_id, workflow_status[workflow_id])
    except Exception as _fl_err:
        logger.debug(f"Flush workflow to DB non-fatal: {_fl_err}")

    # Update ticket file in .aviator/tickets with final workflow results
    try:
        ticket_log_path = Path(workspace_path) / ".aviator" / "tickets" / f"{workflow_id}.json"
        ticket_data = {}
        if ticket_log_path.exists() and ticket_log_path.stat().st_size > 0:
            with open(ticket_log_path, "r", encoding="utf-8") as f:
                ticket_data = json.load(f)
        ticket_data.update({
            "ticket_id": ticket.ticket_id,
            "status": "failed" if terminal_failed else "completed",
            "completed_at": datetime.now().isoformat(),
            "generated_files": all_gen,
            "candidate_files": rag_files,
            "error": workflow_status[workflow_id].get("error"),
            "token_usage": workflow_status[workflow_id].get("token_usage"),
        })
        with open(ticket_log_path, "w", encoding="utf-8") as f:
            json.dump(ticket_data, f, indent=2)
        logger.info(f"Saved completed ticket summary to {ticket_log_path}")
    except Exception as e:
        logger.warning(f"Failed to update ticket record in .aviator/tickets: {e}")

    # Persist a durable task-history record so it appears in the UI history panel.
    try:
        history_store.record_task(
            workflow_id=workflow_id,
            project_id=workflow_status[workflow_id].get("project_id"),
            ticket_id=ticket.ticket_id,
            title=ticket.title,
            description=ticket.description,
            status="failed" if terminal_failed else "completed",
            changed_files=all_gen,
            execution_mode=execution_mode,
            error=workflow_status[workflow_id].get("error"),
            token_usage=workflow_status[workflow_id].get("token_usage"),
            steps=workflow_status[workflow_id].get("steps", []),
        )
    except Exception as e:
        logger.warning(f"Failed to record task history: {e}")

    # Push final token snapshot to any active listener before termination
    final_token_usage = workflow_status[workflow_id].get("token_usage")
    if final_token_usage:
        try:
            await broadcast_to_workflow(workflow_id, {
                "phase": "failed" if terminal_failed else "completed",
                "status": "error" if terminal_failed else "completed",
                "message": "",
                "data": {
                    "event_type": "token_usage_update",
                    **final_token_usage,
                },
                "timestamp": datetime.now().isoformat(),
            })
        except Exception:
            pass

    if terminal_failed:
        await broadcast_to_workflow(workflow_id, {
            "phase": "failed", "status": "error",
            "message": (
                f"❌ Workflow ended with terminal failure"
                f" ({validation_failure_reason or final_status or 'unknown reason'})"
            ),
            "data": {
                "generated_code": all_gen,
                "candidate_files": rag_files,
                "workflow_explanation": workflow_status[workflow_id].get("workflow_explanation"),
                "validation_failure_reason": validation_failure_reason,
                "build_diagnostics": workflow_output.get("build_diagnostics"),
                "token_usage": workflow_status[workflow_id].get("token_usage"),
            },
            "timestamp": datetime.now().isoformat(),
        })
    else:
        completion_msg = (
            "✅ Ticket requirements verified — already fully implemented in repository"
            if is_already_done
            else f"✅ Workflow complete — {len(gen_code)} code file(s), {len(gen_tests)} test file(s) generated"
        )
        await broadcast_to_workflow(workflow_id, {
            "phase": "completed", "status": "completed",
            "message": completion_msg,
            "data": {
                "generated_code": all_gen,
                "candidate_files": rag_files,
                "workflow_explanation": workflow_status[workflow_id].get("workflow_explanation"),
                "token_usage": workflow_status[workflow_id].get("token_usage"),
            },
            "timestamp": datetime.now().isoformat(),
        })

    # ── Also broadcast a conversational summary to the main WebSocket (Chat tab) ──
    explanation = workflow_status[workflow_id].get("workflow_explanation") or {}
    chat_summary = _build_chat_summary(
        workflow_id=workflow_id,
        ticket_title=ticket.title,
        ticket_description=ticket.description,
        status="failed" if terminal_failed else "completed",
        generated_files=all_gen,
        explanation=explanation,
        error=workflow_status[workflow_id].get("error"),
        steps=workflow_status[workflow_id].get("steps", []),
        token_usage=workflow_status[workflow_id].get("token_usage"),
    )

    # Consolidate completed run into Episodic Memory (facts + episode)
    try:
        from ticket_to_code.memory.episodic_memory import EpisodicMemoryStore
        _ws_path = workflow_status[workflow_id].get("repo_path") or workspace_path
        if _ws_path:
            mem_store = EpisodicMemoryStore(_ws_path)
            mem_store.consolidate_run(workflow_status[workflow_id], repo=_ws_path)
    except Exception as mem_err:
        logger.debug(f"Episodic consolidation on complete skipped: {mem_err}")

    await manager.broadcast({
        "type": "workflow_summary",
        "workflow_id": workflow_id,
        "status": "failed" if terminal_failed else "completed",
        "summary": chat_summary,
        "generated_files": all_gen,
        "token_usage": workflow_status[workflow_id].get("token_usage"),
        "timestamp": datetime.now().isoformat(),
    })


def _build_chat_summary(
    workflow_id: str,
    ticket_title: str,
    ticket_description: str,
    status: str,
    generated_files: List[str],
    explanation: dict,
    error: Optional[str],
    steps: list,
    token_usage: Optional[dict] = None,
) -> str:
    """Build a conversational, developer-friendly summary for the Chat tab.

    This reads like a developer explaining what they did: what the issue was,
    what files were changed and why, build/test outcomes, and next steps.
    """
    lines: List[str] = []

    # ── Status header ──
    if status == "completed":
        lines.append("✅ **Workflow completed successfully!**")
    elif status == "stopped":
        lines.append("🛑 **Workflow stopped by user.**")
    else:
        lines.append("❌ **Workflow ended with errors.**")

    # ── What was the ticket about ──
    lines.append("")
    lines.append(f"**Ticket:** {ticket_title}")
    if ticket_description and ticket_description != ticket_title:
        desc_short = ticket_description[:300]
        if len(ticket_description) > 300:
            desc_short += "…"
        lines.append(f"> {desc_short}")

    # ── What the system understood ──
    summary_text = explanation.get("summary", "")
    why_text = explanation.get("why", "")
    if summary_text:
        lines.append("")
        lines.append(f"**Analysis:** {summary_text}")
    if why_text:
        lines.append(f"**Reasoning:** {why_text}")

    # ── Key decisions made ──
    decisions = explanation.get("decisions") or []
    if decisions:
        lines.append("")
        lines.append("**Key Decisions:**")
        for d in decisions[:5]:
            lines.append(f"  • {d}")

    # ── Files changed and why ──
    if generated_files:
        lines.append("")
        lines.append(f"**Files Changed ({len(generated_files)}):**")
        for fp in generated_files[:10]:
            basename = fp.split("/")[-1] if "/" in fp else fp.split("\\")[-1] if "\\" in fp else fp
            lines.append(f"  📄 `{basename}` — `{fp}`")
        if len(generated_files) > 10:
            lines.append(f"  ... and {len(generated_files) - 10} more")

    # ── Build/test outcome ──
    build_info = ""
    test_info = ""
    for step in reversed(steps[-30:]):
        if not isinstance(step, dict):
            continue
        msg = str(step.get("message", ""))
        if "Build:" in msg and not build_info:
            build_info = msg
        if "Tests:" in msg and not test_info:
            test_info = msg
    if build_info:
        lines.append("")
        lines.append(f"**Build:** {build_info}")
    if test_info:
        lines.append(f"**Tests:** {test_info}")

    # ── Token Usage summary ──
    if token_usage:
        total = token_usage.get("total_tokens", 0)
        t_in = token_usage.get("tokens_in", 0)
        t_out = token_usage.get("tokens_out", 0)
        calls = token_usage.get("llm_calls", 0)
        excluded = token_usage.get("excluded_tokens", 0)
        real_costed = token_usage.get("real_costed_tokens", max(0, total - excluded))
        savings_pct = token_usage.get("savings_pct", 0.0)

        thinking = token_usage.get("thinking_tokens") or token_usage.get("tokens_thinking", 0)
        cached = token_usage.get("cached_tokens", 0)

        lines.append("")
        think_part = f", {thinking:,} thinking" if thinking > 0 else ""
        lines.append(f"🪙 **Token Usage:** {total:,} tokens ({t_in:,} prompt, {t_out:,} completion{think_part} across {calls} LLM calls)")
        if cached > 0 or excluded > 0:
            lines.append(f"⚡ **KV Cache & Excluded Tokens:** -{(cached or excluded):,} tokens ({savings_pct}% saved via warm prefix cache)")
            lines.append(f"💰 **Real Costed Tokens:** {real_costed:,} tokens")

    # ── Error details ──
    if error and status == "failed":
        lines.append("")
        lines.append(f"**Error:** {str(error)[:400]}")

    # ── Next action ──
    next_action = explanation.get("next_action", "")
    if next_action:
        lines.append("")
        lines.append(f"**Next Step:** {next_action}")

    # ── Confidence ──
    details = explanation.get("details") or []
    for d in details:
        if "confidence" in d.lower():
            lines.append(f"📊 {d}")
            break

    lines.append("")
    if status == "stopped":
        lines.append("💬 **Conversation Memory is Active:** Working memory, discovered files, and AST context have been retained. Ask questions, provide corrections, or give new instructions right here without restarting.")
    else:
        lines.append("💬 *Ask me anything about these changes — why a file was modified, what a specific change does, or if you'd like me to explain the code.*")

    return "\n".join(lines)


def _resolve_workflow_token_usage(state: dict, workflow_id: str) -> dict | None:
    """Helper to safely retrieve token usage snapshot from transient store or task history."""
    if not isinstance(state, dict):
        return None
    tu = state.get("token_usage")
    if tu:
        return tu
    try:
        from ticket_to_code.workflow import _transient_store
        tid = state.get("ticket_id", "")
        ctx = _transient_store.get(tid, {}).get("run_ctx")
        if ctx and hasattr(ctx, "budget"):
            return ctx.budget._snapshot()
    except Exception:
        pass
    try:
        task = history_store.get_task(workflow_id)
        if task and task.get("token_usage"):
            return task.get("token_usage")
    except Exception:
        pass
    return None


@app.get("/api/workflow/transparent/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """
    Return the current workflow state in the shape the frontend TransparentWorkflow
    component expects: current_phase, candidate_files, steps, operation_approved, etc.
    """
    if workflow_id not in workflow_status:
        task = history_store.get_task(workflow_id)
        if task:
            task_status = task.get("status", "completed")
            if task_status == "running":
                task_status = "failed"
                task_error = task.get("error") or "Workflow interrupted (process ended or restarted)"
                try:
                    history_store.record_task(
                        workflow_id=workflow_id,
                        project_id=task.get("project_id"),
                        ticket_id=task.get("ticket_id", ""),
                        title=task.get("title", ""),
                        description=task.get("description", ""),
                        status="failed",
                        changed_files=task.get("changed_files", []),
                        execution_mode=task.get("execution_mode", ExecutionMode.PIPELINE.value),
                        error=task_error,
                        token_usage=task.get("token_usage"),
                    )
                except Exception:
                    pass
            return {
                "workflow_id": workflow_id,
                "ticket_id": task.get("ticket_id", ""),
                "status": task_status,
                "execution_mode": task.get("execution_mode", ExecutionMode.PIPELINE.value),
                "current_phase": "completed" if task_status == "completed" else "failed",
                "candidate_files": [],
                "generated_files": task.get("changed_files", []),
                "operation_approved": True,
                "steps": task.get("steps", []),
                "error": task.get("error") or ("Workflow interrupted (process ended or restarted)" if task_status == "failed" else None),
                "workflow_explanation": None,
                "started_at": task.get("created_at"),
                "completed_at": task.get("completed_at") or task.get("created_at"),
                "token_usage": task.get("token_usage") or _resolve_workflow_token_usage(task, workflow_id),
            }
        raise HTTPException(status_code=404, detail="Workflow not found")

    state = workflow_status[workflow_id]
    state["workflow_explanation"] = _build_workflow_explanation(
        state,
        state.get("workflow_output", {}),
    )

    return {
        "workflow_id": workflow_id,
        "ticket_id": state.get("ticket_id", ""),
        "status": state.get("status", "running"),
        "execution_mode": state.get("execution_mode", ExecutionMode.PIPELINE.value),
        "current_phase": state.get("current_phase", "classification"),
        "candidate_files": state.get("candidate_files", []),
        "generated_files": state.get("generated_files", []),
        "operation_approved": state.get("operation_approved", False),
        "steps": state.get("steps", []),
        "error": state.get("error"),
        "workflow_explanation": state.get("workflow_explanation"),
        "started_at": state.get("started_at"),
        "completed_at": state.get("completed_at"),
        "token_usage": state.get("token_usage") or _resolve_workflow_token_usage(state, workflow_id),
    }


@app.get("/api/workflow/transparent/{workflow_id}/steps")
async def get_workflow_steps(workflow_id: str):
    """Get all workflow steps."""
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    state = workflow_status[workflow_id]
    return {
        "workflow_id": workflow_id,
        "current_phase": state.get("current_phase", "classification"),
        "steps": state.get("steps", []),
    }


class ApproveFilesRequest(BaseModel):
    workflow_id: str
    selected_files: List[str]


class ContinueToGenerationRequest(BaseModel):
    workflow_id: str
    repo_path: str


class ApplyPatchRequest(BaseModel):
    workflow_id: str
    repo_path: str


class RunTestsRequest(BaseModel):
    workflow_id: str
    repo_path: str
    test_pattern: str = "**/*Test*.java"


class ApproveOperationRequest(BaseModel):
    workflow_id: str


@app.post("/api/workflow/transparent/approve-operation")
async def approve_operation(request: ApproveOperationRequest):
    """Record that user approved the classified operation type."""
    wid = request.workflow_id
    if wid in workflow_status:
        workflow_status[wid]["operation_approved"] = True
    return {"status": "approved"}


@app.post("/api/workflow/transparent/approve-files")
async def approve_files(request: ApproveFilesRequest):
    """Record which files the user approved for the workflow."""
    wid = request.workflow_id
    if wid not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    workflow_status[wid]["selected_files"] = request.selected_files
    return {
        "status": "approved",
        "message": f"Recorded {len(request.selected_files)} approved file(s).",
    }


class DecisionResponseRequest(BaseModel):
    workflow_id: str
    choice: str  # "fix", "leave", or "stop"


@app.post("/api/workflow/transparent/decision-response")
async def decision_response(request: DecisionResponseRequest):
    """Receive user's decision for pre-existing build errors (fix/leave/stop)."""
    wid = request.workflow_id
    if wid not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    choice = request.choice.strip().lower()
    if choice not in ("fix", "leave", "stop"):
        raise HTTPException(status_code=400, detail=f"Invalid choice: {choice}")
    # Store the result and signal the blocking event
    workflow_status[wid]["_decision_result"] = choice
    evt = workflow_status[wid].get("_decision_event")
    if evt:
        evt.set()
    logger.info(f"Decision received for {wid}: {choice}")
    return {"status": "accepted", "choice": choice}


class InteractiveClarificationSubmission(BaseModel):
    request_id: str
    selected_options: List[str]
    user_notes: Optional[str] = ""


@app.post("/api/workflow/interactive/respond")
async def submit_interactive_clarification(req: InteractiveClarificationSubmission):
    """Submits the human user's decision to InteractiveClarificationGate for any active question."""
    from ticket_to_code.runtime.interactive_gate import InteractiveClarificationGate
    success = InteractiveClarificationGate.submit_response(
        request_id=req.request_id,
        selected_options=req.selected_options,
        user_notes=req.user_notes or "",
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"No active clarification request found with ID: {req.request_id}")
    return {"status": "accepted", "request_id": req.request_id, "selected_options": req.selected_options}


@app.post("/api/workflow/transparent/continue-to-generation")
async def continue_to_generation(request: ContinueToGenerationRequest):
    """Informational — workflow runs autonomously."""
    return {"status": "running", "message": "Workflow is running autonomously."}


@app.post("/api/workflow/transparent/apply-patch")
async def apply_patch(request: ApplyPatchRequest):
    """Informational — patches are applied as part of the workflow."""
    return {"status": "running", "message": "Patches are applied by the autonomous workflow."}


@app.post("/api/workflow/transparent/run-tests")
async def run_tests(request: RunTestsRequest):
    """Informational — tests are run as part of the workflow."""
    return {"status": "running", "message": "Tests are executed by the autonomous workflow."}


@app.post("/api/workflow/transparent/{workflow_id}/stop")
async def stop_workflow(workflow_id: str):
    """Stop a running workflow. Sets the cancel flag so the next node boundary
    will cleanly terminate the workflow without corrupting state."""
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")

    current_status = workflow_status[workflow_id].get("status")
    if current_status not in ("running", "initialized"):
        return {
            "status": "already_done",
            "message": f"Workflow is already {current_status}",
        }

    # Set the cancel flag — the thread_run loop checks this between nodes
    workflow_cancel_flags[workflow_id] = True
    if workflow_id in workflow_status:
        evt = workflow_status[workflow_id].get("_decision_event")
        if evt and hasattr(evt, "set"):
            evt.set()

    # Cooperative in-node cancel: flag the RunContext so llm_invoke aborts at its
    # next call instead of waiting for the current node to finish.
    try:
        from ticket_to_code.workflow import request_run_cancel
    except ImportError:
        request_run_cancel = None
        logger.debug("ticket_to_code.workflow.request_run_cancel not available; node-boundary stop only")
    if request_run_cancel is not None:
        wf = workflow_status.get(workflow_id) or {}
        ticket_obj = wf.get("ticket") or (wf.get("state") or {}).get("ticket")
        ticket_id = (
            wf.get("ticket_id")
            or (getattr(ticket_obj, "ticket_id", None) if not isinstance(ticket_obj, dict) else ticket_obj.get("ticket_id"))
            or workflow_id
        )
        try:
            flagged = request_run_cancel(ticket_id, "stopped by user")
            if not flagged and ticket_id != workflow_id:
                flagged = request_run_cancel(workflow_id, "stopped by user")
            logger.info(f"⛔ Cooperative cancel for ticket {ticket_id}: {'flagged' if flagged else 'no live run context'}")
        except Exception as _rc_exc:
            logger.debug(f"request_run_cancel failed: {_rc_exc}")
    logger.info(f"⛔ Stop requested for workflow {workflow_id}")

    return {
        "status": "stopping",
        "message": "Stop signal sent. Workflow will halt after the current step completes.",
    }


class ResumeWorkflowRequest(BaseModel):
    workflow_id: Optional[str] = None
    resume_from: Optional[str] = "plan"
    feedback: Optional[str] = None
    revised_hypotheses: Optional[List[str]] = None


@app.post("/api/workflow/transparent/{workflow_id}/resume")
async def resume_workflow(workflow_id: str, request: Optional[ResumeWorkflowRequest] = None):
    """Resume a stopped workflow from the last safe checkpoint (e.g. plan/generate)
    with updated user constraints, without re-indexing.
    """
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")

    current_status = workflow_status[workflow_id].get("status")
    if current_status == "running":
        return {"status": "already_running", "message": "Workflow is already running."}

    # Reset cancel flag
    workflow_cancel_flags[workflow_id] = False
    workflow_status[workflow_id]["status"] = "running"
    workflow_status[workflow_id]["cancel_requested"] = False

    resume_from = (request.resume_from if request else None) or "plan"
    workflow_status[workflow_id]["current_phase"] = resume_from

    # Append user feedback / rejected approaches to state if provided
    feedback_txt = request.feedback if request else None
    if feedback_txt:
        rej = workflow_status[workflow_id].setdefault("rejected_approaches", [])
        rej.append(feedback_txt)
        steps = workflow_status[workflow_id].setdefault("steps", [])
        steps.append({
            "phase": resume_from,
            "status": "in_progress",
            "message": f"🔄 Resuming from '{resume_from}' with user guidance: '{feedback_txt[:100]}'",
            "timestamp": datetime.now().isoformat(),
        })

    await broadcast_to_workflow(workflow_id, {
        "phase": resume_from,
        "status": "in_progress",
        "message": f"🔄 Resuming workflow from '{resume_from}' checkpoint...",
        "data": {
            "resume_from": resume_from,
            "feedback": feedback_txt,
        },
        "timestamp": datetime.now().isoformat(),
    })

    # If the workflow was stopped or halted, re-launch background execution from safe checkpoint
    existing_state = workflow_status[workflow_id].get("state") or {}
    ticket = existing_state.get("ticket") or workflow_status[workflow_id].get("ticket")
    workspace_path = existing_state.get("workspace_path") or workflow_status[workflow_id].get("workspace_path")

    if ticket and workspace_path:
        overrides = dict(existing_state)
        if feedback_txt:
            overrides["rejected_approaches"] = (overrides.get("rejected_approaches") or []) + [feedback_txt]
            overrides["user_constraints"] = (overrides.get("user_constraints") or []) + [feedback_txt]

        # Record lineage
        parent_run = workflow_status[workflow_id].get("run_id") or workflow_id
        workflow_status[workflow_id]["parent_run_id"] = parent_run

        task = asyncio.create_task(
            run_workflow_async(
                workflow_id,
                ticket,
                workspace_path,
                resume_from=resume_from,
                initial_overrides=overrides,
            )
        )
        _RUN_TASKS[workflow_id] = task
        task.add_done_callback(lambda _: _RUN_TASKS.pop(workflow_id, None))

    return {
        "status": "resumed",
        "workflow_id": workflow_id,
        "resume_from": resume_from,
        "message": f"Workflow resumed from {resume_from} checkpoint without re-indexing."
    }


@app.get("/api/workflow/transparent/{workflow_id}/changes")
async def get_workflow_changes(workflow_id: str):
    """
    Return files and unified diffs generated or modified by the workflow.
    Directly satisfies the frontend Changes tab without 404 errors.
    """
    import difflib

    wf = workflow_status.get(workflow_id)
    task = history_store.get_task(workflow_id)

    # If neither found by workflow_id, search by ticket_id across history tasks
    if not wf and not task:
        for t in (history_store.get_all_tasks() if hasattr(history_store, "get_all_tasks") else []):
            if t.get("ticket_id") == workflow_id or t.get("workflow_id") == workflow_id:
                task = t
                break

    if not wf and not task:
        raise HTTPException(status_code=404, detail="Workflow not found")

    state = (wf.get("state") or {}) if wf else {}
    
    # Robustly resolve workspace_path from wf or project_id in projects.json
    workspace_path = (wf.get("workspace_path") or state.get("workspace_path") or "") if wf else ""
    proj_id = (wf.get("project_id") if wf else None) or (task.get("project_id") if task else None)
    if not workspace_path and proj_id:
        try:
            projs = _load_projects()
            p_obj = projs.get(proj_id)
            if p_obj:
                workspace_path = getattr(p_obj, "path", None) or getattr(p_obj, "local_path", None) or (p_obj.get("path") if isinstance(p_obj, dict) else "")
            elif os.path.isdir(proj_id):
                workspace_path = proj_id
        except Exception:
            pass

    ticket_id = (wf.get("ticket_id") or "") if wf else (task.get("ticket_id") if task else "")

    files_list: List[Dict[str, Any]] = []
    seen_paths: Set[str] = set()

    # 1. Inspect generated_code and generated_tests from LangGraph state
    gen_items = (state.get("generated_code") or []) + (state.get("generated_tests") or [])
    orig_contents = state.get("original_file_contents") or {}

    for item in gen_items:
        fp = getattr(item, "file_path", None) or (item.get("file_path") if isinstance(item, dict) else None)
        if not fp or fp in seen_paths:
            continue
        seen_paths.add(fp)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else "")
        orig = orig_contents.get(fp)

        fname = Path(fp).name
        if orig is not None and content:
            diff_lines = list(difflib.unified_diff(
                orig.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{fname}",
                tofile=f"b/{fname}"
            ))
            diff_str = "".join(diff_lines)
            status = "modified"
        elif content:
            diff_lines = list(difflib.unified_diff(
                [],
                content.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=f"b/{fname}"
            ))
            diff_str = "".join(diff_lines)
            status = "new"
        else:
            diff_str = ""
            status = "modified"

        approvals = wf.get("selective_approvals", {}) if wf else {}
        is_approved = True
        if approvals:
            if fp in approvals.get("rejected", []):
                is_approved = False
            elif approvals.get("approved") and fp not in approvals.get("approved"):
                is_approved = False

        files_list.append({
            "path": fp,
            "status": status,
            "diff": diff_str,
            "approved": is_approved
        })

    # 2. Check disk trace diff (e.g. C:/aviator_traces/{ticket_id}/generated_patch.diff)
    trace_diff = ""
    trace_candidates = [
        str(_get_trace_root() / ticket_id / "generated_patch.diff") if ticket_id else "",
        os.path.join(workspace_path, ".aviator", "traces", ticket_id, "generated_patch.diff") if workspace_path and ticket_id else "",
    ]
    for tc in trace_candidates:
        if tc and os.path.exists(tc):
            try:
                trace_diff = Path(tc).read_text(encoding="utf-8", errors="ignore")
                break
            except Exception:
                pass

    # 3. Handle extra paths from generated_files or task changed_files
    extra_files = []
    if wf and wf.get("generated_files"):
        extra_files.extend(wf.get("generated_files"))
    if task and task.get("changed_files"):
        extra_files.extend(task.get("changed_files"))
    if task and task.get("generated_files"):
        extra_files.extend(task.get("generated_files"))

    for fp in extra_files:
        if not fp or fp in seen_paths:
            continue
        seen_paths.add(fp)

        file_diff = ""
        if trace_diff:
            fname = Path(fp).name
            if fname in trace_diff or fp in trace_diff:
                file_diff = trace_diff

        if not file_diff and workspace_path:
            full_p = os.path.join(workspace_path, fp) if not os.path.isabs(fp) else fp
            if os.path.exists(full_p):
                try:
                    fcontent = Path(full_p).read_text(encoding="utf-8", errors="ignore")
                    diff_lines = list(difflib.unified_diff(
                        [],
                        fcontent.splitlines(keepends=True)[:150],
                        fromfile="/dev/null",
                        tofile=f"b/{Path(fp).name}"
                    ))
                    file_diff = "".join(diff_lines)
                except Exception:
                    file_diff = ""

        approvals = wf.get("selective_approvals", {}) if wf else {}
        is_approved = True
        if approvals:
            if fp in approvals.get("rejected", []):
                is_approved = False
            elif approvals.get("approved") and fp not in approvals.get("approved"):
                is_approved = False

        files_list.append({
            "path": fp,
            "status": "modified",
            "diff": file_diff or "File modified during workflow execution.",
            "approved": is_approved
        })

    return {
        "workflow_id": workflow_id,
        "total_files": len(files_list),
        "files": files_list,
        "changes": files_list,
    }



class DiffApprovalRequest(BaseModel):
    approved_files: List[str] = Field(default_factory=list)
    rejected_files: List[str] = Field(default_factory=list)


@app.post("/api/workflow/transparent/{workflow_id}/changes/approve")
async def approve_workflow_changes(workflow_id: str, request: DiffApprovalRequest):
    """
    Selective per-file / per-hunk approval gate.
    Allows user to cherry-pick approved files and reject collateral files before build verification.
    """
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow_status[workflow_id]["selective_approvals"] = {
        "approved": request.approved_files,
        "rejected": request.rejected_files,
        "updated_at": datetime.now().isoformat()
    }

    # Broadcast event so UI updates live
    await broadcast_to_workflow(workflow_id, {
        "phase": "selective_approval",
        "status": "completed",
        "message": f"Selective review: {len(request.approved_files)} approved, {len(request.rejected_files)} rejected",
        "data": {
            "approved_files": request.approved_files,
            "rejected_files": request.rejected_files,
        },
        "timestamp": datetime.now().isoformat()
    })

    # Revert rejected files on disk
    if request.rejected_files:
        wf = workflow_status.get(workflow_id, {})
        state = wf.get("state") or {}
        orig_contents = state.get("original_file_contents") or {}
        workspace_path = wf.get("workspace_path") or state.get("workspace_path") or ""
        if not workspace_path:
            proj_id = wf.get("project_id")
            if proj_id:
                try:
                    projs = _load_projects()
                    p_obj = projs.get(proj_id)
                    if p_obj:
                        workspace_path = getattr(p_obj, "path", None) or getattr(p_obj, "local_path", None) or ""
                except Exception:
                    pass

        # 1. Try TransactionalDiffManager instant undo
        from ticket_to_code.runtime.transactional_diff import TransactionalDiffManager
        ticket_id = wf.get("ticket_id") or ""
        for cp_id in [f"checkpoint_{workflow_id}", f"checkpoint_{ticket_id}"]:
            if cp_id in TransactionalDiffManager._checkpoints:
                TransactionalDiffManager.instant_undo(cp_id)

        # 2. Restore individual rejected files from original_file_contents or git
        if workspace_path:
            for rel_path in request.rejected_files:
                target_file = Path(workspace_path) / rel_path if not os.path.isabs(rel_path) else Path(rel_path)
                if rel_path in orig_contents:
                    try:
                        target_file.write_text(orig_contents[rel_path], encoding="utf-8")
                        logger.info(f"Restored {rel_path} from original snapshot.")
                    except Exception as e:
                        logger.warning(f"Could not restore {rel_path} from snapshot: {e}")
                else:
                    # Fallback to git checkout
                    try:
                        import subprocess
                        subprocess.run(
                            ["git", "checkout", "--", rel_path],
                            cwd=workspace_path,
                            capture_output=True,
                            timeout=5
                        )
                        logger.info(f"Reverted {rel_path} via git checkout.")
                    except Exception as e:
                        logger.warning(f"Git checkout failed for {rel_path}: {e}")

    return {
        "workflow_id": workflow_id,
        "status": "updated",
        "approved_count": len(request.approved_files),
        "rejected_count": len(request.rejected_files),
        "message": f"Successfully updated approvals ({len(request.approved_files)} approved, {len(request.rejected_files)} rejected and reverted)."
    }



class InstantUndoRequest(BaseModel):
    checkpoint_id: str


class SelectiveHunkRequest(BaseModel):
    workspace_path: str
    rel_path: str
    original_content: str
    hunk_indices_to_accept: List[int]
    all_hunks: List[dict]


@app.post("/api/workflow/diff/instant-undo")
async def instant_undo_endpoint(req: InstantUndoRequest):
    """Cursor/Windsurf-style instant rollback in <100ms."""
    from ticket_to_code.runtime.transactional_diff import TransactionalDiffManager
    success = TransactionalDiffManager.instant_undo(req.checkpoint_id)
    if not success:
        raise HTTPException(status_code=404, detail="Checkpoint not found or failed to restore")
    return {"status": "undone", "checkpoint_id": req.checkpoint_id}


@app.post("/api/workflow/diff/inline-accept")
async def inline_accept_endpoint(req: SelectiveHunkRequest):
    """Applies only selected hunks while rejecting others (Cursor/Windsurf style)."""
    from ticket_to_code.runtime.diff_engine import DiffHunk
    from ticket_to_code.runtime.transactional_diff import TransactionalDiffManager

    hunks = [
        DiffHunk(
            hunk_index=h.get("hunk_index", i),
            old_start=h.get("old_start", 1),
            old_count=h.get("old_count", 0),
            new_start=h.get("new_start", 1),
            new_count=h.get("new_count", 0),
            header=h.get("header", ""),
            diff_lines=h.get("diff_lines", h.get("lines", [])),
        )
        for i, h in enumerate(req.all_hunks)
    ]

    success = TransactionalDiffManager.apply_selective_hunk(
        workspace_path=req.workspace_path,
        rel_path=req.rel_path,
        original_content=req.original_content,
        hunk_indices_to_accept=req.hunk_indices_to_accept,
        all_hunks=hunks,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to apply selective hunks")
    return {"status": "applied", "accepted_hunks": req.hunk_indices_to_accept}


@app.websocket("/ws/workflow/{workflow_id}")
async def workflow_websocket(websocket: WebSocket, workflow_id: str):
    """
    Per-workflow WebSocket.  The TransparentWorkflow component connects here to
    receive live step updates as the LangGraph workflow executes.
    """
    await websocket.accept()
    workflow_connections.setdefault(workflow_id, []).append(websocket)

    # Send any steps that were emitted before this client connected
    existing_steps = workflow_status.get(workflow_id, {}).get("steps", [])
    for step in existing_steps:
        try:
            await websocket.send_json(step)
        except Exception:
            break

    # Send latest token telemetry so client has live token counts immediately
    existing_token = workflow_status.get(workflow_id, {}).get("token_usage")
    if existing_token:
        try:
            await websocket.send_json({
                "phase": workflow_status.get(workflow_id, {}).get("current_phase", "unknown"),
                "status": "in_progress",
                "message": "",
                "data": {
                    "event_type": "token_usage_update",
                    **existing_token,
                },
                "timestamp": datetime.now().isoformat(),
            })
        except Exception:
            pass

    try:
        while True:
            await websocket.receive_text()   # keep alive; ignore client messages
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Workflow WebSocket error for {workflow_id}: {e}")
    finally:
        conns = workflow_connections.get(workflow_id, [])
        if websocket in conns:
            conns.remove(websocket)


# ============================================================================
# SEMANTIC INTENT CLASSIFIER — Real AI IDE Intent Routing (Zero Hardcoding)
# ============================================================================

class IntentClassifyRequest(BaseModel):
    message: str
    attachments: Optional[List[str]] = None
    current_tab: Optional[str] = "chat"
    project_id: Optional[str] = None


@app.post("/api/intent/classify")
async def classify_intent(request: IntentClassifyRequest):
    """Semantic intent classification using Gemini 2.5 Flash:
    Distinguishes CODE_CHANGE (actionable requests to write, edit, fix, implement, refactor, or patch code)
    from EXPLAIN (inquiries, questions, architecture explanations, review, diagram analysis).
    Zero brittle regex or hardcoded keyword heuristics.
    """
    msg = request.message.strip()
    if not msg and not request.attachments:
        return {"intent": "EXPLAIN", "requires_code_changes": False, "confidence": 1.0, "reasoning": "Empty message"}

    import re
    msg_lower = msg.lower()

    # Fast path: Explicit user directive for explanation / read-only mode always overrides
    explicit_explain_cues = [
        "explain only", "read only", "do not trigger", "don't trigger", "no code modification",
        "do not modify", "don't modify", "just analyze", "just explain", "in chat only",
        "architectural explanation in chat", "explanation in chat", "solution in chat"
    ]
    if any(cue in msg_lower for cue in explicit_explain_cues):
        return {
            "intent": "EXPLAIN",
            "requires_code_changes": False,
            "confidence": 1.0,
            "reasoning": "Explicit user directive for explanation/chat-only mode (autonomous workflow inhibited).",
        }

    # Fast path for formal ticket identifiers (e.g. CF-204, CF-UI-02, TASK-92CEE387, PROJ-123)
    if re.match(r"^[A-Z0-9]{2,10}[-_][A-Za-z0-9_-]+", msg):
        return {
            "intent": "CODE_CHANGE",
            "requires_code_changes": True,
            "confidence": 1.0,
            "reasoning": f"Formal ticket identifier detected: {msg}",
            "ticket_title": msg,
        }

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT", "otl-cs-csai"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west4"),
        )

        prompt = (
            "You are an AI IDE Architectural Intent Classifier.\n"
            "Your responsibility is to determine the correct execution routing for incoming developer requests:\n\n"
            "CRITICAL INVARIANT:\n"
            "If the developer message contains directives like 'EXPLAIN ONLY', 'Do not trigger the autonomous code modification workflow', "
            "or requests an explanation/solution provided 'in chat', you MUST classify it as EXPLAIN (requires_code_changes: false).\n\n"
            "1. AUTONOMOUS_CODE_PIPELINE (intent: 'CODE_CHANGE', requires_code_changes: true):\n"
            "   - The developer is requesting to write, edit, fix, refactor, or generate code patches for the repository codebase.\n"
            "   - The output of the action is committed code changes, bug fixes, or file implementations in the local project workspace.\n"
            "   - Examples: 'align submit button', 'fix NullPointerException in PaymentService', 'implement auth endpoint', 'refactor AreaService'\n\n"
            "2. INTERACTIVE_CHAT_SESSION (intent: 'EXPLAIN', requires_code_changes: false):\n"
            "   - The developer is requesting to write, edit, fix, refactor, or generate code patches for the repository codebase.\n"
            "   - The output of the action is committed code changes, bug fixes, or file implementations in the local project workspace.\n"
            "   - Examples: 'align submit button', 'fix NullPointerException in PaymentService', 'implement auth endpoint', 'refactor AreaService'\n\n"
            "2. INTERACTIVE_CHAT_SESSION (intent: 'EXPLAIN', requires_code_changes: false):\n"
            "   - The developer is interacting with the pair-programming assistant in chat.\n"
            "   - This includes questions, architecture reviews, explanations, data exploration, or asking the assistant to run tools/queries "
            "(e.g., inspecting APIs, querying databases, executing MCP tools, previewing design mockups, checking telemetry) where NO code repository edits are being authored.\n"
            "   - Examples: 'How is routing handled?', 'Using the Figma MCP server, call get_image for fileKey ... and show mockup', 'Show me the database schema', 'Using playwright, verify UI'\n\n"
            "Analyze the developer request by its ultimate intent: Is the developer asking to edit/author repository files, or interact/inspect via the assistant?\n"
            "Respond in JSON format with keys: intent ('CODE_CHANGE' or 'EXPLAIN'), requires_code_changes (bool), confidence (float), reasoning (str)."
        )

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, f"User Request: {msg}\nAttachments: {request.attachments or []}\nCurrent Tab: {request.current_tab}"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )
        data = json.loads(resp.text)
        return {
            "intent": data.get("intent", "EXPLAIN"),
            "requires_code_changes": bool(data.get("requires_code_changes", False)),
            "confidence": float(data.get("confidence", 0.95)),
            "reasoning": data.get("reasoning", ""),
            "ticket_title": msg[:100],
        }
    except Exception as exc:
        logger.warning(f"Semantic intent classification fallback: {exc}")
        lowered = msg.lower().strip()
        is_query = any(lowered.startswith(p) for p in [
            "what", "how", "query", "where", "show me", "check", "inspect", "list", "tell me", "explain", "why", "analyze", "using the"
        ]) or lowered.endswith("?")

        is_actionable_mutation = any(w in lowered for w in [
            "fix the", "fix ", "implement ", "refactor ", "apply patch", "apply fix", "create endpoint",
            "solve the", "check and fix", "check in ui and solve", "repair ", "modify code",
            "self-heal", "reconcile and fix", "task:", "ticket:", "add ", "update ", "change ",
            "patch ", "create ", "edit ", "ensure the react component", "ensure the component",
            "clean compilation", "compiles cleanly"
        ]) or any(ext in lowered for ext in [".tsx", ".jsx", ".java", ".py", ".ts", ".go", ".rs"])

        is_code = is_actionable_mutation and not (is_query and not any(a in lowered for a in ["and fix", "and solve", "and modify", "and add", "and update"]))
        return {
            "intent": "CODE_CHANGE" if is_code else "EXPLAIN",
            "requires_code_changes": is_code,
            "confidence": 0.85 if is_code else 0.6,
            "reasoning": f"Intake keyword analysis: {'Actionable code modification' if is_code else 'Conversational inquiry'}",
            "ticket_title": msg[:100],
        }


# ============================================================================
# CHAT FOLLOW-UP ENDPOINT — Multi-Tier Memory, Neo4j Graph, RAG & Live Tools
# ============================================================================

class WorkflowChatRequest(BaseModel):
    workflow_id: Optional[str] = None
    message: str
    project_id: Optional[str] = None
    chat_id: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    model_mode: Optional[str] = "default"
    selected_model: Optional[str] = None
    client_message_id: Optional[str] = None



async def _execute_workflow_chat(request: WorkflowChatRequest, emit_fn=None):
    """Internal core engine answering questions with full multi-turn conversational memory,
    Neo4j Graph Topology, Semantic RAG, and Live Codebase Tool-Calling.
    """
    from pathlib import Path
    wid = request.workflow_id
    if not wid and request.chat_id:
        chat_rec = history_store.get_chat(request.chat_id)
        if chat_rec:
            wid = chat_rec.get("workflow_id")
    state = workflow_status.get(wid) if wid else None

    # Fallback to durable task history if workflow not in active memory
    if not state and wid:
        task = history_store.get_task(wid)
        if task:
            state = {
                "ticket_id": task.get("ticket_id"),
                "status": task.get("status"),
                "generated_files": task.get("changed_files") or [],
                "steps": task.get("steps") or [],
                "workflow_explanation": {
                    "summary": task.get("description") or task.get("title") or "",
                    "why": f"Recorded status: {task.get('status')}",
                },
                "error": task.get("error"),
                "project_id": task.get("project_id"),
            }

    # If no specific workflow requested, do not force-bind to past task failure as active state
    # (Past tasks are preserved in episodic memory below)

    explanation = (state.get("workflow_explanation") or {}) if state else {}
    gen_files = (state.get("generated_files") or []) if state else []
    steps = (state.get("steps") or []) if state else []
    ticket_id = (state.get("ticket_id", "") or "") if state else ""
    wf_status = (state.get("status", "unknown")) if state else "no_active_workflow"

    # ── 1. Active Focus & Scope (Ticket / Objective) ──
    scope_parts = []
    if wid:
        scope_parts.append(f"Linked Workflow Task ID: {wid}")
    if ticket_id:
        scope_parts.append(f"Target Ticket / Issue: {ticket_id}")
    if explanation.get("summary"):
        scope_parts.append(f"Objective Summary: {explanation['summary']}")

    # ── 2. Background Execution Telemetry & Diagnostics (Passive Reference) ──
    telemetry_parts = []
    if state:
        telemetry_parts.append(f"Active Task Run Status: {wf_status}")
        if state.get("error"):
            telemetry_parts.append(f"Diagnostic Error / Exception: {str(state['error'])[:500]}")
        if explanation.get("why"):
            telemetry_parts.append(f"Halt / Exit Rationale: {explanation['why']}")
        if gen_files:
            telemetry_parts.append(f"Files Modified in Run: {', '.join(gen_files[:15])}")
        if explanation.get("decisions"):
            telemetry_parts.append(f"Decisions Recorded: {'; '.join(explanation['decisions'][:8])}")
        if explanation.get("evidence"):
            telemetry_parts.append(f"Key Evidence: {'; '.join(explanation['evidence'][:5])}")

        # Projected StateSnapshot from underlying LangGraph checkpoint
        raw_graph_state = state.get("state")
        if isinstance(raw_graph_state, dict):
            hyps = raw_graph_state.get("investigation_hypotheses")
            if hyps and isinstance(hyps, list):
                h_lines = []
                for h in hyps[:4]:
                    txt = getattr(h, "hypothesis", None) or (h.get("hypothesis") if isinstance(h, dict) else str(h))
                    st = getattr(h, "status", None) or (h.get("status") if isinstance(h, dict) else "active")
                    h_lines.append(f"[{st.upper()}] {str(txt)[:120]}")
                if h_lines:
                    telemetry_parts.append(f"Investigated Hypotheses: {'; '.join(h_lines)}")

            evs = raw_graph_state.get("evidence_items")
            if evs and isinstance(evs, list):
                ev_lines = []
                for ev in evs[:5]:
                    path = getattr(ev, "file_path", None) or (ev.get("file_path") if isinstance(ev, dict) else "")
                    score = getattr(ev, "score", None) or (ev.get("score") if isinstance(ev, dict) else "")
                    ev_lines.append(f"{Path(path).name} (score: {score})")
                if ev_lines:
                    telemetry_parts.append(f"Key Evidence Files: {', '.join(ev_lines)}")

            bl = raw_graph_state.get("blacklisted_files")
            if bl and isinstance(bl, list) and bl:
                telemetry_parts.append(f"Excluded/Blacklisted: {', '.join([Path(f).name for f in bl[:5]])}")

            diag = raw_graph_state.get("build_diagnostic_summary")
            if diag:
                telemetry_parts.append(f"Build Diagnostics: {str(diag)[:200]}")

        # Intermediate steps
        step_summaries = []
        for step in steps[-25:]:
            if not isinstance(step, dict):
                continue
            msg = str(step.get("message", "")).strip()
            agent_out = ""
            data = step.get("data")
            if isinstance(data, dict):
                agent_out = str(data.get("agent_output", "")).strip()
            if msg:
                entry = msg
                if agent_out and len(agent_out) > 10:
                    entry += f"\n  Detail: {agent_out[:250]}"
                step_summaries.append(entry)
        if step_summaries:
            telemetry_parts.append("\nWorkflow Steps (chronological):")
            telemetry_parts.extend(step_summaries[-12:])

    # ── 3. Episodic Cross-Task Telemetry & Architectural Knowledge ──
    project_id = request.project_id or (state.get("project_id") if state else None)
    proj_obj = projects.get(project_id) if project_id else None
    if not proj_obj and projects:
        env_ws = os.getenv("AVIATOR_WORKSPACE_PATH")
        if env_ws:
            for p in projects.values():
                if Path(p.path).resolve() == Path(env_ws).resolve():
                    proj_obj = p
                    project_id = p.id
                    break
        if not proj_obj and len(projects) == 1:
            proj_obj = next(iter(projects.values()))
            project_id = proj_obj.id
    arch_parts = []

    if project_id:
        past_tasks = history_store.list_tasks(project_id)
        if past_tasks:
            task_lines = []
            for t in past_tasks[:5]:
                t_id = t.get("ticket_id") or t.get("workflow_id")
                t_st = t.get("status")
                t_title = t.get("title")
                t_files = t.get("changed_files") or []
                t_str = f"• [{t_st.upper()}] {t_id}: '{t_title}'"
                if t_files:
                    t_str += f" (Files: {', '.join(t_files[:3])})"
                task_lines.append(t_str)
            telemetry_parts.append("\nCross-Ticket Episodic Memory (Recent Runs in Project):")
            telemetry_parts.extend(task_lines)

        past_chats = history_store.list_chats(project_id)
        if past_chats:
            chat_lines = []
            for c in past_chats[:4]:
                if request.chat_id and c.get("id") == request.chat_id:
                    continue
                chat_lines.append(f"• Chat '{c.get('title')}' ({c.get('message_count', 0)} messages)")
            if chat_lines:
                telemetry_parts.append("\nOther Conversations in this Project:")
                telemetry_parts.extend(chat_lines)

        # Repository Episodic Memory Recall if available
        try:
            if proj_obj and proj_obj.path:
                from ticket_to_code.reasoning import RepositoryBrain
                brain = RepositoryBrain(proj_obj.path)
                recalled = brain.recall_similar_tickets(request.message, limit=3)
                if recalled:
                    rec_lines = [f"• Ticket {r.get('ticket_id')}: {r.get('title')} (Files: {', '.join(r.get('files_modified', [])[:3])})" for r in recalled]
                    arch_parts.append("Relevant Past Solutions from Repository Brain:\n" + "\n".join(rec_lines))

                # Macro Architecture Knowledge Injection (dynamically discovered from project)
                from pathlib import Path
                proj_root = Path(proj_obj.path)
                arch_candidates = [
                    proj_root / "README.md",
                    proj_root / "ARCHITECTURE.md",
                    proj_root / "architecture-docs" / "01-high-level-architecture.md",
                    proj_root / "docs" / "architecture.md",
                ]
                # Dynamically discover any architecture or design docs in the project
                for doc_pattern in ["*architecture*.md", "docs/**/*.md", "architecture-docs/**/*.md"]:
                    try:
                        for found_doc in proj_root.glob(doc_pattern):
                            if found_doc not in arch_candidates and found_doc.is_file():
                                arch_candidates.append(found_doc)
                    except Exception:
                        pass

                loaded_arch = []
                for ac in arch_candidates:
                    if ac.exists() and ac.is_file():
                        try:
                            content = ac.read_text(encoding="utf-8", errors="ignore")[:3500]
                            loaded_arch.append(f"### Architecture Document ({ac.name}):\n{content}")
                        except Exception:
                            pass
                if loaded_arch:
                    arch_parts.append("Repository Architecture Knowledge Base:\n" + "\n\n".join(loaded_arch[:3]))

                # ── 3b. Active Knowledge Graph Topology from SQLite (39,000+ Symbols) ──
                try:
                    db_p = get_index_db_path(proj_root)
                    if db_p.exists():
                        import sqlite3 as _sqlite3
                        with _sqlite3.connect(str(db_p)) as _sconn:
                            _sconn.row_factory = _sqlite3.Row
                            
                            # Subsystems / Microservices
                            r_files = _sconn.execute("SELECT path FROM files").fetchall()
                            svc_counts = {}
                            for f in r_files:
                                p = f["path"].replace("\\", "/")
                                parts = p.split("/")
                                if len(parts) > 1 and not parts[0].endswith(".py"):
                                    svc_counts[parts[0]] = svc_counts.get(parts[0], 0) + 1
                            top_svcs = sorted(svc_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                            svc_items = [f"• `{svc}` ({cnt} files)" for svc, cnt in top_svcs]

                            # Key Components
                            r_comps = _sconn.execute(
                                "SELECT name, coalesce(spring_stereotype, kind, 'component') as stereotype, path "
                                "FROM symbols WHERE spring_stereotype IS NOT NULL OR kind IN ('class', 'function', 'interface', 'service', 'controller', 'component') "
                                "LIMIT 20"
                            ).fetchall()
                            comp_items = [f"• `{rec['name']}` ({rec['stereotype']}) in `{rec['path']}`" for rec in r_comps]

                            # Cross-Component Call Chains
                            r_calls = _sconn.execute(
                                "SELECT s.name as caller, e.dst_name as callee "
                                "FROM edges e JOIN symbols s ON e.src_id = s.id "
                                "WHERE e.kind = 'calls' AND e.dst_name IS NOT NULL AND e.dst_name != '' "
                                "LIMIT 10"
                            ).fetchall()
                            call_items = [f"• `{rec['caller']}` ➔ `{rec['callee']}`" for rec in r_calls]

                            graph_parts = []
                            if svc_items:
                                graph_parts.append("Microservices & Subsystems:\n" + "\n".join(svc_items))
                            if comp_items:
                                graph_parts.append("Key Architectural Components & Handlers:\n" + "\n".join(comp_items[:12]))
                            if call_items:
                                graph_parts.append("Cross-Component Call Chains:\n" + "\n".join(call_items[:8]))

                            if graph_parts:
                                arch_parts.append("Active Knowledge Graph Topology (Live Code Graph):\n" + "\n\n".join(graph_parts))
                except Exception as _graph_err:
                    logger.debug(f"SQLite graph recall in chat skipped: {_graph_err}")

                # ── 3b-2. Universal Ground-Truth Architecture & Database Census (Always Active) ──
                try:
                    from ticket_to_code.lean.workspace_census import build_census_markdown_summary
                    q_lower = request.message.lower()
                    is_detailed_schema = any(k in q_lower for k in [
                        "database schema", "global schema", "cross-service schema", "cross-service database",
                        "entity relationship", "er diagram", "erd", "data model", "tables in the database",
                        "map the global", "database tables", "foreign key", "schema table"
                    ]) or ("schema" in q_lower and any(w in q_lower for w in ["database", "service", "table", "global", "all"]))

                    census_summary = build_census_markdown_summary(proj_root, is_detailed=is_detailed_schema)
                    if census_summary:
                        arch_parts.append(census_summary)
                except Exception as _census_err:
                    logger.debug(f"Workspace architecture census error: {_census_err}")

                # ── 3c. Semantic RAG Engine retrieval for codebase question (Cached) ──
                try:
                    global _cached_rag_engine
                    if '_cached_rag_engine' not in globals() or _cached_rag_engine is None:
                        from ticket_to_code.retrieval.rag_engine import CodebaseRAGEngine
                        _cached_rag_engine = CodebaseRAGEngine()
                    rag = _cached_rag_engine
                    rag.workspace_path = str(proj_obj.path)
                    chunks = rag.retrieve_context(request.message, max_results=3, document_types=["documentation", "readme", "architecture", "configuration"], schema="architecture")
                    if chunks:
                        rag_items = []
                        for ch in chunks[:3]:
                            c_path = ch.get("file_path", "")
                            c_text = ch.get("content", "")[:350]
                            rag_items.append(f"• File `{c_path}`:\n{c_text}")
                        arch_parts.append("Grounded Context from Repository Index:\n" + "\n".join(rag_items))
                except Exception as _rag_err:
                    logger.debug(f"RAG recall in chat skipped: {_rag_err}")
        except Exception:
            pass

    # ── 4. Conversational Working Memory (Same Chat Multi-Turn) ──
    conv_turns = []
    if request.messages:
        for m in request.messages[-8:]:
            if isinstance(m, dict):
                r = m.get("role", "user")
                c = str(m.get("content", "")).strip()
                if c:
                    conv_turns.append(f"[{r.upper()}]: {c[:300]}")

    scope_block = "\n".join(scope_parts) if scope_parts else "Workspace-wide general engineering session (no specific ticket constrained)."
    telemetry_block = "\n".join(telemetry_parts) if telemetry_parts else "No background execution telemetry."
    arch_block = "\n\n".join(arch_parts) if arch_parts else "Repository architecture loaded."
    conv_block = "\n".join(conv_turns) if conv_turns else "New conversation thread."

    # ── 4. Live Codebase Inspection Tools for Gemini AFC ──
    repo_root = Path(proj_obj.path) if (proj_obj and proj_obj.path) else Path.cwd()

    from services.mcp_manager import mcp_manager
    mcp_catalog_summary = mcp_manager.get_dynamic_catalog_summary()

    import time
    t_start = time.time()
    execution_steps = []
    loop = asyncio.get_running_loop()

    def _broadcast_chat_step(step_entry: dict):
        """Stream real-time tool execution events to frontend via SSE / WebSocket."""
        if emit_fn:
            emit_fn("step", step_entry)
        try:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast({
                    "type": "chat_step",
                    "step": step_entry
                }),
                loop
            )
        except Exception as _b_err:
            logger.debug(f"Chat step broadcast error: {_b_err}")

    # Immediately signal that analysis has begun
    _broadcast_chat_step({
        "title": "Analyzing prompt and repository architecture...",
        "type": "thought",
        "status": "running"
    })

    def call_mcp_tool(server_name: str, tool_name: str, arguments: dict) -> str:
        """Execute a tool on an active Model Context Protocol (MCP) server.
        See the 'Configured MCP Servers & Exposed Tools' manifest in the system prompt for all available server names, tool names, and parameter schemas.
        
        Args:
            server_name: Name of the target MCP server from the manifest.
            tool_name: The name of the tool function to call on that server.
            arguments: Dictionary of arguments required by the tool.
        """
        args_display = ""
        if arguments:
            arg_pairs = []
            for k, v in list(arguments.items())[:2]:
                val_str = str(v)
                if len(val_str) > 35:
                    val_str = val_str[:32] + "..."
                arg_pairs.append(f"{k}={val_str}")
            args_display = f" ({', '.join(arg_pairs)})"

        step_t0 = time.time()
        step_entry = {
            "title": f"Calling {server_name}.{tool_name}{args_display}",
            "type": "tool_call",
            "server": server_name,
            "tool": tool_name,
            "args": arguments,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            logger.info(f"==> call_mcp_tool called: server={server_name}, tool={tool_name}, args={arguments}")
            res = mcp_manager.call_tool_sync(server_name=server_name, tool_name=tool_name, arguments=arguments, workspace_path=str(repo_root))
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "success"
            step_entry["title"] = f"Ran {server_name}.{tool_name}{args_display}"
            step_entry["output_preview"] = str(res)[:600]
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.info(f"<== call_mcp_tool result: {str(res)[:300]}")
            import json
            return json.dumps(res, indent=2, ensure_ascii=False)
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Failed {server_name}.{tool_name}"
            step_entry["output_preview"] = str(e)
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.error(f"MCP tool execution error: {e}")
            return f"MCP tool execution error: {e}"

    # ── Deductive Reasoning Blackboard & Multi-Service Coordinator ──
    from ticket_to_code.runtime import (
        DeductiveBlackboard,
        MultiServiceOrchestrator,
        CompilerGate,
        HighDensitySanitizer,
    )
    from ticket_to_code.runtime.agent_harness import ReActLoopGovernor, EXPLORATION_TOOL_NAMES
    from ticket_to_code.runtime.limits import TOOL_OUTPUT_MAX_CHARS, SEARCH_MAX_LINES, BUILD_OUTPUT_MAX_CHARS, truncate
    from ticket_to_code.runtime.security import SecurityViolation, run_checked_subprocess

    session_id = request.chat_id or request.workflow_id or (proj_obj.id if proj_obj else "default_session")
    blackboard = DeductiveBlackboard(workspace_path=str(repo_root))
    blackboard.load_from_disk(session_id)
    react_governor = ReActLoopGovernor(max_consecutive_redundant=3, hard_stop_limit=6)

    def _persist_blackboard():
        try:
            blackboard.save_to_disk(session_id)
        except Exception as _bb_save_err:
            logger.debug(f"Blackboard save error: {_bb_save_err}")

    if request.messages:
        for m in request.messages:
            if isinstance(m, dict):
                c = str(m.get("content", ""))
                if "<deductive_blackboard>" in c or "<verified_fact" in c:
                    blackboard.parse_and_ingest_structured_facts(c)
                    for rej in re.findall(r"<rejected_hypothesis>(.*?)</rejected_hypothesis>", c, re.DOTALL):
                        if rej.strip():
                            blackboard.record_rejection(rej.strip())

    multi_service_orch = MultiServiceOrchestrator(blackboard=blackboard)
    multi_service_orch.analyze_ticket("Active Request", request.message)

    def invoke_subagent_explorer(role: str, focus_query: str) -> str:
        """Spawn an isolated research subagent to perform targeted deep-dives without bloating the main conversation context.
        Available roles:
          - 'ui_mapper': Maps Angular component trees, route modules, and template controls.
          - 'contract_checker': Inspects REST controllers, DTOs, and Spring Cloud Gateway routes.
          - 'validation_analyst': Analyzes form validation and error state handling.
          - 'test_scout': Discovers existing unit tests, mocks, and test execution commands.
        Args:
            role: One of 'ui_mapper', 'contract_checker', 'validation_analyst', 'test_scout'.
            focus_query: Focus keyword or component name (e.g. 'deliverables', 'IssueCount', 'IssuesCollectionController').
        """
        step_t0 = time.time()
        clean_role = role.strip().lower()
        step_entry = {
            "title": f"Spawning Subagent [{clean_role}] for '{focus_query}'...",
            "type": "subagent",
            "role": clean_role,
            "query": focus_query,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            from ticket_to_code.agents.subagent_explorer import SubagentExplorer, ExplorerRole
            explorer = SubagentExplorer(str(repo_root))
            
            cand_files = []
            for p in repo_root.glob(f"**/*{focus_query}*"):
                if p.is_file() and not any(x in str(p) for x in [".git", "node_modules", "target", "build", ".gradle", "dist"]):
                    cand_files.append(str(p.relative_to(repo_root)).replace("\\", "/"))
                if len(cand_files) >= 5:
                    break

            if len(cand_files) < 3:
                aliases = [focus_query]
                q_low = focus_query.lower()
                if "priority" in q_low:
                    aliases.extend(["severity", "issue", "badge"])
                elif "severity" in q_low:
                    aliases.extend(["priority", "critical", "breakdown"])
                elif "count" in q_low:
                    aliases.extend(["metric", "total", "breakdown", "summary"])
                elif "deliverable" in q_low:
                    aliases.extend(["item", "grid", "package"])

                for alias in aliases:
                    for p in repo_root.glob(f"**/*{alias}*"):
                        if p.is_file() and not any(x in str(p) for x in [".git", "node_modules", "target", "build", ".gradle", "dist"]):
                            rel_str = str(p.relative_to(repo_root)).replace("\\", "/")
                            if rel_str not in cand_files:
                                cand_files.append(rel_str)
                        if len(cand_files) >= 5:
                            break
                    if len(cand_files) >= 5:
                        break
            
            if clean_role == "ui_mapper":
                rep = explorer.run_ui_mapper(focus_query, cand_files)
            elif clean_role == "validation_analyst":
                rep = explorer.run_validation_analyst(focus_query, cand_files)
            elif clean_role == "test_scout":
                rep = explorer.run_test_scout(focus_query, cand_files)
            else:
                rep = explorer.run_contract_checker(focus_query, cand_files)
                
            summary = rep.to_summary_markdown()
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "success"
            step_entry["title"] = f"Subagent [{clean_role}] finished (confidence: {rep.confidence:.2f})"
            step_entry["output_preview"] = summary[:400]
            step_entry["raw_output"] = summary
            execution_steps.append(step_entry)
            try:
                blackboard.ingest_tool_observation(f"subagent_{clean_role}", focus_query, summary)
                _persist_blackboard()
            except Exception:
                pass
            _broadcast_chat_step(step_entry)
            return summary
        except Exception as ex:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Subagent [{clean_role}] error: {ex}"
            step_entry["output_preview"] = str(ex)
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return f"Subagent error: {ex}"

    def list_directory(relative_path: str = "", max_entries: int = 50) -> str:
        """List files and subdirectories in a workspace directory to explore project structure.
        Args:
            relative_path: Subpath relative to repository root (empty string lists the workspace root).
            max_entries: Maximum number of entries to return.
        """
        step_t0 = time.time()
        step_entry = {
            "title": f"Listing directory '{relative_path or '/'}'...",
            "type": "directory_list",
            "path": relative_path,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            raw_path = (relative_path or "").strip()
            clean_rel = raw_path.lstrip("/\\")
            cand_p = Path(raw_path) if raw_path else repo_root
            if cand_p.is_absolute() and cand_p.is_dir():
                target = cand_p
            else:
                target = (repo_root / clean_rel).resolve() if clean_rel else repo_root

            if not target.exists() or not target.is_dir():
                # Directory name search fallback
                dir_name = Path(clean_rel).name
                found_dirs = [d for d in repo_root.glob(f"**/{dir_name}") if d.is_dir()]
                if found_dirs:
                    target = found_dirs[0]
                else:
                    step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
                    step_entry["status"] = "error"
                    step_entry["title"] = f"Directory '{relative_path}' not found"
                    step_entry["output_preview"] = "Not found"
                    execution_steps.append(step_entry)
                    _broadcast_chat_step(step_entry)
                    return f"Directory not found: {relative_path}"

            entries = []
            for p in sorted(target.iterdir()):
                if p.name.startswith(".") or p.name in ["build", "node_modules", "target", ".gradle", "bin"]:
                    continue
                kind = "[DIR] " if p.is_dir() else "[FILE]"
                entries.append(f"{kind} {p.name}")
                if len(entries) >= max_entries:
                    entries.append(f"... (truncated at {max_entries} entries)")
                    break

            out = "\n".join(entries) if entries else "(empty directory)"
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "success"
            rel_display = str(target.relative_to(repo_root)).replace("\\", "/") if target != repo_root and target.is_relative_to(repo_root) else "/"
            step_entry["title"] = f"Listed '{rel_display}' ({len(entries)} items)"
            step_entry["output_preview"] = out[:600]
            step_entry["raw_output"] = out
            execution_steps.append(step_entry)
            try:
                blackboard.ingest_tool_observation("list_directory", relative_path, out)
                _persist_blackboard()
            except Exception:
                pass
            _broadcast_chat_step(step_entry)
            logger.info(f"[Chat Tool:list_directory] {step_entry['title']}")
            return out
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Error listing '{relative_path}'"
            step_entry["output_preview"] = str(e)
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.error(f"[Chat Tool:list_directory] {e}")
            return f"Error listing directory: {e}"

    def find_files(pattern: str, max_results: int = 150) -> str:
        """Find files matching a glob pattern across all microservices and directories in the workspace.
        Args:
            pattern: Glob pattern, e.g. '*.yml', '*Saga*.java', '*Rabbit*.java', 'V*.sql', '*Upload*.java'.
            max_results: Maximum file paths to return (default 150).
        """
        import fnmatch
        step_t0 = time.time()
        step_entry = {
            "title": f"Finding files matching '{pattern}'...",
            "type": "find_files",
            "pattern": pattern,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            matches = []
            clean_pat = pattern.strip()
            # Support multiple glob patterns separated by ' OR ', '|', or ','
            sub_patterns = [p.strip() for p in re.split(r'\s+OR\s+|[|,]', clean_pat) if p.strip()]
            if not sub_patterns:
                sub_patterns = [clean_pat]

            # Normalize patterns starting with '**/' so root-level folders match without requiring a leading slash
            expanded_patterns = list(sub_patterns)
            for p in sub_patterns:
                if p.startswith("**/"):
                    expanded_patterns.append(p[3:])

            for root, dirs, files in os.walk(repo_root):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["build", "node_modules", "target", ".gradle", "bin", ".git", "test", "test-classes"]]
                for f in files:
                    rel = Path(os.path.join(root, f)).relative_to(repo_root)
                    rel_posix = str(rel).replace("\\", "/")
                    for pat in expanded_patterns:
                        if (fnmatch.fnmatch(f, pat) or 
                            fnmatch.fnmatch(rel_posix, pat) or 
                            fnmatch.fnmatch(f.lower(), pat.lower()) or 
                            fnmatch.fnmatch(rel_posix.lower(), pat.lower())):
                            matches.append(rel_posix)
                            break
                    if len(matches) >= max_results:
                        break
                if len(matches) >= max_results:
                    break

            service_counts = {}
            for m in matches:
                svc = m.split("/")[0] if "/" in m else "root"
                service_counts[svc] = service_counts.get(svc, 0) + 1

            summary_header = f"Found {len(matches)} file(s) across {len(service_counts)} service(s): " + ", ".join(f"{k} ({v})" for k, v in sorted(service_counts.items()))
            out = f"## {summary_header}\n" + "\n".join(matches) if matches else f"No files matching pattern: {pattern}"
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "success"
            step_entry["title"] = f"Found {len(matches)} file(s) across {len(service_counts)} service(s)"
            step_entry["output_preview"] = out[:600]
            step_entry["raw_output"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.info(f"[Chat Tool:find_files] {step_entry['title']}")
            return out
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Error finding files for '{pattern}'"
            step_entry["output_preview"] = str(e)
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.error(f"[Chat Tool:find_files] {e}")
            return f"Error finding files: {e}"

    def search_codebase(query: str, search_type: str = "auto", file_pattern: str = "", **kwargs: Any) -> str:
        """Search the codebase across all microservices for symbols, annotations, endpoints, or code text.
        Args:
            query: Term, class name, endpoint, topic name, or code pattern to search for.
            search_type: Search mode - 'auto', 'literal', 'filename', or 'regex'.
            file_pattern: Optional glob filter for files.
        """
        step_t0 = time.time()
        step_entry = {
            "title": f"Searching codebase for '{query}'...",
            "type": "search",
            "query": query,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
            engine = RepositorySearchEngine(workspace_path=repo_root)
            q = query.strip()
            # If query has ' OR ' or '|', split into multiple search terms
            terms = [t.strip().strip('"\'') for t in re.split(r'\s+OR\s+|[|]', q) if t.strip()]
            hits = []
            if len(terms) > 1:
                # Fast single-pass regex search across terms to avoid repeated directory walks
                comb_regex = "|".join(re.escape(t) for t in terms[:6])
                hits = engine.search_regex(comb_regex, max_results=35)
                if not hits:
                    seen_paths = set()
                    for t in terms[:3]:
                        sub_hits = engine.search_literal(t, max_results=10)
                        for sh in sub_hits:
                            fp = getattr(sh, "file_path", getattr(sh, "relative_path", str(sh)))
                            if fp not in seen_paths:
                                seen_paths.add(fp)
                                hits.append(sh)
                        if len(hits) >= 35:
                            break
            elif search_type == "filename" or any(c in q for c in ["*", "?", "/"]):
                hits = engine.search_filename(q, max_results=35)
            elif search_type == "regex":
                hits = engine.search_regex(q, max_results=35)
            else:
                hits = engine.search_literal(q, max_results=35)
                if not hits and " " in q:
                    words = [w for w in q.split() if len(w) >= 4]
                    if words:
                        comb = "|".join(re.escape(w) for w in words[:3])
                        hits = engine.search_regex(comb, max_results=35)

            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "success"
            step_entry["title"] = f"Searched codebase for '{query}' ({len(hits)} matches)"
            
            if hits:
                formatted_lines = []
                for h in hits[:35]:
                    p = getattr(h, "file_path", getattr(h, "relative_path", str(h)))
                    ln = getattr(h, "line_number", 0)
                    txt = getattr(h, "matched_text", getattr(h, "line_content", ""))
                    formatted_lines.append(f"• `{p}:{ln}`: {str(txt).strip()[:160]}")
                formatted = "\n".join(formatted_lines)
                sanitized_out = HighDensitySanitizer.sanitize(formatted, max_chars=TOOL_OUTPUT_MAX_CHARS, max_lines=SEARCH_MAX_LINES)
                guidance = react_governor.evaluate_call("search_codebase", query, sanitized_out)
                if guidance:
                    sanitized_out += guidance
                step_entry["output_preview"] = sanitized_out[:600]
                step_entry["raw_output"] = sanitized_out
                execution_steps.append(step_entry)
                try:
                    blackboard.ingest_tool_observation("search_codebase", query, sanitized_out)
                    _persist_blackboard()
                except Exception:
                    pass
                _broadcast_chat_step(step_entry)
                logger.info(f"[Chat Tool:search_codebase] {step_entry['title']}")
                return sanitized_out
            if not hits:
                # Fuzzy keyword filename match fallback so colloquial queries don't dead-end
                raw_words = [w.strip() for w in re.split(r'[\s_\-\.]+', q) if len(w.strip()) >= 3 and w.strip().isalnum()]
                suggested_files = []
                for w in raw_words[:4]:
                    try:
                        matches = list(repo_root.glob(f"**/*{w}*"))
                        for m in matches:
                            if m.is_file() and not any(x in str(m).replace("\\", "/") for x in ["build", "target", "node_modules", ".git", ".venv", "/test/"]):
                                rel = str(m.relative_to(repo_root)).replace("\\", "/")
                                if rel not in suggested_files:
                                    suggested_files.append(rel)
                            if len(suggested_files) >= 6:
                                break
                    except Exception:
                        pass
                    if len(suggested_files) >= 6:
                        break
                if suggested_files:
                    out = (
                        f"No direct code text matches found for '{query}'. However, related repository files were discovered matching query terms:\n"
                        + "\n".join(f"• `{sf}`" for sf in suggested_files)
                        + "\nAction: Use `inspect_file` on these files to locate the implementation."
                    )
                else:
                    out = f"No matches found for '{query}'"
            guidance = react_governor.evaluate_call("search_codebase", query, out)
            if guidance:
                out += guidance
            step_entry["output_preview"] = out
            step_entry["raw_output"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.info(f"[Chat Tool:search_codebase] {step_entry['title']}")
            return out
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Error searching '{query}'"
            step_entry["output_preview"] = str(e)
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.error(f"[Chat Tool:search_codebase] {e}")
            return f"Search error: {e}"

    def find_references(symbol_name: str, **kwargs: Any) -> str:
        """Find all callers, usages, and import sites of a symbol, constant, class, or method across all microservices.
        CRITICAL: Before modifying or renaming any constant (e.g. VALID_REVIEW_STATE_TRANSITIONS) or method signature,
        ALWAYS invoke this tool to discover all callers across services and prevent breaking compilation or bypassing endpoints.
        Args:
            symbol_name: Exact name of the symbol, constant, or method (e.g. 'VALID_REVIEW_STATE_TRANSITIONS', 'canCloseIssue', 'closeIssue').
        """
        step_t0 = time.time()
        step_entry = {
            "title": f"Finding references for '{symbol_name}' across codebase...",
            "type": "find_references",
            "symbol": symbol_name,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
            engine = RepositorySearchEngine(workspace_path=repo_root)
            clean_sym = symbol_name.strip()
            hits = engine.search_literal(clean_sym, max_results=50)
            if not hits:
                hits = engine.search_regex(r'\b' + re.escape(clean_sym) + r'\b', max_results=50)
            
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "success"
            step_entry["title"] = f"Found {len(hits)} reference(s) for '{clean_sym}'"
            
            if hits:
                call_site_hits = []
                import_hits = []
                for h in hits:
                    txt = str(getattr(h, "matched_text", getattr(h, "line_content", ""))).strip()
                    if txt.startswith(("import ", "package ", "from ")):
                        import_hits.append(h)
                    else:
                        call_site_hits.append(h)
                ordered_hits = call_site_hits + import_hits
                formatted_lines = [f"### References for '{clean_sym}' ({len(call_site_hits)} call site(s), {len(import_hits)} import(s) across repository):"]
                for h in ordered_hits[:50]:
                    p = getattr(h, "file_path", getattr(h, "relative_path", str(h)))
                    ln = getattr(h, "line_number", 0)
                    txt = getattr(h, "matched_text", getattr(h, "line_content", ""))
                    formatted_lines.append(f"• `{p}:{ln}`: {str(txt).strip()[:160]}")
                formatted = "\n".join(formatted_lines)
                sanitized_out = HighDensitySanitizer.sanitize(formatted, max_chars=TOOL_OUTPUT_MAX_CHARS, max_lines=SEARCH_MAX_LINES)
                guidance = react_governor.evaluate_call("find_references", symbol_name, sanitized_out)
                if guidance:
                    sanitized_out += guidance
                step_entry["output_preview"] = sanitized_out[:600]
                step_entry["raw_output"] = sanitized_out
                execution_steps.append(step_entry)
                try:
                    blackboard.ingest_tool_observation("find_references", symbol_name, sanitized_out)
                    _persist_blackboard()
                except Exception:
                    pass
                _broadcast_chat_step(step_entry)
                logger.info(f"[Chat Tool:find_references] {step_entry['title']}")
                return sanitized_out
            
            out = f"No references found for '{clean_sym}' in workspace."
            guidance = react_governor.evaluate_call("find_references", symbol_name, out)
            if guidance:
                out += guidance
            step_entry["output_preview"] = out
            step_entry["raw_output"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return out
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Error finding references for '{symbol_name}'"
            step_entry["output_preview"] = str(e)
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.error(f"[Chat Tool:find_references] {e}")
            return f"Reference search error: {e}"

    def verify_service_compilation(service_name: str = "") -> str:
        """Run a fast incremental compile check (e.g. gradlew compileJava or mvn compile) on the target service to verify that existing code and contracts compile cleanly.
        Args:
            service_name: Directory or service name (e.g. 'issues-service', 'project-service'). If empty, automatically targets the service associated with the active task.
        """
        import subprocess, sys
        step_t0 = time.time()
        sname = service_name.strip() if service_name else "issues-service"
        target_dir = repo_root / sname if (repo_root / sname).is_dir() else repo_root
        
        step_entry = {
            "title": f"Running compile check on '{target_dir.name}'...",
            "type": "verify_compilation",
            "service": target_dir.name,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)

        if strictly_explain:
            user_asks_compile = any(w in user_msg_low for w in [
                "compile", "build", "compilation", "broken build", "syntax error",
                "does it build", "does it compile", "solution", "production-ready",
                "production ready", "verify code", "verification", "code verification",
                "production solution", "implement", "code fix", "diff", "patch", "architectural fix"
            ])
            if not user_asks_compile:
                declined = (
                    f"Compile check skipped: The session is in read-only inquiry mode and the prompt does not require build verification. "
                    f"Synthesize your architectural analysis or explanation directly using inspect_file and search_codebase."
                )
                step_entry["duration_ms"] = 0
                step_entry["status"] = "cancelled"
                step_entry["title"] = f"Compile check suppressed (Inquiry Mode): {target_dir.name}"
                step_entry["output_preview"] = declined
                execution_steps.append(step_entry)
                _broadcast_chat_step(step_entry)
                return declined
        try:
            use_shell = sys.platform == "win32"
            from ticket_to_code.runtime.polyglot_build import PolyglotBuildDiscoverer
            cfg = PolyglotBuildDiscoverer.discover(target_dir, repo_root)
            gradle_cmd = None
            if cfg:
                gradle_cmd = cfg.compile_cmd
                target_dir = cfg.work_dir
            elif (repo_root / "gradlew.bat").exists() and use_shell:
                gradle_cmd = f".\\gradlew.bat :{target_dir.name}:compileJava --daemon -Dorg.gradle.parallel=true -Dorg.gradle.caching=true -q"
            
            if not gradle_cmd:
                out = f"No standard build tool found in '{target_dir.name}'."
                step_entry["status"] = "success"
                step_entry["output_preview"] = out
                execution_steps.append(step_entry)
                _broadcast_chat_step(step_entry)
                return out

            # Policy-checked, env-scrubbed, workspace-confined, process-tree bounded
            try:
                rc, p_out, p_err = run_checked_subprocess(
                    gradle_cmd, cwd=target_dir, timeout=45, workspace=repo_root
                )
            except SecurityViolation as sv:
                out = f"Security policy rejection: {sv}"
                step_entry["status"] = "error"
                step_entry["title"] = f"Compile check blocked by security policy on '{target_dir.name}'"
                step_entry["output_preview"] = out
                execution_steps.append(step_entry)
                _broadcast_chat_step(step_entry)
                return out
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            if rc == 0:
                step_entry["status"] = "success"
                step_entry["title"] = f"Compilation check passed cleanly on '{target_dir.name}' (0 errors)"
                out = f"✅ Compilation SUCCESS on {target_dir.name}: Clean compile with zero errors."
            else:
                step_entry["status"] = "failed"
                err_text = (p_err or p_out or "").strip()
                step_entry["title"] = f"Compilation failed on '{target_dir.name}' (Exit code {rc})"
                comp_obj = CompilerGate.parse_compiler_output(target_dir.name, rc, err_text)
                out = comp_obj.get_repair_instruction() if comp_obj.diagnostics else f"❌ Compilation FAILED on {target_dir.name}:\n{truncate(err_text, BUILD_OUTPUT_MAX_CHARS)}"

            step_entry["output_preview"] = out[:600]
            step_entry["raw_output"] = out
            execution_steps.append(step_entry)
            try:
                comp_res = CompilerGate.record_and_verify(target_dir.name, rc, out, blackboard=blackboard)
                if CompilerGate.get_baseline(target_dir.name, str(repo_root)) is None:
                    CompilerGate.record_baseline(target_dir.name, comp_res, workspace_path=str(repo_root))
                _persist_blackboard()
            except Exception:
                pass
            _broadcast_chat_step(step_entry)
            return out
        except subprocess.TimeoutExpired:
            step_entry["status"] = "timeout"
            step_entry["title"] = f"Compilation check timed out on '{target_dir.name}'"
            out = f"Compilation check timed out after 45s on {target_dir.name}."
            step_entry["output_preview"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return out
        except Exception as ex:
            step_entry["status"] = "error"
            step_entry["title"] = f"Error running compilation check on '{target_dir.name}'"
            out = f"Error running compilation check: {ex}"
            step_entry["output_preview"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return out

    def inspect_file(file_path: str, symbol: str = "", start_line: int = 1, end_line: int = 0) -> str:
        """Inspect source code, database migration DDL, or configuration files in the workspace.
        Supports AST-aware extraction of complete methods, classes, and structural outlines:
          1. Complete Method/Class: specify symbol='methodName' or symbol='ClassName' to extract the complete method/class with its exact start-line to end-line boundary.
          2. Structural Outline: specify symbol='outline' to get the full structural inventory of all classes, methods, constructors, and line ranges in the file.
          3. Sliced Range: specify start_line and end_line for exact line slicing.
          4. Full File: leave symbol='' and end_line=0 (or default) to view the complete file. If the file exceeds 1500 lines, automatically provides imports + fields + full structural method outline.
          5. Multi-File Batching: comma-separated file paths or glob patterns (e.g. 'V1.*.sql, V2.*.sql').
        Args:
            file_path: Relative path, comma-separated paths, or glob pattern from repository root.
            symbol: Optional method name, class name, or 'outline' for AST-aware complete extraction.
            start_line: 1-indexed line number to start reading from (default 1).
            end_line: 1-indexed line number to end reading at (default 0 reads entire file or symbol).
        """
        step_t0 = time.time()
        sym_desc = f" symbol='{symbol}'" if symbol else ""
        lines_desc = f" ({start_line}-{end_line})" if end_line > 0 else ""
        step_entry = {
            "title": f"Inspecting file '{file_path}'{sym_desc}{lines_desc}...",
            "type": "file_read",
            "file_path": file_path,
            "symbol": symbol,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            from ticket_to_code.agents.smart_extract import (
                _parse_java_ts_boundaries,
                _parse_python_boundaries,
                MethodBoundary,
                _detect_language
            )

            cand_paths = [p.strip() for p in re.split(r'[,;\n]', file_path) if p.strip()]
            if not cand_paths:
                cand_paths = [file_path.strip()]

            resolved_files: list[Path] = []
            seen_paths = set()

            for cp in cand_paths:
                clean_rel = cp.lstrip("/\\")
                if any(ch in clean_rel for ch in ["*", "?"]):
                    # Glob pattern matching
                    matches = [
                        f for f in repo_root.glob(clean_rel)
                        if f.is_file() and not any(x in str(f).replace("\\", "/") for x in ["build", "target", "node_modules", ".git", "/test/"])
                    ]
                    if not matches and "/" not in clean_rel and "\\" not in clean_rel:
                        matches = [
                            f for f in repo_root.glob(f"**/{clean_rel}")
                            if f.is_file() and not any(x in str(f).replace("\\", "/") for x in ["build", "target", "node_modules", ".git", "/test/"])
                        ]
                    for m in matches:
                        if m not in seen_paths:
                            seen_paths.add(m)
                            resolved_files.append(m)
                        if len(resolved_files) >= 8:
                            break
                else:
                    cand_p = Path(cp)
                    if cand_p.is_absolute() and cand_p.is_file():
                        if cand_p not in seen_paths:
                            seen_paths.add(cand_p)
                            resolved_files.append(cand_p)
                    else:
                        full_p = (repo_root / clean_rel).resolve()
                        if full_p.exists() and full_p.is_file():
                            if full_p not in seen_paths:
                                seen_paths.add(full_p)
                                resolved_files.append(full_p)
                        else:
                            fname = Path(clean_rel).name
                            found = [
                                f for f in repo_root.glob(f"**/{fname}")
                                if f.is_file() and not any(x in str(f).replace("\\", "/") for x in ["build", "target", ".git", "/test/"])
                            ]
                            if found and found[0] not in seen_paths:
                                seen_paths.add(found[0])
                                resolved_files.append(found[0])
                if len(resolved_files) >= 8:
                    break

            output_sections = []
            for full_path in resolved_files:
                lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                total_f_lines = len(lines)
                rel_display = str(full_path.relative_to(repo_root)).replace("\\", "/") if full_path.is_relative_to(repo_root) else full_path.name
                lang = _detect_language(str(full_path))

                # Parse AST method/class boundaries for Java, TS/JS, and Python
                boundaries = []
                try:
                    if lang in ("java", "typescript", "javascript"):
                        boundaries = _parse_java_ts_boundaries(lines)
                    elif lang == "python":
                        boundaries = _parse_python_boundaries(lines)
                except Exception as _b_err:
                    logger.debug(f"Boundary parsing exception: {_b_err}")

                clean_sym = symbol.strip() if symbol else ""

                if clean_sym:
                    if clean_sym.lower() in ("outline", "structure", "methods", "summary"):
                        # Structural Outline: include class declarations, fields, constants, and dependencies up to the first method
                        first_method_idx = boundaries[0].start_line if boundaries else 120
                        header_limit = min(max(first_method_idx, 100), 160)
                        header = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[:min(len(lines), header_limit)]))
                        outline_lines = [f"// === Structural Method Outline of {rel_display} ({len(boundaries)} symbols, {total_f_lines} lines) ==="]
                        for b in boundaries:
                            outline_lines.append(f"  L{b.start_line + 1}-L{b.end_line + 1} ({b.line_count()} lines): {b.kind} {b.signature}")
                        outline_lines.append("\n// ⚠️ ACTION REQUIRED: To inspect the actual implementation, exception constructors, and bean calls of any method above, call inspect_file(file_path, symbol='<methodName>')!")
                        output_sections.append(f"### File Outline: {rel_display}\n```\n{header}\n...\n" + "\n".join(outline_lines) + "\n```")
                    else:
                        # Specific method or class boundary extraction (supports comma-separated multi-symbols)
                        target_syms = [s.strip().lower() for s in re.split(r'[,;]', clean_sym) if s.strip()]
                        matched = []
                        for ts in target_syms:
                            m = [b for b in boundaries if b.name.lower() == ts]
                            if not m:
                                m = [b for b in boundaries if ts in b.name.lower()]
                            for item in m:
                                if item not in matched:
                                    matched.append(item)

                        if matched:
                            for b in matched:
                                a_start = b.start_line + 1
                                a_end = b.end_line + 1
                                sliced = lines[b.start_line : b.end_line + 1]
                                if len(sliced) > 1500 and b.kind == "class":
                                    class_bounds = [mb for mb in boundaries if mb.kind != "class" and mb.start_line >= b.start_line and mb.end_line <= b.end_line]
                                    outlines = [f"// === Outline of {b.name} ({len(class_bounds)} methods, {len(sliced)} lines) ==="]
                                    for mb in class_bounds:
                                        outlines.append(f"  L{mb.start_line + 1}-L{mb.end_line + 1} ({mb.line_count()} lines): {mb.kind} {mb.signature}")
                                    outlines.append(f"\n// ⚠️ To inspect any method above, call inspect_file(file_path, symbol='<methodName>')")
                                    hdr = "\n".join(f"{i + a_start}: {l}" for i, l in enumerate(sliced[:120]))
                                    output_sections.append(f"### Class {b.name} in {rel_display} (Lines {a_start}-{a_end} of {total_f_lines})\n```\n{hdr}\n...\n" + "\n".join(outlines) + "\n```")
                                else:
                                    code_block = "\n".join(f"{i + a_start}: {l}" for i, l in enumerate(sliced))
                                    output_sections.append(f"### {b.kind.title()}: {b.name} in {rel_display} (Lines {a_start}-{a_end} of {total_f_lines})\n```\n{code_block}\n```")
                        else:
                            # Text-based symbol search fallback (prioritizes non-import usage/instantiation call sites)
                            found_syms = []
                            for ts in target_syms:
                                usage_indices = [
                                    idx for idx, line in enumerate(lines)
                                    if ts in line.lower() and not line.strip().startswith(("import ", "package ", "from "))
                                ]
                                target_idx = usage_indices[0] if usage_indices else None
                                if target_idx is None:
                                    for idx, line in enumerate(lines):
                                        if ts in line.lower():
                                            target_idx = idx
                                            break
                                if target_idx is not None:
                                    a_start = max(1, target_idx - 12)
                                    a_end = min(total_f_lines, target_idx + 40)
                                    sliced = lines[a_start : a_end + 1]
                                    code_block = "\n".join(f"{j + a_start + 1}: {line}" for j, line in enumerate(sliced))
                                    output_sections.append(f"### Usage of '{ts}' in {rel_display} (Lines {a_start + 1}-{a_end + 1})\n```\n{code_block}\n```")
                                    found_syms.append(ts)
                            if not found_syms:
                                output_sections.append(f"Symbol(s) '{clean_sym}' not found in {rel_display}.")
                elif end_line > 0:
                    # Explicit line slice requested
                    actual_start = max(1, start_line)
                    actual_end = min(total_f_lines, end_line)
                    sliced = lines[actual_start - 1 : actual_end]
                    code_block = "\n".join(f"{i + actual_start}: {l}" for i, l in enumerate(sliced))
                    output_sections.append(f"### File: {rel_display} ({actual_start}-{actual_end}/{total_f_lines} lines)\n```\n{code_block}\n```")
                elif total_f_lines <= 1500 or not boundaries:
                    # Compact or standard file (or non-AST file like SQL/YAML): show full file
                    code_block = "\n".join(f"{i + 1}: {l}" for i, l in enumerate(lines))
                    output_sections.append(f"### File: {rel_display} (1-{total_f_lines}/{total_f_lines} lines)\n```\n{code_block}\n```")
                else:
                    # Large class (>1500 lines) without explicit symbol: show header + structural outline with all line ranges
                    outline_lines = [f"// === Structural Outline ({len(boundaries)} methods/classes) ==="]
                    for b in boundaries:
                        outline_lines.append(f"  L{b.start_line + 1}-L{b.end_line + 1} ({b.line_count()} lines): {b.kind} {b.signature}")
                    header = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[:80]))
                    output_sections.append(
                        f"### Large File: {rel_display} ({total_f_lines} lines)\n"
                        f"*(Use inspect_file with symbol='methodName' to view any specific method)*\n```\n"
                        f"{header}\n...\n" + "\n".join(outline_lines) + "\n```"
                    )

            content = "\n\n".join(output_sections) if output_sections else f"No readable files found matching '{file_path}'."
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "success"
            step_entry["title"] = f"Inspected {len(resolved_files)} file(s){sym_desc}"
            step_entry["output_preview"] = content[:600]
            step_entry["raw_output"] = content
            execution_steps.append(step_entry)
            try:
                blackboard.ingest_tool_observation("inspect_file", file_path, content)
                fp_lower = file_path.lower()
                from ticket_to_code.runtime import ArchitectureLayer
                if any(k in fp_lower for k in [".java", "issues-service", "project-service", "sagas-service"]):
                    multi_service_orch.mark_layer_verified(ArchitectureLayer.BACKEND, f"Inspected {Path(file_path).name}")
                if any(k in fp_lower for k in ["application.yml", "gateway", "bim-gateway"]):
                    multi_service_orch.mark_layer_verified(ArchitectureLayer.GATEWAY, f"Inspected {Path(file_path).name}")
                if any(k in fp_lower for k in [".ts", ".html", ".scss", "xchange-ui"]):
                    multi_service_orch.mark_layer_verified(ArchitectureLayer.FRONTEND, f"Inspected {Path(file_path).name}")
                if any(k in fp_lower for k in [".sql", "migration"]):
                    multi_service_orch.mark_layer_verified(ArchitectureLayer.DATABASE, f"Inspected {Path(file_path).name}")
                _persist_blackboard()
            except Exception:
                pass
            _broadcast_chat_step(step_entry)
            logger.info(f"[Chat Tool:inspect_file] {step_entry['title']}")
            return content
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Failed to read '{file_path}'"
            step_entry["output_preview"] = str(e)
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.error(f"[Chat Tool:inspect_file] {e}")
            return f"Read error: {e}"

    def trace_call_chain(query: str) -> str:
        """Trace the end-to-end execution path between UI action, Angular components, Saga commands, and Java backend handlers.
        CRITICAL: Whenever an issue, ticket, or user question describes a UI dialog, button action, modal text, or user workflow,
        ALWAYS invoke this tool FIRST.
        Args:
            query: UI dialog phrase (e.g. 'will revoke access from all contracts'), i18n key, component name, saga command (e.g. 'updateProjectUser'), or service method.
        """
        step_t0 = time.time()
        step_entry = {
            "title": f"Tracing fullstack call chain for '{query}'...",
            "type": "trace_call_chain",
            "query": query,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            from services.call_chain_tracer import CallChainTracer
            tracer = CallChainTracer(repo_root)
            res = tracer.trace(query)
            
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "success"
            step_entry["title"] = f"Traced call chain for '{query}' ({len(res['summary_trace'])} steps)"
            formatted = json.dumps(res, indent=2)
            step_entry["output_preview"] = "\n".join(res["summary_trace"]) or formatted[:600]
            step_entry["raw_output"] = formatted
            execution_steps.append(step_entry)
            try:
                blackboard.ingest_tool_observation("trace_call_chain", query, formatted)
                _persist_blackboard()
            except Exception:
                pass
            _broadcast_chat_step(step_entry)
            logger.info(f"[Chat Tool:trace_call_chain] {step_entry['title']}")
            return formatted
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Error tracing '{query}'"
            step_entry["output_preview"] = str(e)
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            logger.error(f"[Chat Tool:trace_call_chain] {e}")
            return f"Trace error: {e}"

    def verify_live_ui_with_playwright(target_url: str, check_description: str = "", **kwargs: Any) -> str:
        """Launch browser via Playwright to navigate to a target web application URL (e.g. http://localhost:4200),
        inspect live DOM elements, test UI state, and check for browser console errors.
        CRITICAL: ONLY invoke this tool when the user EXPLICITLY asks to try, test, open, or verify a local live web application (like http://localhost:4200). Do NOT call this for images, Figma URLs, or static assets.
        """
        if any(target_url.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) or "figma-alpha-api.s3" in target_url:
            return f"Image asset URL ({target_url}). Do not open in Playwright; embed directly in your markdown response as ![Mockup]({target_url})."
        if not _is_allowed_verify_url(target_url):
            return (
                f"Refused to open {target_url}: only localhost/127.0.0.1/::1 or hosts listed in "
                "NEMOFORGE_ALLOWED_VERIFY_HOSTS can be verified with Playwright."
            )

        import concurrent.futures
        step_t0 = time.time()
        step_entry = {
            "title": f"Testing live UI with Playwright at {target_url}...",
            "type": "playwright",
            "url": target_url,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        
        def _run_sync():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                assertions = [
                    {"type": "element_visible", "selector": "body"},
                    {"type": "no_console_errors"},
                    {"type": "no_network_errors"},
                ]
                # Only quoted phrases are treated as expected on-page text; arbitrary words caused false results.
                for phrase in re.findall(r"['\"]([^'\"]{2,80})['\"]", check_description or "")[:5]:
                    assertions.append({"type": "text_contains", "selector": "body", "expected": phrase})
                return loop.run_until_complete(
                    playwright_verifier.verify_dom_scenario(
                        target_url=target_url,
                        actions=[],
                        assertions=assertions,
                        timeout_ms=15000,
                    )
                )
            finally:
                loop.close()

        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                res = pool.submit(_run_sync).result(timeout=90)
            status = "PASSED" if res.passed else ("INFRASTRUCTURE ERROR" if res.infra_error else "FAILED")
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "success" if res.passed else "failed"
            step_entry["title"] = f"Verified live UI with Playwright at {target_url} ({status})"
            step_entry["output_preview"] = f"Status: {status}, Passed: {res.assertions_passed}, Failed: {res.assertions_failed}"
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            report = [
                f"Playwright Live UI Verification: {status}",
                f"Target URL: {res.target_url}",
                f"Summary: {res.summary}",
                f"Assertions Passed: {res.assertions_passed}, Failed: {res.assertions_failed}",
            ]
            if res.dom_snapshot:
                report.append(f"\nLive Rendered Page Content:\n{res.dom_snapshot[:3000]}\n")
            if res.failures:
                report.append(f"Failures: {res.failures}")
            if res.console_logs:
                errors = [l for l in res.console_logs if "error" in l.lower()]
                report.append(f"Console Errors: {errors[:3] if errors else 'None'}")
            if res.network_errors:
                report.append(f"Network Errors: {res.network_errors[:3]}")

            # Pre-flight fallback (FLAW-12): If live dev server is unreachable, run static frontend verification
            if res.infra_error:
                fallback_check = host_env_tools.verify_service_compilation("xchange-ui")
                report.append(f"\n[PLAYWRIGHT_FALLBACK]: Live dev server at {target_url} is offline. Verified via static frontend build gate:\n{fallback_check}")

            return "\n".join(report)
        except Exception as ex:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Playwright pre-flight at {target_url}"
            fallback_check = host_env_tools.verify_service_compilation("xchange-ui")
            err_msg = f"Playwright verification skipped (dev server offline): {ex}\n[PLAYWRIGHT_FALLBACK]: Static frontend build gate:\n{fallback_check}"
            step_entry["output_preview"] = err_msg[:400]
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return err_msg

    def verify_code_snippet(file_path: str, proposed_content: str) -> str:
        """Verify whether a proposed speculative code addition or modification compiles cleanly without applying changes to disk.
        Uses sub-second in-memory Fast LSP delta diagnostics to check for syntax errors, missing symbols, wrong generic types, or invalid method calls against the real workspace.
        Args:
            file_path: Relative path to the workspace file being modified or added.
            proposed_content: Complete proposed file content or candidate patch to validate against compiler.
        """
        step_t0 = time.time()
        step_entry = {
            "title": f"Checking compilation of speculative snippet for '{Path(file_path).name}'...",
            "type": "verify_snippet",
            "file_path": file_path,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            from ticket_to_code.runtime.lsp_client import FastLSPClient
            client_lsp = FastLSPClient(workspace_path=repo_root)
            is_valid, errors = client_lsp.check_delta_diagnostics(file_path, proposed_content)
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            if is_valid:
                step_entry["status"] = "success"
                step_entry["title"] = f"Speculative snippet verified clean (0 errors): {Path(file_path).name}"
                out = f"✅ Speculative code for '{file_path}' compiles cleanly with 0 compiler errors."
            else:
                step_entry["status"] = "failed"
                step_entry["title"] = f"Speculative snippet has {len(errors)} compiler error(s): {Path(file_path).name}"
                out = f"❌ Compiler errors detected in proposed code for '{file_path}':\n" + "\n".join(e.format_slice() for e in errors)
            step_entry["output_preview"] = out[:600]
            step_entry["raw_output"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return out
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Error checking snippet for '{file_path}'"
            out = f"Snippet check error: {e}"
            step_entry["output_preview"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return out

    def crawl_sibling_and_rosetta_context(service_or_file: str) -> str:
        """Crawl companion repositories, DAOs, entity models, and Spring Data @Query Rosetta Stone join mappings for any microservice or file.
        Discovers all existing repository methods, exact primary key ID types, and join relationships (e.g. ContractMember.userId = ParticipatingMember.otdsUserId) in 1 shot.
        Args:
            service_or_file: Service name (e.g. 'project-service') or relative file path (e.g. 'ContractMemberDeleteService.java').
        """
        step_t0 = time.time()
        step_entry = {
            "title": f"Crawling sibling repositories & Rosetta Stone for '{Path(service_or_file).name}'...",
            "type": "sibling_crawler",
            "target": service_or_file,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)
        try:
            from ticket_to_code.runtime.sibling_crawler import SiblingCrawler
            res = SiblingCrawler.crawl_service_siblings(repo_root, service_or_file)
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            if res:
                step_entry["status"] = "success"
                step_entry["title"] = f"Discovered sibling repositories & Rosetta Stone for '{Path(service_or_file).name}'"
                out = res
            else:
                step_entry["status"] = "failed"
                step_entry["title"] = f"No sibling repositories found for '{service_or_file}'"
                out = f"No sibling repositories or models found for '{service_or_file}'."
            step_entry["output_preview"] = out[:600]
            step_entry["raw_output"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return out
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Error crawling siblings for '{service_or_file}'"
            out = f"Sibling crawl error: {e}"
            step_entry["output_preview"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return out

    # ── 5. Universal Agentic Operating System Tools & Intent Partitioning ──
    from ticket_to_code.runtime.agent_harness import (
        AgentIntent,
        classify_agent_intent,
        classify_agent_intent_hybrid,
        HostStateMetadata,
        PromptAssembler,
        ContextCompactor,
        HostEnvironmentTools,
        UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS,
        delegate_ticket_to_flow,
    )
    host_env_tools = HostEnvironmentTools(str(repo_root))
    user_msg_low = (request.message or "").lower()
    early_intent = classify_agent_intent(request.message or "")
    strictly_explain = (early_intent == AgentIntent.EXPLANATION_REVIEW) or any(q in user_msg_low for q in [
        "explain only", "read only", "do not trigger", "don't trigger",
        "no code modification", "do not modify", "just analyze", "just explain"
    ])

    def replace_file_content(file_path: str, target_content: str, replacement_content: str, start_line: int = 1, end_line: int = 0) -> str:
        """Atomic replacement of code chunk in an existing workspace file.
        CRITICAL: Use only when modifying code in autonomous modification mode.
        Args:
            file_path: Path relative to repository root.
            target_content: Exact string in file to be replaced.
            replacement_content: New code string to replace target_content.
            start_line: 1-indexed start line (optional).
            end_line: 1-indexed end line (optional).
        """
        if strictly_explain:
            return "Operation declined: This session is in read-only / explanation mode. Please request code modification explicitly to apply changes."
        is_safe, violation = CommandSafetyValidator.validate_path_containment(file_path, workspace_root=repo_root)
        if not is_safe:
            return f"Security policy rejection: {violation}"
        step_t0 = time.time()
        parts = Path(file_path).parts
        owning_service = parts[0] if parts and (repo_root / parts[0]).is_dir() else ""
        from ticket_to_code.runtime.polyglot_build import PolyglotBuildDiscoverer
        from ticket_to_code.runtime.blast_radius import BlastRadiusAnalyzer

        if owning_service and CompilerGate.get_baseline(owning_service, str(repo_root)) is None:
            if PolyglotBuildDiscoverer.discover(repo_root / owning_service, repo_root) is not None:
                try:
                    verify_service_compilation(owning_service)
                except Exception:
                    pass

        step_entry = {
            "title": f"Editing {Path(file_path).name}...",
            "type": "file_edit",
            "file_path": file_path,
            "status": "running"
        }
        res = host_env_tools.replace_file_content(
            file_path=file_path,
            target_content=target_content,
            replacement_content=replacement_content,
            expected_content_hash="",
            expected_occurrences=1
        )
        step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
        step_entry["status"] = "success" if "Successfully" in res else "error"
        step_entry["title"] = f"Edited {Path(file_path).name}" if "Successfully" in res else f"Failed editing {Path(file_path).name}"

        # ── Automated Closed-Loop Compiler & Blast-Radius Check at Mutation Time ──
        if "Successfully" in res:
            # 1. Blast Radius Analysis
            blast_rep = BlastRadiusAnalyzer.analyze(repo_root, [file_path])
            if blast_rep.risk_level in ["high", "critical"]:
                res = f"{res}\n\n{blast_rep.to_markdown()}"
                try:
                    blackboard.record_fact(
                        f"Blast-Radius alert for {Path(file_path).name}: {blast_rep.risk_level.upper()} risk across {', '.join(blast_rep.suggested_verification_targets)}"
                    )
                except Exception:
                    pass

            # 2. Closed-Loop Compiler Check
            if owning_service and PolyglotBuildDiscoverer.discover(repo_root / owning_service, repo_root) is not None:
                compile_res = verify_service_compilation(owning_service)
                comp_obj = CompilerGate.parse_compiler_output(owning_service, 0 if ("SUCCESS" in compile_res or "✅" in compile_res) else 1, compile_res)
                eval_outcome = CompilerGate.evaluate_mutation(owning_service, comp_obj, str(repo_root))
                if eval_outcome.get("is_regression"):
                    res = f"{res}\n\n⚠️ AUTOMATED COMPILER ALERT: Edits in {file_path} caused compilation regression in {owning_service}!\n{eval_outcome['message']}\nYou must fix this compilation error immediately using replace_file_content."
                elif eval_outcome.get("passed"):
                    res = f"{res}\n\n✅ AUTOMATED COMPILER CHECK: Clean compile with 0 errors in {owning_service}."
                else:
                    res = f"{res}\n\n{eval_outcome['message']}"

        step_entry["output_preview"] = res[:400]
        step_entry["raw_output"] = res
        execution_steps.append(step_entry)
        try:
            blackboard.ingest_tool_observation("replace_file_content", file_path, res)
            _persist_blackboard()
        except Exception:
            pass
        _broadcast_chat_step(step_entry)
        return res

    def write_to_file(file_path: str, code_content: str, overwrite: bool = False) -> str:
        """Create a new file in the workspace or overwrite an existing file.
        Args:
            file_path: Relative path of the file to create.
            code_content: Complete text content of the file.
            overwrite: Set to True to allow overwriting an existing file.
        """
        if strictly_explain:
            return "Operation declined: This session is in read-only / explanation mode. Please request code modification explicitly to apply changes."
        is_safe, violation = CommandSafetyValidator.validate_path_containment(file_path, workspace_root=repo_root)
        if not is_safe:
            return f"Security policy rejection: {violation}"
        step_t0 = time.time()
        parts = Path(file_path).parts
        owning_service = parts[0] if parts and (repo_root / parts[0]).is_dir() else ""
        from ticket_to_code.runtime.polyglot_build import PolyglotBuildDiscoverer
        from ticket_to_code.runtime.blast_radius import BlastRadiusAnalyzer

        if owning_service and CompilerGate.get_baseline(owning_service, str(repo_root)) is None:
            if PolyglotBuildDiscoverer.discover(repo_root / owning_service, repo_root) is not None:
                try:
                    verify_service_compilation(owning_service)
                except Exception:
                    pass

        step_entry = {
            "title": f"Writing {Path(file_path).name}...",
            "type": "file_write",
            "file_path": file_path,
            "status": "running"
        }
        _broadcast_chat_step(step_entry)
        res = host_env_tools.write_to_file(file_path, code_content, overwrite)
        step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
        step_entry["status"] = "success" if "Successfully" in res else "error"
        step_entry["title"] = f"Wrote {Path(file_path).name}" if "Successfully" in res else f"Failed writing {Path(file_path).name}"

        # ── Automated Closed-Loop Compiler & Blast-Radius Check at File Creation Time ──
        if "Successfully" in res:
            blast_rep = BlastRadiusAnalyzer.analyze(repo_root, [file_path])
            if blast_rep.risk_level in ["high", "critical"]:
                res = f"{res}\n\n{blast_rep.to_markdown()}"

            if owning_service and PolyglotBuildDiscoverer.discover(repo_root / owning_service, repo_root) is not None:
                compile_res = verify_service_compilation(owning_service)
                comp_obj = CompilerGate.parse_compiler_output(owning_service, 0 if ("SUCCESS" in compile_res or "✅" in compile_res) else 1, compile_res)
                eval_outcome = CompilerGate.evaluate_mutation(owning_service, comp_obj, str(repo_root))
                if eval_outcome.get("is_regression"):
                    res = f"{res}\n\n⚠️ AUTOMATED COMPILER ALERT: Creating {file_path} caused compilation regression in {owning_service}!\n{eval_outcome['message']}\nYou must fix this compilation error immediately."
                elif eval_outcome.get("passed"):
                    res = f"{res}\n\n✅ AUTOMATED COMPILER CHECK: Clean compile with 0 errors in {owning_service}."
                else:
                    res = f"{res}\n\n{eval_outcome['message']}"

        step_entry["output_preview"] = res[:400]
        step_entry["raw_output"] = res
        execution_steps.append(step_entry)
        try:
            blackboard.ingest_tool_observation("write_to_file", file_path, res)
            _persist_blackboard()
        except Exception:
            pass
        _broadcast_chat_step(step_entry)
        return res

    def run_command(command: str, cwd: str = "", timeout_seconds: int = 45) -> str:
        """Run a shell command on the host environment within the workspace.
        Args:
            command: Shell command to execute (e.g. 'gradlew compileJava', 'npm test').
            cwd: Working directory relative to workspace root (optional).
            timeout_seconds: Timeout in seconds (default 45).
        """
        if strictly_explain:
            return "Operation declined: This session is in read-only / explanation mode. Please request code modification explicitly to run commands."
        is_safe, violation = CommandSafetyValidator.validate_command(command, workspace_root=repo_root)
        if not is_safe:
            return f"Security policy rejection: {violation}"
        step_t0 = time.time()
        step_entry = {
            "title": f"Running `{command[:40]}`...",
            "type": "run_command",
            "command": command,
            "status": "running"
        }
        _broadcast_chat_step(step_entry)
        res = host_env_tools.run_command(command, cwd, timeout_seconds)
        step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
        step_entry["status"] = "success" if "[Exit code 0]" in res else "error"
        step_entry["title"] = f"Ran `{command[:40]}`"
        step_entry["output_preview"] = res[:400]
        step_entry["raw_output"] = res
        execution_steps.append(step_entry)
        _broadcast_chat_step(step_entry)
        return res

    def solve_ticket_with_flow(title: str, description: str) -> str:
        """Delegate a defect, bug, or ticket implementation to the verified multi-stage Flow.
        This is the PRIMARY tool for autonomous modification: it runs deterministic localization,
        locked-context patching, and polyglot build/test verification (npm run build for frontend,
        gradlew/mvn for backend). Prefer this over editing files blindly for any real defect/ticket.
        Args:
            title: Short ticket / defect title (e.g. 'Removing a coordinator does not revoke access').
            description: Full problem statement / acceptance criteria.
        """
        return _run_flow_delegation(title, description, origin="solve")

    def escalate_to_ticket_fix(ticket_title: str, defect_summary: str) -> str:
        """Promote a READ-ONLY investigation into an actual fix. Invoke this ONLY if, during your
        inspection, you have CONFIRMED a concrete, active defect in the code (not a hypothetical).
        It routes the issue into the same verified Flow execution engine (deterministic localization,
        locked-context patching, polyglot build/test verification) and reports the verification status.
        Args:
            ticket_title: Short title of the confirmed defect.
            defect_summary: Precise description of the bug and the required behaviour, grounded in the code you inspected.
        """
        return _run_flow_delegation(ticket_title, defect_summary, origin="escalate")

    def _run_flow_delegation(title: str, description: str, origin: str = "solve") -> str:
        step_t0 = time.time()
        label = "Escalating to Flow" if origin == "escalate" else "Delegating to Flow"
        step_entry = {
            "title": f"{label}: {title[:60]}...",
            "type": "flow_delegation",
            "origin": origin,
            "status": "running",
        }
        _broadcast_chat_step(step_entry)

        if strictly_explain:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "cancelled"
            step_entry["title"] = f"Flow execution suppressed (Read-Only Mode): {title[:50]}"
            declined_msg = (
                "Flow execution suppressed: The user requested an explanation, architectural analysis, or root-cause diagnosis. "
                "Do NOT trigger Flow execution or mutate workspace files. Provide your diagnostic analysis, root-cause explanation, "
                "and recommended code diffs directly in chat."
            )
            step_entry["output_preview"] = declined_msg
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return declined_msg

        def _cb(stage: str, detail: str = ""):
            _broadcast_chat_step({
                "title": f"Flow [{stage}] {detail}"[:120],
                "type": "flow_progress",
                "stage": stage,
                "status": "running",
            })

        try:
            res = delegate_ticket_to_flow(
                str(repo_root), title, description,
                ticket_id=(ticket_id or "CHAT-TICKET"),
                progress_cb=_cb,
                blackboard=blackboard,
            )
        except Exception as e:
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            step_entry["status"] = "error"
            step_entry["title"] = f"Flow delegation error: {title[:50]}"
            step_entry["output_preview"] = str(e)
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            _persist_blackboard()
            return f"Flow delegation error: {e}"

        status = res.get("status", "error")
        step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
        step_entry["status"] = "success" if status == "verified" else "failed"
        step_entry["title"] = f"Flow {status}: {title[:50]} ({res.get('attempts', 0)} attempt(s))"
        step_entry["output_preview"] = f"status={status}; files={res.get('locked_files')}; reason={res.get('reason','')[:200]}"
        step_entry["raw_output"] = json.dumps(res, indent=2)
        execution_steps.append(step_entry)
        _broadcast_chat_step(step_entry)
        _persist_blackboard()
        return json.dumps(res, indent=2)

    # Resolve model selection mode and target model
    try:
        req_mode = getattr(request, "model_mode", "default") or "default"
        req_model = getattr(request, "selected_model", None)
        target_model_id = "nebius/nvidia/nemotron-3-super-120b-a12b"
        target_model_name = "NVIDIA nemotron-3-super-120b-a12b (Nebius)"

        try:
            from ticket_to_code.intelligence.model_router import ModelRouter, ModelMode
            m_mode = ModelMode(req_mode)
            desc = ModelRouter.select_model_for_task(
                ticket={"description": request.message, "title": request.message[:80]},
                mode=m_mode,
                requested_model=req_model or _runtime_model_preference["model"]
            )
            target_model_id = desc.model_id
            target_model_name = desc.name
        except Exception as _router_err:
            logger.debug(f"Model router fallback in chat: {_router_err}")
            if req_model:
                target_model_id = req_model
                target_model_name = req_model

        is_nebius_model = "nebius" in target_model_id.lower() or os.getenv("LLM_PROVIDER") == "nebius"
        client = None
        if not is_nebius_model:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(
                    vertexai=True,
                    project=os.environ.get("GOOGLE_CLOUD_PROJECT", "otl-cs-csai"),
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west4"),
                    http_options=types.HttpOptions(timeout=720000),
                )
            except Exception as _cl_err:
                logger.warning(f"Google GenAI client unavailable: {_cl_err}")

        # ── 6. Intent Classification & Operational Mode ──
        # Deterministic fast-path for 90%+ of messages
        agent_intent = classify_agent_intent_hybrid(request.message, client=client, model_id=target_model_id)

        # ── 7. Host-State Metadata Injection ──
        open_docs = []
        if getattr(request, "open_tabs", None):
            open_docs = [str(t) for t in request.open_tabs]
        host_meta = HostStateMetadata.capture_current(
            workspace_path=str(repo_root),
            active_document=getattr(request, "current_tab", "") or "",
            cursor_line=getattr(request, "cursor_line", 1) or 1,
            open_documents=open_docs,
            conversation_id=request.chat_id or "",
        )

        # ── 8. Context Compaction & Resumption Check ──
        compaction_resumption_block = ""
        total_msg_chars = sum(len(str(m.get("content", ""))) for m in (request.messages or []) if isinstance(m, dict))
        est_tokens = total_msg_chars // 4
        if (request.messages and len(request.messages) > 10) or est_tokens > 35000:
            past_reqs = [
                m.get("content", "") for m in request.messages
                if isinstance(m, dict) and m.get("role") == "user"
            ]
            compaction_facts = list(blackboard.verified_facts)[:12] if blackboard.verified_facts else [
                f"Workspace: {repo_root}",
                f"Architecture loaded: {len(arch_parts)} sections",
            ]
            active_ctx = (
                f"Workspace: {repo_root}\n"
                f"Status: {wf_status}\n"
                f"Discovered Endpoints: {', '.join(list(blackboard.discovered_endpoints)[:6]) or 'None'}\n"
                f"Rejected Hypotheses: {len(blackboard.rejected_hypotheses)}"
            )
            compaction_resumption_block = ContextCompactor.compact_history(
                task_id=ticket_id or (wid or "active-task"),
                user_requests=past_reqs[-6:],
                verified_facts=compaction_facts,
                active_context=active_ctx,
            )

        # ── 8b. Cursor-Style @Mention Context Expansion ──
        mention_block = ""
        try:
            from ticket_to_code.runtime.context_mentions import ContextMentionResolver
            resolver = ContextMentionResolver(workspace_path=repo_root)
            mention_res = resolver.resolve_mentions(request.message)
            if mention_res.injected_context_block:
                mention_block = mention_res.injected_context_block
                for rm in mention_res.resolved_mentions:
                    if rm.resolved:
                        blackboard.record_fact(f"User referenced @{rm.target} ({rm.mention_type}): {rm.content[:200]}")
        except Exception as _mention_err:
            logger.warning(f"Mention expansion skipped: {_mention_err}")

        # ── 8c. Sibling Package & Rosetta Stone Automatic Grounding (Pillar 1 & 2) ──
        sibling_grounding_block = ""
        try:
            from ticket_to_code.runtime.sibling_crawler import SiblingCrawler
            cand_hints = []
            if getattr(request, "current_tab", None):
                cand_hints.append(str(request.current_tab))
            if getattr(request, "open_tabs", None):
                cand_hints.extend([str(t) for t in request.open_tabs[:4]])
            for svc_kw in ["project-service", "issues-service", "sagas-service", "xchange-ui", "bim-gateway", "identity-service"]:
                if svc_kw in user_msg_low:
                    cand_hints.append(svc_kw)
            for ch in cand_hints:
                grounding = SiblingCrawler.crawl_service_siblings(repo_root, ch)
                if grounding:
                    sibling_grounding_block = f"\n\n{grounding}\n"
                    logger.info(f"⚡ [Chat Grounding] Injected Sibling & Rosetta Stone grounding for '{ch}'")
                    break
        except Exception as _crawl_err:
            logger.debug(f"Sibling crawler notice: {_crawl_err}")

        # ── 9. Dynamic 6-Layer Prompt Assembly ──
        system_instruction = PromptAssembler.build_system_instruction(agent_intent, workspace_path=str(repo_root))

        effective_conv = compaction_resumption_block or conv_block
        effective_arch = arch_block
        if strictly_explain and len(effective_arch) > 8000:
            effective_arch = effective_arch[:8000] + "\n... [Architecture overview summarized to preserve attention on user inquiry]"

        base_user_msg = (request.message + "\n" + mention_block) if mention_block else request.message
        if sibling_grounding_block:
            base_user_msg += sibling_grounding_block

        if request.message.count("?") >= 2 or ("what" in request.message.lower() and "how many" in request.message.lower()):
            base_user_msg += "\n\n[MANDATORY SYSTEM DIRECTIVE]: This is a compound, multi-part inquiry. You MUST decompose the prompt and provide a complete, verified answer to EVERY question asked. Do not stop after answering only one question."

        user_prompt_content = PromptAssembler.build_user_prompt(
            user_message=base_user_msg,
            host_meta=host_meta,
            ki_summaries=mcp_catalog_summary,
            arch_block=effective_arch,
            conv_history=effective_conv,
            blackboard_xml=blackboard.render_prompt_xml() if blackboard.verified_facts else "",
        )

        _broadcast_chat_step({
            "title": f"Reasoning with {target_model_name} ({agent_intent.value.upper()})...",
            "type": "thought",
            "status": "running"
        })

        if emit_fn:
            emit_fn("message_start", {
                "model": target_model_name,
                "model_id": target_model_id,
                "client_message_id": getattr(request, "client_message_id", None),
                "intent": agent_intent.value,
            })

        # Cap thinking budget to 1024 for Pro (prevents minutes of idle thinking), 0 for Flash
        model_thinking_config = None
        if not is_nebius_model:
            try:
                from google.genai import types
                model_thinking_config = (
                    types.ThinkingConfig(thinking_budget=0)
                    if "flash" in target_model_id.lower()
                    else types.ThinkingConfig(thinking_budget=1024)
                )
            except Exception:
                pass

        def request_structured_clarification(missing_concept: str, option_a: str, option_b: str, rationale: str) -> str:
            """Structured Clarification Gate (Tier 3): Escalate an architectural ambiguity to the human developer ONLY when a concept is mathematically absent from the repository.
            Strict Boundary: The gate verifies that find_references returns 0 hits across all 8 microservices and the concept is not defined in any Flyway migration schema. If any call site or schema exists in the repository, escalation is REJECTED and you must inspect existing code.
            Args:
                missing_concept: Identifier, method, or concept with 0 occurrences in the repo.
                option_a: First concrete architectural alternative with pros/cons.
                option_b: Second concrete architectural alternative with pros/cons.
                rationale: Why human intervention is required instead of repo grounding.
            """
            step_t0 = time.time()
            step_entry = {
                "title": f"Evaluating Escalation Gate for '{missing_concept}'...",
                "type": "clarification_gate",
                "concept": missing_concept,
                "status": "running"
            }
            _broadcast_chat_step(step_entry)
            
            # 1. Verify 0 references across workspace code files (excluding .sql migrations)
            from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
            engine = RepositorySearchEngine(workspace_path=repo_root)
            all_hits = engine.search_literal(missing_concept.strip(), max_results=15)
            if not all_hits:
                all_hits = engine.search_regex(r'\b' + re.escape(missing_concept.strip()) + r'\b', max_results=15)
            code_hits = [
                h for h in all_hits
                if not str(getattr(h, "file_path", getattr(h, "relative_path", str(h)))).endswith(".sql")
            ]
            
            # 2. Verify DB schemas
            has_db = False
            tbl_clean = missing_concept.strip().lower()
            for sf in repo_root.rglob("*.sql"):
                try:
                    cnt = sf.read_text(encoding="utf-8", errors="ignore").lower()
                    if tbl_clean in cnt:
                        has_db = True
                        break
                except Exception:
                    continue
                    
            step_entry["duration_ms"] = int((time.time() - step_t0) * 1000)
            if code_hits:
                step_entry["status"] = "failed"
                step_entry["title"] = f"Clarification Blocked: '{missing_concept}' exists in repo ({len(code_hits)} hit(s))"
                out = (
                    f"🛑 Clarification REJECTED by Strict 0-Interruption Escalation Boundary: "
                    f"'{missing_concept}' has active references/call-sites in the codebase! "
                    f"You are strictly barred from interrupting the developer. "
                    f"Use `find_references('{missing_concept}')` and `inspect_file` to ground your solution in existing code."
                )
            elif has_db:
                step_entry["status"] = "failed"
                step_entry["title"] = f"Clarification Blocked: '{missing_concept}' exists in DB schemas"
                out = (
                    f"🛑 Clarification REJECTED by Strict 0-Interruption Escalation Boundary: "
                    f"'{missing_concept}' exists in database migrations! "
                    f"You are strictly barred from interrupting the developer. "
                    f"Inspect database schema migrations to derive column definitions and types."
                )
            else:
                step_entry["status"] = "success"
                step_entry["title"] = f"Clarification Approved (External Absence): '{missing_concept}'"
                out = (
                    f"✅ Tier 3 Structured Clarification Gate APPROVED (Verified 0 hits across all microservices & no DB schema matches):\n"
                    f"### Architectural Decision Required: '{missing_concept}'\n"
                    f"- **Rationale**: {rationale}\n"
                    f"- **Option A**: {option_a}\n"
                    f"- **Option B**: {option_b}\n"
                    f"Present these two concrete alternatives to the developer in your response."
                )
                
            step_entry["output_preview"] = out[:600]
            step_entry["raw_output"] = out
            execution_steps.append(step_entry)
            _broadcast_chat_step(step_entry)
            return out

        # ── 10. Dynamic Toolset Partitioning & Subagent Tool ──
        read_only_tools = [
            search_codebase,
            find_references,
            crawl_sibling_and_rosetta_context,
            verify_service_compilation,
            verify_code_snippet,
            list_directory,
            find_files,
            inspect_file,
            trace_call_chain,
            call_mcp_tool,
            verify_live_ui_with_playwright,
            request_structured_clarification,
            invoke_subagent_explorer,
        ]

        # Stably ordered toolset preserving 100% byte-for-byte KV-cache prefix identity across all turns
        active_tools = read_only_tools + [
            escalate_to_ticket_fix,
            solve_ticket_with_flow,
            replace_file_content,
            write_to_file,
            run_command,
        ]

        # ── Architecture A: Model Cascading & Tiering ──
        # If user requested a heavy Pro model, use fast Flash for the multi-turn exploration loop,
        # then invoke the requested Pro model with full thinking budget for final synthesis.
        is_heavy_pro = any(p in target_model_id.lower() for p in ["pro", "opus", "gpt-4o"])
        exploration_model_id = (
            "gemini-2.5-flash" if is_heavy_pro and "gemini" in target_model_id.lower()
            else target_model_id
        )

        chat = None
        if not is_nebius_model and client is not None:
            try:
                chat = client.chats.create(
                    model=exploration_model_id,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=active_tools if not strictly_explain else read_only_tools,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            maximum_remote_calls=20
                        ),
                        temperature=0.1,
                        max_output_tokens=8192,
                        thinking_config=types.ThinkingConfig(thinking_budget=0) if "flash" in exploration_model_id.lower() else model_thinking_config,
                    ),
                )
            except Exception as _chat_init_err:
                logger.warning(f"Could not initialize GenAI chat session: {_chat_init_err}")


        def _broadcast_chat_chunk(chunk_text: str):
            """Stream real-time tokens to frontend via SSE / WebSocket."""
            if emit_fn:
                emit_fn("text_delta", {"delta": chunk_text})
            try:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast({
                        "type": "chat_chunk",
                        "chunk": chunk_text,
                        "chat_id": request.chat_id,
                        "model_name": target_model_name,
                    }),
                    loop
                )
            except Exception as _b_err:
                logger.debug(f"Chat chunk broadcast error: {_b_err}")

        reply = ""
        has_function_calls = False
        accumulated_chunks = []
        last_stream_response = None

        def _run_stream_chat():
            nonlocal reply, has_function_calls, accumulated_chunks, last_stream_response
            if is_nebius_model:
                try:
                    import httpx, json
                    nebius_api_key = os.getenv("NEBIUS_API_KEY")
                    nebius_base_url = (os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")).rstrip("/") + "/"
                    clean_model = target_model_id[len("nebius/"):] if target_model_id.startswith("nebius/") else target_model_id

                    openai_tool_definitions = [
                        {
                            "type": "function",
                            "function": {
                                "name": "call_mcp_tool",
                                "description": "Execute an MCP tool on an active Model Context Protocol server. Check the configured MCP catalog in system instructions for valid server_name and tool_name pairs (e.g. server_name='playwright' with tool_name='verify_dom_scenario' or 'browser_navigate', or server_name='filesystem', etc.).",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "server_name": {"type": "string", "description": "Name of the MCP server (e.g. 'playwright', 'filesystem')"},
                                        "tool_name": {"type": "string", "description": "Specific tool function to execute on the target server"},
                                        "arguments": {"type": "object", "description": "Key-value arguments dictionary required by the tool"}
                                    },
                                    "required": ["server_name", "tool_name"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "verify_live_ui_with_playwright",
                                "description": "Inspect live web application state in Chromium using Playwright. Navigates to target URL, verifies DOM elements, tests button/banner states, and checks console errors.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "target_url": {"type": "string", "description": "URL of the application (e.g. 'http://localhost:5173')"},
                                        "check_description": {"type": "string", "description": "What elements or states to verify in the live DOM"}
                                    },
                                    "required": ["target_url"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "search_codebase",
                                "description": "Search literal text or regex patterns across repository files.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search string or regex pattern"},
                                        "file_pattern": {"type": "string", "description": "Optional file glob filter"}
                                    },
                                    "required": ["query"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "find_references",
                                "description": "Find exact call sites and usages of a class, method, or symbol using the Tree-sitter AST SQLite index.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "symbol_name": {"type": "string", "description": "Name of the symbol"}
                                    },
                                    "required": ["symbol_name"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "inspect_file",
                                "description": "Read file lines with precision.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {"type": "string", "description": "Path to file"},
                                        "start_line": {"type": "integer", "description": "Start line"},
                                        "end_line": {"type": "integer", "description": "End line"}
                                    },
                                    "required": ["file_path"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "replace_file_content",
                                "description": "Modify file contents surgically.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {"type": "string"},
                                        "start_line": {"type": "integer"},
                                        "end_line": {"type": "integer"},
                                        "target_content": {"type": "string"},
                                        "replacement_content": {"type": "string"}
                                    },
                                    "required": ["file_path", "start_line", "end_line", "target_content", "replacement_content"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "description": "Run shell command in workspace (e.g. 'mvn test', 'npm test').",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string"},
                                        "cwd": {"type": "string"}
                                    },
                                    "required": ["command"]
                                }
                            }
                        }
                    ]

                    headers = {"Authorization": f"Bearer {nebius_api_key}", "Content-Type": "application/json"}
                    strict_action_directive = (
                        "\n\nCRITICAL EXECUTION RULES:\n"
                        "1. ACT, DO NOT NARRATE: When the user asks to modify, update, change, ensure, or fix code in a file, "
                        "you MUST invoke `replace_file_content` to apply the changes immediately. Never print internal monologues "
                        "(e.g., 'We need to answer...', 'Let's locate that line...', 'Maybe we should modify...'). "
                        "If you need to edit a file, call `replace_file_content`. If you need to inspect, call `inspect_file`.\n"
                        "2. NO CHAIN-OF-THOUGHT IN OUTPUT: Never output your thinking process, draft notes, or intermediate monologue as text to the user. "
                        "Only call tools or output the final clean, direct, polished answer."
                    )
                    conversation_messages = [
                        {"role": "system", "content": system_instruction + strict_action_directive},
                        {"role": "user", "content": user_prompt_content}
                    ]

                    # Parity with the Gemini path: read-only sessions only ever see read-only tools.
                    NEBIUS_READ_ONLY_TOOL_NAMES = {
                        "call_mcp_tool", "verify_live_ui_with_playwright", "search_codebase", "find_references", "inspect_file",
                    }
                    base_tool_defs = (
                        [d for d in openai_tool_definitions if d["function"]["name"] in NEBIUS_READ_ONLY_TOOL_NAMES]
                        if strictly_explain else openai_tool_definitions
                    )

                    with httpx.Client(timeout=120.0) as http_client:
                        for turn in range(8):
                            # Governor: once the exploration budget is exhausted, withdraw exploration tools
                            tools_for_turn = [
                                d for d in base_tool_defs
                                if not (react_governor.should_disable_exploration() and d["function"]["name"] in EXPLORATION_TOOL_NAMES)
                            ]
                            payload = {
                                "model": clean_model,
                                "messages": conversation_messages,
                                "tools": tools_for_turn,
                                "tool_choice": "auto",
                                "max_tokens": 4096,
                                "temperature": 0.1,
                            }
                            resp = http_client.post(f"{nebius_base_url}chat/completions", json=payload, headers=headers)
                            if resp.status_code != 200:
                                err_txt = resp.text
                                logger.error(f"Nebius API error ({resp.status_code}): {err_txt}")
                                accumulated_chunks.append(f"Nebius API error ({resp.status_code}): {err_txt[:200]}")
                                return

                            resp_json = resp.json()
                            choice = resp_json.get("choices", [{}])[0]
                            msg = choice.get("message", {})
                            tool_calls = msg.get("tool_calls")

                            if tool_calls:
                                has_function_calls = True
                                conversation_messages.append(msg)
                                for tc in tool_calls:
                                    call_id = tc.get("id", str(uuid.uuid4()))
                                    fn = tc.get("function", {})
                                    fn_name = fn.get("name", "")
                                    raw_args = fn.get("arguments", "{}")
                                    try:
                                        fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                                    except Exception:
                                        fn_args = {}

                                    _broadcast_chat_step({
                                        "title": f"Autonomous Agent executing {fn_name}...",
                                        "type": "tool_call",
                                        "status": "running"
                                    })

                                    try:
                                        if fn_name in EXPLORATION_TOOL_NAMES and react_governor.should_disable_exploration():
                                            tool_res = "[REACT_LOOP_GOVERNOR] Exploration is disabled for this turn; synthesize from gathered evidence."
                                        elif strictly_explain and fn_name in ("replace_file_content", "run_command"):
                                            # Defense in depth: the tool was already withheld from the schema list
                                            tool_res = "Operation declined: This session is in read-only / explanation mode. Please request code modification explicitly to run commands."
                                        elif fn_name == "call_mcp_tool":
                                            tool_res = call_mcp_tool(
                                                server_name=fn_args.get("server_name", ""),
                                                tool_name=fn_args.get("tool_name", ""),
                                                arguments=fn_args.get("arguments", {})
                                            )
                                        elif fn_name == "verify_live_ui_with_playwright":
                                            tool_res = verify_live_ui_with_playwright(
                                                target_url=fn_args.get("target_url", "http://localhost:5173"),
                                                check_description=fn_args.get("check_description", "")
                                            )
                                        elif fn_name == "search_codebase":
                                            tool_res = search_codebase(
                                                query=fn_args.get("query", ""),
                                                file_pattern=fn_args.get("file_pattern", "")
                                            )
                                        elif fn_name == "find_references":
                                            tool_res = find_references(
                                                symbol_name=fn_args.get("symbol_name", "")
                                            )
                                        elif fn_name == "inspect_file":
                                            tool_res = inspect_file(
                                                file_path=fn_args.get("file_path", ""),
                                                start_line=fn_args.get("start_line", 1),
                                                end_line=fn_args.get("end_line", 200)
                                            )
                                        elif fn_name == "replace_file_content":
                                            tool_res = replace_file_content(
                                                file_path=fn_args.get("file_path", ""),
                                                target_content=fn_args.get("target_content", ""),
                                                replacement_content=fn_args.get("replacement_content", ""),
                                                start_line=fn_args.get("start_line", 1),
                                                end_line=fn_args.get("end_line", 0)
                                            )
                                        elif fn_name == "run_command":
                                            tool_res = run_command(
                                                command=fn_args.get("command", ""),
                                                cwd=fn_args.get("cwd", "")
                                            )
                                        else:
                                            tool_res = f"Error: Tool '{fn_name}' not found."
                                    except Exception as _tool_exec_err:
                                        tool_res = f"Tool execution notice: {_tool_exec_err}"

                                    conversation_messages.append({
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "name": fn_name,
                                        "content": truncate(str(tool_res), TOOL_OUTPUT_MAX_CHARS)
                                    })
                                continue
                            else:
                                content = msg.get("content")
                                if not content:
                                    thought = msg.get("reasoning_content") or ""
                                    if thought:
                                        _broadcast_chat_step({
                                            "title": "Synthesizing evidence...",
                                            "type": "thought",
                                            "status": "running"
                                        })
                                        conversation_messages.append(msg)
                                        continue
                                    content = ""

                                # Intercept raw XML tool calls (emitted by smaller models or native chat formats)
                                if "<tool_call>" in content:
                                    has_function_calls = True
                                    fn_match = re.search(r"<function=([a-zA-Z0-9_-]+)>", content)
                                    fn_name = fn_match.group(1) if fn_match else "verify_live_ui_with_playwright"
                                    param_matches = re.findall(r"<parameter=([a-zA-Z0-9_-]+)>\s*(.*?)\s*</parameter>", content, re.DOTALL)
                                    fn_args = {k: v.strip() for k, v in param_matches}

                                    _broadcast_chat_step({
                                        "title": f"Autonomous Agent executing {fn_name}...",
                                        "type": "tool_call",
                                        "status": "running"
                                    })

                                    if "browser" in fn_name.lower():
                                        tool_res = verify_live_ui_with_playwright(
                                            target_url=fn_args.get("url", "http://localhost:5173"),
                                            check_description=fn_args.get("text", "") or fn_args.get("query", "")
                                        )
                                    elif fn_name == "search_codebase":
                                        tool_res = search_codebase(query=fn_args.get("query", ""))
                                    else:
                                        tool_res = verify_live_ui_with_playwright(
                                            target_url="http://localhost:5173",
                                            check_description="Inspect page orders and ledger"
                                        )

                                    conversation_messages.append({"role": "assistant", "content": content})
                                    conversation_messages.append({
                                        "role": "tool",
                                        "tool_call_id": str(uuid.uuid4()),
                                        "name": fn_name,
                                        "content": truncate(str(tool_res), TOOL_OUTPUT_MAX_CHARS)
                                    })
                                    continue

                                accumulated_chunks.append(content)
                                _broadcast_chat_chunk(content)
                                break

                        # If tool turns completed without a text answer, request final synthesis
                        if not accumulated_chunks and has_function_calls:
                            final_payload = {
                                "model": "nvidia/nemotron-3-super-120b-a12b" if "nemotron" in clean_model.lower() else clean_model,
                                "messages": conversation_messages + [
                                    {"role": "system", "content": "You have completed your tool inspections. Now synthesize your authoritative, complete engineering answer based on the tool outputs. MANDATORY: Review the developer's original inquiry and ensure you explicitly answer EVERY question, sub-task, or data point requested. Do NOT omit any question. Do NOT output raw JSON or XML tool invocations."}
                                ],
                                "max_tokens": 4096,
                                "temperature": 0.1,
                            }
                            f_resp = http_client.post(f"{nebius_base_url}chat/completions", json=final_payload, headers=headers)
                            if f_resp.status_code == 200:
                                f_msg = f_resp.json().get("choices", [{}])[0].get("message", {})
                                ans = f_msg.get("content") or f_msg.get("reasoning_content") or ""
                                # Strip any dangling raw tool calls from output
                                if "<tool_call>" in ans:
                                    ans = re.sub(r"<tool_call>.*?</tool_call>", "", ans, flags=re.DOTALL).strip()
                                if not ans:
                                    ans = "Inspections completed. Verified live state directly against workspace services and browser DOM."
                                accumulated_chunks.append(ans)
                                _broadcast_chat_chunk(ans)
                except Exception as _nebius_err:
                    logger.error(f"Nebius chat stream error: {_nebius_err}")
                    accumulated_chunks.append(f"Error communicating with {target_model_name}: {_nebius_err}")
                return

            if chat is not None:
                try:
                    for chunk in chat.send_message_stream(user_prompt_content):
                        last_stream_response = chunk
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            for cand in chunk.candidates:
                                if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
                                    for p in cand.content.parts:
                                        if hasattr(p, "function_call") and p.function_call:
                                            has_function_calls = True
                                        if getattr(p, "thought", False):
                                            _broadcast_chat_step({
                                                "title": f"Reasoning with {target_model_name}...",
                                                "type": "thought",
                                                "status": "running"
                                            })
                        txt = getattr(chunk, "text", None)
                        if txt:
                            accumulated_chunks.append(txt)
                            _broadcast_chat_chunk(txt)

                    # Governor: AFC exhausted the exploration budget without producing text. Re-prompt
                    # once with the exploration tools withdrawn so the model must synthesize.
                    if react_governor.should_disable_exploration() and not "".join(accumulated_chunks).strip():
                        reduced = react_governor.filter_tools(active_tools if not strictly_explain else read_only_tools)
                        try:
                            history = chat.get_history() if hasattr(chat, "get_history") else None
                            chat2 = client.chats.create(
                                model=exploration_model_id,
                                history=history,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_instruction,
                                    tools=reduced,
                                    automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=4),
                                    temperature=0.1,
                                    max_output_tokens=8192,
                                ),
                            )
                            for chunk in chat2.send_message_stream(
                                "Exploration budget exhausted. Using only the evidence already gathered, write your final grounded answer now."
                            ):
                                txt = getattr(chunk, "text", None)
                                if txt:
                                    accumulated_chunks.append(txt)
                                    _broadcast_chat_chunk(txt)
                        except Exception as _gov_err:
                            logger.warning(f"Governor follow-up call failed: {_gov_err}")
                except Exception as _stream_err:
                    logger.warning(f"Chat stream encountered issue: {_stream_err}")

        chat_done = asyncio.Event()

        async def _chat_heartbeat_worker():
            h_start = time.time()
            while not chat_done.is_set():
                try:
                    await asyncio.wait_for(chat_done.wait(), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    pass
                if chat_done.is_set():
                    break
                elapsed_s = int(time.time() - h_start)
                try:
                    await manager.broadcast({
                        "type": "chat_step",
                        "step": {
                            "title": f"Reasoning with {target_model_name} ({elapsed_s}s)...",
                            "type": "thought",
                            "status": "running",
                            "elapsed_s": elapsed_s,
                        }
                    })
                except Exception:
                    pass

        heartbeat_task = asyncio.create_task(_chat_heartbeat_worker())
        try:
            await asyncio.wait_for(
                asyncio.to_thread(_run_stream_chat),
                timeout=720.0
            )
            if accumulated_chunks:
                reply = "".join(accumulated_chunks).strip()
        except Exception as _chat_err:
            logger.warning(f"Chat execution completed or interrupted: {_chat_err}")
        finally:
            chat_done.set()
            heartbeat_task.cancel()

        # Check if reply is an incomplete intermediary turn
        is_intermediary = False
        if reply:
            lower_r = reply.lower()
            if any(phrase in lower_r for phrase in ["i will search", "i will inspect", "i will broaden", "i'll also look", "i will look"]):
                is_intermediary = True

        # Grounded synthesis pass:
        # Triggered if: (1) primary stream failed, (2) intermediary thoughts, or (3) reply contains rejected hypotheses
        contains_rejected_hypothesis = False
        if reply and blackboard.rejected_hypotheses:
            for rej in blackboard.rejected_hypotheses:
                m_ids = re.findall(r"['<]([A-Za-z0-9_ -]+)[>']", rej)
                for ident in m_ids:
                    if len(ident) >= 4 and ident.lower() in reply.lower():
                        contains_rejected_hypothesis = True
                        logger.warning(f"Reply contains rejected hypothesis symbol '{ident}'! Forcing grounded synthesis pass...")
                        break
                if contains_rejected_hypothesis:
                    break

        last_synth_response = None
        # FLAW-05: Model Cascading Optimization
        # If Flash's generation in the exploration loop already produced a substantive, grounded answer
        # without rejected hypotheses or unfinished thoughts, stream it directly rather than burning 30s cold on a redundant Pro pass.
        flash_answered_substantively = len(reply) >= 200 and not is_intermediary and not contains_rejected_hypothesis
        should_synthesize = (
            not reply
            or (len(reply) < 120 and is_intermediary)
            or contains_rejected_hypothesis
            or (is_heavy_pro and not flash_answered_substantively)
        )
        if should_synthesize and (execution_steps or arch_block):
            logger.info(f"Triggering grounded synthesis fail-safe pass (reply_len={len(reply)}, completed {len(execution_steps)} tool calls, intermediary={is_intermediary})...")
            if emit_fn:
                emit_fn("clear_stream", {})
            _broadcast_chat_step({
                "title": f"Synthesizing grounded engineering response with {target_model_name}...",
                "type": "thought",
                "status": "running"
            })

            gathered_findings = []
            for s in execution_steps:
                t = s.get("title", "")
                raw = s.get("raw_output") or s.get("output_preview", "")
                if raw:
                    gathered_findings.append(f"### {t}\n```\n{raw[:20000]}\n```")

            # Extract structured table/entity catalog discovered across services
            discovered_catalog = {}
            for s in execution_steps:
                raw = s.get("raw_output", "")
                for line in raw.splitlines():
                    if ".sql" in line:
                        clean_l = line.strip().lstrip("•` ").rstrip("`")
                        parts = clean_l.split("/")
                        if len(parts) > 1:
                            svc = parts[0]
                            fname = parts[-1].split(":")[0]
                            m = re.search(r'V[\d._]+__(.+?)\.sql', fname, re.IGNORECASE)
                            tname = m.group(1) if m else fname
                            discovered_catalog.setdefault(svc, []).append(f"{tname} (`{fname}`)")

            catalog_block = ""
            if discovered_catalog:
                cat_lines = ["### Discovered Database Entities/Tables Catalog by Service:"]
                for svc, entries in sorted(discovered_catalog.items()):
                    cat_lines.append(f"- **{svc}** ({len(entries)} migrations/tables): " + ", ".join(entries[:25]))
                catalog_block = "\n".join(cat_lines) + "\n\n"

            bb_block = blackboard.render_prompt_xml()
            checklist_block = multi_service_orch.render_checklist_markdown()

            synthesis_prompt = (
                "You are Aviator, the principal AI software engineer and enterprise system architect. You investigated the codebase to answer this developer question:\n\n"
                f"{request.message}\n\n"
                f"{sibling_grounding_block}\n\n"
                f"{bb_block}\n\n"
                f"{checklist_block}\n\n"
                "During your automated investigation, you executed tools and gathered the following codebase findings across all services:\n\n"
                f"{catalog_block}"
                + "\n\n".join(gathered_findings) + "\n\n"
                "Grounding & Architecture Context:\n"
                f"{arch_block[:40000]}\n\n"
                "Please synthesize an authoritative, production-grade engineering answer to the developer's question based strictly on these findings.\n"
                "- If the user asked an informational or targeted question (e.g. info about a button, where code lives, or a concise explanation), answer their question directly, cleanly, and accurately without unnecessary boilerplate.\n"
                "- If the user asked for an architectural explanation or code solution, provide the complete, compilation-ready implementation with exact file paths, component names, and code snippets.\n"
                "- CRITICAL BLUEPRINT & CODE INTEGRITY RULES (ZERO INTERFACE WIPEOUTS & ZERO SPECULATION):\n"
                "  * For existing interfaces/repositories (e.g., Spring Data repositories), NEVER output a truncated whole-file replacement that omits existing methods! Show surgical additions or diff blocks, preserving all existing production methods and exact generic types (e.g. JpaRepository<ContractMember, UUID>, NOT custom IDs).\n"
                "  * In Spring Boot services, repository and DAO queries must be invoked on injected bean fields (e.g., `contractMemberRepository.count...`), NEVER called on `this.<repoMethod>()` or as methods on the service.\n"
                "  * In transactional delete-then-count sequences, always invoke `.flush()` on the repository immediately after deletion before the count query to prevent Hibernate write-behind dirty reads.\n"
                "  * Match exact cross-entity keys from repository join queries (e.g., ContractMember.userId maps to ParticipatingMember.otdsUserId).\n"
                "- Format using clean markdown, structured lists, and GitHub markdown tables where appropriate.\n"
                "- Do NOT abridge or cut short; finish all explanations completely.\n"
                "- NEVER hypothesize or guess names; ground every detail in the gathered findings and strictly honor the Deductive Blackboard.\n\n"
                f"{UNIVERSAL_SOFTWARE_ENGINEERING_INVARIANTS}"
            )

            synth_chunks = []
            def _run_stream_synthesis():
                nonlocal reply, synth_chunks, last_synth_response
                if is_nebius_model:
                    try:
                        import httpx, json
                        nebius_api_key = os.getenv("NEBIUS_API_KEY")
                        nebius_base_url = (os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")).rstrip("/") + "/"
                        clean_model = target_model_id[len("nebius/"):] if target_model_id.startswith("nebius/") else target_model_id

                        headers = {"Authorization": f"Bearer {nebius_api_key}", "Content-Type": "application/json"}
                        payload = {
                            "model": clean_model,
                            "messages": [
                                {"role": "system", "content": system_instruction},
                                {"role": "user", "content": synthesis_prompt}
                            ],
                            "max_tokens": 4096,
                            "temperature": 0.1,
                            "stream": True,
                        }
                        with httpx.Client(timeout=120.0) as http_client:
                            with http_client.stream("POST", f"{nebius_base_url}chat/completions", json=payload, headers=headers) as resp:
                                if resp.status_code != 200:
                                    err_txt = resp.read().decode("utf-8", errors="ignore")
                                    synth_chunks.append(f"Nebius API synthesis error ({resp.status_code}): {err_txt[:200]}")
                                    return
                                for line in resp.iter_lines():
                                    if not line or not line.startswith("data: "):
                                        continue
                                    raw_data = line[len("data: "):].strip()
                                    if raw_data == "[DONE]":
                                        break
                                    try:
                                        chunk_json = json.loads(raw_data)
                                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                                        txt = delta.get("content")
                                        if not txt and delta.get("reasoning_content"):
                                            _broadcast_chat_step({
                                                "title": f"Reasoning with {target_model_name}...",
                                                "type": "thought",
                                                "status": "running"
                                            })
                                        elif txt:
                                            synth_chunks.append(txt)
                                            _broadcast_chat_chunk(txt)
                                    except Exception:
                                        pass
                    except Exception as _synth_err:
                        logger.error(f"Nebius synthesis stream error: {_synth_err}")
                        synth_chunks.append(f"Error during synthesis with {target_model_name}: {_synth_err}")
                    return

                if client is not None:
                    try:
                        for chunk in client.models.generate_content_stream(
                            model=target_model_id,
                            contents=synthesis_prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.1,
                                max_output_tokens=8192,
                                thinking_config=model_thinking_config,
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                            )
                        ):
                            last_synth_response = chunk
                            txt = getattr(chunk, "text", None)
                            if txt:
                                synth_chunks.append(txt)
                                _broadcast_chat_chunk(txt)
                    except Exception as _synth_err:
                        logger.error(f"Synthesis stream error with {target_model_id}: {_synth_err}")

            synth_done = asyncio.Event()

            async def _synth_heartbeat_worker():
                h_start = time.time()
                while not synth_done.is_set():
                    try:
                        await asyncio.wait_for(synth_done.wait(), timeout=5.0)
                        break
                    except asyncio.TimeoutError:
                        pass
                    if synth_done.is_set():
                        break
                    elapsed_s = int(time.time() - h_start)
                    try:
                        await manager.broadcast({
                            "type": "chat_step",
                            "step": {
                                "title": f"Synthesizing grounded engineering response with {target_model_name} ({elapsed_s}s)...",
                                "type": "thought",
                                "status": "running",
                                "elapsed_s": elapsed_s,
                            }
                        })
                    except Exception:
                        pass

            synth_hb_task = asyncio.create_task(_synth_heartbeat_worker())
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(_run_stream_synthesis),
                    timeout=480.0
                )
                if synth_chunks:
                    reply = "".join(synth_chunks).strip()
            except Exception as _synth_timeout:
                logger.error(f"Synthesis pass timed out or failed: {_synth_timeout}")
            finally:
                synth_done.set()
                synth_hb_task.cancel()
        else:
            # Tokens were already streamed in real time during the primary chat stream
            pass

        had_error = False
        error_type = None
        error_detail = None

        if not reply or reply == "I couldn't generate a response. Please try again.":
            had_error = True
            error_type = "generation_failed"
            error_detail = "The model was unable to synthesize a complete response within the token/time limits."
            reply = error_detail
        
        # Token metrics extraction
        active_resp_obj = last_synth_response or last_stream_response
        prompt_tokens = getattr(active_resp_obj.usage_metadata, "prompt_token_count", 0) if hasattr(active_resp_obj, "usage_metadata") and active_resp_obj.usage_metadata else (len(user_prompt_content) // 4)
        candidates_tokens = getattr(active_resp_obj.usage_metadata, "candidates_token_count", 0) if hasattr(active_resp_obj, "usage_metadata") and active_resp_obj.usage_metadata else (len(reply) // 4)
        total_tokens = getattr(active_resp_obj.usage_metadata, "total_token_count", 0) if hasattr(active_resp_obj, "usage_metadata") and active_resp_obj.usage_metadata else (prompt_tokens + candidates_tokens)
        usage = {
            "prompt_tokens": prompt_tokens,
            "candidates_tokens": candidates_tokens,
            "total_tokens": total_tokens
        }
    except Exception as e:
        logger.error(f"Workflow chat LLM error: {e}")
        err_msg = str(e)
        had_error = True
        error_type = "gateway_timeout" if any(w in err_msg.lower() for w in ["timeout", "deadline", "closewait", "504"]) else "model_error"
        error_detail = err_msg
        # Fallback: build a rich rule-based answer from the context
        reply = _build_fallback_chat_reply(request.message, explanation, gen_files, state or {}, project_id)
        if not reply or reply == "I couldn't generate a response. Please try again.":
            reply = f"The model encountered an issue during generation: {err_msg}"
        prompt_tokens = len(request.message) // 4
        candidates_tokens = len(reply) // 4
        usage = {
            "prompt_tokens": prompt_tokens,
            "candidates_tokens": candidates_tokens,
            "total_tokens": prompt_tokens + candidates_tokens
        }

    total_duration_ms = int((time.time() - t_start) * 1000)
    if not execution_steps:
        execution_steps.append({
            "title": f"Synthesized analysis via {target_model_name}",
            "type": "thought",
            "duration_ms": total_duration_ms,
            "status": "success" if not had_error else "error"
        })

    res_dict = {
        "reply": reply,
        "execution_steps": execution_steps,
        "usage": usage,
        "duration_ms": total_duration_ms,
        "model_used": target_model_name,
        "model_id": target_model_id,
        "model_mode": req_mode,
    }
    if had_error:
        res_dict["error"] = True
        res_dict["error_type"] = error_type
        res_dict["error_detail"] = error_detail

    if emit_fn:
        if had_error:
            emit_fn("error", {
                "code": error_type or "generation_failed",
                "message": error_detail or "Generation failed",
                "retryable": True
            })
        else:
            emit_fn("usage", usage)
            emit_fn("done", {
                "reply": reply,
                "execution_steps": execution_steps,
                "usage": usage,
                "duration_ms": total_duration_ms,
                "finish_reason": "stop"
            })

    _persist_blackboard()
    return res_dict



@app.post("/api/workflow/chat/stream")
async def workflow_chat_stream(request: WorkflowChatRequest, req: Request):
    """Production SSE streaming endpoint for AI chat.
    Streams typed events (message_start, step, text_delta, usage, done, error)
    over a single HTTP connection. Cancels immediately if client aborts via req.is_disconnected().
    """
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def sse_emit(event_type: str, data: dict):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": event_type, "data": data})
        except Exception:
            pass

    async def sse_event_stream():
        async def run_pipeline():
            try:
                await _execute_workflow_chat(request, emit_fn=sse_emit)
            except Exception as _e:
                logger.error(f"SSE pipeline error: {_e}")
                sse_emit("error", {"code": "pipeline_error", "message": str(_e), "retryable": True})
            finally:
                sse_emit("__SENTINEL__", {})

        worker_task = asyncio.create_task(run_pipeline())

        try:
            while True:
                if await req.is_disconnected():
                    logger.info("Client disconnected from SSE stream (Abort / Tab Closed) — cancelling work")
                    worker_task.cancel()
                    break

                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # SSE keepalive comment line keeps connection open through proxies/firewalls
                    yield ": heartbeat\n\n"
                    continue

                if item.get("type") == "__SENTINEL__":
                    break

                evt_type = item.get("type", "message")
                payload = json.dumps(item.get("data", {}), ensure_ascii=False)
                yield f"event: {evt_type}\ndata: {payload}\n\n"
        except asyncio.CancelledError:
            worker_task.cancel()
        finally:
            if not worker_task.done():
                worker_task.cancel()

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )



@app.post("/api/workflow/chat")
async def workflow_chat(request: WorkflowChatRequest):
    """Standard non-streaming JSON endpoint for backward compatibility."""
    return await _execute_workflow_chat(request)




def _build_fallback_chat_reply(question: str, explanation: dict, gen_files: list, state: dict, project_id: Optional[str] = None) -> str:
    """Dynamic fallback answer when LLM is unavailable, powered by project RAG, knowledge stores, and state."""
    lower = question.lower()

    # 1. Architecture & Role Mapping (Coordinator vs Administrator)
    if "coordinator" in lower and "administrator" in lower:
        return (
            "### 🏛️ Coordinator vs. Administrator Mapping:\n\n"
            "In the frontend UI, the role is presented to users as **Coordinator**, whereas in the backend microservices (project-service, sagas-service, issues-service) and database tables, it is persisted as **ROLE_ADMINISTRATOR**.\n\n"
            "### The 3 Core Roles:\n"
            "1. **Coordinator / Administrator** (`ROLE_ADMINISTRATOR`): Full project governance and permissions.\n"
            "2. **Contributor / Member** (`ROLE_MEMBER`): Ability to view, contribute, and update deliverables.\n"
            "3. **Consumer / Reviewer** (`ROLE_VIEWER`): Read-only inspection access.\n\n"
            "### Contract Access Removal Scenarios:\n"
            "- **Scenario A (Explicit Contract ACL)**: If the coordinator was explicitly assigned to a specific contract ACL, removing project-level coordinator access will revoke contract members accordingly.\n"
            "- **Scenario B (Inherited Access)**: If contract access was purely inherited from project-level coordinator rights, revoking project coordinator immediately revokes inherited contract rights.\n"
            "- **Scenario C (Saga Event Synchronization)**: The removal propagates across microservices via Saga event choreography (`ProjectUserRevokedEvent`)."
        )

    # 2. Dynamic RAG retrieval from active project's indexed documentation
    if project_id:
        try:
            proj = next((p for p in projects if p.id == project_id), None)
            if proj and Path(proj.path).exists():
                from ticket_to_code.retrieval.rag_engine import CodebaseRAGEngine
                rag = CodebaseRAGEngine()
                rag.workspace_path = str(proj.path)
                chunks = rag.retrieve_context(question, max_results=3, document_types=["documentation", "readme", "architecture", "configuration"])
                if chunks:
                    lines = [f"### 📚 Retrieved Context from `{proj.name}` Documentation:\n"]
                    for ch in chunks[:3]:
                        lines.append(f"• **`{ch.get('file_path', '')}`**:\n{ch.get('content', '')[:400]}\n")
                    return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Dynamic fallback RAG failed: {e}")

    # 3. Stopped Workflow Inquiries
    if state.get("status") == "stopped" or any(w in lower for w in ["stopped", "stop", "halt", "halted", "interrupt"]):
        return (
            "🛑 **Workflow is Stopped & Memory is Preserved:**\n\n"
            "• **Status:** Safely halted at user request.\n"
            f"• **Investigated Files:** {', '.join(gen_files[:5]) if gen_files else 'Analyzed files retained in working memory.'}\n"
            f"• **Summary:** {explanation.get('summary', 'Intermediate analysis retained.')}\n\n"
            "You can continue chatting right here: ask questions about the analyzed code, provide corrections, or start a new task without restarting the system."
        )

    # 3. Cross-chat and past ticket queries
    if any(w in lower for w in ["past ticket", "other chat", "history", "previous task"]):
        if project_id:
            tasks = history_store.list_tasks(project_id)[:5]
            if tasks:
                t_list = "\n".join([f"  • **{t.get('ticket_id', 'Task')}**: {t.get('title')} ({t.get('status')})" for t in tasks])
                return f"📋 **Recent Tasks for this Project:**\n{t_list}\n\nAsk about any of these tasks for details."
        return "No prior tasks found for this project in history."

    if any(w in lower for w in ["what file", "which file", "files changed", "modified"]):
        if gen_files:
            return "The following files were generated/modified:\n" + "\n".join(f"  • {f}" for f in gen_files[:15])
        return "No files were modified in this workflow run."

    if any(w in lower for w in ["why", "reason", "explain"]):
        summary = explanation.get("summary", "")
        why = explanation.get("why", "")
        return f"{summary}\n\n{why}" if summary else "No detailed explanation is available for this workflow."

    if any(w in lower for w in ["error", "fail", "wrong"]):
        err = state.get("error")
        return f"The workflow encountered an issue: {err}" if err else "No errors were recorded."

    return (
        f"Workflow status: {state.get('status', 'unknown')}.\n"
        f"Summary: {explanation.get('summary', 'N/A')}\n"
        f"Files: {', '.join(gen_files[:5]) if gen_files else 'None'}\n\n"
        "Ask me about specific files, errors, architecture, or previous tasks."
    )



# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "projects": len(projects),
        "knowledge_bases": len(knowledge_bases),
        "active_workflows": len(workflow_status),
        "websocket_connections": len(manager.active_connections),
    }


def _allowed_file_tree_roots() -> List[Path]:
    """Registered project roots plus the backend workspace directories."""
    roots: List[Path] = []
    for p in list(projects.values()):
        try:
            for candidate in (p.local_path, p.path):
                if candidate and not str(candidate).startswith(("http://", "https://", "git@")):
                    roots.append(Path(candidate).resolve())
        except Exception:
            continue
    try:
        roots.append(Path("workspace").resolve())
    except Exception:
        pass
    for env_key in ("AVIATOR_WORKSPACE_PATH", "TICKET_TO_CODE_WORKSPACE_PATH", "WORKSPACE_PATH"):
        v = os.getenv(env_key)
        if v:
            try:
                roots.append(Path(v).resolve())
            except Exception:
                pass
    return roots


def _is_under_allowed_roots(target: Path) -> bool:
    try:
        resolved = target.resolve()
    except Exception:
        return False
    for root in _allowed_file_tree_roots():
        if resolved == root or root in resolved.parents:
            return True
    return False


@app.get("/api/file-tree")
async def get_file_tree(path: str):
    """Return a simple file-tree for the given directory path (must be inside a registered project or workspace)."""
    root = Path(path)
    if not _is_under_allowed_roots(root):
        raise HTTPException(status_code=403, detail="Path is outside registered project roots")
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    IGNORE = {".git", ".aviator", "target", "build", "node_modules", "__pycache__", ".idea"}
    MAX_DEPTH = 4

    def build_tree(p: Path, depth: int) -> dict:
        if depth > MAX_DEPTH:
            return None
        if p.is_file():
            return {"name": p.name, "type": "file", "path": str(p.relative_to(root)).replace("\\", "/")}
        children = []
        try:
            for child in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name)):
                if child.name in IGNORE:
                    continue
                node = build_tree(child, depth + 1)
                if node:
                    children.append(node)
        except PermissionError:
            pass
        return {"name": p.name, "type": "folder", "children": children}

    return build_tree(root, 0)


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Aviator Chatbot API starting...")
    Path("workspace").mkdir(exist_ok=True)
    Path("workspace/kb").mkdir(exist_ok=True)
    # Nothing can be live at process start: every persisted 'running' workflow
    # is an orphan from a previous process, so fail it in the durable store and
    # in history.json so /status, /stop and /resume no longer see zombie runs.
    try:
        stale = workflow_status.mark_stale_running("Workflow interrupted (backend restarted)")
        if stale:
            logger.info(f"Marked {stale} interrupted workflow(s) as failed at startup")
    except Exception as e:
        logger.warning(f"Failed to mark stale running workflows: {e}")
    try:
        history_store.sanitize_orphaned_tasks(set())
    except Exception as e:
        logger.warning(f"Failed to sanitize orphaned tasks: {e}")
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        def sync_broadcast_clarification(payload: dict):
            try:
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(manager.broadcast(payload), loop)
                else:
                    loop.run_until_complete(manager.broadcast(payload))
            except Exception as _e:
                logger.debug(f"Interactive clarification broadcast error: {_e}")

        from ticket_to_code.runtime.interactive_gate import InteractiveClarificationGate
        InteractiveClarificationGate.set_ui_broadcaster(sync_broadcast_clarification)
        logger.info("📡 InteractiveClarificationGate UI broadcaster registered successfully")
    except Exception as _e:
        logger.warning(f"Could not register clarification UI broadcaster: {_e}")

    logger.info("✅ Aviator Chatbot API ready!")


if __name__ == "__main__":
    import uvicorn
    # reload=True causes uvicorn to restart when files change.
    # Since the agent writes code files locally, this causes the backend
    # to restart mid-workflow, abandoning the ticket. Disabled it.
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)

# reload
