"""
History Store — durable chat & task history for the Aviator chatbot.

Mirrors the lightweight JSON persistence used for projects.json so previous
conversations and completed tasks survive server restarts, exactly like the
chat/task history in Copilot / Cursor / other AI IDEs.

Two record kinds:
  • chats  — a conversation: id, project_id, title, messages[], timestamps
  • tasks  — a completed workflow run: workflow_id, project_id, ticket, status,
             changed_files, execution_mode, timestamps

Thread-safe (RLock) with atomic writes. No external dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HISTORY_FILE = Path(__file__).parent / "history.json"
_lock = threading.RLock()


def _now() -> str:
    return datetime.now().isoformat()


def _empty() -> Dict[str, Any]:
    return {"chats": {}, "tasks": {}}


def _load() -> Dict[str, Any]:
    if _HISTORY_FILE.exists():
        try:
            data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            data.setdefault("chats", {})
            data.setdefault("tasks", {})
            # Sanitize any lingering "Thinking..." messages from stored chats
            for chat in data["chats"].values():
                msgs = chat.get("messages", [])
                cleaned = []
                for idx, m in enumerate(msgs):
                    txt = str(m.get("content", "")).strip()
                    if txt in ("🤔 Thinking...", "Thinking...") or m.get("isThinking"):
                        prev_user = next((x for x in reversed(msgs[:idx]) if x.get("role") == "user"), None)
                        cleaned.append({
                            "role": "assistant",
                            "isError": True,
                            "errorType": "timeout",
                            "errorTitle": "⏱️ Gateway Response Timeout (Interrupted)",
                            "errorMessage": "The model response was interrupted or timed out before completion. Click Retry below to resume generation.",
                            "failedMessage": prev_user.get("content", "") if prev_user else "",
                            "modelUsed": m.get("modelUsed") or "gemini-2.5-pro",
                            "timestamp": m.get("timestamp") or _now(),
                        })
                    else:
                        cleaned.append(m)
                chat["messages"] = cleaned
            return data
        except Exception as exc:
            logger.warning(f"Could not load history.json: {exc}")
    return _empty()


def _save(data: Dict[str, Any]) -> None:
    tmp = _HISTORY_FILE.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, _HISTORY_FILE)
    except Exception as exc:
        logger.warning(f"Could not save history.json: {exc}")


def _derive_title(messages: List[Dict[str, Any]], fallback: str = "New chat") -> str:
    for m in messages:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            text = m["content"].strip().replace("\n", " ")
            return (text[:60] + "…") if len(text) > 60 else text
    return fallback


# ── Chats ─────────────────────────────────────────────────────────────────────

def list_chats(project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return chat summaries (no message bodies), newest first."""
    with _lock:
        data = _load()
        out = []
        for chat in data["chats"].values():
            if project_id and chat.get("project_id") != project_id:
                continue
            out.append({
                "id": chat["id"],
                "project_id": chat.get("project_id"),
                "title": chat.get("title") or "New chat",
                "workflow_id": chat.get("workflow_id"),
                "created_at": chat.get("created_at"),
                "updated_at": chat.get("updated_at"),
                "message_count": len(chat.get("messages", [])),
            })
        out.sort(key=lambda c: c.get("updated_at") or "", reverse=True)
        return out


def get_chat(chat_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _load()["chats"].get(chat_id)


def upsert_chat(
    chat_id: Optional[str],
    project_id: Optional[str],
    messages: List[Dict[str, Any]],
    title: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a chat. Returns the stored record."""
    # Filter out in-flight thinking placeholders before saving to disk
    sanitized_messages = []
    for idx, m in enumerate(messages):
        txt = str(m.get("content", "")).strip()
        if txt in ("🤔 Thinking...", "Thinking...") or m.get("isThinking"):
            continue
        sanitized_messages.append(m)
    messages = sanitized_messages

    with _lock:
        data = _load()
        chats = data["chats"]
        now = _now()
        if not chat_id or chat_id not in chats:
            chat_id = chat_id or f"chat_{uuid.uuid4().hex[:12]}"
            record = {
                "id": chat_id,
                "project_id": project_id,
                "title": title or _derive_title(messages),
                "messages": messages,
                "workflow_id": workflow_id,
                "created_at": now,
                "updated_at": now,
            }
        else:
            record = chats[chat_id]
            record["messages"] = messages
            record["project_id"] = project_id or record.get("project_id")
            record["title"] = title or record.get("title") or _derive_title(messages)
            record["updated_at"] = now
            if workflow_id:
                record["workflow_id"] = workflow_id
        chats[chat_id] = record
        _save(data)
        return record


def delete_chat(chat_id: str) -> bool:
    with _lock:
        data = _load()
        if chat_id in data["chats"]:
            del data["chats"][chat_id]
            _save(data)
            return True
        return False


# ── Tasks (completed workflow runs) ────────────────────────────────────────────

def record_task(
    workflow_id: str,
    project_id: Optional[str],
    ticket_id: str,
    title: str,
    description: str,
    status: str,
    changed_files: List[str],
    execution_mode: str = "pipeline",
    error: Optional[str] = None,
    token_usage: Optional[Dict[str, Any]] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Persist a completed task/run so it shows up in history."""
    with _lock:
        data = _load()
        existing = data["tasks"].get(workflow_id, {})
        record = {
            "workflow_id": workflow_id,
            "project_id": project_id,
            "ticket_id": ticket_id,
            "title": (title or description or "Task")[:100],
            "description": description,
            "status": status,
            "changed_files": changed_files or [],
            "execution_mode": execution_mode,
            "error": error,
            "token_usage": token_usage if token_usage is not None else existing.get("token_usage"),
            "steps": steps if steps is not None else existing.get("steps", []),
            "created_at": existing.get("created_at") or _now(),
            "completed_at": _now() if status in ("completed", "failed", "stopped") else existing.get("completed_at"),
        }
        data["tasks"][workflow_id] = record
        _save(data)
        return record


def get_task(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single task record by workflow_id."""
    with _lock:
        return _load()["tasks"].get(workflow_id)


def list_tasks(project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with _lock:
        data = _load()
        out = [
            t for t in data["tasks"].values()
            if not project_id or t.get("project_id") == project_id
        ]
        out.sort(key=lambda t: t.get("created_at") or "", reverse=True)
        return out


def delete_task(workflow_id: str) -> bool:
    with _lock:
        data = _load()
        if workflow_id in data["tasks"]:
            del data["tasks"][workflow_id]
            _save(data)
            return True
        return False


def sanitize_orphaned_tasks(active_workflow_ids: Optional[set] = None) -> None:
    """
    Reconcile tasks stuck in 'running' state after a server restart or crash.
    Any task in history.json with status == 'running' that is not actively in
    active_workflow_ids is automatically marked as 'failed'.
    """
    with _lock:
        data = _load()
        changed = False
        active_ids = active_workflow_ids or set()
        for task in data.get("tasks", {}).values():
            if task.get("status") == "running" and task.get("workflow_id") not in active_ids:
                task["status"] = "failed"
                task["error"] = task.get("error") or "Workflow interrupted (process ended or restarted)"
                task["completed_at"] = task.get("completed_at") or _now()
                changed = True
        if changed:
            _save(data)

