"""
Interactive Clarification & Human-in-the-Loop (HITL) Protocol.

Enables the agent to solicit structured human guidance when facing:
- Ambiguous architectural requirements (e.g. soft vs hard delete).
- High-consequence or irreversible file mutations.
- Unresolved compiler/test conflicts requiring human domain expertise.

Provides structured multi-choice options with a recommended default,
timeout fallback for non-interactive CI/CD runs, and blackboard persistence.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClarificationOption:
    label: str
    description: str = ""
    is_recommended: bool = False


@dataclass
class ClarificationRequest:
    request_id: str
    question: str
    options: List[ClarificationOption]
    is_multi_select: bool = False
    timeout_seconds: float = 120.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ClarificationResponse:
    request_id: str
    selected_options: List[str]
    user_notes: str = ""
    responded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    was_fallback: bool = False


class InteractiveClarificationGate:
    """Manages structured user clarification requests across Chat and Flow engines."""

    _active_requests: Dict[str, ClarificationRequest] = {}
    _pending_events: Dict[str, threading.Event] = {}
    _completed_responses: Dict[str, ClarificationResponse] = {}
    _ui_broadcast_fn: Optional[Callable[[Dict[str, Any]], None]] = None

    @classmethod
    def set_ui_broadcaster(cls, fn: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Sets the thread-safe broadcast function to push clarification modals to the UI."""
        cls._ui_broadcast_fn = fn

    @classmethod
    def request_clarification(
        cls,
        question: str,
        options: List[str],
        recommended_option: Optional[str] = None,
        descriptions: Optional[List[str]] = None,
        is_multi_select: bool = False,
        timeout_seconds: float = 60.0,
        blackboard: Optional[Any] = None,
    ) -> str:
        """Solicits structured multiple-choice guidance from the user.

        If a UI listener is connected, blocks on a threading.Event until response or timeout.
        If timeout expires or running in automated mode, automatically picks the recommended option.
        """
        req_id = f"clarif_{uuid.uuid4().hex[:8]}"
        desc_list = descriptions or [""] * len(options)

        rec_opt = recommended_option or (options[0] if options else "Proceed")
        parsed_options = [
            ClarificationOption(
                label=opt,
                description=desc_list[i] if i < len(desc_list) else "",
                is_recommended=(opt == rec_opt),
            )
            for i, opt in enumerate(options)
        ]

        req = ClarificationRequest(
            request_id=req_id,
            question=question,
            options=parsed_options,
            is_multi_select=is_multi_select,
            timeout_seconds=timeout_seconds,
        )

        cls._active_requests[req_id] = req
        evt = threading.Event()
        cls._pending_events[req_id] = evt

        # Broadcast modal request to UI if broadcaster is available
        if cls._ui_broadcast_fn:
            payload = {
                "event_type": "clarification_requested",
                "request_id": req_id,
                "question": question,
                "options": [
                    {"label": o.label, "description": o.description, "is_recommended": o.is_recommended}
                    for o in parsed_options
                ],
                "recommended": rec_opt,
                "is_multi_select": is_multi_select,
                "timeout_seconds": timeout_seconds,
                "timestamp": datetime.now().isoformat(),
            }
            try:
                cls._ui_broadcast_fn(payload)
            except Exception as e:
                logger.warning(f"Failed to broadcast clarification modal: {e}")

        effective_timeout = timeout_seconds if cls._ui_broadcast_fn else 0.05
        # Wait for user response
        logger.info(f"⏳ Waiting up to {effective_timeout}s for user clarification on '{question}' (ID: {req_id})...")
        answered = evt.wait(timeout=effective_timeout)

        if answered and req_id in cls._completed_responses:
            resp = cls._completed_responses[req_id]
            selection_str = ", ".join(resp.selected_options)
            result_msg = f"User selected: [{selection_str}]. User note: '{resp.user_notes}'"
            logger.info(f"✅ User clarification received for {req_id}: {result_msg}")
        else:
            # Fallback to recommended option
            resp = ClarificationResponse(
                request_id=req_id,
                selected_options=[rec_opt],
                user_notes="Auto-fallback: Selected recommended default after timeout / non-interactive run.",
                was_fallback=True,
            )
            cls._completed_responses[req_id] = resp
            result_msg = f"[AUTO_FALLBACK] User response timed out. Proceeded with recommended option: '{rec_opt}'."
            logger.info(f"⚠️ {result_msg}")

        # Clean up
        cls._active_requests.pop(req_id, None)
        cls._pending_events.pop(req_id, None)

        # Record decision in DeductiveBlackboard as an immutable fact
        if blackboard and hasattr(blackboard, "record_fact"):
            blackboard.record_fact(
                f"Interactive decision on '{question}': {', '.join(resp.selected_options)} (fallback={resp.was_fallback})"
            )

        return result_msg

    @classmethod
    def submit_response(
        cls,
        request_id: str,
        selected_options: List[str],
        user_notes: str = "",
    ) -> bool:
        """Submits the human user's decision from the UI REST or WebSocket endpoint."""
        if request_id not in cls._pending_events:
            return False

        resp = ClarificationResponse(
            request_id=request_id,
            selected_options=selected_options,
            user_notes=user_notes,
            was_fallback=False,
        )
        cls._completed_responses[request_id] = resp
        evt = cls._pending_events[request_id]
        evt.set()
        return True
