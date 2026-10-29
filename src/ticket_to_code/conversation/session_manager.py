from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from ticket_to_code.agents.knowledge_extractor import KnowledgeExtractor
from ticket_to_code.models import (
    ClassifiedType,
    ConversationContext,
    ConversationMessage,
    ConversationState,
    MessageRole,
    TechnicalFacts,
    TicketPriority,
    ValueEdgeTicket,
)


class SessionStage(str, Enum):
    INTAKE = "intake"
    FACT_EXTRACTION = "fact_extraction"
    HYPOTHESIS = "hypothesis"
    READY_TO_RUN = "ready_to_run"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class ReasoningSnapshot(BaseModel):
    stage: SessionStage
    confidence: float = Field(ge=0.0, le=1.0)
    primary_hypothesis: str
    secondary_hypotheses: List[str] = Field(default_factory=list)
    elimination_rules: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)


class TicketConversationSession(BaseModel):
    session_id: str
    workspace_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    stage: SessionStage = SessionStage.INTAKE
    conversation: ConversationState = Field(default_factory=ConversationState)

    technical_facts: Optional[TechnicalFacts] = None
    classified_type: Optional[ClassifiedType] = None

    primary_hypothesis: str = "Gathering context from user conversation."
    secondary_hypotheses: List[str] = Field(default_factory=list)
    elimination_rules: List[str] = Field(default_factory=list)
    planned_actions: List[str] = Field(default_factory=list)

    synthesized_ticket: Optional[ValueEdgeTicket] = None
    last_workflow_status: Optional[str] = None
    last_workflow_errors: List[str] = Field(default_factory=list)


class ConversationSessionManager:
    """Interactive session manager that turns chat into a transparent ticket pipeline."""

    def __init__(self) -> None:
        self._sessions: Dict[str, TicketConversationSession] = {}
        self._extractor = KnowledgeExtractor()
        self._lock = RLock()

    def create_session(self, workspace_path: str, goal: str = "", initial_message: str = "") -> TicketConversationSession:
        with self._lock:
            session_id = f"T2C-{uuid4().hex[:10]}"
            session = TicketConversationSession(session_id=session_id, workspace_path=workspace_path)
            if goal:
                session.conversation.context.current_goal = goal.strip()
            if initial_message:
                self._append_message(session, MessageRole.USER, initial_message)
            self._refresh_intelligence(session)
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> TicketConversationSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session: {session_id}")
            return self._sessions[session_id]

    def add_user_message(self, session_id: str, text: str, images: Optional[List[str]] = None, files: Optional[List[str]] = None) -> TicketConversationSession:
        with self._lock:
            session = self.get_session(session_id)
            self._append_message(session, MessageRole.USER, text, images=images, files=files)
            self._refresh_intelligence(session)
            return session

    def apply_feedback(self, session_id: str, feedback: str) -> TicketConversationSession:
        with self._lock:
            session = self.get_session(session_id)
            self._append_message(session, MessageRole.USER, f"Feedback: {feedback}")
            session.conversation.context.rejected_decisions.append(feedback.strip())
            session.stage = SessionStage.FACT_EXTRACTION
            self._refresh_intelligence(session)
            return session

    def mark_running(self, session_id: str) -> None:
        with self._lock:
            session = self.get_session(session_id)
            session.stage = SessionStage.RUNNING
            session.updated_at = datetime.utcnow()

    def mark_stopped(self, session_id: str, reason: str = "Stopped by user") -> TicketConversationSession:
        with self._lock:
            session = self.get_session(session_id)
            session.stage = SessionStage.STOPPED
            session.last_workflow_status = "stopped"
            session.conversation.context.rejected_decisions.append(f"Pipeline stopped: {reason}")
            self._append_message(session, MessageRole.ASSISTANT, f"🛑 Workflow stopped by user ({reason}). Discovered context and working memory are retained.")
            session.updated_at = datetime.utcnow()
            return session

    def attach_workflow_result(self, session_id: str, status: str, errors: Optional[List[str]] = None) -> TicketConversationSession:
        with self._lock:
            session = self.get_session(session_id)
            session.last_workflow_status = status
            session.last_workflow_errors = errors or []
            session.stage = SessionStage.COMPLETED if status == "success" else SessionStage.FAILED
            session.updated_at = datetime.utcnow()
            return session

    def synthesize_ticket(self, session_id: str, priority: TicketPriority = TicketPriority.MEDIUM) -> ValueEdgeTicket:
        with self._lock:
            session = self.get_session(session_id)
            transcript = self._conversation_transcript(session)
            goal = session.conversation.context.current_goal.strip() or "Interactive ticket from conversation"

            acceptance_criteria = list(session.conversation.context.extracted_requirements)
            if not acceptance_criteria:
                acceptance_criteria = [
                    "Root cause is identified with evidence",
                    "Changes are limited to relevant files only",
                    "Build/test verification is provided",
                ]

            description = (
                f"Goal: {goal}\n\n"
                f"Primary hypothesis: {session.primary_hypothesis}\n"
                f"Secondary hypotheses: {session.secondary_hypotheses}\n"
                f"Elimination rules: {session.elimination_rules}\n\n"
                f"Conversation transcript:\n{transcript}\n"
            )

            ticket = ValueEdgeTicket(
                ticket_id=f"{session.session_id}-TICKET",
                title=goal[:120],
                description=description,
                acceptance_criteria=acceptance_criteria,
                priority=priority,
                labels=["interactive", "chat-context", "ticket-to-code"],
            )
            session.synthesized_ticket = ticket
            session.conversation.synthesized_ticket = ticket
            session.stage = SessionStage.READY_TO_RUN
            session.updated_at = datetime.utcnow()
            return ticket

    def reasoning_snapshot(self, session_id: str) -> ReasoningSnapshot:
        with self._lock:
            session = self.get_session(session_id)
            return ReasoningSnapshot(
                stage=session.stage,
                confidence=self._compute_confidence(session),
                primary_hypothesis=session.primary_hypothesis,
                secondary_hypotheses=session.secondary_hypotheses,
                elimination_rules=session.elimination_rules,
                next_actions=session.planned_actions,
                open_questions=session.conversation.context.open_questions,
            )

    def _append_message(
        self,
        session: TicketConversationSession,
        role: MessageRole,
        text: str,
        images: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
    ) -> None:
        message = ConversationMessage(
            role=role,
            text=text,
            images=images or [],
            files=files or [],
        )
        session.conversation.messages.append(message)
        session.updated_at = datetime.utcnow()

    def _conversation_transcript(self, session: TicketConversationSession) -> str:
        lines: List[str] = []
        for message in session.conversation.messages[-30:]:
            lines.append(f"[{message.role.value}] {message.text}")
        return "\n".join(lines)

    def _refresh_intelligence(self, session: TicketConversationSession) -> None:
        transcript = self._conversation_transcript(session)
        context = session.conversation.context

        if not context.current_goal and transcript:
            context.current_goal = transcript.split("\n", 1)[0][:160]

        requirements = self._extract_requirements(transcript)
        if requirements:
            context.extracted_requirements = requirements

        if "?" in transcript:
            context.open_questions = [line for line in transcript.splitlines() if "?" in line][-5:]
        else:
            context.open_questions = []

        synthetic_ticket = ValueEdgeTicket(
            ticket_id=f"{session.session_id}-CTX",
            title=context.current_goal or "conversation-context",
            description=transcript or "No context yet",
            acceptance_criteria=context.extracted_requirements,
            priority=TicketPriority.MEDIUM,
            labels=["context-extraction"],
        )
        facts = self._extractor.extract(synthetic_ticket)
        classified = ClassifiedType(
            primary=facts.classified_type,
            confidence=0.9 if facts.has_facts() else 0.4,
            signals=[
                f"versions={len(facts.versions)}",
                f"files={len(facts.files)}",
                f"literals={len(facts.high_priority_literals)}",
            ],
        )

        session.technical_facts = facts
        session.classified_type = classified

        session.primary_hypothesis = self._primary_hypothesis(facts)
        session.secondary_hypotheses = self._secondary_hypotheses(facts)
        session.elimination_rules = self._elimination_rules(classified, facts)
        session.planned_actions = self._next_actions(facts)

        if facts.has_facts():
            session.stage = SessionStage.HYPOTHESIS
        elif session.conversation.messages:
            session.stage = SessionStage.FACT_EXTRACTION
        else:
            session.stage = SessionStage.INTAKE

    def _compute_confidence(self, session: TicketConversationSession) -> float:
        facts = session.technical_facts
        if not facts:
            return 0.2
        score = 0.25
        score += min(0.25, len(facts.files) * 0.08)
        score += min(0.2, len(facts.versions) * 0.12)
        score += min(0.2, len(facts.error_codes) * 0.08)
        score += min(0.1, len(facts.high_priority_literals) * 0.01)
        score += 0.1 if facts.has_facts() else 0.0
        return round(min(0.95, score), 2)

    def _extract_requirements(self, transcript: str) -> List[str]:
        requirements: List[str] = []
        for line in transcript.splitlines():
            clean = line.strip()
            if re.match(r"^[-*]\s+", clean):
                requirements.append(re.sub(r"^[-*]\s+", "", clean))
            elif re.match(r"^\d+[.)]\s+", clean):
                requirements.append(re.sub(r"^\d+[.)]\s+", "", clean))
        return requirements[-15:]

    def _primary_hypothesis(self, facts: TechnicalFacts) -> str:
        if facts.versions:
            return "Version transition is incomplete or inconsistent across code/config surfaces."
        if facts.error_codes:
            return "A deterministic failure path is likely triggered by specific validation or runtime checks."
        if facts.files:
            return "The issue is likely localized in explicitly referenced files and their shared dependencies."
        return "More domain context is needed to localize a reliable primary owner file."

    def _secondary_hypotheses(self, facts: TechnicalFacts) -> List[str]:
        hypotheses: List[str] = []
        if facts.paths:
            hypotheses.append("Routing or module boundary mismatch causes state divergence.")
        if facts.config_keys:
            hypotheses.append("Configuration defaults or env-specific overrides conflict with expected behavior.")
        if facts.classes or facts.methods:
            hypotheses.append("Service-level state or method-side effects are leaking across flows.")
        if not hypotheses:
            hypotheses.append("UI expectation and backend state transitions may be out of sync.")
        return hypotheses[:3]

    def _elimination_rules(self, classified: ClassifiedType, facts: TechnicalFacts) -> List[str]:
        rules = [
            "Reject files with no lexical, symbol, or dependency evidence.",
            "Prefer minimal blast-radius edits in primary-owner files first.",
            "Treat generated/minified/vendor files as context-only unless explicitly referenced.",
        ]
        if classified.primary == "UI":
            rules.append("Penalize backend and migration files unless ticket includes API/DB evidence.")
        if classified.primary in {"DATABASE", "SECURITY"}:
            rules.append("Do not allow UI-only fixes to satisfy server-side acceptance criteria.")
        if facts.versions:
            rules.append("Search and update current-state literals before touching unrelated logic.")
        return rules

    def _next_actions(self, facts: TechnicalFacts) -> List[str]:
        actions = [
            "Confirm ticket acceptance criteria from conversation context.",
            "Run repository discovery with extracted literals and symbols.",
            "Rank candidate files and select primary owner before planning.",
        ]
        if facts.files:
            actions.insert(1, "Open explicitly referenced files and verify real code ownership.")
        if facts.error_codes:
            actions.append("Map error code path to controller/service/validator chain.")
        return actions
