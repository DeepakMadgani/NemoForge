"""
Chat Interaction Manager — Interactive Feedback Loop like Copilot/Claude

Maintains conversation context, asks clarifying questions, and provides
interactive feedback to user during code generation.

Key insight: Store conversation state so agent can ask:
  - "Should I create this class or modify existing one?"
  - "I found 3 properties, add all of them?"
  - "This property type is ambiguous, boolean or Observable<boolean>?"
  - "I made changes to 5 files, should I proceed?"

Author: Deepak Madgani
Date: August 2026
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PromptType(str, Enum):
    """Type of question to ask user."""
    CONFIRM_ACTION = "confirm_action"  # Yes/No question
    CHOOSE_OPTION = "choose_option"  # Multiple choice
    CLARIFY_AMBIGUITY = "clarify_ambiguity"  # Resolve ambiguous situation
    REPORT_PROGRESS = "report_progress"  # Informational
    REQUEST_FEEDBACK = "request_feedback"  # Open-ended feedback


@dataclass
class ConversationMessage:
    """Single message in conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: Optional[str] = None  # e.g., "error", "success", "question"
    requires_response: bool = False
    response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "requires_response": self.requires_response,
            "response": self.response,
            "metadata": self.metadata,
        }


@dataclass
class InteractivePrompt:
    """A prompt that requires user interaction."""
    prompt_id: str
    prompt_type: PromptType
    question: str
    options: Optional[List[str]] = None  # For choose_option
    default_option: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)  # Why is this being asked?
    requires_response: bool = True
    user_response: Optional[str] = None
    response_timestamp: Optional[datetime] = None

    def to_dict(self):
        return {
            "prompt_id": self.prompt_id,
            "prompt_type": self.prompt_type.value,
            "question": self.question,
            "options": self.options,
            "default_option": self.default_option,
            "context": self.context,
            "requires_response": self.requires_response,
            "user_response": self.user_response,
            "response_timestamp": self.response_timestamp.isoformat() if self.response_timestamp else None,
        }


class ChatInteractionManager:
    """
    Manages interactive chat with user during code generation.

    Maintains conversation history, asks clarifying questions, and provides
    real-time feedback like Copilot or Claude Code.

    Usage:
        chat = ChatInteractionManager(ticket_id="TICKET-123")
        
        # Send message to user
        chat.add_message(
            role=MessageRole.ASSISTANT,
            content="I'm about to create 3 new properties. Proceed?",
            message_type="question",
        )
        
        # Ask for confirmation
        prompt = chat.ask_confirmation(
            question="Add property isUserProjectMember?",
            context={"file": "add-members.component.ts", "type": "boolean"},
        )
        
        # Wait for response
        user_response = prompt.user_response  # Will be set when user responds
        
        # Log interaction
        chat.log_to_file("/path/to/conversation.json")
    """

    def __init__(
        self,
        ticket_id: str,
        storage_path: Optional[str] = None,
    ):
        self.ticket_id = ticket_id
        self.storage_path = Path(storage_path) if storage_path else None
        self.conversation_history: List[ConversationMessage] = []
        self.pending_prompts: List[InteractivePrompt] = []
        self.answered_prompts: List[InteractivePrompt] = []
        self.session_start = datetime.now()

    def add_message(
        self,
        role: MessageRole,
        content: str,
        message_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> ConversationMessage:
        """Add message to conversation history."""
        message = ConversationMessage(
            role=role,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )
        self.conversation_history.append(message)

        logger.debug(f"[{role.value}] {content}")

        return message

    def ask_confirmation(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        auto_yes_if_unknown: bool = True,
    ) -> InteractivePrompt:
        """
        Ask user for yes/no confirmation.

        Example:
            prompt = chat.ask_confirmation(
                question="Add property isUserProjectMember?",
                context={"file": "add-members.component.ts", "type": "boolean"},
            )
            # Return prompt object; when user responds, prompt.user_response is set
        """
        prompt_id = f"confirm_{len(self.pending_prompts)}"

        prompt = InteractivePrompt(
            prompt_id=prompt_id,
            prompt_type=PromptType.CONFIRM_ACTION,
            question=question,
            options=["Yes", "No"],
            default_option="Yes",
            context=context or {},
            requires_response=True,
        )

        self.pending_prompts.append(prompt)

        # Log the question
        self.add_message(
            role=MessageRole.ASSISTANT,
            content=question,
            message_type="question",
            metadata={"prompt_id": prompt_id, "context": context or {}},
        )

        logger.info(f"❓ {question}")

        return prompt

    def ask_choice(
        self,
        question: str,
        options: List[str],
        default_option: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> InteractivePrompt:
        """
        Ask user to choose from multiple options.

        Example:
            prompt = chat.ask_choice(
                question="Type hint for this property?",
                options=["boolean", "Observable<boolean>", "string"],
                default_option="boolean",
                context={"property": "isLoading", "usage": "template binding"},
            )
        """
        prompt_id = f"choice_{len(self.pending_prompts)}"

        prompt = InteractivePrompt(
            prompt_id=prompt_id,
            prompt_type=PromptType.CHOOSE_OPTION,
            question=question,
            options=options,
            default_option=default_option or (options[0] if options else None),
            context=context or {},
            requires_response=True,
        )

        self.pending_prompts.append(prompt)

        self.add_message(
            role=MessageRole.ASSISTANT,
            content=f"{question}\nOptions: {', '.join(options)}",
            message_type="question",
            metadata={"prompt_id": prompt_id, "options": options},
        )

        logger.info(f"🔹 {question}")
        for i, opt in enumerate(options, 1):
            logger.info(f"   {i}. {opt}")

        return prompt

    def respond_to_prompt(
        self,
        prompt_id: str,
        response: str,
    ) -> bool:
        """
        Record user's response to a prompt.

        Returns: True if prompt was found and updated, False otherwise
        """
        for prompt in self.pending_prompts:
            if prompt.prompt_id == prompt_id:
                prompt.user_response = response
                prompt.response_timestamp = datetime.now()
                self.pending_prompts.remove(prompt)
                self.answered_prompts.append(prompt)

                # Log response
                self.add_message(
                    role=MessageRole.USER,
                    content=response,
                    message_type="response",
                    metadata={"prompt_id": prompt_id},
                )

                logger.info(f"✅ Response: {response}")
                return True

        logger.warning(f"Prompt not found: {prompt_id}")
        return False

    def report_progress(
        self,
        stage: str,
        status: str,
        details: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Report progress to user (informational, no response required).

        Example:
            chat.report_progress(
                stage="code_generation",
                status="completed",
                details="Generated TypeScript component with 8 properties",
                metadata={"file": "add-members.component.ts", "properties": 8},
            )
        """
        message = f"[{stage}] {status}"
        if details:
            message += f" - {details}"

        self.add_message(
            role=MessageRole.ASSISTANT,
            content=message,
            message_type="progress",
            metadata=metadata or {},
        )

        logger.info(f"📊 {message}")

    def report_error(
        self,
        error_message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        recoverable: bool = True,
    ) -> None:
        """
        Report error to user.

        Example:
            chat.report_error(
                error_message="TS2339: Property 'isLoading' not found",
                file_path="add-members.component.ts",
                line_number=42,
                recoverable=True,
            )
        """
        message = f"❌ {error_message}"
        if file_path:
            message += f" in {file_path}"
            if line_number:
                message += f":{line_number}"

        self.add_message(
            role=MessageRole.ASSISTANT,
            content=message,
            message_type="error",
            metadata={
                "file_path": file_path,
                "line_number": line_number,
                "recoverable": recoverable,
            },
        )

        if recoverable:
            logger.warning(f"⚠️ {message} (recoverable)")
        else:
            logger.error(f"❌ {message} (UNRECOVERABLE)")

    def clarify_ambiguity(
        self,
        ambiguity_description: str,
        possible_resolutions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> InteractivePrompt:
        """
        Ask user to clarify an ambiguous situation.

        Example:
            prompt = chat.clarify_ambiguity(
                ambiguity_description="Property 'isLoading' type is ambiguous",
                possible_resolutions=[
                    {"name": "boolean", "reason": "Simple flag"},
                    {"name": "Observable<boolean>", "reason": "Async stream"},
                    {"name": "BehaviorSubject<boolean>", "reason": "With initial value"},
                ],
                context={"property": "isLoading", "usage": "HTML binding"},
            )
        """
        prompt_id = f"clarify_{len(self.pending_prompts)}"

        options = [r["name"] for r in possible_resolutions]

        prompt = InteractivePrompt(
            prompt_id=prompt_id,
            prompt_type=PromptType.CLARIFY_AMBIGUITY,
            question=ambiguity_description,
            options=options,
            context=context or {"resolutions": possible_resolutions},
            requires_response=True,
        )

        self.pending_prompts.append(prompt)

        message = f"{ambiguity_description}\n"
        for i, resolution in enumerate(possible_resolutions, 1):
            message += f"  {i}. {resolution['name']}: {resolution.get('reason', '')}\n"

        self.add_message(
            role=MessageRole.ASSISTANT,
            content=message,
            message_type="question",
            metadata={"prompt_id": prompt_id, "resolutions": possible_resolutions},
        )

        logger.info(f"🔀 {ambiguity_description}")

        return prompt

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation."""
        total_messages = len(self.conversation_history)
        user_messages = sum(1 for m in self.conversation_history if m.role == MessageRole.USER)
        assistant_messages = sum(
            1 for m in self.conversation_history if m.role == MessageRole.ASSISTANT
        )
        total_questions = len(self.answered_prompts) + len(self.pending_prompts)

        return {
            "ticket_id": self.ticket_id,
            "session_start": self.session_start.isoformat(),
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_questions": total_questions,
            "answered_questions": len(self.answered_prompts),
            "pending_questions": len(self.pending_prompts),
        }

    def export_to_json(self) -> str:
        """Export conversation to JSON string."""
        data = {
            "ticket_id": self.ticket_id,
            "session_start": self.session_start.isoformat(),
            "summary": self.get_conversation_summary(),
            "conversation_history": [m.to_dict() for m in self.conversation_history],
            "answered_prompts": [p.to_dict() for p in self.answered_prompts],
            "pending_prompts": [p.to_dict() for p in self.pending_prompts],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def log_to_file(self, file_path: str) -> None:
        """Export conversation to JSON file."""
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(self.export_to_json(), encoding="utf-8")
        logger.info(f"Conversation logged: {output_file}")

    def get_context_for_llm(self) -> str:
        """
        Get conversation context to pass to LLM for consistency.

        The LLM should see previous decisions made during conversation
        so it doesn't ask the same questions or make contradictory choices.
        """
        context_lines = [
            "=== PREVIOUS DECISIONS IN THIS SESSION ===\n",
        ]

        for prompt in self.answered_prompts:
            context_lines.append(f"Q: {prompt.question}")
            context_lines.append(f"A: {prompt.user_response}\n")

        if not self.answered_prompts:
            context_lines.append("(No previous decisions yet)\n")

        return "\n".join(context_lines)

    def require_all_responses(self) -> bool:
        """
        Check if all pending prompts have responses.

        Returns: True if all prompts answered, False if any pending
        """
        return len(self.pending_prompts) == 0

    def wait_for_response(
        self,
        prompt_id: str,
        timeout_seconds: int = 300,
    ) -> Optional[str]:
        """
        Block until user responds to prompt (or timeout).

        Returns: User's response, or None if timeout
        """
        import time

        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            for prompt in self.answered_prompts:
                if prompt.prompt_id == prompt_id:
                    return prompt.user_response

            time.sleep(0.5)  # Check every 500ms

        logger.warning(f"Timeout waiting for response to prompt: {prompt_id}")
        return None
