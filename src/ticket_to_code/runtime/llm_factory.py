"""
Standalone LLM Factory for Ticket-to-Code.

Provides direct multi-provider LLM initialization (Google Gemini, OpenAI, Anthropic)
with zero external dependencies on Aviator ADT or enterprise microservices.
Gracefully degrades to a dummy mock chat model when no API keys are configured or in tests.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

_DEFAULT_NEBIUS_MAX_OUTPUT_TOKENS = 16384


def _nebius_default_max_tokens() -> int:
    """Default Nebius output cap: NEBIUS_MAX_OUTPUT_TOKENS (int) else 16384."""
    raw = os.getenv("NEBIUS_MAX_OUTPUT_TOKENS", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid NEBIUS_MAX_OUTPUT_TOKENS=%r; using %d", raw, _DEFAULT_NEBIUS_MAX_OUTPUT_TOKENS)
    return _DEFAULT_NEBIUS_MAX_OUTPUT_TOKENS


def _require_gcp_project(explicit: Optional[str] = None) -> str:
    """Resolve the GCP project for Vertex AI; never fall back to a hardcoded project."""
    project = explicit or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise RuntimeError(
            "Vertex AI requires GOOGLE_CLOUD_PROJECT to be set (no default project is assumed)."
        )
    return project


def _dummy_llm_allowed() -> bool:
    return os.getenv("AVIATOR_ALLOW_DUMMY_LLM", "") == "1" or bool(os.getenv("PYTEST_CURRENT_TEST"))


def _resolve_model_name(model_name: Optional[str], assistant: bool, default: str) -> str:
    """Resolve the model name: explicit arg > LLM_MODEL_ASSISTANT (assistant=True) > LLM_MODEL > default."""
    if model_name:
        return model_name
    if assistant:
        assistant_model = os.getenv("LLM_MODEL_ASSISTANT", "").strip()
        if assistant_model:
            return assistant_model
    return os.getenv("LLM_MODEL", default)


class DummyChatModel(BaseChatModel):
    """Offline/test mock chat model used when no external LLM credentials are set."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="OK"))])

    @property
    def _llm_type(self) -> str:
        return "dummy-chat-model"


class VertexGenAIChatModel(BaseChatModel):
    """LangChain-compatible ChatModel using Google GenAI SDK (Vertex AI mode)."""

    model_name: str = "gemini-2.5-pro"
    project: str = ""
    location: str = "europe-west4"
    temperature: float = 0.0
    # Cached SDK client (non-pydantic): built once per model instance instead of per call.
    _client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        project: Optional[str] = None,
        location: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            project=_require_gcp_project(project),
            location=location or os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west4"),
            temperature=temperature,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "vertex-genai-chat"

    def _get_client(self):
        """Lazily build and cache the google-genai client (one per model instance)."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(
                vertexai=True,
                project=_require_gcp_project(self.project),
                location=self.location or os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west4"),
            )
        return self._client

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        from google.genai import types

        client = self._get_client()

        def _content(role: str, text: str) -> Any:
            return types.Content(role=role, parts=[types.Part.from_text(text=str(text))])

        # Proper multi-turn transcript: human/other → "user", assistant → "model".
        system_instruction = None
        contents: List[Any] = []
        if isinstance(messages, str):
            contents.append(_content("user", messages))
        else:
            for m in messages:
                if isinstance(m, SystemMessage):
                    system_instruction = str(m.content)
                elif isinstance(m, AIMessage):
                    contents.append(_content("model", m.content))
                elif isinstance(m, HumanMessage):
                    contents.append(_content("user", m.content))
                elif isinstance(m, str):
                    contents.append(_content("user", m))
                else:
                    contents.append(_content("user", getattr(m, "content", "")))
        if not contents:
            contents = [_content("user", " ")]

        max_output_tokens = kwargs.get("max_output_tokens") or kwargs.get("max_tokens")
        temp = kwargs.get("temperature", self.temperature)

        config_args: dict[str, Any] = {"temperature": temp}
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if stop:
            config_args["stop_sequences"] = stop
        if max_output_tokens:
            config_args["max_output_tokens"] = int(max_output_tokens)

        # Configure thinking for Gemini models
        if "flash" in self.model_name.lower():
            config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        else:
            config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=1024)

        cfg = types.GenerateContentConfig(**config_args)
        resp = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=cfg,
        )

        pt = 0
        ct = 0
        tt = 0
        th = 0
        cd = 0
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            um = resp.usage_metadata
            pt = getattr(um, "prompt_token_count", 0) or 0
            ct = getattr(um, "candidates_token_count", 0) or 0
            tt = getattr(um, "total_token_count", 0) or 0
            th = getattr(um, "thoughts_token_count", 0) or 0
            cd = getattr(um, "cached_content_token_count", 0) or 0

        token_usage = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt or (pt + ct),
            "cached_tokens": cd,
            "thinking_tokens": th,
        }

        usage_metadata = {
            "input_tokens": pt,
            "output_tokens": ct,
            "total_tokens": tt or (pt + ct),
            "input_token_details": {"cached_tokens": cd},
            "output_token_details": {"reasoning": th},
        }

        msg = AIMessage(
            content=resp.text or "",
            response_metadata={
                "model_name": self.model_name,
                "token_usage": token_usage,
                "usage_metadata": token_usage,
            },
            usage_metadata=usage_metadata,
        )

        return ChatResult(generations=[ChatGeneration(message=msg)])


class NebiusTokenFactoryChatModel(BaseChatModel):
    """
    Direct, lightweight, dependency-free ChatModel for Nebius Token Factory
    hosting NVIDIA Nemotron models.
    """
    model_name: str = "nvidia/nemotron-3-super-120b-a12b"
    api_key: Optional[str] = None
    base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    temperature: float = 0.0
    max_tokens: int = _DEFAULT_NEBIUS_MAX_OUTPUT_TOKENS

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3-super-120b-a12b",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        raw = model_name or "nvidia/nemotron-3-super-120b-a12b"
        cleaned_model = raw[len("nebius/"):] if raw.startswith("nebius/") else raw
        super().__init__(
            model_name=cleaned_model,
            api_key=api_key or os.getenv("NEBIUS_API_KEY"),
            base_url=(base_url or os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")).rstrip("/") + "/",
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else _nebius_default_max_tokens(),
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "nebius-tokenfactory-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        import httpx

        formatted = []
        for m in messages:
            if isinstance(m, SystemMessage):
                formatted.append({"role": "system", "content": str(m.content)})
            elif isinstance(m, HumanMessage):
                formatted.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage):
                formatted.append({"role": "assistant", "content": str(m.content)})
            elif isinstance(m, str):
                formatted.append({"role": "user", "content": m})
            else:
                formatted.append({"role": "user", "content": str(getattr(m, "content", ""))})

        # Ensure individual messages do not exceed safety threshold
        for msg in formatted:
            c = msg.get("content", "")
            if len(c) > 160_000:
                head = c[:80_000]
                tail = c[-60_000:]
                msg["content"] = f"{head}\n\n... [middle truncated for context limit] ...\n\n{tail}"

        # Proactively bound total payload context length to ~96,000 tokens (~384,000 chars)
        total_chars = sum(len(m.get("content", "")) for m in formatted)
        if total_chars > 384_000 and len(formatted) > 2:
            sys_msg = formatted[0] if formatted[0].get("role") == "system" else None
            latest_msg = formatted[-1]
            middle_msgs = formatted[1:-1] if sys_msg else formatted[:-1]
            while sum(len(m.get("content", "")) for m in middle_msgs) + len(latest_msg.get("content", "")) + (len(sys_msg.get("content", "")) if sys_msg else 0) > 384_000 and middle_msgs:
                middle_msgs.pop(0)
            formatted = ([sys_msg] if sys_msg else []) + middle_msgs + [latest_msg]

        payload = {
            "model": self.model_name,
            "messages": formatted,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}chat/completions"

        timeout_sec = float(os.getenv("LLM_TIMEOUT", "120.0"))
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Nebius Token Factory error ({resp.status_code}): {resp.text[:300]}")
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        msg_obj = choice.get("message", {})
        content = msg_obj.get("content")
        if not content:
            content = msg_obj.get("reasoning_content") or msg_obj.get("reasoning") or ""

        usage = data.get("usage", {})
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            logger.warning(
                "Nebius response for model %s was TRUNCATED (finish_reason=length) at "
                "max_tokens=%s, completion_tokens=%s. Raise NEBIUS_MAX_OUTPUT_TOKENS or "
                "pass a larger max_tokens if the output is incomplete.",
                self.model_name,
                payload.get("max_tokens"),
                token_usage["completion_tokens"],
            )

        ai_msg = AIMessage(
            content=content,
            response_metadata={
                "model_name": self.model_name,
                "token_usage": token_usage,
                "finish_reason": finish_reason,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


class LLMFactory:
    """Universal LLM Factory providing direct access to foundation models."""

    @classmethod
    def get_llm(
        cls,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = True,
        assistant: bool = False,
        **kwargs: Any
    ) -> BaseChatModel:
        """
        Instantiate and return a standard LangChain chat model.
        
        Resolution Priority:
        1. Nebius Token Factory (NVIDIA Nemotron / Open Models on Nebius AI Cloud)
        2. Google Gemini via Google GenAI SDK (Vertex AI with service account credentials)
        3. Google Gemini via API key (langchain-google-genai)
        4. Anthropic Claude (via ANTHROPIC_API_KEY)
        5. OpenAI GPT (via OPENAI_API_KEY)
        6. DummyChatModel ONLY when AVIATOR_ALLOW_DUMMY_LLM=1 or running under pytest;
           otherwise a RuntimeError is raised so misconfiguration fails loudly.

        When ``assistant`` is True and ``model_name`` is not given, the model is
        taken from LLM_MODEL_ASSISTANT (falling back to LLM_MODEL).
        """
        provider = os.getenv("LLM_PROVIDER", "").lower()
        has_nebius = bool(os.getenv("NEBIUS_API_KEY"))
        has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        has_vertex_creds = bool(cred_path and os.path.exists(cred_path)) or bool(os.getenv("GOOGLE_CLOUD_PROJECT"))

        # 1. Nebius Token Factory (NVIDIA Nemotron / Open Models on Nebius AI Cloud)
        if provider == "nebius" or (not provider and has_nebius):
            try:
                raw_model = _resolve_model_name(model_name, assistant, "nvidia/nemotron-3-super-120b-a12b")
                if any(x in raw_model.lower() for x in ["gemini", "claude", "gpt-", "openai"]):
                    nebius_default = os.getenv("NEBIUS_MODEL") or "nvidia/nemotron-3-super-120b-a12b"
                    logger.warning(f"Nebius provider cannot run non-Nebius model '{raw_model}'. Falling back to '{nebius_default}'.")
                    raw_model = nebius_default
                resolved_model = raw_model[len("nebius/"):] if raw_model.startswith("nebius/") else raw_model
                logger.info("🚀 Initializing Nebius Token Factory (NVIDIA model): %s", resolved_model)
                return NebiusTokenFactoryChatModel(
                    model_name=resolved_model,
                    api_key=os.getenv("NEBIUS_API_KEY"),
                    base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
                    temperature=temperature,
                    max_tokens=kwargs.pop("max_tokens", _nebius_default_max_tokens()),
                    **kwargs
                )
            except Exception as e:
                logger.warning("Failed to initialize Nebius Token Factory model: %s", e)

        # 2. Google Gemini via Google GenAI SDK (Vertex AI with service account credentials)
        if has_vertex_creds and (provider in ("google", "vertex", "vertexai")):
            try:
                resolved_model = _resolve_model_name(model_name, assistant, "gemini-2.5-pro")
                logger.info("Initializing Google Vertex GenAI model from credentials: %s", resolved_model)
                return VertexGenAIChatModel(
                    model_name=resolved_model,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception as e:
                logger.warning("Failed to initialize Google Vertex GenAI model: %s", e)

        # 2. Google Gemini via API key
        if provider == "google" or (not provider and has_gemini):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                resolved_model = _resolve_model_name(model_name, assistant, "gemini-2.5-pro")
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                logger.info("Initializing Google GenAI model: %s", resolved_model)
                return ChatGoogleGenerativeAI(
                    model=resolved_model,
                    api_key=api_key,
                    temperature=temperature,
                    streaming=streaming,
                    **kwargs
                )
            except Exception as e:
                logger.warning("Failed to initialize Google GenAI model: %s", e)

        # 3. Anthropic Claude
        if provider == "anthropic" or (not provider and has_anthropic):
            try:
                from langchain_anthropic import ChatAnthropic
                resolved_model = _resolve_model_name(model_name, assistant, "claude-3-5-sonnet-20241022")
                logger.info("Initializing Anthropic model: %s", resolved_model)
                return ChatAnthropic(
                    model=resolved_model,
                    temperature=temperature,
                    streaming=streaming,
                    **kwargs
                )
            except Exception as e:
                logger.warning("Failed to initialize Anthropic model: %s", e)

        # 4. OpenAI GPT
        if provider == "openai" or (not provider and has_openai):
            try:
                from langchain_openai import ChatOpenAI
                resolved_model = _resolve_model_name(model_name, assistant, "gpt-4o")
                base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
                logger.info("Initializing OpenAI model: %s", resolved_model)
                return ChatOpenAI(
                    model=resolved_model,
                    temperature=temperature,
                    streaming=streaming,
                    base_url=base_url,
                    **kwargs
                )
            except Exception as e:
                logger.warning("Failed to initialize OpenAI model: %s", e)

        # 5. Offline / testing fallback — only when explicitly allowed.
        if _dummy_llm_allowed():
            logger.info("No active LLM credentials configured. Operating with DummyChatModel.")
            return DummyChatModel()
        raise RuntimeError(
            "No LLM provider could be initialized"
            + (f" (LLM_PROVIDER={provider!r})" if provider else "")
            + ". Configure one of: NEBIUS_API_KEY (Nebius Token Factory), "
            "GEMINI_API_KEY / GOOGLE_API_KEY (Google GenAI), "
            "GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT with LLM_PROVIDER=vertex (Vertex AI), "
            "ANTHROPIC_API_KEY (Anthropic), or OPENAI_API_KEY (OpenAI). "
            "Set AVIATOR_ALLOW_DUMMY_LLM=1 to deliberately run offline with the DummyChatModel."
        )

    @classmethod
    def get_model(
        cls,
        with_provider: bool = False,
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> BaseChatModel:
        """Compatibility method matching aviator.services.llm.LLMRegistry.get_model."""
        return cls.get_llm(model_name=model_name, **kwargs)


# Backward-compatible alias for any legacy call sites
LLMRegistry = LLMFactory

