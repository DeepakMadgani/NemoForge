"""DORMANT: not used by any production code path (only by tests) as of 2026-10-29.
Kept because tests pin its behaviour; wire it in or delete it together with its tests.


Model-Adaptive Harness with Canonical Normalization.

Decouples the orchestrator from model-specific dialects (Anthropic, OpenAI, Gemini).
The orchestrator interacts solely with `NormalizedRequest` and receives `NormalizedEvent`.
Concrete adapters translate wire payloads at the edges.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import json
import logging

from ticket_to_code.intelligence.model_router import (
    ModelDescriptor,
    ModelRouter,
    AVAILABLE_MODELS,
    DEFAULT_MODEL_ID
)

logger = logging.getLogger(__name__)


class CanonicalToolCall(BaseModel):
    id: str
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class NormalizedRequest(BaseModel):
    system_prompt: str = ""
    messages: List[Dict[str, str]] = Field(default_factory=list)  # [{"role": "user"|"assistant", "content": "..."}]
    tools: List[Dict[str, Any]] = Field(default_factory=list)      # [{"name": ..., "description": ..., "parameters": ...}]
    max_tokens: Optional[int] = None
    temperature: float = 0.0


class NormalizedEvent(BaseModel):
    event_type: str  # "text", "tool_call", "usage", "error"
    text: Optional[str] = None
    tool_calls: List[CanonicalToolCall] = Field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    raw_finish_reason: Optional[str] = None


class BaseAdapter(ABC):
    def __init__(self, descriptor: ModelDescriptor):
        self.descriptor = descriptor

    @abstractmethod
    def to_wire(self, req: NormalizedRequest) -> Dict[str, Any]:
        """Convert canonical request to provider-specific wire payload."""
        pass

    @abstractmethod
    def from_wire(self, raw_resp: Any) -> NormalizedEvent:
        """Parse provider response into canonical event."""
        pass


class AnthropicAdapter(BaseAdapter):
    """Adapter for Claude models (3.5 Sonnet, 3.7 Sonnet)."""

    def to_wire(self, req: NormalizedRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.descriptor.model_id,
            "system": req.system_prompt,
            "messages": [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in req.messages
            ],
            "temperature": req.temperature,
        }
        if req.max_tokens:
            payload["max_tokens"] = min(req.max_tokens, 8192)
        else:
            payload["max_tokens"] = 4096

        if req.tools:
            anthropic_tools = []
            for t in req.tools:
                anthropic_tools.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}})
                })
            payload["tools"] = anthropic_tools

        return payload

    def from_wire(self, raw_resp: Any) -> NormalizedEvent:
        tool_calls: List[CanonicalToolCall] = []
        text_parts: List[str] = []

        # Handle dict or object response
        content = getattr(raw_resp, "content", raw_resp.get("content", []) if isinstance(raw_resp, dict) else [])
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                b_type = getattr(block, "type", block.get("type") if isinstance(block, dict) else "")
                if b_type == "text":
                    b_text = getattr(block, "text", block.get("text", "") if isinstance(block, dict) else "")
                    text_parts.append(b_text)
                elif b_type == "tool_use":
                    b_id = getattr(block, "id", block.get("id", "") if isinstance(block, dict) else "")
                    b_name = getattr(block, "name", block.get("name", "") if isinstance(block, dict) else "")
                    b_input = getattr(block, "input", block.get("input", {}) if isinstance(block, dict) else {})
                    tool_calls.append(CanonicalToolCall(id=b_id, name=b_name, args=b_input if isinstance(b_input, dict) else {}))

        usage = None
        raw_usage = getattr(raw_resp, "usage", raw_resp.get("usage") if isinstance(raw_resp, dict) else None)
        if raw_usage:
            usage = {
                "prompt_tokens": getattr(raw_usage, "input_tokens", raw_usage.get("input_tokens", 0) if isinstance(raw_usage, dict) else 0),
                "completion_tokens": getattr(raw_usage, "output_tokens", raw_usage.get("output_tokens", 0) if isinstance(raw_usage, dict) else 0),
                "cached_tokens": getattr(raw_usage, "cache_read_input_tokens", raw_usage.get("cache_read_input_tokens", 0) if isinstance(raw_usage, dict) else 0),
                "thinking_tokens": 0,
            }

        return NormalizedEvent(
            event_type="tool_call" if tool_calls else "text",
            text="".join(text_parts).strip() if text_parts else None,
            tool_calls=tool_calls,
            usage=usage
        )


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI models (GPT-4o, GPT-4o-mini)."""

    def to_wire(self, req: NormalizedRequest) -> Dict[str, Any]:
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        for m in req.messages:
            messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        payload: Dict[str, Any] = {
            "model": self.descriptor.model_id,
            "messages": messages,
            "temperature": req.temperature,
        }
        if req.max_tokens:
            payload["max_tokens"] = req.max_tokens

        if req.tools:
            openai_tools = []
            for t in req.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}})
                    }
                })
            payload["tools"] = openai_tools

        return payload

    def from_wire(self, raw_resp: Any) -> NormalizedEvent:
        tool_calls: List[CanonicalToolCall] = []
        text = None

        choices = getattr(raw_resp, "choices", raw_resp.get("choices", []) if isinstance(raw_resp, dict) else [])
        if choices:
            first = choices[0]
            msg = getattr(first, "message", first.get("message") if isinstance(first, dict) else None)
            if msg:
                text = getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else None)
                raw_tc = getattr(msg, "tool_calls", msg.get("tool_calls") if isinstance(msg, dict) else None)
                if raw_tc:
                    for tc in raw_tc:
                        tc_id = getattr(tc, "id", tc.get("id", "") if isinstance(tc, dict) else "")
                        fn = getattr(tc, "function", tc.get("function") if isinstance(tc, dict) else None)
                        if fn:
                            fn_name = getattr(fn, "name", fn.get("name", "") if isinstance(fn, dict) else "")
                            raw_args = getattr(fn, "arguments", fn.get("arguments", "{}") if isinstance(fn, dict) else "{}")
                            try:
                                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                            except Exception:
                                parsed_args = {"raw": raw_args}
                            tool_calls.append(CanonicalToolCall(id=tc_id, name=fn_name, args=parsed_args))

        usage = None
        raw_usage = getattr(raw_resp, "usage", raw_resp.get("usage") if isinstance(raw_resp, dict) else None)
        if raw_usage:
            pt = getattr(raw_usage, "prompt_tokens", raw_usage.get("prompt_tokens", 0) if isinstance(raw_usage, dict) else 0)
            ct = getattr(raw_usage, "completion_tokens", raw_usage.get("completion_tokens", 0) if isinstance(raw_usage, dict) else 0)
            pt_details = getattr(raw_usage, "prompt_tokens_details", raw_usage.get("prompt_tokens_details", {}) if isinstance(raw_usage, dict) else {})
            ct_details = getattr(raw_usage, "completion_tokens_details", raw_usage.get("completion_tokens_details", {}) if isinstance(raw_usage, dict) else {})
            cached = getattr(pt_details, "cached_tokens", pt_details.get("cached_tokens", 0) if isinstance(pt_details, dict) else 0)
            thinking = getattr(ct_details, "reasoning_tokens", ct_details.get("reasoning_tokens", 0) if isinstance(ct_details, dict) else 0)
            usage = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cached_tokens": cached,
                "thinking_tokens": thinking,
            }

        return NormalizedEvent(
            event_type="tool_call" if tool_calls else "text",
            text=text.strip() if text else None,
            tool_calls=tool_calls,
            usage=usage
        )


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini models (Gemini 2.0 Flash, Gemini 1.5 Pro)."""

    def to_wire(self, req: NormalizedRequest) -> Dict[str, Any]:
        contents = []
        for m in req.messages:
            role = "user" if m.get("role") == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        payload: Dict[str, Any] = {
            "model": self.descriptor.model_id,
            "contents": contents,
        }
        if req.system_prompt:
            payload["system_instruction"] = {"parts": [{"text": req.system_prompt}]}

        generation_config: Dict[str, Any] = {"temperature": req.temperature}
        if req.max_tokens:
            generation_config["max_output_tokens"] = req.max_tokens
        payload["generation_config"] = generation_config

        if req.tools:
            function_declarations = []
            for t in req.tools:
                function_declarations.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}})
                })
            payload["tools"] = [{"function_declarations": function_declarations}]

        return payload

    def from_wire(self, raw_resp: Any) -> NormalizedEvent:
        tool_calls: List[CanonicalToolCall] = []
        text_parts: List[str] = []

        candidates = getattr(raw_resp, "candidates", raw_resp.get("candidates", []) if isinstance(raw_resp, dict) else [])
        if candidates:
            first = candidates[0]
            content = getattr(first, "content", first.get("content") if isinstance(first, dict) else None)
            if content:
                parts = getattr(content, "parts", content.get("parts", []) if isinstance(content, dict) else [])
                for p in parts:
                    p_text = getattr(p, "text", p.get("text") if isinstance(p, dict) else None)
                    if p_text:
                        text_parts.append(p_text)
                    fn_call = getattr(p, "function_call", p.get("function_call") if isinstance(p, dict) else None)
                    if fn_call:
                        fn_name = getattr(fn_call, "name", fn_call.get("name", "") if isinstance(fn_call, dict) else "")
                        fn_args = getattr(fn_call, "args", fn_call.get("args", {}) if isinstance(fn_call, dict) else {})
                        tool_calls.append(CanonicalToolCall(
                            id=f"call_{fn_name}",
                            name=fn_name,
                            args=dict(fn_args) if hasattr(fn_args, "__iter__") else {}
                        ))

        # Direct .text attribute fallback for google-genai client
        if not text_parts and hasattr(raw_resp, "text") and raw_resp.text:
            text_parts.append(raw_resp.text)

        usage = None
        raw_usage = getattr(raw_resp, "usage_metadata", raw_resp.get("usage_metadata") if isinstance(raw_resp, dict) else None)
        if raw_usage:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_token_count", raw_usage.get("prompt_token_count", 0) if isinstance(raw_usage, dict) else 0),
                "completion_tokens": getattr(raw_usage, "candidates_token_count", raw_usage.get("candidates_token_count", 0) if isinstance(raw_usage, dict) else 0),
                "cached_tokens": getattr(raw_usage, "cached_content_token_count", raw_usage.get("cached_content_token_count", 0) if isinstance(raw_usage, dict) else 0),
                "thinking_tokens": getattr(raw_usage, "thoughts_token_count", raw_usage.get("thoughts_token_count", 0) if isinstance(raw_usage, dict) else 0),
            }

        return NormalizedEvent(
            event_type="tool_call" if tool_calls else "text",
            text="".join(text_parts).strip() if text_parts else None,
            tool_calls=tool_calls,
            usage=usage
        )


def get_adapter_for_model(model_id_or_desc: Any) -> BaseAdapter:
    """Factory to retrieve the appropriate model adapter."""
    if isinstance(model_id_or_desc, ModelDescriptor):
        desc = model_id_or_desc
    else:
        desc = ModelRouter.get_model_descriptor(str(model_id_or_desc))

    if desc.provider == "anthropic":
        return AnthropicAdapter(desc)
    elif desc.provider == "openai":
        return OpenAIAdapter(desc)
    elif desc.provider == "google":
        return GeminiAdapter(desc)
    return AnthropicAdapter(desc)
