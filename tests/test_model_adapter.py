"""
Unit tests for Model-Adaptive Harness (Anthropic, OpenAI, Gemini adapters).
"""
import pytest
from ticket_to_code.intelligence.model_adapter import (
    NormalizedRequest,
    NormalizedEvent,
    CanonicalToolCall,
    AnthropicAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    get_adapter_for_model,
)
from ticket_to_code.intelligence.model_router import AVAILABLE_MODELS


def test_anthropic_adapter_wire_format():
    adapter = get_adapter_for_model("claude-3-5-sonnet")
    assert isinstance(adapter, AnthropicAdapter)

    req = NormalizedRequest(
        system_prompt="You are an autonomous engineer.",
        messages=[{"role": "user", "content": "Fix bug in login.ts"}],
        tools=[{
            "name": "view_file",
            "description": "Read file lines",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }],
        max_tokens=2048,
        temperature=0.2
    )

    wire = adapter.to_wire(req)
    assert wire["model"] == "claude-3-5-sonnet"
    assert wire["system"] == "You are an autonomous engineer."
    assert len(wire["messages"]) == 1
    assert len(wire["tools"]) == 1
    assert wire["tools"][0]["name"] == "view_file"
    assert "input_schema" in wire["tools"][0]

    # Test from_wire tool_use parsing
    raw_response = {
        "content": [
            {"type": "text", "text": "I will inspect the file."},
            {
                "type": "tool_use",
                "id": "tool_123",
                "name": "view_file",
                "input": {"path": "src/login.ts"}
            }
        ],
        "usage": {"input_tokens": 150, "output_tokens": 45}
    }
    event = adapter.from_wire(raw_response)
    assert event.event_type == "tool_call"
    assert event.text == "I will inspect the file."
    assert len(event.tool_calls) == 1
    assert event.tool_calls[0].id == "tool_123"
    assert event.tool_calls[0].name == "view_file"
    assert event.tool_calls[0].args == {"path": "src/login.ts"}
    assert event.usage["prompt_tokens"] == 150


def test_openai_adapter_wire_format():
    adapter = get_adapter_for_model("gpt-4o")
    assert isinstance(adapter, OpenAIAdapter)

    req = NormalizedRequest(
        system_prompt="You are an expert pair programmer.",
        messages=[{"role": "user", "content": "Update pom.xml"}],
        tools=[{
            "name": "replace_file_content",
            "description": "Surgical diff",
            "parameters": {"type": "object", "properties": {"target": {"type": "string"}}}
        }],
        max_tokens=1000
    )

    wire = adapter.to_wire(req)
    assert wire["model"] == "gpt-4o"
    assert wire["messages"][0]["role"] == "system"
    assert wire["messages"][1]["role"] == "user"
    assert wire["tools"][0]["type"] == "function"
    assert wire["tools"][0]["function"]["name"] == "replace_file_content"

    # Test from_wire tool_calls parsing
    raw_response = {
        "choices": [{
            "message": {
                "content": "Applying patch now.",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "replace_file_content",
                        "arguments": '{"target": "pom.xml"}'
                    }
                }]
            }
        }],
        "usage": {"prompt_tokens": 200, "completion_tokens": 50}
    }
    event = adapter.from_wire(raw_response)
    assert event.event_type == "tool_call"
    assert event.text == "Applying patch now."
    assert len(event.tool_calls) == 1
    assert event.tool_calls[0].name == "replace_file_content"
    assert event.tool_calls[0].args == {"target": "pom.xml"}


def test_gemini_adapter_wire_format():
    adapter = get_adapter_for_model("gemini-2.0-flash")
    assert isinstance(adapter, GeminiAdapter)

    req = NormalizedRequest(
        system_prompt="Fast classifier",
        messages=[{"role": "user", "content": "Triage ticket"}],
        tools=[{"name": "search_symbols", "description": "AST lookup"}]
    )

    wire = adapter.to_wire(req)
    assert wire["model"] == "gemini-2.0-flash"
    assert wire["system_instruction"]["parts"][0]["text"] == "Fast classifier"
    assert wire["contents"][0]["role"] == "user"
    assert "tools" in wire

    # Test from_wire function_call parsing
    raw_response = {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Locating symbol..."},
                    {"function_call": {"name": "search_symbols", "args": {"query": "User"}}}
                ]
            }
        }],
        "usage_metadata": {"prompt_token_count": 80, "candidates_token_count": 25}
    }
    event = adapter.from_wire(raw_response)
    assert event.event_type == "tool_call"
    assert len(event.tool_calls) == 1
    assert event.tool_calls[0].name == "search_symbols"
    assert event.tool_calls[0].args == {"query": "User"}
