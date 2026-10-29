import pytest
from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from ticket_to_code.agents.error_resolution_agent import ErrorResolutionAgent
from ticket_to_code.agents.diagnostic_context_builder import DiagnosticContextBuilder, DiagnosticContext


def test_diagnostic_context_builder_is_available_at_runtime():
    """Verify DiagnosticContextBuilder can be imported and instantiated at runtime."""
    builder = DiagnosticContextBuilder()
    assert builder is not None
    assert hasattr(builder, "build_context")
    assert hasattr(builder, "update_file_hash")


def test_resolve_errors_executes_diagnostic_context_builder(tmp_path):
    """Verify resolve_errors executes DiagnosticContextBuilder without NameError."""
    # Create a mock workspace with a typescript file
    src_dir = tmp_path / "src" / "app"
    src_dir.mkdir(parents=True, exist_ok=True)
    ts_file = src_dir / "test.component.ts"
    ts_file.write_text(
        "export class TestComponent {\n"
        "  title: string = 'test';\n"
        "  testMethod() {\n"
        "    console.log('error here');\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    # Use a dummy LLM object
    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(content="Analysis complete, no action taken.")

    agent = ErrorResolutionAgent(llm=FakeLLM(), workspace_path=tmp_path)
    agent.our_modified_files = {str(ts_file.relative_to(tmp_path)).replace("\\", "/")}

    build_errors = [
        f"src/app/test.component.ts(4,5): error TS2304: Cannot find name 'foo'."
    ]

    # This executes resolve_errors, which instantiates DiagnosticContextBuilder
    # and builds diagnostic contexts at line 104+
    modified = agent.resolve_errors(build_errors, max_iter=1)

    # Completed without NameError, modified_files returned as a dict
    assert isinstance(modified, dict)
