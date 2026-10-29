"""
Unit tests for MCP Two-Level Catalog Index and JIT Schema Injection.
"""
import sys
from pathlib import Path
import pytest

backend_dir = Path(__file__).resolve().parents[1] / "chatbot" / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from services.mcp_manager import McpManager


def test_mcp_two_level_catalog_index():
    manager = McpManager()
    catalog = manager.get_catalog_index()
    assert len(catalog) > 0
    
    # Check structure
    first = catalog[0]
    assert "name" in first
    assert "server" in first
    assert "description" in first
    # Verify namespacing
    assert "." in first["name"]


def test_mcp_shortlist_tools_for_ui_task():
    manager = McpManager()
    ui_task = "Verify modal dialog button click and inspect DOM error in Chrome browser"
    shortlisted = manager.shortlist_tools_for_task(ui_task, max_tools=3)
    assert len(shortlisted) > 0
    # Should include browser/playwright tools
    assert any("browser" in name or "playwright" in name for name in shortlisted)


def test_mcp_shortlist_tools_for_file_task():
    manager = McpManager()
    file_task = "Inspect repository directory and read file content"
    shortlisted = manager.shortlist_tools_for_task(file_task, max_tools=3)
    assert len(shortlisted) > 0
    # Should include filesystem tools
    assert any("filesystem" in name or "file" in name for name in shortlisted)


def test_mcp_jit_schema_injection():
    manager = McpManager()
    tools = ["playwright.browser_inspect_dom", "filesystem.read_file"]
    schemas = manager.get_injected_schemas(tools)
    assert len(schemas) >= 1
    
    for s in schemas:
        assert "name" in s
        assert "parameters" in s
        assert "description" in s
        assert s["name"] in tools
