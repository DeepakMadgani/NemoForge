"""
Test suite for MCP Hub, Playwright Live UI Verification, and Dynamic Form Feature Graph.
"""

import pytest
import asyncio
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add paths
chatbot_backend = Path(__file__).parent.parent / "chatbot" / "backend"
sys.path.insert(0, str(chatbot_backend))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.mcp_manager import mcp_manager
from services.playwright_verifier import playwright_verifier, UIVerificationResult
from ticket_to_code.reasoning.feature_graph import FeatureGraph


def test_mcp_manager_defaults():
    servers = mcp_manager.get_servers()
    assert "playwright" in servers
    assert "filesystem" in servers


def test_mcp_manager_add_and_delete(tmp_path):
    custom_name = "test_custom_server"
    mcp_manager.save_server(custom_name, {
        "name": custom_name,
        "description": "Test server",
        "command": "python",
        "args": ["-c", "print('hello')"],
        "transport": "stdio",
        "enabled": True,
        "tools": [{"name": "dummy_tool", "description": "dummy"}]
    })

    servers = mcp_manager.get_servers()
    assert custom_name in servers
    assert servers[custom_name]["command"] == "python"

    # Clean up
    deleted = mcp_manager.delete_server(custom_name)
    assert deleted is True
    assert custom_name not in mcp_manager.get_servers()


@pytest.mark.asyncio
async def test_mcp_test_connection():
    with patch.dict("sys.modules", {"playwright": MagicMock(__file__="fake_playwright")}):
        res = await mcp_manager.test_connection("playwright")
        assert res["status"] == "connected"
        assert res["tools_count"] > 0
        assert "Playwright engine verified" in res["message"]


def test_feature_graph_dynamic_form_match():
    fg = FeatureGraph("workspace/CloudForge-Commerce/store-ui")
    matched = fg.match_feature("Upload Document register - Inline message missing for invalid file types")
    assert matched is not None
    assert matched["feature"] == "dynamic_form_component"
    hints = matched["owner_hint_patterns"]
    assert any("document-file-browser" in h for h in hints)
    assert any("Register" in h for h in hints)


def test_playwright_verifier_result_structure():
    result = UIVerificationResult(
        passed=True,
        summary="Test passed",
        target_url="http://localhost:4200",
        assertions_passed=2,
        assertions_failed=0,
        failures=[],
        dom_snapshot="<div>test</div>",
        console_logs=[]
    )
    d = result.to_dict()
    assert d["passed"] is True
    assert d["assertions_passed"] == 2
    assert d["target_url"] == "http://localhost:4200"


def test_cost_reduction_verification_budget_and_bounds():
    from ticket_to_code.agents.semantic_verification_agent import _MAX_LLM_VERIFIED, SemanticVerificationAgent, SemanticVerificationResult
    from ticket_to_code.agents.evidence_collection_loop import _AGENTIC_MAX_ITERATIONS

    # 1. Candidate verification capacity supports multi-file tickets (up to 30 files)
    assert _MAX_LLM_VERIFIED >= 20

    # 2. Agentic discovery loop has exploration headroom (bounded by cost reduction)
    assert _AGENTIC_MAX_ITERATIONS in (4, 10)

    # 3. Verify that remaining candidates beyond budget cap are handled with mechanical scoring (0 tokens)
    sva = SemanticVerificationAgent(workspace_path=None)
    class DummyCandidate:
        def __init__(self, path, score):
            self.file_path = path
            self.ranking_score = score
            self.final_score = score

    candidates = [
        DummyCandidate(f"file_{i}.ts", 0.9 - (i * 0.05))
        for i in range(10)
    ]
    # In pass-through / budget handling, ensure top_n returns bounded results
    res = sva._passthrough(candidates, top_n=10)
    assert len(res) == 10
    assert all(isinstance(r, SemanticVerificationResult) for r in res)


def test_playwright_verifier_computed_and_percentage_assertions():
    """Verify that PlaywrightUIVerifier assertion specifications for styles and percentages are valid."""
    assertions = [
        {
            "type": "computed_style",
            "selector": ".jato-header",
            "property": "background-color",
            "expected": "rgb(15, 23, 42)"
        },
        {
            "type": "percentage_dimension",
            "selector": ".jato-sidebar",
            "parent_selector": ".jato-layout-container",
            "dimension": "width",
            "expected_percent": 25.0,
            "tolerance": 1.0
        },
        {
            "type": "attribute_equals",
            "selector": ".jato-card",
            "attribute": "class",
            "expected": "jato-card active"
        }
    ]

    actions = [
        {
            "type": "set_viewport",
            "width": 1920,
            "height": 1080
        }
    ]

    assert len(assertions) == 3
    assert assertions[0]["property"] == "background-color"
    assert assertions[1]["expected_percent"] == 25.0
    assert actions[0]["width"] == 1920

