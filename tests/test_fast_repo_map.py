import pytest
from pathlib import Path

from ticket_to_code.indexing.fast_repo_map import FastRepoMap
from ticket_to_code.models import ValueEdgeTicket
from ticket_to_code.runtime.fast_ide_engine import FastIDEEngine


def test_fast_repo_map_budget_and_extraction(tmp_path):
    """Verify FastRepoMap extracts symbols and respects the token budget."""
    # Create sample Java class
    service_file = tmp_path / "src" / "UserService.java"
    service_file.parent.mkdir(parents=True, exist_ok=True)
    service_file.write_text(
        "package com.example;\n"
        "public class UserService {\n"
        "    public User getUser(String id) {\n"
        "        return null;\n"
        "    }\n"
        "    public void deleteUser(String id) {\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    # Create sample Python file
    py_file = tmp_path / "src" / "auth.py"
    py_file.write_text(
        "class AuthManager:\n"
        "    def authenticate(self, token: str) -> bool:\n"
        "        pass\n",
        encoding="utf-8",
    )

    repo_map = FastRepoMap(workspace_path=str(tmp_path), max_tokens=200)
    outline = repo_map.build_map(focus_keywords=["User", "Auth"])

    assert "UserService.java" in outline
    assert "public class UserService" in outline
    assert "public User getUser(String id)" in outline
    # Token budget test: character length must be roughly within budget (~4 chars/token)
    assert len(outline) <= 200 * 5


def test_fast_ide_engine_includes_repo_map_in_prompt(tmp_path):
    """Verify FastIDEEngine passes the compact repo outline into the prompt without blowing token budget."""
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    f1 = src_dir / "OrderService.java"
    f1.write_text("public class OrderService { public void placeOrder() {} }", encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="ORD-1",
        title="Update placeOrder in OrderService.java",
        description="Fix the placeOrder method in src/OrderService.java",
    )

    captured_prompts = []

    def mock_llm(prompt: str):
        captured_prompts.append(prompt)
        return (
            "FILE: src/OrderService.java\n"
            "<<<<<<< SEARCH\n"
            "public void placeOrder() {}\n"
            "=======\n"
            "public void placeOrder() { validate(); }\n"
            ">>>>>>> REPLACE\n"
        )

    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
        llm_client=mock_llm,
        max_repair_turns=1,
    )

    from unittest.mock import patch
    with patch.object(engine, "_run_fast_compile", return_value=(True, "")):
        with patch.object(engine, "_run_fast_test", return_value=(True, "")):
            res = engine.run()

    assert res.success is True
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "REPOSITORY SYMBOL MAP:" in prompt
    assert "OrderService.java" in prompt
    assert "public class OrderService" in prompt
    # Guarantee the prompt is strictly compact (< 2,500 words / tokens)
    assert len(prompt.split()) < 1000
