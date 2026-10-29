"""
Unit tests for ContextBudget admission control, authority-based prompt sizing,
and explicit LLM max_tokens bounding.
"""

from unittest.mock import MagicMock, patch
import os

from ticket_to_code.agents.context_assembler import (
    ContextAssembler,
    ContextBlock,
    ContextBudget,
    AUTHORITY_LEVELS,
)
from ticket_to_code.llm_utils import llm_invoke


def test_context_budget_admission_control_sheds_low_authority():
    """Verify that when context exceeds ContextBudget, high-authority blocks are kept and low-authority are shed."""
    assembler = ContextAssembler()

    task = MagicMock()
    task.id = "TASK-01"

    # Block 1: Filesystem (Authority 1) - 400 tokens (~1600 chars)
    b_fs = ContextBlock(
        label="Disk Truth",
        content="public class OrderService { ... }" * 40,
        source="filesystem",
        authority=AUTHORITY_LEVELS["filesystem"],
        stage="repo",
    )

    # Block 2: Compiler Diagnostics (Authority 2) - 300 tokens (~1200 chars)
    b_diag = ContextBlock(
        label="Compiler Errors",
        content="Line 42: Cannot find symbol PaymentGateway\n" * 25,
        source="compiler",
        authority=AUTHORITY_LEVELS["compiler"],
        stage="validation",
    )

    # Block 3: RAG Semantic Snippets (Authority 6) - 1000 tokens (~4000 chars)
    b_rag = ContextBlock(
        label="RAG Chunks",
        content="Historical ticket note regarding payment systems.\n" * 80,
        source="rag",
        authority=AUTHORITY_LEVELS["rag"],
        stage="repo_evidence",
    )

    # Block 4: Speculative Planner Output (Authority 7) - 1000 tokens (~4000 chars)
    b_plan = ContextBlock(
        label="Planner Guidance",
        content="Maybe we should consider refactoring the whole module.\n" * 80,
        source="planner",
        authority=AUTHORITY_LEVELS["planner"],
        stage="task_metadata",
    )

    # Set budget to 800 tokens (~3200 chars) — enough for filesystem and compiler, but NOT RAG and planner
    budget = ContextBudget(max_tokens=800, max_chars_per_block=5000)

    blocks = [b_plan, b_rag, b_diag, b_fs]  # Intentionally pass out of order
    assembled = assembler._stage_final_assembly(blocks, task, budget)

    # Verify high authority blocks are admitted
    assert "[Disk Truth]" in assembled
    assert "[Compiler Errors]" in assembled

    # Verify low authority blocks were shed due to budget cap
    assert "[Planner Guidance]" not in assembled


def test_context_budget_per_block_truncation():
    """Verify that very large single blocks are truncated to max_chars_per_block."""
    assembler = ContextAssembler()
    task = MagicMock()
    task.id = "TASK-02"

    huge_block = ContextBlock(
        label="Giant File",
        content="X" * 15000,
        source="filesystem",
        authority=AUTHORITY_LEVELS["filesystem"],
        stage="repo",
    )

    budget = ContextBudget(max_tokens=10000, max_chars_per_block=2000)
    assembled = assembler._stage_final_assembly([huge_block], task, budget)

    assert "[Giant File]" in assembled
    assert "[truncated to per-block budget]" in assembled
    assert len(assembled) < 3000


class FakeChatModel:
    def __init__(self):
        self.last_kwargs = None
        self.last_messages = None

    def invoke(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        mock_resp = MagicMock()
        mock_resp.content = "result"
        return mock_resp


def test_llm_invoke_explicit_max_tokens_passed():
    """Verify llm_invoke passes max_tokens to the model."""
    fake_llm = FakeChatModel()
    messages = [MagicMock(content="hello")]
    llm_invoke(fake_llm, messages, max_tokens=2048)

    assert fake_llm.last_kwargs == {"max_tokens": 2048}


def test_llm_invoke_env_default_max_output_tokens():
    """Verify llm_invoke respects AVIATOR_DEFAULT_MAX_OUTPUT_TOKENS when max_tokens is None."""
    fake_llm = FakeChatModel()
    messages = [MagicMock(content="hello")]
    with patch.dict(os.environ, {"AVIATOR_DEFAULT_MAX_OUTPUT_TOKENS": "4096"}):
        llm_invoke(fake_llm, messages)

    assert fake_llm.last_kwargs == {"max_tokens": 4096}


def test_context_budget_target_file_exemption():
    """Verify that the target file being edited is exempted from shedding and uses generous 45k allowance."""
    assembler = ContextAssembler()
    task = MagicMock()
    task.id = "TASK-TARGET"
    task.file_path = "src/services/OrderService.java"

    # Target file block: 25k chars (> 16k standard block limit, < 45k target limit)
    target_block = ContextBlock(
        label="Target File (OrderService.java)",
        content="public class OrderService {\n" + ("    // Important logic line\n" * 900) + "}",
        source="filesystem",
        authority=AUTHORITY_LEVELS["filesystem"],
        stage="repo",
        file_path="src/services/OrderService.java",
    )

    # Low-authority auxiliary block
    aux_block = ContextBlock(
        label="Planner Guidance",
        content="Consider refactoring OrderService.\n" * 100,
        source="planner",
        authority=AUTHORITY_LEVELS["planner"],
        stage="task_metadata",
    )

    budget = ContextBudget(max_tokens=8000, max_chars_per_block=5000, max_target_file_chars=45000)
    assembled = assembler._stage_final_assembly([target_block, aux_block], task, budget)

    # Target file must be fully preserved without truncation
    assert "[Target File (OrderService.java)]" in assembled
    assert "[truncated to per-block budget]" not in assembled
    assert "public class OrderService" in assembled
    assert len(assembled) > 20000
