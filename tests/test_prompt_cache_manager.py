"""
Unit tests for Prompt Cache Prefix Stabilization and KV-cache Optimization.
"""
from ticket_to_code.intelligence.prompt_cache_manager import (
    PromptCacheManager,
    StablePromptStructure
)


def test_tool_schema_deterministic_sorting():
    # Tools in order A
    tools_a = [
        {"name": "view_file", "description": "Read file"},
        {"name": "apply_patch", "description": "Write diff"},
        {"name": "list_dir", "description": "List directory"}
    ]
    struct_a = StablePromptStructure(
        system_instruction="Static instruction",
        tool_schemas=tools_a
    )
    res_a = PromptCacheManager.build_cache_stable_prompt(struct_a)

    # Tools in order B (reversed)
    tools_b = list(reversed(tools_a))
    struct_b = StablePromptStructure(
        system_instruction="Static instruction",
        tool_schemas=tools_b
    )
    res_b = PromptCacheManager.build_cache_stable_prompt(struct_b)

    # Both must result in identically sorted tools
    assert [t["name"] for t in res_a["tools"]] == ["apply_patch", "list_dir", "view_file"]
    assert [t["name"] for t in res_b["tools"]] == ["apply_patch", "list_dir", "view_file"]
    assert res_a["prefix_hash"] == res_b["prefix_hash"]


def test_volatile_scratchpad_placed_at_tail():
    struct = StablePromptStructure(
        system_instruction="Instructions",
        conversation_turns=[
            {"role": "user", "content": "Initial request"},
            {"role": "assistant", "content": "Inspecting..."},
            {"role": "user", "content": "Follow up"}
        ],
        live_scratchpad="COMPILER ERROR: TS2307 Cannot find module"
    )
    res = PromptCacheManager.build_cache_stable_prompt(struct)

    # The volatile compiler error must be appended strictly at the tail of the final message
    last_msg = res["messages"][-1]
    assert "COMPILER ERROR: TS2307" in last_msg["content"]
    assert last_msg["content"].endswith("COMPILER ERROR: TS2307 Cannot find module")


def test_prefix_hash_stability():
    struct1 = StablePromptStructure(
        system_instruction="Fixed system rule.",
        repo_skeletons=[{"path": "a.ts", "skeleton": "export class A {}"}],
        episodic_memories=["Fix: always import CommonModule in Angular"]
    )
    struct2 = StablePromptStructure(
        system_instruction="Fixed system rule.",
        repo_skeletons=[{"path": "a.ts", "skeleton": "export class A {}"}],
        episodic_memories=["Fix: always import CommonModule in Angular"],
        live_scratchpad="Volatile error at t=100s"
    )

    res1 = PromptCacheManager.build_cache_stable_prompt(struct1)
    res2 = PromptCacheManager.build_cache_stable_prompt(struct2)

    # Static prefix hash MUST be identical even when live scratchpad differs!
    assert res1["prefix_hash"] == res2["prefix_hash"]
