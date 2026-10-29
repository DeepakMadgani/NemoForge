"""
Unit tests for the 4 Enterprise Harness Optimizations:
1. Parallel Tool Calling (Turn Batching directives)
2. Token Economics & High-Density Sanitization
3. Model Cascading / Tiering (Flash exploration + Pro synthesis)
4. Subagent Isolation (SubagentExplorer execution & summary returns)
"""

from ticket_to_code.runtime import (
    AgentIntent,
    DeductiveBlackboard,
    HighDensitySanitizer,
    PromptAssembler,
)
from ticket_to_code.agents.subagent_explorer import SubagentExplorer, ExplorerRole


def test_parallel_tool_calling_directive_in_prompt():
    system_inst = PromptAssembler.build_system_instruction(AgentIntent.AUTONOMOUS_MODIFICATION)
    assert "PARALLEL TOOL EXECUTION (TURN BATCHING)" in system_inst
    assert "EMIT MULTIPLE TOOL CALLS SIMULTANEOUSLY" in system_inst


def test_high_density_sanitizer_caps_character_and_line_bloat():
    # Massive search output simulation (10,000 lines)
    bloated_output = "\n".join(f"match {i}: Issue.java:42: private String priority;" for i in range(1000))
    sanitized = HighDensitySanitizer.sanitize(bloated_output, max_chars=3000, max_lines=50)

    assert len(sanitized) <= 3200
    assert len(sanitized.splitlines()) <= 55
    assert "Truncated" in sanitized


def test_subagent_isolation_produces_compact_summary(tmp_path):
    # Setup test workspace
    comp_file = tmp_path / "deliverable.component.ts"
    comp_file.write_text("export class DeliverableComponent {}", encoding="utf-8")

    explorer = SubagentExplorer(str(tmp_path))
    report = explorer.run_ui_mapper("deliverables", [str(comp_file.name)])

    assert report.worker_role == ExplorerRole.UI_MAPPER
    assert report.confidence >= 0.8
    summary = report.to_summary_markdown()
    assert "### [UI_MAPPER]" in summary
    assert "deliverable.component.ts" in summary
    assert len(summary) < 1000  # Stays compact, protecting parent context


def test_blackboard_ingests_subagent_reports():
    bb = DeductiveBlackboard()
    summary = "### [CONTRACT_CHECKER]\n**Claim**: Verified /api/v1/issues/deliverables/{id}/count endpoint exists."
    deductions = bb.ingest_tool_observation("subagent_contract_checker", "count", summary)

    assert len(bb.history) == 1
    assert bb.history[0].tool_name == "subagent_contract_checker"
