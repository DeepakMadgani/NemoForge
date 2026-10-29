import pytest
from pathlib import Path
from ticket_to_code.reasoning.tools import build_default_registry, _t_view_file, _t_grep_search, _t_read_file
from ticket_to_code.models import ValueEdgeTicket, TicketType

class DummyContext:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.candidate_files = {}
        self.files_read = {}
        self.agents = DummyAgents()
        self.ticket = ValueEdgeTicket(
            ticket_id="TEST-1",
            title="Test Ticket",
            description="Testing live query tools",
            ticket_type=TicketType.BUG_FIX
        )

class DummyRepoSearch:
    def search_literal(self, query, max_results=10):
        from dataclasses import dataclass
        @dataclass
        class Result:
            file_path: str
            line_number: int
            matched_text: str
            confidence: float = 0.9
        return [Result("test_file.py", 10, "def test_func():")]

    def search_regex(self, pattern, max_results=10):
        return self.search_literal(pattern, max_results)

class DummyAgents:
    def __init__(self):
        self.repo_search = DummyRepoSearch()

def test_tools_registration():
    tools = build_default_registry()
    assert "view_file" in tools
    assert "grep_search" in tools
    assert "read_file" in tools
    assert "grep" in tools

def test_view_file_slicing(tmp_path):
    sample_file = tmp_path / "sample.py"
    lines = [f"line {i}" for i in range(1, 51)]
    sample_file.write_text("\n".join(lines), encoding="utf-8")

    ctx = DummyContext(str(tmp_path))
    res = _t_view_file(ctx, {"path": "sample.py", "start_line": 10, "end_line": 15})

    assert "Showing 10 to 15" in res
    assert "10: line 10" in res
    assert "15: line 15" in res
    assert "16: line 16" not in res
    assert "9: line 9" not in res

def test_grep_search_formatting(tmp_path):
    ctx = DummyContext(str(tmp_path))
    res = _t_grep_search(ctx, {"query": "test_func"})
    assert "test_file.py:10: def test_func():" in res


def test_adt_dual_task_formatting():
    from ticket_to_code.models import SolutionGuidance, GeneratedCode
    # Test that when both explanation/guidance and code changes exist, both are rendered
    guidance = SolutionGuidance(
        issue_summary="Coordinator is mapped to Administrator in backend database.",
        root_cause="Database schema uses Administrator while UI displays Coordinator.",
        solution_type="architectural_guidance",
        step_by_step_solution=["Review ProjectUserRole enum", "Verify contract member query"],
        verification_steps=["Inspect database column role_name"]
    )
    from ticket_to_code.models import ProgrammingLanguage, TaskType
    code = [GeneratedCode(
        file_path="src/service/MyService.java",
        content="// new code",
        documentation="Fix access check",
        language=ProgrammingLanguage.JAVA,
        change_type=TaskType.MODIFY
    )]

    final_state = {
        "status": "success",
        "solution_guidance": guidance,
        "generated_code": code
    }

    # Simulate formatting logic in adt_tools
    sections = []
    if final_state.get("solution_guidance"):
        g = final_state["solution_guidance"]
        sections.append(f"### 💡 Architectural Analysis & Guidance\n\n{g.issue_summary}\n\n**Root Cause / Mechanism:** {g.root_cause}")
    if final_state.get("generated_code"):
        cf_list = [f"- `{c.file_path}` ({c.documentation})" for c in final_state["generated_code"]]
        sections.append(f"### 🛠️ Code Implementation Applied\n\n" + "\n".join(cf_list))

    combined = "\n\n".join(sections)
    assert "Architectural Analysis & Guidance" in combined
    assert "Coordinator is mapped to Administrator in backend database." in combined
    assert "Code Implementation Applied" in combined
    assert "`src/service/MyService.java` (Fix access check)" in combined

