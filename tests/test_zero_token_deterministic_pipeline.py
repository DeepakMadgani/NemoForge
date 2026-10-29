"""
Unit tests for Zero-Token Deterministic Pipeline & Compiler-as-Truth Early Exit.

Verifies:
1. investigate_node performs deterministic zero-token triage on code tickets.
2. unified_analysis_node extracts structured requirements with 100% fidelity from acceptance criteria without LLM calls.
3. outcome_check_node performs compiler-as-truth early exit in Direct AI IDE Mode when build passes cleanly.
4. edit_loop_agent lowers windowing threshold for verbatim methods to prevent prompt token bloat.
"""

import pytest
from unittest.mock import MagicMock, patch
from ticket_to_code.models import ValueEdgeTicket, TicketType, InvestigationResult, StructuredRequirements, GeneratedCode
from ticket_to_code.workflow import investigate_node, unified_analysis_node, outcome_check_node


def test_investigate_node_deterministic_triage():
    ticket = ValueEdgeTicket(
        id="TICK-001",
        ticket_id="TICK-001",
        title="Fix hover tooltip on member action button",
        description="The action button tooltip should show 'Remove member' when hovered.",
        acceptance_criteria=["Tooltip displays on hover", "Button triggers remove modal"],
    )
    state = {
        "ticket": ticket,
        "workspace_path": "C:/dummy/workspace",
    }
    agents = MagicMock()

    result = investigate_node(state, agents)
    inv = result["investigation_result"]

    assert inv.requires_code_changes is True
    assert inv.ticket_type == TicketType.BUG_FIX
    assert inv.confidence == 1.0
    assert "Fix hover tooltip" in inv.goal
    assert result["status"] == "investigation_complete"
    # Verify legacy LLM investigation agent was NOT called
    agents.investigation.investigate.assert_not_called()


def test_investigate_node_feature_triage():
    ticket = ValueEdgeTicket(
        id="TICK-002",
        ticket_id="TICK-002",
        title="Add export to CSV button in table toolbar",
        description="Implement CSV export for all filtered rows in the table.",
        acceptance_criteria=["Export button visible in toolbar", "Clicking downloads CSV"],
    )
    state = {
        "ticket": ticket,
        "workspace_path": "C:/dummy/workspace",
    }
    agents = MagicMock()

    result = investigate_node(state, agents)
    inv = result["investigation_result"]

    assert inv.requires_code_changes is True
    assert inv.ticket_type == TicketType.FEATURE_REQUEST
    assert inv.confidence == 1.0
    agents.investigation.investigate.assert_not_called()


def test_unified_analysis_node_deterministic_requirements():
    ticket = ValueEdgeTicket(
        id="TICK-003",
        ticket_id="TICK-003",
        title="Fix member status display",
        description="Ensure active and inactive members show correct badge colors.",
        acceptance_criteria=[
            "Active members display green badge",
            "Inactive members display grey badge",
        ],
    )
    inv = InvestigationResult(
        ticket_type=TicketType.BUG_FIX,
        requires_code_changes=True,
        root_cause_hypothesis="Badge styling mismatch",
        recommended_action="Update SCSS",
    )
    state = {
        "ticket": ticket,
        "investigation_result": inv,
        "workspace_path": "C:/dummy/workspace",
    }
    agents = MagicMock()

    result = unified_analysis_node(state, agents)
    reqs: StructuredRequirements = result["requirements"]

    assert len(reqs.functional_requirements) == 2
    assert "Active members display green badge" in reqs.functional_requirements
    assert "Inactive members display grey badge" in reqs.functional_requirements
    assert result["status"] == "analysis_complete"
    # Verify legacy LLM analyzer was NOT called
    agents.analyzer.analyze_ticket.assert_not_called()


def test_outcome_check_node_direct_mode_compiler_as_truth():
    ticket = ValueEdgeTicket(
        id="TICK-004",
        ticket_id="TICK-004",
        title="Update member status styling",
        description="Update badge colors.",
    )
    gen_file = GeneratedCode(
        file_path="src/app/member/member.component.scss",
        content=".badge { color: green; }",
        language="scss",
        change_type="modify",
    )
    state = {
        "ticket": ticket,
        "direct_mode": True,
        "build_status": "success",
        "build_errors": [],
        "generated_code": [gen_file],
        "allowed_files": ["src/app/member/member.component.scss"],
        "workspace_path": "C:/dummy/workspace",
    }
    agents = MagicMock()

    with patch("ticket_to_code.llm_utils.llm_invoke") as mock_llm_invoke:
        result = outcome_check_node(state, agents)
        
        # Verify 0 LLM calls were made
        mock_llm_invoke.assert_not_called()
        
        # Verify verdict is CORRECT via compiler-as-truth
        check_res = result["outcome_check_result"]
        assert check_res["status"] == "CORRECT"
        assert "Direct AI IDE Mode" in check_res["details"]
        assert len(result["unsatisfied_requirements"]) == 0
        assert result["outcome_remediation"] is None


def test_edit_loop_target_replacement_parser(tmp_path):
    from ticket_to_code.agents.edit_loop_agent import EditLoopAgent
    agent = EditLoopAgent(workspace_path=str(tmp_path), llm=MagicMock())

    original_code = "def greet(name):\n    print('Hello ' + name)\n    return True\n"
    llm_response = """SUMMARY: Fix greeting to include exclamation mark.

TARGET:
    print('Hello ' + name)
REPLACEMENT:
    print(f'Hello {name}!')
"""
    result = agent._apply_search_replace(original_code, llm_response, is_create=False)
    assert result.success is True
    assert result.parser_format == "target_replacement"
    assert "print(f'Hello {name}!')" in result.content
    assert "def greet(name):" in result.content
    assert "return True" in result.content


def test_live_compile_file_telemetry_and_py_compile(tmp_path):
    from ticket_to_code.agents.edit_loop_agent import EditLoopAgent

    events = []
    def mock_ui_cb(evt):
        events.append(evt)

    agent = EditLoopAgent(workspace_path=str(tmp_path), llm=MagicMock(), ui_callback=mock_ui_cb)

    # 1. Write clean python file
    clean_file = tmp_path / "valid.py"
    clean_file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    agent._written_files["valid.py"] = clean_file.read_text(encoding="utf-8")

    errors = agent._live_compile_file("valid.py")
    assert len(errors) == 0
    assert any(e.get("event_type") == "live_compile_passed" for e in events)

    # 2. Write invalid python file (syntax error)
    bad_file = tmp_path / "syntax_err.py"
    bad_file.write_text("def broken(:\n    pass\n", encoding="utf-8")
    agent._written_files["syntax_err.py"] = bad_file.read_text(encoding="utf-8")

    events.clear()
    errors = agent._live_compile_file("syntax_err.py")
    assert len(errors) > 0
    assert any(e.get("event_type") == "live_compile_failed" for e in events)


def test_code_generator_target_replacement_and_commenting():
    from ticket_to_code.agents.code_generator import CodeGeneratorAgent
    from ticket_to_code.models import DevelopmentTask, TaskType, ProgrammingLanguage

    generator = CodeGeneratorAgent(llm=MagicMock())
    task = DevelopmentTask(
        id="task_1",
        title="Update calculate total",
        description="Update tax calculation",
        file_path="src/calc.py",
        task_type=TaskType.MODIFY,
        language=ProgrammingLanguage.PYTHON,
    )
    generator._current_existing_content = """def calculate(price, tax):
    total = price + tax
    return total
"""

    # Test TARGET / REPLACEMENT with deprecated code commented out
    llm_output = """<<<AVIATOR_CODE_START>>>
TARGET:
    total = price + tax
REPLACEMENT:
    # legacy: total = price + tax
    total = price * (1 + tax)
<<<AVIATOR_CODE_END>>>
<<<AVIATOR_META_START>>>
{"imports": [], "documentation": "Updated tax formula"}
<<<AVIATOR_META_END>>>
"""
    result = generator._parse_response(llm_output, task)
    assert "# legacy: total = price + tax" in result.content
    assert "total = price * (1 + tax)" in result.content
    assert "def calculate(price, tax):" in result.content
    assert "return total" in result.content


def test_local_code_syntax_checker():
    from ticket_to_code.workflow import _check_code_syntax

    # 1. Clean code with comments and strings containing braces
    clean_ts = """
    import { Component } from '@angular/core';
    // Comment with { and (
    /* Multi-line
       comment { }
    */
    @Component({
        selector: 'app-root',
        template: '<div (click)="run()">Text {value}</div>'
    })
    export class AppComponent {
        calc(a: number, b: number): number {
            return (a + b);
        }
    }
    """
    errs = _check_code_syntax(clean_ts, "app.component.ts")
    assert len(errs) == 0

    # 2. Unbalanced braces
    broken_braces = """
    export class AppComponent {
        calc() { return 1;
    """
    errs = _check_code_syntax(broken_braces, "broken.ts")
    assert any("Unbalanced curly braces" in e for e in errs)

    # 3. Leaked format marker
    leaked_marker = """
    export class AppComponent {
<<<<<<< SEARCH
        oldCode();
=======
        newCode();
>>>>>>> REPLACE
    }
    """
    errs = _check_code_syntax(leaked_marker, "leaked.ts")
    assert any("Leaked LLM format marker" in e for e in errs)


def test_patch_gate_node_proportionality_check(tmp_path):
    from ticket_to_code.workflow import patch_gate_node
    from ticket_to_code.models import GeneratedCode, ProgrammingLanguage, TaskType

    # Create original 100-line file
    target_file = tmp_path / "service.ts"
    orig_lines = [f"line_{i} = {i};" for i in range(100)]
    target_file.write_text("\n".join(orig_lines), encoding="utf-8")

    # A disproportionate SEARCH block covering 95 lines (>90%)
    disproportionate_patch = f"""<<<<<<< SEARCH\n{chr(10).join(orig_lines[:95])}\n=======\nreplacement = true;\n>>>>>>> REPLACE"""

    gen = GeneratedCode(
        file_path="service.ts",
        content=disproportionate_patch,
        language=ProgrammingLanguage.TYPESCRIPT,
        change_type=TaskType.MODIFY,
    )

    state = {
        "workspace_path": str(tmp_path),
        "generated_code": [gen],
        "original_file_contents": {"service.ts": "\n".join(orig_lines)},
        "allowed_files": ["service.ts"],
    }
    agents = MagicMock()

    result = patch_gate_node(state, agents)
    assert result.get("status") == "candidates_invalid"
    assert "proportionality_violation" in result.get("validation_failure_reason", "")


def test_edit_loop_companion_deferral_and_angular_auto_registration(tmp_path):
    from ticket_to_code.agents.edit_loop_agent import EditLoopAgent, EditDecision

    agent = EditLoopAgent(workspace_path=str(tmp_path), llm=MagicMock())

    # Create a module file
    module_file = tmp_path / "app.module.ts"
    module_file.write_text("""import { NgModule } from '@angular/core';

@NgModule({
  declarations: [],
  imports: [],
  providers: []
})
export class AppModule {}
""", encoding="utf-8")

    # 1. When companion template is in pending queue, tsc is deferred
    agent._pending_queue_ref = [
        EditDecision(file_path="user.component.html", reason="Template", edit_description="Template", is_new_file=True)
    ]
    agent._written_files["user.component.ts"] = """import { Component } from '@angular/core';
@Component({ selector: 'app-user', templateUrl: './user.component.html' })
export class UserComponent {}
"""
    # Writing user.component.ts should defer cross-file tsc
    errs = agent._verify_file("user.component.ts")
    assert len(errs) == 0  # Deferred without calling symbol_index.get_ts_diagnostics

    # 2. Verify auto-registration in NgModule
    agent._pending_queue_ref = []  # Companion done
    agent._auto_register_angular_module_for_file("user.component.ts", agent._written_files["user.component.ts"])
    updated_mod = module_file.read_text(encoding="utf-8")
    assert "UserComponent" in updated_mod
    assert "import { UserComponent } from './user.component';" in updated_mod


def test_workflow_graph_wiring_post_plan_pipeline():
    from ticket_to_code.workflow import create_ticket_to_code_graph

    graph = create_ticket_to_code_graph("C:/dummy/workspace", technology="python")
    # Verify fix_build routes through patch_gate
    # LangGraph edges inspection
    edges = [(edge.source, edge.target) for edge in graph.builder.edges if hasattr(edge, 'source')] if hasattr(graph, 'builder') else [(src, tgt) for src, tgt in getattr(graph, 'edges', [])]
    assert any(tgt == "patch_gate" for src, tgt in edges if src == "fix_build")


def test_investigate_node_informational_question_triage():
    from ticket_to_code.models import ValueEdgeTicket, TicketType
    from ticket_to_code.workflow import investigate_node, route_after_unified_analysis
    from langgraph.graph import END

    ticket = ValueEdgeTicket(
        id="TASK-Q1",
        ticket_id="TASK-Q1",
        title="What microservices make up CloudForge and how do they communicate?",
        description="What microservices make up CloudForge and how do they communicate?",
    )
    state = {
        "ticket": ticket,
        "workspace_path": "C:/dummy/workspace",
    }
    agents = MagicMock()

    result = investigate_node(state, agents)
    inv = result["investigation_result"]

    assert inv.requires_code_changes is False
    assert inv.ticket_type == TicketType.USAGE_QUESTION
    assert "Informational / architectural inquiry" in inv.explanation

    # Verify route_after_unified_analysis routes straight to END
    route_state = {
        "investigation_result": inv,
        "solution_guidance": MagicMock(),
        "requirements": None,
    }
    assert route_after_unified_analysis(route_state) == END
