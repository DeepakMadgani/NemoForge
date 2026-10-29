"""Unit tests for CodeGeneratorAgent and PlanningAgent workspace_path robustness.

Verifies:
1. Universal safe access to workspace_path and _workspace_path across all initialization pathways.
2. Property synchronization: assigning to workspace_path updates _workspace_path, and vice versa.
3. Resilience against __new__ bypasses and uninitialized instances.
4. _build_session_context companion resolution for HTML templates without AttributeError.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ticket_to_code.agents.code_generator import CodeGeneratorAgent
from ticket_to_code.agents.planning_agent import PlanningAgent
from ticket_to_code.models import DevelopmentTask, TaskType


def test_code_generator_workspace_path_defaults_and_property_sync():
    """Verify CodeGeneratorAgent safely defaults workspace_path and keeps both attributes in sync."""
    # Test 1: Instantiation without args
    gen = CodeGeneratorAgent.__new__(CodeGeneratorAgent)
    # Bypassing __init__ completely must not raise AttributeError
    assert gen.workspace_path is None
    assert getattr(gen, "_workspace_path", None) is None
    assert str(gen.workspace_path or "") == ""

    # Test 2: Setting workspace_path via property
    gen.workspace_path = "/custom/project/root"
    assert gen._workspace_path == "/custom/project/root"
    assert gen.workspace_path == Path("/custom/project/root")

    # Test 3: Setting _workspace_path directly updates workspace_path property
    gen._workspace_path = "/direct/update"
    assert gen.workspace_path == Path("/direct/update")

    # Test 4: Setting workspace_path to None
    gen.workspace_path = None
    assert gen._workspace_path is None
    assert gen.workspace_path is None


def test_code_generator_init_with_workspace_path():
    """Verify CodeGeneratorAgent properly stores workspace_path on __init__."""
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("ticket_to_code.agents.code_generator.LLMRegistry.get_llm", lambda assistant=True: MagicMock())
        gen = CodeGeneratorAgent("/test/ws")
        assert gen.workspace_path == Path("/test/ws")
        assert gen._workspace_path == "/test/ws"


def test_code_generator_html_session_context_resolution_no_attribute_error():
    """Verify _build_session_context for .html tasks resolves companions safely without AttributeError."""
    gen = CodeGeneratorAgent.__new__(CodeGeneratorAgent)
    gen.workspace_path = "/mock/workspace"

    # Sibling typescript component in session_files
    session_files = {
        "xchange-ui/src/app/modules/members/add-members/add-members.component.ts": """
import { Component } from '@angular/core';

@Component({
  selector: 'app-add-members',
  templateUrl: './add-members.component.html'
})
export class AddMembersComponent {
  isExistingMember: boolean = false;
  existingOrganizationName: string = '';
}
"""
    }

    gen._session_files = session_files

    task = DevelopmentTask(
        id="task-2",
        title="Update HTML template",
        description="Render organization name",
        file_path="xchange-ui/src/app/modules/members/add-members/add-members.component.html",
        task_type=TaskType.MODIFY,
        language="html",
    )

    # Calling _build_session_context must not raise AttributeError
    ctx = gen._build_session_context("<div>original</div>", task)
    assert ctx is not None
    assert "isExistingMember" in ctx
    assert "existingOrganizationName" in ctx


def test_planning_agent_workspace_path_property_sync():
    """Verify PlanningAgent safely exposes and synchronizes workspace_path."""
    planner = PlanningAgent.__new__(PlanningAgent)
    assert planner.workspace_path is None

    planner.workspace_path = "/planning/ws"
    assert planner._workspace_path == "/planning/ws"
    assert planner.workspace_path == Path("/planning/ws")

    planner._workspace_path = "/another/ws"
    assert planner.workspace_path == Path("/another/ws")

    planner.workspace_path = None
    assert planner._workspace_path is None
    assert planner.workspace_path is None


def test_code_generator_resilient_diff_recovery_on_full_file_output():
    """Verify that when an LLM outputs a full-file rewrite instead of SEARCH/REPLACE
    blocks on a MODIFY task, Resilient Diff Recovery extracts the surgical diff
    and patches the file without throwing an error or wasting a retry.
    """
    gen = CodeGeneratorAgent.__new__(CodeGeneratorAgent)
    gen.workspace_path = "/mock/ws"

    original_html = """<div class="container">
  <h1>Add Members</h1>
  <ot-item-select #organization [options]="organizationItemSelect.options"></ot-item-select>
  <button (click)="save()">Save</button>
</div>"""

    # LLM mistakenly returned the whole file with just 1 modification instead of SEARCH/REPLACE delimiters
    llm_full_output = """<div class="container">
  <h1>Add Members</h1>
  <ot-item-select *ngIf="!isExistingMember" #organization [options]="organizationItemSelect.options"></ot-item-select>
  <div *ngIf="isExistingMember">{{ existingOrganizationName }}</div>
  <button (click)="save()">Save</button>
</div>"""

    task = DevelopmentTask(
        id="task-html",
        title="Update HTML",
        description="Add isExistingMember check",
        file_path="src/app/add-members.component.html",
        task_type=TaskType.MODIFY,
        language="html",
    )

    gen._current_existing_content = original_html

    # Simulated LLM response in delimiter format without <<<<<<< SEARCH delimiters
    response_text = f"<<<AVIATOR_CODE_START>>>\n{llm_full_output}\n<<<AVIATOR_CODE_END>>>\n<<<AVIATOR_META_START>>>\n{{\"imports\": [], \"documentation\": \"Added check\"}}\n<<<AVIATOR_META_END>>>"

    result = gen._parse_response(response_text, task)

    # Must have recovered cleanly and applied the change!
    assert result is not None
    assert "*ngIf=\"!isExistingMember\"" in result.content
    assert "existingOrganizationName" in result.content
    assert "<div class=\"container\">" in result.content


