import pytest
from unittest.mock import MagicMock, patch
from ticket_to_code.workflow import (
    _group_tasks_into_batches,
    context_expand_node,
    validate_candidates_node
)
from ticket_to_code.agents.code_generator import CodeGeneratorAgent
from ticket_to_code.runtime.run_context import RunContext, RunBudget

class MockTask:
    def __init__(self, id, file_path, dependencies=None):
        self.id = id
        self.file_path = file_path
        self.dependencies = dependencies or []
        self.task_type = MagicMock(value="create")
        self.title = "Mock title"
        self.description = "Mock description"

def test_batch_grouping_independent():
    """Test that independent tasks are grouped together in Batch 0."""
    tasks = [
        MockTask("t1", "file1.ts"),
        MockTask("t2", "file2.ts"),
        MockTask("t3", "file3.ts")
    ]
    batches = _group_tasks_into_batches(tasks)
    assert len(batches) == 1
    assert len(batches[0]) == 3
    assert set(t.id for t in batches[0]) == {"t1", "t2", "t3"}

def test_batch_grouping_dependent():
    """Test that dependent tasks are separated and ordering is preserved."""
    # A (t1), B (t2) -> C (t3) -> D (t4)
    t1 = MockTask("t1", "file1.ts")
    t2 = MockTask("t2", "file2.ts")
    t3 = MockTask("t3", "file3.ts", dependencies=["t1", "t2"])
    t4 = MockTask("t4", "file4.ts", dependencies=["t3"])
    
    tasks = [t1, t2, t3, t4]
    batches = _group_tasks_into_batches(tasks)
    
    assert len(batches) == 3
    # Batch 0: A and B
    assert len(batches[0]) == 2
    assert set(t.id for t in batches[0]) == {"t1", "t2"}
    # Batch 1: C
    assert len(batches[1]) == 1
    assert batches[1][0].id == "t3"
    # Batch 2: D
    assert len(batches[2]) == 1
    assert batches[2][0].id == "t4"

from ticket_to_code.workflow import plan_node

@patch('ticket_to_code.workflow.write_trace_artifact')
def test_planner_contract_completeness_no_gap(mock_write_trace):
    """Test that when a consumed capability is produced by another task, no gap is reported."""
    plan = MagicMock()

    t1 = MagicMock(id="t1", file_path="f1.ts", dependencies=[])
    t1.task_type = MagicMock(value="create")
    t1.cross_file_contract = {"consumes": [], "produces": [{"capability": "isProjectMember"}]}
    
    t2 = MagicMock(id="t2", file_path="f2.ts", dependencies=["t1"])
    t2.task_type = MagicMock(value="create")
    t2.cross_file_contract = {"consumes": [{"capability": "isProjectMember"}], "produces": []}
    
    plan.tasks = [t1, t2]
    
    agents = MagicMock()
    # Ensure create_plan returns our mock plan, or we can just mock out everything before it.
    # Actually, plan_node reads from state["architectural_plan"] if it's already there? No, it overwrites it.
    agents.planner.create_plan.return_value = plan
    
    state = {
        "status": "in_progress",
        "ticket": MagicMock(ticket_id="ticket123"),
        "workspace_path": "/mock/path",
        "discovered_files": [{"path": "f1.ts", "semantic_relevance_score": 0.9}],
        "requirements": MagicMock(),
    }
    
    result = plan_node(state, agents)
    assert "contract_gaps" not in result or result["contract_gaps"] is None or len(result["contract_gaps"]) == 0

@patch('ticket_to_code.workflow.write_trace_artifact')
def test_planner_contract_completeness_gap_detected(mock_write_trace):
    """Test that when a consumed capability is NOT produced, a gap is detected."""
    plan = MagicMock()
    t1 = MagicMock(id="t1", file_path="f1.ts", dependencies=[])
    t1.task_type = MagicMock(value="create")
    t1.cross_file_contract = {"consumes": [], "produces": [{"capability": "someOtherFunction"}]}
    
    t2 = MagicMock(id="t2", file_path="f2.ts", dependencies=["t1"])
    t2.task_type = MagicMock(value="create")
    t2.cross_file_contract = {"consumes": [{"capability": "isProjectMember"}], "produces": []}
    
    plan.tasks = [t1, t2]
    
    agents = MagicMock()
    agents.planner.create_plan.return_value = plan
    
    state = {
        "status": "in_progress",
        "ticket": MagicMock(ticket_id="ticket123"),
        "workspace_path": "/mock/path",
        "discovered_files": [{"path": "f1.ts", "semantic_relevance_score": 0.9}],
        "requirements": MagicMock(),
    }
    
    result = plan_node(state, agents)
    assert "contract_gaps" in result and result["contract_gaps"] is not None
    assert len(result["contract_gaps"]) == 1
    assert result["contract_gaps"][0]["capability"] == "isprojectmember"

def test_second_sufficiency_routing_insufficient():
    """Test that if the combined evidence lacks keywords, it assesses INSUFFICIENT and routes to Tier 2."""
    state = {
        "outcome_check_result": {
            "unsatisfied_requirements": ["MUST implement the isProjectMember function"]
        },
        "discovered_files": [], # Start empty so Step 1 promotes
        "evidence_items": [
            MagicMock(file_path="unrelated.ts", facts="just some unrelated stuff")
        ],
        "tier2_candidates": [
            {"path": "isprojectmember_service.ts"}
        ]
    }
    
    result = context_expand_node(state, MagicMock())
    
    # 'unrelated.ts' is NOT promoted in Step 1 because it doesn't match the unsatisfied keywords!
    # So promoted=0. Thus, Step 2 runs (but we don't mock it so it finds nothing).
    # Step 3 evaluates INSUFFICIENT because promoted=0 and new_files=0.
    # Therefore, Tier 2 fallback runs and promotes isprojectmember_service.ts (semantically verified)
    assert len(result["discovered_files"]) == 1
    assert result["discovered_files"][0]["path"] == "isprojectmember_service.ts"
    assert result["discovered_files"][0]["_tier"] == "verified_candidate_promoted"

def test_second_sufficiency_routing_sufficient():
    """Test that if the combined evidence HAS keywords, it assesses LIKELY_SUFFICIENT and skips Tier 2."""
    state = {
        "outcome_check_result": {
            "unsatisfied_requirements": ["MUST implement the isProjectMember function"]
        },
        "discovered_files": [], # Start empty so Step 1 promotes
        "evidence_items": [
            MagicMock(file_path="isprojectmember_function.ts", facts="implements the isProjectMember function logic", candidate_role="function")
        ],
        "tier2_candidates": [
            {"path": "other.ts"}
        ]
    }
    
    result = context_expand_node(state, MagicMock())
    
    # Step 1 matches the keywords and promotes project.service.ts
    # Step 3 evaluates it, sees keywords match, and assesses LIKELY_SUFFICIENT/PARTIALLY_COVERED
    # Therefore, Tier 2 fallback DOES NOT run.
    assert len(result["discovered_files"]) == 1
    assert result["discovered_files"][0]["path"] == "isprojectmember_function.ts"
    assert result["discovered_files"][0]["_tier"] == "promoted_from_evidence"
    
def test_per_file_token_accounting():
    """Test that token telemetry correctly snapshots and computes the delta."""
    with patch.object(CodeGeneratorAgent, '__init__', lambda self: None):
        generator = CodeGeneratorAgent()
        task = MagicMock()
        task.file_path = "test.ts"
        task.task_type.value = "create"
        task.language.value = "typescript"
        task.dependencies = []
        task.title = "test title"
        task.description = "test description"
        task.id = "t1"
        task.target_class = "TestClass"
        task.target_method = "testMethod"
        task.allowed_methods = []
        task.edit_anchors = []
        
        context_data = []
        existing_content = ""
        requirements = MagicMock()
        requirements.functional_requirements = []
        requirements.technical_requirements = []
        requirements.edge_cases = []
        
        # Mock the internal prompt component builders
        generator._build_session_context = MagicMock(return_value="session data")
        generator._build_contract_context = MagicMock(return_value="contract data")
        generator._build_import_context = MagicMock(return_value="")
        generator._format_list = MagicMock(return_value="list")
        generator._format_context = MagicMock(return_value="")
        
        # Need to provide some minimal attributes to avoid errors during prompt build
        generator._symbol_resolver = None
        generator._workspace_path = None
        
        # Build prompt
        prompt = generator._build_user_prompt(
            task=task,
            requirements=requirements,
            context=context_data,
            existing_content=existing_content
        )
        
        # Verify telemetry attributes are set correctly
        assert generator._last_session_context == "session data"
        assert generator._last_contract_context == "contract data"
        assert hasattr(generator, "_last_impl_context")

def test_new_file_count_nonzero_after_evidence_promotion():
    """Verify the aliasing fix: _new_file_count must reflect actual additions, not always 0."""
    # Start with 1 existing discovered file
    initial_discovered = [{"path": "existing.ts", "semantic_relevance_score": 0.9}]
    state = {
        "outcome_check_result": {
            "unsatisfied_requirements": ["MUST implement the isProjectMember function"]
        },
        "discovered_files": initial_discovered,
        "evidence_items": [
            # This evidence file matches "isprojectmember" keyword from the unsatisfied requirement
            MagicMock(file_path="project.service.ts", facts="isProjectMember function logic")
        ],
        "tier2_candidates": []
    }
    
    result = context_expand_node(state, MagicMock())
    
    # The evidence file should be promoted, so we have 2 discovered files now
    assert len(result["discovered_files"]) == 2
    paths = [f.get("path") for f in result["discovered_files"]]
    assert "existing.ts" in paths
    assert "project.service.ts" in paths
    
    # Crucially: the original state's discovered_files must NOT have been mutated
    # (this proves the shallow copy fix works)
    assert len(initial_discovered) == 1, (
        f"Original list was mutated (len={len(initial_discovered)}), aliasing bug still present"
    )

