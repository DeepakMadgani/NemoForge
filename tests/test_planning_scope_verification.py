import pytest
from unittest.mock import MagicMock
from pathlib import Path

from ticket_to_code.models import VerificationStatus
from ticket_to_code.agents.planning_scope_verifier import PlanningScopeVerifier


@pytest.fixture
def mock_agents():
    agents = MagicMock()
    agents.workspace_path = MagicMock()
    agents.localizer.sqlite_store.search_symbols.return_value = []
    agents.repo_search.search_literal.return_value = []
    agents.relationship_provider.get_relationships_for_file.return_value = []
    return agents


def test_existing_capability_discovered(mock_agents):
    """Test 1: Existing capability discovered -> VERIFIED_EXISTS"""
    verifier = PlanningScopeVerifier(mock_agents)
    
    # Mocking SQLite to return a hit
    mock_agents.localizer.sqlite_store.search_symbols.return_value = [{"path": "backend/project_service.py", "name": "ProjectMembershipService"}]
    # Mock file existence
    mock_agents.workspace_path.joinpath.return_value.exists.return_value = True
    # mock_agents.workspace_path / ... also handled via MagicMock
    mock_agents.workspace_path.__truediv__.return_value.exists.return_value = True
    
    guidance = [{"requirement": "Members", "what_to_do": "Create a new backend endpoint for project membership"}]
    
    results = verifier.verify_guidance(guidance, [], [])
    
    assert len(results) == 1
    assert results[0].status == VerificationStatus.VERIFIED_EXISTS
    assert len(results[0].evidence) == 1
    assert results[0].evidence[0].file == "backend/project_service.py"
    assert "project" in results[0].evidence[0].reason


def test_strongly_proven_absence(mock_agents):
    """Test 2: Strongly proven absence -> VERIFIED_ABSENT
    Currently unsupported in the generic implementation as per constraint. Should fall back to UNKNOWN.
    """
    verifier = PlanningScopeVerifier(mock_agents)
    guidance = [{"requirement": "Members", "what_to_do": "Create a new backend endpoint"}]
    
    # Generic missing -> UNKNOWN (fallback for safety)
    results = verifier.verify_guidance(guidance, [], [])
    assert results[0].status == VerificationStatus.UNKNOWN


def test_search_returns_zero_results(mock_agents):
    """Test 3: Search returns zero results but absence is not proven -> UNKNOWN"""
    verifier = PlanningScopeVerifier(mock_agents)
    
    # SQLite returns empty
    mock_agents.localizer.sqlite_store.search_symbols.return_value = []
    mock_agents.repo_search.search_literal.return_value = []
    
    guidance = [{"requirement": "Members", "what_to_do": "Create a new backend endpoint for project membership"}]
    results = verifier.verify_guidance(guidance, [], [])
    
    assert len(results) == 1
    assert results[0].status == VerificationStatus.UNKNOWN
    assert len(results[0].evidence) == 0


def test_no_scope_changing_claim(mock_agents):
    """Test 4: No scope-changing claim -> verifier no-op"""
    verifier = PlanningScopeVerifier(mock_agents)
    
    guidance = [{"requirement": "UI Update", "what_to_do": "Update the UI to consume the existing endpoint"}]
    results = verifier.verify_guidance(guidance, [], [])
    
    assert len(results) == 0


def test_existing_evidence(mock_agents):
    """Test 5: Existing evidence already proves capability -> VERIFIED_EXISTS -> no repository search"""
    verifier = PlanningScopeVerifier(mock_agents)
    
    evidence_item = MagicMock()
    evidence_item.content_snippet = "class ProjectMembershipController"
    evidence_item.file_path = "backend/controller.py"
    
    guidance = [{"requirement": "Members", "what_to_do": "Create a new backend controller for project membership"}]
    
    results = verifier.verify_guidance(guidance, [evidence_item], [])
    
    assert len(results) == 1
    assert results[0].status == VerificationStatus.VERIFIED_EXISTS
    assert results[0].evidence[0].file == "backend/controller.py"
    # Ensure no repository search was triggered
    mock_agents.localizer.sqlite_store.search_symbols.assert_not_called()


def test_existing_capability_different_layer(mock_agents):
    """Test 6: Existing capability established through a different technical layer -> VERIFIED_EXISTS"""
    verifier = PlanningScopeVerifier(mock_agents)
    
    mock_rel = MagicMock()
    mock_rel.source_file = "graphql/project_membership_resolver.py"
    
    mock_agents.relationship_provider.get_relationships_for_file.return_value = [mock_rel]
    
    guidance = [{"requirement": "REST API", "what_to_do": "Create a new REST endpoint for project membership", "target_file": "backend/controller.py"}]
    
    results = verifier.verify_guidance(guidance, [], [])
    
    assert len(results) == 1
    assert results[0].status == VerificationStatus.VERIFIED_EXISTS
    assert results[0].evidence[0].file == "graphql/project_membership_resolver.py"
    assert "relationship" in results[0].evidence[0].reason


def test_multiple_claims(mock_agents):
    """Test 7: Multiple claims -> only planning-scope-changing claims are verified"""
    verifier = PlanningScopeVerifier(mock_agents)
    
    guidance = [
        {"requirement": "Req 1", "what_to_do": "Update CSS styling"},
        {"requirement": "Req 2", "what_to_do": "Create a new backend service"}
    ]
    
    results = verifier.verify_guidance(guidance, [], [])
    
    assert len(results) == 1
    assert "service" in results[0].claim


def test_add_members_regression(mock_agents):
    """Test 8: Add Members regression pattern -> Preflight claims new backend capability -> capability exists in actual repository"""
    verifier = PlanningScopeVerifier(mock_agents)
    
    # Not found in evidence ledger (evidence_items = [])
    # Found in repository search:
    mock_agents.localizer.sqlite_store.search_symbols.return_value = [{"path": "src/ProjectMembership.java", "name": "ProjectMembershipService"}]
    mock_agents.workspace_path.joinpath.return_value.exists.return_value = True
    mock_agents.workspace_path.__truediv__.return_value.exists.return_value = True
    
    guidance = [{"requirement": "Backend", "what_to_do": "A new backend endpoint is required for project membership"}]
    
    results = verifier.verify_guidance(guidance, [], [])
    
    assert len(results) == 1
    assert results[0].status == VerificationStatus.VERIFIED_EXISTS
    assert results[0].evidence[0].file == "src/ProjectMembership.java"
    assert "Consume existing capability" in results[0].planning_implication


def test_planning_scope_verification_node_integration(mock_agents):
    """Test 9: Verify planning_scope_verification_node workflow node function executes and serializes properly."""
    from ticket_to_code.workflow import planning_scope_verification_node
    
    mock_agents.localizer.sqlite_store.search_symbols.return_value = [{"path": "src/ProjectMembership.java", "name": "ProjectMembershipService"}]
    mock_agents.workspace_path.joinpath.return_value.exists.return_value = True
    mock_agents.workspace_path.__truediv__.return_value.exists.return_value = True

    state = {
        "preflight_implementation_guidance": [
            {"requirement": "Backend", "what_to_do": "A new backend endpoint is required for project membership"}
        ],
        "evidence_items": [],
        "discovered_files": []
    }
    
    out = planning_scope_verification_node(state, mock_agents)
    assert "planning_scope_verification_results" in out
    res = out["planning_scope_verification_results"]
    assert len(res) == 1
    assert res[0]["status"] == "VERIFIED_EXISTS"
    assert res[0]["evidence"][0]["file"] == "src/ProjectMembership.java"


def test_route_after_preflight_routing():
    """Test 10: Verify route_after_preflight routes already_implemented to END, and anything else to planning_scope_verification."""
    from ticket_to_code.workflow import route_after_preflight, END

    # When already implemented -> END
    state_done = {"status": "already_implemented"}
    assert route_after_preflight(state_done) == END

    # When not done / partially done -> planning_scope_verification
    state_not_done = {"status": "proceed_to_plan"}
    assert route_after_preflight(state_not_done) == "planning_scope_verification"
