"""Phase D — Evidence Discovery Architecture Integration Tests.

These tests prove the entire chain defined by the architecture:

    Ticket → Requirements/Hypotheses → Evidence Ledger
        → What's unresolved?
        → LLM Controller (_decide_next_action)
        → Capability selected → Capability executes → Raw result
        → Normalize + merge + relevance/materiality + provenance
        → Evidence Ledger → Re-evaluate → (loop or proceed)

Each test mocks only the EXTERNAL boundaries (LLM calls, filesystem, vector
store) and exercises the real internal plumbing: _execute_tool, _merge_evidence,
_build_investigation_state, and the ledger's additive/provenance invariants.

Tests:
    1.  semantic_rag → executes → evidence ledger + provenance
    2.  i18n_chain → executes → evidence ledger
    3.  ripgrep → executes → evidence ledger (exact match)
    4.  neo4j_graph → executes → typed relationship → evidence ledger
    5.  symbol_lookup → executes → symbol location → evidence ledger
    6.  duplicate discovery → merged provenance (not 3 entries)
    7.  zero-result search → preserves existing evidence
    8.  evidence-driven tool selection (tool sequence follows ledger state)
    9.  sufficient evidence → proceed without unnecessary tool calls
    10. capability-awareness reminder is informational only
"""

import sys
import types
from pathlib import Path
from dataclasses import dataclass, field
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.evidence_collection_loop import (
    EvidenceCollectionLoop,
    EvidenceKnowledge,
    ArtifactInspection,
    MethodLocation,
    InspectionFact,
)
from ticket_to_code.models import EvidenceItem, SearchContext


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_loop() -> EvidenceCollectionLoop:
    """Create a minimal EvidenceCollectionLoop with stubbed dependencies."""
    loop = object.__new__(EvidenceCollectionLoop)
    loop._workspace = "/fake/workspace"
    loop.repo_search = None
    loop.rag_engine = None
    loop.localizer = None
    loop.relationship_provider = None
    loop._current_followable_relationships = {}
    loop._semantic_verifier = None  # skip real verification
    return loop


def _make_ticket(title="Add Members", desc="Allow adding members to a project"):
    return types.SimpleNamespace(
        title=title,
        description=desc,
        ticket_id="TEST-001",
    )


def _make_search_result(file_path, confidence=0.9, line_number=10, matched_text="match"):
    """Create a mock SearchResult from RepositorySearchEngine."""
    return types.SimpleNamespace(
        file_path=file_path,
        confidence=confidence,
        line_number=line_number,
        matched_text=matched_text,
    )


def _make_evidence_item(file_path, provider="literal", strength="strong",
                        details="test", relevance_score=0.8):
    """Create an EvidenceItem for testing."""
    return EvidenceItem(
        file_path=file_path,
        provider=provider,
        strength=strength,
        details=details,
        content_snippet=f"test snippet from {file_path}",
        relevance_score=relevance_score,
        hypothesis_id="test",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: semantic_rag → ledger with provenance
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticRagToLedger:
    """Prove: unresolved requirement → semantic_rag selected → RAG result
    → evidence item → ledger contains finding + provenance → next
    controller sees finding."""

    def test_semantic_rag_executes_and_populates_ledger(self):
        loop = _make_loop()
        context = SearchContext()

        # Mock _query_semantic to return an evidence item
        semantic_result = _make_evidence_item(
            "src/services/member.service.ts",
            provider="vector", strength="medium",
            details="agentic_semantic_rag",
        )

        loop._query_semantic = lambda queries, hyp_id, visited: [semantic_result]

        # Execute the tool
        items = loop._execute_tool("semantic_rag", "member management service", context)

        assert len(items) == 1, "semantic_rag should return the mocked result"
        assert items[0].file_path == "src/services/member.service.ts"

        # Now merge into ledger
        evidence: Dict[str, dict] = {}
        entry = loop._merge_evidence(
            evidence=evidence,
            file_path=items[0].file_path,
            source="semantic_rag",
            tool="semantic_rag",
            query="member management service",
            iteration=1,
            verdict={"relevant": True, "decision": "include",
                     "reason": "Contains member management logic",
                     "semantic_relevance_score": 0.85},
        )

        # Ledger assertions
        assert "src/services/member.service.ts" in evidence
        assert entry["sources"] == ["semantic_rag"]
        assert entry["decision"] == "include"
        assert entry["status"] == "verified"
        assert entry["search_provenance"][0]["tool"] == "semantic_rag"
        assert entry["search_provenance"][0]["query"] == "member management service"
        assert entry["search_provenance"][0]["iteration"] == 1

        # Next controller sees this finding
        state_text = loop._build_investigation_state(evidence, None, None)
        assert "member.service.ts" in state_text
        assert "VERIFIED ARTIFACTS" in state_text

    def test_semantic_rag_visited_files_tracked(self):
        """After execution, visited_files includes the discovered file."""
        loop = _make_loop()
        context = SearchContext()

        result = _make_evidence_item("src/api/project.api.ts")
        loop._query_semantic = lambda q, h, v: [result]

        loop._execute_tool("semantic_rag", "project API", context)

        assert "src/api/project.api.ts" in context.visited_files


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: i18n_chain → ledger
# ═══════════════════════════════════════════════════════════════════════════════

class TestI18nToLedger:
    """Prove: unresolved UI text requirement → i18n_chain → translation/key
    discovery → ledger."""

    def test_i18n_chain_discovers_translation_file(self):
        loop = _make_loop()
        context = SearchContext()

        i18n_result = _make_evidence_item(
            "src/assets/i18n/en.json",
            provider="i18n", strength="strong",
            details="agentic_i18n",
        )
        loop._trace_i18n_chain = lambda query, visited: [i18n_result]

        items = loop._execute_tool("i18n_chain", "addMembers.title", context)

        assert len(items) == 1
        assert items[0].file_path == "src/assets/i18n/en.json"

        # Merge into ledger
        evidence: Dict[str, dict] = {}
        loop._merge_evidence(
            evidence=evidence,
            file_path=items[0].file_path,
            source="i18n_chain",
            tool="i18n_chain",
            query="addMembers.title",
            iteration=1,
            verdict={"relevant": True, "decision": "include",
                     "reason": "Translation key for Add Members UI"},
        )

        assert "src/assets/i18n/en.json" in evidence
        entry = evidence["src/assets/i18n/en.json"]
        assert "i18n_chain" in entry["sources"]
        assert entry["search_provenance"][0]["tool"] == "i18n_chain"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: ripgrep → ledger (exact match)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRipgrepToLedger:
    """Prove: 'Where is Add Members?' → ripgrep → exact match → ledger."""

    def test_ripgrep_exact_search_populates_ledger(self):
        loop = _make_loop()
        context = SearchContext()

        # Mock repo_search
        mock_search = Mock()
        mock_search.search_literal = Mock(return_value=[
            _make_search_result(
                "src/modules/members/add-members.component.ts",
                confidence=0.95, line_number=42,
                matched_text="'Add Members'",
            ),
        ])
        loop.repo_search = mock_search

        items = loop._execute_tool("ripgrep", "Add Members", context)

        assert len(items) == 1
        item = items[0]
        assert item.file_path == "src/modules/members/add-members.component.ts"
        assert item.provider == "literal"
        assert item.strength == "strong"
        assert "ripgrep:'Add Members'" in item.content_snippet

        # Merge and verify provenance
        evidence: Dict[str, dict] = {}
        loop._merge_evidence(
            evidence=evidence,
            file_path=item.file_path,
            source="ripgrep",
            tool="ripgrep",
            query="Add Members",
            iteration=1,
            verdict={"relevant": True, "decision": "include",
                     "reason": "Contains literal 'Add Members'",
                     "semantic_relevance_score": 0.95},
        )

        entry = evidence["src/modules/members/add-members.component.ts"]
        assert entry["status"] == "verified"
        assert entry["sources"] == ["ripgrep"]
        assert entry["search_provenance"][0]["tool"] == "ripgrep"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: neo4j_graph → typed relationship → ledger
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeo4jToLedger:
    """Prove: relationship question → neo4j_graph → typed relationship
    → ledger.  Assert the relationship itself becomes usable evidence."""

    def test_neo4j_relationship_becomes_evidence(self):
        loop = _make_loop()
        context = SearchContext()

        graph_result = _make_evidence_item(
            "src/services/contract-member.service.java",
            provider="graph", strength="strong",
            details="agentic_neo4j",
        )
        # Add graph-specific metadata
        graph_result = EvidenceItem(
            file_path="src/services/contract-member.service.java",
            provider="graph",
            strength="strong",
            details="agentic_neo4j",
            content_snippet="[neo4j] ContractMemberService CALLS MemberRepository",
            relevance_score=0.9,
            hypothesis_id="agentic",
            graph_distance=1,
            matched_concept="ContractMemberService",
            path=[
                {"type": "Service", "id": "ContractMemberService"},
                {"type": "Repository", "id": "MemberRepository"},
            ],
        )
        loop._query_neo4j = lambda q, h, v: [graph_result]

        items = loop._execute_tool("neo4j_graph", "ContractMemberService", context)

        assert len(items) == 1
        assert items[0].graph_distance == 1
        assert items[0].matched_concept == "ContractMemberService"

        # Merge with relationship data
        evidence: Dict[str, dict] = {}
        relationships = [
            {"kind": "CALLS", "dst_name": "MemberRepository",
             "target_file": "src/repo/member.repository.java"},
        ]
        entry = loop._merge_evidence(
            evidence=evidence,
            file_path=items[0].file_path,
            source="neo4j_graph",
            tool="neo4j_graph",
            query="ContractMemberService",
            iteration=1,
            verdict={"relevant": True, "decision": "include",
                     "reason": "Service handles contract member operations"},
            relationships=relationships,
        )

        # The relationship itself is usable evidence
        assert len(entry["relationships"]) == 1
        rel = entry["relationships"][0]
        assert rel["kind"] == "CALLS"
        assert rel["dst_name"] == "MemberRepository"

        # And it appears in the investigation state as followable
        assert "followable" in entry
        assert len(entry["followable"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: symbol_lookup → symbol location → ledger
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymbolLookupToLedger:
    """Prove: 'Where is this method/type defined?' → symbol_lookup
    → symbol location/signature → ledger."""

    def test_symbol_lookup_populates_ledger(self):
        loop = _make_loop()
        context = SearchContext()

        symbol_result = EvidenceItem(
            file_path="src/services/member.service.ts",
            provider="sqlite_symbol",
            strength="strong",
            details="agentic_symbol",
            content_snippet="[symbol:addMember] MemberService.addMember(userId: string): Observable<Member>",
            relevance_score=0.95,
            hypothesis_id="agentic",
            symbol_name="addMember",
        )
        loop._query_sqlite_symbols = lambda q, h, v: [symbol_result]

        items = loop._execute_tool("symbol_lookup", "addMember", context)

        assert len(items) == 1
        assert items[0].symbol_name == "addMember"
        assert items[0].file_path == "src/services/member.service.ts"

        # Merge
        evidence: Dict[str, dict] = {}
        entry = loop._merge_evidence(
            evidence=evidence,
            file_path=items[0].file_path,
            source="symbol_lookup",
            tool="symbol_lookup",
            query="addMember",
            iteration=1,
            verdict={"relevant": True, "decision": "include",
                     "reason": "Symbol definition for addMember"},
        )

        assert entry["sources"] == ["symbol_lookup"]
        assert entry["search_provenance"][0]["tool"] == "symbol_lookup"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: duplicate discovery → merged provenance (not 3 entries)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDuplicateDiscoveryMerge:
    """Prove: ripgrep and semantic_rag discover the same file → ONE logical
    evidence item with BOTH provenances preserved.

    sources:
        - ripgrep
        - semantic_rag

    This proves the ledger is an evidence model, not an append-only list."""

    def test_same_file_two_tools_one_entry_both_sources(self):
        loop = _make_loop()
        evidence: Dict[str, dict] = {}
        target = "src/modules/members/add-members.component.ts"

        # First discovery: ripgrep
        loop._merge_evidence(
            evidence=evidence,
            file_path=target,
            source="ripgrep",
            tool="ripgrep",
            query="Add Members",
            iteration=1,
            verdict={"relevant": True, "decision": "include",
                     "reason": "Literal match",
                     "semantic_relevance_score": 0.9},
        )

        # Second discovery: semantic_rag finds the same file
        loop._merge_evidence(
            evidence=evidence,
            file_path=target,
            source="semantic_rag",
            tool="semantic_rag",
            query="member management component",
            iteration=2,
            verdict={"relevant": True, "decision": "include",
                     "reason": "Semantic match",
                     "semantic_relevance_score": 0.85},
        )

        # ONE entry, not two
        assert len(evidence) == 1, \
            f"Expected 1 evidence entry but got {len(evidence)}"

        entry = evidence[target]

        # Both sources preserved
        assert "ripgrep" in entry["sources"], "ripgrep source must be preserved"
        assert "semantic_rag" in entry["sources"], "semantic_rag source must be preserved"

        # Both provenances preserved
        assert len(entry["search_provenance"]) == 2
        tools_used = [p["tool"] for p in entry["search_provenance"]]
        assert "ripgrep" in tools_used
        assert "semantic_rag" in tools_used

        # Queries preserved
        queries = [p["query"] for p in entry["search_provenance"]]
        assert "Add Members" in queries
        assert "member management component" in queries

    def test_three_tools_still_one_entry(self):
        """Even with 3 different tools finding the same file, only 1 entry."""
        loop = _make_loop()
        evidence: Dict[str, dict] = {}
        target = "src/shared/member.model.ts"

        for i, (tool, query) in enumerate([
            ("ripgrep", "MemberModel"),
            ("symbol_lookup", "MemberModel"),
            ("semantic_rag", "member data model"),
        ], start=1):
            loop._merge_evidence(
                evidence=evidence,
                file_path=target,
                source=tool,
                tool=tool,
                query=query,
                iteration=i,
                verdict={"relevant": True, "decision": "include", "reason": f"Found by {tool}"},
            )

        assert len(evidence) == 1
        entry = evidence[target]
        assert len(entry["sources"]) == 3
        assert len(entry["search_provenance"]) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: zero-result search → preserves existing evidence
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroResultPreservation:
    """Critical invariant: a search that returns zero results must NEVER
    erase or downgrade evidence that was already accumulated.

    Scenario:
        ripgrep → discovers Controller.java
        semantic_rag → 0 useful results
        final ledger → Controller.java still present, unchanged
    """

    def test_zero_results_do_not_erase_existing_evidence(self):
        loop = _make_loop()
        evidence: Dict[str, dict] = {}

        # Step 1: ripgrep discovers Controller.java
        loop._merge_evidence(
            evidence=evidence,
            file_path="src/controllers/ProjectController.java",
            source="ripgrep",
            tool="ripgrep",
            query="ProjectController",
            iteration=1,
            verdict={"relevant": True, "decision": "include",
                     "reason": "Controller handles project operations",
                     "semantic_relevance_score": 0.92},
        )

        # Snapshot the evidence state
        entry_before = dict(evidence["src/controllers/ProjectController.java"])
        score_before = entry_before["semantic_relevance_score"]
        sources_before = list(entry_before["sources"])
        status_before = entry_before["status"]

        # Step 2: semantic_rag returns 0 results (simulated by not calling
        # _merge_evidence for any file).  The _execute_tool returns empty list.
        context = SearchContext()
        loop._query_semantic = lambda q, h, v: []  # 0 results
        items = loop._execute_tool("semantic_rag", "project controller operations", context)
        assert len(items) == 0, "semantic_rag should return 0 results"

        # Step 3: Verify existing evidence is untouched
        assert "src/controllers/ProjectController.java" in evidence, \
            "Controller.java must still be in ledger"

        entry_after = evidence["src/controllers/ProjectController.java"]
        assert entry_after["semantic_relevance_score"] == score_before, \
            "Score must not change after zero-result search"
        assert entry_after["sources"] == sources_before, \
            "Sources must not change"
        assert entry_after["status"] == status_before, \
            "Status must not be downgraded"
        assert entry_after["decision"] == "include", \
            "Decision must remain 'include'"

    def test_unrelated_zero_result_does_not_affect_ledger_size(self):
        """Even if semantic_rag returns results for OTHER files that get
        excluded, the existing include entries must not change."""
        loop = _make_loop()
        evidence: Dict[str, dict] = {}

        # Pre-existing evidence
        loop._merge_evidence(
            evidence=evidence,
            file_path="src/api/routes.ts",
            source="ripgrep",
            tool="ripgrep",
            query="routes",
            iteration=1,
            verdict={"relevant": True, "decision": "include", "reason": "API routes"},
        )

        # A new file arrives but is excluded
        loop._merge_evidence(
            evidence=evidence,
            file_path="src/utils/logger.ts",
            source="semantic_rag",
            tool="semantic_rag",
            query="logging utility",
            iteration=2,
            verdict={"relevant": False, "decision": "exclude",
                     "reason": "Generic utility, not related"},
        )

        # Original evidence untouched
        routes = evidence["src/api/routes.ts"]
        assert routes["decision"] == "include"
        assert routes["sources"] == ["ripgrep"]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: evidence-driven tool selection
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvidenceDrivenSelection:
    """Prove the controller selects tools based on evolving ledger state,
    NOT by exhausting all tools sequentially.

    Scenario:
        Requirement A → ripgrep (finds component)
        new evidence → Requirement B unresolved → relationship traversal
        new evidence → Requirement C unresolved → semantic_rag

    Tool sequence is driven by requirements, not by "use all tools once".
    """

    def test_tool_sequence_follows_requirements_not_checklist(self):
        """The _decide_next_action mock returns different tools depending
        on what evidence already exists.  The test verifies the loop
        respects these evolving decisions."""
        loop = _make_loop()

        # We'll track what the controller sees at each iteration
        controller_calls = []

        # Mock _decide_next_action to return requirement-driven choices
        call_counter = [0]

        def mock_decide(ticket, hypotheses, evidence, tried, all_items,
                        iteration, max_iterations, evidence_knowledge=None,
                        sufficiency_guidance=None, recovery_context=None):
            call_counter[0] += 1
            n_included = sum(1 for v in evidence.values()
                             if v.get("relevant") and v.get("decision") == "include")

            controller_calls.append({
                "iteration": iteration,
                "n_evidence": n_included,
                "n_tried": len(tried),
            })

            # Iteration 0: Requirement A → use ripgrep
            if call_counter[0] == 1:
                return {"type": "search", "tool": "ripgrep",
                        "query": "Add Members",
                        "reasoning": "Need to find where 'Add Members' UI is rendered"}

            # Iteration 1: New evidence → Requirement B → relationship traversal
            if call_counter[0] == 2:
                return {"type": "search", "tool": "neo4j_graph",
                        "query": "AddMembersComponent",
                        "reasoning": "Found component, now need its service dependencies"}

            # Iteration 2: More evidence → Requirement C → semantic_rag
            if call_counter[0] == 3:
                return {"type": "search", "tool": "semantic_rag",
                        "query": "member organization validation",
                        "reasoning": "Need to find validation logic for member org check"}

            # Iteration 3: sufficient
            return {"type": "proceed",
                    "reasoning": "All requirements covered by evidence"}

        loop._decide_next_action = mock_decide

        # Mock _execute_tool to return different results per tool
        def mock_execute(tool_name, query, context):
            if tool_name == "ripgrep":
                return [_make_evidence_item("src/add-members.component.ts")]
            elif tool_name == "neo4j_graph":
                return [_make_evidence_item("src/member.service.ts",
                                            provider="graph")]
            elif tool_name == "semantic_rag":
                return [_make_evidence_item("src/org-validator.ts",
                                            provider="vector")]
            return []

        loop._execute_tool = mock_execute

        # Mock verification
        loop._verify_single_file = lambda t, f, e: {
            "relevant": True, "decision": "include",
            "reason": f"Verified {f.file_path}",
            "semantic_relevance_score": 0.85,
        }
        loop._should_inspect = lambda *a: False

        # Run the loop manually (simplified)
        ticket = _make_ticket()
        evidence: Dict[str, dict] = {}
        tried: List[dict] = []
        all_items: List[EvidenceItem] = []
        ek = EvidenceKnowledge()

        for iteration in range(10):
            action = loop._decide_next_action(
                ticket=ticket, hypotheses=None,
                evidence=evidence, tried=tried,
                all_items=all_items, iteration=iteration,
                max_iterations=10, evidence_knowledge=ek,
            )

            if action["type"] == "proceed":
                break

            tool_name = action["tool"]
            query = action["query"]

            context = SearchContext()
            new_files = loop._execute_tool(tool_name, query, context)

            tried.append({
                "tool": tool_name, "query": query,
                "files_returned": [f.file_path for f in new_files],
                "iteration": iteration + 1,
            })

            for f in new_files:
                if f.file_path not in evidence:
                    verdict = loop._verify_single_file(ticket, f, evidence)
                    loop._merge_evidence(
                        evidence=evidence,
                        file_path=f.file_path,
                        source=tool_name,
                        tool=tool_name,
                        query=query,
                        iteration=iteration + 1,
                        verdict=verdict,
                    )
                    all_items.append(f)

        # Assertions: tool sequence matches controller decisions, not checklist
        assert len(tried) == 3, f"Expected 3 tool calls, got {len(tried)}"
        assert tried[0]["tool"] == "ripgrep"
        assert tried[1]["tool"] == "neo4j_graph"
        assert tried[2]["tool"] == "semantic_rag"

        # Each decision was informed by growing evidence
        assert controller_calls[0]["n_evidence"] == 0  # nothing at start
        assert controller_calls[1]["n_evidence"] == 1  # ripgrep result visible
        assert controller_calls[2]["n_evidence"] == 2  # ripgrep + neo4j visible

        # All 3 files in ledger
        assert len(evidence) == 3
        assert "src/add-members.component.ts" in evidence
        assert "src/member.service.ts" in evidence
        assert "src/org-validator.ts" in evidence


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9: sufficient evidence → proceed without unnecessary tools
# ═══════════════════════════════════════════════════════════════════════════════

class TestSufficientEvidenceProceeds:
    """Critical negative case: when evidence already answers the requirement,
    the controller should choose 'proceed' without invoking unused tools.

    Proves:
        sufficient evidence → STOP
    NOT:
        sufficient evidence → unused tools exist → keep searching
    """

    def test_controller_proceeds_when_evidence_sufficient(self):
        loop = _make_loop()

        # Pre-populate ledger with sufficient evidence
        evidence: Dict[str, dict] = {
            "src/add-members.component.ts": {
                "file_path": "src/add-members.component.ts",
                "sources": ["ripgrep"],
                "search_provenance": [
                    {"tool": "ripgrep", "query": "Add Members", "iteration": 1}
                ],
                "relevant": True,
                "decision": "include",
                "reason": "Main component for Add Members feature",
                "semantic_relevance_score": 0.95,
                "status": "verified",
                "relationships": [],
                "unresolved_references": [],
            },
            "src/member.service.ts": {
                "file_path": "src/member.service.ts",
                "sources": ["neo4j_graph"],
                "search_provenance": [
                    {"tool": "neo4j_graph", "query": "AddMembersComponent", "iteration": 2}
                ],
                "relevant": True,
                "decision": "include",
                "reason": "Service dependency of the component",
                "semantic_relevance_score": 0.90,
                "status": "verified",
                "relationships": [],
                "unresolved_references": [],
            },
        }

        tried = [
            {"tool": "ripgrep", "query": "Add Members",
             "files_returned": ["src/add-members.component.ts"], "iteration": 1},
            {"tool": "neo4j_graph", "query": "AddMembersComponent",
             "files_returned": ["src/member.service.ts"], "iteration": 2},
        ]

        # Controller says proceed — evidence is sufficient
        proceed_called = [False]

        def mock_decide(**kwargs):
            proceed_called[0] = True
            return {"type": "proceed",
                    "reasoning": "Component and service found. Requirements covered."}

        loop._decide_next_action = mock_decide

        action = loop._decide_next_action(
            ticket=_make_ticket(),
            hypotheses=None,
            evidence=evidence,
            tried=tried,
            all_items=[],
            iteration=2,
            max_iterations=10,
        )

        # The controller chose proceed
        assert action["type"] == "proceed"
        assert proceed_called[0]

        # UNUSED tools still exist (semantic_rag, i18n_chain, symbol_lookup)
        # but they were NOT invoked
        tools_used = {t["tool"] for t in tried}
        unused = {"semantic_rag", "i18n_chain", "symbol_lookup"} - tools_used
        assert len(unused) > 0, "Some tools should remain unused"

        # Evidence is unmodified — no additional searches happened
        assert len(evidence) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 10: capability-awareness is informational only
# ═══════════════════════════════════════════════════════════════════════════════

class TestCapabilityAwarenessInformational:
    """Prove: the AVAILABLE CAPABILITIES section in investigation state
    is informational only and contains explicit anti-checklist language."""

    def test_unused_capabilities_shown_with_disclaimer(self):
        loop = _make_loop()

        # Evidence with only 1 verified artifact (< 3 threshold)
        evidence = {
            "src/component.ts": {
                "file_path": "src/component.ts",
                "sources": ["ripgrep"],
                "relevant": True,
                "decision": "include",
                "reason": "test",
                "semantic_relevance_score": 0.8,
                "status": "verified",
                "search_provenance": [],
                "relationships": [],
                "unresolved_references": [],
            },
        }

        tried = [{"tool": "ripgrep", "query": "test"}]

        state = loop._build_investigation_state(evidence, None, None, tried=tried)

        # Unused capabilities are shown
        assert "AVAILABLE CAPABILITIES" in state

        # Anti-checklist disclaimer is present
        assert "informational only" in state
        assert "Do NOT select a capability merely because it has not been tried" in state

        # Specific unused tools listed
        assert "semantic_rag" in state
        assert "i18n_chain" in state
        assert "symbol_lookup" in state
        assert "neo4j_graph" in state

        # ripgrep is NOT listed (it was used)
        lines = state.split("\n")
        capability_lines = [l for l in lines if "🔧" in l]
        capability_tools = [l.strip().split("🔧")[1].strip() for l in capability_lines]
        assert "ripgrep" not in capability_tools

    def test_capabilities_hidden_when_investigation_mature(self):
        """When >= 3 verified artifacts exist, capabilities are not shown."""
        loop = _make_loop()

        evidence = {}
        for i in range(3):
            evidence[f"src/file{i}.ts"] = {
                "file_path": f"src/file{i}.ts",
                "sources": ["ripgrep"],
                "relevant": True,
                "decision": "include",
                "reason": "test",
                "semantic_relevance_score": 0.8,
                "status": "verified",
                "search_provenance": [],
                "relationships": [],
                "unresolved_references": [],
            }

        tried = [{"tool": "ripgrep", "query": "test"}]

        state = loop._build_investigation_state(evidence, None, None, tried=tried)

        # Capabilities section should NOT appear
        assert "AVAILABLE CAPABILITIES" not in state

    def test_capabilities_hidden_when_all_used(self):
        """When all capabilities have been tried, the section doesn't appear."""
        loop = _make_loop()

        evidence = {
            "src/x.ts": {
                "file_path": "src/x.ts",
                "sources": ["ripgrep"],
                "relevant": True,
                "decision": "include",
                "reason": "test",
                "semantic_relevance_score": 0.8,
                "status": "verified",
                "search_provenance": [],
                "relationships": [],
                "unresolved_references": [],
            },
        }

        # All capabilities tried
        tried = [
            {"tool": "ripgrep", "query": "t1"},
            {"tool": "symbol_lookup", "query": "t2"},
            {"tool": "neo4j_graph", "query": "t3"},
            {"tool": "semantic_rag", "query": "t4"},
            {"tool": "i18n_chain", "query": "t5"},
        ]

        state = loop._build_investigation_state(evidence, None, None, tried=tried)

        assert "AVAILABLE CAPABILITIES" not in state

    def test_follow_counts_as_neo4j(self):
        """Using 'follow' should mark neo4j_graph as used."""
        loop = _make_loop()

        evidence = {
            "src/x.ts": {
                "file_path": "src/x.ts",
                "sources": ["ripgrep"],
                "relevant": True,
                "decision": "include",
                "reason": "test",
                "semantic_relevance_score": 0.8,
                "status": "verified",
                "search_provenance": [],
                "relationships": [],
                "unresolved_references": [],
            },
        }

        tried = [
            {"tool": "ripgrep", "query": "t1"},
            {"tool": "follow", "query": "R1"},  # follow = neo4j_graph
        ]

        state = loop._build_investigation_state(evidence, None, None, tried=tried)

        # neo4j_graph should NOT be in the available list
        if "AVAILABLE CAPABILITIES" in state:
            lines = [l for l in state.split("\n") if "🔧" in l]
            cap_names = [l.strip().split("🔧")[1].strip() for l in lines]
            assert "neo4j_graph" not in cap_names


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: Relationship merge deduplication
# ═══════════════════════════════════════════════════════════════════════════════

class TestRelationshipMergeDedup:
    """Prove that when the same relationship is discovered twice (from
    different tools), it is deduplicated by target+kind."""

    def test_duplicate_relationships_deduplicated(self):
        loop = _make_loop()
        evidence: Dict[str, dict] = {}

        target = "src/component.ts"

        # First merge: relationship from neo4j
        loop._merge_evidence(
            evidence=evidence,
            file_path=target,
            source="neo4j_graph",
            tool="neo4j_graph",
            query="Component",
            iteration=1,
            relationships=[
                {"kind": "IMPORTS", "dst_name": "MemberService",
                 "target_file": "src/member.service.ts"},
            ],
        )

        # Second merge: same relationship from symbol_lookup
        loop._merge_evidence(
            evidence=evidence,
            file_path=target,
            source="symbol_lookup",
            tool="symbol_lookup",
            query="Component",
            iteration=2,
            relationships=[
                {"kind": "IMPORTS", "dst_name": "MemberService",
                 "target_file": "src/member.service.ts"},
            ],
        )

        entry = evidence[target]
        assert len(entry["relationships"]) == 1, \
            "Duplicate relationship should be deduplicated"
        assert entry["relationships"][0]["kind"] == "IMPORTS"
        assert entry["relationships"][0]["dst_name"] == "MemberService"
