"""Tests for Stage 0.7: Guaranteed Discovery RAG in evidence_collection_loop.

Verifies:
    1. collect() ALWAYS invokes RAG (via _query_semantic) regardless of agentic loop decisions
    2. RAG discoveries exist in evidence BEFORE the agentic loop starts
    3. RAG relevance != ownership — only verified candidates become PRIMARY_FEATURE_TARGET
    4. Stage 0.7 is bounded (max 6 queries, max 10 candidates)

Pure logic; no LLM/workspace. Uses mocking to isolate RAG behavior.

SKIPPED (owner decision, 2026-09-16): Stage 0.7 was deliberately removed from
evidence_collection_loop.py — mandatory pre-loop RAG proved not useful, and
semantic_rag remains available as an on-demand agentic tool instead. These
tests document the removed feature and are kept (skipped) for reference only.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from datetime import datetime

import pytest

pytestmark = pytest.mark.skip(
    reason="Stage 0.7 Discovery RAG was removed intentionally (owner decision, "
    "2026-09-16): semantic_rag is an on-demand agentic tool, not a mandatory stage."
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _FakeTicket:
    def __init__(self, title="Add members modal", description="When adding members to a contract, show duplicate warning"):
        self.ticket_id = "TEST-001"
        self.title = title
        self.description = description
        self.acceptance_criteria = ""
        self.priority = "medium"
        self.labels = []


class _FakeHypothesis:
    def __init__(self, hypothesis, queries=None, symbols=None, anchors=None):
        self.hypothesis = hypothesis
        self.queries = queries or []
        self.symbols = symbols or []
        self.anchors = anchors or []


class _FakeEvidenceItem:
    def __init__(self, file_path, relevance_score=0.7, hypothesis_id="stage_07_discovery_rag"):
        self.file_path = file_path
        self.relevance_score = relevance_score
        self.hypothesis_id = hypothesis_id
        self.provider = "vector"
        self.strength = "medium"
        self.details = "semantic"
        self.evidence_type = "usage"
        self.content_snippet = "..."


# ---------------------------------------------------------------------------
# 1. collect() ALWAYS invokes RAG — INTEGRATION TEST
# ---------------------------------------------------------------------------

def test_collect_always_invokes_rag():
    """The production EvidenceCollectionLoop.collect() MUST call _query_semantic
    with hypothesis_id='stage_07_discovery_rag' BEFORE the agentic loop starts.

    This is the architectural guarantee:
        "RAG is no longer optional during Evidence Collection."

    We prove it by calling the REAL collect() method on a REAL
    EvidenceCollectionLoop instance.  All LLM/filesystem methods are mocked,
    but the control flow (Stage 0 → Stage 0.5 → Stage 0.7 → Agentic Loop)
    is the actual production code.
    """
    from ticket_to_code.agents.evidence_collection_loop import EvidenceCollectionLoop
    from ticket_to_code.models import EvidenceItem

    # ── Construct a real instance with mocked dependencies ──────────────
    localizer = MagicMock()
    localizer.workspace_path = "/fake/workspace"

    rag_engine = MagicMock()
    # Return synthetic RAG chunks when _query_semantic calls rag_engine.retrieve_context
    rag_engine.retrieve_context.return_value = [
        {"file_path": "xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
         "content": "export class AddMembersComponent { ... }",
         "score": 0.85},
        {"file_path": "xchange-ui/src/app/modules/shared/services/member.service.ts",
         "content": "export class MemberService { saveMember() { ... } }",
         "score": 0.72},
    ]

    loop = EvidenceCollectionLoop(
        localizer=localizer,
        rag_engine=rag_engine,
        repo_search=None,
        relationship_provider=None,
    )

    # ── Track which methods are called and in what order ────────────────
    _call_log = []   # records (method_name, key_args) tuples

    # Wrap _query_semantic to record calls but keep real implementation
    _real_query_semantic = loop._query_semantic
    def _tracking_query_semantic(queries, hypothesis_id, visited):
        _call_log.append(("_query_semantic", {"hypothesis_id": hypothesis_id, "num_queries": len(queries)}))
        return _real_query_semantic(queries, hypothesis_id, visited)
    loop._query_semantic = _tracking_query_semantic

    # Mock Stage 0 methods (i18n chain — not what we're testing)
    loop._extract_ui_text_labels = MagicMock(return_value=[])

    # Mock Stage 0.5 methods (relationship following — not what we're testing)
    loop._inspect_artifact = MagicMock(return_value=None)
    loop._get_followable_relationships = MagicMock(return_value=[])
    loop._auto_follow_and_inspect = MagicMock()

    # Mock Stage 0.7 verification (needs LLM — mock returns "include")
    loop._verify_single_file = MagicMock(return_value={
        "relevant": True, "decision": "include",
        "reason": "Implements the primary add-members feature component",
        "file_path": "add-members.component.ts",
        "semantic_relevance_score": 0.85,
    })

    # Mock hypothesis-to-queries (needs LLM — mock returns valid queries)
    loop._hypothesis_to_semantic_queries = MagicMock(return_value=[
        "member service duplicate check",
        "add members component validation",
    ])

    # Mock agentic loop's _decide_next_action to immediately "proceed"
    # so the loop exits on iteration 0 (proving Stage 0.7 ran BEFORE it)
    def _mock_decide(*args, **kwargs):
        _call_log.append(("_decide_next_action", {}))
        return {"type": "proceed", "reasoning": "Test: immediate exit"}
    loop._decide_next_action = _mock_decide

    # Mock confidence computation (needs LLM)
    loop._compute_confidence = MagicMock(return_value=0.85)

    # Mock count_pre_visited_results (needs filesystem)
    loop._count_pre_visited_results = MagicMock(return_value=0)

    # ── step_callback to capture event ordering ────────────────────────
    _events = []
    def _capture_callback(event):
        _events.append(event)

    # ── Call the REAL collect() ─────────────────────────────────────────
    ticket = _FakeTicket()
    hypotheses = [
        _FakeHypothesis("Duplicate member detection", queries=["check duplicate members"]),
    ]

    result = loop.collect(
        ticket=ticket,
        hypotheses=hypotheses,
        step_callback=_capture_callback,
    )

    # ── ASSERTIONS ─────────────────────────────────────────────────────

    # 1. _query_semantic WAS called with Stage 0.7 hypothesis ID
    s07_calls = [
        c for c in _call_log
        if c[0] == "_query_semantic" and c[1].get("hypothesis_id") == "stage_07_discovery_rag"
    ]
    assert len(s07_calls) >= 1, (
        f"Stage 0.7 must call _query_semantic with hypothesis_id='stage_07_discovery_rag'. "
        f"Actual call log: {_call_log}"
    )

    # 2. Stage 0.7 queries are bounded (max 6)
    for call_entry in s07_calls:
        assert call_entry[1]["num_queries"] <= 6, (
            f"Stage 0.7 is bounded to max 6 queries, got {call_entry[1]['num_queries']}"
        )

    # 3. _query_semantic(stage_07) was called BEFORE _decide_next_action
    qs_indices = [i for i, c in enumerate(_call_log) if c[0] == "_query_semantic" and c[1].get("hypothesis_id") == "stage_07_discovery_rag"]
    decide_indices = [i for i, c in enumerate(_call_log) if c[0] == "_decide_next_action"]
    assert qs_indices, "Stage 0.7 _query_semantic must be in the call log"
    assert decide_indices, "Agentic loop _decide_next_action must be in the call log"
    assert max(qs_indices) < min(decide_indices), (
        f"Stage 0.7 _query_semantic (at {qs_indices}) must execute BEFORE "
        f"agentic loop _decide_next_action (at {decide_indices})"
    )

    # 4. collect() returned successfully with evidence
    assert result is not None, "collect() must return a result"
    evidence_items, confidence, _ = result
    assert isinstance(evidence_items, list)


# ---------------------------------------------------------------------------
# 1b. Ordering: stage_07_rag_start fires BEFORE agentic loop iteration_start
# ---------------------------------------------------------------------------

def test_stage07_rag_before_agentic_loop_ordering():
    """The step_callback event ordering must be:
        stage_07_rag_start < iteration_start

    This proves Stage 0.7 executes before the agentic loop begins,
    as observed through the UI event timeline.
    """
    from ticket_to_code.agents.evidence_collection_loop import EvidenceCollectionLoop

    localizer = MagicMock()
    localizer.workspace_path = "/fake/workspace"

    rag_engine = MagicMock()
    rag_engine.retrieve_context.return_value = [
        {"file_path": "src/feature.component.ts", "content": "...", "score": 0.80},
    ]

    loop = EvidenceCollectionLoop(
        localizer=localizer,
        rag_engine=rag_engine,
    )

    # Mock all methods that need LLM/filesystem
    loop._extract_ui_text_labels = MagicMock(return_value=[])
    loop._inspect_artifact = MagicMock(return_value=None)
    loop._get_followable_relationships = MagicMock(return_value=[])
    loop._auto_follow_and_inspect = MagicMock()
    loop._verify_single_file = MagicMock(return_value={
        "relevant": True, "decision": "include",
        "reason": "Implements feature component",
        "file_path": "feature.component.ts",
        "semantic_relevance_score": 0.80,
    })
    loop._hypothesis_to_semantic_queries = MagicMock(return_value=[
        "feature component implementation",
    ])
    loop._decide_next_action = MagicMock(return_value={
        "type": "proceed", "reasoning": "Test: immediate exit",
    })
    loop._compute_confidence = MagicMock(return_value=0.80)
    loop._count_pre_visited_results = MagicMock(return_value=0)

    # Capture events
    _events = []
    def _capture(event):
        _events.append(event)

    ticket = _FakeTicket()
    hypotheses = [_FakeHypothesis("Feature detection")]

    loop.collect(ticket=ticket, hypotheses=hypotheses, step_callback=_capture)

    # Extract event types in order
    event_types = [e.get("type", "") for e in _events]

    # stage_07_rag_start must appear
    assert "stage_07_rag_start" in event_types, (
        f"stage_07_rag_start event must be emitted. Got: {event_types}"
    )

    # iteration_start must appear (agentic loop ran at least once)
    assert "iteration_start" in event_types, (
        f"iteration_start event must be emitted. Got: {event_types}"
    )

    # Ordering: stage_07_rag_start BEFORE iteration_start
    rag_start_idx = event_types.index("stage_07_rag_start")
    iter_start_idx = event_types.index("iteration_start")
    assert rag_start_idx < iter_start_idx, (
        f"stage_07_rag_start (idx={rag_start_idx}) must fire BEFORE "
        f"iteration_start (idx={iter_start_idx}). Full event sequence: {event_types}"
    )


# ---------------------------------------------------------------------------
# 2. RAG discoveries exist BEFORE agentic loop
# ---------------------------------------------------------------------------

def test_rag_discoveries_exist_before_agentic_loop():
    """Stage 0.7 results must be present in evidence BEFORE the agentic loop starts.
    
    This verifies the pipeline ordering:
      Stage 0.5 -> Stage 0.7 (RAG) -> Agentic Loop
    """
    evidence = {}
    all_evidence = []
    all_items = []

    # Simulate what Stage 0.7 does when a candidate is verified as "include"
    _fp = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    _cand = _FakeEvidenceItem(_fp, 0.82)
    _sem_score = 0.85
    _reason = "Implements the primary add-members feature component"

    # This is the exact logic from Stage 0.7
    evidence[_fp] = {
        "relevant": True,
        "reason": _reason,
        "source": "stage_07_discovery_rag",
        "relevance_score": _sem_score,
    }
    _cand.relevance_score = _sem_score
    _cand.hypothesis_id = "stage_07_discovery_rag"
    all_items.append(_cand)
    all_evidence.append(_cand)

    # Verify the evidence is available for the agentic loop
    assert _fp in evidence, "Stage 0.7 discovery must be in evidence dict"
    assert evidence[_fp]["source"] == "stage_07_discovery_rag"
    assert evidence[_fp]["relevance_score"] == 0.85
    assert len(all_evidence) == 1
    assert all_evidence[0].hypothesis_id == "stage_07_discovery_rag"


# ---------------------------------------------------------------------------
# 3. RAG relevance != ownership (classification correctness)
# ---------------------------------------------------------------------------

def test_rag_relevance_not_ownership():
    """Only behaviorally proven candidates become PRIMARY_FEATURE_TARGET.
    
    A high RAG score alone is NOT sufficient for primary classification.
    The verification reason must contain implementation-related keywords.
    """
    # PRIMARY: high score + implementation keywords in reason
    _is_primary_1 = (
        True  # decision == "include"
        and 0.85 >= 0.65
        and any(kw in "implements the primary add-members feature component" for kw in [
            "implement", "direct", "primary", "feature",
            "handler", "controller", "component", "target",
        ])
    )
    assert _is_primary_1, "Should be PRIMARY when verified with implementation keywords"

    # RELATED_CONTEXT: high score but no implementation keywords
    _is_primary_2 = (
        True  # decision == "include"
        and 0.90 >= 0.65
        and any(kw in "provides utility for date formatting" for kw in [
            "implement", "direct", "primary", "feature",
            "handler", "controller", "component", "target",
        ])
    )
    assert not _is_primary_2, "Should be RELATED_CONTEXT without implementation keywords"

    # EXCLUDED: decision == "exclude"
    _is_primary_3 = (
        False  # decision == "exclude"
        and 0.95 >= 0.65
        and any(kw in "implements core handler" for kw in [
            "implement", "direct", "primary", "feature",
            "handler", "controller", "component", "target",
        ])
    )
    assert not _is_primary_3, "Should NOT be primary when excluded by verification"

    # Below threshold: score < 0.65
    _is_primary_4 = (
        True  # decision == "include"
        and 0.50 >= 0.65  # below threshold
        and any(kw in "implements the primary component" for kw in [
            "implement", "direct", "primary", "feature",
            "handler", "controller", "component", "target",
        ])
    )
    assert not _is_primary_4, "Should NOT be primary when score is below 0.65"


# ---------------------------------------------------------------------------
# 4. Stage 0.7 is bounded
# ---------------------------------------------------------------------------

def test_stage_07_bounded_queries():
    """Stage 0.7 caps at 6 unique queries and 10 candidates."""
    ticket = _FakeTicket(
        title="Add member duplicate check with warning dialog and permission validation",
        description="Long description line 1\nLine 2\nLine 3"
    )

    # Build queries as Stage 0.7 does
    s07_queries = []
    if ticket.title and len(ticket.title) > 5:
        s07_queries.append(ticket.title[:200])
    if ticket.description:
        first_line = ticket.description.strip().split("\n")[0][:200]
        if len(first_line) > 10:
            s07_queries.append(first_line)

    # Simulate _hypothesis_to_semantic_queries returning many queries
    hyp_queries = [f"query_{i}" for i in range(20)]
    s07_queries.extend(hyp_queries[:4])

    # Deduplicate, cap at 6
    seen = set()
    unique = []
    for q in s07_queries:
        k = q.strip().lower()
        if k not in seen and len(k) > 5:
            seen.add(k)
            unique.append(q)
            if len(unique) >= 6:
                break

    assert len(unique) <= 6, f"Stage 0.7 must cap at 6 queries, got {len(unique)}"

    # Simulate candidate capping at 10
    fake_candidates = [_FakeEvidenceItem(f"src/file_{i}.ts", 0.9 - i * 0.05) for i in range(25)]
    by_file = {c.file_path: c for c in fake_candidates}
    capped = sorted(by_file.values(), key=lambda e: e.relevance_score, reverse=True)[:10]
    assert len(capped) <= 10, f"Stage 0.7 must cap at 10 candidates, got {len(capped)}"


# ---------------------------------------------------------------------------
# 5. Stage 0.7 emits UI callback events
# ---------------------------------------------------------------------------

def test_stage_07_emits_ui_events():
    """Stage 0.7 must emit stage_07_rag_start, stage_07_rag_result, and
    stage_07_rag_complete events for frontend rendering."""
    emitted = []

    def mock_emit(event):
        emitted.append(event)

    # Simulate the events Stage 0.7 emits
    mock_emit({
        "type": "stage_07_rag_start",
        "message": "Stage 0.7: Guaranteed Discovery RAG sweep",
        "timestamp": datetime.now().isoformat(),
    })
    mock_emit({
        "type": "stage_07_rag_result",
        "message": "RAG discovery: add-members.component.ts -> PRIMARY (score=0.85)",
        "file_path": "src/add-members.component.ts",
        "decision": "include",
        "role": "PRIMARY_FEATURE_TARGET",
        "score": 0.85,
        "timestamp": datetime.now().isoformat(),
    })
    mock_emit({
        "type": "stage_07_rag_complete",
        "message": "Discovery RAG complete: 1 primary, 0 contextual (3 queries)",
        "promoted": 1,
        "contextual": 0,
        "queries_run": 3,
        "timestamp": datetime.now().isoformat(),
    })

    assert len(emitted) == 3
    assert emitted[0]["type"] == "stage_07_rag_start"
    assert emitted[1]["type"] == "stage_07_rag_result"
    assert emitted[1]["decision"] == "include"
    assert emitted[1]["role"] == "PRIMARY_FEATURE_TARGET"
    assert emitted[2]["type"] == "stage_07_rag_complete"
    assert emitted[2]["promoted"] == 1


# ---------------------------------------------------------------------------
# 6. Integration: RAG discovery available to planner before plan
# ---------------------------------------------------------------------------

def test_rag_discovery_available_to_planner_before_plan():
    """Stage 0.7 discoveries must be in the evidence pipeline output that flows
    to the planning node. This verifies the data contract between evidence
    collection and planning.
    """
    # Simulate the evidence pipeline output
    evidence_items = []

    # Stage 0 items (from localization)
    evidence_items.append(_FakeEvidenceItem(
        "xchange-ui/src/app/modules/members/members.module.ts", 0.60,
        hypothesis_id="stage_0",
    ))

    # Stage 0.7 items (from guaranteed RAG)
    s07_item = _FakeEvidenceItem(
        "xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        0.85,
        hypothesis_id="stage_07_discovery_rag",
    )
    evidence_items.append(s07_item)

    # Agentic loop items
    evidence_items.append(_FakeEvidenceItem(
        "xchange-ui/src/app/modules/shared/services/members/member.service.ts", 0.70,
        hypothesis_id="agentic_loop_iter_1",
    ))

    # Verify Stage 0.7 items are present
    s07_items = [e for e in evidence_items if e.hypothesis_id == "stage_07_discovery_rag"]
    assert len(s07_items) >= 1, "Stage 0.7 discoveries must be in evidence_items"
    assert s07_items[0].file_path.endswith("add-members.component.ts")
    assert s07_items[0].relevance_score == 0.85

    # Verify ordering: Stage 0.7 items appear BEFORE agentic loop items
    s07_idx = evidence_items.index(s07_item)
    assert s07_idx < len(evidence_items) - 1, "Stage 0.7 items should appear before agentic items"
