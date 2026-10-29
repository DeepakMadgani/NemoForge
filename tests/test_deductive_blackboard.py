"""
Tests for DeductiveBlackboard (Reasoning & Hypothesis Tracker).
Verifies that tool observations are automatically transformed into verified facts,
rejected hypotheses, and anti-hallucination constraints.
"""

from ticket_to_code.runtime.deductive_blackboard import DeductiveBlackboard


def test_blackboard_deduces_java_string_priority_and_rejects_enum():
    bb = DeductiveBlackboard()
    
    # Tool observation from Issue.java
    tool_output = """
    package com.cloudforge.services.services.issues.entity;
    @Entity
    @Table(name = "issue")
    public class Issue extends BaseEntity {
        @Column(name = "priority")
        private String priority;
        @Column(name = "state")
        private String state;
    }
    """
    deductions = bb.ingest_tool_observation("inspect_file", "issues-service/src/main/.../Issue.java", tool_output)
    
    assert any("priority" in d and "String" in d for d in deductions)
    assert any("IssuePriority" in r for r in bb.rejected_hypotheses)
    assert any("UPPER(i.priority)" in r for r in bb.rejected_hypotheses)


def test_blackboard_deduces_dto_constructor_backward_compatibility():
    bb = DeductiveBlackboard()
    
    # Tool observation from IssueCount.java
    tool_output = """
    public class IssueCount {
        public IssueCount(long publicIssue, long privateIssue, long openedIssue,
                          long acceptedIssue, long rejectedIssue, long proposedRejectedIssue,
                          long closedIssue) {
            this.publicIssue = publicIssue;
        }
    }
    """
    deductions = bb.ingest_tool_observation("inspect_file", "issues-service/src/.../dto/IssueCount.java", tool_output)
    
    assert any("IssueCount" in d and "7 parameters" in d for d in deductions)
    assert any("DO NOT delete the existing 7-argument constructor" in r for r in bb.rejected_hypotheses)


def test_blackboard_deduces_enterprise_ui_and_rejects_angular_material():
    bb = DeductiveBlackboard()
    
    # Tool observation from deliverable.component.html
    tool_output = """
    <td cdk-cell *cdkCellDef="let row">
      <ot-icon-button size="md" icon="issue" (click)="navigateToOverlay(row, 'Issues')"></ot-icon-button>
    </td>
    """
    deductions = bb.ingest_tool_observation("view_file", "xchange-ui/.../deliverable.component.html", tool_output)
    
    assert bb.ui_framework_detected == "Enterprise UX Components (@cloudforge/components-lib)"
    assert any("Enterprise UX Components" in d for d in deductions)
    assert any("<mat-chip>" in r for r in bb.rejected_hypotheses)
    assert any("navigateToOverlay" in d for d in deductions)


def test_blackboard_deduces_spring_cloud_gateway_passthrough():
    bb = DeductiveBlackboard()
    
    # Tool observation from application.yml
    tool_output = """
          - id: issues
            predicates:
              - Path=/api/v1/issues/**
            uri: ${ISSUES_SERVICE_URL:http://localhost:8901}
    """
    deductions = bb.ingest_tool_observation("view_file", "bim-gateway/.../application.yml", tool_output)
    
    assert any("/api/v1/issues/**" in d for d in deductions)
    assert any("DO NOT modify Gateway routing" in r for r in bb.rejected_hypotheses)


def test_blackboard_renders_xml_block():
    bb = DeductiveBlackboard()
    bb.record_fact("Issue.priority is String")
    bb.record_rejection("Do not use IssuePriority enum")
    bb.record_endpoint("ISSUES_COUNT_URL: /{itemType}/{itemId}/count")
    
    xml = bb.render_prompt_xml()
    assert "<deductive_blackboard>" in xml
    assert "</deductive_blackboard>" in xml
    assert "Issue.priority is String" in xml
    assert "Do not use IssuePriority enum" in xml
    assert "ISSUES_COUNT_URL" in xml


def test_blackboard_structured_parsing():
    bb = DeductiveBlackboard()
    xml_data = """
    Here are the findings:
    <verified_fact file="src/main/IssueCount.java" entity="IssueCount">IssueCount DTO has 7 constructor arguments.</verified_fact>
    <verified_fact file="xchange-ui/deliverables.service.ts">Service calls /api/v1/issues/count via HttpClient.</verified_fact>
    ```json
    [
      {"file": "bim-gateway/application.yml", "fact": "Gateway route proxies /api/v1/issues/**"}
    ]
    ```
    """
    extracted = bb.parse_and_ingest_structured_facts(xml_data)
    assert len(extracted) == 3
    assert len(bb.verified_facts) == 3
    assert any("7 constructor arguments" in f for f in bb.verified_facts)
    assert any("HttpClient" in f for f in bb.verified_facts)
    assert any("Gateway route" in f for f in bb.verified_facts)


def test_blackboard_obsolescence_eviction():
    bb = DeductiveBlackboard()
    bb.record_fact("IssueCount constructor has 7 arguments", file_path="src/main/IssueCount.java")
    bb.record_fact("DeliverableComponent uses ot-icon-button", file_path="xchange-ui/deliverable.component.html")
    assert len(bb.verified_facts) == 2

    # Mutate IssueCount.java -> should evict old fact about IssueCount.java
    evicted = bb.evict_facts_for_path("src/main/IssueCount.java")
    assert evicted == 1
    assert len(bb.verified_facts) == 1
    assert "DeliverableComponent" in bb.verified_facts[0]

    # Contradiction eviction
    bb.record_fact("Old assumption about priority is true")
    evicted_c = bb.evict_contradicted_facts(r"Old assumption about priority")
    assert evicted_c == 1
    assert len(bb.verified_facts) == 1


def test_blackboard_disk_persistence_roundtrip(tmp_path):
    bb = DeductiveBlackboard(workspace_path=str(tmp_path))
    bb.record_fact("Issue.priority is String type", file_path="Issue.java", entity="Issue")
    bb.record_rejection("Do not use IssuePriority enum")
    bb.record_endpoint("GET /api/v1/issues/count")
    bb.record_entity_field("IssueCount", "criticalCount")
    bb.ui_framework_detected = "Enterprise UX Components"

    # Save to disk
    persisted_path = bb.save_to_disk("test_session_123")
    assert persisted_path is not None
    assert "test_session_123.json" in persisted_path

    # Hydrate fresh blackboard from disk
    fresh_bb = DeductiveBlackboard(workspace_path=str(tmp_path))
    loaded = fresh_bb.load_from_disk("test_session_123")
    assert loaded is True

    # Verify all deductions survived the roundtrip
    assert len(fresh_bb.verified_facts) == 1
    assert "Issue.priority is String type" in fresh_bb.verified_facts
    assert "Do not use IssuePriority enum" in fresh_bb.rejected_hypotheses
    assert "GET /api/v1/issues/count" in fresh_bb.discovered_endpoints
    assert "criticalCount" in fresh_bb.discovered_entities.get("IssueCount", [])
    assert fresh_bb.ui_framework_detected == "Enterprise UX Components"

