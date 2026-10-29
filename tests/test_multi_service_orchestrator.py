"""
Unit tests for MultiServiceOrchestrator.
Verifies cross-service layer detection (Backend + Gateway + Frontend) and gate verification.
"""

from ticket_to_code.runtime.deductive_blackboard import DeductiveBlackboard
from ticket_to_code.runtime.multi_service_orchestrator import ArchitectureLayer, MultiServiceOrchestrator


def test_multi_service_orchestrator_detects_all_layers_for_ticket_3():
    bb = DeductiveBlackboard()
    orchestrator = MultiServiceOrchestrator(blackboard=bb)

    title = "CF-COMPLEX-03: Deliverables & Issues: Cross-Service Severity Breakdown & Priority Badging"
    desc = """
    Currently, deliverables grid displays a simple issue count.
    We need to extend the backend issues-service to break down issue counts by priority (critical, high, medium, low).
    Verify bim-gateway routing for /api/v1/issues/** endpoints.
    In the Angular xchange-ui deliverables grid, render color-coded priority badges with hover tooltips.
    """

    layers = orchestrator.analyze_ticket(title, desc)

    # Should detect BACKEND, GATEWAY, and FRONTEND
    assert ArchitectureLayer.BACKEND in layers
    assert ArchitectureLayer.GATEWAY in layers
    assert ArchitectureLayer.FRONTEND in layers

    # Initially, not all layers are verified
    assert not orchestrator.are_all_layers_verified()
    assert len(orchestrator.get_pending_layers()) == 3

    # Mark layers verified one by one
    orchestrator.mark_layer_verified(ArchitectureLayer.BACKEND, "IssuesServiceImpl JPQL query extended to 12 expressions")
    assert not orchestrator.are_all_layers_verified()

    orchestrator.mark_layer_verified(ArchitectureLayer.GATEWAY, "bim-gateway application.yml proxies /api/v1/issues/** without payload rewrite")
    orchestrator.mark_layer_verified(ArchitectureLayer.FRONTEND, "deliverable.component.html renders .priority-badge with custom styling")

    # Now all layers are verified
    assert orchestrator.are_all_layers_verified()
    assert len(orchestrator.get_pending_layers()) == 0

    # Blackboard received the verified facts
    assert any("BACKEND" in f for f in bb.verified_facts)
    assert any("GATEWAY" in f for f in bb.verified_facts)
    assert any("FRONTEND" in f for f in bb.verified_facts)

    # Checklist renders clean markdown
    md = orchestrator.render_checklist_markdown()
    assert "✅ **BACKEND**" in md
    assert "✅ **GATEWAY**" in md
    assert "✅ **FRONTEND**" in md


def test_multi_service_orchestrator_manifest_driven_unconventional_names(tmp_path):
    """Verifies that services with unconventional names (auth-hub, billing-engine, orders-client)
    are discovered directly from workspace manifests without relying on regex suffixes."""
    # 1. Create realistic polyglot workspace
    auth_dir = tmp_path / "auth-hub"
    auth_dir.mkdir()
    (auth_dir / "pom.xml").write_text("<project></project>", encoding="utf-8")

    client_dir = tmp_path / "orders-client"
    client_dir.mkdir()
    (client_dir / "package.json").write_text('{"name": "orders-client", "dependencies": {"@angular/core": "16.0"}}', encoding="utf-8")

    db_dir = tmp_path / "migrations"
    db_dir.mkdir()
    (db_dir / "V1__init.sql").write_text("CREATE TABLE t (id INT);", encoding="utf-8")

    orchestrator = MultiServiceOrchestrator(workspace_path=str(tmp_path))

    title = "Refresh authentication token in auth-hub and render notification in orders-client"
    desc = "Update auth token refresh cycle in auth-hub and update migrations in migrations/"

    layers = orchestrator.analyze_ticket(title, desc)

    assert ArchitectureLayer.BACKEND in layers
    assert ArchitectureLayer.FRONTEND in layers
    assert ArchitectureLayer.DATABASE in layers

    # Real service names from manifests were bound, not placeholders!
    assert orchestrator.detected_layers[ArchitectureLayer.BACKEND].service_name == "auth-hub"
    assert orchestrator.detected_layers[ArchitectureLayer.FRONTEND].service_name == "orders-client"
    assert orchestrator.detected_layers[ArchitectureLayer.DATABASE].service_name == "migrations"

