"""Unit tests for MultiZoneOrchestrator and DynamicZoneDiscoverer.

Verifies:
1. Dynamic discovery of compilation zones across polyglot manifests without hardcoding.
2. Topological wave ordering (Contracts -> Producers -> Consumers -> UI).
3. Sandboxed wave execution with atomic rollback on failure.
4. Cross-zone contract drift detection.
"""

from pathlib import Path
from ticket_to_code.runtime.multi_zone_orchestrator import (
    DynamicZoneDiscoverer,
    MultiZoneOrchestrator,
    ZoneTier,
)


def test_dynamic_zone_discovery(tmp_path: Path):
    # Setup 4 distinct zones
    # 1. DB Schema zone
    db_dir = tmp_path / "db" / "migration"
    db_dir.mkdir(parents=True)
    (db_dir / "V1__init.sql").write_text("CREATE TABLE users (id INT);", encoding="utf-8")

    # 2. Java Producer Core
    producer_dir = tmp_path / "orders-service"
    producer_dir.mkdir()
    (producer_dir / "build.gradle").write_text("// gradle", encoding="utf-8")

    # 3. Consumer Service
    consumer_dir = tmp_path / "payment-consumer"
    consumer_dir.mkdir()
    (consumer_dir / "package.json").write_text('{"name": "consumer", "dependencies": {}}', encoding="utf-8")

    # 4. React UI
    ui_dir = tmp_path / "admin-portal"
    ui_dir.mkdir()
    (ui_dir / "package.json").write_text('{"name": "ui", "dependencies": {"react": "^18.0.0"}}', encoding="utf-8")

    zones = DynamicZoneDiscoverer.discover_zones(tmp_path)
    zone_names = {z.name: z.tier for z in zones}

    assert "migration" in zone_names or "db" in zone_names
    assert zone_names.get("migration", zone_names.get("db")) == ZoneTier.CONTRACT_SCHEMA
    assert zone_names.get("orders-service") == ZoneTier.PRODUCER_CORE
    assert zone_names.get("payment-consumer") == ZoneTier.CONSUMER_SERVICE
    assert zone_names.get("admin-portal") == ZoneTier.CLIENT_UI


def test_topological_wave_sorting(tmp_path: Path):
    # Setup 3 zones
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "schema.sql").write_text("CREATE TABLE t (id INT);", encoding="utf-8")

    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "pom.xml").write_text("<project></project>", encoding="utf-8")

    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text('{"dependencies": {"vite": "1.0"}}', encoding="utf-8")

    orch = MultiZoneOrchestrator(workspace_path=tmp_path)
    modified_files = [
        "schemas/schema.sql",
        "backend/src/main/java/App.java",
        "frontend/src/App.tsx",
    ]

    zone_map = orch.map_files_to_zones(modified_files)
    waves = orch.build_topological_waves(zone_map)

    # Must produce 3 ordered waves
    assert len(waves) == 3
    # Wave 0 must be schema/contract
    assert waves[0][0].tier == ZoneTier.CONTRACT_SCHEMA
    # Wave 1 must be backend producer
    assert waves[1][0].tier == ZoneTier.PRODUCER_CORE
    # Wave 2 must be frontend UI
    assert waves[2][0].tier == ZoneTier.CLIENT_UI


def test_multi_zone_execution_atomic_rollback_on_failure(tmp_path: Path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "pom.xml").write_text("<project></project>", encoding="utf-8")

    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text('{"dependencies": {"react": "18"}}', encoding="utf-8")

    orch = MultiZoneOrchestrator(workspace_path=tmp_path)
    modified_files = ["backend/App.java", "frontend/App.tsx"]

    # Mock verifier where backend succeeds but frontend fails
    def mock_verifier(zone, files):
        if zone.name == "backend":
            return True, "Backend build passed."
        return False, "Frontend syntax error in App.tsx: line 12."

    report = orch.execute_multi_zone_pipeline(modified_files, zone_verifier_fn=mock_verifier)

    assert not report.success
    assert report.rolled_back
    assert len(report.zone_results) == 2
    assert report.zone_results[0].success
    assert not report.zone_results[1].success
    assert "Frontend syntax error" in report.zone_results[1].error_message


def test_cross_zone_contract_drift_detection(tmp_path: Path):
    # Setup schema zone with a table
    mig_dir = tmp_path / "db" / "migration"
    mig_dir.mkdir(parents=True)
    (mig_dir / "V1__create_users.sql").write_text("""
    CREATE TABLE users (
        id VARCHAR(36) PRIMARY KEY,
        real_user_id VARCHAR(64) NOT NULL
    );
    """, encoding="utf-8")

    # Setup backend entity referencing non-existent column
    svc_dir = tmp_path / "user-service" / "src" / "main" / "java" / "com" / "models"
    svc_dir.mkdir(parents=True)
    (tmp_path / "user-service" / "build.gradle").write_text("// gradle", encoding="utf-8")
    (svc_dir / "User.java").write_text("""
    package com.models;
    import javax.persistence.*;

    @Entity
    @Table(name = "users")
    public class User {
        @Id
        private String id;
        @Column(name = "non_existent_column")
        private String userRef;
    }
    """, encoding="utf-8")

    orch = MultiZoneOrchestrator(workspace_path=tmp_path)
    modified_files = [
        "db/migration/V1__create_users.sql",
        "user-service/src/main/java/com/models/User.java",
    ]

    report = orch.execute_multi_zone_pipeline(
        modified_files,
        zone_verifier_fn=lambda z, f: (True, "Compiled successfully"),
    )

    # Even though both local builds passed, cross-zone contract check catches drift!
    assert not report.success
    assert report.rolled_back
    assert any("Cross-Zone Contract Drift" in v for v in report.contract_violations)
