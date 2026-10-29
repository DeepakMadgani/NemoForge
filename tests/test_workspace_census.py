"""Unit tests for the Dynamic Workspace Architecture & Database Census Engine."""

from pathlib import Path
import pytest

from ticket_to_code.lean.workspace_census import (
    discover_services,
    scan_database_entities_and_migrations,
    run_workspace_census,
    build_census_markdown_summary,
)


def test_generic_polyglot_monorepo_discovery(tmp_path):
    """Verify that multi-ecosystem monorepos are discovered dynamically without hardcoding."""
    # 1. Spring Boot microservice with Flyway migrations
    orders = tmp_path / "orders-service"
    orders.mkdir()
    (orders / "pom.xml").write_text("<project><dependencies><dependency><artifactId>spring-boot-starter-web</artifactId></dependency></dependencies></project>", encoding="utf-8")
    mig_dir = orders / "src" / "main" / "resources" / "db" / "migration"
    mig_dir.mkdir(parents=True)
    (mig_dir / "V1__init.sql").write_text("""
    CREATE TABLE customer_orders (
        id VARCHAR(36) PRIMARY KEY,
        customer_id VARCHAR(36) NOT NULL,
        total_amount NUMERIC(10, 2)
    );
    CREATE TABLE order_items (
        id VARCHAR(36) PRIMARY KEY,
        order_id VARCHAR(36) NOT NULL,
        sku VARCHAR(50) NOT NULL,
        FOREIGN KEY (order_id) REFERENCES customer_orders(id)
    );
    """, encoding="utf-8")

    # 2. Angular frontend
    ui = tmp_path / "storefront-web"
    ui.mkdir()
    (ui / "package.json").write_text('{"dependencies": {"@angular/core": "^18.0.0"}}', encoding="utf-8")
    (ui / "angular.json").write_text("{}", encoding="utf-8")

    # 3. Python FastAPI service with Prisma
    auth = tmp_path / "auth-service"
    auth.mkdir()
    (auth / "pyproject.toml").write_text('[project]\ndependencies = ["fastapi", "uvicorn"]', encoding="utf-8")
    prisma_dir = auth / "prisma"
    prisma_dir.mkdir()
    (prisma_dir / "schema.prisma").write_text("""
    model UserAccount {
        id String @id
        email String @unique
    }
    model SessionToken {
        token String @id
    }
    """, encoding="utf-8")

    # 4. Go API service
    payments = tmp_path / "payments-service"
    payments.mkdir()
    (payments / "go.mod").write_text("module github.com/example/payments\nrequire github.com/gin-gonic/gin v1.9.0", encoding="utf-8")

    # Run census
    census = run_workspace_census(tmp_path, force_refresh=True)

    # Assertions
    svc_names = {s.name for s in census.services}
    assert svc_names == {"orders-service", "storefront-web", "auth-service", "payments-service"}

    # Framework detection
    orders_svc = next(s for s in census.services if s.name == "orders-service")
    assert orders_svc.framework == "Spring Boot Microservice"
    assert orders_svc.is_stateful is True
    assert set(orders_svc.tables) == {"customer_orders", "order_items"}
    assert any("order_id -> customer_orders(id)" in fk for fk in orders_svc.foreign_keys)

    ui_svc = next(s for s in census.services if s.name == "storefront-web")
    assert ui_svc.framework == "Angular Frontend"
    assert ui_svc.is_stateful is False

    auth_svc = next(s for s in census.services if s.name == "auth-service")
    assert auth_svc.framework == "FastAPI Service"
    assert auth_svc.is_stateful is True
    assert set(auth_svc.tables) == {"useraccount", "sessiontoken"}

    pay_svc = next(s for s in census.services if s.name == "payments-service")
    assert pay_svc.framework == "Go Gin API"
    assert pay_svc.is_stateful is False

    # Summary markdown verification
    md = build_census_markdown_summary(tmp_path, is_detailed=True)
    assert "orders-service" in md
    assert "Spring Boot Microservice" in md
    assert "customer_orders" in md
    assert "order_id -> customer_orders(id)" in md
    assert "storefront-web" in md
    assert "Angular Frontend" in md
    assert "auth-service" in md
    assert "FastAPI Service" in md
    assert "payments-service" in md
    assert "Go Gin API" in md
    # No hardcoded CC4E references
    assert "CC4E" not in md
    assert "project-service" not in md


def test_nested_monorepo_discovery(tmp_path):
    """Verify that nested monorepo structures (apps/*, services/*) are detected."""
    apps_dir = tmp_path / "apps"
    services_dir = tmp_path / "services"
    apps_dir.mkdir()
    services_dir.mkdir()

    # app 1: React web app
    portal = apps_dir / "portal"
    portal.mkdir()
    (portal / "package.json").write_text('{"dependencies": {"react": "^18.0.0", "react-dom": "^18.0.0"}}', encoding="utf-8")

    # service 1: Express API
    gateway = services_dir / "api-gateway"
    gateway.mkdir()
    (gateway / "package.json").write_text('{"dependencies": {"express": "^4.18.0"}}', encoding="utf-8")

    census = run_workspace_census(tmp_path, force_refresh=True)
    svc_names = {s.name for s in census.services}
    assert "portal" in svc_names
    assert "api-gateway" in svc_names

    portal_svc = next(s for s in census.services if s.name == "portal")
    assert portal_svc.framework == "React Application"

    gw_svc = next(s for s in census.services if s.name == "api-gateway")
    assert gw_svc.framework == "Express API Service"


def test_jpa_entity_fallback_when_no_sql_migrations(tmp_path):
    """Verify that JPA @Table annotations are detected when SQL migrations are absent."""
    svc = tmp_path / "inventory-service"
    svc.mkdir()
    (svc / "pom.xml").write_text("<project></project>", encoding="utf-8")
    java_dir = svc / "src" / "main" / "java" / "com" / "example" / "domain"
    java_dir.mkdir(parents=True)
    (java_dir / "WarehouseItem.java").write_text("""
    package com.example.domain;
    import javax.persistence.Entity;
    import javax.persistence.Table;

    @Entity
    @Table(name = "warehouse_items")
    public class WarehouseItem {
        private String id;
    }
    """, encoding="utf-8")

    census = run_workspace_census(tmp_path, force_refresh=True)
    inv = next(s for s in census.services if s.name == "inventory-service")
    assert inv.is_stateful is True
    assert "warehouse_items" in inv.tables
