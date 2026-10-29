"""
Cross-Service Multi-Layer Coordinator.

Coordinates architectural tickets that span multiple microservices, gateways,
and frontend applications (e.g. Java Spring Boot + Spring Cloud Gateway + Angular UI).
Prevents single-service blind spots by enforcing multi-layer investigation checklists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ticket_to_code.runtime.deductive_blackboard import DeductiveBlackboard
from ticket_to_code.runtime.multi_zone_orchestrator import DynamicZoneDiscoverer, ZoneTier


class ArchitectureLayer(str, Enum):
    BACKEND = "backend"      # Java / Spring Boot microservices
    GATEWAY = "gateway"      # Spring Cloud Gateway / reverse proxies
    FRONTEND = "frontend"    # Angular / TypeScript UI
    DATABASE = "database"    # Flyway SQL migrations / PostgreSQL schema


@dataclass
class LayerRequirement:
    layer: ArchitectureLayer
    service_name: str
    target_artifacts: List[str]
    verified: bool = False
    evidence: List[str] = field(default_factory=list)


class MultiServiceOrchestrator:
    """Orchestrates cross-service full-stack tickets across Backend, Gateway, Frontend, and Database layers."""

    def __init__(self, blackboard: Optional[DeductiveBlackboard] = None, workspace_path: Optional[str] = None):
        self.blackboard = blackboard or DeductiveBlackboard()
        self.workspace_path = workspace_path
        self.detected_layers: Dict[ArchitectureLayer, LayerRequirement] = {}

    def _extract_services_and_artifacts(self, text: str) -> Tuple[Dict[ArchitectureLayer, str], Dict[ArchitectureLayer, List[str]]]:
        """Dynamically extracts real service names and target artifacts by inspecting workspace manifests."""
        text_lower = text.lower()
        services_by_layer: Dict[ArchitectureLayer, str] = {}
        artifacts_by_layer: Dict[ArchitectureLayer, List[str]] = {
            ArchitectureLayer.BACKEND: [],
            ArchitectureLayer.GATEWAY: [],
            ArchitectureLayer.FRONTEND: [],
            ArchitectureLayer.DATABASE: [],
        }

        # 1. Ground truth: Inspect real workspace compilation zones and build manifests
        if self.workspace_path and Path(self.workspace_path).exists():
            try:
                zones = DynamicZoneDiscoverer.discover_zones(Path(self.workspace_path))
                for z in zones:
                    z_name_lower = z.name.lower()
                    # Check if the real service / zone name or key tokens appear in the ticket
                    tokens = [t for t in re.split(r"[-_.]", z_name_lower) if len(t) >= 3]
                    is_match = (z_name_lower in text_lower) or any(t in text_lower for t in tokens)

                    if is_match or len(zones) == 1:
                        if z.tier == ZoneTier.CLIENT_UI:
                            services_by_layer.setdefault(ArchitectureLayer.FRONTEND, z.name)
                        elif z.tier == ZoneTier.CONTRACT_SCHEMA:
                            services_by_layer.setdefault(ArchitectureLayer.DATABASE, z.name)
                        elif "gateway" in z_name_lower or "proxy" in z_name_lower or "router" in z_name_lower:
                            services_by_layer.setdefault(ArchitectureLayer.GATEWAY, z.name)
                        else:
                            services_by_layer.setdefault(ArchitectureLayer.BACKEND, z.name)
            except Exception:
                pass

        # 2. Extract explicit service names mentioned in text as secondary discovery
        found_services = re.findall(r"\b([a-zA-Z0-9_\-]+(?:-service|-ui|-gateway|-api|-worker|-app|-backend|-frontend|-hub|-core))\b", text, re.IGNORECASE)
        for svc in found_services:
            svc_lower = svc.lower()
            if any(k in svc_lower for k in ("ui", "frontend", "client", "web")):
                services_by_layer.setdefault(ArchitectureLayer.FRONTEND, svc)
            elif any(k in svc_lower for k in ("gateway", "proxy", "router")):
                services_by_layer.setdefault(ArchitectureLayer.GATEWAY, svc)
            elif any(k in svc_lower for k in ("service", "api", "backend", "worker", "hub", "core")):
                services_by_layer.setdefault(ArchitectureLayer.BACKEND, svc)

        # 3. Extract and partition target artifacts by file type
        found_artifacts = re.findall(r"\b([A-Za-z0-9_\-]+\.(?:java|ts|html|scss|sql|yml|yaml|py|go|rs))\b", text, re.IGNORECASE)
        for art in found_artifacts:
            art_lower = art.lower()
            if art_lower.endswith(".sql"):
                artifacts_by_layer[ArchitectureLayer.DATABASE].append(art)
            elif art_lower.endswith((".ts", ".html", ".scss", ".jsx", ".tsx", ".vue")):
                artifacts_by_layer[ArchitectureLayer.FRONTEND].append(art)
            elif art_lower.endswith((".yml", ".yaml")):
                artifacts_by_layer[ArchitectureLayer.GATEWAY].append(art)
            elif art_lower.endswith((".java", ".py", ".go", ".rs", ".kt")):
                artifacts_by_layer[ArchitectureLayer.BACKEND].append(art)

        return services_by_layer, artifacts_by_layer

    def analyze_ticket(self, title: str, description: str) -> List[ArchitectureLayer]:
        """Analyze ticket problem statement to identify required architectural layers."""
        text = f"{title}\n{description}"
        text_lower = text.lower()
        layers: List[ArchitectureLayer] = []
        services_by_layer, artifacts_by_layer = self._extract_services_and_artifacts(text)

        # Backend layer: if discovered manifest service OR explicit backend keywords
        if ArchitectureLayer.BACKEND in services_by_layer or any(w in text_lower for w in ["issue", "backend", "endpoint", "api", "service", "controller", "count", "dto", "query"]):
            layers.append(ArchitectureLayer.BACKEND)
            backend_svc = services_by_layer.get(ArchitectureLayer.BACKEND, "backend-service")
            backend_arts = artifacts_by_layer[ArchitectureLayer.BACKEND] or ["Service.java", "Controller.java"]
            self.detected_layers[ArchitectureLayer.BACKEND] = LayerRequirement(
                layer=ArchitectureLayer.BACKEND,
                service_name=backend_svc,
                target_artifacts=backend_arts,
            )

        # Gateway layer: if discovered manifest gateway OR explicit gateway keywords
        if ArchitectureLayer.GATEWAY in services_by_layer or any(w in text_lower for w in ["gateway", "route", "proxy", "/api/v1"]):
            layers.append(ArchitectureLayer.GATEWAY)
            gw_svc = services_by_layer.get(ArchitectureLayer.GATEWAY, "api-gateway")
            gw_arts = artifacts_by_layer[ArchitectureLayer.GATEWAY] or ["application.yml"]
            self.detected_layers[ArchitectureLayer.GATEWAY] = LayerRequirement(
                layer=ArchitectureLayer.GATEWAY,
                service_name=gw_svc,
                target_artifacts=gw_arts,
            )

        # Frontend layer: if discovered manifest UI OR explicit frontend keywords
        if ArchitectureLayer.FRONTEND in services_by_layer or any(w in text_lower for w in ["ui", "badge", "badging", "tooltip", "angular", "component", "grid", "deliverable", "frontend", "color"]):
            layers.append(ArchitectureLayer.FRONTEND)
            fe_svc = services_by_layer.get(ArchitectureLayer.FRONTEND, "frontend-ui")
            fe_arts = artifacts_by_layer[ArchitectureLayer.FRONTEND] or ["component.ts", "component.html"]
            self.detected_layers[ArchitectureLayer.FRONTEND] = LayerRequirement(
                layer=ArchitectureLayer.FRONTEND,
                service_name=fe_svc,
                target_artifacts=fe_arts,
            )

        # Database layer: if discovered manifest schema OR explicit migration/db keywords
        if ArchitectureLayer.DATABASE in services_by_layer or any(w in text_lower for w in ["migration", "table", "column", "database", "postgres", "sql"]):
            layers.append(ArchitectureLayer.DATABASE)
            db_svc = services_by_layer.get(ArchitectureLayer.DATABASE) or services_by_layer.get(ArchitectureLayer.BACKEND, "database-service")
            db_arts = artifacts_by_layer[ArchitectureLayer.DATABASE] or ["V1__schema.sql"]
            self.detected_layers[ArchitectureLayer.DATABASE] = LayerRequirement(
                layer=ArchitectureLayer.DATABASE,
                service_name=db_svc,
                target_artifacts=db_arts,
            )

        return layers

    def mark_layer_verified(self, layer: ArchitectureLayer, evidence_summary: str) -> None:
        """Mark a specific architectural layer as verified with concrete evidence."""
        if layer in self.detected_layers:
            req = self.detected_layers[layer]
            req.verified = True
            req.evidence.append(evidence_summary)
            self.blackboard.record_fact(f"Architecture Layer '{layer.value.upper()}' verified: {evidence_summary}")

    def are_all_layers_verified(self) -> bool:
        """Checks if all detected layers have been inspected and verified."""
        if not self.detected_layers:
            return True
        return all(req.verified for req in self.detected_layers.values())

    def get_pending_layers(self) -> List[ArchitectureLayer]:
        """Returns any architectural layers that still lack verified evidence."""
        return [layer for layer, req in self.detected_layers.items() if not req.verified]

    def render_checklist_markdown(self) -> str:
        """Renders an architectural checklist of all required layers and their verification status."""
        lines = ["### Cross-Service Architectural Verification Checklist:"]
        for layer, req in self.detected_layers.items():
            status_icon = "✅" if req.verified else "⏳"
            targets = ", ".join(f"`{a}`" for a in req.target_artifacts[:3])
            lines.append(f"- {status_icon} **{layer.value.upper()}** ({req.service_name}): Targets [{targets}]")
            for ev in req.evidence:
                lines.append(f"    * Verified: {ev}")
        return "\n".join(lines)
