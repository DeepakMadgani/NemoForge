"""
Domain Ownership & Architecture Grounding Layer.

This module provides the missing "is this the right service for this feature?"
abstraction.  The existing pipeline gates ask "is this file related?" (keyword
similarity, file existence, module legality) — but never ask whether a
microservice is AUTHORIZED to own a particular business capability.

Core Design Principles (from user review):

    1. UNKNOWN ≠ ALLOW.  Unknown ownership → REQUIRE_VERIFICATION, not blind ALLOW.
    2. Relevance ≠ Ownership.  A file can be highly relevant (read for context)
       but architecturally wrong (must not modify).  These are separate dimensions.
    3. Primary / Secondary / Reference.  Don't use anti_services as core abstraction.
       Instead: PRIMARY_OWNER > SECONDARY_PARTICIPANT > REFERENCE > WRONG_DOMAIN.
    4. CREATE paths.  Ownership must work for files that don't exist yet —
       resolve via path prefix, not file existence.
    5. Hard when known, soft when unknown.  Known violations = hard block.
       Unknown ownership = investigate, don't guess.
    6. Provenance.  Every capability fact tracks its source (DECLARATIVE, DISCOVERED,
       LLM_INFERRED) so only verified facts create hard blocks.

Usage Flow:
    1. WorkspaceIntelligenceAgent loads ArchitectureModel from architecture_model.yaml
    2. DomainResolver extracts ticket domains from title/description
    3. OwnershipResolver classifies each candidate → verdict + policy
    4. plan_node splits candidates into IMPLEMENTATION vs REFERENCE pools
    5. run_gating_logic hard-rejects WRONG_DOMAIN tasks (Pass 1.6)
    6. IncrementalValidator catches post-generation violations
"""

from __future__ import annotations

import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class OwnershipVerdict(str, Enum):
    """Result of checking whether a candidate file is in the right service.

    Ordered from most permissive to most restrictive.
    """
    PRIMARY_OWNER = "PRIMARY_OWNER"          # This service is the primary owner of this capability
    SECONDARY_PARTICIPANT = "SECONDARY_PARTICIPANT"  # Legitimate participant, may need changes
    RELATED = "RELATED"                      # Can reference/read, should not create new logic
    REFERENCE_ONLY = "REFERENCE_ONLY"        # Can read for context, MUST NOT modify
    WRONG_DOMAIN = "WRONG_DOMAIN"            # Hard reject — known wrong bounded context
    UNKNOWN = "UNKNOWN"                      # No ownership data — requires verification


class ModificationPolicy(str, Enum):
    """Controls whether a candidate file can be planned for modification.

    Critical design: UNKNOWN → REQUIRE_VERIFICATION, not ALLOW.
    Only known ownership produces ALLOW.  Unknown ownership is investigated,
    not blindly permitted.
    """
    ALLOW = "ALLOW"                              # Known owner — can be planned for modification
    ALLOW_WITH_EVIDENCE = "ALLOW_WITH_EVIDENCE"  # Secondary participant — needs supporting evidence
    REFERENCE_ONLY = "REFERENCE_ONLY"            # Can read content but CANNOT create tasks
    REQUIRE_VERIFICATION = "REQUIRE_VERIFICATION"  # Unknown ownership — investigate before allowing
    DO_NOT_MODIFY = "DO_NOT_MODIFY"              # Hard block — known wrong domain


class CapabilityProvenance(str, Enum):
    """Source of a capability fact — determines whether violations are hard or soft.

    Only DECLARATIVE and VERIFIED facts produce hard blocks.
    DISCOVERED and LLM_INFERRED produce soft warnings + investigation.
    """
    DECLARATIVE = "DECLARATIVE"       # Developer-maintained in architecture_model.yaml
    DISCOVERED = "DISCOVERED"         # Auto-discovered from repository structure / APIs
    LLM_INFERRED = "LLM_INFERRED"    # Inferred by LLM during analysis
    VERIFIED = "VERIFIED"             # Was DISCOVERED or LLM_INFERRED, then confirmed


# ── Data Models ───────────────────────────────────────────────────────────────

class DomainCapability(BaseModel):
    """A business capability owned by a specific service.

    Example:
        name: "project-membership"
        keywords: ["member", "membership", "project member", "add member"]
        primary_owners: ["project-service"]
        secondary_participants: ["issues-service"]
        reference_only: ["notification-service"]
        provenance: "DECLARATIVE"
        confidence: 1.0
    """
    name: str = Field(..., description="Unique capability identifier, e.g. 'project-membership'")
    keywords: List[str] = Field(default_factory=list, description="Tokens/phrases that indicate this capability")
    primary_owners: List[str] = Field(default_factory=list, description="Services that OWN this capability")
    secondary_participants: List[str] = Field(
        default_factory=list,
        description="Services that legitimately participate (consume/react) but don't own the state",
    )
    reference_only: List[str] = Field(
        default_factory=list,
        description="Services that may read for context but should not implement here",
    )
    provenance: CapabilityProvenance = Field(
        default=CapabilityProvenance.DISCOVERED,
        description="Source of this capability fact",
    )
    confidence: float = Field(
        default=0.5,
        description="How confident the system is in this mapping (0.0-1.0). "
                    "Only high-confidence facts produce hard blocks.",
    )

    @property
    def is_hard_authority(self) -> bool:
        """Whether violations of this capability should be hard-blocked.

        Only DECLARATIVE and VERIFIED facts with high confidence produce hard blocks.
        """
        return (
            self.provenance in (CapabilityProvenance.DECLARATIVE, CapabilityProvenance.VERIFIED)
            and self.confidence >= 0.8
        )


class ServiceDomain(BaseModel):
    """A microservice's bounded context — its domain and what it owns.

    Example:
        name: "project-service"
        path_prefix: "project-service/"
        domain_description: "Project CRUD, lifecycle, project membership"
        capabilities: [DomainCapability(...)]
    """
    name: str = Field(..., description="Service name matching directory name")
    path_prefix: str = Field(..., description="Path prefix for files in this service, e.g. 'project-service/'")
    domain_description: str = Field(default="", description="Human-readable bounded context description")
    capabilities: List[DomainCapability] = Field(default_factory=list)

    @property
    def owned_capability_names(self) -> Set[str]:
        """All capability names this service primarily owns."""
        return {cap.name for cap in self.capabilities}

    @property
    def keyword_set(self) -> Set[str]:
        """All keywords across all capabilities, lowered."""
        result: Set[str] = set()
        for cap in self.capabilities:
            for kw in cap.keywords:
                result.add(kw.lower())
        return result


class CandidateOwnership(BaseModel):
    """Result of classifying a single candidate file against ticket domains.

    Separates relevance (how related is this file?) from ownership
    (is this service authorized to implement this capability?).
    These are fundamentally different dimensions.
    """
    verdict: OwnershipVerdict = OwnershipVerdict.UNKNOWN
    policy: ModificationPolicy = ModificationPolicy.REQUIRE_VERIFICATION
    service_name: str = ""          # The service this file belongs to
    expected_service: str = ""      # The service that SHOULD own this capability
    matched_capability: str = ""    # Which capability was matched
    confidence: float = 0.0         # How confident the match is (0.0-1.0)
    is_hard: bool = False           # Whether this verdict produces a hard block
    evidence: List[str] = Field(default_factory=list)  # Human-readable evidence trail

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "policy": self.policy.value,
            "service": self.service_name,
            "expected": self.expected_service,
            "capability": self.matched_capability,
            "confidence": self.confidence,
            "is_hard": self.is_hard,
            "evidence": self.evidence,
        }


class DomainResolution(BaseModel):
    """Result of resolving a ticket's business domain(s).

    Tracks primary vs secondary domains, authorized services, and
    the resolution method + confidence.
    """
    primary_domains: List[str] = Field(
        default_factory=list,
        description="Primary capability names, e.g. ['project-membership']",
    )
    secondary_domains: List[str] = Field(
        default_factory=list,
        description="Secondary/related capability names",
    )
    confidence: float = Field(default=0.0, description="Overall confidence in domain resolution")
    method: str = Field(default="none", description="Resolution method: 'keyword', 'evidence', 'hybrid', 'none'")
    authorized_services: List[str] = Field(
        default_factory=list,
        description="Services authorized as PRIMARY_OWNER for primary domains",
    )
    secondary_services: List[str] = Field(
        default_factory=list,
        description="Services that are SECONDARY_PARTICIPANT",
    )
    reference_services: List[str] = Field(
        default_factory=list,
        description="Services that may reference but not own/participate",
    )

    @property
    def has_resolution(self) -> bool:
        return len(self.primary_domains) > 0 and self.confidence > 0.0


# ── Architecture Model ────────────────────────────────────────────────────────

class ArchitectureModel(BaseModel):
    """Complete domain ownership model for the workspace.

    Loaded from `brain/knowledge/architecture_model.yaml` or auto-generated
    from WorkspaceIntelligenceAgent service discovery.

    Design: every service has a bounded context (domain). Every capability
    has primary owners, secondary participants, and reference-only services.
    Ownership is tracked with provenance and confidence so that only
    high-confidence declarative facts produce hard blocks.
    """
    services: List[ServiceDomain] = Field(default_factory=list)

    # ── Lookup Methods ────────────────────────────────────────────────────

    def get_service_for_file(self, file_path: str) -> Optional[ServiceDomain]:
        """Find which service a file path belongs to.

        Uses longest-prefix matching so that nested paths resolve correctly.
        Works for both existing files AND new CREATE target paths.
        e.g. 'project-service/src/main/java/...' → ServiceDomain(name='project-service')
        """
        normalized = file_path.replace("\\", "/").lower()

        best_match: Optional[ServiceDomain] = None
        best_len = 0

        for svc in self.services:
            prefix = svc.path_prefix.replace("\\", "/").lower().rstrip("/") + "/"
            if normalized.startswith(prefix) or normalized == prefix.rstrip("/"):
                if len(prefix) > best_len:
                    best_match = svc
                    best_len = len(prefix)

        return best_match

    def get_all_capabilities(self) -> List[DomainCapability]:
        """Flatten all capabilities across all services."""
        caps: List[DomainCapability] = []
        for svc in self.services:
            caps.extend(svc.capabilities)
        return caps

    def get_capability_by_name(self, name: str) -> Optional[DomainCapability]:
        """Find a capability by its unique name."""
        for svc in self.services:
            for cap in svc.capabilities:
                if cap.name == name:
                    return cap
        return None

    def get_primary_owners(self, capability_name: str) -> List[str]:
        """Get the service names that primarily own a capability."""
        cap = self.get_capability_by_name(capability_name)
        return list(cap.primary_owners) if cap else []

    def get_secondary_participants(self, capability_name: str) -> List[str]:
        """Get services that are legitimate secondary participants."""
        cap = self.get_capability_by_name(capability_name)
        return list(cap.secondary_participants) if cap else []

    def is_hard_authority_for(self, capability_name: str) -> bool:
        """Check if a capability has hard (declarative/verified) authority."""
        cap = self.get_capability_by_name(capability_name)
        return cap.is_hard_authority if cap else False

    def to_prompt_block(self) -> str:
        """Render a compact prompt block for LLM injection."""
        if not self.services:
            return ""
        lines = ["=== DOMAIN OWNERSHIP MODEL ==="]
        for svc in self.services:
            lines.append(f"  Service: {svc.name} ({svc.domain_description})")
            for cap in svc.capabilities:
                owners = ", ".join(cap.primary_owners) if cap.primary_owners else svc.name
                secondary = ", ".join(cap.secondary_participants)
                sec_str = f" | secondary: [{secondary}]" if secondary else ""
                provenance = cap.provenance.value
                lines.append(
                    f"    → {cap.name}: owned by [{owners}]{sec_str} "
                    f"(source: {provenance}, confidence: {cap.confidence:.1f})"
                )
        lines.append("===========================")
        return "\n".join(lines)

    # ── Loading ───────────────────────────────────────────────────────────

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "ArchitectureModel":
        """Load architecture model from a YAML file.

        Expected format:
            services:
              - name: project-service
                path_prefix: project-service/
                domain: "Project CRUD, lifecycle, membership"
                capabilities:
                  - name: project-membership
                    keywords: [member, membership]
                    primary_owners: [project-service]
                    secondary_participants: [issues-service]
                    reference_only: [notification-service]
                    provenance: DECLARATIVE
                    confidence: 1.0
        """
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"[ArchitectureModel] YAML not found: {yaml_path}")
            return cls()
        except Exception as e:
            logger.warning(f"[ArchitectureModel] Failed to load YAML: {e}")
            return cls()

        services: List[ServiceDomain] = []
        for svc_data in data.get("services", []):
            capabilities: List[DomainCapability] = []
            for cap_data in svc_data.get("capabilities", []):
                # Support both old format (owner/related/anti) and new format
                primary = cap_data.get("primary_owners", cap_data.get("owner", [svc_data.get("name", "")]))
                secondary = cap_data.get("secondary_participants", cap_data.get("related", []))
                ref_only = cap_data.get("reference_only", [])
                prov_str = cap_data.get("provenance", "DECLARATIVE")
                try:
                    provenance = CapabilityProvenance(prov_str)
                except ValueError:
                    provenance = CapabilityProvenance.DISCOVERED

                capabilities.append(DomainCapability(
                    name=cap_data.get("name", ""),
                    keywords=cap_data.get("keywords", []),
                    primary_owners=primary,
                    secondary_participants=secondary,
                    reference_only=ref_only,
                    provenance=provenance,
                    confidence=float(cap_data.get("confidence", 0.9 if prov_str == "DECLARATIVE" else 0.5)),
                ))
            services.append(ServiceDomain(
                name=svc_data.get("name", ""),
                path_prefix=svc_data.get("path_prefix", svc_data.get("name", "") + "/"),
                domain_description=svc_data.get("domain", ""),
                capabilities=capabilities,
            ))

        model = cls(services=services)
        logger.info(
            f"[ArchitectureModel] Loaded {len(services)} services, "
            f"{sum(len(s.capabilities) for s in services)} capabilities "
            f"from {yaml_path}"
        )
        return model

    @classmethod
    def from_workspace_knowledge(
        cls,
        services: List[Any],
        architecture_map_path: Optional[str] = None,
    ) -> "ArchitectureModel":
        """Auto-generate a basic architecture model from discovered services.

        This is the fallback when no architecture_model.yaml exists.
        It creates ServiceDomains with empty capabilities. Because
        UNKNOWN → REQUIRE_VERIFICATION (not ALLOW), the ownership gate
        will flag candidates for investigation rather than silently allowing.

        If architecture_map_path is provided (e.g. architecture_map.md),
        we parse it for routing rules to bootstrap basic capabilities.
        These are marked as DISCOVERED provenance (soft authority).
        """
        svc_domains: List[ServiceDomain] = []
        for svc in services:
            name = getattr(svc, "name", str(svc)) if not isinstance(svc, str) else svc
            path = getattr(svc, "path", name) if not isinstance(svc, str) else name
            svc_domains.append(ServiceDomain(
                name=name,
                path_prefix=path + "/" if not path.endswith("/") else path,
                domain_description="",
                capabilities=[],
            ))

        model = cls(services=svc_domains)

        # Try to bootstrap from architecture map markdown
        if architecture_map_path and os.path.exists(architecture_map_path):
            try:
                model._bootstrap_from_architecture_map(architecture_map_path)
            except Exception as e:
                logger.warning(f"[ArchitectureModel] Failed to bootstrap from map: {e}")

        return model

    def _bootstrap_from_architecture_map(self, map_path: str) -> None:
        """Parse an architecture_map.md style file to extract routing rules.

        Looks for lines like:
            - Tickets about Project API (project creation, lifecycle) → project-service/
            - Tickets about Area/Taxonomy API (area config, metadata) → area-service/

        These are marked as DISCOVERED provenance (soft authority, not hard blocks).
        """
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return

        # Match routing rules: "Tickets about X → service/"
        pattern = re.compile(
            r"Tickets about\s+(.+?)\s*(?:\(([^)]+)\))?\s*->\s*(\S+)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(content):
            topic = match.group(1).strip()
            details = match.group(2) or ""
            target_svc = match.group(3).strip().rstrip("/")

            # Find or create the target service
            svc = next((s for s in self.services if s.name == target_svc), None)
            if not svc:
                continue

            # Create a basic capability from the routing rule
            keywords = [w.strip().lower() for w in (topic + ", " + details).split(",") if w.strip()]
            cap_name = topic.lower().replace(" ", "-").replace("/", "-")
            svc.capabilities.append(DomainCapability(
                name=cap_name,
                keywords=keywords,
                primary_owners=[target_svc],
                secondary_participants=[],
                reference_only=[],
                provenance=CapabilityProvenance.DISCOVERED,
                confidence=0.5,  # Soft authority — won't create hard blocks
            ))

        logger.info(f"[ArchitectureModel] Bootstrapped from {map_path}")
