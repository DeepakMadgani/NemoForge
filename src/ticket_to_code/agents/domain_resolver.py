"""
Domain Resolver — extracts business domain(s) from ticket text.

Maps a ticket's title + description to DomainCapability names using:
  1. Keyword matching (fast, deterministic) — primary strategy
  2. N-gram overlap scoring for multi-word capability keywords
  3. Primary/secondary service propagation from matched capabilities

The resolver does NOT use LLM calls — it is purely deterministic so it
can run at zero cost/latency before RAG and planning.

Design notes (from user review):
  - Keyword matching alone is insufficient for ambiguous terms like "member"
    that exist across multiple bounded contexts. The resolver scores ALL
    matching capabilities and lets the OwnershipResolver disambiguate using
    repository evidence + architecture authority.
  - The resolver produces primary_domains and secondary_domains, plus
    the authorized / secondary / reference service lists derived from
    the ArchitectureModel.

Usage:
    from ticket_to_code.agents.architecture_model import ArchitectureModel
    from ticket_to_code.agents.domain_resolver import DomainResolver

    model = ArchitectureModel.load_from_yaml("architecture_model.yaml")
    resolver = DomainResolver(model)
    resolution = resolver.resolve("existing project member should show org name", "...")
    # resolution.primary_domains == ["project-membership"]
    # resolution.authorized_services == ["project-service"]
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ticket_to_code.agents.architecture_model import (
    ArchitectureModel,
    DomainCapability,
    DomainResolution,
)

logger = logging.getLogger(__name__)

# Minimum keyword match confidence to consider a capability resolved
_MIN_CONFIDENCE = 0.15

# Minimum confidence for a domain to be considered "primary" vs "secondary"
_PRIMARY_THRESHOLD = 0.4


class DomainResolver:
    """Resolves ticket text → business domain(s) → authorized services.

    The resolver works purely on keyword overlap between the ticket text
    and the DomainCapability.keywords defined in the ArchitectureModel.
    It is deterministic and fast (no LLM calls).

    For ambiguous cases (e.g. "member" matching both project-membership
    and area-membership), the resolver returns ALL matching capabilities
    ranked by score. The OwnershipResolver downstream uses architecture
    authority to disambiguate.
    """

    def __init__(self, architecture_model: ArchitectureModel):
        self._model = architecture_model
        # Pre-build keyword → capability index for fast lookup
        self._keyword_index: Dict[str, List[DomainCapability]] = defaultdict(list)
        self._build_index()

    def _build_index(self) -> None:
        """Build an inverted index: keyword → [capabilities that use it]."""
        for cap in self._model.get_all_capabilities():
            for kw in cap.keywords:
                normalized = kw.strip().lower()
                if normalized:
                    self._keyword_index[normalized].append(cap)

    def resolve(
        self,
        ticket_title: str,
        ticket_description: str = "",
    ) -> DomainResolution:
        """Determine which bounded contexts this ticket belongs to.

        Args:
            ticket_title: The ticket's title/summary.
            ticket_description: The ticket's full description (optional).

        Returns:
            DomainResolution with primary/secondary domains, authorized
            services, secondary services, reference services, and confidence.
        """
        if not self._model.services:
            return DomainResolution(method="none")

        # Combine and normalize ticket text
        full_text = f"{ticket_title} {ticket_description}".lower()
        full_text = re.sub(r"[^\w\s]", " ", full_text)  # strip punctuation
        tokens = set(full_text.split())

        # Score each capability by keyword overlap
        scores: Dict[str, float] = {}
        cap_lookup: Dict[str, DomainCapability] = {}

        for cap in self._model.get_all_capabilities():
            if not cap.keywords:
                continue
            score = self._score_capability(cap, full_text, tokens)
            if score >= _MIN_CONFIDENCE:
                # If multiple capabilities have the same name (across services),
                # keep the highest score
                if cap.name not in scores or score > scores[cap.name]:
                    scores[cap.name] = score
                    cap_lookup[cap.name] = cap

        if not scores:
            logger.info("[DomainResolver] No domain matched — ownership will be UNKNOWN")
            return DomainResolution(method="keyword", confidence=0.0)

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Split into primary vs secondary
        primary: List[str] = []
        secondary: List[str] = []
        for name, score in ranked:
            if score >= _PRIMARY_THRESHOLD:
                primary.append(name)
            else:
                secondary.append(name)

        # If nothing reached primary threshold, promote the best match
        if not primary and ranked:
            best_name, best_score = ranked[0]
            primary.append(best_name)
            if best_name in secondary:
                secondary.remove(best_name)

        # Collect authorized / secondary / reference services
        authorized: Set[str] = set()      # PRIMARY_OWNER services
        sec_services: Set[str] = set()     # SECONDARY_PARTICIPANT services
        ref_services: Set[str] = set()     # REFERENCE_ONLY services

        for domain_name in primary:
            cap = cap_lookup.get(domain_name)
            if cap:
                authorized.update(cap.primary_owners)
                sec_services.update(cap.secondary_participants)
                ref_services.update(cap.reference_only)

        for domain_name in secondary:
            cap = cap_lookup.get(domain_name)
            if cap:
                ref_services.update(cap.primary_owners)
                ref_services.update(cap.secondary_participants)

        # Hierarchy: authorized > secondary > reference (no overlaps)
        sec_services -= authorized
        ref_services -= authorized
        ref_services -= sec_services

        best_confidence = ranked[0][1] if ranked else 0.0

        resolution = DomainResolution(
            primary_domains=primary,
            secondary_domains=secondary,
            confidence=best_confidence,
            method="keyword",
            authorized_services=sorted(authorized),
            secondary_services=sorted(sec_services),
            reference_services=sorted(ref_services),
        )

        logger.info(
            f"[DomainResolver] Resolved → primary={primary}, "
            f"authorized={sorted(authorized)}, secondary={sorted(sec_services)}, "
            f"reference={sorted(ref_services)}, confidence={best_confidence:.2f}"
        )
        return resolution

    # ── Scoring ───────────────────────────────────────────────────────────

    def _score_capability(
        self,
        cap: DomainCapability,
        full_text: str,
        tokens: Set[str],
    ) -> float:
        """Score how well a capability's keywords match the ticket text.

        Uses two strategies:
        1. Single-token match: each keyword token found in ticket tokens
        2. Multi-word match: multi-word keywords found as substrings

        Multi-word keywords are weighted higher because they're more specific
        and less ambiguous (e.g. "project member" > "member").

        Returns a score in [0.0, 1.0].
        """
        if not cap.keywords:
            return 0.0

        total_weight = 0.0
        matched_weight = 0.0

        for kw in cap.keywords:
            kw_lower = kw.strip().lower()
            if not kw_lower:
                continue

            # Weight multi-word keywords higher (more specific = more valuable)
            word_count = len(kw_lower.split())
            weight = 1.0 + (word_count - 1) * 0.5  # 1.0 for 1-word, 1.5 for 2-word, etc.
            total_weight += weight

            if " " in kw_lower:
                # Multi-word keyword: substring match in full text
                if kw_lower in full_text:
                    matched_weight += weight
            else:
                # Single-word keyword: token match
                if kw_lower in tokens:
                    matched_weight += weight

        if total_weight == 0:
            return 0.0

        # Weighted overlap ratio
        return matched_weight / total_weight

    # ── Utility ───────────────────────────────────────────────────────────

    def get_authorized_services(
        self,
        ticket_title: str,
        ticket_description: str = "",
    ) -> List[str]:
        """Convenience: resolve and return just the authorized service names."""
        resolution = self.resolve(ticket_title, ticket_description)
        return resolution.authorized_services

    def is_service_authorized(
        self,
        service_name: str,
        ticket_title: str,
        ticket_description: str = "",
    ) -> bool:
        """Check if a specific service is authorized for this ticket's domain."""
        resolution = self.resolve(ticket_title, ticket_description)
        if not resolution.has_resolution:
            return True  # No domain resolved → no constraint
        return service_name in resolution.authorized_services
