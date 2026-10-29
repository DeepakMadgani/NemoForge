"""
Ownership Resolver — classifies candidate files against ticket domains.

Given a candidate file path + resolved ticket domain(s), determines:
  1. OwnershipVerdict: PRIMARY_OWNER / SECONDARY_PARTICIPANT / RELATED /
     REFERENCE_ONLY / WRONG_DOMAIN / UNKNOWN
  2. ModificationPolicy: ALLOW / ALLOW_WITH_EVIDENCE / REFERENCE_ONLY /
     REQUIRE_VERIFICATION / DO_NOT_MODIFY

Critical design decisions (from user review):
  - UNKNOWN → REQUIRE_VERIFICATION, not ALLOW.
  - Hard blocks only for DECLARATIVE/VERIFIED facts with high confidence.
  - Works for both existing files AND new CREATE target paths (via path prefix).
  - Separate from semantic relevance — a file can be highly relevant but
    architecturally wrong.

Usage:
    from ticket_to_code.agents.ownership_resolver import OwnershipResolver

    resolver = OwnershipResolver(architecture_model)
    ownership = resolver.classify("area-service/MembersService.java", ["project-membership"])
    # ownership.verdict == WRONG_DOMAIN
    # ownership.policy == DO_NOT_MODIFY
    # ownership.is_hard == True (if declarative)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from ticket_to_code.agents.architecture_model import (
    ArchitectureModel,
    CandidateOwnership,
    DomainCapability,
    DomainResolution,
    ModificationPolicy,
    OwnershipVerdict,
    ServiceDomain,
)

logger = logging.getLogger(__name__)


class OwnershipResolver:
    """Classifies candidate files against ticket domains.

    The resolver answers: "Is this service AUTHORIZED to implement
    this capability?" — a fundamentally different question from
    "Is this file relevant?"

    Resolution logic:
        1. Resolve candidate file → service (via path prefix)
        2. For each ticket domain:
           a. If service is in primary_owners → PRIMARY_OWNER / ALLOW
           b. If service is in secondary_participants → SECONDARY / ALLOW_WITH_EVIDENCE
           c. If service is in reference_only → REFERENCE_ONLY / REFERENCE_ONLY
        3. If service exists but doesn't own ANY domain → WRONG_DOMAIN / DO_NOT_MODIFY
           (only if capability has hard authority; otherwise UNKNOWN / REQUIRE_VERIFICATION)
        4. If service unknown → UNKNOWN / REQUIRE_VERIFICATION
    """

    def __init__(self, architecture_model: ArchitectureModel):
        self._model = architecture_model

    def classify(
        self,
        file_path: str,
        ticket_domains: List[str],
    ) -> CandidateOwnership:
        """Classify a candidate file's relationship to the ticket's domain.

        Args:
            file_path: Relative path to the candidate file.
            ticket_domains: Primary domain capability names from DomainResolver.

        Returns:
            CandidateOwnership with verdict, policy, evidence, and is_hard flag.
        """
        if not ticket_domains:
            # No domains resolved — we cannot enforce ownership
            return CandidateOwnership(
                verdict=OwnershipVerdict.UNKNOWN,
                policy=ModificationPolicy.REQUIRE_VERIFICATION,
                evidence=["No ticket domain was resolved — ownership cannot be determined"],
            )

        # Step 1: Resolve file → service
        service = self._model.get_service_for_file(file_path)
        if not service:
            # File doesn't belong to any known service
            return CandidateOwnership(
                verdict=OwnershipVerdict.UNKNOWN,
                policy=ModificationPolicy.REQUIRE_VERIFICATION,
                service_name="<unknown>",
                evidence=[f"File '{file_path}' does not match any known service path prefix"],
            )

        # Step 2: Check each ticket domain against this service
        best_verdict = OwnershipVerdict.WRONG_DOMAIN
        best_policy = ModificationPolicy.DO_NOT_MODIFY
        best_capability = ""
        best_expected = ""
        best_confidence = 0.0
        is_hard = False
        evidence: List[str] = []

        for domain_name in ticket_domains:
            cap = self._model.get_capability_by_name(domain_name)
            if not cap:
                evidence.append(f"Capability '{domain_name}' not found in architecture model")
                continue

            # Check if this service is in the capability's ownership hierarchy
            if service.name in cap.primary_owners:
                # Primary owner — this is the RIGHT place
                return CandidateOwnership(
                    verdict=OwnershipVerdict.PRIMARY_OWNER,
                    policy=ModificationPolicy.ALLOW,
                    service_name=service.name,
                    expected_service=service.name,
                    matched_capability=domain_name,
                    confidence=cap.confidence,
                    is_hard=False,
                    evidence=[
                        f"Service '{service.name}' is PRIMARY_OWNER of '{domain_name}'",
                        f"Provenance: {cap.provenance.value}, confidence: {cap.confidence:.1f}",
                    ],
                )

            if service.name in cap.secondary_participants:
                # Secondary participant — may need changes but isn't the primary
                if best_verdict not in (OwnershipVerdict.PRIMARY_OWNER,):
                    best_verdict = OwnershipVerdict.SECONDARY_PARTICIPANT
                    best_policy = ModificationPolicy.ALLOW_WITH_EVIDENCE
                    best_capability = domain_name
                    best_expected = ", ".join(cap.primary_owners) if cap.primary_owners else "<unknown>"
                    best_confidence = cap.confidence
                    evidence.append(
                        f"Service '{service.name}' is SECONDARY_PARTICIPANT for '{domain_name}' "
                        f"(primary: {best_expected})"
                    )

            elif service.name in cap.reference_only:
                # Reference only — can read but should not modify
                if best_verdict not in (
                    OwnershipVerdict.PRIMARY_OWNER,
                    OwnershipVerdict.SECONDARY_PARTICIPANT,
                ):
                    best_verdict = OwnershipVerdict.REFERENCE_ONLY
                    best_policy = ModificationPolicy.REFERENCE_ONLY
                    best_capability = domain_name
                    best_expected = ", ".join(cap.primary_owners) if cap.primary_owners else "<unknown>"
                    best_confidence = cap.confidence
                    evidence.append(
                        f"Service '{service.name}' is REFERENCE_ONLY for '{domain_name}' "
                        f"(primary: {best_expected})"
                    )

            else:
                # Service not mentioned in this capability at all
                primary = ", ".join(cap.primary_owners) if cap.primary_owners else "<unknown>"
                evidence.append(
                    f"Service '{service.name}' is NOT listed for capability '{domain_name}' "
                    f"(primary owners: {primary})"
                )
                if not best_capability:
                    best_capability = domain_name
                    best_expected = primary
                    best_confidence = cap.confidence

                # Determine hard vs soft based on provenance
                if cap.is_hard_authority:
                    is_hard = True

        # If we fell through all domains without finding PRIMARY or SECONDARY...
        if best_verdict == OwnershipVerdict.WRONG_DOMAIN:
            if is_hard:
                # Known wrong domain with hard authority → DO_NOT_MODIFY
                evidence.append(
                    f"HARD BLOCK: Service '{service.name}' is not authorized for "
                    f"any of {ticket_domains} (high-confidence declarative fact)"
                )
                return CandidateOwnership(
                    verdict=OwnershipVerdict.WRONG_DOMAIN,
                    policy=ModificationPolicy.DO_NOT_MODIFY,
                    service_name=service.name,
                    expected_service=best_expected,
                    matched_capability=best_capability,
                    confidence=best_confidence,
                    is_hard=True,
                    evidence=evidence,
                )
            else:
                # Unknown / low-confidence → REQUIRE_VERIFICATION
                evidence.append(
                    f"SOFT: Service '{service.name}' not listed for {ticket_domains} "
                    f"but authority is low-confidence — requires investigation"
                )
                return CandidateOwnership(
                    verdict=OwnershipVerdict.UNKNOWN,
                    policy=ModificationPolicy.REQUIRE_VERIFICATION,
                    service_name=service.name,
                    expected_service=best_expected,
                    matched_capability=best_capability,
                    confidence=best_confidence,
                    is_hard=False,
                    evidence=evidence,
                )

        return CandidateOwnership(
            verdict=best_verdict,
            policy=best_policy,
            service_name=service.name,
            expected_service=best_expected,
            matched_capability=best_capability,
            confidence=best_confidence,
            is_hard=is_hard,
            evidence=evidence,
        )

    def classify_batch(
        self,
        candidates: List[Dict],
        ticket_domains: List[str],
        path_key: str = "path",
    ) -> List[Dict]:
        """Classify a batch of candidates, enriching each dict in-place.

        Adds 'ownership_verdict', 'modification_policy', 'ownership_evidence',
        and 'ownership_is_hard' keys to each candidate dict.

        Args:
            candidates: List of candidate dicts with a path field.
            ticket_domains: Primary domain capability names.
            path_key: Key in each dict that contains the file path.

        Returns:
            The same list, enriched with ownership data.
        """
        for cand in candidates:
            fp = cand.get(path_key, "")
            if not fp:
                cand["ownership_verdict"] = OwnershipVerdict.UNKNOWN.value
                cand["modification_policy"] = ModificationPolicy.REQUIRE_VERIFICATION.value
                cand["ownership_evidence"] = ["No path available"]
                cand["ownership_is_hard"] = False
                continue

            ownership = self.classify(fp, ticket_domains)
            cand["ownership_verdict"] = ownership.verdict.value
            cand["modification_policy"] = ownership.policy.value
            cand["ownership_evidence"] = ownership.evidence
            cand["ownership_is_hard"] = ownership.is_hard

        return candidates

    def split_candidates(
        self,
        candidates: List[Dict],
        ticket_domains: List[str],
        path_key: str = "path",
    ) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Split candidates into implementation, reference, and blocked pools.

        Args:
            candidates: List of candidate dicts (already classified or not).
            ticket_domains: Primary domain capability names.
            path_key: Key in each dict that contains the file path.

        Returns:
            Tuple of (implementation_candidates, reference_candidates, blocked_candidates)
            - implementation: ALLOW or ALLOW_WITH_EVIDENCE → planner can create tasks
            - reference: REFERENCE_ONLY or REQUIRE_VERIFICATION → planner can read
            - blocked: DO_NOT_MODIFY → completely removed from planner context
        """
        # Ensure all candidates are classified
        self.classify_batch(candidates, ticket_domains, path_key)

        implementation: List[Dict] = []
        reference: List[Dict] = []
        blocked: List[Dict] = []

        for cand in candidates:
            policy = cand.get("modification_policy", ModificationPolicy.REQUIRE_VERIFICATION.value)

            if policy in (ModificationPolicy.ALLOW.value, ModificationPolicy.ALLOW_WITH_EVIDENCE.value):
                implementation.append(cand)
            elif policy == ModificationPolicy.DO_NOT_MODIFY.value:
                blocked.append(cand)
            else:
                # REFERENCE_ONLY and REQUIRE_VERIFICATION → reference pool
                reference.append(cand)

        logger.info(
            f"[OwnershipResolver] Split {len(candidates)} candidates → "
            f"{len(implementation)} implementation, {len(reference)} reference, "
            f"{len(blocked)} blocked"
        )

        return implementation, reference, blocked
