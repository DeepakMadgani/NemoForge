"""Cross-File Dataflow Contract & Structural Invariants Layer.

Enforces deterministic end-to-end data propagation:
  Source (API / Service Query)
     ↓
  Producer (Controller State / Method)
     ↓
  ViewModel / Staged Item (e.g. DisplayedMember.organization)
     ↓
  Sink (Template / Consumer Binding, e.g. dmember.organization)

Prevents stranded state variables (e.g., user.existingOrganizationName stored
on a temporary object but never assigned to DisplayedMember.organization before push).
Provides structural defect fingerprinting for causal repair loops and stagnation guards.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CausalDefectFingerprint:
    """Semantic root-cause fingerprint for defect tracking and stagnation detection."""
    failure_type: str        # e.g., "DATAFLOW_MISMATCH", "UNPOPULATED_SINK", "FORBIDDEN_MUTATION"
    requirement_id: str      # e.g., "REQ-1"
    source_symbol: str       # e.g., "existingMember.company.name"
    producer_symbol: str     # e.g., "DisplayedMember.organization"
    sink_symbol: str         # e.g., "dmember.organization"
    missing_link: str        # e.g., "displayedMember.organization = existingMember.company?.name"

    def to_key(self) -> str:
        """Deterministic fingerprint key ignoring superficial whitespace or prompt differences."""
        raw = f"{self.failure_type}|{self.requirement_id}|{self.source_symbol.strip()}|{self.producer_symbol.strip()}|{self.sink_symbol.strip()}|{self.missing_link.strip()}".lower()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class NegativeRule:
    """Explicit MUST / MUST_NOT behavioral constraints."""
    target_state: str         # e.g. "EXISTING_PROJECT_MEMBER" or "NEW_PROJECT_MEMBER"
    must_do: List[str]        # e.g. ["display static text showing existing organization name"]
    must_not_do: List[str]    # e.g. ["display editable dropdown", "allow editing organization"]


@dataclass
class DataFlowLink:
    """A single end-to-end dataflow contract requirement."""
    source: str               # Origin API / Service query path
    producer: str             # Controller assignment target
    sink: str                 # Template / Consumer binding path
    collection_item: str      # ViewModel item name in collection (e.g. "DisplayedMember")
    required: bool = True
    forbidden_sources: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class DataFlowContract:
    """Component-level DataFlow and Invariant Contract."""
    ticket_id: str = ""
    links: List[DataFlowLink] = field(default_factory=list)
    negative_rules: List[NegativeRule] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def render_pre_generation_prompt_block(self) -> str:
        """Render deterministic contract constraints for pre-generation prompt injection."""
        lines = [
            "════════════════════════════════════════════════════════════════",
            "CROSS-FILE DATAFLOW & STRUCTURAL CONTRACT (PRE-GENERATION MANDATE)",
            "════════════════════════════════════════════════════════════════",
            "You MUST satisfy this deterministic dataflow contract across controller and template:",
        ]

        if self.links:
            lines.append("\nREQUIRED DATAFLOW LINKS:")
            for idx, link in enumerate(self.links, 1):
                lines.append(f"  {idx}. SINK: `{link.sink}`")
                lines.append(f"     ← MUST BE SOURCED FROM: `{link.producer}` (on `{link.collection_item}`)")
                lines.append(f"     ← ORIGIN DATA: `{link.source}`")
                lines.append(f"     ← CRITICAL RULE: When adding to collection (e.g. `displayedMembers.push`),")
                lines.append(f"       you MUST explicitly assign `{link.producer}`.")
                if link.forbidden_sources:
                    lines.append(f"     ⛔ FORBIDDEN PATTERNS: {', '.join(link.forbidden_sources)}")

        if self.negative_rules:
            lines.append("\nBEHAVIORAL INVARIANTS (MUST vs MUST NOT):")
            for rule in self.negative_rules:
                lines.append(f"  [{rule.target_state}]:")
                for m in rule.must_do:
                    lines.append(f"    ✓ MUST: {m}")
                for mn in rule.must_not_do:
                    lines.append(f"    ✗ MUST NOT: {mn}")

        lines.append("════════════════════════════════════════════════════════════════\n")
        return "\n".join(lines)

    @classmethod
    def extract_from_requirements(cls, requirements: Any, ticket_description: str = "") -> "DataFlowContract":
        """Extract dataflow contract based on ticket description and structured requirements."""
        contract = cls()
        desc_lower = (ticket_description or "").lower()

        # Check for organization member pattern
        if "organization" in desc_lower and ("member" in desc_lower or "contract" in desc_lower):
            contract.links.append(DataFlowLink(
                source="existingMember.company?.name || existingMember.companyName",
                producer="displayedMember.organization",
                sink="dmember.organization",
                collection_item="DisplayedMember",
                required=True,
                forbidden_sources=[
                    "Leaving existingOrganizationName stranded on temporary user without assigning to displayedMember.organization",
                    "Binding template to undefined existingOrganizationName",
                ],
                description="Existing project member organization must propagate directly into displayedMember.organization before push"
            ))

            contract.negative_rules.append(NegativeRule(
                target_state="EXISTING PROJECT MEMBER",
                must_do=[
                    "display existing organization as read-only static text",
                    "persist existing organization on save",
                ],
                must_not_do=[
                    "render editable ot-dropdown or ot-item-select",
                    "allow changing existing organization",
                    "leave dmember.organization empty or undefined",
                ]
            ))

            contract.negative_rules.append(NegativeRule(
                target_state="NEW PROJECT MEMBER",
                must_do=[
                    "display editable organization dropdown",
                    "allow selecting organization",
                ],
                must_not_do=[
                    "display read-only static text",
                    "strip organization dropdown options",
                ]
            ))

        return contract

    def validate_code_artifacts(
        self,
        controller_content: str,
        template_content: str = "",
    ) -> Tuple[bool, List[str], List[CausalDefectFingerprint]]:
        """
        Deterministically validate that the generated code fulfills the dataflow contract.
        Returns: (passed, error_messages, defect_fingerprints)
        """
        errors = []
        fingerprints = []

        if not controller_content:
            return True, [], []

        for link in self.links:
            # Check if template references the sink (e.g. dmember.organization)
            sink_prop = link.sink.split(".")[-1]  # e.g. "organization"
            sink_referenced = False
            if template_content:
                # Check for interpolation or attribute binding like dmember.organization
                pattern = re.compile(r"""\b[A-Za-z0-9_$]+\.""" + re.escape(sink_prop) + r"""\b""")
                if pattern.search(template_content):
                    sink_referenced = True

            # If template binds to sink or link is explicitly required
            if link.required or sink_referenced:
                # Check if controller assigns the producer property in collection addition method
                # e.g. displayedMember.organization = ... or organization: ... inside object literal
                assign_pattern = re.compile(
                    r"""(?:(?:displayedMember|newMember|member|dmember|item)\.""" + re.escape(sink_prop) + r"""\s*=|""" +
                    r"""\b""" + re.escape(sink_prop) + r"""\s*:\s*[^,}\n]+)"""
                )
                has_assignment = bool(assign_pattern.search(controller_content))

                # Check for anti-pattern: saving to isolated variable (e.g. user.existingOrganizationName)
                # while failing to assign to displayedMember.organization
                isolated_pattern = re.compile(r"""\b(?:user|searchData|this)\.existing(?:OrganizationName|Org)\b""")
                has_isolated = bool(isolated_pattern.search(controller_content))

                if not has_assignment and has_isolated:
                    err = (
                        f"Broken Dataflow: Staged item '{link.collection_item}' is missing assignment to "
                        f"'{sink_prop}'. Found isolated state '{link.forbidden_sources[0]}' "
                        f"which is never assigned to the item before collection push."
                    )
                    errors.append(err)
                    fingerprints.append(CausalDefectFingerprint(
                        failure_type="DATAFLOW_MISMATCH",
                        requirement_id="REQ-DATAFLOW-1",
                        source_symbol=link.source,
                        producer_symbol=link.producer,
                        sink_symbol=link.sink,
                        missing_link=f"displayedMember.{sink_prop} = existingMember.company?.name || user.existingOrganizationName",
                    ))
                elif not has_assignment and sink_referenced:
                    err = (
                        f"Unpopulated Sink: Template binds to '{link.sink}', but controller does not assign "
                        f"'{sink_prop}' onto collection item '{link.collection_item}'."
                    )
                    errors.append(err)
                    fingerprints.append(CausalDefectFingerprint(
                        failure_type="UNPOPULATED_SINK",
                        requirement_id="REQ-DATAFLOW-2",
                        source_symbol=link.source,
                        producer_symbol=link.producer,
                        sink_symbol=link.sink,
                        missing_link=f"displayedMember.{sink_prop} = ...",
                    ))

        # Check negative contract violations in template
        if template_content:
            # If template has *ngIf checking isExistingMember, verify it does NOT render editable dropdown
            # inside the existing member branch
            dropdown_in_existing = re.search(
                r"""\*(?:ngIf|if)=["'][^"']*\bisExisting(?:Member|ProjectMember)\b[^"']*["'][^>]*>[\s\S]*?<(?:ot-dropdown|ot-item-select|select)\b""",
                template_content,
                re.IGNORECASE
            )
            if dropdown_in_existing:
                err = "Negative Contract Violation: Rendered editable dropdown inside existing member block (*ngIf='...isExistingMember...')."
                errors.append(err)
                fingerprints.append(CausalDefectFingerprint(
                    failure_type="FORBIDDEN_MUTATION",
                    requirement_id="REQ-NEGATIVE-1",
                    source_symbol="isExistingMember",
                    producer_symbol="readonly_static_text",
                    sink_symbol="ot-dropdown",
                    missing_link="Existing members must show static text, not an editable dropdown",
                ))

        passed = (len(errors) == 0)
        return passed, errors, fingerprints

    @staticmethod
    def build_causal_repair_prompt(fingerprints: List[CausalDefectFingerprint]) -> str:
        """Construct structured root-cause repair prompt with causal locks."""
        if not fingerprints:
            return ""

        sections = [
            "════════════════════════════════════════════════════════════════",
            "ROOT CAUSE DEFECT ANALYSIS (CAUSAL REPAIR MANDATE)",
            "════════════════════════════════════════════════════════════════",
            "The semantic verification detected that code compiled, but runtime dataflow is broken.",
            "You MUST fix the root cause producer without altering unrelated templates or introducing aliases.",
        ]

        for idx, fp in enumerate(fingerprints, 1):
            sections.append(f"\nDEFECT #{idx}: [{fp.failure_type}] (Requirement: {fp.requirement_id})")
            sections.append(f"  • Template Sink:   `{fp.sink_symbol}`")
            sections.append(f"  • Expected Target: `{fp.producer_symbol}`")
            sections.append(f"  • Data Source:     `{fp.source_symbol}`")
            sections.append(f"  • Broken Link:     Value was captured in a temporary variable but not assigned to the ViewModel item.")
            sections.append(f"  • REQUIRED ACTION: In controller, explicitly add:")
            sections.append(f"      `{fp.missing_link}`")
            sections.append(f"  • FORBIDDEN STRATEGIES (SCOPE LOCK):")
            sections.append(f"      - Do NOT rename template bindings to match the wrong variable.")
            sections.append(f"      - Do NOT introduce another alias property.")
            sections.append(f"      - Do NOT modify backend services.")

        sections.append("════════════════════════════════════════════════════════════════\n")
        return "\n".join(sections)
