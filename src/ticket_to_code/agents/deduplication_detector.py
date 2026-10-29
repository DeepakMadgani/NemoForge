"""
Duplication Detector — Prevent HTML/TS Property Conflicts

Analyzes planned changes and detects when properties would be duplicated across
files (e.g., HTML wants to add `isUserProjectMember` but TS already has it, or vice versa).

Key insight: Instead of naively generating properties in both files, detect the conflict
BEFORE generation and resolve it intelligently:
  - If TS has property, don't duplicate in HTML
  - If HTML references property, ensure TS declares it
  - If both need property, declare once in TS only

Author: Deepak Madgani
Date: August 2026
"""

import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictType(str, Enum):
    """Types of property/method conflicts."""
    DUPLICATE_IN_BOTH = "duplicate_in_both"  # Already exists in both files
    DUPLICATE_IN_TS = "duplicate_in_ts"  # TS has it, HTML wants to add
    DUPLICATE_IN_HTML = "duplicate_in_html"  # HTML has it, TS wants to add
    MISSING_IN_TS = "missing_in_ts"  # HTML uses it, but TS doesn't have it
    MISSING_IN_HTML = "missing_in_html"  # TS creates it, but HTML needs it
    NAME_MISMATCH = "name_mismatch"  # HTML uses `isUserProjectMember`, TS uses `isProjectMember`
    TYPE_MISMATCH = "type_mismatch"  # TS: boolean, HTML expects string


class ConflictResolution(str, Enum):
    """How to resolve a conflict."""
    SKIP_DUPLICATE = "skip_duplicate"  # Don't generate (already exists)
    ADD_TO_TS_ONLY = "add_to_ts_only"  # Add property to TS, reference in HTML
    ADD_TO_HTML_ONLY = "add_to_html_only"  # Only update HTML template
    RENAME = "rename"  # Rename the conflicting property
    MERGE = "merge"  # Combine both definitions


@dataclass
class PropertyConflict:
    """A single property conflict."""
    property_name: str
    ts_file: str
    html_file: str
    conflict_type: ConflictType
    resolution: ConflictResolution
    ts_has_property: bool = False
    html_references_property: bool = False
    ts_type_hint: Optional[str] = None
    html_type_hint: Optional[str] = None
    reason: str = ""

    def to_dict(self):
        return {
            "property_name": self.property_name,
            "ts_file": self.ts_file,
            "html_file": self.html_file,
            "conflict_type": self.conflict_type.value,
            "resolution": self.resolution.value,
            "ts_has_property": self.ts_has_property,
            "html_references_property": self.html_references_property,
            "ts_type_hint": self.ts_type_hint,
            "html_type_hint": self.html_type_hint,
            "reason": self.reason,
        }


@dataclass
class MethodConflict:
    """A single method conflict."""
    method_name: str
    ts_file: str
    html_file: str
    conflict_type: ConflictType
    resolution: ConflictResolution
    ts_has_method: bool = False
    html_calls_method: bool = False
    reason: str = ""

    def to_dict(self):
        return {
            "method_name": self.method_name,
            "ts_file": self.ts_file,
            "html_file": self.html_file,
            "conflict_type": self.conflict_type.value,
            "resolution": self.resolution.value,
            "ts_has_method": self.ts_has_method,
            "html_calls_method": self.html_calls_method,
            "reason": self.reason,
        }


class DeduplicationDetector:
    """
    Analyzes planned generation tasks and symbol index to detect conflicts.

    Usage:
        detector = DeduplicationDetector(symbol_index)
        
        # Detect conflicts in planned tasks
        conflicts = detector.detect_conflicts(
            ts_file="add-members.component.ts",
            html_file="add-members.component.html",
            planned_ts_properties=["isUserProjectMember", "selectedOrganizationName"],
            planned_html_properties=["isUserProjectMember"],
        )
        
        # Resolve conflicts
        for conflict in conflicts:
            print(f"⚠️  {conflict.property_name}: {conflict.conflict_type}")
            print(f"   Resolution: {conflict.resolution}")
    """

    def __init__(self, symbol_index):
        self.symbol_index = symbol_index

    def detect_conflicts(
        self,
        ts_file: str,
        html_file: str,
        planned_ts_properties: Optional[List[str]] = None,
        planned_html_properties: Optional[List[str]] = None,
        planned_ts_methods: Optional[List[str]] = None,
        planned_html_methods: Optional[List[str]] = None,
    ) -> Tuple[List[PropertyConflict], List[MethodConflict]]:
        """
        Detect conflicts between planned TS and HTML changes.

        Returns: (property_conflicts, method_conflicts)
        """
        planned_ts_properties = planned_ts_properties or []
        planned_html_properties = planned_html_properties or []
        planned_ts_methods = planned_ts_methods or []
        planned_html_methods = planned_html_methods or []

        property_conflicts = []
        method_conflicts = []

        # Get current state from symbol index
        ts_properties = {
            p.name for p in self.symbol_index.get_properties(ts_file)
        }
        ts_methods = {m.name for m in self.symbol_index.get_methods(ts_file)}
        html_properties = set(self.symbol_index.get_template_properties(html_file))
        html_methods = set()  # HTML doesn't have "methods" in the same sense

        # Detect property conflicts
        for prop_name in planned_ts_properties:
            # Case 1: Property already exists in TS
            if prop_name in ts_properties:
                property_conflicts.append(
                    PropertyConflict(
                        property_name=prop_name,
                        ts_file=ts_file,
                        html_file=html_file,
                        conflict_type=ConflictType.DUPLICATE_IN_TS,
                        resolution=ConflictResolution.SKIP_DUPLICATE,
                        ts_has_property=True,
                        html_references_property=prop_name in html_properties,
                        reason=f"Property '{prop_name}' already exists in TS, skipping",
                    )
                )

        for prop_name in planned_html_properties:
            # Case 2: Property already exists in HTML template
            if prop_name in html_properties:
                property_conflicts.append(
                    PropertyConflict(
                        property_name=prop_name,
                        ts_file=ts_file,
                        html_file=html_file,
                        conflict_type=ConflictType.DUPLICATE_IN_HTML,
                        resolution=ConflictResolution.SKIP_DUPLICATE,
                        html_references_property=True,
                        ts_has_property=prop_name in ts_properties,
                        reason=f"Property '{prop_name}' already used in HTML, skipping",
                    )
                )

            # Case 3: HTML wants property that doesn't exist in TS
            elif prop_name not in ts_properties and prop_name not in planned_ts_properties:
                property_conflicts.append(
                    PropertyConflict(
                        property_name=prop_name,
                        ts_file=ts_file,
                        html_file=html_file,
                        conflict_type=ConflictType.MISSING_IN_TS,
                        resolution=ConflictResolution.ADD_TO_TS_ONLY,
                        html_references_property=True,
                        ts_has_property=False,
                        reason=f"HTML references '{prop_name}' but TS doesn't have it — adding to TS",
                    )
                )

        # Detect method conflicts
        for method_name in planned_ts_methods:
            if method_name in ts_methods:
                method_conflicts.append(
                    MethodConflict(
                        method_name=method_name,
                        ts_file=ts_file,
                        html_file=html_file,
                        conflict_type=ConflictType.DUPLICATE_IN_TS,
                        resolution=ConflictResolution.SKIP_DUPLICATE,
                        ts_has_method=True,
                        html_calls_method=method_name in html_methods,
                        reason=f"Method '{method_name}' already exists in TS, skipping",
                    )
                )

        logger.info(
            f"Conflict detection: {len(property_conflicts)} property conflicts, "
            f"{len(method_conflicts)} method conflicts for {ts_file} ↔ {html_file}"
        )

        return property_conflicts, method_conflicts

    def resolve_conflicts(
        self,
        property_conflicts: List[PropertyConflict],
        method_conflicts: List[MethodConflict],
    ) -> Dict[str, object]:
        """
        Generate a resolution plan for all conflicts.

        Returns:
            {
                "skip_properties": ["isUserProjectMember"],
                "add_to_ts_only": ["selectedOrganizationName"],
                "add_to_html_only": [],
                "rename_required": [],
            }
        """
        resolution_plan = {
            "skip_properties": [],
            "add_to_ts_only": [],
            "add_to_html_only": [],
            "rename_required": [],
            "skip_methods": [],
        }

        for conflict in property_conflicts:
            if conflict.resolution == ConflictResolution.SKIP_DUPLICATE:
                resolution_plan["skip_properties"].append(conflict.property_name)
            elif conflict.resolution == ConflictResolution.ADD_TO_TS_ONLY:
                resolution_plan["add_to_ts_only"].append(conflict.property_name)
            elif conflict.resolution == ConflictResolution.ADD_TO_HTML_ONLY:
                resolution_plan["add_to_html_only"].append(conflict.property_name)
            elif conflict.resolution == ConflictResolution.RENAME:
                resolution_plan["rename_required"].append(
                    {
                        "old_name": conflict.property_name,
                        "new_name": f"{conflict.property_name}_generated",
                    }
                )

        for conflict in method_conflicts:
            if conflict.resolution == ConflictResolution.SKIP_DUPLICATE:
                resolution_plan["skip_methods"].append(conflict.method_name)

        return resolution_plan

    def get_safe_property_list(
        self,
        ts_file: str,
        planned_properties: List[str],
    ) -> List[str]:
        """
        Filter out properties that already exist in TS.
        This prevents duplicate declarations.

        Returns: [property_name, ...] (properties safe to generate)
        """
        existing_properties = {
            p.name for p in self.symbol_index.get_properties(ts_file)
        }

        safe = []
        duplicates = []
        for prop in planned_properties:
            if prop in existing_properties:
                duplicates.append(prop)
            else:
                safe.append(prop)

        if duplicates:
            logger.warning(
                f"Duplicate detection: skipping {len(duplicates)} properties "
                f"already in {ts_file}: {duplicates}"
            )

        return safe

    def get_required_property_list(
        self,
        html_file: str,
        ts_file: str,
    ) -> List[str]:
        """
        Get all properties referenced in HTML that MUST be in TS.

        This ensures HTML can always resolve its property bindings.

        Returns: [property_name, ...] (properties that must exist in TS)
        """
        html_properties = set(self.symbol_index.get_template_properties(html_file))
        ts_properties = {
            p.name for p in self.symbol_index.get_properties(ts_file)
        }

        missing_in_ts = html_properties - ts_properties

        if missing_in_ts:
            logger.warning(
                f"Missing properties: HTML {html_file} references {len(missing_in_ts)} "
                f"properties not in TS {ts_file}: {missing_in_ts}"
            )

        return sorted(list(missing_in_ts))
