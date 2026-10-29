"""
Integration Guide: Using Symbol Scanner, Deduplication, Refactoring, and UI

This module shows how to wire together all the new components:
  1. WorkspaceSymbolScanner - inspects existing code
  2. DeduplicationDetector - finds conflicts
  3. CrossFileRefactorer - coordinates changes across files
  4. AnalysisVisualization UI - displays everything

Author: Deepak Madgani
Date: August 2026
"""

import logging
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class TicketToCodeAnalysisPipeline:
    """
    Complete pipeline: Analyze workspace → Detect conflicts → Plan → Generate → Refactor

    Usage:
        pipeline = TicketToCodeAnalysisPipeline(workspace_path, lsp_client, symbol_index)
        
        # Phase 1: Analyze workspace
        analysis = pipeline.analyze_workspace()
        
        # Phase 2: Detect potential conflicts
        conflicts = pipeline.detect_planning_conflicts(
            ts_files=["add-members.component.ts"],
            html_files=["add-members.component.html"],
            planned_properties=["isUserProjectMember", "selectedOrganizationName"],
        )
        
        # Phase 3: Generate safely
        generation_plan = pipeline.create_safe_generation_plan(conflicts)
        
        # Phase 4: Track progress
        pipeline.update_progress(file_path, generated_properties, errors)
    """

    def __init__(
        self,
        workspace_path: str,
        lsp_client=None,
        symbol_index=None,
    ):
        self.workspace_path = Path(workspace_path)
        self.lsp_client = lsp_client
        self.symbol_index = symbol_index

        # Import new components
        from ticket_to_code.agents.workspace_symbol_scanner import (
            WorkspaceSymbolScanner,
        )
        from ticket_to_code.agents.deduplication_detector import (
            DeduplicationDetector,
        )
        from ticket_to_code.agents.cross_file_refactorer import (
            CrossFileRefactorer,
        )

        self.scanner = None
        self.deduplicator = None
        self.refactorer = None

        # Will be initialized on-demand
        self._analysis_cache: Optional[Dict] = None
        self._progress_log: List[Dict] = []

    def analyze_workspace(self, force_rescan: bool = False) -> Dict:
        """
        Phase 1: Scan entire workspace and extract all symbols.

        Returns:
            {
                "summary": { "files": 42, "symbols": 1234, "properties": 567, "methods": 667 },
                "by_file": { "file.ts": FileSymbols, ... },
                "index": WorkspaceSymbolIndex,
            }
        """
        if self._analysis_cache and not force_rescan:
            logger.info("  Using cached workspace analysis")
            return self._analysis_cache

        logger.info(f"Analyzing workspace: {self.workspace_path}")

        from ticket_to_code.agents.workspace_symbol_scanner import (
            WorkspaceSymbolScanner,
        )

        scanner = WorkspaceSymbolScanner(
            workspace_path=str(self.workspace_path),
            lsp_client=self.lsp_client,
        )
        self.scanner = scanner

        # Run scan
        index = scanner.scan_workspace()

        # Get summary
        summary = index.get_summary()

        logger.info(f"  ✅ Analysis complete: {summary}")

        self._analysis_cache = {
            "summary": summary,
            "by_file": scanner.file_symbols,
            "index": index,
        }

        return self._analysis_cache

    def detect_planning_conflicts(
        self,
        ts_file: str,
        html_file: str,
        planned_ts_properties: Optional[List[str]] = None,
        planned_html_properties: Optional[List[str]] = None,
    ) -> Dict:
        """
        Phase 2: Detect potential conflicts before generation.

        Returns:
            {
                "property_conflicts": [PropertyConflict, ...],
                "method_conflicts": [MethodConflict, ...],
                "resolution_plan": { "skip_properties": [...], "add_to_ts_only": [...], ... },
            }
        """
        if not self.symbol_index:
            raise ValueError("symbol_index required for conflict detection")

        logger.info(
            f"Detecting conflicts: {ts_file} ↔ {html_file}"
        )

        from ticket_to_code.agents.deduplication_detector import (
            DeduplicationDetector,
        )

        detector = DeduplicationDetector(self.symbol_index)
        self.deduplicator = detector

        # Detect conflicts
        prop_conflicts, method_conflicts = detector.detect_conflicts(
            ts_file=ts_file,
            html_file=html_file,
            planned_ts_properties=planned_ts_properties or [],
            planned_html_properties=planned_html_properties or [],
        )

        # Get resolution plan
        resolution_plan = detector.resolve_conflicts(prop_conflicts, method_conflicts)

        logger.info(
            f"  Found {len(prop_conflicts)} property conflicts, "
            f"{len(method_conflicts)} method conflicts"
        )

        return {
            "property_conflicts": prop_conflicts,
            "method_conflicts": method_conflicts,
            "resolution_plan": resolution_plan,
        }

    def create_safe_generation_plan(
        self,
        conflicts_analysis: Dict,
        planned_properties: List[str],
        ts_file: str,
    ) -> List[str]:
        """
        Phase 3: Filter out duplicates and create safe property list.

        Takes the planned properties and removes ones that would be duplicated.

        Returns:
            safe_properties: List of properties that can be generated without conflict
        """
        if not self.symbol_index or not self.deduplicator:
            raise ValueError("symbol_index and deduplicator required")

        logger.info(f"Creating safe generation plan for {ts_file}")

        # Get safe property list (removes duplicates)
        safe_properties = self.deduplicator.get_safe_property_list(
            ts_file=ts_file,
            planned_properties=planned_properties,
        )

        logger.info(
            f"  Safe properties: {len(safe_properties)}/{len(planned_properties)} "
            f"(skipped {len(planned_properties) - len(safe_properties)} duplicates)"
        )

        return safe_properties

    def plan_refactoring(
        self,
        refactoring_type: str,
        old_name: str,
        new_name: str,
        ts_file: str,
        html_file: Optional[str] = None,
    ) -> List:
        """
        Phase 4a: Plan a cross-file refactoring operation.

        Supports:
          - rename_property: isUserProjectMember → isProjectMember
          - rename_method: getAllUsers() → fetchAllUsers()
          - rename_class: AddMembersComponent → ManageMembersComponent

        Returns:
            refactoring_changes: List of RefactoringChange objects
        """
        if not self.symbol_index:
            raise ValueError("symbol_index required for refactoring")

        logger.info(
            f"Planning refactoring: {refactoring_type} {old_name} → {new_name}"
        )

        from ticket_to_code.agents.cross_file_refactorer import (
            CrossFileRefactorer,
        )

        refactorer = CrossFileRefactorer(
            workspace_path=str(self.workspace_path),
            symbol_index=self.symbol_index,
            lsp_client=self.lsp_client,
        )
        self.refactorer = refactorer

        # Plan the refactoring
        if refactoring_type == "rename_property":
            changes = refactorer.rename_property(
                old_name=old_name,
                new_name=new_name,
                ts_file=ts_file,
                html_file=html_file or "",
            )
        elif refactoring_type == "rename_method":
            changes = refactorer.rename_method(
                old_name=old_name,
                new_name=new_name,
                ts_file=ts_file,
            )
        elif refactoring_type == "rename_class":
            changes = refactorer.rename_class(
                old_name=old_name,
                new_name=new_name,
                ts_file=ts_file,
            )
        else:
            raise ValueError(f"Unknown refactoring type: {refactoring_type}")

        logger.info(f"  Planned {len(changes)} changes")

        return changes

    def update_progress(
        self,
        iteration: int,
        current_file: str,
        status: str,
        generated_properties: Optional[List[str]] = None,
        verified_errors: Optional[List[str]] = None,
    ) -> Dict:
        """
        Phase 5: Track edit loop progress for UI display.

        Called during code generation to record what's happening.

        Returns:
            progress_entry: Dict entry for UI display
        """
        from datetime import datetime

        progress_entry = {
            "iteration": iteration,
            "current_file": current_file,
            "status": status,
            "generated_properties": generated_properties or [],
            "verified_errors": verified_errors or [],
            "timestamp": datetime.now().isoformat(),
        }

        self._progress_log.append(progress_entry)

        logger.info(
            f"  Edit loop #{iteration}: {current_file} → {status} "
            f"({len(verified_errors or [])} errors)"
        )

        return progress_entry

    def get_ui_data(self) -> Dict:
        """
        Collect all data needed for AnalysisVisualization UI component.

        Returns:
            {
                "workspaceSymbols": { file_path: FileSymbols.to_dict(), ... },
                "propertyConflicts": [conflict.to_dict(), ...],
                "plannerDecisions": { file_path: decision, ... },
                "editLoopProgress": [progress, ...],
            }
        """
        ui_data = {
            "workspaceSymbols": {},
            "propertyConflicts": [],
            "plannerDecisions": {},
            "editLoopProgress": self._progress_log,
        }

        # Add workspace symbols if available
        if self._analysis_cache and self._analysis_cache.get("by_file"):
            ui_data["workspaceSymbols"] = {
                fpath: fsyms.to_dict()
                for fpath, fsyms in self._analysis_cache["by_file"].items()
            }

        # Add conflicts if available
        if hasattr(self, "_last_conflicts"):
            ui_data["propertyConflicts"] = [
                c.to_dict() for c in self._last_conflicts
            ]

        return ui_data

    def export_analysis_report(self, output_path: str) -> None:
        """Export complete analysis to JSON for archival/debugging."""
        import json
        from datetime import datetime

        report = {
            "timestamp": datetime.now().isoformat(),
            "workspace_path": str(self.workspace_path),
            "analysis": self.get_ui_data(),
            "refactoring_summary": (
                self.refactorer.get_refactoring_summary()
                if self.refactorer
                else None
            ),
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        logger.info(f"Analysis report exported: {output_file}")


# ============================================================================
# EXAMPLE WORKFLOW: How to use in plan_node or generate_code_node
# ============================================================================

def example_workflow_usage():
    """
    Example of how to use the TicketToCodeAnalysisPipeline in workflow.py

    This would go in plan_node or generate_code_node:
    """

    # Initialize pipeline
    pipeline = TicketToCodeAnalysisPipeline(
        workspace_path="/path/to/workspace",
        lsp_client=agents.lsp_client,  # From workflow agents
        symbol_index=agents.symbol_index,  # From workflow agents
    )

    # Phase 1: Analyze workspace
    analysis = pipeline.analyze_workspace()
    logger.info(f"Workspace analysis: {analysis['summary']}")

    # Phase 2: Detect conflicts before planning
    conflicts = pipeline.detect_planning_conflicts(
        ts_file="add-members.component.ts",
        html_file="add-members.component.html",
        planned_ts_properties=["isUserProjectMember", "selectedOrganizationName"],
        planned_html_properties=["isUserProjectMember"],  # This will conflict!
    )

    # Phase 3: Filter to safe list
    safe_properties = pipeline.create_safe_generation_plan(
        conflicts_analysis=conflicts,
        planned_properties=["isUserProjectMember", "selectedOrganizationName"],
        ts_file="add-members.component.ts",
    )
    logger.info(f"Safe to generate: {safe_properties}")

    # Phase 4: (Optional) Plan refactoring if needed
    if conflicts["resolution_plan"]["rename_required"]:
        refactoring_changes = pipeline.plan_refactoring(
            refactoring_type="rename_property",
            old_name="isUserProjectMember",
            new_name="isProjectMember",
            ts_file="add-members.component.ts",
            html_file="add-members.component.html",
        )
        logger.info(f"Planned {len(refactoring_changes)} refactoring changes")

    # Phase 5: During code generation, track progress
    for iteration in range(1, 26):  # Edit loop MAX_ITER = 25
        current_file = "add-members.component.ts"
        generated_props = ["isUserProjectMember", "selectedOrganizationName"]
        errors = []  # Would be filled if compilation failed

        progress = pipeline.update_progress(
            iteration=iteration,
            current_file=current_file,
            status="completed" if iteration < 5 else "in-progress",
            generated_properties=generated_props,
            verified_errors=errors,
        )

        # Pass to UI
        # agents.websocket.send({"type": "edit_loop_progress", "data": progress})

    # At end: Export everything for debugging
    pipeline.export_analysis_report("/path/to/report.json")

    # Return UI data
    ui_data = pipeline.get_ui_data()
    return {
        "status": "planning_complete",
        "analysis": ui_data,
        "safe_properties": safe_properties,
        "conflicts": conflicts,
    }
