"""
Cross-File Refactoring Engine — Coordinate Changes Across Files

Handles complex refactoring operations that span multiple files:
  - Rename property across TypeScript + HTML
  - Move method from one class to another (update imports + calls)
  - Update property types consistently
  - Rename class + update all references

Key insight: Changes must be coordinated so all files stay in sync.
A rename in TS must update all HTML templates that reference it.

Author: Deepak Madgani
Date: August 2026
"""

import logging
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class RefactoringType(str, Enum):
    """Types of refactoring operations."""
    RENAME_PROPERTY = "rename_property"  # Rename a property (TS + HTML)
    RENAME_METHOD = "rename_method"  # Rename a method (TS + all callers)
    RENAME_CLASS = "rename_class"  # Rename a class (TS + imports + usages)
    MOVE_PROPERTY = "move_property"  # Move property to another class
    UPDATE_TYPE = "update_type"  # Change property type (TS only)
    UPDATE_IMPORTS = "update_imports"  # Fix import paths after move


@dataclass
class RefactoringOperation:
    """A single refactoring operation."""
    operation_type: RefactoringType
    old_name: str
    new_name: str
    affected_files: List[str]  # Files to be modified
    reason: str = ""
    is_breaking: bool = False  # True if renaming will require code changes


@dataclass
class RefactoringChange:
    """A change to be made in a file."""
    file_path: str
    old_text: str  # Exact text to replace
    new_text: str  # Replacement text
    line_number: Optional[int] = None
    change_type: str = "replacement"  # "replacement", "addition", "deletion"


class CrossFileRefactorer:
    """
    Coordinates refactoring operations across multiple files.

    Usage:
        refactorer = CrossFileRefactorer(workspace_path, symbol_index, lsp_client)
        
        # Rename a property across TypeScript and HTML
        changes = refactorer.rename_property(
            old_name="isUserProjectMember",
            new_name="isProjectMember",
            affected_files=[
                "add-members.component.ts",
                "add-members.component.html",
            ],
        )
        
        # Apply all changes atomically
        for change in changes:
            apply_change(change)
    """

    def __init__(
        self,
        workspace_path: str,
        symbol_index,
        lsp_client=None,
    ):
        self.workspace_path = Path(workspace_path)
        self.symbol_index = symbol_index
        self.lsp_client = lsp_client
        self.refactoring_log: List[Dict] = []

    def rename_property(
        self,
        old_name: str,
        new_name: str,
        ts_file: str,
        html_file: str,
    ) -> List[RefactoringChange]:
        """
        Rename a property in TypeScript and update all HTML references.

        Example:
            isUserProjectMember → isProjectMember
            Updates:
              - TS class property declaration
              - HTML template bindings {{ isUserProjectMember }} → {{ isProjectMember }}
              - HTML property bindings [property]="isUserProjectMember" → [property]="isProjectMember"

        Returns: List of changes to apply
        """
        changes = []

        logger.info(f"Planning property rename: {old_name} → {new_name}")

        # Change 1: Update TypeScript property declaration
        ts_path = self.workspace_path / ts_file
        if ts_path.exists():
            ts_content = ts_path.read_text(encoding="utf-8", errors="ignore")

            # Find the property declaration line
            # Pattern: propertyName: type;  or  propertyName = defaultValue;
            ts_pattern = rf"\b{re.escape(old_name)}\s*[:=]"
            if re.search(ts_pattern, ts_content):
                new_ts_content = re.sub(
                    ts_pattern,
                    f"{new_name}:",
                    ts_content,
                    count=1,  # Only replace the first occurrence (the declaration)
                )

                changes.append(
                    RefactoringChange(
                        file_path=ts_file,
                        old_text=ts_content,
                        new_text=new_ts_content,
                        change_type="replacement",
                    )
                )
                logger.info(f"  TS: Updated property declaration in {ts_file}")

        # Change 2: Update HTML template bindings
        html_path = self.workspace_path / html_file
        if html_path.exists():
            html_content = html_path.read_text(encoding="utf-8", errors="ignore")

            # Replace all occurrences of the property name in templates
            patterns = [
                (r"\{\{\s*" + re.escape(old_name) + r"\s*\}\}", "{{ " + new_name + " }}"),
                (r"\[(\w+)\]=\"" + re.escape(old_name), f'[\\1]="' + new_name),
                (
                    r"\*ng\w+\s*=\s*[\"'].*" + re.escape(old_name),
                    lambda m: m.group(0).replace(old_name, new_name),
                ),
            ]

            new_html_content = html_content
            for pattern, replacement in patterns:
                if isinstance(replacement, str):
                    new_html_content = re.sub(pattern, replacement, new_html_content)
                else:
                    new_html_content = re.sub(pattern, replacement, new_html_content)

            if new_html_content != html_content:
                changes.append(
                    RefactoringChange(
                        file_path=html_file,
                        old_text=html_content,
                        new_text=new_html_content,
                        change_type="replacement",
                    )
                )
                logger.info(f"  HTML: Updated {html_file}")

        # Log the operation
        self.refactoring_log.append(
            {
                "operation": "rename_property",
                "old_name": old_name,
                "new_name": new_name,
                "affected_files": [ts_file, html_file],
                "changes_generated": len(changes),
            }
        )

        return changes

    def rename_method(
        self,
        old_name: str,
        new_name: str,
        ts_file: str,
        calling_files: Optional[List[str]] = None,
    ) -> List[RefactoringChange]:
        """
        Rename a method in TypeScript and update all callers.

        Example:
            getAllSubscriptionUsers() → fetchSubscriptionUsers()
            Updates:
              - TS method declaration
              - All HTML event handler calls (click)="getAllSubscriptionUsers()" → (click)="fetchSubscriptionUsers()"
              - Any TS file that calls this.getAllSubscriptionUsers() → this.fetchSubscriptionUsers()

        Returns: List of changes to apply
        """
        changes = []

        logger.info(f"Planning method rename: {old_name} → {new_name}")

        # Change 1: Update TypeScript method declaration
        ts_path = self.workspace_path / ts_file
        if ts_path.exists():
            ts_content = ts_path.read_text(encoding="utf-8", errors="ignore")

            # Find method declaration: methodName(
            ts_pattern = rf"\b{re.escape(old_name)}\s*\("
            if re.search(ts_pattern, ts_content):
                new_ts_content = re.sub(
                    ts_pattern,
                    f"{new_name}(",
                    ts_content,
                    count=1,  # Only the first occurrence (the definition)
                )

                changes.append(
                    RefactoringChange(
                        file_path=ts_file,
                        old_text=ts_content,
                        new_text=new_ts_content,
                        change_type="replacement",
                    )
                )
                logger.info(f"  TS: Updated method declaration in {ts_file}")

        # Change 2: Update all callers (both TS and HTML)
        calling_files = calling_files or self._find_callers(old_name, ts_file)

        for caller_file in calling_files:
            caller_path = self.workspace_path / caller_file
            if not caller_path.exists():
                continue

            content = caller_path.read_text(encoding="utf-8", errors="ignore")

            # Replace method calls
            # Pattern: methodName( or .methodName( or this.methodName(
            call_pattern = rf"(\w*\.)?{re.escape(old_name)}\("
            if re.search(call_pattern, content):
                new_content = re.sub(
                    call_pattern,
                    rf"\g<1>{new_name}(",
                    content,
                )

                changes.append(
                    RefactoringChange(
                        file_path=caller_file,
                        old_text=content,
                        new_text=new_content,
                        change_type="replacement",
                    )
                )
                logger.info(f"  Updated method calls in {caller_file}")

        # Log the operation
        self.refactoring_log.append(
            {
                "operation": "rename_method",
                "old_name": old_name,
                "new_name": new_name,
                "affected_files": [ts_file] + calling_files,
                "changes_generated": len(changes),
            }
        )

        return changes

    def rename_class(
        self,
        old_name: str,
        new_name: str,
        ts_file: str,
    ) -> List[RefactoringChange]:
        """
        Rename a class and update all imports/references.

        Example:
            AddMembersComponent → ManageMembersComponent
            Updates:
              - TS class declaration
              - All imports: import { AddMembersComponent } from ... → import { ManageMembersComponent } from ...
              - All usages in templates and code

        Returns: List of changes to apply
        """
        changes = []

        logger.info(f"Planning class rename: {old_name} → {new_name}")

        # Change 1: Update class declaration
        ts_path = self.workspace_path / ts_file
        if ts_path.exists():
            ts_content = ts_path.read_text(encoding="utf-8", errors="ignore")

            # Find: export class ClassName
            class_pattern = rf"\bclass\s+{re.escape(old_name)}\b"
            if re.search(class_pattern, ts_content):
                new_ts_content = re.sub(
                    class_pattern,
                    f"class {new_name}",
                    ts_content,
                    count=1,
                )

                changes.append(
                    RefactoringChange(
                        file_path=ts_file,
                        old_text=ts_content,
                        new_text=new_ts_content,
                        change_type="replacement",
                    )
                )
                logger.info(f"  TS: Updated class declaration in {ts_file}")

        # Change 2: Update all imports
        importing_files = self._find_importers(old_name, ts_file)

        for importing_file in importing_files:
            importer_path = self.workspace_path / importing_file
            if not importer_path.exists():
                continue

            content = importer_path.read_text(encoding="utf-8", errors="ignore")

            # Find: import { OldName } from ... or import OldName from ...
            import_pattern = rf"import\s+(?:\{{\s*)?{re.escape(old_name)}(?:\s*\}})?(\s+from|;)"
            if re.search(import_pattern, content):
                new_content = re.sub(
                    import_pattern,
                    rf"import \1" + new_name + r"\2",
                    content,
                )

                changes.append(
                    RefactoringChange(
                        file_path=importing_file,
                        old_text=content,
                        new_text=new_content,
                        change_type="replacement",
                    )
                )
                logger.info(f"  Updated import in {importing_file}")

        # Log the operation
        self.refactoring_log.append(
            {
                "operation": "rename_class",
                "old_name": old_name,
                "new_name": new_name,
                "affected_files": [ts_file] + importing_files,
                "changes_generated": len(changes),
            }
        )

        return changes

    def update_property_type(
        self,
        property_name: str,
        old_type: str,
        new_type: str,
        ts_file: str,
    ) -> List[RefactoringChange]:
        """
        Change a property's type and update any related code.

        Example:
            isLoading: boolean → isLoading$: Observable<boolean>
            Updates type hint in property declaration

        Returns: List of changes to apply
        """
        changes = []

        logger.info(f"Planning type update: {property_name}: {old_type} → {new_type}")

        ts_path = self.workspace_path / ts_file
        if ts_path.exists():
            ts_content = ts_path.read_text(encoding="utf-8", errors="ignore")

            # Find property with old type
            # Pattern: propertyName: oldType;
            type_pattern = (
                rf"(\b{re.escape(property_name)}\s*:\s*){re.escape(old_type)}(\s*[;=])"
            )
            if re.search(type_pattern, ts_content):
                new_ts_content = re.sub(
                    type_pattern,
                    rf"\g<1>{new_type}\2",
                    ts_content,
                )

                changes.append(
                    RefactoringChange(
                        file_path=ts_file,
                        old_text=ts_content,
                        new_text=new_ts_content,
                        change_type="replacement",
                    )
                )
                logger.info(f"  TS: Updated type for {property_name} in {ts_file}")

        return changes

    def _find_callers(self, method_name: str, ts_file: str) -> List[str]:
        """Find all files that call a method."""
        callers = []

        # Use symbol index to find references (if available)
        if hasattr(self.symbol_index, "find_symbol_references"):
            refs = self.symbol_index.find_symbol_references(method_name)
            callers.extend(refs)

        # Also check for sibling HTML files (they always call TS methods)
        ts_path = Path(ts_file)
        html_sibling = ts_path.with_suffix(".html")
        if self.workspace_path / html_sibling.as_posix() in [
            self.workspace_path / f
            for f in self.symbol_index.file_symbols.keys()
        ]:
            callers.append(html_sibling.as_posix())

        return list(set(callers))

    def _find_importers(self, class_name: str, ts_file: str) -> List[str]:
        """Find all files that import a class."""
        importers = []

        # Use symbol index to find references (if available)
        if hasattr(self.symbol_index, "find_symbol_references"):
            refs = self.symbol_index.find_symbol_references(class_name)
            importers.extend(refs)

        return list(set(importers))

    def get_refactoring_summary(self) -> Dict:
        """Get summary of all refactoring operations performed."""
        return {
            "total_operations": len(self.refactoring_log),
            "operations": self.refactoring_log,
        }
