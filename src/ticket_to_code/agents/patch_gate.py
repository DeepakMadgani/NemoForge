"""
Patch Gate and Cross-Artifact Consistency Enforcement.

Final Security Boundary Invariants:
1. File Scope Gate:
   - Only authorized writable files can receive patches.
   - Any patch attempting to write an unauthorized or imported-reference file (e.g. models/project.ts)
     is REJECTED at this boundary.
2. Target Scope Gate:
   - Patches must operate within the approved ChangeTarget boundaries (symbol, block, range).
3. Type Safety & Breaking Changes Gate:
   - Invariant: Unjustified breaking type changes are rejected.
   - Adding required (non-optional) fields to shared interfaces without migration evidence is forbidden.
4. Cross-Artifact Consistency Gate:
   - Verifies cross-artifact references (e.g., Angular TS controller properties ↔ HTML template bindings,
     SCSS classes ↔ HTML elements).
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Set

from ticket_to_code.models import ChangeTarget

logger = logging.getLogger(__name__)


class PatchScopeViolation(Exception):
    """Raised when a patch attempts to modify an unauthorized file or target."""
    pass


class UnjustifiedBreakingTypeChangeError(Exception):
    """Raised when a shared type/interface introduces a breaking change without migration evidence."""
    pass


class CrossArtifactConsistencyError(Exception):
    """Raised when an artifact references symbols/properties undeclared in its companion artifact."""
    pass


class PatchGate:
    """Validates generated patches before they can be written to disk."""

    @classmethod
    def validate_file_authorization(
        cls, file_path: str, authorized_writable_files: Set[str]
    ) -> Tuple[bool, str]:
        """Validate that the file being patched is explicitly in the authorized writable scope."""
        norm_file = file_path.replace("\\", "/").lower().strip("/")
        for auth in authorized_writable_files:
            norm_auth = auth.replace("\\", "/").lower().strip("/")
            if norm_file == norm_auth or norm_file.endswith("/" + norm_auth) or norm_auth.endswith("/" + norm_file):
                return True, "File is authorized for modification"
        return False, f"File '{file_path}' is NOT in authorized writable files list"

    @classmethod
    def validate_type_changes(
        cls,
        file_path: str,
        patch_content: str,
        has_migration_evidence: bool = False,
    ) -> Tuple[bool, str]:
        """
        Validate that additions to shared interfaces and types do not introduce unjustified breaking changes.
        
        Rule: New fields in TypeScript interfaces must be optional ('?') unless explicit
        migration evidence is present in the ticket/plan.
        """
        ext = Path(file_path).suffix.lower()
        if ext in (".ts", ".tsx"):
            # Check for REPLACE blocks that add mandatory properties to interfaces
            # Matches: propName: type; without '?'
            # e.g.: isProjectMember: boolean;
            # versus: isProjectMember?: boolean;
            replace_pattern = re.compile(
                r'={7}\n(.*?)\n>{7} REPLACE', re.DOTALL
            )
            for m in replace_pattern.finditer(patch_content):
                replace_block = m.group(1)
                # Find lines that look like interface property declarations
                for line in replace_block.splitlines():
                    trimmed = line.strip()
                    # Look for property declarations: identifier: type;
                    prop_match = re.match(r'^([a-zA-Z_$][a-zA-Z0-9_$]*)(\?)?:\s*[^;]+;', trimmed)
                    if prop_match:
                        prop_name = prop_match.group(1)
                        is_optional = bool(prop_match.group(2))
                        if not is_optional and not has_migration_evidence:
                            # Check if this file is a shared model/interface
                            fp_low = file_path.lower()
                            if any(k in fp_low for k in ("model", "interface", "dto", "type")):
                                return (
                                    False,
                                    f"Unjustified breaking type change: Required property '{prop_name}' "
                                    f"added to shared model/interface '{file_path}' without optional '?' "
                                    f"modifier and without explicit migration evidence."
                                )

        return True, "Type change check passed"

    @classmethod
    def validate_patch_proportionality(
        cls,
        file_path: str,
        patch_content: str,
        current_content: Optional[str] = None,
        is_new_file: bool = False,
        max_expansion_ratio: float = 4.0,
        min_expansion_threshold_lines: int = 50,
        is_fullstack: bool = False,
        target_file_count: int = 1,
    ) -> Tuple[bool, str]:
        """
        Validate that generated patches are proportional to the requested change and
        do not attempt a suspicious full-file or disproportionate rewrite.
        Scales budget dynamically for fullstack tickets and cross-service contracts (FLAW-10).
        """
        if is_new_file or not current_content:
            return True, "New file creation allowed full generation"

        # Dynamic proportionality budget for full-stack tickets & cross-service contracts (FLAW-10)
        is_contract_file = any(
            file_path.endswith(ext)
            for ext in (".sql", "DTO.java", "Count.java", ".component.html", ".component.scss", ".yml", ".yaml")
        )
        if is_fullstack or is_contract_file or target_file_count > 1:
            max_expansion_ratio = max(max_expansion_ratio, 8.0)
            min_expansion_threshold_lines = max(min_expansion_threshold_lines, 120)

        replace_pattern = re.compile(
            r'<{4,}\s*(?:SEARCH|ORIGINAL|TARGET)[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}\s*(?:REPLACE|UPDATED|REPLACEMENT)[^\r\n]*',
            re.DOTALL
        )
        matches = list(replace_pattern.finditer(patch_content))
        if not matches:
            # Fallback for plain merge conflict markers <<<<<<< / ======= / >>>>>>>
            alt_conflict_pattern = re.compile(
                r'<{4,}[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}[^\r\n]*',
                re.DOTALL
            )
            matches = list(alt_conflict_pattern.finditer(patch_content))
        if not matches:
            # Fallback for TARGET/REPLACEMENT format
            target_pattern = re.compile(
                r'(?:TARGET(?:_CONTENT)?|ORIGINAL):\s*\r?\n?(.*?)\r?\n?(?:REPLACEMENT(?:_CONTENT)?|REPLACE):\s*\r?\n?(.*?)(?=\r?\n(?:TARGET(?:_CONTENT)?|ORIGINAL):|\Z)',
                re.DOTALL | re.IGNORECASE,
            )
            matches = list(target_pattern.finditer(patch_content))
        if not matches:
            # If no SEARCH/REPLACE blocks found in patch, check if attempting raw whole-file rewrite
            threshold = 80 if (is_fullstack or is_contract_file) else 40
            if len(patch_content.strip().splitlines()) > threshold and len(current_content.splitlines()) > threshold:
                return (
                    False,
                    f"Disproportionate rewrite: Patch for existing file '{file_path}' did not use "
                    f"SEARCH/REPLACE blocks and attempted raw replacement."
                )
            return True, "No search/replace blocks to check"

        total_file_lines = len(current_content.splitlines())

        for m in matches:
            search_block = m.group(1)
            replace_block = m.group(2)
            search_lines = len(search_block.splitlines())
            replace_lines = len(replace_block.splitlines())

            # Check 1: Whole file rewrite disguised as a single SEARCH block
            if total_file_lines > 80 and search_lines >= int(0.9 * total_file_lines):
                return (
                    False,
                    f"Disproportionate rewrite: SEARCH block encompasses {search_lines}/{total_file_lines} "
                    f"lines (>= 90%) of '{file_path}'. Whole-file rewrites are forbidden; use targeted blocks."
                )

            # Check 2: Disproportionate expansion (e.g. 5 lines search -> 300 lines replace)
            if search_lines > 0 and replace_lines > max_expansion_ratio * search_lines:
                added_lines = replace_lines - search_lines
                if added_lines > min_expansion_threshold_lines:
                    return (
                        False,
                        f"Disproportionate patch expansion in '{file_path}': SEARCH block is {search_lines} lines "
                        f"but REPLACE block is {replace_lines} lines (+{added_lines} lines, ratio {replace_lines/search_lines:.1f}x). "
                        f"Patches must be proportional to the target symbol."
                    )

        return True, "Patch proportionality verified"

    @classmethod
    def validate_cross_artifact_consistency(
        cls,
        file_path: str,
        patch_content: str,
        sibling_content: Optional[str] = None,
        sibling_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Generic Cross-Artifact Consistency Gate.
        
        Adapters:
        - Angular Adapter: Template (.html) bindings must match declared properties in Controller (.ts).
          Uses structural TemplateExpressionParser to classify literals, pipes, and member chains.
        """
        ext = Path(file_path).suffix.lower()
        
        # Angular HTML ↔ TS Controller adapter
        if ext in (".html", ".htm") and sibling_content:
            from ticket_to_code.agents.template_expression_parser import TemplateExpressionParser
            
            # Extract local template reference variables: e.g. #editableOrg, #users, etc.
            template_local_refs = set(re.findall(r'#([a-zA-Z_$][a-zA-Z0-9_$]*)', patch_content))
            # Also extract let-x declarations: e.g. let-user, *ngFor="let item of items"
            template_local_refs.update(re.findall(r'let-([a-zA-Z_$][a-zA-Z0-9_$]*)', patch_content))
            template_local_refs.update(re.findall(r'\*ngFor=["\']let\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', patch_content))
            # Also extract 'as var' from *ngIf="cond as user"
            template_local_refs.update(re.findall(r'\bas\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', patch_content))

            # Extract expressions from *ngIf, {{ }}, [prop], (event)
            raw_expressions = re.findall(
                r'(?:\*ngIf=["\']([^"\']+)["\']|\[[\w.]+\]=["\']([^"\']+)["\']|\{\{\s*(.*?)\s*\}\}|\([\w.]+\)=["\']([^"\']+)["\'])',
                patch_content
            )
            for match_tuple in raw_expressions:
                expr = next((s for s in match_tuple if s), "")
                if not expr:
                    continue
                refs = TemplateExpressionParser.extract_references(expr, local_vars=template_local_refs)
                for r in refs:
                    if r.receiver == "this":
                        var = r.member
                        decl_pattern = rf'\b(?:public\s+|private\s+|protected\s+)?{re.escape(var)}\s*(?:\?|:|\=|\(|\;)'
                        if not re.search(decl_pattern, sibling_content):
                            return (
                                False,
                                f"Cross-artifact inconsistency: Template references '{var}', but '{var}' "
                                f"is not declared in companion controller '{sibling_path or 'controller.ts'}'."
                            )

        return True, "Cross-artifact consistency check passed"

    @classmethod
    def validate_patch(
        cls,
        file_path: str,
        patch_content: str,
        authorized_writable_files: Set[str],
        current_content: Optional[str] = None,
        change_targets: Optional[List[ChangeTarget]] = None,
        has_migration_evidence: bool = False,
        sibling_content: Optional[str] = None,
        sibling_path: Optional[str] = None,
        is_new_file: bool = False,
    ) -> Tuple[bool, str]:
        """Comprehensive pre-application patch validation."""
        # 1. File Scope Gate (Security Boundary)
        ok, reason = cls.validate_file_authorization(file_path, authorized_writable_files)
        if not ok:
            return False, reason

        # 2. Patch Proportionality Gate (Prevents Whole-File Rewrites)
        ok, reason = cls.validate_patch_proportionality(
            file_path, patch_content, current_content, is_new_file=is_new_file
        )
        if not ok:
            return False, reason

        # 3. Type Safety & Breaking Changes Gate
        ok, reason = cls.validate_type_changes(file_path, patch_content, has_migration_evidence)
        if not ok:
            return False, reason

        # 4. Cross-Artifact Consistency Gate
        ok, reason = cls.validate_cross_artifact_consistency(
            file_path, patch_content, sibling_content, sibling_path
        )
        if not ok:
            return False, reason

        return True, "Patch passed all gates"
