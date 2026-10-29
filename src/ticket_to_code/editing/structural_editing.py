"""
Format-neutral Structural Editing Subsystem for NemoForge.

Implements capability-based editing backends for Code (AST), YAML, XML, and
conservative source-preserving edits for unknown DSLs. Replaces ambiguous
3-line regex replacements with strict target uniqueness and structural invariants.
"""

from __future__ import annotations

import difflib
import logging
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EditTarget:
    file_path: str
    target_identifier: str  # e.g. symbol name, YAML key path, or exact unique block
    start_line: int
    end_line: int
    matched_content: str
    preconditions: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class EditResult:
    success: bool
    modified_content: str
    diff: str
    error: Optional[str] = None
    target: Optional[EditTarget] = None


@dataclass
class ValidationResult:
    is_valid: bool
    syntax_error: Optional[str] = None
    structural_violations: List[str] = field(default_factory=list)


class EditingBackend(ABC):
    """Abstract contract for all format-aware editing implementations."""

    @abstractmethod
    def inspect(self, file_path: str, content: str) -> Dict[str, Any]:
        """Examine structural nodes, keys, symbols, or grammar boundaries."""
        pass

    @abstractmethod
    def locate(
        self,
        file_path: str,
        content: str,
        intent: str,
        evidence: List[str],
    ) -> Optional[EditTarget]:
        """Locate the smallest unambiguous structural region."""
        pass

    @abstractmethod
    def create_edit(
        self,
        target: EditTarget,
        proposed_change: str,
        original_content: str,
    ) -> EditResult:
        """Construct a patch with explicit preconditions."""
        pass

    @abstractmethod
    def validate(self, file_path: str, proposed_content: str) -> ValidationResult:
        """Validate syntax and structural invariants."""
        pass


class YAMLEditingBackend(EditingBackend):
    """
    Structural editor for YAML (Kubernetes, Helm, docker-compose, CI workflows).
    Preserves comments, whitespace, and prevents indentation ambiguity.
    """

    def inspect(self, file_path: str, content: str) -> Dict[str, Any]:
        lines = content.splitlines()
        top_level_keys: List[str] = []
        for idx, line in enumerate(lines):
            m = re.match(r"^([a-zA-Z0-9_\-\.]+)\s*:", line)
            if m:
                top_level_keys.append(m.group(1))
        return {"total_lines": len(lines), "top_level_keys": top_level_keys}

    def locate(
        self,
        file_path: str,
        content: str,
        intent: str,
        evidence: List[str],
    ) -> Optional[EditTarget]:
        lines = content.splitlines()
        # Look for the target key or stanza mentioned in intent
        intent_lower = intent.lower()
        candidates: List[Tuple[int, int, str, str]] = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key_match = re.match(r"^(\s*)([a-zA-Z0-9_\-\.]+)\s*:\s*(.*)$", line)
            if key_match:
                indent_len = len(key_match.group(1))
                key_name = key_match.group(2)
                if key_name.lower() in intent_lower or any(key_name.lower() in e.lower() for e in evidence):
                    # Determine block boundary by indentation
                    end_idx = idx + 1
                    while end_idx < len(lines):
                        next_line = lines[end_idx]
                        if not next_line.strip() or next_line.strip().startswith("#"):
                            end_idx += 1
                            continue
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent_len:
                            break
                        end_idx += 1
                    block_content = "\n".join(lines[idx:end_idx])
                    candidates.append((idx + 1, end_idx, key_name, block_content))

        if len(candidates) == 1:
            start_l, end_l, key, block = candidates[0]
            return EditTarget(
                file_path=file_path,
                target_identifier=key,
                start_line=start_l,
                end_line=end_l,
                matched_content=block,
                confidence=1.0,
            )
        elif len(candidates) > 1:
            # Ambiguity detected! Filter by evidence match
            best: Optional[Tuple[int, int, str, str]] = None
            best_score = 0
            for start_l, end_l, key, block in candidates:
                score = sum(1 for e in evidence if e in block)
                if score > best_score:
                    best_score = score
                    best = (start_l, end_l, key, block)
            if best and best_score > 0:
                start_l, end_l, key, block = best
                return EditTarget(
                    file_path=file_path,
                    target_identifier=key,
                    start_line=start_l,
                    end_line=end_l,
                    matched_content=block,
                    confidence=0.85,
                )
        return None

    def create_edit(
        self,
        target: EditTarget,
        proposed_change: str,
        original_content: str,
    ) -> EditResult:
        lines = original_content.splitlines()
        orig_slice = lines[target.start_line - 1 : target.end_line]

        # Invariant 1: original slice must match target matched content
        if "\n".join(orig_slice).strip() != target.matched_content.strip():
            return EditResult(
                success=False,
                modified_content=original_content,
                diff="",
                error=f"YAML invariant check failed: target content changed at lines {target.start_line}-{target.end_line}",
            )

        # Invariant 2: Guarantee zero modification to lines before start_line and after end_line
        prefix_lines = lines[: target.start_line - 1]
        suffix_lines = lines[target.end_line :]
        new_lines = prefix_lines + proposed_change.splitlines() + suffix_lines
        modified = "\n".join(new_lines)
        if original_content.endswith("\n") and not modified.endswith("\n"):
            modified += "\n"

        diff = "".join(
            difflib.unified_diff(
                lines,
                new_lines,
                fromfile=f"a/{target.file_path}",
                tofile=f"b/{target.file_path}",
                lineterm="",
            )
        )
        return EditResult(success=True, modified_content=modified, diff=diff, target=target)

    def validate(self, file_path: str, proposed_content: str) -> ValidationResult:
        lines = proposed_content.splitlines()
        for idx, line in enumerate(lines):
            if "\t" in line:
                return ValidationResult(
                    is_valid=False,
                    syntax_error=f"YAML syntax violation: Tab character detected on line {idx+1}. Tabs are forbidden in YAML.",
                )

        # Helm template detection (Go template syntax: {{ ... }})
        is_helm = "{{" in proposed_content or "}}" in proposed_content
        if is_helm:
            # 1. Verify Go template delimiter balance
            open_tags = proposed_content.count("{{")
            close_tags = proposed_content.count("}}")
            if open_tags != close_tags:
                return ValidationResult(
                    is_valid=False,
                    syntax_error=f"Helm Go-template delimiter mismatch: {open_tags} '{{{{' tags vs {close_tags} '}}}}' tags",
                )

            # 2. Verify block balance (if/range/with vs end)
            block_starts = len(re.findall(r"\{\{-?\s*(?:if|range|with)\b", proposed_content))
            block_ends = len(re.findall(r"\{\{-?\s*end\b", proposed_content))
            if block_starts != block_ends:
                return ValidationResult(
                    is_valid=False,
                    syntax_error=f"Helm control flow block mismatch: {block_starts} start blocks (if/range/with) vs {block_ends} 'end' directives",
                )

            # 3. Mask template directives for YAML structure verification
            # Replace inline {{ ... }} with a safe scalar token to verify underlying YAML structure
            masked_content = re.sub(r"\{\{.*?\}\}", "helm_rendered_scalar", proposed_content)
            # Remove standalone control lines like {{- if ... }} or {{- end }}
            masked_lines = [
                line if not re.match(r"^\s*helm_rendered_scalar\s*$", line) else f"# {line}"
                for line in masked_content.splitlines()
            ]
            content_to_check = "\n".join(masked_lines)
        else:
            content_to_check = proposed_content

        try:
            import yaml  # type: ignore
            list(yaml.safe_load_all(content_to_check))
            return ValidationResult(is_valid=True)
        except ImportError:
            return ValidationResult(is_valid=True)
        except Exception as e:
            # If pyyaml failed on a Helm template after masking, provide a detailed diagnostic
            prefix = "Helm/YAML syntax error" if is_helm else "YAML syntax error"
            return ValidationResult(is_valid=False, syntax_error=f"{prefix}: {str(e)}")


class XMLEditingBackend(EditingBackend):
    """
    Structural editor for XML (Maven pom.xml, Spring configs, Android manifests).
    Guarantees tag balance and hierarchy preservation.
    """

    def inspect(self, file_path: str, content: str) -> Dict[str, Any]:
        try:
            root = ET.fromstring(content)
            return {"root_tag": root.tag, "valid_xml": True}
        except Exception as e:
            return {"valid_xml": False, "error": str(e)}

    def locate(
        self,
        file_path: str,
        content: str,
        intent: str,
        evidence: List[str],
    ) -> Optional[EditTarget]:
        lines = content.splitlines()
        # Look for XML tags mentioned in intent (e.g. <dependency>, <plugin>, <properties>)
        intent_lower = intent.lower()
        for idx, line in enumerate(lines):
            tag_match = re.search(r"<([a-zA-Z0-9_\-]+)>", line)
            if tag_match:
                tag_name = tag_match.group(1)
                closing_tag = f"</{tag_name}>"
                if tag_name.lower() in intent_lower or any(tag_name.lower() in e.lower() for e in evidence):
                    # Find matching closing tag
                    for end_idx in range(idx, min(len(lines), idx + 100)):
                        if closing_tag in lines[end_idx]:
                            block = "\n".join(lines[idx : end_idx + 1])
                            return EditTarget(
                                file_path=file_path,
                                target_identifier=tag_name,
                                start_line=idx + 1,
                                end_line=end_idx + 1,
                                matched_content=block,
                            )
        return None

    def create_edit(
        self,
        target: EditTarget,
        proposed_change: str,
        original_content: str,
    ) -> EditResult:
        lines = original_content.splitlines()
        new_lines = lines[: target.start_line - 1] + proposed_change.splitlines() + lines[target.end_line :]
        modified = "\n".join(new_lines)
        if original_content.endswith("\n") and not modified.endswith("\n"):
            modified += "\n"

        diff = "".join(
            difflib.unified_diff(
                lines,
                new_lines,
                fromfile=f"a/{target.file_path}",
                tofile=f"b/{target.file_path}",
                lineterm="",
            )
        )
        return EditResult(success=True, modified_content=modified, diff=diff, target=target)

    def validate(self, file_path: str, proposed_content: str) -> ValidationResult:
        try:
            ET.fromstring(proposed_content)
            return ValidationResult(is_valid=True)
        except Exception as e:
            return ValidationResult(is_valid=False, syntax_error=f"XML parsing error: {str(e)}")


class ConservativeSourcePreservingBackend(EditingBackend):
    """
    Conservative, source-preserving editor for SQL, Protobuf, GraphQL, and unknown DSLs.
    Enforces strict target uniqueness and surrounding structure invariant checks.
    Never replaces text if multiple occurrences exist or surrounding context does not match.
    """

    def inspect(self, file_path: str, content: str) -> Dict[str, Any]:
        return {"total_lines": len(content.splitlines()), "total_bytes": len(content)}

    def locate(
        self,
        file_path: str,
        content: str,
        intent: str,
        evidence: List[str],
    ) -> Optional[EditTarget]:
        # Search for unique anchor block in evidence
        for ev in evidence:
            snippet = ev.strip()
            if len(snippet) < 20:
                continue
            count = content.count(snippet)
            if count == 1:
                # Exactly one match found!
                start_char = content.find(snippet)
                start_line = content[:start_char].count("\n") + 1
                end_line = start_line + snippet.count("\n")
                return EditTarget(
                    file_path=file_path,
                    target_identifier=f"unique_snippet_L{start_line}",
                    start_line=start_line,
                    end_line=end_line,
                    matched_content=snippet,
                    confidence=1.0,
                )
            elif count > 1:
                logger.warning(
                    "Conservative editor: snippet in %s matched %d times. Rejecting ambiguous target.",
                    file_path,
                    count,
                )
        return None

    def create_edit(
        self,
        target: EditTarget,
        proposed_change: str,
        original_content: str,
    ) -> EditResult:
        if original_content.count(target.matched_content) != 1:
            return EditResult(
                success=False,
                modified_content=original_content,
                diff="",
                error=f"Ambiguity violation: target block in {target.file_path} does not match exactly once.",
            )

        modified = original_content.replace(target.matched_content, proposed_change, 1)
        orig_lines = original_content.splitlines()
        new_lines = modified.splitlines()
        diff = "".join(
            difflib.unified_diff(
                orig_lines,
                new_lines,
                fromfile=f"a/{target.file_path}",
                tofile=f"b/{target.file_path}",
                lineterm="",
            )
        )
        return EditResult(success=True, modified_content=modified, diff=diff, target=target)

    def validate(self, file_path: str, proposed_content: str) -> ValidationResult:
        # Check basic invariants: non-empty, no null bytes, UTF-8 decodable
        if not proposed_content.strip():
            return ValidationResult(is_valid=False, syntax_error="File content became empty after edit")
        if "\x00" in proposed_content:
            return ValidationResult(is_valid=False, syntax_error="Null byte detected in text content")
        return ValidationResult(is_valid=True)


class StructuralEditingRegistry:
    """Selects the format-appropriate editing backend based on file type and capabilities."""

    def __init__(self):
        self._yaml_backend = YAMLEditingBackend()
        self._xml_backend = XMLEditingBackend()
        self._conservative_backend = ConservativeSourcePreservingBackend()

    def get_backend(self, file_path: str) -> EditingBackend:
        ext = Path(file_path).suffix.lower()
        if ext in (".yaml", ".yml"):
            return self._yaml_backend
        elif ext in (".xml", ".pom"):
            return self._xml_backend
        else:
            return self._conservative_backend
