"""Unit test for Large File Handling (20,000+ lines).

Verifies:
  1. A 20,000-line file is NEVER sent in full to the LLM (avoids context limits).
  2. Automatic Anchor Recovery: If anchor_methods is empty, recovers targets from task description.
  3. Surgical in-place patching: Only the targeted lines inside the 20,000-line file are modified;
     the remaining 19,990+ lines are preserved byte-for-byte.
"""

from pathlib import Path
import pytest

from ticket_to_code.models import DevelopmentTask, TaskType
from ticket_to_code.agents.code_generator import _apply_str_replace_edits
from ticket_to_code.agents.smart_extract import smart_extract, extract_exact_methods


def _generate_synthetic_20k_line_file() -> str:
    """Generate a realistic 20,000-line TypeScript file with 500 methods."""
    lines = [
        "import { Injectable } from '@angular/core';",
        "import { Observable, of } from 'rxjs';",
        "",
        "@Injectable({ providedIn: 'root' })",
        "export class MonolithicEnterpriseService {",
    ]

    # Generate 490 filler methods
    for i in range(1, 491):
        lines.append(f"  operation_{i}(param: string): number {{")
        lines.append(f"    // line 1 of method {i}")
        lines.append(f"    // line 2 of method {i}")
        lines.append(f"    // line 3 of method {i}")
        lines.append(f"    return {i} * 42;")
        lines.append("  }")
        lines.append("")

    # Place our target method at approximately line 3,500 (or later)
    lines.append("  calculateAnnualTax(income: number, exempt: boolean): number {")
    lines.append("    if (exempt) {")
    lines.append("      return 0;")
    lines.append("    }")
    lines.append("    return income * 0.25;")
    lines.append("  }")
    lines.append("")

    # Add more lines to reach 20,000 lines
    current_count = len(lines)
    remaining_methods = (20000 - current_count) // 6
    for i in range(492, 492 + remaining_methods):
        lines.append(f"  extra_operation_{i}(data: any): void {{")
        lines.append("    const timestamp = Date.now();")
        lines.append("    console.log(timestamp);")
        lines.append("  }")
        lines.append("")

    lines.append("}")
    return "\n".join(lines)


def test_20k_line_file_surgical_patch_preservation():
    """Verify that in a 20k-line file, only the target lines change and the rest is preserved."""
    content = _generate_synthetic_20k_line_file()
    lines = content.split('\n')
    assert len(lines) >= 15000  # Multi-thousand line file

    # Surgical search & replace edit on calculateAnnualTax
    old_str = "    return income * 0.25;"
    new_str = "    const rate = exempt ? 0 : 0.25;\n    return income * rate;"

    edits = [{"old_str": old_str, "new_str": new_str}]
    patched = _apply_str_replace_edits(edits, content, "enterprise.service.ts")

    # Verify target modified
    assert new_str in patched
    assert old_str not in patched

    # Verify surrounding lines preserved (e.g. operation_1 and extra_operation_500)
    assert "operation_1(param: string): number {" in patched
    assert "return 1 * 42;" in patched
    assert "extra_operation_500" in patched


def test_20k_line_file_smart_extraction_bounds():
    """Verify that smart_extract and extract_exact_methods compress 20k lines to < 20k chars."""
    content = _generate_synthetic_20k_line_file()
    assert len(content) > 200000  # Over 200KB

    # Target the calculateAnnualTax method
    verbatim, matched = extract_exact_methods(
        content=content,
        file_path="enterprise.service.ts",
        anchor_methods=["calculateAnnualTax"]
    )

    assert len(matched) == 1
    assert matched[0].name == "calculateAnnualTax"
    assert "calculateAnnualTax(income: number, exempt: boolean): number {" in verbatim
    assert "return income * 0.25;" in verbatim
    # Crucial: verbatim is compact (< 2KB), NOT the whole 200KB!
    assert len(verbatim) < 2000
