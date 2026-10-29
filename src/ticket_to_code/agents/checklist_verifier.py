"""
Phase 5: ChecklistVerifier — Verify generated code against Phase 0's definition-of-done.

For each checklist item:
  PASS: LLM cites exact file + line number + code snippet
  FAIL: LLM cannot find implementation

MECHANICAL CITATION VERIFICATION:
  After the LLM declares PASS with a cited snippet, we do a cheap grep to
  confirm the cited snippet ACTUALLY APPEARS in the file near the cited line.
  LLMs under retry pressure will sometimes cite a line that doesn't say what
  they claim — this catches that.

BOUNDED RETRY:
  Unresolved items are fed back to CodeGeneratorAgent for a targeted fix pass.
  Maximum 2 fix iterations. After that, remaining unresolved items are written
  to state["unresolved_checklist_items"] for Phase 6 escalation.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying generated code against the definition-of-done."""
    total_items: int = 0
    passed: int = 0
    failed: int = 0
    items: list = field(default_factory=list)  # list of {id, description, status, evidence}
    
    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.total_items > 0
    
    @property
    def pass_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.passed / self.total_items


class ChecklistVerifier:
    """
    Phase 5: Verify generated code against Phase 0's definition-of-done.
    
    Two-stage verification:
    1. LLM-based: Ask the LLM to find each checklist item in the generated code
    2. Mechanical: Grep the cited file to confirm the citation is real
    
    Integration point: Called after build succeeds, before memory_update.
    """

    MAX_FIX_ITERATIONS = 2

    SYSTEM_PROMPT = """You are a code reviewer verifying that generated code satisfies a checklist.

For EACH checklist item below, examine the provided files and determine:
- PASS: The item is implemented. Cite the exact file, line number, and code snippet.
- FAIL: The item is NOT implemented or is incomplete. Explain what's missing.

Respond with JSON ONLY:
{
  "results": [
    {
      "item_id": 1,
      "status": "pass",
      "file": "path/to/file.ts",
      "line": 47,
      "snippet": "allProjectMembers: any[] = [];",
      "reasoning": "Property is declared on line 47"
    },
    {
      "item_id": 2,
      "status": "fail",
      "file": "",
      "line": 0,
      "snippet": "",
      "reasoning": "No API call found to fetch project members"
    }
  ]
}

RULES:
- Be STRICT. "Declares a property" and "has logic that SETS the property" are different items.
- A property that is declared but never assigned a value is NOT a pass for "has logic that sets X".
- The snippet must be the ACTUAL code from the file, not a paraphrase.
- Line numbers must be accurate — you will be checked mechanically.
- No markdown, only JSON."""

    def __init__(self, llm):
        self.llm = llm

    def verify(
        self,
        checklist_items: list,
        generated_files: Dict[str, str],
    ) -> VerificationResult:
        """
        Verify generated code against checklist items.
        
        Args:
            checklist_items: List of ChecklistItem from Phase 0
            generated_files: Dict of {file_path: file_content} for all generated files
        
        Returns:
            VerificationResult with pass/fail for each item
        """
        if not checklist_items:
            return VerificationResult()

        # Build the verification prompt
        checklist_text = "\n".join(
            f"  {item.id}. {item.description}"
            + (f" [expected in: {item.file_hint}]" if item.file_hint else "")
            + (f" [expected symbol: {item.symbol_hint}]" if item.symbol_hint else "")
            for item in checklist_items
        )

        files_text = ""
        for fp, content in generated_files.items():
            # Truncate very large files but keep enough for verification
            _truncated = content[:8000] + "\n...[TRUNCATED]" if len(content) > 8000 else content
            files_text += f"\n\nFILE: {fp}\n```\n{_truncated}\n```\n"

        user_prompt = (
            f"CHECKLIST TO VERIFY:\n{checklist_text}\n\n"
            f"GENERATED FILES:{files_text}\n\n"
            f"Verify each item. JSON only."
        )

        try:
            from ticket_to_code.llm_utils import llm_invoke
            response = llm_invoke(
                self.llm,
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            raw = response.content if hasattr(response, "content") else str(response)
            result = self._parse_and_verify(raw, checklist_items, generated_files)
            return result

        except Exception as e:
            logger.warning(f"ChecklistVerifier failed: {e}")
            return VerificationResult(
                total_items=len(checklist_items),
                failed=len(checklist_items),
                items=[
                    {"id": item.id, "description": item.description,
                     "status": "fail", "evidence": f"Verification error: {e}"}
                    for item in checklist_items
                ],
            )

    def _parse_and_verify(
        self,
        raw: str,
        checklist_items: list,
        generated_files: Dict[str, str],
    ) -> VerificationResult:
        """Parse LLM response and apply mechanical citation verification."""
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            logger.warning("ChecklistVerifier: no JSON in response")
            return VerificationResult(
                total_items=len(checklist_items),
                failed=len(checklist_items),
            )

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            logger.warning("ChecklistVerifier: JSON parse failed")
            return VerificationResult(
                total_items=len(checklist_items),
                failed=len(checklist_items),
            )

        results_data = data.get("results", [])
        items_out = []
        passed = 0
        failed = 0

        # Index LLM results by item_id
        llm_results = {r.get("item_id"): r for r in results_data}

        for item in checklist_items:
            llm_r = llm_results.get(item.id, {})
            status = llm_r.get("status", "fail")
            cited_file = llm_r.get("file", "")
            cited_line = llm_r.get("line", 0)
            cited_snippet = llm_r.get("snippet", "")
            reasoning = llm_r.get("reasoning", "")

            # Mechanical citation check for PASS results
            if status == "pass" and cited_file and cited_snippet:
                if not self._mechanical_citation_check(
                    item, cited_file, cited_line, cited_snippet, generated_files
                ):
                    # LLM claimed PASS but citation doesn't check out
                    status = "fail"
                    reasoning = (
                        f"CITATION FAILED: LLM claimed line {cited_line} of "
                        f"{cited_file} contains '{cited_snippet[:60]}' but "
                        f"mechanical check found no match or lacked required semantics. Original: {reasoning}"
                    )
                    logger.warning(
                        f"  ⚠️ [Verifier] Item {item.id}: citation check FAILED — "
                        f"'{cited_snippet[:40]}' failed mechanical/semantic checks in {cited_file}"
                    )

            if status == "pass":
                passed += 1
                item.status = "pass"
                item.evidence = f"L{cited_line} in {cited_file}: {cited_snippet[:80]}"
            else:
                failed += 1
                item.status = "fail"
                item.evidence = reasoning

            items_out.append({
                "id": item.id,
                "description": item.description,
                "status": status,
                "evidence": item.evidence,
            })

        result = VerificationResult(
            total_items=len(checklist_items),
            passed=passed,
            failed=failed,
            items=items_out,
        )

        logger.info(
            f"  ✅ ChecklistVerifier: {passed}/{len(checklist_items)} items passed "
            f"({result.pass_rate:.0%})"
        )
        if failed > 0:
            _failed_descs = [
                it["description"][:60] for it in items_out if it["status"] == "fail"
            ]
            logger.warning(
                f"  ⚠️ [Verifier] {failed} item(s) FAILED: "
                + "; ".join(_failed_descs[:3])
            )

        return result

    def _mechanical_citation_check(
        self,
        item: "ChecklistItem",
        cited_file: str,
        cited_line: int,
        cited_snippet: str,
        actual_files: Dict[str, str],
    ) -> bool:
        """
        Confirm the cited snippet actually appears near the cited line.
        
        This catches LLMs that claim PASS with a fabricated citation —
        especially under retry pressure where they're incentivized to
        say "yes it's there" when it isn't.
        """
        # Find the file content (try exact match, then basename match)
        content = actual_files.get(cited_file, "")
        if not content:
            # Try matching by basename
            cited_basename = cited_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
            for fp, fc in actual_files.items():
                fp_basename = fp.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
                if fp_basename == cited_basename:
                    content = fc
                    break

        if not content:
            return False

        lines = content.splitlines()
        snippet_clean = cited_snippet.strip()

        if not snippet_clean:
            return False

        found = False
        # Check ±5 lines around the cited line number
        if cited_line > 0:
            window_start = max(0, cited_line - 6)
            window_end = min(len(lines), cited_line + 5)
            window = lines[window_start:window_end]
            if any(snippet_clean in line for line in window):
                found = True

        # Fallback: search entire file (line number might be off but snippet is real)
        if not found:
            found = any(snippet_clean in line for line in lines)
            
        if not found:
            return False
            
        # Semantic check for assignments (Priority 2)
        if getattr(item, "verification_type", "") == "assignment" and getattr(item, "symbol_hint", ""):
            # Ensure the symbol appears on the left side of an assignment operator (=)
            # We explicitly exclude `:` to avoid false-PASSing on type declarations
            import re
            symbol = item.symbol_hint
            # Regex: word boundary + symbol + word boundary + optional whitespace + =
            if not re.search(r'\b' + re.escape(symbol) + r'\b\s*=', snippet_clean):
                return False
                
        return True
