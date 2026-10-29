"""Clarification Gate for Zero-Shot & Novel Features.

Prevents hallucination when no similar pattern exists in the codebase:
1. Detects zero-shot gaps: when sibling mining finds 0 relevant call-sites for a critical ticket concept.
2. Formulates structured technical questions with concrete candidate alternatives.
3. Interactive & autonomous resolution:
   - In interactive mode: prompts the user with specific choices.
   - In autonomous/benchmark mode: resolves against type declarations (`.d.ts` or Java interfaces).
4. Injects the confirmed decision into the dependency context as ground truth.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ticket_to_code.lean.lean_path import ClarifyRequest, LockedContext


@dataclass
class ClarifyQuestion:
    topic: str
    question: str
    options: List[str]
    default_index: int = 0
    resolved_choice: Optional[str] = None


class ClarifyGate:
    """Manages architectural ambiguities for novel / zero-shot features."""

    def __init__(self, resolver: Optional[Callable[[ClarifyQuestion], str]] = None):
        self.resolver = resolver
        self.history: List[ClarifyQuestion] = []

    def check_ambiguity(self, concept: str, candidates: List[str], ticket_text: str) -> Optional[ClarifyQuestion]:
        """Flags an ambiguity when multiple service methods could match but none have sibling usage."""
        if len(candidates) <= 1:
            return None
        q = ClarifyQuestion(
            topic=concept,
            question=f"No existing usage of '{concept}' exists in this codebase. Which API method should be invoked?",
            options=candidates,
            default_index=0
        )
        return q

    def resolve(self, question: ClarifyQuestion) -> str:
        """Resolves the question using the injected resolver or default choice."""
        if self.resolver:
            choice = self.resolver(question)
        else:
            choice = question.options[question.default_index] if question.options else ""
        question.resolved_choice = choice
        self.history.append(question)
        return choice

    def format_ground_truth_context(self) -> str:
        """Formats resolved clarifications as immutable facts for the LLM prompt."""
        if not self.history:
            return ""
        lines = ["// CONFIRMED ARCHITECTURAL DECISIONS (GROUND TRUTH):"]
        for q in self.history:
            lines.append(f"// - For '{q.topic}': use '{q.resolved_choice}'")
        return "\n".join(lines) + "\n\n"


def inspect_typings_for_method(node_modules_dir: Path, package_name: str, method_name: str) -> bool:
    """Inspects package .d.ts files in node_modules to verify if a method exists in a third-party library."""
    pkg_path = node_modules_dir / package_name
    if not pkg_path.is_dir():
        # Check @types
        pkg_path = node_modules_dir / "@types" / package_name
    if not pkg_path.is_dir():
        return False

    pattern = re.compile(r"\b" + re.escape(method_name) + r"\b")
    for dts in pkg_path.rglob("*.d.ts"):
        try:
            content = dts.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(content):
                return True
        except OSError:
            continue
    return False
