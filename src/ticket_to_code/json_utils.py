"""Utility helpers for resilient JSON parsing of LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def clean_llm_json_text(text: str) -> str:
    """Remove markdown code fences and trim whitespace."""
    cleaned = (text or "").strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    return cleaned


def repair_truncated_json(text: str) -> Optional[str]:
    """Repair a truncated JSON document by closing open strings/containers.

    LLM responses are frequently cut off mid-string or mid-object (producing
    'Unterminated string' errors). This scans for structurally safe truncation
    boundaries (after a completed value, a comma, or a container close) and,
    for each candidate from longest to shortest, appends the minimal closing
    tokens needed to make it parseable. Returns the first fragment that parses.
    """
    if not text:
        return None

    # Find the first structural opener so we ignore leading prose.
    start = None
    for idx, ch in enumerate(text):
        if ch in "{[":
            start = idx
            break
    if start is None:
        return None

    # Collect structurally safe truncation boundaries (exclusive end indices).
    boundaries: list[int] = []
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                boundaries.append(idx + 1)  # end of a complete string value/key
            continue
        if ch == '"':
            in_string = True
        elif ch in "}]":
            boundaries.append(idx + 1)  # container just closed
        elif ch == ",":
            boundaries.append(idx)  # element separator (exclude the comma)
        elif ch.isdigit() or ch in "eE.+-" or ch in "truefalsn":
            boundaries.append(idx + 1)  # possible end of a literal

    # Always consider the full text (covers already-balanced-but-noisy cases).
    boundaries.append(len(text))
    # De-dup while keeping descending order (longest fragment first).
    seen: set[int] = set()
    ordered = [b for b in sorted(boundaries, reverse=True) if not (b in seen or seen.add(b))]

    def _close_fragment(fragment: str) -> Optional[str]:
        body = fragment.rstrip().rstrip(",").rstrip()
        if not body:
            return None
        stack: list[str] = []
        s_in = False
        s_esc = False
        for c in body:
            if s_in:
                if s_esc:
                    s_esc = False
                elif c == "\\":
                    s_esc = True
                elif c == '"':
                    s_in = False
                continue
            if c == '"':
                s_in = True
            elif c == "{":
                stack.append("}")
            elif c == "[":
                stack.append("]")
            elif c in "}]" and stack:
                stack.pop()
        if s_in:
            body += '"'
        # Drop a dangling object key with no value: `..., "key"` or `..., "key":`
        body = re.sub(r'([\{,]\s*"(?:[^"\\]|\\.)*"\s*:?)\s*$', "", body).rstrip().rstrip(",")
        for closer in reversed(stack):
            body += closer
        return body

    for end in ordered:
        fragment = text[start:end]
        repaired = _close_fragment(fragment)
        if not repaired:
            continue
        for candidate in (repaired, re.sub(r",\s*([}\]])", r"\1", repaired)):
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue

    return None


def extract_json_payload(text: str, expect_list: bool = False) -> Optional[Any]:
    """Best-effort JSON extraction from noisy/truncated LLM output."""
    if not text:
        return None

    pattern = r"\[.*\]" if expect_list else r"\{.*\}"
    match = re.search(pattern, text, re.DOTALL)

    candidates: list[str] = []
    if match:
        raw = match.group(0)
        candidates.extend([raw, re.sub(r",\s*([}\]])", r"\1", raw)])

    # Add a repaired variant for truncated payloads (no closing token present).
    repaired = repair_truncated_json(text)
    if repaired:
        candidates.append(repaired)
        candidates.append(re.sub(r",\s*([}\]])", r"\1", repaired))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if expect_list and isinstance(parsed, list):
            return parsed
        if not expect_list and isinstance(parsed, dict):
            return parsed

    return None


def parse_llm_json(text: str, expect_list: bool = False) -> Any:
    """Parse JSON from LLM output with cleanup and fallback extraction."""
    cleaned = clean_llm_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = extract_json_payload(cleaned, expect_list=expect_list)
        if candidate is not None:
            return candidate
        raise
