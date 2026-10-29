# Smart File Extraction — Production-Grade Implementation
# 
# How AI IDEs (Cursor, Copilot Workspace, Aider, Devin) Handle Large Files
# ═══════════════════════════════════════════════════════════════════════════
#
# THE CORE PROBLEM:
#   LLMs have a "Lost in the Middle" flaw. When you send a 112K file,
#   they pay attention to the first ~5K and last ~5K, and IGNORE the middle.
#   This means the exact method you need edited (buried at line 2500) gets lost.
#
# THE SOLUTION (3-Phase Architecture used by ALL top AI IDEs):
#
#   ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
#   │  Phase 1:   │ ──▶ │  Phase 2:    │ ──▶ │  Phase 3:    │
#   │  SEARCH     │     │  EXTRACT     │     │  EDIT        │
#   │  (LLM #1)   │     │  (Pure Code) │     │  (LLM #2)    │
#   └─────────────┘     └──────────────┘     └──────────────┘
#
#   Phase 1 — SEARCH (uses LLM):
#     The Planner Agent reads the ticket + file candidates (just paths
#     and signatures, NOT full contents). It outputs a structured plan:
#       {
#         "file_path": "DeliverablesServiceImpl.java",
#         "allowed_methods": ["validateDeliverableName", "saveDeliverable"],
#         "target_class": "DeliverablesServiceImpl"
#       }
#     This is the "Scout" — it tells us WHERE to look, not HOW to fix.
#
#   Phase 2 — EXTRACT (pure Python, NO LLM):
#     This module! Given the file content + the method names from Phase 1,
#     we use regex-based AST parsing to:
#       a) Build a STRUCTURAL OUTLINE of the entire file (every method signature
#          with line numbers, but bodies collapsed to "{ ... }")
#       b) EXTRACT IN FULL the specific methods from allowed_methods
#       c) FOLLOW CALL CHAINS: if validateDeliverableName() calls
#          checkNameUniqueness(), we include that method too (2 levels deep)
#       d) Always include IMPORTS + CLASS HEADER for context
#
#   Phase 3 — EDIT (uses LLM):
#     The Code Generator Agent receives the extracted content (typically
#     5-15K chars instead of 112K) and writes SEARCH/REPLACE blocks.
#     Because the LLM can see the FULL body of the target method + its
#     call chain, it writes precise, correct patches.
#
# WHY THIS WORKS:
#   - The LLM in Phase 1 never reads the 112K file. It reads signatures
#     and search snippets (~5K total) and outputs method names.
#   - Phase 2 is pure Python regex — instant, deterministic, no LLM cost.
#   - The LLM in Phase 3 reads only the relevant ~15K, so it's fast and
#     accurate. The structural outline gives it enough context to understand
#     the class architecture without reading every unrelated method.
#
# EXAMPLE:
#   Input:  DeliverablesServiceImpl.java (112,000 chars, 3,000 lines, 45 methods)
#   Target: allowed_methods = ["validateDeliverableName"]
#
#   Output (what the LLM sees):
#     ┌──────────────────────────────────────────────────────────┐
#     │ // === IMPORTS & CLASS HEADER (lines 1-35) ===           │
#     │ package com.cloudforge.services.services...               │
#     │ import java.util.*;                                      │
#     │ ...                                                      │
#     │ public class DeliverablesServiceImpl implements ...  {   │
#     │                                                          │
#     │ // === FILE OUTLINE (45 methods) ===                     │
#     │   L36-L85   (50 lines):  public void initService() ...  │
#     │   L86-L220  (135 lines): public Deliverable get(...) ... │
#     │   ...                                                    │
#     │   L2501-L2510 (10 lines): validateDeliverableName(...)   │  ← TARGET
#     │   ...                                                    │
#     │                                                          │
#     │ // === TARGET METHODS (full content) ===                 │
#     │ // L2501-L2510: validateDeliverableName                  │
#     │ public void validateDeliverableName(String name) {       │
#     │     List<Deliverable> existing = repo.findByName(name);  │
#     │     if (!existing.isEmpty()) {                           │
#     │         throw new ValidationException("Name already.."); │
#     │     }                                                    │
#     │ }                                                        │
#     │                                                          │
#     │ // === CALLED BY TARGET (call chain) ===                 │
#     │ // L1800-L1825: findByName (called by above)             │
#     │ private List<Deliverable> findByName(String name) {      │
#     │     return repository.findAll().stream()                 │
#     │         .filter(d -> d.getName().equals(name))           │
#     │         .collect(Collectors.toList());                   │
#     │ }                                                        │
#     └──────────────────────────────────────────────────────────┘
#
#   Total sent to LLM: ~8,000 chars instead of 112,000 chars
#   Speed improvement: ~10x faster generation
#   Accuracy improvement: LLM focuses ONLY on the bug
#
# ═══════════════════════════════════════════════════════════════════════════

import re
import logging
from typing import List, Tuple, Optional, Set, Dict, Any

logger = logging.getLogger(__name__)


def _capture_full_signature(lines: List[str], start_line: int, max_lines: int = 5) -> str:
    """Capture a full method/class signature, including multi-line throws clauses.

    Java methods often have long signatures that span multiple lines:
        private Page<Deliverable> getDeliverables(
            List<FilterQueryParam> queryParams, String areaId, ...
        ) throws MappingException, AreaServiceException, ... {

    Truncating at 150 chars loses the throws clause entirely, which means
    the LLM never sees what exceptions a method declares.  This function
    captures everything from start_line up to (and including) the opening
    '{' brace, joining continuation lines with a single space.
    """
    total = len(lines)
    sig_parts: List[str] = []
    for offset in range(max_lines):
        idx = start_line + offset
        if idx >= total:
            break
        line = lines[idx].strip()
        sig_parts.append(line)
        if '{' in line:
            break
    return ' '.join(sig_parts)


# ─── Data Types ───────────────────────────────────────────────────────────

class MethodBoundary:
    """Represents a parsed method/function/class boundary in source code."""
    __slots__ = ("start_line", "end_line", "name", "signature", "kind")

    def __init__(self, start_line: int, end_line: int, name: str, signature: str, kind: str = "method"):
        self.start_line = start_line
        self.end_line = end_line
        self.name = name
        self.signature = signature
        self.kind = kind  # "class", "method", "function", "constructor"

    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def __repr__(self):
        return f"<{self.kind} {self.name} L{self.start_line+1}-L{self.end_line+1} ({self.line_count()} lines)>"


# ─── Language-Specific Parsers ────────────────────────────────────────────

# Regex patterns for Java/TypeScript method detection
_JAVA_TS_METHOD_RE = re.compile(
    r'^\s*(?:@\w+(?:\([^)]*\))?\s*)*'                       # optional annotations
    r'(?:public|private|protected|static|async|abstract|'
    r'override|final|synchronized|default|native|readonly|\s)*' # modifiers
    r'(?:[\w<>\[\],\s]+\s+)?'                                # return type (optional Java)
    r'(\w+)\s*\([^)]*\)\s*'                                  # method name + params
    r'(?::\s*[\w<>\[\],\s|&?]+)?\s*'                         # return type (TypeScript)
    r'(?:throws\s+[\w,\s]+)?\s*\{',                          # optional throws + opening brace
    re.MULTILINE
)

_JAVA_TS_CLASS_RE = re.compile(
    r'^\s*(?:export\s+)?(?:public|private|protected|abstract|final|static|\s)*'
    r'(?:class|interface|enum|record)\s+(\w+)',
    re.MULTILINE
)

_JAVA_TS_CONSTRUCTOR_RE = re.compile(
    r'^\s*(?:(?:public|private|protected)\s+)?constructor\s*\([^)]*\)\s*\{',
    re.MULTILINE
)

_PYTHON_FUNC_RE = re.compile(r'^(\s*)(def|class|async\s+def)\s+(\w+)')


def _parse_java_ts_boundaries(lines: List[str]) -> List[MethodBoundary]:
    """
    Parse Java/TypeScript files to find class, constructor, and method boundaries.
    Uses brace-counting to accurately find class and method end lines.
    """
    boundaries: List[MethodBoundary] = []
    total = len(lines)

    # First pass: find all class/interface/enum declarations with accurate end-line tracking
    for i, line in enumerate(lines):
        cm = _JAVA_TS_CLASS_RE.match(line)
        if cm:
            c_name = cm.group(1)
            c_sig = _capture_full_signature(lines, i)
            c_depth = 0
            has_opened = False
            c_end = i
            for j in range(i, total):
                c_depth += lines[j].count('{') - lines[j].count('}')
                if '{' in lines[j]:
                    has_opened = True
                if has_opened and c_depth <= 0:
                    c_end = j
                    break
            if not has_opened or c_depth > 0:
                c_end = total - 1
            boundaries.append(MethodBoundary(i, c_end, c_name, c_sig, "class"))

    # Second pass: find methods using brace-depth tracking
    brace_depth = 0
    current_start: Optional[int] = None
    current_name: Optional[str] = None
    current_sig: Optional[str] = None
    current_kind = "method"

    for i, line in enumerate(lines):
        # Only look for new methods at class-body level (brace_depth <= 1)
        if current_start is None and brace_depth <= 1:
            # Check constructor first
            cm = _JAVA_TS_CONSTRUCTOR_RE.match(line)
            if cm:
                current_start = i
                current_name = "constructor"
                current_sig = _capture_full_signature(lines, i)
                current_kind = "constructor"
            else:
                # Check regular method
                mm = _JAVA_TS_METHOD_RE.match(line)
                if mm:
                    current_start = i
                    current_name = mm.group(1)
                    current_sig = _capture_full_signature(lines, i)
                    current_kind = "method"

        # Track braces
        brace_depth += line.count('{') - line.count('}')

        # Method ends when we return to class-body level
        if current_start is not None and brace_depth <= 1:
            boundaries.append(MethodBoundary(current_start, i, current_name, current_sig, current_kind))
            current_start = None
            current_name = None
            current_sig = None

    # Handle unclosed method at EOF
    if current_start is not None:
        boundaries.append(MethodBoundary(current_start, total - 1, current_name, current_sig, current_kind))

    return boundaries


def _parse_python_boundaries(lines: List[str]) -> List[MethodBoundary]:
    """Parse Python files to find function/class boundaries using indentation."""
    boundaries: List[MethodBoundary] = []
    total = len(lines)
    current_start: Optional[int] = None
    current_name: Optional[str] = None
    current_sig: Optional[str] = None
    current_indent = 0
    current_kind = "function"

    for i, line in enumerate(lines):
        m = _PYTHON_FUNC_RE.match(line)
        if m:
            # Close previous block
            if current_start is not None:
                boundaries.append(MethodBoundary(current_start, i - 1, current_name, current_sig, current_kind))
            current_start = i
            current_kind = "class" if m.group(2) == "class" else "function"
            current_name = m.group(3)
            current_sig = line.strip()[:150]
            current_indent = len(m.group(1))

    # Close last block
    if current_start is not None:
        boundaries.append(MethodBoundary(current_start, total - 1, current_name, current_sig, current_kind))

    return boundaries


def _detect_language(file_path: str) -> str:
    """Detect language from file extension."""
    fp = file_path.lower()
    if fp.endswith('.java'):
        return "java"
    elif fp.endswith(('.ts', '.tsx')):
        return "typescript"
    elif fp.endswith(('.js', '.jsx')):
        return "javascript"
    elif fp.endswith('.py'):
        return "python"
    elif fp.endswith(('.html', '.htm')):
        return "html"
    elif fp.endswith(('.scss', '.css', '.less')):
        return "css"
    else:
        return "unknown"


# ─── Call Chain Follower ──────────────────────────────────────────────────

_CALL_PATTERN = re.compile(r'(?:this\.|self\.)?(\w+)\s*\(')

def _follow_call_chain(
    target_boundaries: List[MethodBoundary],
    all_boundaries: List[MethodBoundary],
    lines: List[str],
    max_depth: int = 2
) -> List[MethodBoundary]:
    """
    Given the target methods, find other methods in the SAME file that
    they call (up to max_depth levels deep).

    This is how Cursor and Aider work — they follow the call graph within
    a file so the LLM has the complete picture of the code path.
    """
    # Build method lookup by name
    method_by_name: Dict[str, MethodBoundary] = {}
    for mb in all_boundaries:
        if mb.kind in ("method", "function", "constructor"):
            method_by_name[mb.name] = mb

    included_names: Set[str] = {mb.name for mb in target_boundaries}
    chain_results: List[MethodBoundary] = []

    frontier = list(target_boundaries)

    for depth in range(max_depth):
        next_frontier: List[MethodBoundary] = []
        for mb in frontier:
            body = '\n'.join(lines[mb.start_line:mb.end_line + 1])
            calls = _CALL_PATTERN.findall(body)
            for call_name in calls:
                if call_name in method_by_name and call_name not in included_names:
                    called_mb = method_by_name[call_name]
                    chain_results.append(called_mb)
                    included_names.add(call_name)
                    next_frontier.append(called_mb)
        frontier = next_frontier
        if not frontier:
            break

    return chain_results


# ─── Main Smart Extraction Function ──────────────────────────────────────

def smart_extract(
    content: str,
    file_path: str,
    allowed_methods: Optional[List[str]] = None,
    target_method: Optional[str] = None,
    edit_description: Optional[str] = None,
    max_output_chars: int = 55000,
) -> str:
    """
    Production-grade smart file extraction for large files.

    This is the Phase 2 "EXTRACT" step used by AI IDEs:
      1. Parse the file into method boundaries (pure regex, no LLM)
      2. Build a structural outline (all method signatures with line numbers)
      3. Extract FULL content of target methods (from allowed_methods)
      4. Follow call chains 2 levels deep (method A calls B, B calls C)
      5. Always include imports + class header for context

    Args:
        content:           Full file content (can be 112K+ chars)
        file_path:         Path to file (for language detection)
        allowed_methods:   Method names the Planner said to edit (from Phase 1)
        target_method:     Primary target method name (optional, from Planner)
        edit_description:  Free-text description of the edit (fallback matching)
        max_output_chars:  Safety cap on output size

    Returns:
        Extracted content suitable for LLM consumption (typically 5-15K chars)
    """
    lines = content.split('\n')
    total_lines = len(lines)
    content_len = len(content)
    lang = _detect_language(file_path)

    # ── Step 1: Parse method boundaries ───────────────────────────────────
    if lang in ("java", "typescript", "javascript"):
        boundaries = _parse_java_ts_boundaries(lines)
    elif lang == "python":
        boundaries = _parse_python_boundaries(lines)
    else:
        # For HTML/CSS/unknown: no method-level extraction possible
        # Return first chunk + last chunk
        if content_len <= max_output_chars:
            return content
        half = max_output_chars // 2
        return (
            content[:half]
            + f"\n\n// ... [{content_len - max_output_chars:,} chars omitted] ...\n\n"
            + content[-half:]
        )

    if not boundaries:
        # No methods found — return as much as we can
        if content_len <= max_output_chars:
            return content
        return content[:max_output_chars] + f"\n// [TRUNCATED — file is {content_len:,} chars total]"

    # ── Step 2: Build method name lookup for targeting ────────────────────
    methods_to_find: List[str] = []
    if target_method:
        methods_to_find.append(target_method)
    if allowed_methods:
        for m in allowed_methods:
            if m not in methods_to_find:
                methods_to_find.append(m)

    # ── Step 3: Find target methods ───────────────────────────────────────
    target_boundaries: List[MethodBoundary] = []
    target_starts: Set[int] = set()

    # Strategy A: Direct name match from allowed_methods (highest confidence)
    if methods_to_find:
        for mb in boundaries:
            if mb.name in methods_to_find and mb.start_line not in target_starts:
                target_boundaries.append(mb)
                target_starts.add(mb.start_line)

    # Strategy B: If no direct match, try keyword matching from edit_description
    if not target_boundaries and edit_description:
        keywords = set(re.findall(r'[a-zA-Z]\w{3,}', edit_description.lower()))
        # Remove common noise words
        keywords -= {"this", "that", "with", "from", "have", "been", "should",
                      "edit", "editing", "methods", "method", "file", "class",
                      "none", "true", "false", "null", "return", "string", "void"}

        scored: List[Tuple[int, MethodBoundary]] = []
        for mb in boundaries:
            if mb.kind == "class":
                continue
            sig_words = set(re.findall(r'[a-zA-Z]\w{3,}', mb.signature.lower()))
            # Also check method body for keyword hits
            body = '\n'.join(lines[mb.start_line:mb.end_line + 1]).lower()
            body_hits = sum(1 for kw in keywords if kw in body)
            sig_hits = len(keywords & sig_words)
            score = sig_hits * 3 + body_hits  # signature matches weigh 3x
            if score >= 2:
                scored.append((score, mb))

        # Take top 5 matches
        scored.sort(key=lambda x: -x[0])
        for _, mb in scored[:5]:
            if mb.start_line not in target_starts:
                target_boundaries.append(mb)
                target_starts.add(mb.start_line)

    # ── Step 4: Follow call chains (2 levels deep) ────────────────────────
    chain_methods: List[MethodBoundary] = []
    if target_boundaries:
        chain_methods = _follow_call_chain(target_boundaries, boundaries, lines, max_depth=2)

    # ── Step 5: Build the extracted output ────────────────────────────────
    parts: List[str] = []

    # 5a. Always include imports + class header (capped at 120 lines for safety)
    first_callable = next((mb for mb in boundaries if mb.kind in ("method", "constructor", "function")), None)
    if first_callable is not None:
        header_end = min(first_callable.start_line, 120)
    else:
        header_end = min(boundaries[0].start_line, 80) if boundaries else min(80, total_lines)
    parts.append(f"// === IMPORTS & CLASS HEADER (lines 1-{header_end}) ===")
    parts.append('\n'.join(lines[:header_end]))

    # 5b. Structural outline (every method, just signature + line numbers)
    method_only = [mb for mb in boundaries if mb.kind != "class"]
    parts.append("")
    parts.append(f"// === FILE OUTLINE: {file_path} ({content_len:,} chars, {total_lines} lines, {len(method_only)} methods) ===")
    parts.append("//")
    for mb in boundaries:
        marker = ""
        if mb.start_line in target_starts:
            marker = "  ◀◀◀ TARGET"
        elif mb in chain_methods:
            marker = "  ◀ CALLED BY TARGET"
        parts.append(f"//   L{mb.start_line+1}-L{mb.end_line+1} ({mb.line_count():>4} lines)  [{mb.kind}]  {mb.signature}{marker}")
    parts.append("//")

    # 5c. Target methods — FULL content
    if target_boundaries:
        parts.append("")
        parts.append(f"// === TARGET METHODS ({len(target_boundaries)} methods — shown in FULL) ===")
        for mb in target_boundaries:
            parts.append(f"\n// --- L{mb.start_line+1}-L{mb.end_line+1}: {mb.name} ---")
            parts.append('\n'.join(lines[mb.start_line:mb.end_line + 1]))

    # 5d. Call chain methods — FULL content
    if chain_methods:
        parts.append("")
        parts.append(f"// === CALLED BY TARGET ({len(chain_methods)} methods — shown in FULL for context) ===")
        for mb in chain_methods:
            parts.append(f"\n// --- L{mb.start_line+1}-L{mb.end_line+1}: {mb.name} [called by target] ---")
            parts.append('\n'.join(lines[mb.start_line:mb.end_line + 1]))

    # 5e. Fallback: if nothing matched, show head + tail
    if not target_boundaries and not chain_methods:
        parts.append("")
        parts.append("// === NO SPECIFIC METHODS MATCHED — showing head + tail ===")
        head_chars = min(30000, max_output_chars // 2)
        parts.append(content[:head_chars])
        parts.append(f"\n// ... [{content_len - head_chars * 2:,} chars omitted — use outline above to navigate] ...\n")
        parts.append(content[-10000:])

    result = '\n'.join(parts)

    # Safety cap
    if len(result) > max_output_chars:
        result = result[:max_output_chars - 200] + (
            f"\n\n// [TRUNCATED at {max_output_chars:,} chars — "
            f"file is {content_len:,} total, {total_lines} lines]"
        )

    logger.info(
        f"smart_extract: {file_path} — "
        f"{content_len:,} → {len(result):,} chars "
        f"({len(result)/content_len*100:.0f}%), "
        f"{len(target_boundaries)} target methods, "
        f"{len(chain_methods)} call-chain methods"
    )

    return result


# ─── Tree-Sitter Based Boundary Parsers ──────────────────────────────────
# Grammar-correct method boundary detection. Unlike the regex+brace-counting
# parser above, tree-sitter handles nested generics, annotations with braces,
# string literals containing braces, and comments containing braces correctly.
#
# These are used by extract_exact_methods() for the EDITING role.
# The regex parser is retained as a fallback and continues to be used by
# smart_extract() for the AWARENESS role (where minor boundary errors are
# acceptable since that content is never used as an edit source).

def _parse_java_boundaries_treesitter(source: str) -> List[MethodBoundary]:
    """
    Parse Java source into method boundaries using tree-sitter.
    
    Returns grammar-correct boundaries that handle:
    - Nested classes with generic parameters (Map<String, List<Foo>>)
    - Annotations containing braces (@Query("{...}"))
    - String literals and comments containing braces
    """
    try:
        import tree_sitter
        import tree_sitter_java
    except ImportError:
        logger.warning("tree-sitter-java not available, falling back to regex parser")
        return []

    try:
        lang = tree_sitter.Language(tree_sitter_java.language())
        parser = tree_sitter.Parser()
        parser.language = lang
        tree = parser.parse(source.encode("utf-8"))
        root = tree.root_node
    except Exception as e:
        logger.warning(f"tree-sitter Java parse failed: {e}")
        return []

    boundaries: List[MethodBoundary] = []

    def _visit(node):
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf-8") if name_node else "?"
            # Use byte offsets to get accurate line numbers
            start_line = node.start_point[0]
            # For class, just record the declaration line (not full body)
            boundaries.append(MethodBoundary(
                start_line, start_line, name,
                source.split('\n')[start_line].strip()[:150], "class"
            ))
        elif node.type in ("method_declaration", "constructor_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode("utf-8")
            elif node.type == "constructor_declaration":
                name = "constructor"
            else:
                name = "?"
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            sig = source.split('\n')[start_line].strip()[:150]
            kind = "constructor" if node.type == "constructor_declaration" else "method"
            boundaries.append(MethodBoundary(start_line, end_line, name, sig, kind))

        for child in node.children:
            _visit(child)

    _visit(root)
    return boundaries


def _parse_typescript_boundaries_treesitter(source: str, file_path: str = "") -> List[MethodBoundary]:
    """
    Parse TypeScript source into method boundaries using tree-sitter.
    
    Handles decorators, generic types, template literals, and arrow functions
    that the regex parser can mis-parse.
    """
    try:
        import tree_sitter
        import tree_sitter_typescript
    except ImportError:
        logger.warning("tree-sitter-typescript not available, falling back to regex parser")
        return []

    try:
        is_tsx = file_path.endswith((".tsx", ".jsx")) or ("<" in source and hasattr(tree_sitter_typescript, "language_tsx"))
        if is_tsx and hasattr(tree_sitter_typescript, "language_tsx"):
            ts_grammar = tree_sitter_typescript.language_tsx()
        else:
            ts_grammar = tree_sitter_typescript.language_typescript()
        lang = tree_sitter.Language(ts_grammar)
        parser = tree_sitter.Parser()
        parser.language = lang
        tree = parser.parse(source.encode("utf-8"))
        root = tree.root_node
    except Exception as e:
        logger.warning(f"tree-sitter TypeScript parse failed: {e}")
        return []

    boundaries: List[MethodBoundary] = []
    lines = source.split('\n')

    def _visit(node):
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf-8") if name_node else "?"
            start_line = node.start_point[0]
            boundaries.append(MethodBoundary(
                start_line, start_line, name,
                lines[start_line].strip() if start_line < len(lines) else "", "class"
            ))
        elif node.type == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode("utf-8")
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                sig = _capture_full_signature(lines, start_line) if start_line < len(lines) else ""
                kind = "constructor" if name == "constructor" else "method"
                boundaries.append(MethodBoundary(start_line, end_line, name, sig, kind))
        elif node.type in ("public_field_definition", "property_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode("utf-8")
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                sig = _capture_full_signature(lines, start_line) if start_line < len(lines) else ""
                val_node = node.child_by_field_name("value")
                kind = "method" if (val_node and val_node.type in ("arrow_function", "function_expression")) else "property"
                boundaries.append(MethodBoundary(start_line, end_line, name, sig, kind))
        elif node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode("utf-8")
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                sig = _capture_full_signature(lines, start_line) if start_line < len(lines) else ""
                boundaries.append(MethodBoundary(start_line, end_line, name, sig, "function"))

        for child in node.children:
            _visit(child)

    _visit(root)
    return boundaries


def _get_reliable_boundaries(
    content: str,
    file_path: str,
    lines: List[str],
) -> List[MethodBoundary]:
    """
    Get method boundaries using tree-sitter (preferred) with regex fallback.
    
    Tree-sitter provides grammar-correct boundaries.
    Regex+brace-counting is the fallback for unsupported languages or failures.
    """
    lang = _detect_language(file_path)

    if lang == "java":
        boundaries = _parse_java_boundaries_treesitter(content)
        if boundaries:
            logger.info(f"  [boundaries] tree-sitter Java: {len(boundaries)} symbols")
            return boundaries
        logger.info("  [boundaries] tree-sitter Java failed, falling back to regex")

    elif lang in ("typescript", "javascript"):
        boundaries = _parse_typescript_boundaries_treesitter(content, file_path)
        if boundaries:
            logger.info(f"  [boundaries] tree-sitter TypeScript: {len(boundaries)} symbols")
            return boundaries
        logger.info("  [boundaries] tree-sitter TS failed, falling back to regex")

    # Fallback to regex+brace-counting (existing logic)
    if lang in ("java", "typescript", "javascript"):
        boundaries = _parse_java_ts_boundaries(lines)
        logger.info(f"  [boundaries] regex fallback: {len(boundaries)} symbols")
        return boundaries
    elif lang == "python":
        boundaries = _parse_python_boundaries(lines)
        logger.info(f"  [boundaries] Python indent parser: {len(boundaries)} symbols")
        return boundaries

    return []


# ─── Verbatim Method Extraction (for SEARCH/REPLACE editing) ─────────────

def extract_exact_methods(
    content: str,
    file_path: str,
    anchor_methods: List[str],
    max_output_chars: int = 55000,
) -> Tuple[str, List[MethodBoundary]]:
    """
    Extract EXACT, UNCOMPRESSED method bodies for SEARCH/REPLACE editing.
    
    Unlike smart_extract(), this returns REAL source bytes that are GUARANTEED
    to be substrings of the original file. No markers, no compression,
    no comment headers injected into the source.
    
    This is the EDITING half of the skeleton+verbatim split.
    smart_extract() handles the AWARENESS half.
    
    Uses tree-sitter for grammar-correct boundaries (Java, TypeScript).
    Falls back to regex+brace-counting if tree-sitter is unavailable.
    
    Args:
        content:         Full file content (original bytes)
        file_path:       Path to file (for language detection)
        anchor_methods:  Method names to extract (from edit_anchors + allowed_methods).
                         For MODIFY tasks: the methods being edited.
                         For INSERT tasks: the anchor neighbor methods (e.g. the method
                         AFTER which a new method will be placed).
        max_output_chars: Safety cap on total output size
    
    Returns:
        (verbatim_content, matched_boundaries)
        
        verbatim_content: The exact source text of matched methods, with small
                          non-content headers between them for the LLM to orient.
                          The method bodies themselves are verbatim substrings of `content`.
        matched_boundaries: The MethodBoundary objects that were found and extracted.
                            Empty list = anchor methods not found (caller should escalate).
    """
    lines = content.split('\n')
    total_lines = len(lines)
    content_len = len(content)

    # ── Step 1: Get reliable boundaries ───────────────────────────────────
    boundaries = _get_reliable_boundaries(content, file_path, lines)
    if not boundaries:
        logger.warning(f"extract_exact_methods: no boundaries found in {file_path}")
        return "", []

    # ── Step 2: Find target methods by name ───────────────────────────────
    target_boundaries: List[MethodBoundary] = []
    target_starts: Set[int] = set()

    for mb in boundaries:
        if mb.name in anchor_methods and mb.start_line not in target_starts:
            target_boundaries.append(mb)
            target_starts.add(mb.start_line)

    if not target_boundaries:
        logger.warning(
            f"extract_exact_methods: none of {anchor_methods} found in {file_path}. "
            f"Available methods: {[mb.name for mb in boundaries if mb.kind != 'class'][:20]}"
        )
        return "", []

    # ── Step 3: Follow call chains (same as smart_extract) ────────────────
    chain_methods = _follow_call_chain(target_boundaries, boundaries, lines, max_depth=2)

    # ── Step 4: Build output with VERBATIM method bodies ──────────────────
    # Critical: the method body text MUST be an exact substring of `content`.
    # We achieve this by joining the original `lines[start:end+1]`, which
    # reconstructs the exact bytes (minus the final newline split artifact).
    parts: List[str] = []

    # 4a. Always include imports + class header (verbatim)
    first_callable = next((mb for mb in boundaries if mb.kind in ("method", "constructor", "function")), None)
    if first_callable is not None:
        header_end = first_callable.start_line
    else:
        header_end = min(boundaries[0].start_line, 80) if boundaries else min(80, total_lines)
    parts.append('\n'.join(lines[:header_end]))

    # 4b. Target methods — VERBATIM content (exact bytes from original file)
    if target_boundaries:
        parts.append("")
        for mb in target_boundaries:
            # This comment line is NOT part of the source — it's a separator
            # for the LLM to understand which method it's looking at.
            # The actual source starts on the next line.
            parts.append(f"// ▼▼▼ Method: {mb.name} (L{mb.start_line+1}-L{mb.end_line+1}) — VERBATIM SOURCE ▼▼▼")
            parts.append('\n'.join(lines[mb.start_line:mb.end_line + 1]))
            parts.append(f"// ▲▲▲ End: {mb.name} ▲▲▲")

    # 4c. Call chain methods — VERBATIM content
    if chain_methods:
        parts.append("")
        for mb in chain_methods:
            parts.append(f"// ▼▼▼ Called by target: {mb.name} (L{mb.start_line+1}-L{mb.end_line+1}) — VERBATIM SOURCE ▼▼▼")
            parts.append('\n'.join(lines[mb.start_line:mb.end_line + 1]))
            parts.append(f"// ▲▲▲ End: {mb.name} ▲▲▲")

    result = '\n'.join(parts)

    # Safety cap
    if len(result) > max_output_chars:
        result = result[:max_output_chars - 200] + (
            f"\n\n// [TRUNCATED at {max_output_chars:,} chars — "
            f"file is {content_len:,} total, {total_lines} lines]"
        )

    logger.info(
        f"extract_exact_methods: {file_path} — "
        f"{content_len:,} → {len(result):,} chars "
        f"({len(result)/content_len*100:.0f}%), "
        f"{len(target_boundaries)} target methods, "
        f"{len(chain_methods)} call-chain methods"
    )

    return result, target_boundaries


# ─── Public Tree-sitter API (for reuse by code_generator.py) ─────────────
# These functions provide FRESH tree-sitter parsing at patch-apply time.
# We cannot reuse the indexed AST from project-add time because files change
# task-to-task during a run. Like Cursor, we parse the CURRENT file content
# on the fly at the moment we need it.

def detect_language(file_path: str) -> str:
    """Public wrapper for language detection. Reuse across modules."""
    return _detect_language(file_path)


def parse_with_treesitter(source: str, file_path: str):
    """
    Parse source code with tree-sitter and return the tree + language name.

    Returns (tree, language_name) or (None, language_name) if tree-sitter
    is unavailable or parsing fails. Callers should handle None gracefully.

    This is the FRESH parse that runs on the CURRENT file content at
    patch-apply time — not the stale indexed AST from project-add time.
    """
    lang = _detect_language(file_path)

    if lang == "java":
        try:
            import tree_sitter
            import tree_sitter_java
            ts_lang = tree_sitter.Language(tree_sitter_java.language())
            parser = tree_sitter.Parser()
            parser.language = ts_lang
            tree = parser.parse(source.encode("utf-8"))
            return tree, lang
        except (ImportError, Exception) as e:
            logger.debug(f"tree-sitter Java parse failed: {e}")
            return None, lang

    elif lang in ("typescript", "javascript"):
        try:
            import tree_sitter
            import tree_sitter_typescript
            ext = Path(file_path).suffix.lower()
            if ext in (".tsx", ".jsx") and hasattr(tree_sitter_typescript, "language_tsx"):
                ts_lang = tree_sitter.Language(tree_sitter_typescript.language_tsx())
            else:
                ts_lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
            parser = tree_sitter.Parser()
            parser.language = ts_lang
            tree = parser.parse(source.encode("utf-8"))
            return tree, lang
        except (ImportError, Exception) as e:
            logger.debug(f"tree-sitter TS parse failed: {e}")
            return None, lang

    elif lang == "python":
        try:
            import tree_sitter
            import tree_sitter_python
            ts_lang = tree_sitter.Language(tree_sitter_python.language())
            parser = tree_sitter.Parser()
            parser.language = ts_lang
            tree = parser.parse(source.encode("utf-8"))
            return tree, lang
        except (ImportError, Exception) as e:
            logger.debug(f"tree-sitter Python parse failed: {e}")
            return None, lang

    return None, lang


def count_ast_nodes(tree, node_types: list[str]) -> int:
    """Count the number of AST nodes of specific types in a tree-sitter tree."""
    if tree is None:
        return -1  # Sentinel: tree-sitter unavailable
    count = 0
    def _walk(node):
        nonlocal count
        if node.type in node_types:
            count += 1
        for child in node.children:
            _walk(child)
    _walk(tree.root_node)
    return count


def has_ast_errors(tree) -> bool:
    """Check if a tree-sitter tree contains any ERROR or MISSING nodes."""
    if tree is None:
        return False  # Can't check — assume OK
    def _walk(node):
        if node.type == "ERROR" or node.is_missing:
            return True
        for child in node.children:
            if _walk(child):
                return True
        return False
    return _walk(tree.root_node)


def get_ast_error_ranges(tree) -> list[tuple[int, int]]:
    """Return (start_line, end_line) pairs for all ERROR nodes in the tree."""
    if tree is None:
        return []
    errors = []
    def _walk(node):
        if node.type == "ERROR" or node.is_missing:
            errors.append((node.start_point[0] + 1, node.end_point[0] + 1))
        for child in node.children:
            _walk(child)
    _walk(tree.root_node)
    return errors


def extract_concrete_symbol_signatures(file_paths: list[str], repo_root: Optional[Any] = None) -> dict[str, Any]:
    """
    Extracts concrete symbol signatures (classes, method signatures, exception constructors,
    and database columns) from the given files, formatting them as strict ground-truth constraints.
    """
    from pathlib import Path
    root = Path(repo_root) if repo_root else Path.cwd()
    results = {
        "classes": [],
        "methods": [],
        "constructors": [],
        "enums": {},
        "database_tables": {},
        "summary_constraints": []
    }

    for fp in file_paths:
        p = (root / fp.lstrip("/\\")).resolve() if not Path(fp).is_absolute() else Path(fp)
        if not p.is_file():
            # Try glob
            m = list(root.glob(f"**/{Path(fp).name}"))
            if m:
                p = m[0]
            else:
                continue

        content = p.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        fname = p.name

        # SQL DDL parsing
        if p.suffix.lower() == ".sql":
            table_matches = re.finditer(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_.]+)\s*\((.*?)\);', content, re.DOTALL | re.IGNORECASE)
            for tm in table_matches:
                tname = tm.group(1).split(".")[-1].strip('"')
                col_defs = tm.group(2)
                cols = []
                for col_line in col_defs.splitlines():
                    col_line = col_line.strip().rstrip(",")
                    if not col_line or col_line.upper().startswith(("PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE", "CHECK")):
                        continue
                    col_parts = col_line.split()
                    if col_parts:
                        cname = col_parts[0].strip('"')
                        ctype = col_parts[1] if len(col_parts) > 1 else "UNKNOWN"
                        cols.append(f"{cname} ({ctype})")
                results["database_tables"][tname] = cols
                results["summary_constraints"].append(f"DB Table `{tname}` ({fname}): columns [{', '.join(cols[:15])}]")
            continue

        # Java / TypeScript / Python AST boundary parsing
        lang = _detect_language(str(p))
        boundaries = []
        if lang in ("java", "typescript", "javascript"):
            boundaries = _parse_java_ts_boundaries(lines)
        elif lang == "python":
            boundaries = _parse_python_boundaries(lines)

        for b in boundaries:
            sig = b.signature
            if b.kind in ("class", "interface", "record"):
                results["classes"].append(f"{b.kind} {b.name} in `{fname}`")
                results["summary_constraints"].append(f"Verified Type: `{b.kind} {b.name}` in `{fname}`")
            elif b.kind == "enum":
                # Extract enum values
                enum_vals = []
                enum_body = lines[b.start_line:b.end_line + 1]
                for eline in enum_body:
                    clean_el = eline.split("//")[0].strip()
                    for val_match in re.findall(r'\b[A-Z][A-Z0-9_]+\b', clean_el):
                        if val_match not in ("PUBLIC", "ENUM", "CLASS", "STATIC", "FINAL", "DEFAULT"):
                            enum_vals.append(val_match)
                results["enums"][b.name] = enum_vals
                results["summary_constraints"].append(f"Enum `{b.name}` ({fname}) values: [{', '.join(enum_vals)}]")
            elif b.kind in ("constructor",):
                results["constructors"].append(f"`{sig}` in `{fname}`")
                results["summary_constraints"].append(f"Constructor Contract: `{sig}` in `{fname}`")
            elif b.kind in ("method", "function"):
                results["methods"].append(f"`{sig}` in `{fname}` (L{b.start_line + 1}-L{b.end_line + 1})")

    return results



