"""
Ticket Fact Extractor

Pure Python, zero LLM calls.
Deterministically extracts structured technical facts from any ticket before
the LLM investigation step, so the LLM cannot ignore version numbers, file
names, class names, or error codes buried in "Technical Context" sections.

Author: Deepak Madgani
Date: June 2026
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import PurePosixPath

from ticket_to_code.models import TechnicalFacts, VersionFact, FileFact, ValueEdgeTicket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section header keywords — used by the Section Splitter
# ---------------------------------------------------------------------------
_SECTION_PATTERNS = [
    ("technical_context",   re.compile(r"(?i)technical\s*context|tech\s*context|implementation\s*note")),
    ("files",               re.compile(r"(?i)files?\s*(to\s*modify)?:|affected\s*files?|changed?\s*files?")),
    ("steps",               re.compile(r"(?i)steps?\s*(to\s*reproduce|to\s*fix)?:")),
    ("logs",                re.compile(r"(?i)log\s*(output)?:|stack\s*trace:|error\s*log:")),
    ("acceptance_criteria", re.compile(r"(?i)acceptance\s*criteria:|expected\s*behavior:")),
    ("current_behavior",    re.compile(r"(?i)current\s*behavior:|observed\s*behavior:")),
    ("environment",         re.compile(r"(?i)environment:|version\s*info:")),
    ("description",         re.compile(r"(?i)description:|issue:|problem:")),
]

# ---------------------------------------------------------------------------
# Regex patterns for fact extraction
# ---------------------------------------------------------------------------

# Version: 6-digit (e.g., 260200), semantic (e.g., 26.2.0, 26.2), v-prefixed, underscore
_VERSION_INT      = re.compile(r"\b(\d{6})\b")
_VERSION_DOTTED   = re.compile(r"\b(\d{1,3}\.\d{1,3}(?:\.\d{1,5})?)\b")
_VERSION_PREFIXED = re.compile(r"\b[vV](\d[\d.]+)\b")

# Version transition: "from X to Y", "X -> Y", "X → Y", "X ➔ Y"
_VERSION_TRANS = re.compile(
    r"(?:from\s+|update\s+to\s+)?(?P<old>\d[\d._v]+)\s*(?:->|→|➔|to)\s*(?P<new>\d[\d._v]+)"
)

# File names with common code/config/script extensions
_FILE_PATTERN = re.compile(
    r"(?<!\w)([\w\-./]+\.(?:"
    r"ts|tsx|js|jsx|mjs|vue|svelte|"        # Frontend
    r"java|kt|scala|groovy|"               # JVM
    r"py|rb|go|rs|cs|cpp|c|h|swift|"      # Other langs
    r"sh|bash|ps1|bat|cmd|"               # Scripts
    r"json|yaml|yml|xml|toml|ini|conf|properties|env|"  # Config
    r"sql|"                                # Database
    r"md|txt|rst|"                         # Docs
    r"scss|css|less|sass|html|htm"         # UI
    r"))(?!\w)",
    re.IGNORECASE
)

# UI component names before common suffixes (e.g. "deliverable add reviewer page", "reviewers tab")
_UI_PATTERN = re.compile(r"\b([a-zA-Z0-9\-]+(?:\s+[a-zA-Z0-9\-]+){0,2})\s+(page|component|dialog|modal|tab|button|screen|view)\b", re.IGNORECASE)

# High-signal UI workflow/state literals used for deterministic repository search.
_QUOTED_LITERAL_PATTERN = re.compile(r"['\"]([^'\"]{3,80})['\"]")
_UI_CONTROL_LITERAL_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\s\-/]{1,50})\s+"
    r"(tab|button|icon|column|label|dialog|modal|view|screen)\b",
    re.IGNORECASE,
)
_OBSERVATION_LITERAL_PATTERN = re.compile(
    r"\b(?:observe|observed|expected|expectation\s+is)\s+([^\.\n]{5,120})",
    re.IGNORECASE,
)
_STATE_ASSERTION_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\s\-/]{1,50}\s+button\s+(?:is\s+)?(?:enabled|disabled))\b",
    re.IGNORECASE,
)

# PascalCase class / component names (at least 2 capital-letter segments, ≥5 chars)
_CLASS_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")

# camelCase method / variable names (starts lowercase, has at least one capital)
_METHOD_PATTERN = re.compile(r"\b([a-z][a-z0-9]+[A-Z][a-zA-Z0-9]+)\b")

# Config keys: dotted lowercase (e.g., application.version, spring.datasource.url)
_CONFIG_KEY = re.compile(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,5})\b")

# Paths: relative paths that look like source directories
_PATH_PATTERN = re.compile(r"(?<!\w)((?:src|com|org|net|io|main|test|app|lib|pkg)/[\w./\-]+)(?!\w)")

# Error codes: uppercase letters + digits (e.g., IDX10000, HTTP404, NRE-1234)
_ERROR_CODE = re.compile(r"\b([A-Z]{2,}\d{4,}|[A-Z]{2,}-\d{3,})\b")

# Java/C# exception names
_EXCEPTION = re.compile(r"\b(\w+(?:Exception|Error|Fault|Warning))\b")

# URLs
_URL_PATTERN = re.compile(r"https?://[^\s\"'>]+")

# SQL keywords (detect SQL context)
_SQL_PATTERN = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE)\b", re.IGNORECASE)

# Security keywords
_SECURITY_KEYWORDS = re.compile(
    r"\b(auth|authentication|authorization|token|jwt|oauth|ssl|tls|certificate|"
    r"password|secret|api.key|xss|csrf|injection|vulnerability|cve)\b",
    re.IGNORECASE
)

# Dependency keywords
_DEPENDENCY_KEYWORDS = re.compile(
    r"\b(upgrade|downgrade|bump\s+version|dependency|package|library|"
    r"maven|gradle|npm|pip|nuget|gem)\b",
    re.IGNORECASE
)


_DELETE_OPERATION = re.compile(r"\b(delete|remove|drop)\b", re.IGNORECASE)
_CREATE_OPERATION = re.compile(r"\b(create|add|new)\b", re.IGNORECASE)
_MODIFY_OPERATION = re.compile(
    r"\b(update|change|modify|edit|rename|replace|align|set|patch|write)\b",
    re.IGNORECASE,
)


def _generate_version_permutations(raw: str) -> List[str]:
    """
    Given a raw version string, generate all codebase forms.

    Examples:
        "260200"  → ["260200", "26.2.0", "26.2", "26_2", "26_2_0", "v26.2.0", "v26.2"]
        "26.3"    → ["26.3", "260300", "26_3", "v26.3"]
        "26.3.0"  → ["26.3.0", "260300", "26_3_0", "26_3", "v26.3.0"]
    """
    raw = raw.strip().lstrip("vV")
    results = set([raw])

    # 6-digit integer → dotted form
    m = re.fullmatch(r"(\d{2})(\d{2})(\d{2})", raw)
    if m:
        major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        results.update([
            f"{major}.{minor}.{patch}",
            f"{major}.{minor}",
            f"{major}_{minor}_{patch}",
            f"{major}_{minor}",
            f"v{major}.{minor}.{patch}",
            f"v{major}.{minor}",
        ])
        return sorted(results)

    # Dotted form → integer and underscore forms
    m = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", raw)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        patch = int(m.group(3)) if m.group(3) else 0
        results.update([
            f"{major:02d}{minor:02d}{patch:02d}",   # 6-digit
            f"{major}_{minor}_{patch}",
            f"{major}_{minor}",
            f"v{raw}",
            f"{major}.{minor}",                     # without patch
        ])
        return sorted(results)

    return sorted(results)


class TicketSectionSplitter:
    """
    Splits a ticket description into labeled sections.

    Recognizes common enterprise ticket patterns:
        Description / Technical Context / Files / Steps / Logs / Environment / Acceptance Criteria
    """

    def split(self, ticket: ValueEdgeTicket) -> Dict[str, str]:
        """Return a mapping of section_name → section_text."""
        full_text = f"{ticket.title}\n{ticket.description}"
        if ticket.acceptance_criteria:
            full_text += "\n" + "\n".join(ticket.acceptance_criteria)

        # Find all section header positions
        sections: Dict[str, str] = {}
        boundaries: List[Tuple[int, str]] = []

        for name, pattern in _SECTION_PATTERNS:
            for m in pattern.finditer(full_text):
                boundaries.append((m.start(), name))

        # Sort by position
        boundaries.sort(key=lambda x: x[0])

        if not boundaries:
            sections["description"] = full_text
            return sections

        # Extract text between consecutive boundaries
        for i, (start, name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            section_text = full_text[start:end].strip()
            # Concatenate if section appears multiple times
            if name in sections:
                sections[name] += "\n" + section_text
            else:
                sections[name] = section_text

        # Anything before the first section = description
        first_pos = boundaries[0][0]
        if first_pos > 0:
            leading = full_text[:first_pos].strip()
            if leading:
                sections["description"] = leading + "\n" + sections.get("description", "")

        return sections


class KnowledgeExtractor:
    """
    Deterministically extracts all technical facts from any input.

    Zero LLM calls. Runs in milliseconds.
    Input-agnostic: works for Jira tickets, chat, PDFs, logs, build output.
    Outputs a TechnicalFacts object that is injected into every downstream LLM
    prompt so critical facts cannot be missed.
    """

    def __init__(self):
        self.splitter = TicketSectionSplitter()

    def extract(self, ticket: ValueEdgeTicket) -> TechnicalFacts:
        """
        Main entry point. Extract all technical facts from a ticket.

        Returns:
            TechnicalFacts with all extracted facts and a classified ticket type.
        """
        logger.info(f"[FactExtractor] Extracting facts from ticket: {ticket.ticket_id}")

        # Full text — concatenate all sections for fact extraction
        full_text = f"{ticket.title}\n{ticket.description}"
        if ticket.acceptance_criteria:
            full_text += "\n" + "\n".join(ticket.acceptance_criteria)
        if ticket.labels:
            full_text += "\n" + " ".join(ticket.labels)

        # Split into sections — technical_context gets higher priority
        section_map = self.splitter.split(ticket)

        # Gather high-priority text (technical_context + files sections first)
        priority_text = (
            section_map.get("technical_context", "") + "\n" +
            section_map.get("files", "") + "\n" +
            section_map.get("environment", "")
        )

        # === Extract facts ===
        versions     = self._extract_versions(full_text, priority_text)
        files        = self._extract_files(full_text, priority_text)
        
        # UI components extracted as filename hints
        ui_comps = self._extract_ui_components(full_text)
        for comp in ui_comps:
            files.append(FileFact(name=comp, path=None, operation="modify"))
            
        classes      = self._extract_classes(full_text)
        methods      = self._extract_methods(full_text)
        config_keys  = self._extract_config_keys(full_text)
        paths        = self._extract_paths(full_text)
        error_codes  = self._extract_error_codes(full_text)
        urls         = _URL_PATTERN.findall(full_text)
        sql_snippets = _SQL_PATTERN.findall(full_text)
        api_routes   = self._extract_api_routes(full_text)
        state_literals = self._extract_state_literals(full_text)

        # Classify ticket type deterministically
        classified_type = TicketTypeClassifier.classify(
            full_text=full_text,
            versions=versions,
            files=files,
            error_codes=error_codes,
            sql_snippets=sql_snippets,
        )

        # Build the flattened high-priority literals list for query expansion
        high_priority = []
        high_priority.extend(state_literals)
        for v in versions:
            high_priority.extend(v.permutations_old)
        for f in files:
            high_priority.append(f.name)
            if f.path:
                high_priority.append(f.path)
        high_priority.extend(ui_comps)
        high_priority.extend(classes[:10])
        high_priority.extend(methods[:10])
        high_priority.extend(config_keys[:5])
        high_priority.extend(error_codes[:5])
        high_priority.extend(api_routes[:5])

        # Deduplicate while preserving order
        seen = set()
        unique_priority = []
        for item in high_priority:
            if item and item not in seen:
                seen.add(item)
                unique_priority.append(item)

        facts = TechnicalFacts(
            versions=versions,
            files=files,
            classes=classes,
            methods=methods,
            config_keys=config_keys,
            paths=paths,
            error_codes=error_codes,
            urls=urls[:10],
            sql_snippets=[s[0] if isinstance(s, tuple) else s for s in sql_snippets[:5]],
            classified_type=classified_type,
            section_map={k: v[:500] for k, v in section_map.items()},
            high_priority_literals=unique_priority,
        )

        logger.info(
            f"[FactExtractor] Ticket {ticket.ticket_id}: "
            f"type={classified_type}, "
            f"versions={len(versions)}, files={len(files)}, "
            f"classes={len(classes)}, methods={len(methods)}, "
            f"literals={len(unique_priority)}"
        )
        return facts

    # ------------------------------------------------------------------
    # Private extraction methods
    # ------------------------------------------------------------------

    def _context_window(self, text: str, start: int, end: int) -> str:
        """Return the containing line or sentence around a file match."""
        left_line = text.rfind("\n", 0, start)
        right_line = text.find("\n", end)
        if left_line == -1:
            left_line = 0
        else:
            left_line += 1
        if right_line == -1:
            right_line = len(text)

        line = text[left_line:right_line].strip()
        if len(line) >= 12:
            return line

        left_sent = max(text.rfind(".", 0, start), text.rfind(":", 0, start), text.rfind(";", 0, start))
        right_candidates = [p for p in (text.find(".", end), text.find("\n", end), text.find(";", end)) if p != -1]
        right_sent = min(right_candidates) if right_candidates else len(text)
        if left_sent == -1:
            left_sent = 0
        else:
            left_sent += 1
        return text[left_sent:right_sent].strip()

    def _infer_file_operation(self, context: str) -> str:
        """Infer create/modify/delete from nearby ticket context."""
        if not context:
            return "modify"
        if _DELETE_OPERATION.search(context):
            return "delete"
        if _CREATE_OPERATION.search(context):
            return "create"
        if _MODIFY_OPERATION.search(context):
            return "modify"
        return "modify"

    def _extract_versions(self, full_text: str, priority_text: str) -> List[VersionFact]:
        """Extract version transitions and standalone version strings."""
        facts: List[VersionFact] = []
        seen_pairs: set = set()

        # Priority: explicit transitions first ("260200 -> 260300")
        for m in _VERSION_TRANS.finditer(full_text):
            old_raw = m.group("old").strip()
            new_raw = m.group("new").strip()
            pair_key = (old_raw, new_raw)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                facts.append(VersionFact(
                    old_value=old_raw,
                    new_value=new_raw,
                    permutations_old=_generate_version_permutations(old_raw),
                    permutations_new=_generate_version_permutations(new_raw),
                ))

        # Standalone 6-digit versions from priority text
        if not facts:
            int_versions = _VERSION_INT.findall(priority_text) or _VERSION_INT.findall(full_text)
            if len(int_versions) >= 2:
                old_raw, new_raw = int_versions[0], int_versions[1]
                pair_key = (old_raw, new_raw)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    facts.append(VersionFact(
                        old_value=old_raw,
                        new_value=new_raw,
                        permutations_old=_generate_version_permutations(old_raw),
                        permutations_new=_generate_version_permutations(new_raw),
                    ))

        # Standalone dotted versions from priority text (e.g., "26.2")
        if not facts:
            dotted = _VERSION_DOTTED.findall(priority_text) or _VERSION_DOTTED.findall(full_text)
            # Deduplicate while preserving first-occurrence order
            unique_dotted = list(dict.fromkeys(dotted))
            if len(unique_dotted) >= 2:
                # Use contextual clues to determine which is old vs new
                # "currently displayed as X" / "is X" → old
                # "update to Y" / "needs to be Y" → new
                _old_ctx = re.compile(
                    r"(?:currently|displayed\s+as|is\s+currently|shows?|was)\s+"
                    + re.escape(unique_dotted[0]),
                    re.IGNORECASE,
                )
                _new_ctx = re.compile(
                    r"(?:update(?:d)?\s+to|change(?:d)?\s+to|needs?\s+to\s+be|set\s+to)\s+"
                    + re.escape(unique_dotted[0]),
                    re.IGNORECASE,
                )
                # If the first unique version appears in a "new" context and
                # the second in an "old" context, swap them.
                first_is_new = bool(_new_ctx.search(full_text))
                first_is_old = bool(_old_ctx.search(full_text))

                if first_is_new and not first_is_old:
                    old_raw, new_raw = unique_dotted[1], unique_dotted[0]
                else:
                    old_raw, new_raw = unique_dotted[0], unique_dotted[1]

                pair_key = (old_raw, new_raw)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    facts.append(VersionFact(
                        old_value=old_raw,
                        new_value=new_raw,
                        permutations_old=_generate_version_permutations(old_raw),
                        permutations_new=_generate_version_permutations(new_raw),
                    ))

        return facts

    def _extract_files(self, full_text: str, priority_text: str) -> List[FileFact]:
        """Extract explicit file names/paths from ticket text."""
        found: Dict[str, FileFact] = {}

        # Check priority text first (technical_context, files sections)
        for text in (priority_text, full_text):
            for m in _FILE_PATTERN.finditer(text):
                raw = m.group(1)
                name = PurePosixPath(raw).name
                if len(name) < 3 or name.startswith("."):
                    continue
                if name not in found:
                    # Infer operation from the containing line/sentence in the same source text.
                    ctx = self._context_window(text, m.start(), m.end())
                    op = self._infer_file_operation(ctx)

                    path_val = raw if "/" in raw or "\\" in raw else None
                    found[name] = FileFact(name=name, path=path_val, operation=op)

        return list(found.values())

    def _extract_ui_components(self, text: str) -> List[str]:
        """Extract UI component names (e.g. 'deliverable add reviewer' from 'deliverable add reviewer page')."""
        components = []
        for match in _UI_PATTERN.finditer(text):
            comp_name = match.group(1).strip().lower()
            # Replace spaces with hyphens to match common frontend file naming conventions
            comp_name_kebab = comp_name.replace(' ', '-')
            if comp_name_kebab not in components:
                components.append(comp_name_kebab)
        return components[:10]

    def _extract_state_literals(self, text: str) -> List[str]:
        """
        Extract deterministic, high-signal UI workflow/state literals.

        These literals are critical for tickets describing multi-step behavior
        regressions across tabs/pages (e.g., "Upload button is disabled").
        """
        candidates: List[str] = []

        # 1) Quoted literals (button labels, tab names, menu items)
        for m in _QUOTED_LITERAL_PATTERN.finditer(text):
            lit = m.group(1).strip()
            if 3 <= len(lit) <= 80:
                candidates.append(lit)

        # 2) UI controls with semantic suffix
        for m in _UI_CONTROL_LITERAL_PATTERN.finditer(text):
            prefix = m.group(1).strip()
            suffix = m.group(2).strip().lower()
            lit = f"{prefix} {suffix}".strip()
            candidates.append(lit)

        # 3) Observation/expectation lines (what is currently wrong)
        for m in _OBSERVATION_LITERAL_PATTERN.finditer(text):
            lit = m.group(1).strip()
            if 5 <= len(lit) <= 120:
                candidates.append(lit)

        # 4) Explicit enabled/disabled assertions (very high value)
        for m in _STATE_ASSERTION_PATTERN.finditer(text):
            candidates.append(m.group(1).strip())

        # Normalize + dedupe + remove generic fragments
        generic_fragments = {
            "the application",
            "click on",
            "select button",
            "save button",
            "cancel button",
            "close icon",
        }
        out: List[str] = []
        seen = set()
        for raw in candidates:
            lit = re.sub(r"\s+", " ", raw).strip(" .:-\n\t")
            low = lit.lower()
            if len(lit) < 4:
                continue
            if low in generic_fragments:
                continue
            if low not in seen:
                seen.add(low)
                out.append(lit)

        # Keep deterministic cap for prompt/search budget
        return out[:20]

    def _extract_classes(self, text: str) -> List[str]:
        """Extract PascalCase class/component names."""
        matches = _CLASS_PATTERN.findall(text)
        # Filter out common English words that happen to be PascalCase
        _ENGLISH_STOPWORDS = {
            "Login", "Password", "Button", "Cancel", "Submit", "Dialog",
            "Click", "Select", "Navigate", "Navigate", "Review", "Update",
            "Current", "Expected", "Technical", "Context", "Business",
            "Impact", "Criteria", "Acceptance", "Steps", "Behavior",
        }
        return [m for m in dict.fromkeys(matches) if m not in _ENGLISH_STOPWORDS][:15]

    def _extract_methods(self, text: str) -> List[str]:
        """Extract camelCase method/variable names."""
        matches = _METHOD_PATTERN.findall(text)
        # Filter out very short or common words
        return [m for m in dict.fromkeys(matches) if len(m) >= 5][:15]

    def _extract_config_keys(self, text: str) -> List[str]:
        """Extract dotted config keys like 'application.version'."""
        matches = _CONFIG_KEY.findall(text)
        # Filter: must have at least 2 segments, no HTTP domains
        filtered = [
            m for m in dict.fromkeys(matches)
            if "." in m
            and not m.startswith("http")
            and not m.endswith(".com")
            and not m.endswith(".net")
            and len(m) >= 6
        ]
        return filtered[:10]

    def _extract_api_routes(self, text: str) -> List[str]:
        """Extract API routes like /api/v1/users."""
        pattern = re.compile(r'(?:/[a-zA-Z0-9_\-]+)+')
        routes = []
        for match in pattern.findall(text):
            if len(match) > 3:
                routes.append(match)
        return list(dict.fromkeys(routes))

    def _extract_paths(self, text: str) -> List[str]:
        """Extract relative source paths like 'src/app'."""
        matches = _PATH_PATTERN.findall(text)
        return list(dict.fromkeys(matches))[:10]

    def _extract_error_codes(self, text: str) -> List[str]:
        """Extract error codes and exception names."""
        codes = _ERROR_CODE.findall(text)
        exceptions = _EXCEPTION.findall(text)
        all_errors = list(dict.fromkeys(codes + exceptions))
        return all_errors[:10]


class TicketTypeClassifier:
    """
    Pure-Python, deterministic ticket type classifier.

    Uses extracted TechnicalFacts to classify the ticket. No LLM needed.
    This classification is then passed to the Planner as a penalty map.
    """

    @staticmethod
    def classify(
        full_text: str,
        versions: List[VersionFact],
        files: List[FileFact],
        error_codes: List[str],
        sql_snippets: List[str],
    ) -> str:
        """
        Return one of: VERSION_BUMP | UI | BUG | CONFIG | API | DATABASE |
                       SECURITY | DEPENDENCY | FEATURE
        Rules are applied in priority order.
        """
        text_lower = full_text.lower()

        # Rule 1: Explicit version transition → always VERSION_BUMP
        if versions:
            return "VERSION_BUMP"

        # Rule 2: Stack trace / error code → BUG
        if error_codes:
            return "BUG"

        # Rule 3: SQL patterns → DATABASE
        if sql_snippets:
            return "DATABASE"

        # Rule 4: Security keywords → SECURITY
        if _SECURITY_KEYWORDS.search(full_text):
            return "SECURITY"

        # Rule 5: Dependency keywords → DEPENDENCY
        if _DEPENDENCY_KEYWORDS.search(full_text):
            return "DEPENDENCY"

        # Rule 6: API/endpoint patterns → API
        api_patterns = re.compile(r"\b(endpoint|api|rest|graphql|grpc|swagger|openapi|http\s+\d{3})\b", re.IGNORECASE)
        if api_patterns.search(full_text):
            return "API"

        # Rule 7: Config file types and no code → CONFIG
        config_files = [f for f in files if f.name.endswith((".yml", ".yaml", ".json", ".properties", ".conf", ".ini", ".env"))]
        code_files   = [f for f in files if not f.name.endswith((".yml", ".yaml", ".json", ".properties", ".conf", ".ini", ".env", ".md"))]
        if config_files and not code_files:
            return "CONFIG"

        # Rule 8: Pure UI — CSS/SCSS/HTML keywords
        ui_keywords = re.compile(r"\b(css|scss|layout|alignment|superimposed|padding|margin|responsive|flex|grid|ui|ux|button|modal|dialog|overlay)\b", re.IGNORECASE)
        logic_bug_keywords = re.compile(r"\b(incorrect|wrong|count|displays|calculate|value|fails)\b", re.IGNORECASE)
        
        if ui_keywords.search(full_text) and not versions and not error_codes:
            if logic_bug_keywords.search(full_text):
                return "BUG"
            return "UI_STYLING"

        # Rule 9: General logic bug without stack trace
        if logic_bug_keywords.search(full_text) and not versions:
            return "BUG"

        # Default → FEATURE
        return "FEATURE"
