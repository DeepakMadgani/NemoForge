"""
Companion and Path Resolver Utility for Ticket-to-Code Pipeline.

Implements industry-standard companion file discovery (used in modern AI IDEs like Cursor,
Devin, and Claude Code) and robust path canonicalization across microservice boundaries.
"""

import os
import re
from typing import List, Optional, Set


def normalize_path(path: str) -> str:
    """Normalize path separators and remove surrounding whitespace/slashes."""
    if not path:
        return ""
    p = str(path).replace("\\", "/").strip()
    return p.strip("/")


def _suffix_overlap(diag_segments: List[str], cand_segments: List[str]) -> int:
    """Count right-to-left contiguous segment overlap."""
    min_len = min(len(diag_segments), len(cand_segments))
    overlap = 0
    for i in range(1, min_len + 1):
        if diag_segments[-i] == cand_segments[-i]:
            overlap += 1
        else:
            break
    return overlap


def _forward_overlap(diag_segments: List[str], cand_segments: List[str]) -> int:
    """Best left-to-right contiguous overlap of diagnostic segments in candidate."""
    if not diag_segments or not cand_segments:
        return 0
    best = 0
    for idx, seg in enumerate(cand_segments):
        if seg != diag_segments[0]:
            continue
        overlap = 0
        while (
            overlap < len(diag_segments)
            and idx + overlap < len(cand_segments)
            and cand_segments[idx + overlap] == diag_segments[overlap]
        ):
            overlap += 1
        if overlap > best:
            best = overlap
    return best


def _candidate_score(diag_segments: List[str], cand_segments: List[str]) -> tuple:
    """Return a sortable score tuple for candidate ranking."""
    suffix = _suffix_overlap(diag_segments, cand_segments)
    forward = _forward_overlap(diag_segments, cand_segments)
    min_len = min(len(diag_segments), len(cand_segments))
    full_tail = int(suffix == min_len)
    full_forward = int(forward == len(diag_segments))
    same_len_bonus = -abs(len(cand_segments) - len(diag_segments))
    return (full_forward, full_tail, suffix, forward, same_len_bonus)


def resolve_candidate_paths(diag_path: str, candidates: List[str], max_results: int = 3) -> List[str]:
    """
    Resolve a diagnostic path to a ranked list of plausible workspace paths.

    This is generic and ambiguity-safe: if multiple files are plausible, returns a
    short ranked list instead of forcing a single potentially wrong match.
    """
    if not diag_path or not candidates:
        return []

    norm_diag = normalize_path(diag_path).lower()
    diag_segments = [s for s in norm_diag.split("/") if s]

    exact = [c for c in candidates if normalize_path(c).lower() == norm_diag]
    if exact:
        return exact[:1]

    scored = []
    for cand in candidates:
        norm_cand = normalize_path(cand).lower()
        cand_segments = [s for s in norm_cand.split("/") if s]
        score = _candidate_score(diag_segments, cand_segments)

        # Accept only if there is strong evidence of containment.
        suffix, forward = score[2], score[3]
        min_len = min(len(diag_segments), len(cand_segments))
        strong_tail = suffix == min_len and min_len >= 2
        strong_forward = forward == len(diag_segments) and len(diag_segments) >= 2
        if strong_tail or strong_forward:
            scored.append((score, cand))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    top_score = scored[0][0]
    tied = [cand for score, cand in scored if score == top_score]

    def _module_root(path: str) -> str:
        n = normalize_path(path)
        parts = [p for p in n.split("/") if p]
        return parts[0].lower() if parts else ""

    # Preserve determinism for the top tier and limit fanout.
    tied_sorted = sorted(tied, key=lambda c: (len(normalize_path(c)), normalize_path(c)))

    # Safety/recall balance used by modern AI IDEs:
    # when top-ranked matches are tied across multiple module roots,
    # keep those ties in context so downstream code-fix can disambiguate,
    # while write scope remains enforced by planner checks.
    module_roots = {_module_root(c) for c in tied_sorted}
    if len(module_roots) > 1:
        return tied_sorted[: max(max_results, 8)]

    return tied_sorted[:max_results]


def resolve_canonical_path(diag_path: str, candidates: List[str]) -> Optional[str]:
    """
    Matches a compiler diagnostic path against a list of known candidates (from workspace or state).
    
    Uses segment-level right-to-left subpath matching to resolve workspace prefixes
    (e.g., 'src/app/foo.ts' matching 'xchange-ui/src/app/foo.ts').
    
    If multiple candidates match, selects the candidate with the highest segment overlap.
    """
    ranked = resolve_candidate_paths(diag_path, candidates, max_results=1)
    return ranked[0] if ranked else None


def get_companion_files(
    file_path: str,
    known_files: Optional[List[str]] = None,
    file_content: Optional[str] = None
) -> List[str]:
    """
    Discovers companion/sister files for a given source file based on framework conventions,
    AST decorator links (e.g. Angular templateUrl/styleUrls), and import relationships.
    """
    if not file_path:
        return []

    norm_path = normalize_path(file_path)
    base_no_ext, ext = os.path.splitext(norm_path)
    companions: Set[str] = set()

    # ── 1. Framework Sister File Mapping (Angular / Web / Native) ────────────
    if ext == ".html":
        # Companion TypeScript / Script / Style files
        companions.add(f"{base_no_ext}.ts")
        companions.add(f"{base_no_ext}.tsx")
        companions.add(f"{base_no_ext}.scss")
        companions.add(f"{base_no_ext}.css")
        companions.add(f"{base_no_ext}.spec.ts")
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        # Companion Template / Style / Spec files
        companions.add(f"{base_no_ext}.html")
        companions.add(f"{base_no_ext}.scss")
        companions.add(f"{base_no_ext}.css")
        if not base_no_ext.endswith(".spec"):
            companions.add(f"{base_no_ext}.spec{ext}")
        # Fix D: TypeScript interface/model import companion discovery
        if file_content and ext in (".ts", ".tsx"):
            _ts_import_re = re.compile(
                r"import\s*\{([^}]+)\}\s*from\s*['\"]([./][^'\"]*)['\"]",
                re.MULTILINE,
            )
            _dirname = os.path.dirname(norm_path)
            for _match in _ts_import_re.finditer(file_content):
                _rel_module = _match.group(2)
                _candidates = [
                    os.path.normpath(os.path.join(_dirname, _rel_module + ".ts")).replace("\\", "/"),
                    os.path.normpath(os.path.join(_dirname, _rel_module + ".tsx")).replace("\\", "/"),
                ]
                for _c in _candidates:
                    companions.add(_c)
    elif ext in (".scss", ".css", ".sass", ".less"):
        companions.add(f"{base_no_ext}.html")
        companions.add(f"{base_no_ext}.ts")
        companions.add(f"{base_no_ext}.tsx")
    elif ext == ".java":
        if base_no_ext.endswith("Test"):
            companions.add(f"{base_no_ext[:-4]}.java")
        elif base_no_ext.endswith("Impl"):
            companions.add(f"{base_no_ext[:-4]}.java")
            companions.add(f"{base_no_ext[:-4]}Test.java")
        else:
            companions.add(f"{base_no_ext}Impl.java")
            companions.add(f"{base_no_ext}Test.java")
        # Fix D: import-based DTO/Model/Entity companion discovery
        if file_content:
            _java_import_type_re = re.compile(
                r'import\s+[\w.]+\.([\w]+(?:Model|Dto|Request|Response|Entity|Input|Output|Payload|Data|VO|PO))\s*;',
                re.IGNORECASE,
            )
            for _cls in _java_import_type_re.findall(file_content):
                companions.add(f"{_cls}.java")
    elif ext == ".py":
        # Fix D: Python import companion discovery for dataclasses / Pydantic models
        if file_content:
            _py_from_import_re = re.compile(
                r"from\s+(\.[\w.]*)\s+import\s+([\w,\s]+)",
                re.MULTILINE,
            )
            _dirname = os.path.dirname(norm_path)
            for _match in _py_from_import_re.finditer(file_content):
                _rel = _match.group(1).lstrip(".")
                _module_path = os.path.join(_dirname, _rel.replace(".", "/") + ".py").replace("\\", "/")
                companions.add(os.path.normpath(_module_path).replace("\\", "/"))
        dirname = os.path.dirname(norm_path)
        filename = os.path.basename(norm_path)
        if filename.startswith("test_"):
            companions.add(os.path.join(dirname, filename[5:]).replace("\\", "/"))
        else:
            companions.add(os.path.join(dirname, f"test_{filename}").replace("\\", "/"))

    # ── 2. Explicit Angular @Component / Template Links ──────────────────────
    if file_content:
        # Match templateUrl: './name.component.html'
        template_matches = re.findall(r"templateUrl\s*:\s*['\"]([^'\"]+)['\"]", file_content)
        for tpl in template_matches:
            dirname = os.path.dirname(norm_path)
            tpl_clean = tpl.lstrip("./").replace("\\", "/")
            resolved = os.path.normpath(os.path.join(dirname, tpl_clean)).replace("\\", "/")
            companions.add(resolved)

        # Match styleUrls: ['./name.component.scss', ...]
        style_matches = re.findall(r"styleUrls\s*:\s*\[([^\]]+)\]", file_content)
        for group in style_matches:
            for style in re.findall(r"['\"]([^'\"]+)['\"]", group):
                dirname = os.path.dirname(norm_path)
                style_clean = style.lstrip("./").replace("\\", "/")
                resolved = os.path.normpath(os.path.join(dirname, style_clean)).replace("\\", "/")
                companions.add(resolved)

    # ── 3. Filter against known workspace files if provided ──────────────────
    if known_files:
        valid_companions: List[str] = []
        for comp in companions:
            matched = resolve_canonical_path(comp, known_files)
            if matched and matched not in valid_companions and matched != file_path:
                valid_companions.append(matched)
        return valid_companions

    return [c for c in companions if c != norm_path]
