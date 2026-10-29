"""
Sibling Package & Schema Rosetta Stone Crawler.

Discovers companion repositories, DAOs, entity models, and Spring Data / JPA @Query
join relationships for any given target service or file.
Provides deterministic grounding so neither Autonomous Flow nor Chat can hallucinate
custom IDs, miss existing repository methods, or misalign foreign key joins.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class SiblingCrawler:
    """Discovers sibling repositories, entity models, and Rosetta Stone query relationships."""

    @classmethod
    def crawl_service_siblings(
        cls,
        workspace_path: str | Path,
        target_path_or_name: str,
        max_methods: int = 15,
    ) -> str:
        """Crawls sibling repositories and models for a given file or service name.
        Returns a rich markdown block containing:
          1. Exact repository interface declarations (with generic ID types).
          2. Complete list of existing repository method signatures (preventing interface wipeout).
          3. Rosetta Stone join relationships extracted from @Query annotations.
          4. Entity primary keys and foreign key columns.
        """
        ws = Path(workspace_path).resolve()
        target_str = str(target_path_or_name).strip()
        if not target_str:
            return ""

        # 1. Resolve owning service directory
        service_dir: Optional[Path] = None
        cand_p = (ws / target_str) if not Path(target_str).is_absolute() else Path(target_str)

        if cand_p.is_file() or cand_p.is_dir():
            # Find the top-level service folder under workspace
            try:
                rel = cand_p.relative_to(ws)
            except ValueError:
                rel = cand_p
            parts = rel.parts
            if parts and (ws / parts[0]).is_dir():
                service_dir = ws / parts[0]
        else:
            # Check if target_str matches any top-level service
            for child in ws.iterdir():
                if child.is_dir() and (child.name.lower() in target_str.lower() or target_str.lower() in child.name.lower()):
                    service_dir = child
                    break

        if not service_dir or not service_dir.is_dir():
            service_dir = ws

        # 2. Locate repository and model directories
        repo_files: List[Path] = []
        model_files: List[Path] = []

        # If cand_p is a Java file, extract its injected fields and imports directly
        injected_repo_names: Set[str] = set()
        if cand_p.is_file() and cand_p.suffix == ".java":
            try:
                c_text = cand_p.read_text(encoding="utf-8", errors="ignore")
                for m in re.finditer(r"(?:private|protected|public)?\s+(?:final\s+)?([A-Z]\w*(?:Repository|Dao))\s+\w+;", c_text):
                    injected_repo_names.add(m.group(1))
                for m in re.finditer(r"import\s+[\w.]+\.([A-Z]\w*(?:Repository|Dao|Member|Contract|Project));", c_text):
                    injected_repo_names.add(m.group(1))
            except Exception:
                pass

        for root, dirs, files in os.walk(service_dir):
            if any(p in ("target", "build", ".git", "node_modules", "test", "specifications") for p in Path(root).parts):
                continue
            r_lower = root.replace("\\", "/").lower()
            for f in files:
                if not f.endswith(".java") or "specification" in f.lower():
                    continue
                full_f = Path(root) / f
                if "/repository" in r_lower or "/dao" in r_lower or f.endswith("Repository.java") or f.endswith("Dao.java"):
                    repo_files.append(full_f)
                elif "/model" in r_lower or "/entity" in r_lower or "/domain" in r_lower:
                    model_files.append(full_f)

        if not repo_files and not model_files:
            return ""

        # Score relevance based on injected fields and target tokens
        keywords = set(re.findall(r"[A-Za-z0-9_]+", target_str.lower()))
        def repo_score(p: Path) -> int:
            score = 0
            stem = p.stem
            stem_low = stem.lower()
            if stem in injected_repo_names:
                score += 150
            if "contractmember" in stem_low:
                score += 80
            if "participatingmember" in stem_low:
                score += 80
            if "member" in stem_low and "member" in keywords:
                score += 50
            if "contract" in stem_low and "contract" in keywords:
                score += 40
            for k in keywords:
                if len(k) > 3 and k in stem_low:
                    score += 15
            return score

        repo_files.sort(key=repo_score, reverse=True)
        active_repos = repo_files[:8]

        def model_score(p: Path) -> int:
            score = 0
            stem = p.stem
            stem_low = stem.lower()
            if stem in injected_repo_names:
                score += 100
            if "contractmember" in stem_low:
                score += 60
            if "participatingmember" in stem_low:
                score += 60
            if "member" in stem_low and "member" in keywords:
                score += 40
            for k in keywords:
                if len(k) > 3 and k in stem_low:
                    score += 15
            return score

        model_files.sort(key=model_score, reverse=True)
        active_models = model_files[:8]

        out_lines: List[str] = [
            f"### ⚡ Sibling Package & Schema Rosetta Stone Grounding ({service_dir.name}):",
            "> **CRITICAL ARCHITECTURAL CONSTRAINTS:**",
            "> - Do NOT create full-file replacements that wipe out the existing repository methods below.",
            "> - Preserves exact entity ID types (e.g. `UUID`, not custom ID classes).",
            "> - Honor the Rosetta Stone join relationships extracted from production `@Query` annotations.",
            ""
        ]

        rosetta_joins: List[str] = []

        # 3. Parse Repositories
        out_lines.append("#### Sibling Repositories & Declared Contracts:")
        for rf in active_repos[:6]:
            try:
                content = rf.read_text(encoding="utf-8", errors="ignore")
                try:
                    rel_disp = rf.relative_to(ws).as_posix()
                except ValueError:
                    rel_disp = rf.name
                
                # Interface signature
                sig_match = re.search(r"public\s+interface\s+([A-Za-z0-9_]+(?:\s+extends\s+[^{]+)?)", content)
                sig_str = sig_match.group(1).strip() if sig_match else rf.stem

                # Extract queries
                queries = re.findall(r'@Query\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']', content)
                for q in queries:
                    # Look for join conditions like cm.userId = pm.otdsUserId
                    for jm in re.finditer(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)\s*=\s*([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', q):
                        pair = f"{jm.group(1)} <-> {jm.group(2)}"
                        if pair not in rosetta_joins:
                            rosetta_joins.append(pair)

                # Extract method signatures
                methods: List[str] = []
                for line in content.splitlines():
                    clean_l = line.strip()
                    if clean_l.startswith(("//", "/*", "*", "@", "package ", "import ", "public interface")):
                        continue
                    if "(" in clean_l and clean_l.endswith(";"):
                        methods.append(clean_l)

                out_lines.append(f"- **`{rf.name}`** (`{rel_disp}`):")
                out_lines.append(f"  * Declaration: `public interface {sig_str}`")
                if queries:
                    out_lines.append("  * Production Queries:")
                    for q in queries[:4]:
                        out_lines.append(f"    - `{q[:140]}`")
                if methods:
                    out_lines.append(f"  * Existing Methods ({len(methods)} declared - MUST BE PRESERVED):")
                    for m in methods[:max_methods]:
                        out_lines.append(f"    - `{m}`")
                    if len(methods) > max_methods:
                        out_lines.append(f"    - ... [{len(methods) - max_methods} more existing methods]")
                out_lines.append("")
            except Exception:
                continue

        # 4. Rosetta Stone Join Matrix
        if rosetta_joins:
            out_lines.append("#### ⚡ Rosetta Stone Schema & Join Map (Production Ground Truth):")
            for j in rosetta_joins[:8]:
                out_lines.append(f"- ` {j} `")
            out_lines.append("")

        # 5. Entity Models & ID Types
        if active_models:
            out_lines.append("#### Sibling Entity Models & ID Types:")
            for mf in active_models[:6]:
                try:
                    content = mf.read_text(encoding="utf-8", errors="ignore")
                    # ID field extraction
                    id_match = re.search(r"@Id\b(?:[^;]+;)?\s*(?:@[^;]+;\s*)*private\s+([A-Za-z0-9_<>]+)\s+id\s*;", content)
                    id_type = id_match.group(1) if id_match else "UUID"

                    # Key columns
                    col_matches = re.findall(r'@Column\s*\(\s*name\s*=\s*["\']([^"\']+)["\']\s*\)\s*(?:private|protected)\s+([A-Za-z0-9_<>]+)\s+([A-Za-z0-9_]+)\s*;', content)
                    cols_summary = ", ".join(f"{c[2]}: {c[1]} (`{c[0]}`)" for c in col_matches[:5]) if col_matches else "Standard fields"

                    out_lines.append(f"- **`{mf.name}`**: PrimaryKey=`{id_type} id`, KeyColumns=[{cols_summary}]")
                except Exception:
                    continue
            out_lines.append("")

        return "\n".join(out_lines)
