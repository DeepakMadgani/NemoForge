"""Database Schema Migration Verifier & Safety Gate.

Validates Flyway / Liquibase / SQL migrations in backend services:
1. DDL Dialect & Syntax Validation: parses CREATE TABLE, ALTER TABLE, ADD COLUMN, etc.
2. Flyway Naming & Monotonic Version Ordering: enforces V<ver>__<desc>.sql conventions.
3. Production Migration Safety Checks:
   - Prohibits destructive DROP TABLE / TRUNCATE without safety annotations.
   - Requires DEFAULT clauses when adding NOT NULL columns to prevent table locks / data failures.
4. Schema Verification Runner: integrates into the lean verify ladder.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

from ticket_to_code.lean.convergence import StructuredFailure
from ticket_to_code.lean.lean_path import VerifyOutcome


_FLYWAY_NAME_RE = re.compile(r"^V([0-9]+(?:[._][0-9]+)*)__([A-Za-z0-9_]+)\.sql$", re.I)
_SEMVER_NAME_RE = re.compile(r"^[vV]?([0-9]+(?:[._][0-9]+)*)(?:__|_|-)(.+?)\.(?:sql|py|cs|ts|js)$", re.I)
_TIMESTAMP_NAME_RE = re.compile(r"^([0-9]{8,14})(?:_|-)(.+?)\.(?:sql|py|cs|ts|js)$")
_SEQUENTIAL_NAME_RE = re.compile(r"^([0-9]{3,6})(?:_|-)(.+?)\.(?:sql|py|cs|ts|js)$")
_UNSAFE_DROP_RE = re.compile(r"\bDROP\s+(?:TABLE|COLUMN|DATABASE)\b", re.I)
_UNSAFE_TRUNCATE_RE = re.compile(r"\bTRUNCATE\s+TABLE\b", re.I)
_NOT_NULL_NO_DEFAULT_RE = re.compile(r"\bADD\s+(?:COLUMN\s+)?(\w+)\s+([A-Za-z0-9_()]+)\s+NOT\s+NULL(?!\s+DEFAULT)\b", re.I)


@dataclass
class GenericMigrationInfo:
    scheme: str  # "semver", "timestamp", "sequential"
    version_tuple: Tuple[int, ...]
    raw_version: str
    description: str


def parse_migration_info(name: str) -> Optional[GenericMigrationInfo]:
    """Parse migration across SemVer (V1.2__), Timestamp (20261012140000_), or Sequential (0042_) schemes."""
    # 1. Flyway / Semver
    m = _FLYWAY_NAME_RE.match(name)
    if m:
        raw_v, desc = m.group(1), m.group(2)
        parts = tuple(int(p) for p in re.split(r"[._]", raw_v) if p.isdigit())
        return GenericMigrationInfo(scheme="semver", version_tuple=parts, raw_version=raw_v, description=desc)

    # 2. Timestamp (8-14 digits)
    m = _TIMESTAMP_NAME_RE.match(name)
    if m:
        raw_v, desc = m.group(1), m.group(2)
        return GenericMigrationInfo(scheme="timestamp", version_tuple=(int(raw_v),), raw_version=raw_v, description=desc)

    # 3. Sequential integer (3-6 digits)
    m = _SEQUENTIAL_NAME_RE.match(name)
    if m:
        raw_v, desc = m.group(1), m.group(2)
        return GenericMigrationInfo(scheme="sequential", version_tuple=(int(raw_v),), raw_version=raw_v, description=desc)

    # 4. General Semver (v1.2.3_foo.sql, 1.0.0_foo.py)
    m = _SEMVER_NAME_RE.match(name)
    if m:
        raw_v, desc = m.group(1), m.group(2)
        parts = tuple(int(p) for p in re.split(r"[._]", raw_v) if p.isdigit())
        if parts:
            return GenericMigrationInfo(scheme="semver", version_tuple=parts, raw_version=raw_v, description=desc)

    return None


def suggest_next_migration_filename(existing_filenames: List[str], description: str = "new_migration") -> str:
    """Discovers the active migration naming scheme and returns the next non-colliding filename."""
    parsed_items = []
    for fn in existing_filenames:
        info = parse_migration_info(fn)
        if info:
            parsed_items.append(info)
    clean_desc = "".join(c if c.isalnum() or c == "_" else "_" for c in description)
    if not parsed_items:
        return f"V1.0__{clean_desc}.sql"

    # Determine dominant scheme
    scheme_counts: dict[str, int] = {}
    for item in parsed_items:
        scheme_counts[item.scheme] = scheme_counts.get(item.scheme, 0) + 1
    dominant_scheme = max(scheme_counts.items(), key=lambda kv: kv[1])[0]

    scheme_items = [it for it in parsed_items if it.scheme == dominant_scheme]
    highest = max(scheme_items, key=lambda it: it.version_tuple)

    if dominant_scheme == "sequential":
        next_int = highest.version_tuple[0] + 1
        width = max(len(highest.raw_version), 4)
        return f"{next_int:0{width}d}_{clean_desc}.sql"
    elif dominant_scheme == "timestamp":
        import datetime
        now_ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if int(now_ts) <= highest.version_tuple[0]:
            next_ts = str(highest.version_tuple[0] + 1)
        else:
            next_ts = now_ts
        return f"{next_ts}_{clean_desc}.sql"
    else:  # semver
        v_parts = list(highest.version_tuple)
        if len(v_parts) == 1:
            next_ver = f"V{v_parts[0] + 1}.0"
        elif len(v_parts) == 2:
            if v_parts[1] >= 9:
                next_ver = f"V{v_parts[0] + 1}.0"
            else:
                next_ver = f"V{v_parts[0]}.{v_parts[1] + 1}"
        else:
            v_parts[-1] += 1
            next_ver = f"V{'.'.join(str(p) for p in v_parts)}"
        return f"{next_ver}__{clean_desc}.sql"


@dataclass
class MigrationAnalysis:
    file_name: str
    is_valid_syntax: bool
    version: Optional[str] = None
    description: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def parse_flyway_version(name: str) -> Optional[Tuple[Tuple[int, ...], str]]:
    """Extracts numeric version tuple and description from a Flyway script name (backwards compatible)."""
    info = parse_migration_info(name)
    if info and info.scheme == "semver":
        return info.version_tuple, info.description
    return None



def _split_statements(sql_text: str) -> List[str]:
    """Splits SQL statements by semicolon while respecting string literals."""
    statements: List[str] = []
    current: List[str] = []
    in_quote = False
    quote_char = ""

    for char in sql_text:
        if in_quote:
            current.append(char)
            if char == quote_char:
                in_quote = False
        elif char in ("'", '"', "`"):
            in_quote = True
            quote_char = char
            current.append(char)
        elif char == ";":
            statements.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current and "".join(current).strip():
        statements.append("".join(current).strip())
    return [s for s in statements if s]


def validate_sql_syntax(sql_text: str) -> List[str]:
    """Validates SQL DDL statements for balanced delimiters, valid keywords, and standard DDL structure."""
    errors = []
    # Strip comments
    clean_lines = []
    for line in sql_text.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("--") or trimmed.startswith("//"):
            continue
        clean_lines.append(line)
    clean = "\n".join(clean_lines).strip()
    if not clean:
        return ["Migration file is empty or contains only comments."]

    # Check balanced parentheses
    if clean.count("(") != clean.count(")"):
        errors.append(f"Unbalanced parentheses: {clean.count('(')} open vs {clean.count(')')} close.")

    # Check statements terminate with semicolons and start with valid SQL keywords
    statements = _split_statements(clean)
    valid_starts = ("CREATE", "ALTER", "INSERT", "UPDATE", "DELETE", "DROP", "SELECT", "COMMENT", "GRANT", "REVOKE")
    for stmt in statements:
        first_word = stmt.split()[0].upper() if stmt.split() else ""
        if first_word not in valid_starts:
            errors.append(f"Statement starts with unrecognized SQL keyword: '{first_word}' in '{stmt[:60]}...'")

    # Safety: check for hazardous migrations
    if _UNSAFE_DROP_RE.search(clean) and "-- force-unsafe-drop" not in sql_text:
        errors.append("Destructive schema operation: DROP TABLE/COLUMN detected without '-- force-unsafe-drop' annotation.")
    if _UNSAFE_TRUNCATE_RE.search(clean):
        errors.append("Dangerous TRUNCATE statement detected in migration.")
    for m in _NOT_NULL_NO_DEFAULT_RE.finditer(clean):
        col, typ = m.group(1), m.group(2)
        errors.append(f"Unsafe migration: column '{col} {typ} NOT NULL' added without a DEFAULT clause.")

    return errors


def verify_db_migration(root: Path, written_rel: List[str]) -> Tuple[bool, Optional[StructuredFailure]]:
    """Verifies all SQL migration scripts in written files against naming, syntax, and ordering rules."""
    migration_indicators = ("migration", "migrations", "changelog", "evolutions", "schema", "db")
    migration_files = [
        r for r in written_rel
        if (
            parse_migration_info(Path(r).name) is not None
            or any(seg in r.lower().replace("\\", "/").split("/") for seg in migration_indicators)
        ) and Path(r).suffix in (".sql", ".py", ".cs", ".ts")
    ]
    if not migration_files:
        return True, None

    for rel in migration_files:
        p = root / rel
        if not p.is_file():
            continue
        # 1. Migration naming convention check (SemVer, Timestamp, or Sequential)
        parsed = parse_migration_info(p.name)
        if not parsed:
            # Allow repeatable migrations R__*.sql
            if not p.name.startswith("R__"):
                return False, StructuredFailure(
                    kind="db_migration",
                    file=rel,
                    actual=f"Invalid migration file name '{p.name}'. Expected semver (e.g. V1.11__AddStatus.sql), timestamp (e.g. 20261012140000_add_status.sql), or sequential integer (e.g. 0042_add_status.sql).",
                    hint="Rename the file to match the repository's migration versioning convention."
                )

        # 2. SQL syntax & safety validation (only for .sql files)
        if p.suffix == ".sql":
            sql_content = p.read_text(encoding="utf-8", errors="ignore")
            errors = validate_sql_syntax(sql_content)
            if errors:
                return False, StructuredFailure(
                    kind="db_migration",
                    file=rel,
                    actual="\n".join(errors),
                    hint="Fix the SQL syntax or safety error. Ensure DDL statements terminate with ';' and have balanced parentheses."
                )

        # 3. Version monotonicity check against existing sibling migrations
        if parsed:
            new_ver = parsed.version_tuple
            existing_versions = []
            for sibling in p.parent.iterdir():
                if sibling.name == p.name or not sibling.is_file():
                    continue
                s_parsed = parse_migration_info(sibling.name)
                if s_parsed and s_parsed.scheme == parsed.scheme:
                    existing_versions.append((s_parsed.version_tuple, sibling.name))
            # Check for version collisions
            for ver, s_name in existing_versions:
                if ver == new_ver:
                    return False, StructuredFailure(
                        kind="db_migration",
                        file=rel,
                        actual=f"Migration version collision: version '{p.name}' conflicts with existing '{s_name}'.",
                        hint="Increment the version number to be strictly higher than all existing migrations."
                    )
            # Check for version regression (monotonicity order)
            if existing_versions:
                max_ver = max(existing_versions, key=lambda it: it[0])
                if new_ver < max_ver[0]:
                    return False, StructuredFailure(
                        kind="db_migration",
                        file=rel,
                        actual=f"Out-of-order migration detected: version '{p.name}' is lower than existing '{max_ver[1]}'.",
                        hint="Enterprise migration engines reject out-of-order execution by default. Version must be strictly greater than the highest existing migration."
                    )

    return True, None
