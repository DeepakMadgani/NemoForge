"""
Semantic Concept Indexer & Graph RAG Bridge.

Implements the "Semantic Concept Layer" and "Graph RAG" patterns from:
- Sean's AI Stories: "AI Agent Graph Engineering In 21 Min | Loop, Harness, System Design"
- "Memory Layers | Graph RAG, Vector DB, SQLite"

Connects natural-language ticket concepts ("acknowledgement", "date range validation",
"case-insensitive user", "deliverable export") to code symbols and files in the SQLite index,
completing the missing write-half of the graph.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Core domain ontology for enterprise web applications
CORE_DOMAIN_CONCEPTS: List[Dict[str, Any]] = [
    {
        "id": "concept_transmittal_ack",
        "name": "acknowledgement",
        "category": "workflow",
        "description": "Transmittal acknowledgement action, buttons, dialogs and backend sagas",
        "keywords": ["acknowledgement", "acknowledge", "acknowledgment", "ack"],
        "symbol_patterns": ["%Ack%", "%ack%", "%Acknowledgement%", "%Acknowledgment%"],
        "path_patterns": ["%transmittal%edit%", "%transmittal%ack%"]
    },
    {
        "id": "concept_transmittal",
        "name": "transmittal",
        "category": "domain_entity",
        "description": "Transmittal creation, properties, review, and migration",
        "keywords": ["transmittal", "transmittals"],
        "symbol_patterns": ["%Transmittal%", "%transmittal%"],
        "path_patterns": ["%transmittal%"]
    },
    {
        "id": "concept_deliverable",
        "name": "deliverable",
        "category": "domain_entity",
        "description": "Deliverable grid, sets, bulk upload, review and download",
        "keywords": ["deliverable", "deliverables", "deliverable-set"],
        "symbol_patterns": ["%Deliverable%", "%deliverable%"],
        "path_patterns": ["%deliverable%"]
    },
    {
        "id": "concept_user_email",
        "name": "user_email",
        "category": "identity",
        "description": "User identity, email lookups, case sensitivity, and membership",
        "keywords": ["email", "user", "case", "insensitive", "identity"],
        "symbol_patterns": ["%getUserByEmail%", "%findByEmail%", "%Email%"],
        "path_patterns": ["%user%", "%member%"]
    },
    {
        "id": "concept_project_options",
        "name": "project_options",
        "category": "configuration",
        "description": "Project level options, settings, allow creation flags, checkboxes",
        "keywords": ["option", "options", "creation", "allow", "project-settings"],
        "symbol_patterns": ["%ProjectOption%", "%allowCreation%", "%projectSettings%"],
        "path_patterns": ["%project-option%", "%project-settings%"]
    },
    {
        "id": "concept_bulk_upload",
        "name": "bulk_upload",
        "category": "feature",
        "description": "Deliverable and template bulk upload, processing and errors",
        "keywords": ["upload", "bulk", "template", "heading"],
        "symbol_patterns": ["%BulkUpload%", "%bulkUpload%"],
        "path_patterns": ["%bulk-upload%", "%bulkupload%"]
    },
    {
        "id": "concept_issue_viewer",
        "name": "issues",
        "category": "feature",
        "description": "Issues review panel, list, publishing and Brava viewer annotations",
        "keywords": ["issue", "issues", "review-panel", "viewer"],
        "symbol_patterns": ["%IssueList%", "%issueList%", "%IssueReview%"],
        "path_patterns": ["%issue-list%", "%issues%"]
    },
    {
        "id": "concept_contract_library",
        "name": "library_delete",
        "category": "action",
        "description": "Library deletion confirmation, dialogs, and contract management",
        "keywords": ["library", "delete", "confirmation", "contract"],
        "symbol_patterns": ["%LibraryDelete%", "%deleteLibrary%"],
        "path_patterns": ["%library-delete%", "%library%"]
    },
    {
        "id": "concept_date_picker",
        "name": "date_component",
        "category": "ui_widget",
        "description": "Date range picker, locale formatting, date filters",
        "keywords": ["date", "datepicker", "range", "calendar", "locale"],
        "symbol_patterns": ["%DateComponent%", "%dateRange%"],
        "path_patterns": ["%date%", "%calendar%"]
    }
]


class SemanticConceptIndexer:
    """
    Manages the Semantic Concept Layer in the SQLite index.
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.db_path = self.workspace_path / ".aviator" / "index.db"

    def ensure_concept_tables(self) -> None:
        """Creates semantic concept tables if they do not exist."""
        if not self.db_path.is_file():
            logger.warning(f"Index database not found at {self.db_path}")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_concepts (
                    concept_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    description TEXT,
                    keywords TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concept_symbols (
                    concept_id TEXT,
                    symbol_id INTEGER,
                    symbol_name TEXT,
                    file_path TEXT,
                    relevance REAL,
                    PRIMARY KEY (concept_id, symbol_id),
                    FOREIGN KEY (concept_id) REFERENCES semantic_concepts(concept_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_symbols_path ON concept_symbols(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_symbols_concept ON concept_symbols(concept_id)")
            conn.commit()

    def populate_concepts(self) -> int:
        """Populates the semantic concept layer by linking domain concepts to symbols."""
        self.ensure_concept_tables()
        if not self.db_path.is_file():
            return 0

        links_created = 0
        with sqlite3.connect(self.db_path) as conn:
            # 1. Insert or update defined concepts
            for c in CORE_DOMAIN_CONCEPTS:
                conn.execute("""
                    INSERT OR REPLACE INTO semantic_concepts (concept_id, name, category, description, keywords)
                    VALUES (?, ?, ?, ?, ?)
                """, (c["id"], c["name"], c["category"], c["description"], " ".join(c["keywords"])))

                # 2. Find matching symbols in SQLite index
                for sym_pat in c["symbol_patterns"]:
                    cursor = conn.execute("""
                        SELECT id, name, path FROM symbols
                        WHERE name LIKE ? AND kind IN ('class', 'method', 'interface')
                        LIMIT 50
                    """, (sym_pat,))
                    rows = cursor.fetchall()
                    for r in rows:
                        sym_id, sym_name, sym_path = r
                        conn.execute("""
                            INSERT OR IGNORE INTO concept_symbols (concept_id, symbol_id, symbol_name, file_path, relevance)
                            VALUES (?, ?, ?, ?, 1.0)
                        """, (c["id"], sym_id, sym_name, sym_path))
                        links_created += 1

                # 3. Find matching file paths
                for path_pat in c["path_patterns"]:
                    cursor = conn.execute("""
                        SELECT path FROM files WHERE path LIKE ? LIMIT 30
                    """, (path_pat,))
                    rows = cursor.fetchall()
                    for r in rows:
                        f_path = r[0]
                        # Link a pseudo symbol 0 to associate the file directly
                        conn.execute("""
                            INSERT OR IGNORE INTO concept_symbols (concept_id, symbol_id, symbol_name, file_path, relevance)
                            VALUES (?, 0, 'file_level', ?, 0.8)
                        """, (c["id"], f_path))
                        links_created += 1

            conn.commit()

        logger.info(f"Populated {len(CORE_DOMAIN_CONCEPTS)} Semantic Concepts with {links_created} code links in SQLite")
        return links_created

    def query_concept_candidates(self, keywords: List[str], top_n: int = 15) -> List[Dict[str, Any]]:
        """Queries the semantic concept layer in SQLite for files matching given keywords."""
        if not self.db_path.is_file():
            return []

        lower_kws = [k.lower() for k in keywords if len(k) >= 3]
        if not lower_kws:
            return []

        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Check which concepts match our keywords
            placeholders = ",".join("?" for _ in lower_kws)
            sql = f"""
                SELECT cs.file_path, cs.symbol_name, sc.name as concept_name, cs.relevance
                FROM concept_symbols cs
                JOIN semantic_concepts sc ON cs.concept_id = sc.concept_id
                WHERE sc.name IN ({placeholders})
                   OR EXISTS (
                       SELECT 1 FROM json_each('["' || replace(sc.keywords, ' ', '","') || '"]')
                       WHERE value IN ({placeholders})
                   )
                ORDER BY cs.relevance DESC
                LIMIT ?
            """
            try:
                # pass lower_kws twice for the two placeholders
                cursor = conn.execute(sql, lower_kws + lower_kws + [top_n])
                rows = cursor.fetchall()
                for r in rows:
                    results.append({
                        "file_path": r["file_path"],
                        "symbol_name": r["symbol_name"],
                        "concept_name": r["concept_name"],
                        "relevance": r["relevance"],
                    })
            except Exception as e:
                # Fallback simple query
                simple_sql = f"""
                    SELECT cs.file_path, cs.symbol_name, sc.name as concept_name, cs.relevance
                    FROM concept_symbols cs
                    JOIN semantic_concepts sc ON cs.concept_id = sc.concept_id
                    WHERE sc.name IN ({placeholders})
                    ORDER BY cs.relevance DESC
                    LIMIT ?
                """
                try:
                    cursor = conn.execute(simple_sql, lower_kws + [top_n])
                    for r in cursor.fetchall():
                        results.append({
                            "file_path": r["file_path"],
                            "symbol_name": r["symbol_name"],
                            "concept_name": r["concept_name"],
                            "relevance": r["relevance"],
                        })
                except Exception as ex2:
                    logger.debug(f"Concept query failed: {ex2}")

        return results
