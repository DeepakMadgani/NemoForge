"""
Episodic Memory System for Autonomous Coding Agents.

Implements cross-run episodic learning inspired by:
- Sean's AI Stories: "AI Agent Memory System | Semantic & Episodic Memory, RAG"
- Hermes Agent Harness: "Loop Engineering, Memory System, Self-Improving AI"

Differentiates Semantic Memory (what is true in the codebase right now) from
Episodic Memory (the historical record of past ticket resolutions, failures,
winning files, patch patterns, and measured execution costs).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Represents a single historical run / intervention episode."""
    episode_id: str
    ticket_id: str
    title: str
    description: str
    symptoms: List[str] = field(default_factory=list)
    error_signature: str = ""
    resolved_files: List[str] = field(default_factory=list)
    outcome: str = "SUCCESS"  # SUCCESS | FAILED_VERIFICATION | DECLINED | TIMEOUT
    verifier_result: str = ""  # Maven / Surefire / npm status
    patch_summary: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    wall_clock_s: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    # How the outcome was verified: "test_verified" | "compile_only" | None (unknown/unverified).
    # Only test_verified / compile_only episodes may carry outcome="SUCCESS".
    verification_level: Optional[str] = None

    @property
    def fingerprint(self) -> str:
        """Normalized fingerprint for fast exact or near-duplicate lookup."""
        normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", f"{self.title} {self.error_signature}").lower()
        tokens = sorted(set(t for t in normalized.split() if len(t) >= 4))
        return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


class EpisodicMemoryStore:
    """
    Persistent SQLite storage and retrieval for cross-run episodes.
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.db_dir = self.workspace_path / ".aviator" / "memory"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "episodes.db"
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS episodes (
                        episode_id TEXT PRIMARY KEY,
                        ticket_id TEXT,
                        fingerprint TEXT,
                        title TEXT,
                        description TEXT,
                        error_signature TEXT,
                        resolved_files TEXT,
                        outcome TEXT,
                        verifier_result TEXT,
                        patch_summary TEXT,
                        tokens_in INTEGER,
                        tokens_out INTEGER,
                        llm_calls INTEGER,
                        wall_clock_s REAL,
                        timestamp TEXT,
                        verification_level TEXT
                    )
                """)
                # Migration for databases created before verification_level existed.
                cols = {r[1] for r in conn.execute("PRAGMA table_info(episodes)")}
                if "verification_level" not in cols:
                    conn.execute("ALTER TABLE episodes ADD COLUMN verification_level TEXT")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ticket_id ON episodes(ticket_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_fingerprint ON episodes(fingerprint)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome)")

                # Full-text search table for semantic episodic recall
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                        episode_id UNINDEXED,
                        title,
                        description,
                        error_signature,
                        tokenize='porter unicode61'
                    )
                """)

                # Structured, exact-match facts table with LFU hit counting
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS episodic_facts (
                        repo TEXT,
                        kind TEXT,
                        key TEXT,
                        value TEXT,
                        ticket_id TEXT,
                        ts INTEGER,
                        hit_count INTEGER DEFAULT 1,
                        PRIMARY KEY (repo, kind, key)
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_lookup ON episodic_facts(repo, kind, key)")

                # Verified patch store for zero-token fast paths (Category 6.1)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS verified_patches (
                        patch_id TEXT PRIMARY KEY,
                        repo TEXT,
                        error_sig TEXT,
                        ticket_summary TEXT,
                        patch_content TEXT,
                        verified INTEGER DEFAULT 1,
                        hit_count INTEGER DEFAULT 1,
                        last_used INTEGER
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_verified_patches_lookup ON verified_patches(repo, error_sig)")
        finally:
            conn.close()

    def record_episode(self, episode: Episode) -> None:
        """Stores or updates an episode in persistent memory."""
        resolved_json = json.dumps(episode.resolved_files)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO episodes (
                    episode_id, ticket_id, fingerprint, title, description,
                    error_signature, resolved_files, outcome, verifier_result,
                    patch_summary, tokens_in, tokens_out, llm_calls, wall_clock_s, timestamp,
                    verification_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.episode_id, episode.ticket_id, episode.fingerprint,
                episode.title, episode.description, episode.error_signature,
                resolved_json, episode.outcome, episode.verifier_result,
                episode.patch_summary, episode.tokens_in, episode.tokens_out,
                episode.llm_calls, episode.wall_clock_s, episode.timestamp,
                episode.verification_level,
            ))
            # Keep FTS in sync
            conn.execute("DELETE FROM episodes_fts WHERE episode_id = ?", (episode.episode_id,))
            conn.execute("""
                INSERT INTO episodes_fts (episode_id, title, description, error_signature)
                VALUES (?, ?, ?, ?)
            """, (episode.episode_id, episode.title, episode.description, episode.error_signature))
            conn.commit()
        logger.info(f"💾 Recorded Episode {episode.episode_id} for ticket {episode.ticket_id} (outcome={episode.outcome})")

    def recall_similar_episodes(
        self,
        query_text: str,
        error_signature: str = "",
        top_k: int = 5,
        only_successful: bool = True
    ) -> List[Tuple[Episode, float]]:
        """
        Recalls past episodes similar to the given ticket text or error signature.
        Returns a list of (Episode, score) tuples.
        """
        clean_query = re.sub(r"[^\w\s]", " ", query_text).strip()
        tokens = [t for t in clean_query.split() if len(t) >= 4]
        if not tokens:
            return []

        # Form FTS query using OR matching for top terms
        fts_query = " OR ".join(tokens[:10])

        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = """
                SELECT e.*, rank
                FROM episodes_fts fts
                JOIN episodes e ON fts.episode_id = e.episode_id
                WHERE episodes_fts MATCH ?
            """
            if only_successful:
                sql += " AND e.outcome = 'SUCCESS'"
            sql += " ORDER BY rank LIMIT ?"

            try:
                cursor = conn.execute(sql, (fts_query, top_k * 2))
                rows = cursor.fetchall()
            except Exception as e:
                logger.debug(f"Episodic FTS query failed: {e}")
                return []

            query_set = set(t.lower() for t in tokens)
            for r in rows:
                files = json.loads(r["resolved_files"]) if r["resolved_files"] else []
                ep = Episode(
                    episode_id=r["episode_id"],
                    ticket_id=r["ticket_id"],
                    title=r["title"],
                    description=r["description"],
                    error_signature=r["error_signature"] or "",
                    resolved_files=files,
                    outcome=r["outcome"],
                    verifier_result=r["verifier_result"] or "",
                    patch_summary=r["patch_summary"] or "",
                    tokens_in=r["tokens_in"] or 0,
                    tokens_out=r["tokens_out"] or 0,
                    llm_calls=r["llm_calls"] or 0,
                    wall_clock_s=r["wall_clock_s"] or 0.0,
                    timestamp=r["timestamp"] or "",
                    verification_level=(
                        r["verification_level"] if "verification_level" in r.keys() else None
                    ),
                )
                # Compute Jaccard overlap on title + description
                text_set = set(re.findall(r"\b[a-zA-Z]{4,}\b", f"{ep.title} {ep.description}".lower()))
                overlap = len(query_set & text_set) / max(1, len(query_set | text_set))
                # BM25 rank is negative (lower = better)
                bm25_score = 1.0 / (1.0 + abs(float(r["rank"])))
                combined_score = round(0.6 * overlap + 0.4 * bm25_score, 3)

                results.append((ep, combined_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def seed_from_answer_key(self, answer_key_path: str | Path) -> int:
        """
        Seeds episodic memory from a curated or git-derived answer key.
        Returns the number of episodes recorded.
        """
        p = Path(answer_key_path)
        if not p.is_file():
            return 0
        data = json.loads(p.read_text(encoding="utf-8"))
        tickets = data.get("tickets", [])
        count = 0
        for t in tickets:
            t_id = t.get("id", "")
            title = t.get("title", "")
            desc = t.get("evidence", "") or t.get("detail", "") or title
            exp_files = [f["path"] for f in t.get("expected_changed_files", []) if isinstance(f, dict) and "path" in f]
            if not exp_files and t.get("primary"):
                exp_files = [t["primary"]]
            if not t_id or not exp_files:
                continue

            ep = Episode(
                episode_id=f"seed_git_{t_id}",
                ticket_id=t_id,
                title=title,
                description=desc,
                resolved_files=exp_files,
                outcome="SUCCESS",
                patch_summary=f"Git historical fix for ticket {t_id}",
                verification_level="test_verified",  # git history = shipped, verified fixes
            )
            self.record_episode(ep)
            count += 1
        logger.info(f"🌱 Seeded {count} historical episodes into EpisodicMemoryStore from {p.name}")
        return count

    # ── Structured Exact-Match Facts (Tier 3) ──────────────────────────────────

    def upsert_fact(
        self,
        repo: str,
        kind: str,
        key: str,
        value: str,
        ticket_id: str = "",
    ) -> None:
        """
        Upsert a structured fact into episodic memory.
        Increments hit_count on repeat matches and updates timestamp.
        """
        now_ts = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO episodic_facts (repo, kind, key, value, ticket_id, ts, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(repo, kind, key) DO UPDATE SET
                    value = excluded.value,
                    ticket_id = excluded.ticket_id,
                    ts = excluded.ts,
                    hit_count = episodic_facts.hit_count + 1
            """, (repo, kind, key, value, ticket_id, now_ts))
            conn.commit()

    def lookup_facts(
        self,
        repo: str,
        kind: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve structured facts matching repo, kind, or key.
        Ranked by hit_count DESC, ts DESC.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT repo, kind, key, value, ticket_id, ts, hit_count FROM episodic_facts WHERE repo = ?"
            params: List[Any] = [repo]
            if kind:
                query += " AND kind = ?"
                params.append(kind)
            if key:
                query += " AND key = ?"
                params.append(key)
            query += " ORDER BY hit_count DESC, ts DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def decay_stale_facts(
        self,
        repo: str,
        max_age_days: int = 45,
        min_hit_count: int = 1,
    ) -> int:
        """
        LFU Eviction: Removes or decays facts that haven't been reinforced within max_age_days.
        """
        cutoff_ts = int(time.time()) - (max_age_days * 86400)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM episodic_facts
                WHERE repo = ? AND ts < ? AND hit_count <= ?
            """, (repo, cutoff_ts, min_hit_count))
            deleted = cursor.rowcount
            conn.commit()
            if deleted:
                logger.info(f"🧹 Decayed {deleted} stale episodic facts from {repo}")
            return deleted

    _VERIFIED_LEVELS = ("test_verified", "compile_only")

    def consolidate_run(
        self,
        state: Dict[str, Any],
        repo: str,
        verification_level: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Distills a completed or stopped workflow state into durable facts and an episode.
        Mines:
        • file_touched: Candidate diffs and generated files
        • build_fix: Compilation errors and resolutions
        • ownership: High-confidence grounded owner files
        • pref: User rejection rules and preferences

        ``verification_level`` ("test_verified" | "compile_only" | None) gates the
        episode outcome: a run is recorded as SUCCESS only when it both finished
        and was verified. Unknown verification never yields SUCCESS, so later
        episodic recall cannot be poisoned by unverified "completions".
        """
        ticket_obj = state.get("ticket")
        ticket_id = getattr(ticket_obj, "ticket_id", "") if ticket_obj else str(state.get("ticket_id") or "UNKNOWN")
        title = getattr(ticket_obj, "title", "") if ticket_obj else str(state.get("title") or "")
        desc = getattr(ticket_obj, "description", "") if ticket_obj else str(state.get("description") or "")
        status = state.get("status", "unknown")
        vl = verification_level or state.get("verification_level")

        facts_added = 0

        # 1. Distill files touched
        diffs = state.get("candidate_diffs") or []
        gen_files = state.get("generated_code") or []
        touched: List[str] = []
        for d in diffs:
            fp = d.get("path") if isinstance(d, dict) else str(d)
            if fp and fp not in touched:
                touched.append(fp)
                self.upsert_fact(repo, "file_touched", fp, ticket_id, ticket_id)
                facts_added += 1

        for g in gen_files:
            fp = g.get("file_path") if isinstance(g, dict) else getattr(g, "file_path", "")
            if fp and fp not in touched:
                touched.append(fp)
                self.upsert_fact(repo, "file_touched", fp, ticket_id, ticket_id)
                facts_added += 1

        # 2. Mine build fixes and error resolutions
        exec_logs = state.get("exec_logs") or []
        for log_entry in exec_logs:
            # Look for typical compiler errors: TS2307, CS0246, etc.
            matches = re.findall(r"\b([A-Z]{2,4}\d{4,5})\b", log_entry)
            for err_code in matches:
                self.upsert_fact(repo, "build_fix", err_code, log_entry[:200], ticket_id)
                facts_added += 1

        # 3. Mine user rejection rules / preferences
        rejected = state.get("rejected_approaches") or []
        for rej in rejected:
            rej_str = str(rej).strip()
            if rej_str:
                self.upsert_fact(repo, "pref", f"reject_{ticket_id[:8]}", rej_str, ticket_id)
                facts_added += 1

        # 4. Save episode
        finished_ok = (
            status in ("completed", "memory_updated", "success")
            or (state.get("outcome_check_result") or {}).get("status") in ("CORRECT", "all_passed")
        )
        if finished_ok and vl in self._VERIFIED_LEVELS:
            outcome = "SUCCESS"
        elif status == "stopped":
            outcome = "INTERRUPTED"
        else:
            outcome = "FAILED"
        if finished_ok and outcome != "SUCCESS":
            logger.info(
                f"Episode for {ticket_id} finished ({status}) but verification_level={vl!r} "
                "is not verified — recorded as FAILED, not SUCCESS"
            )
        ep = Episode(
            episode_id=f"ep_{ticket_id}_{int(time.time())}",
            ticket_id=ticket_id,
            title=title,
            description=desc,
            resolved_files=touched,
            outcome=outcome,
            patch_summary=f"Run {status} with {len(touched)} file(s) touched",
            verification_level=vl,
        )
        self.record_episode(ep)

        return {"facts_added": facts_added, "files_touched": len(touched)}

    def prime_scratchpad(self, repo: str, ticket_text: str) -> Dict[str, Any]:
        """
        Primes the Tier-1 scratchpad before starting a run.
        Combines exact-match facts from episodic_facts with semantic recall of past episodes.
        """
        # Exact-match facts
        recent_files = self.lookup_facts(repo, kind="file_touched", limit=5)
        known_build_fixes = self.lookup_facts(repo, kind="build_fix", limit=5)
        preferences = self.lookup_facts(repo, kind="pref", limit=5)

        # Semantic recall of similar past runs
        similar_episodes = self.recall_similar_episodes(ticket_text, top_k=3, only_successful=False)

        return {
            "known_files": [f["key"] for f in recent_files],
            "known_fixes": {f["key"]: f["value"] for f in known_build_fixes},
            "user_preferences": [f["value"] for f in preferences],
            "similar_past_episodes": [
                {
                    "ticket_id": ep.ticket_id,
                    "title": ep.title,
                    "outcome": ep.outcome,
                    "files": ep.resolved_files,
                    "relevance": score,
                }
                for ep, score in similar_episodes
            ],
        }

    # ── Zero-Token Fast Path (Category 6.1) ────────────────────────────────────

    def record_verified_patch(
        self,
        repo: str,
        error_sig: str,
        ticket_summary: str,
        patch_content: str,
        verified: bool = True
    ) -> str:
        """
        Record a verified diff patch associated with a normalized error signature.
        Enables zero-token replay on recurring bugs.
        """
        patch_id = hashlib.sha256(f"{repo}:{error_sig}".encode("utf-8")).hexdigest()[:16]
        now_ts = int(time.time())
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    INSERT INTO verified_patches (
                        patch_id, repo, error_sig, ticket_summary, patch_content, verified, hit_count, last_used
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(patch_id) DO UPDATE SET
                        patch_content = excluded.patch_content,
                        verified = excluded.verified,
                        ticket_summary = excluded.ticket_summary,
                        hit_count = verified_patches.hit_count + 1,
                        last_used = excluded.last_used
                """, (patch_id, repo, error_sig, ticket_summary, patch_content, 1 if verified else 0, now_ts))
        finally:
            conn.close()
        logger.info(f"💾 Stored verified patch {patch_id} for error signature {error_sig[:12]} in {repo}")
        return patch_id

    def lookup_zero_token_patch(self, repo: str, error_sig: str) -> Optional[Dict[str, Any]]:
        """
        Zero-token fast path: Retrieves a verified patch for exact-matching normalized error signature.
        """
        if not error_sig:
            return None
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT patch_id, repo, error_sig, ticket_summary, patch_content, hit_count
                FROM verified_patches
                WHERE repo = ? AND error_sig = ? AND verified = 1
                ORDER BY hit_count DESC LIMIT 1
            """, (repo, error_sig))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                now_ts = int(time.time())
                with conn:
                    conn.execute(
                        "UPDATE verified_patches SET hit_count = hit_count + 1, last_used = ? WHERE patch_id = ?",
                        (now_ts, row["patch_id"])
                    )
                return res
        finally:
            conn.close()
        return None

    def demote_patch(self, repo: str, error_sig: str) -> bool:
        """
        Demotes a patch when its replay fails verification to prevent poisoning memory.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute("""
                    UPDATE verified_patches
                    SET verified = 0
                    WHERE repo = ? AND error_sig = ?
                """, (repo, error_sig))
                return cursor.rowcount > 0
        finally:
            conn.close()
