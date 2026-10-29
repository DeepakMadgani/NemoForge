"""Item 28: episodic memory only records SUCCESS for verified runs.

- Episode.verification_level column (fresh DB + legacy DB migration)
- consolidate_run(): completed-but-unverified -> FAILED, verified -> SUCCESS
- recall returns the stored level; localization weights recalled files by it.
"""

import json
import sqlite3

import pytest

from ticket_to_code.memory.episodic_memory import Episode, EpisodicMemoryStore


def _columns(db_path):
    with sqlite3.connect(db_path) as conn:
        return {r[1] for r in conn.execute("PRAGMA table_info(episodes)")}


def _episode_row(store, ticket_id):
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT outcome, verification_level FROM episodes WHERE ticket_id = ? ORDER BY timestamp DESC LIMIT 1",
            (ticket_id,),
        ).fetchone()
    return dict(row) if row else None


def _state(ticket_id, status="completed", **extra):
    s = {
        "ticket_id": ticket_id,
        "title": "Fix coordinator modal",
        "description": "Coordinator modal shows ghost members",
        "status": status,
        "candidate_diffs": [{"path": "src/app/coordinators.component.ts"}],
    }
    s.update(extra)
    return s


def test_fresh_db_has_verification_level_column(tmp_path):
    store = EpisodicMemoryStore(str(tmp_path))
    assert "verification_level" in _columns(store.db_path)


def test_legacy_db_is_migrated(tmp_path):
    db_dir = tmp_path / ".aviator" / "memory"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "episodes.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE episodes (
                episode_id TEXT PRIMARY KEY, ticket_id TEXT, fingerprint TEXT, title TEXT,
                description TEXT, error_signature TEXT, resolved_files TEXT, outcome TEXT,
                verifier_result TEXT, patch_summary TEXT, tokens_in INTEGER, tokens_out INTEGER,
                llm_calls INTEGER, wall_clock_s REAL, timestamp TEXT
            )
        """)
        conn.execute(
            "INSERT INTO episodes (episode_id, ticket_id, title, description, resolved_files, outcome) "
            "VALUES ('old1', 'T-OLD', 'Legacy episode', 'desc', '[]', 'SUCCESS')"
        )
    assert "verification_level" not in _columns(db_path)

    store = EpisodicMemoryStore(str(tmp_path))  # must migrate, not crash
    assert "verification_level" in _columns(store.db_path)

    # Old rows read back with level None; new rows store the level.
    store.record_episode(Episode(
        episode_id="new1", ticket_id="T-NEW", title="Coordinator modal fix",
        description="fix", resolved_files=["a.ts"], verification_level="compile_only",
    ))
    with sqlite3.connect(store.db_path) as conn:
        rows = dict(conn.execute("SELECT episode_id, verification_level FROM episodes").fetchall())
    assert rows == {"old1": None, "new1": "compile_only"}


def test_consolidate_completed_without_verification_is_not_success(tmp_path):
    store = EpisodicMemoryStore(str(tmp_path))
    summary = store.consolidate_run(_state("T-1"), repo="cloudforge")
    assert summary["files_touched"] == 1
    row = _episode_row(store, "T-1")
    assert row["outcome"] == "FAILED"
    assert row["verification_level"] is None


@pytest.mark.parametrize("level", ["test_verified", "compile_only"])
def test_consolidate_verified_levels_are_success_and_stored(tmp_path, level):
    store = EpisodicMemoryStore(str(tmp_path))
    store.consolidate_run(_state("T-2"), repo="cloudforge", verification_level=level)
    row = _episode_row(store, "T-2")
    assert row == {"outcome": "SUCCESS", "verification_level": level}


def test_consolidate_reads_verification_level_from_state(tmp_path):
    store = EpisodicMemoryStore(str(tmp_path))
    store.consolidate_run(_state("T-3", verification_level="test_verified"), repo="cloudforge")
    assert _episode_row(store, "T-3")["outcome"] == "SUCCESS"


def test_consolidate_outcome_check_correct_still_requires_verification(tmp_path):
    store = EpisodicMemoryStore(str(tmp_path))
    store.consolidate_run(
        _state("T-4", status="unknown", outcome_check_result={"status": "CORRECT"}), repo="cloudforge",
    )
    assert _episode_row(store, "T-4")["outcome"] == "FAILED"
    store.consolidate_run(
        _state("T-5", status="unknown", outcome_check_result={"status": "CORRECT"}),
        repo="cloudforge", verification_level="compile_only",
    )
    assert _episode_row(store, "T-5")["outcome"] == "SUCCESS"


def test_consolidate_stopped_is_interrupted_and_default_kwarg_keeps_legacy_callers(tmp_path):
    store = EpisodicMemoryStore(str(tmp_path))
    summary = store.consolidate_run(_state("T-6", status="stopped"), repo="cloudforge")  # legacy 2-arg call
    assert summary["files_touched"] == 1
    assert _episode_row(store, "T-6")["outcome"] == "INTERRUPTED"


def test_recall_returns_verification_level(tmp_path):
    store = EpisodicMemoryStore(str(tmp_path))
    store.record_episode(Episode(
        episode_id="ep1", ticket_id="T-7", title="Fix coordinator modal ghost members",
        description="Coordinator modal lists removed members", resolved_files=["x.ts"],
        outcome="SUCCESS", verification_level="test_verified",
    ))
    results = store.recall_similar_episodes("coordinator modal ghost members", top_k=3, only_successful=False)
    assert results, "FTS recall should find the episode"
    ep, _score = results[0]
    assert ep.episode_id == "ep1"
    assert ep.verification_level == "test_verified"


def test_seed_from_answer_key_is_test_verified(tmp_path):
    store = EpisodicMemoryStore(str(tmp_path))
    key = tmp_path / "answer_key.json"
    key.write_text(json.dumps({"tickets": [
        {"id": "T-SEED", "title": "Seeded fix", "evidence": "detail",
         "expected_changed_files": [{"path": "src/seed.ts"}]},
    ]}), encoding="utf-8")
    assert store.seed_from_answer_key(key) == 1
    assert _episode_row(store, "T-SEED") == {"outcome": "SUCCESS", "verification_level": "test_verified"}


def test_localization_episodic_confidence_weighting():
    from ticket_to_code.agents.localization_agent import _episodic_confidence
    assert _episodic_confidence("test_verified") == pytest.approx(0.90)
    assert _episodic_confidence("compile_only") == pytest.approx(0.75)
    assert _episodic_confidence(None) == pytest.approx(0.5)
    assert _episodic_confidence("bogus") == pytest.approx(0.5)
