"""
Unit tests for Episodic Fix Memory & Zero-Token Fast Path (Category 6.1).
"""
import tempfile
import pytest
from ticket_to_code.memory.episodic_memory import EpisodicMemoryStore


def test_zero_token_fast_path_record_and_lookup():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = EpisodicMemoryStore(tmp_dir)

        error_sig = "TS2307_cannot_find_module_core"
        patch = "--- a/auth.ts\n+++ b/auth.ts\n@@ -1,1 +1,2 @@\n+import { Core } from '@bim/core';\n"
        
        # 1. Record verified patch
        patch_id = store.record_verified_patch(
            repo="cc4e",
            error_sig=error_sig,
            ticket_summary="Fix missing core import in auth",
            patch_content=patch,
            verified=True
        )
        assert patch_id is not None

        # 2. Zero-token lookup
        hit = store.lookup_zero_token_patch("cc4e", error_sig)
        assert hit is not None
        assert hit["patch_content"] == patch
        assert hit["hit_count"] >= 1

        # 3. Non-existent signature returns None
        miss = store.lookup_zero_token_patch("cc4e", "unknown_error_sig")
        assert miss is None

        # 4. Demote patch upon verification failure -> future lookups must return None
        demoted = store.demote_patch("cc4e", error_sig)
        assert demoted is True
        assert store.lookup_zero_token_patch("cc4e", error_sig) is None
