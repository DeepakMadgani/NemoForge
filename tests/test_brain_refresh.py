"""Tests for the self-updating two-layer repository brain.

Covers the mechanical scan (the "what"), the content-signature semantic cache (the
"why"), and every add / change / undo / delete edge case that keeps the brain correct.
"""

import json
import os
import sys
import importlib.util
from pathlib import Path

import pytest

# Load the brain module directly from its file so we do NOT trigger the heavy
# `ticket_to_code` package __init__ (which imports the full aviator LLM stack). The
# scan/refresh path is intentionally stdlib-only, so this import is self-contained.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "ticket_to_code" / "brain" / "generate_repository_brain.py"
)
_spec = importlib.util.spec_from_file_location("gen_brain_standalone", _MODULE_PATH)
brain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brain)

scan_source_directory_brain = brain.scan_source_directory_brain
refresh_repository_brain = brain.refresh_repository_brain
_folder_signature = brain._folder_signature
_bound_descriptions = brain._bound_descriptions
SEMANTIC_CACHE_FILENAME = brain.SEMANTIC_CACHE_FILENAME
ensure_repository_brain = brain.ensure_repository_brain


# ──────────────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Records call count; returns a distinct description per invocation."""

    def __init__(self, content_fn=None):
        self.calls = 0
        self._content_fn = content_fn

    def invoke(self, prompt):
        self.calls += 1
        if self._content_fn is not None:
            return _Resp(self._content_fn(self, prompt))
        return _Resp(json.dumps({
            "business_domain": f"Domain{self.calls}",
            "technical_role": f"Role{self.calls}",
            "core_responsibilities": f"Responsibility {self.calls}",
        }))


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_brain(workspace: Path):
    p = workspace / "brain" / "knowledge" / "generated_directory_brain.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _load_cache(workspace: Path):
    p = workspace / "brain" / "knowledge" / SEMANTIC_CACHE_FILENAME
    return json.loads(p.read_text(encoding="utf-8"))


def _entry(brain_list, directory_path):
    return next((e for e in brain_list if e["directory_path"] == directory_path), None)


# ──────────────────────────────────────────────────────────────────────────────
# Signature behavior
# ──────────────────────────────────────────────────────────────────────────────

def test_signature_is_deterministic_and_order_independent():
    a = {"key_artifacts": ["b.py", "a.py"], "intent_terms": ["Y", "X"]}
    b = {"key_artifacts": ["a.py", "b.py"], "intent_terms": ["X", "Y"]}
    assert _folder_signature(a) == _folder_signature(b)


def test_signature_changes_when_structure_changes():
    a = {"key_artifacts": ["a.py"], "intent_terms": ["X"]}
    b = {"key_artifacts": ["a.py", "c.py"], "intent_terms": ["X"]}
    assert _folder_signature(a) != _folder_signature(b)


def test_signature_handles_missing_fields():
    assert _folder_signature({}) == _folder_signature({"key_artifacts": [], "intent_terms": []})


# ──────────────────────────────────────────────────────────────────────────────
# Mechanical scan
# ──────────────────────────────────────────────────────────────────────────────

def test_scan_nonexistent_workspace_returns_empty(tmp_path):
    assert scan_source_directory_brain(str(tmp_path / "nope")) == []


def test_scan_empty_workspace_returns_empty(tmp_path):
    assert scan_source_directory_brain(str(tmp_path)) == []


def test_scan_collects_sources_and_skips_ignored_dirs(tmp_path):
    _write(tmp_path / "src" / "api" / "user_controller.py", "class UserController:\n    pass\n")
    _write(tmp_path / "node_modules" / "pkg" / "index.js", "export const x = 1;")
    _write(tmp_path / "docker-compose.yml", "services: {}")  # not a source ext

    result = scan_source_directory_brain(str(tmp_path))
    dirs = {e["directory_path"] for e in result}

    assert "src/api" in dirs
    assert not any("node_modules" in d for d in dirs)
    api = _entry(result, "src/api")
    assert api["key_artifacts"] == ["src/api/user_controller.py"]
    assert "UserController" in api["intent_terms"]


# ──────────────────────────────────────────────────────────────────────────────
# Refresh: creation, enrichment, reuse
# ──────────────────────────────────────────────────────────────────────────────

def test_refresh_creates_brain_and_cache(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    summary = refresh_repository_brain(str(tmp_path), llm=None, enrich=False)

    assert summary["status"] == "refreshed"
    assert (tmp_path / "brain" / "knowledge" / "generated_directory_brain.json").exists()
    assert (tmp_path / "brain" / "knowledge" / SEMANTIC_CACHE_FILENAME).exists()


def test_refresh_enriches_with_llm(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    llm = FakeLLM()
    summary = refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)

    assert summary["enriched"] == 1
    assert llm.calls == 1
    entry = _entry(_load_brain(tmp_path), "app/mod")
    assert entry["business_domain"] == "Domain1"
    assert entry["technical_role"] == "Role1"


def test_refresh_reuses_cache_with_zero_llm_on_second_run(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    llm = FakeLLM()

    first = refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    assert first["enriched"] == 1 and llm.calls == 1

    second = refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    assert second["reused"] == 1
    assert second["enriched"] == 0
    assert llm.calls == 1  # no new LLM call for unchanged structure


def test_refresh_reenriches_on_structure_change(tmp_path):
    mod = tmp_path / "app" / "mod"
    _write(mod / "thing.py", "class Thing:\n    pass\n")
    llm = FakeLLM()
    refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    assert llm.calls == 1

    _write(mod / "extra.py", "class Extra:\n    pass\n")  # structural change
    summary = refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    assert summary["enriched"] == 1
    assert llm.calls == 2


# ──────────────────────────────────────────────────────────────────────────────
# Undo / revert / delete — the heart of "make it perfect"
# ──────────────────────────────────────────────────────────────────────────────

def test_refresh_revert_reuses_history_with_zero_new_llm(tmp_path):
    mod = tmp_path / "app" / "mod"
    _write(mod / "thing.py", "class Thing:\n    pass\n")
    llm = FakeLLM()

    # Run 1: original structure -> Domain1
    refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    assert _entry(_load_brain(tmp_path), "app/mod")["business_domain"] == "Domain1"

    # Run 2: add a file (structure B) -> Domain2
    extra = mod / "extra.py"
    _write(extra, "class Extra:\n    pass\n")
    refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    assert _entry(_load_brain(tmp_path), "app/mod")["business_domain"] == "Domain2"
    assert llm.calls == 2

    # Run 3: undo the addition -> back to structure A. Must reuse Domain1 for free.
    extra.unlink()
    summary = refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    assert summary["reused"] == 1
    assert llm.calls == 2  # NO new LLM call on revert
    assert _entry(_load_brain(tmp_path), "app/mod")["business_domain"] == "Domain1"


def test_refresh_prunes_deleted_folder(tmp_path):
    _write(tmp_path / "app" / "keep" / "a.py", "class A:\n    pass\n")
    _write(tmp_path / "app" / "drop" / "b.py", "class B:\n    pass\n")
    llm = FakeLLM()

    refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    cache = _load_cache(tmp_path)
    assert "app/keep" in cache["folders"] and "app/drop" in cache["folders"]

    # Delete one folder entirely (undo of a whole feature)
    import shutil
    shutil.rmtree(tmp_path / "app" / "drop")

    summary = refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    assert summary["pruned"] == 1

    brain_list = _load_brain(tmp_path)
    dirs = {e["directory_path"] for e in brain_list}
    assert "app/drop" not in dirs
    assert "app/keep" in dirs

    cache = _load_cache(tmp_path)
    assert "app/drop" not in cache["folders"]  # orphan cache entry pruned


# ──────────────────────────────────────────────────────────────────────────────
# Safety / graceful degradation
# ──────────────────────────────────────────────────────────────────────────────

def test_refresh_skips_nonexistent_workspace(tmp_path):
    summary = refresh_repository_brain(str(tmp_path / "missing"))
    assert summary["status"] == "skipped_no_workspace"


def test_refresh_empty_scan_preserves_existing_brain(tmp_path):
    # Seed a good brain first.
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    refresh_repository_brain(str(tmp_path), llm=FakeLLM(), enrich=True)
    good = _load_brain(tmp_path)
    assert good

    # Now remove all source files; a refresh must NOT wipe the existing brain.
    (tmp_path / "app" / "mod" / "thing.py").unlink()
    summary = refresh_repository_brain(str(tmp_path), enrich=False)
    assert summary["status"] == "skipped_empty_scan"
    assert _load_brain(tmp_path) == good  # untouched


def test_refresh_handles_corrupt_cache(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    cache_path = tmp_path / "brain" / "knowledge" / SEMANTIC_CACHE_FILENAME
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{ this is : not json", encoding="utf-8")

    summary = refresh_repository_brain(str(tmp_path), llm=FakeLLM(), enrich=True)
    assert summary["status"] == "refreshed"
    assert summary["enriched"] == 1  # recovered by treating cache as empty


def test_refresh_malformed_llm_json_keeps_mechanical(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    bad_llm = FakeLLM(content_fn=lambda self, prompt: "not valid json at all")

    summary = refresh_repository_brain(str(tmp_path), llm=bad_llm, enrich=True)
    assert summary["enriched"] == 0
    assert summary["unenriched"] == 1
    # Mechanical default retained, and nothing bad cached.
    cache = _load_cache(tmp_path)
    assert cache["folders"] == {}


def test_refresh_llm_extra_keys_are_ignored(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    content = json.dumps({
        "business_domain": "Payments",
        "technical_role": "Service",
        "core_responsibilities": "Handles money",
        "unexpected": "ignore me",
    })
    llm = FakeLLM(content_fn=lambda self, prompt: content)

    summary = refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    assert summary["enriched"] == 1
    entry = _entry(_load_brain(tmp_path), "app/mod")
    assert entry["business_domain"] == "Payments"


def test_refresh_no_enrich_uses_mechanical_only(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    llm = FakeLLM()
    summary = refresh_repository_brain(str(tmp_path), llm=llm, enrich=False)
    assert summary["enriched"] == 0
    assert llm.calls == 0


def test_refresh_is_idempotent(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    llm = FakeLLM()
    refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    first = _load_brain(tmp_path)
    refresh_repository_brain(str(tmp_path), llm=llm, enrich=True)
    second = _load_brain(tmp_path)
    assert first == second


# ──────────────────────────────────────────────────────────────────────────────
# History bounding
# ──────────────────────────────────────────────────────────────────────────────

def test_bound_descriptions_caps_history_and_keeps_current():
    descriptions = {f"sig{i}": {"business_domain": str(i)} for i in range(12)}
    current = "sig11"
    bounded = _bound_descriptions(descriptions, current, max_keep=8)
    assert len(bounded) == 8
    assert current in bounded


def test_bound_descriptions_forces_current_when_absent():
    descriptions = {f"sig{i}": {"business_domain": str(i)} for i in range(10)}
    descriptions["current"] = {"business_domain": "now"}
    bounded = _bound_descriptions(descriptions, "current", max_keep=3)
    assert "current" in bounded
    assert len(bounded) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap: ensure_repository_brain (new project / re-add)
# ──────────────────────────────────────────────────────────────────────────────

def test_ensure_creates_brain_with_llm_when_missing(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    llm = FakeLLM()

    summary = ensure_repository_brain(str(tmp_path), llm=llm)
    assert summary["bootstrap"] == "created"
    assert summary["enriched"] == 1
    assert llm.calls == 1
    assert (tmp_path / "brain" / "knowledge" / "generated_directory_brain.json").exists()


def test_ensure_skips_when_brain_present(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    ensure_repository_brain(str(tmp_path), llm=FakeLLM())  # first bootstrap

    llm = FakeLLM()
    summary = ensure_repository_brain(str(tmp_path), llm=llm)  # second time
    assert summary["status"] == "exists"
    assert summary["bootstrap"] == "reused"
    assert llm.calls == 0  # no scan, no LLM when knowledge already exists


def test_ensure_regenerates_after_project_removed_and_readded(tmp_path):
    import shutil
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")

    first = ensure_repository_brain(str(tmp_path), llm=FakeLLM())
    assert first["bootstrap"] == "created"

    # Simulate removing the project (its brain/knowledge goes with it).
    shutil.rmtree(tmp_path / "brain")

    llm = FakeLLM()
    second = ensure_repository_brain(str(tmp_path), llm=llm)
    assert second["bootstrap"] == "created"  # re-added project regenerated with LLM
    assert llm.calls == 1


def test_ensure_no_enrich_creates_mechanical_only(tmp_path):
    _write(tmp_path / "app" / "mod" / "thing.py", "class Thing:\n    pass\n")
    llm = FakeLLM()
    summary = ensure_repository_brain(str(tmp_path), llm=llm, enrich=False)
    assert summary["bootstrap"] == "created"
    assert summary["enriched"] == 0
    assert llm.calls == 0


def test_ensure_uses_legacy_brain_when_primary_missing(tmp_path):
    legacy = tmp_path / ".agents" / "repository_brain.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = [{
        "directory_path": "service/api",
        "business_domain": "Legacy",
        "technical_role": "Controller",
        "core_responsibilities": "Legacy brain",
        "intent_terms": ["Api"],
        "key_artifacts": ["service/api/UserController.java"],
    }]
    legacy.write_text(json.dumps(legacy_payload, indent=2), encoding="utf-8")

    llm = FakeLLM()
    summary = ensure_repository_brain(str(tmp_path), llm=llm)
    assert summary["status"] == "exists"
    assert summary["bootstrap"] == "migrated_legacy"
    assert llm.calls == 0

    primary = tmp_path / "brain" / "knowledge" / "generated_directory_brain.json"
    assert primary.exists()
    assert json.loads(primary.read_text(encoding="utf-8")) == legacy_payload

