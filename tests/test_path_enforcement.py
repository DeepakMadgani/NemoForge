"""
Tests for post-plan path enforcement (Task 1) and smart blacklisting (Task 2).

Run with:
  cd aviator-plugin-sample
  python -m pytest tests/test_path_enforcement.py -v
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

# ──────────────────────────────────────────────────────────────────────────
# Lightweight stubs — we test the LOGIC, not the full workflow integration
# ──────────────────────────────────────────────────────────────────────────

class TaskType(Enum):
    MODIFY = "modify"
    CREATE = "create"
    READ_ONLY = "read_only"
    DELETE = "delete"


@dataclass
class FakeTask:
    id: str
    title: str
    description: str
    file_path: str
    task_type: TaskType
    new_file_creation_allowed: bool = False
    dependencies: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════
# TASK 1 TESTS: Post-plan Path Enforcement
# ══════════════════════════════════════════════════════════════════════════

def _run_path_enforcement(tasks, discovered_for_planner, workspace_dir):
    """
    Extract of the exact logic from plan_node() lines 1279-1391.
    Isolated here so we can test without the full workflow.
    """
    discovered_paths = {d["path"].replace("\\", "/") for d in discovered_for_planner}
    discovered_basenames = {}
    for dp in discovered_paths:
        bn = dp.rsplit("/", 1)[-1].lower()
        discovered_basenames.setdefault(bn, []).append(dp)

    ws_root = Path(workspace_dir) if workspace_dir else None

    corrected_tasks = []
    for task in tasks:
        task_norm = task.file_path.replace("\\", "/")
        task_type_val = getattr(task.task_type, "value", str(task.task_type)).lower()

        # READ_ONLY: pass through
        if task_type_val == "read_only":
            corrected_tasks.append(task)
            continue

        # CREATE: smart handling
        if task_type_val == "create":
            if ws_root and (ws_root / task.file_path).exists():
                desc_lower = (task.description or "").lower()
                create_intent_keywords = [
                    "create new", "implement new", "add new file",
                    "new class", "new module", "new service",
                ]
                is_genuine_create = any(kw in desc_lower for kw in create_intent_keywords)

                if is_genuine_create:
                    base, ext = os.path.splitext(task.file_path)
                    new_path = f"{base}_new{ext}"
                    old_path = task.file_path
                    task.file_path = new_path
                    for other_task in tasks:
                        if other_task.id != task.id:
                            if old_path in (other_task.description or ""):
                                other_task.description = other_task.description.replace(old_path, new_path)
                            if old_path in (other_task.dependencies or []):
                                other_task.dependencies = [
                                    new_path if d == old_path else d
                                    for d in other_task.dependencies
                                ]
                else:
                    task.task_type = TaskType.MODIFY
                    task.new_file_creation_allowed = False

            corrected_tasks.append(task)
            continue

        # MODIFY: path MUST exist in discovered files or on disk
        if task_norm in discovered_paths:
            corrected_tasks.append(task)
            continue
        if ws_root and (ws_root / task.file_path).exists():
            corrected_tasks.append(task)
            continue

        # Path is hallucinated — try auto-correction by basename
        task_basename = task_norm.rsplit("/", 1)[-1].lower()
        candidates_for_fix = discovered_basenames.get(task_basename, [])

        if len(candidates_for_fix) == 1:
            task.file_path = candidates_for_fix[0]
            corrected_tasks.append(task)
        elif len(candidates_for_fix) > 1:
            best = max(candidates_for_fix, key=lambda p: next(
                (d.get("confidence", 0) for d in discovered_for_planner
                 if d["path"].replace("\\", "/") == p), 0
            ))
            task.file_path = best
            corrected_tasks.append(task)
        else:
            pass  # dropped

    return corrected_tasks


class TestTask1_ModifyAutoCorrection:
    """MODIFY task with hallucinated path gets auto-corrected to real path."""

    def test_basename_match_single(self, tmp_path):
        """app/app.py -> src/app.py when src/app.py is in discovered files."""
        discovered = [
            {"path": "src/app.py", "confidence": 0.9},
            {"path": "src/models.py", "confidence": 0.8},
        ]
        tasks = [
            FakeTask(id="t1", title="Fix bug", description="Fix the bug in app.py",
                     file_path="app/app.py", task_type=TaskType.MODIFY),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "src/app.py"

    def test_basename_match_ambiguous_picks_best_confidence(self, tmp_path):
        """When multiple files match basename, pick highest confidence."""
        discovered = [
            {"path": "src/app.py", "confidence": 0.5},
            {"path": "lib/app.py", "confidence": 0.9},
        ]
        tasks = [
            FakeTask(id="t1", title="Fix", description="Fix",
                     file_path="wrong/app.py", task_type=TaskType.MODIFY),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "lib/app.py"

    def test_no_basename_match_drops_task(self, tmp_path):
        """MODIFY with completely fake basename gets dropped."""
        discovered = [
            {"path": "src/app.py", "confidence": 0.9},
        ]
        tasks = [
            FakeTask(id="t1", title="Fix", description="Fix",
                     file_path="app/config.py", task_type=TaskType.MODIFY),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 0

    def test_correct_path_passes_through(self, tmp_path):
        """MODIFY with correct discovered path is untouched."""
        discovered = [
            {"path": "src/app.py", "confidence": 0.9},
        ]
        tasks = [
            FakeTask(id="t1", title="Fix", description="Fix",
                     file_path="src/app.py", task_type=TaskType.MODIFY),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "src/app.py"

    def test_path_on_disk_but_not_discovered_passes(self, tmp_path):
        """MODIFY for a file that exists on disk (but not in discovery) still passes."""
        (tmp_path / "extra").mkdir(parents=True, exist_ok=True)
        (tmp_path / "extra" / "helper.py").write_text("# helper")
        discovered = [
            {"path": "src/app.py", "confidence": 0.9},
        ]
        tasks = [
            FakeTask(id="t1", title="Fix", description="Fix",
                     file_path="extra/helper.py", task_type=TaskType.MODIFY),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "extra/helper.py"

    def test_case_insensitive_basename_match(self, tmp_path):
        """Basename matching is case-insensitive."""
        discovered = [
            {"path": "src/App.py", "confidence": 0.9},
        ]
        tasks = [
            FakeTask(id="t1", title="Fix", description="Fix",
                     file_path="wrong/app.py", task_type=TaskType.MODIFY),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "src/App.py"


class TestTask1_CreateSmartHandling:
    """CREATE task smart handling when file exists vs doesn't."""

    def test_create_new_file_passes_through(self, tmp_path):
        """CREATE for a file that doesn't exist -> pass through (genuinely new)."""
        discovered = [{"path": "src/app.py", "confidence": 0.9}]
        tasks = [
            FakeTask(id="t1", title="Create config", description="Create config validator",
                     file_path="src/config_validator.py", task_type=TaskType.CREATE,
                     new_file_creation_allowed=True),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "src/config_validator.py"
        assert result[0].task_type == TaskType.CREATE

    def test_create_existing_file_modification_intent_converts(self, tmp_path):
        """CREATE for existing file with modification intent -> converted to MODIFY."""
        (tmp_path / "requirements.txt").write_text("flask\n")
        discovered = [{"path": "requirements.txt", "confidence": 0.8}]
        tasks = [
            FakeTask(id="t1", title="Add dependency", 
                     description="Add requests dependency to requirements.txt",
                     file_path="requirements.txt", task_type=TaskType.CREATE,
                     new_file_creation_allowed=True),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].task_type == TaskType.MODIFY
        assert result[0].new_file_creation_allowed == False
        assert result[0].file_path == "requirements.txt"

    def test_create_existing_file_genuine_create_renames(self, tmp_path):
        """CREATE for existing file with genuine create intent -> rename to _new."""
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src" / "service.py").write_text("# old service")
        discovered = [{"path": "src/service.py", "confidence": 0.8}]
        tasks = [
            FakeTask(id="t1", title="Create new service",
                     description="Create new service module for notifications",
                     file_path="src/service.py", task_type=TaskType.CREATE,
                     new_file_creation_allowed=True),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "src/service_new.py"
        assert result[0].task_type == TaskType.CREATE

    def test_create_rename_updates_cross_task_references(self, tmp_path):
        """When CREATE gets renamed, other tasks that reference it get updated."""
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src" / "helper.py").write_text("# helper")
        discovered = [{"path": "src/app.py", "confidence": 0.9}]
        tasks = [
            FakeTask(id="t1", title="Create new helper",
                     description="Implement new helper module",
                     file_path="src/helper.py", task_type=TaskType.CREATE,
                     new_file_creation_allowed=True),
            FakeTask(id="t2", title="Update app",
                     description="Import from src/helper.py in the main app",
                     file_path="src/app.py", task_type=TaskType.MODIFY,
                     dependencies=["src/helper.py"]),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))

        # t1 should be renamed
        create_task = next(t for t in result if t.id == "t1")
        assert create_task.file_path == "src/helper_new.py"

        # t2's description and dependencies should be updated
        modify_task = next(t for t in result if t.id == "t2")
        assert "src/helper_new.py" in modify_task.description
        assert "src/helper_new.py" in modify_task.dependencies
        assert "src/helper.py" not in modify_task.dependencies

    def test_read_only_passes_through(self, tmp_path):
        """READ_ONLY tasks always pass through regardless of path."""
        discovered = [{"path": "src/app.py", "confidence": 0.9}]
        tasks = [
            FakeTask(id="t1", title="Read context",
                     description="Read for context",
                     file_path="some/nonexistent/file.py", task_type=TaskType.READ_ONLY),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "some/nonexistent/file.py"


class TestTask1_MixedScenarios:
    """End-to-end scenarios with multiple task types."""

    def test_mixed_correct_hallucinated_create(self, tmp_path):
        """Mix of correct MODIFY, hallucinated MODIFY, and legitimate CREATE."""
        (tmp_path / "config").mkdir(parents=True, exist_ok=True)
        (tmp_path / "config" / "settings.json").write_text('{"key": "val"}')
        discovered = [
            {"path": "src/app.py", "confidence": 0.9},
            {"path": "config/settings.json", "confidence": 0.7},
        ]
        tasks = [
            # Correct MODIFY -> pass through
            FakeTask(id="t1", title="Fix app", description="Fix bug",
                     file_path="src/app.py", task_type=TaskType.MODIFY),
            # Hallucinated MODIFY -> should be dropped (no basename match)
            FakeTask(id="t2", title="Fix config", description="Fix config",
                     file_path="app/config.py", task_type=TaskType.MODIFY),
            # Legitimate CREATE -> new file, pass through
            FakeTask(id="t3", title="Create validator", description="Create config validator",
                     file_path="src/validator.py", task_type=TaskType.CREATE,
                     new_file_creation_allowed=True),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 2  # t1 + t3 (t2 dropped)
        assert result[0].file_path == "src/app.py"       # t1 unchanged
        assert result[1].file_path == "src/validator.py"  # t3 pass through

    def test_hallucinated_path_with_backslashes(self, tmp_path):
        """Handles Windows-style backslashes in task paths."""
        discovered = [{"path": "src/app.py", "confidence": 0.9}]
        tasks = [
            FakeTask(id="t1", title="Fix", description="Fix",
                     file_path="wrong\\app.py", task_type=TaskType.MODIFY),
        ]
        result = _run_path_enforcement(tasks, discovered, str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "src/app.py"

    def test_empty_tasks_list(self, tmp_path):
        """No tasks -> no crash."""
        discovered = [{"path": "src/app.py", "confidence": 0.9}]
        result = _run_path_enforcement([], discovered, str(tmp_path))
        assert result == []

    def test_empty_discovered_list(self, tmp_path):
        """No discovered files -> MODIFY tasks only pass if on disk."""
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src" / "real.py").write_text("# real")
        tasks = [
            FakeTask(id="t1", title="Fix", description="Fix",
                     file_path="src/real.py", task_type=TaskType.MODIFY),
            FakeTask(id="t2", title="Fix", description="Fix",
                     file_path="src/fake.py", task_type=TaskType.MODIFY),
        ]
        result = _run_path_enforcement(tasks, [], str(tmp_path))
        assert len(result) == 1
        assert result[0].file_path == "src/real.py"


# ══════════════════════════════════════════════════════════════════════════
# TASK 2 TESTS: Smart Blacklisting
# ══════════════════════════════════════════════════════════════════════════

def _run_smart_blacklisting(rejected_tasks, accepted_tasks, task_logs, 
                             existing_blacklist, candidate_retry_count, max_retries=2):
    """
    Extract of the smart blacklisting logic from validate_candidates_node()
    lines 1615-1688. Isolated for testing.
    """
    rejected_reasons = {}
    for log_entry in task_logs:
        if log_entry["status"] == "rejected":
            rejected_reasons[log_entry["task"]] = log_entry.get("reason", "Unknown reason")

    immediately_blacklist = []
    for t in rejected_tasks:
        t_path = t.file_path
        reject_reason = rejected_reasons.get(t_path, "")

        is_ungrounded = (
            "completely ungrounded" in reject_reason.lower()
            or "score < 0.05" in reject_reason.lower()
        )
        is_boundary = (
            "architectural" in reject_reason.lower()
            or "boundary" in reject_reason.lower()
        )

        if is_ungrounded:
            has_dependents = False
            for acc in accepted_tasks:
                acc_desc = (acc.description or "").lower()
                acc_deps = acc.dependencies or []
                if t_path.lower() in acc_desc or t_path in acc_deps:
                    has_dependents = True
                    break
            if not has_dependents:
                immediately_blacklist.append(t_path)
        elif is_boundary:
            immediately_blacklist.append(t_path)

    new_blacklist = list(set(existing_blacklist + immediately_blacklist))
    new_retry_count = candidate_retry_count + 1

    if new_retry_count >= max_retries:
        new_blacklist = list(set(new_blacklist + [t.file_path for t in rejected_tasks]))

    return new_blacklist, immediately_blacklist


class TestTask2_UngroundedBlacklisting:
    """Ungrounded files (score < 0.05) with zero role get blacklisted immediately."""

    def test_ungrounded_zero_role_blacklisted_immediately(self):
        """File doesn't exist, no task depends on it -> blacklist now."""
        rejected = [
            FakeTask(id="r1", title="Bad", description="Bad task",
                     file_path="app/config.py", task_type=TaskType.MODIFY),
        ]
        accepted = [
            FakeTask(id="a1", title="Good", description="Fix the main app",
                     file_path="src/app.py", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "app/config.py", "status": "rejected",
             "reason": "File is completely ungrounded (score < 0.05)"},
            {"task": "src/app.py", "status": "accepted", "reason": "OK"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, accepted, task_logs, [], 0, max_retries=2
        )
        assert "app/config.py" in immediate
        assert "app/config.py" in new_bl

    def test_ungrounded_with_dependent_not_blacklisted(self):
        """File doesn't exist, BUT an accepted task references it -> don't blacklist."""
        rejected = [
            FakeTask(id="r1", title="Create helper", description="Create helper",
                     file_path="src/helper.py", task_type=TaskType.CREATE),
        ]
        accepted = [
            FakeTask(id="a1", title="Update app",
                     description="Import from src/helper.py for new functionality",
                     file_path="src/app.py", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "src/helper.py", "status": "rejected",
             "reason": "File is completely ungrounded (score < 0.05)"},
            {"task": "src/app.py", "status": "accepted", "reason": "OK"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, accepted, task_logs, [], 0, max_retries=2
        )
        assert "src/helper.py" not in immediate
        assert "src/helper.py" not in new_bl

    def test_ungrounded_with_dependency_in_deps_list(self):
        """Accepted task has rejected file in its dependencies list -> don't blacklist."""
        rejected = [
            FakeTask(id="r1", title="Create config", description="",
                     file_path="src/config.py", task_type=TaskType.CREATE),
        ]
        accepted = [
            FakeTask(id="a1", title="Update app", description="",
                     file_path="src/app.py", task_type=TaskType.MODIFY,
                     dependencies=["src/config.py"]),
        ]
        task_logs = [
            {"task": "src/config.py", "status": "rejected",
             "reason": "File is completely ungrounded (score < 0.05)"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, accepted, task_logs, [], 0, max_retries=2
        )
        assert "src/config.py" not in immediate


class TestTask2_BoundaryBlacklisting:
    """Architectural boundary violations are always blacklisted immediately."""

    def test_boundary_violation_blacklisted(self):
        rejected = [
            FakeTask(id="r1", title="Cross-boundary", description="Bad",
                     file_path="other-module/Service.java", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "other-module/Service.java", "status": "rejected",
             "reason": "Architectural boundary violation: cannot depend on other-module"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, [], task_logs, [], 0, max_retries=2
        )
        assert "other-module/Service.java" in immediate
        assert "other-module/Service.java" in new_bl

    def test_boundary_blacklisted_even_with_dependents(self):
        """Boundary violations always blacklisted, even if other tasks reference them."""
        rejected = [
            FakeTask(id="r1", title="Bad cross", description="",
                     file_path="other/api.java", task_type=TaskType.MODIFY),
        ]
        accepted = [
            FakeTask(id="a1", title="Consumer", description="Uses other/api.java",
                     file_path="src/main.java", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "other/api.java", "status": "rejected",
             "reason": "Architectural boundary violation"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, accepted, task_logs, [], 0, max_retries=2
        )
        assert "other/api.java" in immediate


class TestTask2_LLMJudgmentDelay:
    """LLM judgment failures delay blacklisting (give retry chance)."""

    def test_llm_judgment_not_immediately_blacklisted(self):
        """File exists but failed contextual LLM check -> NOT blacklisted on first retry."""
        rejected = [
            FakeTask(id="r1", title="Wrong context", description="",
                     file_path="src/utils.py", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "src/utils.py", "status": "rejected",
             "reason": "REJECTED: Mechanical Post-Check Failed (Hallucinated text evidence)"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, [], task_logs, [], 0, max_retries=2
        )
        assert "src/utils.py" not in immediate
        assert "src/utils.py" not in new_bl

    def test_llm_judgment_blacklisted_on_max_retry(self):
        """After max retries exhausted, even LLM judgment failures get blacklisted."""
        rejected = [
            FakeTask(id="r1", title="Wrong context", description="",
                     file_path="src/utils.py", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "src/utils.py", "status": "rejected",
             "reason": "REJECTED: Mechanical Post-Check Failed"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, [], task_logs, [], 1, max_retries=2
        )
        assert "src/utils.py" not in immediate  # not "immediately"
        assert "src/utils.py" in new_bl  # but IS in final blacklist

    def test_unknown_reason_treated_as_llm_judgment(self):
        """Unknown/unrecognized reason -> treated as LLM judgment (delay)."""
        rejected = [
            FakeTask(id="r1", title="Unknown", description="",
                     file_path="src/mystery.py", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "src/mystery.py", "status": "rejected",
             "reason": "Some unknown reason we haven't seen before"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, [], task_logs, [], 0, max_retries=2
        )
        assert "src/mystery.py" not in immediate
        assert "src/mystery.py" not in new_bl


class TestTask2_ExistingBlacklist:
    """Existing blacklist entries are preserved."""

    def test_existing_blacklist_preserved(self):
        rejected = [
            FakeTask(id="r1", title="Bad", description="",
                     file_path="new/bad.py", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "new/bad.py", "status": "rejected",
             "reason": "File is completely ungrounded (score < 0.05)"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, [], task_logs, ["old/blacklisted.py"], 0, max_retries=2
        )
        assert "old/blacklisted.py" in new_bl
        assert "new/bad.py" in new_bl

    def test_no_duplicates_in_blacklist(self):
        """Same file rejected twice doesn't create duplicates."""
        rejected = [
            FakeTask(id="r1", title="Bad", description="",
                     file_path="dup/file.py", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "dup/file.py", "status": "rejected",
             "reason": "File is completely ungrounded (score < 0.05)"},
        ]
        # File already in existing blacklist
        new_bl, _ = _run_smart_blacklisting(
            rejected, [], task_logs, ["dup/file.py"], 0, max_retries=2
        )
        assert new_bl.count("dup/file.py") == 1


class TestTask2_MixedRejections:
    """Multiple rejections with different reasons in the same cycle."""

    def test_mixed_ungrounded_boundary_llm(self):
        """Ungrounded + boundary -> both immediate. LLM judgment -> delayed."""
        rejected = [
            FakeTask(id="r1", title="Ghost", description="",
                     file_path="ghost/file.py", task_type=TaskType.MODIFY),
            FakeTask(id="r2", title="Boundary", description="",
                     file_path="other/service.java", task_type=TaskType.MODIFY),
            FakeTask(id="r3", title="Context", description="",
                     file_path="src/utils.py", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "ghost/file.py", "status": "rejected",
             "reason": "File is completely ungrounded (score < 0.05)"},
            {"task": "other/service.java", "status": "rejected",
             "reason": "Architectural boundary violation"},
            {"task": "src/utils.py", "status": "rejected",
             "reason": "REJECTED: Mechanical Post-Check Failed"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            rejected, [], task_logs, [], 0, max_retries=2
        )
        assert "ghost/file.py" in immediate
        assert "other/service.java" in immediate
        assert "src/utils.py" not in immediate
        assert "ghost/file.py" in new_bl
        assert "other/service.java" in new_bl
        assert "src/utils.py" not in new_bl  # retry 1 < max 2

    def test_empty_rejected_list(self):
        """No rejected tasks -> empty blacklist additions."""
        new_bl, immediate = _run_smart_blacklisting(
            [], [], [], [], 0, max_retries=2
        )
        assert immediate == []
        assert new_bl == []

    def test_all_accepted_no_rejected(self):
        """All tasks accepted, no rejections -> no change."""
        accepted = [
            FakeTask(id="a1", title="Good", description="",
                     file_path="src/app.py", task_type=TaskType.MODIFY),
        ]
        task_logs = [
            {"task": "src/app.py", "status": "accepted", "reason": "OK"},
        ]
        new_bl, immediate = _run_smart_blacklisting(
            [], accepted, task_logs, ["old.py"], 0, max_retries=2
        )
        assert immediate == []
        assert new_bl == ["old.py"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
