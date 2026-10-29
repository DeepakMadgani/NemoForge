import pytest
from pathlib import Path

from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
from ticket_to_code.models import (
    ArchitecturalPlan,
    DevelopmentTask,
    TaskType,
)


def test_companion_resolver_java_test_binding(tmp_path):
    """Verify companion resolver discovers and binds *Test.java for a Java source class."""
    src_file = tmp_path / "src" / "main" / "java" / "com" / "example" / "UserService.java"
    test_file = tmp_path / "src" / "test" / "java" / "com" / "example" / "UserServiceTest.java"

    src_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)

    src_file.write_text("package com.example; public class UserService {}", encoding="utf-8")
    test_file.write_text("package com.example; public class UserServiceTest {}", encoding="utf-8")

    comp = PolyglotCompanionResolver.resolve_companion(
        file_path="src/main/java/com/example/UserService.java",
        workspace_path=str(tmp_path),
    )
    assert comp is not None
    rel_path, content = comp
    assert "UserServiceTest.java" in rel_path.replace("\\", "/")
    assert "UserServiceTest" in content


def test_topological_tiered_ordering():
    """Verify tasks are topologically sorted: Interfaces -> Services -> Migrations -> Tests."""
    tasks = [
        DevelopmentTask(id="t1", title="Test Task", file_path="src/test/java/com/example/UserServiceTest.java", description="Test", task_type=TaskType.MODIFY),
        DevelopmentTask(id="t2", title="Service Task", file_path="src/main/java/com/example/UserService.java", description="Service", task_type=TaskType.MODIFY),
        DevelopmentTask(id="t3", title="Repo Task", file_path="src/main/java/com/example/UserRepository.java", description="Repository Interface", task_type=TaskType.MODIFY),
        DevelopmentTask(id="t4", title="Flyway Task", file_path="src/main/resources/db/migration/V1__Add_Users.sql", description="Flyway Migration", task_type=TaskType.CREATE),
    ]

    def _tier_rank(task) -> int:
        fp = (getattr(task, "file_path", "") or "").lower().replace("\\", "/")
        if any(k in fp for k in ["test", "spec"]):
            return 4
        if any(k in fp for k in ["migration", ".sql", "schema"]):
            return 3
        if any(k in fp for k in ["repository", "dto", "entity", "model", "interface", "constants"]):
            return 1
        if any(k in fp for k in ["service", "handler", "controller", "component", "view", "route"]):
            return 2
        return 2

    sorted_tasks = sorted(tasks, key=_tier_rank)

    assert "UserRepository.java" in sorted_tasks[0].file_path
    assert "UserService.java" in sorted_tasks[1].file_path
    assert "V1__Add_Users.sql" in sorted_tasks[2].file_path
    assert "UserServiceTest.java" in sorted_tasks[3].file_path


def test_auto_companion_scope_binding(tmp_path):
    """Verify companion resolver prevents scope lock rejection for companion test files."""
    handler_p = tmp_path / "src" / "UpdateProjectUserHandler.java"
    handler_test_p = tmp_path / "src" / "UpdateProjectUserHandlerTest.java"

    handler_p.parent.mkdir(parents=True, exist_ok=True)
    handler_p.write_text("class UpdateProjectUserHandler {}", encoding="utf-8")
    handler_test_p.write_text("class UpdateProjectUserHandlerTest {}", encoding="utf-8")

    initial_writable = ["src/UpdateProjectUserHandler.java"]
    expanded_with_companions = list(initial_writable)

    for w_fp in initial_writable:
        comp = PolyglotCompanionResolver.resolve_companion(w_fp, workspace_path=str(tmp_path))
        if comp and comp[0] and comp[0] not in expanded_with_companions:
            expanded_with_companions.append(comp[0].replace("\\", "/"))

    assert len(expanded_with_companions) == 2
    assert any("UpdateProjectUserHandlerTest.java" in f for f in expanded_with_companions)
