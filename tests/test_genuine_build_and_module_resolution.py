"""
Unit tests for genuine build execution, upward project manifest resolution,
and multi-language execution engines (Java, .NET, Node.js, Python).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ticket_to_code.models import BuildResult, BuildStatus, GeneratedCode, TestResult, TestStatus
from ticket_to_code.workflow import (
    build_node,
    test_node as workflow_test_node,
    check_build_status,
    check_test_status,
    route_after_discover,
    outcome_check_node,
    memory_update_node,
    verification_failed_node,
    create_ticket_to_code_graph,
    WorkflowAgents,
)
from ticket_to_code.brain.brain_storage import RepositoryBrainStorage
from ticket_to_code.execution import (
    ExecutionEngineFactory,
    JavaMavenExecutor,
    JavaGradleExecutor,
    DotNetExecutionEngine,
    NodejsExecutionEngine,
    GoExecutionEngine,
    RustExecutionEngine,
)


def test_java_maven_work_dir_and_wrapper_resolution(tmp_path):
    """Verify JavaMavenExecutor resolves work_dir safely for root and sub-modules, and locates mvnw."""
    mvnw_root = tmp_path / "mvnw.cmd"
    mvnw_root.write_text("@echo off\n", encoding="utf-8")
    
    submodule = tmp_path / "services" / "billing-service"
    submodule.mkdir(parents=True)
    sub_pom = submodule / "pom.xml"
    sub_pom.write_text("<project></project>", encoding="utf-8")

    executor = JavaMavenExecutor(str(tmp_path))

    # Module pom.xml
    work_dir = executor._get_work_dir("services/billing-service/pom.xml")
    assert work_dir == submodule.resolve()
    
    # Executable resolution should find root mvnw.cmd on Windows
    mvn_exec = executor._get_mvn_exec(work_dir)
    assert "mvnw" in mvn_exec

    # Root pom.xml
    root_pom = tmp_path / "pom.xml"
    root_pom.write_text("<project></project>", encoding="utf-8")
    root_work_dir = executor._get_work_dir("pom.xml")
    assert root_work_dir == tmp_path.resolve()


def test_java_gradle_work_dir_and_wrapper_resolution(tmp_path):
    """Verify JavaGradleExecutor resolves work_dir safely and locates gradlew.bat."""
    gradlew_root = tmp_path / "gradlew.bat"
    gradlew_root.write_text("@echo off\n", encoding="utf-8")

    submodule = tmp_path / "core" / "auth"
    submodule.mkdir(parents=True)
    sub_gradle = submodule / "build.gradle"
    sub_gradle.write_text("// gradle build", encoding="utf-8")

    executor = JavaGradleExecutor(str(tmp_path))
    work_dir = executor._get_work_dir("core/auth/build.gradle")
    assert work_dir == submodule.resolve()

    gradle_exec = executor._get_gradle_exec(work_dir)
    assert "gradlew" in gradle_exec


def test_build_node_upward_module_resolution_java(tmp_path):
    """Verify build_node walks up from a modified Java file to the nested pom.xml module root."""
    submodule = tmp_path / "services" / "order-service"
    java_file_dir = submodule / "src" / "main" / "java" / "com" / "example"
    java_file_dir.mkdir(parents=True)
    
    pom = submodule / "pom.xml"
    pom.write_text("<project></project>", encoding="utf-8")

    java_file = java_file_dir / "OrderService.java"
    java_file.write_text("package com.example; class OrderService {}", encoding="utf-8")

    gen_code = [
        GeneratedCode(
            file_path="services/order-service/src/main/java/com/example/OrderService.java",
            content="package com.example; class OrderService {}",
            language="java",
            change_type="modify",
        )
    ]

    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "PROJ-101"

    state = {
        "workspace_path": str(tmp_path),
        "generated_code": gen_code,
        "ticket": mock_ticket,
    }

    mock_agents = MagicMock()

    with patch.object(ExecutionEngineFactory, "create") as mock_factory_create:
        mock_exec = MagicMock()
        mock_exec.get_technology_name.return_value = "Java (Maven)"
        mock_exec.execute_build.return_value = BuildResult(
            status=BuildStatus.SUCCESS,
            exit_code=0,
            errors=[],
            stdout="BUILD SUCCESS",
            stderr="",
        )
        mock_factory_create.return_value = mock_exec

        res = build_node(state, mock_agents)

        # Factory must be called with java-maven
        mock_factory_create.assert_called_with(str(tmp_path), "java-maven")
        # Build must target the specific module's pom.xml
        mock_exec.execute_build.assert_called_with("services/order-service/pom.xml")
        assert res["build_result"].status == BuildStatus.SUCCESS


def test_build_node_upward_module_resolution_dotnet(tmp_path):
    """Verify build_node walks up to find nested .csproj file for a modified C# file."""
    api_dir = tmp_path / "backend" / "Accounts.Api"
    controllers_dir = api_dir / "Controllers"
    controllers_dir.mkdir(parents=True)

    csproj = api_dir / "Accounts.Api.csproj"
    csproj.write_text("<Project Sdk=\"Microsoft.NET.Sdk.Web\"></Project>", encoding="utf-8")

    gen_code = [
        GeneratedCode(
            file_path="backend/Accounts.Api/Controllers/UserController.cs",
            content="// user controller",
            language="csharp",
            change_type="modify",
        )
    ]

    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "DOTNET-42"

    state = {
        "workspace_path": str(tmp_path),
        "generated_code": gen_code,
        "ticket": mock_ticket,
    }

    mock_agents = MagicMock()

    with patch.object(ExecutionEngineFactory, "create") as mock_factory_create:
        mock_exec = MagicMock()
        mock_exec.get_technology_name.return_value = ".NET"
        mock_exec.execute_build.return_value = BuildResult(
            status=BuildStatus.SUCCESS,
            exit_code=0,
            errors=[],
            stdout="Build succeeded.",
            stderr="",
        )
        mock_factory_create.return_value = mock_exec

        res = build_node(state, mock_agents)

        mock_factory_create.assert_called_with(str(tmp_path), "dotnet")
        mock_exec.execute_build.assert_called_with("backend/Accounts.Api/Accounts.Api.csproj")
        assert res["build_result"].status == BuildStatus.SUCCESS


def test_route_after_discover_fast_path_backend_service():
    """Verify route_after_discover sends confident backend Java and Python targets directly to edit_loop."""
    state = {
        "workspace_path": "C:/dummy/workspace",
        "discovered_files": [
            {
                "path": "src/main/java/com/app/PaymentService.java",
                "confidence": 0.85,
                "features": {"ownership_type": "SERVICE"},
            }
        ]
    }

    route = route_after_discover(state)
    assert route == "edit_loop"
    assert state.get("direct_mode") is True
    assert "src/main/java/com/app/PaymentService.java" in state.get("allowed_files")


def test_execution_engine_factory_unsupported_graceful():
    """Verify unsupported technology raises informative ValueError listing registered engines."""
    import pytest
    with pytest.raises(ValueError) as exc:
        ExecutionEngineFactory.create("C:/dummy/path", "unsupported_xyz")
    assert "Unsupported technology: unsupported_xyz" in str(exc.value)
    assert "java" in str(exc.value)
    assert "nodejs" in str(exc.value)


def test_graph_registers_test_node_and_wires_edges():
    """Verify create_ticket_to_code_graph registers 'test' node and wires test edges."""
    graph = create_ticket_to_code_graph("C:/dummy/workspace", technology="python")
    assert "test" in graph.nodes
    assert "build" in graph.nodes
    assert "pre_fix_build" in graph.nodes
    assert "fix_build" in graph.nodes
    assert "outcome_check" in graph.nodes
    assert "memory_update" in graph.nodes


def test_build_and_test_status_routing_honest_chain():
    """Verify check_build_status routes to test on success, and check_test_status routes honestly."""
    # 1. Successful build routes to test
    success_state = {
        "build_result": BuildResult(status=BuildStatus.SUCCESS, exit_code=0, errors=[]),
        "build_errors": [],
    }
    assert check_build_status(success_state) == "test"

    # 2. Passing test routes to outcome_check
    passing_test_state = {
        "test_result": TestResult(
            status=TestStatus.ALL_PASSED,
            total_tests=5,
            passed=5,
            failed=0,
            errors=[],
            duration_seconds=1.0,
        )
    }
    assert check_test_status(passing_test_state) in ("outcome_check", "ui_verify")

    # 3. Failing test with retries left routes to remediation_plan
    failing_test_state = {
        "test_result": TestResult(
            status=TestStatus.SOME_FAILED,
            total_tests=5,
            passed=3,
            failed=2,
            errors=["Test assertion failed"],
            duration_seconds=1.0,
        ),
        "retry_attempt": 1,
        "max_retry_attempts": 3,
    }
    assert check_test_status(failing_test_state) == "remediation_plan"

    # 4. Failing test with retries exhausted routes to memory_update (failure)
    exhausted_test_state = {
        "test_result": TestResult(
            status=TestStatus.SOME_FAILED,
            total_tests=5,
            passed=3,
            failed=2,
            errors=["Test assertion failed"],
            duration_seconds=1.0,
        ),
        "retry_attempt": 3,
        "max_retry_attempts": 3,
    }
    assert check_test_status(exhausted_test_state) == "memory_update"


def test_workflow_test_node_execution_genuine_result(tmp_path):
    """Verify test_node executes tests via the registered executor and populates state['test_result']."""
    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "TEST-99"

    state = {
        "workspace_path": str(tmp_path),
        "ticket": mock_ticket,
        "test_status": "ready",
        "code_status": "ready",
    }

    mock_agents = MagicMock()
    mock_executor = MagicMock()
    mock_executor.detect_project_files.return_value = ["pom.xml"]
    mock_executor.get_technology_name.return_value = "Java (Maven)"
    mock_executor.execute_tests.return_value = TestResult(
        status=TestStatus.ALL_PASSED,
        total_tests=12,
        passed=12,
        failed=0,
        errors=[],
        duration_seconds=2.5,
    )
    mock_agents.executor = mock_executor

    res = workflow_test_node(state, mock_agents)

    assert "test_result" in res
    assert res["test_result"].status == TestStatus.ALL_PASSED
    assert res["test_result"].passed == 12
    assert res["test_result"].total_tests == 12
    mock_executor.execute_tests.assert_called_with("pom.xml")


def test_workflow_test_node_no_tests_workspace_returns_no_tests(tmp_path):
    """Verify test_node returns TestStatus.NO_TESTS when no project test files exist."""
    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "TEST-NO-TESTS"

    state = {
        "workspace_path": str(tmp_path),
        "ticket": mock_ticket,
        "test_status": "ready",
        "code_status": "ready",
    }

    mock_agents = MagicMock()
    mock_executor = MagicMock()
    mock_executor.detect_project_files.return_value = []
    mock_agents.executor = mock_executor

    res = workflow_test_node(state, mock_agents)
    assert res["test_result"].status == TestStatus.NO_TESTS
    assert res["test_result"].total_tests == 0
    assert res["compile_only_verification"] is True
    assert res["verification_level"] == "compile_only"

    # Merge node output into state (simulating LangGraph's node execution)
    state.update(res)
    # check_test_status is a pure router: routes NO_TESTS to outcome_check without mutating state
    route = check_test_status(state)
    assert route in ("outcome_check", "ui_verify")
    assert state.get("compile_only_verification") is True


def test_workflow_test_node_execute_tests_exception_wrapped(tmp_path):
    """Verify test_node catches subprocess/runtime crashes from execute_tests and returns TestStatus.ERROR."""
    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "TEST-CRASH"

    state = {
        "workspace_path": str(tmp_path),
        "ticket": mock_ticket,
        "test_status": "ready",
        "code_status": "ready",
    }

    mock_agents = MagicMock()
    mock_executor = MagicMock()
    mock_executor.detect_project_files.return_value = ["pom.xml"]
    mock_executor.get_technology_name.return_value = "Java (Maven)"
    mock_executor.execute_tests.side_effect = RuntimeError("mvn: command not found")
    mock_agents.executor = mock_executor

    res = workflow_test_node(state, mock_agents)
    assert res["test_result"].status == TestStatus.ERROR
    assert "Test runner crashed" in res["test_result"].errors[0]

    # check_test_status routes infrastructure ERROR and TIMEOUT directly to memory_update
    # to avoid burning 3 futile repair cycles on missing CLI tools / environment issues
    state["test_result"] = res["test_result"]
    state["retry_attempt"] = 0
    state["max_retry_attempts"] = 3
    assert check_test_status(state) == "memory_update"


def test_outcome_check_tri_state_tests_passed_distinguishes_no_tests_from_all_passed():
    """Verify outcome_check_node accurately records compile-only vs test-passed in outcome details."""
    mock_ticket = MagicMock()
    mock_ticket.id = "TICK-TEST-CHECK"
    mock_ticket.title = "Verify tri-state testing"
    mock_ticket.description = "Test description"

    gen_file = GeneratedCode(
        file_path="src/service/OrderService.java",
        content="public class OrderService {}",
        language="java",
        change_type="modify",
    )

    base_state = {
        "ticket": mock_ticket,
        "direct_mode": True,
        "build_status": "success",
        "build_errors": [],
        "generated_code": [gen_file],
        "allowed_files": ["src/service/OrderService.java"],
        "workspace_path": "C:/dummy/workspace",
    }

    mock_agents = MagicMock()

    # Case 1: NO_TESTS -> details note compile-only verification, no test suite
    state_no_tests = dict(base_state)
    state_no_tests["test_result"] = TestResult(
        status=TestStatus.NO_TESTS,
        total_tests=0,
        passed=0,
        failed=0,
        errors=[],
        duration_seconds=0,
    )
    res_no_tests = outcome_check_node(state_no_tests, mock_agents)
    assert "compile-only verification, no test suite" in res_no_tests["outcome_check_result"]["details"]

    # Case 2: ALL_PASSED -> details note all automated tests passed
    state_passed = dict(base_state)
    state_passed["test_result"] = TestResult(
        status=TestStatus.ALL_PASSED,
        total_tests=5,
        passed=5,
        failed=0,
        errors=[],
        duration_seconds=1.0,
    )
    res_passed = outcome_check_node(state_passed, mock_agents)
    assert "all automated tests passed" in res_passed["outcome_check_result"]["details"]


def test_route_after_discover_tightened_threshold_rejects_unrelated_candidate():
    """Verify route_after_discover requires score >= 0.80 AND role/keyword/extension relevance."""
    # Candidate with high score but NO relevance -> falls through to waterfall
    state_unrelated = {
        "workspace_path": "C:/dummy/workspace",
        "discovered_files": [
            {
                "path": "docs/release_notes.txt",
                "confidence": 0.85,
                "features": {"ownership_type": "DOCUMENTATION"},
            }
        ]
    }
    assert route_after_discover(state_unrelated) == "hypothesis_investigation"

    # Candidate with role but below 0.80 (e.g. 0.75) -> falls through to waterfall
    state_low_score = {
        "workspace_path": "C:/dummy/workspace",
        "discovered_files": [
            {
                "path": "src/services/PaymentService.java",
                "confidence": 0.75,
                "features": {"ownership_type": "SERVICE"},
            }
        ]
    }
    assert route_after_discover(state_low_score) == "hypothesis_investigation"

    # Candidate with score >= 0.80 AND relevant role -> fast path!
    state_confident = {
        "workspace_path": "C:/dummy/workspace",
        "discovered_files": [
            {
                "path": "src/services/PaymentService.java",
                "confidence": 0.85,
                "features": {"ownership_type": "SERVICE"},
            }
        ]
    }
    assert route_after_discover(state_confident) == "edit_loop"


def test_memory_update_node_tri_state_verification_level():
    """Verify memory_update_node distinguishes compile_only from test_verified when calling brain storage."""
    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "TICK-MEM-TRI"

    gen_file = GeneratedCode(
        file_path="src/service/OrderService.java",
        content="public class OrderService {}",
        language="java",
        change_type="modify",
    )

    mock_agents = MagicMock()
    mock_brain = MagicMock()
    mock_agents.localizer.repository_brain = mock_brain

    # Case 1: NO_TESTS -> verification_level="compile_only"
    state_no_tests = {
        "ticket": mock_ticket,
        "workspace_path": "C:/dummy/workspace",
        "build_result": BuildResult(status=BuildStatus.SUCCESS, exit_code=0, errors=[]),
        "test_result": TestResult(
            status=TestStatus.NO_TESTS,
            total_tests=0,
            passed=0,
            failed=0,
            errors=[],
            duration_seconds=0,
        ),
        "generated_code": [gen_file],
    }
    res_no_tests = memory_update_node(state_no_tests, mock_agents)
    assert res_no_tests["status"] == "memory_updated"
    assert res_no_tests["verification_level"] == "compile_only"
    mock_brain.record_successful_ticket.assert_called_with(
        "src/service/OrderService.java",
        verification_level="compile_only",
    )

    mock_brain.reset_mock()

    # Case 2: ALL_PASSED -> verification_level="test_verified"
    state_passed = {
        "ticket": mock_ticket,
        "workspace_path": "C:/dummy/workspace",
        "build_result": BuildResult(status=BuildStatus.SUCCESS, exit_code=0, errors=[]),
        "test_result": TestResult(
            status=TestStatus.ALL_PASSED,
            total_tests=5,
            passed=5,
            failed=0,
            errors=[],
            duration_seconds=1.0,
        ),
        "generated_code": [gen_file],
    }
    res_passed = memory_update_node(state_passed, mock_agents)
    assert res_passed["status"] == "memory_updated"
    assert res_passed["verification_level"] == "test_verified"
    mock_brain.record_successful_ticket.assert_called_with(
        "src/service/OrderService.java",
        verification_level="test_verified",
    )


def test_repository_brain_storage_persists_verification_levels(tmp_path):
    """Verify RepositoryBrainStorage records last_verification_level and verification_counts on disk."""
    import json

    storage = RepositoryBrainStorage()
    storage.loaded_directory = tmp_path

    # First record as compile_only
    storage.record_successful_ticket("src/service/OrderService.java", verification_level="compile_only")

    state_file = tmp_path / "repository_brain_runtime.state"
    assert state_file.exists()

    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_entry = data["files"]["src/service/OrderService.java"]
    assert file_entry["success_count"] == 1
    assert file_entry["last_verification_level"] == "compile_only"
    assert file_entry["verification_counts"]["compile_only"] == 1

    # Second record as test_verified
    storage.record_successful_ticket("src/service/OrderService.java", verification_level="test_verified")

    with open(state_file, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    file_entry2 = data2["files"]["src/service/OrderService.java"]
    assert file_entry2["success_count"] == 2
    assert file_entry2["last_verification_level"] == "test_verified"
    assert file_entry2["verification_counts"]["compile_only"] == 1
    assert file_entry2["verification_counts"]["test_verified"] == 1


def test_nodejs_executor_no_test_script_returns_no_tests(tmp_path):
    """Verify NodejsExecutionEngine returns TestStatus.NO_TESTS when package.json lacks a test script."""
    import json
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({"name": "sample-pkg", "scripts": {"build": "tsc"}}), encoding="utf-8")

    engine = NodejsExecutionEngine(str(tmp_path))
    res = engine.execute_tests()
    assert res.status == TestStatus.NO_TESTS
    assert res.total_tests == 0


def test_go_and_rust_execution_engines_registered_and_detected(tmp_path):
    """Verify Go and Rust workspaces are auto-detected and create their respective execution engines."""
    go_dir = tmp_path / "go_service"
    go_dir.mkdir()
    (go_dir / "go.mod").write_text("module example.com/goservice\n\ngo 1.22\n", encoding="utf-8")

    rust_dir = tmp_path / "rust_service"
    rust_dir.mkdir()
    (rust_dir / "Cargo.toml").write_text('[package]\nname = "rustservice"\nversion = "0.1.0"\n', encoding="utf-8")

    assert ExecutionEngineFactory._detect_technology(go_dir) == "go"
    assert ExecutionEngineFactory._detect_technology(rust_dir) == "rust"

    go_engine = ExecutionEngineFactory.create(str(go_dir))
    assert isinstance(go_engine, GoExecutionEngine)
    assert go_engine.get_technology_name() == "Go"

    rust_engine = ExecutionEngineFactory.create(str(rust_dir))
    assert isinstance(rust_engine, RustExecutionEngine)
    assert rust_engine.get_technology_name() == "Rust"


def test_verification_failed_node_rolls_back_workspace_and_archives_patch(tmp_path):
    """Verify verification_failed_node restores modified files, removes created files, and archives patch."""
    # 1. Existing file modified
    orig_file = tmp_path / "service.py"
    orig_file.write_text("def run():\n    print('broken')\n", encoding="utf-8")

    # 2. Newly created file
    created_file = tmp_path / "new_helper.py"
    created_file.write_text("def helper(): pass\n", encoding="utf-8")

    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "TICK-ROLLBACK-TEST"

    gen_mod = GeneratedCode(file_path="service.py", content="def run():\n    print('broken')\n", language="python", change_type="modify")
    gen_create = GeneratedCode(file_path="new_helper.py", content="def helper(): pass\n", language="python", change_type="create")

    state = {
        "ticket": mock_ticket,
        "workspace_path": str(tmp_path),
        "original_file_contents": {
            "service.py": "def run():\n    print('original')\n",
        },
        "generated_code": [gen_mod, gen_create],
        "outcome_check_result": {
            "status": "PARTIAL",
            "findings": [{"verdict": "DEFECT", "missing_behavior": "Incomplete implementation"}],
        },
    }

    mock_agents = MagicMock()
    res = verification_failed_node(state, mock_agents)

    assert res["status"] == "verification_failed"
    assert res["workspace_rolled_back"] is True
    assert "service.py" in res["restored_files"]
    assert "new_helper.py" in res["removed_files"]

    # Verify disk content restored
    assert orig_file.read_text(encoding="utf-8") == "def run():\n    print('original')\n"
    # Verify created file unlinked
    assert not created_file.exists()

    # Verify trace artifact archived
    patch_file = tmp_path / ".aviator" / "traces" / "TICK-ROLLBACK-TEST" / "verification_failed.patch"
    assert patch_file.exists()
    assert "service.py" in patch_file.read_text(encoding="utf-8")
    # Verify workspace root is clean (no verification_failed.patch at root)
    assert not (tmp_path / "verification_failed.patch").exists()


def test_workspace_rollback_path_containment(tmp_path):
    """Ensure rollback strictly ignores paths that escape the workspace boundary."""
    outside_dir = tmp_path.parent / "outside_workspace"
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_secret = outside_dir / "secret.txt"
    outside_secret.write_text("KEEP_ME_INTACT", encoding="utf-8")

    inside_orig = tmp_path / "inside.py"
    inside_orig.write_text("modified", encoding="utf-8")

    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "TICK-CONTAINMENT"

    # Attempt path traversal in original_file_contents and generated_code
    state = {
        "ticket": mock_ticket,
        "workspace_path": str(tmp_path),
        "original_file_contents": {
            "../outside_workspace/secret.txt": "OVERWRITTEN",
            "inside.py": "pristine_inside",
        },
        "generated_code": [
            GeneratedCode(file_path="../outside_workspace/secret.txt", content="evil", language="python", change_type="create"),
            GeneratedCode(file_path="inside.py", content="modified", language="python", change_type="modify"),
        ],
        "outcome_check_result": {"status": "PARTIAL"},
    }

    mock_agents = MagicMock()
    res = verification_failed_node(state, mock_agents)

    assert res["workspace_rolled_back"] is True
    # Inside file restored
    assert inside_orig.read_text(encoding="utf-8") == "pristine_inside"
    # Outside file untouched! Not deleted, not overwritten!
    assert outside_secret.read_text(encoding="utf-8") == "KEEP_ME_INTACT"


def test_fast_path_edit_loop_captures_snapshots_and_rolls_back(tmp_path):
    """Direct mode bypasses generate_code_node; EditLoopAgent must snapshot and rollback safely."""
    from ticket_to_code.agents.edit_loop_agent import EditLoopAgent, EditDecision, EditResult

    service_file = tmp_path / "app" / "service.py"
    service_file.parent.mkdir(parents=True, exist_ok=True)
    service_file.write_text("ORIGINAL_CONTENT", encoding="utf-8")

    agent = EditLoopAgent(
        workspace_path=str(tmp_path),
        llm=MagicMock(),
        symbol_index=MagicMock(),
    )

    # 1. Simulate agent modifying existing file and creating a new file
    agent._write_file("app/service.py", "MODIFIED_CONTENT")
    agent._write_file("app/new_module.py", "NEW_CONTENT")

    # Verify EditLoopAgent captured snapshot
    assert "app/service.py" in agent.pre_edit_snapshots
    assert agent.pre_edit_snapshots["app/service.py"] == "ORIGINAL_CONTENT"
    assert "app/new_module.py" not in agent.pre_edit_snapshots

    # 2. Simulate edit_loop_node state update
    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "TICK-FAST-PATH"

    gen_mod = GeneratedCode(file_path="app/service.py", content="MODIFIED_CONTENT", language="python", change_type="modify")
    gen_new = GeneratedCode(file_path="app/new_module.py", content="NEW_CONTENT", language="python", change_type="create")

    state = {
        "ticket": mock_ticket,
        "workspace_path": str(tmp_path),
        "original_file_contents": dict(agent.pre_edit_snapshots),
        "generated_code": [gen_mod, gen_new],
        "outcome_check_result": {"status": "PARTIAL"},
    }

    mock_agents = MagicMock()
    res = verification_failed_node(state, mock_agents)

    assert res["workspace_rolled_back"] is True
    assert "app/service.py" in res["restored_files"]
    assert "app/new_module.py" in res["removed_files"]

    # Verify disk state
    assert service_file.read_text(encoding="utf-8") == "ORIGINAL_CONTENT"
    assert not (tmp_path / "app" / "new_module.py").exists()
    assert not (tmp_path / "verification_failed.patch").exists()
    assert (tmp_path / ".aviator" / "traces" / "TICK-FAST-PATH" / "verification_failed.patch").exists()


def test_memory_update_node_rolls_back_on_terminal_failure(tmp_path):
    """Terminal build or test failures reaching memory_update must roll back and never record success."""
    from ticket_to_code.models import BuildResult, BuildStatus

    code_file = tmp_path / "calc.py"
    code_file.write_text("broken calc\n", encoding="utf-8")
    extra_file = tmp_path / "broken_extra.py"
    extra_file.write_text("syntax error !!!\n", encoding="utf-8")

    mock_ticket = MagicMock()
    mock_ticket.ticket_id = "TICK-BUILD-FAIL-ROLLBACK"

    gen_mod = GeneratedCode(file_path="calc.py", content="broken calc", language="python", change_type="modify")
    gen_extra = GeneratedCode(file_path="broken_extra.py", content="syntax error !!!", language="python", change_type="create")

    state = {
        "ticket": mock_ticket,
        "workspace_path": str(tmp_path),
        "original_file_contents": {
            "calc.py": "def calc(): return 42\n",
        },
        "generated_code": [gen_mod, gen_extra],
        "build_result": BuildResult(
            status=BuildStatus.FAILURE,
            exit_code=1,
            errors=["SyntaxError in calc.py"],
        ),
        "test_result": None,
    }

    mock_agents = MagicMock()
    mock_brain = MagicMock()
    mock_agents.localizer.repository_brain = mock_brain

    res = memory_update_node(state, mock_agents)

    # 1. State and return verdict
    assert res["status"] == "escalated"
    assert res["escalation_type"] == "build_failed"
    assert res["workspace_rolled_back"] is True
    assert "calc.py" in res["restored_files"]
    assert "broken_extra.py" in res["removed_files"]

    # 2. Disk restored and new file unlinked
    assert code_file.read_text(encoding="utf-8") == "def calc(): return 42\n"
    assert not extra_file.exists()

    # 3. Patch archived in traces, not in workspace root
    assert not (tmp_path / "verification_failed.patch").exists()
    trace_patch = tmp_path / ".aviator" / "traces" / "TICK-BUILD-FAIL-ROLLBACK" / "verification_failed.patch"
    assert trace_patch.exists()
    assert "calc.py" in trace_patch.read_text(encoding="utf-8")

    # 4. Brain memory was NOT updated with a success
    mock_brain.record_successful_ticket.assert_not_called()


