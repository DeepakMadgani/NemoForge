"""
Unit and Integration Tests for Universal Hardening and Generalization Upgrades.

Verifies:
1. PolyglotBuildDiscoverer across Java, Python, Go, Rust, .NET, Make.
2. GitWorktreeSandbox ephemeral isolation and cleanup.
3. BlastRadiusAnalyzer cross-service contract impact detection.
4. Tolerant code replacement across CRLF/LF line endings and whitespace indentation mismatch.
5. CompilerGate polyglot error parsing across Python, Go, Rust, and .NET.
6. CompilerGate baseline disk persistence roundtrip.
"""

import json
from pathlib import Path

import pytest
from ticket_to_code.runtime.agent_harness import HostEnvironmentTools
from ticket_to_code.runtime.blast_radius import BlastRadiusAnalyzer
from ticket_to_code.runtime.compiler_gate import CompilerGate
from ticket_to_code.runtime.git_sandbox import GitWorktreeSandbox
from ticket_to_code.runtime.polyglot_build import PolyglotBuildDiscoverer
from ticket_to_code.runtime.tool_registry import UnifiedToolRegistry


def test_polyglot_build_discoverer_python(tmp_path: Path):
    py_proj = tmp_path / "python_service"
    py_proj.mkdir()
    (py_proj / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")

    cfg = PolyglotBuildDiscoverer.discover(py_proj, tmp_path)
    assert cfg is not None
    assert cfg.ecosystem == "python"
    assert "pytest" in cfg.test_cmd


def test_polyglot_build_discoverer_rust(tmp_path: Path):
    rust_proj = tmp_path / "rust_cli"
    rust_proj.mkdir()
    (rust_proj / "Cargo.toml").write_text("[package]\nname='rust_cli'\n", encoding="utf-8")

    cfg = PolyglotBuildDiscoverer.discover(rust_proj, tmp_path)
    assert cfg is not None
    assert cfg.ecosystem == "rust"
    assert cfg.compile_cmd == "cargo check --quiet"


def test_polyglot_build_discoverer_go(tmp_path: Path):
    go_proj = tmp_path / "go_service"
    go_proj.mkdir()
    (go_proj / "go.mod").write_text("module example.com/goservice\n", encoding="utf-8")

    cfg = PolyglotBuildDiscoverer.discover(go_proj, tmp_path)
    assert cfg is not None
    assert cfg.ecosystem == "go"
    assert "go build" in cfg.compile_cmd


def test_polyglot_build_discoverer_dotnet(tmp_path: Path):
    dotnet_proj = tmp_path / "dotnet_service"
    dotnet_proj.mkdir()
    (dotnet_proj / "Service.csproj").write_text("<Project Sdk=\"Microsoft.NET.Sdk\"/>", encoding="utf-8")

    cfg = PolyglotBuildDiscoverer.discover(dotnet_proj, tmp_path)
    assert cfg is not None
    assert cfg.ecosystem == "dotnet"
    assert "dotnet build" in cfg.compile_cmd


def test_git_sandbox_creation_and_cleanup(tmp_path: Path):
    repo = tmp_path / "source_repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "App.java").write_text("public class App {}", encoding="utf-8")

    sandbox_env = GitWorktreeSandbox.create_sandbox(repo, "TICKET-123", sandbox_parent_dir=tmp_path / "sandboxes")
    assert sandbox_env.sandbox_path.exists()
    assert (sandbox_env.sandbox_path / "src" / "App.java").exists()

    # Mutate file in sandbox
    (sandbox_env.sandbox_path / "src" / "App.java").write_text("public class App { // modified }", encoding="utf-8")

    # Apply back
    success = GitWorktreeSandbox.apply_sandbox_to_original(sandbox_env, ["src/App.java"])
    assert success is True
    assert "// modified" in (repo / "src" / "App.java").read_text(encoding="utf-8")

    # Cleanup
    cleaned = GitWorktreeSandbox.cleanup_sandbox(sandbox_env)
    assert cleaned is True
    assert not sandbox_env.sandbox_path.exists()


def test_blast_radius_analyzer_detects_critical_cross_boundary(tmp_path: Path):
    # Setup backend controller
    issues = tmp_path / "issues-service"
    issues.mkdir()
    (issues / "IssueController.java").write_text(
        '@RestController\n@RequestMapping("/api/v1/issues")\n'
        'public class IssueController {\n'
        '  @GetMapping("/count")\n'
        '  public IssueCount getCount() { return null; }\n'
        '}\n',
        encoding="utf-8"
    )

    # Setup Gateway config
    gateway = tmp_path / "bim-gateway"
    gateway.mkdir()
    (gateway / "application.yml").write_text(
        "routes:\n  - id: issues\n    predicates:\n      - Path=/api/v1/issues/**\n",
        encoding="utf-8"
    )

    # Setup Frontend caller
    ui = tmp_path / "xchange-ui"
    ui.mkdir()
    (ui / "deliverables.service.ts").write_text(
        "this.http.get('/api/v1/issues/count')",
        encoding="utf-8"
    )

    report = BlastRadiusAnalyzer.analyze(tmp_path, ["issues-service/IssueController.java"])
    assert report.risk_level == "critical"
    assert "issues-service" in report.modified_services
    assert any("bim-gateway" in g for g in report.affected_gateways)
    assert any("xchange-ui" in u for u in report.affected_frontend_callers)
    md = report.to_markdown()
    assert "CRITICAL" in md
    assert "Affected Gateway Routes" in md
    assert "Affected UI Callers" in md


def test_tolerant_patcher_crlf_and_indentation(tmp_path: Path):
    tools = HostEnvironmentTools(str(tmp_path))
    target = tmp_path / "Component.ts"
    # Target file with CRLF and 4 spaces
    target.write_bytes(b"export class Component {\r\n    public value = 1;\r\n    public title = 'test';\r\n}\r\n")

    # Agent provides LF and slightly differing leading indentation
    agent_target = "public value = 1;\npublic title = 'test';"
    agent_replacement = "public value = 2;\npublic title = 'updated';"

    res = tools.replace_file_content("Component.ts", agent_target, agent_replacement)
    assert "Successfully" in res
    updated_bytes = target.read_bytes()
    assert b"value = 2" in updated_bytes
    assert b"title = 'updated'" in updated_bytes


def test_compiler_gate_parses_polyglot_languages():
    # 1. Python
    py_err = "File \"models.py\", line 15, in <module>\nNameError: name 'UndefinedType' is not defined"
    res_py = CompilerGate.parse_compiler_output("python-svc", 1, py_err)
    assert not res_py.passed
    assert len(res_py.diagnostics) >= 1
    assert "models.py" in res_py.diagnostics[0].file_path
    assert res_py.diagnostics[0].line_number == 15

    # 2. Go
    go_err = "server/main.go:24:9: undefined: HandleRequest"
    res_go = CompilerGate.parse_compiler_output("go-svc", 1, go_err)
    assert not res_go.passed
    assert len(res_go.diagnostics) >= 1
    assert "main.go" in res_go.diagnostics[0].file_path
    assert res_go.diagnostics[0].line_number == 24

    # 3. Rust
    rust_err = "error[E0425]: cannot find value `foo` in this scope\n  --> src/main.rs:18:5"
    res_rust = CompilerGate.parse_compiler_output("rust-svc", 1, rust_err)
    assert not res_rust.passed
    assert len(res_rust.diagnostics) >= 1
    assert "main.rs" in res_rust.diagnostics[0].file_path
    assert res_rust.diagnostics[0].line_number == 18

    # 4. .NET
    cs_err = "Controllers\\MainController.cs(32,14): error CS0103: The name 'NotFound' does not exist in the current context"
    res_cs = CompilerGate.parse_compiler_output("dotnet-svc", 1, cs_err)
    assert not res_cs.passed
    assert len(res_cs.diagnostics) >= 1
    assert "MainController.cs" in res_cs.diagnostics[0].file_path
    assert res_cs.diagnostics[0].line_number == 32


def test_compiler_gate_baseline_disk_persistence(tmp_path: Path):
    CompilerGate.clear_baselines()
    sample_err = "Main.java:10: error: cannot find symbol"
    res = CompilerGate.parse_compiler_output("test-svc", 1, sample_err)
    CompilerGate.record_baseline("test-svc", res, workspace_path=str(tmp_path))

    # Check baseline file on disk
    base_file = tmp_path / ".aviator" / "compiler_baselines.json"
    assert base_file.exists()

    # Clear memory and reload from disk
    CompilerGate.clear_baselines()
    assert CompilerGate.get_baseline("test-svc") is None

    loaded = CompilerGate.load_baselines_from_disk(str(tmp_path))
    assert loaded is True
    reloaded = CompilerGate.get_baseline("test-svc")
    assert reloaded is not None
    assert reloaded.exit_code == 1
    assert len(reloaded.diagnostics) == 1
    assert reloaded.diagnostics[0].line_number == 10


def test_focused_test_command_discovery(tmp_path: Path):
    src_dir = tmp_path / "src" / "main" / "java" / "com" / "example"
    src_dir.mkdir(parents=True)
    svc = src_dir / "PaymentService.java"
    svc.write_text("public class PaymentService {}", encoding="utf-8")

    test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
    test_dir.mkdir(parents=True)
    test_svc = test_dir / "PaymentServiceTest.java"
    test_svc.write_text("public class PaymentServiceTest {}", encoding="utf-8")

    # Mock gradle project
    (tmp_path / "build.gradle").write_text("// gradle", encoding="utf-8")

    focused_info = PolyglotBuildDiscoverer.get_focused_test_command(svc, tmp_path)
    assert focused_info is not None
    cmd, work_dir = focused_info
    assert "test" in cmd
    assert "--tests PaymentServiceTest" in cmd
    assert work_dir == tmp_path


def test_workflow_safe_write_or_patch_file_undo_stack(tmp_path: Path):
    from ticket_to_code.workflow import _safe_write_or_patch_file
    target_file = tmp_path / "Service.ts"
    target_file.write_text("const a = 1;", encoding="utf-8")

    state = {"_surgical_diff_engine": None}
    _safe_write_or_patch_file(tmp_path, "Service.ts", "const a = 2;", state)

    assert target_file.read_text(encoding="utf-8") == "const a = 2;"
    diff_engine = state["_surgical_diff_engine"]
    assert len(diff_engine.undo_stack) == 1
    assert diff_engine.undo_stack[0].original_content == "const a = 1;"

    # Test rollback
    diff_engine.rollback_all()
    assert target_file.read_text(encoding="utf-8") == "const a = 1;"

