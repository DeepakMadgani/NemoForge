import sys
import tempfile
from pathlib import Path
import pytest

from ticket_to_code.runtime.container_sandbox import (
    ContainerSandboxEngine,
    ToolchainProfile,
    DockerSandboxExecutor,
    ContainerExecutionResult,
)


def test_detect_requirements_pom_xml_java_21():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        pom = ws / "pom.xml"
        pom.write_text(
            """<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>demo</artifactId>
  <version>1.0.0</version>
  <properties>
    <java.version>21</java.version>
  </properties>
</project>""",
            encoding="utf-8",
        )

        engine = ContainerSandboxEngine(ws)
        profile = engine.detect_requirements(ws)

        assert profile.ecosystem == "java"
        assert profile.build_tool == "maven"
        assert profile.required_version == "21"
        assert "temurin-21" in profile.container_image
        assert "pom.xml" in profile.declared_source


def test_detect_requirements_pom_xml_java_8():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        pom = ws / "pom.xml"
        pom.write_text(
            """<project>
  <properties>
    <maven.compiler.source>1.8</maven.compiler.source>
  </properties>
</project>""",
            encoding="utf-8",
        )

        engine = ContainerSandboxEngine(ws)
        profile = engine.detect_requirements(ws)

        assert profile.ecosystem == "java"
        assert profile.required_version == "1.8"
        assert "temurin-8" in profile.container_image


def test_detect_requirements_gradle():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        gradle = ws / "build.gradle"
        gradle.write_text(
            """plugins { id 'java' }
sourceCompatibility = '17'
targetCompatibility = '17'""",
            encoding="utf-8",
        )

        engine = ContainerSandboxEngine(ws)
        profile = engine.detect_requirements(ws)

        assert profile.ecosystem == "java"
        assert profile.build_tool == "gradle"
        assert profile.required_version == "17"
        assert "jdk17" in profile.container_image


def test_detect_requirements_package_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        pkg = ws / "package.json"
        pkg.write_text(
            """{
  "name": "my-ui",
  "engines": {
    "node": "18.x"
  },
  "packageManager": "pnpm@8.1.0"
}""",
            encoding="utf-8",
        )
        (ws / "pnpm-lock.yaml").write_text("lockfileVersion: 5.4", encoding="utf-8")

        engine = ContainerSandboxEngine(ws)
        profile = engine.detect_requirements(ws)

        assert profile.ecosystem == "nodejs"
        assert profile.build_tool == "pnpm"
        assert profile.required_version == "18.x"
        assert "node:18" in profile.container_image
        assert profile.lockfile == "pnpm-lock.yaml"


def test_wrapper_detection_and_preference():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        (ws / "pom.xml").write_text("<project></project>", encoding="utf-8")
        wrap_name = "mvnw.cmd" if sys.platform == "win32" else "mvnw"
        (ws / wrap_name).write_text("echo wrapper", encoding="utf-8")

        engine = ContainerSandboxEngine(ws)
        profile = engine.detect_requirements(ws)

        assert profile.wrapper_cmd is not None
        assert wrap_name in profile.wrapper_cmd

        # Verify command substitution prefers wrapper
        subbed_cmd = engine._apply_wrapper_preference("mvn clean test", profile, ws)
        assert wrap_name in subbed_cmd
        assert not subbed_cmd.startswith("mvn ")


def test_host_compatibility_check_catches_mismatch(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        engine = ContainerSandboxEngine(ws)
        profile = ToolchainProfile(
            ecosystem="java",
            build_tool="maven",
            required_version="21",
            declared_source="pom.xml (Java 21)",
        )

        # Mock java -version returning Java 11
        class MockSubprocessRes:
            stdout = 'openjdk version "11.0.18" 2023-01-17'
            stderr = ""
            returncode = 0

        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: MockSubprocessRes()
        )
        monkeypatch.setattr("shutil.which", lambda cmd: "C:\\fake\\java.exe" if cmd == "java" else "C:\\fake\\mvn.cmd")

        is_compat, reason = engine._verify_host_compatibility(profile)
        assert not is_compat
        assert "Host Toolchain Mismatch" in reason
        assert "Java 21" in reason
        assert "Java 11" in reason


def test_container_sandbox_engine_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        (ws / "package.json").write_text('{"name": "test-pkg"}', encoding="utf-8")

        engine = ContainerSandboxEngine(ws)
        ok, profile, msg = engine.prepare_environment(ws)
        assert profile.ecosystem == "nodejs"

        # Execute simple cross-platform echo command
        echo_cmd = "echo SandboxTestOK"
        res = engine.execute_build(ws, echo_cmd)
        assert res.exit_code == 0
        assert "SandboxTestOK" in res.stdout or res.isolated_via in ("docker", "process_sandbox")

        summary = engine.collect_results()
        assert summary["total_executions"] == 1
        assert summary["all_passed"] is True
        assert summary["active_profile"]["ecosystem"] == "nodejs"
        assert len(summary["executions"]) == 1
