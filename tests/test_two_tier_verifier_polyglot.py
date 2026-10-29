import pytest
from ticket_to_code.verification.two_tier_verifier import (
    TwoTierVerifier,
    Tier1Result,
    Tier2Result,
)

def test_tier1_polyglot_python_syntax(tmp_path):
    verifier = TwoTierVerifier(workspace_path=str(tmp_path))

    # Valid Python file
    valid_py = tmp_path / "valid.py"
    valid_py.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    res_valid = verifier.run_tier1_fast_check(["valid.py"])
    assert res_valid.passed is True
    assert len(res_valid.diagnostics) == 0

    # Invalid Python syntax
    invalid_py = tmp_path / "invalid.py"
    invalid_py.write_text("def hello(\n    return 'world'\n", encoding="utf-8")
    res_invalid = verifier.run_tier1_fast_check(["invalid.py"])
    assert res_invalid.passed is False
    assert len(res_invalid.diagnostics) > 0
    assert res_invalid.diagnostics[0].code == "PYTHON_SYNTAX_ERROR"


def test_tier1_polyglot_typescript_delimiters(tmp_path):
    verifier = TwoTierVerifier(workspace_path=str(tmp_path))

    # TypeScript with unclosed template literal backtick
    invalid_ts = tmp_path / "component.ts"
    invalid_ts.write_text("const msg = `hello world;\nconst b = 1;\n", encoding="utf-8")
    res_ts = verifier.run_tier1_fast_check(["component.ts"])
    assert res_ts.passed is False
    assert any("TS_" in d.code or d.code == "UNCLOSED_TEMPLATE_LITERAL" for d in res_ts.diagnostics)


def test_tier1_polyglot_java_class_mismatch(tmp_path):
    verifier = TwoTierVerifier(workspace_path=str(tmp_path))

    # Java class where declared public class does not match file name
    java_file = tmp_path / "DocumentService.java"
    java_file.write_text("public class AreaService {\n    public void save() {}\n}\n", encoding="utf-8")
    res_java = verifier.run_tier1_fast_check(["DocumentService.java"])
    assert res_java.passed is False
    assert any(d.code == "JAVA_TYPE_NAME_MISMATCH" for d in res_java.diagnostics)


def test_tier2_agent_style_pytest_execution(tmp_path):
    # Setup mock python project
    sub_proj = tmp_path / "my_service"
    tests_dir = sub_proj / "tests"
    tests_dir.mkdir(parents=True)

    test_file = tests_dir / "test_simple.py"
    test_file.write_text("def test_ok():\n    assert 1 == 1\n", encoding="utf-8")

    verifier = TwoTierVerifier(workspace_path=str(tmp_path))
    res = verifier.run_tier2_scoped_build(target_service="my_service", test_filter="tests/test_simple.py")
    assert res.passed is True
    assert res.exit_code == 0
    assert len(res.diagnostics) == 0
