"""
Unit tests for CompilerGate.
Verifies parsing of Java and TypeScript compiler diagnostics, self-repair prompt generation,
and DeductiveBlackboard recording.
"""

from ticket_to_code.runtime.compiler_gate import CompilerGate
from ticket_to_code.runtime.deductive_blackboard import DeductiveBlackboard


def test_compiler_gate_parses_java_cannot_find_symbol():
    bb = DeductiveBlackboard()
    raw_error = """
    > Task :issues-service:compileJava FAILED
    /workspace/CloudForge-Commerce/order-service/src/main/java/com/cloudforge/solutions/services/issues/service/impl/IssuesServiceImpl.java:480: error: cannot find symbol
        if (issue.getPriority() == IssuePriority.CRITICAL) {
                                   ^
      symbol:   variable IssuePriority
      location: class IssuesServiceImpl
    1 error
    """

    res = CompilerGate.record_and_verify("issues-service", 1, raw_error, blackboard=bb)

    assert not res.passed
    assert res.exit_code == 1
    assert len(res.diagnostics) >= 1
    diag = res.diagnostics[0]
    assert "IssuesServiceImpl.java" in diag.file_path
    assert diag.line_number == 480
    assert "cannot find symbol" in diag.message

    # Test repair instruction generation
    repair_prompt = res.get_repair_instruction()
    assert "❌ COMPILATION FAILED" in repair_prompt
    assert "IssuesServiceImpl.java:480" in repair_prompt
    assert "cannot find symbol" in repair_prompt

    # Verify blackboard recorded rejection
    assert any("cannot find symbol" in r for r in bb.rejected_hypotheses)


def test_compiler_gate_parses_typescript_syntax_error():
    bb = DeductiveBlackboard()
    raw_error = """
    Error: src/app/modules/deliverables/deliverable/deliverable.component.ts(245,18): error TS2304: Cannot find name 'IssuePriority'.
    """

    res = CompilerGate.record_and_verify("xchange-ui", 1, raw_error, blackboard=bb)

    assert not res.passed
    assert len(res.diagnostics) >= 1
    diag = res.diagnostics[0]
    assert "deliverable.component.ts" in diag.file_path
    assert diag.line_number == 245
    assert diag.error_code == "TS2304"
    assert "Cannot find name 'IssuePriority'" in diag.message


def test_compiler_gate_records_clean_compilation():
    bb = DeductiveBlackboard()
    res = CompilerGate.record_and_verify("issues-service", 0, "BUILD SUCCESSFUL in 2s", blackboard=bb)

    assert res.passed
    assert "✅ Compilation succeeded with 0 errors." in res.get_repair_instruction()
    assert any("compiled cleanly with 0 errors" in f for f in bb.verified_facts)


def test_compiler_gate_baseline_clean_to_regression():
    CompilerGate.clear_baselines()
    clean_baseline = CompilerGate.parse_compiler_output("issues-service", 0, "BUILD SUCCESSFUL")
    CompilerGate.record_baseline("issues-service", clean_baseline)

    # Agent edit causes a syntax error
    err_out = """
    /C:/repo/issues-service/src/main/java/IssueController.java:42: error: cannot find symbol
    """
    mutation_res = CompilerGate.parse_compiler_output("issues-service", 1, err_out)
    eval_res = CompilerGate.evaluate_mutation("issues-service", mutation_res)

    assert not eval_res["passed"]
    assert eval_res["is_regression"] is True
    assert len(eval_res["new_diagnostics"]) == 1
    assert "cannot find symbol" in eval_res["message"]


def test_compiler_gate_baseline_pre_existing_errors_not_flagged():
    CompilerGate.clear_baselines()
    pre_existing_err = """
    /C:/repo/issues-service/src/main/java/OldBrokenService.java:10: error: cannot find symbol
    """
    baseline = CompilerGate.parse_compiler_output("issues-service", 1, pre_existing_err)
    CompilerGate.record_baseline("issues-service", baseline)

    # After edit, the exact same pre-existing error is present, with NO new errors
    post_edit = CompilerGate.parse_compiler_output("issues-service", 1, pre_existing_err)
    eval_res = CompilerGate.evaluate_mutation("issues-service", post_edit)

    assert not eval_res["passed"]
    assert eval_res["is_regression"] is False
    assert len(eval_res["new_diagnostics"]) == 0
    assert len(eval_res["pre_existing_diagnostics"]) == 1
    assert "0 new regressions" in eval_res["message"]


def test_compiler_gate_baseline_delta_regression_isolated():
    CompilerGate.clear_baselines()
    pre_existing_err = """
    /C:/repo/issues-service/src/main/java/OldBrokenService.java:10: error: cannot find symbol
    """
    baseline = CompilerGate.parse_compiler_output("issues-service", 1, pre_existing_err)
    CompilerGate.record_baseline("issues-service", baseline)

    # Agent edit introduces a second error in a new file
    mixed_errors = """
    /C:/repo/issues-service/src/main/java/OldBrokenService.java:10: error: cannot find symbol
    /C:/repo/issues-service/src/main/java/NewController.java:99: error: ';' expected
    """
    post_edit = CompilerGate.parse_compiler_output("issues-service", 1, mixed_errors)
    eval_res = CompilerGate.evaluate_mutation("issues-service", post_edit)

    assert not eval_res["passed"]
    assert eval_res["is_regression"] is True
    assert len(eval_res["new_diagnostics"]) == 1
    assert eval_res["new_diagnostics"][0].line_number == 99
    assert len(eval_res["pre_existing_diagnostics"]) == 1
    assert "[NEW] `NewController.java:99`" in eval_res["message"]
