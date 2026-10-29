"""
Tests for EditLoopAgent — Safe Edit Engine, Matching Safety Invariants, and Retry Feedback.

Verifies all 16 safety requirements from the EditLoopAgent Safety Specification:
TEST 1: Exact SEARCH match applies successfully.
TEST 2: CRLF/LF normalization succeeds when it results in exactly one match.
TEST 3: Whitespace normalization succeeds only when exactly one safe match exists.
TEST 4: No SEARCH match produces structured failure.
TEST 5: Multiple possible matches produce structured failure and NO edit.
TEST 6: SequenceMatcher/fuzzy similarity does NOT cause an automatic edit.
TEST 7: _generate_edit() retries exactly once after SEARCH failure.
TEST 8: Retry receives the actual failure feedback.
TEST 9: Second failed attempt does not trigger another retry.
TEST 10: _modify action with explicit edits succeeds.
TEST 11: _modify action returning arbitrary full-file content is rejected.
TEST 12: That rejection is added to attempt_rejections / retry feedback.
TEST 13: Canonical prompt format exactly matches canonical parser format.
TEST 14: Existing legacy format tests continue passing if those formats are still supported.
TEST 15: Regression test proving a failed edit does NOT cause the next iteration to treat the file as completed.
TEST 16: Regression test proving already-correct code remains unchanged after a failed edit attempt.
"""

from pathlib import Path
from unittest.mock import Mock
from langchain_core.messages import AIMessage

from ticket_to_code.agents.edit_loop_agent import (
    EditLoopAgent,
    EditDecision,
    EditResult,
    ApplyEditResult,
)


class FakeLLM:
    """Mock-free test double for LLM invocation that satisfies llm_invoke runtime checks."""

    def __init__(self, responses=None):
        self.responses = list(responses) if responses else []
        self.call_count = 0
        self.call_args_list = []

    def invoke(self, messages):
        self.call_count += 1
        self.call_args_list.append(((messages,), {}))
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, str):
                return AIMessage(content=item)
            return item
        return AIMessage(content="")


def _make_agent(tmp_path: Path, fake_llm=None) -> EditLoopAgent:
    llm = fake_llm or FakeLLM()
    symbol_index = Mock()
    symbol_index.get_ts_diagnostics.return_value = []
    symbol_index.get_members_for_file.return_value = None
    return EditLoopAgent(
        workspace_path=str(tmp_path),
        llm=llm,
        symbol_index=symbol_index,
    )


# ==============================================================================
# TEST 1: Exact SEARCH match applies successfully
# ==============================================================================
def test_1_exact_search_match_applies(tmp_path):
    agent = _make_agent(tmp_path)
    original = (
        "export class MemberService {\n"
        "  getMembers() {\n"
        "    return this.members;\n"
        "  }\n"
        "}\n"
    )
    llm_resp = (
        "<<<<<<< SEARCH\n"
        "  getMembers() {\n"
        "    return this.members;\n"
        "  }\n"
        "=======\n"
        "  getMembers() {\n"
        "    return this.activeMembers;\n"
        "  }\n"
        ">>>>>>> REPLACE"
    )
    res = agent._apply_search_replace(original, llm_resp, is_create=False)
    assert res.success is True
    assert res.applied_count == 1
    assert res.parser_format == "aider"
    assert "return this.activeMembers;" in res.content
    assert "return this.members;" not in res.content


# ==============================================================================
# TEST 2: CRLF/LF normalization succeeds when it results in exactly one match
# ==============================================================================
def test_2_crlf_lf_normalization_succeeds(tmp_path):
    agent = _make_agent(tmp_path)
    # File uses CRLF
    original_crlf = "line 1\r\nline 2\r\nline 3\r\n"
    # LLM outputs LF
    llm_resp = (
        "<<<<<<< SEARCH\n"
        "line 2\n"
        "=======\n"
        "line 2 updated\n"
        ">>>>>>> REPLACE"
    )
    res = agent._apply_search_replace(original_crlf, llm_resp, is_create=False)
    assert res.success is True
    assert res.applied_count == 1
    assert "line 2 updated" in res.content

    # Reverse: File uses LF, LLM outputs CRLF
    original_lf = "alpha\nbeta\ngamma\n"
    llm_resp_crlf = (
        "<<<<<<< SEARCH\r\n"
        "beta\r\n"
        "=======\r\n"
        "beta updated\r\n"
        ">>>>>>> REPLACE"
    )
    res2 = agent._apply_search_replace(original_lf, llm_resp_crlf, is_create=False)
    assert res2.success is True
    assert "beta updated" in res2.content


# ==============================================================================
# TEST 3: Whitespace normalization succeeds only when exactly one safe match exists
# ==============================================================================
def test_3_whitespace_normalization_single_safe_match(tmp_path):
    agent = _make_agent(tmp_path)
    # Original has 4-space indentation
    original = (
        "class Foo {\n"
        "    calculateTotal() {\n"
        "        const x = 10;\n"
        "        return x * 2;\n"
        "    }\n"
        "}\n"
    )
    # LLM uses 2-space indentation
    llm_resp = (
        "<<<<<<< SEARCH\n"
        "  calculateTotal() {\n"
        "    const x = 10;\n"
        "    return x * 2;\n"
        "  }\n"
        "=======\n"
        "  calculateTotal() {\n"
        "    const x = 20;\n"
        "    return x * 2;\n"
        "  }\n"
        ">>>>>>> REPLACE"
    )
    res = agent._apply_search_replace(original, llm_resp, is_create=False)
    assert res.success is True
    assert res.applied_count == 1
    assert "const x = 20;" in res.content

    # Ambiguous whitespace match: multiple blocks normalize to the same form
    original_ambiguous = (
        "class Foo {\n"
        "    foo() {\n"
        "        return 1;\n"
        "    }\n"
        "  foo() {\n"
        "    return 1;\n"
        "  }\n"
        "}\n"
    )
    llm_resp_ambiguous = (
        "<<<<<<< SEARCH\n"
        "foo() {\n"
        "return 1;\n"
        "}\n"
        "=======\n"
        "foo() {\n"
        "return 2;\n"
        "}\n"
        ">>>>>>> REPLACE"
    )
    res_amb = agent._apply_search_replace(original_ambiguous, llm_resp_ambiguous, is_create=False)
    assert res_amb.success is False
    assert "ambiguous" in res_amb.failure_reason
    assert res_amb.content == original_ambiguous  # untouched!


# ==============================================================================
# TEST 4: No SEARCH match produces structured failure
# ==============================================================================
def test_4_no_search_match_structured_failure(tmp_path):
    agent = _make_agent(tmp_path)
    original = "function hello() { return 'world'; }\n"
    llm_resp = (
        "<<<<<<< SEARCH\n"
        "function missingFunction() {\n"
        "=======\n"
        "function replacement() {\n"
        ">>>>>>> REPLACE"
    )
    res = agent._apply_search_replace(original, llm_resp, is_create=False)
    assert res.success is False
    assert "did not match the current file" in res.failure_reason
    assert "missingFunction" in res.failed_search
    assert res.content == original  # Original code preserved


# ==============================================================================
# TEST 5: Multiple possible matches produce structured failure and NO edit
# ==============================================================================
def test_5_multiple_matches_produce_failure_and_no_edit(tmp_path):
    agent = _make_agent(tmp_path)
    original = (
        "item = 1;\n"
        "doSomething();\n"
        "item = 1;\n"
    )
    llm_resp = (
        "<<<<<<< SEARCH\n"
        "item = 1;\n"
        "=======\n"
        "item = 2;\n"
        ">>>>>>> REPLACE"
    )
    res = agent._apply_search_replace(original, llm_resp, is_create=False)
    assert res.success is False
    assert "ambiguous" in res.failure_reason
    assert res.content == original  # ZERO edits applied


# ==============================================================================
# TEST 6: SequenceMatcher/fuzzy similarity does NOT cause an automatic edit
# ==============================================================================
def test_6_no_fuzzy_sequencematcher_automatic_edit(tmp_path):
    agent = _make_agent(tmp_path)
    # Original has variable calculateTotal
    original = "const total = calculateTotal(items, discount);\n"
    # LLM made a typo or guessed: calcTotal (very high character similarity >85%)
    llm_resp = (
        "<<<<<<< SEARCH\n"
        "const total = calcTotal(items, discount);\n"
        "=======\n"
        "const total = computeTotal(items, discount);\n"
        ">>>>>>> REPLACE"
    )
    res = agent._apply_search_replace(original, llm_resp, is_create=False)
    # Must FAIL safely without guessing or approximate replacement!
    assert res.success is False
    assert res.content == original
    assert "calculateTotal" in res.content


# ==============================================================================
# TEST 7: _generate_edit() retries exactly once after SEARCH failure
# ==============================================================================
def test_7_generate_edit_retries_exactly_once(tmp_path):
    original = "export function compute() { return 42; }\n"

    # Attempt 1 returns non-matching SEARCH; Attempt 2 returns valid SEARCH
    bad_edit = (
        "<<<<<<< SEARCH\n"
        "export function wrong() { return 0; }\n"
        "=======\n"
        "export function compute() { return 100; }\n"
        ">>>>>>> REPLACE"
    )
    good_edit = (
        "<<<<<<< SEARCH\n"
        "export function compute() { return 42; }\n"
        "=======\n"
        "export function compute() { return 100; }\n"
        ">>>>>>> REPLACE"
    )
    llm = FakeLLM(responses=[bad_edit, good_edit])

    agent = _make_agent(tmp_path, fake_llm=llm)
    decision = EditDecision(file_path="src/calc.ts", reason="Update value", edit_description="change 42 to 100")
    ticket = Mock(title="Fix value", description="description")

    new_content = agent._generate_edit(
        decision=decision,
        current_content=original,
        ticket=ticket,
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    assert llm.call_count == 2
    assert new_content is not None
    assert "return 100;" in new_content


# ==============================================================================
# TEST 8: Retry receives the actual failure feedback
# ==============================================================================
def test_8_retry_receives_actual_failure_feedback(tmp_path):
    original = "export function compute() { return 42; }\n"

    bad_edit = (
        "<<<<<<< SEARCH\n"
        "function missingTarget() {}\n"
        "=======\n"
        "function updated() {}\n"
        ">>>>>>> REPLACE"
    )
    good_edit = (
        "<<<<<<< SEARCH\n"
        "export function compute() { return 42; }\n"
        "=======\n"
        "export function compute() { return 100; }\n"
        ">>>>>>> REPLACE"
    )
    llm = FakeLLM(responses=[bad_edit, good_edit])

    agent = _make_agent(tmp_path, fake_llm=llm)
    decision = EditDecision(file_path="src/calc.ts", reason="Update", edit_description="change value")
    ticket = Mock(title="Fix value", description="description")

    agent._generate_edit(
        decision=decision,
        current_content=original,
        ticket=ticket,
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    assert llm.call_count == 2
    # Verify the second call received feedback about missingTarget and failure reason
    second_call_messages = llm.call_args_list[1][0][0]
    feedback_msg = second_call_messages[-1].content
    assert "FAILED TO APPLY" in feedback_msg
    assert "missingTarget" in feedback_msg
    assert "did not match the current file" in feedback_msg


# ==============================================================================
# TEST 9: Second failed attempt does not trigger another retry
# ==============================================================================
def test_9_second_failed_attempt_no_further_retry(tmp_path):
    original = "export function compute() { return 42; }\n"
    bad_edit = (
        "<<<<<<< SEARCH\n"
        "function nonexistent() {}\n"
        "=======\n"
        "function replacement() {}\n"
        ">>>>>>> REPLACE"
    )
    llm = FakeLLM(responses=[bad_edit, bad_edit])

    agent = _make_agent(tmp_path, fake_llm=llm)
    decision = EditDecision(file_path="src/calc.ts", reason="Update", edit_description="change value")
    ticket = Mock(title="Fix value", description="description")

    res = agent._generate_edit(
        decision=decision,
        current_content=original,
        ticket=ticket,
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    # Exactly 2 calls (initial + 1 retry), never a 3rd call
    assert llm.call_count == 2
    assert res is None  # Safe failure!


# ==============================================================================
# TEST 10: _modify action with explicit edits succeeds
# ==============================================================================
def test_10_modify_action_with_explicit_edits_succeeds(tmp_path):
    target_file = tmp_path / "src" / "sample.ts"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("let count = 0;\n", encoding="utf-8")

    json_response = (
        '{\n'
        '  "reasoning": "fix count",\n'
        '  "fixes": [\n'
        '    {\n'
        '      "file": "src/sample.ts",\n'
        '      "action": "modify",\n'
        '      "edits": [\n'
        '        {"search": "let count = 0;", "replace": "let count = 5;"}\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    llm = FakeLLM(responses=[json_response])

    agent = _make_agent(tmp_path, fake_llm=llm)
    fixed = agent._fix_errors(
        file_path="src/sample.ts",
        current_content="let count = 0;\n",
        errors=["count is wrong"],
        ticket=Mock(title="T", description="D"),
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    assert fixed is not None
    assert "let count = 5;" in fixed
    assert target_file.read_text(encoding="utf-8") == "let count = 5;\n"


# ==============================================================================
# TEST 11: _modify action returning arbitrary full-file content is rejected
# ==============================================================================
def test_11_modify_action_returning_content_is_rejected(tmp_path):
    target_file = tmp_path / "src" / "service.ts"
    target_file.parent.mkdir(parents=True)
    initial_content = "class MyService { existingMethod() { return 1; } }\n"
    target_file.write_text(initial_content, encoding="utf-8")

    # LLM returns action="modify" but provides "content" instead of "edits"
    json_response = (
        '{\n'
        '  "reasoning": "rewriting entire file",\n'
        '  "fixes": [\n'
        '    {\n'
        '      "file": "src/service.ts",\n'
        '      "action": "modify",\n'
        '      "content": "class MyService { fullRewrite() {} }"\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    llm = FakeLLM(responses=[json_response, json_response, json_response])

    agent = _make_agent(tmp_path, fake_llm=llm)
    agent._fix_errors(
        file_path="src/service.ts",
        current_content=initial_content,
        errors=["some error"],
        ticket=Mock(title="T", description="D"),
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    # The file on disk MUST NOT have been overwritten with the full rewrite!
    disk_content = target_file.read_text(encoding="utf-8")
    assert disk_content == initial_content
    assert "fullRewrite" not in disk_content


# ==============================================================================
# TEST 12: Rejection is added to attempt_rejections / retry feedback
# ==============================================================================
def test_12_rejection_added_to_retry_feedback(tmp_path):
    target_file = tmp_path / "src" / "service.ts"
    target_file.parent.mkdir(parents=True)
    initial_content = "class MyService { existingMethod() { return 1; } }\n"
    target_file.write_text(initial_content, encoding="utf-8")

    # Attempt 1: bad modify with full content
    bad_json = (
        '{\n'
        '  "reasoning": "bad attempt",\n'
        '  "fixes": [\n'
        '    {"file": "src/service.ts", "action": "modify", "content": "full file content"}\n'
        '  ]\n'
        '}'
    )
    # Attempt 2: corrected with explicit edits
    good_json = (
        '{\n'
        '  "reasoning": "corrected",\n'
        '  "fixes": [\n'
        '    {\n'
        '      "file": "src/service.ts",\n'
        '      "action": "modify",\n'
        '      "edits": [{"search": "return 1;", "replace": "return 2;"}]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    llm = FakeLLM(responses=[bad_json, good_json])

    agent = _make_agent(tmp_path, fake_llm=llm)
    fixed = agent._fix_errors(
        file_path="src/service.ts",
        current_content=initial_content,
        errors=["some error"],
        ticket=Mock(title="T", description="D"),
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    assert llm.call_count == 2
    # Verify attempt 2 prompt contains the rejection feedback
    second_prompt = llm.call_args_list[1][0][0][1].content
    assert "requires explicit search/replace edits" in second_prompt
    assert "Full-file content was returned" in second_prompt
    assert "return 2;" in fixed


# ==============================================================================
# TEST 13: Canonical prompt format exactly matches canonical parser format
# ==============================================================================
def test_13_canonical_prompt_format_matches_parser_format(tmp_path):
    llm = FakeLLM(responses=["<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE"])
    agent = _make_agent(tmp_path, fake_llm=llm)
    decision = EditDecision(file_path="src/app.ts", reason="r", edit_description="d")

    agent._generate_edit(
        decision=decision,
        current_content="foo\n",
        ticket=Mock(title="T", description="D"),
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    system_prompt = llm.call_args_list[0][0][0][0].content
    # Prompt describes Aider git-merge-conflict style
    assert "<<<<<<< SEARCH" in system_prompt
    assert "=======" in system_prompt
    assert ">>>>>>> REPLACE" in system_prompt


# ==============================================================================
# TEST 14: Existing legacy format tests continue passing if supported
# ==============================================================================
def test_14_legacy_formats_continue_passing(tmp_path):
    agent = _make_agent(tmp_path)
    original = "var x = 10;\nvar y = 20;\n"

    # Fallback 1: <<<SEARCH>>>
    delimiters_resp = (
        "<<<SEARCH>>>\n"
        "var x = 10;\n"
        "<<<REPLACE>>>\n"
        "var x = 15;\n"
        "<<<END>>>"
    )
    res1 = agent._apply_search_replace(original, delimiters_resp, is_create=False)
    assert res1.success is True
    assert res1.parser_format == "delimiters"
    assert "var x = 15;" in res1.content

    # Fallback 2: legacy EDIT:
    legacy_resp = (
        "EDIT:\n"
        "old_str:\n"
        "<<<\n"
        "var y = 20;\n"
        ">>>\n"
        "new_str:\n"
        "<<<\n"
        "var y = 25;\n"
        ">>>"
    )
    res2 = agent._apply_search_replace(original, legacy_resp, is_create=False)
    assert res2.success is True
    assert res2.parser_format == "legacy"
    assert "var y = 25;" in res2.content


# ==============================================================================
# TEST 15: Failed edit does NOT cause next iteration to treat file as completed
# ==============================================================================
def test_15_failed_edit_not_treated_as_completed(tmp_path):
    # LLM always returns invalid search block
    bad_resp = "<<<<<<< SEARCH\nconst WRONG = 0;\n=======\nconst WRONG = 1;\n>>>>>>> REPLACE"
    llm = FakeLLM(responses=[bad_resp, bad_resp, bad_resp, bad_resp, bad_resp, bad_resp])
    agent = _make_agent(tmp_path, fake_llm=llm)

    test_file = tmp_path / "src" / "target.ts"
    test_file.parent.mkdir(parents=True)
    initial_content = "export const PI = 3.14;\n"
    test_file.write_text(initial_content, encoding="utf-8")

    # Initial plan has one task for target.ts
    plan_task = Mock(
        file_path="src/target.ts",
        title="Update PI",
        description="Update PI value",
        task_type=Mock(value="modify"),
    )
    initial_plan = Mock(tasks=[plan_task])

    result = agent.run(
        ticket=Mock(ticket_id="T-1", title="Update PI", description="Update PI value"),
        requirements=Mock(functional_requirements=[]),
        initial_plan=initial_plan,
    )

    # Must NOT be marked as solved!
    assert result["solved"] is False
    # Must have a failed edit recorded
    assert any(r.reason == "edit application failed" for r in result["edit_results"])
    # File content must remain original
    assert test_file.read_text(encoding="utf-8") == initial_content


# ==============================================================================
# TEST 16: Already-correct code remains unchanged after a failed edit attempt
# ==============================================================================
def test_16_correct_code_remains_unchanged_after_failed_edit(tmp_path):
    target_file = tmp_path / "src" / "good_file.ts"
    target_file.parent.mkdir(parents=True)
    pristine_code = (
        "// Correct production code\n"
        "export function getOrg(user: any): string {\n"
        "  return user.existingOrganizationName || '';\n"
        "}\n"
    )
    target_file.write_text(pristine_code, encoding="utf-8")

    bad_edit = (
        "<<<<<<< SEARCH\n"
        "// Corrupted search block that doesn't exist\n"
        "=======\n"
        "// Corrupted replacement\n"
        ">>>>>>> REPLACE"
    )
    llm = FakeLLM(responses=[bad_edit, bad_edit])

    agent = _make_agent(tmp_path, fake_llm=llm)
    decision = EditDecision(file_path="src/good_file.ts", reason="r", edit_description="d")

    res = agent._generate_edit(
        decision=decision,
        current_content=pristine_code,
        ticket=Mock(title="T", description="D"),
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    assert res is None
    # Verify disk content is untouched byte-for-byte
    assert target_file.read_text(encoding="utf-8") == pristine_code


# ==============================================================================
# TEST 17: Multi-edit batch atomicity [valid, invalid, valid] in _safe_match_and_replace
# ==============================================================================
def test_17_multi_edit_batch_atomicity_valid_invalid_valid(tmp_path):
    agent = _make_agent(tmp_path)
    original = (
        "const a = 1;\n"
        "const b = 2;\n"
        "const c = 3;\n"
    )
    edits = [
        {"old_str": "const a = 1;\n", "new_str": "const a = 100;\n"},  # Valid
        {"old_str": "const nonExistent = 999;\n", "new_str": "const b = 200;\n"},  # Invalid
        {"old_str": "const c = 3;\n", "new_str": "const c = 300;\n"},  # Valid
    ]
    res = agent._safe_match_and_replace(original, edits, parser_format="aider")
    # Must fail safely and return the UNMODIFIED original byte-for-byte
    assert res.success is False
    assert res.content == original
    assert "const a = 100;" not in res.content
    assert "const c = 300;" not in res.content


# ==============================================================================
# TEST 18: _fix_errors atomic edits rejected on partial failure (no partial write)
# ==============================================================================
def test_18_fix_errors_atomic_edits_rejected_on_partial_failure(tmp_path):
    target_file = tmp_path / "src" / "service.ts"
    target_file.parent.mkdir(parents=True)
    initial_content = (
        "class Service {\n"
        "  methodA() { return 1; }\n"
        "  methodB() { return 2; }\n"
        "}\n"
    )
    target_file.write_text(initial_content, encoding="utf-8")

    # LLM returns 2 edits: edit 1 is valid, edit 2 is invalid
    partial_failure_json = (
        '{\n'
        '  "reasoning": "partial fix",\n'
        '  "fixes": [\n'
        '    {\n'
        '      "file": "src/service.ts",\n'
        '      "action": "modify",\n'
        '      "edits": [\n'
        '        {"search": "methodA() { return 1; }", "replace": "methodA() { return 10; }"},\n'
        '        {"search": "nonExistentMethod()", "replace": "newMethod()"}\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    llm = FakeLLM(responses=[partial_failure_json, partial_failure_json, partial_failure_json])
    agent = _make_agent(tmp_path, fake_llm=llm)

    agent._fix_errors(
        file_path="src/service.ts",
        current_content=initial_content,
        errors=["compiler error"],
        ticket=Mock(title="T", description="D"),
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    # Disk content must remain pristine — NO partial application of methodA!
    disk_content = target_file.read_text(encoding="utf-8")
    assert disk_content == initial_content
    assert "return 10;" not in disk_content


# ==============================================================================
# TEST 19: _fix_errors rejects ambiguous search string (no guessing first occurrence)
# ==============================================================================
def test_19_fix_errors_ambiguous_edits_rejected(tmp_path):
    target_file = tmp_path / "src" / "service.ts"
    target_file.parent.mkdir(parents=True)
    initial_content = (
        "class Service {\n"
        "  flag = true;\n"
        "  flag = true;\n"
        "}\n"
    )
    target_file.write_text(initial_content, encoding="utf-8")

    ambiguous_json = (
        '{\n'
        '  "reasoning": "ambiguous fix",\n'
        '  "fixes": [\n'
        '    {\n'
        '      "file": "src/service.ts",\n'
        '      "action": "modify",\n'
        '      "edits": [\n'
        '        {"search": "flag = true;", "replace": "flag = false;"}\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    llm = FakeLLM(responses=[ambiguous_json, ambiguous_json, ambiguous_json])
    agent = _make_agent(tmp_path, fake_llm=llm)

    agent._fix_errors(
        file_path="src/service.ts",
        current_content=initial_content,
        errors=["compiler error"],
        ticket=Mock(title="T", description="D"),
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    # Disk content must remain untouched — ambiguous edit rejected!
    disk_content = target_file.read_text(encoding="utf-8")
    assert disk_content == initial_content
    assert "flag = false;" not in disk_content


# ==============================================================================
# TEST 20: Existing clean files trigger fast-exit with 0 LLM calls
# ==============================================================================
def test_20_existing_file_map_fast_exit_zero_llm_calls(tmp_path):
    llm = FakeLLM()
    agent = _make_agent(tmp_path, fake_llm=llm)

    target_file = tmp_path / "src" / "clean_comp.ts"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    content = "export class CleanComponent { isReady = true; }\n"
    target_file.write_text(content, encoding="utf-8")

    plan_task = Mock(
        file_path="src/clean_comp.ts",
        title="Update clean component",
        description="Update description",
        task_type=Mock(value="modify"),
    )
    initial_plan = Mock(tasks=[plan_task])

    result = agent.run(
        ticket=Mock(ticket_id="T-20", title="Clean Ticket", description="Desc"),
        requirements=Mock(functional_requirements=[]),
        initial_plan=initial_plan,
        existing_file_map={"src/clean_comp.ts": content},
    )

    # Must fast-exit without burning LLM calls!
    assert result["solved"] is True
    assert result["iterations"] == 0
    assert llm.call_count == 0


# ==============================================================================
# TEST 21: _generate_edit provides verbatim source without smart_extract outlines
# ==============================================================================
def test_21_generate_edit_uses_verbatim_for_normal_files(tmp_path):
    captured_messages = []

    class CapturingLLM(FakeLLM):
        def invoke(self, messages):
            captured_messages.append(messages)
            return AIMessage(content="SUMMARY: Updated value\n<<<<<<< SEARCH\n  val = 1;\n=======\n  val = 2;\n>>>>>>> REPLACE")

    llm = CapturingLLM()
    agent = _make_agent(tmp_path, fake_llm=llm)

    file_content = (
        "export class SampleService {\n"
        "  val = 1;\n"
        "  doSomething() {\n"
        "    return this.val;\n"
        "  }\n"
        "}\n"
    )
    decision = EditDecision(file_path="src/sample.ts", reason="Update val", edit_description="Change val to 2")

    res = agent._generate_edit(
        decision=decision,
        current_content=file_content,
        ticket=Mock(title="T", description="D"),
        requirements=None,
        code_rag_context="",
        lsp_context="",
    )

    assert res is not None
    assert "val = 2;" in res
    # Verify the prompt sent to LLM contains the verbatim code without smart_extract comment banners
    assert len(captured_messages) >= 1
    user_prompt = captured_messages[0][1].content
    assert "export class SampleService {" in user_prompt
    assert "// === FILE OUTLINE" not in user_prompt

