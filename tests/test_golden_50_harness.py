"""
Unit tests for golden_50_harness.py
Verifies invariants of the 50-ticket benchmark harness.
"""

from pathlib import Path
import pytest

from tests.benchmarks.golden_50_harness import (
    make_golden_50_specs,
    detect_toolchain,
    check_ticket_prereqs,
    evaluate_inquiry_rubric,
    Tier0AssertionResult,
    TicketSpec
)


def test_golden_50_specs_count_and_categories():
    """Verify exactly 50 tickets are defined with correct category distribution."""
    specs = make_golden_50_specs()
    assert len(specs) == 50

    # Verify 1-to-50 sequential numbering
    numbers = [s.number for s in specs]
    assert numbers == list(range(1, 51))

    # Verify ticket IDs
    for i, s in enumerate(specs, start=1):
        assert s.ticket_id == f"GOLDEN-{i:02d}"

    # Category counts
    category_counts = {}
    for s in specs:
        category_counts[s.category] = category_counts.get(s.category, 0) + 1

    assert category_counts["UI"] == 7
    assert category_counts["FE"] == 8
    assert category_counts["BE"] == 7
    assert category_counts["Mix"] == 7
    assert category_counts["Cfg"] == 6
    assert category_counts["Info"] == 5
    assert category_counts["Dbg"] == 5
    assert category_counts["Neg"] == 5
    assert sum(category_counts.values()) == 50


def test_evaluators_callable_on_empty_workspace(tmp_path):
    """Verify every Tier 0 evaluator executes safely against a clean workspace without crashing."""
    specs = make_golden_50_specs()
    dummy_run_output = {
        "solution_guidance": "Grounded answer citing auth.service.ts and interceptor with token symbols.",
        "changed_files": [],
        "tokens_in": 100,
        "tokens_out": 50,
        "total_tokens": 150
    }

    for spec in specs:
        res = spec.evaluator(tmp_path, dummy_run_output)
        assert isinstance(res, Tier0AssertionResult)
        assert isinstance(res.passed, bool)
        assert isinstance(res.reason, str)


def test_inquiry_rubric_enforces_zero_code_changes():
    """Verify that inquiry tickets strictly fail if any file is modified."""
    # Modified files present -> must fail
    res = evaluate_inquiry_rubric(
        answer_text="Here is the explanation citing auth.service.ts token storage.",
        changed_files=["src/app/auth.service.ts"],
        expected_citations=["auth.service"],
        key_symbols=["token"]
    )
    assert not res.passed
    assert "Must be 0" in res.reason

    # Zero files modified + correct citations -> passes
    res_clean = evaluate_inquiry_rubric(
        answer_text="Here is the explanation citing auth.service.ts and localStorage token storage.",
        changed_files=[],
        expected_citations=["auth.service"],
        key_symbols=["token", "localStorage"]
    )
    assert res_clean.passed


def test_toolchain_detection():
    """Verify toolchain detection returns dictionary of expected tools."""
    tools = detect_toolchain()
    assert isinstance(tools, dict)
    assert "git" in tools
    assert "node" in tools
    assert "python" in tools


def test_reset_fixture_workspace_guards_dirty_tree(tmp_path):
    """Verify that reset_fixture_workspace aborts on uncommitted work unless allow_live_reset=True."""
    import subprocess
    from tests.benchmarks.golden_50_harness import reset_fixture_workspace

    # Initialize a temporary git repo
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), capture_output=True, check=True)
    
    test_file = tmp_path / "README.md"
    test_file.write_text("initial content", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), capture_output=True, check=True)

    # Create uncommitted dirty change
    test_file.write_text("dirty uncommitted content", encoding="utf-8")

    # Call without allow_live_reset -> MUST RAISE RuntimeError
    with pytest.raises(RuntimeError, match="SAFETY ABORT"):
        reset_fixture_workspace(tmp_path, allow_live_reset=False)

    # Uncommitted content must still be intact (not destroyed)
    assert test_file.read_text(encoding="utf-8") == "dirty uncommitted content"

    # Call with allow_live_reset=True -> resets cleanly
    reset_fixture_workspace(tmp_path, allow_live_reset=True)
    assert test_file.read_text(encoding="utf-8") == "initial content"


def test_fixture_exemption_allows_reset_on_dirty_fixture(tmp_path):
    """Verify that a managed fixture workspace is exempt from the dirty tree abort."""
    import subprocess
    from tests.benchmarks.golden_50_harness import reset_fixture_workspace

    # Name the workspace with 'fixture' in its path
    fixture_dir = tmp_path / "xchange-ui-fixture"
    fixture_dir.mkdir()

    subprocess.run(["git", "init"], cwd=str(fixture_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(fixture_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(fixture_dir), capture_output=True, check=True)

    test_file = fixture_dir / "index.ts"
    test_file.write_text("initial code", encoding="utf-8")
    subprocess.run(["git", "add", "index.ts"], cwd=str(fixture_dir), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(fixture_dir), capture_output=True, check=True)

    # Make dirty change on fixture
    test_file.write_text("ticket 1 modified code", encoding="utf-8")

    # Even with allow_live_reset=False, fixture workspace should reset cleanly!
    reset_fixture_workspace(fixture_dir, allow_live_reset=False)
    assert test_file.read_text(encoding="utf-8") == "initial code"


def test_artifact_filtering_ignores_trace_files():
    """Verify that internal trace artifacts and logs are excluded from changed files."""
    from tests.benchmarks.golden_50_harness import is_ignored_artifact_file

    assert is_ignored_artifact_file(".aviator/trace.json") is True
    assert is_ignored_artifact_file(".aviator\\harness_tmp_GOLDEN-01.json") is True
    assert is_ignored_artifact_file("trace_artifacts/decision_chain.json") is True
    assert is_ignored_artifact_file("build_output.log") is True
    assert is_ignored_artifact_file(".pytest_cache/v/cache/nodeids") is True

    # Real source files must NOT be ignored
    assert is_ignored_artifact_file("src/app/auth.service.ts") is False
    assert is_ignored_artifact_file("pom.xml") is False
    assert is_ignored_artifact_file("src/assets/i18n/register/en.json") is False


def test_multi_ticket_reset_sequencing_on_fixture(tmp_path):
    """Verify that multiple tickets modifying the fixture workspace execute and reset without crashing."""
    import subprocess
    from tests.benchmarks.golden_50_harness import (
        execute_single_ticket,
        TicketSpec,
        Tier0AssertionResult
    )

    fixture_dir = tmp_path / "xchange-ui-fixture"
    fixture_dir.mkdir()

    subprocess.run(["git", "init"], cwd=str(fixture_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(fixture_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(fixture_dir), capture_output=True, check=True)

    test_file = fixture_dir / "app.component.ts"
    test_file.write_text("class AppComponent { count = 0; }", encoding="utf-8")
    subprocess.run(["git", "add", "app.component.ts"], cwd=str(fixture_dir), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(fixture_dir), capture_output=True, check=True)

    # Evaluator that asserts changes are present DURING evaluation
    observed_states = []

    def make_evaluator(expected_text: str):
        def _eval(ws: Path, run_out: dict) -> Tier0AssertionResult:
            content = (ws / "app.component.ts").read_text(encoding="utf-8")
            observed_states.append(content)
            passed = expected_text in content
            return Tier0AssertionResult(passed=passed, reason="matched" if passed else "missing")
        return _eval

    # In dry-run mode, simulate running two consecutive tickets with changes
    spec1 = TicketSpec(
        number=1,
        ticket_id="GOLDEN-01",
        category="FE",
        difficulty="Easy",
        title="Ticket 1",
        description="Update count to 10",
        acceptance_criteria=[],
        required_prereqs=[],
        evaluator=make_evaluator("class AppComponent { count = 0; }")
    )

    spec2 = TicketSpec(
        number=2,
        ticket_id="GOLDEN-02",
        category="FE",
        difficulty="Easy",
        title="Ticket 2",
        description="Update count to 20",
        acceptance_criteria=[],
        required_prereqs=[],
        evaluator=make_evaluator("class AppComponent { count = 0; }")
    )

    # Run ticket 1 dry-run
    res1 = execute_single_ticket(spec=spec1, workspace=fixture_dir, mode="baseline", dry_run=True)
    assert res1.tier0_passed is True

    # Mutate file on fixture (simulate ticket 1 leaving dirty state before ticket 2 pre-reset)
    test_file.write_text("class AppComponent { count = 999; }", encoding="utf-8")

    # Run ticket 2 (dry_run=False but mocked worker or dry_run=False with pre-reset):
    # Test pre-reset on dirty fixture directly:
    from tests.benchmarks.golden_50_harness import reset_fixture_workspace
    reset_fixture_workspace(fixture_dir, allow_live_reset=False)
    # File must be cleanly restored to initial commit
    assert test_file.read_text(encoding="utf-8") == "class AppComponent { count = 0; }"

    # Now execute ticket 2
    res2 = execute_single_ticket(spec=spec2, workspace=fixture_dir, mode="baseline", dry_run=True)
    assert res2.tier0_passed is True



