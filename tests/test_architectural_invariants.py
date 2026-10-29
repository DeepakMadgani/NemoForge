"""
End-to-End Architectural Invariants Test Suite for NemoForge.

Verifies the 8 core invariants:
1. Patch integrity (no ambiguous silent edits)
2. Scope integrity (format-neutral structural boundaries)
3. Build correctness (classification before repair)
4. Baseline awareness (pre-existing vs new regressions)
5. State boundedness (artifact references and token budget)
6. Environment isolation (ephemeral ports & browser discovery)
7. Verification honesty (inconclusive / blocked != pass)
8. Reproducibility & Policy arbitration
"""

import tempfile
from pathlib import Path
import pytest

from ticket_to_code.intelligence.repository_intelligence import RepositoryIntelligence
from ticket_to_code.editing.structural_editing import (
    StructuralEditingRegistry,
    YAMLEditingBackend,
    XMLEditingBackend,
    ConservativeSourcePreservingBackend,
    EditTarget,
)
from ticket_to_code.execution.build_orchestrator import (
    BuildFailureClassifier,
    FailureCategory,
    ClassifiedBuildResult,
)
from ticket_to_code.verification.baseline_verifier import (
    BaselineReport,
    FailureSignature,
    DifferentialStatus,
    BaselineDifferentialVerifier,
)
from ticket_to_code.verification.environment_manager import VerificationEnvironmentManager
from ticket_to_code.memory.bounded_state import (
    ContextAssembler,
    BoundedTicketToCodeState,
)
from ticket_to_code.policy.policy_engine import PolicyEngine, PolicyAction


def test_invariant_1_patch_integrity_ambiguity_rejection():
    """Invariant: No ambiguous target is silently edited."""
    backend = ConservativeSourcePreservingBackend()
    repeated_content = "SELECT * FROM orders;\nSELECT * FROM orders;\n"
    target = EditTarget(
        file_path="query.sql",
        target_identifier="test",
        start_line=1,
        end_line=1,
        matched_content="SELECT * FROM orders;",
    )
    res = backend.create_edit(target, "SELECT id FROM orders;", repeated_content)
    assert not res.success
    assert "Ambiguity violation" in (res.error or "")


def test_invariant_2_scope_integrity_yaml_formatting():
    """Invariant: YAML editing preserves structure and rejects invalid tabs."""
    backend = YAMLEditingBackend()
    val_bad = backend.validate("service.yaml", "services:\n\t- name: order")
    assert not val_bad.is_valid
    err_str = val_bad.syntax_error or ""
    assert "Tab character" in err_str or "\\t" in err_str

    val_good = backend.validate("service.yaml", "services:\n  - name: order\n    port: 8080\n")
    assert val_good.is_valid


def test_invariant_3_build_correctness_pre_repair_classification():
    """Invariant: Build failures are classified before source repair."""
    cat_src, _, actionable_src, _ = BuildFailureClassifier.classify(
        1, "[ERROR] OrderService.java:[45,12] error: cannot find symbol", "", ["mvn"]
    )
    assert cat_src == FailureCategory.SOURCE_COMPILATION_ERROR
    assert actionable_src is True

    cat_dep, _, actionable_dep, _ = BuildFailureClassifier.classify(
        1, "Could not resolve dependencies for project com.cloudforge:order-service", "", ["mvn"]
    )
    assert cat_dep == FailureCategory.DEPENDENCY_RESOLUTION_FAILURE
    assert actionable_dep is False

    cat_infra, _, actionable_infra, _ = BuildFailureClassifier.classify(
        1, "Docker client is not available in test runner", "", ["mvn"]
    )
    assert cat_infra == FailureCategory.INFRASTRUCTURE_FAILURE
    assert actionable_infra is False


def test_invariant_4_baseline_awareness_flaky_and_pre_existing():
    """Invariant: Existing baseline failures are distinguished from new regressions."""
    class MockOrchestrator:
        def execute_build(self, files):
            return ClassifiedBuildResult(
                success=False,
                exit_code=1,
                category=FailureCategory.TEST_ASSERTION_REGRESSION,
                summary="1 test failed",
                reproducible_evidence="",
                actionable_for_code_repair=True,
                command_executed=["mvn"],
                working_directory=".",
                error_locations=[{"test_case": "OrderServiceTest#testFlaky", "snippet": "Connection reset"}],
            )

    verifier = BaselineDifferentialVerifier(MockOrchestrator())
    base_report = verifier.capture_baseline(["OrderService.java"])
    assert len(base_report.failures) == 1

    diff = verifier.evaluate_candidate(["OrderService.java"], baseline=base_report)
    assert diff.status == DifferentialStatus.NO_REGRESSION_OBSERVED
    assert diff.is_safe_to_commit is True
    assert len(diff.new_failures) == 0
    assert len(diff.pre_existing_failures) == 1


def test_invariant_5_state_boundedness_and_artifact_storage():
    """Invariant: Heavy logs are offloaded to disk; model context remains bounded."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ca = ContextAssembler(Path(tmp_dir))
        heavy_log = "Error stack trace line...\n" * 2000  # ~50 KB
        ref = ca.store_artifact(heavy_log, "Compilation failure on OrderService.java:32")
        assert ref.byte_size > 40000

        state = BoundedTicketToCodeState(
            ticket_id="CF-204",
            ticket_title="Order hold release",
            ticket_description="Release hold on cancel",
            workspace_root=".",
            requirements_summary="OrderService must call payment client hold release.",
            evidence_refs=[ref],
        )
        prompt = ca.assemble_planning_context(state)
        # Bounded prompt must only contain the summary reference, not the 50 KB dump!
        assert len(prompt) < 1000
        assert "OrderService must call payment client" in prompt
        assert "Compilation failure on OrderService.java:32" in prompt


def test_invariant_6_environment_isolation_ephemeral_ports():
    """Invariant: Concurrency-safe ephemeral ports are allocated dynamically."""
    mgr = VerificationEnvironmentManager("run_alpha")
    p1 = mgr.allocate_ephemeral_port()
    p2 = mgr.allocate_ephemeral_port()
    assert p1 != p2
    assert p1 > 1024
    assert p2 > 1024


def test_invariant_7_verification_honesty_blocked_abstention():
    """Invariant: Blocked or inconclusive environments are never treated as passes."""
    ri = RepositoryIntelligence(tempfile.gettempdir())
    model = ri.analyze()
    policy = PolicyEngine(model)

    b_res = ClassifiedBuildResult(
        success=False,
        exit_code=1,
        category=FailureCategory.MISSING_TOOLCHAIN,
        summary="mvn: command not found",
        reproducible_evidence="bash: mvn: command not found",
        actionable_for_code_repair=False,
        command_executed=["mvn"],
        working_directory=".",
    )
    decision = policy.evaluate_build_failure(b_res, remaining_retries=3)
    assert decision.action == PolicyAction.ESCALATE_ENVIRONMENT_ERROR
    assert "MISSING_TOOLCHAIN" in decision.limitations[0]


def test_invariant_8_reproducible_repo_intelligence():
    """Invariant: Modules, build systems, and reactor scope are discovered dynamically."""
    # Run against the CloudForge-Commerce workspace that sits next to nemoforge/
    workspace = Path(__file__).resolve().parents[2] / "workspace" / "CloudForge-Commerce"
    if not workspace.is_dir():
        pytest.skip(f"CloudForge-Commerce workspace not present at {workspace}")
    ri = RepositoryIntelligence(str(workspace))
    model = ri.analyze()
    assert "order-service" in model.modules
    assert "cloudforge-store-ui" in model.modules
    cmd, cwd = model.get_reactor_build_scope(["order-service/src/main/java/OrderService.java"])
    assert len(cmd) > 0
    assert "order-service" in model.get_owning_module("order-service/src/main/java/OrderService.java").name
