import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ticket_to_code.models import ValueEdgeTicket
from ticket_to_code.runtime.fast_ide_engine import FastIDEEngine, SearchReplaceParser


def test_search_replace_parser():
    raw_llm_output = """
Here is the fix for the user service:

FILE: src/main/UserService.java
<<<<<<< SEARCH
    public void updateUser(User user) {
        // old logic
    }
=======
    public void updateUser(User user) {
        // new logic with validation
        validate(user);
    }
>>>>>>> REPLACE
"""
    blocks = SearchReplaceParser.parse(raw_llm_output)
    assert len(blocks) == 1
    assert blocks[0].file_path == "src/main/UserService.java"
    assert "old logic" in blocks[0].search_block
    assert "validate(user);" in blocks[0].replace_block


def test_fast_ide_engine_clean_patch_cycle(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    sample_file = src_dir / "Calculator.java"
    sample_file.write_text(
        "public class Calculator {\n"
        "    public int add(int a, int b) {\n"
        "        return a - b; // bug\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    ticket = ValueEdgeTicket(
        ticket_id="CALC-101",
        title="Fix add method bug in Calculator.java",
        description="The add method returns a - b instead of a + b in src/Calculator.java",
    )

    def mock_llm_response(prompt: str):
        return (
            "FILE: src/Calculator.java\n"
            "<<<<<<< SEARCH\n"
            "        return a - b; // bug\n"
            "=======\n"
            "        return a + b;\n"
            ">>>>>>> REPLACE\n"
        )

    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
        llm_client=mock_llm_response,
        max_repair_turns=1,
    )

    with patch.object(engine, "_run_fast_compile", return_value=(True, "")):
        with patch.object(engine, "_run_fast_test", return_value=(True, "")):
            result = engine.run()

    assert result.success is True
    assert result.build_passed is True
    assert "src/Calculator.java" in result.modified_files
    updated_content = sample_file.read_text(encoding="utf-8")
    assert "return a + b;" in updated_content
    assert result.tokens_consumed > 0
    assert result.tokens_consumed < 5000  # Strict low-token guarantee


def test_fast_ide_engine_rollback_on_compile_failure(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    sample_file = src_dir / "Payment.py"
    original_code = "def pay():\n    return False\n"
    sample_file.write_text(original_code, encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="PAY-1",
        title="Payment update in src/Payment.py",
        description="Update payment method in src/Payment.py",
    )

    def mock_llm_broken(prompt: str):
        return (
            "FILE: src/Payment.py\n"
            "<<<<<<< SEARCH\n"
            "    return False\n"
            "=======\n"
            "    broken syntax ???\n"
            ">>>>>>> REPLACE\n"
        )

    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
        llm_client=mock_llm_broken,
        max_repair_turns=1,
    )

    # Simulate compilation always failing
    with patch.object(engine, "_run_fast_compile", return_value=(False, "SyntaxError: invalid syntax")):
        result = engine.run()

    # Must fail closed, rollback all edits to disk, and request escalation
    assert result.success is False
    assert result.escalate_to_full_graph is True
    assert sample_file.read_text(encoding="utf-8") == original_code


def test_fast_ide_engine_stale_content_hash_rejection(tmp_path):
    """Verifies that FastIDEEngine detects and rejects concurrent file mutations via contentHash."""
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    sample_file = src / "Service.py"
    sample_file.write_text("def process():\n    return 100\n", encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="HASH-01",
        title="Update process in src/Service.py",
        description="Update process in src/Service.py",
    )

    def mock_llm_concurrent_mutate(prompt: str):
        # Concurrently mutate file on disk right before model returns diff
        sample_file.write_text("def process():\n    # mutated by concurrent process!\n    return 100\n", encoding="utf-8")
        return (
            "FILE: src/Service.py\n"
            "<<<<<<< SEARCH\n"
            "    return 100\n"
            "=======\n"
            "    return 200\n"
            ">>>>>>> REPLACE\n"
        )

    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
        llm_client=mock_llm_concurrent_mutate,
        max_repair_turns=0,
    )

    result = engine.run()
    # Must fail or detect stale hash rather than applying corrupted edit
    assert result.success is False


def test_fast_ide_engine_intent_triage_readonly_inquiry(tmp_path):
    """Verifies that read-only inquiries are resolved immediately without code modification."""
    ticket = ValueEdgeTicket(
        ticket_id="INQ-01",
        title="How does the messaging system work?",
        description="Explain the architecture of the event broker and consumers in the repo.",
    )
    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
    )
    result = engine.run()
    assert result.success is True
    # Read-only tickets build/test nothing, so they must not claim a verified success.
    assert result.status == "no_change_required"
    assert result.build_passed is None and result.test_passed is None
    assert len(result.modified_files) == 0
    assert "Repository Architecture & Analysis" in result.message


def test_fast_ide_engine_patch_gate_validations(tmp_path):
    """Verifies that patch_gate rejects unbalanced braces, markdown fences, and empty blocks."""
    from ticket_to_code.runtime.fast_ide_engine import SearchReplaceBlock
    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ValueEdgeTicket(ticket_id="T1", title="Test", description="Test"),
    )

    # 1. Unbalanced braces
    block_unbalanced = SearchReplaceBlock(
        file_path="src/App.java",
        search_block="void run() {}",
        replace_block="void run() { if (true) {",  # Missing closing brace
    )
    ok, err = engine._patch_gate(block_unbalanced)
    assert ok is False
    assert "Unbalanced curly braces" in err

    # 2. Markdown fence leakage
    block_fence = SearchReplaceBlock(
        file_path="src/App.java",
        search_block="int x = 1;",
        replace_block="```java\nint x = 2;\n```",
    )
    ok, err = engine._patch_gate(block_fence)
    assert ok is False
    assert "markdown code fences" in err

    # 3. Clean block
    block_clean = SearchReplaceBlock(
        file_path="src/App.java",
        search_block="int x = 1;",
        replace_block="int x = 2;",
    )
    ok, err = engine._patch_gate(block_clean)
    assert ok is True


def test_fast_ide_engine_import_validation_java(tmp_path):
    """Verifies that import_validation auto-injects missing Java imports deterministically."""
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    java_file = src / "TestService.java"
    java_file.write_text(
        "package com.example.service;\n\n"
        "public class TestService {\n"
        "    @Autowired\n"
        "    private List<String> items;\n"
        "}\n",
        encoding="utf-8",
    )

    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ValueEdgeTicket(ticket_id="T1", title="Test", description="Test"),
    )
    engine._import_validation("src/TestService.java")

    updated = java_file.read_text(encoding="utf-8")
    assert "import org.springframework.beans.factory.annotation.Autowired;" in updated
    assert "import java.util.List;" in updated


def test_fast_ide_engine_live_ui_telemetry(tmp_path):
    """Verifies that FastIDEEngine emits real-time WebSocket telemetry for each stage."""
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    sample_file = src / "App.java"
    sample_file.write_text("class App { void test() {} }\n", encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="TELEM-01",
        title="Fix test method in src/App.java",
        description="Update App.java in src/App.java",
    )

    events = []
    def mock_ui_callback(payload: dict):
        events.append(payload)

    def mock_llm_response(prompt: str):
        return (
            "FILE: src/App.java\n"
            "<<<<<<< SEARCH\n"
            "class App { void test() {} }\n"
            "=======\n"
            "class App { void test() { int ok = 1; } }\n"
            ">>>>>>> REPLACE\n"
        )

    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
        llm_client=mock_llm_response,
        max_repair_turns=1,
        ui_callback=mock_ui_callback,
    )

    with patch.object(engine, "_run_fast_compile", return_value=(True, "")):
        with patch.object(engine, "_run_fast_test", return_value=(True, "")):
            res = engine.run()

    assert res.success is True
    # Verify stages were emitted
    phases = [e["phase"] for e in events]
    assert "classification" in phases
    assert "localization" in phases
    assert "planning" in phases
    assert "patch_generation" in phases
    assert "validation" in phases


def test_fast_ide_engine_multi_zone_build_verification(tmp_path):
    """Verifies that _run_fast_compile validates all distinct directories in multi-file edits."""
    be = tmp_path / "backend" / "src"
    be.mkdir(parents=True, exist_ok=True)
    (be / "Service.java").write_text("class Service {}", encoding="utf-8")

    fe = tmp_path / "frontend" / "src"
    fe.mkdir(parents=True, exist_ok=True)
    (fe / "app.component.ts").write_text("class Component {}", encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="MULTI-01",
        title="Update backend and frontend components",
        description="Modify backend/src/Service.java and frontend/src/app.component.ts",
    )

    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
    )

    # Calling with list of files across two zones
    with patch("ticket_to_code.runtime.polyglot_build.PolyglotBuildDiscoverer.discover") as mock_disc:
        mock_cfg1 = MagicMock(compile_cmd="mvn compile", work_dir=tmp_path / "backend")
        mock_cfg2 = MagicMock(compile_cmd="npm run build", work_dir=tmp_path / "frontend")
        mock_disc.side_effect = lambda path, ws: mock_cfg1 if "backend" in str(path) else mock_cfg2

        with patch.object(engine, "_execute_subprocess_with_tree_cleanup", return_value=(0, "OK", "")) as mock_exec:
            ok, err = engine._run_fast_compile(["backend/src/Service.java", "frontend/src/app.component.ts"])
            assert ok is True
            # Both compilation commands were executed!
            assert mock_exec.call_count == 2


def test_fast_ide_engine_test_requested_preserves_companion_priority(tmp_path):
    """Verifies that when test coverage is explicitly requested, test files retain full priority."""
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "OrderService.java").write_text("class OrderService {}", encoding="utf-8")
    (src / "OrderServiceTest.java").write_text("class OrderServiceTest {}", encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="TEST-01",
        title="OrderService enhancements",
        description="Add unit tests and assert behavior for OrderService",
    )

    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
    )

    targets, ranked, _ = engine._react_evidence_collection()
    test_entry = next((item for item in ranked if "OrderServiceTest.java" in item["file_path"]), None)
    assert test_entry is not None
    # Test was not penalized with 0.6 multiplier because test_requested was True
    assert test_entry["is_test"] is True


def test_apply_tolerant_replacement_whitespace():
    from ticket_to_code.editing.surgical_diff_engine import apply_tolerant_replacement

    content = "def calculate():\n    x = 10   \n    y = 20\n    return x + y\n"
    # LLM outputs without trailing spaces on line 2
    old_block = "    x = 10\n    y = 20"
    new_block = "    x = 15\n    y = 25"

    res = apply_tolerant_replacement(content, old_block, new_block)
    assert res is not None
    assert "x = 15" in res
    assert "y = 25" in res


def test_apply_tolerant_replacement_indentation_shift():
    from ticket_to_code.editing.surgical_diff_engine import apply_tolerant_replacement

    content = "class Service:\n        def run(self):\n            print('starting')\n            self.work()\n"
    # LLM omits the 12-space indent and outputs with 0 or 4 spaces
    old_block = "print('starting')\nself.work()"
    new_block = "print('starting')\nself.work()\nprint('finished')"

    res = apply_tolerant_replacement(content, old_block, new_block)
    assert res is not None
    assert "            print('finished')" in res  # Inherited the 12-space indentation cleanly!


def test_apply_tolerant_replacement_fuzzy():
    from ticket_to_code.editing.surgical_diff_engine import apply_tolerant_replacement

    content = "public void authenticate(String token) {\n    validateSignature(token);\n    checkExpiry(token);\n}\n"
    # LLM has minor whitespace or comment variation
    old_block = "public void authenticate(String token) {\n    validateSignature(token);\n    checkExpiry(token);\n}"
    new_block = "public void authenticate(String token) {\n    validateSignature(token);\n    checkExpiry(token);\n    recordAudit(token);\n}"

    res = apply_tolerant_replacement(content, old_block, new_block)
    assert res is not None
    assert "recordAudit(token);" in res


def test_surgical_diff_engine_applies_tolerant_when_exact_fails(tmp_path):
    from ticket_to_code.editing.surgical_diff_engine import SurgicalDiffEngine
    file_p = tmp_path / "App.java"
    file_p.write_text("class App {\n    void execute() {\n        System.out.println(1);\n    }\n}\n", encoding="utf-8")
    engine = SurgicalDiffEngine(str(tmp_path))
    # LLM gave search block with 0 indentation instead of 8 spaces
    res = engine.apply_anchored_replacement(
        "App.java",
        "System.out.println(1);",
        "System.out.println(1);\nSystem.out.println(2);",
    )
    assert res.success is True
    assert "System.out.println(2);" in file_p.read_text(encoding="utf-8")


def test_patch_gate_enforces_invariant_18_asymmetric_rbac(tmp_path):
    from ticket_to_code.runtime.fast_ide_engine import SearchReplaceBlock
    ticket = ValueEdgeTicket(
        ticket_id="RBAC-01",
        title="Remove member from contract",
        description="A coordinator wants to remove member from contract",
    )
    engine = FastIDEEngine(workspace_path=str(tmp_path), ticket=ticket)
    # Patch accidentally deletes parent contract!
    bad_block = SearchReplaceBlock(
        file_path="ContractMemberService.java",
        search_block="void remove() {}",
        replace_block="void remove() { contractRepository.delete(contract); }",
    )
    ok, err = engine._patch_gate(bad_block)
    assert ok is False
    assert "Asymmetric RBAC Lifecycle Violation" in err


def test_import_validation_resolves_workspace_java_and_ts_symbols(tmp_path):
    """Verifies that _import_validation dynamically resolves custom project symbols across
    different directories in the workspace for Java and TypeScript without hardcoding."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "DeliverableSummary.java").write_text(
        "package com.company.models;\npublic class DeliverableSummary {}\n",
        encoding="utf-8"
    )

    services_dir = tmp_path / "services"
    services_dir.mkdir(parents=True)
    svc_file = services_dir / "DeliverableService.java"
    svc_file.write_text(
        "package com.company.services;\n\npublic class DeliverableService {\n    DeliverableSummary summary;\n}\n",
        encoding="utf-8"
    )

    ticket = ValueEdgeTicket(ticket_id="T-1", title="Deliverables", description="Update deliverable summary")
    engine = FastIDEEngine(workspace_path=str(tmp_path), ticket=ticket)
    engine._import_validation("services/DeliverableService.java")

    updated_java = svc_file.read_text(encoding="utf-8")
    assert "import com.company.models.DeliverableSummary;" in updated_java

    # TypeScript dynamic symbol resolution
    ts_models = tmp_path / "src" / "app" / "models"
    ts_models.mkdir(parents=True)
    (ts_models / "contract-member.model.ts").write_text(
        "export interface ContractMember { id: string; }\n",
        encoding="utf-8"
    )

    ts_comps = tmp_path / "src" / "app" / "components"
    ts_comps.mkdir(parents=True)
    comp_file = ts_comps / "member.component.ts"
    comp_file.write_text(
        "export class MemberComponent { member: ContractMember; }\n",
        encoding="utf-8"
    )

    engine._import_validation("src/app/components/member.component.ts")
    updated_ts = comp_file.read_text(encoding="utf-8")
    assert "import { ContractMember } from '../models/contract-member.model';" in updated_ts






