"""
Tests for Multi-File, Cross-Folder Working Set and Strict Test Verification Gates

Simulates enterprise tickets like CF-REFUND-101:
1. Multi-File Working Set Resolution (Repository + Service + Test + Migration)
2. Coordinated Multi-File SEARCH/REPLACE Execution
3. New File Creation (Flyway Migration SQL)
4. Strict Non-Zero Test Verification Gate
5. Atomic Transaction Rollback across Multiple Folders
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ticket_to_code.models import ValueEdgeTicket
from ticket_to_code.runtime.fast_ide_engine import FastIDEEngine, FastIDEResult
from ticket_to_code.editing.surgical_diff_engine import SurgicalDiffEngine, EditStatus


def test_surgical_diff_engine_creates_new_file_and_rolls_back(tmp_path):
    """Verifies that SurgicalDiffEngine can create a new file (e.g. Flyway SQL) and cleanly delete it on rollback."""
    diff_engine = SurgicalDiffEngine(str(tmp_path))
    new_sql_path = "db/migration/V11__Remove_ghost_members.sql"
    
    # 1. Create new file with empty old_block
    res = diff_engine.apply_anchored_replacement(
        rel_path=new_sql_path,
        old_block="",
        new_block="DELETE FROM participating_member WHERE role_name = 'Member';\n",
    )
    assert res.success is True
    created_file = tmp_path / "db" / "migration" / "V11__Remove_ghost_members.sql"
    assert created_file.is_file()
    assert "DELETE FROM participating_member" in created_file.read_text(encoding="utf-8")

    # 2. Rollback should cleanly delete the newly created file
    diff_engine.rollback_all()
    assert not created_file.exists()


def test_surgical_diff_engine_stale_content_hash_rejection(tmp_path):
    """Verifies that SurgicalDiffEngine rejects modifications if expected_content_hash is stale."""
    diff_engine = SurgicalDiffEngine(str(tmp_path))
    file_path = tmp_path / "Model.java"
    file_path.write_text("class Model {\n    int id = 1;\n}\n", encoding="utf-8")

    real_hash = diff_engine.compute_content_hash(file_path.read_text(encoding="utf-8"))

    # Stale hash attempt -> MUST FAIL
    res_stale = diff_engine.apply_anchored_replacement(
        rel_path="Model.java",
        old_block="    int id = 1;",
        new_block="    int id = 2;",
        expected_content_hash="deadbeef12345678",
    )
    assert res_stale.success is False
    assert res_stale.status == EditStatus.STALE_CONTENT_HASH
    assert "Stale Content Hash" in res_stale.message

    # Valid hash attempt -> MUST SUCCEED
    res_ok = diff_engine.apply_anchored_replacement(
        rel_path="Model.java",
        old_block="    int id = 1;",
        new_block="    int id = 2;",
        expected_content_hash=real_hash,
    )
    assert res_ok.success is True
    assert "int id = 2;" in file_path.read_text(encoding="utf-8")


def test_multi_file_working_set_discovery(tmp_path):
    """Verifies that FastIDEEngine discovers multiple files across folders including companion tests."""
    repo = tmp_path / "project-service"
    
    # 1. Create Repository in repository/
    repo_dir = repo / "src" / "main" / "java" / "com" / "cloudforge" / "repository"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "ContractMemberRepository.java").write_text("public interface ContractMemberRepository {}", encoding="utf-8")
    
    # 2. Create Service in service/
    svc_dir = repo / "src" / "main" / "java" / "com" / "cloudforge" / "service"
    svc_dir.mkdir(parents=True, exist_ok=True)
    (svc_dir / "ParticipatingMemberService.java").write_text("public class ParticipatingMemberService {}", encoding="utf-8")
    
    # 3. Create Companion Test in test/
    test_dir = repo / "src" / "test" / "java" / "com" / "cloudforge" / "service"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "ParticipatingMemberServiceTest.java").write_text("public class ParticipatingMemberServiceTest {}", encoding="utf-8")

    ticket = ValueEdgeTicket(
        ticket_id="CF-REFUND-101",
        title="Prevent Ghost Members in ParticipatingMemberService and ContractMemberRepository",
        description="Check ContractMemberRepository before demoting coordinators in ParticipatingMemberService.",
    )

    engine = FastIDEEngine(workspace_path=str(tmp_path), ticket=ticket)
    targets = engine._locate_target_files()

    assert any("ContractMemberRepository.java" in t for t in targets)
    assert any("ParticipatingMemberService.java" in t for t in targets)
    assert any("ParticipatingMemberServiceTest.java" in t for t in targets)


def test_multi_file_coordinated_execution_and_strict_test_gate(tmp_path):
    """
    Simulates CF-REFUND-101:
    Coordinates edits across ContractMemberRepository, ParticipatingMemberService, and Flyway migration SQL.
    Verifies that the strict test gate rejects '0 tests run' and succeeds when >= 1 test passes.
    """
    repo = tmp_path / "project-service"
    repo_dir = repo / "src" / "main" / "java" / "com" / "cloudforge" / "repository"
    repo_dir.mkdir(parents=True, exist_ok=True)
    repo_file = repo_dir / "ContractMemberRepository.java"
    repo_file.write_text(
        "package com.cloudforge.repository;\n"
        "public interface ContractMemberRepository {\n"
        "    // existing methods\n"
        "}\n",
        encoding="utf-8",
    )

    svc_dir = repo / "src" / "main" / "java" / "com" / "cloudforge" / "service"
    svc_dir.mkdir(parents=True, exist_ok=True)
    svc_file = svc_dir / "ParticipatingMemberService.java"
    svc_file.write_text(
        "package com.cloudforge.service;\n"
        "public class ParticipatingMemberService {\n"
        "    public void updateMember() {\n"
        "        save(); // Unchecked\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    ticket = ValueEdgeTicket(
        ticket_id="CF-REFUND-101",
        title="Project Service: Prevent Ghost Members by Removing Project-Level Coordinators with No Contracts",
        description="Update ContractMemberRepository.java, ParticipatingMemberService.java, and add Flyway migration.",
    )

    # Multi-file LLM output
    rel_repo = str(repo_file.relative_to(tmp_path)).replace("\\", "/")
    rel_svc = str(svc_file.relative_to(tmp_path)).replace("\\", "/")
    rel_sql = "project-service/src/main/resources/db/migration/V11__Remove_ghost_members.sql"

    multi_file_llm_response = f"""
FILE: {rel_repo}
<<<<<<< SEARCH
    // existing methods
=======
    long countByUserIdAndProjectId(String userId, String projectId);
>>>>>>> REPLACE

FILE: {rel_svc}
<<<<<<< SEARCH
        save(); // Unchecked
=======
        if (contractCount == 0) deleteMember(); else save();
>>>>>>> REPLACE

FILE: {rel_sql}
<<<<<<< SEARCH
=======
DELETE FROM participating_member WHERE role_name = 'Member';
>>>>>>> REPLACE
"""

    engine = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
        llm_client=lambda prompt: multi_file_llm_response,
        max_repair_turns=1,
    )

    # 1. Test Strict Test Gate: Rejects 0 tests run
    with patch.object(engine, "_run_fast_compile", return_value=(True, "")):
        with patch.object(engine, "_run_fast_test", return_value=(False, "Strict Verification Gate Rejection: 0 tests completed.")):
            res_fail = engine.run()
            assert res_fail.success is False
            # Verify atomic rollback
            assert not (tmp_path / rel_sql).exists()

    # 2. Test Success Path: >= 1 test passed
    engine_success = FastIDEEngine(
        workspace_path=str(tmp_path),
        ticket=ticket,
        llm_client=lambda prompt: multi_file_llm_response,
        max_repair_turns=1,
    )
    with patch.object(engine_success, "_run_fast_compile", return_value=(True, "")):
        with patch.object(engine_success, "_run_fast_test", return_value=(True, "Tests: 2 passed, 0 failed.")):
            res_success = engine_success.run()
            assert res_success.success is True
            assert len(res_success.modified_files) == 3
            assert (tmp_path / rel_sql).is_file()
            assert "countByUserIdAndProjectId" in repo_file.read_text(encoding="utf-8")
            assert "deleteMember()" in svc_file.read_text(encoding="utf-8")
