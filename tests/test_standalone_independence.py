"""
Test Standalone Independence and Runtime Modules.

Validates that all standalone ticket_to_code.runtime modules function
correctly with zero external dependency on aviator or aviator_core.
"""

import tempfile
from pathlib import Path
import pytest

from ticket_to_code.runtime.compat_models import StateModel, AgentState, WorkflowState
from ticket_to_code.runtime.settings import settings, LLMProvider
from ticket_to_code.runtime.storage import SqliteStore, Neo4jStore
from ticket_to_code.runtime.vector_store import VectorStoreManager
from ticket_to_code.runtime.indexer import index_file, index_repository
from ticket_to_code.runtime.localizer import HybridLocalizer, FileCandidate, SpringFilter, SpringContext
from ticket_to_code.runtime.patch_engine import (
    EnhancedPatchEngine,
    EnhancedPatchApplicator,
    ContextExpander,
    FileCandidate as PEFileCandidate,
)


def test_compat_models():
    class CustomState(StateModel):
        count: int = 0
        notes: list[str] = []

    state = CustomState(count=5, notes=["a", "b"])
    assert state.count == 5
    assert state.notes == ["a", "b"]

    agent_state = AgentState(active_file="foo.py", iteration=2)
    assert agent_state.active_file == "foo.py"

    wf_state = WorkflowState(status="running")
    assert wf_state.status == "running"


def test_settings_singleton():
    assert settings is not None
    assert hasattr(settings, "llm_provider")
    assert hasattr(settings, "database_url")
    assert LLMProvider.GOOGLE_GENAI == "google-genai"


def test_sqlite_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SqliteStore(str(db_path))

        with store.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            cursor.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
            conn.commit()

        with store.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users WHERE id = 1")
            row = cursor.fetchone()
            assert row[0] == "Alice"
        store.close()


def test_vector_store_manager():
    vsm = VectorStoreManager()
    store = vsm.get_vector_store("test_collection")
    assert store is not None
    # Add texts should succeed (mock/noop or connected)
    ids = store.add_texts(["def hello(): pass"], metadatas=[{"file": "test.py"}])
    assert isinstance(ids, list)
    results = store.similarity_search("hello", k=2)
    assert isinstance(results, list)


def test_standalone_indexer():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        py_file = root / "sample.py"
        py_file.write_text(
            "class SampleClass:\n"
            "    def sample_method(self):\n"
            "        pass\n"
            "\n"
            "def standalone_func():\n"
            "    pass\n",
            encoding="utf-8"
        )

        db_path = root / "index.db"
        index_repository(root, db_path)

        assert db_path.exists()

        # Query SQLite to ensure symbols were extracted
        store = SqliteStore(str(db_path))
        with store.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, type FROM symbols ORDER BY name")
            symbols = cursor.fetchall()
        store.close()

        names = [s[0] for s in symbols]
        assert "SampleClass" in names
        assert "sample_method" in names
        assert "standalone_func" in names


def test_standalone_localizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        file1 = root / "service.py"
        file1.write_text("class PaymentService:\n    def process_payment(): pass", encoding="utf-8")
        file2 = root / "controller.py"
        file2.write_text("class PaymentController:\n    pass", encoding="utf-8")

        db_path = root / "index.db"
        index_repository(root, db_path)

        localizer = HybridLocalizer(root, db_path)
        candidates = localizer.localize("Fix PaymentService process_payment", top_k=5)
        localizer.close()

        assert len(candidates) > 0
        top_paths = [c.path for c in candidates]
        assert any("service.py" in p for p in top_paths)


def test_patch_engine_and_applicator():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target = root / "math_lib.py"
        target.write_text(
            "def add(a, b):\n"
            "    return a - b\n",
            encoding="utf-8"
        )

        patch_engine = EnhancedPatchEngine()
        diff = (
            "--- a/math_lib.py\n"
            "+++ b/math_lib.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b\n"
            "+    return a + b\n"
        )

        hunknodes = patch_engine.parse(diff)
        assert len(hunknodes) >= 1

        applicator = EnhancedPatchApplicator(root)
        success, msg = applicator.apply_patch(diff)
        assert success

        new_content = target.read_text(encoding="utf-8")
        assert "return a + b" in new_content


def test_zero_aviator_imports():
    import ast
    src_dir = Path(__file__).parent.parent / "src" / "ticket_to_code"

    violations = []
    for py_file in src_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "aviator" or alias.name.startswith("aviator.") or alias.name == "aviator_core" or alias.name.startswith("aviator_core."):
                        violations.append(f"{py_file.name}:{node.lineno} -> import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "aviator" or node.module.startswith("aviator.") or node.module == "aviator_core" or node.module.startswith("aviator_core.")):
                    violations.append(f"{py_file.name}:{node.lineno} -> from {node.module} import ...")

    assert not violations, f"Found forbidden aviator imports: {violations}"
