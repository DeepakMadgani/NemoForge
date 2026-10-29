import pytest
from pathlib import Path
from ticket_to_code.runtime.fast_ide_engine import FastIDEEngine
from ticket_to_code.runtime.invariant_gates import InvariantEnforcementGates
from ticket_to_code.runtime.multi_zone_orchestrator import DynamicZoneDiscoverer, ZoneTier
from ticket_to_code.lean.call_chain_tracer import CallChainTracer


def test_invariant_18_generalized_rbac(tmp_path: Path):
    ticket = "Remove member user from organization workspace"
    
    # 1. Negative test: deleting parent aggregate with generic DAO
    bad_diff = """
+    public void removeUser(String orgId, String userId) {
+        organizationDao.delete(orgId);
+    }
"""
    passed, msgs = InvariantEnforcementGates.run_all_gates(tmp_path, ticket, "src/service.java", bad_diff)
    assert not passed
    assert any("Invariant 18" in m for m in msgs)

    # 2. Positive test: deleting subordinate join entity
    good_diff = """
+    public void removeUser(String orgId, String userId) {
+        organizationMemberDao.deleteByOrgIdAndUserId(orgId, userId);
+    }
"""
    passed, msgs = InvariantEnforcementGates.run_all_gates(tmp_path, ticket, "src/service.java", good_diff)
    assert passed
    assert len(msgs) == 0


def test_invariant_14_secrets_gate(tmp_path: Path):
    ticket = "Update auth configuration"
    
    # Negative test: hardcoded secret
    bad_diff = """
+    String apiKey = "sk-live-1234567890abcdef123456";
"""
    passed, msgs = InvariantEnforcementGates.run_all_gates(tmp_path, ticket, "src/auth.java", bad_diff)
    assert not passed
    assert any("Invariant 14" in m for m in msgs)


def test_dynamic_zone_discoverer_polyglot(tmp_path: Path):
    # Setup Nx app + Bazel build + .NET csproj
    (tmp_path / "apps" / "web").mkdir(parents=True)
    (tmp_path / "apps" / "web" / "project.json").write_text('{"projectType": "application"}', encoding="utf-8")
    
    (tmp_path / "services" / "gateway").mkdir(parents=True)
    (tmp_path / "services" / "gateway" / "BUILD.bazel").write_text('# Bazel build', encoding="utf-8")
    
    (tmp_path / "services" / "dotnet_api").mkdir(parents=True)
    (tmp_path / "services" / "dotnet_api" / "Api.csproj").write_text('<Project />', encoding="utf-8")

    zones = DynamicZoneDiscoverer.discover_zones(tmp_path)
    zone_names = {z.name for z in zones}
    assert "web" in zone_names
    assert "gateway" in zone_names
    assert "dotnet_api" in zone_names


def test_python_import_validation(tmp_path: Path):
    engine = FastIDEEngine(tmp_path, "Fix ticket")
    py_file = tmp_path / "service.py"
    py_file.write_text("def fetch_user(uid: str) -> Optional[Dict[str, Any]]:\n    return None\n", encoding="utf-8")
    
    engine._import_validation("service.py")
    content = py_file.read_text(encoding="utf-8")
    assert "from typing import Optional" in content
    assert "from typing import Dict" in content
    assert "from typing import Any" in content


def test_csharp_using_validation(tmp_path: Path):
    engine = FastIDEEngine(tmp_path, "Fix ticket")
    cs_file = tmp_path / "UserController.cs"
    cs_file.write_text("public class UserController {\n    public async Task<IActionResult> Get() {\n        return Ok();\n    }\n}\n", encoding="utf-8")
    
    engine._import_validation("UserController.cs")
    content = cs_file.read_text(encoding="utf-8")
    assert "using System.Threading.Tasks;" in content
    assert "using Microsoft.AspNetCore.Mvc;" in content


def test_polyglot_rest_endpoint_tracing(tmp_path: Path):
    tracer = CallChainTracer(tmp_path)
    # Create Python FastAPI route
    py_dir = tmp_path / "backend"
    py_dir.mkdir(parents=True)
    py_file = py_dir / "routes.py"
    py_file.write_text("""
from fastapi import APIRouter
router = APIRouter()

@router.delete("/users/{user_id}")
def delete_user(user_id: str):
    return {"status": "ok"}
""", encoding="utf-8")

    res = tracer.trace("delete user from users endpoint")
    assert len(res["rest_endpoints"]) > 0
    ep = res["rest_endpoints"][0]
    assert ep["http_method"] == "DELETE"
    assert "/users/{user_id}" in ep["path"]
    assert ep["framework"] == "FastAPI/Flask"
    assert res["is_fully_resolved"]

