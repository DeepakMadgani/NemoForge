"""Tests for Step 2: Transitive Type Hydration & TypeContract.

Verifies:
1. ImportResolver cleanly parses named imports and resolves relative TypeScript files.
2. TypeScriptTypeResolver resolves method parameters and hydrates Direct + Transitive types.
3. SEMANTIC SHAPE INVARIANT:
   - FilterParticipantMemberInput.email is an ARRAY: FilterStringInput[]
   - FilterStringInput exposes eq?: string
   - TypeDependencyEdge records is_array=True
4. Rendered TypeContract contains authoritative repository declarations and explicit array guidance.
5. End-to-end hydration on the real CC4E xchange-ui codebase (MemberService.members).
"""

from pathlib import Path
import pytest

from ticket_to_code.intelligence.typescript.import_resolver import TypeScriptImportResolver
from ticket_to_code.intelligence.typescript.type_resolver import TypeScriptTypeResolver
from ticket_to_code.intelligence.typescript.type_contract import MethodTypeContract


# ── Synthetic Multi-File Type Hierarchy ─────────────────────────────────────

@pytest.fixture
def synthetic_ts_repo(tmp_path: Path):
    """Build a synthetic TypeScript project modeling the real CC4E structure."""
    models_dir = tmp_path / "src" / "app" / "models"
    models_dir.mkdir(parents=True)
    services_dir = tmp_path / "src" / "app" / "services"
    services_dir.mkdir(parents=True)

    # 1. graphql.types.ts (Leaf transitive definitions)
    gql_types = models_dir / "graphql.types.ts"
    gql_types.write_text(
        "export interface FilterStringInput {\n"
        "  eq?: string;\n"
        "  ne?: string;\n"
        "  contains?: string;\n"
        "}\n\n"
        "export interface FilterIdInput {\n"
        "  eq?: string;\n"
        "  ne?: string;\n"
        "}\n",
        encoding="utf-8",
    )

    # 2. participating-members.ts (Intermediate parameter & return type)
    part_members = models_dir / "participating-members.ts"
    part_members.write_text(
        "import { FilterIdInput, FilterStringInput } from './graphql.types';\n\n"
        "export interface ParticipatingMember {\n"
        "  id?: string;\n"
        "  email: string;\n"
        "  name: string;\n"
        "  companyName?: string;\n"
        "}\n\n"
        "export interface FilterParticipantMemberInput {\n"
        "  projectId: FilterIdInput;\n"
        "  email?: FilterStringInput[];\n"
        "  userId?: FilterIdInput;\n"
        "  name?: FilterStringInput;\n"
        "}\n",
        encoding="utf-8",
    )

    # 3. member.service.ts (Service calling members())
    svc_file = services_dir / "member.service.ts"
    svc_file.write_text(
        "import { Observable } from 'rxjs';\n"
        "import { FilterParticipantMemberInput, ParticipatingMember } from '../models/participating-members';\n\n"
        "export class MemberService {\n"
        "  public members(filter: FilterParticipantMemberInput): Observable<ParticipatingMember[]> {\n"
        "    return null;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    return {
        "root": tmp_path,
        "service": svc_file,
        "model": part_members,
        "gql": gql_types,
    }


def test_import_resolver_parses_imports():
    content = (
        "import { FilterParticipantMemberInput, ParticipatingMember as PM } from '../../models/participating-members';\n"
        "import * as Rx from 'rxjs';\n"
    )
    imports = TypeScriptImportResolver.parse_named_imports(content)
    assert imports["FilterParticipantMemberInput"] == "../../models/participating-members"
    assert imports["PM"] == "../../models/participating-members"


def test_import_resolver_resolves_relative_path(synthetic_ts_repo):
    svc = synthetic_ts_repo["service"]
    resolved = TypeScriptImportResolver.resolve_module_path("../models/participating-members", svc)
    assert resolved is not None
    assert resolved.exists()
    assert resolved.name == "participating-members.ts"


def test_transitive_type_hydration_members(synthetic_ts_repo):
    """CRITICAL ACCEPTANCE TEST:
    Verify MemberService.members hydrates FilterParticipantMemberInput AND transitive FilterStringInput
    with EXACT semantic shape (email is an array, not a scalar).
    """
    svc = synthetic_ts_repo["service"]
    contract: MethodTypeContract = TypeScriptTypeResolver.build_method_type_contract(
        svc, "members", workspace_root=synthetic_ts_repo["root"]
    )

    assert contract.service_class == "MemberService"
    assert contract.method_name == "members"
    assert contract.parameter_name == "filter"
    assert contract.parameter_type == "FilterParticipantMemberInput"

    # 1. Direct type verified
    assert "FilterParticipantMemberInput" in contract.direct_types
    direct_def = contract.direct_types["FilterParticipantMemberInput"]
    assert direct_def.kind == "interface"
    assert "projectId: FilterIdInput" in direct_def.raw_declaration
    # STRICT ASSERTION: email MUST be an array
    assert "email?: FilterStringInput[]" in direct_def.raw_declaration

    # 2. Transitive types verified
    assert "FilterStringInput" in contract.transitive_types
    assert "FilterIdInput" in contract.transitive_types
    str_input_def = contract.transitive_types["FilterStringInput"]
    assert "eq?: string" in str_input_def.raw_declaration

    # 3. Dependency edges verified
    edge_map = {(e.parent_type, e.field_name): e for e in contract.dependency_edges}
    assert ("FilterParticipantMemberInput", "email") in edge_map
    email_edge = edge_map[("FilterParticipantMemberInput", "email")]
    assert email_edge.target_type == "FilterStringInput"
    assert email_edge.is_array is True, "email must be recognized as an ARRAY"

    # 4. Return type verified
    assert "ParticipatingMember" in contract.return_types
    assert "companyName" in contract.return_types["ParticipatingMember"].properties

    # 5. Rendered prompt block verification
    prompt_block = contract.render_prompt_block()
    assert "AUTHORITATIVE REPOSITORY TYPE CONTRACT" in prompt_block
    assert "FilterParticipantMemberInput" in prompt_block
    assert "FilterStringInput" in prompt_block
    assert "email[] → FilterStringInput (ARRAY" in prompt_block
    assert "pass an array: [ { eq: value } ]" in prompt_block
    assert "[RETURN TYPE] ParticipatingMember" in prompt_block
    assert "companyName?: string" in prompt_block


def test_live_cc4e_member_service_hydration():
    """Verify type hydration directly against the real CC4E repository if present."""
    cc4e_service = Path(r"C:\CC4E\xchange-ui\src\app\modules\shared\services\members\member.service.ts")
    if not cc4e_service.exists():
        pytest.skip("C:\\CC4E\\xchange-ui not present in environment")

    contract = TypeScriptTypeResolver.build_method_type_contract(
        cc4e_service, "members", workspace_root=r"C:\CC4E"
    )
    assert not contract.is_empty()
    assert "FilterParticipantMemberInput" in contract.direct_types
    assert "FilterStringInput" in contract.transitive_types

    # Ensure email is proven to be an array in the real model
    direct = contract.direct_types["FilterParticipantMemberInput"]
    assert "FilterStringInput[]" in direct.properties.get("email", "")

    # Ensure return type is hydrated and contains companyName
    assert "ParticipatingMember" in contract.return_types
    ret_def = contract.return_types["ParticipatingMember"]
    assert "companyName" in ret_def.properties
    assert "companyName?: String" in ret_def.raw_declaration

    # Verify prompt block renders return type
    prompt_block = contract.render_prompt_block()
    assert "[RETURN TYPE] ParticipatingMember" in prompt_block
    assert "companyName?: String" in prompt_block


def test_reproduce_actual_production_failure_and_enforce_array_shape(synthetic_ts_repo):
    """REPRODUCE ACTUAL PRODUCTION FAILURE & VERIFY ENFORCEMENT:

    // WRONG — observed production failure that caused TS2322:
    const filter = {
      projectId: { eq: this.contractId },
      email: { eq: searchDataElement }
    };

    // CORRECT — enforced by TypeContract:
    const filter = {
      projectId: { eq: this.contractId },
      email: [{ eq: searchDataElement }]
    };
    """
    svc = synthetic_ts_repo["service"]
    contract = TypeScriptTypeResolver.build_method_type_contract(
        svc, "members", workspace_root=synthetic_ts_repo["root"]
    )

    wrong_code = """
    const filter: FilterParticipantMemberInput = {
      projectId: { eq: this.contractId },
      email: { eq: searchDataElement }
    };
    this.memberService.members(filter).subscribe(...);
    """

    correct_code = """
    const filter: FilterParticipantMemberInput = {
      projectId: { eq: this.contractId },
      email: [{ eq: searchDataElement }]
    };
    this.memberService.members(filter).subscribe(...);
    """

    # 1. The WRONG code MUST be caught and flagged with a violation
    violations = contract.validate_code_snippet(wrong_code)
    assert len(violations) > 0, "Production bug (email as object instead of array) must be caught!"
    assert any("email" in v and "ARRAY" in v and "TS2322" in v for v in violations)

    # 2. The CORRECT code MUST pass validation cleanly
    clean_violations = contract.validate_code_snippet(correct_code)
    assert len(clean_violations) == 0, "Correct code with array wrapping must pass without violations"

