"""Tests for PolyglotCompanionResolver."""
from pathlib import Path
import pytest
from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver


def test_resolve_angular_companion(tmp_path: Path):
    comp_ts = tmp_path / "user-profile.component.ts"
    comp_ts.write_text("export class UserProfileComponent {}", encoding="utf-8")
    comp_html = tmp_path / "user-profile.component.html"
    comp_html.write_text("<div>Profile</div>", encoding="utf-8")

    # Resolve from HTML to TS
    res = PolyglotCompanionResolver.resolve_companion(str(comp_html), workspace_path=str(tmp_path))
    assert res is not None
    assert res[0].endswith("user-profile.component.ts")
    assert "UserProfileComponent" in res[1]

    # Resolve from TS to HTML
    res_html = PolyglotCompanionResolver.resolve_companion(str(comp_ts), workspace_path=str(tmp_path))
    assert res_html is not None
    assert res_html[0].endswith("user-profile.component.html")
    assert "Profile" in res_html[1]


def test_resolve_blazor_companion(tmp_path: Path):
    razor_cs = tmp_path / "Counter.razor.cs"
    razor_cs.write_text("public partial class Counter { int count = 0; }", encoding="utf-8")
    razor = tmp_path / "Counter.razor"
    razor.write_text("<button @onclick=\"Increment\">@count</button>", encoding="utf-8")

    res = PolyglotCompanionResolver.resolve_companion(str(razor), workspace_path=str(tmp_path))
    assert res is not None
    assert res[0].endswith("Counter.razor.cs")
    assert "count = 0" in res[1]


def test_resolve_react_companion(tmp_path: Path):
    css = tmp_path / "Button.module.css"
    css.write_text(".primary { color: blue; }", encoding="utf-8")
    tsx = tmp_path / "Button.tsx"
    tsx.write_text("export const Button = () => <button />;", encoding="utf-8")

    res = PolyglotCompanionResolver.resolve_companion(str(tsx), workspace_path=str(tmp_path))
    assert res is not None
    assert res[0].endswith("Button.module.css")
    assert ".primary" in res[1]


def test_resolve_python_views_companion(tmp_path: Path):
    templates = tmp_path / "templates"
    templates.mkdir()
    tpl = templates / "dashboard.html"
    tpl.write_text("<h1>Dashboard</h1>", encoding="utf-8")
    views = tmp_path / "views.py"
    views.write_text("def dashboard(request): pass", encoding="utf-8")

    res = PolyglotCompanionResolver.resolve_companion(str(tpl), workspace_path=str(tmp_path))
    assert res is not None
    assert res[0].endswith("views.py")
    assert "dashboard" in res[1]


def test_resolve_from_session_memory_priority(tmp_path: Path):
    comp_html = tmp_path / "test.component.html"
    comp_html.write_text("<div></div>", encoding="utf-8")

    # Disk has old version
    comp_ts = tmp_path / "test.component.ts"
    comp_ts.write_text("export class OldComponent {}", encoding="utf-8")

    # Session memory has uncommitted active changes
    session_files = {
        str(comp_ts): "export class UpdatedComponent { public inMemory: boolean = true; }"
    }

    res = PolyglotCompanionResolver.resolve_companion(
        str(comp_html),
        session_files=session_files,
        workspace_path=str(tmp_path)
    )
    assert res is not None
    # Must prioritize session memory over disk
    assert "UpdatedComponent" in res[1]
    assert "inMemory" in res[1]


def test_resolve_all_companions_angular(tmp_path: Path):
    comp_ts = tmp_path / "add-members.component.ts"
    comp_ts.write_text("export class AddMembersComponent {}", encoding="utf-8")
    comp_html = tmp_path / "add-members.component.html"
    comp_html.write_text("<div>Modal</div>", encoding="utf-8")
    comp_scss = tmp_path / "add-members.component.scss"
    comp_scss.write_text(".modal { display: block; }", encoding="utf-8")

    comps = PolyglotCompanionResolver.resolve_all_companions(str(comp_ts), workspace_path=str(tmp_path))
    assert len(comps) == 2
    comp_names = [Path(c).name for c in comps]
    assert "add-members.component.html" in comp_names
    assert "add-members.component.scss" in comp_names


def test_cross_service_contract_resolver(tmp_path: Path):
    from ticket_to_code.intelligence.contracts.companion_resolver import CrossServiceContractResolver

    # Create issues-service controller
    ctrl_dir = tmp_path / "issues-service/src/main/java/com/cloudforge/controller"
    ctrl_dir.mkdir(parents=True)
    ctrl_file = ctrl_dir / "IssuesCollectionController.java"
    ctrl_file.write_text(
        '@RestController\n@RequestMapping("/api/v1/issues")\npublic class IssuesCollectionController {\n'
        '  @GetMapping("/count")\n  public IssueCount getCount() { return null; }\n}',
        encoding="utf-8"
    )

    # Create bim-gateway config
    gw_dir = tmp_path / "bim-gateway/src/main/resources"
    gw_dir.mkdir(parents=True)
    gw_file = gw_dir / "application.yml"
    gw_file.write_text(
        'spring:\n  cloud:\n    gateway:\n      routes:\n        - id: issues\n          predicates:\n            - Path=/api/v1/issues/**',
        encoding="utf-8"
    )

    # Create xchange-ui service
    ui_dir = tmp_path / "xchange-ui/src/app/services"
    ui_dir.mkdir(parents=True)
    ui_file = ui_dir / "deliverables.service.ts"
    ui_file.write_text(
        'export class DeliverablesService {\n  getCount() {\n    return this.http.get("/api/v1/issues/count");\n  }\n}',
        encoding="utf-8"
    )

    # Test resolving contracts from controller
    contracts = CrossServiceContractResolver.resolve_contracts(str(ctrl_file), str(tmp_path))
    assert any("application.yml" in g for g in contracts["gateway_routes"])
    assert any("deliverables.service.ts" in s for s in contracts["frontend_services"])

    # Test resolving companions from UI service
    ui_contracts = CrossServiceContractResolver.resolve_contracts(str(ui_file), str(tmp_path))
    assert any("IssuesCollectionController.java" in c for c in ui_contracts["controllers"])
    assert any("application.yml" in g for g in ui_contracts["gateway_routes"])

    # Test resolve_all_companions integrates cross-service companions
    all_comps = PolyglotCompanionResolver.resolve_all_companions(str(ui_file), workspace_path=str(tmp_path))
    comp_names = [Path(c).name for c in all_comps]
    assert "IssuesCollectionController.java" in comp_names
    assert "application.yml" in comp_names


