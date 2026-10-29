from ticket_to_code.agents.subagent_explorer import (
    SubagentExplorer,
    ExplorerRole,
    EvidenceAnchor,
    SubagentReport,
)

def test_subagent_explorer_workers_and_firewall(tmp_path):
    # Setup mock workspace files
    comp_dir = tmp_path / "apps" / "ui" / "document"
    comp_dir.mkdir(parents=True)

    ts_file = comp_dir / "document-register.component.ts"
    ts_file.write_text("export class DocumentRegisterComponent { uploadControl = new FormControl(); }", encoding="utf-8")

    html_file = comp_dir / "document-register.component.html"
    html_file.write_text("<div *ngIf='uploadControl.invalid'>Please select a valid document</div>", encoding="utf-8")

    spec_file = comp_dir / "document-register.component.spec.ts"
    spec_file.write_text("describe('DocumentRegisterComponent', () => { it('renders', () => {}); });", encoding="utf-8")

    explorer = SubagentExplorer(workspace_root=str(tmp_path))

    # Worker 1: UI Mapper
    rep_ui = explorer.run_ui_mapper(
        ticket_description="Upload inline validation error",
        candidate_components=["apps/ui/document/document-register.component.ts"],
    )
    assert rep_ui.worker_role == ExplorerRole.UI_MAPPER
    assert rep_ui.confidence > 0.8
    assert "document-register.component.ts" in rep_ui.recommended_files[0]

    # Worker 2: Validation Analyst
    rep_val = explorer.run_validation_analyst(
        component_ts_path="apps/ui/document/document-register.component.ts",
        component_html_path="apps/ui/document/document-register.component.html",
    )
    assert rep_val.worker_role == ExplorerRole.VALIDATION_ANALYST
    assert len(rep_val.evidence) >= 2
    assert any("uploadControl" in (e.snippet or "") for e in rep_val.evidence)

    # Worker 3: Contract Checker
    rep_contract = explorer.run_contract_checker(api_dto_path=None)
    assert rep_contract.worker_role == ExplorerRole.CONTRACT_CHECKER
    assert len(rep_contract.recommended_files) == 0  # No backend edits recommended

    # Worker 4: Test Scout
    rep_test = explorer.run_test_scout(component_dir="apps/ui/document")
    assert rep_test.worker_role == ExplorerRole.TEST_SCOUT
    assert len(rep_test.recommended_files) == 1
    assert "document-register.component.spec.ts" in rep_test.recommended_files[0]

    # Context Firewall Summary Synthesis
    summary = explorer.synthesize_investigation([rep_ui, rep_val, rep_contract, rep_test])
    assert "## Sub-Agent Exploration Findings" in summary
    assert "[UI_MAPPER]" in summary
    assert "[VALIDATION_ANALYST]" in summary
    assert "[CONTRACT_CHECKER]" in summary
    assert "[TEST_SCOUT]" in summary
    assert "document-register.component.html" in summary
    assert "document-register.component.spec.ts" in summary
    # Assert size is small (context firewall prevents 50k token blowup)
    assert len(summary) < 2000
