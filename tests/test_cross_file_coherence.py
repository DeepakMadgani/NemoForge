"""Unit tests for Cross-File Coherence and Immediate In-Memory Session Propagation.

Verifies:
  1. Dual companion context injection: BOTH AST symbol contract AND companion source are injected.
  2. In-memory session propagation: newly generated files are immediately available in memory
     for downstream tasks, resolving companions without needing prior disk writes.
  3. Zero hardcoding: completely dynamic symbol extraction and binding validation across components.
"""

from pathlib import Path
import pytest

from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
from ticket_to_code.intelligence.contracts.component_contract import (
    ComponentContract,
    TemplateContractValidator,
)


def test_in_memory_session_propagation_and_companion_resolution():
    """Verify that in-memory changes in session_files are prioritized and resolved accurately."""
    # Simulate in-memory generation of a controller file (not written to disk)
    new_ts_content = """
import { Component } from '@angular/core';

@Component({
  selector: 'app-user-profile',
  templateUrl: './user-profile.component.html'
})
export class UserProfileComponent {
  userStatus: string = 'ACTIVE';
  isAccountVerified: boolean = true;
  accountTier: number = 1;

  refreshStatus(userId: string): void {
    console.log(userId);
  }
}
"""
    session_files = {
        "src/app/features/profile/user-profile.component.ts": new_ts_content
    }

    # Resolve companion for the HTML template purely from session_files
    template_path = "src/app/features/profile/user-profile.component.html"
    companion = PolyglotCompanionResolver.resolve_companion(
        template_path,
        session_files=session_files,
        workspace_path="/mock/workspace"
    )

    assert companion is not None
    comp_path, comp_text = companion
    assert comp_path == "src/app/features/profile/user-profile.component.ts"
    assert comp_text == new_ts_content

    # Extract contract dynamically (zero hardcoding)
    contract = ComponentContract.extract_from_ts(comp_text, file_path=comp_path)
    assert contract.class_name == "UserProfileComponent"
    assert "userStatus" in contract.properties
    assert "isAccountVerified" in contract.properties
    assert "accountTier" in contract.properties
    assert "refreshStatus" in contract.methods

    # Render prompt block - verify dynamic invariants
    prompt = contract.render_prompt_block()
    assert "CONTROLLER CONTRACT FOR TEMPLATE: UserProfileComponent" in prompt
    assert "userStatus" in prompt
    assert "isAccountVerified" in prompt
    assert "Do NOT invent phantom sub-properties" in prompt

    # Verify template contract validator against the in-memory contract
    valid_html = """
    <div *ngIf="isAccountVerified">
      <span>{{ userStatus }}</span>
      <button (click)="refreshStatus('123')">Refresh</button>
    </div>
    """
    violations = TemplateContractValidator.validate(valid_html, contract)
    assert len(violations) == 0

    # Verify template contract validator catches hallucinated property
    invalid_html = """
    <div *ngIf="userIsSuperAdmin">
      <span>{{ userStatus }}</span>
    </div>
    """
    violations = TemplateContractValidator.validate(invalid_html, contract)
    assert len(violations) == 1
    assert "userIsSuperAdmin" in violations[0]
