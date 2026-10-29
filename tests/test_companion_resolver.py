import pytest
from ticket_to_code.utils.companion_resolver import (
    normalize_path,
    resolve_canonical_path,
    resolve_candidate_paths,
    get_companion_files,
)


def test_normalize_path():
    assert normalize_path("src\\app\\foo.ts") == "src/app/foo.ts"
    assert normalize_path("/xchange-ui/src/app/foo.ts/") == "xchange-ui/src/app/foo.ts"


def test_resolve_canonical_path_exact():
    candidates = [
        "xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        "project-service/src/main/java/com/cloudforge/ContractMemberService.java"
    ]
    diag = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
    assert resolve_canonical_path(diag, candidates) == diag


def test_resolve_canonical_path_suffix():
    candidates = [
        "xchange-ui/src/app/modules/members/add-members/add-members.component.html",
        "xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
    ]
    # Compiler gives path without workspace prefix
    diag = "src/app/modules/members/add-members/add-members.component.html"
    assert resolve_canonical_path(diag, candidates) == "xchange-ui/src/app/modules/members/add-members/add-members.component.html"


def test_resolve_canonical_path_windows_backslashes():
    candidates = [
        "xchange-ui/src/app/modules/members/add-members/add-members.component.html",
    ]
    diag = "src\\app\\modules\\members\\add-members\\add-members.component.html"
    assert resolve_canonical_path(diag, candidates) == "xchange-ui/src/app/modules/members/add-members/add-members.component.html"


def test_get_companion_files_angular_html():
    companions = get_companion_files("xchange-ui/src/app/modules/members/add-members/add-members.component.html")
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.ts" in companions
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.scss" in companions


def test_get_companion_files_from_ts_content():
    content = """
    @Component({
      selector: 'se-add-members',
      templateUrl: './add-members.component.html',
      styleUrls: ['./add-members.component.scss']
    })
    export class AddMembersComponent {}
    """
    known_files = [
        "xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        "xchange-ui/src/app/modules/members/add-members/add-members.component.html",
        "xchange-ui/src/app/modules/members/add-members/add-members.component.scss",
    ]
    companions = get_companion_files(
        "xchange-ui/src/app/modules/members/add-members/add-members.component.ts",
        known_files=known_files,
        file_content=content
    )
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.html" in companions
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.scss" in companions


def test_resolve_candidate_paths_returns_ranked_subset_when_ambiguous():
    candidates = [
        "xchange-ui/src/app/modules/members/add-members/add-members.component.html",
        "legacy-ui/src/app/modules/members/add-members/add-members.component.html",
        "project-service/src/main/java/com/cloudforge/projectservice/service/ContractMemberService.java",
    ]
    diag = "src/app/modules/members/add-members/add-members.component.html"
    matches = resolve_candidate_paths(diag, candidates, max_results=2)

    assert len(matches) == 2
    assert all(m.endswith("add-members.component.html") for m in matches)


def test_resolve_candidate_paths_prefers_exact_match():
    candidates = [
        "xchange-ui/src/app/modules/members/add-members/add-members.component.html",
        "src/app/modules/members/add-members/add-members.component.html",
    ]
    diag = "src/app/modules/members/add-members/add-members.component.html"
    matches = resolve_candidate_paths(diag, candidates)

    assert matches == ["src/app/modules/members/add-members/add-members.component.html"]


def test_resolve_candidate_paths_keeps_cross_module_top_ties():
    candidates = [
        "xchange-ui/src/app/modules/members/add-members/add-members.component.html",
        "legacy-ui/src/app/modules/members/add-members/add-members.component.html",
        "ui-next/src/app/modules/members/add-members/add-members.component.html",
        "project-service/src/main/java/com/cloudforge/projectservice/service/ContractMemberService.java",
    ]
    diag = "src/app/modules/members/add-members/add-members.component.html"
    matches = resolve_candidate_paths(diag, candidates, max_results=2)

    assert len(matches) == 3
    assert "xchange-ui/src/app/modules/members/add-members/add-members.component.html" in matches
    assert "legacy-ui/src/app/modules/members/add-members/add-members.component.html" in matches
    assert "ui-next/src/app/modules/members/add-members/add-members.component.html" in matches


def test_resolve_candidate_paths_caps_same_module_ties_by_max_results():
    candidates = [
        "xchange-ui/src/app/a/b/c/file.html",
        "xchange-ui/src/app/a/b/c/file.html.bak",  # non-matching tail, should be excluded by scorer
        "xchange-ui/src/app/a/b/c/file.html",      # duplicate candidate path scenario
        "xchange-ui/src/app/a/b/c/file.html",
    ]
    diag = "src/app/a/b/c/file.html"
    matches = resolve_candidate_paths(diag, candidates, max_results=1)

    assert len(matches) == 1
    assert matches[0] == "xchange-ui/src/app/a/b/c/file.html"
