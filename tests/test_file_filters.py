"""
Unit tests for centralized file_filters.
"""

from pathlib import Path
from ticket_to_code.utils.file_filters import is_ignored_path, STANDARD_SKIP_DIRS, prune_walk_dirs


def test_standard_skip_dirs_present():
    assert "public" in STANDARD_SKIP_DIRS
    assert "dist" in STANDARD_SKIP_DIRS
    assert "node_modules" in STANDARD_SKIP_DIRS
    assert "wwwroot" in STANDARD_SKIP_DIRS


def test_minified_chunk_files_ignored():
    # Webpack hashed chunk files
    assert is_ignored_path("server/public/105.88261251d83cecc2.js")
    assert is_ignored_path("server/public/62.f9938d5a65f0a2f3.js")
    assert is_ignored_path("server/public/main.8dc01088f9db2756.js")
    assert is_ignored_path("dist/bundle.min.js")
    assert is_ignored_path("src/styles/vendor.min.css")
    assert is_ignored_path("scripts/bundle.js.map")


def test_valid_source_files_not_ignored():
    assert not is_ignored_path("src/app/modules/members/member/member.component.scss")
    assert not is_ignored_path("src/app/modules/members/member/member.component.ts")
    assert not is_ignored_path("src/app/modules/members/member/member.component.html")
    assert not is_ignored_path("backend/src/main/java/com/example/App.java")


def test_prune_walk_dirs():
    dirs = ["node_modules", "src", "public", ".git", "target", "components"]
    prune_walk_dirs(dirs)
    assert dirs == ["src", "components"]
