"""
Generate Ground-Truth Answer Key from Git History in C:/CC4E repositories.

Scans all microservices and xchange-ui for commit messages containing 6-7 digit ticket IDs,
extracts fix diffs, verifies existing workspace files, and constructs a high-power answer key.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any

WORKSPACE = Path("C:/CC4E")
REPOS = [
    "xchange-ui",
    "se-connector-apis",
    "area-service",
    "issues-service",
    "project-service",
    "ene-load-service",
    "bim-gateway",
    "sagas-service",
]

TICKET_PATTERN = re.compile(r"\b([0-9]{6,7})\b")
IGNORE_EXTS = {".md", ".txt", ".png", ".jpg", ".svg", ".ico", ".gitignore"}
IGNORE_TITLES = {"merge", "bump", "version", "release", "jenkins", "revert"}


def run_git(repo_dir: Path, args: List[str]) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir)] + args,
            capture_output=True,
            text=True,
            errors="ignore",
            check=True,
        )
        return res.stdout
    except Exception:
        return ""


def extract_repo_commits(repo_name: str) -> List[Dict[str, Any]]:
    repo_dir = WORKSPACE / repo_name
    if not (repo_dir / ".git").exists():
        return []

    log_output = run_git(
        repo_dir,
        [
            "log",
            "--all",
            "--no-merges",
            "--format=%H%x09%s",
            "--grep=[0-9]\\{6,7\\}",
            "-n",
            "1000",
        ],
    )

    commits = []
    for line in log_output.strip().split("\n"):
        if not line or "\t" not in line:
            continue
        commit_hash, subject = line.split("\t", 1)
        m = TICKET_PATTERN.search(subject)
        if not m:
            continue
        ticket_id = m.group(1)

        # Clean subject
        cleaned_sub = subject
        for token in [ticket_id, ":", "-", "[", "]", "#"]:
            cleaned_sub = cleaned_sub.replace(token, " ")
        cleaned_title = " ".join(cleaned_sub.split()).strip()

        if len(cleaned_title) < 8:
            continue
        if any(cleaned_title.lower().startswith(bad) for bad in IGNORE_TITLES):
            continue

        # Get changed files
        diff_files_out = run_git(
            repo_dir,
            ["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
        )
        changed_rel = [
            f.strip()
            for f in diff_files_out.strip().split("\n")
            if f.strip()
        ]

        # Map to workspace root
        valid_files = []
        for rel in changed_rel:
            ws_rel = f"{repo_name}/{rel}".replace("\\", "/")
            full_p = WORKSPACE / ws_rel
            if full_p.suffix.lower() in IGNORE_EXTS:
                continue
            if full_p.exists() and full_p.is_file():
                valid_files.append(ws_rel)

        if not valid_files:
            continue

        commits.append(
            {
                "id": ticket_id,
                "commit": commit_hash[:8],
                "repo": repo_name,
                "raw_subject": subject,
                "title": cleaned_title,
                "files": valid_files,
            }
        )

    return commits


def build_answer_key(target_count: int = 100) -> Dict[str, Any]:
    all_commits = []
    for r in REPOS:
        commits = extract_repo_commits(r)
        all_commits.extend(commits)

    # Group by ticket ID
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_commits:
        tid = c["id"]
        grouped.setdefault(tid, []).append(c)

    # Load existing hand-annotated key to preserve anchors
    key_path = Path("tests/benchmarks/cc4e_real_ticket_answer_key.json")
    existing_data = {}
    if key_path.exists():
        existing_data = json.loads(key_path.read_text(encoding="utf-8"))

    existing_ids = {t["id"]: t for t in existing_data.get("tickets", [])}
    excluded_ids = {x["id"] for x in existing_data.get("excluded", [])}

    curated_tickets: List[Dict[str, Any]] = []

    # 1. First add all existing hand-annotated tickets
    for tid, t in existing_ids.items():
        curated_tickets.append(t)

    # 2. Add git-derived tickets
    for tid, items in grouped.items():
        if tid in existing_ids or tid in excluded_ids:
            continue

        # Collect unique files
        all_files = []
        for it in items:
            for f in it["files"]:
                if f not in all_files:
                    all_files.append(f)

        if not all_files or len(all_files) > 10:
            # Skip massive refactors or empty
            continue

        # Pick best title (longest, most descriptive)
        best_title = max((it["title"] for it in items), key=len)
        repo = items[0]["repo"]
        primary_file = all_files[0]
        ext = Path(primary_file).suffix.lower()
        lang = "typescript" if ext in {".ts", ".html", ".scss"} else "java" if ext == ".java" else "json"

        # Determine confidence
        lower_title = best_title.lower()
        is_confirmed = any(w in lower_title for w in ["fix", "bug", "error", "issue", "fail", "null", "not displayed", "wrong"])
        conf_str = "CONFIRMED_DEFECT" if is_confirmed else "PLAUSIBLE"

        # Build expected_changed_files
        expected = []
        for idx, f in enumerate(all_files):
            role = "primary-defect" if idx == 0 else "related-change"
            expected.append({"path": f, "role": role, "rank": idx + 1})

        curated_tickets.append(
            {
                "id": tid,
                "title": f"[{repo}] {best_title}",
                "confidence": conf_str,
                "language": lang,
                "primary": primary_file,
                "evidence": f"Git commit {items[0]['commit']} in {repo}: '{items[0]['raw_subject']}'",
                "expected_changed_files": expected,
                "forbidden_files": [],
                "source": "git_history",
            }
        )

        if len(curated_tickets) >= target_count:
            break

    result = {
        "meta": {
            "version": 2,
            "workspace_root": "C:/CC4E",
            "total_tickets": len(curated_tickets),
            "hand_annotated_count": len(existing_ids),
            "git_derived_count": len(curated_tickets) - len(existing_ids),
            "description": "Expanded ground-truth answer key derived from git commit history across CC4E repos.",
        },
        "tickets": curated_tickets,
        "excluded": existing_data.get("excluded", []),
    }
    return result


if __name__ == "__main__":
    out_dir = Path("tests/benchmarks")
    key = build_answer_key(target_count=300)
    out_file = out_dir / "cc4e_git_curated_answer_key.json"
    out_file.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"Generated {len(key['tickets'])} tickets into {out_file}")
    print(f"  Hand-annotated : {key['meta']['hand_annotated_count']}")
    print(f"  Git-derived    : {key['meta']['git_derived_count']}")
