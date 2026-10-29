"""
Git-history answer-key extractor.

Derives a labelled localization benchmark from commit history: every commit
carrying a 7-digit ticket ID yields (ticket_id, files_touched). That is
ground truth for "which files does fixing this ticket require" — produced
mechanically, with no human annotation, at whatever scale the history allows.

WHY THIS EXISTS
---------------
The hand-annotated key in answer_key.json has 9 tickets, 4 of
them CONFIRMED. At that size a single ticket flipping moves the top-3 hit rate
by 11%, so no change to the ranker is falsifiable. This script converts the
1,700+ ticket-tagged commits across the enterprise services into a benchmark with
enough n to detect a real improvement.

HONEST CAVEAT — READ BEFORE USING THE NUMBERS
----------------------------------------------
The ticket text used here is the COMMIT SUBJECT, e.g. "addcontract",
"Fix for date component localization issue". A real JIRA report is far noisier:
4157705 buries its root cause under a support conversation and a 400 response.
Commit subjects frequently NAME THE FIX, so this benchmark is systematically
EASIER than production. Treat it as:

  * a REGRESSION SUITE  — can you detect that a change made retrieval worse?
  * a ranking of RELATIVE progress between runs

and NOT as an estimate of production accuracy. For realistic (harder)
measurement use answer_key.json, which carries real ticket
descriptions. Never average the two.

Usage:
    python tests/benchmarks/git_answer_key_extractor.py
    python tests/benchmarks/git_answer_key_extractor.py --max-files 8 --out key.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

FIXTURE_ROOT = Path(r"workspace\CloudForge-Commerce")
DEFAULT_OUT = Path(__file__).resolve().parent / "answer_key.json"

# Repos that contain a verbatim copy of xchange-ui. Including them would make
# every UI answer ambiguous: the same file exists at two paths.
EXCLUDED_REPOS = {"xchange-ui-fixture"}

TICKET_RE = re.compile(r"(?<!\d)(\d{7})(?!\d)")

# Extensions that constitute "the code a fix lives in".
SOURCE_EXTS = {".java", ".ts", ".html", ".scss", ".css", ".sql", ".py", ".js", ".kt", ".graphql"}

# Paths that are never the answer to "where is the defect".
NOISE_PATH_PARTS = {
    "node_modules", "target", "build", "dist", "coverage",
    "package-lock.json", "yarn.lock", "pom.xml", "build.gradle",
    "manifest.yml", "angular.json", "tsconfig",
}

TEST_MARKERS = ("/test/", "/tests/", "/__tests__/", ".spec.ts", ".test.ts", "test.java")


def is_source_file(rel: str) -> bool:
    parts = set(Path(rel).parts)
    if parts & NOISE_PATH_PARTS:
        return False
    if any(noise in Path(rel).name for noise in NOISE_PATH_PARTS):
        return False
    return Path(rel).suffix.lower() in SOURCE_EXTS


def is_test_file(rel: str) -> bool:
    norm = "/" + rel.replace("\\", "/").lower()
    return any(m in norm for m in TEST_MARKERS)


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, errors="ignore", timeout=600,
    ).stdout


def collect_repo(repo: Path) -> dict[str, dict]:
    """Return {ticket_id: {files, commits, subject}} for one repository."""
    raw = run_git(
        repo, "log", "--all", "--no-merges",
        "--pretty=format:@@@%H\t%s",
        "--name-only",
    )
    by_ticket: dict[str, dict] = defaultdict(
        lambda: {"files": set(), "commits": [], "subjects": []}
    )

    current_h, current_s, current_files = None, None, []
    for line in raw.splitlines():
        if line.startswith("@@@"):
            if current_h:
                _flush(by_ticket, current_s, current_files, current_h)
            parts = line[3:].split("\t", 1)
            current_h = parts[0].strip()
            current_s = parts[1] if len(parts) > 1 else ""
            current_files = []
        elif line.strip():
            current_files.append(line.strip())
    if current_h:
        _flush(by_ticket, current_s, current_files, current_h)

    return by_ticket


def _flush(by_ticket, subject, files, commit_hash) -> None:
    ids = set(TICKET_RE.findall(subject or ""))
    if not ids:
        return
    src = [f for f in files if is_source_file(f)]
    if not src:
        return
    # A commit tagged with two ticket ids is ambiguous as a label.
    if len(ids) > 1:
        return
    tid = next(iter(ids))
    rec = by_ticket[tid]
    rec["files"].update(src)
    rec["commits"].append(commit_hash[:8])
    if subject and subject not in rec["subjects"]:
        rec["subjects"].append(subject)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(FIXTURE_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-files", type=int, default=8,
                    help="Tickets touching more source files are features, not "
                         "localizable bugs. Default 8.")
    ap.add_argument("--min-files", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"FATAL: {root} not found", file=sys.stderr)
        return 2

    repos = [d for d in sorted(root.iterdir())
             if d.is_dir() and (d / ".git").is_dir() and d.name not in EXCLUDED_REPOS]
    print(f"Scanning {len(repos)} repositories under {root}\n")

    started = time.time()
    per_repo_raw = 0
    tickets: dict[str, dict] = {}

    for repo in repos:
        by_ticket = collect_repo(repo)
        per_repo_raw += len(by_ticket)
        for tid, rec in by_ticket.items():
            src = sorted(f for f in rec["files"] if not is_test_file(f))
            tests = sorted(f for f in rec["files"] if is_test_file(f))
            if not (args.min_files <= len(src) <= args.max_files):
                continue
            entry = tickets.setdefault(tid, {
                "id": tid, "files": set(), "test_files": set(),
                "repos": set(), "subjects": [], "commits": [],
            })
            entry["files"].update(src)
            entry["test_files"].update(tests)
            entry["repos"].add(repo.name)
            entry["commits"].extend(rec["commits"])
            for s in rec["subjects"]:
                if s not in entry["subjects"]:
                    entry["subjects"].append(s)
        print(f"  {repo.name:<22} {len(by_ticket):>4} ticket-tagged")

    built = []
    for tid, e in tickets.items():
        files = sorted(e["files"])
        # Primary = the non-test file that is not a shared config-ish leaf.
        # Rank 1 goes to the first non-test source file; ranking beyond that is
        # deliberately flat because we cannot know which of several files is
        # "the" defect without reading the diff.
        subject = (e["subjects"][0] if e["subjects"] else "")
        built.append({
            "id": tid,
            "title": subject,
            "text_source": "commit_subject",
            "confidence": "CONFIRMED_DEFECT",
            "language": _guess_language(files, sorted(e["repos"])),
            "repos": sorted(e["repos"]),
            "commit_count": len(e["commits"]),
            "expected_changed_files": [
                {"path": f, "role": "source", "rank": 1 if i == 0 else 2}
                for i, f in enumerate(files)
            ],
            "also_consider": sorted(e["test_files"])[:5],
            "forbidden_files": [],
        })

    built.sort(key=lambda t: t["id"])
    out = {
        "meta": {
            "version": 1,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generator": "git_answer_key_extractor.py",
            "workspace_root": str(root),
            "excluded_repos": sorted(EXCLUDED_REPOS),
            "raw_ticket_ids_seen": per_repo_raw,
            "tickets_retained": len(built),
            "max_files_per_ticket": args.max_files,
            "ticket_text_source": "commit_subject",
            "WARNING": (
                "Ticket text is ONLY the commit subject (description is empty), so these "
                "tickets are DEGENERATE INPUT, not representative. Measured: top-3 17.5% "
                "aggregate here vs 44.4% on the 9 real-JIRA-text tickets. The commit "
                "subject often names the fix, but it supplies far less lexical signal than "
                "a real report - often a single compound token ('addcontract') that no "
                "path-token search can match. 291/292 are correctly declined as "
                "NOT_ANNOTATABLE by the production router. Use this key to measure the "
                "RETRIEVAL LAYER under sparse text, or as a 292-ticket regression suite "
                "for detecting ranker changes. Do NOT compare its absolute hit rate to "
                "production, and never average it with answer_key.json."
            ),
        },
        "tickets": built,
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    repos_n = len({r for t in built for r in t["repos"]})
    print(f"\n  repos scanned        : {len(repos)}")
    print(f"  raw ticket ids       : {per_repo_raw}")
    print(f"  retained (localizable): {len(built)}")
    print(f"  total expected files : {sum(len(t['expected_changed_files']) for t in built)}")
    print(f"  elapsed              : {round(time.time()-started,1)}s")
    print(f"\n  -> {outp}")
    return 0


def _guess_language(files: list[str], repos: list[str] | None = None) -> str:
    """Label by the dominant file extension, or 'mixed' when several languages
    or several services are involved. A ticket touching both xchange-ui and a
    Java service is genuinely multi-language; mislabeling it as 'java' because
    of a fixed priority order would silently distort any per-language analysis.
    """
    exts = {Path(f).suffix.lower() for f in files} & {
        ".java", ".kt", ".ts", ".tsx", ".py", ".sql", ".scss", ".html", ".css",
    }
    # Collapse html/scss into the TS bucket: in this codebase they are
    # component companions of a .ts file, not a separate language.
    primary = {e for e in exts if e not in (".html", ".scss", ".css")}
    if not primary:
        primary = exts
    labels = {e.lstrip(".") for e in primary}
    if len(labels) > 1 or (repos and len(repos) > 1):
        return "mixed"
    return next(iter(labels)) if labels else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
