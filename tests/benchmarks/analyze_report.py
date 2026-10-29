"""
Post-hoc analysis of a cc4e_scope_localization report.

Separates the numbers that answer different questions:
  * by ANSWER WIDTH  — a 13-file ticket is easier to hit than a 1-file ticket,
                        so the aggregate overstates accuracy
  * by LANGUAGE      — the git-derived key is 64% xchange-ui, so the aggregate
                        is dominated by Angular
  * by ROUTE         — precision when the system commits vs recall when it defers

Usage:
    python tests/benchmarks/analyze_report.py --report <report.json> --key <key.json>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()

    rep = load(args.report)
    key = load(args.key)

    width = {e["id"]: len(e.get("expected_changed_files", [])) for e in key["tickets"]}
    lang = {e["id"]: e.get("language", "unknown") for e in key["tickets"]}
    rows = [r for r in rep["results"] if r.get("status") == "OK"]

    print(f"report : {args.report}")
    print(f"key    : {args.key}")
    print(f"graded : {len(rows)} tickets\n")

    def block(title: str, groups: dict) -> None:
        print(f"-- {title} " + "-" * max(0, 58 - len(title)))
        print(f"  {'bucket':<14}{'n':>5}{'top-1':>9}{'top-3':>9}")
        for k in sorted(groups, key=lambda x: (len(x), x)):
            rs = groups[k]
            if not rs:
                continue
            t1 = 100.0 * sum(1 for r in rs if r["first_hit_index"] == 0) / len(rs)
            t3 = 100.0 * sum(1 for r in rs if r["top3_hit"]) / len(rs)
            print(f"  {k:<14}{len(rs):>5}{t1:>8.1f}%{t3:>8.1f}%")
        print()

    by_width = defaultdict(list)
    for r in rows:
        n = width.get(r["id"], 0)
        by_width["1 file" if n == 1 else ("2-3 files" if n <= 3 else "4+ files")].append(r)
    block("BY ANSWER WIDTH (the 1-file bucket is the honest headline)", by_width)

    by_lang = defaultdict(list)
    for r in rows:
        by_lang[lang.get(r["id"], "unknown")].append(r)
    block("BY LANGUAGE", by_lang)

    by_route = defaultdict(list)
    for r in rows:
        by_route[str(r.get("fast_path"))].append(r)
    block("BY ROUTE (precision when committing, recall when deferring)", by_route)

    # Fast-path calibration: committing on a wrong top-1 is the costly error.
    fp = [r for r in rows if r.get("fast_path") == "edit_loop"]
    if fp:
        good = sum(1 for r in fp if r["first_hit_index"] == 0)
        print(f"FAST-PATH CALIBRATION")
        print(f"  committed            : {len(fp)}")
        print(f"  top-1 was correct    : {good}/{len(fp)}")
        print(f"  top-3 contained hit  : {sum(1 for r in fp if r['top3_hit'])}/{len(fp)}")
        print(f"  top-1 WRONG (costly) : {sum(1 for r in fp if r['first_hit_index'] != 0)}")
    else:
        print("FAST-PATH CALIBRATION: never fired — cannot measure precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
