"""
Ticket to Code: 50-Ticket Golden Set Benchmark Harness
======================================================
Week 0 Baseline & Continuous Regression Evaluation Engine

Key architectural principles:
1. Harness-Verifies-Independently: Tier 0 assertions evaluate pipeline results
   using decoupled AST/regex/exit-code checks, without invoking the pipeline's internal tools.
2. Alternative-Solution Tolerance: Tier 0 checks allow any valid equivalent pattern (any-of).
3. Pinned Inquiry Judge: Versioned prompt and deterministic citation checks for info tickets.
4. Generous Baseline Caps: 500k tokens / 900s per ticket in baseline mode to avoid censoring telemetry.
5. Between-Ticket Clean Reset: Hard git reset and untracked file pruning between runs.
6. Environment Prerequisites: Missing toolchains yield SKIPPED_PREREQ rather than pipeline failures.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Configure UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("golden_50_harness")

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Tier0AssertionResult:
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TicketSpec:
    number: int
    ticket_id: str
    category: str       # UI, FE, BE, Mix, Cfg, Info, Dbg, Neg
    difficulty: str     # Easy, Med, Hard
    title: str
    description: str
    acceptance_criteria: List[str]
    required_prereqs: List[str]  # e.g. ["git", "node"], ["java", "mvn"]
    evaluator: Callable[[Path, Dict[str, Any]], Tier0AssertionResult]
    expected_files: List[str] = field(default_factory=list)


@dataclass
class TicketExecutionResult:
    ticket_id: str
    number: int
    category: str
    difficulty: str
    title: str
    status: str          # RESOLVED, FAILED_ASSERTION, PIPELINE_ERROR, TIMEOUT, SKIPPED_PREREQ
    wall_clock_s: float
    tokens_in: int
    tokens_out: int
    total_tokens: int
    tier0_passed: bool
    tier0_reason: str
    changed_files: List[str] = field(default_factory=list)
    phase_tokens: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class BenchmarkReport:
    timestamp: str
    mode: str
    workspace_path: str
    total_tickets: int
    evaluated: int
    resolved: int
    skipped_prereq: int
    failed: int
    resolution_rate_pct: float
    total_tokens: int
    avg_tokens_per_ticket: float
    avg_tokens_per_resolved_ticket: float
    avg_wall_clock_s: float
    category_summary: Dict[str, Dict[str, Any]]
    tickets: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def check_service_online(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """Fast probe socket connection to verify if a background service is reachable."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# ============================================================================
# TOOLCHAIN & PREREQUISITES VERIFICATION
# ============================================================================

def detect_toolchain() -> Dict[str, bool]:
    """Detect available executables on host machine."""
    tools = {
        "git": shutil.which("git") is not None,
        "node": shutil.which("node") is not None,
        "npm": shutil.which("npm") is not None,
        "java": shutil.which("java") is not None,
        "mvn": shutil.which("mvn") is not None,
        "docker": shutil.which("docker") is not None,
        "python": shutil.which("python") is not None or shutil.which("python3") is not None,
    }
    return tools


def check_ticket_prereqs(spec: TicketSpec, available_tools: Dict[str, bool]) -> Tuple[bool, List[str]]:
    missing = [req for req in spec.required_prereqs if not available_tools.get(req, False)]
    return (len(missing) == 0, missing)


# ============================================================================
# WORKSPACE RESET PROCEDURES & DEDICATED FIXTURE MANAGEMENT
# ============================================================================

def is_fixture_workspace(workspace: Path) -> bool:
    """Check if workspace is a dedicated fixture or disposable benchmark copy."""
    norm = str(workspace.resolve()).replace("\\", "/").lower()
    name = workspace.resolve().name.lower()
    if name.endswith(("-fixture", "_fixture", ".fixture")) or name == "fixtures":
        return True
    if "/fixtures/" in norm or "/.aviator/fixtures/" in norm or "\\.aviator\\fixtures\\" in norm:
        return True
    if (workspace / ".aviator_fixture").exists():
        return True
    return False


def is_ignored_artifact_file(path_str: str) -> bool:
    """Check if a changed file is an internal Aviator trace/metric/log artifact."""
    norm = path_str.replace("\\", "/").lower().strip('"').strip("'")
    if norm.startswith((".aviator/", "trace_artifacts/", ".system_generated/", ".pytest_cache/")):
        return True
    if "/.aviator/" in norm or "/trace_artifacts/" in norm or "/.system_generated/" in norm:
        return True
    if norm.endswith((".log",)):
        return True
    return False


def reset_fixture_workspace(workspace: Path, allow_live_reset: bool = False) -> None:
    """
    Hard-reset and prune untracked files on the fixture workspace between runs.
    Guards against accidental data loss on live repositories with uncommitted changes.
    Exempts dedicated fixture copies (which are disposable by definition).
    """
    if not (workspace / ".git").exists():
        logger.warning(f"Workspace {workspace} is not a git repo; skipping git reset.")
        return

    is_fixture = is_fixture_workspace(workspace)
    # Safety Guard: If it is NOT a fixture and contains uncommitted work, refuse without explicit allow_live_reset
    if not is_fixture and not allow_live_reset:
        changed = get_git_status_changed_files(workspace, filter_artifacts=False)
        if changed:
            msg = (
                f"SAFETY ABORT: Refusing destructive reset on non-fixture workspace '{workspace}'.\n"
                f"Found {len(changed)} uncommitted change(s):\n"
                f"  {changed[:5]}\n"
                f"The harness will NOT run 'git reset --hard / clean -fd' on a dirty live tree without --allow-live-reset.\n"
                f"Please run against a dedicated fixture clone (default: workspace/CloudForge-Commerce) "
                f"or pass --allow-live-reset if you are intentionally resetting this repo."
            )
            logger.error(msg)
            raise RuntimeError(msg)

    logger.info(f"Resetting workspace: {workspace} (fixture={is_fixture})")
    try:
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=True
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as exc:
        logger.error(f"Failed to reset workspace {workspace}: {exc.stderr}")
        raise


def ensure_fixture_workspace(source_repo: Path, fixture_path: Path) -> Path:
    """Ensure a dedicated fixture clone exists. Clones from source if missing and junctions node_modules."""
    if not fixture_path.exists():
        logger.info(f"Creating dedicated fixture clone: {source_repo} -> {fixture_path}")
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", str(source_repo), str(fixture_path)],
            check=True, capture_output=True, text=True
        )
        # Create Windows directory junction for node_modules to avoid slow reinstall
        source_nm = source_repo / "node_modules"
        target_nm = fixture_path / "node_modules"
        if source_nm.exists() and not target_nm.exists():
            try:
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(target_nm), str(source_nm)],
                    check=False, capture_output=True
                )
                logger.info(f"Junctioned node_modules: {target_nm} -> {source_nm}")
            except Exception as e:
                logger.warning(f"Could not junction node_modules: {e}")
    return fixture_path


def get_git_status_changed_files(workspace: Path, filter_artifacts: bool = True) -> List[str]:
    """Retrieve list of modified/added files from git status, optionally excluding trace artifacts."""
    if not (workspace / ".git").exists():
        return []
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=True
        )
        changed = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                file_rel = parts[1].strip().strip('"').strip("'")
                if filter_artifacts and is_ignored_artifact_file(file_rel):
                    continue
                changed.append(file_rel)
        return changed
    except Exception:
        return []


# ============================================================================
# HELPER EVALUATOR UTILITIES (WITH ANY-OF TOLERANCE)
# ============================================================================

def _read_file_text(workspace: Path, rel_path: str) -> Optional[str]:
    target = workspace / rel_path
    if not target.exists():
        # Search recursively if path is relative stem
        fname = Path(rel_path).name
        matches = list(workspace.glob(f"**/{fname}"))
        if not matches:
            # Handle singular/plural discrepancies (e.g., member vs members)
            stem_core = fname.split(".")[0].rstrip("s")
            suffix = "".join(Path(rel_path).suffixes)
            matches = list(workspace.glob(f"**/*{stem_core}*{suffix}"))
        if matches:
            target = matches[0]
        else:
            return None
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _check_any_pattern(text: str, patterns: List[str | re.Pattern]) -> bool:
    for p in patterns:
        if isinstance(p, str):
            if p in text:
                return True
        else:
            if p.search(text):
                return True
    return False


# NOTE: inquiry tickets are scored by a deterministic substring rubric; there is no LLM judge.

def evaluate_inquiry_rubric(
    answer_text: str,
    changed_files: List[str],
    expected_citations: List[str],
    key_symbols: List[str]
) -> Tier0AssertionResult:
    # Invariant 1: Inquiry tickets MUST NOT modify any files
    if len(changed_files) > 0:
        return Tier0AssertionResult(
            passed=False,
            reason=f"Inquiry ticket modified {len(changed_files)} files: {changed_files}. Must be 0.",
            details={"changed_files": changed_files}
        )

    if not answer_text or len(answer_text.strip()) < 20:
        return Tier0AssertionResult(
            passed=False,
            reason="Inquiry answer is empty or trivially short.",
            details={"answer_length": len(answer_text or "")}
        )

    # Citation & symbol presence check
    citations_found = [c for c in expected_citations if c.lower() in answer_text.lower()]
    symbols_found = [s for s in key_symbols if s.lower() in answer_text.lower()]

    citation_ratio = len(citations_found) / max(1, len(expected_citations))
    symbol_ratio = len(symbols_found) / max(1, len(key_symbols))

    passed = (citation_ratio >= 0.5 and symbol_ratio >= 0.5) or (len(citations_found) >= 1 and len(symbols_found) >= 1)
    return Tier0AssertionResult(
        passed=passed,
        reason=f"Found citations: {citations_found}/{expected_citations}, symbols: {symbols_found}/{key_symbols}",
        details={
            "citations_found": citations_found,
            "symbols_found": symbols_found,
            "citation_ratio": citation_ratio,
            "symbol_ratio": symbol_ratio
        }
    )


# ============================================================================
# 50-TICKET SPECIFICATIONS & INDEPENDENT TIER 0 EVALUATORS
# ============================================================================

def make_golden_50_specs() -> List[TicketSpec]:
    specs: List[TicketSpec] = []

    # ------------------------------------------------------------------------
    # Category 1: UI / Styling / Viewports (Tickets 1–7)
    # ------------------------------------------------------------------------
    specs.append(TicketSpec(
        number=1,
        ticket_id="GOLDEN-01",
        category="UI",
        difficulty="Easy",
        title="Button hover transition & active state on member table",
        description="Add smooth CSS transition on member table action buttons for :hover and :active states.",
        acceptance_criteria=["Hover state has transition duration >= 150ms", "Active state shows elevation or transform change"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=any(
                    _check_any_pattern(
                        _read_file_text(ws, p) or "",
                        [
                            "transition",
                            re.compile(r"&:hover\s*\{[^}]*transition", re.DOTALL),
                            re.compile(r":hover\s*\{[^}]*(?:background|transform|opacity)", re.DOTALL),
                            re.compile(r":active\s*\{[^}]*transform", re.DOTALL),
                        ]
                    )
                    for p in [
                        "src/app/modules/members/member/member.component.scss",
                        "src/jato/members/jmembers.component.scss",
                        "src/app/organization/members/members.component.scss",
                        "member.component.scss",
                        "members.component.scss",
                    ]
                ),
                reason="SCSS AST contains button hover/active transition rules",
            )
        )
    ))

    specs.append(TicketSpec(
        number=2,
        ticket_id="GOLDEN-02",
        category="UI",
        difficulty="Easy",
        title="Navbar mobile viewport overflow at 375px width",
        description="Navbar items wrap or hide behind a hamburger toggle when viewport is <= 375px.",
        acceptance_criteria=["No horizontal scrollbar at 375px viewport", "Media query handles <= 375px or 576px"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/core/components/navbar/navbar.component.scss") or
                    _read_file_text(ws, "navbar.component.scss") or "",
                    [
                        re.compile(r"@media\s*\([^)]*max-width:\s*(?:375|480|576|768)px\)", re.IGNORECASE),
                        "overflow-x: hidden",
                        "flex-wrap: wrap",
                        re.compile(r"@include\s+media-breakpoint-down", re.IGNORECASE)
                    ]
                ),
                reason="Navbar SCSS contains responsive media query or wrap rule for mobile viewports",
            )
        )
    ))

    specs.append(TicketSpec(
        number=3,
        ticket_id="GOLDEN-03",
        category="UI",
        difficulty="Med",
        title="Table header contrast ratio in high-contrast dark theme",
        description="Ensure table header text color meets WCAG AA contrast against dark background.",
        acceptance_criteria=["Table header text color uses theme variable or high-contrast hex code"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/styles/themes/dark-theme.scss") or
                    _read_file_text(ws, "src/app/shared/table/table.component.scss") or "",
                    [
                        "var(--ot-theme-table-header-text)",
                        "var(--ot-color-high-contrast)",
                        re.compile(r"th\s*\{[^}]*color:\s*(?:#fff|#f\w+|var\([^)]*\))", re.DOTALL),
                        "contrast("
                    ]
                ),
                reason="Table header contrast rule or theme variable applied",
            )
        )
    ))

    specs.append(TicketSpec(
        number=4,
        ticket_id="GOLDEN-04",
        category="UI",
        difficulty="Med",
        title="Modal backdrop z-index clipping under sticky header",
        description="Modal overlay backdrop is clipped under sticky header; fix z-index stacking context.",
        acceptance_criteria=["Modal z-index elevated above header z-index (e.g. >= 1050)"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/shared/modal/modal.component.scss") or
                    _read_file_text(ws, "src/styles/components/_modal.scss") or "",
                    [
                        re.compile(r"z-index:\s*(?:1050|1060|2000|9999|\$zindex-modal)", re.IGNORECASE),
                        "var(--ot-z-index-modal)"
                    ]
                ),
                reason="Modal backdrop z-index adjusted to clear sticky header",
            )
        )
    ))

    specs.append(TicketSpec(
        number=5,
        ticket_id="GOLDEN-05",
        category="UI",
        difficulty="Med",
        title="Missing ellipsis truncation on long deliverable filenames",
        description="Long filenames in deliverable cards push metadata out of view; add text ellipsis truncation.",
        acceptance_criteria=["Filename container applies text-overflow: ellipsis, overflow: hidden, white-space: nowrap"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/projects/deliverables/deliverable-item/deliverable-item.component.scss") or
                    _read_file_text(ws, "deliverable-item.component.scss") or "",
                    [
                        re.compile(r"text-overflow:\s*ellipsis", re.IGNORECASE),
                        "ot-text-truncate",
                        re.compile(r"@include\s+text-truncate", re.IGNORECASE)
                    ]
                ),
                reason="Text ellipsis truncation rule present in deliverable filename style",
            )
        )
    ))

    specs.append(TicketSpec(
        number=6,
        ticket_id="GOLDEN-06",
        category="UI",
        difficulty="Hard",
        title="SVG icon misaligned in dynamic form dropdown options",
        description="SVG icons inside dynamic form select dropdown options are vertically offset by 4px.",
        acceptance_criteria=["Dropdown option template or CSS vertically centers icon with vertical-align or flexbox"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/shared/form/dropdown/dropdown.component.scss") or
                    _read_file_text(ws, "src/shared/form/dynamic-form.component.scss") or "",
                    [
                        re.compile(r"vertical-align:\s*(?:middle|-2px|-3px|-4px)", re.IGNORECASE),
                        re.compile(r"align-items:\s*center", re.IGNORECASE),
                        "display: inline-flex"
                    ]
                ),
                reason="Dropdown option icon vertical alignment rule present",
            )
        )
    ))

    specs.append(TicketSpec(
        number=7,
        ticket_id="GOLDEN-07",
        category="UI",
        difficulty="Hard",
        title="Grid layout collapse when table displays > 10 columns",
        description="Table grid collapses into unreadable squished columns when more than 10 columns are rendered.",
        acceptance_criteria=["Minimum column width enforced (min-width >= 100px) or horizontal scroll enabled"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/shared/table/table.component.scss") or
                    _read_file_text(ws, "table.component.scss") or "",
                    [
                        re.compile(r"min-width:\s*(?:100|120|140|150)px", re.IGNORECASE),
                        "overflow-x: auto",
                        re.compile(r"grid-template-columns:\s*repeat\([^)]*minmax", re.IGNORECASE)
                    ]
                ),
                reason="Table column minimum width or overflow container rule present",
            )
        )
    ))

    # ------------------------------------------------------------------------
    # Category 2: Frontend Angular / Reactive Forms / RxJS (Tickets 8–15)
    # ------------------------------------------------------------------------
    specs.append(TicketSpec(
        number=8,
        ticket_id="GOLDEN-08",
        category="FE",
        difficulty="Easy",
        title="Disable 'Add Member' button when email control is invalid",
        description="Disable submit button on add-members form when email control or form is invalid.",
        acceptance_criteria=["Submit button is disabled when email is invalid or form is invalid"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/organization/members/add-members/add-members.component.html") or
                    _read_file_text(ws, "add-members.component.html") or "",
                    [
                        re.compile(r'\[disabled\]\s*=\s*["\'](?:form\.invalid|!form\.valid|addMemberForm\.invalid|emailControl\.invalid|isFormInvalid)', re.IGNORECASE),
                        re.compile(r'disabled\s*=\s*["\']{{[^}]*invalid[^}]*}}["\']', re.IGNORECASE)
                    ]
                ),
                reason="Add Member button contains [disabled] condition tied to form/email validity",
            )
        )
    ))

    specs.append(TicketSpec(
        number=9,
        ticket_id="GOLDEN-09",
        category="FE",
        difficulty="Easy",
        title="Add French & German translations for deletion dialog",
        description="Add translation keys for deletion confirmation dialog into fr.json and de.json.",
        acceptance_criteria=["de.json and fr.json both contain deletion confirmation keys"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=(
                    _check_any_pattern(_read_file_text(ws, "src/assets/i18n/de.json") or "", ["delete", "lÃ¶schen", "loeschen", "dialog", "confirm"]) and
                    _check_any_pattern(_read_file_text(ws, "src/assets/i18n/fr.json") or "", ["delete", "supprimer", "dialog", "confirmer"])
                ),
                reason="de.json and fr.json contain deletion dialog translation keys",
            )
        )
    ))

    specs.append(TicketSpec(
        number=10,
        ticket_id="GOLDEN-10",
        category="FE",
        difficulty="Med",
        title="Upload Document register - Inline error for invalid file types",
        description="Show inline validation error when uploading unsupported file types in dynamic form template.",
        acceptance_criteria=["Dynamic form template or validator marks invalid file type and displays error message"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/assets/data/uploadDocumentRegister.json") or
                    _read_file_text(ws, "src/shared/document-file-browser/document-file-browser/document-file-browser.component.ts") or "",
                    [
                        "fileTypeValidator",
                        "INVALID_FILE_TYPE",
                        re.compile(r"allowedExtensions", re.IGNORECASE),
                        "file_type_error",
                        re.compile(r"invalidFileType", re.IGNORECASE)
                    ]
                ),
                reason="Dynamic form or file browser validator checks file type extensions",
            )
        )
    ))

    specs.append(TicketSpec(
        number=11,
        ticket_id="GOLDEN-11",
        category="FE",
        difficulty="Med",
        title="Member dropdown options not updating on Org change",
        description="When organization dropdown changes, member dropdown does not refresh. Link with switchMap.",
        acceptance_criteria=["organization change triggers member service reload via switchMap or subscribe"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/organization/members/members.component.ts") or
                    _read_file_text(ws, "members.component.ts") or "",
                    [
                        re.compile(r"switchMap\([^)]*getMembers", re.IGNORECASE),
                        re.compile(r"org(?:Id)?\.valueChanges", re.IGNORECASE),
                        re.compile(r"onOrgChange\([^)]*\)\s*\{[^}]*getMembers", re.DOTALL)
                    ]
                ),
                reason="Organization change reactively triggers members refresh via switchMap/event",
            )
        )
    ))

    specs.append(TicketSpec(
        number=12,
        ticket_id="GOLDEN-12",
        category="FE",
        difficulty="Med",
        title="Async validator for project code causing memory leak",
        description="Async validator subscription is never cancelled on rapid typing. Add cancellation/debounce.",
        acceptance_criteria=["Validator adds debounceTime or switchMap or takeUntil to cancel in-flight check"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/projects/validators/project-code.validator.ts") or
                    _read_file_text(ws, "project-code.validator.ts") or "",
                    [
                        "debounceTime",
                        "switchMap",
                        "take(1)",
                        "takeUntil",
                        re.compile(r"first\(\)", re.IGNORECASE)
                    ]
                ),
                reason="Async project code validator incorporates debounce or subscription cancellation",
            )
        )
    ))

    specs.append(TicketSpec(
        number=13,
        ticket_id="GOLDEN-13",
        category="FE",
        difficulty="Hard",
        title="Multi-step wizard losing form state on step 3 back-nav",
        description="Navigating back from step 3 in wizard resets step 1 and 2 inputs. Preserve form model in service.",
        acceptance_criteria=["Wizard service or component preserves form value across steps"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/wizard/wizard.component.ts") or
                    _read_file_text(ws, "src/app/wizard/wizard.service.ts") or "",
                    [
                        re.compile(r"patchValue\([^)]*savedState", re.IGNORECASE),
                        re.compile(r"wizardState", re.IGNORECASE),
                        "retainFormState",
                        re.compile(r"formData\[[^\]]*\]\s*=", re.IGNORECASE)
                    ]
                ),
                reason="Wizard preserves form state on backwards navigation",
            )
        )
    ))

    specs.append(TicketSpec(
        number=14,
        ticket_id="GOLDEN-14",
        category="FE",
        difficulty="Hard",
        title="Drag-and-drop listener unsubscription on ngOnDestroy",
        description="Global document dragover/drop listeners leak memory when component unmounts. Remove on destroy.",
        acceptance_criteria=["Component unregisters event listeners inside ngOnDestroy"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/shared/drag-drop/drag-drop.directive.ts") or
                    _read_file_text(ws, "src/app/shared/drag-drop.directive.ts") or "",
                    [
                        "removeEventListener",
                        re.compile(r"ngOnDestroy\(\)[^{]*\{[^}]*(?:destroy|unbind|remove|unsubscribe)", re.DOTALL),
                        "destroy$.next()"
                    ]
                ),
                reason="Drag and drop directive properly cleans up listeners on ngOnDestroy",
            )
        )
    ))

    specs.append(TicketSpec(
        number=15,
        ticket_id="GOLDEN-15",
        category="FE",
        difficulty="Hard",
        title="Table pagination resets to page 1 on inline row edit",
        description="Editing a row inline triggers full table re-query and resets current page to 1. Preserve page index.",
        acceptance_criteria=["Row edit preserves current page index and page size during reload"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/shared/table/table.component.ts") or
                    _read_file_text(ws, "table.component.ts") or "",
                    [
                        re.compile(r"pageIndex:\s*this\.pageIndex", re.IGNORECASE),
                        re.compile(r"currentPage\s*=\s*pageIndex", re.IGNORECASE),
                        "preservePageIndex",
                        re.compile(r"loadPage\(this\.currentPage\)", re.IGNORECASE)
                    ]
                ),
                reason="Table row edit preserves current page index",
            )
        )
    ))

    # ------------------------------------------------------------------------
    # Category 3: Backend Java Spring Boot / JPA (Tickets 16–22)
    # ------------------------------------------------------------------------
    specs.append(TicketSpec(
        number=16,
        ticket_id="GOLDEN-16",
        category="BE",
        difficulty="Easy",
        title="Add @NotNull and @Size validation to MemberDto.email",
        description="Add JSR-380 validation annotations to MemberDto email field.",
        acceptance_criteria=["MemberDto.java has @NotNull and @Size or @Email annotations on email field"],
        required_prereqs=["git", "java"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "project-service/src/main/java/com/cloudforge/project/dto/MemberDto.java") or
                    _read_file_text(ws, "MemberDto.java") or "",
                    [
                        re.compile(r"@NotNull[^\n]*\n[^\n]*@Size[^\n]*\n[^\n]*private\s+String\s+email", re.MULTILINE),
                        re.compile(r"@NotBlank[^\n]*\n[^\n]*private\s+String\s+email", re.MULTILINE),
                        re.compile(r"@Email[^\n]*private\s+String\s+email", re.MULTILINE)
                    ]
                ),
                reason="MemberDto email field annotated with validation constraints",
            )
        )
    ))

    specs.append(TicketSpec(
        number=17,
        ticket_id="GOLDEN-17",
        category="BE",
        difficulty="Easy",
        title="Update project status enum to include 'ARCHIVED'",
        description="Add ARCHIVED value to ProjectStatus enum in project-service.",
        acceptance_criteria=["ProjectStatus.java enum includes ARCHIVED"],
        required_prereqs=["git", "java"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "project-service/src/main/java/com/cloudforge/project/model/ProjectStatus.java") or
                    _read_file_text(ws, "ProjectStatus.java") or "",
                    ["ARCHIVED"]
                ),
                reason="ProjectStatus enum includes ARCHIVED constant",
            )
        )
    ))

    specs.append(TicketSpec(
        number=18,
        ticket_id="GOLDEN-18",
        category="BE",
        difficulty="Med",
        title="Filter members endpoint by organizationId query param",
        description="Update GET /api/members controller endpoint to accept optional organizationId query parameter.",
        acceptance_criteria=["Controller endpoint defines @RequestParam(value = 'organizationId', required = false)"],
        required_prereqs=["git", "java"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "project-service/src/main/java/com/cloudforge/project/controller/MemberController.java") or
                    _read_file_text(ws, "MemberController.java") or "",
                    [
                        re.compile(r"@RequestParam\([^)]*organizationId[^)]*\)", re.IGNORECASE),
                        re.compile(r"findByOrganizationId", re.IGNORECASE)
                    ]
                ),
                reason="MemberController includes organizationId request parameter",
            )
        )
    ))

    specs.append(TicketSpec(
        number=19,
        ticket_id="GOLDEN-19",
        category="BE",
        difficulty="Med",
        title="Duplicate email throws 500 instead of 409 Conflict",
        description="Add exception handler for DuplicateKeyException or DataIntegrityViolationException mapping to HTTP 409.",
        acceptance_criteria=["ControllerAdvice maps unique constraint violations to HttpStatus.CONFLICT"],
        required_prereqs=["git", "java"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "project-service/src/main/java/com/cloudforge/project/exception/GlobalExceptionHandler.java") or
                    _read_file_text(ws, "GlobalExceptionHandler.java") or "",
                    [
                        re.compile(r"@ResponseStatus\(\s*HttpStatus\.CONFLICT\s*\)", re.IGNORECASE),
                        "HttpStatus.CONFLICT",
                        "409"
                    ]
                ),
                reason="GlobalExceptionHandler maps duplicate constraint exception to HTTP 409 Conflict",
            )
        )
    ))

    specs.append(TicketSpec(
        number=20,
        ticket_id="GOLDEN-20",
        category="BE",
        difficulty="Med",
        title="Soft-delete deliverables via JPA cascade on project delete",
        description="Configure soft deletion on Deliverable entity when parent project is deleted.",
        acceptance_criteria=["Deliverable entity configured with @SQLDelete or deleted flag update on cascade"],
        required_prereqs=["git", "java"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "project-service/src/main/java/com/cloudforge/project/entity/Deliverable.java") or
                    _read_file_text(ws, "Deliverable.java") or "",
                    [
                        "@SQLDelete",
                        re.compile(r"@Where\(clause\s*=\s*[\"']deleted\s*=\s*false[\"']\)", re.IGNORECASE),
                        "isDeleted",
                        re.compile(r"deleted\s*=\s*true", re.IGNORECASE)
                    ]
                ),
                reason="Deliverable entity annotated with soft-delete clause",
            )
        )
    ))

    specs.append(TicketSpec(
        number=21,
        ticket_id="GOLDEN-21",
        category="BE",
        difficulty="Hard",
        title="Deadlock in member batch import under concurrent load",
        description="Concurrent batch imports cause database deadlock. Enforce deterministic sorting before lock acquisition.",
        acceptance_criteria=["Batch import sorts member records by key or acquires pessimistic write locks"],
        required_prereqs=["git", "java"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "project-service/src/main/java/com/cloudforge/project/service/MemberBatchService.java") or
                    _read_file_text(ws, "MemberBatchService.java") or "",
                    [
                        re.compile(r"sort\([^)]*Comparator", re.IGNORECASE),
                        "LockModeType.PESSIMISTIC_WRITE",
                        "orderedBatch",
                        re.compile(r"Comparator\.comparing", re.IGNORECASE)
                    ]
                ),
                reason="Member batch import sorts items or locks rows deterministically",
            )
        )
    ))

    specs.append(TicketSpec(
        number=22,
        ticket_id="GOLDEN-22",
        category="BE",
        difficulty="Hard",
        title="Add role-based @PreAuthorize to deliverable download",
        description="Secure deliverable download endpoint with @PreAuthorize check for ADMIN or PROJECT_LEAD.",
        acceptance_criteria=["Endpoint method annotated with @PreAuthorize('hasRole(...)')"],
        required_prereqs=["git", "java"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "project-service/src/main/java/com/cloudforge/project/controller/DeliverableController.java") or
                    _read_file_text(ws, "DeliverableController.java") or "",
                    [
                        re.compile(r"@PreAuthorize\([^)]*hasRole[^)]*\)", re.IGNORECASE),
                        re.compile(r"@PreAuthorize\([^)]*hasAnyAuthority[^)]*\)", re.IGNORECASE),
                        re.compile(r"@Secured\([^)]*\)", re.IGNORECASE)
                    ]
                ),
                reason="Deliverable download endpoint guarded with role authorization",
            )
        )
    ))

    # ------------------------------------------------------------------------
    # Category 4: Full-Stack Cross-System Boundaries (Tickets 23–29)
    # ------------------------------------------------------------------------
    specs.append(TicketSpec(
        number=23,
        ticket_id="GOLDEN-23",
        category="Mix",
        difficulty="Med",
        title="Format backend ISO timestamp to localized UI date",
        description="Deliverable cards display raw ISO timestamp '2026-09-21T14:30:00Z'. Pipe to localized format.",
        acceptance_criteria=["Template uses DatePipe or localized date formatting"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/projects/deliverables/deliverable-item/deliverable-item.component.html") or
                    _read_file_text(ws, "deliverable-item.component.html") or "",
                    [
                        re.compile(r"\|\s*date(?::['\"][^'\"]*['\"])?", re.IGNORECASE),
                        "formatDate",
                        "toLocaleDateString"
                    ]
                ),
                reason="Template applies date pipe or formatting method to deliverable timestamp",
            )
        )
    ))

    specs.append(TicketSpec(
        number=24,
        ticket_id="GOLDEN-24",
        category="Mix",
        difficulty="Hard",
        title="Add 'department' from backend Employee to frontend card",
        description="Add 'department' field to backend Employee DTO and display it on frontend member card.",
        acceptance_criteria=["Backend Employee DTO has department field and frontend template renders department"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=(
                    _check_any_pattern(_read_file_text(ws, "EmployeeDto.java") or _read_file_text(ws, "Employee.java") or "", ["department"]) or
                    _check_any_pattern(_read_file_text(ws, "member-card.component.html") or _read_file_text(ws, "member.model.ts") or "", ["department"])
                ),
                reason="Department field added across backend DTO or frontend component",
            )
        )
    ))

    specs.append(TicketSpec(
        number=25,
        ticket_id="GOLDEN-25",
        category="Mix",
        difficulty="Hard",
        title="Frontend download fails: backend returns multipart zip",
        description="Frontend HTTP call expects JSON, failing on zip binary response. Set responseType: 'blob'.",
        acceptance_criteria=["HTTP GET request specifies responseType: 'blob'"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/projects/deliverables/deliverable.service.ts") or
                    _read_file_text(ws, "deliverable.service.ts") or "",
                    [
                        "responseType: 'blob'",
                        'responseType: "blob"',
                        "responseType: ('blob' as const)"
                    ]
                ),
                reason="HTTP request configures responseType: 'blob' for binary download",
            )
        )
    ))

    specs.append(TicketSpec(
        number=26,
        ticket_id="GOLDEN-26",
        category="Mix",
        difficulty="Hard",
        title="Contract status change not reflected in UI without refresh",
        description="Listen to WebSocket / SSE contract status update events to update UI reactively.",
        acceptance_criteria=["Component subscribes to contract status event stream and updates state"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/contracts/contract-detail/contract-detail.component.ts") or
                    _read_file_text(ws, "contract-detail.component.ts") or "",
                    [
                        re.compile(r"eventSource|webSocket|contractStatusUpdates|onStatusChange", re.IGNORECASE),
                        re.compile(r"subscribe\([^)]*status\s*=>", re.IGNORECASE)
                    ]
                ),
                reason="Contract component subscribes to status update event stream",
            )
        )
    ))

    specs.append(TicketSpec(
        number=27,
        ticket_id="GOLDEN-27",
        category="Mix",
        difficulty="Hard",
        title="Pagination totalElements mismatch between Spring & UI",
        description="Spring Page object returns totalElements; frontend expects totalCount. Map response in adapter.",
        acceptance_criteria=["Service maps response.totalElements to table paginator totalCount"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/shared/table/table-data-source.ts") or
                    _read_file_text(ws, "member.service.ts") or "",
                    [
                        "totalElements",
                        re.compile(r"totalCount:\s*res\.totalElements", re.IGNORECASE),
                        re.compile(r"map\([^)]*res\.totalElements", re.IGNORECASE)
                    ]
                ),
                reason="Service maps Spring totalElements to table total count",
            )
        )
    ))

    specs.append(TicketSpec(
        number=28,
        ticket_id="GOLDEN-28",
        category="Mix",
        difficulty="Hard",
        title="OAuth2 token expiry during upload freezes UI",
        description="401 Unauthorized during upload leaves spinner hanging. Interceptor should refresh token and retry.",
        acceptance_criteria=["HTTP Interceptor catches 401 error, attempts token refresh, and retries request"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/core/interceptors/auth.interceptor.ts") or
                    _read_file_text(ws, "auth.interceptor.ts") or "",
                    [
                        re.compile(r"status\s*===\s*401", re.IGNORECASE),
                        "refreshToken",
                        re.compile(r"catchError\([^)]*401", re.IGNORECASE)
                    ]
                ),
                reason="Auth interceptor handles 401 status with token refresh",
            )
        )
    ))

    specs.append(TicketSpec(
        number=29,
        ticket_id="GOLDEN-29",
        category="Mix",
        difficulty="Hard",
        title="GraphQL mutation for member role missing input field",
        description="Add optional permissions input field to updateMemberRole GraphQL mutation.",
        acceptance_criteria=[".graphql schema or mutation definition includes permissions field"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/graphql/mutations/updateMemberRole.graphql") or
                    _read_file_text(ws, "schema.graphql") or "",
                    ["permissions", "rolePermissions", "$permissions"]
                ),
                reason="GraphQL schema or mutation incorporates permissions parameter",
            )
        )
    ))

    # ------------------------------------------------------------------------
    # Category 5: Configuration / Build / CI (Tickets 30–35)
    # ------------------------------------------------------------------------
    specs.append(TicketSpec(
        number=30,
        ticket_id="GOLDEN-30",
        category="Cfg",
        difficulty="Easy",
        title="Update backend application-dev.yml server port to 8085",
        description="Change server.port in application-dev.yml from 8080 to 8085.",
        acceptance_criteria=["application-dev.yml defines server.port as 8085"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "project-service/src/main/resources/application-dev.yml") or
                    _read_file_text(ws, "application-dev.yml") or "",
                    [
                        re.compile(r"port:\s*8085", re.IGNORECASE),
                        "server.port: 8085",
                        "server.port=8085"
                    ]
                ),
                reason="application-dev.yml sets server.port to 8085",
            )
        )
    ))

    specs.append(TicketSpec(
        number=31,
        ticket_id="GOLDEN-31",
        category="Cfg",
        difficulty="Easy",
        title="Add CORS allowed origin for staging domain",
        description="Add 'https://staging.cloudforge.internal' to allowed CORS origins in backend configuration.",
        acceptance_criteria=["CORS configuration allows staging.cloudforge.internal origin"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "project-service/src/main/java/com/cloudforge/project/config/CorsConfig.java") or
                    _read_file_text(ws, "CorsConfig.java") or
                    _read_file_text(ws, "application.yml") or "",
                    ["https://staging.cloudforge.internal", "staging.cloudforge.internal"]
                ),
                reason="CORS configuration contains staging domain origin",
            )
        )
    ))

    specs.append(TicketSpec(
        number=32,
        ticket_id="GOLDEN-32",
        category="Cfg",
        difficulty="Med",
        title="Angular proxy.conf.json routing /api/areas to wrong port",
        description="Update proxy.conf.json to route /api/areas to target port 8082 instead of 8080.",
        acceptance_criteria=["proxy.conf.json maps /api/areas to localhost:8082"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "proxy.conf.json") or
                    _read_file_text(ws, "src/proxy.conf.json") or "",
                    [
                        re.compile(r'"/api/areas"[^}]*"target":\s*"http://localhost:8082"', re.DOTALL),
                        "localhost:8082"
                    ]
                ),
                reason="proxy.conf.json routes /api/areas to port 8082",
            )
        )
    ))

    specs.append(TicketSpec(
        number=33,
        ticket_id="GOLDEN-33",
        category="Cfg",
        difficulty="Med",
        title="Docker Compose RabbitMQ healthcheck command fix",
        description="Update RabbitMQ healthcheck command in docker-compose.yml to rabbitmq-diagnostics check_running.",
        acceptance_criteria=["docker-compose.yml specifies rabbitmq-diagnostics check_running"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "docker-compose.yml") or
                    _read_file_text(ws, "docker-compose.test.yml") or "",
                    ["rabbitmq-diagnostics check_running", "rabbitmq-diagnostics -q ping"]
                ),
                reason="docker-compose.yml contains rabbitmq healthcheck diagnostic command",
            )
        )
    ))

    specs.append(TicketSpec(
        number=34,
        ticket_id="GOLDEN-34",
        category="Cfg",
        difficulty="Hard",
        title="Maven build failure: dependency convergence on jackson",
        description="Lock jackson-databind version in root pom.xml dependencyManagement to 2.15.2.",
        acceptance_criteria=["pom.xml defines dependencyManagement for jackson-databind version 2.15.2"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "pom.xml") or
                    _read_file_text(ws, "project-service/pom.xml") or "",
                    ["jackson-databind", "2.15.2"]
                ),
                reason="pom.xml locks jackson-databind dependency version",
            )
        )
    ))

    specs.append(TicketSpec(
        number=35,
        ticket_id="GOLDEN-35",
        category="Cfg",
        difficulty="Hard",
        title="Node 20 upgrade breaks Angular build (OpenSSL error)",
        description="Add --openssl-legacy-provider to package.json build script for Node 20 compatibility.",
        acceptance_criteria=["package.json build script includes --openssl-legacy-provider or Node crypto options"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "package.json") or "",
                    ["--openssl-legacy-provider", "NODE_OPTIONS"]
                ),
                reason="package.json build command includes openssl legacy provider flag",
            )
        )
    ))

    # ------------------------------------------------------------------------
    # Category 6: Inquiry / Information-Only (Tickets 36–40)
    # ------------------------------------------------------------------------
    specs.append(TicketSpec(
        number=36,
        ticket_id="GOLDEN-36",
        category="Info",
        difficulty="Easy",
        title="Where is the JWT token stored and validated on frontend?",
        description="Explain where the JWT token is persisted (sessionStorage / localStorage) and which interceptor validates it.",
        acceptance_criteria=["Identifies storage location and citations to auth.service.ts and auth.interceptor.ts without editing files"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            evaluate_inquiry_rubric(
                answer_text=res.get("solution_guidance") or res.get("answer") or "",
                changed_files=res.get("changed_files", []),
                expected_citations=["auth.service", "interceptor"],
                key_symbols=["token", "sessionStorage", "localStorage"]
            )
        )
    ))

    specs.append(TicketSpec(
        number=37,
        ticket_id="GOLDEN-37",
        category="Info",
        difficulty="Easy",
        title="Which microservice owns the 'deliverables' DB table?",
        description="Identify which backend service repository defines and manages the deliverables database entity.",
        acceptance_criteria=["Identifies project-service as owner of deliverables table without modifying files"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            evaluate_inquiry_rubric(
                answer_text=res.get("solution_guidance") or res.get("answer") or "",
                changed_files=res.get("changed_files", []),
                expected_citations=["project-service", "Deliverable"],
                key_symbols=["deliverables", "project"]
            )
        )
    ))

    specs.append(TicketSpec(
        number=38,
        ticket_id="GOLDEN-38",
        category="Info",
        difficulty="Med",
        title="Explain the data flow when clicking 'Register Document'",
        description="Trace architectural sequence from user click in document browser through service layer to backend API.",
        acceptance_criteria=["Provides sequential flow across UI component, service, and backend endpoint with 0 code changes"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            evaluate_inquiry_rubric(
                answer_text=res.get("solution_guidance") or res.get("answer") or "",
                changed_files=res.get("changed_files", []),
                expected_citations=["document-file-browser", "uploadDocument"],
                key_symbols=["register", "document", "upload"]
            )
        )
    ))

    specs.append(TicketSpec(
        number=39,
        ticket_id="GOLDEN-39",
        category="Info",
        difficulty="Med",
        title="List all endpoints protected by ADMIN in area-service",
        description="Provide a grounded list of all REST controller endpoints protected by ADMIN role in area-service.",
        acceptance_criteria=["Lists matching endpoints with exact URI paths and annotations with 0 file modifications"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            evaluate_inquiry_rubric(
                answer_text=res.get("solution_guidance") or res.get("answer") or "",
                changed_files=res.get("changed_files", []),
                expected_citations=["area-service", "AreaController"],
                key_symbols=["ADMIN", "PreAuthorize"]
            )
        )
    ))

    specs.append(TicketSpec(
        number=40,
        ticket_id="GOLDEN-40",
        category="Info",
        difficulty="Hard",
        title="Compare permission checks between v1 and v2 controllers",
        description="Compare authorization mechanisms and permission checks between v1 and v2 API controllers.",
        acceptance_criteria=["Presents comparative breakdown of security checks across API versions with 0 code writes"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            evaluate_inquiry_rubric(
                answer_text=res.get("solution_guidance") or res.get("answer") or "",
                changed_files=res.get("changed_files", []),
                expected_citations=["v1", "v2"],
                key_symbols=["permission", "controller", "role"]
            )
        )
    ))

    # ------------------------------------------------------------------------
    # Category 7: Debugging & Diagnostic Investigations (Tickets 41–45)
    # ------------------------------------------------------------------------
    specs.append(TicketSpec(
        number=41,
        ticket_id="GOLDEN-41",
        category="Dbg",
        difficulty="Easy",
        title="TypeError: Cannot read property 'name' of undefined in card",
        description="Template crashes when card data is loading. Add Angular safe navigation operator (?.) to item.name.",
        acceptance_criteria=["Template safely guards property access with item?.name or ngIf"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/shared/card/card.component.html") or
                    _read_file_text(ws, "card.component.html") or "",
                    [
                        "item?.name",
                        "card?.name",
                        re.compile(r"\*\s*ngIf\s*=\s*['\"][^'\"]*name['\"]", re.IGNORECASE)
                    ]
                ),
                reason="Template uses safe navigation operator or null check for item name",
            )
        )
    ))

    specs.append(TicketSpec(
        number=42,
        ticket_id="GOLDEN-42",
        category="Dbg",
        difficulty="Med",
        title="Save button stays disabled when required fields filled",
        description="Form validation status is not recalculated after programmatic patchValue. Add updateValueAndValidity.",
        acceptance_criteria=["Form calls updateValueAndValidity() or recalculates validity after patch"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/organization/members/add-members/add-members.component.ts") or
                    _read_file_text(ws, "add-members.component.ts") or "",
                    [
                        "updateValueAndValidity",
                        "markAsDirty",
                        "markAsTouched"
                    ]
                ),
                reason="Component invokes form validity update method after changes",
            )
        )
    ))

    specs.append(TicketSpec(
        number=43,
        ticket_id="GOLDEN-43",
        category="Dbg",
        difficulty="Med",
        title="HTTP 500 on GET /api/members when billingAddress is null",
        description="Null pointer exception when building MemberDto from entity with null billing address. Add null-safe mapper.",
        acceptance_criteria=["Mapper checks if billingAddress is null before accessing nested fields"],
        required_prereqs=["git", "java"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "MemberMapper.java") or _read_file_text(ws, "MemberService.java") or "",
                    [
                        "Optional.ofNullable",
                        re.compile(r"getBillingAddress\(\)\s*!=\s*null", re.IGNORECASE),
                        re.compile(r"billingAddress\s*==\s*null", re.IGNORECASE)
                    ]
                ),
                reason="Mapper guards against null billing address",
            )
        )
    ))

    specs.append(TicketSpec(
        number=44,
        ticket_id="GOLDEN-44",
        category="Dbg",
        difficulty="Hard",
        title="Race condition: fast tab switching displays stale data",
        description="Fast tab clicking causes stale asynchronous responses to overwrite newer selections.",
        acceptance_criteria=["Data fetch uses switchMap to cancel prior requests, or uses takeUntil/cancellation flag"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/projects/tabs/project-tabs.component.ts") or
                    _read_file_text(ws, "project-tabs.component.ts") or "",
                    [
                        "switchMap",
                        "takeUntil",
                        re.compile(r"activeRequestId|requestSequence|cancelPrevious", re.IGNORECASE)
                    ]
                ),
                reason="Component implements switchMap or cancellation mechanism to resolve tab race condition",
            )
        )
    ))

    specs.append(TicketSpec(
        number=45,
        ticket_id="GOLDEN-45",
        category="Dbg",
        difficulty="Hard",
        title="File upload hangs at 99% with no error reported",
        description="Upload progress reaches 99% and hangs indefinitely on server connection drop. Handle error event.",
        acceptance_criteria=["Upload observable catches error event and sets error status"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/shared/document-file-browser/document-file-browser/document-file-browser.component.ts") or
                    _read_file_text(ws, "document-file-browser.component.ts") or "",
                    [
                        "HttpEventType.ResponseHeader",
                        "catchError",
                        re.compile(r"uploadFailed|uploadError|onUploadError", re.IGNORECASE)
                    ]
                ),
                reason="Upload routine handles HTTP error events and emits error state",
            )
        )
    ))

    # ------------------------------------------------------------------------
    # Category 8: Negative Outcomes & Missing Errors (Tickets 46–50)
    # ------------------------------------------------------------------------
    specs.append(TicketSpec(
        number=46,
        ticket_id="GOLDEN-46",
        category="Neg",
        difficulty="Easy",
        title="No error banner shown when file exceeds 25MB limit",
        description="File input silently ignores files > 25MB without notifying user. Display error banner.",
        acceptance_criteria=["Validation checks file.size > 25MB and sets error message"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/shared/document-file-browser/document-file-browser/document-file-browser.component.ts") or
                    _read_file_text(ws, "src/assets/data/uploadDocumentRegister.json") or "",
                    [
                        "26214400",       # 25 * 1024 * 1024
                        "25 * 1024",
                        re.compile(r"maxFileSize|fileSizeExceeded|FILE_TOO_LARGE", re.IGNORECASE)
                    ]
                ),
                reason="File size validator bounds uploads to 25MB with error signal",
            )
        )
    ))

    specs.append(TicketSpec(
        number=47,
        ticket_id="GOLDEN-47",
        category="Neg",
        difficulty="Med",
        title="Empty catch block in project.service.ts swallows errors",
        description="Empty catch block in project.service.ts silently swallows network failures. Log and rethrow.",
        acceptance_criteria=["Catch block in project.service.ts logs error or rethrows with throwError"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/projects/services/project.service.ts") or
                    _read_file_text(ws, "project.service.ts") or "",
                    [
                        "console.error",
                        "this.logger.error",
                        "throwError",
                        re.compile(r"catchError\([^)]*=>\s*\{[^}]+", re.MULTILINE)
                    ]
                ),
                reason="Project service catch block logs or rethrows error instead of empty swallow",
            )
        )
    ))

    specs.append(TicketSpec(
        number=48,
        ticket_id="GOLDEN-48",
        category="Neg",
        difficulty="Med",
        title="Backend returns 403 but frontend shows generic 500 toast",
        description="Frontend HTTP error handler catches 403 Forbidden and displays generic 500 'Internal Server Error' toast. Differentiate 403.",
        acceptance_criteria=["Error handler distinguishes HTTP 403 and shows Access Denied or Forbidden notification"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/core/services/error-handler.service.ts") or
                    _read_file_text(ws, "error-handler.service.ts") or "",
                    [
                        re.compile(r"status\s*===\s*403", re.IGNORECASE),
                        "Access Denied",
                        "Forbidden",
                        "403"
                    ]
                ),
                reason="Error handler explicitly differentiates 403 status from generic 500 errors",
            )
        )
    ))

    specs.append(TicketSpec(
        number=49,
        ticket_id="GOLDEN-49",
        category="Neg",
        difficulty="Hard",
        title="Form submission fails silently without scrolling to error",
        description="When form has invalid off-screen inputs, clicking submit does nothing silently. Mark all controls as touched and scroll to first error.",
        acceptance_criteria=["Submit marks all controls as touched (markAllAsTouched) and invokes scrollIntoView"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/shared/form/dynamic-form.component.ts") or
                    _read_file_text(ws, "add-members.component.ts") or "",
                    [
                        "markAllAsTouched",
                        "scrollIntoView",
                        re.compile(r"markAsTouched\(\)", re.IGNORECASE)
                    ]
                ),
                reason="Form submit routine marks invalid controls and brings error into viewport",
            )
        )
    ))

    specs.append(TicketSpec(
        number=50,
        ticket_id="GOLDEN-50",
        category="Neg",
        difficulty="Hard",
        title="Session timeout silently drops unsaved contract draft",
        description="Session expiry redirect wipes in-progress contract draft without prompting or auto-saving. Hook auto-save to auth expiry.",
        acceptance_criteria=["Draft service saves form state to storage upon session timeout event"],
        required_prereqs=["git"],
        evaluator=lambda ws, res: (
            Tier0AssertionResult(
                passed=_check_any_pattern(
                    _read_file_text(ws, "src/app/contracts/contract-form/contract-form.component.ts") or
                    _read_file_text(ws, "contract-form.component.ts") or "",
                    [
                        re.compile(r"saveDraft|autoSave|persistDraft", re.IGNORECASE),
                        re.compile(r"sessionStorage\.setItem\([^)]*draft", re.IGNORECASE),
                        "onSessionTimeout"
                    ]
                ),
                reason="Contract form implements draft auto-save hook on session expiry",
            )
        )
    ))

    return specs


# ============================================================================
# BENCHMARK EXECUTION RUNNER
# ============================================================================

def execute_single_ticket(
    spec: TicketSpec,
    workspace: Path,
    mode: str,
    dry_run: bool = False,
    allow_live_reset: bool = False,
    available_tools: Optional[Dict[str, bool]] = None
) -> TicketExecutionResult:
    """Execute evaluation for a single benchmark ticket."""
    start_time = time.time()
    available = available_tools or detect_toolchain()

    # Step 1: Toolchain check
    has_prereqs, missing = check_ticket_prereqs(spec, available)
    if not has_prereqs:
        logger.warning(f"[{spec.ticket_id}] SKIPPED: Missing required tools: {missing}")
        return TicketExecutionResult(
            ticket_id=spec.ticket_id,
            number=spec.number,
            category=spec.category,
            difficulty=spec.difficulty,
            title=spec.title,
            status="SKIPPED_PREREQ",
            wall_clock_s=0.0,
            tokens_in=0,
            tokens_out=0,
            total_tokens=0,
            tier0_passed=False,
            tier0_reason=f"Missing prerequisites: {missing}",
            error_message=f"Missing toolchain: {missing}"
        )

    # Configure limits based on mode
    if mode == "baseline":
        token_limit = 800_000
        # Real-pipeline data (GOLDEN-01: 43.5min / 402k tokens for an EASY
        # ticket) shows 15min/500k saturated. Uncensored baseline headroom:
        wall_limit = 5400.0  # 90 mins
    else:
        token_limit = 120_000
        wall_limit = 180.0  # 3 mins

    # Step 2: Reset workspace before run (Week 0 between-ticket reset procedure)
    if not dry_run:
        reset_fixture_workspace(workspace, allow_live_reset=allow_live_reset)

    # Step 3: Run pipeline or dry-run
    run_output: Dict[str, Any] = {
        "solution_guidance": "",
        "changed_files": [],
        "tokens_in": 0,
        "tokens_out": 0,
        "total_tokens": 0,
        "phase_tokens": {}
    }
    status = "RESOLVED"
    error_msg = None

    if dry_run:
        logger.info(f"[{spec.ticket_id}] DRY-RUN: Evaluating existing workspace state without invoking pipeline.")
        run_output["changed_files"] = get_git_status_changed_files(workspace)
    else:
        logger.info(f"[{spec.ticket_id}] START: Running pipeline worker (Timeout: {wall_limit:.0f}s, MaxTokens: {token_limit})...")
        tmp_output = workspace / ".aviator" / f"harness_tmp_{spec.ticket_id}.json"
        tmp_output.parent.mkdir(parents=True, exist_ok=True)
        if tmp_output.exists():
            try:
                tmp_output.unlink()
            except Exception:
                pass

        repo_root = Path(__file__).resolve().parent.parent.parent
        src_path = str(repo_root / "src")
        worker_env = os.environ.copy()
        existing_pp = worker_env.get("PYTHONPATH", "")
        worker_env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pp}" if existing_pp else src_path

        # Prefer the repo virtualenv interpreter: the pipeline depends on
        # packages installed there (aviator, langchain, langgraph) that a
        # system Python cannot import — unit tests don't catch this because
        # tests/conftest.py stubs all aviator.* modules.
        worker_python = sys.executable
        for cand in (repo_root / ".venv" / "Scripts" / "python.exe", repo_root / ".venv" / "bin" / "python"):
            if cand.exists():
                worker_python = str(cand)
                break

        cmd = [
            worker_python, "-m", "tests.benchmarks.golden_50_harness",
            "--worker-ticket", spec.ticket_id,
            "--workspace", str(workspace),
            "--token-limit", str(token_limit),
            "--output-file", str(tmp_output)
        ]

        proc = None
        # Stream worker output to a per-ticket log so it stays inspectable
        # even after a timeout kill (PIPE buffers would be lost).
        # Keep worker logs OUTSIDE the fixture workspace: the post-ticket
        # `git clean -fd` would otherwise destroy the diagnostics we need
        # when a run fails or times out.
        worker_log = Path(__file__).resolve().parent / "logs" / f"harness_worker_{spec.ticket_id}.log"
        worker_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(worker_log, "w", encoding="utf-8") as log_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=worker_env,
                    cwd=str(repo_root)
            )
            proc.communicate(timeout=wall_limit)

            if proc.returncode != 0:
                tail = _read_worker_log_tail(worker_log)
                logger.error(f"[{spec.ticket_id}] Worker failed with exit code {proc.returncode}:\n{tail}")
                status = "PIPELINE_ERROR"
                error_msg = tail or f"Worker exited with code {proc.returncode}"

            if tmp_output.exists():
                try:
                    run_output = json.loads(tmp_output.read_text(encoding="utf-8"))
                except Exception as je:
                    logger.error(f"Failed to parse worker output JSON: {je}")
                    status = "PIPELINE_ERROR"
                    error_msg = f"Worker JSON parse error: {je}"
            elif status != "PIPELINE_ERROR":
                status = "PIPELINE_ERROR"
                error_msg = "Worker exited without creating output JSON"

        except subprocess.TimeoutExpired:
            logger.error(f"[{spec.ticket_id}] TIMEOUT: Worker exceeded wall-clock limit of {wall_limit:.0f}s!")
            status = "TIMEOUT"
            error_msg = f"Timed out after {wall_limit:.0f}s (worker log: {worker_log.name})"
            if proc:
                try:
                    # Windows process tree termination
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                except Exception:
                    proc.kill()
        finally:
            if tmp_output.exists():
                try:
                    tmp_output.unlink()
                except Exception:
                    pass

    wall_clock = time.time() - start_time
    if status == "TIMEOUT":
        wall_clock = max(wall_clock, wall_limit)

    # Step 4: Run independent Tier 0 evaluator ON THE MODIFIED WORKSPACE
    tier0_res = spec.evaluator(workspace, run_output)
    if status == "RESOLVED":
        if not tier0_res.passed:
            status = "FAILED_ASSERTION"

    logger.info(
        f"[{spec.ticket_id}] RESULT: {status} in {wall_clock:.1f}s | "
        f"Tier0: {tier0_res.passed} ({tier0_res.reason}) | "
        f"Tokens: {run_output.get('total_tokens', 0):,}"
    )

    # Step 5: Post-ticket clean reset AFTER evaluation has completed
    if not dry_run:
        try:
            reset_fixture_workspace(workspace, allow_live_reset=allow_live_reset)
        except Exception as clean_err:
            logger.warning(f"[{spec.ticket_id}] Post-ticket reset failed (non-fatal): {clean_err}")

    return TicketExecutionResult(
        ticket_id=spec.ticket_id,
        number=spec.number,
        category=spec.category,
        difficulty=spec.difficulty,
        title=spec.title,
        status=status,
        wall_clock_s=wall_clock,
        tokens_in=run_output.get("tokens_in", 0),
        tokens_out=run_output.get("tokens_out", 0),
        total_tokens=run_output.get("total_tokens", 0),
        tier0_passed=tier0_res.passed,
        tier0_reason=tier0_res.reason,
        changed_files=run_output.get("changed_files", []),
        phase_tokens=run_output.get("phase_tokens", {}),
        error_message=error_msg
    )


def _load_worker_environment() -> None:
    """Load .env and resolve GOOGLE_APPLICATION_CREDENTIALS for the worker subprocess.

    Mirrors the loading logic in production_readiness_benchmark.py so headless
    workers authenticate exactly like the chatbot backend does.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials and not Path(credentials).is_absolute():
        resolved = repo_root / credentials
        if resolved.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved.resolve())
    elif not credentials:
        default_credentials = repo_root / "otl-cs-csai.json"
        if default_credentials.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(default_credentials.resolve())


def _read_worker_log_tail(worker_log: Path, max_chars: int = 2000) -> str:
    """Read the tail of a worker log file for error surfacing."""
    try:
        if worker_log.exists():
            return worker_log.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except Exception:
        pass
    return ""


def run_worker_ticket(
    ticket_id: str,
    workspace: Path,
    output_file: Path,
    token_limit: int = 500_000
) -> None:
    """Worker process entrypoint: runs a single ticket headlessly and writes metrics to JSON."""
    os.environ["HEADLESS_MODE"] = "1"
    os.environ["AUTO_APPROVE"] = "1"
    os.environ["CI"] = "1"
    os.environ["AVIATOR_MAX_TOKENS"] = str(token_limit)

    _load_worker_environment()

    specs = {s.ticket_id: s for s in make_golden_50_specs()}
    if ticket_id not in specs:
        raise ValueError(f"Unknown ticket ID for worker: {ticket_id}")
    spec = specs[ticket_id]

    # Ensure src directory is in sys.path
    src_dir = Path(__file__).resolve().parent.parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from ticket_to_code import ValueEdgeTicket, run_autonomous_workflow

    ticket_input = ValueEdgeTicket(
        ticket_id=spec.ticket_id,
        title=spec.title,
        description=spec.description,
        acceptance_criteria=spec.acceptance_criteria
    )

    state = run_autonomous_workflow(
        ticket=ticket_input,
        workspace_path=str(workspace)
    )

    out = {
        "status": state.get("status", "unknown") if isinstance(state, dict) else "unknown",
        "solution_guidance": str(state.get("solution_guidance") or "") if isinstance(state, dict) else "",
        "changed_files": get_git_status_changed_files(workspace),
        "tokens_in": 0,
        "tokens_out": 0,
        "total_tokens": 0,
        "phase_tokens": {}
    }

    if isinstance(state, dict):
        tu = state.get("token_usage")
        if tu:
            out["tokens_in"] = tu.get("tokens_in", 0)
            out["tokens_out"] = tu.get("tokens_out", 0)
            out["total_tokens"] = tu.get("total_tokens", out["tokens_in"] + out["tokens_out"])
            out["phase_tokens"] = tu.get("phase_breakdown", {})
        elif state.get("run_context") and hasattr(state["run_context"], "budget"):
            b = state["run_context"].budget
            out["tokens_in"] = getattr(b, "tokens_in", 0)
            out["tokens_out"] = getattr(b, "tokens_out", 0)
            out["total_tokens"] = out["tokens_in"] + out["tokens_out"]

    output_file.write_text(json.dumps(out, indent=2), encoding="utf-8")


# ============================================================================
# HARNESS MAIN DRIVER & CLI
# ============================================================================

def run_benchmark(
    workspace: Path,
    ticket_filter: str = "all",
    mode: str = "baseline",
    dry_run: bool = False,
    allow_live_reset: bool = False,
    report_path: Optional[Path] = None,
    max_consecutive_errors: int = 5
) -> BenchmarkReport:
    all_specs = make_golden_50_specs()
    tools = detect_toolchain()
    logger.info(f"Detected Toolchain: {tools}")

    # Parse ticket filter
    selected_specs: List[TicketSpec] = []
    if ticket_filter == "all":
        selected_specs = all_specs
    else:
        # Support ranges like 1-5 or comma-separated lists like GOLDEN-01,GOLDEN-02
        parts = [p.strip() for p in ticket_filter.split(",") if p.strip()]
        for p in parts:
            if "-" in p and not p.startswith("GOLDEN"):
                start_str, end_str = p.split("-", 1)
                try:
                    s_idx, e_idx = int(start_str), int(end_str)
                    selected_specs.extend([s for s in all_specs if s_idx <= s.number <= e_idx])
                except ValueError:
                    pass
            else:
                # Match by ticket_id or number
                matched = [s for s in all_specs if s.ticket_id.upper() == p.upper() or str(s.number) == p]
                selected_specs.extend(matched)

    # De-duplicate while preserving order
    seen_ids: Set[str] = set()
    specs_to_run: List[TicketSpec] = []
    for s in selected_specs:
        if s.ticket_id not in seen_ids:
            seen_ids.add(s.ticket_id)
            specs_to_run.append(s)

    logger.info(f"Running benchmark with {len(specs_to_run)} tickets (Mode: {mode}, Dry-Run: {dry_run})")

    results: List[TicketExecutionResult] = []
    circuit_breaker_failures = 0

    for spec in specs_to_run:
        try:
            res = execute_single_ticket(
                spec=spec,
                workspace=workspace,
                mode=mode,
                dry_run=dry_run,
                allow_live_reset=allow_live_reset,
                available_tools=tools
            )
        except Exception as run_err:
            logger.error(f"[{spec.ticket_id}] UNHANDLED ERROR in ticket runner: {run_err}", exc_info=True)
            res = TicketExecutionResult(
                ticket_id=spec.ticket_id,
                number=spec.number,
                category=spec.category,
                difficulty=spec.difficulty,
                title=spec.title,
                status="PIPELINE_ERROR",
                wall_clock_s=0.0,
                tokens_in=0,
                tokens_out=0,
                total_tokens=0,
                tier0_passed=False,
                tier0_reason=f"Runner exception: {run_err}",
                error_message=str(run_err)
            )
        results.append(res)

        if res.status == "PIPELINE_ERROR":
            circuit_breaker_failures += 1
            if circuit_breaker_failures >= max_consecutive_errors and not dry_run:
                logger.error(f"CIRCUIT BREAKER: {max_consecutive_errors} consecutive pipeline errors. Halting benchmark run.")
                break
        else:
            circuit_breaker_failures = 0

    # Compile Summary
    total_evaluated = len([r for r in results if r.status != "SKIPPED_PREREQ"])
    resolved_count = len([r for r in results if r.status == "RESOLVED"])
    skipped_count = len([r for r in results if r.status == "SKIPPED_PREREQ"])
    failed_count = len([r for r in results if r.status in ("FAILED_ASSERTION", "PIPELINE_ERROR", "TIMEOUT")])

    res_rate = (resolved_count / max(1, total_evaluated)) * 100.0 if total_evaluated > 0 else 0.0
    total_tokens = sum(r.total_tokens for r in results)
    avg_tokens = total_tokens / max(1, len(results))
    avg_resolved_tokens = (sum(r.total_tokens for r in results if r.status == "RESOLVED") / max(1, resolved_count)) if resolved_count > 0 else 0.0
    avg_latency = sum(r.wall_clock_s for r in results) / max(1, len(results))

    # Category breakdown
    categories: Dict[str, Dict[str, Any]] = {}
    for cat in ["UI", "FE", "BE", "Mix", "Cfg", "Info", "Dbg", "Neg"]:
        cat_results = [r for r in results if r.category == cat]
        cat_eval = len([r for r in cat_results if r.status != "SKIPPED_PREREQ"])
        cat_res = len([r for r in cat_results if r.status == "RESOLVED"])
        categories[cat] = {
            "total": len(cat_results),
            "evaluated": cat_eval,
            "resolved": cat_res,
            "resolution_rate_pct": (cat_res / max(1, cat_eval)) * 100.0 if cat_eval > 0 else 0.0
        }

    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        mode=mode,
        workspace_path=str(workspace),
        total_tickets=len(specs_to_run),
        evaluated=total_evaluated,
        resolved=resolved_count,
        skipped_prereq=skipped_count,
        failed=failed_count,
        resolution_rate_pct=round(res_rate, 2),
        total_tokens=total_tokens,
        avg_tokens_per_ticket=round(avg_tokens, 1),
        avg_tokens_per_resolved_ticket=round(avg_resolved_tokens, 1),
        avg_wall_clock_s=round(avg_latency, 2),
        category_summary=categories,
        tickets=[dataclasses.asdict(r) for r in results],
        metadata={
            "vector_rag": "online" if check_service_online(5433) else "degraded",
            "neo4j": "online" if check_service_online(7687) else "offline",
        },
    )

    out_file = report_path or Path("benchmark_baseline_report.json")
    out_file.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info(f"Saved benchmark report to: {out_file.resolve()}")
    logger.info(f"Resolution Rate: {report.resolution_rate_pct}% ({resolved_count}/{total_evaluated})")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="50-Ticket Golden Set Benchmark Harness")
    parser.add_argument("--workspace", type=str, default=r"workspace\CloudForge-Commerce", help="Target fixture workspace path (default: workspace/CloudForge-Commerce)")
    parser.add_argument("--source-repo", type=str, default=r"workspace\CloudForge-Commerce\store-ui", help="Source repository to clone fixture from if missing")
    parser.add_argument("--auto-fixture", action="store_true", default=True, help="Auto-create dedicated fixture clone if missing")
    parser.add_argument("--allow-live-reset", action="store_true", default=False, help="Explicitly allow destructive reset on live or dirty workspace")
    parser.add_argument("--tickets", type=str, default="all", help="Ticket filter (e.g. 'all', '1-5', 'GOLDEN-01')")
    parser.add_argument("--mode", type=str, choices=["baseline", "enforce"], default="baseline", help="Budget mode")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate existing workspace without running pipeline")
    parser.add_argument("--report-path", type=str, default="benchmark_baseline_report.json", help="Output JSON report path")
    parser.add_argument("--max-consecutive-errors", type=int, default=5, help="Circuit breaker threshold for consecutive pipeline errors")

    # Worker-specific arguments
    parser.add_argument("--worker-ticket", type=str, default=None, help="Execute worker for single ticket ID")
    parser.add_argument("--token-limit", type=int, default=500000, help="Token limit for worker")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON path for worker")

    args = parser.parse_args()

    # Worker dispatch branch
    if args.worker_ticket:
        if not args.output_file:
            sys.exit("Error: --output-file is required when --worker-ticket is specified")
        run_worker_ticket(
            ticket_id=args.worker_ticket,
            workspace=Path(args.workspace).resolve(),
            output_file=Path(args.output_file).resolve(),
            token_limit=args.token_limit
        )
        sys.exit(0)

    ws_path = Path(args.workspace).resolve()
    rep_path = Path(args.report_path).resolve()

    # If not dry-run and workspace doesn't exist, handle auto-fixture
    if not args.dry_run and not ws_path.exists() and args.auto_fixture:
        src_path = Path(args.source_repo).resolve()
        if src_path.exists():
            ensure_fixture_workspace(src_path, ws_path)
        else:
            logger.warning(f"Source repository '{src_path}' not found; cannot auto-create fixture.")

    run_benchmark(
        workspace=ws_path,
        ticket_filter=args.tickets,
        mode=args.mode,
        dry_run=args.dry_run,
        allow_live_reset=args.allow_live_reset,
        report_path=rep_path,
        max_consecutive_errors=args.max_consecutive_errors
    )
