"""
Architecture Model Setup — Leverages the Repository Brain for configuration.

Uses the existing brain infrastructure (WorkspaceIntelligence + Repository Brain)
to auto-generate a starter architecture_model.yaml. The brain already discovers
services, extracts business domains, intent terms, and LLM-enriched descriptions
— all of which produce a FAR richer result than naive directory scanning.

Usage:
    python -m ticket_to_code.skills.project-architecture.setup_architecture /path/to/workspace

    # Or from Python:
    from ticket_to_code.skills.setup_architecture import generate_architecture_from_brain
    config = generate_architecture_from_brain("/path/to/workspace")

How it works:
    1. Calls WorkspaceIntelligence.analyze() → gets ServiceInfo list
    2. Loads the Repository Brain (generated_directory_brain.json) → gets business_domain,
       intent_terms, core_responsibilities per directory
    3. Groups brain entries by service (via path_prefix matching)
    4. Generates a starter architecture_model.yaml with:
       - Services from WorkspaceIntelligence (name, path, language, framework)
       - Capabilities derived from brain entries (business_domain + intent_terms)
       - Keywords from the brain's intent_terms (already includes symbol names, selectors, etc.)
    5. All capabilities start as DISCOVERED/0.5 confidence (soft authority)
       → user promotes to DECLARATIVE/1.0 after review
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


def _load_brain_entries(workspace_path: str) -> List[dict]:
    """Load the repository brain's directory entries.

    Search order:
        1. brain/knowledge/generated_directory_brain.json (primary)
        2. .agents/repository_brain.json (legacy)
    """
    ws = Path(workspace_path)
    for candidate in [
        ws / "brain" / "knowledge" / "generated_directory_brain.json",
        ws / ".agents" / "repository_brain.json",
    ]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
    return []


def _discover_services_via_brain(workspace_path: str) -> List[dict]:
    """Use WorkspaceIntelligence to discover services (rich metadata).

    Falls back to brain directory entries if WorkspaceIntelligence is unavailable.
    """
    try:
        from ticket_to_code.agents.workspace_intelligence import WorkspaceIntelligenceAgent
        agent = WorkspaceIntelligenceAgent(workspace_path)
        knowledge = agent.analyze()
        return [
            {
                "name": s.name,
                "path_prefix": s.path + "/" if not s.path.endswith("/") else s.path,
                "language": s.language,
                "framework": s.framework,
            }
            for s in knowledge.services
        ]
    except Exception:
        # Fallback: derive services from brain directory entries
        brain_entries = _load_brain_entries(workspace_path)
        if not brain_entries:
            return []

        # Group by top-level directory
        services: Dict[str, dict] = {}
        for entry in brain_entries:
            dir_path = entry.get("directory_path", "")
            parts = dir_path.split("/")
            if not parts or parts[0] == ".":
                continue
            svc_name = parts[0]
            if svc_name not in services:
                services[svc_name] = {
                    "name": svc_name,
                    "path_prefix": svc_name + "/",
                    "language": "",
                    "framework": "",
                }
        return list(services.values())


def _group_brain_by_service(
    brain_entries: List[dict],
    services: List[dict],
) -> Dict[str, List[dict]]:
    """Group brain directory entries by their owning service."""
    grouped: Dict[str, List[dict]] = {s["name"]: [] for s in services}

    for entry in brain_entries:
        dir_path = entry.get("directory_path", "")
        for svc in services:
            prefix = svc["path_prefix"]
            if dir_path.startswith(prefix) or dir_path == prefix.rstrip("/"):
                grouped[svc["name"]].append(entry)
                break

    return grouped


def _extract_capabilities(
    service_name: str,
    brain_entries: List[dict],
) -> List[dict]:
    """Derive capabilities from brain entries for a service.

    Each unique (business_domain, technical_role) pair becomes a capability.
    Keywords come from the brain's intent_terms (which include real symbol names,
    component selectors, and extracted identifiers — far richer than naive guessing).
    """
    cap_map: Dict[str, dict] = {}

    for entry in brain_entries:
        domain = entry.get("business_domain", "").strip()
        role = entry.get("technical_role", "").strip()
        intent_terms = entry.get("intent_terms", [])
        responsibilities = entry.get("core_responsibilities", "")

        if not domain or domain.lower() in ("source module", "unknown"):
            continue

        # Capability key: domain + role
        cap_key = f"{domain}:{role}" if role and role.lower() != "source module" else domain
        cap_name = cap_key.lower().replace(" ", "-").replace(":", "-").replace("/", "-")
        # Sanitize cap_name
        cap_name = "".join(c for c in cap_name if c.isalnum() or c == "-")[:40]

        if cap_name not in cap_map:
            cap_map[cap_name] = {
                "name": cap_name,
                "keywords": [],
                "domain_hint": domain,
                "role_hint": role,
                "responsibilities": [],
            }

        cap = cap_map[cap_name]

        # Add intent terms as keywords (deduplicated, capped)
        for term in intent_terms:
            term_lower = term.lower().strip()
            if term_lower and len(term_lower) > 2 and term_lower not in cap["keywords"]:
                cap["keywords"].append(term_lower)

        # Add responsibilities
        if responsibilities and isinstance(responsibilities, str):
            for r in responsibilities.split(";"):
                r = r.strip()
                if r and r not in cap["responsibilities"]:
                    cap["responsibilities"].append(r)

    # Limit keywords per capability
    result = []
    for cap in cap_map.values():
        cap["keywords"] = cap["keywords"][:12]  # Cap at 12 keywords
        cap["responsibilities"] = cap["responsibilities"][:3]
        result.append(cap)

    return result


def generate_architecture_from_brain(workspace_path: str) -> str:
    """Generate a starter architecture_model.yaml using the Repository Brain.

    This produces MUCH richer output than naive directory scanning because:
    - Services come from WorkspaceIntelligence (language, framework, build system)
    - Capabilities come from brain entries (business_domain, intent_terms)
    - Keywords include real symbol names, selectors, and extracted identifiers
    - Business domains are LLM-enriched (if brain was enriched)
    """
    services = _discover_services_via_brain(workspace_path)
    brain_entries = _load_brain_entries(workspace_path)
    grouped = _group_brain_by_service(brain_entries, services)

    lines = [
        "# Architecture Model — Domain Ownership Configuration",
        f"# Auto-generated from Repository Brain for: {Path(workspace_path).name}",
        "#",
        "# This starter config was generated using the brain's deep understanding",
        "# of your codebase. Each capability was derived from the brain's",
        "# business_domain + intent_terms — NOT from naive directory scanning.",
        "#",
        "# REVIEW INSTRUCTIONS:",
        "#   1. Check each capability's primary_owners — is this the right service?",
        "#   2. Add secondary_participants where other services legitimately react",
        "#   3. Add reference_only for services that can READ but must NOT modify",
        "#   4. Change provenance to DECLARATIVE and confidence to 1.0 after review",
        "#   5. Remove any TODO capabilities that aren't real business features",
        "#",
        "# Provenance: DISCOVERED (soft warnings) → DECLARATIVE (hard blocks)",
        "# Confidence: 0.5 (investigate) → 1.0 (hard reject violations)",
        "",
        "services:",
    ]

    for svc in services:
        name = svc["name"]
        lang = svc.get("language", "")
        fw = svc.get("framework", "")
        tech_hint = f" ({fw}/{lang})" if fw else f" ({lang})" if lang else ""

        lines.extend([
            f"  # ── {name} {'─' * max(1, 60 - len(name))}",
            f"  - name: {name}",
            f"    path_prefix: {svc['path_prefix']}",
        ])

        # Domain description from brain
        svc_entries = grouped.get(name, [])
        domains = set()
        for e in svc_entries:
            d = e.get("business_domain", "").strip()
            if d and d.lower() not in ("source module", "unknown"):
                domains.add(d)
        domain_desc = "; ".join(sorted(domains)[:3]) if domains else f"{name}{tech_hint}"
        lines.append(f"    domain: \"{domain_desc}\"")

        # Capabilities from brain
        capabilities = _extract_capabilities(name, svc_entries)
        if capabilities:
            lines.append("    capabilities:")
            for cap in capabilities[:6]:  # Max 6 capabilities per service
                kw_str = ", ".join(cap["keywords"][:8])
                resp_comment = ""
                if cap.get("responsibilities"):
                    resp_comment = f"  # {cap['responsibilities'][0][:60]}"
                lines.extend([
                    f"      - name: {cap['name']}{resp_comment}",
                    f"        keywords: [{kw_str}]",
                    f"        primary_owners: [{name}]",
                    f"        secondary_participants: []  # TODO: services that react to this",
                    f"        reference_only: []  # TODO: services that can READ but NOT modify",
                    f"        provenance: DISCOVERED  # → Change to DECLARATIVE after review",
                    f"        confidence: 0.5  # → Change to 1.0 after review",
                ])
        else:
            lines.extend([
                "    capabilities:",
                f"      - name: {name}-core",
                f"        keywords: []  # TODO: add keywords from your tickets",
                f"        primary_owners: [{name}]",
                f"        secondary_participants: []",
                f"        reference_only: []",
                f"        provenance: DISCOVERED",
                f"        confidence: 0.5",
            ])

        lines.append("")

    if not services:
        lines.extend([
            "  # No services discovered by the brain.",
            "  # Run: python -m ticket_to_code.brain.generate_repository_brain \\",
            "  #          --workspace /path/to/workspace --ensure",
            "  # Then re-run this setup script.",
        ])

    return "\n".join(lines) + "\n"


def main():
    """CLI entry point: generate architecture config from brain."""
    if len(sys.argv) < 2:
        print("Usage: python -m ticket_to_code.skills.project-architecture.setup_architecture /path/to/workspace")
        print()
        print("Generates a starter architecture_model.yaml using the Repository Brain.")
        print("The brain must be generated first (it usually is, on first pipeline run).")
        print()
        print("If no brain exists, run this first:")
        print("  python -m ticket_to_code.brain.generate_repository_brain --workspace /path/to/workspace --ensure")
        sys.exit(1)

    workspace = sys.argv[1]
    if not os.path.isdir(workspace):
        print(f"Error: {workspace} is not a directory")
        sys.exit(1)

    # Ensure brain exists before generating architecture
    brain_path = Path(workspace) / "brain" / "knowledge" / "generated_directory_brain.json"
    legacy_path = Path(workspace) / ".agents" / "repository_brain.json"
    if not brain_path.exists() and not legacy_path.exists():
        print("⚠️  No Repository Brain found. Generating brain first...")
        try:
            from ticket_to_code.brain.generate_repository_brain import ensure_repository_brain
            result = ensure_repository_brain(workspace, enrich=True)
            print(f"   Brain: {result.get('status', 'unknown')}")
        except Exception as e:
            print(f"   Brain generation failed: {e}")
            print("   Continuing with WorkspaceIntelligence service discovery only...")

    config = generate_architecture_from_brain(workspace)

    # Write to workspace
    out_dir = Path(workspace) / "brain" / "knowledge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "architecture_model.yaml"

    if out_path.exists():
        print(f"⚠️  {out_path} already exists. Printing to stdout instead.")
        print("─" * 70)
        print(config)
    else:
        out_path.write_text(config, encoding="utf-8")
        print(f"✓ Generated {out_path}")
        print()
        print("NEXT STEPS:")
        print("  1. Open the generated file and review each capability")
        print("  2. Add secondary_participants and reference_only services")
        print("  3. Change provenance to DECLARATIVE and confidence to 1.0")
        print("  4. The pipeline will auto-load this on next run")


if __name__ == "__main__":
    main()
