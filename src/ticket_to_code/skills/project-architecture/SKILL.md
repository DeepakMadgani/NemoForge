---
name: project-architecture
description: |
  Configures domain ownership boundaries for ticket-to-code.
  Teaches the system which microservice OWNS each business capability,
  so it doesn't accidentally modify files in the wrong bounded context.
  
  Use this skill when:
    - Setting up ticket-to-code for a new project/workspace
    - Adding or updating service ownership rules
    - Debugging why a file was excluded or wrongly included
    - Understanding how domain ownership affects the pipeline
---

# Project Architecture Configuration

## What This Does

The **Domain Ownership Layer** prevents the pipeline from modifying files in the wrong
microservice, even when those files are semantically relevant. It answers:

> "Is this service **AUTHORIZED** to implement this capability?"

This is fundamentally different from "Is this file **relevant**?" — a file can be
highly relevant (for context) but architecturally wrong (must not modify).

## Quick Setup

### Step 1: Generate your architecture model from the brain

The setup script leverages the **Repository Brain** (service discovery, business domains,
intent terms, LLM-enriched descriptions) to auto-generate a rich starter config:

```bash
python -m ticket_to_code.skills.project-architecture.setup_architecture /path/to/workspace
```

This will:
1. Use `WorkspaceIntelligence` to discover services (language, framework, build system)
2. Load the Repository Brain (`generated_directory_brain.json`) for business domains
3. Group brain entries by service and extract capabilities with real keywords
4. Write a starter `brain/knowledge/architecture_model.yaml`

> **Note:** If no brain exists yet, the script auto-bootstraps one via
> `ensure_repository_brain()`. If WorkspaceIntelligence is unavailable, it falls
> back to deriving services from brain directory entries.

**Alternative: Manual from template**

```bash
# Copy the blank template and fill it in manually
cp src/ticket_to_code/config/architecture_model.template.yaml \
   brain/knowledge/architecture_model.yaml
```

### Step 2: Review and promote

The auto-generated config starts with `DISCOVERED` provenance and `0.5` confidence
(soft warnings only). After reviewing:

```yaml
services:
  - name: your-service
    capabilities:
      - name: auto-detected-capability
        keywords: [brain, extracted, terms]  # ← verify these
        primary_owners: [your-service]
        secondary_participants: []   # ← add services that react
        reference_only: []           # ← add services that must NOT modify
        provenance: DECLARATIVE      # ← change from DISCOVERED
        confidence: 1.0              # ← change from 0.5
```

### Step 3: Verify it works

Run the pipeline with `TRACE_MODE=1`. Check the logs for:
```
🏗️ Architecture model loaded from architecture_model.yaml
🏗️ Domain resolution: primary=[...] authorized=[...]
⛔ REFERENCE_ONLY: some-service/SomeFile.java (verdict=WRONG_DOMAIN, ...)
```

## How It Affects the Pipeline

The architecture model creates a **5-layer defense**:

| Layer | Where | What Happens |
|-------|-------|-------------|
| 1. Semantic Verification | `semantic_verification_agent.py` | Domain-wrong files get score zeroed |
| 2. Candidate Classification | `workflow.py` → `plan_node()` | Files re-tagged as `REFERENCE` |
| 3. Planner Prompt | `planning_agent.py` | LLM told not to modify ❌ files |
| 4. Hard Gate | `plan_gating.py` → Pass 1.6 | Tasks for wrong-domain files rejected |
| 5. Post-Generation | `incremental_validator.py` | Last-resort catch for violations |

## Configuration Reference

### Provenance Levels

| Level | Meaning | Effect |
|-------|---------|--------|
| `DECLARATIVE` | You manually defined this | **Hard blocks** (if confidence ≥ 0.8) |
| `VERIFIED` | Auto-discovered, then you confirmed | **Hard blocks** |
| `DISCOVERED` | Auto-discovered from code structure | Soft warnings only |
| `LLM_INFERRED` | LLM guessed this | Soft warnings only |

### Confidence Levels

| Range | Effect |
|-------|--------|
| `0.0 - 0.7` | Soft warnings — investigate but don't block |
| `0.8 - 1.0` | Hard blocks — reject tasks in wrong service |

### Ownership Verdicts

| Verdict | Policy | Meaning |
|---------|--------|---------|
| `PRIMARY_OWNER` | `ALLOW` | This service owns the capability |
| `SECONDARY_PARTICIPANT` | `ALLOW_WITH_EVIDENCE` | Legitimate consumer/reactor |
| `RELATED` | `REFERENCE_ONLY` | Can read, should not create new logic |
| `REFERENCE_ONLY` | `REFERENCE_ONLY` | Can read for context, MUST NOT modify |
| `WRONG_DOMAIN` | `DO_NOT_MODIFY` | Hard reject — wrong bounded context |
| `UNKNOWN` | `REQUIRE_VERIFICATION` | No data — investigate first |

## Important: UNKNOWN ≠ ALLOW

If a service has no ownership data, the system does **NOT** silently allow it.
Instead it produces `REQUIRE_VERIFICATION` — which means "investigate before allowing."
This is a critical safety property.

## File Search Order

The system looks for `architecture_model.yaml` in this order:
1. `{workspace}/brain/knowledge/architecture_model.yaml` (workspace-level)
2. `{ticket-to-code}/config/architecture_model.yaml` (plugin-level fallback)

Workspace-level is preferred because it survives plugin updates.

## Troubleshooting

### "No architecture model available"
→ Run `python -m ticket_to_code.skills.project-architecture.setup_architecture /path/to/workspace`
→ Or manually copy the template from `config/architecture_model.template.yaml`.

### "No Repository Brain found"
→ The brain must exist for rich auto-generation. Run:
  `python -m ticket_to_code.brain.generate_repository_brain --workspace /path/to/workspace --ensure`

### Setup script produces empty capabilities
→ The brain has no `business_domain` or `intent_terms` for your directories.
→ Re-generate with LLM enrichment: `--ensure` (not `--no-enrich`).

### "Domain resolution: no confident match"
→ The ticket keywords don't match any capability. Add more keywords to your capabilities.

### Files being wrongly excluded
→ Check if the service is listed in `reference_only` for a capability it should own.
→ Reduce `confidence` to `< 0.8` to make it a soft warning instead of a hard block.

### Files NOT being excluded when they should
→ Ensure `provenance: DECLARATIVE` and `confidence: 1.0` for hard blocks.
→ Check the `reference_only` list includes the wrong service.
