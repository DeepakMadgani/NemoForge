# Production Benchmark Guide

## Goal
Use empirical evidence to evaluate production readiness of the autonomous ticket solver.

## What production-ready means here
The system is considered production-ready for autonomous ticket handling only when:
- It reliably completes runs without environment/parser crashes.
- It solves representative tickets across categories with stable quality.
- It produces explainable reports with actionable failure taxonomy.
- A/B behavior is predictable (experience on/off does not cause regressions).

This benchmark is the safety gate before scaling ticket volume.

## What this benchmark measures
- Localization accuracy (when expected file ground truth is provided)
- Owner identification rate (when expected owner files are provided)
- Patch production and terminal outcomes
- Verification pass rate
- Failure taxonomy and stage bottlenecks
- Token usage (prompt, completion, total)
- Human acceptance (if labeled in the dataset)

## Dataset format
Use [benchmark_dataset_template.json](benchmark_dataset_template.json).

Required fields per ticket:
- ticket_id
- title
- description
- category
- complexity
- acceptance_criteria

Strongly recommended for high-quality metrics:
- expected_changed_files
- expected_owner_files
- repo_scope
- technology (explicitly set per ticket)
- human_acceptance

Supported technology values:
- java
- java-maven
- java-gradle
- dotnet (or .net)
- csharp (or c#)
- nodejs
- angular

Notes:
- Avoid python for contract benchmark tickets in this runner path, because runtime contract execution currently supports the values above.
- If technology is omitted, the runner applies best-effort inference from category/labels/repo_scope, but explicit values are safer for large datasets.

## Run single benchmark
```powershell
"C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample/run_production_readiness_benchmark.bat" `
  --workspace "C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample" `
  --tickets "C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample/src/ticket_to_code/benchmarks/benchmark_dataset_template.json" `
  --report-dir "C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample/benchmark_reports" `
  --execution-mode contract
```

Smoke shortcut:
```powershell
"C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample/run_production_readiness_smoke.bat"
```

## Run A/B benchmark (experience on vs off)
```powershell
"C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample/run_production_readiness_benchmark.bat" `
  --workspace "C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample" `
  --tickets "C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample/src/ticket_to_code/benchmarks/benchmark_dataset_template.json" `
  --report-dir "C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample/benchmark_reports" `
  --execution-mode contract `
  --lightweight `
  --ab-experience
```

`--lightweight` forces `max_fix_attempts=1` for each ticket run. Use it for faster, more deterministic pilot scoring before full-depth retries.

Smoke A/B shortcut:
```powershell
"C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample/run_production_readiness_ab_smoke.bat"
```

## Output files
Single run:
- production_readiness_report.json
- production_readiness_report.md

A/B run:
- production_readiness_ab_report.json
- production_readiness_ab_report.md

## Core KPI definitions
- success_rate_effect_achieved:
  - solved with verified patch for code-change tickets
  - no_action_required for evidence-backed no-op or non-code tickets
- localization_accuracy_mean:
  - Mean overlap of expected_changed_files with changed_files
- owner_identification_rate:
  - Expected owner files marked REQUIRED by planner decisions
- top_bottlenecks:
  - Highest failure stage counts from taxonomy

## Recommended readiness thresholds
- localization_accuracy_mean >= 0.90
- verification_pass_rate >= 0.90
- success_rate_effect_achieved >= 0.85
- human_acceptance_rate >= 0.85
- false positive solved states = 0

## Go-live gate before 100-ticket run
- Run a 10-ticket pilot (2 per category) first.
- Require 0 environment failures and 0 unsupported-technology failures.
- Require deterministic ticket metadata: repo_scope, technology, expected_changed_files, expected_owner_files.
- Only then run the full 100-ticket benchmark.

## Next-step optimization rule
Only change components that rank highest in top_bottlenecks and show measurable before/after gains.
