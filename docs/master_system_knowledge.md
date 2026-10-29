# NemoForge Master System Knowledge & Architecture Specification

> **Authoritative Technical Specification**: System Internals, ReAct Agent Lifecycle, Tree-sitter Symbol Engine, Compiler Gate Sandbox, and Nebius Token Factory Integration.

---

## 1. Overview
NemoForge is an autonomous software engineering agent combining deterministic static analysis (Tree-sitter AST symbol graphs stored in embedded SQLite WAL) with deep LLM reasoning (NVIDIA Nemotron 3 Super 120B MoE on Nebius Token Factory) and a closed-loop multi-zone compiler verification sandbox.

## 2. Architectural Pillars
- **Zero-Database Indexing**: Embedded SQLite WAL database (`.nemoforge/index.db`) indexes 4,600+ multi-language files in <1 second without external vector DBs.
- **Inference Engine**: Native OpenAI-compatible client connecting to Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1/`).
- **Surgical Patch Engine**: Fail-closed 3-line search/replace anchors prevent file drift and accidental line deletion.
- **Polyglot Compiler Gate**: Automated sandboxed execution of `mvn compile`, `gradle build`, and `npm test` with diagnostic error feedback loops.
- **Developer Interface**: React 18 + Vite frontend with live ReAct DAG visualizer (Flow tab) and native Git diff inspector with virtual buffer controls (Changes tab).
