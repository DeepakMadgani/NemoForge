# NemoForge: Autonomous SWE Agent Core Framework

Core Python framework providing AST symbol graph extraction, multi-agent ReAct orchestration, surgical anchor patch generation, and deterministic compiler verification.

---

## Architecture Overview

```text
src/ticket_to_code/
|-- agents/                 # ReAct agent modules
|   |-- ticket_analyzer.py  # Intent extraction & functional requirements
|   |-- planning_agent.py   # Multi-file architectural planning
|   |-- rag_engine.py       # AST symbol subgraph retrieval
|   `-- code_generator.py   # Surgical 3-line anchor patch generation
|-- indexer/                # Tree-sitter AST symbol extraction & SQLite WAL
|-- runtime/                # Sandboxed compiler execution & polyglot build gates
`-- models.py               # Pydantic data schemas for tickets, plans, and patches
```

---

## Core Capabilities

1. **Tree-sitter AST Symbol Slicing**: Extracts method declarations, class hierarchies, and Spring/React dependencies into embedded SQLite WAL in under 1 second.
2. **NVIDIA Nemotron Reasoning**: Leverages Nemotron 3 Super (120B MoE) via Nebius Token Factory for deep multi-file planning and surgical code synthesis.
3. **Fail-Closed Patch Application**: Enforces 3-line search/replace anchors; aborts cleanly if context drifts rather than corrupting target files.
4. **Multi-Zone Compiler Gate**: Validates synthesized patches against live build tools (`mvn compile`, `gradle build`, `npm test`) with up to 3 automated self-repair turns.

---

## License

MIT License - see the root [LICENSE](../../LICENSE) file for details.
