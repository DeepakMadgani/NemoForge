# Data Storage Organization

## Directory Structure

```
src/ticket_to_code/data/
├── rag_documents/              # RAG knowledge base documents
│   ├── {project_name}/
│   │   ├── architecture.md     # Architecture docs
│   │   ├── api_reference.md    # API documentation
│   │   ├── guidelines.md       # Coding guidelines
│   │   └── examples/          # Code examples
│   └── global/
│       └── best_practices.md  # Global best practices
│
├── indexed_projects/          # Project-specific indexes
│   └── {project_id}/
│       ├── .aviator/
│       │   └── index.db       # SQLite: symbols, edges, AST
│       ├── metadata.json      # Project metadata
│       └── stats.json         # Indexing statistics
│
└── cache/                     # Temporary cache (not committed)
    ├── embeddings/           # Cached embeddings
    └── llm_responses/        # Cached LLM responses
```

## Storage Locations by Type

### 1. AST Structure (Symbols, Edges, Relationships)
**Location:** `{project_path}/.aviator/index.db` (SQLite)
**What:** Classes, methods, fields, imports, method calls, inheritance
**Size:** ~5-10 MB per 1000 files
**Used for:** Fast symbol lookup, dependency analysis

### 2. Vector Embeddings (Semantic Search)
**Location:** PostgreSQL Docker (port 5433) - table: `embeddings`
**What:** Code chunks with 768D vector embeddings
**Size:** ~50 MB per 1000 chunks
**Used for:** Semantic similarity search during RAG

### 3. Graph Relationships (Optional)
**Location:** Neo4j (cloud or local)
**What:** Visual graph of dependencies
**Used for:** Complex traversal queries

### 4. RAG Documents (Knowledge Base)
**Location:** `src/ticket_to_code/data/rag_documents/{project_name}/`
**What:** Markdown documentation, guidelines, examples
**Size:** Variable (typically 1-10 MB)
**Used for:** Additional context beyond code

### 5. Project Metadata
**Location:** `src/ticket_to_code/data/indexed_projects/{project_id}/`
**What:** Indexing stats, file hashes, configuration
**Size:** ~1-5 MB
**Used for:** Incremental indexing, project management

## Collection Names (PostgreSQL/pgvector)

- `codebase_chunks` - Main codebase embeddings
- `test_chunks` - Test file embeddings (separate for RAG isolation)
- `architectural_guidelines` - Architecture documentation
- `{project_name}_custom` - Project-specific knowledge

## Cache Policy

- **embeddings/**: LRU cache, max 1000 entries, 7-day TTL
- **llm_responses/**: LRU cache, max 500 entries, 1-day TTL
- Cache cleared on: force reindex, schema changes

## Backup & Sync

- SQLite indexes: Backed up with project
- PostgreSQL: Managed by Ticket-to-Code runtime (auto-backup)
- Neo4j: Cloud-managed or manual backup
- RAG documents: Version controlled (git)
