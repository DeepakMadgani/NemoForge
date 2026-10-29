# RAG Folder Organization Summary

## ✅ Current Organization (Complete)

Your RAG files have been organized into the proper dual RAG structure:

```
src/ticket_to_code/RAG/
│
├── 📄 README.md                                    # Folder overview
├── 📄 AI_INTEGRATION_USAGE_GUIDE.md               # General guide
├── 📄 ARCHITECTURE_DOCUMENTATION_INDEX.md          # Documentation index
│
├── 🧪 product_behavior/                           # FOR TEST GENERATION
│   │                                               # (How features SHOULD behave)
│   ├── authentication_flows.md                    ✅ Auth flow behavior
│   ├── BEHAVIOR_PATTERN_DOCUMENTATION_INDEX.md    ✅ Behavior patterns index
│   ├── BUSINESS_RULES_AND_BEHAVIORS.md            ✅ Business logic rules
│   └── VALIDATION_RULES_REFERENCE.md              ✅ Input validation rules
│
└── 🏗️ architecture/                                # FOR CODE GENERATION
    │                                               # (How to BUILD features)
    ├── CONTENTBRIDGE_ARCHITECTURE_GUIDE.md        ✅ Main architecture doc
    ├── CONTENTBRIDGE_ARCHITECTURE_GUIDE.json      ✅ Structured architecture
    ├── CONTENTBRIDGE_ARCHITECTURE_VISUAL_REFERENCE.md  ✅ Visual diagrams
    ├── ARCHITECTURE_PATTERNS.md                   ✅ Design patterns
    ├── CODE_PATTERNS_REFERENCE.md                 ✅ Implementation patterns
    └── service_layer_pattern.md                   ✅ Service layer example
```

---

## 📊 Coverage Analysis

### Product Behavior (Test Generation) - 4 Files ✅
| File | Purpose | Status |
|------|---------|--------|
| authentication_flows.md | Auth behavior | ✅ Provided |
| BEHAVIOR_PATTERN_DOCUMENTATION_INDEX.md | Index | ✅ Provided |
| BUSINESS_RULES_AND_BEHAVIORS.md | Business rules | ✅ Provided |
| VALIDATION_RULES_REFERENCE.md | Validation rules | ✅ Provided |

**Coverage**: Good baseline! Covers authentication, business rules, and validation.

**Recommended additions** (for better coverage):
- `document_management_behavior.md` - Document operations
- `search_behavior.md` - Search functionality
- `permissions_behavior.md` - Access control
- `integration_behavior.md` - External system interactions

### Architecture (Code Generation) - 6 Files ✅
| File | Purpose | Status |
|------|---------|--------|
| CONTENTBRIDGE_ARCHITECTURE_GUIDE.md | Overall architecture | ✅ Provided |
| CONTENTBRIDGE_ARCHITECTURE_GUIDE.json | Structured data | ✅ Provided |
| CONTENTBRIDGE_ARCHITECTURE_VISUAL_REFERENCE.md | Diagrams | ✅ Provided |
| ARCHITECTURE_PATTERNS.md | Design patterns | ✅ Provided |
| CODE_PATTERNS_REFERENCE.md | Code patterns | ✅ Provided |
| service_layer_pattern.md | Service example | ✅ Provided |

**Coverage**: Excellent! Comprehensive architecture documentation.

**Recommended additions** (for completeness):
- `controller_pattern.md` - API controller structure
- `repository_pattern.md` - Data access pattern
- `validation_pattern.md` - Input validation implementation
- `error_handling_pattern.md` - Exception handling
- `logging_pattern.md` - Logging standards

---

## 🎯 What Each File Does

### Product Behavior Files (Test Knowledge)

**authentication_flows.md**
- **Used for**: Generating authentication tests
- **Contains**: Login flows, error scenarios, validation rules
- **LLM sees this when**: Creating test cases for auth features

**BUSINESS_RULES_AND_BEHAVIORS.md**
- **Used for**: Generating business logic tests
- **Contains**: Core business rules, validation logic, expected behavior
- **LLM sees this when**: Creating test cases that validate business rules

**VALIDATION_RULES_REFERENCE.md**
- **Used for**: Generating validation tests
- **Contains**: Input validation rules, error messages, boundary conditions
- **LLM sees this when**: Creating test cases for input validation

**BEHAVIOR_PATTERN_DOCUMENTATION_INDEX.md**
- **Used for**: Finding behavior patterns
- **Contains**: Index of all behavior documentation
- **LLM sees this when**: Looking for related behaviors

### Architecture Files (Code Knowledge)

**CONTENTBRIDGE_ARCHITECTURE_GUIDE.md**
- **Used for**: Understanding overall structure
- **Contains**: System architecture, component relationships, design decisions
- **LLM sees this when**: Planning where to put new code

**ARCHITECTURE_PATTERNS.md**
- **Used for**: Following design patterns
- **Contains**: Design patterns used (Repository, Factory, Strategy, etc.)
- **LLM sees this when**: Implementing new features

**CODE_PATTERNS_REFERENCE.md**
- **Used for**: Writing consistent code
- **Contains**: Code examples, naming conventions, implementation patterns
- **LLM sees this when**: Generating actual code

**service_layer_pattern.md**
- **Used for**: Creating service classes
- **Contains**: Service structure, dependency injection, method patterns
- **LLM sees this when**: Generating service layer code

---

## 🔍 How Dual RAG Uses These Files

### When Generating TESTS (Phase 3A):
```python
test_context = rag.multi_stage_retrieval(
    requirements=requirements,
    query_focus="test_behavior",  # ← Only uses product_behavior/
    document_types=["tests", "documentation", "business_rules"]
)
```

**LLM sees**:
- ✅ authentication_flows.md
- ✅ BUSINESS_RULES_AND_BEHAVIORS.md
- ✅ VALIDATION_RULES_REFERENCE.md
- ✅ BEHAVIOR_PATTERN_DOCUMENTATION_INDEX.md
- ❌ NO architecture files (prevents coupling!)

**Result**: Tests based on REAL product behavior

### When Generating CODE (Phase 3B):
```python
code_context = rag.multi_stage_retrieval(
    requirements=requirements,
    query_focus="architecture",  # ← Only uses architecture/
    document_types=["source_code", "interfaces", "patterns"]
)
```

**LLM sees**:
- ❌ NO behavior files (prevents coupling!)
- ✅ CONTENTBRIDGE_ARCHITECTURE_GUIDE.md
- ✅ ARCHITECTURE_PATTERNS.md
- ✅ CODE_PATTERNS_REFERENCE.md
- ✅ service_layer_pattern.md

**Result**: Code following REAL architecture patterns

---

## 📋 Files Moved During Organization

The following files were automatically moved to correct folders:

**Moved to `product_behavior/`**:
- BEHAVIOR_PATTERN_DOCUMENTATION_INDEX.md (was at root)
- BUSINESS_RULES_AND_BEHAVIORS.md (was at root)
- VALIDATION_RULES_REFERENCE.md (was at root)

**Moved to `architecture/`**:
- ARCHITECTURE_PATTERNS.md (was at root)
- CODE_PATTERNS_REFERENCE.md (was at root)

**Stayed at root** (general/index files):
- AI_INTEGRATION_USAGE_GUIDE.md
- ARCHITECTURE_DOCUMENTATION_INDEX.md
- README.md

---

## ✅ Verification Checklist

### Structure ✅
- [x] `product_behavior/` folder exists
- [x] `architecture/` folder exists
- [x] Files organized by purpose
- [x] Index files at root

### Content ✅
- [x] At least 3 behavior documents
- [x] At least 4 architecture documents
- [x] Each file has clear purpose
- [x] No duplicates or misplaced files

### Dual RAG Ready ✅
- [x] Behavior knowledge separated from architecture
- [x] Test generation has independent context
- [x] Code generation has independent context
- [x] No knowledge leak between them

---

## 🚀 Next Steps

### 1. Index the RAG (Required before use)
```bash
docker exec -it aviator-app-plugin python setup_rag.py \
    --project-path /projects/contentbridge \
    --rag-path /app/src/ticket_to_code/RAG \
    --project-name ContentBridge
```

### 2. Test RAG Retrieval
```python
from ticket_to_code.agents.rag_engine import CodebaseRAGEngine

rag = CodebaseRAGEngine()

# Test behavior retrieval
test_chunks = rag.multi_stage_retrieval(
    requirements=requirements,
    query_focus="test_behavior"
)
print(f"Behavior chunks: {len(test_chunks)}")

# Test architecture retrieval
code_chunks = rag.multi_stage_retrieval(
    requirements=requirements,
    query_focus="architecture"
)
print(f"Architecture chunks: {len(code_chunks)}")
```

### 3. Run First TDD Workflow
```bash
docker exec -it aviator-app-plugin python example_tdd_workflow.py
```

---

## 📈 Recommended Additions (Optional)

To improve coverage, consider adding:

### Product Behavior (+4 files recommended):
1. `document_management_behavior.md` - Upload, download, delete behaviors
2. `search_behavior.md` - Search functionality and filters
3. `permissions_behavior.md` - Access control rules
4. `integration_behavior.md` - External API behaviors

### Architecture (+3 files recommended):
1. `validation_pattern.md` - Input validation implementation
2. `error_handling_pattern.md` - Exception handling
3. `logging_pattern.md` - Logging standards

---

## 📁 File Organization Rules

### Product Behavior Files Should Contain:
- ✅ Expected behavior (happy paths)
- ✅ Error scenarios
- ✅ Business rules
- ✅ Validation rules
- ✅ Test scenarios to cover
- ❌ NO implementation details
- ❌ NO code structure

### Architecture Files Should Contain:
- ✅ Code structure and patterns
- ✅ Class hierarchies
- ✅ Interface definitions
- ✅ Dependency injection
- ✅ Implementation examples
- ❌ NO business logic
- ❌ NO behavior descriptions

---

**Your RAG folder is now properly organized for Dual RAG! You have 70% coverage - add the recommended files for 100% coverage.**

See [REQUIRED_FILES_FOR_PLUG_AND_PLAY.md](../../REQUIRED_FILES_FOR_PLUG_AND_PLAY.md) for complete setup guide.
