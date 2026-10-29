# RAG Knowledge Base - Dual RAG Architecture

This folder contains knowledge documents organized for **Dual RAG** - separate knowledge sources for test generation vs code generation.

## Folder Structure

```
RAG/
├── README.md                                # This file
├── AI_INTEGRATION_USAGE_GUIDE.md           # General usage guide
├── ARCHITECTURE_DOCUMENTATION_INDEX.md      # Index of all docs
│
├── product_behavior/                        # For TEST generation (how it SHOULD work)
│   ├── authentication_flows.md             # Auth behavior patterns
│   ├── BEHAVIOR_PATTERN_DOCUMENTATION_INDEX.md  # Behavior index
│   ├── BUSINESS_RULES_AND_BEHAVIORS.md     # Core business rules
│   └── VALIDATION_RULES_REFERENCE.md       # Validation rules
│
└── architecture/                            # For CODE generation (how to BUILD it)
    ├── CONTENTBRIDGE_ARCHITECTURE_GUIDE.md      # Main architecture doc
    ├── CONTENTBRIDGE_ARCHITECTURE_GUIDE.json    # Structured architecture
    ├── CONTENTBRIDGE_ARCHITECTURE_VISUAL_REFERENCE.md  # Visual diagrams
    ├── ARCHITECTURE_PATTERNS.md             # Design patterns
    ├── CODE_PATTERNS_REFERENCE.md           # Code implementation patterns
    └── service_layer_pattern.md             # Service layer example
```

---

## Product Behavior (product_behavior/)

**Purpose**: Knowledge for generating TEST CASES

**Contains**:
- ✅ How features SHOULD behave
- ✅ Business rules and validation
- ✅ Expected error messages
- ✅ Test scenarios to cover
- ✅ Edge cases and boundary conditions

**Used by**: `query_focus="test_behavior"`

**Example Documents to Add**:
- `discount_validation_behavior.md` - How discount codes work
- `checkout_flow_behavior.md` - Checkout process steps
- `error_handling_behavior.md` - Standard error responses
- `business_rules.md` - All business validations

---

## Architecture (architecture/)

**Purpose**: Knowledge for generating IMPLEMENTATION CODE

**Contains**:
- ✅ Code structure and patterns
- ✅ Class hierarchies
- ✅ Interface implementations
- ✅ Dependency injection patterns
- ✅ Service layer examples

**Used by**: `query_focus="architecture"`

**Example Documents to Add**:
- `validation_pattern.md` - How to validate inputs
- `controller_pattern.md` - API controller structure
- `repository_pattern.md` - Data access patterns
- `logging_pattern.md` - How to log operations

---

## Why Dual RAG?

### Traditional (Single RAG)
```
RAG → Everything → LLM → Tests + Code
     ↓
Tests know code structure
Code knows test expectations
     ↓
COUPLED & BIASED
```

### Dual RAG (Independent)
```
Product Behavior RAG → LLM → Tests (behavior-driven)
        ↓
    Independent!
        ↓
Architecture RAG → LLM → Code (pattern-driven)
        ↓
    Tests + Code must match behavior independently
```

---

## How It Works

### Test Generation (TDD - First)
```python
test_context = rag.multi_stage_retrieval(
    requirements=requirements,
    query_focus="test_behavior",  # ← Only product_behavior/
    document_types=["tests", "documentation", "business_rules"]
)

# LLM generates tests using ONLY behavior knowledge
# Tests = "What should happen?"
```

### Code Generation (Second)
```python
code_context = rag.multi_stage_retrieval(
    requirements=requirements,
    query_focus="architecture",  # ← Only architecture/
    document_types=["source_code", "interfaces", "patterns"]
)

# LLM generates code using ONLY architecture knowledge
# Code = "How to build it?"
```

---

## Adding New Documents

### For Product Behavior (Tests)
1. Create `.md` file in `product_behavior/`
2. Document HOW it should behave
3. Include business rules
4. List expected errors
5. Show test scenarios

**Template**:
```markdown
# Feature Name Behavior

## Expected Behavior
- Happy path scenario
- Error scenarios
- Edge cases

## Business Rules
1. Rule 1
2. Rule 2

## Validation Rules
| Field | Rule | Error Message |
|-------|------|---------------|

## Test Scenarios
[Fact] void Scenario_Condition_ExpectedResult()
```

### For Architecture (Code)
1. Create `.md` file in `architecture/`
2. Document HOW to implement
3. Include code examples
4. Show patterns

**Template**:
```markdown
# Pattern Name

## Implementation
```csharp
// Code example
```

## Usage
// How to use

## Integration
// How it fits with other code
```

---

## Indexing

After adding documents, re-index:

```bash
docker exec -it aviator-app-plugin python setup_rag.py \
    --project-path /projects/contentbridge \
    --rag-path /app/src/ticket_to_code/RAG \
    --project-name ContentBridge
```

This creates embeddings and stores in pgvector with metadata:
- `document_type`: "product_behavior" or "architecture"
- `file_path`: Full path
- `folder`: Which folder it came from

---

## Verification

Check what's indexed:

```python
from ticket_to_code.agents.rag_engine import CodebaseRAGEngine

rag = CodebaseRAGEngine()

# Test retrieval
test_chunks = rag._get_test_behavior_context(requirements)
print(f"Product behavior chunks: {len(test_chunks)}")

code_chunks = rag._get_architecture_context(requirements)
print(f"Architecture chunks: {len(code_chunks)}")
```

---

## Guidelines

### ✅ DO:
- Keep behavior and architecture separate
- Document REAL product behavior (not imagined)
- Include actual code examples in architecture
- Update when product changes

### ❌ DON'T:
- Mix behavior and implementation
- Add test code to architecture/
- Add implementation details to product_behavior/
- Copy code without context

---

**This separation ensures AI-generated tests and code are independently grounded in real knowledge!**
