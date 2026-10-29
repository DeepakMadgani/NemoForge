# 🚀 Autonomous Ticket-to-Code System

**A fully autonomous AI system that converts ValueEdge tickets into production-ready, tested, and integrated code using multi-agent architecture with Aviator ADT.**

---

## 📦 Package Structure

```
ticket_to_code/
├── __init__.py               # Package exports
├── models.py                 # All Pydantic data models (650 lines)
├── agents/                   # Multi-agent AI system
│   ├── __init__.py
│   ├── ticket_analyzer.py    # Agent 1: Requirement extraction
│   ├── planning_agent.py     # Agent 2: Architectural planning
│   ├── rag_engine.py         # Agent 3: Multi-stage RAG
│   └── code_generator.py     # Agent 5: Code generation
└── execution/                # Technology-agnostic execution
    ├── __init__.py
    ├── base_executor.py      # Abstract base + Factory
    ├── java_executor.py      # Maven & Gradle support
    ├── build_executor.py     # .NET support (legacy)
    └── microservices_executor.py  # Multi-service orchestration
```

---

## 🎯 Quick Start

### Installation

```python
# Add to PYTHONPATH
import sys
sys.path.append('C:/Users/dmadgani/Desktop/My_Aviator/aviator-plugin-sample/src')

# Import
from ticket_to_code import ValueEdgeTicket, TicketPriority
from ticket_to_code.agents import TicketAnalyzerAgent, PlanningAgent
from ticket_to_code.execution import ExecutionEngineFactory
```

### Basic Usage

```python
from ticket_to_code.models import ValueEdgeTicket, TicketPriority
from ticket_to_code.agents import (
    TicketAnalyzerAgent,
    PlanningAgent,
    CodebaseRAGEngine,
    CodeGeneratorAgent
)
from ticket_to_code.execution import MicroserviceExecutionEngine

# 1. Create ticket
ticket = ValueEdgeTicket(
    ticket_id="VE-12345",
    title="Add discount code to checkout",
    description="Users should enter discount codes at checkout...",
    acceptance_criteria=[
        "Discount field visible",
        "Invalid codes show error",
        "Valid codes apply discount"
    ],
    priority=TicketPriority.HIGH
)

# 2. Analyze → Extract requirements
analyzer = TicketAnalyzerAgent()
requirements = analyzer.analyze_ticket(ticket)

# 3. Plan → Create architecture
planner = PlanningAgent()
plan = planner.create_plan(ticket, requirements)

# 4. Retrieve → Get code context
rag = CodebaseRAGEngine()
context = rag.multi_stage_retrieval(requirements)

# 5. Generate → Create code
generator = CodeGeneratorAgent()
generated = generator.generate_code(
    task=plan.tasks[0],
    requirements=requirements,
    context=context
)

# 6. Execute → Build & test (Java)
engine = MicroserviceExecutionEngine(
    workspace_path="/workspace",
    technology="java"
)
build_result = engine.base_executor.execute_build("pom.xml")
test_result = engine.base_executor.execute_tests("pom.xml")

print(f"✅ {test_result.passed}/{test_result.total_tests} tests passed")
```

---

## 🧠 Core Components

### 1. Data Models (`models.py`)

Complete Pydantic models for:
- **Tickets**: `ValueEdgeTicket`, `StructuredRequirements`
- **Planning**: `DevelopmentTask`, `ArchitecturalPlan`, `APIChange`
- **RAG**: `CodeChunk`, `ContextEvaluation`
- **Generation**: `GeneratedCode`, `GeneratedTests`
- **Execution**: `BuildResult`, `TestResult`, `ExecutionError`
- **Workflow**: `AutonomousWorkflowState`, `WorkflowStatus`

### 2. Agents (`agents/`)

#### **Ticket Analyzer Agent**
- Converts natural language → structured JSON
- Extracts functional/technical requirements
- Identifies edge cases

```python
from ticket_to_code.agents import TicketAnalyzerAgent

analyzer = TicketAnalyzerAgent()
requirements = analyzer.analyze_ticket(ticket)
```

#### **Planning Agent**
- Architectural decision making
- Task decomposition with dependencies
- API/database change detection

```python
from ticket_to_code.agents import PlanningAgent

planner = PlanningAgent()
plan = planner.create_plan(ticket, requirements)
```

#### **RAG Engine** (Advanced Multi-Stage Retrieval)
- Semantic search via pgvector
- Context expansion based on dependencies
- Code graph traversal

```python
from ticket_to_code.agents import CodebaseRAGEngine

rag = CodebaseRAGEngine()
context = rag.multi_stage_retrieval(requirements, max_iterations=3)
```

#### **Code Generator Agent**
- Context-aware generation (no hallucination)
- Language-specific best practices (Java, C#, TypeScript, Python)
- SOLID principles, error handling

```python
from ticket_to_code.agents import CodeGeneratorAgent

generator = CodeGeneratorAgent()
code = generator.generate_code(task, requirements, context)
```

### 3. Execution Engine (`execution/`)

**Pluggable architecture** for multiple technology stacks:

#### **Auto-Detection**
```python
from ticket_to_code.execution import ExecutionEngineFactory

# Auto-detects Java from pom.xml or build.gradle
executor = ExecutionEngineFactory.create("/workspace")
```

#### **Java (Maven/Gradle)**
```python
from ticket_to_code.execution import JavaMavenExecutor

executor = JavaMavenExecutor("/workspace")
build_result = executor.execute_build("pom.xml")
test_result = executor.execute_tests("pom.xml")
```

#### **Microservices Orchestration**
```python
from ticket_to_code.execution import MicroserviceExecutionEngine

engine = MicroserviceExecutionEngine("/workspace", technology="java")
results = engine.execute_full_validation()
# Build all → Test all → Docker up → Integration → Docker down
```

---

## 🔥 Key Innovations

### 1. **Multi-Stage RAG** (Unlike Any Other System)
```
Traditional:  Query → Single Retrieval → Use
Our System:   Query → Retrieve → Evaluate → Refine → Expand → Use
```

### 2. **Context Validation**
Validates retrieved context **before** code generation to prevent hallucination

### 3. **Technology-Agnostic Execution**
Supports Java, .NET, Node.js, Python, Go with factory pattern

### 4. **Self-Healing Architecture** (Future)
```
Generate → Build → Fail → Analyze → Fix → Rebuild → Success
```

---

## 📊 Supported Technologies

| Technology | Build Tool | Test Framework | Status |
|------------|-----------|----------------|--------|
| **Java (Maven)** | `mvn clean install` | `mvn test` | ✅ Implemented |
| **Java (Gradle)** | `gradle build` | `gradle test` | ✅ Implemented |
| **.NET** | `dotnet build` | `dotnet test` | ✅ Implemented |
| **Node.js** | `npm build` | `npm test` | 🔜 Future |
| **Python** | `pytest` | `pytest` | 🔜 Future |

---

## 🎯 Use Cases

### Java Microservices (Supplier + Exchange)
```python
from ticket_to_code.execution import MicroserviceExecutionEngine

engine = MicroserviceExecutionEngine("/workspace", technology="java")

# Discovers services automatically
print(f"Found {len(engine.services)} services")

# Build all services
build_results = engine.build_all_services()

# Test all services
test_results = engine.test_all_services()

# Full Docker validation
results = engine.execute_full_validation()
```

### Single Service Development
```python
from ticket_to_code.execution import JavaMavenExecutor

executor = JavaMavenExecutor("/service-workspace")
build = executor.execute_build("pom.xml")
tests = executor.execute_tests("pom.xml")
```

---

## 📚 Documentation

**Root-level documentation** (in `aviator-plugin-sample/`):
- **[AUTONOMOUS_TICKET_TO_CODE_ARCHITECTURE.md](../../AUTONOMOUS_TICKET_TO_CODE_ARCHITECTURE.md)** - Complete system design (1,200 lines)
- **[IMPLEMENTATION_SUMMARY.md](../../IMPLEMENTATION_SUMMARY.md)** - What we built (400 lines)
- **[QUICKSTART_GUIDE.md](../../QUICKSTART_GUIDE.md)** - Usage examples (500 lines)
- **[JAVA_MICROSERVICES_GUIDE.md](../../JAVA_MICROSERVICES_GUIDE.md)** - Java-specific guide (600 lines)
- **[EXECUTION_ENGINE_UPDATE.md](../../EXECUTION_ENGINE_UPDATE.md)** - Why technology-agnostic matters (400 lines)
- **[PROJECT_STRUCTURE.md](../../PROJECT_STRUCTURE.md)** - Complete project overview

---

## 🧪 Testing

```python
# Run tests for specific agent
pytest tests/test_ticket_analyzer.py

# Run all tests
pytest tests/

# Test with Docker
docker compose -f docker-compose.test.yml run --rm test
```

---

## 🔧 Configuration

### Environment Variables

```bash
# .env file
NEBIUS_API_KEY=your_nebius_api_key
CONTENT_SYSTEM=ticket_to_code
```

### Aviator Plugin Registration

```toml
# pyproject.toml
[project.entry-points."aviator.tool_modifiers"]
"ticket_to_code" = "ticket_to_code.extensions:tool_extension"

[project.entry-points."aviator.routers"]
"ticket_to_code" = "ticket_to_code.extensions:router"
```

---

## 🚀 Roadmap

### ✅ Completed (40%)
- Data models (Pydantic)
- Ticket Analyzer Agent
- Planning Agent
- RAG Engine (multi-stage)
- Code Generator Agent
- Execution Engine (Java, .NET)
- Microservices support

### 🔜 Coming Soon (60%)
- Context Evaluator Agent
- Test Generator Agent
- Debug & Fix Agent (self-healing)
- Integration Agent (Git PR, ValueEdge)
- LangGraph workflow orchestration
- Full end-to-end automation

---

## 🤝 Contributing

Part of the NemoForge autonomous coding agent ecosystem.

---

## 📄 License

MIT License

---

## 👤 Author

**Deepak Madgani**  
NemoForge Team  
April 2026

---

**Built with Nebius Token Factory & NVIDIA Nemotron** | **NemoForge**
