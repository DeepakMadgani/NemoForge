# ⚡ NemoForge: Autonomous SWE Agent

> **Powered by NVIDIA Nemotron on Nebius Token Factory**  
> An autonomous software engineering agent that ingests Jira/Git issues, localizes target code via deterministic AST symbol graphs, performs surgical edits using NVIDIA Nemotron, and guarantees correctness through automated multi-tier verification gates.

[![Nebius Token Factory](https://img.shields.io/badge/Powered%20By-Nebius%20Token%20Factory-00D47E?style=flat-square)](https://nebius.com)
[![NVIDIA Nemotron](https://img.shields.io/badge/Model-NVIDIA%20Nemotron%203%20Super-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

---

## 🌟 Overview

**NemoForge** is a production-grade autonomous software engineering platform submitted for the **Nebius x NVIDIA Global AI Hackathon** (*Coding and Agentic Engineering Track*).

Unlike superficial coding assistants that rely on fuzzy vector retrieval and prompt-and-pray patching, NemoForge merges **deterministic AST compilation & call-graph analysis** with **NVIDIA's flagship Nemotron reasoning models** hosted on **Nebius Token Factory**.

```
                      ┌──────────────────────────────────────────────┐
                      │    Ticket / Issue / Feature Description      │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │       Stage 1: Deterministic Discovery       │
                      │  (Tree-sitter AST Slicing + SQLite Symbols)  │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    Stage 2: NVIDIA Nemotron Reasoning        │
                      │  (Nemotron 3 Super 120B MoE via Nebius API)  │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │        Stage 3: Surgical Patch Engine        │
                      │   (Fail-Closed 3-Line Anchor Replacements)   │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │     Stage 4: Two-Tier Verification Gate      │
                      │   (Maven/Gradle Compiler + Playwright E2E)   │
                      └──────────────────────────────────────────────┘
```

---

## ✨ Architectural Pillars

### 1. 🟢 NVIDIA Nemotron on Nebius Token Factory
- **Flagship Intelligence**: Utilizes **NVIDIA Nemotron 3 Super (120B MoE)** for deep architectural understanding, cross-microservice dependency navigation, and surgical code synthesis.
- **Fast Discovery**: Routes lightweight intake checks and log diagnosis to **NVIDIA Nemotron 70B Instruct**.
- **Nebius Token Factory Integration**: Native OpenAI-compatible client targeting `https://api.tokenfactory.nebius.com/v1/`.

### 2. ⚡ Zero-Database Fast Indexing (SQLite + Tree-sitter)
- Completely abandoned bloated graph databases (Neo4j) and vector search engines.
- Indexes 4,600+ multi-language files in **13.7 seconds flat** into local embedded SQLite (`.nemoforge/index.db`).
- Slices method boundaries, `@Autowired` dependencies, and class skeletons on-the-fly in 3ms.

### 3. 🔒 Zero-Leak Key Architecture
- API credentials are typed directly in the frontend UI (`⚙️ LLM Settings`) and stored exclusively in local untracked `.env.local`.
- Zero credentials are ever committed to Git or pushed to remote repositories.

### 4. 🛡️ Fail-Closed Surgical Patching & Verification
- Employs strict 3-line context anchors to eliminate silent hallucinations.
- Validates code changes with live build executions (`mvn compile`, `npm test`, Playwright DOM assertions) before marking any ticket complete.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Node.js 18+ & npm
- Nebius Token Factory API Key ([Get Nebius Key](https://tokenfactory.nebius.com))

### 1. Clone & Setup
```bash
git clone https://github.com/YourUsername/NemoForge-NVIDIA.git
cd NemoForge-NVIDIA
```

### 2. Configure Environment
Copy the example environment file:
```bash
cp .env.example .env.local
```
Or simply launch the app and enter your key in the UI:
```env
LLM_PROVIDER=nebius
NEBIUS_API_KEY=your_nebius_api_key_here
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
LLM_MODEL=nebius/nvidia/nemotron-3-super-120b-a12b
```

### 3. Start Backend & UI
In terminal 1 (Backend):
```bash
cd chatbot/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8002
```

In terminal 2 (Frontend):
```bash
cd chatbot/frontend
npm install
npm run dev
```

Open your browser to **`http://localhost:3000`**.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Inference & Reasoning** | NVIDIA Nemotron 3 Super / 70B | Flagship code synthesis & architectural reasoning |
| **Hosting Infrastructure** | Nebius Token Factory | High-throughput serverless open-model inference |
| **AST Code Intelligence** | Tree-sitter & Embedded SQLite | Sub-millisecond syntax parsing & call graph indexing |
| **Backend API** | FastAPI & Uvicorn | Async WebSocket orchestration & workflow dispatch |
| **Frontend UI** | React, Vite & Vanilla CSS | Live streaming workflow monitor & human checkpoint gate |
| **Verification Gate** | Subprocess Sandbox & Playwright | Live polyglot compilation & E2E DOM verification |

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
