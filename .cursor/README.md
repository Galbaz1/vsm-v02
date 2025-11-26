# Cursor AI Toolkit for Agentic RAG Debugging

A reusable package of AI sub-agents for debugging complex agentic RAG systems.
**This toolkit keeps your Cursor context clean while providing deep analysis.**

## For Users: How to Report Issues

When a query fails, use the `/debug` command to trigger the debugging workflow:

```
/debug Query "What is the maximum timeout for Branch Test" keeps looping
```

Or be explicit:
```
Debug why the system failed on query X (don't try to answer X, diagnose the system)
```

## For Agents: Quick Start

```bash
# 1. Find trace
ls -la logs/query_traces/

# 2. Run intelligent analysis (Gemini reads EVERYTHING)
python scripts/analyze_with_llm.py --gemini-only <trace_id_prefix>

# 3. Apply the suggested fix, then verify
python scripts/run_benchmark.py --output results.json
```

**DO NOT** grep data files or read traces manually - let the sub-agent handle it.

## Architecture

```
.cursor/
├── agents/              # 🤖 AI Sub-Agents Package (Python)
│   ├── analyzer.py      # Dual-model analyzer (Gemini 3 Pro + GPT-5.1)
│   ├── config.py        # Project configuration (critical files list)
│   ├── context.py       # Context gathering utilities
│   └── cli.py           # Command-line interface
│
├── hooks/               # 🪝 Cursor Lifecycle Hooks (TypeScript)
│   ├── before-prompt.ts # Loop detection warning
│   ├── after-edit.ts    # Benchmark reminder for agent files
│   └── stop.ts          # Session summary
│
├── rules/               # 📋 Cursor Rules (.mdc)
│   ├── debugging.mdc    # ⚠️ MUST READ before debugging
│   └── *.mdc            # Project conventions
│
└── README.md            # This file
```

## How It Works

### The Problem
When debugging agentic RAG issues, you need to read:
- Full query traces (JSON, can be 10KB+)
- Multiple source files
- Logs and environment state

This pollutes your Cursor context and makes you less effective.

### The Solution
**Sub-agents do the heavy lifting:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  YOU: "Why is query X failing?"                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  python scripts/analyze_with_llm.py <trace_id>                          │
│                                                                          │
│  GEMINI 3 PRO (1M context window) loads:                                │
│  ├── Entire codebase (api/services/*.py)                                │
│  ├── All recent query traces                                            │
│  └── Full decision history with environment state                       │
│                                                                          │
│  Returns: Concise diagnosis with specific file:line and fix             │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  YOUR CONTEXT: Clean! Only ~500 words of diagnosis                      │
│  YOU CAN NOW: Apply the fix, verify, move on                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Debugging Workflow

### Step 1: Reproduce and Get Trace ID
```bash
# Run the failing query
curl "http://localhost:8001/agentic_search?query=YOUR+QUERY"

# Find the trace
ls -la logs/query_traces/  # Most recent is your query
```

### Step 2: Quick Trace Summary (no LLM)
```bash
python scripts/analyze_traces.py <trace_id_prefix>
```
Shows: iterations, tools used, outcome - helps identify obvious issues.

### Step 3: Deep Analysis (LLM-powered)
```bash
# Gemini only (faster, uses 1M context)
python scripts/analyze_with_llm.py --gemini-only <trace_id>

# Full dual-model (Gemini gathers, GPT-5.1 reasons deeply)
python scripts/analyze_with_llm.py <trace_id>
```

### Step 4: Apply Fix and Verify
```bash
# After fixing, run benchmark
python scripts/run_benchmark.py --output results.json
```

## Common Issues the Analyzer Finds

| Symptom | Typical Root Cause | Location |
|---------|-------------------|----------|
| Agent loops on fast_vector_search | Environment truncation hides answer | `environment.py:222` |
| "Failed to parse LLM response" | num_ctx too small, prompt truncated | `llm.py:168` |
| Wrong tool selected | Decision prompt missing context | `llm.py:262-282` |
| Hybrid search returns nothing | Error swallowed silently | `search_tools.py:265` |

## Configuration

Edit `.cursor/agents/config.py` to list your project's critical files:

```python
VSM_CONFIG = AgentConfig(
    critical_files=[
        "api/services/agent.py",     # Decision loop
        "api/services/llm.py",       # LLM client
        "api/services/environment.py", # State management
        # Add your files here
    ],
)
```

## Requirements

- **Python 3.12+** in conda env `vsm-hva`
- **API Keys** in `.env`:
  - `GEMINI_API_KEY` - For Gemini 3 Pro (1M context)
  - `OPENAI_API_KEY` - For GPT-5.1 (optional, for dual-model)

```bash
conda activate vsm-hva
pip install google-genai openai python-dotenv
```

## Installing in Another Project

```bash
./.cursor/install.sh /path/to/new/project
```

Then edit `config.py` to list that project's critical files.
