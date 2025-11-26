# VSM Debugging Guide

Complete guide for debugging the agentic RAG system.

---

## Quick Reference

### Services
| Service | Port | Health Check |
|---------|------|--------------|
| API | 8001 | `curl localhost:8001/docs` |
| Weaviate | 8080 | `curl localhost:8080/v1/.well-known/ready` |
| Ollama | 11434 | `curl localhost:11434/api/tags` |
| Frontend | 3000 | `curl localhost:3000` |

### Key Commands
```bash
# Start everything
./scripts/start.sh

# Stop everything
./scripts/stop.sh

# Run benchmark
python scripts/run_benchmark.py --output results.json

# Analyze query traces
python scripts/analyze_traces.py          # List recent
python scripts/analyze_traces.py --loops  # Find looping queries
python scripts/analyze_traces.py <id>     # Show specific trace
```

---

## Debugging Workflow

### 1. Query Not Working?

**Step 1: Test direct search (bypass agent)**
```bash
curl "http://localhost:8001/search?query=YOUR_QUERY&limit=10" | jq '.hits[].content'
```
- ✅ Results? → Data exists, agent routing is the issue
- ❌ No results? → Data not indexed or wrong collection

**Step 2: Check query trace**
```bash
python scripts/analyze_traces.py --loops
# Find your query, then:
python scripts/analyze_traces.py <query_id>
```

**Step 3: Check source data**
```bash
grep -i "YOUR_KEYWORD" data/output_techman.json | head -5
```

### 2. Agent Loops Forever?

**Symptoms:** Query hits max iterations (10), never generates response.

**Check:**
```bash
python scripts/analyze_traces.py --loops
```

**Common causes:**
1. **Environment truncation** (`api/services/environment.py:222`)
   - Objects truncated to 200 chars, hiding answer from LLM
2. **Missing termination logic** (`api/services/llm.py`)
   - LLM never sets `should_end: true`
3. **Poor retrieval** 
   - Answer exists but not in top results

**Quick fix test:**
```python
# In environment.py line 222, try:
preview = str(obj)[:500] + "..."  # was 200
```

### 3. Wrong Tool Selected?

**Check decision history:**
```bash
python scripts/analyze_traces.py <query_id>
```

Look at the reasoning for each iteration. The LLM prompt is in `api/services/llm.py:262-282`.

---

## Log Files

| Log | Location | Purpose |
|-----|----------|---------|
| API | `/tmp/vsm-api.log` | HTTP requests, errors |
| Ollama | `/tmp/vsm-ollama.log` | LLM calls, model loading |
| Frontend | `/tmp/vsm-frontend.log` | Next.js output |
| File edits | `/tmp/vsm-edits.log` | Cursor hook: edited files |
| Query traces | `logs/query_traces/` | Full agent decision history |
| Cursor sessions | `logs/cursor_sessions/` | Session metadata |

**Watch logs:**
```bash
tail -f /tmp/vsm-api.log
tail -f /tmp/vsm-ollama.log
```

---

## Cursor Hooks (Auto-Running)

Hooks run automatically during Cursor agent sessions.

| Hook | When | What |
|------|------|------|
| `beforeSubmitPrompt` | Before prompt sent | Injects context if debugging |
| `afterFileEdit` | After file edit | Logs edits, warns on agent files |
| `stop` | Task complete | Saves session, links traces |

**Check hook output:** Look for messages in Cursor's agent panel.

---

## Benchmark Testing

```bash
# Run full benchmark (20 questions)
python scripts/run_benchmark.py

# Save results for comparison
python scripts/run_benchmark.py --output results_$(date +%Y%m%d).json

# Compare metrics
cat results.json | jq '.regular_rag'
```

**Key metrics:**
- **Hit@1**: Correct page in top result
- **Hit@3**: Correct page in top 3
- **MRR**: Mean Reciprocal Rank
- **Manual Accuracy**: Correct manual selected

---

## Context Agent Workflow

When a bug is reported:

1. **Gather context** using the `.cursor/rules/context_agent.mdc` rule
2. **Check traces** in `logs/query_traces/`
3. **Check sessions** in `logs/cursor_sessions/`
4. **Create context file** in `context_agent/`
5. **Hand off** to coding agent with full context

**Context Agent outputs:** `context_agent/*.md`

---

## Architecture Quick Reference

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│            AgentOrchestrator                │
│   api/services/agent.py                     │
│                                             │
│   ┌─────────────────────────────────────┐   │
│   │         Decision Loop               │   │
│   │   max_iterations: 10                │   │
│   │                                     │   │
│   │   1. Get available tools            │   │
│   │   2. LLM decides tool + inputs      │   │
│   │   3. Execute tool                   │   │
│   │   4. Store in Environment           │   │
│   │   5. Check should_end               │   │
│   └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
    │
    ├── fast_vector_search → Weaviate (AssetManual)
    ├── colqwen_search → ColQwen API (PDFDocuments)
    ├── text_response → Generate answer from Environment
    └── visual_interpretation → MLX VLM
```

---

## Files to Know

| File | Purpose |
|------|---------|
| `api/services/agent.py` | Main orchestrator, decision loop |
| `api/services/llm.py` | LLM client, decision prompt |
| `api/services/environment.py` | State management, `to_llm_context()` |
| `api/services/tracer.py` | Query trace logging |
| `api/services/tools/search_tools.py` | Search tool implementations |
| `api/services/search.py` | Weaviate query logic |
| `scripts/analyze_traces.py` | CLI for trace analysis |
| `scripts/run_benchmark.py` | Benchmark evaluation |

