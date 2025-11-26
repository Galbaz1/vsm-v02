# Observability & Logging Plan for Context Agent

## Current State Assessment

### ✅ What Exists
| Component | Location | Purpose |
|-----------|----------|---------|
| Service logs | `/tmp/vsm-{ollama,api,frontend}.log` | Basic stdout/stderr |
| Benchmark script | `scripts/run_benchmark.py` | Retrieval accuracy (MRR, Hit@k) |
| Benchmark data | `data/benchmarks/benchmark*.json` | 20 ground-truth Q&A pairs |
| Agent streaming | NDJSON to frontend | Decisions visible in UI |

### ❌ What's Missing for Debugging
| Gap | Impact |
|-----|--------|
| No decision trace logs | Can't see LLM input/output for each iteration |
| No environment state capture | Don't know what data LLM saw when deciding |
| No query-level persistence | Can't replay or analyze failed queries |
| No timing breakdown | Don't know where time is spent per tool |
| Benchmark tests retrieval only | Doesn't test agent loop behavior |

---

## Proposed Logging Architecture

### 1. Query Trace Log (NEW)
**Location:** `logs/query_traces/{query_id}.json`

```json
{
  "query_id": "abc-123",
  "timestamp": "2025-11-26T12:12:41Z",
  "user_query": "What is the supply voltage of S-130?",
  "iterations": [
    {
      "iteration": 1,
      "decision": {
        "tool_name": "fast_vector_search",
        "inputs": {"query": "...", "limit": 5},
        "reasoning": "...",
        "should_end": false
      },
      "llm_prompt_preview": "First 500 chars of prompt...",
      "environment_state": {
        "is_empty": true,
        "token_estimate": 0
      },
      "tool_result": {
        "count": 5,
        "time_ms": 450,
        "llm_message": "Found 5 chunks..."
      }
    },
    // ... more iterations
  ],
  "total_iterations": 10,
  "total_time_ms": 70000,
  "final_outcome": "max_iterations_reached",
  "errors": []
}
```

### 2. Decision Logger (Add to `agent.py`)
```python
import json
from pathlib import Path
from datetime import datetime

class QueryTracer:
    def __init__(self, query_id: str, user_query: str):
        self.trace = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "iterations": [],
        }
    
    def log_iteration(self, iteration: int, decision: dict, 
                      env_state: dict, result: dict):
        self.trace["iterations"].append({
            "iteration": iteration,
            "decision": decision,
            "environment_state": env_state,
            "tool_result": result,
        })
    
    def save(self, outcome: str):
        self.trace["final_outcome"] = outcome
        self.trace["total_iterations"] = len(self.trace["iterations"])
        
        log_dir = Path("logs/query_traces")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        with open(log_dir / f"{self.trace['query_id']}.json", "w") as f:
            json.dump(self.trace, f, indent=2)
```

### 3. Enhanced Environment Logging
Add to `api/services/environment.py`:
```python
def to_debug_state(self) -> dict:
    """Full state dump for debugging."""
    return {
        "is_empty": self.is_empty(),
        "token_estimate": self.estimate_tokens(),
        "tools_with_data": list(self.environment.keys()),
        "total_objects": len(self.get_all_objects()),
        "full_context_preview": self.to_llm_context(max_tokens=1000),
    }
```

---

## Implementation Priority

### Phase 1: Immediate (Debug Current Issue)
1. [ ] Add `QueryTracer` class to `agent.py`
2. [ ] Log each decision with full reasoning
3. [ ] Capture environment state before each decision
4. [ ] Save traces to `logs/query_traces/`

### Phase 2: Analysis Tools
1. [ ] Script to analyze failed queries: `scripts/analyze_traces.py`
2. [ ] Add `--trace` flag to benchmark script
3. [ ] Dashboard/CLI to view recent query traces

### Phase 3: Alerting & Metrics
1. [ ] Track loop count distribution
2. [ ] Alert on queries exceeding N iterations
3. [ ] Prometheus metrics for tool usage

---

## Files to Modify

| File | Changes |
|------|---------|
| `api/services/agent.py` | Add QueryTracer, log each iteration |
| `api/services/environment.py` | Add `to_debug_state()` method |
| `api/services/llm.py` | Log LLM prompt/response (DEBUG level) |
| `scripts/run_benchmark.py` | Add `--trace` flag, save traces |
| `api/main.py` | Configure logging levels |

---

## Context Agent Data Sources

When debugging, check these locations in order:

### 1. Query Traces (MOST USEFUL - after implementation)
```bash
ls logs/query_traces/
cat logs/query_traces/{query_id}.json | jq .
```

### 2. Terminal Logs
```bash
tail -f /tmp/vsm-api.log        # API errors
tail -f /tmp/vsm-ollama.log     # LLM calls (timing)
```

### 3. Benchmark Results
```bash
python scripts/run_benchmark.py --output results.json
cat results.json | jq '.results[] | select(.regular_rag_rank == null)'
```

### 4. Raw Data Verification
```bash
# Check if data exists in Weaviate
python scripts/weaviate_search_manual.py "S-130 supply voltage"

# Check source JSON
grep -i "supply voltage" data/output_techman.json
```

### 5. Agent Source Code
| File | What to check |
|------|---------------|
| `api/services/agent.py` | Decision loop, max_iterations |
| `api/services/llm.py` | System prompt, decision parsing |
| `api/services/environment.py` | Context truncation (line 222) |
| `api/services/tools/search_tools.py` | Search tool behavior |

---

## Quick Debug Commands

```bash
# Replay a query with verbose output
curl -N "http://localhost:8001/agentic_search?query=YOUR_QUERY" | jq -c '.'

# Check Weaviate collections
curl http://localhost:8080/v1/schema | jq '.classes[].class'

# Test direct search (bypass agent)
curl "http://localhost:8001/search?query=S-130+supply+voltage&limit=10" | jq '.hits[].content'

# Run benchmark with trace
python scripts/run_benchmark.py --trace --output results.json
```

