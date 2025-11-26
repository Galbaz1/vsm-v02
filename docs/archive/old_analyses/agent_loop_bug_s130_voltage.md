# Agent Loop Bug: S-130 Supply Voltage Query

## User Query
```
What is the supply voltage range of the SR unit S130?
```

## Expected Answer
**Source:** Benchmark `benchmark_corrected.json` (id: 8)
```
The S-130 operates on a supply voltage between 11 and 16 V DC.
Reference: 91001002 Techman, page 45
```

---

## Problem Summary
The agent entered an infinite loop (10 iterations max), always choosing `fast_vector_search` but never:
1. Synthesizing a response from retrieved data
2. Calling `text_response` to end the conversation
3. Setting `should_end: true` in decisions

---

## Agent Decision Log (10 iterations)

| # | Tool | Reasoning |
|---|------|-----------|
| 1 | fast_vector_search | "specific specification (supply voltage range) of a unit, which is a factual text query" |
| 2 | fast_vector_search | "previous search returned general description but not the specific supply voltage range" |
| 3 | fast_vector_search | "previous search did not return the specific supply voltage range" |
| 4 | fast_vector_search | "previous search returned relevant sections but did not show the exact supply voltage range" |
| 5 | fast_vector_search | "previous search returned truncated chunks" |
| 6 | fast_vector_search | "chunk_type 'table' will retrieve the relevant spec table" |
| 7 | fast_vector_search | "previous searches returned relevant sections but not specific voltage details" |
| 8 | fast_vector_search | "targeted text search for phrase 'Supply voltage'" |
| 9 | fast_vector_search | "previous searches returned only truncated snippets" |
| 10| fast_vector_search | "higher limit should retrieve precise specification" |

---

## Data Exists in Source

The specification **IS** in `data/output_techman.json` (page 45, table):

```json
{
  "markdown": "<table id=\"45-1\">...<tr><td id=\"45-e\">Supply voltage:</td><td id=\"45-f\">11 to 16 V DC (DC+, DC- or Vdd, 0V).</td></tr>...",
  "type": "table",
  "id": "acf56b24-2ba9-4557-9d23-51f8a5bd6e24",
  "grounding": {
    "page": 45
  }
}
```

---

## Root Cause Hypotheses

### 1. Environment Context Truncation (HIGH LIKELIHOOD)
**File:** `api/services/environment.py:221-223`
```python
for obj in objects[:5]:  # Limit to 5 objects per entry
    preview = str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj)
    lines.append(f"  - {preview}")
```
- Each object is truncated to 200 chars
- The table with "11 to 16 V DC" may exceed this, hiding the answer from the LLM
- LLM sees "truncated" data and keeps searching

### 2. Missing Termination Logic (HIGH LIKELIHOOD)
**File:** `api/services/llm.py` - DecisionPromptBuilder system prompt
```
Guidelines:
4. When you have enough information: Use text_response to answer
```
- The LLM never decides `should_end: true`
- Possible prompt issue: LLM doesn't recognize when it HAS the answer
- No explicit "after N failed searches, synthesize from what you have"

### 3. Chunk Indexing Issue (MEDIUM)
- The table chunk may not be vectorized correctly
- Query "supply voltage S-130" may not semantically match table content well
- bge-m3 embeddings might not handle technical tables optimally

### 4. Rule-Based Fallback Never Triggers (LOW)
**File:** `api/services/agent.py:229-237`
```python
if not tree_data.environment.is_empty():
    if tree_data.num_iterations > 0:
        return Decision(
            tool_name="text_response",
            should_end=True,
        )
```
- This rule only runs if LLM fails
- LLM is succeeding (returning valid JSON) but making poor decisions

---

## Relevant Files

| File | Purpose |
|------|---------|
| `api/services/agent.py` | Main orchestrator, decision loop, max_iterations=10 |
| `api/services/llm.py` | DecisionPromptBuilder, OllamaClient |
| `api/services/environment.py` | TreeData, Environment.to_llm_context() |
| `api/services/tools/search_tools.py` | FastVectorSearchTool, Result yield |
| `api/services/search.py` | Weaviate near_text query |

---

## Log Sources

| Log | Location |
|-----|----------|
| Ollama | `/tmp/vsm-ollama.log` (shows 10x `/api/chat` + 10x `/api/embed` calls) |
| API | `/tmp/vsm-api.log` (shows single agentic_search request) |
| Frontend | `/tmp/vsm-frontend.log` (no errors) |

---

## Ollama Log Evidence
Terminal 10 shows the loop pattern:
```
[GIN] 2025/11/26 - 12:12:55 | POST "/api/chat"   # Decision 1
[GIN] 2025/11/26 - 12:12:56 | POST "/api/embed"  # Search 1
[GIN] 2025/11/26 - 12:12:59 | POST "/api/chat"   # Decision 2
[GIN] 2025/11/26 - 12:13:00 | POST "/api/embed"  # Search 2
... (repeats 10 times)
[GIN] 2025/11/26 - 12:14:02 | POST "/api/embed"  # Search 10
```
Total time: ~70 seconds for failed query

---

## Suggested Fixes to Investigate

### Fix 1: Increase context preview length
```python
# environment.py:222
preview = str(obj)[:500] + "..." if len(str(obj)) > 500 else str(obj)
```

### Fix 2: Add forced termination after N searches
```python
# agent.py - in decision loop
if tree_data.tasks_completed.get("fast_vector_search", []) >= 3:
    # Force text_response after 3 searches
    return Decision(tool_name="text_response", should_end=True)
```

### Fix 3: Update LLM prompt with explicit termination rules
```python
# llm.py - SYSTEM_PROMPT
"""
CRITICAL: After 2-3 search attempts, you MUST use text_response 
to synthesize an answer from available data, even if incomplete.
"""
```

### Fix 4: Improve table chunk retrieval
- Check if tables are ingested with proper metadata
- Consider adding keyword search fallback for specification queries

---

## Testing Commands

```bash
# Direct search test
curl "http://localhost:8001/search?query=supply+voltage+S-130&limit=10"

# Agentic search
curl "http://localhost:8001/agentic_search?query=What+is+the+supply+voltage+range+of+the+S-ART+Unit+S-130"

# Check Weaviate data for S-130 table
python scripts/weaviate_search_manual.py "S-130 supply voltage"
```

---

## Priority
**HIGH** - This is a core agent functionality bug affecting answer quality and response times.

