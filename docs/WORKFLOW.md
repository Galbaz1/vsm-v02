# VSM Development Workflow

## Two-Agent Pattern (Recommended)

For complex issues, use two separate chats:

### Chat 1: Analysis Agent
```
/analyze Query "X" is failing/looping
```
- Investigates the issue
- Uses sub-agents (Gemini) to analyze traces
- Creates: `context_agent/analysis_<timestamp>.md`
- **Does NOT edit code**

### Chat 2: Coding Agent
```
/implement context_agent/analysis_<timestamp>.md
```
- Reads the analysis file
- Implements the recommended fix
- Runs verification
- **Does NOT do research**

**Why this works:**
- Clean context separation
- Analysis agent can read lots of files without polluting coding context
- Coding agent has focused, actionable instructions

---

## Quick Debug (Single Chat)

For simple issues:
```
/debug Query "X" is failing - diagnose the system
```

---

## For Agents

1. **Find trace:** `ls logs/query_traces/`
2. **Run analyzer:** `python scripts/analyze_with_llm.py --gemini-only <trace_id>`
3. **Apply fix, verify:** `python scripts/run_benchmark.py`

**DO NOT** grep through data files or read raw traces.

---

## The Key Insight: Sub-Agents Keep Context Clean

When you report an issue, I **don't read raw trace files** (they're 10KB+ and pollute context).

Instead, I run:
```bash
python scripts/analyze_with_llm.py --gemini-only <trace_id>
```

This loads **everything** into Gemini 3 Pro (1M context window):
- Entire codebase (api/services/*.py)
- All 20 recent traces
- Full target trace JSON

And returns a **concise diagnosis** (~500 words) with:
- Root cause with evidence
- Exact file:line to fix
- Specific code change
- Confidence level

**My context stays clean. I get actionable output.**

---

## Typical Debugging Session

### 1. User Reports Issue
*"The query 'Branch Test timeout' keeps looping"*

### 2. I Find the Trace
```bash
ls -la logs/query_traces/  # Find most recent
```

### 3. I Run the Analyzer
```bash
python scripts/analyze_with_llm.py --gemini-only 9827c6df
```

### 4. I Get Diagnosis Like:
```
### Root Cause
The Ollama num_ctx is not configured, defaulting to 2048 tokens.
As context grew to 7675 tokens, the prompt was truncated, removing
the JSON instruction. LLM returned plain text → parse failure.

### File to Fix
api/services/llm.py:168

### Code Change
Add `"num_ctx": 8192` to the payload options.

### Confidence
High (95%)
```

### 5. I Apply the Fix
Edit `api/services/llm.py:168`, add the line.

### 6. Verify
```bash
python scripts/run_benchmark.py --output results.json
```

---

## FAQ

### "Won't reading logs pollute context?"

**Yes, raw logs would.** That's why we use analyzers:

| ❌ Don't Do | ✅ Do Instead |
|-------------|---------------|
| `cat logs/query_traces/*.json` | `python scripts/analyze_with_llm.py --gemini-only <id>` |
| Read full `/tmp/vsm-api.log` | `grep "error" /tmp/vsm-api.log \| head -10` |
| Read 500-line trace manually | Let Gemini read it (1M context) |

### "Can you see the frontend?"

**Yes!** I have built-in browser tools:
```
browser_navigate → Go to URL
browser_snapshot → Page structure
browser_take_screenshot → Visual capture
browser_console_messages → JS errors
```

### "When should I start a new chat?"

✅ **New chat when:**
- This chat is confused
- Very long (>50 messages)
- Completely different task

❌ **Don't new chat for:**
- Switching between debugging/coding
- Each small fix

---

## Rules (Auto-Apply by File)

| Rule | Activates When | Purpose |
|------|----------------|---------|
| `conventions.mdc` | Always | Stack, ports, structure |
| `philosophy.mdc` | Always | Architecture principles |
| `debugging.mdc` | Editing `api/services/**`, `logs/**` | Debug workflow |

---

## Hooks (Auto-Run in Background)

| Hook | When | What You See |
|------|------|--------------|
| `beforeSubmitPrompt` | Before prompt | Warning if queries looped |
| `afterFileEdit` | After edit | "Run benchmark" reminder |
| `stop` | Task complete | Summary notification |

---

## Quick Reference Commands

```bash
# Find all traces
ls -la logs/query_traces/

# Analyze a trace (use first 8 chars of ID)
python scripts/analyze_with_llm.py --gemini-only abc12345

# Find looping queries
python scripts/analyze_traces.py --loops

# Basic trace summary (no LLM)
python scripts/analyze_traces.py <trace_id>

# Test search directly
curl "localhost:8001/search?query=YOUR_QUERY&limit=5" | jq '.hits[].content'

# Run benchmark
python scripts/run_benchmark.py --output results.json

# Check services
curl -s localhost:8001/docs > /dev/null && echo "API ✅" || echo "API ❌"
```

---

## Flow Diagram

```
User: "Query X is failing"
          │
          ▼
┌─────────────────────────────────┐
│ ls logs/query_traces/           │  ← Find trace ID
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ python scripts/analyze_with_    │  ← Gemini reads EVERYTHING
│   llm.py --gemini-only <id>     │    (1M context window)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Get diagnosis:                  │
│ - Root cause                    │  ← ~500 words, actionable
│ - file:line                     │
│ - Code fix                      │
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Apply fix                       │
│ Run benchmark                   │  ← Verify
└─────────────────────────────────┘
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| **Traces** | Auto-saved for every `/agentic_search` |
| **Analyzer** | Gemini reads everything, returns diagnosis |
| **Rules** | Auto-apply conventions by file path |
| **Hooks** | Auto-run at lifecycle events |
| **Browser** | Built-in for UI testing |
| **Context** | Kept clean via analyzers, not raw dumps |
