# /debug - Debug a Failing Query

## What This Command Does

When the user reports a query that failed, looped, or gave wrong results, **DO NOT try to answer the query yourself**. Instead, use the intelligent analyzer to diagnose WHY the system failed.

## Steps

1. **Find the trace ID**
   ```bash
   ls -la logs/query_traces/
   ```
   The most recent trace is likely the failing query.

2. **Run the intelligent analyzer**
   ```bash
   python scripts/analyze_with_llm.py --gemini-only <trace_id_prefix>
   ```
   Use the first 8 characters of the trace filename.

3. **Wait for the diagnosis**
   The analyzer will return:
   - Root cause with evidence
   - Exact file:line to fix
   - Specific code change
   - Confidence level

4. **Apply the fix suggested by the analyzer**

5. **Verify with benchmark**
   ```bash
   python scripts/run_benchmark.py --output results.json
   ```

## Important

- **DO NOT** grep through data files
- **DO NOT** read raw trace JSON files  
- **DO NOT** try to answer the user's original query
- **DO** use the sub-agent (analyzer script) which has 1M context

The sub-agent keeps your context clean by reading everything itself and returning only a concise diagnosis.

## Example Usage

User: "The query 'What is the maximum timeout for Branch Test' keeps looping"

You should:
```bash
ls -la logs/query_traces/
# Find: 9827c6df-ae84-4d89-b7bd-3162d2aa7c4c.json

python scripts/analyze_with_llm.py --gemini-only 9827c6df
# Returns diagnosis with exact fix
```

Then apply the fix and run benchmark.

