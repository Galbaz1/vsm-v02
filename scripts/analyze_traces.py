#!/usr/bin/env python
"""
Analyze Query Traces - Debug tool for Context Agent.

Reads query traces from logs/query_traces/ and provides analysis.

Usage:
    python scripts/analyze_traces.py                    # List recent traces
    python scripts/analyze_traces.py {query_id}        # Show specific trace
    python scripts/analyze_traces.py --loops           # Show queries that looped
    python scripts/analyze_traces.py --summary         # Summary statistics
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

TRACE_DIR = Path("logs/query_traces")


def load_trace(trace_path: Path) -> dict:
    """Load a single trace file."""
    with open(trace_path) as f:
        return json.load(f)


def list_traces(limit: int = 20) -> list:
    """List recent traces sorted by time."""
    if not TRACE_DIR.exists():
        print(f"No traces found. Directory {TRACE_DIR} does not exist.")
        return []
    
    traces = []
    for path in TRACE_DIR.glob("*.json"):
        try:
            trace = load_trace(path)
            traces.append({
                "path": path,
                "query_id": trace.get("query_id", "unknown"),
                "query": trace.get("user_query", "")[:50],
                "iterations": trace.get("total_iterations", 0),
                "outcome": trace.get("final_outcome", "unknown"),
                "time_ms": trace.get("total_time_ms", 0),
                "timestamp": trace.get("timestamp", ""),
            })
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    # Sort by timestamp descending
    traces.sort(key=lambda x: x["timestamp"], reverse=True)
    return traces[:limit]


def show_trace(query_id: str):
    """Show detailed trace for a query."""
    # Find trace file
    matches = list(TRACE_DIR.glob(f"{query_id}*.json"))
    if not matches:
        print(f"No trace found for query_id: {query_id}")
        return
    
    trace = load_trace(matches[0])
    
    print("=" * 70)
    print(f"Query ID: {trace.get('query_id')}")
    print(f"Query: {trace.get('user_query')}")
    print(f"Timestamp: {trace.get('timestamp')}")
    print(f"Outcome: {trace.get('final_outcome')}")
    print(f"Total Iterations: {trace.get('total_iterations')}")
    print(f"Total Time: {trace.get('total_time_ms', 0):.0f}ms")
    print("=" * 70)
    
    print("\nDecision History:")
    print("-" * 70)
    
    for it in trace.get("iterations", []):
        decision = it.get("decision", {})
        env = it.get("environment_state", {})
        
        print(f"\n[Iteration {it.get('iteration')}]")
        print(f"  Tool: {decision.get('tool_name')}")
        print(f"  Reasoning: {decision.get('reasoning', '')[:100]}")
        print(f"  Should End: {decision.get('should_end')}")
        print(f"  Inputs: {json.dumps(decision.get('inputs', {}))[:80]}")
        print(f"  Environment: {env.get('total_objects', 0)} objects, "
              f"~{env.get('token_estimate', 0)} tokens")
        
        if env.get("first_object_preview"):
            print(f"  First Object: {env.get('first_object_preview', '')[:100]}...")
    
    if trace.get("errors"):
        print("\nErrors:")
        for err in trace.get("errors", []):
            print(f"  - {err.get('message')}")


def show_loops():
    """Show queries that hit max iterations."""
    traces = list_traces(limit=100)
    loops = [t for t in traces if t["outcome"] == "max_iterations"]
    
    if not loops:
        print("No queries have hit max iterations.")
        return
    
    print(f"Found {len(loops)} queries that looped:\n")
    print(f"{'Query ID':<12} {'Iterations':<12} {'Time (ms)':<12} {'Query':<40}")
    print("-" * 80)
    
    for t in loops:
        print(f"{t['query_id'][:10]:<12} {t['iterations']:<12} "
              f"{t['time_ms']:<12.0f} {t['query']:<40}")


def show_summary():
    """Show summary statistics."""
    traces = list_traces(limit=500)
    
    if not traces:
        print("No traces found.")
        return
    
    outcomes = Counter(t["outcome"] for t in traces)
    avg_iterations = sum(t["iterations"] for t in traces) / len(traces)
    avg_time = sum(t["time_ms"] for t in traces) / len(traces)
    
    tools_used = Counter()
    for path in TRACE_DIR.glob("*.json"):
        try:
            trace = load_trace(path)
            for it in trace.get("iterations", []):
                tool = it.get("decision", {}).get("tool_name")
                if tool:
                    tools_used[tool] += 1
        except:
            pass
    
    print("=" * 50)
    print("Query Trace Summary")
    print("=" * 50)
    print(f"\nTotal Traces: {len(traces)}")
    print(f"Average Iterations: {avg_iterations:.1f}")
    print(f"Average Time: {avg_time:.0f}ms")
    
    print("\nOutcomes:")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome}: {count} ({count/len(traces)*100:.1f}%)")
    
    print("\nTool Usage:")
    for tool, count in tools_used.most_common():
        print(f"  {tool}: {count}")


def main():
    if len(sys.argv) < 2:
        # List recent traces
        traces = list_traces()
        if not traces:
            return
        
        print(f"{'Query ID':<12} {'Iter':<6} {'Outcome':<15} {'Time':<10} {'Query':<35}")
        print("-" * 80)
        for t in traces:
            print(f"{t['query_id'][:10]:<12} {t['iterations']:<6} "
                  f"{t['outcome']:<15} {t['time_ms']:<10.0f} {t['query']:<35}")
        
        print(f"\nUse: python scripts/analyze_traces.py <query_id> for details")
        return
    
    arg = sys.argv[1]
    
    if arg == "--loops":
        show_loops()
    elif arg == "--summary":
        show_summary()
    elif arg == "--help" or arg == "-h":
        print(__doc__)
    else:
        show_trace(arg)


if __name__ == "__main__":
    main()

