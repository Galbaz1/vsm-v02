"""
Context gathering utilities for sub-agents.

Provides functions to load codebase files and traces without
polluting the main Cursor agent's context.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import get_config, AgentConfig


def load_codebase(
    config: Optional[AgentConfig] = None,
    extended: bool = True
) -> str:
    """
    Load codebase files as formatted string.
    
    Args:
        config: Agent configuration (uses default if None)
        extended: If True, load extended files (for Gemini). If False, critical only.
    
    Returns:
        Formatted string with all file contents.
    """
    if config is None:
        config = get_config()
    
    files_to_load = config.extended_files if extended else config.critical_files
    files_content = []
    
    for file_path in files_to_load:
        full_path = config.project_root / file_path
        if full_path.exists():
            try:
                content = full_path.read_text()
                # Truncate if needed (for GPT-only mode)
                if not extended:
                    lines = content.split("\n")[:config.max_code_lines_per_file]
                    content = "\n".join(lines)
                files_content.append(f"### FILE: {file_path}\n```python\n{content}\n```\n")
            except Exception as e:
                files_content.append(f"### FILE: {file_path}\n[Error reading: {e}]\n")
    
    return "\n\n".join(files_content)


def load_traces(
    config: Optional[AgentConfig] = None,
    max_traces: Optional[int] = None
) -> str:
    """
    Load recent trace files.
    
    Args:
        config: Agent configuration
        max_traces: Maximum number of traces to load (default from config)
    
    Returns:
        Formatted string with trace contents.
    """
    if config is None:
        config = get_config()
    
    if max_traces is None:
        max_traces = config.max_traces_to_load
    
    if not config.trace_dir.exists():
        return "No traces directory found."
    
    traces = []
    sorted_paths = sorted(
        config.trace_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for path in sorted_paths[:max_traces]:
        try:
            with open(path) as f:
                trace = json.load(f)
            traces.append(f"### TRACE: {path.name}\n```json\n{json.dumps(trace, indent=2)}\n```")
        except Exception as e:
            traces.append(f"### TRACE: {path.name}\n[Error: {e}]")
    
    return "\n\n".join(traces)


def load_single_trace(
    trace_id: str,
    config: Optional[AgentConfig] = None
) -> Optional[Dict[str, Any]]:
    """
    Load a single trace by ID prefix.
    
    Args:
        trace_id: Trace ID or prefix
        config: Agent configuration
    
    Returns:
        Trace dict or None if not found.
    """
    if config is None:
        config = get_config()
    
    if not config.trace_dir.exists():
        return None
    
    matches = list(config.trace_dir.glob(f"{trace_id}*.json"))
    if not matches:
        return None
    
    with open(matches[0]) as f:
        return json.load(f)


def format_trace(trace: Dict[str, Any]) -> str:
    """
    Format a trace dict into a readable string.
    
    Args:
        trace: The trace dictionary
    
    Returns:
        Formatted string for LLM consumption.
    """
    lines = [
        f"Query: {trace.get('user_query', 'unknown')}",
        f"Outcome: {trace.get('final_outcome', 'unknown')}",
        f"Iterations: {trace.get('total_iterations', 0)}",
        f"Time: {trace.get('total_time_ms', 0):.0f}ms",
        "",
        "Decision History:",
    ]
    
    for it in trace.get("iterations", []):
        decision = it.get("decision", {})
        env = it.get("environment_state", {})
        
        lines.append(f"  [{it.get('iteration')}] {decision.get('tool_name')}")
        lines.append(f"      Reasoning: {decision.get('reasoning', '')[:150]}")
        lines.append(f"      should_end: {decision.get('should_end')}")
        lines.append(f"      Env: {env.get('total_objects', 0)} objects")
        
        if env.get("first_object_preview"):
            lines.append(f"      Preview: {env.get('first_object_preview', '')[:150]}...")
    
    return "\n".join(lines)


def find_similar_trace(
    query: str,
    config: Optional[AgentConfig] = None
) -> Optional[Dict[str, Any]]:
    """
    Find the most similar trace to a given query.
    
    Args:
        query: The query to match
        config: Agent configuration
    
    Returns:
        Best matching trace or None.
    """
    if config is None:
        config = get_config()
    
    if not config.trace_dir.exists():
        return None
    
    best_match = None
    best_score = 0
    
    for path in config.trace_dir.glob("*.json"):
        try:
            with open(path) as f:
                trace = json.load(f)
            user_query = trace.get("user_query", "").lower()
            # Simple word overlap scoring
            query_words = set(query.lower().split())
            trace_words = set(user_query.split())
            score = len(query_words & trace_words)
            if score > best_score:
                best_score = score
                best_match = trace
        except:
            pass
    
    return best_match

