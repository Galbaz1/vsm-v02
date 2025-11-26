"""
Cursor AI Sub-Agents Toolkit

A reusable package of AI sub-agents that the main Cursor agent can invoke
to get intelligent analysis without polluting context.

Usage:
    from .cursor.agents import analyze_trace, gather_context

    # In hooks or scripts:
    result = await analyze_trace("query_id")
"""

from .analyzer import (
    analyze_single,
    analyze_loops,
    diagnose_query,
    dual_model_analysis,
    gemini_only_analysis,
    gpt_only_analysis,
)
from .context import (
    load_codebase,
    load_traces,
    format_trace,
)
from .config import AgentConfig

__all__ = [
    "analyze_single",
    "analyze_loops", 
    "diagnose_query",
    "dual_model_analysis",
    "gemini_only_analysis",
    "gpt_only_analysis",
    "load_codebase",
    "load_traces",
    "format_trace",
    "AgentConfig",
]

__version__ = "1.0.0"

