"""
Query Tracer - Captures full decision trace for debugging agent loops.

Each query gets a JSON trace file in logs/query_traces/ containing:
- All iterations with decisions and reasoning
- Environment state at each step
- Tool results and timing
- Final outcome

Usage in agent.py:
    tracer = QueryTracer(query_id, user_prompt)
    # ... in loop ...
    tracer.log_iteration(iteration, decision, env_state, result)
    # ... after loop ...
    tracer.save("completed" | "max_iterations" | "error")
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default trace directory - use absolute path relative to this file's location
# This ensures traces save to the right place regardless of working directory
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent  # api/services/tracer.py -> project root
TRACE_DIR = _PROJECT_ROOT / "logs" / "query_traces"

# Log the path on import for debugging
logger.info(f"QueryTracer will save to: {TRACE_DIR}")


class QueryTracer:
    """
    Captures full execution trace for a single query.
    
    Saves to logs/query_traces/{query_id}.json for post-mortem analysis.
    """
    
    def __init__(
        self,
        query_id: str,
        user_query: str,
        enabled: bool = True,
        trace_dir: Optional[Path] = None,
    ):
        self.query_id = query_id
        self.enabled = enabled
        self.trace_dir = trace_dir or TRACE_DIR
        self.start_time = datetime.now()
        
        self.trace: Dict[str, Any] = {
            "query_id": query_id,
            "timestamp": self.start_time.isoformat(),
            "user_query": user_query,
            "iterations": [],
            "errors": [],
        }
    
    def log_iteration(
        self,
        iteration: int,
        decision: Dict[str, Any],
        environment_state: Dict[str, Any],
        tool_result: Optional[Dict[str, Any]] = None,
        llm_prompt_preview: Optional[str] = None,
    ) -> None:
        """
        Log a single iteration of the decision loop.
        
        Args:
            iteration: Current iteration number (1-indexed)
            decision: The Decision object as dict
            environment_state: Output of environment.to_debug_state()
            tool_result: Result metadata from tool execution
            llm_prompt_preview: First N chars of LLM prompt (for debugging)
        """
        if not self.enabled:
            return
        
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "decision": {
                "tool_name": decision.get("tool_name"),
                "inputs": decision.get("inputs"),
                "reasoning": decision.get("reasoning"),
                "should_end": decision.get("should_end", False),
            },
            "environment_state": environment_state,
        }
        
        if tool_result:
            entry["tool_result"] = tool_result
        
        if llm_prompt_preview:
            entry["llm_prompt_preview"] = llm_prompt_preview[:500]
        
        self.trace["iterations"].append(entry)
        
        # Also log to standard logger for real-time visibility
        logger.debug(
            f"[{self.query_id[:8]}] Iteration {iteration}: "
            f"{decision.get('tool_name')} - {decision.get('reasoning', '')[:50]}"
        )
    
    def log_error(self, error_message: str, recoverable: bool = True) -> None:
        """Log an error that occurred during execution."""
        if not self.enabled:
            return
        
        self.trace["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "message": error_message,
            "recoverable": recoverable,
        })
    
    def save(self, outcome: str, total_time_ms: Optional[float] = None) -> Optional[Path]:
        """
        Save the trace to disk.
        
        Args:
            outcome: Final outcome - "completed", "max_iterations", "error", "impossible"
            total_time_ms: Total query time in milliseconds
            
        Returns:
            Path to saved trace file, or None if disabled
        """
        if not self.enabled:
            return None
        
        end_time = datetime.now()
        
        self.trace["final_outcome"] = outcome
        self.trace["total_iterations"] = len(self.trace["iterations"])
        self.trace["total_time_ms"] = total_time_ms or (
            (end_time - self.start_time).total_seconds() * 1000
        )
        self.trace["end_timestamp"] = end_time.isoformat()
        
        # Create directory if needed
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trace
        trace_path = self.trace_dir / f"{self.query_id}.json"
        try:
            with open(trace_path, "w") as f:
                json.dump(self.trace, f, indent=2, default=str)
            
            logger.info(f"Query trace saved: {trace_path}")
            return trace_path
            
        except Exception as e:
            logger.error(f"Failed to save query trace: {e}")
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current trace state."""
        tools_used = [
            it["decision"]["tool_name"] 
            for it in self.trace["iterations"]
        ]
        
        return {
            "query_id": self.query_id,
            "iterations": len(self.trace["iterations"]),
            "tools_used": tools_used,
            "errors": len(self.trace["errors"]),
            "unique_tools": list(set(tools_used)),
        }


def get_environment_debug_state(environment) -> Dict[str, Any]:
    """
    Extract debug state from an Environment object.
    
    Can be called with environment instance to capture state
    before each decision.
    """
    try:
        all_objects = environment.get_all_objects()
        
        return {
            "is_empty": environment.is_empty(),
            "token_estimate": environment.estimate_tokens(),
            "tools_with_data": list(environment.environment.keys()),
            "total_objects": len(all_objects),
            # Include first object preview to help debug truncation
            "first_object_preview": (
                str(all_objects[0])[:300] if all_objects else None
            ),
        }
    except Exception as e:
        return {"error": str(e)}


# Singleton for trace configuration
_tracing_enabled: bool = True


def set_tracing_enabled(enabled: bool) -> None:
    """Enable or disable query tracing globally."""
    global _tracing_enabled
    _tracing_enabled = enabled


def is_tracing_enabled() -> bool:
    """Check if query tracing is enabled."""
    return _tracing_enabled

