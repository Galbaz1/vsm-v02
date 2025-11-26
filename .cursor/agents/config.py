"""
Configuration for the Cursor AI Sub-Agents Toolkit.

This file should be customized per-project to specify which files
are critical for debugging your specific application.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# Calculate project root from this file's location
# .cursor/agents/config.py -> project root
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent


@dataclass
class AgentConfig:
    """Configuration for sub-agents. Customize for your project."""
    
    # Paths - use absolute paths to avoid working directory issues
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    trace_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "logs" / "query_traces")
    
    # API Keys (loaded from environment)
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    # Models
    gemini_model: str = "gemini-3-pro-preview"  # 1M context window
    gpt_model: str = "gpt-5.1"  # High reasoning
    gpt_fallback_model: str = "gpt-4o"  # If 5.1 unavailable
    
    # Critical files for your project (customize this!)
    critical_files: List[str] = field(default_factory=lambda: [
        # Default files - override for your project
        "api/services/agent.py",
        "api/services/llm.py",
        "api/services/environment.py",
    ])
    
    # Extended files for full context (Gemini can handle more)
    extended_files: List[str] = field(default_factory=lambda: [])
    
    # Analysis settings
    max_traces_to_load: int = 20
    max_code_lines_per_file: int = 500  # For GPT-only mode
    
    def __post_init__(self):
        """Combine critical and extended files."""
        if not self.extended_files:
            self.extended_files = self.critical_files.copy()


# ============================================================================
# VSM Project Configuration (customize this for your project)
# ============================================================================

VSM_CONFIG = AgentConfig(
    project_root=_PROJECT_ROOT,
    trace_dir=_PROJECT_ROOT / "logs" / "query_traces",
    critical_files=[
        "api/services/agent.py",
        "api/services/llm.py",
        "api/services/environment.py",
        "api/services/tools/search_tools.py",
        "api/services/tools/base.py",
        "api/services/search.py",
        "api/services/tracer.py",
    ],
    extended_files=[
        "api/services/agent.py",
        "api/services/llm.py",
        "api/services/environment.py",
        "api/services/tools/search_tools.py",
        "api/services/tools/base.py",
        "api/services/search.py",
        "api/services/tracer.py",
        "api/endpoints/agentic.py",
        "api/core/config.py",
        "api/services/tools/__init__.py",
        "scripts/run_benchmark.py",
        "data/benchmarks/benchmark_corrected.json",
    ],
)


def get_config() -> AgentConfig:
    """Get the project configuration."""
    return VSM_CONFIG

