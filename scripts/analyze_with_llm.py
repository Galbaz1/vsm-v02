#!/usr/bin/env python
"""
LLM-Powered Trace Analyzer - Using the Cursor Agents Toolkit.

This is a thin wrapper around the .cursor/agents package.
For the full implementation, see .cursor/agents/

Usage:
    python scripts/analyze_with_llm.py <query_id>
    python scripts/analyze_with_llm.py --loops
    python scripts/analyze_with_llm.py --diagnose "query"
    python scripts/analyze_with_llm.py --gemini-only <id>
    python scripts/analyze_with_llm.py --gpt-only <id>

Architecture:
    Phase 1: Gemini 3 Pro (1M context window) - Gathers ALL context
    Phase 2: GPT-5.1 (high reasoning) - Deep analysis and fixes
"""

import sys
import os

# Add project root to path so we can import .cursor.agents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the agents package
from importlib import import_module


def main():
    """Run the CLI."""
    # Dynamic import to handle the dot in .cursor
    try:
        # Try direct import first
        from _cursor.agents.cli import main as cli_main
        cli_main()
    except ImportError:
        # Fall back to manual import
        import asyncio
        
        # Add .cursor to path
        cursor_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".cursor"
        )
        sys.path.insert(0, cursor_path)
        
        from agents.cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
