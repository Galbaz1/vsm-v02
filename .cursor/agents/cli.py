#!/usr/bin/env python
"""
CLI entry point for the Cursor AI Sub-Agents Toolkit.

This provides the command-line interface for running analysis.
Can be run directly or via: python -m .cursor.agents

Usage:
    python -m .cursor.agents <query_id>
    python -m .cursor.agents --loops
    python -m .cursor.agents --diagnose "query"
    python -m .cursor.agents --gemini-only <id>
    python -m .cursor.agents --gpt-only <id>
"""

import asyncio
import sys


def main():
    """CLI entry point."""
    from .analyzer import analyze_single, analyze_loops, diagnose_query
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCursor AI Sub-Agents Toolkit v1.0.0")
        print("\nThis package provides intelligent analysis without polluting context.")
        print("\nArchitecture:")
        print("  Phase 1: Gemini 3 Pro (1M context) - Gathers everything")
        print("  Phase 2: GPT-5.1 (high reasoning) - Deep analysis")
        return
    
    arg = sys.argv[1]
    
    async def run():
        if arg == "--loops":
            print(await analyze_loops())
        elif arg == "--gemini-only" and len(sys.argv) > 2:
            print(await analyze_single(sys.argv[2], mode="gemini"))
        elif arg == "--gpt-only" and len(sys.argv) > 2:
            print(await analyze_single(sys.argv[2], mode="gpt"))
        elif arg == "--diagnose" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            print(await diagnose_query(query))
        elif arg in ("--help", "-h"):
            print(__doc__)
        else:
            print(await analyze_single(arg))
    
    asyncio.run(run())


if __name__ == "__main__":
    main()

