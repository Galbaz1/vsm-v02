"""
Dual-Model Analyzer - Gemini 3 Pro + GPT-5.1 working together.

Architecture:
  Phase 1: Gemini 3 Pro (1M context) - Gathers and synthesizes ALL context
  Phase 2: GPT-5.1 (high reasoning) - Deep analysis and specific fixes
"""

import json
import sys
from typing import Optional, Dict, Any

from .config import get_config, AgentConfig
from .context import (
    load_codebase,
    load_traces,
    load_single_trace,
    format_trace,
    find_similar_trace,
)


# ============================================================================
# Prompts
# ============================================================================

GEMINI_INSTRUCTIONS = """You are a context synthesis expert for debugging AI systems.

Your role is to GATHER and SYNTHESIZE context. Another model (GPT-5.1) will do deep analysis.

## Output Format

### Query Under Analysis
[The user's original query]

### Trace Summary
- Outcome: [completed/max_iterations/error]
- Iterations: [N]
- Tools used: [list]
- Key decisions: [reasoning patterns]

### Relevant Code Sections
For each relevant file, extract EXACT lines that matter:
```
File: path/to/file.py
Lines 220-225:
[exact code]
Potential issue: [what looks suspicious]
```

### Retrieved Content Analysis
[What did searches return? Was the answer present?]

### Patterns Detected
- [List patterns: repeated searches, truncation, tool misuse]

### Context for Deep Analysis
[Synthesize into clear narrative for GPT-5.1]

Be thorough - you have 1M tokens of context."""


GPT_INSTRUCTIONS = """You are an expert debugger analyzing context synthesized by Gemini 3 Pro.

## Output Format (strict)

## Root Cause
[Single paragraph with evidence]

## Code Location
`file_path:line_number`
```python
[exact problematic code]
```

## Fix
```python
[exact replacement code]
```

## Verification
[How to verify the fix works]

## Confidence
[HIGH/MEDIUM/LOW] - [brief justification]

Be precise and actionable. Under 400 words total."""


# ============================================================================
# Gemini 3 Pro (Context Gatherer)
# ============================================================================

async def gemini_gather_context(
    trace_data: Dict[str, Any],
    query: str,
    config: Optional[AgentConfig] = None
) -> str:
    """Use Gemini 3 Pro to gather and synthesize massive context."""
    if config is None:
        config = get_config()
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=config.gemini_api_key)
        
        # Load everything - Gemini can handle 1M tokens
        codebase = load_codebase(config, extended=True)
        all_traces = load_traces(config)
        current_trace = json.dumps(trace_data, indent=2)
        
        prompt = f"""Synthesize context for debugging this query failure.

## User's Question
{query}

## Current Trace
```json
{current_trace}
```

## Full Codebase
{codebase}

## Recent Traces
{all_traces}

Follow your instructions to produce a comprehensive context synthesis."""

        print("Phase 1: Gemini 3 Pro gathering context...", file=sys.stderr)
        
        response = client.models.generate_content(
            model=config.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_INSTRUCTIONS,
                temperature=0.2,
                max_output_tokens=64000,  # Gemini 3 uses many tokens for thinking
            ),
        )
        
        # Gemini SDK bug: response.text can be None with config, access via candidates
        if response.text:
            return response.text
        elif response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return f"Gemini returned empty response: {response}"
        
    except Exception as e:
        print(f"Gemini error: {e}", file=sys.stderr)
        return _fallback_context_gather(trace_data, query, config)


def _fallback_context_gather(
    trace_data: Dict[str, Any],
    query: str,
    config: AgentConfig
) -> str:
    """Fallback if Gemini unavailable."""
    lines = [
        "### Query Under Analysis",
        query,
        "",
        "### Trace Summary",
        f"- Outcome: {trace_data.get('final_outcome', 'unknown')}",
        f"- Iterations: {trace_data.get('total_iterations', 0)}",
        "",
        "### Decision History",
    ]
    
    for it in trace_data.get("iterations", []):
        decision = it.get("decision", {})
        lines.append(f"[{it.get('iteration')}] {decision.get('tool_name')}: {decision.get('reasoning', '')[:100]}")
    
    lines.append("\n### Critical Code Sections")
    codebase = load_codebase(config, extended=False)
    lines.append(codebase[:10000])  # Limit for fallback
    
    return "\n".join(lines)


# ============================================================================
# GPT-5.1 (Deep Analyzer)
# ============================================================================

async def gpt_deep_analysis(
    gemini_context: str,
    query: str,
    config: Optional[AgentConfig] = None
) -> str:
    """Use GPT-5.1 with high reasoning for deep analysis."""
    if config is None:
        config = get_config()
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.openai_api_key)
        
        prompt = f"""Based on the context synthesis below, diagnose the root cause and provide a specific fix.

## Context from Gemini 3 Pro
{gemini_context}

## Original Question
{query}

Provide your deep analysis following the output format in your instructions."""

        print("Phase 2: GPT-5.1 deep analysis (high reasoning)...", file=sys.stderr)
        
        # Try Responses API (GPT-5.1)
        try:
            response = client.responses.create(
                model=config.gpt_model,
                input=prompt,
                instructions=GPT_INSTRUCTIONS,
                reasoning={"effort": "high"},
                text={"verbosity": "low"},
            )
            return response.output_text
        except AttributeError:
            # Fall back to chat completions
            response = client.chat.completions.create(
                model=config.gpt_fallback_model,
                messages=[
                    {"role": "system", "content": GPT_INSTRUCTIONS},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.2,
            )
            return f"(Using {config.gpt_fallback_model})\n\n{response.choices[0].message.content}"
            
    except Exception as e:
        print(f"GPT error: {e}", file=sys.stderr)
        return f"GPT analysis failed: {e}\n\nContext gathered:\n{gemini_context[:2000]}..."


# ============================================================================
# Analysis Modes
# ============================================================================

async def dual_model_analysis(
    trace_data: Dict[str, Any],
    query: str,
    config: Optional[AgentConfig] = None
) -> str:
    """Full dual-model pipeline: Gemini gathers, GPT-5.1 analyzes."""
    gemini_context = await gemini_gather_context(trace_data, query, config)
    gpt_analysis = await gpt_deep_analysis(gemini_context, query, config)
    
    return f"""# Dual-Model Analysis

## Phase 1: Context Synthesis (Gemini 3 Pro)
{gemini_context}

---

## Phase 2: Deep Analysis (GPT-5.1 High Reasoning)
{gpt_analysis}"""


async def gemini_only_analysis(
    trace_data: Dict[str, Any],
    query: str,
    config: Optional[AgentConfig] = None
) -> str:
    """Analysis with just Gemini 3 Pro (faster, good context)."""
    if config is None:
        config = get_config()
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=config.gemini_api_key)
        
        codebase = load_codebase(config, extended=True)
        current_trace = json.dumps(trace_data, indent=2)
        
        prompt = f"""Analyze this query failure and provide a specific diagnosis.

## Query
{query}

## Trace
```json
{current_trace}
```

## Codebase
{codebase}

Provide:
1. Root cause with evidence
2. Exact file:line to fix
3. Specific code change
4. Confidence level"""

        response = client.models.generate_content(
            model=config.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=64000,  # Gemini 3 uses many tokens for thinking
            ),
        )
        
        # Gemini SDK bug: response.text can be None with config, access via candidates
        if response.text:
            return response.text
        elif response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return f"Gemini returned empty response: {response}"
        
    except Exception as e:
        return f"Gemini analysis failed: {e}"


async def gpt_only_analysis(
    trace_data: Dict[str, Any],
    query: str,
    config: Optional[AgentConfig] = None
) -> str:
    """Analysis with just GPT-5.1 (limited context, deeper reasoning)."""
    context = _fallback_context_gather(trace_data, query, config or get_config())
    return await gpt_deep_analysis(context, query, config)


# ============================================================================
# High-Level API
# ============================================================================

async def analyze_single(
    trace_id: str,
    mode: str = "dual",
    config: Optional[AgentConfig] = None
) -> str:
    """Analyze a single trace by ID."""
    if config is None:
        config = get_config()
    
    trace = load_single_trace(trace_id, config)
    if not trace:
        return f"No trace found for: {trace_id}"
    
    query = f"Why did this query fail? Query: {trace.get('user_query', 'unknown')}"
    
    if mode == "gemini":
        return await gemini_only_analysis(trace, query, config)
    elif mode == "gpt":
        return await gpt_only_analysis(trace, query, config)
    else:
        return await dual_model_analysis(trace, query, config)


async def analyze_loops(
    mode: str = "dual",
    config: Optional[AgentConfig] = None
) -> str:
    """Analyze all looping queries."""
    if config is None:
        config = get_config()
    
    if not config.trace_dir.exists():
        return "No traces directory found."
    
    results = []
    for path in config.trace_dir.glob("*.json"):
        try:
            with open(path) as f:
                trace = json.load(f)
            if trace.get("final_outcome") == "max_iterations":
                print(f"Analyzing: {path.name}...", file=sys.stderr)
                query = "Why did this query loop without answering?"
                
                if mode == "gemini":
                    analysis = await gemini_only_analysis(trace, query, config)
                elif mode == "gpt":
                    analysis = await gpt_only_analysis(trace, query, config)
                else:
                    analysis = await dual_model_analysis(trace, query, config)
                
                results.append(f"### {trace.get('user_query', '')[:50]}...\n\n{analysis}")
        except Exception as e:
            print(f"Error with {path}: {e}", file=sys.stderr)
    
    if not results:
        return "No looping queries found."
    
    return f"# Analysis of {len(results)} Looping Queries\n\n" + "\n\n---\n\n".join(results)


async def diagnose_query(
    query: str,
    mode: str = "dual",
    config: Optional[AgentConfig] = None
) -> str:
    """Diagnose why a specific query might fail."""
    if config is None:
        config = get_config()
    
    trace = find_similar_trace(query, config)
    
    if trace:
        if mode == "gemini":
            return await gemini_only_analysis(trace, f"Diagnose: {query}", config)
        elif mode == "gpt":
            return await gpt_only_analysis(trace, f"Diagnose: {query}", config)
        else:
            return await dual_model_analysis(trace, f"Diagnose: {query}", config)
    else:
        return "No matching trace found. Run the query first to generate a trace."

