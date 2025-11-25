# Agent Architecture - Future Design

**Last Updated:** 2025-11-25  
**Status:** Design document for Phase 4-5 implementation

---

## Overview

This document describes the target agent architecture, adapting Weaviate's Elysia patterns for our dual-pipeline RAG system with visual grounding.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Decision Agent                              │
│                        (gpt-oss-120B via Ollama)                    │
│                                                                      │
│  Input: user_prompt, available_tools, environment, errors, history  │
│  Output: {tool, inputs, reasoning, should_end}                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ FastVector    │   │ ColQwen       │   │ TextResponse  │
│ SearchTool    │   │ SearchTool    │   │ Tool          │
│               │   │               │   │               │
│ is_available: │   │ is_available: │   │ is_available: │
│   always      │   │   always      │   │   env.has_    │
│               │   │               │   │   results     │
│ run_if_true:  │   │ run_if_true:  │   │               │
│   false       │   │   false       │   │ end: true     │
└───────┬───────┘   └───────┬───────┘   └───────────────┘
        │                   │
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ Environment   │   │ VisualInterp  │
│               │   │ Tool          │
│ fast_vector:  │   │               │
│   [results]   │   │ is_available: │
│               │   │   colqwen in  │
│ colqwen:      │   │   environment │
│   [pages]     │   │               │
│               │   │ Input: pages  │
│ visual_interp:│   │ from env      │
│   [text]      │   └───────────────┘
└───────────────┘
```

---

## Core Components

### 1. TreeData (Shared State)

```python
@dataclass
class TreeData:
    """All state passed between decision agent and tools."""
    
    user_prompt: str
    environment: Environment
    conversation_history: List[Dict]  # Previous turns
    tasks_completed: Dict[str, List[float]]  # tool -> execution times
    errors: List[str]  # Error messages for LLM context
    collection_names: List[str]  # Available Weaviate collections
```

### 2. Environment (Result Storage)

```python
class Environment:
    """Centralized storage for all tool outputs.
    
    Structure:
        environment[tool_name][result_name] = [
            {"objects": [...], "metadata": {...}},
        ]
    
    Also stores hidden_environment for internal state not shown to LLM.
    """
    
    def add(self, tool_name: str, result: Result) -> None
    def find(self, tool_name: str, name: str = None) -> Optional[Dict]
    def is_empty(self) -> bool
    def estimate_tokens(self) -> int
    def to_llm_context(self, max_tokens: int = 10000) -> str
```

### 3. Result Objects

```python
@dataclass
class Result:
    """Typed output from tools."""
    
    objects: List[Dict]          # Retrieved items
    metadata: Dict = None        # Query info, counts, etc.
    name: str = None             # Key for environment storage
    payload_type: str = "result" # Frontend rendering hint
    llm_message: str = None      # Message shown to decision agent
    
    # llm_message supports placeholders:
    # - {num_objects}: len(objects)
    # - {name}: result name
    # - Any key in metadata
    #
    # Example: "Found {num_objects} chunks matching '{query}'"
```

### 4. Error Objects

```python
@dataclass
class Error:
    """Recoverable error that informs LLM without crashing."""
    
    message: str
    recoverable: bool = True
    suggestion: str = None  # What LLM should try next
    
    # Examples:
    # - "No results found" + "Try broader search terms"
    # - "Weaviate connection failed" + recoverable=False
```

---

## Tool Specifications

### FastVectorSearchTool

```python
class FastVectorSearchTool(Tool):
    name = "fast_vector_search"
    description = "Quick semantic search over text chunks. Use for factual queries about specs, definitions, procedures. Returns in ~0.5s."
    end = False
    
    async def is_tool_available(self, tree_data: TreeData) -> bool:
        return True  # Always available
    
    async def run_if_true(self, tree_data: TreeData) -> tuple[bool, dict]:
        return False, {}  # Never auto-trigger
    
    async def __call__(self, tree_data: TreeData, inputs: dict):
        query = inputs["query"]
        results = await search_service.search(query, limit=5)
        
        if not results:
            yield Error(
                message=f"No text results for: {query}",
                recoverable=True,
                suggestion="Try ColQwen visual search for diagrams"
            )
            return
        
        yield Result(
            objects=results,
            metadata={"query": query, "source": "AssetManual"},
            name="text_results",
            payload_type="chunks",
            llm_message="Found {num_objects} text chunks. Top result from page {page}."
        )
```

### ColQwenSearchTool

```python
class ColQwenSearchTool(Tool):
    name = "colqwen_visual_search"
    description = "Visual document search using multi-vector embeddings. Use for diagrams, schematics, charts, visual layouts. SLOWER (~3-5s) but accurate for visual content."
    end = False
    
    async def is_tool_available(self, tree_data: TreeData) -> bool:
        return True  # Always available
    
    async def __call__(self, tree_data: TreeData, inputs: dict):
        query = inputs["query"]
        pages = await colqwen_service.search(query, top_k=3)
        
        if not pages:
            yield Error(
                message=f"No visual matches for: {query}",
                recoverable=True,
                suggestion="Try fast_vector_search for text content"
            )
            return
        
        yield Result(
            objects=pages,
            metadata={"query": query, "source": "PDFDocuments"},
            name="visual_pages",
            payload_type="colqwen_pages",
            llm_message="Found {num_objects} pages with visual matches. Pages: {page_numbers}."
        )
```

### VisualInterpretationTool

```python
class VisualInterpretationTool(Tool):
    name = "visual_interpretation"
    description = "Use VLM to interpret page images. Describes diagrams, charts, tables found in ColQwen results."
    end = False
    
    async def is_tool_available(self, tree_data: TreeData) -> bool:
        # Only available after ColQwen has retrieved pages
        return tree_data.environment.find("colqwen_visual_search") is not None
    
    async def __call__(self, tree_data: TreeData, inputs: dict):
        # Get pages from environment
        colqwen_data = tree_data.environment.find("colqwen_visual_search", "visual_pages")
        if not colqwen_data:
            yield Error(message="No visual pages to interpret", recoverable=False)
            return
        
        pages = colqwen_data[-1]["objects"]  # Most recent
        interpretations = []
        
        for page in pages[:2]:  # Limit to top 2 for speed
            image_path = page["image_path"]
            interpretation = await vlm_service.interpret(
                image_path=image_path,
                query=tree_data.user_prompt
            )
            interpretations.append({
                "page_number": page["page_number"],
                "interpretation": interpretation
            })
        
        yield Result(
            objects=interpretations,
            metadata={"source": "Qwen3-VL-8B"},
            name="visual_interpretations",
            payload_type="interpretations",
            llm_message="VLM analyzed {num_objects} pages. Summary: {summary}"
        )
```

### TextResponseTool

```python
class TextResponseTool(Tool):
    name = "text_response"
    description = "Generate final response to user based on retrieved information."
    end = True  # This tool can end the conversation
    
    async def is_tool_available(self, tree_data: TreeData) -> bool:
        # Only available when environment has some results
        return not tree_data.environment.is_empty()
    
    async def __call__(self, tree_data: TreeData, inputs: dict):
        # Build context from environment
        context = tree_data.environment.to_llm_context(max_tokens=8000)
        
        prompt = f"""Based on the retrieved information, answer the user's question.

USER QUESTION: {tree_data.user_prompt}

RETRIEVED INFORMATION:
{context}

INSTRUCTIONS:
- Answer based ONLY on the retrieved information
- Include page references
- If information is insufficient, say so
- Be concise but thorough

ANSWER:"""
        
        # Stream response
        async for token in llm_service.stream_generate(prompt):
            yield {"type": "token", "content": token}
```

### SummarizeTool (Auto-trigger)

```python
class SummarizeTool(Tool):
    name = "summarize"
    description = "Summarize environment when it gets too large."
    end = False
    
    async def is_tool_available(self, tree_data: TreeData) -> bool:
        return not tree_data.environment.is_empty()
    
    async def run_if_true(self, tree_data: TreeData) -> tuple[bool, dict]:
        # Auto-trigger when environment exceeds token limit
        tokens = tree_data.environment.estimate_tokens()
        if tokens > 50000:
            return True, {"max_summary_tokens": 5000}
        return False, {}
    
    async def __call__(self, tree_data: TreeData, inputs: dict):
        # Summarize and replace environment content
        summary = await llm_service.summarize(
            tree_data.environment.to_llm_context(),
            max_tokens=inputs.get("max_summary_tokens", 5000)
        )
        
        # Store in hidden environment, clear main
        tree_data.environment.hidden_environment["summary"] = summary
        tree_data.environment.environment.clear()
        
        yield Result(
            objects=[{"summary": summary}],
            name="summarized",
            llm_message="Environment summarized due to size. Key points preserved."
        )
```

---

## Decision Agent Prompt

```python
DECISION_PROMPT = """You are the decision agent for a technical manual search system.

CURRENT STATE:
- User query: {user_prompt}
- Environment (retrieved data): {environment_context}
- Previous errors: {errors}
- Tools already used: {tasks_completed}

AVAILABLE TOOLS:
{available_tools}

INSTRUCTIONS:
1. Analyze what the user needs
2. Consider what's already been retrieved (environment)
3. Choose the best tool to call next
4. If you have enough information, use text_response to answer

Output JSON:
{{
    "tool": "<tool_name>",
    "inputs": {{"<param>": "<value>"}},
    "reasoning": "<why this tool>",
    "should_end": false
}}

If the task is impossible, set:
{{
    "tool": "text_response",
    "inputs": {{}},
    "reasoning": "Cannot complete: <reason>",
    "should_end": true,
    "impossible": true
}}
"""
```

---

## Streaming Output Format

Each message is NDJSON with a type field:

```json
{"type": "decision", "tool": "fast_vector_search", "reasoning": "User asked factual question"}
{"type": "status", "message": "Searching text chunks..."}
{"type": "result", "tool": "fast_vector_search", "objects": [...], "metadata": {...}}
{"type": "decision", "tool": "text_response", "reasoning": "Have enough context"}
{"type": "token", "content": "The"}
{"type": "token", "content": " operating"}
{"type": "token", "content": " voltage..."}
{"type": "complete"}
```

Error case:

```json
{"type": "decision", "tool": "fast_vector_search", "reasoning": "..."}
{"type": "error", "message": "No results found", "recoverable": true, "suggestion": "Try visual search"}
{"type": "decision", "tool": "colqwen_visual_search", "reasoning": "Fast search failed, trying visual"}
```

---

## Information Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           /agentic_search                           │
│                          (FastAPI Endpoint)                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AgentOrchestrator                           │
│                                                                      │
│  1. Create TreeData with user_prompt                                │
│  2. Loop until max_iterations or should_end:                        │
│     a. Get available tools (check is_tool_available)                │
│     b. Check auto-triggers (run_if_true)                            │
│     c. Call Decision Agent LLM → get tool choice                    │
│     d. Execute tool → get Result/Error                              │
│     e. Add to Environment                                           │
│     f. Yield NDJSON messages                                        │
│  3. Yield "complete" when done                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    Ollama     │   │  Weaviate     │   │     MLX       │
│  gpt-oss-120B │   │  (8080)       │   │  VLM (8000)   │
│   (11434)     │   │               │   │               │
│               │   │ - AssetManual │   │ Qwen3-VL-8B   │
│ Decision +    │   │ - PDFDocs     │   │               │
│ Synthesis     │   │               │   │ Page interp   │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

## Memory Budget

| Component | Estimated | Notes |
|-----------|-----------|-------|
| gpt-oss-120B | ~65 GB | MoE, only 5.1B active per token |
| Qwen3-VL-8B | ~8 GB | Loaded on-demand for visual queries |
| ColQwen2.5-v0.2 | ~4 GB | Always loaded for retrieval |
| nomic-embed-text | ~2 GB | Embeddings |
| KV Cache | ~150 GB | Multi-turn conversations |
| **Total** | ~229 GB | Fits M3 256GB with headroom |

---

## Implementation Phases

### Phase 4: LLM Decision Agent (Current Sprint)

1. Create `api/services/llm.py` with Ollama client
2. Implement decision prompt with tool formatting
3. Replace `_make_decision()` with LLM call
4. Test full loop with FastVectorSearchTool

### Phase 5: Visual Interpretation

1. Add MLX VLM service (`api/services/vlm.py`)
2. Implement VisualInterpretationTool
3. Wire conditional availability after ColQwen
4. Test visual query end-to-end

### Phase 6: Polish

1. Add SummarizeTool with auto-trigger
2. Implement conversation persistence
3. Frontend integration for new payload types
4. Benchmark and optimize

---

**Version:** 1.0  
**Author:** Agent Optimization Research  
**See Also:** [ARCHITECTURE.md](ARCHITECTURE.md), [AGENT_OPTIMIZATION_PLAN.md](AGENT_OPTIMIZATION_PLAN.md)

