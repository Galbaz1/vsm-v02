# Agent Optimization Plan for VSM Demo v02

**Last Updated:** 2025-11-25  
**Target Hardware:** M3 Mac Studio, 256GB Unified Memory  
**Research Sources:** 
- Exa AI deep search (Nov 2025)
- Weaviate Elysia framework (Aug 2025)
- Qwen technical reports (2025)

---

## Executive Summary

Your current `agent.py` is a **skeleton with rule-based routing**. Based on deep research into the Qwen ecosystem and Weaviate's Elysia agentic RAG framework (Nov 2025), here's a comprehensive plan to transform it into a production-grade agentic RAG system.

### Key Findings from Elysia Research (NEW)

| Elysia Pattern | Description | Why It Matters |
|----------------|-------------|----------------|
| **Decision Tree** | Pre-defined node structure, not flat tool list | Enables sophisticated multi-step workflows |
| **Tool Availability** | `is_tool_available()` / `run_if_true()` | Context-aware tool exposure |
| **Environment State** | Centralized `Environment` object | Persistent state across tool executions |
| **Self-Healing Errors** | `Error` objects inform LLM, not crash | Graceful recovery, retry logic |
| **Reasoning Transparency** | LLM outputs decision reasoning | Debuggable, explainable routing |

### Model & Infrastructure Findings

| Area | Current State | Recommended Upgrade |
|------|--------------|---------------------|
| **Routing** | Keyword-based | Qwen3 with thinking mode |
| **Answer Generation** | Mock/passthrough | Qwen3-14B via MLX |
| **Visual Interpretation** | None | Qwen3-VL-8B or Qwen2.5-VL-7B |
| **ColQwen Retrieval** | ColQwen2.5 (good) | Upgrade to ColQwen2.5-v0.2 |
| **Agent Architecture** | Simple routing | Elysia-style decision tree |
| **Inference** | Ollama | MLX (native Apple Silicon) |

---

## Part 1: Model Selection for M3 256GB

### Why MLX Over Ollama?

Recent Apple research (Nov 2025) shows MLX delivers:
- **Up to 4x faster TTFT** (Time to First Token) vs previous generation
- **Native unified memory** - no CPU↔GPU memory copies
- **Quantization built-in** - 4-bit with minimal quality loss
- **Direct Hugging Face integration** - `pip install mlx-lm`

With **256GB unified memory**, you can run:

| Model | Size (4-bit) | Memory Usage | Use Case |
|-------|-------------|--------------|----------|
| Qwen3-14B-Instruct | ~9 GB | ~12 GB | Primary agent reasoning |
| Qwen3-30B-A3B (MoE) | ~17 GB | ~20 GB | High-quality routing (only 3B active) |
| Qwen3-VL-8B-Instruct | ~5 GB | ~8 GB | Visual interpretation |
| Qwen2.5-VL-32B-Instruct | ~18 GB | ~24 GB | Premium visual understanding |
| ColQwen2.5 | ~2 GB | ~4 GB | Visual document retrieval |

**Total for full stack:** ~48 GB (leaves 200+ GB for caching and concurrent requests)

### Recommended Model Stack

```python
# Primary configuration for M3 256GB
AGENT_MODELS = {
    # Router/Reasoner - MoE gives 30B quality with 3B cost
    "router": "mlx-community/Qwen3-30B-A3B-Instruct-4bit",
    
    # Answer synthesizer - balance of speed and quality
    "synthesizer": "mlx-community/Qwen3-14B-Instruct-4bit",
    
    # Visual interpreter - for understanding retrieved pages
    "vision": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    
    # Alternative: Qwen3-VL-8B when MLX version available
    # "vision": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    
    # Embeddings - keep using Ollama for now (stable)
    "embeddings": "nomic-embed-text"  # via Ollama
}
```

---

## Part 2: What's Missing in Current Agent

### Gap Analysis

| Feature | Current Code | What's Needed |
|---------|-------------|---------------|
| **LLM Loading** | `self.model = None` | MLX model initialization |
| **Routing Logic** | Keyword matching | LLM-based with thinking trace |
| **Answer Synthesis** | Mock text | Actual generation from context |
| **Visual Understanding** | None | Feed ColQwen results to VLM |
| **Multi-turn Context** | Unused param | KV cache + history management |
| **Tool Definitions** | Good structure | Add actual execution |
| **Streaming** | Mock delays | Real token streaming |

### Current Code Gaps

```python
# Line 41 - Model never loaded
self.model = None  # Loaded on demand

# Lines 93-108 - Rule-based routing (should be LLM)
visual_keywords = ["diagram", "image", "picture", ...]
is_visual = any(kw in query_lower for kw in visual_keywords)

# Lines 144-162 - Mock streaming (should be real)
for char in response:
    yield {"type": "token", "content": char}
    await asyncio.sleep(0.01)  # Fake delay
```

---

## Part 3: Implementation Plan

### Phase 1: MLX Integration (Week 1)

**Goal:** Replace Ollama with MLX for inference

#### 1.1 Install MLX Dependencies

```bash
# In your vsm-hva conda environment
pip install mlx mlx-lm
```

#### 1.2 Create MLX Service

```python
# api/services/mlx_inference.py
"""MLX-based inference service for Apple Silicon."""

import mlx.core as mx
from mlx_lm import load, generate, stream_generate
from typing import AsyncGenerator, Optional
import asyncio

class MLXInferenceService:
    """Singleton service for MLX model inference."""
    
    _instance = None
    _models = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.loaded_models = {}
    
    async def load_model(self, model_id: str, alias: str = None):
        """Load a model from Hugging Face via MLX."""
        key = alias or model_id
        if key not in self.loaded_models:
            # Load in background thread to not block
            loop = asyncio.get_event_loop()
            model, tokenizer = await loop.run_in_executor(
                None, load, model_id
            )
            self.loaded_models[key] = (model, tokenizer)
        return self.loaded_models[key]
    
    async def generate(
        self,
        model_key: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate text completion."""
        model, tokenizer = self.loaded_models[model_key]
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature
            )
        )
        return response
    
    async def stream_generate(
        self,
        model_key: str,
        prompt: str,
        max_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        """Stream text generation token by token."""
        model, tokenizer = self.loaded_models[model_key]
        
        # MLX stream_generate is synchronous, wrap in executor
        for token in stream_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens
        ):
            yield token
            await asyncio.sleep(0)  # Yield control


# Singleton accessor
def get_mlx_service() -> MLXInferenceService:
    return MLXInferenceService.get_instance()
```

### Phase 2: LLM-Based Routing (Week 2)

**Goal:** Replace keyword matching with actual reasoning

#### 2.1 Routing Prompt Design

```python
# api/services/agent.py - Updated analyze_query method

ROUTING_PROMPT = """You are a query router for a technical manual search system.

Given a user query about asset manuals (electrical equipment, firmware, etc.), decide the best search strategy:

STRATEGIES:
- "fast_only": Use for factual text queries (definitions, specs, procedures, parameters)
  Examples: "What is the operating voltage?", "Define firmware version"
  
- "colqwen_only": Use for visual/spatial queries (diagrams, schematics, charts, layouts)
  Examples: "Show me the wiring diagram", "What does the control panel look like?"
  
- "fast_then_colqwen": Use for complex queries that benefit from both text AND visual context
  Examples: "How do I troubleshoot error E05?", "Explain the installation process"

<query>{query}</query>

Think through your decision step by step, then output JSON:
{{"strategy": "...", "reasoning": "..."}}
"""

async def analyze_query(self, query: str, user_history: Optional[List] = None) -> SearchDecision:
    """LLM-based query analysis with thinking."""
    
    mlx = get_mlx_service()
    
    # Format prompt
    prompt = ROUTING_PROMPT.format(query=query)
    
    # Get LLM decision with explicit reasoning
    response = await mlx.generate(
        model_key="router",
        prompt=prompt,
        max_tokens=256,
        temperature=0.3  # Low temp for consistent routing
    )
    
    # Parse JSON from response
    try:
        # Extract JSON from response (may have thinking prefix)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group())
        else:
            # Fallback to fast_then_colqwen for safety
            decision = {"strategy": "fast_then_colqwen", "reasoning": "Parse error, using safe default"}
    except json.JSONDecodeError:
        decision = {"strategy": "fast_then_colqwen", "reasoning": "JSON parse error"}
    
    strategy = decision.get("strategy", "fast_then_colqwen")
    
    return SearchDecision(
        use_fast_vector=strategy in ["fast_only", "fast_then_colqwen"],
        use_colqwen=strategy in ["colqwen_only", "fast_then_colqwen"],
        strategy=strategy,
        reasoning=decision.get("reasoning", "")
    )
```

### Phase 3: Answer Synthesis (Week 3)

**Goal:** Generate natural language answers from retrieved context

#### 3.1 Synthesis Prompt Design

```python
SYNTHESIS_PROMPT = """You are a technical documentation assistant. Based on the retrieved content from asset manuals, provide a helpful answer.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Include specific page references (e.g., "See page 45")
3. If diagrams are relevant, describe what they show
4. If the context doesn't contain the answer, say so clearly
5. Be concise but thorough

ANSWER:"""

async def synthesize_answer(
    self,
    query: str,
    fast_results: List[Dict],
    colqwen_results: List[Dict]
) -> AsyncGenerator[str, None]:
    """Generate streaming answer from retrieved context."""
    
    # Format context from results
    context_parts = []
    
    if fast_results:
        for r in fast_results[:5]:
            context_parts.append(
                f"[Page {r.get('page_number', '?')}, {r.get('chunk_type', 'text')}]\n"
                f"{r.get('content', '')[:500]}"
            )
    
    if colqwen_results:
        for r in colqwen_results[:3]:
            context_parts.append(
                f"[Visual: Page {r.get('page_number', '?')}]\n"
                f"Image available at: {r.get('image_path', 'N/A')}"
            )
    
    context = "\n\n---\n\n".join(context_parts)
    prompt = SYNTHESIS_PROMPT.format(context=context, query=query)
    
    mlx = get_mlx_service()
    async for token in mlx.stream_generate("synthesizer", prompt, max_tokens=1024):
        yield token
```

### Phase 4: Visual Interpretation (Week 4)

**Goal:** Use VLM to understand retrieved page images

#### 4.1 MLX-VLM Integration

```python
# api/services/vision.py
"""Vision-language model service using MLX-VLM."""

from mlx_vlm import load as load_vlm, generate as generate_vlm
from PIL import Image
from pathlib import Path

class VisionService:
    """Service for visual understanding of manual pages."""
    
    def __init__(self):
        self.model = None
        self.processor = None
    
    async def load_model(self, model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"):
        """Load VLM model."""
        if self.model is None:
            self.model, self.processor = load_vlm(model_id)
    
    async def interpret_page(
        self,
        image_path: str,
        query: str
    ) -> str:
        """Interpret a manual page image in context of query."""
        
        image = Image.open(image_path)
        
        prompt = f"""Look at this technical manual page and answer the question.

QUESTION: {query}

Describe any relevant:
- Diagrams or schematics
- Tables or specifications
- Warning labels or callouts
- Visual instructions

ANSWER:"""
        
        response = generate_vlm(
            self.model,
            self.processor,
            prompt=prompt,
            image=image,
            max_tokens=512
        )
        
        return response
    
    async def verify_relevance(
        self,
        image_path: str,
        query: str
    ) -> dict:
        """Check if a page is actually relevant to the query."""
        
        image = Image.open(image_path)
        
        prompt = f"""Is this manual page relevant to the query: "{query}"?

Respond in JSON:
{{"relevant": true/false, "reason": "brief explanation", "confidence": 0.0-1.0}}"""
        
        response = generate_vlm(
            self.model, self.processor,
            prompt=prompt, image=image,
            max_tokens=128
        )
        
        # Parse JSON response
        try:
            return json.loads(response)
        except:
            return {"relevant": True, "reason": "Parse error", "confidence": 0.5}
```

### Phase 5: Qwen-Agent Integration (Week 5-6)

**Goal:** Leverage official Qwen-Agent framework with MCP support

#### 5.1 Why Qwen-Agent?

The official Qwen-Agent framework (Nov 2025) provides:
- **Native MCP support** - Connect to any MCP server
- **Built-in tools** - code_interpreter, image_gen, web_search
- **Thinking mode** - Explicit reasoning traces
- **Streaming** - Real-time response generation
- **Multi-agent** - Orchestrate multiple specialized agents

#### 5.2 Installation

```bash
pip install qwen-agent

# For MCP support
brew install uv git sqlite3  # macOS
```

#### 5.3 Qwen-Agent Based Implementation

```python
# api/services/qwen_agent_service.py
"""Qwen-Agent based orchestrator with MCP support."""

from qwen_agent.agents import Assistant
from qwen_agent.tools import BaseTool
from typing import Dict, Any, List
import json

class FastVectorSearchTool(BaseTool):
    """Tool for fast text-based vector search."""
    
    name = "fast_vector_search"
    description = (
        "Quick semantic search over manual content. "
        "Use for factual queries about specs, definitions, procedures. "
        "Returns in ~0.5s."
    )
    parameters = [{
        "name": "query",
        "type": "string",
        "description": "Search query",
        "required": True
    }]
    
    def call(self, params: dict, **kwargs) -> str:
        # Import here to avoid circular imports
        from api.services.search import SearchService
        
        search = SearchService()
        results = search.search(params["query"], limit=5)
        return json.dumps(results, indent=2)


class ColQwenSearchTool(BaseTool):
    """Tool for visual document search."""
    
    name = "colqwen_visual_search"
    description = (
        "Deep visual search using ColQwen multi-vector embeddings. "
        "Use for diagram queries, visual troubleshooting, spatial layouts. "
        "SLOWER (~3-5s) but more accurate for visual content."
    )
    parameters = [{
        "name": "query",
        "type": "string",
        "description": "Search query",
        "required": True
    }]
    
    def call(self, params: dict, **kwargs) -> str:
        from api.services.colqwen import ColQwenService
        
        colqwen = ColQwenService()
        results = colqwen.search(params["query"], top_k=3)
        return json.dumps(results, indent=2)


class QwenAgentOrchestrator:
    """Production agent using Qwen-Agent framework."""
    
    def __init__(self):
        # Configure LLM - use local MLX endpoint
        self.llm_cfg = {
            'model': 'qwen3-14b',
            'model_server': 'http://localhost:8000/v1',  # MLX server
            'api_key': 'EMPTY',
            'generate_cfg': {
                'extra_body': {
                    'chat_template_kwargs': {'enable_thinking': True}
                }
            }
        }
        
        # Define tools
        self.tools = [
            FastVectorSearchTool(),
            ColQwenSearchTool(),
        ]
        
        # System prompt
        self.system = """You are a technical documentation assistant for asset manuals.

Your job is to help users find information in technical manuals about electrical equipment, 
firmware, and industrial assets.

WORKFLOW:
1. Analyze the user's query to understand what they need
2. Choose the appropriate search tool:
   - fast_vector_search: For text-based factual queries
   - colqwen_visual_search: For visual/diagram queries
3. Interpret the search results
4. Provide a clear, helpful answer with page references

Always cite your sources with page numbers."""
        
        # Create assistant
        self.assistant = Assistant(
            llm=self.llm_cfg,
            system_message=self.system,
            function_list=self.tools
        )
    
    async def run(self, query: str) -> AsyncGenerator[Dict, None]:
        """Run the agent and stream responses."""
        
        messages = [{'role': 'user', 'content': query}]
        
        for response in self.assistant.run(messages=messages):
            # Qwen-Agent yields incremental responses
            yield {
                "type": "agent_response",
                "content": response
            }
```

---

## Part 4: ColQwen Upgrade

### Current: ColQwen2.5
Your current setup uses ColQwen2.5 which is good, but there's a newer version.

### Recommended: ColQwen2.5-v0.2

```python
# scripts/colqwen_mlx_ingest.py - Update model reference
MODEL_NAME = "vidore/colqwen2.5-v0.2"  # Latest as of Nov 2025
```

Key improvements in v0.2:
- **Dynamic resolution** - No aspect ratio distortion
- **768 image patches** - Higher detail retention
- **Improved multi-vector quality** - Better late interaction scores

### Alternative: ColPali 1.3 + Qwen3-VL

For even better results, consider the dual-model approach:
1. **ColPali 1.3** for retrieval (lighter, faster)
2. **Qwen3-VL-8B** for interpretation (richer understanding)

```python
# Two-stage visual RAG
async def visual_rag(query: str):
    # Stage 1: Fast retrieval with ColPali
    retrieved_pages = await colpali_search(query, top_k=5)
    
    # Stage 2: Deep understanding with Qwen3-VL
    for page in retrieved_pages:
        interpretation = await qwen3_vl_interpret(page.image, query)
        page.interpretation = interpretation
    
    return retrieved_pages
```

---

## Part 5: Performance Optimization

### Memory Management

With 256GB RAM, you can keep multiple models loaded:

```python
# api/services/model_manager.py
"""Intelligent model loading and memory management."""

class ModelManager:
    """Manages model lifecycle for optimal memory usage."""
    
    # Models to keep always loaded (hot)
    HOT_MODELS = ["router", "embeddings"]
    
    # Models to load on demand (warm)
    WARM_MODELS = ["synthesizer", "vision"]
    
    # Estimated memory per model (GB)
    MEMORY_MAP = {
        "router": 20,        # Qwen3-30B-A3B
        "synthesizer": 12,   # Qwen3-14B
        "vision": 8,         # Qwen2.5-VL-7B
        "embeddings": 2,     # nomic-embed-text
    }
    
    def __init__(self, max_memory_gb: int = 200):
        self.max_memory = max_memory_gb
        self.loaded = {}
        self.last_used = {}
    
    async def ensure_loaded(self, model_key: str):
        """Ensure a model is loaded, evicting if necessary."""
        if model_key in self.loaded:
            self.last_used[model_key] = time.time()
            return
        
        # Check if we need to evict
        current_usage = sum(
            self.MEMORY_MAP.get(k, 10) 
            for k in self.loaded
        )
        needed = self.MEMORY_MAP.get(model_key, 10)
        
        while current_usage + needed > self.max_memory:
            # Evict least recently used (not in HOT_MODELS)
            evictable = [
                k for k in self.loaded 
                if k not in self.HOT_MODELS
            ]
            if not evictable:
                raise MemoryError("Cannot load model, hot models exceed limit")
            
            lru = min(evictable, key=lambda k: self.last_used.get(k, 0))
            await self.unload(lru)
            current_usage -= self.MEMORY_MAP.get(lru, 10)
        
        await self._load_model(model_key)
```

### Batch Processing

For high-throughput scenarios:

```python
async def batch_search(queries: List[str]) -> List[SearchResult]:
    """Process multiple queries efficiently."""
    
    # Batch embed queries
    query_embeddings = await batch_embed(queries)
    
    # Parallel search
    results = await asyncio.gather(*[
        search_with_embedding(emb) 
        for emb in query_embeddings
    ])
    
    return results
```

### KV Cache for Multi-Turn

```python
class ConversationCache:
    """Maintains KV cache for multi-turn conversations."""
    
    def __init__(self, max_conversations: int = 100):
        self.caches = {}
        self.max = max_conversations
    
    def get_or_create(self, session_id: str):
        if session_id not in self.caches:
            if len(self.caches) >= self.max:
                # Evict oldest
                oldest = min(self.caches, key=lambda k: self.caches[k].last_access)
                del self.caches[oldest]
            self.caches[session_id] = KVCache()
        return self.caches[session_id]
```

---

## Part 6: Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│                    Streaming NDJSON Consumer                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Qwen-Agent Orchestrator                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Router    │  │ Synthesizer │  │   Vision    │     │   │
│  │  │ Qwen3-30B   │  │  Qwen3-14B  │  │ Qwen3-VL-8B │     │   │
│  │  │   (MoE)     │  │             │  │             │     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │   │
│  │         │                │                │             │   │
│  │         └────────────────┼────────────────┘             │   │
│  │                          │                               │   │
│  │                    MLX Inference                         │   │
│  │                   (Apple Silicon)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐             │
│  │   Fast Vector RAG   │  │     ColQwen RAG     │             │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │             │
│  │  │  AssetManual  │  │  │  │ PDFDocuments  │  │             │
│  │  │  (Weaviate)   │  │  │  │  (Weaviate)   │  │             │
│  │  │ nomic-embed   │  │  │  │ ColQwen2.5    │  │             │
│  │  └───────────────┘  │  │  └───────────────┘  │             │
│  └─────────────────────┘  └─────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Static Assets                                │
│  ┌─────────────────────┐  ┌─────────────────────┐             │
│  │   Preview PNGs      │  │   Source PDFs       │             │
│  │ static/previews/*   │  │  static/manuals/*   │             │
│  └─────────────────────┘  └─────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | MLX Integration | MLX service, model loading, basic generation |
| 2 | **Elysia Patterns** (NEW) | Environment, TreeData, Error objects, base Tool class |
| 3 | LLM Routing + Decision Tree | Decision nodes, `is_tool_available`, reasoning output |
| 4 | Answer Synthesis | Context formatting, streaming generation |
| 5 | Visual Interpretation | VLM integration, page understanding |
| 6 | Qwen-Agent Integration | Full framework, MCP tools |
| 7 | Testing & Optimization | Benchmarks, memory tuning, edge cases |

### Updated Priority Order

Based on Elysia research, implement in this order:

1. **Environment & State (Week 2)** - Foundation for everything else
2. **Tool Base Class (Week 2)** - `is_tool_available`, `run_if_true`, Error handling
3. **Decision Tree Structure (Week 3)** - Node navigation, branch logic
4. **LLM Decision Agent (Week 3)** - Replace keyword matching
5. **MLX Inference (Week 1, parallel)** - Can be done alongside Week 2-3
6. **Visual Interpretation (Week 5)** - Once foundation is solid

---

## Part 8: Quick Wins (Do Today)

### 1. Install MLX (5 minutes)

```bash
conda activate vsm-hva
pip install mlx mlx-lm
```

### 2. Test MLX with Qwen (10 minutes)

```python
# test_mlx.py
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
response = generate(
    model, tokenizer,
    prompt="What is the capital of France?",
    max_tokens=100
)
print(response)
```

### 3. Serve MLX as OpenAI-compatible API (5 minutes)

```bash
# Start MLX server (OpenAI-compatible)
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8000
```

Then your existing code can hit `http://localhost:8000/v1/chat/completions`.

---

## References

### Research Sources (Nov 2025)

1. **Apple MLX Research** - "Exploring LLMs with MLX and Neural Accelerators in M5 GPU"
   - URL: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
   - Key finding: 4x TTFT speedup, native quantization

2. **Qwen3-VL** - Latest multimodal model (Sept 2025)
   - GitHub: https://github.com/QwenLM/Qwen3-VL
   - Sizes: 8B, 32B, 235B-A22B (MoE)

3. **ColPali ICLR 2025** - Visual Document Retrieval
   - Paper: https://proceedings.iclr.cc/paper_files/paper/2025/ColPali
   - Latest: ColQwen2.5-v0.2

4. **Qwen-Agent Framework** - Official agent framework
   - Docs: https://qwen.readthedocs.io/en/latest/framework/qwen_agent.html
   - Features: MCP support, thinking mode, streaming

5. **Weaviate Late Interaction** - Multi-vector retrieval
   - Blog: https://weaviate.io/blog/late-interaction-overview

6. **Elysia Agentic RAG Framework** - Decision tree based agent system
   - Blog: https://weaviate.io/blog/elysia-agentic-rag
   - GitHub: https://github.com/weaviate/elysia
   - Key patterns: Decision trees, tool availability, environment state

---

## Part 9: Elysia Architecture Patterns (NEW)

Based on deep research into [Weaviate's Elysia framework](https://weaviate.io/blog/elysia-agentic-rag), here are **production-proven patterns** to adopt for your agent:

### 9.1 Decision Tree Architecture (vs. Simple Routing)

**Current Problem:** Your agent treats all tools as equally available at all times.

**Elysia Pattern:** Pre-defined tree of decision nodes with branches.

```python
# Elysia-style decision tree structure
"""
📁 Base (root)
├── 🔧 fast_vector_search (for text queries)
│   └── 🔧 relevance_check (verify results)
├── 🔧 colqwen_visual_search (for visual queries)
│   └── 🔧 visual_interpretation (VLM analysis)
├── 🔧 summarize (when environment has data)
└── 🔧 text_response (end conversation)
"""

@dataclass
class DecisionNode:
    """Node in the decision tree."""
    id: str
    tools: List[Tool]
    children: Dict[str, "DecisionNode"]  # tool_id -> next node
    instruction: str  # How to choose between tools
    
class DecisionTree:
    """Elysia-style decision tree for tool routing."""
    
    def __init__(self):
        self.root = self._build_tree()
        self.current_node = self.root
        
    def _build_tree(self) -> DecisionNode:
        """Build the decision tree structure."""
        
        # Post-search verification node
        relevance_check = DecisionNode(
            id="relevance_check",
            tools=[RelevanceCheckTool()],
            children={},
            instruction="Verify retrieved results are relevant to query"
        )
        
        visual_interpret = DecisionNode(
            id="visual_interpret", 
            tools=[VisualInterpretationTool()],
            children={},
            instruction="Use VLM to interpret the retrieved page images"
        )
        
        # Root node with main tools
        root = DecisionNode(
            id="root",
            tools=[
                FastVectorSearchTool(),
                ColQwenSearchTool(),
                SummarizeTool(),
                TextResponseTool()
            ],
            children={
                "fast_vector_search": relevance_check,
                "colqwen_visual_search": visual_interpret
            },
            instruction="""Choose the best tool for the user's query:
- fast_vector_search: Text-based factual queries
- colqwen_visual_search: Visual/diagram queries  
- summarize: When environment has retrieved data
- text_response: End conversation with final answer"""
        )
        
        return root
```

### 9.2 Tool Availability Control

**Key Elysia Pattern:** Tools can be conditionally available based on environment state.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TreeData:
    """Central state object passed to all tools."""
    user_prompt: str
    environment: "Environment"
    conversation_history: List[Dict]
    tasks_completed: Dict[str, List[float]]  # tool -> execution times
    errors: List[str]
    
class Tool(ABC):
    """Base tool class with Elysia-style availability control."""
    
    name: str
    description: str
    end: bool = False  # Can this tool end the conversation?
    
    async def is_tool_available(
        self,
        tree_data: TreeData,
        **kwargs
    ) -> bool:
        """Override to control when this tool is available.
        
        Examples:
        - Summarize only when environment is non-empty
        - Visual interpretation only after ColQwen search
        """
        return True
    
    async def run_if_true(
        self,
        tree_data: TreeData,
        **kwargs
    ) -> tuple[bool, dict]:
        """Override to auto-trigger this tool under certain conditions.
        
        Returns:
            (should_run: bool, inputs: dict)
            
        Examples:
        - Auto-summarize when environment exceeds token limit
        - Auto-cleanup when error count > 3
        """
        return False, {}
    
    @abstractmethod
    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        **kwargs
    ):
        """Execute the tool."""
        pass


class SummarizeTool(Tool):
    """Tool that's only available when environment has data."""
    
    name = "summarize"
    description = "Summarize retrieved information for the user"
    end = True
    
    async def is_tool_available(self, tree_data: TreeData, **kwargs) -> bool:
        """Only available when environment is non-empty."""
        return not tree_data.environment.is_empty()
    
    async def run_if_true(self, tree_data: TreeData, **kwargs) -> tuple[bool, dict]:
        """Auto-trigger when environment gets too large."""
        token_count = tree_data.environment.estimate_tokens()
        if token_count > 50000:
            return True, {"max_tokens": 1000}
        return False, {}


class VisualInterpretationTool(Tool):
    """Tool that's only available after ColQwen search."""
    
    name = "visual_interpretation"
    description = "Interpret page images using VLM"
    
    async def is_tool_available(self, tree_data: TreeData, **kwargs) -> bool:
        """Only available after ColQwen has retrieved pages."""
        return tree_data.environment.find("colqwen_visual_search") is not None
```

### 9.3 Environment as Central State

**Elysia Pattern:** All retrieved data lives in a structured environment that tools can read/write.

```python
from typing import Dict, List, Any, Optional

class Environment:
    """Persistent state across all tool executions.
    
    Structure:
    {
        tool_name: {
            result_name: [
                {"objects": [...], "metadata": {...}},
                {"objects": [...], "metadata": {...}},
            ]
        }
    }
    """
    
    def __init__(self):
        self.environment: Dict[str, Dict[str, List[Dict]]] = {}
        self.hidden_environment: Dict[str, Any] = {}  # Not shown to LLM
        
    def add(self, tool_name: str, result: "Result"):
        """Add a result from a tool execution."""
        if tool_name not in self.environment:
            self.environment[tool_name] = {}
        
        name = result.name or "default"
        if name not in self.environment[tool_name]:
            self.environment[tool_name][name] = []
            
        self.environment[tool_name][name].append({
            "objects": result.objects,
            "metadata": result.metadata
        })
    
    def find(
        self, 
        tool_name: str, 
        name: str = None, 
        index: int = None
    ) -> Optional[Dict]:
        """Retrieve data from environment."""
        if tool_name not in self.environment:
            return None
        
        if name is None:
            return self.environment[tool_name]
        
        if name not in self.environment[tool_name]:
            return None
            
        data = self.environment[tool_name][name]
        
        if index is not None:
            return data[index] if index < len(data) else None
        return data
    
    def is_empty(self) -> bool:
        """Check if environment has any data."""
        return len(self.environment) == 0
    
    def estimate_tokens(self) -> int:
        """Estimate token count of environment content."""
        import json
        return len(json.dumps(self.environment)) // 4
    
    def to_llm_context(self, max_tokens: int = 10000) -> str:
        """Format environment for LLM consumption."""
        import json
        content = json.dumps(self.environment, indent=2)
        if len(content) // 4 > max_tokens:
            # Truncate oldest entries
            content = content[:max_tokens * 4] + "\n... (truncated)"
        return content


@dataclass
class Result:
    """Result object yielded from tools."""
    objects: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None
    name: str = None
    llm_message: str = None  # Message shown to LLM about this result
    
    def __post_init__(self):
        self.metadata = self.metadata or {}
```

### 9.4 Self-Healing Errors

**Elysia Pattern:** Errors don't crash the system—they inform the LLM to retry or try different approaches.

```python
@dataclass
class Error:
    """Error object that informs the LLM without crashing."""
    message: str
    recoverable: bool = True
    suggestion: str = None

class FastVectorSearchTool(Tool):
    """Example tool with self-healing error handling."""
    
    async def __call__(self, tree_data: TreeData, inputs: dict, **kwargs):
        query = inputs.get("query", "")
        
        try:
            results = await self._search(query)
            
            if not results:
                # Yield error to inform LLM
                yield Error(
                    message=f"No results found for query: {query}",
                    recoverable=True,
                    suggestion="Try broadening the search terms or using different keywords"
                )
                return
                
            yield Result(
                objects=results,
                metadata={"query": query, "count": len(results)},
                name="search_results"
            )
            
        except WeaviateConnectionError as e:
            yield Error(
                message=f"Weaviate connection failed: {e}",
                recoverable=False,
                suggestion="Check if Weaviate is running: docker compose up -d"
            )
        except InvalidFilterError as e:
            # LLM might have generated invalid filter
            yield Error(
                message=f"Invalid filter generated: {e}",
                recoverable=True,
                suggestion="Try searching without filters or use different filter values"
            )
```

### 9.5 Decision Agent with Reasoning

**Elysia Pattern:** Each decision node has an LLM agent that outputs structured decisions with reasoning.

```python
from pydantic import BaseModel, Field
from typing import Literal
import dspy

class DecisionOutput(BaseModel):
    """Structured output from decision agent."""
    tool: str = Field(description="Name of tool to use")
    inputs: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = Field(description="Why this tool was chosen")
    should_end: bool = Field(default=False, description="End after this tool?")
    impossible: bool = Field(default=False, description="Is task impossible?")

class DecisionAgent:
    """LLM agent that decides which tool to use."""
    
    def __init__(self, model: str = "qwen3-14b"):
        self.lm = dspy.LM(model)
        
    async def decide(
        self,
        tree_data: TreeData,
        available_tools: List[Tool],
        node_instruction: str
    ) -> DecisionOutput:
        """Make a decision about which tool to use."""
        
        # Build prompt with full context
        prompt = f"""You are a decision agent in a technical manual search system.

CURRENT STATE:
- User query: {tree_data.user_prompt}
- Environment: {tree_data.environment.to_llm_context(max_tokens=5000)}
- Previous errors: {tree_data.errors[-3:] if tree_data.errors else "None"}
- Tasks completed: {list(tree_data.tasks_completed.keys())}

AVAILABLE TOOLS:
{self._format_tools(available_tools)}

INSTRUCTION:
{node_instruction}

Decide which tool to use. Output JSON:
{{"tool": "...", "inputs": {{...}}, "reasoning": "...", "should_end": false, "impossible": false}}
"""
        
        response = await self.lm.acomplete(prompt)
        return self._parse_response(response)
    
    def _format_tools(self, tools: List[Tool]) -> str:
        lines = []
        for t in tools:
            lines.append(f"- {t.name}: {t.description}")
            if t.end:
                lines.append(f"  (Can end conversation)")
        return "\n".join(lines)
```

### 9.6 Updated Agent Implementation

Here's how to integrate these Elysia patterns into your existing `agent.py`:

```python
# api/services/agent.py - Elysia-style refactor
"""Agent service using Elysia-style decision tree patterns."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, AsyncGenerator, Optional
import asyncio

@dataclass
class TreeData:
    user_prompt: str
    environment: Environment = field(default_factory=Environment)
    conversation_history: List[Dict] = field(default_factory=list)
    tasks_completed: Dict[str, List[float]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    collection_names: List[str] = field(default_factory=list)

class ElysiaStyleAgent:
    """Agent orchestrator using Elysia decision tree patterns."""
    
    def __init__(self):
        self.tree = DecisionTree()
        self.decision_agent = DecisionAgent()
        self.max_iterations = 10
        
    async def run(
        self,
        query: str,
        collection_names: List[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the decision tree until completion."""
        
        tree_data = TreeData(
            user_prompt=query,
            collection_names=collection_names or []
        )
        
        current_node = self.tree.root
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Get available tools at current node
            available_tools = [
                t for t in current_node.tools
                if await t.is_tool_available(tree_data)
            ]
            
            # Check for auto-trigger tools
            for tool in available_tools:
                should_run, inputs = await tool.run_if_true(tree_data)
                if should_run:
                    yield {"type": "auto_trigger", "tool": tool.name}
                    async for result in self._execute_tool(tool, tree_data, inputs):
                        yield result
            
            # Get decision from LLM
            decision = await self.decision_agent.decide(
                tree_data,
                available_tools,
                current_node.instruction
            )
            
            yield {
                "type": "decision",
                "tool": decision.tool,
                "reasoning": decision.reasoning
            }
            
            if decision.impossible:
                yield {
                    "type": "impossible",
                    "message": "Task cannot be completed with available data"
                }
                break
            
            # Execute chosen tool
            tool = next((t for t in available_tools if t.name == decision.tool), None)
            if tool:
                async for result in self._execute_tool(tool, tree_data, decision.inputs):
                    yield result
                
                # Navigate to next node if exists
                if decision.tool in current_node.children:
                    current_node = current_node.children[decision.tool]
                elif decision.should_end and tool.end:
                    yield {"type": "complete"}
                    break
                else:
                    # Stay at current node for another iteration
                    pass
    
    async def _execute_tool(
        self,
        tool: Tool,
        tree_data: TreeData,
        inputs: dict
    ) -> AsyncGenerator[Dict, None]:
        """Execute a tool and handle results/errors."""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async for output in tool(tree_data, inputs):
                if isinstance(output, Result):
                    tree_data.environment.add(tool.name, output)
                    yield {
                        "type": "result",
                        "tool": tool.name,
                        "data": output.objects,
                        "metadata": output.metadata
                    }
                elif isinstance(output, Error):
                    tree_data.errors.append(output.message)
                    yield {
                        "type": "error",
                        "message": output.message,
                        "recoverable": output.recoverable,
                        "suggestion": output.suggestion
                    }
                else:
                    yield {"type": "output", "content": output}
                    
        except Exception as e:
            tree_data.errors.append(str(e))
            yield {"type": "exception", "message": str(e)}
        
        # Track execution time
        elapsed = asyncio.get_event_loop().time() - start_time
        if tool.name not in tree_data.tasks_completed:
            tree_data.tasks_completed[tool.name] = []
        tree_data.tasks_completed[tool.name].append(elapsed)
```

### 9.7 Key Differences from Current Implementation

| Aspect | Current `agent.py` | Elysia-Style |
|--------|-------------------|--------------|
| Routing | Keyword matching | LLM decision with reasoning |
| Tool availability | All tools always available | Conditional via `is_tool_available` |
| Auto-triggers | None | `run_if_true` for conditions |
| State management | Scattered | Centralized `Environment` |
| Error handling | Exceptions | `Error` objects inform LLM |
| Tree navigation | None | Node-based traversal |
| Transparency | Black box | Full reasoning exposed |

---

## Appendix: Model Comparison

### Routing Model Options

| Model | Active Params | Quality | Speed | Memory |
|-------|--------------|---------|-------|--------|
| Qwen3-4B | 4B | Good | Fast | 3 GB |
| Qwen3-8B | 8B | Very Good | Medium | 5 GB |
| Qwen3-14B | 14B | Excellent | Medium | 9 GB |
| Qwen3-30B-A3B | 3B (active) | Excellent | Fast | 17 GB |

**Recommendation:** Qwen3-30B-A3B gives 30B quality with 3B inference cost.

### Vision Model Options

| Model | Params | DocVQA | ChartQA | Memory |
|-------|--------|--------|---------|--------|
| Qwen2.5-VL-3B | 3B | 89.3 | 82.1 | 3 GB |
| Qwen2.5-VL-7B | 7B | 92.1 | 85.7 | 5 GB |
| Qwen2.5-VL-32B | 32B | 95.2 | 89.3 | 18 GB |
| Qwen3-VL-8B | 8B | 93.5 | 87.2 | 6 GB |

**Recommendation:** Qwen2.5-VL-7B for balance, Qwen3-VL-8B when MLX version available.

---

**Document Version:** 2.0  
**Author:** AI Research Assistant  
**Last Major Update:** 2025-11-25 (Added Elysia architecture patterns)  
**Next Review:** 2025-12-25

