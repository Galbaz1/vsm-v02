# Tool Routing Architecture

**Last Updated:** 2025-11-26  
**Module:** `api/services/tools/`

---

## Overview

Tools are the **execution layer** of the agentic RAG system. They:
1. Perform actual retrieval/generation operations
2. Use providers transparently (local or cloud)
3. Control their own availability based on context
4. Yield structured outputs for the environment

The key insight is that **tools don't change based on mode** - they use whichever provider is configured.

---

## Tool Architecture

```mermaid
flowchart TB
    subgraph AgentLayer["Agent Layer"]
        Agent[AgentOrchestrator]
        Decision["Decision Module<br/>(DSPy)"]
    end

    subgraph ToolRegistry["Tool Registry"]
        direction TB
        FVS["FastVectorSearchTool"]
        CQS["ColQwenSearchTool"]
        HYB["HybridSearchTool"]
        VIS["VisualInterpretationTool"]
        TXT["TextResponseTool"]
        SUM["SummarizeTool"]
    end

    subgraph ToolBase["Tool Base Class"]
        IsAvailable["is_tool_available()"]
        RunIfTrue["run_if_true()"]
        Call["__call__()"]
    end

    subgraph Providers["Provider Layer"]
        GetEmb["get_embeddings()"]
        GetVDB["get_vectordb()"]
        GetLLM["get_llm()"]
        GetVLM["get_vlm()"]
    end

    Agent --> Decision
    Decision --> FVS
    Decision --> CQS
    Decision --> HYB
    Decision --> VIS
    Decision --> TXT
    Decision --> SUM

    FVS --> ToolBase
    CQS --> ToolBase
    HYB --> ToolBase
    VIS --> ToolBase
    TXT --> ToolBase
    SUM --> ToolBase

    FVS --> GetEmb
    FVS --> GetVDB
    CQS --> GetEmb
    CQS --> GetVDB
    HYB --> GetEmb
    HYB --> GetVDB
    VIS --> GetVLM
    TXT --> GetLLM
    SUM --> GetLLM

    style AgentLayer fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style ToolRegistry fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Providers fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## Tool Availability Flow

```mermaid
flowchart TD
    Start([Agent Decision Loop])
    
    Start --> GetTools["Get all registered tools"]
    GetTools --> LoopStart{{"For each tool"}}
    
    LoopStart --> CheckAvail["await tool.is_tool_available(tree_data)"]
    
    CheckAvail --> IsAvail{{"Available?"}}
    
    IsAvail -->|Yes| AddToList["Add to available_tools"]
    IsAvail -->|No| SkipTool["Skip tool"]
    
    AddToList --> NextTool
    SkipTool --> NextTool
    
    NextTool{{"More tools?"}}
    NextTool -->|Yes| LoopStart
    NextTool -->|No| PassToDecision["Pass available_tools to Decision Module"]
    
    PassToDecision --> DecisionMakes["Decision Module selects tool"]
    DecisionMakes --> Execute["Execute selected tool"]
    
    Execute --> End([Continue Loop])

    style Start fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style End fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style CheckAvail fill:#fff9c4,stroke:#f57f17,stroke-width:2px
```

---

## Tool Definitions

### FastVectorSearchTool

```mermaid
flowchart LR
    subgraph Inputs
        Query["query: str"]
        Limit["limit: int = 5"]
        Collection["collection: str = 'AssetManual'"]
    end

    subgraph Tool["FastVectorSearchTool"]
        Embed["Embed query"]
        Search["Vector search"]
        Format["Format results"]
    end

    subgraph Providers
        Embedder["get_embeddings()"]
        VectorDB["get_vectordb()"]
    end

    subgraph Output
        Result["Result<br/>objects: List[dict]<br/>metadata: {query, count, time_ms}"]
    end

    Query --> Embed
    Limit --> Search
    Collection --> Search
    
    Embed --> Embedder
    Embedder --> Search
    Search --> VectorDB
    VectorDB --> Format
    Format --> Result

    style Tool fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Providers fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

**Availability:** Always available (default search tool)

**Mode Behavior:**
| Mode | Embedder | Vector DB |
|------|----------|-----------|
| Local | bge-m3 via Ollama | Weaviate Docker |
| Cloud | Jina v4 Dense | Weaviate Cloud |

---

### ColQwenSearchTool

```mermaid
flowchart LR
    subgraph Inputs
        Query["query: str"]
        TopK["top_k: int = 3"]
    end

    subgraph Tool["ColQwenSearchTool"]
        EmbedQuery["Embed query<br/>(multi-vector)"]
        MultiVecSearch["Multi-vector search<br/>(late interaction)"]
        Format["Format with<br/>preview URLs"]
    end

    subgraph Providers
        Embedder["get_embeddings()"]
        VectorDB["get_vectordb()"]
    end

    subgraph Output
        Result["Result<br/>objects: List[{page, score, preview_url}]<br/>metadata: {query, time_ms}"]
    end

    Query --> EmbedQuery
    TopK --> MultiVecSearch
    
    EmbedQuery --> Embedder
    Embedder --> MultiVecSearch
    MultiVecSearch --> VectorDB
    VectorDB --> Format
    Format --> Result

    style Tool fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Providers fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

**Availability:** Always available

**Mode Behavior:**
| Mode | Embedder | Vector DB | Multi-Vector Support |
|------|----------|-----------|---------------------|
| Local | ColQwen2.5-v0.2 (PyTorch) | Weaviate Docker | Native multi-vector |
| Cloud | **Serverless Worker** (Jina v4) | Weaviate Cloud | Named vectors |

> **Note:** Jina API does NOT expose multi-vector for images. Cloud mode requires a self-hosted worker running Jina v4 with `return_multivector=True`.

---

### HybridSearchTool

```mermaid
flowchart TB
    subgraph Inputs
        Query["query: str"]
        TextLimit["text_limit: int = 3"]
        VisualLimit["visual_limit: int = 2"]
    end

    subgraph Tool["HybridSearchTool"]
        direction TB
        Parallel["asyncio.gather()"]
        
        subgraph VectorPath["Vector Search Path"]
            VEmbed["Embed query"]
            VSearch["Hybrid search<br/>(vector + BM25)"]
        end
        
        subgraph VisualPath["Visual Search Path"]
            MEmbed["Embed query<br/>(multi-vector)"]
            MSearch["Multi-vector search"]
        end
        
        Merge["Merge + Deduplicate"]
    end

    subgraph Output
        TextResult["text_results: List"]
        VisualResult["visual_results: List"]
    end

    Query --> Parallel
    TextLimit --> VSearch
    VisualLimit --> MSearch
    
    Parallel --> VectorPath
    Parallel --> VisualPath
    
    VectorPath --> Merge
    VisualPath --> Merge
    
    Merge --> TextResult
    Merge --> VisualResult

    style Tool fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style VectorPath fill:#fff3e0,stroke:#ef6c00,stroke-width:1px
    style VisualPath fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px
```

**Availability:** Always available

**Key Feature:** Runs both searches **in parallel** using `asyncio.gather()` for efficiency.

---

### VisualInterpretationTool

```mermaid
flowchart LR
    subgraph Inputs
        ImagePath["image_path: str"]
        Prompt["prompt: str"]
    end

    subgraph Tool["VisualInterpretationTool"]
        CheckEnv["Check environment<br/>for ColQwen results"]
        SelectPage["Select best<br/>matching page"]
        Interpret["VLM interpretation"]
    end

    subgraph Providers
        VLM["get_vlm()"]
    end

    subgraph Output
        Result["Result<br/>objects: [{interpretation, page, image_path}]"]
    end

    ImagePath --> SelectPage
    Prompt --> Interpret
    
    CheckEnv --> SelectPage
    SelectPage --> Interpret
    Interpret --> VLM
    VLM --> Result

    style Tool fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Providers fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

**Availability:** Only when environment contains ColQwen results

```python
async def is_tool_available(self, tree_data, **kwargs) -> bool:
    # Only available if we have visual search results to interpret
    colqwen_data = tree_data.environment.find("colqwen_search")
    return colqwen_data is not None and len(colqwen_data) > 0
```

**Mode Behavior:**
| Mode | VLM Provider |
|------|--------------|
| Local | Qwen3-VL-8B via MLX |
| Cloud | Gemini 2.5 Flash (Vision) |

---

### TextResponseTool

```mermaid
flowchart LR
    subgraph Inputs
        IncludeSources["include_sources: bool = True"]
    end

    subgraph Tool["TextResponseTool"]
        BuildContext["Build context from<br/>environment"]
        BuildPrompt["Build response prompt"]
        Generate["Stream LLM generation"]
    end

    subgraph Providers
        LLM["get_llm()"]
    end

    subgraph Output
        Response["Response<br/>text: str<br/>sources: List[{page, manual}]"]
    end

    IncludeSources --> BuildContext
    BuildContext --> BuildPrompt
    BuildPrompt --> Generate
    Generate --> LLM
    LLM --> Response

    style Tool fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Providers fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

**Availability:** When environment has data OR after at least one iteration

```python
async def is_tool_available(self, tree_data, **kwargs) -> bool:
    return not tree_data.environment.is_empty() or tree_data.num_iterations > 0
```

**End Tool:** Yes - this tool ends the conversation with a final response.

---

### SummarizeTool

```mermaid
flowchart LR
    subgraph Inputs
        MaxLength["max_length: int = 200"]
    end

    subgraph Tool["SummarizeTool"]
        GatherData["Gather all objects<br/>from environment"]
        BuildPrompt["Build summary prompt"]
        Generate["LLM summarization"]
    end

    subgraph Providers
        LLM["get_llm()"]
    end

    subgraph Output
        Response["Response<br/>text: str (summary)"]
    end

    MaxLength --> BuildPrompt
    GatherData --> BuildPrompt
    BuildPrompt --> Generate
    Generate --> LLM
    LLM --> Response

    style Tool fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Providers fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

**Availability:** Only when environment has data

**Auto-Trigger:** When environment exceeds 30,000 estimated tokens

```python
async def run_if_true(self, tree_data, **kwargs) -> Tuple[bool, dict]:
    if tree_data.environment.estimate_tokens() > 30000:
        return True, {"max_length": 500}
    return False, {}
```

---

## Tool Selection Matrix

```mermaid
flowchart TD
    Query([User Query])
    
    Query --> Analysis{{"Query Analysis"}}
    
    Analysis -->|"Visual keywords:<br/>diagram, figure, schematic"| VisualPath
    Analysis -->|"Table keywords:<br/>bit code, menu, specs"| HybridPath
    Analysis -->|"Simple factual:<br/>what is, define, voltage"| FastPath
    Analysis -->|"Complex/ambiguous"| HybridPath
    
    subgraph VisualPath["Visual Path"]
        ColQwen["ColQwenSearchTool"]
        ColQwen --> HasResults{{"Results found?"}}
        HasResults -->|Yes| Visual["VisualInterpretationTool"]
        HasResults -->|No| Fallback1["HybridSearchTool"]
    end
    
    subgraph HybridPath["Hybrid Path"]
        Hybrid["HybridSearchTool"]
        Hybrid --> EnoughData{{"Sufficient data?"}}
        EnoughData -->|Yes| Respond1["TextResponseTool"]
        EnoughData -->|No| MoreSearch["Additional search"]
    end
    
    subgraph FastPath["Fast Path"]
        FastVec["FastVectorSearchTool"]
        FastVec --> CheckData{{"Data adequate?"}}
        CheckData -->|Yes| Respond2["TextResponseTool"]
        CheckData -->|No| UpgradeHybrid["HybridSearchTool"]
    end

    Visual --> FinalResponse["TextResponseTool"]
    Respond1 --> End([Response])
    Respond2 --> End
    FinalResponse --> End

    style Query fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style End fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style Analysis fill:#fff9c4,stroke:#f57f17,stroke-width:2px
```

---

## Tool Implementation Pattern

```python
# api/services/tools/search_tools.py
from api.services.tools.base import Tool
from api.core.providers import get_embeddings, get_vectordb
from api.schemas.agent import Result, Error, Status

class FastVectorSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="fast_vector_search",
            description=(
                "Search AssetManual collection using fast vector similarity. "
                "Best for: definitions, specifications, simple factual queries."
            ),
            status="Searching text content...",
            inputs={
                "query": {
                    "description": "Search query text",
                    "type": "str",
                    "required": True,
                },
                "limit": {
                    "description": "Maximum results",
                    "type": "int",
                    "default": 5,
                },
            },
        )
    
    async def __call__(self, tree_data, inputs, **kwargs):
        import time
        start = time.time()
        
        yield Status("Embedding query...")
        
        # Get providers (mode-agnostic)
        embedder = get_embeddings()
        vectordb = get_vectordb()
        
        query = inputs.get("query", tree_data.user_prompt)
        limit = inputs.get("limit", 5)
        
        try:
            # Embed query
            query_vector = await embedder.embed_query(query)
            
            yield Status("Searching AssetManual...")
            
            # Search
            results = await vectordb.hybrid_search(
                collection="AssetManual",
                query=query,
                query_vector=query_vector,
                limit=limit,
                alpha=0.5,  # Balance vector and BM25
            )
            
            elapsed = (time.time() - start) * 1000
            
            yield Result(
                objects=results,
                name="AssetManual",
                metadata={
                    "query": query,
                    "count": len(results),
                    "time_ms": elapsed,
                },
                llm_message=f"Found {len(results)} text results for '{query}'",
            )
            
        except Exception as e:
            yield Error(
                message=f"Search failed: {str(e)}",
                recoverable=True,
                suggestion="Try a different query or check service availability",
            )
```

---

## Output Types

```mermaid
classDiagram
    class ToolOutput {
        <<interface>>
        +to_frontend(query_id) dict
    }
    
    class Result {
        +objects: List[dict]
        +name: str
        +metadata: dict
        +llm_message: str
    }
    
    class Error {
        +message: str
        +recoverable: bool
        +suggestion: str
    }
    
    class Response {
        +text: str
        +sources: List[dict]
    }
    
    class Status {
        +message: str
    }
    
    ToolOutput <|-- Result
    ToolOutput <|-- Error
    ToolOutput <|-- Response
    ToolOutput <|-- Status
```

---

## Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Provider Agnostic** | Tools call `get_*()` factories, not specific implementations |
| **Conditional Availability** | `is_tool_available()` controls when tool appears in decisions |
| **Auto-Trigger** | `run_if_true()` bypasses LLM decision when conditions met |
| **Structured Output** | All tools yield `Result`, `Error`, `Response`, or `Status` |
| **Environment Integration** | Results automatically added to `tree_data.environment` |
| **LLM Context** | `llm_message` field provides context for next decision |

