# Cloud Migration Architecture Overview

**Last Updated:** 2025-11-26  
**Status:** Architecture Freeze  
**Target Version:** VSM v0.3

---

## Executive Summary

This document outlines the architectural changes required to support both **local** and **cloud** deployment modes for VSM's dual-pipeline agentic RAG system. The goal is modular, mode-switchable infrastructure that preserves the existing Elysia-inspired architecture.

**Key Challenge:** The public Jina API does not support multi-vector output for images (required for high-precision visual search).
**Solution:** A **Serverless Custom Worker** (on RunPod/Modal) will host the Jina v4 model to provide this specific capability in the cloud, ensuring parity with the local ColQwen pipeline.

---

## Current vs Target Architecture

### Mode Comparison

| Component | Local Mode | Cloud Mode |
|-----------|------------|------------|
| **LLM (Decision + Response)** | gpt-oss:120b (Ollama) | Gemini 2.5 Flash |
| **VLM (Visual Interpretation)** | Qwen3-VL-8B (MLX) | Gemini 2.5 Flash (Vision) |
| **Text Embeddings** | bge-m3 (Ollama) | Jina Embeddings v4 (Dense) |
| **Visual Embeddings** | ColQwen2.5-v0.2 (PyTorch) | **Jina v4 Multi-Vector (Serverless)** |
| **Vector Database** | Weaviate (Docker) | Weaviate Cloud |
| **RAM Required** | ~80GB | ~4GB |
| **Latency Profile** | High throughput, no network | Network-dependent, lower memory |

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph UserLayer["User Interface Layer"]
        UI[Next.js Frontend]
        API[FastAPI Backend]
    end

    subgraph OrchestratorLayer["Orchestrator Layer"]
        Agent[AgentOrchestrator]
        TreeData[TreeData State]
        Env[Environment Store]
    end

    subgraph DSPyLayer["DSPy Layer"]
        Sig[Signatures]
        ModLocal["Compiled Module<br/>local/decision.json"]
        ModCloud["Compiled Module<br/>cloud/decision.json"]
    end

    subgraph ToolLayer["Tool Layer"]
        FVS[FastVectorSearchTool]
        CQS[ColQwenSearchTool]
        HYB[HybridSearchTool]
        VIS[VisualInterpretationTool]
        TXT[TextResponseTool]
    end

    subgraph ProviderLayer["Provider Layer"]
        direction LR
        subgraph LocalProviders["Local Providers"]
            OllamaLLM[OllamaLLM]
            MLXVLM[MLXVLM]
            OllamaEmb[OllamaEmbeddings]
            WeaviateLocal[WeaviateLocal]
        end
        subgraph CloudProviders["Cloud Providers"]
            GeminiLLM[GeminiLLM]
            GeminiVLM[GeminiVLM]
            JinaEmb[JinaEmbeddings]
            JinaWorker[ServerlessWorker]
            WeaviateCloud[WeaviateCloud]
        end
    end

    UI --> API
    API --> Agent
    Agent --> TreeData
    Agent --> Env
    Agent --> Sig
    
    Sig --> ModLocal
    Sig --> ModCloud
    
    Agent --> FVS
    Agent --> CQS
    Agent --> HYB
    Agent --> VIS
    Agent --> TXT
    
    FVS --> OllamaEmb
    FVS --> JinaEmb
    FVS --> WeaviateLocal
    FVS --> WeaviateCloud
    
    CQS --> OllamaEmb
    CQS --> JinaWorker
    CQS --> WeaviateLocal
    CQS --> WeaviateCloud
    
    VIS --> MLXVLM
    VIS --> GeminiVLM
    
    TXT --> OllamaLLM
    TXT --> GeminiLLM

    style LocalProviders fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style CloudProviders fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style DSPyLayer fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style ToolLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style JinaWorker fill:#ffccbc,stroke:#d84315,stroke-width:2px,stroke-dasharray: 5 5
```

---

## Layer Responsibilities

```mermaid
flowchart TB
    subgraph L1["Layer 1: User Interface"]
        direction LR
        NextJS["Next.js 16"]
        FastAPI["FastAPI"]
    end

    subgraph L2["Layer 2: Orchestrator"]
        direction LR
        Agent["AgentOrchestrator"]
        TreeData["TreeData"]
        Environment["Environment"]
    end

    subgraph L3["Layer 3: DSPy Signatures"]
        direction LR
        Signatures["Model-agnostic contracts"]
        CompiledLocal["Compiled (local)"]
        CompiledCloud["Compiled (cloud)"]
    end

    subgraph L4["Layer 4: Tools"]
        direction LR
        SearchTools["Search Tools"]
        ResponseTools["Response Tools"]
        VisualTools["Visual Tools"]
    end

    subgraph L5["Layer 5: Providers"]
        direction LR
        LLM["LLM Provider"]
        VLM["VLM Provider"]
        Embed["Embedding Provider"]
        VDB["VectorDB Provider"]
    end

    L1 --> L2
    L2 --> L3
    L2 --> L4
    L4 --> L5
    L3 --> L5

    style L1 fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style L2 fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style L3 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style L4 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style L5 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

| Layer | Responsibility | Changes Between Modes |
|-------|---------------|----------------------|
| **User Interface** | HTTP requests, streaming | None |
| **Orchestrator** | Decision loop, state management | None |
| **DSPy Signatures** | Define I/O contracts | None (signatures shared) |
| **DSPy Compiled** | Optimized prompts | Different per model |
| **Tools** | Execute retrieval/generation | None (use providers) |
| **Providers** | Connect to services | Different implementations |

---

## Decision Flow

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant Agent as AgentOrchestrator
    participant DSPy as DSPy Module
    participant Tool as Selected Tool
    participant Provider as Provider Layer

    User->>API: POST /agentic_search?query=...
    API->>Agent: run(query)
    
    loop Decision Loop (max 10 iterations)
        Agent->>Agent: Get available tools
        Agent->>DSPy: Decision Module(query, tools, environment)
        Note over DSPy: Uses compiled prompts<br/>for current mode
        DSPy-->>Agent: Decision(tool_name, inputs)
        
        Agent->>Tool: Execute(tree_data, inputs)
        Tool->>Provider: get_embeddings() / get_llm() / ...
        Note over Provider: Returns local or cloud<br/>based on VSM_MODE
        Provider-->>Tool: Provider instance
        Tool->>Provider: embed_query() / search() / generate()
        Provider-->>Tool: Results
        Tool-->>Agent: Yield Result/Error/Response
        
        alt should_end = true
            Agent-->>API: Stream complete
        end
    end
    
    API-->>User: NDJSON stream
```

---

## Data Flow by Mode

### Local Mode Data Flow

```mermaid
flowchart LR
    subgraph Input["User Query"]
        Query["'Show wiring diagram'"]
    end

    subgraph Embedding["Local Embedding"]
        Ollama["Ollama<br/>bge-m3"]
    end

    subgraph Search["Local Search"]
        Weaviate["Weaviate<br/>Docker :8080"]
    end

    subgraph VLM["Local VLM"]
        MLX["MLX<br/>Qwen3-VL-8B"]
    end

    subgraph LLM["Local LLM"]
        GPT["Ollama<br/>gpt-oss:120b"]
    end

    subgraph Output["Response"]
        Response["Streaming NDJSON"]
    end

    Query --> Ollama
    Ollama --> Weaviate
    Weaviate --> MLX
    MLX --> GPT
    GPT --> Response

    style Input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

### Cloud Mode Data Flow

```mermaid
flowchart LR
    subgraph Input["User Query"]
        Query["'Show wiring diagram'"]
    end

    subgraph Embedding["Cloud Embedding"]
        Jina["Jina API<br/>v4 Dense"]
        Worker["Serverless Worker<br/>v4 Multi-Vector"]
    end

    subgraph Search["Cloud Search"]
        WeaviateCloud["Weaviate Cloud"]
    end

    subgraph VLM["Cloud VLM"]
        GeminiVision["Gemini 2.5 Flash<br/>(Vision)"]
    end

    subgraph LLM["Cloud LLM"]
        GeminiLLM["Gemini 2.5 Flash<br/>(thinking=4096)"]
    end

    subgraph Output["Response"]
        Response["Streaming NDJSON"]
    end

    Query --> Jina
    Query --> Worker
    Jina --> WeaviateCloud
    Worker --> WeaviateCloud
    WeaviateCloud --> GeminiVision
    GeminiVision --> GeminiLLM
    GeminiLLM --> Response

    style Input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Worker fill:#ffccbc,stroke:#d84315,stroke-width:2px
```

---

## Separation of Concerns Summary

```mermaid
flowchart TD
    subgraph Changes["What Changes Between Modes"]
        Providers["Provider implementations"]
        CompiledPrompts["DSPy compiled prompts"]
        Config["Configuration values"]
    end

    subgraph NoChanges["What Stays The Same"]
        Orchestrator["Agent orchestrator logic"]
        Tools["Tool definitions"]
        Signatures["DSPy signatures"]
        Environment["Environment/TreeData"]
        API["API endpoints"]
        Frontend["Frontend code"]
    end

    style Changes fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style NoChanges fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## Migration Path

```mermaid
flowchart TD
    Start([Current State])
    
    Start --> Phase1["Phase 1: Provider Layer"]
    
    subgraph Phase1Details["Create api/core/providers/"]
        P1A["Define interfaces (base.py)"]
        P1B["Move local code to local/"]
        P1C["Implement cloud providers"]
        P1D["Create Serverless Worker"]
    end
    
    Phase1 --> Phase1Details
    Phase1Details --> Phase2
    
    Phase2["Phase 2: DSPy Integration"]
    
    subgraph Phase2Details["Create api/prompts/"]
        P2A["Define signatures"]
        P2B["Create VSMChainOfThought"]
        P2C["Optimize for local model"]
        P2D["Optimize for cloud model"]
    end
    
    Phase2 --> Phase2Details
    Phase2Details --> Phase3
    
    Phase3["Phase 3: Tool Updates"]
    
    subgraph Phase3Details["Update api/services/tools/"]
        P3A["Replace direct calls with providers"]
        P3B["Update availability logic"]
        P3C["Test both modes"]
    end
    
    Phase3 --> Phase3Details
    Phase3Details --> Phase4
    
    Phase4["Phase 4: Cloud Ingestion"]
    
    subgraph Phase4Details["Create scripts/cloud_ingest.py"]
        P4A["Jina API text ingestion"]
        P4B["Serverless worker visual ingestion"]
        P4C["Weaviate Cloud setup"]
    end
    
    Phase4 --> Phase4Details
    Phase4Details --> End([Both Modes Working])

    style Start fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    style End fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
```

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [02-provider-layer.md](./02-provider-layer.md) | Provider interfaces and implementations |
| [03-dspy-prompt-optimization.md](./03-dspy-prompt-optimization.md) | DSPy signatures and compilation |
| [04-tool-routing.md](./04-tool-routing.md) | Tool availability and selection |
| [05-search-pipelines.md](./05-search-pipelines.md) | Text and visual search flows |
| [06-configuration-guide.md](./06-configuration-guide.md) | Environment setup |

---

## Key Takeaways

1. **Single codebase, two modes**: `VSM_MODE` environment variable controls everything
2. **Serverless Worker**: Solves the "missing API" problem for visual search
3. **Providers are thin**: Just connection adapters, no business logic
4. **Tools are mode-agnostic**: They call `get_*()` factories
5. **DSPy handles prompt differences**: Compiled modules per model
6. **Orchestrator unchanged**: Agent logic works identically in both modes
