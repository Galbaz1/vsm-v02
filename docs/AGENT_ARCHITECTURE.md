# Agent Architecture

**Last Updated:** 2025-11-25

```mermaid
flowchart TB
    subgraph Data["📄 Data Ingestion"]
        PDF[PDF Documents]
        PDF --> LandingAI[LandingAI ADE]
        PDF --> Previews[generate_previews.py]
        LandingAI --> JSON[Parsed JSON]
    end

    subgraph Weaviate["🗄️ Weaviate Local (8080)"]
        subgraph Collections
            AM[AssetManual<br/>text chunks]
            PD[PDFDocuments<br/>page images]
        end
        subgraph Metadata
            CD[Chunk Data<br/>bbox, section]
            CM[ColQwen Vectors<br/>1024×128 multi-vec]
        end
    end

    JSON --> AM
    Previews --> PD

    subgraph Context["📋 Context (TreeData)"]
        UP[User Prompt]
        CH[Conversation History]
        ENV[Environment<br/>retrieved results]
        ERR[Errors]
        TC[Tasks Completed]
    end

    UQ[/"💬 User Query"/] --> Context

    subgraph Tree["🌳 Decision Tree"]
        DA[Decision Agent<br/>gpt-oss-120B]
        
        subgraph Loop["Self-Healing Loop"]
            TS[Tool Selection]
            TE[Tool Execution]
            RA[Result Assessment]
            RE[Reasoning & Eval]
            TS --> TE --> RA --> RE --> TS
        end
        
        DA --> Loop
    end

    subgraph Tools["🔧 Available Tools"]
        FVS[FastVectorSearch<br/>~0.5s]
        CQS[ColQwenSearch<br/>~3s]
        VIT[VisualInterpret<br/>Qwen3-VL-8B]
        TR[TextResponse<br/>end=true]
    end

    Context --> DA
    Tools --> Loop
    
    FVS -.-> AM
    CQS -.-> PD
    VIT -.-> MLX[MLX Server<br/>8000]

    subgraph Response["📤 Response"]
        RD[Retrieved Data<br/>+ bbox overlays]
        TS2[Text Summary<br/>streaming tokens]
    end

    Loop --> Response
    Response --> UF[/"👍 User Feedback"/]

    style Data fill:#1a3a2a,stroke:#4ade80
    style Weaviate fill:#1a2a4a,stroke:#60a5fa
    style Context fill:#2a2a3a,stroke:#a78bfa
    style Tree fill:#1a3a3a,stroke:#22d3d3
    style Tools fill:#3a2a1a,stroke:#fb923c
    style Response fill:#1a3a3a,stroke:#22d3d3
    style Loop fill:#0a2a2a,stroke:#14b8a6
```

---

## Models

| Role | Model | Runtime | Memory |
|------|-------|---------|--------|
| Decision + Synthesis | gpt-oss:120b | Ollama | ~65GB |
| Vision | Qwen3-VL-8B | MLX | ~8GB |
| Retrieval | ColQwen2.5-v0.2 | PyTorch | ~4GB |
| Embeddings | nomic-embed-text | Ollama | ~2GB |

---

## Tool Availability

| Tool | is_available | run_if_true | end |
|------|--------------|-------------|-----|
| FastVectorSearch | always | no | no |
| ColQwenSearch | always | no | no |
| VisualInterpret | after ColQwen | no | no |
| TextResponse | env not empty | no | **yes** |
| Summarize | env not empty | env > 50K tokens | no |

---

## NDJSON Stream

```json
{"type": "decision", "tool": "fast_vector_search", "reasoning": "..."}
{"type": "status", "message": "Searching..."}
{"type": "result", "objects": [...], "metadata": {...}}
{"type": "error", "message": "...", "recoverable": true}
{"type": "token", "content": "The"}
{"type": "complete"}
```

---

**See:** [ARCHITECTURE.md](ARCHITECTURE.md) for full system docs
