# Agent Architecture

**Last Updated:** 2025-11-25

```mermaid
flowchart TB
    subgraph Data[Data Ingestion]
        PDF[PDF Documents]
        PDF --> LandingAI[LandingAI ADE]
        PDF --> Previews[generate_previews]
        LandingAI --> JSON[Parsed JSON]
    end

    subgraph Weaviate[Weaviate Local 8080]
        AM[AssetManual]
        PD[PDFDocuments]
    end

    JSON --> AM
    Previews --> PD

    subgraph Context[Context TreeData]
        UP[User Prompt]
        CH[Conversation History]
        ENV[Environment]
        ERR[Errors]
        TC[Tasks Completed]
    end

    UQ([User Query]) --> Context

    subgraph Tree[Decision Tree]
        DA[Decision Agent gpt-oss-120B]
        
        subgraph Loop[Self-Healing Loop]
            TS[Tool Selection]
            TE[Tool Execution]
            RA[Result Assessment]
            RE[Reasoning]
        end
        
        DA --> TS
        TS --> TE
        TE --> RA
        RA --> RE
        RE --> TS
    end

    subgraph Tools[Available Tools]
        FVS[FastVectorSearch]
        CQS[ColQwenSearch]
        VIT[VisualInterpret]
        TR[TextResponse]
    end

    Context --> DA
    Tools -.-> TE
    
    FVS -.-> AM
    CQS -.-> PD
    VIT -.-> MLX[MLX VLM 8000]

    subgraph Response[Response]
        RD[Retrieved Data]
        TS2[Text Summary]
    end

    RA --> Response
    Response --> UF([User Feedback])
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
