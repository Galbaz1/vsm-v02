# VSM Agent Architecture

```mermaid
flowchart TB
    %% ==================== DATA INGESTION ====================
    subgraph Data["Data Ingestion"]
        PDF[(PDF Documents)]
        PDF --> LandingAI[LandingAI ADE]
        PDF --> Previews[generate_previews.py]
        LandingAI --> JSON[Parsed JSON]
    end

    %% ==================== WEAVIATE ====================
    subgraph Weaviate["Weaviate :8080"]
        AM[(AssetManual<br/>nomic-embed-text)]
        PD[(PDFDocuments<br/>ColQwen multi-vector)]
    end

    JSON --> AM
    Previews --> PD

    %% ==================== CONTEXT ====================
    subgraph Context["TreeData Context"]
        UP[User Prompt]
        CH[Conversation History]
        ENV[Environment<br/>tool results accumulate]
        ERR[Errors<br/>recoverable + suggestions]
    end

    UQ([User Query]) --> Context

    %% ==================== DECISION TREE ====================
    subgraph Tree["Decision Tree"]
        DA{Decision Agent<br/>gpt-oss-120B}
        
        subgraph Loop["Self-Healing Loop"]
            direction LR
            TS[Tool Selection] --> TE[Tool Execution]
            TE --> RA[Result Assessment]
            RA --> RE[Reasoning]
            RE --> TS
        end
        
        DA --> TS
    end

    %% ==================== TOOLS ====================
    subgraph Tools["Available Tools"]
        FVS[FastVectorSearchTool]
        CQS[ColQwenSearchTool]
        VIT[VisualInterpretationTool]
        SUM[SummarizeTool]
        TR[TextResponseTool]
    end

    Context --> DA
    Tools -.-> TE
    
    %% Tool connections to services
    FVS -.-> AM
    CQS -.-> PD
    VIT -.-> MLX[MLX VLM :8000<br/>Qwen3-VL-8B]

    %% ==================== STREAMING OUTPUT ====================
    subgraph Response["Streaming NDJSON"]
        ST[status]
        DC[decision]
        RS[result]
        ER[error]
    end

    RA --> Response
    Response --> FE([Frontend :3000])

    %% ==================== STYLING ====================
    classDef database fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef process fill:#f5f5f5,stroke:#424242,stroke-width:1px
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef tool fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef external fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef stream fill:#ffebee,stroke:#c62828,stroke-width:1px
    classDef terminal fill:#eeeeee,stroke:#616161,stroke-width:2px

    class AM,PD,PDF database
    class DA decision
    class FVS,CQS,VIT,SUM,TR tool
    class MLX external
    class ST,DC,RS,ER stream
    class UQ,FE terminal
```

## Tool Availability Rules

| Tool | `is_tool_available` Condition |
|------|------------------------------|
| **FastVectorSearchTool** | Always available |
| **ColQwenSearchTool** | Always available |
| **VisualInterpretationTool** | `env["ColQwenSearchTool"]` has results |
| **SummarizeTool** | Environment > 50K tokens |
| **TextResponseTool** | Environment non-empty |

## Streaming Payload Types

```json
{"type": "status", "data": {"phase": "searching", "tool": "FastVectorSearchTool"}}
{"type": "decision", "data": {"tool": "ColQwenSearchTool", "reasoning": "..."}}
{"type": "result", "data": {"objects": [...], "llm_message": "Found 5 pages"}}
{"type": "error", "data": {"message": "...", "recoverable": true}}
```
