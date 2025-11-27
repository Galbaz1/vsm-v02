# Search Pipeline Architecture

**Last Updated:** 2025-11-26  
**Modules:** `api/services/search.py`, `api/services/colqwen.py`

---

## Overview

VSM implements a **dual-pipeline** search architecture:
1. **Text RAG Pipeline**: Fast semantic search over chunked text
2. **Visual RAG Pipeline**: Late-interaction multi-vector search over page images

Both pipelines must work identically regardless of local/cloud mode, using the provider abstraction layer.

---

## Pipeline Comparison

| Aspect | Text RAG (Fast Vector) | Visual RAG (ColQwen) |
|--------|----------------------|---------------------|
| **Input** | Text chunks with metadata | Full page images |
| **Embedding** | Single dense vector (1024d) | Multi-vector (~1024 × 128d) |
| **Search** | Cosine similarity + BM25 | MaxSim (late interaction) |
| **Speed** | ~0.5s | ~3-5s |
| **Best For** | Factual queries, definitions | Diagrams, tables, spatial layout |
| **Collection** | AssetManual | PDFDocuments |

---

## Text RAG Pipeline

### Local Mode

```mermaid
sequenceDiagram
    participant Tool as FastVectorSearchTool
    participant Emb as OllamaEmbeddings<br/>(bge-m3)
    participant VDB as WeaviateLocal<br/>(:8080)
    participant Ollama as Ollama API<br/>(:11434)

    Tool->>Emb: embed_query("voltage specs")
    Emb->>Ollama: POST /api/embeddings<br/>model: bge-m3
    Note over Ollama: Generate 1024-dim vector
    Ollama-->>Emb: embedding[1024]
    Emb-->>Tool: query_vector

    Tool->>VDB: hybrid_search(collection="AssetManual",<br/>query="voltage specs",<br/>query_vector=...,<br/>alpha=0.5)
    
    VDB->>VDB: Vector similarity search
    VDB->>VDB: BM25 keyword search
    VDB->>VDB: Fusion (alpha=0.5)
    
    VDB-->>Tool: results[{content, page_number, bbox, ...}]
    Tool-->>Tool: Yield Result
```

### Cloud Mode

```mermaid
sequenceDiagram
    participant Tool as FastVectorSearchTool
    participant Emb as JinaEmbeddings<br/>(jina-embeddings-v4)
    participant VDB as WeaviateCloud
    participant Jina as Jina API<br/>(api.jina.ai)

    Tool->>Emb: embed_query("voltage specs")
    Emb->>Jina: POST /v1/embeddings<br/>model: jina-embeddings-v4<br/>task: retrieval.query<br/>dimensions: 1024
    Note over Jina: Generate 1024-dim vector<br/>optimized for query
    Jina-->>Emb: embedding[1024]
    Emb-->>Tool: query_vector

    Tool->>VDB: hybrid_search(collection="AssetManual",<br/>query="voltage specs",<br/>query_vector=...,<br/>alpha=0.5)
    
    VDB->>VDB: Vector similarity search
    VDB->>VDB: BM25 keyword search
    VDB->>VDB: Fusion (alpha=0.5)
    
    VDB-->>Tool: results[{content, page_number, bbox, ...}]
    Tool-->>Tool: Yield Result
```

---

## Visual RAG Pipeline

### Local Mode

```mermaid
sequenceDiagram
    participant Tool as ColQwenSearchTool
    participant Ret as ColQwenRetriever<br/>(PyTorch)
    participant VDB as WeaviateLocal<br/>(:8080)
    participant GPU as Metal GPU

    Tool->>Ret: search("wiring diagram", top_k=3)
    
    Ret->>GPU: Tokenize + encode query
    Note over GPU: ColQwen2.5-v0.2<br/>Generate ~32 query vectors
    GPU-->>Ret: query_vectors[32][128]

    Ret->>VDB: Fetch all page vectors<br/>from PDFDocuments
    VDB-->>Ret: pages[{vectors[1024][128], page_num, ...}]

    Ret->>GPU: MaxSim computation
    Note over GPU: For each page:<br/>max(query · page_patches)
    GPU-->>Ret: scores[n_pages]

    Ret->>Ret: Sort by score, top_k
    Ret-->>Tool: results[{page, score, image_path}]
    Tool-->>Tool: Yield Result
```

### Cloud Mode (Serverless Worker)

```mermaid
sequenceDiagram
    participant Tool as ColQwenSearchTool
    participant Emb as JinaHybridEmbeddings
    participant VDB as WeaviateCloud
    participant Worker as Serverless Worker

    Tool->>Emb: embed_query_multivec("wiring diagram")
    Emb->>Worker: POST /embed_query<br/>text="wiring diagram"
    Note over Worker: Runs Jina v4 on GPU<br/>Returns multi-vectors
    Worker-->>Emb: query_vectors[n][128]
    Emb-->>Tool: query_vectors

    Tool->>VDB: multi_vector_search(<br/>collection="PDFDocuments",<br/>query_vectors=...,<br/>limit=3)
    
    Note over VDB: Weaviate late-interaction<br/>MaxSim scoring
    
    VDB-->>Tool: results[{page, score, image_path}]
    Tool-->>Tool: Yield Result
```

---

## Hybrid Search Pipeline

```mermaid
flowchart TB
    Query([User Query])
    
    Query --> Split["Split into parallel tasks"]
    
    Split --> TextTask["Text Search Task"]
    Split --> VisualTask["Visual Search Task"]
    
    subgraph TextPipeline["Text Pipeline (asyncio)"]
        TextEmbed["Embed query<br/>(dense)"]
        TextSearch["Hybrid search<br/>AssetManual"]
        TextEmbed --> TextSearch
    end
    
    subgraph VisualPipeline["Visual Pipeline (asyncio)"]
        VisualEmbed["Embed query<br/>(multi-vector)"]
        VisualSearch["Multi-vector search<br/>PDFDocuments"]
        VisualEmbed --> VisualSearch
    end
    
    TextTask --> TextPipeline
    VisualTask --> VisualPipeline
    
    TextPipeline --> Gather["asyncio.gather()"]
    VisualPipeline --> Gather
    
    Gather --> Merge["Merge Results"]
    
    subgraph MergeLogic["Merge Logic"]
    Dedupe["Deduplicate by page"]
    Score["Normalize scores"]
    Combine["Combine text + visual"]
    end
    
    Merge --> MergeLogic
    MergeLogic --> Results([Combined Results])
    
    style Query fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style Results fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style TextPipeline fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style VisualPipeline fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

---

## Ingestion Pipelines

### Text Ingestion (Cloud)

```mermaid
flowchart LR
    subgraph Input["Input"]
    PDF["PDF Document"]
    JSON["LandingAI ADE JSON"]
    end
    
    subgraph Processing["Processing"]
    Parse["Parse JSON chunks"]
    Clean["Clean content"]
    Chunk["Create chunk objects"]
    end
    
    subgraph Embedding["Embedding"]
    JinaEmb["Jina API<br/>task: retrieval.passage"]
    end
    
    subgraph Storage["Storage"]
    WeaviateCloud["Weaviate Cloud"]
    end
    
    PDF --> JSON
    JSON --> Parse
    Parse --> Clean
    Clean --> Chunk
    Chunk --> JinaEmb
    JinaEmb --> WeaviateCloud
    
    style Embedding fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

### Visual Ingestion (Cloud - Serverless)

```mermaid
flowchart LR
    subgraph Input["Input"]
    PDF["PDF Document"]
    end
    
    subgraph Preview["Preview Generation"]
    GenPreviews["generate_previews.py"]
    PNGs["Page PNGs"]
    end
    
    subgraph Embedding["Embedding"]
    Worker["Serverless Worker<br/>Jina v4 Multi-Vector"]
    MultiVec["Multi-vectors"]
    end
    
    subgraph Storage["Storage"]
    WeaviateCloud["Weaviate Cloud<br/>Named vectors"]
    end
    
    PDF --> GenPreviews
    GenPreviews --> PNGs
    PNGs --> Worker
    Worker --> MultiVec
    MultiVec --> WeaviateCloud
    
    style Embedding fill:#ffccbc,stroke:#d84315,stroke-width:2px
```

---

## Cloud Ingestion Script

```python
# scripts/cloud_ingest.py
"""
Cloud-mode ingestion script using Jina Embeddings and Weaviate Cloud.
"""
import asyncio
from pathlib import Path
from api.core.config import get_settings
from api.core.providers import get_embeddings, get_vectordb

async def ingest_page_images(preview_dir: str, manual_name: str):
    """Ingest page images to Weaviate Cloud using Jina Multi-Vectors via Serverless Worker."""
    embedder = get_embeddings()
    vectordb = get_vectordb()
    
    preview_path = Path(preview_dir)
    images = sorted(preview_path.glob("page-*.png"))
    
    print(f"Processing {len(images)} page images...")
    
    for i, image_path in enumerate(images):
        page_num = int(image_path.stem.split("-")[1])
        
        # Get multi-vector embedding from Serverless Worker
        vectors = await embedder.embed_images_multivec([str(image_path)])
        
        # Upsert to Weaviate Cloud
        await vectordb.upsert_multivec(
            collection="PDFDocuments",
            object_data={
                "page_id": i,
                "asset_manual": manual_name,
                "page_number": page_num,
                "image_path": str(image_path),
            },
            vectors=vectors[0],
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Ingested {i + 1}/{len(images)}")
    
    print(f"✅ Ingested {len(images)} pages to Weaviate Cloud")
```
