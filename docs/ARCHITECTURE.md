# VSM Demo v02 - System Architecture

**Last Updated:** 2025-11-25

This document provides the authoritative, up-to-date architecture overview of the VSM Demo v02 system.

---

## System Overview

VSM Demo v02 is a **dual-pipeline agentic RAG system** for searching technical asset manuals with visual grounding. It combines text-based semantic search with multimodal visual search.

### Key Components

```mermaid
graph TB
    subgraph Data Ingestion
        PDF[PDF Documents] --> LandingAI[LandingAI ADE Parser]
        PDF --> Previews[generate_previews.py]
        LandingAI --> JSON[Parsed JSON]
        JSON --> WeaviateIngest[weaviate_ingest_manual.py]
        WeaviateIngest --> AssetManual[Weaviate: AssetManual Collection]
        
        PDF --> ColQwenIngest[colqwen_ingest.py]
        Previews --> ColQwenIngest
        ColQwenIngest --> PDFDocs[Weaviate: PDFDocuments Collection]
    end
    
    subgraph Backend
        AssetManual --> FastVector[Fast Vector Search]
        PDFDocs --> ColQwenSearch[ColQwen Late Interaction]
        FastVector --> Agent[Agent Orchestrator]
        ColQwenSearch --> Agent
        Agent --> Streaming[Streaming NDJSON API]
    end
    
    subgraph Frontend
        Streaming --> NextJS[Next.js 16 Frontend]
        NextJS --> UI[React 19 UI Components]
    end
```

---

## Two RAG Pipelines

### Pipeline 1: Text-Based RAG (Regular)

**Purpose:** Fast semantic search over text chunks

| Stage | Tool/Component | Details |
|-------|---------------|---------|
| **Parsing** | LandingAI ADE | Extracts text, tables, figures with layout awareness |
| **Chunking** | `weaviate_ingest_manual.py` | Creates chunks with metadata (page, bbox, section) |
| **Embedding** | Ollama `nomic-embed-text` | Single 768-dim vector per chunk |
| **Storage** | Weaviate `AssetManual` collection | ~811 chunks per 132-page PDF |
| **Search** | `near_text` vector search | Cosine similarity, ~0.5s query time |

**Source:** [docs/RAG_PIPELINE_EXPLAINED.md](file:///Users/lab/Documents/vsm_demo_v02/docs/RAG_PIPELINE_EXPLAINED.md)

### Pipeline 2: Multimodal ColQwen RAG

**Purpose:** Visual grounding for diagrams, charts, spatial queries

| Stage | Tool/Component | Details |
|-------|---------------|---------|
| **Image conversion** | `generate_previews.py` | PDF → PNGs (reused for both pipelines) |
| **Embedding** | ColQwen2.5 | ~1024 vectors × 128-dim per page (multi-vector) |
| **Storage** | Weaviate `PDFDocuments` collection | 132 pages per PDF |
| **Search** | Late interaction (MaxSim) | Patch-level matching, ~3-5s query time |

**Source:** [docs/COLQWEN_INGESTION_EXPLAINED.md](file:///Users/lab/Documents/vsm_demo_v02/docs/COLQWEN_INGESTION_EXPLAINED.md)

---

## Technology Stack

### Infrastructure (Docker)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **Weaviate** | `semitechnologies/weaviate:1.34.0` | 8080 (HTTP), 50051 (gRPC) | Vector database |
| **Ollama** | `ollama/ollama:latest` | 11434 | LLM (gpt-oss) + embeddings |

**Configuration:** [docker-compose.yml](file:///Users/lab/Documents/vsm_demo_v02/docker-compose.yml)

### Models

| Role | Model | Runtime | Active Params | Memory |
|------|-------|---------|---------------|--------|
| **Decision Agent** | gpt-oss:120b | Ollama | 5.1B (MoE) | ~65GB |
| **Complex Tools** | gpt-oss:120b | Ollama | 5.1B | shared |
| **Vision** | Qwen3-VL-8B | MLX | 8B | ~8GB |
| **Retrieval** | ColQwen2.5-v0.2 | PyTorch | 2B | ~4GB |
| **Embeddings** | nomic-embed-text | Ollama | - | ~2GB |

**Why gpt-oss-120B?** Elysia-recommended MoE model (5.1B active parameters) optimized for agentic reasoning. Single model for both decision agent and complex tools simplifies architecture.

### Backend (Python 3.12)

| Component | Technology | Location |
|-----------|------------|----------|
| **API Framework** | FastAPI | [api/main.py](file:///Users/lab/Documents/vsm_demo_v02/api/main.py) |
| **Regular RAG** | [api/services/search.py](file:///Users/lab/Documents/vsm_demo_v02/api/services/search.py) | Text-based vector search |
| **ColQwen RAG** | [api/services/colqwen.py](file:///Users/lab/Documents/vsm_demo_v02/api/services/colqwen.py) | Multimodal visual search |
| **Agent** | [api/services/agent.py](file:///Users/lab/Documents/vsm_demo_v02/api/services/agent.py) | Rule-based orchestrator |
| **Schemas** | Pydantic | [api/schemas/search.py](file:///Users/lab/Documents/vsm_demo_v02/api/schemas/search.py) |

**Environment:** Conda env `vsm-hva`

### Frontend (Node.js 18+)

| Component | Version | Purpose |
|-----------|---------|---------|
| **Framework** | Next.js 16 | App Router, SSR |
| **UI Library** | React 19 | Component rendering |
| **Styling** | Tailwind CSS v4 | Utility-first styling |
| **Components** | shadcn/ui | Accessible component library |
| **Data Fetching** | React Query | Caching, state management |

**Location:** [frontend/](file:///Users/lab/Documents/vsm_demo_v02/frontend)

---

## Data Flow

### Ingestion Flow (One-time setup)

```
1. PDF File (e.g., uk_firmware.pdf)
   ↓
2a. LandingAI ADE → JSON → weaviate_ingest_manual.py → AssetManual collection
2b. generate_previews.py → PNGs → colqwen_ingest.py → PDFDocuments collection
   ↓
3. Both pipelines ready for search
```

### Search Flow (Runtime - Current)

```
User Query → Agent (rule-based)
           ├→ FastVectorSearchTool → Environment
           └→ ColQwenSearchTool → Environment
           → Mock text response
           → Stream NDJSON to frontend
```

### Search Flow (Target - LLM Agent)

```
User Query → Decision Agent (gpt-oss-120B)
           │
           ├─[text query]→ FastVectorSearchTool
           │               └→ Result(objects, llm_message) → Environment
           │
           ├─[visual query]→ ColQwenSearchTool
           │                 └→ Result(pages, llm_message) → Environment
           │                     │
           │                     └→ VisualInterpretationTool (Qwen3-VL)
           │                         └→ Result(interpretation) → Environment
           │
           └─[has results]→ TextResponseTool
                            └→ Streaming response to frontend
```

**Key patterns:** 
- Environment accumulates results across tool calls
- Each Result has `llm_message` for decision agent context
- Tools expose `is_tool_available` for conditional routing

---

## API Endpoints

### Production Endpoints (Port 8001)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/search` | GET | Regular vector search (fast) |
| `/agentic_search` | GET | Streaming agentic search (progressive) |
| `/healthz` | GET | Health check |
| `/static/manuals/*.pdf` | GET | PDF files |
| `/static/previews/*/page-*.png` | GET | Preview images |

**Documentation:** Auto-generated Swagger at `http://localhost:8001/docs`

---

## Scripts Reference

### Ingestion Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| [parse_with_landingai.py](file:///Users/lab/Documents/vsm_demo_v02/scripts/parse_with_landingai.py) | Parse PDF with LandingAI ADE | `python scripts/parse_with_landingai.py <pdf> <output.json>` |
| [generate_previews.py](file:///Users/lab/Documents/vsm_demo_v02/scripts/generate_previews.py) | Create PNG previews | `python scripts/generate_previews.py <pdf> <output_dir>` |
| [weaviate_ingest_manual.py](file:///Users/lab/Documents/vsm_demo_v02/scripts/weaviate_ingest_manual.py) | Ingest to AssetManual | `python scripts/weaviate_ingest_manual.py <json> "<Manual Name>"` |
| [colqwen_ingest.py](file:///Users/lab/Documents/vsm_demo_v02/scripts/colqwen_ingest.py) | Ingest to PDFDocuments | `python scripts/colqwen_ingest.py "<Manual Name>"` |

**Note:** All scripts run from project root. Preview PNGs must exist before ColQwen ingestion (optimized 2025-11-25).

### Search Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| [weaviate_search_manual.py](file:///Users/lab/Documents/vsm_demo_v02/scripts/weaviate_search_manual.py) | CLI search | `python scripts/weaviate_search_manual.py "<query>"` |
| [test_api.py](file:///Users/lab/Documents/vsm_demo_v02/test_api.py) | Test API endpoints | `python test_api.py` |
| [test_both_manuals.py](file:///Users/lab/Documents/vsm_demo_v02/test_both_manuals.py) | Verify ingestion | `python test_both_manuals.py` |

---

## Environment Variables

### Backend (`.env` in project root)

```bash
LANDINGAI_API_KEY=your_key_here  # Required for parse_with_landingai.py
```

### Frontend (`frontend/.env.local`)

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8001
```

---

## Weaviate Collections Schema

### AssetManual (Regular RAG)

| Field | Type | Vectorized | Purpose |
|-------|------|------------|---------|
| `content` | TEXT | ✅ | Chunk text content |
| `manual_name` | TEXT | ✅ | Manual identifier |
| `anchor_id` | TEXT | ❌ | Chunk ID from ADE |
| `chunk_type` | TEXT | ❌ | text/table/figure/title |
| `page_number` | INT | ❌ | Page index (0-based) |
| `bbox` | TEXT | ❌ | JSON bounding box |
| `section_title` | TEXT | ❌ | Section heading |

**Vectorizer:** `text2vec-ollama` (nomic-embed-text)

### PDFDocuments (ColQwen RAG)

| Field | Type | Vectorized | Purpose |
|-------|------|------------|---------|
| `page_id` | INT | ❌ | Page identifier |
| `asset_manual` | TEXT | ❌ | Manual identifier |
| `page_number` | INT | ❌ | Page index (1-based) |
| `image_path` | TEXT | ❌ | Path to preview PNG |
| *vector* | MULTI-VECTOR | ✅ | ~1024 × 128-dim embeddings |

**Vectorizer:** Manual (ColQwen2.5 generates embeddings)

---

## Current Implementation Status

### ✅ Completed (Phase 1-2: Search Pipelines)

- [x] Regular RAG pipeline (LandingAI → Weaviate → FastAPI)
- [x] ColQwen ingestion with multi-vector support
- [x] Streaming NDJSON endpoint (`/agentic_search`)
- [x] Frontend UI with visual grounding + bbox overlay
- [x] Multi-manual support (uk_firmware.pdf, techman.pdf)
- [x] Benchmark: Hit@5 90%, MRR 0.72

### ✅ Completed (Phase 3: Agent Foundation)

- [x] Elysia-style `Environment` class (`api/services/environment.py`)
- [x] `Tool` base class with `is_tool_available`, `run_if_true`
- [x] `FastVectorSearchTool`, `ColQwenSearchTool` wrappers
- [x] `Result`/`Error`/`Response`/`Status` schemas
- [x] Agent refactored to decision tree pattern

### 🚧 In Progress (Phase 4: LLM Integration)

- [ ] gpt-oss-120B via Ollama for decision agent
- [ ] Replace rule-based routing with LLM decisions
- [ ] Implement `TextResponseTool` with real generation

### 📋 Future (Phase 5: Visual Interpretation)

- [ ] Qwen3-VL-8B via MLX for page interpretation
- [ ] `VisualInterpretationTool` after ColQwen search
- [ ] Synthesis combining text + visual results

**See:** [TODO.md](/TODO.md) for detailed sprint items

---

## Deployment Workflow

### Quick Start (Local Development)

```bash
# 1. Start infrastructure
docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text

# 2. Activate Python environment
conda activate vsm-hva
pip install -r requirements.txt

# 3. Ingest documents (example: techman.pdf)
python scripts/parse_with_landingai.py data/techman.pdf output_techman.json
python scripts/generate_previews.py data/techman.pdf static/previews/techman
python scripts/weaviate_ingest_manual.py output_techman.json "Technical Manual"
python scripts/colqwen_ingest.py "Technical Manual"

# 4. Start backend
uvicorn api.main:app --reload --port 8001

# 5. Start frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:3000`  
API docs: `http://localhost:8001/docs`

---

## Key Decisions and Trade-offs

### Why Two Pipelines?

| Aspect | Text RAG | ColQwen RAG |
|--------|----------|-------------|
| **Speed** | Fast (0.5s) | Slower (3-5s) |
| **Use case** | Factual queries | Visual/spatial queries |
| **Storage** | 4 MB per PDF | 68 MB per PDF |
| **Accuracy** | Good for text | Better for diagrams/charts |

**Decision:** Use agent to route queries intelligently rather than always running both.

### Why Elysia Patterns (Not Full Elysia)?

We adopt Elysia's architecture patterns but not the framework itself:

**What We Use:**
- ✅ `Environment` for centralized state across tools
- ✅ `Tool` base class with `is_tool_available`, `run_if_true`
- ✅ `Result`/`Error` objects for typed outputs with `llm_message`
- ✅ Decision tree pattern: Agent → Tools → Environment → Agent

**Why Not `pip install elysia-ai`:**
- ❌ Need custom bbox/preview metadata in Results
- ❌ Elysia uses DSPy/LiteLLM; we want direct Ollama/MLX
- ❌ ColQwen multi-vector retrieval is custom
- ❌ Visual interpretation (VLM) not in Elysia

**Reference:** Elysia cloned to `~/elysia-reference` for pattern reference

### Why Local-First?

All components run locally to:
- 🔒 Keep proprietary manual data private
- 💰 Avoid cloud API costs
- ⚡ Reduce latency (no network round-trips)
- 🛠️ Full control over model versions

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: colpali_engine` | `pip install -r requirements.txt` |
| Weaviate connection refused | `docker compose up -d` |
| Port 8001 in use | Change port: `uvicorn api.main:app --port 8002` |
| ColQwen out of memory | Reduce batch size in `colqwen_ingest.py` |
| LandingAI API error | Check `LANDINGAI_API_KEY` in `.env` |

### Verification Commands

```bash
# Check Weaviate is running
curl http://localhost:8080/v1/meta

# Check Ollama is running
curl http://localhost:11434/api/tags

# Test regular RAG
curl "http://localhost:8001/search?query=voltage&limit=3"

# Test ColQwen (after ingestion)
curl -N "http://localhost:8001/agentic_search?query=show%20diagram"
```

---

## Documentation Map

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](file:///Users/lab/Documents/vsm_demo_v02/README.md) | Quick start guide | New users |
| **ARCHITECTURE.md** (this doc) | System design | Developers |
| [RAG_PIPELINE_EXPLAINED.md](file:///Users/lab/Documents/vsm_demo_v02/docs/RAG_PIPELINE_EXPLAINED.md) | Regular RAG deep-dive | Technical |
| [COLQWEN_INGESTION_EXPLAINED.md](file:///Users/lab/Documents/vsm_demo_v02/docs/COLQWEN_INGESTION_EXPLAINED.md) | ColQwen deep-dive | Technical |
| [TESTING.md](file:///Users/lab/Documents/vsm_demo_v02/TESTING.md) | Testing procedures | QA/Developers |
| [frontend/README.md](file:///Users/lab/Documents/vsm_demo_v02/frontend/README.md) | Frontend details | Frontend devs |

---

## Related Projects

### ColQwen Standalone (Deprecated)

**Location:** `colqwen/` directory

This was an initial standalone FastAPI service (port 8002) for ColQwen. It has been **integrated into the main API** at `api/services/colqwen.py`.

**Status:** The `colqwen/` directory contains legacy notebooks and deployment docs. Ignore for current work.

---

## Contact and Support

For questions or issues:
1. Check this architecture doc first
2. Review specific technical docs linked above
3. Consult code comments in [api/](file:///Users/lab/Documents/vsm_demo_v02/api)
4. Test with provided verification scripts

---

**Version:** 1.0  
**Last Updated:** 2025-11-25  
**Maintainer:** Lab Team
