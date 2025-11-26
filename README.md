# VSM - Visual Search Manual

A **local-first agentic RAG system** for searching technical asset manuals with visual grounding. Runs entirely on Apple Silicon (M3 256GB Mac Studio).

## Features

- 🔍 **Dual RAG Pipelines**: Fast text search (bge-m3) + Visual search (ColQwen2.5)
- 🤖 **Agentic LLM**: gpt-oss:120b makes tool selection decisions
- 📄 **Visual Grounding**: Bounding boxes overlay on page images
- ⚡ **Streaming Responses**: Real-time NDJSON output
- 🏠 **100% Local**: No cloud dependencies, all inference on-device

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Mac Studio (Host)                        │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────────┐│
│  │  Native Ollama  │    │         Application             ││
│  │  (0.0.0.0:11434)│    │  - API (FastAPI, port 8001)     ││
│  │                 │    │  - Frontend (Next.js, port 3000)││
│  │  Models:        │    └─────────────────────────────────┘│
│  │  - gpt-oss:120b │                                       │
│  │  - bge-m3       │    host.docker.internal:11434         │
│  └────────┬────────┘              ▼                        │
│           │           ┌─────────────────┐                  │
│           └──────────▶│ Weaviate (8080) │ Docker           │
│                       │ 1,614 documents │                  │
│                       └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decision**: Ollama runs **natively** on macOS (not in Docker) to access full 256GB RAM and Metal GPU acceleration. Weaviate runs in Docker and connects to Native Ollama via `host.docker.internal`.

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- [Ollama](https://ollama.ai) installed natively
- Docker Desktop
- Node.js 18+
- Conda (with `vsm-hva` environment)

### 1. Pull Required Models

```bash
# LLM for decision-making and text generation (~65GB)
ollama pull gpt-oss:120b

# Embeddings for text search (~1.2GB)
ollama pull bge-m3
```

### 2. Start All Services

```bash
conda activate vsm-hva
./scripts/start.sh
```

This will:
1. Kill any conflicting Ollama/Weaviate containers from other projects
2. Start Native Ollama with optimized settings
3. Start Weaviate (Docker) connected to Native Ollama
4. Start the FastAPI backend
5. Start the Next.js frontend

### 3. Open the App

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8001/docs

### 4. Stop All Services

```bash
./scripts/stop.sh
```

## Ingesting Documents

### Text-based RAG (Fast Search)

```bash
# Parse PDF with LandingAI ADE
python scripts/parse_with_landingai.py data/manual.pdf data/output.json

# Generate page previews
python scripts/generate_previews.py data/manual.pdf static/previews/manual

# Ingest into Weaviate
python scripts/weaviate_ingest_manual.py data/output.json "Manual Name"
```

### Visual RAG (ColQwen - Optional)

```bash
# Requires preview PNGs from above
python scripts/colqwen_ingest.py "Manual Name"
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /search?query=...` | Fast vector search |
| `GET /agentic_search?query=...` | Agentic streaming search (SSE) |
| `GET /healthz` | Health check |

## Project Structure

```
vsm-v02/
├── api/                    # FastAPI backend
│   ├── services/agent.py   # Decision tree orchestrator
│   ├── services/llm.py     # Ollama/MLX clients
│   └── services/tools/     # Tool implementations
├── frontend/               # Next.js 16 + React 19
├── scripts/
│   ├── start.sh            # Start all services
│   ├── stop.sh             # Stop all services
│   └── weaviate_ingest_manual.py
├── docker-compose.yml      # Weaviate only (no Ollama!)
└── static/previews/        # Page PNG images
```

## Models & Memory

| Model | Size | Purpose |
|-------|------|---------|
| gpt-oss:120b | ~65GB | LLM (decisions + generation) |
| bge-m3 | ~1.2GB | Text embeddings (8K context) |
| ColQwen2.5-v0.2 | ~4GB | Visual retrieval |
| Qwen3-VL-8B | ~8GB | Visual interpretation (MLX) |

**Total**: ~78GB, leaving ~178GB for KV cache on 256GB Mac Studio.

## Troubleshooting

### "404 Not Found" for Ollama API

**Cause**: Wrong Ollama instance running (Docker instead of Native).

**Fix**: Run `./scripts/start.sh` - it automatically kills conflicting containers.

### "lookup ollama on 127.0.0.11:53: no such host"

**Cause**: Weaviate container from another project with wrong config.

**Fix**: 
```bash
docker stop $(docker ps -q --filter name=weaviate)
docker rm $(docker ps -aq --filter name=weaviate)
./scripts/start.sh
```

### Embedding failures during ingestion

**Cause**: Model swapping instability in Ollama.

**Fix**: Pre-warm the embedding model:
```bash
curl -s http://localhost:11434/api/embed -d '{"model":"bge-m3","input":"warmup","keep_alive":"15m"}'
```

## Logs

```bash
tail -f /tmp/vsm-ollama.log    # Ollama
tail -f /tmp/vsm-api.log       # Backend API
tail -f /tmp/vsm-frontend.log  # Frontend
```

## Debugging Agent Issues

Query traces are auto-saved to `logs/query_traces/` for every `/agentic_search` call.

### When a Query Fails

```bash
# 1. Find the trace
ls -la logs/query_traces/

# 2. Run intelligent analysis (uses Gemini 3 Pro's 1M context)
python scripts/analyze_with_llm.py --gemini-only <trace_id_prefix>

# 3. Apply the suggested fix, then verify
python scripts/run_benchmark.py --output results.json
```

The analyzer loads the entire codebase + all traces into Gemini 3 Pro and returns a concise diagnosis with exact file:line to fix.

See [.cursor/README.md](.cursor/README.md) for the full debugging toolkit documentation.

## Documentation

- [Development Workflow](docs/WORKFLOW.md) - How to debug and develop
- [System Architecture](docs/ARCHITECTURE.md)
- [Agent Flow Diagram](docs/agent_diagram.md)
- [RAG Pipeline Explained](docs/RAG_PIPELINE_EXPLAINED.md)
- [ColQwen Ingestion](docs/COLQWEN_INGESTION_EXPLAINED.md)

## Tech Stack

- **LLM**: gpt-oss:120b (Native Ollama)
- **Embeddings**: bge-m3 (Native Ollama)
- **Vector DB**: Weaviate 1.34 (Docker)
- **Backend**: Python 3.12, FastAPI
- **Frontend**: Next.js 16, React 19, Tailwind v4
- **Parsing**: LandingAI ADE
