# VSM - Visual Search Manual

A **hybrid agentic RAG system** for searching technical asset manuals with visual grounding. Supports both local (Mac Studio) and cloud (MacBook Air) deployment modes.

## Features

- 🔍 **Dual RAG Pipelines**: Text search + Visual search (separate pipelines)
- 🤖 **Agentic LLM**: DSPy-powered tool selection (model-agnostic)
- 📄 **Visual Grounding**: Bounding boxes overlay on page images
- ⚡ **Streaming Responses**: Real-time NDJSON output
- 🔄 **Mode-Switchable**: `VSM_MODE=local|cloud` via Provider abstraction

## Deployment Modes

| Component | Local (Mac Studio) | Cloud (MacBook Air) |
|-----------|-------------------|---------------------|
| **LLM** | gpt-oss:120b (Ollama) | Gemini 2.5 Flash |
| **VLM** | Qwen3-VL-8B (MLX) | Gemini 2.5 Flash |
| **Embeddings** | bge-m3 (Ollama) | Jina v4 |
| **Visual Search** | ColQwen2.5-v0.2 | Weaviate + Jina CLIP v2 |
| **Vector DB** | Weaviate (Docker) | Weaviate Cloud |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Mac Studio (Local Mode)                  │
│  ┌─────────────────┐    ┌─────────────────────────────────┐│
│  │  Native Ollama  │    │  API (8001) / Frontend (3000)   ││
│  │  (0.0.0.0:11434)│    └─────────────────────────────────┘│
│  │  - gpt-oss:120b │                                       │
│  │  - bge-m3       │    host.docker.internal:11434         │
│  └────────┬────────┘              ▼                        │
│           │           ┌─────────────────┐                  │
│           └──────────▶│ Weaviate (8080) │ Docker           │
│                       └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  MacBook Air (Cloud Mode)                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  API (8001) / Frontend (3000)                           ││
│  └────────┬────────────────────────────────────────────────┘│
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ Gemini API  │  │  Jina API   │  │  Weaviate Cloud   │   │
│  │ (LLM + VLM) │  │ (Embeddings)│  │ (+ Jina CLIP v2)  │   │
│  └─────────────┘  └─────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Conda (with `vsm-hva` environment)
- **Local Mode**: [Ollama](https://ollama.ai), Docker Desktop
- **Cloud Mode**: API keys for Gemini, Jina, Weaviate Cloud

### Option A: Local Mode (Mac Studio)

```bash
./scripts/start.sh
```

### Option B: Cloud Mode (MacBook Air)

```bash
./scripts/start_cloud.sh
```

> **Note:** Cloud mode requires `.env` with `GEMINI_API_KEY`, `JINA_API_KEY`, `WEAVIATE_URL`, `WEAVIATE_API_KEY`

## Environment Variables

### Mode Selection

```bash
VSM_MODE=local   # Default: use Ollama + MLX + ColQwen
VSM_MODE=cloud   # Use Gemini + Jina + Weaviate Cloud
```

### Local Mode

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_EMBED_MODEL=bge-m3
MLX_VLM_BASE_URL=http://localhost:8000
WEAVIATE_LOCAL_URL=http://localhost:8080
```

### Cloud Mode

```bash
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_THINKING_BUDGET=-1  # -1=dynamic, 0=off, 1-24576=tokens
JINA_API_KEY=jina_...
WEAVIATE_URL=https://xxx.weaviate.cloud
WEAVIATE_API_KEY=xxx
```

## Ingesting Documents

### Local Mode

```bash
# Parse PDF with LandingAI ADE
python scripts/parse_with_landingai.py data/manual.pdf data/output.json

# Generate page previews
python scripts/generate_previews.py data/manual.pdf static/previews/manual

# Ingest text into Weaviate
python scripts/weaviate_ingest_manual.py data/output.json "Manual Name"

# Ingest visuals (ColQwen)
python scripts/colqwen_ingest.py "Manual Name"
```

### Cloud Mode

```bash
# Ingest both text and visuals to Weaviate Cloud
export VSM_MODE=cloud
python scripts/cloud_ingest.py --text data/output.json "Manual Name"
python scripts/cloud_ingest.py --visual data/manual.pdf "Manual Name"
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /search?query=...` | Fast hybrid search |
| `GET /agentic_search?query=...` | Agentic streaming search (SSE) |
| `GET /healthz` | Health check |

## Project Structure

```
vsm-v02/
├── api/
│   ├── core/
│   │   ├── config.py           # Settings + VSM_MODE
│   │   ├── providers/          # LLM/VLM/Embed/VectorDB abstractions
│   │   └── dspy_config.py      # DSPy LM configuration
│   ├── prompts/                # DSPy signatures
│   ├── services/
│   │   ├── agent.py            # Decision tree orchestrator
│   │   └── tools/              # Tool implementations
│   └── endpoints/              # FastAPI routes
├── frontend/                   # Next.js 16 + React 19
├── scripts/
│   ├── start.sh                # Start local services
│   ├── stop.sh                 # Stop services
│   ├── cloud_ingest.py         # Cloud ingestion
│   └── weaviate_ingest_manual.py
├── docs/cloud-migration/       # Architecture docs
└── static/previews/            # Page PNG images
```

## Models & Memory (Local Mode)

| Model | Size | Purpose |
|-------|------|---------|
| gpt-oss:120b | ~65GB | LLM (decisions + generation) |
| bge-m3 | ~1.2GB | Text embeddings (8K context) |
| ColQwen2.5-v0.2 | ~4GB | Visual retrieval |
| Qwen3-VL-8B | ~8GB | Visual interpretation (MLX) |

**Total**: ~78GB, leaving ~178GB for KV cache on 256GB Mac Studio.

## Troubleshooting

### "404 Not Found" for Ollama API (Local)

**Cause**: Wrong Ollama instance running (Docker instead of Native).

**Fix**: Run `./scripts/start.sh` - it automatically kills conflicting containers.

### Search returns 0 results (Cloud)

**Cause**: Data not ingested to cloud Weaviate.

**Fix**: Run `python scripts/cloud_ingest.py ...` to ingest data.

### "GEMINI_API_KEY not set" (Cloud)

**Cause**: Missing environment variables.

**Fix**: Ensure all cloud API keys are exported before starting.

## Logs

```bash
tail -f /tmp/vsm-ollama.log    # Ollama (local)
tail -f /tmp/vsm-api.log       # Backend API
tail -f /tmp/vsm-frontend.log  # Frontend
```

## Debugging Agent Issues

Query traces are auto-saved to `logs/query_traces/` for every `/agentic_search` call.

```bash
# 1. Find the trace
ls -la logs/query_traces/

# 2. Run intelligent analysis (uses Gemini 3 Pro's 1M context)
python scripts/analyze_with_llm.py --gemini-only <trace_id_prefix>
```

## Documentation

- [Cloud Migration Architecture](docs/cloud-migration/README.md)
- [Provider Layer Design](docs/cloud-migration/02-provider-layer.md)
- [Configuration Guide](docs/cloud-migration/06-configuration-guide.md)
- [Agent Flow Diagram](docs/agent_diagram.md)
- [RAG Pipeline Explained](docs/RAG_PIPELINE_EXPLAINED.md)

## Tech Stack

- **LLM**: gpt-oss:120b (local) / Gemini 2.5 Flash (cloud)
- **Embeddings**: bge-m3 (local) / Jina v4 (cloud)
- **Visual Search**: ColQwen2.5 (local) / Jina CLIP v2 (cloud)
- **Vector DB**: Weaviate 1.34
- **Backend**: Python 3.12, FastAPI, DSPy
- **Frontend**: Next.js 16, React 19, Tailwind v4
- **Parsing**: LandingAI ADE
