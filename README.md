# VSM Demo v02 - Technical Asset Manual Search

A local, **dual-pipeline agentic RAG stack** for searching technical asset manuals with visual grounding. Combines fast text-based search with multimodal ColQwen retrieval.

📖 **[Full System Architecture →](docs/ARCHITECTURE.md)**

## Architecture

- **Dual RAG Pipelines**: Fast vector search (Ollama) + Visual search (ColQwen2.5)
- **Parsing**: LandingAI Agentic Document Extraction (ADE) for PDF parsing
- **Vector DB**: Local Weaviate (Docker) with multi-vector support
- **Backend API**: FastAPI with agentic orchestration and streaming responses
- **Frontend**: Next.js 16, React 19, Tailwind CSS v4, and shadcn/ui

## Quick Start

### Prerequisites

- Python 3.12+ (Conda env `vsm-hva`)
- Node.js 18.17+
- Docker & Docker Compose
- LandingAI API key (set in `.env`)

### 1. Start Infrastructure

```bash
# Start Weaviate and Ollama
docker compose up -d

# Pull embedding model
docker compose exec ollama ollama pull nomic-embed-text
```

### 2. Ingest Documents

#### Regular RAG Pipeline (Text-based)

```bash
# Activate conda environment
conda activate vsm-hva

# Parse PDF with LandingAI ADE
python scripts/parse_with_landingai.py data/uk_firmware.pdf output_landingai.json

# Generate page previews (needed for both pipelines)
python scripts/generate_previews.py data/uk_firmware.pdf static/previews/uk_firmware

# Ingest into Weaviate
python scripts/weaviate_ingest_manual.py output_landingai.json "UK Firmware Manual"
```

#### ColQwen RAG Pipeline (Multimodal - Optional)

```bash
# Requires preview PNGs (generated above)
# Downloads ColQwen2.5 model (~8GB) on first run
python scripts/colqwen_ingest.py "UK Firmware Manual"
```

**Note:** ColQwen ingestion takes ~10-15 minutes on first run (model download + embedding generation).

### 3. Start Backend API

```bash
# In one terminal
uvicorn api.main:app --reload --port 8001
```

### 4. Start Frontend

```bash
# In another terminal
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to use the search interface.

## Project Structure

```
vsm_demo_v02/
├── api/                    # FastAPI backend
│   └── main.py            # Search API endpoints
├── frontend/              # Next.js frontend
│   ├── app/               # Pages and layouts
│   ├── components/        # React components
│   └── lib/              # API client and hooks
├── data/                  # Source PDFs
├── static/                # Generated assets
│   ├── manuals/          # PDF files
│   └── previews/         # Page preview images
├── scripts/               # Utility scripts
│   ├── parse_with_landingai.py
│   ├── weaviate_ingest_manual.py
│   ├── generate_previews.py
│   └── weaviate_search_manual.py
├── docker-compose.yml         # Weaviate + Ollama setup
```

## Scripts

### Parsing
- `scripts/parse_with_landingai.py <pdf_path> <output_json>` - Parse PDF with LandingAI ADE

### Ingestion
- `scripts/weaviate_ingest_manual.py <json_path> <manual_name>` - Ingest parsed chunks into Weaviate
- `scripts/generate_previews.py [pdf_path] [output_dir]` - Generate PNG previews from PDF

### Search
- `scripts/weaviate_search_manual.py <query>` - CLI search (for testing)

## API Endpoints

- `GET /search?query=<query>&limit=<limit>` - Fast vector search (text-based RAG)
- `GET /agentic_search?query=<query>` - Agentic streaming search (both pipelines)
- `GET /healthz` - Health check
- `GET /static/manuals/<manual>.pdf` - PDF files
- `GET /static/previews/<manual>/page-<n>.png` - Preview images

**Interactive docs:** `http://localhost:8001/docs`

## Environment Variables

### Backend (`.env`)
```
LANDINGAI_API_KEY=your_key_here
```

### Frontend (`frontend/.env.local`)

> [!TIP]
> You can copy the example environment file: `cp frontend/.env.example frontend/.env.local`
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8001
```

## Documentation

- **[System Architecture](docs/ARCHITECTURE.md)** - Complete system design, data flows, deployment
- **[Regular RAG Pipeline](docs/RAG_PIPELINE_EXPLAINED.md)** - Text-based search deep-dive
- **[ColQwen Ingestion](docs/COLQWEN_INGESTION_EXPLAINED.md)** - Multimodal search deep-dive
- **[Testing Guide](TESTING.md)** - Testing procedures and verification
- **[Frontend README](frontend/README.md)** - Frontend development guide
- **[Scripts README](scripts/README.md)** - Script usage reference

## Features

- ✅ Semantic search over manual content
- ✅ Visual grounding with bounding boxes
- ✅ Page preview images with highlighted regions
- ✅ Direct PDF page links
- ✅ Keyword highlighting in results
- ✅ Responsive UI

## Tech Stack

- **LandingAI ADE**: Document parsing with layout awareness
- **Weaviate**: Vector database with Ollama embeddings
- **FastAPI**: REST API with Pydantic validation
- **Next.js 16**: React framework with App Router
- **React 19**: Library for web and native user interfaces
- **Tailwind CSS v4**: Utility-first styling
- **shadcn/ui**: Accessible component library
- **React Query**: Data fetching and caching

