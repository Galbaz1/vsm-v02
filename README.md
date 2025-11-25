# VSM Demo v02 - Technical Asset Manual Search

A local, agentic RAG stack for searching technical asset manuals with visual grounding. Built with LandingAI ADE, Weaviate, FastAPI, and Next.js.

## Architecture

- **Parsing**: LandingAI Agentic Document Extraction (ADE) for PDF parsing
- **Vector DB**: Local Weaviate (Docker) with Ollama embeddings
- **Backend API**: FastAPI serving search endpoints and static assets
- **Frontend**: Next.js 14 with Tailwind CSS and shadcn/ui

## Quick Start

### Prerequisites

- Python 3.12+ (Conda env `vsm-hva`)
- Node.js 18+
- Docker & Docker Compose
- LandingAI API key (set in `.env`)

### 1. Start Infrastructure

```bash
# Start Weaviate and Ollama
docker compose up -d

# Pull embedding model
docker compose exec ollama ollama pull nomic-embed-text
```

### 2. Parse and Ingest Manuals

```bash
# Activate conda environment
conda activate vsm-hva

# Parse PDF with LandingAI ADE
python parse_with_landingai.py data/uk_firmware.pdf output_landingai.json

# Generate page previews
python generate_previews.py

# Ingest into Weaviate
python weaviate_ingest_manual.py output_landingai.json "UK Firmware Manual"
```

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
├── parse_with_landingai.py    # ADE parsing script
├── weaviate_ingest_manual.py  # Weaviate ingestion
├── generate_previews.py       # Preview generation
└── docker-compose.yml         # Weaviate + Ollama setup
```

## Scripts

### Parsing
- `parse_with_landingai.py <pdf_path> <output_json>` - Parse PDF with LandingAI ADE

### Ingestion
- `weaviate_ingest_manual.py <json_path> <manual_name>` - Ingest parsed chunks into Weaviate
- `generate_previews.py [pdf_path] [output_dir]` - Generate PNG previews from PDF

### Search
- `weaviate_search_manual.py <query>` - CLI search (for testing)

## API Endpoints

- `GET /search?query=<query>&limit=<limit>` - Semantic search
- `GET /healthz` - Health check
- `GET /static/manuals/<manual>.pdf` - PDF files
- `GET /static/previews/<manual>/page-<n>.png` - Preview images

## Environment Variables

### Backend (`.env`)
```
LANDINGAI_API_KEY=your_key_here
```

### Frontend (`frontend/.env.local`)
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8001
```

## Development

See individual READMEs:
- [Frontend README](frontend/README.md)
- [Elysia Integration Guide](docs/USING_WITH_ELYSIA.md)

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
- **Next.js 14**: React framework with App Router
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Accessible component library
- **React Query**: Data fetching and caching

