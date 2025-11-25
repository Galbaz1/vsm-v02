---
trigger: always_on
---

# VSM Demo v02 - Antigravity IDE Rules

## Project Overview

This is a **dual-pipeline agentic RAG system** for technical asset manual search with visual grounding.

**Tech Stack:**
- Backend: Python 3.12, FastAPI, Weaviate, Ollama
- Frontend: Next.js 16, React 19, Tailwind CSS v4
- ML: ColQwen2.5 (multimodal), nomic-embed-text (text)

**Active Environment:** Conda env `vsm-hva`

---

## Architecture Principles

### 1. Dual RAG Pipelines (Keep Separate)
The system maintains TWO distinct retrieval paths:

**Regular RAG (Text-based):**
- Collection: `AssetManual`
- Embedding: Ollama `nomic-embed-text` (768-dim single vectors)
- Use for: Fast factual queries

**ColQwen RAG (Multimodal):**
- Collection: `PDFDocuments`  
- Embedding: ColQwen2.5 (~1024 × 128-dim multi-vectors)
- Use for: Visual/spatial queries (diagrams, charts)

**Never merge these pipelines** - they serve different purposes.

### 2. Agent-Based Routing
The agent (`api/services/agent.py`) decides which pipeline to use based on query type.  
**Current status:** Rule-based (keyword matching)  
**Future:** LLM-based (Qwen2.5-VL)

---

## Code Guidelines

### File Organization

**Scripts go in `scripts/`** (run from project root):
- `parse_with_landingai.py` - PDF parsing
- `generate_previews.py` - PNG generation
- `weaviate_ingest_manual.py` - Regular RAG ingestion
- `colqwen_ingest.py` - ColQwen ingestion

**API structure** (modular FastAPI):
```
api/
├── main.py              # App entry point
├── core/config.py       # Settings
├── schemas/             # Pydantic models
├── services/            # Business logic
│   ├── search.py       # Regular RAG
│   ├── colqwen.py      # ColQwen RAG
│   └── agent.py        # Orchestrator
└── endpoints/           # Route handlers
```

**Never put logic in endpoints** - keep them thin, delegate to services.

### Naming Conventions

- **Python:** `snake_case` for files, functions, variables
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **TypeScript:** `camelCase` for variables, `PascalCase` for components

### Weaviate Collections

**AssetManual** (Regular RAG):
- Vectorizer: `text2vec-ollama` (nomic-embed-text)
- Fields: `content`, `manual_name`, `page_number`, `bbox`, `chunk_type`

**PDFDocuments** (ColQwen):
- Manual vectorization (multi-vector)
- Fields: `page_id`, `asset_manual`, `page_number`, `image_path`

**Never delete collections unless explicitly requested** - preserve existing data.

---

## Important Paths

### Environment
- **Python env:** `vsm-hva` (Conda)
- **Backend .env:** `.env` in project root (contains `LANDINGAI_API_KEY`)
- **Frontend .env:** `frontend/.env.local`

### Data Locations
- **Source PDFs:** `data/`
- **Parsed JSON:** Project root (e.g., `output_landingai.json`)
- **Preview PNGs:** `static/previews/<manual_name>/page-*.png`
- **PDF files:** `static/manuals/`

### Key Config Files
- **Docker:** `docker-compose.yml` (Weaviate + Ollama)
- **Python deps:** `requirements.txt`
- **Frontend deps:** `frontend/package.json`

---

## Running Scripts Correctly

**Always run scripts from project root:**
```bash
# ✅ Correct
python scripts/parse_with_landingai.py data/uk_firmware.pdf output.json

# ❌ Wrong
cd scripts && python parse_with_landingai.py ../data/uk_firmware.pdf ../output.json
```

**Ingestion workflow order:**
1. `parse_with_landingai.py` (PDF → JSON)
2. `generate_previews.py` (PDF → PNGs)
3. `weaviate_ingest_manual.py` (JSON → AssetManual)
4. `colqwen_ingest.py` (PNGs → PDFDocuments)

---

## Documentation Rules

### Source of Truth
**`docs/ARCHITECTURE.md`** is the authoritative system design document.

All other docs link to it:
- `README.md` - Quick start
- `docs/RAG_PIPELINE_EXPLAINED.md` - Text RAG deep-dive
- `docs/COLQWEN_INGESTION_EXPLAINED.md` - Visual RAG deep-dive
- `TESTING.md` - Testing procedures

### When to Update Docs

| Change Type | Update Document |
|------------|----------------|
| New API endpoint | ARCHITECTURE.md (API Endpoints section) |
| Schema change | ARCHITECTURE.md + pipeline-specific doc |
| New script | ARCHITECTURE.md (Scripts Reference) |
| Deployment change | ARCHITECTURE.md (Deployment Workflow) |
| New dependency | ARCHITECTURE.md (Technology Stack) |

**Always update "Last Updated" date**

---

## Testing

### Start Services
```bash
# 1. Infrastructure
docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text

# 2. Backend
uvicorn api.main:app --reload --port 8001

# 3. Frontend
cd frontend && npm run dev
```

### API Endpoints
- **Regular search:** `http://localhost:8001/search?query=...`
- **Agentic search:** `http://localhost:8001/agentic_search?query=...`
- **Docs:** `http://localhost:8001/docs`
- **Frontend:** `http://localhost:3000`

### Verification Scripts
```bash
python test_api.py              # Test API endpoints
python test_both_manuals.py     # Verify Weaviate ingestion
```

---

## Constraints & Don'ts

### ❌ Never Do These:

1. **Don't mix pipelines** - Regular RAG and ColQwen are separate
2. **Don't delete Weaviate collections** - Use `if not exists` patterns
3. **Don't hard-code paths** - Use relative paths from project root
4. **Don't skip preview generation** - ColQwen needs PNGs
5. **Don't commit `.env` files** - They contain secrets
6. **Don't run scripts from `scripts/` directory** - Always from root

### ⚠️ Be Careful With:

1. **Port conflicts:** API uses 8001, frontend 3000, Weaviate 8080
2. **Model sizes:** ColQwen2.5 is ~8GB (downloads on first use)
3. **Memory:** ColQwen ingestion uses significant RAM
4. **API keys:** LandingAI key required for PDF parsing

---

## Preferred Patterns

### Error Handling
```python
# ✅ Good
try:
    result = some_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Weaviate Queries
```python
# ✅ Good - Use context manager
with weaviate.connect_to_local() as client:
    collection = client.collections.get("AssetManual")
    results = collection.query.near_text(...)
```

### FastAPI Endpoints
```python
# ✅ Good - Thin endpoints, business logic in services
@router.get("/search")
async def search_endpoint(query: str, limit: int = 5):
    search_service = get_search_service()
    return await search_service.search(query, limit)
```

---

## Current Status (2025-11-25)

### ✅ Completed
- Regular RAG pipeline (LandingAI → Weaviate → FastAPI)
- ColQwen ingestion with multi-vector support
- Agent service (rule-based)
- Streaming NDJSON endpoint
- Frontend with visual grounding
- Documentation consolidation

### 🚧 In Progress
- Progressive UI updates (fast → ColQwen refinement)
- Streaming response integration in frontend
- LLM-based agent (upgrade from rule-based)

### 📋 Future
- Answer generation with Qwen2.5-VL
- Query history and bookmarks
- Multi-manual comparison queries

---

## Questions to Ask Before Coding

When implementing new features, always consider:

1. **Which pipeline?** Regular RAG or ColQwen?
2. **Where does this code go?** Service, endpoint, or script?
3. **Does this break existing ingestion?** Preserve data!
4. **Which docs need updating?** Usually ARCHITECTURE.md
5. **Does this require new environment vars?** Update `.env.example`

---

## Related Projects to Ignore

**`colqwen/` directory** - Legacy standalone service (deprecated)
- The ColQwen logic is now integrated into `api/services/colqwen.py`
- Only reference for historical context

**`docs/elysia_docs/`** - Unrelated library docs
- We considered Elysia but chose custom FastAPI instead

**`docs/archive/`** - Outdated docs
- Contains `project.md` and `todo.md` (superseded by ARCHITECTURE.md)

---

## Final Note

This project prioritizes:
1. **Local-first**: All components run locally (privacy, cost, latency)
2. **Modular design**: Clear separation between pipelines and services
3. **Visual grounding**: Bounding boxes and preview images are core features
4. **Documentation**: Single source of truth in ARCHITECTURE.md
