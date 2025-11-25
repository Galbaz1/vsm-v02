# Using This Project with Elysia

Elysia is Weaviate's agentic RAG framework that provides both a backend orchestration layer and a pre-built Next.js frontend for exploring Weaviate collections interactively.

## What is Elysia?

Elysia is a decision-tree-based agent framework that:
- Connects to Weaviate and dynamically chooses which tools to use (query, aggregate, text generation, etc.)
- Comes with a web UI (Next.js frontend) for interactive searches
- Can be extended with custom tools

**Key repositories:**
- Backend: https://github.com/weaviate/elysia
- Frontend: https://github.com/weaviate/elysia-frontend

## Option 1: Use Your Custom FastAPI + Frontend (Recommended)

This is the approach we've taken in this repo because:
- You already have a tailored FastAPI service (`api/main.py`) with `/search` exposing rich metadata (page numbers, bounding boxes).
- Your ingestion pipeline is custom (LandingAI ADE → Weaviate), and you control the schema directly.
- Building a lightweight Next.js UI (as planned) gives you full control over UX without needing to fork/modify Elysia's codebase.

**When to use this:** You want complete control over the UI/UX and your backend logic is already specialized.

## Option 2: Integrate with Elysia for Agentic RAG

If you want Elysia's agentic decision-making capabilities (automatic tool selection, chain-of-thought reasoning), you can wire your existing Weaviate collection into Elysia:

### Setup Steps

1. **Install Elysia**
   ```bash
   pip install elysia-ai
   ```

2. **Configure Elysia for Local Weaviate**
   
   Create or update your `.env` file:
   ```bash
   # Weaviate connection
   WEAVIATE_IS_LOCAL=True
   WCD_URL=localhost
   LOCAL_WEAVIATE_PORT=8080
   LOCAL_WEAVIATE_GRPC_PORT=50051
   
   # Models (use Ollama for local)
   BASE_PROVIDER=ollama
   COMPLEX_PROVIDER=ollama
   BASE_MODEL=qwen2.5:7b
   COMPLEX_MODEL=qwen2.5:7b
   MODEL_API_BASE=http://localhost:11434
   ```

3. **Preprocess Your Collection**
   
   Elysia requires preprocessing to understand your schema and create summaries:
   ```python
   from elysia import configure, preprocess
   
   configure(
       weaviate_is_local=True,
       wcd_url="localhost",
       base_provider="ollama",
       complex_provider="ollama",
       base_model="qwen2.5:7b",
       complex_model="qwen2.5:7b",
       model_api_base="http://localhost:11434",
   )
   
   # This creates metadata for Elysia's agent to understand AssetManual
   preprocess("AssetManual")
   ```

4. **Use Elysia's Backend + Frontend**
   
   Start Elysia's web UI:
   ```bash
   elysia start --port 8002
   ```
   
   Then open http://localhost:8002 to interact with your `AssetManual` collection through Elysia's chat-style interface.

### What You Get with Elysia

**Pros:**
- Agentic decision-making: Elysia automatically decides whether to use `query`, `aggregate`, or other tools.
- Pre-built UI: Interactive chat interface with collection browsing.
- Multi-collection support: Query across multiple Weaviate collections.

**Cons:**
- Less control over UI/UX (would need to fork/customize the frontend repo).
- Preprocessing step required (LLM-generated summaries of your schema).
- Doesn't natively expose your custom FastAPI features (bounding box overlays, preview images).

## Option 3: Hybrid Approach

You can run both:
1. **Your FastAPI service** (port 8001) for the custom manual search UI with PDF previews and bounding box highlights.
2. **Elysia** (port 8002) for exploratory agentic queries across collections.

This gives you:
- A polished, production-ready manual search experience (your custom UI).
- An exploratory/admin tool (Elysia) for ad-hoc analysis or debugging collection data.

## Recommendation

Since your use case focuses on **showing exact PDF sources with visual highlighting**, stick with **Option 1** (custom FastAPI + Next.js frontend). Elysia is powerful for general-purpose agentic RAG, but your pipeline already delivers the precise grounding metadata (page, bbox) that Elysia's generic UI wouldn't leverage out-of-the-box.

If you later want agentic capabilities (e.g., "compare safety procedures across 3 manuals"), you can add Elysia as a complementary tool.

## Quick Command Reference

```bash
# Start your custom FastAPI
uvicorn api.main:app --reload --port 8001

# Start Elysia (if you want to try it)
elysia start --port 8002

# Preprocess your collection for Elysia
python -c "from elysia import configure, preprocess; configure(weaviate_is_local=True); preprocess('AssetManual')"
```

