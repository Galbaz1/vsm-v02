---
alwaysApply: true
---

---

### Project context (for Cursor)

We are building a **local, agentic RAG stack** for technical asset manuals:

* **Vector DB:** Local Weaviate (Docker) with gRPC enabled, reachable at `http://localhost:8080` (HTTP) and `localhost:50051` (gRPC). Primary collection(s) live here and store chunked manual content plus metadata (page, coordinates, section type, etc.).

  * Docs: [https://docs.weaviate.io/weaviate/quickstart/local](https://docs.weaviate.io/weaviate/quickstart/local)
  * Python client: [https://weaviate-python-client.readthedocs.io](https://weaviate-python-client.readthedocs.io)

* **Parsing / Chunking:** We use **LandingAI Agentic Document Extraction (ADE)** to parse *large, complex PDFs* (100–1000 pages, with images, tables, forms). ADE gives us layout-aware JSON (and optionally Markdown) with page + coordinate “groundings”.

  * Overview: [https://docs.landing.ai/ade/ade-overview](https://docs.landing.ai/ade/ade-overview)
  * Quickstart: [https://docs.landing.ai/ade/ade-quickstart](https://docs.landing.ai/ade/ade-quickstart)
  * Python library: [https://github.com/landing-ai/ade-python](https://github.com/landing-ai/ade-python)
  * Goal: write standalone scripts that

    1. Call ADE’s async parse for big PDFs.
    2. Normalize chunks (text spans, table cells, captions) into a simple schema.
    3. Upsert those chunks into Weaviate with embeddings.

* **Models / Embeddings:**

  * Local models are served via **Ollama** with at least:

    * `bge-m3` for embeddings
    * `qwen2.5:7b` (text LLM)
    * `qwen3-vl:8b` (vision-language, for later multimodal stuff)
  * Weaviate uses `text2vec-ollama` pointing to `http://ollama:11434` and `bge-m3` as the default vectorizer for text fields.

* **Agent / UI:**

  * We use **Elysia** (Weaviate’s agentic RAG framework) as the orchestration + frontend layer on top of Weaviate.
  * Elysia runs in Python and connects to Weaviate via its own config; it decides which tools to call (`query`, `aggregate`, `text_response`, etc.) and exposes a browser UI for interactive RAG.
  * Docs: [https://weaviate.github.io/elysia/](https://weaviate.github.io/elysia/)
  * Blog explainer: [https://weaviate.io/blog/elysia-agentic-rag](https://weaviate.io/blog/elysia-agentic-rag) ([weaviate.io][1])

---

### Files / scripts the agent should expect

We are standardizing on three main script types (Python):

1. **`landing_parse_and_store.py`**

   * Uses `ade-python` to:

     * Accept a local PDF path.
     * Run ADE async parse for large PDFs.
     * Transform ADE JSON into normalized chunk structures (including page index and, where useful, bounding boxes).
     * Write chunks into Weaviate `TestDocs` or a dedicated manual collection (e.g. `AssetManualChunk`) using the Python client and automatic `text2vec-ollama` embeddings.

2. **`weaviate_setup.py`**

   * Creates / updates Weaviate collections with:

     * `text` / `content` field
     * optional `page`, `section_type`, `asset_id`, etc.
     * `text2vec-ollama` vector config (`bge-m3` model, `apiEndpoint=http://ollama:11434`).
   * This script is idempotent (safe to run multiple times).

3. **`search_demo.py`**

   * Connects to local Weaviate via `weaviate.connect_to_local()`.
   * Performs simple `near_text` and `hybrid` queries against the manual collection.
   * Optionally wraps queries in a small helper for Elysia or for quick CLI debugging (print top-k hits with page numbers and snippets).

---

### Current implementation status (v0 prototype)

In this repo we already have a first working, end-to-end prototype using three concrete scripts:

1. **`parse_with_landingai.py`**

   * Uses the ADE **Parse Jobs** HTTP API to:
     * Accept a local PDF path (e.g. `data/uk_firmware.pdf`).
     * Create an async ADE parse job (`POST /v1/ade/parse/jobs` with `model=dpt-2-latest`).
     * Poll the job until completion and download the full JSON result (`markdown`, `chunks`, `grounding`, `metadata`, etc.).
     * Save that JSON to disk (e.g. `output_landingai.json`).
   * Loads `LANDINGAI_API_KEY` from a local `.env` file via `python-dotenv`.

2. **`weaviate_ingest_manual.py`**

   * Reads the ADE JSON output (`output_landingai.json`).
   * Normalizes ADE chunks into a simple schema with:
     * `content` (chunk text/markdown),
     * `anchor_id` (chunk ID),
     * `chunk_type` (e.g. `text`, `table`, `figure`, `marginalia`),
     * `page_number` (from ADE grounding),
     * `bbox` (stringified bounding box from ADE grounding).
   * Connects to local Weaviate via `weaviate.connect_to_local()`.
   * Ensures a collection `AssetManual` exists with:
     * `content` and `manual_name` vectorized via **`text2vec-ollama`** (`nomic-embed-text`, `api_endpoint=http://ollama:11434`),
     * metadata-only fields (`anchor_id`, `chunk_type`, `page_number`, `bbox`) with `skip_vectorization=True` where appropriate.
   * Batch-ingests all chunks into `AssetManual`.

3. **`weaviate_search_manual.py`**

   * Connects to local Weaviate and runs simple search queries over the `AssetManual` collection (near-text / hybrid style).
   * Returns top-k chunks with their `content`, `page_number`, and other metadata for debugging and RAG experiments.

Over time, these prototype scripts can be refactored/renamed into the more polished targets
(`landing_parse_and_store.py`, `weaviate_setup.py`, `search_demo.py`), but agents should treat the
current three scripts as the **source of truth** for the working PDF → ADE JSON → Weaviate pipeline.

---

### Important assumptions for Cursor

* **Environment:**

  * Python 3.12, Conda env `vsm-hva`.
  * Weaviate and Ollama are started via `docker-compose.yml` in the project root.
* **Endpoints:**

  * Weaviate: `http://localhost:8080` (HTTP), `localhost:50051` (gRPC).
  * Ollama: `http://localhost:11434`.
  * ADE: authenticated via `LANDINGAI_API_KEY` env var; follow ADE docs for host/base URL.
* **Code style expectations:**

  * Use the **official Weaviate Python client** (v4+).
  * Use the **official LandingAI `ade-python` client** (no custom HTTP unless absolutely necessary).
  * Prefer small, composable functions for: `parse_pdf_with_ade`, `flatten_ade_chunks`, `ensure_weaviate_collection`, `upsert_chunks_to_weaviate`, `search_manual`.


[1]: https://weaviate.io/blog/elysia-agentic-rag?utm_source=chatgpt.com "Elysia: Building an end-to-end agentic RAG app"
