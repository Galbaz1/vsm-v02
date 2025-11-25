In the doxs folder you find a file that explains Elysia, it also lists the backend and frontend githubs. I am considering using our local weaviate cluster and local models with Elysia and saomehow add the ColQwen .

## ColQwen + Local Weaviate Integration Plan

### 1. Weaviate Configuration
- **Requirement**: Weaviate v1.29+ (Local instance is v1.34.0, so we are good).
- **Schema**: The collection must be configured for multi-vector embeddings.
    - Use `Configure.VectorIndex.hnsw(multivector=Configure.MultiVector(aggregation=MultiVectorAggregation.MAX_SIM))` (or similar v4 client syntax).
    - Enable `ref2vec-centroid` is NOT needed for ColPali/ColQwen as we provide embeddings manually.

### 2. Ingestion
- **Missing Script**: The `notebooks/` directory referenced in `colqwen/README.md` is missing. We need to recreate the ingestion logic.
- **Logic**:
    1. Load PDF.
    2. Convert pages to images.
    3. Use `ColQwen2_5` (via `colqwen/api/rag.py` classes) to generate multi-vector embeddings for each page.
    4. Ingest into Weaviate using `client.collections.create(...)` with the multi-vector config.

### 3. Retrieval
- **Logic**:
    1. Receive query.
    2. Use `ColQwen2_5` to generate multi-vector embedding for the query.
    3. Query Weaviate using `near_vector` (it supports multi-vector inputs if the index is configured correctly).
    4. Pass results to `Qwen2.5-VL` for generation (already implemented in `colqwen/api/rag.py`).

### 4. Integration with Elysia
- The `colqwen` app is a separate FastAPI service (port 8002).
- We can keep it separate and call it from the main API (or Elysia backend) OR merge the logic.
- Given the "modular structure" request, keeping it as a microservice or a distinct module in the main API is best.