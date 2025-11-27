# VSM v0.3 - Cloud Migration & DSPy Architecture

**Status:** 🟢 Phases 0-2 Complete | 🟡 Phase 3 Ready to Start  
**Goal:** Unified Local/Cloud Architecture with DSPy Agentic Logic  
**Last Updated:** 2025-11-26

---

## Critical Issues to Fix First

| Issue | Location | Impact | Phase |
|-------|----------|--------|-------|
| **Hardcoded paths** | `api/services/tools/visual_tools.py:194,390` | Breaks on any machine not `/Users/lab/...` | 3.0 |
| **Empty config.py** | `api/core/config.py` | Missing all VSM_MODE and provider settings | 3.1 |

---

## Phase 0: Prerequisites & Setup (NOT STARTED)
> **Status:** Dependencies not installed, no DSPy code exists yet

- [ ] Add `dspy-ai` to `requirements.txt`
- [ ] Add `mlflow` to `requirements.txt` (optional, for experiment tracking)
- [ ] Run `pip install -r requirements.txt` in `vsm-hva` conda env
- [ ] Verify DSPy import: `python -c "import dspy; print(dspy.__version__)"`

## Phase 1: Knowledge Module (Atlas) ✅
> **Completed:** 2025-11-26  
> **Location:** `api/knowledge/`

- [x] Atlas Pydantic Model
- [x] ThorGuard Domain Knowledge
- [x] Factory Pattern

**Verification:**
```bash
python -c "from api.knowledge import get_atlas; a = get_atlas(); print(a.agent_description[:100])"
```

## Phase 2: Enhanced Environment & TreeData ✅
> **Completed:** 2025-11-26  
> **Location:** `api/services/environment.py`

- [x] Rich `tasks_completed` structure (List[Dict])
- [x] Task tracking methods (`update_tasks_completed`)
- [x] Context serialization (`tasks_completed_string`)
- [x] Error handling with `current_tool` context
- [x] Agent wiring for Atlas

**Verification:**
```bash
python -c "from api.services.environment import TreeData; td = TreeData(); print(type(td.tasks_completed))"
```

---

## Phase 3: Provider Layer (Infrastructure Foundation)
> **Goal:** Decouple business logic from specific models. Enable `VSM_MODE`.  
> **Ref:** `docs/cloud-migration/02-provider-layer.md`

### 3.0 Critical Bug Fixes
- [ ] Fix hardcoded `/Users/lab/Documents/vsm-v02` in `visual_tools.py`
  - Lines 194, 390: Replace with `Path(__file__).parent.parent.parent.parent`
  - **Must do first** - blocks all testing

### 3.1 Configuration Refactor
> **File:** `api/core/config.py`  
> **Current state:** Only has `api_base_url`, `pdf_base_url`, `preview_base_url`, `cors_origins`

- [ ] Add `vsm_mode: Literal["local", "cloud"] = "local"`
- [ ] Add Local settings:
  ```python
  ollama_base_url: str = "http://localhost:11434"
  ollama_model: str = "gpt-oss:120b"
  ollama_embed_model: str = "bge-m3"
  mlx_vlm_base_url: str = "http://localhost:8000"
  weaviate_local_url: str = "http://localhost:8080"
  ```
- [ ] Add Cloud settings:
  ```python
  gemini_api_key: str = ""
  gemini_model: str = "gemini-2.5-flash"
  gemini_thinking_budget: int = -1  # -1=dynamic, 0=off, 1-24576=fixed (Ref: https://ai.google.dev/gemini-api/docs/thinking)
  jina_api_key: str = ""
  jina_worker_url: str = ""  # Serverless endpoint for multi-vector
  weaviate_cloud_url: str = ""
  weaviate_cloud_api_key: str = ""
  ```
- [ ] Update `get_settings()` to load from environment

**Verification:**
```bash
python -c "from api.core.config import get_settings; s = get_settings(); print(s.vsm_mode)"
```

### 3.2 Provider Base Interfaces
> **File:** `api/core/providers/base.py`

- [ ] Create `LLMProvider` ABC:
  - `generate(prompt, temperature, max_tokens) -> LLMResponse`
  - `chat(messages, temperature, max_tokens) -> LLMResponse`
  - `stream_chat(messages) -> AsyncGenerator[str]`
- [ ] Create `VLMProvider` ABC:
  - `interpret_image(image_path, prompt) -> str`
  - `is_available() -> bool`
- [ ] Create `EmbeddingProvider` ABC:
  - `embed_texts(texts, task) -> List[List[float]]`
  - `embed_query(query) -> List[float]`
  - `embed_images_multivec(image_paths) -> List[List[List[float]]]`  # Multi-vector!
- [ ] Create `VectorDBProvider` ABC:
  - `vector_search(collection, query_vector, limit) -> List[Dict]`
  - `hybrid_search(collection, query, query_vector, limit, alpha) -> List[Dict]`
  - `multi_vector_search(collection, query_vectors, limit) -> List[Dict]`
  - `upsert_multivec(collection, object_data, vectors) -> None`

### 3.3 Local Providers (Wrap Existing)
> **Files:** `api/core/providers/local/`

- [ ] `OllamaLLM` - Wrap existing `OllamaClient` from `api/services/llm.py`
- [ ] `MLXVLM` - Wrap existing `MLXVLMClient` from `api/services/llm.py`
- [ ] `OllamaEmbeddings` - New, uses Ollama `/api/embeddings` with `bge-m3`
- [ ] `WeaviateLocal` - Wrap existing search logic from `api/services/search.py`

### 3.4 Cloud Providers (Stubs)
> **Files:** `api/core/providers/cloud/`  
> **Note:** Full implementation in Phase 5. Stubs allow testing the abstraction.

- [ ] `GeminiLLM` - Stub with `raise NotImplementedError("Cloud not configured")`
- [ ] `GeminiVLM` - Stub (Gemini 2.5 Flash handles vision too)
- [ ] `JinaHybridEmbeddings` - Stub
- [ ] `WeaviateCloud` - Stub

### 3.5 Factory Implementation
> **File:** `api/core/providers/__init__.py`

- [ ] Implement `get_llm() -> LLMProvider`
- [ ] Implement `get_vlm() -> VLMProvider`
- [ ] Implement `get_embeddings() -> EmbeddingProvider`
- [ ] Implement `get_vectordb() -> VectorDBProvider`
- [ ] Singleton caching for each provider

**Verification:**
```bash
# With VSM_MODE=local (default)
python -c "from api.core.providers import get_llm; print(type(get_llm()).__name__)"
# Should print: OllamaLLM
```

---

## Phase 4: DSPy Signatures & Modules
> **Goal:** Define model-agnostic logic for the Agent.  
> **Ref:** `docs/cloud-migration/03-dspy-prompt-optimization.md`  
> **Depends on:** Phase 3 (needs `get_settings().vsm_mode`)

### 4.1 Core Signatures
> **File:** `api/prompts/signatures.py`

- [ ] `DecisionSignature`:
  - Inputs: `query`, `available_tools`, `environment_summary`, `iteration`
  - Outputs: `tool_name`, `tool_inputs`, `reasoning`, `should_end`
- [ ] `SearchQuerySignature`:
  - Inputs: `original_query`, `search_type`
  - Outputs: `expanded_query`, `keywords`
- [ ] `ResponseSignature`:
  - Inputs: `query`, `context`
  - Outputs: `answer`, `sources`, `confidence`

### 4.2 Chain of Thought Module
> **File:** `api/prompts/chain_of_thought.py`  
> **Pattern:** Elysia's `ElysiaChainOfThought`

- [ ] Implement `VSMChainOfThought(dspy.Module)`:
  - Auto-inject `user_prompt`, `conversation_history`, `atlas`, `previous_errors`
  - Optional: `environment`, `tasks_completed`

### 4.3 DSPy Configuration
> **File:** `api/core/dspy_config.py`  
> **Critical:** DSPy uses different syntax for Gemini thinking control

- [ ] Implement `configure_dspy()`:
  ```python
  if settings.vsm_mode == "local":
      lm = dspy.LM(f'ollama_chat/{settings.ollama_model}', api_base=settings.ollama_base_url)
  else:
      # For Gemini 2.5 Flash thinking control:
      # Ref: https://stackoverflow.com/questions/79809980/turn-off-geminis-reasoning-in-dspy
      if settings.gemini_thinking_budget == 0:
          lm = dspy.LM(f'gemini/{settings.gemini_model}', api_key=settings.gemini_api_key, reasoning_effort="disable")
      elif settings.gemini_thinking_budget == -1:
          lm = dspy.LM(f'gemini/{settings.gemini_model}', api_key=settings.gemini_api_key)  # Dynamic (default)
      else:
          # Custom budget - may need generation_config
          lm = dspy.LM(f'gemini/{settings.gemini_model}', api_key=settings.gemini_api_key)
  ```

---

## Phase 5: Cloud Implementation (The "Hacker" Layer)
> **Goal:** Implement the actual Cloud providers, including the Serverless Worker.  
> **Ref:** `docs/cloud-migration/05-search-pipelines.md`  
> **Depends on:** Phase 3.2 (interfaces), Phase 3.5 (factory)

### 5.1 Serverless Worker (Visual Multi-Vector Search)
> **Critical:** Jina API does NOT expose multi-vector for images. Must self-host.  
> **Ref:** https://huggingface.co/jinaai/jina-embeddings-v4

- [ ] Create `worker/jina_worker.py` (Modal or RunPod script)
- [ ] Implement Jina v4 model loading with `trust_remote_code=True`
- [ ] Implement `/embed` endpoint with `return_multivector=True`
- [ ] Deploy and get endpoint URL
- [ ] Add `JINA_WORKER_URL` to `.env`

**Worker pseudocode:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jinaai/jina-embeddings-v4", trust_remote_code=True)

@app.post("/embed")
def embed(images: List[str], return_multivector: bool = True):
    embeddings = model.encode(images, return_multivector=return_multivector)
    return {"embeddings": embeddings}
```

### 5.2 Cloud Provider: GeminiLLM
> **File:** `api/core/providers/cloud/llm.py`

- [ ] Implement using `google-generativeai` SDK
- [ ] Support `thinkingBudget` parameter (0-24576)
- [ ] Extract thinking output into `LLMResponse.thinking` field

### 5.3 Cloud Provider: GeminiVLM
> **File:** `api/core/providers/cloud/vlm.py`

- [ ] Implement using same Gemini model (multimodal)
- [ ] `interpret_image()` sends image as base64

### 5.4 Cloud Provider: JinaHybridEmbeddings
> **File:** `api/core/providers/cloud/embeddings.py`

- [ ] Implement `embed_texts()` using Jina API `/v1/embeddings`
  - `model: jina-embeddings-v4`
  - `task: retrieval.passage` or `retrieval.query`
- [ ] Implement `embed_query()` using Jina API
  - `task: retrieval.query`
- [ ] Implement `embed_images_multivec()` using **Serverless Worker**
  - POST to `settings.jina_worker_url/embed`
  - `return_multivector: true`

### 5.5 Cloud Provider: WeaviateCloud
> **File:** `api/core/providers/cloud/vectordb.py`  
> **Note:** Already have `WEAVIATE_URL` and `WEAVIATE_API_KEY` in `.env`

- [ ] Implement using `weaviate.connect_to_weaviate_cloud()`
- [ ] Support named vectors for multi-vector search
- [ ] Implement `upsert_multivec()` for ingestion

---

## Phase 6: Tool & Agent Refactor
> **Goal:** Connect the Agent to the new Provider abstraction.  
> **Ref:** `docs/cloud-migration/04-tool-routing.md`  
> **Depends on:** Phase 3.5 (factory), Phase 4 (DSPy), Phase 5 (cloud providers)

### 6.1 Refactor Tools to Use Providers
> **Files:** `api/services/tools/search_tools.py`, `api/services/tools/visual_tools.py`

- [ ] `FastVectorSearchTool`: Replace direct Weaviate calls with:
  ```python
  from api.core.providers import get_embeddings, get_vectordb
  embedder = get_embeddings()
  vectordb = get_vectordb()
  ```
- [ ] `ColQwenSearchTool`: Same pattern, use `embed_images_multivec()` for query
- [ ] `VisualInterpretationTool`: Use `get_vlm()` instead of direct `MLXVLMClient`
- [ ] `TextResponseTool`: Use `get_llm()` instead of direct `OllamaClient`
- [ ] `SummarizeTool`: Use `get_llm()`
- [ ] Verify `HybridSearchTool` still works (uses other tools internally)

### 6.2 Refactor Agent Orchestrator
> **File:** `api/services/agent.py`

- [ ] Replace `DecisionPromptBuilder` + `OllamaClient` with DSPy module:
  ```python
  from api.prompts import get_compiled_module
  decision_module = get_compiled_module("decision")
  ```
- [ ] Keep rule-based fallback for reliability
- [ ] Remove direct imports of `get_ollama_client()`

---

## Phase 7: Cloud Ingestion
> **Goal:** Populate the Cloud Vector DB.  
> **Depends on:** Phase 5 (cloud providers working)

- [ ] Create `scripts/cloud_ingest.py`
- [ ] Implement Text Ingestion:
  - Use `JinaHybridEmbeddings.embed_texts()` → `WeaviateCloud.batch_upsert()`
- [ ] Implement Visual Ingestion:
  - Use `JinaHybridEmbeddings.embed_images_multivec()` → `WeaviateCloud.upsert_multivec()`
- [ ] Run ingestion for "TechMan" manual
- [ ] Verify with test queries

---

## Phase 8: Benchmarking System
> **Goal:** Evaluate performance of both modes.

- [ ] Implement `TechnicalJudge` (DSPy module)
- [ ] Create Benchmark API (`/benchmark/evaluate`)
- [ ] Create Frontend Benchmark Mode UI
- [ ] Create `scripts/run_benchmark.py`

---

## Phase 9: Optimization Loop
> **Goal:** Tune prompts for Gemini and OSS models.

- [ ] Create training dataset (examples of good decisions)
- [ ] Create `scripts/optimize_prompts.py`:
  ```python
  from dspy.teleprompt import MIPROv2
  optimizer = MIPROv2(metric=decision_accuracy, auto="medium")
  optimized = optimizer.compile(module, trainset=examples)
  optimized.save(f"api/prompts/{mode}/decision.json")
  ```
- [ ] Run optimization for Local (gpt-oss)
- [ ] Run optimization for Cloud (gemini-flash)
- [ ] Commit optimized JSON files to `api/prompts/local/`, `api/prompts/cloud/`

---

## Phase 10: Cleanup & Release

- [ ] Remove legacy `OllamaClient` class (replaced by `OllamaLLM` provider)
- [ ] Remove legacy `MLXVLMClient` class (replaced by `MLXVLM` provider)
- [ ] Remove `DecisionPromptBuilder` (replaced by DSPy)
- [ ] Update `.cursor/rules/philosophy.mdc` - change "DSPy not used" to "DSPy active"
- [ ] Final Documentation Update

---

## Key References

### Verified Sources

| Topic | Source | Key Finding |
|-------|--------|-------------|
| **Jina v4 Multi-Vector** | [HuggingFace Model Card](https://huggingface.co/jinaai/jina-embeddings-v4) | Multi-vector via `return_multivector=True` in Python only, NOT via API |
| **DSPy + Gemini Thinking** | [Stack Overflow](https://stackoverflow.com/questions/79809980/turn-off-geminis-reasoning-in-dspy) | Use `reasoning_effort="disable"` to turn off thinking |
| **Gemini thinkingBudget** | [Google AI Docs](https://ai.google.dev/gemini-api/docs/thinking) | 2.5 Flash: 0-24576, -1=dynamic, 0=off |
| **Weaviate Multi-Vector** | [Weaviate Docs](https://weaviate.io/blog/late-interaction-overview) | Named vectors supported in v1.29+ |

### Architecture Docs

- `docs/cloud-migration/01-architecture-overview.md` - High-level design
- `docs/cloud-migration/02-provider-layer.md` - Provider interfaces
- `docs/cloud-migration/03-dspy-prompt-optimization.md` - DSPy integration
- `docs/cloud-migration/04-tool-routing.md` - Tool architecture
- `docs/cloud-migration/05-search-pipelines.md` - Search flow diagrams
- `docs/cloud-migration/06-configuration-guide.md` - Setup instructions

### Elysia Patterns Used

| Pattern | Elysia Source | VSM Location |
|---------|---------------|--------------|
| `TreeData` state object | `elysia/tree/objects.py` | `api/services/environment.py` |
| `Tool` base class | `elysia/objects.py` | `api/services/tools/base.py` |
| `Environment` store | `elysia/tree/objects.py` | `api/services/environment.py` |
| `update_tasks_completed` | Lines 685-742 | `TreeData.update_tasks_completed()` |
| Decision loop | `elysia/tree/tree.py` | `AgentOrchestrator.run()` |

---

## Quick Start (After Phase 3)

```bash
# Local Mode (default)
export VSM_MODE=local
python -m api.main  # Uses Ollama + MLX + Local Weaviate

# Cloud Mode
export VSM_MODE=cloud
export GEMINI_API_KEY=AIza...
export JINA_API_KEY=jina_...
export JINA_WORKER_URL=https://your-worker.modal.run
export WEAVIATE_CLOUD_URL=https://xxx.weaviate.cloud
export WEAVIATE_CLOUD_API_KEY=xxx
python -m api.main  # Uses Gemini + Jina + Weaviate Cloud
```
