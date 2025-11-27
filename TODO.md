# VSM v0.3 - Cloud Migration & DSPy Architecture

**Status:** 🟢 Phases 0-6 Complete | 🟡 Phase 7 Ready to Start  
**Goal:** Unified Local/Cloud Architecture with DSPy Agentic Logic  
**Last Updated:** 2025-11-27

---

## Critical Issues to Fix First

| Issue | Location | Impact | Status |
|-------|----------|--------|--------|
| ~~Hardcoded paths~~ | `api/services/tools/visual_tools.py` | ~~Breaks on non-lab machines~~ | ✅ Fixed |
| ~~Empty config.py~~ | `api/core/config.py` | ~~Missing VSM_MODE~~ | ✅ Fixed |

---

## Phase 0: Prerequisites & Setup ✅
> **Status:** ✅ Complete - DSPy 3.0.4 installed

- [x] Add `dspy-ai` to `requirements.txt`
- [x] Add `mlflow` to `requirements.txt` (optional, for experiment tracking)
- [x] Run `pip install -r requirements.txt` in `vsm-hva` conda env
- [x] Verify DSPy import: `python -c "import dspy; print(dspy.__version__)"` → **v3.0.4**

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

## Phase 3: Provider Layer (Infrastructure Foundation) ✅
> **Completed:** 2025-11-27  
> **Goal:** Decouple business logic from specific models. Enable `VSM_MODE`.  
> **Ref:** `docs/cloud-migration/02-provider-layer.md`

### 3.0 Critical Bug Fixes ✅
- [x] Fixed hardcoded `/Users/lab/Documents/vsm-v02` in `visual_tools.py`
  - Lines 194, 390: Now uses `Path(__file__).parent.parent.parent.parent`
  - Works on any machine

### 3.1 Configuration Refactor ✅
- [x] Added `vsm_mode: Literal["local", "cloud"] = "local"`
- [x] Added Local settings (Ollama, MLX, Weaviate Docker)
- [x] Added Cloud settings (Gemini, Jina, Weaviate Cloud)
- [x] Updated `get_settings()` to load from environment

### 3.2 Provider Base Interfaces ✅
- [x] `LLMProvider` ABC with generate, chat, stream_chat
- [x] `VLMProvider` ABC with interpret_image, is_available
- [x] `EmbeddingProvider` ABC for TEXT only (embed_texts, embed_query)
- [x] `VectorDBProvider` ABC for TEXT search (vector_search, hybrid_search, batch_upsert)
- [x] `VisualSearchProvider` ABC for full visual RAG (search, ingest_page)

> **Key Design Decision:** Visual search uses a separate provider because:
> - Local: ColQwen does embedding + MaxSim in one shot (can't split)
> - Cloud: Jina Worker (embed) + Weaviate Cloud (search)

### 3.3 Local Providers ✅
- [x] `OllamaLLM` - Wraps existing `OllamaClient`
- [x] `MLXVLM` - Wraps existing `MLXVLMClient`
- [x] `OllamaEmbeddings` - TEXT only, Ollama `/api/embeddings` with bge-m3
- [x] `WeaviateLocal` - TEXT search on AssetManual
- [x] `ColQwenVisualSearch` - Wraps existing `ColQwenRetriever` for visual RAG

### 3.4 Cloud Providers (Stubs) ✅
- [x] `GeminiLLM` - Stub
- [x] `GeminiVLM` - Stub
- [x] `JinaEmbeddings` - Stub (TEXT only)
- [x] `WeaviateCloud` - Stub (TEXT search)
- [x] `JinaVisualSearch` - Stub (Jina Worker + Weaviate Cloud)

### 3.5 Factory Implementation ✅
- [x] `get_llm() -> LLMProvider`
- [x] `get_vlm() -> VLMProvider`
- [x] `get_embeddings() -> EmbeddingProvider`
- [x] `get_vectordb() -> VectorDBProvider`
- [x] `get_visual_search() -> VisualSearchProvider`
- [x] Singleton caching with lazy loading

**Verification Passed:**
```bash
$ python -c "from api.core.providers import get_llm, get_visual_search; print(type(get_llm()).__name__, type(get_visual_search()).__name__)"
OllamaLLM ColQwenVisualSearch
```

---

## Phase 4: DSPy Signatures & Modules ✅
> **Completed:** 2025-11-27  
> **Goal:** Define model-agnostic logic for the Agent.  
> **Ref:** `docs/cloud-migration/03-dspy-prompt-optimization.md`

### 4.1 Core Signatures ✅
> **File:** `api/prompts/signatures/`

- [x] `DecisionSignature`: Tool selection with query, tools, env, iteration
- [x] `SearchQuerySignature`: Query expansion for retrieval
- [x] `ResponseSignature`: Answer generation with sources + confidence

### 4.2 Chain of Thought Module ✅
> **File:** `api/prompts/chain_of_thought.py`

- [x] `VSMChainOfThought(dspy.Module)` with auto-context injection
- [x] Auto-injects: user_prompt, conversation, atlas, errors, environment

### 4.3 DSPy Configuration ✅
> **File:** `api/core/dspy_config.py`

- [x] `configure_dspy()` - Mode-switchable LM config
- [x] Gemini thinking control (0=off, -1=dynamic, 1-24576=fixed)
- [x] `get_dspy_lm()` + `reset_dspy_config()`

### 4.4 Module Loader ✅
> **File:** `api/prompts/__init__.py`

- [x] `get_compiled_module(name)` - Load optimized modules per mode
- [x] `get_vsm_module(name)` - Get VSMChainOfThought with TreeData support
- [x] Caching + state loading from `prompts/{mode}/*.json`

**Verification:**
```bash
python -c "from api.prompts import get_compiled_module, SIGNATURE_MAP; print(list(SIGNATURE_MAP.keys()))"
# ['decision', 'search', 'response']
```

---

## Phase 5: Cloud Implementation ✅
> **Completed:** 2025-11-27  
> **Goal:** Implement the actual Cloud providers.  
> **Ref:** `docs/cloud-migration/05-search-pipelines.md`

### 5.1 Cloud Provider: GeminiLLM ✅
> **File:** `api/core/providers/cloud/llm.py`

- [x] Uses `google-genai` SDK (python-genai)
- [x] Supports `thinkingBudget` (0=off, -1=dynamic, 1-24576=fixed)
- [x] Extracts thinking output into `LLMResponse.thinking`
- [x] `generate()`, `chat()`, `stream_chat()` implemented

### 5.2 Cloud Provider: GeminiVLM ✅
> **File:** `api/core/providers/cloud/vlm.py`

- [x] Uses same Gemini model (multimodal)
- [x] `interpret_image()` sends image as base64 Part
- [x] `is_available()` for health check

### 5.3 Cloud Provider: JinaEmbeddings ✅
> **File:** `api/core/providers/cloud/embeddings.py`

- [x] `embed_texts()` using Jina API `/v1/embeddings`
- [x] `embed_query()` with `task: retrieval.query`
- [x] Matryoshka dimensions (1024 default)
- [x] Async httpx client

### 5.4 Cloud Provider: JinaVisualSearch ✅
> **File:** `api/core/providers/cloud/visual_search.py`

- [x] `search()`: Weaviate `near_text` with `multi2vec-jinaai`
- [x] `ingest_page()`: Base64 blob storage, auto-embed
- [x] `get_page_image()`: Decoded blob retrieval
- [x] `ensure_collection_exists()`: PDFDocuments with Jina CLIP

### 5.5 Cloud Provider: WeaviateCloud ✅
> **File:** `api/core/providers/cloud/vectordb.py`

- [x] `connect()` using `weaviate.connect_to_weaviate_cloud()`
- [x] `vector_search()` with near_vector
- [x] `hybrid_search()` for AssetManual
- [x] `batch_upsert()` for text ingestion

**Verification:**
```bash
python -c "from api.core.providers.cloud.llm import GeminiLLM; print('OK')"
```

---

## Phase 6: Tool & Agent Refactor ✅
> **Completed:** 2025-11-27  
> **Goal:** Connect the Agent to the new Provider abstraction.  
> **Ref:** `docs/cloud-migration/04-tool-routing.md`

### 6.1 Refactor Tools to Use Providers ✅
> **Files:** `api/services/tools/`

- [x] `FastVectorSearchTool`: Uses `get_embeddings()` + `get_vectordb()`
- [x] `ColQwenSearchTool`: Uses `get_visual_search().search()`
- [x] `VisualInterpretationTool`: Uses `get_vlm()` + cloud image retrieval
- [x] `TextResponseTool`: Uses `get_llm()`
- [x] `SummarizeTool`: Uses `get_llm()`
- [x] `HybridSearchTool`: Uses refactored tools (parallel execution)

### 6.2 Refactor Agent Orchestrator ✅
> **File:** `api/services/agent.py`

- [x] Replaced `DecisionPromptBuilder` with DSPy module `get_vsm_module("decision")`
- [x] Maintained rule-based fallback for reliability
- [x] Removed direct imports of `get_ollama_client()`
- [x] Async execution of DSPy modules via `asyncio.to_thread`

### 6.3 API Services Refactor ✅
> **File:** `api/services/search.py`, `api/endpoints/search.py`

- [x] `perform_search` is now async and uses providers
- [x] `search_manual` endpoint is now async
- [x] Fixed WeaviateCloud filter handling

---

## Phase 7: Cloud Ingestion ✅
> **Completed:** 2025-11-27  
> **Goal:** Populate the Cloud Vector DB.  
> **Depends on:** Phase 5 (cloud providers working)

- [x] Create `scripts/cloud_ingest.py`
- [x] Implement Text Ingestion:
  - Uses `JinaEmbeddings.embed_texts()` → `WeaviateCloud.batch_upsert()`
  - Parses markdown from LandingAI JSON
- [x] Implement Visual Ingestion:
  - Uses `pdf2image` to convert PDF pages
  - Uses `get_visual_search().ingest_page()` (handles Blob + Auto-embed)
- [x] Run ingestion for both manuals:
  - ThorGuard TechMan: 132 pages (text + visual)
  - ThorGuard UK Firmware: 128 pages (text + visual)
- [x] Verified with test queries (text + visual search working)

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
| **Jina CLIP + Weaviate** | [Weaviate Docs](https://docs.weaviate.io/weaviate/model-providers/jinaai/embeddings-multimodal) | Native integration via `multi2vec-jinaai`, images as blobs, auto-embed |
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
export WEAVIATE_CLOUD_URL=https://xxx.weaviate.cloud
export WEAVIATE_CLOUD_API_KEY=xxx
python -m api.main  # Uses Gemini + Jina + Weaviate Cloud
```
