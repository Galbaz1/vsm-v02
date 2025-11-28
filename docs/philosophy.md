---
alwaysApply: true
---

# Philosophy & Vision

## Core Principles
1. **Mode-switchable** - `VSM_MODE=local|cloud` via Provider abstraction
2. **Native Ollama (local)** - Must run natively (not Docker) to access full RAM/Metal
3. **Dual-pipeline** - Text RAG + Visual RAG stay separate (both modes)
4. **Visual grounding** - BBoxes and page images are core, not optional

## Memory Budget (M3 256GB)
| Model | Size | Notes |
|-------|------|-------|
| gpt-oss-120B | ~65GB | LLM for decisions + text generation |
| bge-m3 | ~1.2GB | Embeddings (8K context, retrieval-optimized) |
| Qwen3-VL-8B | ~8GB | VLM for visual interpretation (MLX) |
| ColQwen2.5-v0.2 | ~4GB | Visual retrieval (PyTorch) |
| **Total** | ~78GB | Leaves 178GB for KV cache |

## Agent Architecture (Elysia-inspired)

### Decision Loop
```
User Query → LLM Decision → Tool Execution → Result
                ↑                              ↓
                └──── Self-healing if error ───┘
```

### Tool Pattern
```python
class Tool:
    async def __call__(self, tree_data, inputs) -> AsyncGenerator:
        yield Result(
            objects=[...],
            metadata={"query": query},
            llm_message="Found {n} results"  # For LLM context
        )
```

### Environment (Centralized State)
```python
environment[tool_name][result_name] = [{"objects": [...], "metadata": {...}}]
```

## What We DON'T Use
- **Dockerized Ollama** - Can't access full RAM/GPU (local mode)
- **Multiple LLMs per mode** - Single LLM for all text tasks

## Cloud Migration (Phase 3 Complete ✅)
> See `TODO.md` and `docs/cloud-migration/` for full plan

**Status:** Phase 3 Complete | Phase 4 (DSPy) Ready

| Mode | LLM | VLM | Text Embeddings | Visual Search |
|------|-----|-----|-----------------|---------------|
| Local | gpt-oss:120b (Ollama) | Qwen3-VL-8B (MLX) | bge-m3 | ColQwen2.5-v0.2 |
| Cloud | Gemini 2.5 Flash | Gemini 2.5 Flash | Jina v4 | Weaviate + Jina CLIP v2 |

**Architecture:** Visual search is a separate provider (`get_visual_search()`) that handles embedding + search + ingestion. Cloud uses native Weaviate `multi2vec-jinaai` module with base64 image blobs.

## DSPy Integration (Phase 4)
- `api/knowledge/` ✅ Atlas + ThorGuard domain knowledge
- `api/prompts/` → DSPy Signatures (pending)
- Compiled prompts saved per-mode: `api/prompts/local/`, `api/prompts/cloud/`

## Anti-Patterns
- Don't add abstractions for one-time operations
- Don't future-proof hypothetical requirements
- Don't refactor unrelated code when fixing bugs
