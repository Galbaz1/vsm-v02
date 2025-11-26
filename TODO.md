# TODO

**Last Updated:** 2024-11-26

## Current State

See `.cursor/rules/progress.mdc` for recent fixes and known issues.

---

## Active Work

### In Progress
- [ ] Benchmark system after hybrid search fix
- [ ] Verify LED query now returns correct answer

### Next Up
- [ ] Run full benchmark suite on all 20 Q&A pairs
- [ ] Document benchmark results in progress.mdc

---

## Backlog

### Agent Improvements
- [ ] Auto-terminate after 3 searches with same results
- [ ] Better termination logic (synthesize after finding answer)
- [ ] Few-shot examples from successful queries

### Performance
- [ ] Benchmark: TTFT, tokens/sec
- [ ] KV cache optimization

### Visual Pipeline
- [ ] MLX VLM integration (Qwen3-VL-8B)
- [ ] VisualInterpretationTool

---

## Completed

### 2024-11-26
- [x] Fixed table content stripped during ingestion (`weaviate_ingest_manual.py`)
- [x] Changed vector search to hybrid (vector + BM25)
- [x] Removed chunk_type filter from tool
- [x] Created two-agent workflow (`/analyze`, `/implement`)
- [x] Set up Cursor hooks (beforeSubmitPrompt, afterFileEdit, stop)
- [x] Created progress tracking rule
- [x] Cleaned up documentation (archived old plans)

### 2024-11-25
- [x] Implemented QueryTracer for debugging
- [x] Created `analyze_with_llm.py` (Gemini sub-agent)
- [x] Implemented Elysia-style agent architecture
- [x] Created Tool base classes
- [x] Refactored agent.py to decision tree pattern

### 2024-11-24
- [x] Initial benchmark: Hit@5 90%, MRR 0.72
