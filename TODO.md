# TODO

## Active Sprint

### Agent LLM Integration
- [ ] Install MLX: `pip install mlx mlx-lm`
- [ ] Test Qwen3-14B-4bit loading on M3
- [ ] Replace `_make_decision()` with LLM call
- [ ] Implement `TextResponseTool` with actual generation

### Frontend Streaming
- [ ] Connect `/agentic_search` to new agent
- [ ] Handle `Decision`, `Status`, `Result` payload types
- [ ] Progressive UI: show fast results, then ColQwen refinement

---

## Backlog

### MLX Model Stack
- [ ] Serve Qwen via `mlx_lm.server --port 8000`
- [ ] Add MLX service wrapper in `api/services/mlx.py`
- [ ] Benchmark TTFT and throughput on M3

### Visual Interpretation
- [ ] Load Qwen2.5-VL-7B for page understanding
- [ ] Add `VisualInterpretationTool` after ColQwen search
- [ ] Pass page images to VLM for diagram explanation

### Agent Improvements
- [ ] Add `run_if_true` logic for auto-summarize (>50K tokens)
- [ ] Implement conversation history in TreeData
- [ ] Add retry logic when Error.recoverable=True

---

## Completed

### 2025-11-25
- [x] Implement Elysia-style Environment class
- [x] Create Tool base with `is_tool_available`, `run_if_true`
- [x] Wrap FastVectorSearch and ColQwenSearch as Tools
- [x] Refactor agent.py to decision tree pattern
- [x] Add Result/Error/Response/Status schemas
- [x] Clone Elysia repo to ~/elysia-reference
- [x] Update AGENT_OPTIMIZATION_PLAN.md with Elysia patterns

### 2025-11-24
- [x] Benchmark system with 20 Q&A pairs
- [x] Fix page number references in ground truth
- [x] Hit@5: 90%, MRR: 0.72 for both pipelines

---

## Notes

**MLX Quick Test:**
```bash
pip install mlx mlx-lm
python -c "from mlx_lm import load, generate; m,t = load('mlx-community/Qwen2.5-7B-Instruct-4bit'); print(generate(m,t,'Hello!'))"
```

**Elysia Reference:** `~/elysia-reference`

**Key Files:**
- Agent: `api/services/agent.py`
- Tools: `api/services/tools/`
- Environment: `api/services/environment.py`
- Schemas: `api/schemas/agent.py`

