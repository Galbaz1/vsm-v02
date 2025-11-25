# TODO

## Sprint 1: Core Agent (This Week)

### 1.1 Model Setup
- [ ] Pull gpt-oss:120b via Ollama (~65GB download)
- [ ] Test: `ollama run gpt-oss:120b "What is 2+2?"`
- [ ] Verify MoE performance (should be fast despite size)

### 1.2 Decision Agent LLM
- [ ] Create `api/services/llm.py` with Ollama client wrapper
- [ ] Replace `_make_decision()` in agent.py with LLM call
- [ ] Format prompt with: tools, environment, history, errors
- [ ] Parse JSON response: `{tool, inputs, reasoning, should_end}`

### 1.3 Tool Execution Loop
- [ ] Implement `_execute_tool()` with Result/Error handling
- [ ] Add results to Environment after each tool
- [ ] Stream `decision`, `status`, `result`, `error` payloads
- [ ] Test with `/agentic_search?query=voltage`

---

## Sprint 2: Visual Interpretation (Next Week)

### 2.1 VLM Setup
- [ ] Install: `pip install mlx-vlm`
- [ ] Download Qwen3-VL-8B: `mlx_vlm.download mlx-community/Qwen3-VL-8B-Instruct-4bit`
- [ ] Start server: `mlx_vlm.server --model ... --port 8000`

### 2.2 VisualInterpretationTool
- [ ] Create tool that calls MLX VLM endpoint
- [ ] `is_tool_available`: Only when ColQwen results exist
- [ ] Input: page image paths from Environment
- [ ] Output: Result with VLM's description of diagrams/charts

### 2.3 Integration
- [ ] After ColQwenSearchTool, decision agent can choose VLM
- [ ] VLM result goes to Environment
- [ ] SummarizeTool sees both text + visual interpretations

---

## Sprint 3: Synthesis & Polish (Week After)

### 3.1 TextResponseTool
- [ ] Real streaming generation from gpt-oss
- [ ] Include page citations from Environment
- [ ] Handle `end=True` to stop decision loop

### 3.2 SummarizeTool
- [ ] `is_tool_available`: Only when Environment non-empty
- [ ] `run_if_true`: Auto-trigger when env > 50K tokens
- [ ] Synthesize text results + VLM interpretations

### 3.3 Frontend Integration
- [ ] Handle new payload types in Next.js
- [ ] Show VLM interpretations alongside page images
- [ ] Progressive rendering: fast results → ColQwen → VLM

---

## Backlog

### Agent Improvements
- [ ] Conversation history persistence (export to Weaviate)
- [ ] Retry logic for recoverable errors
- [ ] Few-shot examples from past successful runs

### Performance
- [ ] Benchmark: TTFT, tokens/sec on M3 256GB
- [ ] KV cache optimization for multi-turn

### Multi-User
- [ ] TreeManager for session management
- [ ] Conversation timeout cleanup

---

## Completed

### 2025-11-25
- [x] Research: gpt-oss-120B vs Qwen3 model selection
- [x] Research: Elysia architecture deep dive
- [x] Research: DSPy integration path (via Ollama OpenAI-compat)
- [x] Decision: gpt-oss-120B for LLM, Qwen3-VL-8B for VLM
- [x] Update conventions.mdc with final model stack
- [x] Update philosophy.mdc with weakspots addressed
- [x] Implement Elysia-style Environment class
- [x] Create Tool base with is_tool_available, run_if_true
- [x] Wrap FastVectorSearch and ColQwenSearch as Tools
- [x] Refactor agent.py to decision tree pattern
- [x] Add Result/Error/Response/Status schemas
- [x] Clone Elysia repo to ~/elysia-reference

### 2025-11-24
- [x] Benchmark system with 20 Q&A pairs
- [x] Hit@5: 90%, MRR: 0.72 for both pipelines

---

## Quick Reference

**Start Everything:**
```bash
docker compose up -d           # Weaviate
ollama serve                   # LLM (port 11434)
mlx_vlm.server --port 8000     # VLM (port 8000)
uvicorn api.main:app --port 8001
```

**Test Agent:**
```bash
curl -N "http://localhost:8001/agentic_search?query=show+wiring+diagram"
```

**Key Files:**
- Agent: `api/services/agent.py`
- Tools: `api/services/tools/`
- Environment: `api/services/environment.py`
- Schemas: `api/schemas/agent.py`
- LLM Service: `api/services/llm.py` (TODO)
