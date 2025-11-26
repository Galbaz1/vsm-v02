# VSM v02 - DSPy Migration & Benchmarking TODO

**Created:** 2025-11-26  
**Plan Reference:** `PLAN_DSPY_MIGRATION.md`  
**Status:** 🟡 Planning Complete, Implementation Not Started

---

## Overview

This TODO tracks the DSPy migration and interactive benchmarking system implementation. Phases are ordered by risk (lowest first) and dependency (prerequisites first). Each phase is testable independently.

---

## Phase 0: Prerequisites & Setup
> **Risk:** LOW | **Effort:** 1 hour | **Dependencies:** None

### 0.1 Environment Setup
- [ ] Add `dspy-ai` to `requirements.txt`
- [ ] Add `mlflow` to `requirements.txt` (optional, for experiment tracking)
- [ ] Run `pip install -r requirements.txt` in `vsm-hva` conda env
- [ ] Verify DSPy import works: `python -c "import dspy; print(dspy.__version__)"`

### 0.2 DSPy + Ollama Configuration
- [ ] Create `api/core/dspy_config.py` with Ollama LM setup
- [ ] Test DSPy → Ollama connection with simple signature
- [ ] Verify async support works (`acall`)

**Test:** `python -c "import dspy; lm = dspy.LM('ollama_chat/gpt-oss:120b', api_base='http://localhost:11434'); dspy.configure(lm=lm); print('OK')"`

---

## Phase 1: Knowledge Module (Atlas)
> **Risk:** LOW | **Effort:** 2-3 hours | **Dependencies:** Phase 0

### 1.1 Create Atlas Class
- [ ] Create `api/knowledge/__init__.py` with `Atlas` Pydantic model
- [ ] Fields: `style`, `agent_description`, `end_goal`, `datetime_reference`
- [ ] Copy pattern from Elysia: `docs/elysia-source-code-for-reference-only/elysia/tree/objects.py:354-375`

### 1.2 Create ThorGuard Knowledge
- [ ] Create `api/knowledge/thorguard.py`
- [ ] Define `AGENT_DESCRIPTION` with:
  - Technical Manual description (audience, topics, use cases)
  - Users Manual description (audience, topics, use cases)
  - Search strategy guidance
- [ ] Create `get_atlas()` factory function

**Test:** `python -c "from api.knowledge import get_atlas; a = get_atlas(); print(a.agent_description[:100])"`

---

## Phase 2: Enhanced Environment & TreeData
> **Risk:** MEDIUM | **Effort:** 4-6 hours | **Dependencies:** Phase 1

### 2.1 Enhance Environment Class
- [ ] Add `hidden_environment: Dict[str, Any]` for cross-tool data
- [ ] Add `_REF_ID` auto-assignment in `add()` method
- [ ] Add `is_empty()` method
- [ ] Update `__str__` to format for LLM consumption

### 2.2 Enhance TreeData Class
- [ ] Add `tasks_completed: List[Dict]` field
- [ ] Add `atlas: Atlas` field
- [ ] Add `update_tasks_completed(task, reasoning, inputs, parsed_info)` method
- [ ] Add `tasks_completed_string()` method for LLM formatting
- [ ] Add `current_tool: str` for error context

### 2.3 Wire Up Atlas
- [ ] Modify TreeData initialization to accept Atlas
- [ ] Update `AgentService.run()` to create TreeData with Atlas

**Test:** Manual test with mock data - verify `tasks_completed_string()` output format matches Elysia pattern.

---

## Phase 3: DSPy Signatures
> **Risk:** MEDIUM | **Effort:** 4-6 hours | **Dependencies:** Phase 0

### 3.1 Create Signature Module
- [ ] Create `api/prompts/__init__.py`
- [ ] Create `api/prompts/signatures.py`

### 3.2 Implement DecisionSignature
- [ ] Define `DecisionSignature(dspy.Signature)`:
  - Inputs: `available_tools: list[dict]`
  - Outputs: `tool_name: str`, `tool_inputs: dict`, `should_end: bool`
- [ ] Add docstring as system prompt (routing agent behavior)

### 3.3 Implement TextResponseSignature
- [ ] Define `TextResponseSignature(dspy.Signature)`:
  - Inputs: `retrieved_context: str`
  - Outputs: `response: str`
- [ ] Add docstring for response generation behavior

### 3.4 Implement VSMChainOfThought
- [ ] Create `api/prompts/chain_of_thought.py`
- [ ] Implement `VSMChainOfThought(dspy.Module)`:
  - Auto-inject: `user_prompt`, `conversation_history`, `atlas`, `previous_errors`
  - Optional: `environment`, `tasks_completed`
  - Pattern from: `docs/elysia-source-code-for-reference-only/elysia/util/elysia_chain_of_thought.py`

**Test:** Unit test signatures with mock inputs, verify structured output parsing.

---

## Phase 4: Judge Service (Benchmarking Foundation)
> **Risk:** LOW | **Effort:** 3-4 hours | **Dependencies:** Phase 0

### 4.1 Create Evaluation Module
- [ ] Create `api/evaluation/__init__.py`
- [ ] Create `api/evaluation/judge.py`

### 4.2 Implement TechnicalJudge
- [ ] Define `TechnicalJudgeSignature(dspy.Signature)`:
  - Inputs: `question`, `ground_truth`, `agent_answer`
  - Outputs: `score: int`, `reasoning: str`, `missing_facts: list`, `incorrect_facts: list`
- [ ] Add scoring guide in docstring (90-100, 70-89, etc.)
- [ ] Wrap in `TechnicalJudge(dspy.Module)` with ChainOfThought
- [ ] Implement `forward()` and `aforward()` methods

**Test:** `python -c "from api.evaluation.judge import TechnicalJudge; j = TechnicalJudge(); r = j('What is X?', 'X is Y', 'X is Y'); print(r.score)"`

---

## Phase 5: Benchmark API Endpoints
> **Risk:** LOW | **Effort:** 3-4 hours | **Dependencies:** Phase 4

### 5.1 Create Benchmark Router
- [ ] Create `api/endpoints/benchmark.py`
- [ ] Define Pydantic models: `SuggestResponse`, `EvaluateRequest`, `EvaluateResponse`

### 5.2 Implement Endpoints
- [ ] `GET /benchmark/suggest` - returns random question (NO answer)
- [ ] `GET /benchmark/suggest?category=X` - filter by category
- [ ] `POST /benchmark/evaluate` - calls Judge, returns score
- [ ] `GET /benchmark/categories` - list available categories

### 5.3 Register Router
- [ ] Add router to `api/main.py`
- [ ] Verify endpoints in Swagger docs

**Test:** 
```bash
curl http://localhost:8001/benchmark/suggest
curl http://localhost:8001/benchmark/categories
curl -X POST http://localhost:8001/benchmark/evaluate \
  -H "Content-Type: application/json" \
  -d '{"benchmark_id": 0, "agent_answer": "test answer"}'
```

---

## Phase 6: Frontend Benchmark Mode
> **Risk:** MEDIUM | **Effort:** 6-8 hours | **Dependencies:** Phase 5

### 6.1 Create Benchmark Hook
- [ ] Create `frontend/lib/hooks/useBenchmark.ts`
- [ ] Implement `suggestQuestion(category?)` function
- [ ] Implement `evaluateAnswer(benchmarkId, answer)` function
- [ ] Handle loading and error states

### 6.2 Create BenchmarkMode Component
- [ ] Create `frontend/components/BenchmarkMode.tsx`
- [ ] "Suggest Question" button (all, Complex, Direct)
- [ ] Display current question with category badge
- [ ] "Evaluate" button (appears after agent response)
- [ ] Score display with color coding (green/yellow/orange/red)
- [ ] Missing/incorrect facts lists
- [ ] Collapsible ground truth reveal

### 6.3 Integrate into Main Page
- [ ] Add "Benchmark Mode" toggle to header (next to "Agentic Mode")
- [ ] Show BenchmarkMode component when enabled
- [ ] Wire `onQuestionSelect` to search bar
- [ ] Pass `agentAnswer` and `isAgentDone` from agentic state

**Test:** Manual UI testing - suggest question → run search → evaluate → verify score display.

---

## Phase 7: Migrate Agent to DSPy
> **Risk:** HIGH | **Effort:** 8-12 hours | **Dependencies:** Phases 2, 3

### 7.1 Add Feature Flag
- [ ] Add `USE_DSPY_DECISION` flag to `api/core/config.py`
- [ ] Default to `False` initially

### 7.2 Update Tool Execution
- [ ] Modify `_execute_tool()` to auto-update environment
- [ ] Modify `_execute_tool()` to auto-update tasks_completed
- [ ] Ensure all tools return `Result` with `llm_message`

### 7.3 Implement DSPy Decision Making
- [ ] Create `_make_dspy_decision()` using `VSMChainOfThought`
- [ ] Inject environment if not empty
- [ ] Inject tasks_completed if any
- [ ] Parse structured output to `Decision` object

### 7.4 Integrate with Agent Loop
- [ ] Modify `_make_decision()` to check feature flag
- [ ] Call `_make_dspy_decision()` if flag enabled
- [ ] Fall back to old `DecisionPromptBuilder` if flag disabled

**Test:** 
1. Run with flag OFF - verify no regression
2. Run with flag ON - verify DSPy decisions work
3. Run benchmark - compare scores

---

## Phase 8: Migrate TextResponseTool
> **Risk:** MEDIUM | **Effort:** 4-6 hours | **Dependencies:** Phase 7

### 8.1 Create TextResponse Signature
- [ ] Add `TextResponseSignature` to `api/prompts/signatures.py`
- [ ] Include environment context in inputs
- [ ] Add source citation requirement in docstring

### 8.2 Update TextResponseTool
- [ ] Use `VSMChainOfThought(TextResponseSignature, tree_data, environment=True)`
- [ ] Ensure streaming still works with DSPy
- [ ] Keep old implementation as fallback

**Test:** Run agentic search, verify response quality and source citations.

---

## Phase 9: Batch Evaluation System
> **Risk:** MEDIUM | **Effort:** 4-6 hours | **Dependencies:** Phase 4

### 9.1 Create Devset Loader
- [ ] Create `api/evaluation/devset.py`
- [ ] Implement `load_benchmark()` → `List[dspy.Example]`
- [ ] Implement `split_devset()` for train/dev split

### 9.2 Create Custom Metrics
- [ ] Create `api/evaluation/metrics.py`
- [ ] Implement `PageHitMetric` (MRR-style)
- [ ] Implement `VisualElementMetric`
- [ ] Implement `VSMCombinedMetric` (weighted combination)

### 9.3 Create VSMEvaluator
- [ ] Create `api/evaluation/evaluator.py`
- [ ] Wrap `dspy.Evaluate` with MLflow tracking
- [ ] Add per-category breakdown

### 9.4 Update Benchmark Script
- [ ] Modify `scripts/run_benchmark.py` to use DSPy evaluation
- [ ] Add `--dspy` flag to switch between old/new evaluation
- [ ] Output results in same format for comparison

**Test:** `python scripts/run_benchmark.py --dspy` - compare to old results.

---

## Phase 10: Optimization Loop
> **Risk:** HIGH | **Effort:** 8-12 hours | **Dependencies:** Phases 7, 9

### 10.1 Create Optimization Script
- [ ] Create `scripts/run_optimization.py`
- [ ] Configure MIPROv2 optimizer
- [ ] Use VSMCombinedMetric

### 10.2 Run Optimization
- [ ] Split benchmark into train (70%) / dev (30%)
- [ ] Run MIPROv2 with `max_bootstrapped_demos=2`
- [ ] Save optimized module to JSON

### 10.3 Evaluate Improvement
- [ ] Compare baseline vs optimized on dev set
- [ ] Document improvement percentage
- [ ] If improvement > 5%, deploy optimized prompts

**Test:** Compare Hit@1, Hit@3, SemanticF1 before/after optimization.

---

## Phase 11: Documentation & Cleanup
> **Risk:** LOW | **Effort:** 2-4 hours | **Dependencies:** All previous phases

### 11.1 Update Architecture Docs
- [ ] Update `docs/ARCHITECTURE.md` with DSPy components
- [ ] Add architecture diagram for DSPy flow
- [ ] Document new API endpoints

### 11.2 Update Rules Files
- [ ] Update `.cursor/rules/philosophy.mdc` - remove "DSPy not used"
- [ ] Update `.cursor/rules/progress.mdc` with DSPy migration status

### 11.3 Remove Old Code
- [ ] Remove `DecisionPromptBuilder` if DSPy validated
- [ ] Remove feature flags once stable
- [ ] Archive `docs/BENCHMARK_SYSTEM_DESIGN.md` (superseded)

### 11.4 Final Benchmark
- [ ] Run full benchmark suite
- [ ] Document final scores in `progress.mdc`
- [ ] Create release notes

---

## Quick Reference: File Changes

### New Files
| File | Phase | Purpose |
|------|-------|---------|
| `api/core/dspy_config.py` | 0 | DSPy + Ollama configuration |
| `api/knowledge/__init__.py` | 1 | Atlas class |
| `api/knowledge/thorguard.py` | 1 | ThorGuard domain knowledge |
| `api/prompts/__init__.py` | 3 | Module init |
| `api/prompts/signatures.py` | 3 | DSPy Signatures |
| `api/prompts/chain_of_thought.py` | 3 | VSMChainOfThought |
| `api/evaluation/__init__.py` | 4 | Module init |
| `api/evaluation/judge.py` | 4 | TechnicalJudge |
| `api/evaluation/devset.py` | 9 | Benchmark loader |
| `api/evaluation/metrics.py` | 9 | Custom metrics |
| `api/evaluation/evaluator.py` | 9 | VSMEvaluator |
| `api/endpoints/benchmark.py` | 5 | Benchmark API |
| `frontend/lib/hooks/useBenchmark.ts` | 6 | Benchmark hook |
| `frontend/components/BenchmarkMode.tsx` | 6 | Benchmark UI |
| `scripts/run_optimization.py` | 10 | MIPROv2 optimization |

### Modified Files
| File | Phase | Changes |
|------|-------|---------|
| `requirements.txt` | 0 | Add dspy-ai, mlflow |
| `api/services/environment.py` | 2 | Enhance Environment, TreeData |
| `api/services/agent.py` | 7 | DSPy decision making |
| `api/services/tools/base.py` | 7 | Ensure llm_message in Result |
| `api/main.py` | 5 | Register benchmark router |
| `frontend/app/page.tsx` | 6 | Add Benchmark Mode toggle |
| `scripts/run_benchmark.py` | 9 | DSPy evaluation option |
| `.cursor/rules/philosophy.mdc` | 11 | Update DSPy status |
| `docs/ARCHITECTURE.md` | 11 | Add DSPy architecture |

---

## Success Criteria

| Metric | Current | Target | Phase |
|--------|---------|--------|-------|
| DSPy + Ollama works | ❌ | ✅ | 0 |
| Atlas loads correctly | ❌ | ✅ | 1 |
| Context accumulates | ❌ | ✅ | 2 |
| Judge returns structured output | ❌ | ✅ | 4 |
| Benchmark API works | ❌ | ✅ | 5 |
| Frontend Benchmark Mode | ❌ | ✅ | 6 |
| Agent uses DSPy decisions | ❌ | ✅ | 7 |
| No benchmark regression | TBD | ≥ baseline | 7 |
| Hit@1 | TBD | ≥70% | 9 |
| Hit@3 | TBD | ≥85% | 9 |
| SemanticF1 | TBD | ≥0.7 | 9 |
| Optimization improvement | N/A | ≥5% | 10 |

---

## Notes

- **Rollback:** Each phase can be rolled back independently. Feature flags protect production.
- **Testing:** Test after each phase before proceeding.
- **Dependencies:** Respect phase order - don't skip prerequisites.
- **Time Estimate:** Total ~40-60 hours across all phases.

