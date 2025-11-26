# VSM v02 - DSPy Migration & Benchmarking TODO

**Created:** 2025-11-26  
**Plan Reference:** `PLAN_DSPY_MIGRATION.md`  
**Status:** 🟢 Phase 0-2 Complete, Ready for Phase 3

---

## Overview

This TODO tracks the DSPy migration and interactive benchmarking system implementation. Phases are ordered by risk (lowest first) and dependency (prerequisites first). Each phase is testable independently.

---

## Phase 0: Prerequisites & Setup ✅
> **Risk:** LOW | **Effort:** 1 hour | **Dependencies:** None | **Completed:** 2025-11-26

### 0.1 Environment Setup
- [x] Add `dspy-ai` to `requirements.txt`
- [x] Add `mlflow` to `requirements.txt` (optional, for experiment tracking)
- [x] Run `pip install -r requirements.txt` in `vsm-hva` conda env
- [x] Verify DSPy import works: `python -c "import dspy; print(dspy.__version__)"`

### 0.2 DSPy + Ollama Configuration
- [x] Create `api/core/dspy_config.py` with Ollama LM setup
- [x] Test DSPy → Ollama connection with simple signature
- [x] Verify async support works (`acall`)

**Test:** `python -c "import dspy; lm = dspy.LM('ollama_chat/gpt-oss:120b', api_base='http://localhost:11434'); dspy.configure(lm=lm); print('OK')"`

---

## Phase 1: Knowledge Module (Atlas) ✅
> **Risk:** LOW | **Effort:** 2-3 hours | **Dependencies:** Phase 0 | **Completed:** 2025-11-26

### 1.1 Create Atlas Class
- [x] Create `api/knowledge/__init__.py` with `Atlas` Pydantic model
- [x] Fields: `style`, `agent_description`, `end_goal`, `datetime_reference`
- [x] Copy pattern from Elysia: `docs/elysia-source-code-for-reference-only/elysia/tree/objects.py:354-375`

### 1.2 Create ThorGuard Knowledge
- [x] Create `api/knowledge/thorguard.py`
- [x] Define `AGENT_DESCRIPTION` with:
  - Technical Manual description (audience, topics, use cases)
  - Users Manual description (audience, topics, use cases)
  - Search strategy guidance
- [x] Create `get_atlas()` factory function

**Test:** `python -c "from api.knowledge import get_atlas; a = get_atlas(); print(a.agent_description[:100])"`

---

## Phase 2: Enhanced Environment & TreeData ✅
> **Risk:** MEDIUM | **Effort:** 4-6 hours | **Dependencies:** Phase 1 | **Completed:** 2025-11-26

### Current State Analysis
The current implementation in `api/services/environment.py` already has:
- ✅ `Environment.hidden_environment` - already implemented (lines 67-77)
- ✅ `Environment.is_empty()` - already implemented (lines 79-85)
- ✅ `Environment._REF_ID` auto-assignment - already implemented in `add_objects()` (lines 127-131)
- ✅ `TreeData.atlas` field - already implemented (line 291)
- ✅ `TreeData.tasks_completed` - restructured to rich format

**What's MISSING** (compared to Elysia pattern at `docs/elysia-source-code-for-reference-only/elysia/tree/objects.py:685-798`):

### 2.1 Restructure `tasks_completed` Format
- [x] Change `tasks_completed` from `Dict[str, List[float]]` to `List[Dict]` format
- [x] New format should match Elysia structure:
  ```python
  [
      {
          "prompt": str,  # The user prompt this task was for
          "task": [
              {
                  "task": str,           # Tool name
                  "iteration": int,      # Which iteration
                  "reasoning": str,      # Why this tool was chosen
                  "inputs": dict,        # Tool inputs used
                  "llm_message": str,    # Output message for context
                  "action": bool,        # True if action tool, False if subcategory
                  "error": bool,         # True if task errored
              },
              ...
          ]
      },
      ...
  ]
  ```

**Reference:** Elysia `update_tasks_completed()` at lines 685-742

### 2.2 Add `update_tasks_completed()` Method
- [x] Implement `update_tasks_completed(prompt, task, num_iterations, **kwargs)` method
- [x] Handle three cases:
  1. New prompt - create new entry
  2. Existing prompt, new task - append to task list
  3. Existing prompt, existing task - update with kwargs
- [x] Support kwargs: `reasoning`, `inputs`, `llm_message`, `action`, `error`

**Elysia Pattern:**
```python
def update_tasks_completed(self, prompt: str, task: str, num_trees_completed: int, **kwargs):
    # Search for existing prompt entry
    prompt_found = False
    task_found = False
    # ... (see elysia/tree/objects.py:685-742)
```

### 2.3 Add `tasks_completed_string()` Method
- [x] Implement `tasks_completed_string()` method for LLM consumption
- [x] Format output with XML-like tags for structure:
  ```
  <prompt_1>
  Prompt: {user_prompt}
  <task_1>
  Chosen action: {task_name} (SUCCESSFUL/UNSUCCESSFUL)
  Reasoning: {reasoning}
  Inputs: {inputs}
  LLM Message: {llm_message}
  </task_1>
  </prompt_1>
  ```

**Reference:** Elysia `tasks_completed_string()` at lines 759-798

### 2.4 Add `current_tool` Field for Error Context
- [x] Add `current_tool: Optional[str] = None` field to TreeData
- [x] Add `set_current_tool(task: str)` method
- [x] Modify `get_errors()` to filter by current tool (for self-healing)
- [x] Update error structure from `List[str]` to `Dict[str, List[str]]`

**Reference:** Elysia pattern at lines 744-757

### 2.5 Wire Up Atlas in Agent
- [x] Update `AgentOrchestrator.run()` (line 364) to pass Atlas to TreeData:
  ```python
  from api.knowledge.thorguard import get_atlas
  
  tree_data = TreeData(
      user_prompt=user_prompt,
      environment=Environment(),
      conversation_history=conversation_history or [],
      collection_names=self.collection_names,
      max_iterations=self.max_iterations,
      atlas=get_atlas(),  # ADD THIS
  )
  ```

### 2.6 Update `_execute_tool()` to Track Tasks
- [x] Modify `_execute_tool()` (line 290) to call `update_tasks_completed()`:
  ```python
  # After successful tool execution
  tree_data.update_tasks_completed(
      prompt=tree_data.user_prompt,
      task=tool.name,
      num_iterations=tree_data.num_iterations,
      reasoning=decision.reasoning if hasattr(decision, 'reasoning') else "",
      inputs=inputs,
      llm_message=result.llm_message if hasattr(result, 'llm_message') else "",
      action=True,
      error=not successful,
  )
  ```

### 2.7 Update Serialization
- [x] Update `TreeData.to_json()` to handle new `tasks_completed` format
- [x] Update `TreeData.from_json()` to deserialize correctly (with backwards compatibility)

---

### DSPy Integration Notes

From `docs/dspy_generated_llms.txt`, key concepts for Phase 2:

1. **Module Pattern** (lines 199-211): TreeData will be passed to DSPy modules
2. **ChainOfThought** (lines 24, 360-368): Will use `tasks_completed_string()` for context
3. **Streaming** (lines 21, 330-352): Ensure updates work with async generators

### Elysia Patterns Used

| Pattern | Elysia Location | VSM Implementation |
|---------|-----------------|-------------------|
| `tasks_completed` structure | lines 616-619 | `TreeData.tasks_completed` |
| `update_tasks_completed()` | lines 685-742 | New method |
| `tasks_completed_string()` | lines 759-798 | New method |
| `current_task` + error filtering | lines 744-757 | New field + method |
| `tree_count_string()` | lines 808-814 | Existing `iteration_status()` ✅ |

### Implementation Order

1. **Step 1:** Restructure `tasks_completed` type definition
2. **Step 2:** Add `update_tasks_completed()` method
3. **Step 3:** Add `tasks_completed_string()` method  
4. **Step 4:** Add `current_tool` and error filtering
5. **Step 5:** Wire Atlas in agent
6. **Step 6:** Update `_execute_tool()` to track tasks
7. **Step 7:** Update serialization methods
8. **Step 8:** Test with mock data

---

**Test Cases:**
```bash
# Test 1: tasks_completed structure
python -c "
from api.services.environment import TreeData
td = TreeData(user_prompt='test')
td.update_tasks_completed('test', 'fast_vector_search', 1, reasoning='test reason', inputs={'query': 'test'})
print(td.tasks_completed)
"

# Test 2: tasks_completed_string output
python -c "
from api.services.environment import TreeData
td = TreeData(user_prompt='test')
td.update_tasks_completed('test', 'fast_vector_search', 1, reasoning='found docs', llm_message='Found 5 results')
print(td.tasks_completed_string())
"

# Test 3: Atlas integration
python -c "
from api.knowledge.thorguard import get_atlas
from api.services.environment import TreeData
atlas = get_atlas()
td = TreeData(user_prompt='test', atlas=atlas)
print(f'Atlas loaded: {td.atlas.style[:50]}...')
"
```

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
| `requirements.txt` | 0 ✅ | Add dspy-ai, mlflow |
| `api/services/environment.py` | 2 | Restructure `tasks_completed`, add `update_tasks_completed()`, `tasks_completed_string()`, `current_tool` |
| `api/services/agent.py` | 2 | Wire Atlas, update `_execute_tool()` for task tracking |
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
| DSPy + Ollama works | ✅ | ✅ | 0 |
| Atlas loads correctly | ✅ | ✅ | 1 |
| Context accumulates | ✅ | ✅ | 2 |
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

