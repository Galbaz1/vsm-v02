# Plan: DSPy Migration & Knowledge Context Management

**Created:** 2024-11-26  
**Status:** PLANNING (not implemented)  
**Reference:** Elysia source at `docs/elysia-source-code-for-reference-only/`

---

## 1. The Problem

### 1.1 Hardcoded Prompts

Currently, prompts are hardcoded strings in `api/services/llm.py`:

```python
class DecisionPromptBuilder:
    SYSTEM_PROMPT = """You are an intelligent agent that helps users find information in technical manuals.
    ...
    """
```

**Issues:**
- Hard to modify without code changes
- No structure for inputs/outputs
- Manual JSON parsing of LLM responses
- No type safety

### 1.2 Missing Knowledge Context

The agent has **no awareness** of what documentation it has access to. It doesn't know:
- There are TWO manuals (Technical Manual + Users Manual)
- They cover the SAME product (ThorGuard Intruder Alarm System)
- They serve DIFFERENT purposes (installation vs operation)
- When to search which manual

**Current state:** The agent blindly searches without understanding its knowledge base.

### 1.3 No Context Accumulation Between Tools

Currently, each tool call is somewhat isolated. When tool A retrieves data, tool B doesn't automatically see what was retrieved. There's no smart mechanism to:
- Track what was already retrieved (avoid duplicate searches)
- Share reasoning/decisions across iterations
- Build up a "train of thought" the LLM can follow
- Self-heal from errors with context

### 1.4 Not Reusable

When deployed for a different company/context:
- Would need to modify Python code
- No clear separation between "agent logic" and "domain knowledge"
- No pattern for adding new knowledge sources

---

## 2. How Elysia Solves This

Elysia uses **three key patterns** for intelligent context management:

1. **DSPy Signatures** - Structured prompts with typed I/O
2. **Atlas** - Agent identity + knowledge context
3. **TreeData + Environment** - Smart context accumulation shared between ALL agents

### 2.1 DSPy Signatures (Structured Prompts)

Instead of string templates, Elysia defines prompts as Python classes:

**Reference:** `docs/elysia-source-code-for-reference-only/elysia/tree/prompt_templates.py`

```python
class DecisionPrompt(dspy.Signature):
    """
    You are a routing agent... (docstring = system prompt)
    """
    
    # Typed inputs
    available_actions: list[dict] = dspy.InputField(
        description="List of possible actions to choose from..."
    )
    previous_errors: list[dict] = dspy.InputField(
        description="Errors from previous attempts..."
    )
    
    # Typed outputs
    function_name: str = dspy.OutputField(
        description="Select exactly one function name..."
    )
    function_inputs: dict[str, Any] = dspy.OutputField(
        description="Inputs for the selected function..."
    )
```

**Benefits:**
- Type-safe inputs/outputs
- Automatic JSON extraction (no manual parsing)
- Self-documenting
- Can be optimized by DSPy

### 2.2 Atlas (Agent Identity + Knowledge Context)

Elysia uses an `Atlas` class to define WHO the agent is and WHAT it knows:

**Reference:** `docs/elysia-source-code-for-reference-only/elysia/tree/objects.py` (lines 354-375)

```python
class Atlas(BaseModel):
    style: str = Field(
        default="No style provided.",
        description="The writing style of the agent."
    )
    agent_description: str = Field(
        default="No description provided.",
        description="The description of the process you are following."
    )
    end_goal: str = Field(
        default="No end goal provided.",
        description="A short description of your overall goal."
    )
    datetime_reference: dict = Field(...)  # Current date/time for decisions
```

**This is where knowledge context goes:**
- `agent_description` contains what the agent knows about
- Can include manual descriptions, available data sources, search strategies

### 2.3 TreeData - Centralized State That ALL Agents Share

**Reference:** `docs/elysia-source-code-for-reference-only/elysia/tree/objects.py` (lines 545-952)

`TreeData` is the **central state object** passed to every tool and decision. It accumulates:

```python
class TreeData:
    user_prompt: str              # Current query
    conversation_history: list    # [{"role": "user/assistant", "content": str}]
    environment: Environment      # All retrieved objects (see below)
    tasks_completed: list         # Train of thought - what was done + reasoning
    atlas: Atlas                  # Agent identity + knowledge
    errors: dict                  # Self-healing error context per tool
    num_trees_completed: int      # Iteration count
    collection_data: CollectionData  # Schema info about data sources
```

**Key methods:**
- `update_tasks_completed()` - Logs each decision with reasoning, inputs, outputs
- `tasks_completed_string()` - Formats the entire train of thought for LLM
- `get_errors()` - Returns errors for current tool (self-healing)

### 2.4 Environment - Smart Object Storage

**Reference:** `docs/elysia-source-code-for-reference-only/elysia/tree/objects.py` (lines 15-343)

The `Environment` stores ALL retrieved objects, keyed by tool and result type:

```python
environment = {
    "tool_name": {
        "result_name": [
            {
                "metadata": {"query": "...", "time": 1.2},
                "objects": [
                    {"_REF_ID": "query_AssetManual_0_0", "content": "...", "page": 5},
                    {"_REF_ID": "query_AssetManual_0_1", "content": "...", "page": 12},
                ]
            }
        ]
    }
}
```

**Smart features:**
- Auto-deduplication with `_REF_ID` (marks repeats instead of storing twice)
- `hidden_environment` for cross-tool data NOT shown to LLM
- `is_empty()` to check if any data has been retrieved

**How it accumulates:**
1. Tool A runs search → returns `Result(objects=[...], metadata={...})`
2. Tree calls `environment.add("tool_a", result)`
3. Tool B runs → sees environment with Tool A's results
4. LLM can reference objects by `_REF_ID` in its response

### 2.5 tasks_completed - Train of Thought

**Reference:** `docs/elysia-source-code-for-reference-only/elysia/tree/objects.py` (lines 685-798)

This is the "reasoning log" that accumulates across iterations:

```python
tasks_completed = [
    {
        "prompt": "How do I wire the battery?",
        "task": [
            {
                "task": "fast_vector_search",
                "iteration": 0,
                "action": True,
                "reasoning": "User asking about wiring, need technical manual",
                "inputs": {"query": "battery wiring connection"},
                "parsed_info": "Found 3 results about battery terminals..."
            },
            {
                "task": "text_response",
                "iteration": 0,
                "action": True,
                "reasoning": "Have enough info to answer"
            }
        ]
    }
]
```

The `tasks_completed_string()` method formats this for LLM consumption:
```
<prompt_1>
Prompt: How do I wire the battery?
<task_1>
Chosen action: fast_vector_search (SUCCESSFUL)
Reasoning: User asking about wiring, need technical manual
Parsed_info: Found 3 results about battery terminals...
</task_1>
</prompt_1>
```

### 2.6 ElysiaChainOfThought (Dynamic Signature Extension)

**Reference:** `docs/elysia-source-code-for-reference-only/elysia/util/elysia_chain_of_thought.py`

This is the **magic module** that automatically injects ALL accumulated context into ANY signature:

```python
class ElysiaChainOfThought(Module):
    def __init__(
        self,
        signature: Type[Signature],
        tree_data: TreeData,          # ← Gets TreeData with all accumulated state
        reasoning: bool = True,
        impossible: bool = True,
        message_update: bool = True,
        environment: bool = False,     # ← Optional: include retrieved objects
        collection_schemas: bool = False,
        tasks_completed: bool = False, # ← Optional: include train of thought
        **config,
    ):
        # AUTOMATICALLY PREPEND common inputs to ANY signature
        extended = signature.prepend(name="user_prompt", field=..., type_=str)
        extended = extended.append(name="conversation_history", field=..., type_=list[dict])
        extended = extended.append(name="atlas", field=..., type_=Atlas)
        extended = extended.append(name="previous_errors", field=..., type_=list[dict])
        
        # OPTIONALLY add rich context
        if environment:
            extended = extended.append(name="environment", field=..., type_=dict)
        if tasks_completed:
            extended = extended.append(name="tasks_completed", field=..., type_=str)
        
        self.predict = dspy.Predict(extended)
    
    def _add_tree_data_inputs(self, kwargs: dict):
        """Called before forward() - injects TreeData into kwargs"""
        kwargs["user_prompt"] = self.tree_data.user_prompt
        kwargs["conversation_history"] = self.tree_data.conversation_history
        kwargs["atlas"] = self.tree_data.atlas
        kwargs["previous_errors"] = self.tree_data.get_errors()
        
        if self.environment:
            kwargs["environment"] = self.tree_data.environment.environment
        if self.tasks_completed:
            kwargs["tasks_completed"] = self.tree_data.tasks_completed_string()
        
        return kwargs
    
    async def aforward(self, **kwargs):
        kwargs = self._add_tree_data_inputs(kwargs)  # ← Auto-inject context
        return await self.predict.acall(**kwargs)
```

**This is how context flows between ALL agents:**
1. Tool A runs → updates `tree_data.environment` and `tree_data.tasks_completed`
2. Tool B uses `ElysiaChainOfThought(signature, tree_data, environment=True, tasks_completed=True)`
3. Tool B's signature automatically sees: user prompt + history + atlas + Tool A's results + reasoning

### 2.7 The Tree's `_update_environment()` Method

**Reference:** `docs/elysia-source-code-for-reference-only/elysia/tree/tree.py` (lines 1236-1271)

When ANY tool returns a `Result`, the Tree automatically:
```python
async def _evaluate_result(self, result, decision):
    if isinstance(result, Result):
        # 1. Add to environment (shared object store)
        self.tree_data.environment.add(decision.function_name, result)
        
        # 2. Add to tasks_completed (train of thought)
        self.tree_data.update_tasks_completed(
            prompt=self.user_prompt,
            task=decision.function_name,
            reasoning=decision.reasoning,
            inputs=decision.function_inputs,
            parsed_info=result.llm_parse(),  # ← Tool's summary for LLM
        )
```

**This means tools don't need to manually share context - it happens automatically.**

---

## 3. Proposed Solution for VSM

### 3.1 Create Knowledge Module (Atlas + Domain Context)

```
api/knowledge/
├── __init__.py          # Atlas class + base loader
└── thorguard.py         # ThorGuard-specific knowledge
```

**`api/knowledge/__init__.py`:**
```python
from pydantic import BaseModel, Field
from datetime import datetime

class Atlas(BaseModel):
    """Agent identity + knowledge context. Passed to EVERY LLM call."""
    style: str = Field(default="Professional, cite page numbers")
    agent_description: str = Field(default="No description provided.")
    end_goal: str = Field(default="User has the information they need")
    datetime_reference: dict = Field(default_factory=lambda: {
        "current_datetime": datetime.now().isoformat(),
        "current_day": datetime.now().strftime("%A"),
    })
```

**`api/knowledge/thorguard.py`:**
```python
AGENT_DESCRIPTION = """
You are a technical assistant for the ThorGuard Intruder Alarm System.

## Available Documentation

### 1. Technical Manual (techman.pdf) → Collection: AssetManual
- Audience: Installers, Electricians
- Focus: Hardware, wiring, mounting, jumpers, voltages
- Use for: "How do I wire...", "What voltage...", "Where to connect..."

### 2. Users Manual (uk_firmware.pdf) → Collection: AssetManual
- Audience: End users, Security guards
- Focus: Software, menus, keypad operation, LED meanings
- Use for: "How do I set...", "What does LED mean...", "Menu navigation..."

## Search Strategy
- Installation/wiring questions → Prioritize Technical Manual
- Operation/menu questions → Prioritize Users Manual
- Component questions (e.g., "RKP keypad") → Search BOTH (they cover different aspects)
- If first search insufficient → Search the OTHER manual

## Important
These manuals are COMPLEMENTARY. The Technical Manual tells you HOW to install hardware.
The Users Manual tells you HOW to operate it once installed.
"""
```

### 3.2 Enhance Environment with Elysia Patterns

**Modify `api/services/environment.py`:**
```python
class Environment:
    """Store of ALL retrieved objects, keyed by tool → result_name."""
    
    def __init__(self):
        self.environment: Dict[str, Dict[str, List[Dict]]] = {}
        self.hidden_environment: Dict[str, Any] = {}  # Cross-tool data NOT shown to LLM
    
    def add(self, tool_name: str, result: Result):
        """Auto-called by agent when tool returns Result."""
        if tool_name not in self.environment:
            self.environment[tool_name] = {}
        
        name = result.name  # e.g., "AssetManual" or "PDFDocuments"
        if name not in self.environment[tool_name]:
            self.environment[tool_name][name] = []
        
        # Add with _REF_ID for LLM reference
        objects_with_refs = []
        for i, obj in enumerate(result.objects):
            ref_id = f"{tool_name}_{name}_{len(self.environment[tool_name][name])}_{i}"
            objects_with_refs.append({"_REF_ID": ref_id, **obj})
        
        self.environment[tool_name][name].append({
            "metadata": result.metadata,
            "objects": objects_with_refs,
        })
    
    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in self.environment.values())
```

**Enhance `TreeData` with tasks_completed:**
```python
@dataclass
class TreeData:
    user_prompt: str = ""
    environment: Environment = field(default_factory=Environment)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    tasks_completed: List[Dict] = field(default_factory=list)  # NEW: Train of thought
    atlas: Atlas = field(default_factory=lambda: Atlas(...))   # NEW: Agent identity
    errors: Dict[str, List[str]] = field(default_factory=dict)
    num_iterations: int = 0
    max_iterations: int = 10
    
    def update_tasks_completed(self, task: str, reasoning: str, inputs: dict, parsed_info: str):
        """Log each decision for LLM context in subsequent calls."""
        self.tasks_completed.append({
            "prompt": self.user_prompt,
            "task": task,
            "iteration": self.num_iterations,
            "reasoning": reasoning,
            "inputs": inputs,
            "parsed_info": parsed_info,  # Tool's summary for LLM
        })
    
    def tasks_completed_string(self) -> str:
        """Format train of thought for LLM consumption."""
        out = ""
        for i, task in enumerate(self.tasks_completed):
            out += f"<task_{i+1}>\n"
            out += f"Action: {task['task']}\n"
            out += f"Reasoning: {task['reasoning']}\n"
            out += f"Result: {task['parsed_info']}\n"
            out += f"</task_{i+1}>\n"
        return out
```

### 3.3 Create DSPy Signatures

```
api/prompts/
├── __init__.py
├── signatures.py         # DecisionSignature, TextResponseSignature
└── chain_of_thought.py   # VSMChainOfThought (Elysia pattern)
```

**`api/prompts/signatures.py`:**
```python
import dspy

class DecisionSignature(dspy.Signature):
    """Route queries to appropriate tools based on user question and available documentation."""
    
    available_tools: list[dict] = dspy.InputField(
        description="List of tools with name, description, inputs schema"
    )
    
    tool_name: str = dspy.OutputField(
        description="Exactly one tool name from available_tools"
    )
    tool_inputs: dict = dspy.OutputField(
        description="Inputs for the selected tool"
    )
    should_end: bool = dspy.OutputField(
        description="True if this response completes the user's request"
    )


class TextResponseSignature(dspy.Signature):
    """Generate final response to user based on retrieved information."""
    
    retrieved_context: str = dspy.InputField(
        description="Information retrieved from manuals"
    )
    
    response: str = dspy.OutputField(
        description="Answer to user's question, citing sources"
    )
```

### 3.4 Create VSMChainOfThought (Context Injection)

**`api/prompts/chain_of_thought.py`:**
```python
import dspy
from dspy import Module, Signature
from typing import Type

class VSMChainOfThought(Module):
    """
    Automatically injects TreeData context into any signature.
    Pattern from: docs/elysia-source-code-for-reference-only/elysia/util/elysia_chain_of_thought.py
    """
    
    def __init__(
        self,
        signature: Type[Signature],
        tree_data: "TreeData",
        environment: bool = False,
        tasks_completed: bool = False,
        reasoning: bool = True,
    ):
        super().__init__()
        self.tree_data = tree_data
        self.include_environment = environment
        self.include_tasks = tasks_completed
        
        # Dynamically extend signature with common inputs
        extended = signature
        extended = extended.prepend("user_prompt", dspy.InputField(desc="User's question"))
        extended = extended.append("conversation_history", dspy.InputField(desc="Previous messages"))
        extended = extended.append("atlas", dspy.InputField(desc="Agent identity and knowledge context"))
        extended = extended.append("previous_errors", dspy.InputField(desc="Errors from previous attempts"))
        
        if environment:
            extended = extended.append("environment", dspy.InputField(desc="Retrieved objects so far"))
        if tasks_completed:
            extended = extended.append("tasks_completed", dspy.InputField(desc="Actions taken so far with reasoning"))
        if reasoning:
            extended = extended.append("reasoning", dspy.OutputField(desc="Brief explanation of decision"))
        
        self.predict = dspy.Predict(extended)
    
    def _inject_context(self, kwargs: dict) -> dict:
        """Inject TreeData into kwargs before prediction."""
        kwargs["user_prompt"] = self.tree_data.user_prompt
        kwargs["conversation_history"] = self.tree_data.conversation_history
        kwargs["atlas"] = self.tree_data.atlas.model_dump()
        kwargs["previous_errors"] = self.tree_data.errors.get(self.tree_data.current_tool, [])
        
        if self.include_environment:
            kwargs["environment"] = self.tree_data.environment.environment
        if self.include_tasks:
            kwargs["tasks_completed"] = self.tree_data.tasks_completed_string()
        
        return kwargs
    
    def forward(self, **kwargs):
        kwargs = self._inject_context(kwargs)
        return self.predict(**kwargs)
    
    async def aforward(self, **kwargs):
        kwargs = self._inject_context(kwargs)
        return await self.predict.acall(**kwargs)
```

### 3.5 Update Agent to Auto-Update Context

**Modify `api/services/agent.py`:**
```python
async def _execute_tool(self, tool, tree_data, inputs, decision):
    """Execute tool and auto-update environment + tasks_completed."""
    
    async for result in tool(tree_data=tree_data, inputs=inputs):
        if isinstance(result, Result):
            # AUTO-UPDATE: Add to environment (shared object store)
            tree_data.environment.add(tool.name, result)
            
            # AUTO-UPDATE: Add to tasks_completed (train of thought)
            tree_data.update_tasks_completed(
                task=tool.name,
                reasoning=decision.reasoning,
                inputs=inputs,
                parsed_info=result.llm_message,  # Tool's summary for LLM
            )
        
        yield result

async def _make_llm_decision(self, tree_data, available_tools):
    """Use DSPy with auto-injected context."""
    module = VSMChainOfThought(
        DecisionSignature,
        tree_data=tree_data,
        environment=not tree_data.environment.is_empty(),  # Include if we have data
        tasks_completed=len(tree_data.tasks_completed) > 0,  # Include if we've done things
    )
    
    result = await module.aforward(
        available_tools=[t.to_dict() for t in available_tools],
    )
    
    return Decision(
        tool_name=result.tool_name,
        inputs=result.tool_inputs,
        should_end=result.should_end,
        reasoning=result.reasoning,
    )
```

---

## 4. Elysia Reference Files

All paths relative to `docs/elysia-source-code-for-reference-only/`

### Context Management (CRITICAL - READ FIRST)

| File | Lines | What to Learn |
|------|-------|---------------|
| `elysia/tree/objects.py` | 15-343 | **Environment class** - smart object storage with `_REF_ID` |
| `elysia/tree/objects.py` | 354-376 | **Atlas class** - agent identity + knowledge context |
| `elysia/tree/objects.py` | 545-800 | **TreeData class** - central state, `tasks_completed`, `update_tasks_completed()` |
| `elysia/tree/objects.py` | 759-798 | `tasks_completed_string()` - formats train of thought for LLM |
| `elysia/util/elysia_chain_of_thought.py` | 1-420 | **ElysiaChainOfThought** - auto context injection pattern |
| `elysia/util/elysia_chain_of_thought.py` | 316-343 | `_add_tree_data_inputs()` - how context is injected |

### Tree Execution (How Context Flows)

| File | Lines | What to Learn |
|------|-------|---------------|
| `elysia/tree/tree.py` | 142-154 | Tree initialization with TreeData + Atlas |
| `elysia/tree/tree.py` | 1236-1271 | `_update_environment()` - auto-updates after tool execution |
| `elysia/tree/tree.py` | 1299-1349 | `_evaluate_result()` - how Results become environment entries |
| `elysia/tree/tree.py` | 1655-1661 | `update_tasks_completed()` call after each decision |

### DSPy Signatures

| File | Lines | What to Learn |
|------|-------|---------------|
| `elysia/tree/prompt_templates.py` | 1-145 | **DecisionPrompt** - main routing signature |
| `elysia/tools/text/prompt_templates.py` | all | TextResponsePrompt, SummarizingPrompt |
| `elysia/tools/retrieval/prompt_templates.py` | all | Query-related signatures |

### Base Classes

| File | What to Learn |
|------|---------------|
| `elysia/objects.py` | Tool, Result, Error, Status base classes |
| `elysia/config.py` | Settings pattern with LM configuration |

### API Layer (Reference)

| File | What to Learn |
|------|---------------|
| `elysia/api/services/tree.py` | How they expose Tree as a service |
| `elysia/api/routes/query.py` | Query endpoint structure |

---

## 5. Implementation Phases

### Phase 1: Knowledge Module + Atlas (LOW RISK)
- [ ] Create `api/knowledge/__init__.py` with Atlas class (copy Elysia pattern)
- [ ] Create `api/knowledge/thorguard.py` with manual descriptions
- [ ] Test Atlas loads correctly

### Phase 2: Enhance Environment + TreeData (MEDIUM RISK)
- [ ] Add `hidden_environment` to Environment class
- [ ] Add `_REF_ID` auto-assignment to `Environment.add()`
- [ ] Add `tasks_completed` list to TreeData
- [ ] Add `update_tasks_completed()` method
- [ ] Add `tasks_completed_string()` method for LLM formatting
- [ ] Add Atlas to TreeData
- [ ] Test context accumulation manually

### Phase 3: DSPy Setup (LOW RISK)
- [ ] Add `dspy-ai` to requirements.txt
- [ ] Configure DSPy with Ollama in `api/services/llm.py`
- [ ] Verify DSPy + Ollama connection works

### Phase 4: Create DSPy Signatures (MEDIUM RISK)
- [ ] Create `api/prompts/__init__.py`
- [ ] Create `api/prompts/signatures.py` (DecisionSignature, TextResponseSignature)
- [ ] Create `api/prompts/chain_of_thought.py` (VSMChainOfThought)
- [ ] Test signatures with mock data

### Phase 5: Migrate Agent (HIGHER RISK)
- [ ] Update `_execute_tool` to auto-update environment + tasks_completed
- [ ] Update `_make_llm_decision` to use VSMChainOfThought
- [ ] Keep old `DecisionPromptBuilder` as fallback initially
- [ ] Add feature flag to switch between old/new decision logic

### Phase 6: Migrate Tools (MEDIUM RISK)
- [ ] Update TextResponseTool to use DSPy signature + environment context
- [ ] Ensure tool Results include `llm_message` for tasks_completed
- [ ] Test tools work with context injection

### Phase 7: Testing + Validation
- [ ] Run benchmark to verify no regression
- [ ] Test multi-step queries (verify context accumulates)
- [ ] Test error recovery (verify errors show in next decision)
- [ ] Test ThorGuard-specific routing (Technical vs Users manual)
- [ ] Remove old `DecisionPromptBuilder` if all tests pass

---

## 6. DSPy + Ollama Configuration

DSPy supports Ollama natively:

```python
import dspy

# Configure for local Ollama
lm = dspy.LM(
    'ollama_chat/gpt-oss:120b',
    api_base='http://localhost:11434',
    api_key=''
)
dspy.configure(lm=lm)
```

---

## 7. Success Criteria

1. **Knowledge-aware routing:** Agent prefers Technical Manual for wiring, Users Manual for menus
2. **Context accumulates:** Second tool call sees results from first tool call in environment
3. **Train of thought:** LLM sees `tasks_completed` with reasoning from previous decisions
4. **Self-healing:** Errors from tool A visible in tool B's next decision
5. **No hardcoded prompts:** All prompts defined as DSPy Signatures
6. **Reusable:** Adding a new domain = creating a new knowledge file (Atlas)
7. **No regression:** Benchmark scores stay same or improve
8. **Type safety:** LLM responses parsed into typed objects automatically

---

## 8. Files to Modify

| File | Change |
|------|--------|
| `requirements.txt` | Add `dspy-ai` |
| `api/services/environment.py` | Enhance Environment (hidden_env, _REF_ID), enhance TreeData (tasks_completed, Atlas) |
| `api/services/llm.py` | Add DSPy LM config, keep DecisionPromptBuilder as fallback |
| `api/services/agent.py` | Auto-update context in `_execute_tool`, use VSMChainOfThought |
| `api/services/tools/base.py` | Ensure Result has `llm_message` field |
| **NEW** `api/knowledge/__init__.py` | Atlas class |
| **NEW** `api/knowledge/thorguard.py` | Manual descriptions + search strategy |
| **NEW** `api/prompts/__init__.py` | Module init |
| **NEW** `api/prompts/signatures.py` | DecisionSignature, TextResponseSignature |
| **NEW** `api/prompts/chain_of_thought.py` | VSMChainOfThought (context injection) |

---

## 9. Context Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                     │
│                    "How do I wire the backup battery?"                   │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TREE DATA (Central State)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐ │
│  │     ATLAS       │  │   ENVIRONMENT   │  │    TASKS_COMPLETED       │ │
│  │ agent_desc:     │  │ (empty at start)│  │ (empty at start)         │ │
│  │ "ThorGuard..."  │  │                 │  │                          │ │
│  └─────────────────┘  └─────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │          VSMChainOfThought(DecisionSignature)         │
              │  AUTO-INJECTS: user_prompt, atlas, conversation_history │
              └───────────────────────────┬───────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DECISION 1: fast_vector_search                                          │
│  Reasoning: "User asking about wiring, need Technical Manual"            │
│  Inputs: {query: "backup battery wiring connection"}                     │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     TOOL EXECUTES + AUTO-UPDATE                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ tree_data.environment.add("fast_vector_search", Result(              ││
│  │     objects=[{page: 45, content: "Connect red wire to..."}],         ││
│  │     metadata={query: "battery wiring", collection: "AssetManual"}    ││
│  │ ))                                                                   ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ tree_data.update_tasks_completed(                                    ││
│  │     task="fast_vector_search",                                       ││
│  │     reasoning="User asking about wiring, need Technical Manual",     ││
│  │     parsed_info="Found 3 results about battery terminals..."         ││
│  │ )                                                                    ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TREE DATA (After Tool 1)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐ │
│  │     ATLAS       │  │   ENVIRONMENT   │  │    TASKS_COMPLETED       │ │
│  │ agent_desc:     │  │ fast_vector_    │  │ [{task: "fast_vector_    │ │
│  │ "ThorGuard..."  │  │   search:       │  │   search", reasoning:    │ │
│  │                 │  │   AssetManual:  │  │   "User asking...",      │ │
│  │                 │  │   [{objects:    │  │   parsed_info: "Found    │ │
│  │                 │  │     [...]}]     │  │   3 results..."}]        │ │
│  └─────────────────┘  └─────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │          VSMChainOfThought(DecisionSignature)         │
              │  AUTO-INJECTS: user_prompt, atlas, conversation_history │
              │  + environment (now has data!)                          │
              │  + tasks_completed (now has reasoning!)                 │
              └───────────────────────────┬───────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DECISION 2: text_response                                               │
│  Reasoning: "Have enough info from Technical Manual to answer"           │
│  should_end: true                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** The LLM in Decision 2 sees EVERYTHING from Decision 1 automatically:
- The retrieved objects in `environment`
- The reasoning and results in `tasks_completed`
- The agent's knowledge context in `atlas`

This is what makes multi-step reasoning work without manual context threading.

---

## 10. Rollback Plan

If DSPy migration causes issues:
1. Keep old `DecisionPromptBuilder` until new approach is validated
2. Feature flag to switch between old/new decision logic
3. Knowledge module (Atlas) can be used independently of DSPy
4. Environment + tasks_completed enhancements work without DSPy

