# Plan: DSPy Migration & Knowledge Context Management

**Created:** 2025-11-26  
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

### 1.3 Context Accumulation: Current State (Researched 2025-11-26)

**What the agent DOES have:**

| Capability | Status | Location |
|------------|--------|----------|
| Tool awareness | ✅ YES | LLM sees all tools with descriptions + inputs in prompt (`llm.py:293-308`) |
| Environment sharing | ✅ YES | All tools read/write to shared `tree_data.environment` (`environment.py:17-249`) |
| Multi-tool looping | ✅ YES | Can chain all 6 tools, up to 10 iterations (`agent.py:378-454`) |
| Error tracking | ✅ YES | `tree_data.errors` shown to LLM (last 3) (`llm.py:331-335`) |
| Auto-triggers | ✅ YES | SummarizeTool auto-runs at 30k tokens (`base.py:417-425`) |

**What the agent DOESN'T have:**

| Gap | Impact | Risk |
|-----|--------|------|
| No decision history | Can't detect loops | Same (tool, query) pair might run 3x |
| No query evolution | Tools always use original prompt | "wiring diagram" searched repeatedly, never refined |
| No duplicate detection | Wastes iterations | ColQwen + FastVector + Hybrid all search same query |
| No result scoring | Can't assess sufficiency | LLM doesn't know if results are "good enough" |
| No manual awareness | Blind routing | Agent doesn't know Tech Manual vs Users Manual purposes |
| No tool chain patterns | Ad-hoc sequencing | Can't plan "search → visualize → respond" |

**Looping Risk Scenarios:**

```
Scenario A: Repetitive Search Loop
  Iter 1: fast_vector_search("wiring diagram") → [text results]
  Iter 2: colqwen_search("wiring diagram") → [same pages as images]
  Iter 3: hybrid_search("wiring diagram") → [combined duplicates]
  ...
  Iter 10: Hit max_iterations with incomplete answer
  
Scenario B: Error Doesn't Learn
  Iter 1: Error: "No results for X"
  Iter 2: LLM sees error but no guidance on HOW to recover
         → Might retry same search with same query
```

**Current Decision Flow:**

```
Per Iteration:
1. _get_available_tools() → Filter by is_tool_available()
2. _check_auto_triggers() → Run tools that should auto-fire
3. _make_decision() → LLM or rule-based fallback
4. _execute_tool() → Run tool, add Result to environment
5. Check should_end → Break or continue loop
```

**Key Code Locations:**
- Tool registration: `agent.py:87-102`
- Tool metadata to LLM: `llm.py:293-308` (format_tools_description)
- LLM decision prompt: `llm.py:262-354` (DecisionPromptBuilder)
- Main loop: `agent.py:378-454`
- Environment storage: `environment.py:84-133`

### 1.4 Not Reusable

When deployed for a different company/context:
- Would need to modify Python code
- No clear separation between "agent logic" and "domain knowledge"
- No pattern for adding new knowledge sources

### 1.5 Short-Term Fixes (Pre-DSPy)

These improvements can be made WITHOUT DSPy migration:

```python
# 1. Add decision history tracking to TreeData
tree_data.decision_history = [
    {"iteration": 1, "tool": "fast_vector_search", "query": "...", "found": 5},
    {"iteration": 2, "tool": "colqwen_search", "query": "...", "found": 3},
]

# 2. Detect duplicate (tool, query) pairs before executing
if (tool_name, query) in [(d["tool"], d["query"]) for d in tree_data.decision_history]:
    # Force different tool or refined query
    
# 3. Add result confidence scoring
yield Result(objects=[...], confidence=0.85, sufficient=False)

# 4. Manual context injection (simple version of Atlas)
tree_data.knowledge_context = {
    "manuals": [
        {"name": "Technical Manual", "topics": ["installation", "wiring", "setup"]},
        {"name": "Users Manual", "topics": ["operation", "menus", "maintenance"]},
    ]
}
```

**Estimated effort:** 1-2 days for decision history + duplicate detection

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

## 10. DSPy-Powered Benchmarking & Interactive Evaluation

### 10.1 Current State

**Benchmark File:** `data/benchmarks/benchmarksv03.json` (canonical - 20 questions)

**Current Schema (Rich):**
```json
{
  "category": "Complex Problem | Direct Question",
  "query": "...",
  "answer": "...",
  "evidence": {
    "document": "techman.pdf | uk_firmware.pdf",
    "locations": [{"chapter": "3", "page": "19"}],
    "section": "3.2.7 Battery and power connections",
    "visual_element": "Table: Default jumper positions"
  }
}
```

**Current Benchmark Script:** `scripts/run_benchmark.py`
- Queries API endpoints directly (`/search`, `/agentic_search`)
- Calculates Hit@1, Hit@3, Hit@5, MRR, Manual Accuracy
- Compares Regular RAG vs ColQwen pipelines
- **Limitation:** Not integrated with DSPy, no optimization loop

**Previous Research:** `docs/BENCHMARK_SYSTEM_DESIGN.md`
- Proposed "Blind Agent & Judge" architecture ✅ REUSE
- Frontend "Benchmark Mode" concept ✅ REUSE
- Identified data leakage risk ✅ ADDRESSED
- Open questions about Judge prompt engineering → SOLVED by DSPy

### 10.2 The "Blind Agent & Judge" Architecture

**Problem:** If the Agent sees the benchmark answers, the test is invalid (data leakage).

**Solution:** Separate the Agent (system under test) from the Judge (evaluator).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Benchmark Mode)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ "Suggest        │  │ Run Search      │  │ "Evaluate" Button       │  │
│  │  Question"      │  │ (normal flow)   │  │ (after response)        │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │
└───────────┼────────────────────┼───────────────────────┼────────────────┘
            │                    │                       │
            ▼                    ▼                       ▼
┌───────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐
│  GET /benchmark/  │  │  GET /agentic_      │  │  POST /benchmark/       │
│  suggest          │  │  search?query=...   │  │  evaluate               │
│                   │  │                     │  │                         │
│  Returns:         │  │  Agent is BLIND     │  │  Input:                 │
│  - question only  │  │  (no access to      │  │  - benchmark_id         │
│  - benchmark_id   │  │   benchmark file)   │  │  - agent_answer         │
│  - category       │  │                     │  │                         │
│  (NO answer!)     │  │                     │  │  Judge compares to      │
│                   │  │                     │  │  ground truth           │
└───────────────────┘  └─────────────────────┘  └─────────────────────────┘
```

### 10.3 DSPy LLM-as-Judge Pattern

DSPy provides the exact pattern we need for semantic evaluation:

```python
import dspy

class TechnicalJudge(dspy.Signature):
    """
    Judge if the agent's answer is factually correct compared to ground truth.
    Focus on technical accuracy: specific values (voltages, resistances, pin numbers).
    Ignore stylistic differences (phrasing, formatting).
    """
    
    question: str = dspy.InputField(desc="The technical question asked")
    ground_truth: str = dspy.InputField(desc="The verified correct answer from documentation")
    agent_answer: str = dspy.InputField(desc="The answer generated by the Agent")
    
    # Structured outputs
    score: int = dspy.OutputField(desc="Score 0-100 based on factual accuracy")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the score")
    missing_facts: list[str] = dspy.OutputField(desc="Key facts from ground truth missing in agent answer")
    incorrect_facts: list[str] = dspy.OutputField(desc="Factually incorrect statements in agent answer")

# Create the Judge module
judge = dspy.ChainOfThought(TechnicalJudge)

# Usage
result = judge(
    question="How do I set jumpers for 24V with 1800mA charging?",
    ground_truth="J1C, J1D, J1F, and J1G must be ON. J1A is not used, J1B is OFF, J1E is OFF.",
    agent_answer="You need to set J1C to ON and J1D to ON for 24V operation."
)
# result.score = 40
# result.reasoning = "Agent correctly identified J1C and J1D but missed J1F and J1G"
# result.missing_facts = ["J1F must be ON", "J1G must be ON"]
# result.incorrect_facts = []
```

**Why DSPy solves the Judge prompt engineering problem:**
- Structured outputs (score, reasoning, missing_facts) - no JSON parsing needed
- Type-safe - DSPy validates the output schema
- Optimizable - can use MIPROv2 to improve Judge accuracy
- Consistent - same signature always produces same output structure

### 10.4 DSPy Evaluation Framework

DSPy provides built-in evaluation utilities that enable:
1. **Structured Metrics** - Define custom metrics as Python functions
2. **Parallel Evaluation** - Multi-threaded evaluation with progress
3. **MLflow Integration** - Experiment tracking and visualization
4. **Optimizer Loop** - Use evaluation results to improve prompts

**Key DSPy Evaluation Components:**
```python
from dspy.evaluate import Evaluate, SemanticF1

# Define custom metric for RAG
def rag_metric(example, pred, trace=None):
    """Evaluate RAG response quality."""
    # Check if correct page was retrieved
    retrieved_pages = [r.get("page") for r in pred.retrieved_docs]
    page_hit = example.expected_page in retrieved_pages
    
    # Check if answer covers key facts (semantic)
    semantic = SemanticF1(decompositional=True)
    semantic_score = semantic(example, pred)
    
    # Combined score
    return (page_hit * 0.5) + (semantic_score * 0.5)

# Create evaluator
evaluate = dspy.Evaluate(
    devset=benchmark_devset,
    metric=rag_metric,
    num_threads=16,
    display_progress=True,
    display_table=5
)

# Run evaluation
result = evaluate(rag_module)
```

### 10.5 Proposed Architecture

```
api/
├── endpoints/
│   └── benchmark.py         # NEW: /benchmark/suggest, /benchmark/evaluate
├── evaluation/
│   ├── __init__.py
│   ├── judge.py             # TechnicalJudge DSPy Signature + module
│   ├── metrics.py           # Custom metrics (PageHit, SemanticAnswer, Combined)
│   ├── devset.py            # Load benchmarksv03.json as dspy.Example list
│   └── evaluator.py         # VSMEvaluator class wrapping dspy.Evaluate

frontend/
├── components/
│   └── BenchmarkMode.tsx    # NEW: Benchmark UI component
├── lib/hooks/
│   └── useBenchmark.ts      # NEW: Hook for benchmark API calls

scripts/
├── run_benchmark.py         # Updated to use DSPy evaluation
└── run_optimization.py      # NEW: Optimize prompts using benchmark results
```

### 10.6 Benchmark API Endpoints

**`api/endpoints/benchmark.py`:**
```python
"""
Benchmark endpoints for interactive evaluation.

The Agent has NO access to this file or the benchmark data.
Only the Judge service can read the ground truth answers.
"""

import json
import random
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path

from api.evaluation.judge import TechnicalJudge

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# Load benchmark data (server-side only, never exposed to Agent)
BENCHMARK_PATH = Path("data/benchmarks/benchmarksv03.json")
_benchmark_data = None

def _load_benchmark():
    global _benchmark_data
    if _benchmark_data is None:
        with open(BENCHMARK_PATH, "r") as f:
            _benchmark_data = json.load(f)
    return _benchmark_data


class SuggestResponse(BaseModel):
    benchmark_id: int
    question: str
    category: str
    # NOTE: answer is NOT included - Agent must remain blind


class EvaluateRequest(BaseModel):
    benchmark_id: int
    agent_answer: str


class EvaluateResponse(BaseModel):
    score: int  # 0-100
    reasoning: str
    missing_facts: list[str]
    incorrect_facts: list[str]
    # For debugging (optional, could be hidden in production)
    ground_truth: str | None = None


@router.get("/suggest", response_model=SuggestResponse)
async def suggest_question(category: str | None = None):
    """
    Get a random benchmark question for testing.
    
    The answer is NOT returned - the Agent must answer blindly.
    """
    benchmark = _load_benchmark()
    
    # Filter by category if specified
    if category:
        filtered = [b for b in benchmark if b.get("category") == category]
        if not filtered:
            raise HTTPException(404, f"No questions in category: {category}")
        item = random.choice(filtered)
    else:
        item = random.choice(benchmark)
    
    # Find index for benchmark_id
    benchmark_id = benchmark.index(item)
    
    return SuggestResponse(
        benchmark_id=benchmark_id,
        question=item["query"],
        category=item.get("category", "Unknown"),
    )


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_answer(request: EvaluateRequest):
    """
    Evaluate the Agent's answer against ground truth using the Judge.
    
    This is the ONLY place where ground truth is accessed.
    """
    benchmark = _load_benchmark()
    
    if request.benchmark_id < 0 or request.benchmark_id >= len(benchmark):
        raise HTTPException(400, "Invalid benchmark_id")
    
    item = benchmark[request.benchmark_id]
    ground_truth = item["answer"]
    question = item["query"]
    
    # Run the DSPy Judge
    judge = TechnicalJudge()
    result = await judge.aforward(
        question=question,
        ground_truth=ground_truth,
        agent_answer=request.agent_answer,
    )
    
    return EvaluateResponse(
        score=result.score,
        reasoning=result.reasoning,
        missing_facts=result.missing_facts,
        incorrect_facts=result.incorrect_facts,
        ground_truth=ground_truth,  # Optional: show after evaluation
    )


@router.get("/categories")
async def list_categories():
    """List available benchmark categories."""
    benchmark = _load_benchmark()
    categories = list(set(b.get("category", "Unknown") for b in benchmark))
    return {"categories": categories, "total": len(benchmark)}
```

### 10.7 Judge Service (DSPy)

**`api/evaluation/judge.py`:**
```python
"""
DSPy-powered Judge for semantic evaluation of Agent answers.

This module is isolated from the Agent - it only runs during evaluation.
"""

import dspy
from typing import List


class TechnicalJudgeSignature(dspy.Signature):
    """
    Judge if the agent's answer is factually correct compared to ground truth.
    
    You are a technical examiner for industrial alarm system documentation.
    Focus on FACTUAL ACCURACY: specific values (voltages, resistances, jumper pins, terminal numbers).
    IGNORE stylistic differences (phrasing, formatting, order of information).
    
    Scoring guide:
    - 90-100: All key facts correct, no errors
    - 70-89: Most facts correct, minor omissions
    - 50-69: Some facts correct, significant omissions
    - 30-49: Few facts correct, major errors or omissions
    - 0-29: Mostly incorrect or irrelevant
    """
    
    question: str = dspy.InputField(desc="The technical question asked")
    ground_truth: str = dspy.InputField(desc="The verified correct answer from documentation")
    agent_answer: str = dspy.InputField(desc="The answer generated by the Agent")
    
    # Structured outputs
    score: int = dspy.OutputField(desc="Score 0-100 based on factual accuracy")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the score (1-2 sentences)")
    missing_facts: List[str] = dspy.OutputField(desc="Key facts from ground truth missing in agent answer")
    incorrect_facts: List[str] = dspy.OutputField(desc="Factually incorrect statements in agent answer")


class TechnicalJudge(dspy.Module):
    """
    Judge module that evaluates Agent answers against ground truth.
    
    Uses ChainOfThought for better reasoning on complex technical comparisons.
    """
    
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(TechnicalJudgeSignature)
    
    def forward(self, question: str, ground_truth: str, agent_answer: str):
        return self.judge(
            question=question,
            ground_truth=ground_truth,
            agent_answer=agent_answer,
        )
    
    async def aforward(self, question: str, ground_truth: str, agent_answer: str):
        """Async version for API endpoints."""
        return await self.judge.acall(
            question=question,
            ground_truth=ground_truth,
            agent_answer=agent_answer,
        )
```

### 10.8 Custom Metrics for Batch Evaluation

**`api/evaluation/metrics.py`:**
```python
import dspy
from dspy.evaluate import SemanticF1
from typing import List, Optional

class PageHitMetric:
    """Check if expected page appears in retrieved results."""
    
    def __init__(self, tolerance: int = 1, top_k: int = 5):
        self.tolerance = tolerance
        self.top_k = top_k
    
    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        expected_page = example.expected_page
        expected_doc = example.expected_document
        
        # Get retrieved pages from prediction
        retrieved = getattr(pred, 'retrieved_docs', [])[:self.top_k]
        
        for rank, doc in enumerate(retrieved, 1):
            page = doc.get("page_number") or doc.get("page")
            doc_name = doc.get("document") or doc.get("manual_name")
            
            if page and abs(page - expected_page) <= self.tolerance:
                if doc_name == expected_doc:
                    return 1.0 / rank  # MRR-style score
        
        return 0.0


class VisualElementMetric:
    """Check if visual element (table, figure) was retrieved."""
    
    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
        expected_visual = example.visual_element.lower() if example.visual_element else None
        if not expected_visual:
            return True  # No visual element expected
        
        # Check if any retrieved chunk mentions the visual element
        for doc in getattr(pred, 'retrieved_docs', []):
            content = doc.get("content", "").lower()
            if expected_visual in content:
                return True
        
        return False


class VSMCombinedMetric:
    """Combined metric for VSM RAG evaluation."""
    
    def __init__(self, page_weight: float = 0.4, semantic_weight: float = 0.4, visual_weight: float = 0.2):
        self.page_metric = PageHitMetric()
        self.semantic_metric = SemanticF1(decompositional=True)
        self.visual_metric = VisualElementMetric()
        self.page_weight = page_weight
        self.semantic_weight = semantic_weight
        self.visual_weight = visual_weight
    
    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        page_score = self.page_metric(example, pred, trace)
        semantic_score = self.semantic_metric(example, pred, trace)
        visual_score = float(self.visual_metric(example, pred, trace))
        
        combined = (
            self.page_weight * page_score +
            self.semantic_weight * semantic_score +
            self.visual_weight * visual_score
        )
        
        # For optimization: require all components to pass
        if trace is not None:
            return page_score >= 0.5 and semantic_score >= 0.5 and visual_score
        
        return combined
```

### 10.9 Frontend Benchmark Mode

**`frontend/components/BenchmarkMode.tsx`:**
```tsx
'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';

interface BenchmarkQuestion {
  benchmark_id: number;
  question: string;
  category: string;
}

interface EvaluationResult {
  score: number;
  reasoning: string;
  missing_facts: string[];
  incorrect_facts: string[];
  ground_truth?: string;
}

interface BenchmarkModeProps {
  onQuestionSelect: (question: string) => void;
  agentAnswer: string | null;
  isAgentDone: boolean;
}

export function BenchmarkMode({ onQuestionSelect, agentAnswer, isAgentDone }: BenchmarkModeProps) {
  const [currentQuestion, setCurrentQuestion] = useState<BenchmarkQuestion | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const suggestQuestion = async (category?: string) => {
    setIsLoading(true);
    setEvaluation(null);
    try {
      const url = category 
        ? `/api/benchmark/suggest?category=${encodeURIComponent(category)}`
        : '/api/benchmark/suggest';
      const res = await fetch(url);
      const data: BenchmarkQuestion = await res.json();
      setCurrentQuestion(data);
      onQuestionSelect(data.question);
    } finally {
      setIsLoading(false);
    }
  };

  const evaluateAnswer = async () => {
    if (!currentQuestion || !agentAnswer) return;
    
    setIsEvaluating(true);
    try {
      const res = await fetch('/api/benchmark/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          benchmark_id: currentQuestion.benchmark_id,
          agent_answer: agentAnswer,
        }),
      });
      const data: EvaluationResult = await res.json();
      setEvaluation(data);
    } finally {
      setIsEvaluating(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    if (score >= 40) return 'text-orange-600 bg-orange-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <Card className="p-4 mb-6 border-2 border-dashed border-blue-300 bg-blue-50/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-lg">🎯</span>
          <h3 className="font-semibold">Benchmark Mode</h3>
          <Badge variant="outline">Testing</Badge>
        </div>
        <div className="flex gap-2">
          <Button 
            size="sm" 
            variant="outline"
            onClick={() => suggestQuestion()}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Suggest Question'}
          </Button>
          <Button 
            size="sm" 
            variant="outline"
            onClick={() => suggestQuestion('Complex Problem')}
            disabled={isLoading}
          >
            Complex
          </Button>
          <Button 
            size="sm" 
            variant="outline"
            onClick={() => suggestQuestion('Direct Question')}
            disabled={isLoading}
          >
            Direct
          </Button>
        </div>
      </div>

      {currentQuestion && (
        <div className="mb-4 p-3 bg-white rounded border">
          <div className="flex items-center gap-2 mb-2">
            <Badge>{currentQuestion.category}</Badge>
            <span className="text-xs text-muted-foreground">
              ID: {currentQuestion.benchmark_id}
            </span>
          </div>
          <p className="text-sm">{currentQuestion.question}</p>
        </div>
      )}

      {isAgentDone && agentAnswer && !evaluation && (
        <Button 
          onClick={evaluateAnswer}
          disabled={isEvaluating}
          className="w-full"
        >
          {isEvaluating ? 'Evaluating...' : '📊 Evaluate Agent Answer'}
        </Button>
      )}

      {evaluation && (
        <div className="mt-4 p-4 bg-white rounded border">
          <div className="flex items-center justify-between mb-3">
            <span className="font-medium">Evaluation Result</span>
            <span className={`text-2xl font-bold px-3 py-1 rounded ${getScoreColor(evaluation.score)}`}>
              {evaluation.score}/100
            </span>
          </div>
          
          <p className="text-sm text-muted-foreground mb-3">{evaluation.reasoning}</p>
          
          {evaluation.missing_facts.length > 0 && (
            <div className="mb-2">
              <span className="text-xs font-medium text-orange-600">Missing Facts:</span>
              <ul className="text-xs text-muted-foreground ml-4 list-disc">
                {evaluation.missing_facts.map((fact, i) => (
                  <li key={i}>{fact}</li>
                ))}
              </ul>
            </div>
          )}
          
          {evaluation.incorrect_facts.length > 0 && (
            <div className="mb-2">
              <span className="text-xs font-medium text-red-600">Incorrect Facts:</span>
              <ul className="text-xs text-muted-foreground ml-4 list-disc">
                {evaluation.incorrect_facts.map((fact, i) => (
                  <li key={i}>{fact}</li>
                ))}
              </ul>
            </div>
          )}
          
          {evaluation.ground_truth && (
            <details className="mt-3">
              <summary className="text-xs cursor-pointer text-blue-600">Show Ground Truth</summary>
              <p className="text-xs mt-2 p-2 bg-gray-50 rounded">{evaluation.ground_truth}</p>
            </details>
          )}
        </div>
      )}
    </Card>
  );
}
```

### 10.10 Devset Loader (for batch evaluation)

**`api/evaluation/devset.py`:**
```python
import json
import dspy
from pathlib import Path
from typing import List

DOC_NAME_MAP = {
    "techman.pdf": "Technical Manual",
    "uk_firmware.pdf": "UK Firmware Manual",
}

def load_benchmark(path: str = "data/benchmarks/benchmarksv03.json") -> List[dspy.Example]:
    """Load benchmark as DSPy Examples."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        evidence = item.get("evidence", {})
        locations = evidence.get("locations", [{}])
        
        example = dspy.Example(
            # Inputs
            question=item["query"],
            category=item.get("category", "Unknown"),
            
            # Expected outputs (for metrics)
            expected_answer=item["answer"],
            expected_document=DOC_NAME_MAP.get(evidence.get("document"), evidence.get("document")),
            expected_page=int(locations[0].get("page", 0)) if locations else 0,
            expected_chapter=locations[0].get("chapter") if locations else None,
            expected_section=evidence.get("section"),
            visual_element=evidence.get("visual_element"),
        ).with_inputs("question", "category")
        
        examples.append(example)
    
    return examples

def split_devset(examples: List[dspy.Example], train_ratio: float = 0.7):
    """Split examples into train and dev sets."""
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]
```

### 10.14 VSM Evaluator (Batch)

**`api/evaluation/evaluator.py`:**
```python
import dspy
from dspy.evaluate import Evaluate
from typing import Optional, Dict, Any
import mlflow

from .metrics import VSMCombinedMetric, PageHitMetric
from .devset import load_benchmark

class VSMEvaluator:
    """Wrapper for DSPy evaluation with MLflow tracking."""
    
    def __init__(
        self,
        benchmark_path: str = "data/benchmarks/benchmarksv03.json",
        metric: Optional[Any] = None,
        num_threads: int = 8,
        mlflow_experiment: str = "vsm-rag-evaluation"
    ):
        self.devset = load_benchmark(benchmark_path)
        self.metric = metric or VSMCombinedMetric()
        self.num_threads = num_threads
        self.mlflow_experiment = mlflow_experiment
        
        self.evaluator = Evaluate(
            devset=self.devset,
            metric=self.metric,
            num_threads=num_threads,
            display_progress=True,
            display_table=5
        )
    
    def evaluate(self, program: dspy.Module, run_name: str = "evaluation") -> Dict[str, Any]:
        """Run evaluation with MLflow tracking."""
        mlflow.set_experiment(self.mlflow_experiment)
        
        with mlflow.start_run(run_name=run_name):
            result = self.evaluator(program)
            
            # Log metrics
            mlflow.log_metric("combined_score", result)
            
            # Log per-category breakdown
            category_scores = self._compute_category_scores(program)
            for category, score in category_scores.items():
                mlflow.log_metric(f"score_{category.lower().replace(' ', '_')}", score)
            
            return {
                "score": result,
                "category_scores": category_scores,
                "devset_size": len(self.devset),
            }
    
    def _compute_category_scores(self, program: dspy.Module) -> Dict[str, float]:
        """Compute scores broken down by question category."""
        categories = {}
        for example in self.devset:
            cat = example.category
            if cat not in categories:
                categories[cat] = {"scores": [], "count": 0}
            
            pred = program(**example.inputs())
            score = self.metric(example, pred)
            categories[cat]["scores"].append(score)
            categories[cat]["count"] += 1
        
        return {
            cat: sum(data["scores"]) / data["count"]
            for cat, data in categories.items()
        }
```

### 10.11 Implementation Phases (Benchmarking)

**Phase B1: Judge Service (LOW RISK)**
- [ ] Create `api/evaluation/__init__.py`
- [ ] Create `api/evaluation/judge.py` (TechnicalJudge DSPy Signature)
- [ ] Test Judge with sample inputs
- [ ] Verify structured output (score, reasoning, facts)

**Phase B2: Benchmark API (LOW RISK)**
- [ ] Create `api/endpoints/benchmark.py`
- [ ] Implement `/benchmark/suggest` (returns question only, NO answer)
- [ ] Implement `/benchmark/evaluate` (calls Judge)
- [ ] Implement `/benchmark/categories`
- [ ] Register router in `main.py`

**Phase B3: Frontend Benchmark Mode (MEDIUM RISK)**
- [ ] Create `frontend/components/BenchmarkMode.tsx`
- [ ] Add "Benchmark Mode" toggle to header (next to "Agentic Mode")
- [ ] Wire up "Suggest Question" → search bar
- [ ] Wire up "Evaluate" → display score card
- [ ] Style evaluation results (score color, facts lists)

**Phase B4: Batch Evaluation (MEDIUM RISK)**
- [ ] Create `api/evaluation/devset.py` (load as dspy.Example)
- [ ] Create `api/evaluation/metrics.py` (PageHit, Visual, Combined)
- [ ] Create `api/evaluation/evaluator.py` (VSMEvaluator)
- [ ] Update `scripts/run_benchmark.py` to use DSPy evaluation

**Phase B5: Optimization Loop (AFTER DSPy Migration)**
- [ ] Create `scripts/run_optimization.py`
- [ ] Run MIPROv2 on benchmark
- [ ] Compare optimized vs baseline
- [ ] Save optimized prompts

### 10.12 What We're Reusing vs. Not Using

**From `docs/BENCHMARK_SYSTEM_DESIGN.md`:**

| Concept | Status | Notes |
|---------|--------|-------|
| "Blind Agent & Judge" architecture | ✅ REUSE | Core pattern - Agent never sees answers |
| Frontend "Benchmark Mode" | ✅ REUSE | Toggle, Suggest, Evaluate buttons |
| "Suggest Question" flow | ✅ REUSE | Returns question only, strips answer |
| "Evaluate" button flow | ✅ REUSE | Sends agent answer to Judge |
| Score (0-100) display | ✅ REUSE | With color coding |
| Judge prompt engineering | ❌ REPLACED | DSPy Signature handles this automatically |
| Manual JSON parsing | ❌ REPLACED | DSPy structured outputs |
| "Is gpt-oss overkill?" question | ✅ RESOLVED | Use same model, DSPy can optimize |
| Visual evidence evaluation | ⏳ DEFERRED | Start with text-only, add later |
| Async vs blocking | ✅ RESOLVED | Use async (DSPy supports it) |

**New from DSPy:**

| Feature | Benefit |
|---------|---------|
| `TechnicalJudgeSignature` | Type-safe, structured output (score, reasoning, facts) |
| `dspy.ChainOfThought` | Better reasoning for complex comparisons |
| `SemanticF1` metric | Built-in semantic similarity for batch eval |
| `dspy.Evaluate` | Parallel batch evaluation with progress |
| `MIPROv2` optimizer | Can improve Judge prompts automatically |

### 10.13 Benchmark Success Criteria

| Metric | Current | Target (Post-DSPy) |
|--------|---------|-------------------|
| Hit@1 | TBD | ≥70% |
| Hit@3 | TBD | ≥85% |
| SemanticF1 | TBD | ≥0.7 |
| Visual Element Hit | TBD | ≥80% |
| Complex Problem Accuracy | TBD | ≥60% |
| Direct Question Accuracy | TBD | ≥80% |
| Judge Consistency | N/A | ≥90% (same score on repeated eval) |

---

## 11. Rollback Plan

If DSPy migration causes issues:
1. Keep old `DecisionPromptBuilder` until new approach is validated
2. Feature flag to switch between old/new decision logic
3. Knowledge module (Atlas) can be used independently of DSPy
4. Environment + tasks_completed enhancements work without DSPy
5. Benchmark evaluation can use DSPy metrics without migrating agent

