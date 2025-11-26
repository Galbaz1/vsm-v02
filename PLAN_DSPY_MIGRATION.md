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

### 1.3 Not Reusable

When deployed for a different company/context:
- Would need to modify Python code
- No clear separation between "agent logic" and "domain knowledge"
- No pattern for adding new knowledge sources

---

## 2. How Elysia Solves This

Elysia uses **DSPy Signatures** for structured prompts and an **Atlas** for agent identity/knowledge context.

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
```

**This is where knowledge context goes:**
- `agent_description` contains what the agent knows about
- Can include manual descriptions, available data sources, search strategies

### 2.3 ElysiaChainOfThought (Dynamic Signature Extension)

Elysia has a custom DSPy Module that **automatically injects** common fields into any signature:

**Reference:** `docs/elysia-source-code-for-reference-only/elysia/util/elysia_chain_of_thought.py`

```python
class ElysiaChainOfThought(Module):
    def __init__(self, signature, tree_data, ...):
        # Dynamically extend signature with common fields
        extended = signature.prepend(name="user_prompt", field=..., type_=str)
        extended = extended.append(name="atlas", field=..., type_=Atlas)
        extended = extended.append(name="environment", field=..., type_=dict)
        
        self.predict = dspy.Predict(extended)
```

**Pattern:**
1. Take any signature
2. Automatically add: `user_prompt`, `conversation_history`, `atlas`, `errors`
3. Optionally add: `environment`, `collection_schemas`, `tasks_completed`
4. Optionally add outputs: `reasoning`, `message_update`, `impossible`

---

## 3. Proposed Solution for VSM

### 3.1 Create Knowledge Module

```
api/knowledge/
├── __init__.py          # Atlas class + loader
└── thorguard.py         # ThorGuard manual descriptions
```

**`api/knowledge/thorguard.py`:**
```python
AGENT_DESCRIPTION = """
You are a technical assistant for the ThorGuard Intruder Alarm System.

## Available Documentation

### 1. Technical Manual (techman.pdf)
- Audience: Installers, Electricians
- Focus: Hardware, wiring, mounting, jumpers
- Use for: "How do I wire...", "What voltage..."

### 2. Users Manual (uk_firmware.pdf)
- Audience: End users, Security guards
- Focus: Software, menus, keypad operation
- Use for: "How do I set...", "What does LED mean..."

### Search Strategy
- Installation questions → Technical Manual
- Operation questions → Users Manual
- Component questions → Search BOTH
"""
```

### 3.2 Create DSPy Signatures

```
api/prompts/
├── __init__.py
├── signatures.py         # DecisionSignature, TextResponseSignature
└── chain_of_thought.py   # VSMChainOfThought (like ElysiaChainOfThought)
```

**`api/prompts/signatures.py`:**
```python
class DecisionSignature(dspy.Signature):
    """Route queries to appropriate tools based on user question and available documentation."""
    
    available_tools: list[dict] = dspy.InputField(...)
    previous_errors: list[str] = dspy.InputField(...)
    
    tool_name: str = dspy.OutputField(...)
    tool_inputs: dict = dspy.OutputField(...)
    should_end: bool = dspy.OutputField(...)
```

### 3.3 Update TreeData with Atlas

**Modify `api/services/environment.py`:**
```python
from api.knowledge import Atlas
from api.knowledge.thorguard import AGENT_DESCRIPTION

@dataclass
class TreeData:
    atlas: Atlas = field(default_factory=lambda: Atlas(
        agent_description=AGENT_DESCRIPTION,
        style="Professional, cite page numbers",
        end_goal="User has the information they need"
    ))
```

### 3.4 Update Agent to Use DSPy

**Modify `api/services/agent.py`:**
```python
from api.prompts.signatures import DecisionSignature
from api.prompts.chain_of_thought import VSMChainOfThought

async def _make_llm_decision(self, tree_data, available_tools):
    module = VSMChainOfThought(
        DecisionSignature,
        tree_data=tree_data,
        environment=not tree_data.environment.is_empty(),
    )
    result = await module.aforward(
        available_tools=[...],
        lm=self.lm,
    )
    return Decision(
        tool_name=result.tool_name,
        inputs=result.tool_inputs,
        should_end=result.should_end,
    )
```

---

## 4. Elysia Reference Files

### Core DSPy Patterns (MUST READ)

| File | Lines | What to Learn |
|------|-------|---------------|
| `elysia/tree/prompt_templates.py` | 1-145 | DecisionPrompt signature structure |
| `elysia/util/elysia_chain_of_thought.py` | 1-420 | Dynamic signature extension pattern |
| `elysia/tree/objects.py` | 354-375 | Atlas class definition |
| `elysia/tree/objects.py` | 545-800 | TreeData class with all state |

### Tool Implementation Patterns

| File | What to Learn |
|------|---------------|
| `elysia/tools/text/prompt_templates.py` | TextResponsePrompt, SummarizingPrompt |
| `elysia/tools/text/text.py` | How TextResponse tool uses DSPy |
| `elysia/tools/retrieval/prompt_templates.py` | Query-related signatures |
| `elysia/objects.py` | Tool, Result, Error base classes |

### API Layer (Reference)

| File | What to Learn |
|------|---------------|
| `elysia/api/services/tree.py` | How they expose Tree as a service |
| `elysia/api/routes/query.py` | Query endpoint structure |
| `elysia/config.py` | Settings pattern with LM configuration |

---

## 5. Implementation Phases

### Phase 1: Add DSPy + Knowledge Module (LOW RISK)
- [ ] Add `dspy-ai` to requirements.txt
- [ ] Create `api/knowledge/__init__.py` with Atlas class
- [ ] Create `api/knowledge/thorguard.py` with manual descriptions
- [ ] Update TreeData to include Atlas

### Phase 2: Create DSPy Signatures (MEDIUM RISK)
- [ ] Create `api/prompts/signatures.py`
- [ ] Create `api/prompts/chain_of_thought.py` (VSMChainOfThought)
- [ ] Add DSPy LM configuration to `api/services/llm.py`

### Phase 3: Migrate Agent (HIGHER RISK)
- [ ] Update `_make_llm_decision` to use DSPy
- [ ] Update TextResponseTool to use DSPy signature
- [ ] Remove old `DecisionPromptBuilder`

### Phase 4: Testing
- [ ] Run benchmark to verify no regression
- [ ] Test with ThorGuard-specific questions
- [ ] Verify knowledge context is being used

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

1. **Knowledge-aware routing:** Agent should prefer Technical Manual for wiring questions, Users Manual for menu questions
2. **No hardcoded prompts:** All prompts defined as DSPy Signatures
3. **Reusable:** Adding a new domain = creating a new knowledge file
4. **No regression:** Benchmark scores should stay same or improve
5. **Type safety:** LLM responses parsed into typed objects automatically

---

## 8. Files to Modify

| File | Change |
|------|--------|
| `requirements.txt` | Add `dspy-ai` |
| `api/services/environment.py` | Add Atlas to TreeData |
| `api/services/llm.py` | Add DSPy LM config, can remove DecisionPromptBuilder later |
| `api/services/agent.py` | Use DSPy for decisions |
| **NEW** `api/knowledge/__init__.py` | Atlas class |
| **NEW** `api/knowledge/thorguard.py` | Manual descriptions |
| **NEW** `api/prompts/signatures.py` | DSPy signatures |
| **NEW** `api/prompts/chain_of_thought.py` | VSMChainOfThought module |

---

## 9. Rollback Plan

If DSPy migration causes issues:
1. Keep old `DecisionPromptBuilder` until new approach is validated
2. Feature flag to switch between old/new decision logic
3. Knowledge module (Atlas) can be used independently of DSPy

