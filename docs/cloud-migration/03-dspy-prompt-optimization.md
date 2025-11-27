# DSPy Prompt Optimization Architecture

**Last Updated:** 2025-11-26  
**Module:** `api/prompts/`

---

## Overview

Different LLM models respond differently to the same prompts. DSPy provides a framework for:
1. **Defining** model-agnostic contracts (Signatures)
2. **Optimizing** prompts per-model (Compilers like MIPROv2)
3. **Saving/Loading** optimized prompts (JSON state files)

This separation allows VSM to maintain **one codebase** with **model-specific prompt tuning**.

---

## The Three-Layer Separation

```mermaid
flowchart TB
    subgraph Layer1["Layer 1: Signatures (Shared Contract)"]
        direction LR
        DecSig["DecisionSignature"]
        SearchSig["SearchQuerySignature"]
        RespSig["ResponseSignature"]
        JudgeSig["TechnicalJudgeSignature"]
    end

    subgraph Layer2["Layer 2: Compiled Modules (Per-Model)"]
        direction LR
        subgraph LocalPrompts["prompts/local/"]
            DecLocal["decision.json<br/>Optimized for gpt-oss:120b"]
            SearchLocal["search.json"]
            RespLocal["response.json"]
        end
        subgraph CloudPrompts["prompts/cloud/"]
            DecCloud["decision.json<br/>Optimized for gemini-2.5-flash"]
            SearchCloud["search.json"]
            RespCloud["response.json"]
        end
    end

    subgraph Layer3["Layer 3: Runtime Module Loader"]
        Loader["get_compiled_module(name)"]
        VSM_MODE{{"VSM_MODE"}}
    end

    DecSig --> DecLocal
    DecSig --> DecCloud
    SearchSig --> SearchLocal
    SearchSig --> SearchCloud
    RespSig --> RespLocal
    RespSig --> RespCloud

    VSM_MODE --> Loader
    Loader --> DecLocal
    Loader --> DecCloud

    style Layer1 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style LocalPrompts fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style CloudPrompts fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Layer3 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

---

## DSPy Compilation Flow

```mermaid
flowchart TD
    subgraph OfflineOptimization["Offline: Prompt Optimization (scripts/optimize_prompts.py)"]
        direction TB
        BaseSig["Base Signature<br/>DecisionSignature"]
        TrainSet["Training Examples<br/>benchmarks/decision_examples.json"]
        Metric["Evaluation Metric<br/>decision_accuracy()"]
        
        BaseSig --> Optimizer
        TrainSet --> Optimizer
        Metric --> Optimizer
        
        Optimizer["MIPROv2 Optimizer"]
        
        Optimizer --> BootstrapPhase
        
        subgraph BootstrapPhase["Phase 1: Bootstrap Few-Shot"]
            Bootstrap["Generate candidate demos<br/>from training data"]
        end
        
        BootstrapPhase --> InstructPhase
        
        subgraph InstructPhase["Phase 2: Instruction Proposals"]
            Instruct["Generate instruction variants<br/>grounded in task dynamics"]
        end
        
        InstructPhase --> BayesPhase
        
        subgraph BayesPhase["Phase 3: Bayesian Optimization"]
            Bayes["Search for optimal<br/>instruction + demo combination"]
        end
        
        BayesPhase --> Compiled["Compiled Module"]
        Compiled --> SaveJSON["Save to prompts/{mode}/decision.json"]
    end

    subgraph RuntimeLoading["Runtime: Module Loading"]
        LoadJSON["Load prompts/{mode}/decision.json"]
        RecreateModule["Recreate Module Structure"]
        LoadState["Load Optimized State"]
        ReadyModule["Ready-to-Use Module"]
        
        LoadJSON --> RecreateModule
        RecreateModule --> LoadState
        LoadState --> ReadyModule
    end

    SaveJSON -.-> LoadJSON

    style OfflineOptimization fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style RuntimeLoading fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## VSMChainOfThought Module

Inspired by Elysia's `ElysiaChainOfThought`, this module automatically injects context into any DSPy Signature.

```mermaid
flowchart LR
    subgraph Inputs["Automatic Inputs (from TreeData)"]
        UserPrompt["user_prompt"]
        ConvHistory["conversation_history"]
        Atlas["atlas (domain knowledge)"]
        PrevErrors["previous_errors"]
        EnvData["environment (optional)"]
        TasksComp["tasks_completed (optional)"]
    end

    subgraph BaseSig["Base Signature Inputs"]
        CustomInputs["Custom Inputs<br/>(e.g., available_tools)"]
    end

    subgraph VSMCoT["VSMChainOfThought"]
        Extend["Extend Signature"]
        Predict["dspy.Predict"]
    end

    subgraph Outputs["Extended Outputs"]
        Reasoning["reasoning<br/>(chain of thought)"]
        MsgUpdate["message_update<br/>(user feedback)"]
        Impossible["impossible<br/>(task feasibility)"]
        OrigOutputs["Original Outputs"]
    end

    UserPrompt --> Extend
    ConvHistory --> Extend
    Atlas --> Extend
    PrevErrors --> Extend
    EnvData --> Extend
    TasksComp --> Extend
    CustomInputs --> Extend
    
    Extend --> Predict
    Predict --> Reasoning
    Predict --> MsgUpdate
    Predict --> Impossible
    Predict --> OrigOutputs

    style Inputs fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style VSMCoT fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Outputs fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## Signature Definitions

### DecisionSignature

```python
# api/prompts/signatures/decision.py
import dspy

class DecisionSignature(dspy.Signature):
    """
    Decide which tool to use next based on query and current state.
    
    Guidelines:
    - For tables, bit codes, menus: prefer hybrid_search
    - For diagrams, schematics, figures: prefer colqwen_search
    - For simple definitions: prefer fast_vector_search
    - When data is gathered: use text_response to answer
    """
    
    # Inputs
    query: str = dspy.InputField(
        desc="The user's question to answer"
    )
    available_tools: list[dict] = dspy.InputField(
        desc="List of available tools with name and description"
    )
    environment_summary: str = dspy.InputField(
        desc="Summary of data already retrieved"
    )
    iteration: str = dspy.InputField(
        desc="Current iteration status (e.g., '2/10')"
    )
    
    # Outputs
    tool_name: str = dspy.OutputField(
        desc="Name of the tool to use"
    )
    tool_inputs: dict = dspy.OutputField(
        desc="Input parameters for the chosen tool"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the decision"
    )
    should_end: bool = dspy.OutputField(
        desc="True if this should be the final action"
    )
```

### SearchQuerySignature

```python
# api/prompts/signatures/search.py
import dspy

class SearchQuerySignature(dspy.Signature):
    """
    Expand or refine a user query for better retrieval.
    """
    
    original_query: str = dspy.InputField(
        desc="The user's original query"
    )
    search_type: str = dspy.InputField(
        desc="Type of search: 'vector', 'visual', or 'hybrid'"
    )
    
    expanded_query: str = dspy.OutputField(
        desc="Expanded query with relevant technical terms"
    )
    keywords: list[str] = dspy.OutputField(
        desc="Key terms for BM25 matching"
    )
```

### ResponseSignature

```python
# api/prompts/signatures/response.py
import dspy

class ResponseSignature(dspy.Signature):
    """
    Generate a helpful response from retrieved information.
    """
    
    query: str = dspy.InputField(
        desc="The user's question"
    )
    context: str = dspy.InputField(
        desc="Retrieved information with page references"
    )
    
    answer: str = dspy.OutputField(
        desc="Direct, helpful answer to the question"
    )
    sources: list[dict] = dspy.OutputField(
        desc="List of source references (page, manual)"
    )
    confidence: str = dspy.OutputField(
        desc="Confidence level: 'high', 'medium', 'low'"
    )
```

---

## Optimization Script

```mermaid
flowchart TD
    Start([Start Optimization])
    
    Start --> LoadConfig["Load Settings"]
    LoadConfig --> CheckMode{{"VSM_MODE?"}}
    
    CheckMode -->|local| ConfigLocal["Configure DSPy<br/>ollama_chat/gpt-oss:120b"]
    CheckMode -->|cloud| ConfigCloud["Configure DSPy<br/>gemini/gemini-2.5-flash"]
    
    ConfigLocal --> LoadData["Load Training Data<br/>benchmarks/decision_examples.json"]
    ConfigCloud --> LoadData
    
    LoadData --> CreateModule["Create Base Module<br/>dspy.ChainOfThought(DecisionSignature)"]
    
    CreateModule --> InitOptimizer["Initialize MIPROv2<br/>metric=decision_accuracy<br/>auto='medium'"]
    
    InitOptimizer --> Compile["optimizer.compile(module, trainset)"]
    
    Compile --> Evaluate["Evaluate on validation set"]
    
    Evaluate --> SaveModule["Save to prompts/{mode}/decision.json"]
    
    SaveModule --> End([Complete])

    style Start fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style End fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style CheckMode fill:#fff9c4,stroke:#f57f17,stroke-width:2px
```

### Script Implementation

```python
# scripts/optimize_prompts.py
import dspy
from dspy.teleprompt import MIPROv2
from api.core.config import get_settings
from api.prompts.signatures import DecisionSignature
import json

def decision_accuracy(example, prediction, trace=None):
    """Metric for decision quality."""
    # Check if predicted tool matches expected
    tool_match = prediction.tool_name == example.expected_tool
    # Check if inputs are reasonable
    inputs_valid = all(k in prediction.tool_inputs for k in example.required_inputs)
    return tool_match and inputs_valid

def main():
    settings = get_settings()
    mode = settings.vsm_mode
    
    # Configure DSPy LM based on mode
    # Ref: https://stackoverflow.com/questions/79809980/turn-off-geminis-reasoning-in-dspy
    # Ref: https://ai.google.dev/gemini-api/docs/thinking
    if mode == "local":
        lm = dspy.LM(
            f'ollama_chat/{settings.ollama_model}',
            api_base=settings.ollama_base_url,
        )
    else:
        # Gemini 2.5 Flash thinking control:
        # - reasoning_effort="disable" turns off thinking
        # - Default (no param) = dynamic thinking
        # - thinkingBudget: 0=off, -1=dynamic, 1-24576=fixed tokens
        if settings.gemini_thinking_budget == 0:
            lm = dspy.LM(
                f'gemini/{settings.gemini_model}',
                api_key=settings.gemini_api_key,
                reasoning_effort="disable",  # DSPy-specific param
            )
        else:
            # Dynamic or custom budget (DSPy handles via LiteLLM)
            lm = dspy.LM(
                f'gemini/{settings.gemini_model}',
                api_key=settings.gemini_api_key,
            )
    
    dspy.configure(lm=lm)
    
    # Load training data
    with open("data/benchmarks/decision_examples.json") as f:
        examples = [dspy.Example(**e).with_inputs("query", "available_tools", "environment_summary", "iteration") 
                    for e in json.load(f)]
    
    # Split train/val
    trainset = examples[:80]
    valset = examples[80:]
    
    # Create base module
    module = dspy.ChainOfThought(DecisionSignature)
    
    # Initialize optimizer
    optimizer = MIPROv2(
        metric=decision_accuracy,
        auto="medium",  # Balance between speed and quality
    )
    
    # Compile (optimize)
    optimized = optimizer.compile(
        module,
        trainset=trainset,
    )
    
    # Save optimized state
    optimized.save(f"api/prompts/{mode}/decision.json", save_program=False)
    
    print(f"✅ Saved optimized decision module to api/prompts/{mode}/decision.json")

if __name__ == "__main__":
    main()
```

---

## Runtime Loading

```python
# api/prompts/__init__.py
import dspy
from pathlib import Path
from api.core.config import get_settings

_compiled_modules: dict = {}

def get_compiled_module(name: str) -> dspy.Module:
    """
    Load a compiled DSPy module for the current mode.
    
    Args:
        name: Module name (e.g., "decision", "search", "response")
    
    Returns:
        Compiled DSPy module with optimized prompts
    """
    global _compiled_modules
    
    settings = get_settings()
    mode = settings.vsm_mode
    cache_key = f"{mode}_{name}"
    
    if cache_key not in _compiled_modules:
        # Get signature class
        from api.prompts.signatures import SIGNATURE_MAP
        signature_cls = SIGNATURE_MAP[name]
        
        # Recreate module structure
        module = dspy.ChainOfThought(signature_cls)
        
        # Load optimized state
        state_path = Path(__file__).parent / mode / f"{name}.json"
        if state_path.exists():
            module.load(str(state_path))
        
        _compiled_modules[cache_key] = module
    
    return _compiled_modules[cache_key]
```

---

## File Structure

```
api/prompts/
├── __init__.py              # get_compiled_module() factory
├── signatures/
│   ├── __init__.py          # SIGNATURE_MAP export
│   ├── decision.py          # DecisionSignature
│   ├── search.py            # SearchQuerySignature
│   ├── response.py          # ResponseSignature
│   └── judge.py             # TechnicalJudgeSignature
├── local/                   # Compiled for gpt-oss:120b
│   ├── decision.json
│   ├── search.json
│   └── response.json
└── cloud/                   # Compiled for gemini-2.5-flash
    ├── decision.json
    ├── search.json
    └── response.json
```

---

## Key Insights

### Why Per-Model Optimization Matters

| Aspect | gpt-oss:120b (Local) | Gemini 2.5 Flash (Cloud) |
|--------|---------------------|--------------------------|
| **Prompt Style** | Concise, direct | Can handle longer context |
| **JSON Reliability** | Needs strict formatting | More flexible parsing |
| **Few-Shot Learning** | Benefits significantly | Already strong zero-shot |
| **Thinking** | N/A | Extended thinking available |

### MIPROv2 Optimization Produces

1. **Optimized Instructions**: Model-specific system prompt variations
2. **Curated Few-Shot Examples**: Demonstrations that improve task performance
3. **Validated Combination**: Bayesian-optimized pairing of instructions + demos

### Saving/Loading Preserves

- Signature structure
- Optimized instructions
- Few-shot demonstrations
- Per-predictor configurations

---

## Integration with Agent

```mermaid
sequenceDiagram
    participant Agent as AgentOrchestrator
    participant Loader as get_compiled_module()
    participant Module as DecisionModule
    participant LLM as LLM Provider

    Agent->>Loader: get_compiled_module("decision")
    Loader->>Loader: Check cache
    Loader->>Loader: Load prompts/{mode}/decision.json
    Loader-->>Agent: Compiled Module

    Agent->>Module: module(query=..., available_tools=..., ...)
    Module->>Module: Apply optimized instructions
    Module->>Module: Include few-shot examples
    Module->>LLM: Generate completion
    LLM-->>Module: Response
    Module->>Module: Parse structured output
    Module-->>Agent: Decision(tool_name, inputs, reasoning, should_end)
```

