"""
DSPy Prompts Module for VSM.

Provides:
- Signatures: Model-agnostic contracts (Decision, Search, Response)
- Modules: VSMChainOfThought with auto-context injection
- Loader: get_compiled_module() for loading optimized prompts

File Structure:
    api/prompts/
    ├── __init__.py              # This file - exports + loader
    ├── chain_of_thought.py      # VSMChainOfThought module
    ├── signatures/
    │   ├── __init__.py          # SIGNATURE_MAP export
    │   ├── decision.py          # DecisionSignature
    │   ├── search.py            # SearchQuerySignature
    │   └── response.py          # ResponseSignature
    ├── local/                   # Compiled for gpt-oss:120b
    │   └── *.json               # Optimized module states
    └── cloud/                   # Compiled for gemini-2.5-flash
        └── *.json               # Optimized module states
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

# Ensure DSPy cache is writable in sandboxed/test environments before importing dspy
_cache_dir = Path(__file__).resolve().parent.parent.parent / ".dspy-cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DSPY_CACHE_DIR", str(_cache_dir))
os.environ.setdefault("DSPY_CACHE_PATH", str(_cache_dir / "cache.sqlite"))
os.environ.setdefault("DSPY_DISABLE_CACHE", "1")

import dspy

from api.core.config import get_settings
from api.prompts.signatures import (
    DecisionSignature,
    SearchQuerySignature,
    ResponseSignature,
    SIGNATURE_MAP,
)
from api.prompts.chain_of_thought import VSMChainOfThought

logger = logging.getLogger(__name__)

# Cache for compiled modules
_compiled_modules: Dict[str, dspy.Module] = {}


def get_compiled_module(
    name: str,
    use_chain_of_thought: bool = True,
) -> dspy.Module:
    """
    Load a compiled DSPy module for the current mode.
    
    Args:
        name: Module name ("decision", "search", "response")
        use_chain_of_thought: If True, wrap in ChainOfThought (default)
    
    Returns:
        Compiled DSPy module with optimized prompts (if available)
    
    Example:
        decision_module = get_compiled_module("decision")
        result = decision_module(
            query="How do I reset the controller?",
            available_tools=tools_json,
            environment_summary="",
            iteration="1/10",
        )
    """
    settings = get_settings()
    mode = settings.vsm_mode
    cache_key = f"{mode}_{name}_{use_chain_of_thought}"
    
    if cache_key not in _compiled_modules:
        # Get signature class
        if name not in SIGNATURE_MAP:
            raise ValueError(f"Unknown signature: {name}. Valid: {list(SIGNATURE_MAP.keys())}")
        
        signature_cls = SIGNATURE_MAP[name]
        
        # Create module
        if use_chain_of_thought:
            module = dspy.ChainOfThought(signature_cls)
        else:
            module = dspy.Predict(signature_cls)
        
        # Try to load optimized state
        state_path = Path(__file__).parent / mode / f"{name}.json"
        if state_path.exists():
            try:
                module.load(str(state_path))
                logger.info(f"Loaded optimized {name} module from {state_path}")
            except Exception as e:
                logger.warning(f"Failed to load optimized state: {e}")
        else:
            logger.debug(f"No optimized state at {state_path}, using base module")
        
        _compiled_modules[cache_key] = module
    
    return _compiled_modules[cache_key]


def get_vsm_module(name: str) -> VSMChainOfThought:
    """
    Get a VSMChainOfThought module with auto-context injection.
    
    Args:
        name: Module name ("decision", "search", "response")
    
    Returns:
        VSMChainOfThought module that auto-injects TreeData context
    
    Example:
        decision = get_vsm_module("decision")
        result = decision(
            tree_data=tree_data,
            atlas=atlas,
            available_tools=tools_json,
            iteration="1/10",
        )
    """
    # Ensure DSPy is configured before creating modules
    from api.core.dspy_config import get_dspy_lm
    get_dspy_lm()
    
    if name not in SIGNATURE_MAP:
        raise ValueError(f"Unknown signature: {name}. Valid: {list(SIGNATURE_MAP.keys())}")
    
    return VSMChainOfThought(SIGNATURE_MAP[name])


def reset_compiled_modules() -> None:
    """Reset the compiled modules cache (for testing or mode switching)."""
    global _compiled_modules
    _compiled_modules = {}
    logger.info("Compiled modules cache reset")


__all__ = [
    # Signatures
    "DecisionSignature",
    "SearchQuerySignature", 
    "ResponseSignature",
    "SIGNATURE_MAP",
    # Modules
    "VSMChainOfThought",
    # Loaders
    "get_compiled_module",
    "get_vsm_module",
    "reset_compiled_modules",
]
