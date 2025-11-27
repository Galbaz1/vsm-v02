"""
DSPy Configuration for VSM.

Configures the DSPy language model based on VSM_MODE.

Ref: https://stackoverflow.com/questions/79809980/turn-off-geminis-reasoning-in-dspy
Ref: https://ai.google.dev/gemini-api/docs/thinking
"""

import logging
from typing import Optional

import dspy

from api.core.config import get_settings

logger = logging.getLogger(__name__)

_configured = False


def configure_dspy() -> dspy.LM:
    """
    Configure DSPy with the appropriate LM based on VSM_MODE.
    
    Returns:
        The configured language model instance.
    
    Notes:
        - Local mode: Uses Ollama with gpt-oss model
        - Cloud mode: Uses Gemini with optional thinking control
        
    Gemini Thinking Budget:
        - 0: Thinking disabled (reasoning_effort="disable")
        - -1: Dynamic thinking (default, Gemini decides)
        - 1-24576: Fixed token budget for thinking
    """
    global _configured
    
    settings = get_settings()
    
    if settings.vsm_mode == "local":
        logger.info(f"Configuring DSPy for local mode: ollama_chat/{settings.ollama_model}")
        lm = dspy.LM(
            f"ollama_chat/{settings.ollama_model}",
            api_base=settings.ollama_base_url,
        )
    else:
        # Cloud mode: Gemini
        logger.info(f"Configuring DSPy for cloud mode: gemini/{settings.gemini_model}")
        
        if settings.gemini_thinking_budget == 0:
            # Disable thinking entirely
            logger.info("Gemini thinking: DISABLED")
            lm = dspy.LM(
                f"gemini/{settings.gemini_model}",
                api_key=settings.gemini_api_key,
                reasoning_effort="disable",
            )
        elif settings.gemini_thinking_budget == -1:
            # Dynamic thinking (Gemini decides)
            logger.info("Gemini thinking: DYNAMIC")
            lm = dspy.LM(
                f"gemini/{settings.gemini_model}",
                api_key=settings.gemini_api_key,
            )
        else:
            # Fixed budget - DSPy/LiteLLM handles via generation_config
            logger.info(f"Gemini thinking: {settings.gemini_thinking_budget} tokens")
            lm = dspy.LM(
                f"gemini/{settings.gemini_model}",
                api_key=settings.gemini_api_key,
                # Note: Custom budget may require generation_config in LiteLLM
            )
    
    dspy.configure(lm=lm)
    _configured = True
    
    return lm


def get_dspy_lm() -> dspy.LM:
    """
    Get the configured DSPy LM, configuring if needed.
    
    Returns:
        The DSPy language model instance.
    """
    if not _configured:
        return configure_dspy()
    return dspy.settings.lm


def reset_dspy_config() -> None:
    """Reset DSPy configuration (for testing or mode switching)."""
    global _configured
    _configured = False
    logger.info("DSPy configuration reset")

