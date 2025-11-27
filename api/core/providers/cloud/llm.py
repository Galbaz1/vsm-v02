"""
Cloud LLM Provider - Gemini 2.5 Flash.

Uses the google-genai SDK (python-genai) for generation.

Ref: https://ai.google.dev/gemini-api/docs/thinking
Ref: https://github.com/googleapis/python-genai
"""

import logging
import time
from typing import AsyncGenerator, List, Dict, Optional

from api.core.config import get_settings
from api.core.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GeminiLLM(LLMProvider):
    """
    Gemini-based LLM provider for cloud deployment.
    
    Uses Gemini 2.5 Flash with optional thinking budget.
    
    Thinking Budget:
    - 0: Disabled (no thinking output)
    - -1: Dynamic (Gemini decides)
    - 1-24576: Fixed token budget
    """
    
    def __init__(self):
        settings = get_settings()
        
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Set VSM_MODE=local or configure GEMINI_API_KEY."
            )
        
        # Import here to avoid loading SDK when not needed
        from google import genai
        from google.genai import types
        
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        self._thinking_budget = settings.gemini_thinking_budget
        
        logger.info(f"Initialized GeminiLLM: model={self._model}, thinking_budget={self._thinking_budget}")
    
    def _get_generation_config(
        self,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Build GenerateContentConfig with optional thinking config."""
        types = self._types
        
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Add thinking config based on budget
        if self._thinking_budget == 0:
            # Disable thinking
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0
            )
        elif self._thinking_budget > 0:
            # Fixed budget
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self._thinking_budget
            )
        # If -1 (dynamic), don't set thinking_config - let Gemini decide
        
        return types.GenerateContentConfig(**config_kwargs)
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate text from a prompt using Gemini."""
        start_time = time.time()
        
        config = self._get_generation_config(temperature, max_tokens)
        
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Extract thinking output if available (accumulate all thinking parts)
        thinking_parts = []
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        thinking_parts.append(part.thought)
        thinking = "\n".join(thinking_parts)
        
        # Get token usage
        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = getattr(response.usage_metadata, "total_token_count", 0)
        
        return LLMResponse(
            text=response.text or "",
            model=self._model,
            tokens_used=tokens_used,
            time_ms=elapsed_ms,
            thinking=thinking,
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion from message history using Gemini."""
        start_time = time.time()
        
        # Convert messages to Gemini format
        # Gemini uses 'user' and 'model' roles
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map OpenAI-style roles to Gemini roles
            if role == "assistant":
                role = "model"
            elif role == "system":
                # Prepend system message to first user message
                # or treat as user message
                role = "user"
            
            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })
        
        config = self._get_generation_config(temperature, max_tokens)
        
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Extract thinking (accumulate all thinking parts)
        thinking_parts = []
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        thinking_parts.append(part.thought)
        thinking = "\n".join(thinking_parts)
        
        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = getattr(response.usage_metadata, "total_token_count", 0)
        
        return LLMResponse(
            text=response.text or "",
            model=self._model,
            tokens_used=tokens_used,
            time_ms=elapsed_ms,
            thinking=thinking,
        )
    
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion token-by-token using Gemini."""
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
            
            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })
        
        config = self._get_generation_config(
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        
        # Use async streaming API
        response_stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )
        
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text
