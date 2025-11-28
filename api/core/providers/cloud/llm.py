"""
Cloud LLM Provider - GPT-5.1 (primary) + Gemini 2.5 Flash (fallback).

Primary: OpenAI GPT-5.1 via Responses API (more reliable, high reasoning)
Fallback: Gemini 2.5 Flash (has intermittent empty response issues)

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
        """
        Stream chat completion using GPT-5.1 (primary) with Gemini fallback.
        
        GPT-5.1 is more reliable than Gemini which has intermittent empty response issues.
        Falls back to Gemini if OpenAI key not set or GPT-5.1 fails.
        """
        # Build prompt for GPT-5.1
        full_prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            full_prompt += f"{role}: {content}\n"
        
        settings = get_settings()
        
        # Try GPT-5.1 first (more reliable)
        if settings.openai_api_key:
            gpt_result = await self._gpt51_primary(full_prompt, kwargs)
            if gpt_result:
                yield gpt_result
                return
            logger.warning("GPT-5.1 failed, falling back to Gemini...")
        else:
            logger.info("OPENAI_API_KEY not set, using Gemini directly")
        
        # Fall back to Gemini
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
        
        try:
            response_stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )
            
            chunk_count = 0
            text_yielded = False
            async for chunk in response_stream:
                chunk_count += 1
                if chunk.text:
                    text_yielded = True
                    yield chunk.text
            
            # If we got chunks but none had text, or no chunks at all, raise
            if not text_yielded:
                error_msg = f"Gemini returned {chunk_count} chunks but no text content"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                    
        except Exception as e:
            logger.error(f"Gemini fallback error: {e}")
            raise
    
    async def _gpt51_primary(
        self,
        prompt: str,
        kwargs: Dict,
    ) -> Optional[str]:
        """
        Primary LLM: GPT-5.1 using OpenAI Responses API.
        
        GPT-5.1 is a thinking model with high reasoning capabilities.
        More reliable than Gemini (no empty response issues).
        """
        settings = get_settings()
        
        try:
            from openai import OpenAI
            import asyncio
            
            client = OpenAI(api_key=settings.openai_api_key)
            
            logger.info(f"GPT-5.1 primary: Using {settings.openai_model}")
            
            # GPT-5.1 uses the Responses API with reasoning
            # Run in thread pool since OpenAI SDK is sync
            def call_gpt51():
                try:
                    # Try Responses API (GPT-5.1)
                    response = client.responses.create(
                        model=settings.openai_model,
                        input=prompt,
                        reasoning={"effort": "high"},
                    )
                    return response.output_text
                except AttributeError:
                    # Fall back to Chat Completions if Responses API unavailable
                    logger.warning("GPT-5.1 Responses API unavailable, using chat completions")
                    response = client.chat.completions.create(
                        model="gpt-4o",  # Fallback model
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=kwargs.get("max_tokens", 2048),
                        temperature=kwargs.get("temperature", 0.7),
                    )
                    return response.choices[0].message.content
            
            result = await asyncio.to_thread(call_gpt51)
            logger.info(f"GPT-5.1 primary successful: {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"GPT-5.1 primary error: {e}")
            return None
