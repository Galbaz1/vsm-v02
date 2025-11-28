"""
Cloud LLM Provider - GPT-5.1 (primary) + Gemini 2.5 Flash (fallback).

Primary: OpenAI GPT-5.1 via Responses API (more reliable, high reasoning)
Fallback: Gemini 2.5 Flash (has intermittent empty response issues)

Ref: https://ai.google.dev/gemini-api/docs/thinking
Ref: https://github.com/googleapis/python-genai
"""

import logging
import time
from typing import AsyncGenerator, List, Dict, Optional, Any

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
        settings = get_settings()
        
        # Try GPT-5.1 first (more reliable)
        if settings.openai_api_key:
            emitted = False
            async for chunk in self._gpt51_responses_stream(messages, kwargs):
                if chunk:
                    emitted = True
                    yield chunk
            if emitted:
                return
            logger.warning("GPT-5.1 Responses API yielded no text, falling back to Gemini...")
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
        
        def _yield_text_parts(chunk) -> List[str]:
            texts: List[str] = []
            # google-genai streaming chunks often surface text inside candidates/parts, not chunk.text
            candidate = getattr(chunk, "candidates", [None])[0]
            if candidate and getattr(candidate, "content", None):
                for part in candidate.content.parts or []:
                    text_val = getattr(part, "text", None)
                    if text_val:
                        texts.append(text_val)
            # Fallback to chunk.text if present
            if getattr(chunk, "text", None):
                texts.append(chunk.text)
            return texts
        
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
                texts = _yield_text_parts(chunk)
                for t in texts:
                    if t:
                        text_yielded = True
                        yield t
            
            # If we got chunks but none had text, or no chunks at all, raise
            if not text_yielded:
                error_msg = f"Gemini returned {chunk_count} chunks but no text content"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                    
        except Exception as e:
            logger.error(f"Gemini fallback error: {e}")
            raise
    
    def _messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Responses API input format.
        
        - user/system -> role:user with input_text
        - assistant -> role:assistant with output_text
        """
        inputs: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if not text:
                continue
            
            if role == "assistant":
                inputs.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })
            else:
                # treat system messages as user context
                inputs.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                })
        
        # Fallback to a minimal user message if somehow empty
        if not inputs:
            inputs.append({
                "role": "user",
                "content": [{"type": "input_text", "text": ""}],
            })
        return inputs

    async def _gpt51_responses_stream(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict,
    ) -> AsyncGenerator[Optional[str], None]:
        """
        Primary LLM: GPT-5.1 using OpenAI Responses API (streaming).
        
        Uses recommended Responses API input format and surfaces only text deltas to avoid
        Pydantic serialization noise.
        
        Note: GPT-5.1 supports temperature/top_p ONLY when reasoning.effort="none".
        For reasoning tasks, use reasoning.effort="low"|"medium"|"high" without temperature.
        
        Ref: https://platform.openai.com/docs/guides/latest-model
        """
        settings = get_settings()
        
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            inputs = self._messages_to_responses_input(messages)
            
            # Determine reasoning effort - for agentic RAG we want some reasoning
            # but "high" can be slow. Use "low" for balance of speed + quality.
            reasoning_effort = kwargs.get("reasoning_effort", "low")
            
            # Build request params
            request_params = {
                "model": settings.openai_model,
                "input": inputs,
                "max_output_tokens": kwargs.get("max_tokens", 2048),
                "stream": True,
            }
            
            # Only add reasoning if effort is not "none"
            if reasoning_effort != "none":
                request_params["reasoning"] = {"effort": reasoning_effort}
            else:
                # When reasoning is "none", we can use temperature
                request_params["temperature"] = kwargs.get("temperature", 0.7)
            
            stream = await client.responses.create(**request_params)
            
            async for event in stream:
                if event.type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta
                elif event.type == "response.error":
                    err_msg = getattr(event, "message", None) or str(event)
                    logger.error(f"GPT-5.1 stream error: {err_msg}")
                    break
                elif event.type == "response.completed":
                    break
            
        except Exception as e:
            logger.error(f"GPT-5.1 Responses API error: {e}")
            yield None

    async def _gpt51_chat_stream(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict,
    ) -> AsyncGenerator[Optional[str], None]:
        """
        Secondary GPT-5.1 path using chat completions streaming (when Responses API rejects params).
        """
        settings = get_settings()
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            stream = await client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_completion_tokens=kwargs.get("max_tokens", 2048),
                stream=True,
            )
            
            async for chunk in stream:
                choice = getattr(chunk, "choices", [None])[0]
                if choice and getattr(choice, "delta", None):
                    delta = choice.delta
                    text = getattr(delta, "content", None)
                    if text:
                        yield text
            
        except Exception as e:
            logger.error(f"GPT-5.1 chat completion error: {e}")
            yield None
