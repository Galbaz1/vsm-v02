"""Local LLM Provider - Wraps Ollama for text generation."""

import logging
from typing import AsyncGenerator, List, Dict

from api.core.providers.base import LLMProvider, LLMResponse
from api.core.config import get_settings
from api.services.llm import OllamaClient

logger = logging.getLogger(__name__)


class OllamaLLM(LLMProvider):
    """
    Ollama-based LLM provider for local deployment.
    
    Wraps the existing OllamaClient to provide the standard LLMProvider interface.
    Uses gpt-oss:120b by default.
    """
    
    def __init__(self):
        settings = get_settings()
        self._client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a prompt using Ollama.
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Ollama options
            
        Returns:
            LLMResponse with generated text
        """
        # OllamaClient.generate returns LLMResponse (not AsyncGenerator) when stream=False
        response = await self._client.generate(
            prompt=prompt,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # The LLMResponse from OllamaClient is compatible (but has no thinking field)
        # We use the provider's LLMResponse to ensure consistency
        return LLMResponse(
            text=response.text,
            model=response.model,
            tokens_used=response.tokens_used,
            time_ms=response.time_ms,
            thinking="",  # Ollama doesn't have thinking output
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        Generate chat completion from message history using Ollama.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Ollama options
            
        Returns:
            LLMResponse with generated text
        """
        response = await self._client.chat(
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            text=response.text,
            model=response.model,
            tokens_used=response.tokens_used,
            time_ms=response.time_ms,
            thinking="",
        )
    
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion token-by-token using Ollama.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            **kwargs: Additional Ollama options
            
        Yields:
            Text chunks as they are generated
        """
        # OllamaClient.chat returns AsyncGenerator directly when stream=True (no await)
        generator = self._client.chat(
            messages=messages,
            stream=True,
            **kwargs
        )
        
        async for chunk in generator:
            yield chunk
    
    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

