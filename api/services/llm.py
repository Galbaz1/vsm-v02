"""
LLM Service - Ollama and MLX VLM clients.

Provides async interfaces for:
- OllamaClient: Text generation with gpt-oss:120b
- MLXVLMClient: Visual interpretation with Qwen3-VL-8B

Follows Elysia patterns for agentic decision-making.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from api.services.environment import TreeData
    from api.services.tools.base import Tool

logger = logging.getLogger(__name__)

# Default configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MLX_VLM_BASE_URL = os.getenv("MLX_VLM_BASE_URL", "http://localhost:8000")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    model: str
    tokens_used: int = 0
    time_ms: float = 0


class OllamaClient:
    """
    Async client for Ollama API.
    
    Supports both streaming and non-streaming generation,
    with special formatting for agentic decision-making.
    """
    
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> AsyncGenerator[str, None] | LLMResponse:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to instance model)
            stream: Whether to stream the response
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            If stream=True: AsyncGenerator yielding text chunks
            If stream=False: LLMResponse with complete text
        """
        model = model or self.model
        client = await self._get_client()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs.get("options", {}),
            },
        }
        
        if stream:
            return self._stream_generate(client, payload)
        else:
            return await self._single_generate(client, payload)
    
    async def _single_generate(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
    ) -> LLMResponse:
        """Non-streaming generation."""
        import time
        start = time.time()
        
        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                text=data.get("response", ""),
                model=data.get("model", payload["model"]),
                tokens_used=data.get("eval_count", 0),
                time_ms=(time.time() - start) * 1000,
            )
        except httpx.HTTPError as e:
            logger.error(f"Ollama generate error: {e}")
            raise
    
    async def _stream_generate(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Streaming generation."""
        try:
            async with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as e:
            logger.error(f"Ollama stream error: {e}")
            raise
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> AsyncGenerator[str, None] | LLMResponse:
        """
        Chat-style generation with message history.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            model: Model to use
            stream: Whether to stream
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            LLMResponse or AsyncGenerator[str]
        """
        model = model or self.model
        client = await self._get_client()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs.get("options", {}),
            },
        }
        
        if stream:
            return self._stream_chat(client, payload)
        else:
            return await self._single_chat(client, payload)
    
    async def _single_chat(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
    ) -> LLMResponse:
        """Non-streaming chat."""
        import time
        start = time.time()
        
        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                text=data.get("message", {}).get("content", ""),
                model=data.get("model", payload["model"]),
                tokens_used=data.get("eval_count", 0),
                time_ms=(time.time() - start) * 1000,
            )
        except httpx.HTTPError as e:
            logger.error(f"Ollama chat error: {e}")
            raise
    
    async def _stream_chat(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Streaming chat."""
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as e:
            logger.error(f"Ollama chat stream error: {e}")
            raise


class DecisionPromptBuilder:
    """
    Builds prompts for the decision agent.
    
    Formats tool descriptions, environment state, and conversation
    history for the LLM to make tool selection decisions.
    """
    
    SYSTEM_PROMPT = """You are an intelligent agent that helps users find information in technical manuals.
You have access to the following tools to retrieve and process information:

{tools_description}

Based on the user's query and the current state, decide which tool to use next.
You must respond with a valid JSON object containing:
- "tool_name": The name of the tool to use
- "inputs": A dictionary of inputs for the tool
- "reasoning": A brief explanation of why you chose this tool
- "should_end": true if this is the final response, false otherwise

Guidelines:
1. For queries about tables, bit codes, status displays, menus, or specifications: Use hybrid_search (finds both text AND visual layout)
2. For explicit visual requests (diagrams, schematics, "show me", figures): Use colqwen_search
3. For simple definitions or short procedures: Use fast_vector_search
4. For complex or technical queries: Use hybrid_search (it runs in parallel, so it's efficient)
5. When you have enough information: Use text_response to answer
6. If the environment has lots of data: Consider using summarize first

PREFER hybrid_search for most technical queries - it finds both text AND visual content efficiently (runs in parallel).

Current iteration: {iteration_status}
"""
    
    USER_TEMPLATE = """Query: {query}

{environment_context}

{error_context}

What tool should I use next? Respond with JSON only."""
    
    @classmethod
    def format_tools_description(cls, tools: List["Tool"]) -> str:
        """Format available tools for the prompt."""
        descriptions = []
        for tool in tools:
            inputs_str = ""
            if tool.inputs:
                inputs_list = []
                for name, spec in tool.inputs.items():
                    req = "(required)" if spec.get("required") else f"(default: {spec.get('default')})"
                    inputs_list.append(f"    - {name}: {spec.get('description', '')} {req}")
                inputs_str = "\n" + "\n".join(inputs_list)
            
            descriptions.append(
                f"- **{tool.name}**: {tool.description}{inputs_str}"
            )
        return "\n\n".join(descriptions)
    
    @classmethod
    def build_decision_prompt(
        cls,
        tree_data: "TreeData",
        available_tools: List["Tool"],
    ) -> List[Dict[str, str]]:
        """
        Build the full prompt for decision-making.
        
        Returns messages in chat format for the LLM.
        """
        # Build system prompt with tools
        tools_desc = cls.format_tools_description(available_tools)
        system_content = cls.SYSTEM_PROMPT.format(
            tools_description=tools_desc,
            iteration_status=tree_data.iteration_status(),
        )
        
        # Build user prompt with context
        env_context = tree_data.environment.to_llm_context(max_tokens=4000)
        
        error_context = ""
        errors = tree_data.get_errors()  # Returns flattened list from Dict
        if errors:
            error_context = "Previous errors (try to recover):\n" + "\n".join(
                f"- {err}" for err in errors[-3:]  # Last 3 errors
            )
        
        user_content = cls.USER_TEMPLATE.format(
            query=tree_data.user_prompt,
            environment_context=env_context,
            error_context=error_context,
        )
        
        messages = [
            {"role": "system", "content": system_content},
        ]
        
        # Add conversation history (limited)
        for msg in tree_data.conversation_history[-6:]:  # Last 6 messages
            messages.append(msg)
        
        # Add current query
        messages.append({"role": "user", "content": user_content})
        
        return messages


@dataclass
class DecisionResult:
    """Parsed decision from LLM."""
    tool_name: str
    inputs: Dict[str, Any]
    reasoning: str
    should_end: bool = False
    
    @classmethod
    def from_json(cls, text: str) -> "DecisionResult":
        """Parse decision from JSON string."""
        # Try to extract JSON from the response
        text = text.strip()
        
        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        
        # Handle leading/trailing non-JSON
        if not text.startswith("{"):
            start = text.find("{")
            if start != -1:
                text = text[start:]
        if not text.endswith("}"):
            end = text.rfind("}") + 1
            if end > 0:
                text = text[:end]
        
        try:
            data = json.loads(text)
            return cls(
                tool_name=data.get("tool_name", "text_response"),
                inputs=data.get("inputs", {}),
                reasoning=data.get("reasoning", ""),
                should_end=data.get("should_end", False),
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse decision JSON: {e}\nText: {text[:200]}")
            # Return a fallback decision
            return cls(
                tool_name="text_response",
                inputs={},
                reasoning="Failed to parse LLM response, defaulting to text response",
                should_end=True,
            )


class MLXVLMClient:
    """
    Client for MLX VLM server (Qwen3-VL-8B).
    
    Used for visual interpretation of page images.
    """
    
    def __init__(
        self,
        base_url: str = MLX_VLM_BASE_URL,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def interpret_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """
        Interpret an image using the VLM.
        
        Args:
            image_path: Path to the image file
            prompt: Question/instruction about the image
            max_tokens: Maximum response tokens
            
        Returns:
            VLM's interpretation of the image
        """
        import base64
        from pathlib import Path
        
        client = await self._get_client()
        
        # Read and encode image
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine image type
        suffix = image_file.suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        
        # Build request for OpenAI-compatible endpoint
        payload = {
            "model": "Qwen3-VL-8B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
        }
        
        try:
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return data["choices"][0]["message"]["content"]
            
        except httpx.HTTPError as e:
            logger.error(f"MLX VLM error: {e}")
            raise
    
    async def is_available(self) -> bool:
        """Check if the VLM server is available."""
        try:
            client = await self._get_client()
            response = await client.get("/v1/models")
            return response.status_code == 200
        except Exception:
            return False


# Singleton instances
_ollama_client: Optional[OllamaClient] = None
_mlx_vlm_client: Optional[MLXVLMClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


def get_mlx_vlm_client() -> MLXVLMClient:
    """Get or create MLX VLM client singleton."""
    global _mlx_vlm_client
    if _mlx_vlm_client is None:
        _mlx_vlm_client = MLXVLMClient()
    return _mlx_vlm_client


async def reset_clients() -> None:
    """Reset all client singletons (for testing)."""
    global _ollama_client, _mlx_vlm_client
    if _ollama_client:
        await _ollama_client.close()
        _ollama_client = None
    if _mlx_vlm_client:
        await _mlx_vlm_client.close()
        _mlx_vlm_client = None

