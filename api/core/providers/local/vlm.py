"""Local VLM Provider - Wraps MLX VLM for visual interpretation."""

import logging

from api.core.providers.base import VLMProvider
from api.core.config import get_settings
from api.services.llm import MLXVLMClient

logger = logging.getLogger(__name__)


class MLXVLM(VLMProvider):
    """
    MLX-based VLM provider for local deployment.
    
    Wraps the existing MLXVLMClient to provide the standard VLMProvider interface.
    Uses Qwen3-VL-8B running on MLX.
    """
    
    def __init__(self):
        settings = get_settings()
        self._client = MLXVLMClient(
            base_url=settings.mlx_vlm_base_url,
        )
    
    async def interpret_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """
        Interpret an image using MLX VLM.
        
        Args:
            image_path: Path to image file (local filesystem)
            prompt: Text instruction/question about the image
            max_tokens: Maximum tokens in response
            
        Returns:
            Text interpretation of the image
        """
        return await self._client.interpret_image(
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
        )
    
    async def is_available(self) -> bool:
        """
        Check if the MLX VLM server is currently available.
        
        Returns:
            True if service is reachable and functional
        """
        return await self._client.is_available()
    
    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

