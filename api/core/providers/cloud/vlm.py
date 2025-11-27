"""
Cloud VLM Provider - Gemini 2.5 Flash Vision.

Uses the google-genai SDK for multimodal generation.
Gemini 2.5 Flash supports both text and image inputs natively.

Ref: https://ai.google.dev/gemini-api/docs/vision
"""

import base64
import logging
import time
from pathlib import Path
from typing import Optional

from api.core.config import get_settings
from api.core.providers.base import VLMProvider

logger = logging.getLogger(__name__)


class GeminiVLM(VLMProvider):
    """
    Gemini-based VLM provider for cloud deployment.
    
    Uses Gemini 2.5 Flash with multimodal capabilities.
    Images are sent as base64-encoded data.
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
        
        logger.info(f"Initialized GeminiVLM: model={self._model}")
    
    async def interpret_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """
        Interpret an image using Gemini Vision.
        
        Args:
            image_path: Path to image file (or base64 string)
            prompt: Text prompt for interpretation
            max_tokens: Maximum output tokens
            
        Returns:
            Text interpretation of the image
        """
        types = self._types
        
        # Read image and convert to base64
        path = Path(image_path)
        if path.exists():
            with open(path, "rb") as f:
                image_bytes = f.read()
            
            # Determine MIME type
            suffix = path.suffix.lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/png")
        else:
            # Assume it's already base64
            logger.warning(f"Image path not found, assuming base64: {image_path[:50]}...")
            image_bytes = base64.b64decode(image_path)
            mime_type = "image/png"
        
        # Create image part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )
        
        # Generate with image + text
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more precise interpretation
        )
        
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt, image_part],
            config=config,
        )
        
        return response.text or ""
    
    async def is_available(self) -> bool:
        """Check if Gemini service is available."""
        try:
            # Quick test with minimal content
            response = self._client.models.generate_content(
                model=self._model,
                contents="Say 'ok'",
                config=self._types.GenerateContentConfig(max_output_tokens=5),
            )
            return response.text is not None
        except Exception as e:
            logger.warning(f"Gemini availability check failed: {e}")
            return False
