"""Local Text Embeddings Provider - Uses Ollama with bge-m3."""

import logging
import httpx
from typing import List

from api.core.providers.base import EmbeddingProvider
from api.core.config import get_settings

logger = logging.getLogger(__name__)


class OllamaEmbeddings(EmbeddingProvider):
    """
    Ollama-based embedding provider for local deployment.
    
    Uses bge-m3 model (8K context, retrieval-optimized).
    Text embeddings only - visual search uses ColQwenVisualSearch.
    """
    
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_embed_model
        self._dimensions = 1024  # bge-m3 produces 1024-dim vectors
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client
    
    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality (1024 for bge-m3)."""
        return self._dimensions
    
    async def embed_texts(
        self,
        texts: List[str],
        task: str = "retrieval.passage",
    ) -> List[List[float]]:
        """
        Embed multiple text strings using Ollama.
        
        Args:
            texts: List of text strings to embed
            task: Task type (ignored for Ollama, kept for API consistency)
            
        Returns:
            List of dense embedding vectors
        """
        client = await self._get_client()
        embeddings = []
        
        for text in texts:
            try:
                response = await client.post(
                    "/api/embeddings",
                    json={"model": self.model, "prompt": text}
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])
            except httpx.HTTPError as e:
                logger.error(f"Ollama embeddings error: {e}")
                raise
        
        return embeddings
    
    async def embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            query: Query text
            
        Returns:
            Dense embedding vector
        """
        client = await self._get_client()
        
        try:
            response = await client.post(
                "/api/embeddings",
                json={"model": self.model, "prompt": query}
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except httpx.HTTPError as e:
            logger.error(f"Ollama embeddings error: {e}")
            raise
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
