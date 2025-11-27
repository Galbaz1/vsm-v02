"""
Cloud Text Embeddings Provider - Jina v4 API.

Uses Jina's Universal Embeddings API for text embeddings.
Images/visual embeddings are handled by JinaVisualSearch.

Ref: https://api.jina.ai/v1/embeddings
Ref: https://jina.ai/embeddings/
"""

import logging
from typing import List

import httpx

from api.core.config import get_settings
from api.core.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Jina API endpoint
JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"


class JinaEmbeddings(EmbeddingProvider):
    """
    Jina-based TEXT embedding provider for cloud deployment.
    
    Uses Jina v4 API for text embeddings:
    - POST /v1/embeddings
    - model: jina-embeddings-v4
    - task: retrieval.query or retrieval.passage
    - dimensions: 1024 (Matryoshka - can be 128-2048)
    """
    
    def __init__(self):
        settings = get_settings()
        
        if not settings.jina_api_key:
            raise ValueError(
                "JINA_API_KEY not set. "
                "Set VSM_MODE=local or configure JINA_API_KEY."
            )
        
        self._api_key = settings.jina_api_key
        self._model = "jina-embeddings-v4"
        self._dimensions_value = 1024  # Default Matryoshka dimension
        
        # Create async HTTP client
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
        )
        
        logger.info(f"Initialized JinaEmbeddings: model={self._model}, dimensions={self._dimensions_value}")
    
    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality (1024 for Jina v4)."""
        return self._dimensions_value
    
    async def embed_texts(
        self,
        texts: List[str],
        task: str = "retrieval.passage",
    ) -> List[List[float]]:
        """
        Embed multiple text strings using Jina API.
        
        Args:
            texts: List of text strings to embed
            task: Task type for asymmetric retrieval
                  - "retrieval.passage": For document chunks
                  - "retrieval.query": For search queries
                  - "text-matching": For symmetric matching
                  
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        payload = {
            "model": self._model,
            "input": texts,
            "task": task,
            "dimensions": self._dimensions_value,
        }
        
        response = await self._client.post(
            JINA_EMBEDDINGS_URL,
            json=payload,
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract embeddings in order
        embeddings = [None] * len(texts)
        for item in data.get("data", []):
            idx = item.get("index", 0)
            embedding = item.get("embedding", [])
            if idx < len(embeddings):
                embeddings[idx] = embedding
        
        # Verify all texts have embeddings
        if any(e is None for e in embeddings):
            missing_count = sum(1 for e in embeddings if e is None)
            raise ValueError(f"Jina API returned incomplete embeddings: missing {missing_count}/{len(texts)}")
            
        return embeddings
    
    async def embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Embed a single query string using Jina API.
        
        Uses task=retrieval.query for asymmetric retrieval.
        """
        embeddings = await self.embed_texts([query], task="retrieval.query")
        if embeddings:
            return embeddings[0]
        return []
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
