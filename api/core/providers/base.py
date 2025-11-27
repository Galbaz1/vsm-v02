"""
Provider Layer Base Interfaces.

Defines abstract base classes for LLM, VLM, Embedding, VectorDB, and VisualSearch providers.
These interfaces enable mode-switchable deployment (local vs. cloud).

Key Design Decisions:
- EmbeddingProvider handles TEXT embeddings only (bge-m3 local, Jina v4 cloud)
- VisualSearchProvider handles the full visual RAG pipeline:
  - Local: ColQwen does query embedding + MaxSim scoring in one pass
  - Cloud: Jina CLIP via native Weaviate integration (images stored as blobs)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Dict, Any, Optional


# === Response Models ===

@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    model: str
    tokens_used: int = 0
    time_ms: float = 0
    thinking: str = ""  # Gemini 2.5 Flash thinking output (when enabled)


@dataclass
class VisualSearchResult:
    """Single result from visual search."""
    page_id: int
    asset_manual: str
    page_number: int
    score: float
    image_path: str = ""      # Local mode: filesystem path
    image_base64: str = ""    # Cloud mode: base64-encoded image from Weaviate blob


# === Provider Interfaces ===

class LLMProvider(ABC):
    """Abstract LLM provider interface for text generation."""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate a single completion from a prompt."""
        ...
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion from message history."""
        ...
    
    @abstractmethod
    async def stream_chat(
        self, 
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion token-by-token."""
        ...


class VLMProvider(ABC):
    """Abstract Vision-Language Model provider interface."""
    
    @abstractmethod
    async def interpret_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """Interpret an image with a text prompt."""
        ...
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the VLM service is currently available."""
        ...


class EmbeddingProvider(ABC):
    """
    Abstract embedding provider for TEXT embeddings only.
    
    Local: bge-m3 via Ollama
    Cloud: Jina v4 via API
    
    Note: Image/visual embeddings are handled by VisualSearchProvider.
    """
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensionality."""
        ...
    
    @abstractmethod
    async def embed_texts(
        self, 
        texts: List[str],
        task: str = "retrieval.passage",
    ) -> List[List[float]]:
        """Embed multiple text strings (dense vectors)."""
        ...
    
    @abstractmethod
    async def embed_query(
        self, 
        query: str,
    ) -> List[float]:
        """Embed a single query string (optimized for retrieval)."""
        ...


class VectorDBProvider(ABC):
    """
    Abstract vector database provider for TEXT search.
    
    Handles AssetManual collection (chunked text with dense vectors).
    Visual search (PDFDocuments) is handled by VisualSearchProvider.
    """
    
    @abstractmethod
    def connect(self) -> Any:
        """Return a database client/connection."""
        ...
    
    @abstractmethod
    async def vector_search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Perform vector similarity search."""
        ...
    
    @abstractmethod
    async def hybrid_search(
        self,
        collection: str,
        query: str,
        query_vector: List[float],
        limit: int = 5,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Perform hybrid search (vector + BM25 keyword search)."""
        ...

    @abstractmethod
    async def batch_upsert(
        self,
        collection: str,
        objects: List[Dict[str, Any]],
    ) -> None:
        """Batch upsert objects with dense embeddings (for text ingestion)."""
        ...


class VisualSearchProvider(ABC):
    """
    Abstract provider for visual RAG pipeline.
    
    Local (ColQwen):
    - Query embedding + MaxSim scoring in one pass
    - Images stored on filesystem, referenced by path
    
    Cloud (Jina CLIP + Weaviate):
    - Native Weaviate integration with multi2vec-jinaai
    - Images stored as base64 blobs in Weaviate
    - Weaviate handles embedding automatically via Jina API
    """
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[VisualSearchResult]:
        """
        Search for visually relevant pages.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of VisualSearchResult with page info and similarity scores
        """
        ...
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the visual search service is available."""
        ...
    
    @abstractmethod
    async def ingest_page(
        self,
        page_id: int,
        asset_manual: str,
        page_number: int,
        image_path: str,
    ) -> None:
        """
        Ingest a single page image into the visual search index.
        
        Local: Generates ColQwen embeddings, stores in Weaviate with image path reference
        Cloud: Reads image, converts to base64, stores blob in Weaviate (Jina embeds automatically)
        
        Args:
            page_id: Unique page identifier
            asset_manual: Manual name
            page_number: Page number within manual
            image_path: Path to page image file (PNG)
        """
        ...
    
    @abstractmethod
    async def get_page_image(
        self,
        page_id: int,
    ) -> Optional[bytes]:
        """
        Retrieve page image bytes for display.
        
        Local: Reads from filesystem using stored path
        Cloud: Returns decoded base64 blob from Weaviate
        
        Args:
            page_id: Page identifier
            
        Returns:
            Image bytes (PNG) or None if not found
        """
        ...
