"""
Provider Layer - Factory Functions.

Provides mode-switchable access to providers based on VSM_MODE env var.

Providers:
- LLM: Text generation (Ollama local, Gemini cloud)
- VLM: Visual interpretation (MLX local, Gemini cloud)
- Embeddings: TEXT embeddings only (Ollama/bge-m3 local, Jina v4 cloud)
- VectorDB: TEXT search on AssetManual (Weaviate Docker local, Weaviate Cloud)
- VisualSearch: Full visual RAG pipeline (ColQwen local, Jina Worker + Weaviate cloud)
"""

import logging
from typing import Optional

from api.core.config import get_settings
from api.core.providers.base import (
    LLMProvider,
    VLMProvider,
    EmbeddingProvider,
    VectorDBProvider,
    VisualSearchProvider,
    LLMResponse,
    VisualSearchResult,
)

logger = logging.getLogger(__name__)

# Singleton cache
_llm_provider: Optional[LLMProvider] = None
_vlm_provider: Optional[VLMProvider] = None
_embedding_provider: Optional[EmbeddingProvider] = None
_vectordb_provider: Optional[VectorDBProvider] = None
_visual_search_provider: Optional[VisualSearchProvider] = None


def get_llm() -> LLMProvider:
    """Get LLM provider (OllamaLLM local, GeminiLLM cloud)."""
    global _llm_provider
    
    if _llm_provider is None:
        settings = get_settings()
        
        if settings.vsm_mode == "local":
            from api.core.providers.local import OllamaLLM
            logger.info("Initializing OllamaLLM (local mode)")
            _llm_provider = OllamaLLM()
        else:
            from api.core.providers.cloud import GeminiLLM
            logger.info("Initializing GeminiLLM (cloud mode)")
            _llm_provider = GeminiLLM()
    
    return _llm_provider


def get_vlm() -> VLMProvider:
    """Get VLM provider (MLXVLM local, GeminiVLM cloud)."""
    global _vlm_provider
    
    if _vlm_provider is None:
        settings = get_settings()
        
        if settings.vsm_mode == "local":
            from api.core.providers.local import MLXVLM
            logger.info("Initializing MLXVLM (local mode)")
            _vlm_provider = MLXVLM()
        else:
            from api.core.providers.cloud import GeminiVLM
            logger.info("Initializing GeminiVLM (cloud mode)")
            _vlm_provider = GeminiVLM()
    
    return _vlm_provider


def get_embeddings() -> EmbeddingProvider:
    """Get TEXT embeddings provider (OllamaEmbeddings local, JinaEmbeddings cloud)."""
    global _embedding_provider
    
    if _embedding_provider is None:
        settings = get_settings()
        
        if settings.vsm_mode == "local":
            from api.core.providers.local import OllamaEmbeddings
            logger.info("Initializing OllamaEmbeddings (local mode)")
            _embedding_provider = OllamaEmbeddings()
        else:
            from api.core.providers.cloud import JinaEmbeddings
            logger.info("Initializing JinaEmbeddings (cloud mode)")
            _embedding_provider = JinaEmbeddings()
    
    return _embedding_provider


def get_vectordb() -> VectorDBProvider:
    """Get TEXT vector database provider (WeaviateLocal local, WeaviateCloud cloud)."""
    global _vectordb_provider
    
    if _vectordb_provider is None:
        settings = get_settings()
        
        if settings.vsm_mode == "local":
            from api.core.providers.local import WeaviateLocal
            logger.info("Initializing WeaviateLocal (local mode)")
            _vectordb_provider = WeaviateLocal()
        else:
            from api.core.providers.cloud import WeaviateCloud
            logger.info("Initializing WeaviateCloud (cloud mode)")
            _vectordb_provider = WeaviateCloud()
    
    return _vectordb_provider


def get_visual_search() -> VisualSearchProvider:
    """
    Get visual search provider for the full visual RAG pipeline.
    
    Local: ColQwenVisualSearch (ColQwen2.5 on GPU - embed + MaxSim in one shot)
    Cloud: JinaVisualSearch (Jina Worker embed → Weaviate Cloud multi-vector search)
    """
    global _visual_search_provider
    
    if _visual_search_provider is None:
        settings = get_settings()
        
        if settings.vsm_mode == "local":
            from api.core.providers.local import ColQwenVisualSearch
            logger.info("Initializing ColQwenVisualSearch (local mode)")
            _visual_search_provider = ColQwenVisualSearch()
        else:
            from api.core.providers.cloud import JinaVisualSearch
            logger.info("Initializing JinaVisualSearch (cloud mode)")
            _visual_search_provider = JinaVisualSearch()
    
    return _visual_search_provider


def reset_providers() -> None:
    """Reset all provider singletons (for testing or mode switching)."""
    global _llm_provider, _vlm_provider, _embedding_provider, _vectordb_provider, _visual_search_provider
    
    logger.info("Resetting all provider singletons")
    
    _llm_provider = None
    _vlm_provider = None
    _embedding_provider = None
    _vectordb_provider = None
    _visual_search_provider = None
    
    # Clear settings cache so new env vars are picked up
    get_settings.cache_clear()


__all__ = [
    # Factory functions
    "get_llm",
    "get_vlm",
    "get_embeddings",
    "get_vectordb",
    "get_visual_search",
    "reset_providers",
    # Base classes (for type hints)
    "LLMProvider",
    "VLMProvider",
    "EmbeddingProvider",
    "VectorDBProvider",
    "VisualSearchProvider",
    # Response models
    "LLMResponse",
    "VisualSearchResult",
]
