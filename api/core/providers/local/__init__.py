"""Local provider package - implementations for Mac Studio deployment."""

from api.core.providers.local.llm import OllamaLLM
from api.core.providers.local.vlm import MLXVLM
from api.core.providers.local.embeddings import OllamaEmbeddings
from api.core.providers.local.vectordb import WeaviateLocal
from api.core.providers.local.visual_search import ColQwenVisualSearch

__all__ = [
    "OllamaLLM",
    "MLXVLM",
    "OllamaEmbeddings",
    "WeaviateLocal",
    "ColQwenVisualSearch",
]
