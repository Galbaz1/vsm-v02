"""Local VectorDB Provider - Wraps Weaviate Docker for TEXT search."""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from api.core.providers.base import VectorDBProvider
from api.core.config import get_settings

if TYPE_CHECKING:
    import weaviate

logger = logging.getLogger(__name__)


class WeaviateLocal(VectorDBProvider):
    """
    Weaviate-based vector database provider for local deployment.
    
    Handles TEXT search on AssetManual collection.
    Visual search (PDFDocuments) is handled by ColQwenVisualSearch.
    """
    
    def __init__(self):
        settings = get_settings()
        self.url = settings.weaviate_local_url
        self._client: "weaviate.WeaviateClient" | None = None
    
    def connect(self) -> "weaviate.WeaviateClient":
        """Get or create Weaviate client connection."""
        if self._client is None:
            import weaviate
            # Parse host from URL (remove http://)
            host = self.url.replace("http://", "").replace("https://", "")
            if ":" in host:
                host = host.split(":")[0]
            self._client = weaviate.connect_to_local(host=host)
        return self._client
    
    async def vector_search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Perform vector similarity search."""
        client = self.connect()
        
        try:
            coll = client.collections.get(collection)
            
            # Build filters if provided
            weaviate_filter = None
            if filters:
                from weaviate.classes.query import Filter
                for key, value in filters.items():
                    if weaviate_filter is None:
                        weaviate_filter = Filter.by_property(key).equal(value)
                    else:
                        weaviate_filter = weaviate_filter & Filter.by_property(key).equal(value)
            
            result = coll.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=weaviate_filter,
            )
            
            hits = []
            for obj in result.objects:
                props = obj.properties or {}
                hits.append({
                    "id": str(obj.uuid),
                    "properties": props,
                    "score": getattr(obj.metadata, "distance", 0.0) if hasattr(obj, "metadata") else 0.0,
                })
            
            return hits
            
        except Exception as e:
            logger.error(f"Weaviate vector search error: {e}")
            raise
    
    async def hybrid_search(
        self,
        collection: str,
        query: str,
        query_vector: List[float],
        limit: int = 5,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Perform hybrid search (vector + BM25)."""
        client = self.connect()
        
        try:
            coll = client.collections.get(collection)
            
            weaviate_filter = None
            if filters:
                from weaviate.classes.query import Filter
                for key, value in filters.items():
                    if weaviate_filter is None:
                        weaviate_filter = Filter.by_property(key).equal(value)
                    else:
                        weaviate_filter = weaviate_filter & Filter.by_property(key).equal(value)
            
            result = coll.query.hybrid(
                query=query,
                vector=query_vector,
                limit=limit,
                alpha=alpha,
                filters=weaviate_filter,
            )
            
            hits = []
            for obj in result.objects:
                props = obj.properties or {}
                hits.append({
                    "id": str(obj.uuid),
                    "properties": props,
                    "score": getattr(obj.metadata, "score", 0.0) if hasattr(obj, "metadata") else 0.0,
                })
            
            return hits
            
        except Exception as e:
            logger.error(f"Weaviate hybrid search error: {e}")
            raise

    async def batch_upsert(
        self,
        collection: str,
        objects: List[Dict[str, Any]],
    ) -> None:
        """Batch upsert objects with dense embeddings (for text ingestion)."""
        client = self.connect()
        
        try:
            coll = client.collections.get(collection)
            
            with coll.batch.dynamic() as batch:
                for obj in objects:
                    vector = obj.pop("vector", None)
                    batch.add_object(properties=obj, vector=vector)
            
        except Exception as e:
            logger.error(f"Weaviate batch upsert error: {e}")
            raise
    
    async def close(self) -> None:
        """Close the Weaviate client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
