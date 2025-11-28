"""
Cloud VectorDB Provider - Weaviate Cloud for TEXT search.

Uses Weaviate Python client v4 for cloud connections.
Handles AssetManual collection (chunked text).

Ref: https://weaviate.io/developers/weaviate/client-libraries/python
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from api.core.config import get_settings
from api.core.providers.base import VectorDBProvider

logger = logging.getLogger(__name__)


class WeaviateCloud(VectorDBProvider):
    """
    Weaviate Cloud-based vector database provider for TEXT search.
    
    Handles AssetManual collection (chunked text).
    Visual search (PDFDocuments) is handled by JinaVisualSearch.
    
    Uses Weaviate Python client v4 with connect_to_weaviate_cloud().
    """
    
    def __init__(self):
        settings = get_settings()
        
        if not settings.weaviate_cloud_url:
            raise ValueError(
                "WEAVIATE_URL not set. "
                "Set VSM_MODE=local or configure WEAVIATE_URL."
            )
        
        if not settings.weaviate_cloud_api_key:
            raise ValueError(
                "WEAVIATE_API_KEY not set. "
                "Configure WEAVIATE_API_KEY for Weaviate Cloud."
            )
        
        self._cluster_url = settings.weaviate_cloud_url
        self._api_key = settings.weaviate_cloud_api_key
        self._jina_api_key = settings.jina_api_key  # For Jina vectorizer
        self._client = None
        
        logger.info(f"Initialized WeaviateCloud: url={self._cluster_url}")
    
    def connect(self) -> Any:
        """Connect to Weaviate Cloud using weaviate.connect_to_weaviate_cloud()."""
        if self._client is not None:
            return self._client
        
        # Lazy import to avoid loading weaviate when not needed
        import weaviate
        from weaviate.classes.init import Auth
        
        # Build additional headers for Jina API (if using Jina vectorizer)
        additional_headers = {}
        if self._jina_api_key:
            additional_headers["X-JinaAI-Api-Key"] = self._jina_api_key  # Official Weaviate header name
        
        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self._cluster_url,
            auth_credentials=Auth.api_key(self._api_key),
            headers=additional_headers,
        )
        
        logger.info("Connected to Weaviate Cloud")
        return self._client
    
    async def vector_search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Perform vector similarity search on Weaviate Cloud.
        
        Args:
            collection: Collection name (e.g., "AssetManual")
            query_vector: Query embedding vector
            limit: Max results to return
            filters: Optional Weaviate filters
            
        Returns:
            List of result objects with properties and metadata
        """
        return await asyncio.to_thread(
            self._vector_search_sync,
            collection,
            query_vector,
            limit,
            filters
        )

    def _vector_search_sync(
        self,
        collection: str,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        client = self.connect()
        
        import weaviate.classes.query as wq
        
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
        
        # Build query
        response = coll.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=weaviate_filter,
            return_metadata=wq.MetadataQuery(distance=True, certainty=True),
        )
        
        results = []
        for obj in response.objects:
            result = {
                "id": str(obj.uuid),
                "properties": dict(obj.properties),
                "distance": getattr(obj.metadata, "distance", None),
                "certainty": getattr(obj.metadata, "certainty", None),
            }
            results.append(result)
        
        return results
    
    async def hybrid_search(
        self,
        collection: str,
        query: str,
        query_vector: List[float],
        limit: int = 5,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Perform hybrid search (vector + BM25) on Weaviate Cloud.
        
        Args:
            collection: Collection name
            query: Text query for BM25
            query_vector: Query embedding for vector search
            limit: Max results
            alpha: Balance between vector (1.0) and keyword (0.0)
            filters: Optional filters
            
        Returns:
            List of result objects
        """
        return await asyncio.to_thread(
            self._hybrid_search_sync,
            collection,
            query,
            query_vector,
            limit,
            alpha,
            filters
        )

    def _hybrid_search_sync(
        self,
        collection: str,
        query: str,
        query_vector: List[float],
        limit: int,
        alpha: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        client = self.connect()
        
        import weaviate.classes.query as wq
        
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
        
        # Hybrid search with both text and vector
        response = coll.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
            filters=weaviate_filter,
            return_metadata=wq.MetadataQuery(score=True, explain_score=True),
        )
        
        results = []
        for obj in response.objects:
            result = {
                "id": str(obj.uuid),
                "properties": dict(obj.properties),
                "score": getattr(obj.metadata, "score", None),
            }
            results.append(result)
        
        return results

    async def batch_upsert(
        self,
        collection: str,
        objects: List[Dict[str, Any]],
    ) -> None:
        """
        Batch upsert objects to Weaviate Cloud.
        
        Args:
            collection: Collection name
            objects: List of objects with 'properties' and optionally 'vector'
        """
        await asyncio.to_thread(self._batch_upsert_sync, collection, objects)

    def _batch_upsert_sync(
        self,
        collection: str,
        objects: List[Dict[str, Any]],
    ) -> None:
        client = self.connect()
        
        coll = client.collections.get(collection)
        
        with coll.batch.dynamic() as batch:
            for obj in objects:
                properties = obj.get("properties", obj)
                vector = obj.get("vector")
                
                if vector:
                    batch.add_object(properties=properties, vector=vector)
                else:
                    batch.add_object(properties=properties)
        
        logger.info(f"Batch upserted {len(objects)} objects to {collection}")
    
    def close(self):
        """Close the Weaviate client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Closed Weaviate Cloud connection")
