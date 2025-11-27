"""
Cloud Visual Search Provider - Jina CLIP + Weaviate Cloud Native Integration.

Uses Weaviate's native multi2vec-jinaai module for multimodal embeddings.
Images are stored as base64 blobs in Weaviate - no external storage needed.
Weaviate handles embedding automatically via Jina API.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import List, Optional

from api.core.providers.base import VisualSearchProvider, VisualSearchResult
from api.core.config import get_settings

logger = logging.getLogger(__name__)

# Collection name for visual search
COLLECTION_NAME = "PDFDocuments"


class JinaVisualSearch(VisualSearchProvider):
    """
    Jina CLIP-based visual search provider for cloud deployment.
    
    Architecture:
    - Weaviate Cloud with multi2vec-jinaai vectorizer
    - Images stored as base64 blobs in Weaviate
    - Jina CLIP v2 generates embeddings automatically at ingestion
    - Near-text and near-image search supported natively
    
    No serverless worker needed - Weaviate handles everything!
    """
    
    def __init__(self):
        settings = get_settings()
        
        if not settings.weaviate_cloud_url:
            raise ValueError(
                "WEAVIATE_URL not configured. "
                "Set WEAVIATE_URL environment variable for cloud mode."
            )
        if not settings.jina_api_key:
            raise ValueError(
                "JINA_API_KEY not configured. "
                "Set JINA_API_KEY environment variable for cloud mode."
            )
        
        self._client = None
        self._settings = settings
    
    def _get_client(self):
        """Get or create Weaviate Cloud client."""
        if self._client is None:
            import weaviate
            from weaviate.classes.init import Auth
            
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self._settings.weaviate_cloud_url,
                auth_credentials=Auth.api_key(self._settings.weaviate_cloud_api_key),
                headers={"X-JinaAI-Api-Key": self._settings.jina_api_key},
            )
        return self._client
    
    async def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[VisualSearchResult]:
        """
        Search for visually relevant pages using Jina CLIP.
        
        Uses Weaviate's near_text which automatically embeds the query
        using the configured Jina CLIP model.
        """
        client = self._get_client()
        
        try:
            coll = client.collections.get(COLLECTION_NAME)
            
            # Near-text search - Weaviate embeds query via Jina CLIP
            response = coll.query.near_text(
                query=query,
                limit=top_k,
                return_properties=["page_id", "asset_manual", "page_number"],
                # Don't return blob by default (large), fetch separately if needed
            )
            
            results = []
            for obj in response.objects:
                props = obj.properties
                # Distance is returned, convert to similarity score
                score = 1.0 - (obj.metadata.distance if obj.metadata and obj.metadata.distance else 0.0)
                
                results.append(VisualSearchResult(
                    page_id=props.get("page_id", 0),
                    asset_manual=props.get("asset_manual", "Unknown"),
                    page_number=props.get("page_number", 0),
                    image_path="",  # Cloud mode doesn't use filesystem paths
                    image_base64="",  # Fetch separately via get_page_image()
                    score=score,
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Jina visual search failed: {e}")
            raise
    
    async def is_available(self) -> bool:
        """Check if Weaviate Cloud and Jina are reachable."""
        try:
            client = self._get_client()
            # Simple check - try to get collection
            client.collections.get(COLLECTION_NAME)
            return True
        except Exception as e:
            logger.error(f"Jina visual search availability check failed: {e}")
            return False
    
    async def ingest_page(
        self,
        page_id: int,
        asset_manual: str,
        page_number: int,
        image_path: str,
    ) -> None:
        """
        Ingest a single page image into Weaviate Cloud.
        
        Reads the image, converts to base64, and stores as blob.
        Weaviate automatically generates Jina CLIP embeddings.
        """
        await asyncio.to_thread(
            self._ingest_page_sync,
            page_id,
            asset_manual,
            page_number,
            image_path,
        )

    def _ingest_page_sync(
        self,
        page_id: int,
        asset_manual: str,
        page_number: int,
        image_path: str,
    ) -> None:
        client = self._get_client()
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image_bytes = path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        try:
            coll = client.collections.get(COLLECTION_NAME)
            
            # Insert with blob - Weaviate auto-embeds via Jina
            coll.data.insert(
                properties={
                    "page_id": page_id,
                    "asset_manual": asset_manual,
                    "page_number": page_number,
                    "page_image": image_b64,  # Blob field for Jina CLIP
                },
            )
            
            logger.info(f"Ingested page {page_number} from {asset_manual} to cloud")
            
        except Exception as e:
            logger.error(f"Failed to ingest page {page_number}: {e}")
            raise
    
    async def get_page_image(
        self,
        page_id: int,
    ) -> Optional[bytes]:
        """
        Retrieve page image bytes from Weaviate blob.
        """
        client = self._get_client()
        
        try:
            from weaviate.classes.query import Filter
            
            coll = client.collections.get(COLLECTION_NAME)
            
            result = coll.query.fetch_objects(
                filters=Filter.by_property("page_id").equal(page_id),
                limit=1,
                return_properties=["page_image"],  # Get the blob
            )
            
            if not result.objects:
                return None
            
            image_b64 = result.objects[0].properties.get("page_image")
            if not image_b64:
                return None
            
            # Decode base64 to bytes
            return base64.b64decode(image_b64)
            
        except Exception as e:
            logger.error(f"Failed to get page image {page_id}: {e}")
            return None
    
    async def ensure_collection_exists(self) -> None:
        """
        Create the PDFDocuments collection with Jina CLIP vectorizer if it doesn't exist.
        
        Call this during cloud setup/ingestion.
        """
        client = self._get_client()
        
        try:
            # Check if collection exists
            if client.collections.exists(COLLECTION_NAME):
                logger.info(f"Collection {COLLECTION_NAME} already exists")
                return
            
            from weaviate.classes.config import Configure, Property, DataType, Multi2VecField
            
            client.collections.create(
                COLLECTION_NAME,
                properties=[
                    Property(name="page_id", data_type=DataType.INT),
                    Property(name="asset_manual", data_type=DataType.TEXT),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="page_image", data_type=DataType.BLOB),
                ],
                vector_config=[
                    Configure.Vectors.multi2vec_jinaai(
                        name="jina_clip",
                        image_fields=[
                            Multi2VecField(name="page_image", weight=1.0)
                        ],
                        model="jina-clip-v2",
                    )
                ],
            )
            
            logger.info(f"Created collection {COLLECTION_NAME} with Jina CLIP vectorizer")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def close(self) -> None:
        """Close Weaviate client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
