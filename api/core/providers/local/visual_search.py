"""Local Visual Search Provider - Wraps ColQwenRetriever."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from api.core.providers.base import VisualSearchProvider, VisualSearchResult

logger = logging.getLogger(__name__)


class ColQwenVisualSearch(VisualSearchProvider):
    """
    ColQwen-based visual search provider for local deployment.
    
    Wraps the existing ColQwenRetriever which handles:
    - Query embedding (multi-vector via ColQwen2.5-v0.2)
    - MaxSim scoring against page vectors in Weaviate
    - All computation on local GPU (MPS/CUDA)
    
    Images are stored on the local filesystem, referenced by path in Weaviate.
    """
    
    def __init__(self):
        self._retriever = None
    
    def _get_retriever(self):
        """Lazy load ColQwen retriever (heavy model)."""
        if self._retriever is None:
            from api.services.colqwen import get_colqwen_retriever
            self._retriever = get_colqwen_retriever()
        return self._retriever
    
    async def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[VisualSearchResult]:
        """
        Search for visually relevant pages using ColQwen late-interaction.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of VisualSearchResult with page info and MaxSim scores
        """
        retriever = self._get_retriever()
        
        # ColQwenRetriever.retrieve is synchronous (runs on GPU)
        raw_results = retriever.retrieve(query, top_k=top_k)
        
        results = []
        for r in raw_results:
            results.append(VisualSearchResult(
                page_id=r.get("page_id", 0),
                asset_manual=r.get("asset_manual", "Unknown"),
                page_number=r.get("page_number", 0),
                image_path=r.get("image_path", ""),
                image_base64="",  # Local mode uses filesystem paths
                score=r.get("maxsim_score", 0.0),
            ))
        
        return results
    
    async def is_available(self) -> bool:
        """Check if ColQwen is available."""
        try:
            retriever = self._get_retriever()
            return retriever is not None
        except Exception as e:
            logger.error(f"ColQwen availability check failed: {e}")
            return False
    
    async def ingest_page(
        self,
        page_id: int,
        asset_manual: str,
        page_number: int,
        image_path: str,
    ) -> None:
        """
        Ingest a single page image into PDFDocuments collection.
        
        Generates ColQwen multi-vectors and stores them in Weaviate
        with the "colqwen" named vector. Image path is stored as reference.
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
        import torch
        import weaviate
        from PIL import Image
        
        retriever = self._get_retriever()
        retriever._ensure_initialized()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        batch = retriever.processor.process_images([image]).to(retriever.device)
        
        with torch.no_grad():
            embedding = retriever.model(**batch)[0]
        
        # Convert to list format for Weaviate
        vectors = embedding.cpu().float().numpy().tolist()
        
        # Store in Weaviate with named vector
        with weaviate.connect_to_local() as client:
            coll = client.collections.get("PDFDocuments")
            
            coll.data.insert(
                properties={
                    "page_id": page_id,
                    "asset_manual": asset_manual,
                    "page_number": page_number,
                    "image_path": str(image_path),  # Store path, not blob
                },
                vector={"colqwen": vectors},
            )
        
        logger.info(f"Ingested page {page_number} from {asset_manual}")
    
    async def get_page_image(
        self,
        page_id: int,
    ) -> Optional[bytes]:
        """
        Retrieve page image bytes from filesystem.
        
        Looks up the image_path in Weaviate and reads the file.
        """
        return await asyncio.to_thread(self._get_page_image_sync, page_id)

    def _get_page_image_sync(self, page_id: int) -> Optional[bytes]:
        import weaviate
        from weaviate.classes.query import Filter
        
        try:
            with weaviate.connect_to_local() as client:
                coll = client.collections.get("PDFDocuments")
                
                result = coll.query.fetch_objects(
                    filters=Filter.by_property("page_id").equal(page_id),
                    limit=1,
                )
                
                if not result.objects:
                    return None
                
                image_path = result.objects[0].properties.get("image_path")
                if not image_path:
                    return None
                
                # Read from filesystem
                path = Path(image_path)
                if not path.is_absolute():
                    # Relative to project root
                    project_root = Path(__file__).parent.parent.parent.parent.parent
                    path = project_root / image_path
                
                if path.exists():
                    return path.read_bytes()
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get page image {page_id}: {e}")
            return None
