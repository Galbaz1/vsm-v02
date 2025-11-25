"""
ColQwen retrieval service for multi-vector late-interaction search.
"""

import os
from typing import List, Dict, Any, Optional
import weaviate
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

COLLECTION_NAME = "PDFDocuments"

class ColQwenRetriever:
    """ColQwen-based retrieval using multi-vector embeddings"""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.model = None
        self.processor = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy load ColQwen models"""
        if self._initialized:
            return
        
        print(f"[ColQwen] Loading model on {self.device}...")
        
        self.model = ColQwen2_5.from_pretrained(
            "vidore/colqwen2.5-v0.2",
            dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        
        self.processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
        
        self._initialized = True
        print("[ColQwen] Model ready")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform late-interaction retrieval using ColQwen embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of search results with page info and distances
        """
        self._ensure_initialized()
        
        # Generate query embedding
        batch = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embedding = self.model(**batch)[0]
        
        # Convert to list for Weaviate
        query_vector = query_embedding.cpu().numpy().tolist()
        
        # Query Weaviate
        with weaviate.connect_to_local() as client:
            coll = client.collections.get(COLLECTION_NAME)
            
            response = coll.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
            )
        
        # Format results
        results = []
        for obj in response.objects:
            props = obj.properties
            results.append({
                "page_id": props.get("page_id"),
                "asset_manual": props.get("asset_manual"),
                "page_number": props.get("page_number"),
                "image_path": props.get("image_path"),
                "distance": obj.metadata.distance,
            })
        
        return results

# Singleton instance
_retriever = None

def get_colqwen_retriever() -> ColQwenRetriever:
    """Get or create ColQwen retriever instance"""
    global _retriever
    if _retriever is None:
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        _retriever = ColQwenRetriever(device=device)
    return _retriever
