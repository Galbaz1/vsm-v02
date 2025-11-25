"""
Search tools for VSM agentic RAG.

Wraps existing search services (fast vector, ColQwen) as Elysia-style tools.
"""

import time
from typing import Any, AsyncGenerator, Dict, Tuple, TYPE_CHECKING

from api.services.tools.base import Tool
from api.schemas.agent import Result, Error, Status

if TYPE_CHECKING:
    from api.services.environment import TreeData


class FastVectorSearchTool(Tool):
    """
    Fast vector search over AssetManual collection.
    
    Uses Ollama nomic-embed-text embeddings for ~0.5s query time.
    Best for: factual queries, text content, quick lookups.
    """
    
    def __init__(self):
        super().__init__(
            name="fast_vector_search",
            description=(
                "Search the AssetManual collection using fast vector similarity. "
                "Best for factual queries about specifications, procedures, or text content. "
                "Returns text chunks with page numbers, bounding boxes, and section titles. "
                "Average query time: ~0.5 seconds."
            ),
            status="Searching text content...",
            inputs={
                "query": {
                    "description": "Search query text",
                    "type": "str",
                    "required": True,
                },
                "limit": {
                    "description": "Maximum number of results",
                    "type": "int",
                    "default": 5,
                },
                "chunk_type": {
                    "description": "Filter by chunk type (text, table, figure, title)",
                    "type": "str",
                    "default": None,
                },
            },
            end=False,
        )
    
    async def is_tool_available(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> bool:
        """Always available for text-based queries."""
        return "AssetManual" in tree_data.collection_names
    
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Execute fast vector search."""
        from api.services.search import perform_search
        
        query = inputs.get("query", tree_data.user_prompt)
        limit = inputs.get("limit", 5)
        chunk_type = inputs.get("chunk_type")
        
        yield Status(f"Searching AssetManual for: {query[:50]}...")
        
        start_time = time.time()
        
        try:
            hits, page_hits = perform_search(
                query=query,
                limit=limit,
                chunk_type=chunk_type,
                group_by_page=True,
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            if not hits:
                yield Error(
                    message=f"No results found for query: '{query}'",
                    recoverable=True,
                    suggestion="Try different keywords or broaden the search terms",
                )
                return
            
            # Convert SearchHit objects to dicts
            objects = []
            for hit in hits:
                obj = {
                    "content": hit.content,
                    "manual_name": hit.manual_name,
                    "page_number": hit.page_number,
                    "chunk_type": hit.chunk_type,
                    "section_title": hit.section_title,
                    "pdf_page_url": hit.pdf_page_url,
                    "page_image_url": hit.page_image_url,
                }
                if hit.bbox:
                    obj["bbox"] = hit.bbox.model_dump()
                objects.append(obj)
            
            yield Result(
                objects=objects,
                metadata={
                    "query": query,
                    "count": len(objects),
                    "time_ms": elapsed_ms,
                    "collection": "AssetManual",
                    "chunk_type_filter": chunk_type,
                },
                name="AssetManual",
                llm_message=f"Found {len(objects)} text chunks matching '{query}' in {elapsed_ms}ms.",
            )
            
        except Exception as e:
            yield Error(
                message=f"Search failed: {str(e)}",
                recoverable=False,
                error_type="weaviate_error",
            )


class ColQwenSearchTool(Tool):
    """
    ColQwen visual search over PDFDocuments collection.
    
    Uses ColQwen2.5 multi-vector embeddings with late-interaction (MaxSim).
    Best for: diagrams, charts, wiring schematics, visual content.
    """
    
    def __init__(self):
        super().__init__(
            name="colqwen_search",
            description=(
                "Search using ColQwen multimodal embeddings for visual content. "
                "Best for queries about diagrams, charts, wiring schematics, figures, or visual layouts. "
                "Returns full page images with MaxSim similarity scores. "
                "Average query time: 3-5 seconds (slower but more accurate for visual queries)."
            ),
            status="Searching visual content...",
            inputs={
                "query": {
                    "description": "Search query (can describe visual content)",
                    "type": "str",
                    "required": True,
                },
                "top_k": {
                    "description": "Number of top pages to return",
                    "type": "int",
                    "default": 3,
                },
            },
            end=False,
        )
    
    async def is_tool_available(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> bool:
        """Check if ColQwen search is available."""
        return "PDFDocuments" in tree_data.collection_names
    
    async def run_if_true(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Auto-trigger for explicitly visual queries.
        
        Detects keywords like 'diagram', 'figure', 'schematic', etc.
        """
        visual_keywords = [
            "diagram", "figure", "schematic", "wiring", "circuit",
            "chart", "graph", "image", "picture", "show me",
            "visual", "layout", "drawing"
        ]
        
        query_lower = tree_data.user_prompt.lower()
        
        # Strong visual indicators - auto-trigger
        if any(kw in query_lower for kw in ["show me", "diagram of", "figure showing"]):
            return True, {"query": tree_data.user_prompt, "top_k": 3}
        
        return False, {}
    
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Execute ColQwen visual search."""
        from api.services.colqwen import get_colqwen_retriever
        from api.core.config import get_settings
        
        query = inputs.get("query", tree_data.user_prompt)
        top_k = inputs.get("top_k", 3)
        
        yield Status(f"Searching visual content for: {query[:50]}...")
        
        start_time = time.time()
        
        try:
            retriever = get_colqwen_retriever()
            results = retriever.retrieve(query=query, top_k=top_k)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            if not results:
                yield Error(
                    message=f"No visual results found for query: '{query}'",
                    recoverable=True,
                    suggestion="Try describing the visual content differently",
                )
                return
            
            # Enhance results with preview URLs
            settings = get_settings()
            objects = []
            for result in results:
                # Build preview URL from image_path
                image_path = result.get("image_path", "")
                if image_path:
                    # image_path is like "static/previews/techman/page-1.png"
                    preview_url = f"{settings.api_base_url}/{image_path}"
                else:
                    preview_url = None
                
                objects.append({
                    "page_number": result.get("page_number"),
                    "asset_manual": result.get("asset_manual"),
                    "maxsim_score": result.get("maxsim_score"),
                    "image_path": image_path,
                    "preview_url": preview_url,
                })
            
            yield Result(
                objects=objects,
                metadata={
                    "query": query,
                    "count": len(objects),
                    "time_ms": elapsed_ms,
                    "collection": "PDFDocuments",
                    "retrieval_type": "colqwen_maxsim",
                },
                name="PDFDocuments",
                llm_message=(
                    f"Found {len(objects)} relevant pages with visual content. "
                    f"Top result: page {objects[0]['page_number']} "
                    f"(score: {objects[0]['maxsim_score']:.3f})"
                ),
            )
            
        except Exception as e:
            yield Error(
                message=f"ColQwen search failed: {str(e)}",
                recoverable=False,
                error_type="colqwen_error",
                suggestion="ColQwen may not be initialized. Try again or use fast_vector_search.",
            )


class HybridSearchTool(Tool):
    """
    Hybrid search that combines fast vector and ColQwen results.
    
    Useful for complex queries that need both text and visual context.
    """
    
    def __init__(self):
        super().__init__(
            name="hybrid_search",
            description=(
                "Perform both text and visual search, combining results. "
                "Use for complex queries that may benefit from both text chunks "
                "and relevant page images. Returns merged results from both pipelines."
            ),
            status="Performing hybrid search...",
            inputs={
                "query": {
                    "description": "Search query",
                    "type": "str",
                    "required": True,
                },
                "text_limit": {
                    "description": "Max text results",
                    "type": "int",
                    "default": 3,
                },
                "visual_limit": {
                    "description": "Max visual results",
                    "type": "int",
                    "default": 2,
                },
            },
            end=False,
        )
    
    async def is_tool_available(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> bool:
        """Available when both collections exist."""
        return (
            "AssetManual" in tree_data.collection_names and
            "PDFDocuments" in tree_data.collection_names
        )
    
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Execute hybrid search."""
        query = inputs.get("query", tree_data.user_prompt)
        text_limit = inputs.get("text_limit", 3)
        visual_limit = inputs.get("visual_limit", 2)
        
        yield Status("Running hybrid search (text + visual)...")
        
        # Run both searches
        fast_tool = FastVectorSearchTool()
        colqwen_tool = ColQwenSearchTool()
        
        text_results = []
        visual_results = []
        errors = []
        
        # Fast vector search
        async for output in fast_tool(tree_data, {"query": query, "limit": text_limit}):
            if isinstance(output, Result):
                text_results = output.objects
            elif isinstance(output, Error):
                errors.append(f"Text: {output.message}")
        
        # ColQwen search
        async for output in colqwen_tool(tree_data, {"query": query, "top_k": visual_limit}):
            if isinstance(output, Result):
                visual_results = output.objects
            elif isinstance(output, Error):
                errors.append(f"Visual: {output.message}")
        
        if not text_results and not visual_results:
            yield Error(
                message="Both search pipelines returned no results",
                recoverable=True,
                suggestion="Try different search terms",
            )
            return
        
        # Combine results
        combined = {
            "text_chunks": text_results,
            "visual_pages": visual_results,
        }
        
        yield Result(
            objects=[combined],
            metadata={
                "query": query,
                "text_count": len(text_results),
                "visual_count": len(visual_results),
                "errors": errors if errors else None,
            },
            name="hybrid_search",
            llm_message=(
                f"Hybrid search found {len(text_results)} text chunks "
                f"and {len(visual_results)} relevant pages."
            ),
        )

