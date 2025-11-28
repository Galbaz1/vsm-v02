"""
Search tools for VSM agentic RAG.

Wraps existing search services (fast vector, ColQwen) as Elysia-style tools.
"""

import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Tuple, TYPE_CHECKING

from api.services.tools.base import Tool
from api.schemas.agent import Result, Error, Status

if TYPE_CHECKING:
    from api.services.environment import TreeData


class FastVectorSearchTool(Tool):
    """
    Fast vector search over AssetManual collection.
    
    Uses configured Embedding and VectorDB providers.
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
                    "description": "Maximum number of results (default 5, use 10+ for tables/specifications)",
                    "type": "int",
                    "default": 5,
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
        """Execute fast vector search with deduplication."""
        from api.services.search import perform_search
        
        query = inputs.get("query", tree_data.user_prompt)
        limit = inputs.get("limit", 5)
        
        # Check if similar query was already executed (includes failed attempts)
        if tree_data.environment.has_query_been_executed(
            query,
            similarity_threshold=0.85,
            tasks_completed=tree_data.tasks_completed,
        ):
            yield Error(
                message=f"Similar query already executed: '{query[:50]}...'",
                recoverable=True,
                suggestion="Try a different query or use text_response to answer from existing data",
            )
            return
        
        yield Status(f"Searching AssetManual for: {query[:50]}...")
        
        start_time = time.time()
        
        try:
            # Search ALL chunk types (text, table, figure, title)
            # perform_search is now async and uses providers
            hits, page_hits = await perform_search(
                query=query,
                limit=limit,
                chunk_type=None,  # Always search all types
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
            
            # Deduplicate against existing content in environment
            existing_hashes = tree_data.environment.get_existing_content_hashes()
            
            # Convert SearchHit objects to dicts, filtering duplicates
            objects = []
            duplicates_skipped = 0
            for hit in hits:
                content_key = hit.content[:100] if hit.content else ""
                if content_key in existing_hashes:
                    duplicates_skipped += 1
                    continue
                    
                obj = {
                    "content": hit.content,
                    "manual_name": hit.manual_name,
                    "page_number": hit.page_number,
                    "score": hit.score,
                    "chunk_type": hit.chunk_type,
                    "section_title": hit.section_title,
                    "pdf_page_url": hit.pdf_page_url,
                    "page_image_url": hit.page_image_url,
                }
                if hit.bbox:
                    obj["bbox"] = hit.bbox.model_dump()
                objects.append(obj)
            
            if not objects and duplicates_skipped > 0:
                yield Error(
                    message=f"All {duplicates_skipped} results were duplicates of existing data",
                    recoverable=True,
                    suggestion="Use text_response to answer from existing data",
                )
                return
            
            llm_msg = f"Found {len(objects)} text chunks matching '{query}' in {elapsed_ms}ms."
            if duplicates_skipped > 0:
                llm_msg += f" ({duplicates_skipped} duplicates filtered)"
            
            yield Result(
                objects=objects,
                metadata={
                    "query": query,
                    "count": len(objects),
                    "time_ms": elapsed_ms,
                    "collection": "AssetManual",
                    "chunk_type_filter": None,  # Searches all types
                    "duplicates_filtered": duplicates_skipped,
                },
                name="AssetManual",
                llm_message=llm_msg,
            )
            
        except Exception as e:
            yield Error(
                message=f"Search failed: {str(e)}",
                recoverable=False,
                error_type="search_error",
            )


class ColQwenSearchTool(Tool):
    """
    Visual search over PDFDocuments collection.
    
    Uses VisualSearchProvider (ColQwen local, Jina CLIP cloud).
    Best for: diagrams, charts, wiring schematics, visual content.
    """
    
    def __init__(self):
        super().__init__(
            name="colqwen_search",
            description=(
                "Search for visual content (diagrams, charts, schematics, figures). "
                "Returns relevant page images with similarity scores. "
                "Use for queries like 'show me', 'diagram of', 'figure showing'. "
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
        """Check if visual search is available."""
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
        """Execute visual search with deduplication."""
        from api.core.providers import get_visual_search
        from api.core.config import get_settings
        
        query = inputs.get("query", tree_data.user_prompt)
        top_k = inputs.get("top_k", 3)
        
        # Check if similar query was already executed (includes failed attempts)
        if tree_data.environment.has_query_been_executed(
            query,
            similarity_threshold=0.85,
            tasks_completed=tree_data.tasks_completed,
        ):
            yield Error(
                message=f"Similar visual query already executed: '{query[:50]}...'",
                recoverable=True,
                suggestion="Try a different query or use text_response to answer from existing data",
            )
            return
        
        yield Status(f"Searching visual content for: {query[:50]}...")
        
        start_time = time.time()
        
        try:
            visual_search = get_visual_search()
            results = await visual_search.search(query=query, top_k=top_k)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            if not results:
                yield Error(
                    message=f"No visual results found for query: '{query}'",
                    recoverable=True,
                    suggestion="Try describing the visual content differently",
                )
                return
            
            # Deduplicate against existing pages in environment
            existing_pages = tree_data.environment.get_existing_page_numbers()
            
            # Enhance results with preview URLs, filtering duplicates
            settings = get_settings()
            objects = []
            duplicates_skipped = 0
            for result in results:
                if result.page_number in existing_pages:
                    duplicates_skipped += 1
                    continue
                    
                preview_url = None
                if result.image_path:
                    preview_url = f"{settings.api_base_url}/{result.image_path}"
                elif result.image_base64:
                    preview_url = f"data:image/png;base64,{result.image_base64}"
                elif result.page_id:
                    preview_url = f"{settings.api_base_url}/images/{result.page_id}"
                
                objects.append({
                    "page_id": result.page_id,  # Needed for cloud image retrieval
                    "page_number": result.page_number,
                    "asset_manual": result.asset_manual,
                    "score": result.score,
                    "maxsim_score": result.score,
                    "image_path": result.image_path,
                    "preview_url": preview_url,
                })
            
            if not objects and duplicates_skipped > 0:
                yield Error(
                    message=f"All {duplicates_skipped} visual results were duplicates of existing pages",
                    recoverable=True,
                    suggestion="Use text_response to answer from existing data",
                )
                return
            
            llm_msg = (
                f"Found {len(objects)} relevant pages with visual content. "
                f"Top result: page {objects[0]['page_number']} "
                f"(score: {objects[0]['score']:.3f})"
            )
            if duplicates_skipped > 0:
                llm_msg += f" ({duplicates_skipped} duplicate pages filtered)"
            
            yield Result(
                objects=objects,
                metadata={
                    "query": query,
                    "count": len(objects),
                    "time_ms": elapsed_ms,
                    "collection": "PDFDocuments",
                    "retrieval_type": "visual_search",
                    "duplicates_filtered": duplicates_skipped,
                },
                name="PDFDocuments",
                llm_message=llm_msg,
            )
            
        except Exception as e:
            yield Error(
                message=f"Visual search failed: {str(e)}",
                recoverable=False,
                error_type="visual_search_error",
                suggestion="Visual search service may not be available.",
            )


class HybridSearchTool(Tool):
    """
    Hybrid search that combines fast vector and ColQwen results.
    
    Runs both searches IN PARALLEL for optimal performance.
    Useful for complex queries that need both text and visual context.
    """
    
    def __init__(self):
        super().__init__(
            name="hybrid_search",
            description=(
                "Perform both text and visual search IN PARALLEL, combining results. "
                "Use for complex queries, tables, specifications, or any query that may "
                "benefit from both text chunks and relevant page images. "
                "Efficient: runs both pipelines simultaneously."
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
    
    async def _collect_results(
        self,
        tool: Tool,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
    ) -> Tuple[List[Dict], List[str]]:
        """Helper to collect all results from a tool (for parallel execution)."""
        results = []
        errors = []
        async for output in tool(tree_data, inputs):
            if isinstance(output, Result):
                results = output.objects
            elif isinstance(output, Error):
                errors.append(output.message)
        return results, errors
    
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Execute hybrid search with PARALLEL execution."""
        query = inputs.get("query", tree_data.user_prompt)
        text_limit = inputs.get("text_limit", 3)
        visual_limit = inputs.get("visual_limit", 2)
        
        yield Status("Running hybrid search (text + visual in parallel)...")
        
        start_time = time.time()
        
        # Create tool instances
        fast_tool = FastVectorSearchTool()
        colqwen_tool = ColQwenSearchTool()
        
        # Run BOTH searches in parallel using asyncio.gather
        (text_results, text_errors), (visual_results, visual_errors) = await asyncio.gather(
            self._collect_results(fast_tool, tree_data, {"query": query, "limit": text_limit}),
            self._collect_results(colqwen_tool, tree_data, {"query": query, "top_k": visual_limit}),
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Combine errors
        errors = [f"Text: {e}" for e in text_errors] + [f"Visual: {e}" for e in visual_errors]
        
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
                "time_ms": elapsed_ms,
                "parallel": True,
                "errors": errors if errors else None,
            },
            name="hybrid_search",
            llm_message=(
                f"Hybrid search found {len(text_results)} text chunks "
                f"and {len(visual_results)} relevant pages in {elapsed_ms}ms (parallel)."
            ),
        )
