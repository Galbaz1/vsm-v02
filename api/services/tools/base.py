"""
Tool base class - Adapted from Weaviate Elysia framework.

Provides the foundation for creating tools with:
- Conditional availability (is_tool_available)
- Auto-trigger conditions (run_if_true)
- Async generator execution

Original: https://github.com/weaviate/elysia/blob/main/elysia/objects.py
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from api.services.environment import TreeData

logger = logging.getLogger(__name__)


class Tool(ABC):
    """
    Base class for all tools in the VSM agentic RAG system.
    
    Tools must implement:
    - __call__: Async generator that yields Result/Error/Response objects
    
    Tools can optionally override:
    - is_tool_available: Control when the tool appears in the decision tree
    - run_if_true: Auto-trigger the tool under certain conditions
    
    Attributes:
        name: Unique identifier for the tool
        description: Detailed description for the LLM to understand when to use it
        status: Status message shown while tool is running
        inputs: Schema of inputs the tool accepts
        end: Whether this tool can end the conversation
    
    Example:
        class FastVectorSearchTool(Tool):
            def __init__(self):
                super().__init__(
                    name="fast_vector_search",
                    description="Search AssetManual collection using fast vector similarity",
                    inputs={
                        "query": {
                            "description": "Search query text",
                            "type": "str",
                            "required": True
                        },
                        "limit": {
                            "description": "Maximum results to return",
                            "type": "int",
                            "default": 5
                        }
                    }
                )
            
            async def __call__(self, tree_data, inputs, **kwargs):
                yield Status("Searching AssetManual...")
                results = await self._search(inputs["query"], inputs.get("limit", 5))
                yield Result(objects=results, name="AssetManual")
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        status: Optional[str] = None,
        inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        end: bool = False,
    ):
        """
        Initialize the tool.
        
        Args:
            name: Unique identifier
            description: Description for LLM decision-making
            status: Status message while running (default: "Running {name}...")
            inputs: Input schema for the tool
            end: Whether this tool can end the conversation
        """
        self.name = name
        self.description = description
        self.status = status or f"Running {name}..."
        self.inputs = inputs or {}
        self.end = end
    
    def get_default_inputs(self) -> Dict[str, Any]:
        """Get default values for all inputs."""
        return {
            key: spec.get("default")
            for key, spec in self.inputs.items()
            if "default" in spec
        }
    
    async def is_tool_available(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> bool:
        """
        Check if this tool should be available in the current context.
        
        Override this to control when the tool appears in the decision tree.
        For example, a "summarize" tool might only be available when the
        environment contains retrieved data.
        
        Args:
            tree_data: Current state of the decision tree
            **kwargs: Additional context (client_manager, etc.)
            
        Returns:
            True if the tool should be available, False otherwise
        
        Example:
            async def is_tool_available(self, tree_data, **kwargs) -> bool:
                # Only available when environment has data
                return not tree_data.environment.is_empty()
        """
        return True
    
    async def run_if_true(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if this tool should auto-trigger.
        
        Override this to automatically run the tool under certain conditions,
        bypassing the normal LLM decision process.
        
        Args:
            tree_data: Current state of the decision tree
            **kwargs: Additional context
            
        Returns:
            Tuple of (should_run, inputs_dict)
            - should_run: True to auto-trigger this tool
            - inputs_dict: Inputs to use when auto-triggering
        
        Example:
            async def run_if_true(self, tree_data, **kwargs) -> Tuple[bool, dict]:
                # Auto-summarize when environment gets too large
                if tree_data.environment.estimate_tokens() > 50000:
                    return True, {"max_tokens": 1000}
                return False, {}
        """
        return False, {}
    
    @abstractmethod
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """
        Execute the tool.
        
        This must be an async generator that yields Result, Error, Response,
        or Status objects.
        
        Args:
            tree_data: Current state including environment, history, etc.
            inputs: Input values for this tool execution
            **kwargs: Additional context (client_manager, model, etc.)
            
        Yields:
            Result, Error, Response, or Status objects
        
        Example:
            async def __call__(self, tree_data, inputs, **kwargs):
                yield Status("Starting search...")
                
                try:
                    results = await self._do_search(inputs["query"])
                    
                    if not results:
                        yield Error(
                            message="No results found",
                            recoverable=True,
                            suggestion="Try different keywords"
                        )
                        return
                    
                    yield Result(
                        objects=results,
                        metadata={"query": inputs["query"]},
                        name="search_results"
                    )
                    
                except Exception as e:
                    yield Error(
                        message=str(e),
                        recoverable=False
                    )
        """
        yield None  # Placeholder for abstract method


class TextResponseTool(Tool):
    """
    Tool that generates a text response to end the conversation.
    
    Uses configured LLM provider (Ollama or Gemini) for response generation.
    This is typically used as the final tool in a chain to synthesize
    an answer from the retrieved data.
    """
    
    RESPONSE_PROMPT = """You are a helpful technical assistant. Answer the user's question based on the retrieved information.

User Question: {query}

Retrieved Information:
{context}

Instructions:
1. Answer the question directly and concisely
2. Use specific details from the retrieved information
3. If referencing a page, mention the page number
4. If the information doesn't fully answer the question, say so
5. Keep your response focused and helpful

Answer:"""
    
    def __init__(self):
        super().__init__(
            name="text_response",
            description=(
                "Generate a final text response to answer the user's question. "
                "Use this when you have gathered enough information from searches "
                "and are ready to provide a complete answer."
            ),
            status="Generating response...",
            inputs={
                "include_sources": {
                    "description": "Whether to include source citations",
                    "type": "bool",
                    "default": True,
                }
            },
            end=True,  # This tool can end the conversation
        )
    
    async def is_tool_available(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> bool:
        """Only available when there's data to respond about."""
        # Available if there's any data in environment or it's been multiple iterations
        return not tree_data.environment.is_empty() or tree_data.num_iterations > 0
    
    def _build_context(self, tree_data: "TreeData", max_chars: int = 8000) -> Tuple[str, List[Dict]]:
        """Build context string from environment data."""
        all_objects = tree_data.environment.get_all_objects()
        
        context_parts = []
        sources = []
        total_chars = 0
        
        # Flatten hybrid search results if present
        flattened_objects = []
        for obj in all_objects:
            # Handle HybridSearchTool nested structure
            if "text_chunks" in obj or "visual_pages" in obj:
                flattened_objects.extend(obj.get("text_chunks", []))
                flattened_objects.extend(obj.get("visual_pages", []))
            else:
                flattened_objects.append(obj)
        
        for obj in flattened_objects:
            # Build source reference
            page = obj.get("page_number")
            manual = obj.get("manual_name") or obj.get("asset_manual", "Manual")
            
            if page:
                sources.append({"page": page, "manual": manual})
            
            # Add content to context - handle both text and visual results
            content = obj.get("content") or obj.get("interpretation", "")
            
            if content:
                # Text content from FastVectorSearch or VLM interpretation
                chunk_text = f"[Page {page}, {manual}]: {content}"
            elif obj.get("maxsim_score") is not None or obj.get("score") is not None:
                # Visual result from ColQwen - describe what was found
                maxsim = obj.get("maxsim_score")
                score = maxsim if maxsim is not None else obj.get("score", 0)
                # Handle both local path and potential future cloud URL
                image_ref = obj.get("preview_url") or obj.get("image_path", "")
                chunk_text = f"[Page {page}, {manual}]: Visual match found (score: {score:.2f}). View page image at: {image_ref}"
            else:
                continue
            
            if total_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        # Add visual interpretation results if present
        visual_data = tree_data.environment.find("visual_interpretation")
        if visual_data:
            for name, entries in visual_data.items():
                for entry in entries:
                    for obj in entry.get("objects", []):
                        interp = obj.get("interpretation")
                        page = obj.get("page_number")
                        if interp:
                            chunk_text = f"[Visual interpretation, Page {page}]: {interp}"
                            if total_chars + len(chunk_text) <= max_chars:
                                context_parts.append(chunk_text)
                                total_chars += len(chunk_text)
        
        return "\n\n".join(context_parts), sources
    
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Generate streaming response using LLM provider."""
        from api.schemas.agent import Response, Status
        from api.core.providers import get_llm
        
        yield Status("Generating response...")
        
        include_sources = inputs.get("include_sources", True)
        
        # Build context from environment
        context, sources = self._build_context(tree_data)
        
        if not context:
            yield Response(
                text="I couldn't find relevant information to answer your question. Please try rephrasing your query.",
            )
            return
        
        # Build the prompt
        prompt = self.RESPONSE_PROMPT.format(
            query=tree_data.user_prompt,
            context=context,
        )
        
        # Try streaming LLM generation
        try:
            llm = get_llm()
            
            # Use stream_chat with a single user message to support both providers
            messages = [{"role": "user", "content": prompt}]
            
            stream_gen = llm.stream_chat(
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            
            # Collect streamed text
            full_text = ""
            async for chunk in stream_gen:
                full_text += chunk
                # Could yield partial responses for real-time streaming
                # but for now we collect and yield at end
            
            yield Response(
                text=full_text.strip(),
                sources=sources[:5] if include_sources else None,
            )
            
        except Exception as e:
            logger.warning(f"LLM generation failed, using fallback: {e}")
            # Fallback to simple response
            yield Response(
                text=f"Based on the retrieved information ({len(sources)} sources), I found relevant content but could not generate a detailed response. Please review the search results directly.",
                sources=sources[:5] if include_sources else None,
            )


class SummarizeTool(Tool):
    """
    Tool that summarizes retrieved data.
    
    Uses configured LLM provider to condense large amounts of retrieved information.
    Only available when the environment contains data.
    """
    
    SUMMARIZE_PROMPT = """Summarize the following retrieved information concisely.

Information to summarize:
{context}

Instructions:
1. Create a coherent summary covering the main points
2. Keep it under {max_length} words
3. Preserve important technical details and numbers
4. Mention relevant page references

Summary:"""
    
    def __init__(self):
        super().__init__(
            name="summarize",
            description=(
                "Summarize the information that has been retrieved so far. "
                "Use this to condense multiple search results into a coherent summary."
            ),
            status="Summarizing results...",
            inputs={
                "max_length": {
                    "description": "Maximum length of summary in words",
                    "type": "int",
                    "default": 200,
                }
            },
            end=True,
        )
    
    async def is_tool_available(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> bool:
        """Only available when environment has data to summarize."""
        return not tree_data.environment.is_empty()
    
    async def run_if_true(
        self,
        tree_data: "TreeData",
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Auto-trigger if environment is very large."""
        if tree_data.environment.estimate_tokens() > 30000:
            return True, {"max_length": 500}
        return False, {}
    
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Summarize retrieved data using LLM provider."""
        from api.schemas.agent import Response, Status
        from api.core.providers import get_llm
        
        yield Status("Summarizing retrieved data...")
        
        max_length = inputs.get("max_length", 200)
        
        # Build context from all retrieved objects
        all_objects = tree_data.environment.get_all_objects()
        
        context_parts = []
        for obj in all_objects:
            content = obj.get("content") or obj.get("interpretation", "")
            page = obj.get("page_number", "?")
            manual = obj.get("manual_name") or obj.get("asset_manual", "Manual")
            
            if content:
                context_parts.append(f"[Page {page}, {manual}]: {content[:500]}")
        
        context = "\n\n".join(context_parts[:20])  # Limit to 20 items
        
        if not context:
            yield Response(text="No information to summarize.")
            return
        
        prompt = self.SUMMARIZE_PROMPT.format(
            context=context,
            max_length=max_length,
        )
        
        try:
            llm = get_llm()
            
            # Use stream_chat with single message
            messages = [{"role": "user", "content": prompt}]
            
            stream_gen = llm.stream_chat(
                messages=messages,
                temperature=0.5,
                max_tokens=max_length * 2,  # Rough token estimate
            )
            
            full_text = ""
            async for chunk in stream_gen:
                full_text += chunk
            
            yield Response(text=full_text.strip())
            
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            # Fallback
            yield Response(
                text=f"Retrieved {len(all_objects)} items from technical manuals. "
                f"Key sources include pages: {', '.join(str(o.get('page_number', '?')) for o in all_objects[:5])}."
            )
