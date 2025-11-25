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
    
    This is typically used as the final tool in a chain to synthesize
    an answer from the retrieved data.
    """
    
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
    
    async def __call__(
        self,
        tree_data: "TreeData",
        inputs: Dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Generate response - to be implemented with actual LLM call."""
        from api.schemas.agent import Response, Status
        
        yield Status("Generating response...")
        
        # Get all retrieved objects for context
        all_objects = tree_data.environment.get_all_objects()
        
        # TODO: Replace with actual LLM call using MLX
        # For now, generate a placeholder response
        if all_objects:
            sources = []
            for obj in all_objects[:3]:
                if "page_number" in obj:
                    sources.append({
                        "page": obj.get("page_number"),
                        "manual": obj.get("manual_name", "Manual"),
                    })
            
            yield Response(
                text=f"Based on the retrieved information ({len(all_objects)} chunks), here is what I found...",
                sources=sources if inputs.get("include_sources", True) else None,
            )
        else:
            yield Response(
                text="I couldn't find relevant information to answer your question. Please try rephrasing your query.",
            )


class SummarizeTool(Tool):
    """
    Tool that summarizes retrieved data.
    
    Only available when the environment contains data.
    """
    
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
        """Summarize - to be implemented with actual LLM call."""
        from api.schemas.agent import Response, Status
        
        yield Status("Summarizing retrieved data...")
        
        all_objects = tree_data.environment.get_all_objects()
        
        # TODO: Replace with actual LLM summarization
        yield Response(
            text=f"Summary of {len(all_objects)} retrieved items: ...",
        )

