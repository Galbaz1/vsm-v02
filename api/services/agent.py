"""
Agent service - Elysia-style decision tree orchestrator.

Refactored from rule-based routing to use Elysia patterns:
- Environment for centralized state
- Tools with availability control
- Decision tree traversal
- Self-healing error handling
- LLM-powered decision making using DSPy (model-agnostic)

Original Elysia: https://github.com/weaviate/elysia
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

from api.services.environment import Environment, TreeData
from api.services.tracer import (
    QueryTracer,
    get_environment_debug_state,
    is_tracing_enabled,
)
from api.knowledge.thorguard import get_atlas
from api.services.tools import (
    Tool,
    FastVectorSearchTool,
    ColQwenSearchTool,
    HybridSearchTool,
    TextResponseTool,
    SummarizeTool,
    VisualInterpretationTool,
)
from api.prompts import get_vsm_module
from api.schemas.agent import (
    Result,
    Error,
    Response,
    Status,
    Decision,
    Complete,
)

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Elysia-style agent orchestrator for VSM dual-pipeline RAG.
    
    Features:
    - Decision tree with tool availability control
    - Centralized Environment for state management
    - Self-healing error handling
    - Progressive response streaming
    - DSPy-based decision making (supports both local/cloud models)
    
    Example:
        agent = get_agent()
        async for output in agent.run("Show me the wiring diagram"):
            if output["type"] == "result":
                print(f"Found {len(output['payload']['objects'])} results")
            elif output["type"] == "response":
                print(output["payload"]["text"])
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        collection_names: Optional[List[str]] = None,
    ):
        """
        Initialize the agent orchestrator.
        
        Args:
            max_iterations: Maximum decision tree iterations before forced stop
            collection_names: Available Weaviate collections
        """
        self.max_iterations = max_iterations
        self.collection_names = collection_names or ["AssetManual", "PDFDocuments"]
        
        # Initialize tools
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default tool set."""
        tools = [
            FastVectorSearchTool(),
            ColQwenSearchTool(),
            HybridSearchTool(),
            VisualInterpretationTool(),
            TextResponseTool(),
            SummarizeTool(),
        ]
        for tool in tools:
            self.tools[tool.name] = tool
    
    def add_tool(self, tool: Tool) -> None:
        """Add a custom tool to the agent."""
        self.tools[tool.name] = tool
    
    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent."""
        if tool_name in self.tools:
            del self.tools[tool_name]
    
    async def _get_available_tools(
        self,
        tree_data: TreeData,
    ) -> List[Tool]:
        """Get tools that are currently available based on state."""
        available = []
        for tool in self.tools.values():
            if await tool.is_tool_available(tree_data):
                available.append(tool)
        return available
    
    async def _check_auto_triggers(
        self,
        tree_data: TreeData,
    ) -> List[tuple[Tool, Dict[str, Any]]]:
        """Check which tools should auto-trigger."""
        triggers = []
        for tool in self.tools.values():
            should_run, inputs = await tool.run_if_true(tree_data)
            if should_run:
                triggers.append((tool, inputs))
        return triggers
    
    async def _make_decision(
        self,
        tree_data: TreeData,
        available_tools: List[Tool],
    ) -> Decision:
        """
        Decide which tool to use based on query and state.
        
        Uses DSPy module for intelligent routing, with rule-based fallback.
        """
        # Try LLM-based decision first
        try:
            return await self._make_llm_decision(tree_data, available_tools)
        except Exception as e:
            logger.warning(f"LLM decision failed, falling back to rules: {e}")
            return await self._make_rule_decision(tree_data, available_tools)
    
    async def _make_llm_decision(
        self,
        tree_data: TreeData,
        available_tools: List[Tool],
    ) -> Decision:
        """
        LLM-powered decision making using DSPy.
        
        Uses the "decision" module which auto-injects context via VSMChainOfThought.
        """
        import dspy
        from api.core.dspy_config import get_dspy_lm
        
        # Ensure DSPy is configured and capture the LM
        lm = get_dspy_lm()
        
        # Get DSPy module
        decision_module = get_vsm_module("decision")
        
        # Serialize tools for the prompt
        tools_list = []
        for tool in available_tools:
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputs": tool.inputs,
            })
        tools_json = json.dumps(tools_list, indent=2)
        
        # Run DSPy prediction in thread pool.
        # IMPORTANT: DSPy settings are thread-local, so we must use dspy.settings.context()
        # to propagate the LM config to the worker thread.
        def run_decision():
            # Apply DSPy settings in this worker thread
            with dspy.settings.context(lm=lm):
                return decision_module(
                    tree_data=tree_data,
                    available_tools=tools_json,
                    iteration=f"{tree_data.num_iterations}/{self.max_iterations}",
                )
            
        result = await asyncio.to_thread(run_decision)
        
        # Parse outputs
        tool_name = result.tool_name
        tool_inputs_str = result.tool_inputs
        reasoning = getattr(result, "reasoning", "No reasoning provided")
        
        # Robust boolean parsing for should_end
        raw_should_end = getattr(result, "should_end", False)
        if isinstance(raw_should_end, str):
            should_end = raw_should_end.lower() in ("true", "yes", "1")
        else:
            should_end = bool(raw_should_end)
        
        # Parse inputs JSON
        try:
            if isinstance(tool_inputs_str, str):
                tool_inputs = json.loads(tool_inputs_str)
            else:
                tool_inputs = tool_inputs_str  # Already a dict?
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool inputs JSON: {tool_inputs_str}")
            tool_inputs = {}
        
        # Validate tool exists
        tool_names = {t.name for t in available_tools}
        if tool_name not in tool_names:
            logger.warning(
                f"LLM chose unavailable tool: {tool_name}. "
                f"Available: {tool_names}"
            )
            # Fall back to text_response if tool doesn't exist
            tool_name = "text_response"
            should_end = True
        
        return Decision(
            tool_name=tool_name,
            inputs=tool_inputs,
            reasoning=reasoning,
            should_end=should_end,
        )
    
    async def _make_rule_decision(
        self,
        tree_data: TreeData,
        available_tools: List[Tool],
    ) -> Decision:
        """
        Rule-based fallback decision making.
        
        Used when LLM is unavailable or fails.
        """
        query = tree_data.user_prompt.lower()
        
        # Visual indicators
        visual_keywords = [
            "diagram", "figure", "schematic", "wiring", "circuit",
            "chart", "graph", "image", "picture", "show me", "visual"
        ]
        
        # Simple factual indicators
        factual_keywords = [
            "what is", "define", "how much", "voltage", "temperature",
            "specification", "model number", "dimension"
        ]
        
        # Check for explicit end request
        end_keywords = ["thank", "done", "finished", "that's all"]
        if any(kw in query for kw in end_keywords):
            return Decision(
                tool_name="text_response",
                inputs={},
                reasoning="User indicated they are done",
                should_end=True,
            )
        
        # Check if we already have enough data
        if not tree_data.environment.is_empty():
            # If we have data, prefer to respond
            if tree_data.num_iterations > 0:
                return Decision(
                    tool_name="text_response",
                    inputs={"include_sources": True},
                    reasoning="Already have retrieved data, generating response",
                    should_end=True,
                )
        
        # Route based on query type
        is_visual = any(kw in query for kw in visual_keywords)
        is_factual = any(kw in query for kw in factual_keywords)
        
        # Check tool availability
        has_fast = any(t.name == "fast_vector_search" for t in available_tools)
        has_colqwen = any(t.name == "colqwen_search" for t in available_tools)
        has_hybrid = any(t.name == "hybrid_search" for t in available_tools)
        
        if is_visual and has_colqwen:
            return Decision(
                tool_name="colqwen_search",
                inputs={"query": tree_data.user_prompt, "top_k": 3},
                reasoning="Query asks for visual content - using ColQwen for visual grounding",
            )
        elif is_factual and has_fast:
            return Decision(
                tool_name="fast_vector_search",
                inputs={"query": tree_data.user_prompt, "limit": 5},
                reasoning="Simple factual query - fast vector search is sufficient",
            )
        elif has_hybrid:
            return Decision(
                tool_name="hybrid_search",
                inputs={
                    "query": tree_data.user_prompt,
                    "text_limit": 3,
                    "visual_limit": 2,
                },
                reasoning="Complex query - using hybrid search for comprehensive results",
            )
        elif has_fast:
            return Decision(
                tool_name="fast_vector_search",
                inputs={"query": tree_data.user_prompt, "limit": 5},
                reasoning="Defaulting to fast vector search",
            )
        else:
            return Decision(
                tool_name="text_response",
                inputs={},
                reasoning="No search tools available, providing direct response",
                should_end=True,
                impossible=True,
            )
    
    async def _execute_tool(
        self,
        tool: Tool,
        tree_data: TreeData,
        inputs: Dict[str, Any],
        query_id: str,
        reasoning: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a tool and handle its outputs."""
        start_time = time.time()
        successful = True
        last_llm_message = ""
        
        # Set current tool for error context
        tree_data.set_current_tool(tool.name)
        
        try:
            async for output in tool(tree_data, inputs):
                if output is None:
                    continue
                
                if isinstance(output, Result):
                    # Add to environment
                    tree_data.environment.add(tool.name, output)
                    # Capture llm_message for task tracking
                    if hasattr(output, 'llm_message') and output.llm_message:
                        last_llm_message = output.llm_message
                    yield output.to_frontend(query_id)
                
                elif isinstance(output, Error):
                    # Add error for self-healing
                    tree_data.add_error(output.message)
                    successful = False
                    yield output.to_frontend(query_id)
                
                elif isinstance(output, (Response, Status)):
                    yield output.to_frontend(query_id)
                
        except Exception as e:
            tree_data.add_error(str(e))
            yield Error(
                message=str(e),
                recoverable=False,
            ).to_frontend(query_id)
            successful = False
        
        # Record task completion with rich context
        elapsed_ms = (time.time() - start_time) * 1000
        tree_data.update_tasks_completed(
            prompt=tree_data.user_prompt,
            task=tool.name,
            num_iterations=tree_data.num_iterations,
            reasoning=reasoning,
            inputs=inputs,
            llm_message=last_llm_message,
            action=True,
            error=not successful,
            duration_ms=elapsed_ms,
        )
        
        # Clear errors for this tool if successful
        if successful:
            tree_data.clear_errors(tool.name)
    
    async def run(
        self,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        query_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the agent decision tree.
        
        Args:
            user_prompt: User's query
            conversation_history: Previous conversation for context
            query_id: Unique ID for this query
        
        Yields:
            Stream of outputs (Result, Error, Response, Status, Decision, Complete)
        """
        query_id = query_id or str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize query tracer for debugging
        tracer = QueryTracer(
            query_id=query_id,
            user_query=user_prompt,
            enabled=is_tracing_enabled(),
        )
        
        # Initialize TreeData with Atlas for domain knowledge
        tree_data = TreeData(
            user_prompt=user_prompt,
            environment=Environment(),
            conversation_history=conversation_history or [],
            collection_names=self.collection_names,
            max_iterations=self.max_iterations,
            atlas=get_atlas(),
        )
        
        # Add user message to history
        tree_data.add_conversation_message("user", user_prompt)
        
        # Track outcome for tracer
        outcome = "completed"
        
        # Main decision loop
        while tree_data.num_iterations < tree_data.max_iterations:
            tree_data.num_iterations += 1
            
            # Get available tools
            available_tools = await self._get_available_tools(tree_data)
            
            if not available_tools:
                tracer.log_error("No tools available", recoverable=False)
                yield Error(
                    message="No tools available",
                    recoverable=False,
                ).to_frontend(query_id)
                outcome = "error"
                break
            
            # Check for auto-triggers
            auto_triggers = await self._check_auto_triggers(tree_data)
            for tool, inputs in auto_triggers:
                yield Status(f"Auto-triggered: {tool.name}").to_frontend(query_id)
                async for output in self._execute_tool(
                    tool, tree_data, inputs, query_id, reasoning=f"Auto-triggered: {tool.name}"
                ):
                    if output:
                        yield output
            
            # Make decision
            decision = await self._make_decision(tree_data, available_tools)
            
            # Log iteration to tracer
            tracer.log_iteration(
                iteration=tree_data.num_iterations,
                decision={
                    "tool_name": decision.tool_name,
                    "inputs": decision.inputs,
                    "reasoning": decision.reasoning,
                    "should_end": decision.should_end,
                },
                environment_state=get_environment_debug_state(tree_data.environment),
            )
            
            # Yield decision for transparency
            yield decision.to_frontend(query_id)
            
            # Check for impossible task
            if decision.impossible:
                yield Response(
                    text="I cannot complete this task with the available tools.",
                ).to_frontend(query_id)
                outcome = "impossible"
                break
            
            # Execute chosen tool
            if decision.tool_name in self.tools:
                tool = self.tools[decision.tool_name]
                yield Status(tool.status).to_frontend(query_id)
                
                async for output in self._execute_tool(
                    tool, tree_data, decision.inputs, query_id, reasoning=decision.reasoning
                ):
                    if output:
                        yield output
            
            # Check if we should end
            if decision.should_end:
                break
        else:
            # Loop exhausted max_iterations
            outcome = "max_iterations"
            logger.warning(
                f"Query {query_id[:8]} hit max iterations ({self.max_iterations})"
            )
        
        # Save trace for debugging
        total_time_ms = (time.time() - start_time) * 1000
        tracer.save(outcome=outcome, total_time_ms=total_time_ms)
        
        # Yield completion signal
        yield Complete().to_frontend(query_id)
    
    # Legacy compatibility methods
    async def analyze_query(self, query: str, user_history: Optional[List] = None):
        """Legacy method for backward compatibility."""
        tree_data = TreeData(
            user_prompt=query,
            conversation_history=user_history or [],
            collection_names=self.collection_names,
        )
        
        available_tools = await self._get_available_tools(tree_data)
        decision = await self._make_decision(tree_data, available_tools)
        
        # Convert to legacy SearchDecision format
        from dataclasses import dataclass
        
        @dataclass
        class SearchDecision:
            use_fast_vector: bool
            use_colqwen: bool
            strategy: str
            reasoning: str
        
        use_fast = decision.tool_name in ["fast_vector_search", "hybrid_search"]
        use_colqwen = decision.tool_name in ["colqwen_search", "hybrid_search"]
        
        if decision.tool_name == "fast_vector_search":
            strategy = "fast_only"
        elif decision.tool_name == "colqwen_search":
            strategy = "colqwen_only"
        elif decision.tool_name == "hybrid_search":
            strategy = "fast_then_colqwen"
        else:
            strategy = "fast_only"
        
        return SearchDecision(
            use_fast_vector=use_fast,
            use_colqwen=use_colqwen,
            strategy=strategy,
            reasoning=decision.reasoning,
        )


# Singleton instance
_agent: Optional[AgentOrchestrator] = None


def get_agent() -> AgentOrchestrator:
    """Get or create agent orchestrator instance."""
    global _agent
    if _agent is None:
        _agent = AgentOrchestrator()
    return _agent


def reset_agent() -> None:
    """Reset the agent instance (useful for testing)."""
    global _agent
    _agent = None
