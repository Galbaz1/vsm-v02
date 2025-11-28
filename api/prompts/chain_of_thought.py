"""
VSM Chain of Thought Module.

Inspired by Elysia's ElysiaChainOfThought, this module automatically
injects context into any DSPy Signature.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

import dspy

if TYPE_CHECKING:
    from api.services.environment import TreeData
    from api.knowledge import Atlas


class VSMChainOfThought(dspy.Module):
    """
    Chain of Thought module that auto-injects VSM context.
    
    Automatically adds to any signature:
    - user_prompt: The original user query
    - conversation_history: Prior conversation context
    - atlas: Domain knowledge (agent description, tool hints)
    - previous_errors: Errors from prior iterations (for self-healing)
    - environment: Retrieved data summary (optional)
    - tasks_completed: Completed task list (optional)
    
    Usage:
        sig = DecisionSignature
        module = VSMChainOfThought(sig)
        result = module(
            tree_data=tree_data,
            available_tools=tools_json,
            iteration="2/10",
        )
    """
    
    def __init__(self, signature: type, **config):
        """
        Initialize the module.
        
        Args:
            signature: DSPy Signature class to wrap
            **config: Additional config passed to dspy.ChainOfThought
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.ChainOfThought(signature, **config)
    
    def forward(
        self,
        tree_data: Optional["TreeData"] = None,
        atlas: Optional["Atlas"] = None,
        **kwargs,
    ) -> dspy.Prediction:
        """
        Run the signature with auto-injected context.
        
        Args:
            tree_data: TreeData state object (provides user_prompt, errors, env)
            atlas: Domain knowledge object
            **kwargs: Additional inputs for the signature
        
        Returns:
            DSPy Prediction with signature outputs
        """
        # Build context from tree_data
        context = self._build_context(tree_data, atlas)
        
        # Merge with explicit kwargs (kwargs take precedence)
        inputs = {**context, **kwargs}
        
        # Run prediction
        return self.predictor(**inputs)
    
    def _build_context(
        self,
        tree_data: Optional["TreeData"],
        atlas: Optional["Atlas"],
    ) -> Dict[str, Any]:
        """Build context dict from TreeData and Atlas."""
        context = {}
        
        if tree_data is not None:
            # User prompt
            context["query"] = tree_data.user_prompt
            
            # Conversation history (if available)
            if hasattr(tree_data, "conversation_history") and tree_data.conversation_history:
                context["conversation_history"] = self._format_conversation(
                    tree_data.conversation_history
                )
            
            # Previous errors (for self-healing)
            errors = tree_data.get_errors()
            if errors:
                context["previous_errors"] = "\n".join(f"- {e}" for e in errors[-3:])
            
            # Environment summary
            if tree_data.environment:
                context["environment_summary"] = tree_data.environment.to_llm_context(
                    max_tokens=4000
                )
            else:
                context["environment_summary"] = "(No data retrieved yet)"
            
            # Previous queries - CRITICAL for avoiding duplicate searches
            context["previous_queries"] = self._extract_previous_queries(tree_data)
            
            # Tasks completed - shows which tools have been called
            if hasattr(tree_data, "tasks_completed") and tree_data.tasks_completed:
                context["tasks_completed"] = self._format_tasks_for_decision(tree_data)
            else:
                context["tasks_completed"] = "(No tools called yet)"
        
        if atlas is not None:
            # Inject domain knowledge
            context["agent_description"] = atlas.agent_description
            if atlas.tool_hints:
                context["tool_hints"] = atlas.tool_hints
        
        return context
    
    def _extract_previous_queries(self, tree_data: "TreeData") -> str:
        """Extract all previous search queries from environment and tasks."""
        queries = set()
        
        # Extract from environment metadata
        if tree_data.environment:
            for tool_name in tree_data.environment.environment:
                for result_name in tree_data.environment.environment[tool_name]:
                    for entry in tree_data.environment.environment[tool_name][result_name]:
                        metadata = entry.get("metadata", {})
                        if "query" in metadata:
                            queries.add(metadata["query"])
        
        # Extract from tasks_completed
        if hasattr(tree_data, "tasks_completed"):
            for task_prompt in tree_data.tasks_completed:
                for task in task_prompt.get("task", []):
                    inputs = task.get("inputs", {})
                    if isinstance(inputs, dict) and "query" in inputs:
                        queries.add(inputs["query"])
        
        if not queries:
            return "(No previous queries)"
        
        return "\n".join(f"- {q}" for q in sorted(queries))
    
    def _format_tasks_for_decision(self, tree_data: "TreeData") -> str:
        """Format tasks completed for decision context - shows tool usage counts."""
        if not tree_data.tasks_completed:
            return "(No tools called yet)"
        
        # Count tool calls
        tool_counts: Dict[str, int] = {}
        tool_results: Dict[str, str] = {}  # success/error status
        
        for task_prompt in tree_data.tasks_completed:
            for task in task_prompt.get("task", []):
                tool_name = task.get("task", "unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                
                # Track if it errored
                if task.get("error"):
                    tool_results[tool_name] = "ERROR"
                elif "llm_message" in task:
                    tool_results[tool_name] = "SUCCESS"
        
        lines = ["Tools already called:"]
        for tool, count in tool_counts.items():
            status = tool_results.get(tool, "")
            warning = ""
            if count >= 2:
                warning = " ⚠️ STOP calling this tool!"
            lines.append(f"- {tool}: {count}x {status}{warning}")
        
        return "\n".join(lines)
    
    def _format_conversation(self, conversation: list) -> str:
        """Format conversation history as string."""
        if not conversation:
            return ""
        
        lines = []
        for msg in conversation[-5:]:  # Last 5 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Truncate
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)

