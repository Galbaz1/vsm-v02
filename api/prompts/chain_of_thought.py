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
            
            # Tasks completed
            if hasattr(tree_data, "tasks_completed") and tree_data.tasks_completed:
                context["tasks_completed"] = tree_data.tasks_completed_string
        
        if atlas is not None:
            # Inject domain knowledge
            context["agent_description"] = atlas.agent_description
            if atlas.tool_hints:
                context["tool_hints"] = atlas.tool_hints
        
        return context
    
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

