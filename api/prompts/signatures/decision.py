"""
Decision Signature for VSM Agent.

Determines which tool to use next based on query and current state.
"""

import dspy


class DecisionSignature(dspy.Signature):
    """
    Decide which tool to use next based on query and current state.
    
    Guidelines:
    - For tables, bit codes, menus, specifications: prefer hybrid_search
    - For diagrams, schematics, figures, "show me": prefer colqwen_search
    - For simple definitions or short procedures: prefer fast_vector_search
    - For complex/technical queries: prefer hybrid_search (runs in parallel)
    - When sufficient data is gathered: use text_response to answer
    - If environment has lots of data: consider summarize first
    
    PREFER hybrid_search for most technical queries - it finds both text AND visual content.
    """
    
    # Inputs
    query: str = dspy.InputField(
        desc="The user's question to answer"
    )
    available_tools: str = dspy.InputField(
        desc="JSON list of available tools with name and description"
    )
    environment_summary: str = dspy.InputField(
        desc="Summary of data already retrieved (may be empty)"
    )
    iteration: str = dspy.InputField(
        desc="Current iteration status (e.g., '2/10')"
    )
    
    # Outputs
    tool_name: str = dspy.OutputField(
        desc="Name of the tool to use (must match one from available_tools)"
    )
    tool_inputs: str = dspy.OutputField(
        desc="JSON object with input parameters for the chosen tool"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this tool was chosen"
    )
    should_end: bool = dspy.OutputField(
        desc="True if this should be the final action (usually with text_response)"
    )

