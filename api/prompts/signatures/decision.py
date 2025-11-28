"""
Decision Signature for VSM Agent.

Determines which tool to use next based on query and current state.
"""

import dspy


class DecisionSignature(dspy.Signature):
    """
    Decide which tool to use next based on query and current state.
    
    TERMINATION RULES (CRITICAL - FOLLOW STRICTLY):
    1. If environment has ANY relevant data, use text_response with should_end=True
    2. If iteration is 3+ and environment has data, MUST use text_response with should_end=True
    3. NEVER call the same tool type more than twice
    4. NEVER call visual_interpretation more than once - synthesize with text_response instead
    5. After ONE search + ONE interpretation attempt, use text_response to answer
    
    ANTI-LOOP RULES:
    1. NEVER repeat a search with the same or very similar query
    2. If you've tried a tool and it didn't provide the answer, DON'T try it again
    3. When in doubt, use text_response to provide the best answer from available data
    
    Tool Selection (IN ORDER OF PREFERENCE):
    1. text_response: USE THIS when environment has ANY data relevant to the query
    2. hybrid_search: First search for complex queries (ONE TIME ONLY)
    3. fast_vector_search: Simple factual lookups (ONE TIME ONLY)
    4. colqwen_search: Visual content (ONE TIME ONLY)
    5. visual_interpretation: ONLY if colqwen found relevant pages AND text_response can't answer
    
    IMPORTANT: The goal is to ANSWER the user, not to gather perfect data. 
    Synthesize an answer from available data rather than searching forever.
    """
    
    # Inputs
    query: str = dspy.InputField(
        desc="The user's question to answer"
    )
    available_tools: str = dspy.InputField(
        desc="JSON list of available tools with name and description"
    )
    environment_summary: str = dspy.InputField(
        desc="Summary of data already retrieved - if this has relevant content, use text_response!"
    )
    previous_queries: str = dspy.InputField(
        desc="List of queries already executed - DO NOT repeat these"
    )
    tasks_completed: str = dspy.InputField(
        desc="Tools already used in this session - avoid calling same tool type repeatedly"
    )
    iteration: str = dspy.InputField(
        desc="Current iteration (e.g., '2/10'). If >= 3 and env has data, MUST use text_response"
    )
    
    # Outputs
    tool_name: str = dspy.OutputField(
        desc="Name of the tool. If iteration >= 3 with data in env, MUST be 'text_response'"
    )
    tool_inputs: str = dspy.OutputField(
        desc="JSON object with input parameters."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation. Must justify why NOT using text_response if env has data."
    )
    should_end: bool = dspy.OutputField(
        desc="True if this is the final action. MUST be True when using text_response."
    )

