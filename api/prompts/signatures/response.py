"""
Response Signature for VSM Agent.

Generates helpful responses from retrieved information.
"""

import dspy


class ResponseSignature(dspy.Signature):
    """
    Generate a helpful response from retrieved information.
    
    Guidelines:
    - Be direct and answer the question first
    - Reference specific pages when citing information
    - If information is incomplete, acknowledge limitations
    - Use technical terminology accurately
    """
    
    # Inputs
    query: str = dspy.InputField(
        desc="The user's question"
    )
    context: str = dspy.InputField(
        desc="Retrieved information with page references"
    )
    
    # Outputs
    answer: str = dspy.OutputField(
        desc="Direct, helpful answer to the question"
    )
    sources: str = dspy.OutputField(
        desc="JSON list of source references [{page, manual, relevance}]"
    )
    confidence: str = dspy.OutputField(
        desc="Confidence level: 'high', 'medium', or 'low'"
    )

