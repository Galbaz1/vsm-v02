"""
Search Query Signature for VSM Agent.

Expands or refines user queries for better retrieval.
"""

import dspy


class SearchQuerySignature(dspy.Signature):
    """
    Expand or refine a user query for better retrieval.
    
    For technical manuals:
    - Add relevant technical terms and synonyms
    - Include component names mentioned in the query
    - Consider abbreviations and their expansions
    """
    
    # Inputs
    original_query: str = dspy.InputField(
        desc="The user's original query"
    )
    search_type: str = dspy.InputField(
        desc="Type of search: 'vector' (text), 'visual' (images), or 'hybrid' (both)"
    )
    
    # Outputs
    expanded_query: str = dspy.OutputField(
        desc="Expanded query with relevant technical terms"
    )
    keywords: str = dspy.OutputField(
        desc="JSON list of key terms for BM25 matching"
    )

