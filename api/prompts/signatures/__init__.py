"""
DSPy Signatures for VSM Agent.

Defines model-agnostic contracts for:
- Decision: Tool selection logic
- SearchQuery: Query expansion for retrieval
- Response: Answer generation from context
"""

from api.prompts.signatures.decision import DecisionSignature
from api.prompts.signatures.search import SearchQuerySignature
from api.prompts.signatures.response import ResponseSignature

# Map names to signature classes for dynamic loading
SIGNATURE_MAP = {
    "decision": DecisionSignature,
    "search": SearchQuerySignature,
    "response": ResponseSignature,
}

__all__ = [
    "DecisionSignature",
    "SearchQuerySignature",
    "ResponseSignature",
    "SIGNATURE_MAP",
]

