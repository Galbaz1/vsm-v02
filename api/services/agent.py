"""
Agent service for orchestrating retrieval methods and generating responses.

The agent uses Qwen2.5-VL (or similar) to:
1. Analyze user queries
2. Decide which retrieval method(s) to use
3. Interpret visual results from ColQwen
4. Stream progressive responses
"""

import os
import json
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio

@dataclass
class SearchDecision:
    """Agent's decision on which search methods to use"""
    use_fast_vector: bool
    use_colqwen: bool
    strategy: str  # "fast_only", "colqwen_only", "fast_then_colqwen", "both_parallel"
    reasoning: str

class AgentOrchestrator:
    """
    Orchestrates retrieval methods based on query analysis.
    
    Uses a reasoning model to decide which search method(s) to use
    and interprets results with visual understanding.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize agent with Qwen2.5-VL or compatible model.
        
        Args:
            model_path: Path to local model or model identifier
        """
        self.model_path = model_path or "qwen/qwen2.5-vl-7b-instruct"
        self.model = None  # Loaded on demand
        self.tools = self._define_tools()
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available tools for the agent."""
        return [
            {
                "name": "fast_vector_search",
                "description": (
                    "Quick semantic search over manual content using Ollama embeddings. "
                    "Use for: factual queries, keyword-heavy questions, when speed is critical. "
                    "Returns in ~0.5-1s."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5},
                        "chunk_type": {"type": "string", "description": "Filter by chunk type (optional)"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "colqwen_late_interaction",
                "description": (
                    "Deep visual-semantic search using ColQwen2.5 multi-vector embeddings. "
                    "Use for: diagram questions, visual troubleshooting, complex spatial queries. "
                    "SLOWER (~3-5s) but more accurate for visual content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Max results", "default": 3}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    async def analyze_query(self, query: str, user_history: Optional[List] = None) -> SearchDecision:
        """
        Analyze user query and decide which retrieval method(s) to use.
        
        Args:
            query: User's search query
            user_history: Previous queries/results for context
        
        Returns:
            SearchDecision with strategy and reasoning
        """
        # For now, implement rule-based logic
        # TODO: Replace with actual LLM-based decision making
        
        # Keywords indicating visual content
        visual_keywords = ["diagram", "image", "picture", "schematic", "wiring", "figure", "show me"]
        
        # Keywords indicating simple factual queries
        simple_keywords = ["what is", "define", "voltage", "temperature", "model number"]
        
        query_lower = query.lower()
        
        # Check for visual indicators
        is_visual = any(kw in query_lower for kw in visual_keywords)
        
        # Check for simple factual
        is_simple = any(kw in query_lower for kw in simple_keywords) and len(query.split()) < 10
        
        if is_visual:
            return SearchDecision(
                use_fast_vector=False,
                use_colqwen=True,
                strategy="colqwen_only",
                reasoning="Query asks for visual content - using ColQwen for accurate visual grounding"
            )
        elif is_simple:
            return SearchDecision(
                use_fast_vector=True,
                use_colqwen=False,
                strategy="fast_only",
                reasoning="Simple factual query - fast vector search is sufficient"
            )
        else:
            # Complex query - use fast first, then optionally ColQwen
            return SearchDecision(
                use_fast_vector=True,
                use_colqwen=True,
                strategy="fast_then_colqwen",
                reasoning="Complex query - showing fast results first, then enriching with visual context"
            )
    
    async def stream_response(
        self,
        query: str,
        fast_results: Optional[List] = None,
        colqwen_results: Optional[List] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream progressive response based on available results.
        
        Yields:
            Chunks of response in format: {"type": "token", "content": "..."}
        """
        # TODO: Implement actual streaming with Qwen2.5-VL
        # For now, return a mock response
        
        if fast_results:
            yield {"type": "metadata", "content": {"source": "fast_vector", "count": len(fast_results)}}
            
            response = f"Based on the manual, "
            for char in response:
                yield {"type": "token", "content": char}
                await asyncio.sleep(0.01)  # Simulate streaming
        
        if colqwen_results:
            yield {"type": "metadata", "content": {"source": "colqwen", "count": len(colqwen_results)}}
            
            refinement = f"\n\nThe diagram on page {colqwen_results[0].get('page_number', 'unknown')} shows additional details."
            for char in refinement:
                yield {"type": "token", "content": char}
                await asyncio.sleep(0.01)

# Singleton instance
_agent = None

def get_agent() -> AgentOrchestrator:
    """Get or create agent orchestrator instance"""
    global _agent
    if _agent is None:
        _agent = AgentOrchestrator()
    return _agent
