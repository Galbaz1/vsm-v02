"""
Agentic search endpoint with progressive results streaming.

Uses the Elysia-style decision tree agent to intelligently:
- Select tools (fast vector, ColQwen, visual interpretation)
- Execute searches and gather results
- Generate synthesized responses

Streams NDJSON payloads matching frontend types:
- status: Progress updates
- decision: Tool selection with reasoning
- result: Search results (text, visual, interpretations)
- response: Final synthesized answer
- error: Recoverable and non-recoverable errors
- complete: Stream finished
"""

import json
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from api.services.agent import get_agent

router = APIRouter()


@router.get("/agentic_search")
async def agentic_search(
    query: str = Query(..., min_length=3, description="User's search query"),
):
    """
    Agentic search using LLM-powered tool selection.
    
    The agent decides which tools to use based on the query:
    - fast_vector_search: For factual text queries
    - colqwen_search: For visual content (diagrams, schematics)
    - visual_interpretation: For analyzing page images
    - hybrid_search: For complex queries needing both
    - text_response: For generating final answers
    
    Response format (NDJSON):
    ```
    {"type": "status", "query_id": "...", "payload": {"message": "..."}}
    {"type": "decision", "query_id": "...", "payload": {"tool": "...", "reasoning": "..."}}
    {"type": "result", "query_id": "...", "payload": {"objects": [...], "name": "...", "count": N}}
    {"type": "response", "query_id": "...", "payload": {"text": "...", "sources": [...]}}
    {"type": "error", "query_id": "...", "payload": {"message": "...", "recoverable": bool}}
    {"type": "complete", "query_id": "...", "payload": {}}
    ```
    """
    
    async def generate():
        agent = get_agent()
        
        async for output in agent.run(query):
            if output:  # Filter out empty outputs
                yield json.dumps(output) + "\n"
    
    return StreamingResponse(
        generate(), 
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
