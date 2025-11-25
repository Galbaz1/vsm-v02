"""
Agentic search endpoint with progressive results streaming.
"""

import asyncio
import json
from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from api.services.agent import get_agent, SearchDecision
from api.services.search import perform_search
from api.services.colqwen import get_colqwen_retriever

router = APIRouter()

@router.get("/agentic_search")
async def agentic_search(
    query: str = Query(..., min_length=3, description="User's search query"),
    limit: int = Query(5, ge=1, le=20, description=" Max results for vector search"),
    top_k: int = Query(3, ge=1, le=10, description="Max results for ColQwen"),
):
    """
    Agentic search that intelligently decides which retrieval method(s) to use
    and streams progressive results to the client.
    
    Response format (NDJSON):
    - {"type": "decision", "data": {...}}  # Agent's search decision
    - {"type": "fast_results", "data": [...]}  # Fast vector search results
    - {"type": "colqwen_results", "data": [...]}  # ColQwen results
    - {"type": "answer_token", "data": "..."} # Streaming answer tokens
    - {"type": "complete", "data": {}}  # Signals completion
    """
    
    async def generate():
        agent = get_agent()
        
        # 1. Agent analyzes query and decides strategy
        decision: SearchDecision = await agent.analyze_query(query)
        
        yield json.dumps({
            "type": "decision",
            "data": {
                "strategy": decision.strategy,
                "reasoning": decision.reasoning,
                "use_fast_vector": decision.use_fast_vector,
                "use_colqwen": decision.use_colqwen,
            }
        }) + "\n"
        
        fast_results = None
        colqwen_results = None
        
        # 2. Execute retrieval based on strategy
        if decision.strategy == "fast_only":
            # Fast vector search only
            fast_results, page_hits = perform_search(query, limit, None, False)
            yield json.dumps({
                "type": "fast_results",
                "data": [hit.dict() for hit in fast_results]
            }) + "\n"
        
        elif decision.strategy == "colqwen_only":
            # ColQwen only
            retriever = get_colqwen_retriever()
            colqwen_results = retriever.retrieve(query, top_k)
            yield json.dumps({
                "type": "colqwen_results",
                "data": colqwen_results
            }) + "\n"
        
        elif decision.strategy == "fast_then_colqwen":
            # Fast first, then ColQwen in background
            fast_results, page_hits = perform_search(query, limit, None, False)
            yield json.dumps({
                "type": "fast_results",
                "data": [hit.dict() for hit in fast_results]
            }) + "\n"
            
            # Start agent answering with fast results
            async for chunk in agent.stream_response(query, fast_results=fast_results):
                if chunk["type"] == "token":
                    yield json.dumps({
                        "type": "answer_token",
                        "data": chunk["content"]
                    }) + "\n"
            
            # Now fetch ColQwen results
            retriever = get_colqwen_retriever()
            colqwen_results = retriever.retrieve(query, top_k)
            yield json.dumps({
                "type": "colqwen_results",
                "data": colqwen_results
            }) + "\n"
            
            # Agent refines answer with visual context
            async for chunk in agent.stream_response(query, colqwen_results=colqwen_results):
                if chunk["type"] == "token":
                    yield json.dumps({
                        "type": "answer_refinement",
                        "data": chunk["content"]
                    }) + "\n"
        
        elif decision.strategy == "both_parallel":
            # Run both in parallel
            fast_task = asyncio.create_task(
                asyncio.to_thread(perform_search, query, limit, None, False)
            )
            
            retriever = get_colqwen_retriever()
            colqwen_task = asyncio.create_task(
                asyncio.to_thread(retriever.retrieve, query, top_k)
            )
            
            # Yield results as they complete
            fast_results, page_hits = await fast_task
            yield json.dumps({
                "type": "fast_results",
                "data": [hit.dict() for hit in fast_results]
            }) + "\n"
            
            colqwen_results = await colqwen_task
            yield json.dumps({
                "type": "colqwen_results",
                "data": colqwen_results
            }) + "\n"
        
        # 3. Signal completion
        yield json.dumps({"type": "complete", "data": {}}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")
