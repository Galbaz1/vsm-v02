from typing import Optional
from fastapi import APIRouter, Query

from api.schemas.search import SearchResponse
from api.services.search import perform_search

router = APIRouter()

DEFAULT_LIMIT = 5
MAX_LIMIT = 20

@router.get("/search", response_model=SearchResponse)
def search_manual(
    query: str = Query(..., min_length=3, description="Natural language search query."),
    limit: int = Query(
        DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description="Maximum number of results to return.",
    ),
    chunk_type: Optional[str] = Query(
        None,
        description="Filter results by chunk type (e.g., 'table', 'figure', 'text').",
    ),
    group_by_page: bool = Query(
        False,
        description="Group results by page and deduplicate by content_hash.",
    ),
) -> SearchResponse:
    hits, page_hits = perform_search(query, limit, chunk_type, group_by_page)
    return SearchResponse(query=query, hits=hits, page_hits=page_hits)
