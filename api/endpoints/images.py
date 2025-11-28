"""
Image retrieval endpoint for visual search results.

Serves page images from either local filesystem (ColQwen) or cloud blobs (Jina/Weaviate).
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from api.core.providers import get_visual_search

router = APIRouter()


@router.get("/images/{page_id}")
async def get_image(page_id: int):
    """
    Return the page image bytes for a given page_id.
    """
    visual_search = get_visual_search()
    
    try:
        image_bytes = await visual_search.get_page_image(page_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch image: {exc}") from exc
    
    if not image_bytes:
        raise HTTPException(status_code=404, detail=f"No image found for page_id={page_id}")
    
    return Response(content=image_bytes, media_type="image/png")
