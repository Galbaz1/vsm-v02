import json
from urllib.parse import quote
from typing import Optional, List, Dict, Set
from fastapi import HTTPException

from api.core.config import get_settings
from api.core.providers import get_embeddings, get_vectordb
from api.schemas.search import SearchHit, PageHit, BoundingBox

COLLECTION_NAME = "AssetManual"

def _slugify_manual(manual_name: str) -> str:
    return quote(manual_name.strip().replace(" ", "_").lower())

def _build_pdf_url(manual_name: str, page: Optional[int]) -> str:
    settings = get_settings()
    slug = _slugify_manual(manual_name)
    page_number = page or 1
    return f"{settings.pdf_base_url}/{slug}.pdf#page={page_number}"

def _build_preview_url(manual_name: str, page: Optional[int]) -> Optional[str]:
    settings = get_settings()
    slug = _slugify_manual(manual_name)
    # Remove "_manual" suffix if present to match preview folder structure
    if slug.endswith("_manual"):
        slug = slug[:-7]  # Remove "_manual"
    page_number = page or 1
    return f"{settings.preview_base_url}/{slug}/page-{page_number}.png"

def _parse_bbox(raw: Optional[object]) -> Optional[BoundingBox]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return BoundingBox(**raw)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return BoundingBox(**data)
        except json.JSONDecodeError:
            return None
    return None

async def perform_search(query: str, limit: int, chunk_type: Optional[str], group_by_page: bool) -> tuple[List[SearchHit], Optional[Dict[str, PageHit]]]:
    """
    Perform text search using the configured VectorDB provider.
    
    Uses hybrid search (vector + keyword).
    """
    try:
        # 1. Generate embedding for the query
        embeddings = get_embeddings()
        query_vector = await embeddings.embed_query(query)
        
        # 2. Build filters
        filters = None
        if chunk_type:
            filters = {"chunk_type": chunk_type}
            
        # 3. Execute search via provider
        vectordb = get_vectordb()
        results = await vectordb.hybrid_search(
            collection=COLLECTION_NAME,
            query=query,
            query_vector=query_vector,
            limit=limit * 2 if group_by_page else limit,
            alpha=0.5,
            filters=filters,
        )
            
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search provider error: {exc}") from exc

    hits: List[SearchHit] = []
    seen_hashes: Dict[str, Set[str]] = {}  # {(manual, page): {content_hash}}
    
    for obj in results:
        # Provider returns dict with "properties", "id", "score"
        props = obj.get("properties", {})
        uuid = obj.get("id")
        score = obj.get("score")
        
        manual_name = props.get("manual_name") or "Unknown Manual"
        page_number = props.get("page_number")
        bbox = _parse_bbox(props.get("bbox"))
        pdf_url = _build_pdf_url(manual_name, page_number)
        preview_url = _build_preview_url(manual_name, page_number)
        content_hash = props.get("content_hash")
        
        # Deduplication logic
        if group_by_page and content_hash and page_number is not None:
            key = f"{manual_name}:{page_number}"
            if key not in seen_hashes:
                seen_hashes[key] = set()
            
            if content_hash in seen_hashes[key]:
                continue  # Skip duplicate
            
            seen_hashes[key].add(content_hash)
        
        hit = SearchHit(
            anchor_id=props.get("anchor_id") or uuid or str(hash(props.get("content", "")))[:16],
            manual_name=manual_name,
            content=props.get("content", ""),
            page_number=page_number,
            score=score,
            bbox=bbox,
            pdf_page_url=pdf_url,
            page_image_url=preview_url,
            chunk_type=props.get("chunk_type"),
            section_title=props.get("section_title"),
            content_hash=content_hash,
        )
        hits.append(hit)
        
        # Limit results after deduplication
        if len(hits) >= limit:
            break
    
    # Build page_hits structure for preview modal
    page_hits: Dict[str, PageHit] = {}
    for hit in hits:
        if hit.page_number is not None:
            key = f"{hit.manual_name}:{hit.page_number}"
            if key not in page_hits:
                page_hits[key] = PageHit(
                    page_number=hit.page_number,
                    manual_name=hit.manual_name,
                    hits=[],
                    bboxes=[],
                )
            page_hits[key].hits.append(hit)
            if hit.bbox:
                page_hits[key].bboxes.append(hit.bbox)

    return hits, page_hits if page_hits else None
