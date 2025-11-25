import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import weaviate
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl
from weaviate.classes.query import Filter


COLLECTION_NAME = "AssetManual"
DEFAULT_LIMIT = 5
MAX_LIMIT = 20


class Settings(BaseModel):
    pdf_base_url: str = "/static/manuals"
    preview_base_url: str = "/static/previews"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        pdf_base_url=os.getenv("PDF_BASE_URL", "/static/manuals"),
        preview_base_url=os.getenv("PREVIEW_BASE_URL", "/static/previews"),
    )


class BoundingBox(BaseModel):
    left: float = Field(ge=0.0, le=1.0)
    top: float = Field(ge=0.0, le=1.0)
    right: float = Field(ge=0.0, le=1.0)
    bottom: float = Field(ge=0.0, le=1.0)


class SearchHit(BaseModel):
    anchor_id: str
    manual_name: str
    content: str
    page_number: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    pdf_page_url: str
    page_image_url: Optional[str] = None
    chunk_type: Optional[str] = None
    section_title: Optional[str] = None
    content_hash: Optional[str] = None


class PageHit(BaseModel):
    """Aggregated hits for a single page"""
    page_number: int
    manual_name: str
    hits: list[SearchHit]
    bboxes: list[BoundingBox]


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    page_hits: Optional[dict[str, PageHit]] = None  # Key: "{manual_name}:{page_number}"


app = FastAPI(
    title="Manual Search API",
    version="0.1.0",
    description="Semantic search API over ADE-parsed manuals stored in Weaviate.",
)

cors_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


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


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
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
    try:
        with weaviate.connect_to_local() as client:
            coll = client.collections.use(COLLECTION_NAME)

            from weaviate.classes.query import Filter

            filters = None
            if chunk_type:
                filters = Filter.by_property("chunk_type").equal(chunk_type)

            result = coll.query.near_text(
                query=query,
                limit=limit * 2 if group_by_page else limit,
                filters=filters,
            )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Weaviate error: {exc}") from exc

    hits: list[SearchHit] = []
    seen_hashes: dict[str, set[str]] = {}  # {(manual, page): {content_hash}}
    
    for obj in result.objects:
        props = obj.properties or {}
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
            anchor_id=props.get("anchor_id", obj.uuid),
            manual_name=manual_name,
            content=props.get("content", ""),
            page_number=page_number,
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
    page_hits: dict[str, PageHit] = {}
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

    return SearchResponse(query=query, hits=hits, page_hits=page_hits if page_hits else None)

