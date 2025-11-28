from typing import Optional, Dict
from pydantic import BaseModel, Field

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
    score: Optional[float] = None
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
    page_hits: Optional[Dict[str, PageHit]] = None  # Key: "{manual_name}:{page_number}"
