// TypeScript types matching FastAPI response models

export interface BoundingBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface SearchHit {
  anchor_id: string;
  manual_name: string;
  content: string;
  page_number: number | null;
  bbox: BoundingBox | null;
  pdf_page_url: string;
  page_image_url: string | null;
  chunk_type?: string | null;
  section_title?: string | null;
  content_hash?: string | null;
}

export interface PageHit {
  page_number: number;
  manual_name: string;
  hits: SearchHit[];
  bboxes: BoundingBox[];
}

export interface SearchResponse {
  query: string;
  hits: SearchHit[];
  page_hits?: Record<string, PageHit> | null;
}

export interface SearchQueryParams {
  query: string;
  limit?: number;
  chunk_type?: string;
  group_by_page?: boolean;
}

