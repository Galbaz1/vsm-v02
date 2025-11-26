#!/usr/bin/env python
"""
weaviate_ingest_manual.py

Usage:
    python weaviate_ingest_manual.py parsed_manual.json "My Freezer Asset Manual"

- Reads JSON output from parse_with_landingai.py (which contains ADE chunks).
- Stores rich chunks (text + metadata like page number, bounding box) in Weaviate.
"""

import sys
import json
import os
import re
import hashlib
import html
from typing import Optional

import weaviate
from weaviate.classes.config import Configure, Property, DataType


COLLECTION_NAME = "AssetManual"


ANCHOR_TAG_RE = re.compile(r"<a\s+id=['\"][^'\"]*['\"][^>]*></a>\s*", re.IGNORECASE)
PAGE_BREAK_RE = re.compile(r"<!--\s*PAGE BREAK\s*-->", re.IGNORECASE)
FLOWCHART_BLOCK_RE = re.compile(r"<::[^>]*::>", re.IGNORECASE | re.DOTALL)
TABLE_TAG_RE = re.compile(r"<table[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)
HEADING_RE = re.compile(r"^#+\s+(.+)$", re.MULTILINE)


def extract_section_title(text: str) -> Optional[str]:
    """
    Extract section title from markdown headings (## 1.1 Title).
    Returns the first heading found, or None.
    """
    if not text:
        return None
    
    # Look for markdown headings
    match = HEADING_RE.search(text)
    if match:
        title = match.group(1).strip()
        # Clean up any remaining HTML/markdown artifacts
        title = HTML_TAG_RE.sub("", title)
        title = html.unescape(title)
        return title.strip() if title else None
    
    return None


def clean_content(text: str) -> str:
    """
    Remove ADE markup (anchor tags, page breaks, flowchart blocks, tables),
    decode HTML entities, and return clean prose.
    """
    if not text:
        return ""
    
    cleaned = text
    
    # Remove anchor tags
    cleaned = ANCHOR_TAG_RE.sub("", cleaned)
    
    # Remove page break markers
    cleaned = PAGE_BREAK_RE.sub("\n", cleaned)
    
    # Remove flowchart blocks (<::...::>)
    cleaned = FLOWCHART_BLOCK_RE.sub("", cleaned)
    
    # For table chunks, extract readable text from cells
    if "<table" in cleaned.lower():
        # Remove opening/closing table tags but KEEP content
        cleaned = re.sub(r"<table[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</table>", "", cleaned, flags=re.IGNORECASE)
        # Convert table structure to readable text
        cleaned = re.sub(r"<tr[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</tr>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<t[dh][^>]*>", " | ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</t[dh]>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<thead[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</thead>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<tbody[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</tbody>", "", cleaned, flags=re.IGNORECASE)
    
    # Remove all remaining HTML tags
    cleaned = HTML_TAG_RE.sub("", cleaned)
    
    # Decode HTML entities
    cleaned = html.unescape(cleaned)
    
    # Clean up whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    
    return cleaned.strip()


def extract_grounding(grounding):
    """
    ADE may return grounding as a dict or list of dicts. Normalize it
    to a (page, bbox) tuple.
    """
    if not grounding:
        return None, None

    if isinstance(grounding, dict):
        return grounding.get("page"), grounding.get("box")

    if isinstance(grounding, list) and len(grounding) > 0:
        first = grounding[0]
        if isinstance(first, dict):
            return first.get("page"), first.get("box")

    return None, None


def compute_content_hash(content: str) -> str:
    """
    Compute a normalized hash of content for deduplication.
    Normalizes by lowercasing and stripping whitespace.
    """
    normalized = content.lower().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_chunks_from_json(json_path: str):
    """
    Load LandingAI ADE chunks from the saved JSON.
    Returns a list of dicts ready for Weaviate ingestion.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ADE JSON structure usually has a "chunks" key
    # Each chunk has: id, text (or markdown), chunk_type, grounding (page, box)
    raw_chunks = data.get("chunks", [])
    
    normalized_chunks = []
    for ch in raw_chunks:
        # Extract text content
        raw_content = ch.get("text") or ch.get("markdown") or ""
        content = clean_content(raw_content)
        if not content.strip():
            continue

        # Extract metadata
        chunk_id = ch.get("chunk_id") or ch.get("id")
        chunk_type = ch.get("chunk_type") or ch.get("type") or "text"
        
        # Extract section title from raw content (before cleaning)
        section_title = extract_section_title(raw_content)

        grounding = ch.get("grounding")
        page_num, bbox = extract_grounding(grounding)
        
        # Compute content hash for deduplication
        content_hash = compute_content_hash(content)

        normalized_chunks.append({
            "anchor_id": chunk_id,
            "content": content,
            "chunk_type": chunk_type,
            "page_number": page_num,
            "bbox": json.dumps(bbox) if bbox else None,
            "section_title": section_title,
            "content_hash": content_hash,
        })

    return normalized_chunks


def ensure_collection(client: weaviate.WeaviateClient):
    """
    Ensure the AssetManual collection exists with appropriate schema.
    Creates collection only if it doesn't exist (preserves existing data).
    """
    existing = client.collections.list_all()
    if COLLECTION_NAME in existing:
        print(f"[Weaviate] Collection {COLLECTION_NAME} already exists, using it...")
        return client.collections.get(COLLECTION_NAME)

    print(f"[Weaviate] Creating collection {COLLECTION_NAME}...")
    coll = client.collections.create(
        name=COLLECTION_NAME,
        vector_config=Configure.Vectors.text2vec_ollama(
            # Connect to Native Ollama on host machine
            api_endpoint="http://host.docker.internal:11434",
            model="bge-m3",  # Better retrieval for technical documentation
        ),
        properties=[
            Property(name="manual_name", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="anchor_id", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="chunk_type", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="bbox", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="section_title", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="content_hash", data_type=DataType.TEXT, skip_vectorization=True),
        ]
    )
    return coll


def main():
    if len(sys.argv) < 3:
        print("Usage: python weaviate_ingest_manual.py parsed_manual.json \"Manual Name\"")
        sys.exit(1)

    json_path = sys.argv[1]
    manual_name = sys.argv[2]

    if not os.path.isfile(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print(f"[Weaviate] Loading chunks from {json_path}...")
    chunks = load_chunks_from_json(json_path)
    print(f"[Weaviate] Loaded {len(chunks)} chunks.")

    with weaviate.connect_to_local() as client:
        coll = ensure_collection(client)
        
        print(f"[Weaviate] Ingesting chunks into collection {COLLECTION_NAME}...")
        count = 0
        # Single-threaded, small batches to prevent Ollama EOF errors
        with coll.batch.fixed_size(batch_size=20, concurrent_requests=1) as batch:
            for ch in chunks:
                props = {
                    "manual_name": manual_name,
                    "content": ch["content"],
                    "anchor_id": ch["anchor_id"],
                    "chunk_type": ch["chunk_type"],
                    "page_number": ch["page_number"],
                    "bbox": ch["bbox"],
                    "section_title": ch.get("section_title"),
                    "content_hash": ch["content_hash"],
                }
                batch.add_object(properties=props)
                count += 1

        print(f"[Weaviate] Ingested {count} objects into {COLLECTION_NAME}.")


if __name__ == "__main__":
    main()
