#!/usr/bin/env python
"""
Cloud Ingestion Script for VSM v0.3.

Ingests technical manuals into Weaviate Cloud for Hybrid (Text + Visual) RAG.

Capabilities:
1. Text Ingestion:
   - Parses LandingAI JSON output (markdown with page breaks)
   - Chunks text by page and section
   - Generates embeddings using Jina Embeddings v4 (via Provider)
   - Upserts to 'AssetManual' collection in Weaviate Cloud

2. Visual Ingestion:
   - Converts PDF pages to images (using pdf2image)
   - Ingests images into 'PDFDocuments' collection
   - Uses Jina CLIP v2 (via Weaviate native integration) for visual embeddings

Usage:
    export VSM_MODE=cloud
    export WEAVIATE_URL=...
    export WEAVIATE_API_KEY=...
    export JINA_API_KEY=...
    
    python scripts/cloud_ingest.py \
        --json data/output_techman.json \
        --pdf data/techman.pdf \
        --manual "ThorGuard TechMan"
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pdf2image import convert_from_path
from tqdm import tqdm

from api.core.config import get_settings
from api.core.providers import get_embeddings, get_vectordb, get_visual_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
PAGE_BREAK_MARKER = "<!-- PAGE BREAK -->"
ANCHOR_RE = re.compile(r"<a id='[^']+'></a>")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Clean markdown/HTML artifacts from text."""
    text = ANCHOR_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    # Remove multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_landingai_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Parse LandingAI JSON output into page chunks.
    
    Handles the markdown format with page breaks.
    Returns list of dicts: {"page_number": int, "content": str}
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    markdown = data.get("markdown", "")
    if not markdown:
        # Fallback to checking 'chunks' if markdown is empty
        # (Legacy support or different output format)
        chunks = data.get("chunks", [])
        if chunks:
            logger.info(f"Found {len(chunks)} pre-defined chunks")
            return [
                {
                    "page_number": c.get("grounding", {}).get("page") or 1,
                    "content": clean_text(c.get("text") or c.get("markdown") or ""),
                    "chunk_type": c.get("chunk_type", "text")
                }
                for c in chunks
            ]
        raise ValueError("No 'markdown' or 'chunks' found in JSON")
    
    # Split by page break
    raw_pages = markdown.split(PAGE_BREAK_MARKER)
    parsed_pages = []
    
    for i, raw_page in enumerate(raw_pages):
        content = clean_text(raw_page)
        if not content:
            continue
            
        parsed_pages.append({
            "page_number": i + 1,  # 1-based index
            "content": content,
            "chunk_type": "text"  # Default for now
        })
        
    logger.info(f"Parsed {len(parsed_pages)} pages from markdown")
    return parsed_pages


def ensure_asset_manual_collection():
    """Ensure AssetManual collection exists with correct schema."""
    client = get_vectordb().connect()
    
    if client.collections.exists("AssetManual"):
        logger.info("Collection AssetManual already exists")
        return

    from weaviate.classes.config import Configure, Property, DataType
    
    client.collections.create(
        name="AssetManual",
        vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
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
    logger.info("Created collection AssetManual")


async def ingest_text(
    pages: List[Dict[str, Any]],
    manual_name: str,
    batch_size: int = 50
):
    """
    Ingest text chunks into AssetManual collection.
    """
    # Ensure collection exists
    ensure_asset_manual_collection()
    
    logger.info(f"Starting text ingestion for {manual_name}...")
    
    embeddings_provider = get_embeddings()
    vectordb_provider = get_vectordb()
    
    total_pages = len(pages)
    
    for i in range(0, total_pages, batch_size):
        batch = pages[i : i + batch_size]
        
        # Prepare texts for embedding
        texts = [p["content"] for p in batch]
        
        try:
            # Generate embeddings (Jina v4)
            vectors = await embeddings_provider.embed_texts(texts, task="retrieval.passage")
            
            # Prepare objects for upsert
            objects_to_upsert = []
            for page, vector in zip(batch, vectors):
                obj = {
                    "properties": {
                        "manual_name": manual_name,
                        "content": page["content"],
                        "page_number": page["page_number"],
                        "chunk_type": page["chunk_type"],
                        "content_hash": str(hash(page["content"])),
                    },
                    "vector": vector,
                }
                objects_to_upsert.append(obj)
            
            # Upsert to Weaviate
            await vectordb_provider.batch_upsert("AssetManual", objects_to_upsert)
            
            logger.info(f"Ingested text batch {i // batch_size + 1}/{(total_pages // batch_size) + 1}")
            
        except Exception as e:
            logger.error(f"Failed to ingest text batch: {e}")
            # Continue to next batch? Or raise?
            # raise e


async def ingest_visuals(
    pdf_path: str,
    manual_name: str,
    start_page: int = 1
):
    """
    Ingest PDF pages as images into PDFDocuments collection.
    """
    logger.info(f"Starting visual ingestion for {manual_name} from {pdf_path}...")
    
    visual_search = get_visual_search()
    
    # Ensure collection exists (create if needed)
    if hasattr(visual_search, "ensure_collection_exists"):
        await visual_search.ensure_collection_exists()
    
    # Convert PDF to images
    # This might be slow/memory intensive for large PDFs, consider batching if needed
    try:
        images = await asyncio.to_thread(convert_from_path, pdf_path)
        logger.info(f"Converted PDF to {len(images)} images")
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        raise
    
    # Temporary directory for images (VisualSearchProvider expects paths currently)
    # Wait, JinaVisualSearch.ingest_page reads from path.
    # We should save them temporarily.
    temp_dir = Path("temp_ingest_images")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        for i, image in enumerate(tqdm(images)):
            page_num = start_page + i
            image_filename = f"page_{page_num}.png"
            image_path = temp_dir / image_filename
            
            # Save image
            await asyncio.to_thread(image.save, image_path, "PNG")
            
            # Deterministic page ID generation
            unique_str = f"{manual_name}_{page_num}"
            # Use first 12 chars of MD5 hash converted to int (fits in standard int64)
            page_id = int(hashlib.md5(unique_str.encode()).hexdigest()[:12], 16)
            
            # Ingest with retry logic for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await visual_search.ingest_page(
                        page_id=page_id,
                        asset_manual=manual_name,
                        page_number=page_num,
                        image_path=str(image_path),
                    )
                    logger.info(f"Ingested page {page_num} from {manual_name} to cloud")
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                        logger.warning(f"Rate limited on page {page_num}, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to ingest page {page_num}: {e}")
                        break
            
            # Cleanup immediately
            await asyncio.to_thread(image_path.unlink)
            
            # Rate limit: 3s delay between pages to avoid Jina API throttling
            await asyncio.sleep(3.0)
            
    finally:
        # Cleanup dir
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except:
                pass


async def main():
    parser = argparse.ArgumentParser(description="Cloud Ingestion for VSM")
    parser.add_argument("--json", required=True, help="Path to LandingAI parsed JSON")
    parser.add_argument("--pdf", required=True, help="Path to original PDF")
    parser.add_argument("--manual", required=True, help="Manual name (e.g. 'TechMan')")
    parser.add_argument("--skip-text", action="store_true", help="Skip text ingestion")
    parser.add_argument("--skip-visual", action="store_true", help="Skip visual ingestion")
    
    args = parser.parse_args()
    
    # Verify mode
    settings = get_settings()
    if settings.vsm_mode != "cloud":
        logger.warning("VSM_MODE is not 'cloud'. Ingestion might fail or use local providers.")
        # Allow continuing if user knows what they are doing (e.g. hybrid testing)
    
    if not args.skip_text:
        pages = parse_landingai_json(args.json)
        await ingest_text(pages, args.manual)
    
    if not args.skip_visual:
        await ingest_visuals(args.pdf, args.manual)
        
    logger.info("Ingestion complete!")


if __name__ == "__main__":
    asyncio.run(main())
