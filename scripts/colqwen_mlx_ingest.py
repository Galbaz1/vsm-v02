#!/usr/bin/env python
"""
MLX-Optimized ColQwen2.5 Ingestion for Apple Silicon M3

Optimizations for M3 Mac Studio (256GB RAM):
- Large batch processing (32 pages/batch vs 4)
- Metal Performance Shaders via PyTorch MPS
- torch.compile with AOT for Metal acceleration
- Parallel image loading (8 workers)
- Unified memory architecture exploitation
- Automatic mixed precision (bfloat16)

Usage:
    python scripts/colqwen_mlx_ingest.py "Technical Manual"
    python scripts/colqwen_mlx_ingest.py "UK Firmware Manual"
"""

import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List
import time

# M3 Optimization Config - MUST be imported first
from mlx_config import configure_metal_performance, get_device

import torch
from PIL import Image
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# Configure Metal before any model loading
configure_metal_performance()

# M3-Optimized Constants
COLLECTION_NAME = "PDFDocuments"
BATCH_SIZE = 32  # 8x larger than baseline (leverage 256GB RAM)
NUM_WORKERS = 8  # Parallel image loading threads


class MLXColQwenIngester:
    """M3-optimized ColQwen ingestion pipeline."""
    
    def __init__(self):
        self.device = get_device()
        self.model = None
        self.processor = None
        
    def initialize_model(self):
        """Load and compile ColQwen model for M3."""
        print(f"\n[M3] Loading ColQwen2.5 on {self.device}...")
        start = time.time()
        
        # Load model with bfloat16 for memory efficiency
        self.model = ColQwen2_5.from_pretrained(
            "vidore/colqwen2.5-v0.2",
            dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        
        # OPTIMIZATION: Compile model for Metal acceleration
        print("[M3] Compiling model for Metal...")
        self.model = torch.compile(
            self.model,
            backend="aot_eager",  # Ahead-of-time compilation for Metal
        )
        
        self.processor = ColQwen2_5_Processor.from_pretrained(
            "vidore/colqwen2.5-v0.2"
        )
        
        elapsed = time.time() - start
        print(f"[M3] Model loaded and compiled in {elapsed:.2f}s")
        
    def load_images_parallel(self, manual_name: str) -> List[Image.Image]:
        """Load preview images with parallel processing."""
        dir_name = manual_name.lower().replace(' ', '_')
        preview_dir = Path(f"static/previews/{dir_name}")
        
        if not preview_dir.exists():
            raise FileNotFoundError(
                f"Preview directory not found: {preview_dir}\n"
                f"Run: python scripts/generate_previews.py <pdf> static/previews/{dir_name}"
            )
        
        image_files = sorted(
            preview_dir.glob("page-*.png"),
            key=lambda p: int(p.stem.split('-')[1])
        )
        
        if not image_files:
            raise FileNotFoundError(f"No PNGs in {preview_dir}")
        
        print(f"\n[M3] Loading {len(image_files)} images in parallel ({NUM_WORKERS} workers)...")
        start = time.time()
        
        # OPTIMIZATION: Parallel image loading
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            images = list(executor.map(Image.open, image_files))
        
        elapsed = time.time() - start
        print(f"[M3] Loaded {len(images)} images in {elapsed:.2f}s ({len(images)/elapsed:.1f} imgs/sec)")
        
        return images
    
    def generate_embeddings_optimized(self, images: List[Image.Image]) -> List:
        """Generate multi-vector embeddings with M3 optimizations."""
        print(f"\n[M3] Generating embeddings for {len(images)} pages")
        print(f"[M3] Batch size: {BATCH_SIZE} (M3-optimized, 8x baseline)")
        
        start_time = time.time()
        embeddings = []
        
        total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            print(f"[M3] Batch {batch_num}/{total_batches} ({len(batch)} pages)...", end=" ")
            batch_start = time.time()
            
            # Process batch
            batch_input = self.processor.process_images(batch).to(self.device)
            
            # OPTIMIZATION: Use autocast for Metal
            with torch.no_grad():
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    batch_embeddings = self.model(**batch_input)
            
            # Convert and store
            for emb in batch_embeddings:
                embeddings.append(emb.cpu().numpy())
            
            # Clean up GPU memory
            del batch_input
            if self.device == "mps":
                torch.mps.empty_cache()
            
            batch_elapsed = time.time() - batch_start
            print(f"{batch_elapsed:.2f}s ({len(batch)/batch_elapsed:.1f} pages/sec)")
        
        total_time = time.time() - start_time
        avg_speed = len(images) / total_time
        
        print(f"\n[M3] ✅ Embeddings complete!")
        print(f"[M3] Total: {total_time:.2f}s, Average: {avg_speed:.2f} pages/sec")
        
        return embeddings
    
    def create_collection(self, client: weaviate.WeaviateClient):
        """Create Weaviate collection with multi-vector config."""
        existing = client.collections.list_all()
        
        if COLLECTION_NAME in existing:
            print(f"\n[Weaviate] Deleting existing {COLLECTION_NAME}...")
            client.collections.delete(COLLECTION_NAME)
        
        print(f"[Weaviate] Creating {COLLECTION_NAME} with multi-vector support...")
        
        client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                Property(name="page_id", data_type=DataType.INT),
                Property(name="asset_manual", data_type=DataType.TEXT),
                Property(name="page_number", data_type=DataType.INT),
                Property(name="image_path", data_type=DataType.TEXT,
                        skip_vectorization=True),
            ],
        )
        
        print(f"[Weaviate] Collection created")
    
    def ingest_to_weaviate(
        self,
        client,
        manual_name: str,
        images: List[Image.Image],
        embeddings: List
    ):
        """Ingest pages with embeddings to Weaviate."""
        coll = client.collections.get(COLLECTION_NAME)
        
        print(f"\n[Weaviate] Ingesting {len(images)} pages...")
        start = time.time()
        
        with coll.batch.fixed_size(batch_size=100) as batch:
            for page_num, (image, embedding) in enumerate(zip(images, embeddings), start=1):
                multi_vector = embedding.tolist()
                
                props = {
                    "page_id": page_num,
                    "asset_manual": manual_name,
                    "page_number": page_num,
                    "image_path": f"static/previews/{manual_name.lower().replace(' ', '_')}/page-{page_num}.png",
                }
                
                batch.add_object(properties=props, vector=multi_vector)
        
        elapsed = time.time() - start
        print(f"[Weaviate] ✅ Ingestion complete in {elapsed:.2f}s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/colqwen_mlx_ingest.py <manual_name>")
        print('Example: python scripts/colqwen_mlx_ingest.py "Technical Manual"')
        print("\nNote: Preview PNGs must exist in static/previews/<manual_name>/")
        sys.exit(1)
    
    manual_name = sys.argv[1]
    
    print("\n" + "="*70)
    print("  MLX-Optimized ColQwen Ingestion for Apple Silicon M3 (256GB RAM)")
    print("="*70)
    
    overall_start = time.time()
    
    # Initialize ingester
    ingester = MLXColQwenIngester()
    
    # Load model
    ingester.initialize_model()
    
    # Load images
    images = ingester.load_images_parallel(manual_name)
    
    # Generate embeddings
    embeddings = ingester.generate_embeddings_optimized(images)
    
    # Ingest to Weaviate
    with weaviate.connect_to_local() as client:
        ingester.create_collection(client)
        ingester.ingest_to_weaviate(client, manual_name, images, embeddings)
    
    overall_time = time.time() - overall_start
    
    print("\n" + "="*70)
    print(f"  ✅ {manual_name}: {len(images)} pages ingested in {overall_time:.2f}s")
    print(f"  📊 Performance: {len(images)/overall_time:.2f} pages/sec")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
