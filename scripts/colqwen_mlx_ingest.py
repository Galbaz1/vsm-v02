#!/usr/bin/env python
"""
MLX-Optimized ColQwen2.5 Ingestion for Apple Silicon M3

Optimizations for M3 Mac Studio (256GB RAM):
- MPS with CPU fallback for unsupported ops (>65536 channels)
- Optimized batch processing (8 pages/batch - balanced for MPS+CPU hybrid)
- Parallel image loading (8 workers)
- Unified memory architecture exploitation
- bfloat16 precision for memory efficiency

Note: torch.compile removed due to MPS compatibility issues with vision encoders.
See: https://github.com/pytorch/pytorch/issues/152278

Usage:
    python scripts/colqwen_mlx_ingest.py "Technical Manual"
    python scripts/colqwen_mlx_ingest.py "UK Firmware Manual"
"""

import sys
import os
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List
import time

# Suppress ResourceWarning from colpali_engine temp directories
warnings.filterwarnings("ignore", category=ResourceWarning, message="Implicitly cleaning up")

# CRITICAL: Enable MPS fallback BEFORE importing torch
# This allows unsupported MPS ops (conv with >65536 channels) to fall back to CPU
# Reference: https://github.com/pytorch/pytorch/issues/134416
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from PIL import Image
import weaviate
from weaviate.classes.config import Configure, Property, DataType
import weaviate.classes.config as wc
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# M3-Optimized Constants
COLLECTION_NAME = "PDFDocuments"
# Batch size 8: Balanced for MPS+CPU hybrid execution
# - Too large (32): CPU fallback ops become bottleneck
# - Too small (1-2): Underutilizes GPU parallelism
# - 8 is optimal for vision models with partial CPU fallback
BATCH_SIZE = 8
NUM_WORKERS = 8  # Parallel image loading threads

# Manual name to directory mapping
MANUAL_DIR_MAP = {
    "Technical Manual": "techman",
    "UK Firmware Manual": "uk_firmware",
}


def configure_m3_optimizations():
    """Configure PyTorch for M3 with MPS fallback."""
    print("[M3 Config] Configuring PyTorch for Apple Silicon...")
    print(f"[M3 Config] PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'not set')}")
    
    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    
    print(f"[M3 Config] MPS available: {mps_available}")
    print(f"[M3 Config] MPS built: {mps_built}")
    
    if mps_available:
        # Use high precision for matrix ops
        torch.set_float32_matmul_precision('high')
        print("[M3 Config] Float32 matmul precision: high")
    
    return "mps" if mps_available else "cpu"


def load_image_safely(img_path: Path) -> Image.Image:
    """Load image and convert to RGB to ensure file is closed."""
    with Image.open(img_path) as img:
        # Convert to RGB and return a copy (closes the file handle)
        return img.convert("RGB").copy()


class MLXColQwenIngester:
    """M3-optimized ColQwen ingestion pipeline with MPS fallback."""
    
    def __init__(self, device: str):
        self.device = device
        self.model = None
        self.processor = None
        
    def initialize_model(self):
        """Load ColQwen model for M3 (no torch.compile - incompatible with MPS fallback)."""
        print(f"\n[M3] Loading ColQwen2.5 on {self.device}...")
        start = time.time()
        
        # Load model with bfloat16 for memory efficiency
        self.model = ColQwen2_5.from_pretrained(
            "vidore/colqwen2.5-v0.2",
            dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        
        # NOTE: torch.compile is DISABLED
        # Reason: AOT compilation breaks with MPS fallback for vision encoder's
        # patch embedding (3D convolution with >65536 output channels)
        # Reference: https://github.com/pytorch/pytorch/issues/152278
        
        self.processor = ColQwen2_5_Processor.from_pretrained(
            "vidore/colqwen2.5-v0.2"
        )
        
        elapsed = time.time() - start
        print(f"[M3] Model loaded in {elapsed:.2f}s (torch.compile disabled for MPS compatibility)")
        
    def load_images_parallel(self, manual_name: str) -> List[Image.Image]:
        """Load preview images with parallel processing."""
        if manual_name not in MANUAL_DIR_MAP:
            raise ValueError(
                f"Unknown manual: '{manual_name}'\n"
                f"Available manuals: {list(MANUAL_DIR_MAP.keys())}"
            )
        
        dir_name = MANUAL_DIR_MAP[manual_name]
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
        
        # OPTIMIZATION: Parallel image loading with proper file closing
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            images = list(executor.map(load_image_safely, image_files))
        
        elapsed = time.time() - start
        print(f"[M3] Loaded {len(images)} images in {elapsed:.2f}s ({len(images)/elapsed:.1f} imgs/sec)")
        
        return images
    
    def generate_embeddings_optimized(self, images: List[Image.Image]) -> List:
        """Generate multi-vector embeddings with MPS + CPU fallback."""
        print(f"\n[M3] Generating embeddings for {len(images)} pages")
        print(f"[M3] Batch size: {BATCH_SIZE} (optimized for MPS+CPU hybrid)")
        print(f"[M3] Note: Some ops will fall back to CPU (expected for >65536 channel convolutions)")
        
        start_time = time.time()
        embeddings = []
        
        total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            print(f"[M3] Batch {batch_num}/{total_batches} ({len(batch)} pages)...", end=" ", flush=True)
            batch_start = time.time()
            
            # Process batch
            batch_input = self.processor.process_images(batch).to(self.device)
            
            # Generate embeddings (no autocast - MPS autocast has limited support)
            with torch.no_grad():
                batch_embeddings = self.model(**batch_input)
            
            # Convert and store (bfloat16 -> float32 for numpy compatibility)
            for emb in batch_embeddings:
                embeddings.append(emb.cpu().float().numpy())
            
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
    
    def ensure_collection(self, client: weaviate.WeaviateClient):
        """Ensure Weaviate collection exists with multi-vector config.
        
        Creates collection if it doesn't exist, otherwise reuses existing.
        Uses Weaviate's multi-vector support for ColBERT-style late interaction.
        Reference: https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings
        """
        existing = client.collections.list_all()
        
        if COLLECTION_NAME in existing:
            print(f"\n[Weaviate] Collection {COLLECTION_NAME} already exists (will append)")
            return
        
        print(f"\n[Weaviate] Creating {COLLECTION_NAME} with multi-vector support...")
        
        # Multi-vector configuration for ColQwen embeddings
        # Each page produces ~750 vectors of 128 dimensions (ColBERT-style)
        client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                wc.Property(name="page_id", data_type=wc.DataType.INT),
                wc.Property(name="asset_manual", data_type=wc.DataType.TEXT),
                wc.Property(name="page_number", data_type=wc.DataType.INT),
                wc.Property(name="image_path", data_type=wc.DataType.TEXT,
                           skip_vectorization=True),
            ],
            # Named multi-vector configuration (required for ColBERT/ColQwen)
            # Using multi_vector_config param (fixes Dep027 deprecation warning)
            vector_config=[
                Configure.MultiVectors.self_provided(
                    name="colqwen",
                    multi_vector_config=Configure.VectorIndex.MultiVector.multi_vector(),
                    vector_index_config=Configure.VectorIndex.hnsw()
                )
            ]
        )
        
        print(f"[Weaviate] Collection created with 'colqwen' multi-vector index")
    
    def delete_manual_pages(self, client: weaviate.WeaviateClient, manual_name: str):
        """Delete existing pages for a specific manual before re-ingesting."""
        coll = client.collections.get(COLLECTION_NAME)
        
        # Delete objects where asset_manual matches
        result = coll.data.delete_many(
            where=weaviate.classes.query.Filter.by_property("asset_manual").equal(manual_name)
        )
        
        if result.successful > 0:
            print(f"[Weaviate] Deleted {result.successful} existing pages for '{manual_name}'")
    
    def ingest_to_weaviate(
        self,
        client,
        manual_name: str,
        images: List[Image.Image],
        embeddings: List
    ):
        """Ingest pages with multi-vector embeddings to Weaviate.
        
        Uses named vector format: {"colqwen": [[v1], [v2], ...]}
        as required by Weaviate multi-vector configuration.
        """
        coll = client.collections.get(COLLECTION_NAME)
        
        print(f"\n[Weaviate] Ingesting {len(images)} pages with multi-vector embeddings...")
        start = time.time()
        
        dir_name = MANUAL_DIR_MAP[manual_name]
        
        with coll.batch.dynamic() as batch:
            for page_num, (image, embedding) in enumerate(zip(images, embeddings), start=1):
                # Convert embedding to list format for Weaviate
                # Shape: (seq_len, 128) -> list of lists
                multi_vector = embedding.tolist()
                
                props = {
                    "page_id": page_num,
                    "asset_manual": manual_name,
                    "page_number": page_num,
                    "image_path": f"static/previews/{dir_name}/page-{page_num}.png",
                }
                
                # Named vector format required for multi-vector collections
                batch.add_object(
                    properties=props,
                    vector={"colqwen": multi_vector}
                )
                
                if page_num % 25 == 0:
                    print(f"[Weaviate] Added {page_num}/{len(images)} pages...")
        
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
    print("  ColQwen Ingestion for Apple Silicon M3 (MPS + CPU Fallback)")
    print("="*70)
    
    overall_start = time.time()
    
    # Configure M3 optimizations and get device
    device = configure_m3_optimizations()
    
    # Initialize ingester
    ingester = MLXColQwenIngester(device)
    
    # Load model
    ingester.initialize_model()
    
    # Load images
    images = ingester.load_images_parallel(manual_name)
    
    # Generate embeddings
    embeddings = ingester.generate_embeddings_optimized(images)
    
    # Ingest to Weaviate
    with weaviate.connect_to_local() as client:
        ingester.ensure_collection(client)
        ingester.delete_manual_pages(client, manual_name)  # Remove old pages for this manual
        ingester.ingest_to_weaviate(client, manual_name, images, embeddings)
    
    overall_time = time.time() - overall_start
    
    print("\n" + "="*70)
    print(f"  ✅ {manual_name}: {len(images)} pages ingested in {overall_time:.2f}s")
    print(f"  📊 Performance: {len(images)/overall_time:.2f} pages/sec")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
