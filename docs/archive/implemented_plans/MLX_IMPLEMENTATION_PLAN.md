# MLX Implementation Plan for ColQwen on M3 Mac Studio

**Decision:** MLX-Optimized PyTorch Implementation  
**Date:** 2025-11-25  
**Hardware:** M3 Mac Studio, 256GB RAM

---

## Executive Summary

**Reality Check:** ColQwen2.5 doesn't have native MLX support (November 2025).

**Our Strategy:**  
Use **MLX-optimized PyTorch** with maximum M3 utilization:
- PyTorch 2.5.1 with Metal Performance Shaders
- Batch size: **32** (8x baseline)
- torch.compile for Metal acceleration
- Parallel processing throughout

**Expected Performance:**
- **5-8x faster** than baseline (15 min → 2-3 min per 130 pages)
- **Native M3 Metal acceleration**
- **256GB RAM fully utilized**

---

## What We Learned from Research

✅ **MLX-VLM exists** - supports Qwen2.5-VL  
❌ **ColQwen-MLX** - does NOT exist yet  
✅ **Weaviate + multi-vector** - works perfectly  
✅ **PyTorch MPS** - mature on M3  
✅ **torch.compile** - gives 20-30% boost  

**Conclusion:** Hybrid approach is optimal for November 2025.

---

## Implementation Plan

### Step 1: Update Dependencies

```txt
# requirements.txt
torch==2.5.1  # NOT 2.6.0 (has MPS bugs)
transformers>=4.45.0
colpali-engine>=0.3.0
mlx>=0.18.0  # For future optimizations
accelerate  # Auto device mapping
```

### Step 2: Create M3 Configuration

Create `scripts/mlx_config.py`:
- Configure Metal Performance Shaders
- Set memory fractions (80% of 256GB)
- Optimize for unified memory

### Step 3: Rewrite Ingestion Script

Create `scripts/colqwen_mlx_ingest.py`:
- Batch size: 32 (up from 4)
- Parallel image loading (8 workers)
- torch.compile with AOT mode
- Automatic mixed precision
- Explicit memory management

---

## Key Optimizations

### 1. Batch Processing
```python
# Before: 4 pages/batch
BATCH_SIZE = 4

# After: 32 pages/batch (8x larger!)
BATCH_SIZE = 32  # Leverage 256GB RAM
```

### 2. Model Compilation
```python
model = torch.compile(
    model,
    backend="aot_eager",  # Metal-optimized
    mode="max-autotune"   # Maximum performance
)
```

### 3. Parallel Image Loading
```python
with ThreadPoolExecutor(max_workers=8) as executor:
    images = list(executor.map(Image.open, image_files))
```

### 4. Memory Management
```python
# Clean GPU memory after each batch
torch.mps.empty_cache()
```

---

## Performance Expectations

| Metric | Baseline | M3-Optimized | Improvement |
|--------|----------|--------------|-------------|
| Batch size | 4 | 32 | 8x |
| torch.compile | No | Yes | +25% |
| Parallel loading | No | Yes (8 workers) | +40% |
| **Total speedup** | 1x | **~6-8x** | 🚀 |

**Time Estimates:**
- Technical Manual (132 pages): 15 min → **~2 min**
- UK Firmware (128 pages): 14 min → **~2 min**
- **Both manuals:** ~30 min → **~4 min**

---

## Implementation Files

```
scripts/
├── mlx_config.py              # NEW: M3 optimization config
├── colqwen_mlx_ingest.py      # NEW: Optimized ingestion
├── colqwen_ingest.py          # OLD: Keep for comparison
└── generate_previews.py       # Unchanged
```

---

## Next Steps

1. **Create `mlx_config.py`** with Metal optimizations
2. **Create `colqwen_mlx_ingest.py`** with all optimizations
3. **Update `requirements.txt`** with pinned versions
4. **Run benchmark** on small test
5. **Ingest both manuals** and measure performance
6. **Document results** in walkthrough

---

## Future: Native MLX Migration

When `colqwen2.5-mlx` becomes available:

1. **Check Hugging Face** for `vidore/colqwen2.5-mlx`
2. **Evaluate performance** vs. our optimized PyTorch
3. **Migrate if** >2x additional speedup
4. **Keep current** until clear benefit

**Monitor:** GitHub issues on colpali-engine and mlx-vlm repos

---

## Success Metrics

✅ Ingestion < 3 min per 130-page PDF  
✅ Memory usage < 40GB peak  
✅ 100% pages successfully ingested  
✅ Embedding quality matches baseline  

---

**Ready to implement!** 🚀
