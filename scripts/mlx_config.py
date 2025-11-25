"""
Apple Silicon M3 Optimization Configuration for ColQwen

Note: The main optimizations are now in colqwen_mlx_ingest.py.
This file is kept for backwards compatibility but the key setting
(PYTORCH_ENABLE_MPS_FALLBACK) must be set BEFORE importing torch.

Key learnings from MPS limitations:
- MPS has a 65536 output channel limit for convolutions
- ColQwen2.5's vision encoder exceeds this limit
- Solution: PYTORCH_ENABLE_MPS_FALLBACK=1 allows CPU fallback
- torch.compile is incompatible with MPS fallback mode

Reference: https://github.com/pytorch/pytorch/issues/152278
"""
import os
import torch


def configure_metal_performance():
    """
    Configure PyTorch for optimal Apple Silicon M3 performance.
    
    IMPORTANT: PYTORCH_ENABLE_MPS_FALLBACK must be set BEFORE importing torch.
    The colqwen_mlx_ingest.py script handles this at the top of the file.
    """
    # Ensure fallback is enabled (should already be set before torch import)
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        print("[M3 Config] WARNING: PYTORCH_ENABLE_MPS_FALLBACK not set!")
        print("[M3 Config] This may cause errors with ColQwen vision encoder.")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Disable tokenizer warnings (improves performance)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("[M3 Config] Metal Performance Shaders configured")
    print(f"[M3 Config] MPS available: {torch.backends.mps.is_available()}")
    print(f"[M3 Config] MPS built: {torch.backends.mps.is_built()}")
    print(f"[M3 Config] MPS fallback enabled: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'not set')}")
    
    if not torch.backends.mps.is_available():
        print("[M3 Config] WARNING: MPS not available! Using CPU only.")
        return False
    
    return True


def get_device():
    """Get optimal device for M3."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        print("[M3 Config] WARNING: No GPU available, using CPU")
        return "cpu"
