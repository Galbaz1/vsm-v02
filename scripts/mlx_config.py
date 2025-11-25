"""
Apple Silicon M3 Optimization Configuration

Configures PyTorch Metal Performance Shaders for maximum M3 performance.
"""
import torch
import os

def configure_metal_performance():
    """
    Configure PyTorch for optimal Apple Silicon M3 performance.
    
    This function should be called BEFORE loading any models.
    """
    # Enable MPS backend
    torch.backends.mps.enabled = True
    
    # Use high-precision matrix multiplication
    torch.set_float32_matmul_precision('high')
    
    # Set Metal memory fraction (use 80% of available 256GB)
    torch.mps.set_per_process_memory_fraction(0.8)
    
    # Disable memory fragmentation warnings
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # Disable tokenizer warnings (improves performance)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("[M3 Config] Metal Performance Shaders configured")
    print(f"[M3 Config] MPS available: {torch.backends.mps.is_available()}")
    print(f"[M3 Config] MPS built: {torch.backends.mps.is_built()}")
    
    if not torch.backends.mps.is_available():
        print("[M3 Config] WARNING: MPS not available! Falling back to CPU")
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
