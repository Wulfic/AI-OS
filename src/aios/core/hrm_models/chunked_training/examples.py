"""
Example usage of chunked training memory estimation utilities.

Demonstrates memory estimation for different training configurations.
"""

from __future__ import annotations

from .memory_estimation import estimate_memory_usage, recommend_chunk_size


def print_memory_estimations():
    """Print memory estimations for various extreme context configurations."""
    print("=" * 60)
    print("Memory Estimation for Extreme Context Length Training")
    print("=" * 60)
    
    configs = [
        ("Conservative (100K context)", 1, 100_000, 1024),
        ("Balanced (50K context)", 2, 50_000, 2048),
        ("Aggressive (10K context)", 4, 10_000, 2048),
    ]
    
    for name, batch_size, seq_len, chunk_size in configs:
        print(f"\n{name}:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len:,}")
        print(f"  Chunk size: {chunk_size}")
        
        mem = estimate_memory_usage(
            batch_size=batch_size,
            seq_len=seq_len,
            chunk_size=chunk_size,
            hidden_size=768,
            vocab_size=50257,
            num_params=124_000_000,
        )
        
        print(f"  Memory breakdown:")
        print(f"    Model: {mem['model_gb']:.2f} GB")
        print(f"    Optimizer: {mem['optimizer_gb']:.2f} GB")
        print(f"    Gradients: {mem['gradients_gb']:.2f} GB")
        print(f"    Carry states: {mem['carry_gb']:.4f} GB")
        print(f"    Chunk activations: {mem['chunk_activations_gb']:.2f} GB")
        print(f"    Chunk logits: {mem['chunk_logits_gb']:.2f} GB")
        print(f"    PyTorch overhead: {mem['pytorch_overhead_gb']:.2f} GB")
        print(f"  TOTAL: {mem['total_gb']:.2f} GB")
        print(f"  Chunks: {mem['num_chunks']}")
        print(f"  Fits in 20GB? {'✓ YES' if mem['total_gb'] <= 18 else '✗ NO (reduce batch or chunk size)'}")
    
    print("\n" + "=" * 60)
    print("Chunk Size Recommendations")
    print("=" * 60)
    
    scenarios = [
        (20, 2, 100_000),
        (20, 4, 50_000),
        (20, 8, 10_000),
        (11, 1, 100_000),
    ]
    
    for vram, batch, seq in scenarios:
        rec_chunk = recommend_chunk_size(vram, batch, seq)
        print(f"VRAM: {vram}GB, Batch: {batch}, Seq: {seq:,} → Recommended chunk: {rec_chunk}")


if __name__ == "__main__":
    print_memory_estimations()
