#!/usr/bin/env python3
"""
Example script demonstrating extreme context length training with HRM.

This script shows how to use the chunked training utilities to train
on sequences much longer than would fit in VRAM using standard approaches.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("=" * 80)
    print("HRM Extreme Context Length Training - Example")
    print("=" * 80)
    print()
    
    # Import the chunked training utilities
    try:
        from aios.core.hrm_models.chunked_training import (
            estimate_memory_usage,
            recommend_chunk_size,
        )
    except ImportError as e:
        print(f"Error importing chunked_training module: {e}")
        print("Make sure you're running from the AI-OS repository root.")
        return 1
    
    # Scenario 1: Estimate memory for different configurations
    print("Scenario 1: Memory Estimation")
    print("-" * 80)
    
    configs = [
        ("Conservative 10K", 2, 10_000, 1024),
        ("Balanced 50K", 2, 50_000, 2048),
        ("Aggressive 100K", 1, 100_000, 1024),
    ]
    
    for name, batch_size, seq_len, chunk_size in configs:
        mem = estimate_memory_usage(
            batch_size=batch_size,
            seq_len=seq_len,
            chunk_size=chunk_size,
            hidden_size=768,
            vocab_size=50257,
            num_params=124_000_000,
        )
        
        print(f"\n{name}:")
        print(f"  Configuration:")
        print(f"    - Sequence length: {seq_len:,} tokens")
        print(f"    - Chunk size: {chunk_size:,} tokens")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Number of chunks: {mem['num_chunks']}")
        print(f"  Memory breakdown:")
        print(f"    - Model weights: {mem['model_gb']:.2f} GB")
        print(f"    - Optimizer states: {mem['optimizer_gb']:.2f} GB")
        print(f"    - Gradients: {mem['gradients_gb']:.2f} GB")
        print(f"    - Carry states: {mem['carry_gb']:.4f} GB (tiny!)")
        print(f"    - Chunk activations: {mem['chunk_activations_gb']:.3f} GB")
        print(f"    - Chunk logits: {mem['chunk_logits_gb']:.2f} GB")
        print(f"    - PyTorch overhead: {mem['pytorch_overhead_gb']:.2f} GB")
        print(f"  TOTAL VRAM: {mem['total_gb']:.2f} GB")
        
        if mem['total_gb'] <= 10:
            status = "✓ Fits on single 11GB GPU"
        elif mem['total_gb'] <= 20:
            status = "✓ Fits on 2x 11GB GPUs"
        else:
            status = "✗ Needs more VRAM"
        print(f"  Status: {status}")
    
    print()
    print("=" * 80)
    print("Scenario 2: Chunk Size Recommendations")
    print("-" * 80)
    
    scenarios = [
        ("Single 11GB GPU", 11, 1, 100_000),
        ("Dual 11GB GPUs", 20, 2, 100_000),
        ("Dual 11GB GPUs (50K)", 20, 4, 50_000),
        ("High-end 24GB GPU", 24, 4, 100_000),
    ]
    
    for name, vram, batch, seq in scenarios:
        recommended = recommend_chunk_size(
            available_vram_gb=vram,
            batch_size=batch,
            seq_len=seq,
            safety_margin=0.15,
        )
        
        print(f"\n{name}:")
        print(f"  Available VRAM: {vram} GB")
        print(f"  Target batch size: {batch}")
        print(f"  Target sequence length: {seq:,} tokens")
        print(f"  → Recommended chunk size: {recommended:,} tokens")
        
        # Show expected memory with this chunk size
        mem = estimate_memory_usage(
            batch_size=batch,
            seq_len=seq,
            chunk_size=recommended,
            hidden_size=768,
            vocab_size=50257,
            num_params=124_000_000,
        )
        print(f"  → Expected VRAM usage: {mem['total_gb']:.2f} GB")
        print(f"  → Number of chunks: {mem['num_chunks']}")
    
    print()
    print("=" * 80)
    print("Scenario 3: Training Command Examples")
    print("-" * 80)
    print()
    
    print("For your setup (2x 11GB GPUs = 20GB total):\n")
    
    print("Conservative (guaranteed to work):")
    print("  aios hrm-hf train-actv1 \\")
    print("    --model gpt2 \\")
    print("    --dataset-file your_data.txt \\")
    print("    --max-seq-len 10000 \\")
    print("    --batch-size 1 \\")
    print("    --gradient-accumulation-steps 32 \\")
    print("    --gradient-checkpointing \\")
    print("    --ddp --cuda-ids \"0,1\" \\")
    print("    --steps 10000\n")
    
    print("Balanced (recommended):")
    print("  aios hrm-hf train-actv1 \\")
    print("    --model gpt2 \\")
    print("    --dataset-file your_data.txt \\")
    print("    --max-seq-len 50000 \\")
    print("    --batch-size 2 \\")
    print("    --gradient-accumulation-steps 16 \\")
    print("    --gradient-checkpointing \\")
    print("    --ddp --cuda-ids \"0,1\" \\")
    print("    --steps 10000\n")
    
    print("Aggressive (maximum context):")
    print("  aios hrm-hf train-actv1 \\")
    print("    --model gpt2 \\")
    print("    --dataset-file your_data.txt \\")
    print("    --max-seq-len 100000 \\")
    print("    --batch-size 1 \\")
    print("    --gradient-accumulation-steps 32 \\")
    print("    --gradient-checkpointing \\")
    print("    --ddp --cuda-ids \"0,1\" \\")
    print("    --steps 20000\n")
    
    print()
    print("=" * 80)
    print("Key Insights")
    print("-" * 80)
    print()
    print("1. HRM's recurrent architecture makes chunking natural")
    print("   - Carry state is only ~1-2 MB per sample")
    print("   - No attention matrix → huge memory savings")
    print()
    print("2. Main memory bottleneck: output logits")
    print("   - batch_size × seq_len × vocab_size × 4 bytes")
    print("   - Chunking reduces seq_len in memory")
    print()
    print("3. Gradient checkpointing is worth it")
    print("   - 40-50% memory savings")
    print("   - Only 30% slower training")
    print()
    print("4. Use gradient accumulation")
    print("   - Small micro-batches for memory")
    print("   - Large effective batch for convergence")
    print()
    print("5. Start conservative, then scale up")
    print("   - Begin with 10K context to validate setup")
    print("   - Increase to 50K-100K once comfortable")
    print()
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Test memory estimation: python scripts/example_chunked_training.py")
    print("  2. Try short training run with 10K context")
    print("  3. Monitor VRAM usage: watch -n 1 nvidia-smi")
    print("  4. Scale up to 50K-100K context")
    print()
    print("Documentation: docs/EXTREME_CONTEXT_LENGTH_TRAINING.md")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
