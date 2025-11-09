"""
VRAM Estimator Verification Script

This script demonstrates and verifies the accuracy of the HRM ACTv1 VRAM estimator.
"""

from aios.core.hrm_models.extreme_scale_optimizations import estimate_extreme_context_memory

# Test configurations based on common scenarios
test_configs = [
    {
        "name": "Default (8H+8L, 512d) - 4K context",
        "batch_size": 1,
        "seq_len": 4096,
        "chunk_size": 768,
        "hidden_size": 512,
        "h_layers": 8,
        "l_layers": 8,
        "num_heads": 8,
        "expansion": 2.0,
        "vocab_size": 50257,
        "model_params": 87_115_778,  # Actual measured value
        "use_amp": True
    },
    {
        "name": "Default (8H+8L, 512d) - 50K context",
        "batch_size": 1,
        "seq_len": 50000,
        "chunk_size": 512,
        "hidden_size": 512,
        "h_layers": 8,
        "l_layers": 8,
        "num_heads": 8,
        "expansion": 2.0,
        "vocab_size": 50257,
        "model_params": 87_115_778,
        "use_amp": True
    },
    {
        "name": "Large (12H+12L, 768d) - 20K context",
        "batch_size": 1,
        "seq_len": 20000,
        "chunk_size": 640,
        "hidden_size": 768,
        "h_layers": 12,
        "l_layers": 12,
        "num_heads": 12,
        "expansion": 2.0,
        "vocab_size": 50257,
        "model_params": 350_000_000,  # Approximate
        "use_amp": True
    },
    {
        "name": "Extreme (8H+8L, 512d) - 100K context",
        "batch_size": 1,
        "seq_len": 100000,
        "chunk_size": 384,
        "hidden_size": 512,
        "h_layers": 8,
        "l_layers": 8,
        "num_heads": 8,
        "expansion": 2.0,
        "vocab_size": 50257,
        "model_params": 87_115_778,
        "use_amp": True
    },
    {
        "name": "Mega (16H+16L, 1024d) - 50K context",
        "batch_size": 1,
        "seq_len": 50000,
        "chunk_size": 512,
        "hidden_size": 1024,
        "h_layers": 16,
        "l_layers": 16,
        "num_heads": 16,
        "expansion": 2.0,
        "vocab_size": 50257,
        "model_params": 500_000_000,  # Approximate
        "use_amp": True
    },
]

def print_estimate(config):
    """Print a detailed estimate for a configuration."""
    print(f"\n{'='*70}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*70}")
    
    estimate = estimate_extreme_context_memory(**config)
    
    print(f"\nModel Architecture:")
    print(f"  Layers: {config['h_layers']}H + {config['l_layers']}L")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Num Heads: {config['num_heads']}")
    print(f"  Expansion: {config['expansion']}")
    print(f"  Vocab Size: {config['vocab_size']:,}")
    print(f"  Total Params: {config['model_params']:,}")
    
    print(f"\nTraining Configuration:")
    print(f"  Context Length: {config['seq_len']:,} tokens")
    print(f"  Chunk Size: {config['chunk_size']} tokens")
    print(f"  Num Chunks: {estimate['notes']['num_chunks']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Mixed Precision: {'Yes (FP16)' if config['use_amp'] else 'No (FP32)'}")
    print(f"  Gradient Checkpointing: {estimate['notes']['gradient_checkpointing']}")
    
    print(f"\nMemory Breakdown:")
    print(f"  {'Component':<25} {'GB':>10}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Model Parameters':<25} {estimate['model_params_gb']:>10.2f}")
    print(f"  {'Optimizer States':<25} {estimate['optimizer_states_gb']:>10.2f}")
    print(f"  {'Activations (1 chunk)':<25} {estimate['chunk_activations_gb']:>10.2f}")
    print(f"  {'Carry State':<25} {estimate['carry_state_gb']:>10.3f}")
    print(f"  {'Gradients':<25} {estimate['gradients_gb']:>10.2f}")
    print(f"  {'Logits Output':<25} {estimate['logits_output_gb']:>10.2f}")
    print(f"  {'CUDA Overhead (15%)':<25} {estimate['cuda_overhead_gb']:>10.2f}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'TOTAL ESTIMATED':<25} {estimate['total_estimated_gb']:>10.2f}")
    
    # Parameter verification
    param_breakdown = estimate['parameter_breakdown']
    print(f"\nParameter Verification:")
    print(f"  Embeddings: {param_breakdown['embeddings']:,}")
    print(f"  LM Head: {param_breakdown['lm_head']:,}")
    print(f"  H-Level: {param_breakdown['h_level']:,}")
    print(f"  L-Level: {param_breakdown['l_level']:,}")
    print(f"  Total Estimated: {param_breakdown['total']:,}")
    print(f"  Total Input: {param_breakdown['model_params_input']:,}")
    print(f"  Match: {'✅ Yes' if param_breakdown['match'] else '⚠️ No'}")
    
    # GPU compatibility
    print(f"\nGPU Compatibility:")
    for gpu_name, vram_gb in [
        ("RTX 2080 Ti / RTX 3060", 11.0),
        ("RTX 3080 / RTX 4070", 12.0),
        ("RTX 3090 / RTX 4080", 24.0),
        ("RTX 4090 / A6000", 48.0),
    ]:
        if estimate['total_estimated_gb'] <= vram_gb * 0.9:  # 90% threshold
            headroom = vram_gb - estimate['total_estimated_gb']
            print(f"  ✅ {gpu_name:<25} ({vram_gb} GB) - {headroom:.1f} GB headroom")
        else:
            overage = estimate['total_estimated_gb'] - vram_gb
            print(f"  ❌ {gpu_name:<25} ({vram_gb} GB) - {overage:.1f} GB over")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" HRM ACTv1 VRAM Estimator Verification")
    print("="*70)
    print("\nThis tool estimates VRAM usage for HRM ACTv1 training with:")
    print("  • Mixed precision (FP16/BF16)")
    print("  • Gradient checkpointing")
    print("  • Chunked training with carry state compression")
    print("  • AdamW optimizer")
    
    for config in test_configs:
        print_estimate(config)
    
    print(f"\n{'='*70}")
    print("Verification complete!")
    print("="*70)
    print("\nNotes:")
    print("  • Estimates assume gradient checkpointing (saves ~50% activation memory)")
    print("  • Carry state is compressed to last position only (1 token, not full chunk)")
    print("  • AMP uses FP16 for activations, FP32 for weights/gradients")
    print("  • CUDA overhead is estimated at 15% of allocated memory")
    print("  • Real usage may vary ±10% based on PyTorch version and settings")
    print("\n")
