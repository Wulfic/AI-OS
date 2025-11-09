"""Quick verification test for gradient checkpointing fix.

Tests that gradient checkpointing now actually reduces VRAM usage.

Expected Results:
- Baseline (no opts): ~2.61 GB
- Gradient checkpointing: ~1.95 GB (~25% savings)
- Difference: ~0.65 GB

If gradient checkpointing shows only 0.02 GB savings, the fix didn't work.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_harness_real import RealTestConfig, run_real_training_test


def verify_gradient_checkpointing_fix():
    """Run two quick tests to verify gradient checkpointing fix."""
    
    print("=" * 80)
    print("GRADIENT CHECKPOINTING FIX VERIFICATION")
    print("=" * 80)
    print()
    print("Running 2 tests:")
    print("  1. Baseline (no optimizations)")
    print("  2. Gradient checkpointing only")
    print()
    print("Expected: ~25% VRAM savings with gradient checkpointing")
    print("=" * 80)
    print()
    
    # Test 1: Baseline (no optimizations) - LARGER MODEL and LONGER CONTEXT
    print("[1/2] Running BASELINE test (no optimizations)...")
    baseline_config = RealTestConfig(
        model_name="artifacts/hf_implant/tokenizers/gpt2",
        h_layers=3,  # More layers
        l_layers=3,  # More layers  
        hidden_size=768,  # Larger hidden size
        num_heads=8,
        context_size=512,  # Longer context
        batch_size=2,  # Larger batch
        steps=10,
        halt_max_steps=1,
        eval_batches=1,
        use_moe=False,
        num_experts=8,
        num_experts_per_tok=2,
        gradient_checkpointing=False,  # DISABLED
        use_amp=False,
        cpu_offload=False,
        use_8bit_optimizer=False,
        use_flash_attention_2=False,
        deepspeed_stage=0,
        use_lora=False,
        context_chunking=False,
        device="cuda:1",
    )
    
    baseline_result = run_real_training_test(baseline_config)
    
    if not baseline_result.success:
        print(f"[ERROR] Baseline test failed: {baseline_result.error_message or 'Unknown error'}")
        return False
    
    baseline_vram = baseline_result.actual_vram_bytes / (1024**3)  # Convert bytes to GB
    print(f"[OK] Baseline VRAM: {baseline_vram:.3f} GB")
    print()
    
    # Test 2: Gradient checkpointing only - LARGER MODEL and LONGER CONTEXT
    print("[2/2] Running GRADIENT CHECKPOINTING test...")
    gradcheck_config = RealTestConfig(
        model_name="artifacts/hf_implant/tokenizers/gpt2",
        h_layers=3,  # More layers
        l_layers=3,  # More layers
        hidden_size=768,  # Larger hidden size
        num_heads=8,
        context_size=512,  # Longer context
        batch_size=2,  # Larger batch
        steps=10,
        halt_max_steps=1,
        eval_batches=1,
        use_moe=False,
        num_experts=8,
        num_experts_per_tok=2,
        gradient_checkpointing=True,  # ENABLED
        use_amp=False,
        cpu_offload=False,
        use_8bit_optimizer=False,
        use_flash_attention_2=False,
        deepspeed_stage=0,
        use_lora=False,
        context_chunking=False,
        device="cuda:1",
    )
    
    gradcheck_result = run_real_training_test(gradcheck_config)
    
    if not gradcheck_result.success:
        print(f"[ERROR] Gradient checkpointing test failed: {gradcheck_result.error_message or 'Unknown error'}")
        return False
    
    gradcheck_vram = gradcheck_result.actual_vram_bytes / (1024**3)  # Convert bytes to GB
    print(f"[OK] Gradient checkpointing VRAM: {gradcheck_vram:.3f} GB")
    print()
    
    # Calculate savings
    vram_saved = baseline_vram - gradcheck_vram
    savings_pct = (vram_saved / baseline_vram) * 100
    
    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Baseline VRAM:           {baseline_vram:.3f} GB")
    print(f"Gradient Checkpointing:  {gradcheck_vram:.3f} GB")
    print(f"VRAM Saved:              {vram_saved:.3f} GB ({savings_pct:.1f}%)")
    print()
    
    # Verify fix worked
    if savings_pct >= 15.0:  # At least 15% savings (conservative threshold)
        print("[SUCCESS] Gradient checkpointing is WORKING!")
        print(f"          Achieved {savings_pct:.1f}% VRAM savings (expected ~25%)")
        print()
        print("Next steps:")
        print("  1. Rerun comprehensive optimization tests with fixes")
        print("  2. Analyze results with analyze_comprehensive_results.py")
        print("  3. Update VRAM estimator with corrected coefficients")
        return True
    elif savings_pct >= 5.0:
        print("[PARTIAL] Gradient checkpointing showing some effect")
        print(f"          Achieved {savings_pct:.1f}% savings (expected ~25%)")
        print("          May need further investigation")
        return False
    else:
        print("[FAILED] Gradient checkpointing NOT working!")
        print(f"        Only {savings_pct:.1f}% savings (expected ~25%)")
        print("        Fix did not resolve the issue")
        return False


if __name__ == "__main__":
    success = verify_gradient_checkpointing_fix()
    sys.exit(0 if success else 1)
