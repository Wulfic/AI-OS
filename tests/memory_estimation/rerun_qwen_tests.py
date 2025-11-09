"""Rerun failed Qwen tests with corrected vocab size.

This script reruns only the Qwen tokenizer tests that failed due to vocab mismatch.
Skips 4096 context tests that are likely to OOM on 11GB GPUs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_harness_real import RealTestConfig, run_real_training_test
from baseline_tests_real import REAL_TOKENIZERS, REAL_MODEL_SIZES

def generate_qwen_retry_configs():
    """Generate configs for failed Qwen tests (excluding 4096 context)."""
    configs = []
    
    tokenizer = REAL_TOKENIZERS["qwen2.5-7b"]
    
    # Test all model sizes
    for model_key, model_info in REAL_MODEL_SIZES.items():
        # Test contexts up to 2048 (4096 will OOM but let it record the error)
        for context in [128, 256, 512, 1024, 2048]:
            config = RealTestConfig(
                model_name=tokenizer["model_name"],
                h_layers=model_info["h_layers"],
                l_layers=model_info["l_layers"],
                hidden_size=model_info["hidden_size"],
                num_heads=model_info["num_heads"],
                context_size=context,
                batch_size=1,
                steps=10,
                halt_max_steps=10,
                eval_batches=1,
                use_moe=True,
                num_experts=8,
                num_experts_per_tok=2,
                gradient_checkpointing=True,
                use_amp=True,
                device="cuda:1",
            )
            configs.append(config)
    
    return configs


def main():
    """Run Qwen retry tests."""
    configs = generate_qwen_retry_configs()
    
    print(f"\n{'='*80}")
    print(f"Rerunning Qwen Tests (vocab size corrected)")
    print(f"{'='*80}")
    print(f"Total tests: {len(configs)}")
    print(f"Note: Skipping 4096 context tests to avoid OOM")
    print(f"Results will be saved to: Z:/AI-OS-Data/memory_test_results/qwen_retry_results.json")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[Test {i}/{len(configs)}]")
        print(f"  Model: Qwen 2.5-7B")
        print(f"  Architecture: {config.h_layers}h/{config.l_layers}l, hidden={config.hidden_size}")
        print(f"  Context: {config.context_size}, Batch: {config.batch_size}")
        
        result = run_real_training_test(config, verbose=True)
        results.append(result)
        
        # Save intermediate results
        output_file = Path("Z:/AI-OS-Data/memory_test_results/qwen_retry_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "test_suite": "qwen_retry",
            "total_tests": len(configs),
            "completed_tests": i,
            "results": [r.__dict__ for r in results],
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
    
    # Final summary
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    
    print(f"\n{'='*80}")
    print(f"Qwen Retry Test Suite Complete")
    print(f"{'='*80}")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(results)*100:.1f}%")
    
    if passed > 0:
        passed_results = [r for r in results if r.success]
        vram_values = [r.actual_vram_bytes for r in passed_results]
        print(f"\nVRAM Statistics:")
        print(f"  Min: {min(vram_values)/1024**3:.2f} GB")
        print(f"  Max: {max(vram_values)/1024**3:.2f} GB")
        print(f"  Avg: {sum(vram_values)/len(vram_values)/1024**3:.2f} GB")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
