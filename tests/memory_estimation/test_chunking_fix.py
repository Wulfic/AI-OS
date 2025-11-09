"""Test context chunking after gradient fix.

This script reruns only the 12 chunking tests that failed with
"element 0 of tensors does not require grad" error.

After fixing the chunked_training/core.py to properly accumulate gradients,
these tests should now pass and give us critical data for massive context support.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_harness_real import run_real_training_test, RealTestConfig
from comprehensive_optimization_matrix import ALL_OPTIMIZATION_CONFIGS


def get_chunking_configs():
    """Extract only the chunking configurations from the comprehensive matrix."""
    all_configs = ALL_OPTIMIZATION_CONFIGS
    
    # Filter for chunking configs
    chunking_configs = [
        cfg for cfg in all_configs 
        if "chunking" in cfg.name.lower()
    ]
    
    print(f"\nFound {len(chunking_configs)} chunking configurations:")
    for cfg in chunking_configs:
        print(f"  - {cfg.name}: {cfg.description}")
    
    return chunking_configs


def run_chunking_tests():
    """Run all chunking tests."""
    configs = get_chunking_configs()
    
    print("\n" + "="*80)
    print("CONTEXT CHUNKING TESTS (After Gradient Fix)")
    print("="*80)
    print(f"Total configurations: {len(configs)}")
    print(f"Expected tests: ~{len(configs) * 3} (each config × 3 contexts)")
    print("="*80 + "\n")
    
    results = []
    test_count = 0
    successful = 0
    failed = 0
    
    for cfg_idx, opt_config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"[Config {cfg_idx+1}/{len(configs)}] {opt_config.name}")
        print(f"  Description: {opt_config.description}")
        print(f"  Contexts: {opt_config.context_sizes}")
        print(f"{'='*80}")
        
        for ctx_idx, context_size in enumerate(opt_config.context_sizes):
            test_count += 1
            
            print(f"\n  [Test {test_count}] ctx={context_size}...")
            
            # Create test config
            test_config = RealTestConfig(
                model_name="artifacts/hf_implant/tokenizers/gpt2",
                h_layers=2,
                l_layers=2,
                hidden_size=512,
                num_heads=8,
                context_size=context_size,
                batch_size=1,
                use_moe=opt_config.use_moe,
                num_experts=8,  # Default
                num_experts_per_tok=2,  # Default
                gradient_checkpointing=opt_config.gradient_checkpointing,
                use_amp=opt_config.use_amp,
                use_flash_attention_2=opt_config.use_flash_attention_2,
                use_8bit_optimizer=opt_config.use_8bit_optimizer,
                cpu_offload=opt_config.cpu_offload,
                context_chunking=opt_config.context_chunking,
                chunk_size=opt_config.chunk_size if opt_config.chunk_size else 2048,
                deepspeed_stage=opt_config.deepspeed_stage,
                use_lora=opt_config.use_lora,
                lora_rank=opt_config.lora_rank,
                lora_alpha=opt_config.lora_alpha if opt_config.lora_alpha else (opt_config.lora_rank * 2 if opt_config.lora_rank else None),
                lora_dropout=0.05 if opt_config.use_lora else None,
                lora_target_modules=["q_proj", "k_proj", "v_proj"] if opt_config.use_lora else None,
                device="cuda:1",
            )
            
            # Run test
            try:
                result = run_real_training_test(test_config, verbose=False)
                
                if result.success:
                    vram_gb = result.actual_vram_bytes / (1024**3)
                    print(f"    [OK] {vram_gb:.2f} GB VRAM")
                    successful += 1
                    
                    results.append({
                        "config_name": opt_config.name,
                        "context_size": context_size,
                        "chunk_size": opt_config.chunk_size,
                        "vram_gb": vram_gb,
                        "success": True,
                        "error": None,
                    })
                else:
                    print(f"    [FAIL] {result.error_message}")
                    failed += 1
                    
                    results.append({
                        "config_name": opt_config.name,
                        "context_size": context_size,
                        "chunk_size": opt_config.chunk_size,
                        "vram_gb": 0.0,
                        "success": False,
                        "error": result.error_message,
                    })
                
                # Brief pause between tests
                time.sleep(2)
                
            except Exception as e:
                print(f"    [ERROR] {str(e)}")
                failed += 1
                
                results.append({
                    "config_name": opt_config.name,
                    "context_size": context_size,
                    "chunk_size": opt_config.chunk_size,
                    "vram_gb": 0.0,
                    "success": False,
                    "error": str(e),
                })
    
    # Summary
    print("\n" + "="*80)
    print("CHUNKING TESTS COMPLETE")
    print("="*80)
    print(f"Total tests: {test_count}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/test_count*100:.1f}%")
    print("="*80 + "\n")
    
    # Save results
    output_file = Path("Z:/AI-OS-Data/memory_test_results/chunking_tests_fixed.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "total": test_count,
                "successful": successful,
                "failed": failed,
                "success_rate": successful/test_count if test_count > 0 else 0,
            },
            "results": results,
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}\n")
    
    return results


if __name__ == "__main__":
    print("Context Chunking Test Suite (After Gradient Fix)")
    print("="*80)
    
    results = run_chunking_tests()
    
    # Show failed tests details
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        print("\nFailed tests details:")
        for test in failed_tests:
            print(f"  - {test['config_name']} ctx={test['context_size']}: {test['error']}")
