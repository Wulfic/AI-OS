"""Run comprehensive optimization tests covering all memory optimization strategies.

This will test 172 configurations (57 unique configs × 3-4 contexts each).
Estimated runtime: 6-8 hours for complete coverage.
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from typing import Dict, Any

# Set environment for C: drive storage (avoid KeyboardInterrupts)
os.environ["HF_HOME"] = "C:\\Users\\tyler\\.cache\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "C:\\Users\\tyler\\.cache\\huggingface\\transformers"
os.environ["AIOS_MINIMAL_LOGGING"] = "1"

from comprehensive_optimization_matrix import ALL_OPTIMIZATION_CONFIGS, count_total_tests
from test_harness_real import RealTestConfig, run_real_training_test

# Storage configuration
MODEL_CACHE_DIR = "C:\\Users\\tyler\\AppData\\Local\\Temp\\aios_opt_test_cache"
TEMP_DIR = "C:\\Users\\tyler\\AppData\\Local\\Temp\\aios_opt_test_temp"
RESULTS_FILE = "Z:/AI-OS-Data/memory_test_results/comprehensive_optimization_results.json"

# Ensure directories exist
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

# Test configuration
MODEL_CONFIG = {
    "h_layers": 3,
    "l_layers": 3,
    "hidden_size": 512,
    "num_heads": 8,
    "model_name": "artifacts/hf_implant/base_model",  # GPT2 tokenizer
    "batch_size": 1,
    "steps": 10,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}


def create_test_config(opt_config, context_size: int) -> RealTestConfig:
    """Create a RealTestConfig from an OptimizationConfig."""
    return RealTestConfig(
        model_name=MODEL_CONFIG["model_name"],
        h_layers=MODEL_CONFIG["h_layers"],
        l_layers=MODEL_CONFIG["l_layers"],
        hidden_size=MODEL_CONFIG["hidden_size"],
        num_heads=MODEL_CONFIG["num_heads"],
        context_size=context_size,
        batch_size=MODEL_CONFIG["batch_size"],
        use_moe=opt_config.use_moe,
        num_experts=MODEL_CONFIG["num_experts"],
        num_experts_per_tok=MODEL_CONFIG["num_experts_per_tok"],
        use_amp=opt_config.use_amp,
        gradient_checkpointing=opt_config.gradient_checkpointing,
        # All additional optimization parameters (now supported!)
        use_flash_attention_2=opt_config.use_flash_attention_2,
        use_8bit_optimizer=opt_config.use_8bit_optimizer,
        cpu_offload=opt_config.cpu_offload,
        context_chunking=opt_config.context_chunking,
        chunk_size=opt_config.chunk_size,
        deepspeed_stage=opt_config.deepspeed_stage,
        use_lora=opt_config.use_lora,
        lora_rank=opt_config.lora_rank if opt_config.lora_rank else 8,
        lora_alpha=opt_config.lora_alpha if opt_config.lora_alpha else (opt_config.lora_rank * 2 if opt_config.lora_rank else 16),
        lora_dropout=0.1,
        lora_target_modules=opt_config.lora_target_modules if opt_config.lora_target_modules else "q_proj,v_proj",
        steps=MODEL_CONFIG["steps"],
        device="cuda:1",
        model_cache_dir=MODEL_CACHE_DIR,
        temp_dir_base=TEMP_DIR,
    )


def save_test_result(results: Dict, test_result: Dict):
    """Save a single test result and update the results file."""
    test_id = test_result["test_id"]
    
    # Update or append
    existing_idx = next(
        (i for i, t in enumerate(results["tests"]) if t.get("test_id") == test_id),
        None
    )
    if existing_idx is not None:
        results["tests"][existing_idx] = test_result
    else:
        results["tests"].append(test_result)
    
    # Save to file
    try:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"  WARNING: Could not save results: {e}")


def run_comprehensive_tests():
    """Run all comprehensive optimization tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE OPTIMIZATION TESTS")
    print("="*80)
    print(f"Total configurations: {len(ALL_OPTIMIZATION_CONFIGS)}")
    print(f"Total tests: {count_total_tests()}")
    print(f"Model: medium (3h/3l, hidden=512)")
    print(f"Tokenizer: gpt2 (50,257 vocab)")
    print(f"Training: 10 steps per test")
    print(f"Results: {RESULTS_FILE}")
    print("="*80)
    print()
    
    # Initialize results
    results = {
        "metadata": {
            "test_suite": "comprehensive_optimizations",
            "total_configs": len(ALL_OPTIMIZATION_CONFIGS),
            "total_tests": count_total_tests(),
            "model": "medium",
            "h_layers": MODEL_CONFIG["h_layers"],
            "l_layers": MODEL_CONFIG["l_layers"],
            "hidden_size": MODEL_CONFIG["hidden_size"],
            "tokenizer": "gpt2",
            "vocab_size": 50257,
            "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tests": []
    }
    
    # Load existing results if available
    if Path(RESULTS_FILE).exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                existing = json.load(f)
                if "tests" in existing:
                    results["tests"] = existing["tests"]
                    print(f"Loaded existing results: {len(results['tests'])} tests\n")
        except Exception as e:
            print(f"Note: Could not load existing results: {e}\n")
    
    total_tests = count_total_tests()
    test_num = 0
    passed = 0
    failed = 0
    skipped = 0
    
    for config_idx, opt_config in enumerate(ALL_OPTIMIZATION_CONFIGS, 1):
        print(f"\n[Config {config_idx}/{len(ALL_OPTIMIZATION_CONFIGS)}] {opt_config.name}")
        print(f"  Description: {opt_config.description}")
        print(f"  Contexts: {opt_config.context_sizes}")
        
        for context_size in opt_config.context_sizes:
            test_num += 1
            test_id = f"{opt_config.name}_ctx{context_size}"
            
            # Check if already completed
            if any(t.get("test_id") == test_id and t.get("success") for t in results["tests"]):
                print(f"    [{test_num}/{total_tests}] SKIP: {test_id} (already completed)")
                skipped += 1
                passed += 1
                continue
            
            print(f"    [{test_num}/{total_tests}] Testing: ctx={context_size}...", end=" ", flush=True)
            
            # Run test with all optimization features enabled
            try:
                test_config = create_test_config(opt_config, context_size)
                result = run_real_training_test(test_config, verbose=False)
                
                if result.success:
                    vram_gb = result.actual_vram_bytes / (1024**3)
                    print(f"{vram_gb:.2f} GB VRAM [OK]")
                    
                    test_result = {
                        "test_id": test_id,
                        "config_name": opt_config.name,
                        "description": opt_config.description,
                        "context_size": context_size,
                        "vram_gb": vram_gb,
                        "duration_seconds": result.test_duration_seconds,
                        "success": True,
                        # Optimization flags
                        "use_moe": opt_config.use_moe,
                        "gradient_checkpointing": opt_config.gradient_checkpointing,
                        "use_amp": opt_config.use_amp,
                        "use_flash_attention_2": opt_config.use_flash_attention_2,
                        "use_8bit_optimizer": opt_config.use_8bit_optimizer,
                        "cpu_offload": opt_config.cpu_offload,
                        "context_chunking": opt_config.context_chunking,
                        "deepspeed_stage": opt_config.deepspeed_stage,
                        "use_lora": opt_config.use_lora,
                        "lora_rank": opt_config.lora_rank,
                    }
                    save_test_result(results, test_result)
                    passed += 1
                else:
                    print(f"FAILED: {result.error_message}")
                    test_result = {
                        "test_id": test_id,
                        "config_name": opt_config.name,
                        "context_size": context_size,
                        "success": False,
                        "error_message": result.error_message,
                    }
                    save_test_result(results, test_result)
                    failed += 1
                    
            except KeyboardInterrupt:
                print("\n\nINTERRUPTED by user")
                raise
            except Exception as e:
                print(f"ERROR: {str(e)[:80]}")
                failed += 1
            
            # Clean up between tests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    time.sleep(5)
                except KeyboardInterrupt:
                    pass
    
    # Final summary
    results["metadata"]["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["metadata"]["summary"] = {
        "total_tests": total_tests,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "success_rate": f"{passed/total_tests*100:.1f}%",
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TESTS COMPLETE")
    print("="*80)
    print(f"Passed: {passed}/{total_tests}")
    print(f"Failed: {failed}/{total_tests}")
    print(f"Skipped: {skipped}/{total_tests}")
    print(f"Success rate: {passed/total_tests*100:.1f}%")
    print(f"\nResults saved: {RESULTS_FILE}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    print("\n[OK] ALL optimization features are now supported:")
    print("  + Mixture of Experts (MoE)")
    print("  + Gradient Checkpointing")
    print("  + Automatic Mixed Precision (AMP)")
    print("  + Flash Attention 2")
    print("  + 8-bit Optimizer (bitsandbytes)")
    print("  + CPU Offload")
    print("  + Context Chunking")
    print("  + DeepSpeed ZeRO stages (1/2/3)")
    print("  + LoRA/PEFT adapters")
    print()
    print("This comprehensive test will cover ALL 172 configurations.")
    print("Estimated runtime: 8-12 hours for complete coverage.")
    print()
    print("Starting comprehensive testing NOW...")
    print()
    
    results = run_comprehensive_tests()
