"""Run optimization comparison tests to measure VRAM impact of different optimization strategies.

This script tests 8 optimization combinations across 3 context sizes (24 total tests):
1. Baseline (no optimizations)
2. MoE only
3. Gradient checkpointing only
4. AMP only
5. MoE + gradient checkpointing
6. MoE + AMP
7. Gradient checkpointing + AMP
8. All optimizations (MoE + gradient checkpointing + AMP)

Context sizes: 128, 512, 1024
Model: medium (512 hidden size, 3h/3l)
Tokenizer: gpt2 (50,257 vocab)
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from typing import List, Dict, Any

# Set HF cache to C: drive to avoid interrupts
os.environ["HF_HOME"] = "C:\\Users\\tyler\\.cache\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "C:\\Users\\tyler\\.cache\\huggingface\\transformers"
os.environ["AIOS_MINIMAL_LOGGING"] = "1"

from test_harness_real import RealTestConfig, run_real_training_test

# Model cache directory
MODEL_CACHE_DIR = "Z:/AI-OS-Data/memory_test_cache"
RESULTS_FILE = "Z:/AI-OS-Data/memory_test_results/optimization_comparison_results.json"

# Optimization configurations
OPTIMIZATIONS = [
    {"name": "baseline", "use_moe": False, "gradient_checkpointing": False, "use_amp": False},
    {"name": "moe_only", "use_moe": True, "gradient_checkpointing": False, "use_amp": False},
    {"name": "gradcheck_only", "use_moe": False, "gradient_checkpointing": True, "use_amp": False},
    {"name": "amp_only", "use_moe": False, "gradient_checkpointing": False, "use_amp": True},
    {"name": "moe_gradcheck", "use_moe": True, "gradient_checkpointing": True, "use_amp": False},
    {"name": "moe_amp", "use_moe": True, "gradient_checkpointing": False, "use_amp": True},
    {"name": "gradcheck_amp", "use_moe": False, "gradient_checkpointing": True, "use_amp": True},
    {"name": "all_opts", "use_moe": True, "gradient_checkpointing": True, "use_amp": True},
]

CONTEXT_SIZES = [128, 512, 1024]

# Medium model configuration
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


def run_optimization_tests():
    """Run all optimization comparison tests."""
    print("\n" + "="*80)
    print("Optimization Comparison Tests")
    print("="*80)
    print(f"Model: medium (3h/3l, hidden=512)")
    print(f"Tokenizer: gpt2 (50,257 vocab)")
    print(f"Optimizations: {len(OPTIMIZATIONS)} configs")
    print(f"Contexts: {CONTEXT_SIZES}")
    print(f"Total tests: {len(OPTIMIZATIONS) * len(CONTEXT_SIZES)}")
    print(f"Results: {RESULTS_FILE}")
    print("="*80)
    print()
    
    results = {
        "test_info": {
            "model": "medium",
            "h_layers": MODEL_CONFIG["h_layers"],
            "l_layers": MODEL_CONFIG["l_layers"],
            "hidden_size": MODEL_CONFIG["hidden_size"],
            "tokenizer": "gpt2",
            "vocab_size": 50257,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
    
    total_tests = len(OPTIMIZATIONS) * len(CONTEXT_SIZES)
    test_num = 0
    passed = 0
    failed = 0
    
    for opt_config in OPTIMIZATIONS:
        for context_size in CONTEXT_SIZES:
            test_num += 1
            
            # Check if test already completed
            test_id = f"{opt_config['name']}_ctx{context_size}"
            if any(t.get("test_id") == test_id and t.get("success") for t in results["tests"]):
                print(f"[{test_num}/{total_tests}] SKIP: {test_id} (already completed)")
                passed += 1
                continue
            
            print(f"\n[{test_num}/{total_tests}] Testing: {opt_config['name']}")
            print(f"  Context: {context_size}")
            print(f"  MoE: {opt_config['use_moe']}, GradCheck: {opt_config['gradient_checkpointing']}, AMP: {opt_config['use_amp']}")
            
            # Create test config
            config = RealTestConfig(
                model_name=MODEL_CONFIG["model_name"],
                h_layers=MODEL_CONFIG["h_layers"],
                l_layers=MODEL_CONFIG["l_layers"],
                hidden_size=MODEL_CONFIG["hidden_size"],
                num_heads=MODEL_CONFIG["num_heads"],
                context_size=context_size,
                batch_size=MODEL_CONFIG["batch_size"],
                use_moe=opt_config["use_moe"],
                num_experts=MODEL_CONFIG["num_experts"],
                num_experts_per_tok=MODEL_CONFIG["num_experts_per_tok"],
                use_amp=opt_config["use_amp"],
                gradient_checkpointing=opt_config["gradient_checkpointing"],
                steps=MODEL_CONFIG["steps"],
                device="cuda:1",
                model_cache_dir=MODEL_CACHE_DIR,
            )
            
            try:
                # Run test
                result = run_real_training_test(config, verbose=False)
                
                if result.success:
                    passed += 1
                    vram_gb = result.actual_vram_bytes / (1024**3)
                    print(f"  PASSED: {vram_gb:.2f} GB VRAM")
                    
                    # Save result
                    test_result = {
                        "test_id": test_id,
                        "optimization": opt_config["name"],
                        "context_size": context_size,
                        "use_moe": opt_config["use_moe"],
                        "gradient_checkpointing": opt_config["gradient_checkpointing"],
                        "use_amp": opt_config["use_amp"],
                        "vram_gb": vram_gb,
                        "duration_seconds": result.test_duration_seconds,
                        "success": True,
                    }
                    
                    # Update or append result
                    existing_idx = next((i for i, t in enumerate(results["tests"]) if t.get("test_id") == test_id), None)
                    if existing_idx is not None:
                        results["tests"][existing_idx] = test_result
                    else:
                        results["tests"].append(test_result)
                    
                    # Save after each successful test
                    try:
                        with open(RESULTS_FILE, 'w') as f:
                            json.dump(results, f, indent=2)
                    except Exception as save_err:
                        print(f"  WARNING: Could not save results: {save_err}")
                else:
                    failed += 1
                    print(f"  FAILED: {result.error_message}")
                    
            except KeyboardInterrupt:
                print(f"\n  INTERRUPTED by user")
                raise
            except Exception as e:
                failed += 1
                print(f"  ERROR: {str(e)[:100]}")
            
            # Clean up CUDA memory between tests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                time.sleep(2)  # Brief pause
    
    print("\n" + "="*80)
    print("Optimization Tests Complete")
    print("="*80)
    print(f"Passed: {passed}/{total_tests}")
    print(f"Failed: {failed}/{total_tests}")
    print(f"Results saved: {RESULTS_FILE}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = run_optimization_tests()
