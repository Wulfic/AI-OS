"""Run baseline tests for the 9 missing tokenizers.

Already tested (from baseline_real_full_results.json):
- gpt2 (22 tests)
- mistral-7b (24 tests)
- starcoder2 (20 tests)
- codellama (18 tests)

Already tested (from qwen_retry_results.json):
- qwen2.5-7b (17 tests passed, 8 OOM)

Missing tokenizers (need to test):
- phi3-mini
- llava-1.5
- deepseek-coder-v2
- clip-vit
- siglip
- biobert
- scibert
- finbert
- legal-bert

This script will run baseline tests for all 9 missing tokenizers.
Using the same test matrix as the original baseline:
- 5 model sizes (micro, tiny, small, medium, large)
- 5 contexts (128, 256, 512, 1024, 2048) - skipping 4096 which OOMs
- Total: 9 tokenizers × 5 model sizes × 5 contexts = 225 tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_harness_real import RealTestConfig, run_real_training_test
from baseline_tests_real import REAL_MODEL_SIZES

# Missing tokenizers with their vocab sizes
MISSING_TOKENIZERS = {
    "phi3-mini": {
        "model_name": "artifacts/hf_implant/tokenizers/phi3-mini",
        "vocab_size": 32011,
    },
    "llava-1.5": {
        "model_name": "artifacts/hf_implant/tokenizers/llava-1.5",
        "vocab_size": 32002,
    },
    "deepseek-coder-v2": {
        "model_name": "artifacts/hf_implant/tokenizers/deepseek-coder-v2",
        "vocab_size": 100018,
    },
    "clip-vit": {
        "model_name": "artifacts/hf_implant/tokenizers/clip-vit",
        "vocab_size": 49408,
    },
    "siglip": {
        "model_name": "artifacts/hf_implant/tokenizers/siglip",
        "vocab_size": 32000,
    },
    "biobert": {
        "model_name": "artifacts/hf_implant/tokenizers/biobert",
        "vocab_size": 28996,
    },
    "scibert": {
        "model_name": "artifacts/hf_implant/tokenizers/scibert",
        "vocab_size": 31090,
    },
    "finbert": {
        "model_name": "artifacts/hf_implant/tokenizers/finbert",
        "vocab_size": 30522,
    },
    "legal-bert": {
        "model_name": "artifacts/hf_implant/tokenizers/legal-bert",
        "vocab_size": 30522,
    },
}


def generate_missing_tokenizer_configs():
    """Generate configs for all missing tokenizers."""
    configs = []
    
    for tokenizer_name, tokenizer_info in MISSING_TOKENIZERS.items():
        for model_key, model_info in REAL_MODEL_SIZES.items():
            # Test contexts up to 2048 (4096 will OOM but we'll record it)
            for context in [128, 256, 512, 1024, 2048]:
                config = RealTestConfig(
                    model_name=tokenizer_info["model_name"],
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
                configs.append((tokenizer_name, config))
    
    return configs


def main():
    """Run missing tokenizer tests."""
    configs = generate_missing_tokenizer_configs()
    
    print(f"\n{'='*80}")
    print(f"Running Tests for Missing Tokenizers")
    print(f"{'='*80}")
    print(f"Total tests: {len(configs)}")
    print(f"Tokenizers: {list(MISSING_TOKENIZERS.keys())}")
    print(f"Model sizes: {list(REAL_MODEL_SIZES.keys())}")
    print(f"Contexts: [128, 256, 512, 1024, 2048]")
    print(f"Results will be saved to: Z:/AI-OS-Data/memory_test_results/missing_tokenizers_results.json")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, (tokenizer_name, config) in enumerate(configs, 1):
        model_size = None
        for size_name, size_info in REAL_MODEL_SIZES.items():
            if (size_info["h_layers"] == config.h_layers and 
                size_info["l_layers"] == config.l_layers and
                size_info["hidden_size"] == config.hidden_size):
                model_size = size_name
                break
        
        print(f"\n[Test {i}/{len(configs)}]")
        print(f"  Tokenizer: {tokenizer_name}")
        print(f"  Model: {model_size} ({config.h_layers}h/{config.l_layers}l, hidden={config.hidden_size})")
        print(f"  Context: {config.context_size}, Batch: {config.batch_size}")
        
        result = run_real_training_test(config, verbose=True)
        results.append({
            "tokenizer": tokenizer_name,
            "model_size": model_size,
            "result": result.__dict__,
        })
        
        # Save intermediate results
        output_file = Path("Z:/AI-OS-Data/memory_test_results/missing_tokenizers_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "test_suite": "missing_tokenizers",
            "tokenizers": list(MISSING_TOKENIZERS.keys()),
            "total_tests": len(configs),
            "completed_tests": i,
            "results": results,
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
    
    # Final summary
    passed = sum(1 for r in results if r["result"]["success"])
    failed = len(results) - passed
    
    # Group by tokenizer
    tokenizer_stats = {}
    for r in results:
        tok = r["tokenizer"]
        if tok not in tokenizer_stats:
            tokenizer_stats[tok] = {"passed": 0, "failed": 0}
        if r["result"]["success"]:
            tokenizer_stats[tok]["passed"] += 1
        else:
            tokenizer_stats[tok]["failed"] += 1
    
    print(f"\n{'='*80}")
    print(f"Missing Tokenizers Test Suite Complete")
    print(f"{'='*80}")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(results)*100:.1f}%")
    
    print(f"\nPer-Tokenizer Results:")
    for tok_name in MISSING_TOKENIZERS.keys():
        if tok_name in tokenizer_stats:
            stats = tokenizer_stats[tok_name]
            total = stats["passed"] + stats["failed"]
            print(f"  {tok_name:20s}: {stats['passed']:2d}/{total:2d} passed ({stats['passed']/total*100:.0f}%)")
    
    if passed > 0:
        passed_results = [r["result"] for r in results if r["result"]["success"]]
        vram_values = [r["actual_vram_bytes"] for r in passed_results]
        print(f"\nVRAM Statistics (successful tests):")
        print(f"  Min: {min(vram_values)/1024**3:.2f} GB")
        print(f"  Max: {max(vram_values)/1024**3:.2f} GB")
        print(f"  Avg: {sum(vram_values)/len(vram_values)/1024**3:.2f} GB")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
