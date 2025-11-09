"""Test one specific tokenizer (single tokenizer mode).

Usage:
    python test_single_tokenizer.py phi3-mini
    python test_single_tokenizer.py biobert
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_harness_real import RealTestConfig, run_real_training_test
from baseline_tests_real import REAL_MODEL_SIZES, REAL_TOKENIZERS


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_single_tokenizer.py <tokenizer_name>")
        print(f"\nAvailable tokenizers:")
        for name in REAL_TOKENIZERS.keys():
            print(f"  - {name}")
        sys.exit(1)
    
    tokenizer_name = sys.argv[1]
    
    if tokenizer_name not in REAL_TOKENIZERS:
        print(f"Error: Unknown tokenizer '{tokenizer_name}'")
        print(f"\nAvailable tokenizers:")
        for name in REAL_TOKENIZERS.keys():
            print(f"  - {name}")
        sys.exit(1)
    
    tokenizer_info = REAL_TOKENIZERS[tokenizer_name]
    
    print(f"\n{'='*80}")
    print(f"Testing Tokenizer: {tokenizer_name}")
    print(f"{'='*80}")
    print(f"Vocab size: {tokenizer_info['vocab_size']}")
    print(f"Model path: {tokenizer_info['model_name']}")
    print(f"Test matrix: 5 model sizes × 5 contexts = 25 tests")
    print(f"Results: Z:/AI-OS-Data/memory_test_results/{tokenizer_name}_results.json")
    print(f"{'='*80}\n")
    
    results = []
    configs = []
    
    # Generate configs
    for model_key, model_info in REAL_MODEL_SIZES.items():
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
            configs.append((model_key, config))
    
    # Run tests
    for i, (model_size, config) in enumerate(configs, 1):
        print(f"\n[Test {i}/{len(configs)}]")
        print(f"  Model: {model_size} ({config.h_layers}h/{config.l_layers}l, hidden={config.hidden_size})")
        print(f"  Context: {config.context_size}, Batch: {config.batch_size}")
        
        try:
            result = run_real_training_test(config, verbose=True)
            results.append({
                "model_size": model_size,
                "context": config.context_size,
                "success": result.success,
                "vram_gb": result.actual_vram_bytes / 1024**3 if result.success else 0,
                "error": result.error if hasattr(result, 'error') else None,
            })
        except Exception as e:
            print(f"\n[X] Test crashed with exception: {e}")
            results.append({
                "model_size": model_size,
                "context": config.context_size,
                "success": False,
                "vram_gb": 0,
                "error": str(e),
            })
        
        # Save intermediate results
        output_file = Path(f"Z:/AI-OS-Data/memory_test_results/{tokenizer_name}_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "tokenizer": tokenizer_name,
            "vocab_size": tokenizer_info["vocab_size"],
            "total_tests": len(configs),
            "completed_tests": i,
            "results": results,
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
    
    # Final summary
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    print(f"\n{'='*80}")
    print(f"Test Suite Complete: {tokenizer_name}")
    print(f"{'='*80}")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(results)*100:.1f}%")
    
    if passed > 0:
        vram_values = [r["vram_gb"] for r in results if r["success"]]
        print(f"\nVRAM Statistics:")
        print(f"  Min: {min(vram_values):.2f} GB")
        print(f"  Max: {max(vram_values):.2f} GB")
        print(f"  Avg: {sum(vram_values)/len(vram_values):.2f} GB")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
