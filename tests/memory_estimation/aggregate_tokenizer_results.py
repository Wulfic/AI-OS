"""Aggregate all tokenizer test results into a single comprehensive dataset.

Combines results from all completed tokenizer tests into a unified analysis file.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Results directory
RESULTS_DIR = "Z:/AI-OS-Data/memory_test_results"
OUTPUT_FILE = "Z:/AI-OS-Data/memory_test_results/aggregated_tokenizer_results.json"
BASELINE_FILE = "Z:/AI-OS-Data/memory_test_results/baseline_real_full_results.json"

# List of tokenizers that were tested
TOKENIZERS = [
    "gpt2",
    "mistral-7b", 
    "qwen2.5-7b",
    "starcoder2",
    "codellama",
    "phi3-mini",
    "llava-1.5",
    "deepseek-coder-v2",
    "clip-vit",
    "siglip",
    "biobert",
    "scibert",
    "finbert",
    "legal-bert",
]

# Known vocab sizes for tokenizers
VOCAB_SIZES = {
    "gpt2": 50257,
    "mistral-7b": 32000,
    "qwen2.5-7b": 151665,
    "starcoder2": 49152,
    "codellama": 32016,
    "phi3-mini": 32064,
    "llava-1.5": 32002,
    "deepseek-coder-v2": 102400,
    "clip-vit": 49408,
    "siglip": 32000,
    "biobert": 28996,
    "scibert": 31090,
    "finbert": 30873,
    "legal-bert": 100018,
}

def load_baseline_results() -> Dict[str, Any]:
    """Load the original baseline test results (first 5 tokenizers)."""
    baseline_file = Path(BASELINE_FILE)
    
    if not baseline_file.exists():
        print(f"  ℹ️  No baseline file found")
        return {}
    
    try:
        with open(baseline_file, 'r') as f:
            data = json.load(f)
        
        print(f"  ✓ Loaded baseline file with {len(data.get('results', []))} tests")
        
        # Reorganize by tokenizer
        by_tokenizer = defaultdict(lambda: {"results": [], "vocab_size": 0})
        
        for test in data.get("results", []):
            # Extract tokenizer from model_name (e.g., "artifacts/hf_implant/tokenizers/gpt2" -> "gpt2")
            model_name = test.get("config", {}).get("model_name", "")
            tokenizer = model_name.split("/")[-1] if model_name else None
            
            if tokenizer and test.get("success"):
                # Restructure test data to match new format
                restructured = {
                    "tokenizer": tokenizer,
                    "model_size": f"{test['config']['hidden_size']}h",  # Will be categorized below
                    "context_size": test["config"]["context_size"],
                    "vram_gb": test["actual_vram_bytes"] / (1024**3),
                    "duration_seconds": test.get("test_duration_seconds", 0),
                    "success": True,
                    "config": test["config"],
                }
                
                # Map hidden size to model size names
                hidden = test["config"]["hidden_size"]
                if hidden == 128:
                    restructured["model_size"] = "micro"
                elif hidden == 256:
                    restructured["model_size"] = "tiny"
                elif hidden == 384:
                    restructured["model_size"] = "small"
                elif hidden == 512:
                    restructured["model_size"] = "medium"
                elif hidden == 768:
                    restructured["model_size"] = "large"
                
                by_tokenizer[tokenizer]["results"].append(restructured)
                by_tokenizer[tokenizer]["vocab_size"] = VOCAB_SIZES.get(tokenizer, 0)
        
        return by_tokenizer
    except Exception as e:
        print(f"  ❌ Error loading baseline file: {e}")
        return {}


def load_tokenizer_results(tokenizer_name: str, baseline_data: Dict) -> Dict[str, Any]:
    """Load results for a specific tokenizer."""
    # Special case for qwen (retry file)
    if tokenizer_name == "qwen2.5-7b":
        retry_file = Path(RESULTS_DIR) / "qwen_retry_results.json"
        if retry_file.exists():
            try:
                with open(retry_file, 'r') as f:
                    data = json.load(f)
                # Add tokenizer name if missing
                if "results" in data:
                    for result in data["results"]:
                        if "tokenizer" not in result:
                            result["tokenizer"] = "qwen2.5-7b"
                return {
                    "tokenizer": "qwen2.5-7b",
                    "vocab_size": VOCAB_SIZES.get("qwen2.5-7b", 0),
                    "total_tests": data.get("total_tests", 0),
                    "completed_tests": data.get("completed_tests", 0),
                    "results": data.get("results", []),
                }
            except:
                pass
    
    # First check if it's in baseline data
    if tokenizer_name in baseline_data:
        results = baseline_data[tokenizer_name]["results"]
        return {
            "tokenizer": tokenizer_name,
            "vocab_size": baseline_data[tokenizer_name]["vocab_size"],
            "total_tests": len(results),
            "completed_tests": sum(1 for r in results if r.get("success")),
            "results": results,
        }
    
    # Otherwise check for individual file
    results_file = Path(RESULTS_DIR) / f"{tokenizer_name}_results.json"
    
    if not results_file.exists():
        print(f"  ⚠️  No results file for {tokenizer_name}")
        return None
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
            
        if "results" not in data:
            print(f"  ⚠️  No results in file for {tokenizer_name}")
            return None
            
        return data
    except Exception as e:
        print(f"  ❌ Error loading {tokenizer_name}: {e}")
        return None


def aggregate_all_results():
    """Aggregate results from all tokenizers."""
    print("\n" + "="*80)
    print("Aggregating Tokenizer Test Results")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Baseline file: {BASELINE_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*80)
    print()
    
    # Load baseline data first
    print("Loading baseline results (first 5 tokenizers)...")
    baseline_data = load_baseline_results()
    print()
    
    aggregated = {
        "metadata": {
            "total_tokenizers_tested": 0,
            "total_tests_collected": 0,
            "total_successful_tests": 0,
            "total_failed_tests": 0,
            "vocab_size_range": {"min": None, "max": None},
            "vram_range_gb": {"min": None, "max": None},
            "context_sizes_tested": set(),
            "model_sizes_tested": set(),
        },
        "tokenizers": {},
        "all_tests": [],
        "summary_by_tokenizer": {},
        "summary_by_context": defaultdict(list),
        "summary_by_model_size": defaultdict(list),
    }
    
    for tokenizer in TOKENIZERS:
        print(f"Loading: {tokenizer}...")
        data = load_tokenizer_results(tokenizer, baseline_data)
        
        if data is None:
            continue
        
        tokenizer_info = {
            "tokenizer": data.get("tokenizer", tokenizer),
            "vocab_size": data.get("vocab_size", 0),
            "total_tests": data.get("total_tests", 0),
            "completed_tests": data.get("completed_tests", 0),
            "results": data.get("results", []),
        }
        
        aggregated["tokenizers"][tokenizer] = tokenizer_info
        aggregated["metadata"]["total_tokenizers_tested"] += 1
        
        # Process each test result
        successful_tests = 0
        failed_tests = 0
        
        for result in tokenizer_info["results"]:
            if result.get("success"):
                successful_tests += 1
                aggregated["all_tests"].append({
                    "tokenizer": tokenizer,
                    "vocab_size": tokenizer_info["vocab_size"],
                    **result
                })
                
                # Track ranges
                vram_gb = result.get("vram_gb", 0)
                if vram_gb > 0:
                    if aggregated["metadata"]["vram_range_gb"]["min"] is None:
                        aggregated["metadata"]["vram_range_gb"]["min"] = vram_gb
                        aggregated["metadata"]["vram_range_gb"]["max"] = vram_gb
                    else:
                        aggregated["metadata"]["vram_range_gb"]["min"] = min(
                            aggregated["metadata"]["vram_range_gb"]["min"], vram_gb
                        )
                        aggregated["metadata"]["vram_range_gb"]["max"] = max(
                            aggregated["metadata"]["vram_range_gb"]["max"], vram_gb
                        )
                
                # Track context sizes and model sizes
                aggregated["metadata"]["context_sizes_tested"].add(result.get("context_size"))
                aggregated["metadata"]["model_sizes_tested"].add(result.get("model_size"))
                
                # Group by context and model size
                aggregated["summary_by_context"][result.get("context_size")].append(vram_gb)
                aggregated["summary_by_model_size"][result.get("model_size")].append(vram_gb)
            else:
                failed_tests += 1
        
        # Update vocab range
        vocab = tokenizer_info["vocab_size"]
        if vocab > 0:
            if aggregated["metadata"]["vocab_size_range"]["min"] is None:
                aggregated["metadata"]["vocab_size_range"]["min"] = vocab
                aggregated["metadata"]["vocab_size_range"]["max"] = vocab
            else:
                aggregated["metadata"]["vocab_size_range"]["min"] = min(
                    aggregated["metadata"]["vocab_size_range"]["min"], vocab
                )
                aggregated["metadata"]["vocab_size_range"]["max"] = max(
                    aggregated["metadata"]["vocab_size_range"]["max"], vocab
                )
        
        # Summary for this tokenizer
        aggregated["summary_by_tokenizer"][tokenizer] = {
            "vocab_size": vocab,
            "total_tests": tokenizer_info["total_tests"],
            "successful": successful_tests,
            "failed": failed_tests,
            "success_rate": f"{successful_tests/tokenizer_info['total_tests']*100:.1f}%" if tokenizer_info["total_tests"] > 0 else "0%",
        }
        
        aggregated["metadata"]["total_successful_tests"] += successful_tests
        aggregated["metadata"]["total_failed_tests"] += failed_tests
        
        print(f"  ✓ {tokenizer}: {successful_tests}/{tokenizer_info['total_tests']} tests")
    
    # Convert sets to sorted lists (filter out None)
    aggregated["metadata"]["context_sizes_tested"] = sorted([x for x in aggregated["metadata"]["context_sizes_tested"] if x is not None])
    aggregated["metadata"]["model_sizes_tested"] = sorted([x for x in aggregated["metadata"]["model_sizes_tested"] if x is not None])
    
    # Calculate context and model size statistics
    for context, vram_values in aggregated["summary_by_context"].items():
        aggregated["summary_by_context"][context] = {
            "count": len(vram_values),
            "min_vram_gb": min(vram_values) if vram_values else 0,
            "max_vram_gb": max(vram_values) if vram_values else 0,
            "avg_vram_gb": sum(vram_values) / len(vram_values) if vram_values else 0,
        }
    
    for model_size, vram_values in aggregated["summary_by_model_size"].items():
        aggregated["summary_by_model_size"][model_size] = {
            "count": len(vram_values),
            "min_vram_gb": min(vram_values) if vram_values else 0,
            "max_vram_gb": max(vram_values) if vram_values else 0,
            "avg_vram_gb": sum(vram_values) / len(vram_values) if vram_values else 0,
        }
    
    aggregated["metadata"]["total_tests_collected"] = (
        aggregated["metadata"]["total_successful_tests"] + 
        aggregated["metadata"]["total_failed_tests"]
    )
    
    # Save aggregated results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("Aggregation Complete")
    print("="*80)
    print(f"Tokenizers tested: {aggregated['metadata']['total_tokenizers_tested']}")
    print(f"Total tests collected: {aggregated['metadata']['total_tests_collected']}")
    print(f"Successful tests: {aggregated['metadata']['total_successful_tests']}")
    print(f"Failed tests: {aggregated['metadata']['total_failed_tests']}")
    print(f"Success rate: {aggregated['metadata']['total_successful_tests']/aggregated['metadata']['total_tests_collected']*100:.1f}%")
    print()
    print(f"Vocab size range: {aggregated['metadata']['vocab_size_range']['min']:,} - {aggregated['metadata']['vocab_size_range']['max']:,}")
    print(f"VRAM range: {aggregated['metadata']['vram_range_gb']['min']:.2f} - {aggregated['metadata']['vram_range_gb']['max']:.2f} GB")
    print(f"Context sizes: {aggregated['metadata']['context_sizes_tested']}")
    print(f"Model sizes: {aggregated['metadata']['model_sizes_tested']}")
    print()
    print(f"Results saved to: {OUTPUT_FILE}")
    print("="*80)
    
    return aggregated


if __name__ == "__main__":
    results = aggregate_all_results()
