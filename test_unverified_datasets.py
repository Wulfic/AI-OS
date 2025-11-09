#!/usr/bin/env python3
"""
Test script to verify unverified datasets work correctly.
This will attempt to load a small sample from each dataset.
"""

import os
os.environ['HF_HOME'] = 'Z:\\huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = 'Z:\\huggingface_cache\\datasets'

from datasets import load_dataset
import time

# Datasets to test (unverified ones from GUI list)
DATASETS_TO_TEST = [
    # Programming datasets
    {
        "name": "GitHub-Code-Clean",
        "path": "codeparrot/github-code-clean",
        "config": None,
        "split": "train",
        "category": "programming",
    },
    {
        "name": "The-Stack-Python",
        "path": "bigcode/the-stack",
        "config": "python",
        "split": "train",
        "category": "programming",
    },
    {
        "name": "The-Stack-JavaScript",
        "path": "bigcode/the-stack",
        "config": "javascript",
        "split": "train",
        "category": "programming",
    },
    {
        "name": "The-Stack-Java",
        "path": "bigcode/the-stack",
        "config": "java",
        "split": "train",
        "category": "programming",
    },
    {
        "name": "CodeParrot-Clean",
        "path": "codeparrot/codeparrot-clean",
        "config": None,
        "split": "train",
        "category": "programming",
    },
    # Additional English datasets
    {
        "name": "FineWeb-Edu-10BT",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "category": "english",
    },
    {
        "name": "FineWeb-Edu-100BT",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "split": "train",
        "category": "english",
    },
    {
        "name": "Wikipedia-EN",
        "path": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "category": "english",
    },
    {
        "name": "CodeSearchNet-All",
        "path": "code_search_net",
        "config": "all",
        "split": "train",
        "category": "programming",
    },
    {
        "name": "APPS",
        "path": "codeparrot/apps",
        "config": "all",
        "split": "train",
        "category": "programming",
    },
    {
        "name": "CodeContests",
        "path": "deepmind/code_contests",
        "config": None,
        "split": "train",
        "category": "programming",
    },
]


def test_dataset(ds_info):
    """Test if a dataset can be loaded."""
    print(f"\n{'='*80}")
    print(f"Testing: {ds_info['name']}")
    print(f"Path: {ds_info['path']}")
    print(f"Category: {ds_info['category']}")
    print(f"{'='*80}")
    
    start = time.time()
    try:
        load_kwargs = {
            "path": ds_info["path"],
            "split": ds_info["split"],
            "streaming": True,  # Use streaming for quick test
            "trust_remote_code": True,
        }
        
        if ds_info["config"]:
            load_kwargs["name"] = ds_info["config"]
        
        print("⏳ Loading dataset (streaming mode)...")
        
        dataset = load_dataset(**load_kwargs)
        
        print("⏳ Fetching first 10 samples...")
        samples = []
        for i, sample in enumerate(dataset):
            samples.append(sample)
            if i >= 9:  # Get 10 samples
                break
        
        elapsed = time.time() - start
        
        if len(samples) > 0:
            print(f"✅ SUCCESS!")
            print(f"   • Loaded {len(samples)} samples in {elapsed:.2f}s")
            print(f"   • Sample keys: {list(samples[0].keys())}")
            
            # Show first sample snippet
            first_key = list(samples[0].keys())[0]
            first_value = str(samples[0][first_key])[:200]
            print(f"   • First sample ({first_key}): {first_value}...")
            
            return {
                "status": "success",
                "samples": len(samples),
                "keys": list(samples[0].keys()),
                "time": elapsed,
            }
        else:
            print(f"❌ FAILED: No samples loaded")
            return {"status": "failed", "error": "No samples"}
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FAILED: {e}")
        
        # Check for common issues
        error_str = str(e).lower()
        if "not found" in error_str or "doesn't exist" in error_str:
            reason = "Dataset not found on HuggingFace"
        elif "deprecated" in error_str or "loading script" in error_str:
            reason = "Deprecated loading script"
        elif "authentication" in error_str or "gated" in error_str:
            reason = "Requires authentication/access"
        elif "config" in error_str:
            reason = "Invalid config name"
        else:
            reason = str(e)[:200]
        
        return {
            "status": "failed",
            "error": reason,
            "time": elapsed,
        }


def main():
    print("="*80)
    print("🔍 Dataset Verification Tool")
    print("="*80)
    print(f"\nTesting {len(DATASETS_TO_TEST)} unverified datasets...")
    print("This will take a few minutes as we load samples from each.\n")
    
    results = {
        "verified": [],
        "failed": [],
    }
    
    for i, ds_info in enumerate(DATASETS_TO_TEST, 1):
        print(f"\n[{i}/{len(DATASETS_TO_TEST)}]")
        result = test_dataset(ds_info)
        
        if result["status"] == "success":
            results["verified"].append({
                "name": ds_info["name"],
                "path": ds_info["path"],
                "config": ds_info["config"],
                "category": ds_info["category"],
                "samples": result["samples"],
                "keys": result["keys"],
                "time": result["time"],
            })
        else:
            results["failed"].append({
                "name": ds_info["name"],
                "path": ds_info["path"],
                "config": ds_info["config"],
                "category": ds_info["category"],
                "error": result["error"],
            })
    
    # Print summary
    print("\n" + "="*80)
    print("📊 VERIFICATION SUMMARY")
    print("="*80)
    
    if results["verified"]:
        print(f"\n✅ VERIFIED ({len(results['verified'])} datasets):")
        for ds in results["verified"]:
            config_str = f" (config: {ds['config']})" if ds['config'] else ""
            print(f"\n   {ds['name']}{config_str}")
            print(f"   • Path: {ds['path']}")
            print(f"   • Category: {ds['category']}")
            print(f"   • Sample keys: {ds['keys']}")
            print(f"   • Load time: {ds['time']:.2f}s")
    
    if results["failed"]:
        print(f"\n❌ FAILED ({len(results['failed'])} datasets):")
        for ds in results["failed"]:
            config_str = f" (config: {ds['config']})" if ds['config'] else ""
            print(f"\n   {ds['name']}{config_str}")
            print(f"   • Path: {ds['path']}")
            print(f"   • Reason: {ds['error']}")
    
    # Generate updated dataset list for GUI
    print("\n" + "="*80)
    print("📝 UPDATED DATASET STATUS")
    print("="*80)
    print("\nDatasets ready to add to GUI (verified ✅):")
    
    all_verified = results["verified"]
    for ds in all_verified:
        config_str = f'"{ds["config"]}"' if ds['config'] else 'None'
        print(f"""
    {{
        "name": "{ds['name']}",
        "path": "{ds['path']}",
        "config": {config_str},
        "split": "train",
        "category": "{ds['category']}",
        "verified": True,
    }},""")
    
    if results["failed"]:
        print("\n\nDatasets to REMOVE or mark deprecated:")
        for ds in results["failed"]:
            print(f"   ❌ {ds['name']}: {ds['error']}")
    
    print("\n" + "="*80)
    print(f"✅ Verification complete!")
    print(f"   • Verified: {len(results['verified'])}")
    print(f"   • Failed: {len(results['failed'])}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
