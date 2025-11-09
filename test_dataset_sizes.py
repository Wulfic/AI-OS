#!/usr/bin/env python3
"""
Test script to determine actual sizes of datasets
This will attempt to download a small sample or get metadata for each dataset
"""

import os
import sys
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

os.environ['HF_HOME'] = 'Z:\\huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = 'Z:\\huggingface_cache\\datasets'

# Test datasets
TEST_DATASETS = [
    # English
    {"name": "WikiText-103", "path": "wikitext", "config": "wikitext-103-raw-v1", "category": "english"},
    {"name": "TinyStories", "path": "roneneldan/TinyStories", "config": None, "category": "english"},
    {"name": "CC-News", "path": "cc_news", "config": None, "category": "english"},
    {"name": "FineWeb-Edu-10BT", "path": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "category": "english"},
    {"name": "FineWeb-Edu-100BT", "path": "HuggingFaceFW/fineweb-edu", "config": "sample-100BT", "category": "english"},
    {"name": "Wikipedia-20231101-en", "path": "wikimedia/wikipedia", "config": "20231101.en", "category": "english"},
    
    # Programming
    {"name": "CodeSearchNet-Python", "path": "code_search_net", "config": "python", "category": "programming"},
    {"name": "MBPP", "path": "mbpp", "config": "sanitized", "category": "programming"},
    {"name": "HumanEval", "path": "openai_humaneval", "config": None, "category": "programming"},
    {"name": "CodeSearchNet-All", "path": "code_search_net", "config": "all", "category": "programming"},
    {"name": "APPS", "path": "codeparrot/apps", "config": "all", "category": "programming"},
    {"name": "CodeContests", "path": "deepmind/code_contests", "config": None, "category": "programming"},
]


def test_dataset(ds_info):
    """Test a dataset to get actual size info."""
    print(f"\n{'='*80}")
    print(f"Testing: {ds_info['name']}")
    print(f"Path: {ds_info['path']}")
    print(f"Config: {ds_info['config']}")
    print(f"{'='*80}")
    
    try:
        # Try to get dataset info without downloading
        load_kwargs = {"path": ds_info["path"], "trust_remote_code": False}
        if ds_info["config"]:
            load_kwargs["name"] = ds_info["config"]
        
        # Try streaming to get info
        print("Attempting streaming load...")
        ds = load_dataset(**load_kwargs, split="train", streaming=True)
        
        # Get a few samples to check
        samples = []
        for i, sample in enumerate(ds):
            samples.append(sample)
            if i >= 2:
                break
        
        print(f"✅ Dataset accessible via streaming")
        print(f"   Sample keys: {list(samples[0].keys())}")
        print(f"   Sample 0 length: {len(str(samples[0]))} chars")
        
        # Try to get size info from dataset card/info
        try:
            from datasets import load_dataset_builder
            builder = load_dataset_builder(ds_info["path"], name=ds_info["config"])
            if hasattr(builder, 'info') and hasattr(builder.info, 'download_size'):
                size_bytes = builder.info.download_size
                if size_bytes:
                    size_gb = size_bytes / (1024 ** 3)
                    print(f"   Download size from metadata: {size_gb:.2f} GB")
            if hasattr(builder, 'info') and hasattr(builder.info, 'dataset_size'):
                size_bytes = builder.info.dataset_size
                if size_bytes:
                    size_gb = size_bytes / (1024 ** 3)
                    print(f"   Dataset size from metadata: {size_gb:.2f} GB")
        except Exception as e:
            print(f"   Could not get size metadata: {e}")
        
        return {"status": "success", "accessible": True, "method": "streaming"}
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    print("="*80)
    print("DATASET SIZE TESTING")
    print("="*80)
    print("\nThis will test each dataset to determine actual sizes.\n")
    
    results = []
    
    for ds_info in TEST_DATASETS:
        result = test_dataset(ds_info)
        result["name"] = ds_info["name"]
        result["category"] = ds_info["category"]
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"\n✅ Accessible: {len(success)}")
    for r in success:
        print(f"   • {r['name']} ({r['category']})")
    
    print(f"\n❌ Failed: {len(failed)}")
    for r in failed:
        print(f"   • {r['name']} ({r['category']}): {r.get('error', 'Unknown')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
