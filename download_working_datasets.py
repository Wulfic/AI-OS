#!/usr/bin/env python3
"""
WORKING Dataset Downloader - Only VERIFIED datasets that actually work!
Downloads curated datasets to Z:\\training_datasets

This version ONLY includes datasets that have been verified to work on HuggingFace Hub.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

# Set cache to Z: drive
os.environ['HF_HOME'] = 'Z:\\huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = 'Z:\\huggingface_cache\\datasets'

try:
    from datasets import load_dataset, DownloadConfig
    from datasets.utils.logging import set_verbosity_info
    set_verbosity_info()
except ImportError:
    print("Installing datasets library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset, DownloadConfig
    from datasets.utils.logging import set_verbosity_info
    set_verbosity_info()


# VERIFIED WORKING DATASETS ONLY!
WORKING_DATASETS = [
    {
        "name": "WikiText-103",
        "path": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "size_gb": 0.5,
        "description": "High-quality Wikipedia articles",
        "tier": 1,
    },
    {
        "name": "TinyStories",
        "path": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "size_gb": 2.5,
        "description": "Simple stories - great for testing",
        "tier": 1,
    },
    {
        "name": "C4-en-validation",
        "path": "allenai/c4",
        "config": "en",
        "split": "validation",
        "size_gb": 8,
        "description": "C4 validation set - cleaner web text",
        "tier": 1,
    },
    {
        "name": "OpenWebText",
        "path": "Skylion007/openwebtext",
        "config": None,
        "split": "train",
        "size_gb": 38,
        "description": "Reddit web content - diverse language",
        "tier": 2,
        "streaming": True,  # Large, download samples
        "max_samples": 100000,
    },
    {
        "name": "FineWeb-Edu-10BT",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "size_gb": 10,
        "description": "Educational web content (10BT sample)",
        "tier": 2,
    },
    {
        "name": "FineWeb-Edu-100BT",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "split": "train",
        "size_gb": 28,
        "description": "Educational web content (100BT sample)",
        "tier": 3,
    },
]


def download_dataset(ds_info: Dict[str, Any]) -> bool:
    """Download a single dataset."""
    output_path = Path("Z:\\training_datasets") / ds_info["name"].lower().replace(" ", "_").replace("-", "_")
    
    # Check if exists
    if output_path.exists():
        data_files = list(output_path.glob("*.arrow")) + list(output_path.glob("data/*.arrow"))
        if data_files:
            size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
            print(f"   ✅ Already exists: {len(data_files)} files, {size_mb:.1f} MB")
            return True
    
    print(f"\n📥 {ds_info['name']} ({ds_info['size_gb']} GB)")
    print(f"   {ds_info['description']}")
    
    try:
        # Handle streaming datasets
        if ds_info.get("streaming", False):
            max_samples = ds_info.get("max_samples", 100000)
            print(f"   📦 Streaming first {max_samples:,} samples...")
            
            load_kwargs = {
                "path": ds_info["path"],
                "split": ds_info["split"],
                "streaming": True,
            }
            if ds_info["config"]:
                load_kwargs["name"] = ds_info["config"]
            
            dataset_stream = load_dataset(**load_kwargs)
            
            samples = []
            print(f"   ⏳ Collecting samples...", end="", flush=True)
            for i, sample in enumerate(dataset_stream):
                samples.append(sample)
                if (i + 1) % 10000 == 0:
                    print(f"\r   ⏳ Collecting samples... {i+1:,}/{max_samples:,}", end="", flush=True)
                if i + 1 >= max_samples:
                    break
            print()
            
            # Convert and save
            from datasets import Dataset
            dataset = Dataset.from_dict({
                key: [s[key] for s in samples]
                for key in samples[0].keys()
            })
            
            output_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(output_path))
            
            size_mb = sum(f.stat().st_size for f in output_path.glob("*.arrow")) / (1024 * 1024)
            print(f"   ✅ SUCCESS: {len(samples):,} samples, {size_mb:.1f} MB")
            return True
            
        else:
            # Direct download
            print(f"   ⏳ Downloading...")
            
            download_config = DownloadConfig(
                resume_download=True,
                max_retries=5,
            )
            
            load_kwargs = {
                "path": ds_info["path"],
                "split": ds_info["split"],
                "download_config": download_config,
            }
            if ds_info["config"]:
                load_kwargs["name"] = ds_info["config"]
            
            start = time.time()
            dataset = load_dataset(**load_kwargs)
            duration = time.time() - start
            
            output_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(output_path))
            
            size_mb = sum(f.stat().st_size for f in output_path.glob("*.arrow")) / (1024 * 1024)
            print(f"   ✅ SUCCESS: {len(dataset):,} samples, {size_mb:.1f} MB ({duration:.1f}s)")
            return True
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def main():
    print("=" * 70)
    print("🚀 WORKING Dataset Downloader")
    print("=" * 70)
    print("\nThis script downloads ONLY verified, working datasets.\n")
    
    # Group by tier
    tier1 = [ds for ds in WORKING_DATASETS if ds["tier"] == 1]
    tier2 = [ds for ds in WORKING_DATASETS if ds["tier"] == 2]
    tier3 = [ds for ds in WORKING_DATASETS if ds["tier"] == 3]
    
    tier1_size = sum(ds["size_gb"] for ds in tier1)
    tier2_size = sum(ds["size_gb"] for ds in tier2)
    tier3_size = sum(ds["size_gb"] for ds in tier3)
    
    print(f"📦 Tier 1 - Essential ({tier1_size:.1f} GB):")
    for ds in tier1:
        method = " [streaming]" if ds.get("streaming") else ""
        print(f"   • {ds['name']}: {ds['description']} ({ds['size_gb']} GB){method}")
    
    print(f"\n📦 Tier 2 - Recommended ({tier2_size:.1f} GB):")
    for ds in tier2:
        method = " [streaming]" if ds.get("streaming") else ""
        print(f"   • {ds['name']}: {ds['description']} ({ds['size_gb']} GB){method}")
    
    print(f"\n📦 Tier 3 - Large ({tier3_size:.1f} GB):")
    for ds in tier3:
        method = " [streaming]" if ds.get("streaming") else ""
        print(f"   • {ds['name']}: {ds['description']} ({ds['size_gb']} GB){method}")
    
    print("\n" + "=" * 70)
    print("OPTIONS:")
    print("=" * 70)
    print(f"1. Essential only ({tier1_size:.1f} GB) - Best for getting started quickly")
    print(f"2. Essential + Recommended ({tier1_size + tier2_size:.1f} GB) - Good balance")
    print(f"3. All datasets ({tier1_size + tier2_size + tier3_size:.1f} GB) - Complete collection")
    print("0. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "0":
        print("Exiting...")
        return
    
    datasets_to_download = []
    if choice == "1":
        datasets_to_download = tier1
    elif choice == "2":
        datasets_to_download = tier1 + tier2
    elif choice == "3":
        datasets_to_download = WORKING_DATASETS
    else:
        print("❌ Invalid choice")
        return
    
    total_size = sum(ds["size_gb"] for ds in datasets_to_download)
    print(f"\n{'='*70}")
    print(f"📥 Downloading {len(datasets_to_download)} datasets (~{total_size:.1f} GB)")
    print(f"{'='*70}")
    
    results = {"success": [], "failed": [], "skipped": []}
    
    for i, ds_info in enumerate(datasets_to_download, 1):
        print(f"\n[{i}/{len(datasets_to_download)}] {ds_info['name']}")
        
        output_path = Path("Z:\\training_datasets") / ds_info["name"].lower().replace(" ", "_").replace("-", "_")
        if output_path.exists():
            data_files = list(output_path.glob("*.arrow")) + list(output_path.glob("data/*.arrow"))
            if data_files:
                results["skipped"].append(ds_info["name"])
                size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
                print(f"   ✅ Already downloaded: {size_mb:.1f} MB")
                continue
        
        success = download_dataset(ds_info)
        if success:
            results["success"].append(ds_info["name"])
        else:
            results["failed"].append(ds_info["name"])
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    if results["success"]:
        print(f"✅ Downloaded: {len(results['success'])}")
        for name in results["success"]:
            print(f"   • {name}")
    
    if results["skipped"]:
        print(f"\n⏭️  Skipped (existing): {len(results['skipped'])}")
        for name in results["skipped"]:
            print(f"   • {name}")
    
    if results["failed"]:
        print(f"\n❌ Failed: {len(results['failed'])}")
        for name in results["failed"]:
            print(f"   • {name}")
    
    print(f"\n✅ Complete! Data saved to: Z:\\training_datasets")
    
    # Show what's available
    print("\n📁 Available datasets:")
    for path in Path("Z:\\training_datasets").iterdir():
        if path.is_dir():
            data_files = list(path.glob("*.arrow")) + list(path.glob("data/*.arrow"))
            if data_files:
                size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
                print(f"   • {path.name}: {size_mb:.1f} MB ({len(data_files)} files)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
