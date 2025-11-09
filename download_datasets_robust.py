#!/usr/bin/env python3
"""
Robust Dataset Downloader for AI-OS Training
Downloads curated datasets to Z:\\training_datasets with proper error handling and progress tracking.

This version focuses on PROVEN, WORKING datasets with smart caching and resume support.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Set cache directory to Z: drive BEFORE importing datasets
os.environ['HF_HOME'] = 'Z:\\huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = 'Z:\\huggingface_cache\\datasets'
os.environ['TRANSFORMERS_CACHE'] = 'Z:\\huggingface_cache\\models'

try:
    from datasets import load_dataset, DatasetDict, Dataset, DownloadConfig
    from datasets.utils.logging import set_verbosity_info
    set_verbosity_info()
except ImportError:
    print("ERROR: datasets library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset, DatasetDict, Dataset, DownloadConfig
    from datasets.utils.logging import set_verbosity_info
    set_verbosity_info()


# Dataset catalog with PROVEN, WORKING configurations
DATASET_CATALOG = [
    {
        "name": "WikiText-103",
        "path": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "size_gb": 0.5,
        "description": "High-quality Wikipedia articles for language modeling",
        "tier": 1,
        "method": "direct"  # Direct download, no streaming needed
    },
    {
        "name": "Wikipedia-EN",
        "path": "wikipedia",
        "config": "20220301.en",
        "split": "train",
        "size_gb": 20,
        "description": "Full English Wikipedia dump (2022-03-01)",
        "tier": 1,
        "method": "direct"
    },
    {
        "name": "BookCorpus",
        "path": "bookcorpusopen/bookcorpusopen",
        "config": None,
        "split": "train",
        "size_gb": 6,
        "description": "Books corpus for general language understanding",
        "tier": 1,
        "method": "direct"
    },
    {
        "name": "C4-en-validation",
        "path": "allenai/c4",
        "config": "en",
        "split": "validation",
        "size_gb": 1.5,
        "description": "C4 validation set (cleaner, smaller subset)",
        "tier": 1,
        "method": "direct"
    },
    {
        "name": "OpenWebText",
        "path": "Skylion007/openwebtext",
        "config": None,
        "split": "train",
        "size_gb": 38,
        "description": "Web text for diverse language patterns",
        "tier": 2,
        "method": "streaming"  # Large dataset, needs streaming approach
    },
    {
        "name": "TinyStories",
        "path": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "size_gb": 2,
        "description": "Simple stories for testing and baseline",
        "tier": 1,
        "method": "direct"
    },
    {
        "name": "Pile-Uncopyrighted",
        "path": "monology/pile-uncopyrighted",
        "config": None,
        "split": "train",
        "size_gb": 100,
        "description": "Large diverse corpus (subset of The Pile)",
        "tier": 3,
        "method": "streaming",
        "streaming_samples": 100000  # Download first 100k samples
    },
    {
        "name": "FineWeb-Edu-sample",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "size_gb": 10,
        "description": "Educational web content (sample)",
        "tier": 2,
        "method": "direct"
    },
]


def get_output_path(dataset_name: str) -> Path:
    """Get output path for a dataset."""
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    return Path("Z:\\training_datasets") / safe_name


def check_dataset_exists(output_path: Path) -> bool:
    """Check if dataset is already downloaded."""
    if not output_path.exists():
        return False
    
    # Check for actual data files (not just config)
    data_files = list(output_path.glob("*.arrow")) + list(output_path.glob("data/*.arrow"))
    
    if len(data_files) > 0:
        total_size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
        print(f"   ✅ Already downloaded: {len(data_files)} files, {total_size_mb:.1f} MB")
        return True
    
    return False


def download_direct(ds_info: Dict[str, Any], output_path: Path) -> bool:
    """Download dataset directly (for smaller datasets)."""
    try:
        print(f"\n📥 Downloading {ds_info['name']} ({ds_info['size_gb']} GB)...")
        print(f"   Source: {ds_info['path']}")
        print(f"   Method: Direct download")
        
        # Configure download with resume support
        download_config = DownloadConfig(
            resume_download=True,
            max_retries=5,
            cache_dir=Path("Z:\\huggingface_cache\\downloads")
        )
        
        # Load dataset
        start_time = time.time()
        
        load_kwargs = {
            "path": ds_info["path"],
            "split": ds_info["split"],
            "download_config": download_config,
            "trust_remote_code": False,  # Security: don't execute arbitrary code
        }
        
        if ds_info["config"]:
            load_kwargs["name"] = ds_info["config"]
        
        print(f"   ⏳ Loading dataset...")
        dataset = load_dataset(**load_kwargs)
        
        duration = time.time() - start_time
        
        # Save to disk
        print(f"   💾 Saving to disk...")
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        
        # Verify
        sample_count = len(dataset)
        size_mb = sum(f.stat().st_size for f in output_path.glob("*.arrow")) / (1024 * 1024)
        
        print(f"   ✅ SUCCESS: {sample_count:,} samples, {size_mb:.1f} MB in {duration:.1f}s")
        return True
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def download_streaming(ds_info: Dict[str, Any], output_path: Path) -> bool:
    """Download large dataset using streaming approach."""
    try:
        print(f"\n📥 Downloading {ds_info['name']} ({ds_info['size_gb']} GB)...")
        print(f"   Source: {ds_info['path']}")
        print(f"   Method: Streaming (large dataset)")
        
        # For very large datasets, download a manageable subset
        max_samples = ds_info.get("streaming_samples", 500000)
        
        print(f"   ⏳ Streaming first {max_samples:,} samples...")
        
        load_kwargs = {
            "path": ds_info["path"],
            "split": ds_info["split"],
            "streaming": True,
            "trust_remote_code": False,
        }
        
        if ds_info["config"]:
            load_kwargs["name"] = ds_info["config"]
        
        # Load as streaming dataset
        start_time = time.time()
        dataset_stream = load_dataset(**load_kwargs)
        
        # Collect samples
        print(f"   📦 Collecting samples...")
        samples = []
        for i, sample in enumerate(dataset_stream):
            samples.append(sample)
            
            if (i + 1) % 10000 == 0:
                print(f"      Progress: {i+1:,} samples collected...")
            
            if i + 1 >= max_samples:
                break
        
        duration = time.time() - start_time
        
        # Convert to Dataset and save
        print(f"   💾 Converting to dataset and saving...")
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_dict({
            key: [sample[key] for sample in samples]
            for key in samples[0].keys()
        })
        
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        
        # Verify
        size_mb = sum(f.stat().st_size for f in output_path.glob("*.arrow")) / (1024 * 1024)
        
        print(f"   ✅ SUCCESS: {len(samples):,} samples, {size_mb:.1f} MB in {duration:.1f}s")
        return True
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def download_dataset(ds_info: Dict[str, Any]) -> bool:
    """Download a single dataset using the appropriate method."""
    output_path = get_output_path(ds_info["name"])
    
    # Check if already exists
    if check_dataset_exists(output_path):
        return True
    
    # Download based on method
    method = ds_info.get("method", "direct")
    
    if method == "streaming":
        return download_streaming(ds_info, output_path)
    else:
        return download_direct(ds_info, output_path)


def show_menu():
    """Display interactive menu."""
    print("\n" + "="*70)
    print("🚀 AI-OS Robust Dataset Downloader")
    print("="*70)
    print("\nAvailable dataset tiers:")
    print()
    
    # Group by tier
    tier1 = [ds for ds in DATASET_CATALOG if ds["tier"] == 1]
    tier2 = [ds for ds in DATASET_CATALOG if ds["tier"] == 2]
    tier3 = [ds for ds in DATASET_CATALOG if ds["tier"] == 3]
    
    tier1_size = sum(ds["size_gb"] for ds in tier1)
    tier2_size = sum(ds["size_gb"] for ds in tier2)
    tier3_size = sum(ds["size_gb"] for ds in tier3)
    
    print(f"📌 Tier 1 - Essential Datasets ({tier1_size:.1f} GB)")
    for ds in tier1:
        print(f"   • {ds['name']}: {ds['description']} ({ds['size_gb']} GB)")
    
    print(f"\n📌 Tier 2 - Recommended Additions ({tier2_size:.1f} GB)")
    for ds in tier2:
        print(f"   • {ds['name']}: {ds['description']} ({ds['size_gb']} GB)")
    
    print(f"\n📌 Tier 3 - Large-Scale Corpus ({tier3_size:.1f} GB)")
    for ds in tier3:
        method = " - streaming" if ds.get("method") == "streaming" else ""
        samples = f" - first {ds.get('streaming_samples', 0):,} samples" if ds.get("streaming_samples") else ""
        print(f"   • {ds['name']}: {ds['description']} ({ds['size_gb']} GB{method}{samples})")
    
    print("\n" + "="*70)
    print("Download Options:")
    print("="*70)
    print(f"1. Essential only ({tier1_size:.1f} GB) - Fast, proven datasets")
    print(f"2. Essential + Recommended ({tier1_size + tier2_size:.1f} GB) - Best balance")
    print(f"3. All tiers ({tier1_size + tier2_size + tier3_size:.1f} GB) - Complete collection")
    print("4. Custom selection")
    print("5. Resume/retry failed downloads")
    print("0. Exit")
    print()


def main():
    """Main entry point."""
    
    # Create output directory
    output_base = Path("Z:\\training_datasets")
    output_base.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path("Z:\\huggingface_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_base}")
    print(f"📁 Cache directory: {cache_dir}")
    
    while True:
        show_menu()
        
        try:
            choice = input("Select option: ").strip()
            
            if choice == "0":
                print("\n👋 Exiting...")
                break
            
            datasets_to_download = []
            
            if choice == "1":
                datasets_to_download = [ds for ds in DATASET_CATALOG if ds["tier"] == 1]
            elif choice == "2":
                datasets_to_download = [ds for ds in DATASET_CATALOG if ds["tier"] <= 2]
            elif choice == "3":
                datasets_to_download = DATASET_CATALOG
            elif choice == "4":
                print("\nAvailable datasets:")
                for i, ds in enumerate(DATASET_CATALOG, 1):
                    print(f"{i}. {ds['name']} ({ds['size_gb']} GB)")
                
                indices = input("\nEnter dataset numbers (comma-separated): ").strip()
                try:
                    selected = [int(x.strip()) - 1 for x in indices.split(",")]
                    datasets_to_download = [DATASET_CATALOG[i] for i in selected if 0 <= i < len(DATASET_CATALOG)]
                except:
                    print("❌ Invalid selection")
                    continue
            elif choice == "5":
                print("\n🔄 Retrying all datasets...")
                datasets_to_download = DATASET_CATALOG
            else:
                print("❌ Invalid choice")
                continue
            
            if not datasets_to_download:
                print("❌ No datasets selected")
                continue
            
            # Download datasets
            total_size = sum(ds["size_gb"] for ds in datasets_to_download)
            print(f"\n{'='*70}")
            print(f"📦 Starting download of {len(datasets_to_download)} datasets ({total_size:.1f} GB)")
            print(f"{'='*70}")
            
            results = {
                "success": [],
                "failed": [],
                "skipped": []
            }
            
            for i, ds_info in enumerate(datasets_to_download, 1):
                print(f"\n[{i}/{len(datasets_to_download)}] Processing: {ds_info['name']}")
                
                output_path = get_output_path(ds_info["name"])
                if check_dataset_exists(output_path):
                    results["skipped"].append(ds_info["name"])
                    continue
                
                success = download_dataset(ds_info)
                
                if success:
                    results["success"].append(ds_info["name"])
                else:
                    results["failed"].append(ds_info["name"])
            
            # Summary
            print("\n" + "="*70)
            print("📊 DOWNLOAD SUMMARY")
            print("="*70)
            print(f"✅ Successful: {len(results['success'])}")
            for name in results["success"]:
                print(f"   • {name}")
            
            if results["skipped"]:
                print(f"\n⏭️  Skipped (already exists): {len(results['skipped'])}")
                for name in results["skipped"]:
                    print(f"   • {name}")
            
            if results["failed"]:
                print(f"\n❌ Failed: {len(results['failed'])}")
                for name in results["failed"]:
                    print(f"   • {name}")
            
            print("\n" + "="*70)
            
            # Ask to continue
            cont = input("\nDownload more datasets? (y/n): ").strip().lower()
            if cont != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Download interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Download session complete!")
    print(f"📁 Datasets saved to: Z:\\training_datasets")


if __name__ == "__main__":
    main()
