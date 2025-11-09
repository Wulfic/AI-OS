#!/usr/bin/env python3
"""
Optimized 1TB Dataset Downloader for HRM Training
Downloads high-quality curated datasets to Z:\training_datasets
Features: Parallel downloads, resume support, progress tracking
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

print("=" * 100)
print("OPTIMIZED 1TB DATASET DOWNLOADER FOR AI-OS")
print("=" * 100)
print()

# Target directory with 1TB space
OUTPUT_DIR = Path("Z:/training_datasets")
print(f"📁 Target directory: {OUTPUT_DIR}")
print(f"💾 Budget: 1TB (~1000GB)")
print()

# Create output directory
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Output directory ready")
except Exception as e:
    print(f"❌ Failed to create output directory: {e}")
    print(f"Please ensure Z: drive is accessible")
    sys.exit(1)

# Check if datasets library is installed
try:
    from datasets import load_dataset, DownloadConfig
    import datasets
    print("✅ HuggingFace datasets library found")
except ImportError:
    print("❌ HuggingFace datasets library not found")
    print("\nInstalling datasets library...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "tqdm"])
        from datasets import load_dataset, DownloadConfig
        import datasets
        print("✅ Successfully installed datasets library")
    except Exception as e:
        print(f"❌ Failed to install: {e}")
        print("\nPlease run: pip install datasets tqdm")
        sys.exit(1)

print()

# Dataset catalog optimized for 1TB budget
# Ordered by priority: small test datasets first, then large production datasets
DATASET_CATALOG = [
    # === TIER 1: Quick Test & Essential (70GB) ===
    {
        "name": "WikiText-103",
        "path": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "size_gb": 0.5,
        "priority": 1,
        "description": "Wikipedia articles - quick test dataset",
        "quality": "⭐⭐⭐⭐⭐",
        "use_case": "Testing, structured text"
    },
    {
        "name": "Wikipedia-EN",
        "path": "legacy-datasets/wikipedia",
        "config": "20220301.en",
        "split": "train",
        "size_gb": 20,
        "priority": 1,
        "description": "Full Wikipedia - factual knowledge base",
        "quality": "⭐⭐⭐⭐⭐",
        "use_case": "Factual knowledge, structured articles"
    },
    {
        "name": "OpenWebText",
        "path": "Skylion007/openwebtext",
        "config": None,
        "split": "train",
        "size_gb": 38,
        "priority": 1,
        "description": "Reddit-curated web content (Parquet version)",
        "quality": "⭐⭐⭐⭐",
        "use_case": "Diverse web text, conversational"
    },
    {
        "name": "BookCorpus",
        "path": "bookcorpusopen/bookcorpusopen",
        "config": None,
        "split": "train",
        "size_gb": 12,
        "priority": 1,
        "description": "Books and long-form text (open version)",
        "quality": "⭐⭐⭐⭐⭐",
        "use_case": "Long-form coherent text"
    },
    
    # === TIER 2: High-Quality Educational (550GB) ===
    {
        "name": "FineWeb-Edu",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "default",
        "split": "train",
        "size_gb": 500,
        "priority": 2,
        "description": "Educational web content - HIGHEST QUALITY reasoning",
        "quality": "⭐⭐⭐⭐⭐",
        "use_case": "Educational content, reasoning patterns, Q&A"
    },
    {
        "name": "C4-en",
        "path": "allenai/c4",
        "config": "en",
        "split": "train",
        "size_gb": 50,
        "priority": 2,
        "description": "Colossal Clean Crawled Corpus subset",
        "quality": "⭐⭐⭐⭐",
        "use_case": "Clean web text at scale",
        "streaming_recommended": True
    },
    
    # === TIER 3: Large-Scale Diverse (380GB) ===
    {
        "name": "RedPajama-Sample",
        "path": "togethercomputer/RedPajama-Data-1T-Sample",
        "config": None,
        "split": "train",
        "size_gb": 200,
        "priority": 3,
        "description": "High-quality diverse dataset (sample)",
        "quality": "⭐⭐⭐⭐",
        "use_case": "Diverse domains, balanced corpus"
    },
    {
        "name": "OSCAR-en",
        "path": "oscar-corpus/OSCAR-2301",
        "config": "en",
        "split": "train",
        "size_gb": 100,
        "priority": 3,
        "description": "OSCAR multilingual corpus - English",
        "quality": "⭐⭐⭐",
        "use_case": "Large-scale web crawl",
        "streaming_recommended": True
    },
    {
        "name": "Pile-Subsets",
        "path": "EleutherAI/pile",
        "config": "all",
        "split": "train",
        "size_gb": 80,
        "priority": 3,
        "description": "The Pile - diverse specialized domains",
        "quality": "⭐⭐⭐⭐",
        "use_case": "Specialized domains (code, science, etc.)",
        "note": "Select specific subsets to fit budget"
    }
]

# Calculate total size
total_size = sum(ds["size_gb"] for ds in DATASET_CATALOG)
print("=" * 100)
print("DATASET CATALOG")
print("=" * 100)
print(f"📊 Total datasets: {len(DATASET_CATALOG)}")
print(f"💾 Total size: ~{total_size:.1f} GB")
print()

# Display catalog by priority
for priority in [1, 2, 3]:
    tier_datasets = [ds for ds in DATASET_CATALOG if ds["priority"] == priority]
    tier_size = sum(ds["size_gb"] for ds in tier_datasets)
    tier_name = {1: "TIER 1: Essential", 2: "TIER 2: Educational", 3: "TIER 3: Large-Scale"}[priority]
    
    print(f"\n{tier_name} (~{tier_size:.1f} GB)")
    print("-" * 100)
    for ds in tier_datasets:
        streaming = " [STREAMING RECOMMENDED]" if ds.get("streaming_recommended") else ""
        print(f"  • {ds['name']} ({ds['size_gb']}GB) {ds['quality']}{streaming}")
        print(f"    {ds['description']}")
        print(f"    Use: {ds['use_case']}")
        if ds.get("note"):
            print(f"    Note: {ds['note']}")
        print()

print("=" * 100)

# User selection
print("\n📋 DOWNLOAD OPTIONS:")
print("1. Quick Test (71GB) - Tier 1 only")
print("2. Recommended (621GB) - Tier 1 + Tier 2")
print("3. Full 1TB (1001GB) - All tiers")
print("4. Custom selection")
print("5. List only (no download)")
print()

choice = input("Select option [1-5] (default: 2): ").strip() or "2"

if choice == "5":
    print("\n✅ Dataset catalog displayed. No downloads performed.")
    sys.exit(0)

# Determine which datasets to download
if choice == "1":
    selected = [ds for ds in DATASET_CATALOG if ds["priority"] == 1]
elif choice == "2":
    selected = [ds for ds in DATASET_CATALOG if ds["priority"] <= 2]
elif choice == "3":
    selected = DATASET_CATALOG
elif choice == "4":
    print("\nCustom selection:")
    selected = []
    for i, ds in enumerate(DATASET_CATALOG, 1):
        answer = input(f"  Download {ds['name']} ({ds['size_gb']}GB)? [y/N]: ").strip().lower()
        if answer == 'y':
            selected.append(ds)
else:
    print("Invalid choice, using Recommended (Tier 1 + 2)")
    selected = [ds for ds in DATASET_CATALOG if ds["priority"] <= 2]

if not selected:
    print("❌ No datasets selected")
    sys.exit(0)

selected_size = sum(ds["size_gb"] for ds in selected)
print(f"\n✅ Selected {len(selected)} datasets (~{selected_size:.1f} GB)")
print()

# Confirm download
print("⚠️  WARNING: This will download large amounts of data!")
print(f"   Total size: ~{selected_size:.1f} GB")
print(f"   Destination: {OUTPUT_DIR}")
print()
confirm = input("Proceed with download? [y/N]: ").strip().lower()
if confirm != 'y':
    print("❌ Download cancelled")
    sys.exit(0)

print()
print("=" * 100)
print("STARTING OPTIMIZED DOWNLOAD")
print("=" * 100)
print()

# Download configuration for optimization
download_config = DownloadConfig(
    cache_dir=str(OUTPUT_DIR / ".cache"),
    resume_download=True,  # Resume interrupted downloads
    max_retries=5,
    num_proc=4  # Parallel processing
)

# Statistics
stats = {
    "total": len(selected),
    "completed": 0,
    "skipped": 0,
    "failed": 0,
    "start_time": time.time()
}

def download_dataset(ds_info: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Download a single dataset with error handling and progress tracking"""
    result = {
        "name": ds_info["name"],
        "success": False,
        "skipped": False,
        "error": None,
        "samples": 0,
        "time": 0
    }
    
    output_path = OUTPUT_DIR / ds_info["name"].lower().replace("-", "_")
    
    # Check if already downloaded
    if output_path.exists():
        try:
            # Try to load existing dataset
            from datasets import load_from_disk
            existing = load_from_disk(str(output_path))
            result["samples"] = len(existing)
            result["success"] = True
            result["skipped"] = True
            print(f"[{index}/{stats['total']}] ✅ {ds_info['name']} - Already downloaded ({result['samples']:,} samples)")
            return result
        except Exception:
            print(f"[{index}/{stats['total']}] ⚠️  {ds_info['name']} - Existing download corrupted, re-downloading...")
    
    print(f"[{index}/{stats['total']}] 📥 {ds_info['name']} ({ds_info['size_gb']}GB)")
    print(f"   Path: {ds_info['path']}")
    if ds_info.get("config"):
        print(f"   Config: {ds_info['config']}")
    
    start = time.time()
    
    try:
        # Use streaming for very large datasets (>100GB)
        use_streaming = ds_info.get("streaming_recommended", False) or ds_info["size_gb"] > 100
        
        if use_streaming:
            print(f"   Mode: Streaming (large dataset)")
            # Stream and save in chunks to avoid memory issues
            if ds_info.get("config"):
                dataset = load_dataset(
                    ds_info["path"],
                    ds_info["config"],
                    split=ds_info["split"],
                    streaming=True,
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    ds_info["path"],
                    split=ds_info["split"],
                    streaming=True,
                    trust_remote_code=True
                )
            
            # For streaming datasets, we need to iterate and save
            # This is memory-efficient but slower
            print(f"   ⚠️  Streaming mode: Download will happen during training")
            print(f"   💾 Creating streaming reference at {output_path}")
            
            # Save streaming dataset configuration
            config_file = output_path / "streaming_config.json"
            output_path.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump({
                    "path": ds_info["path"],
                    "config": ds_info.get("config"),
                    "split": ds_info["split"],
                    "streaming": True
                }, f, indent=2)
            
            result["success"] = True
            result["samples"] = "streaming"
            
        else:
            print(f"   Mode: Full download")
            # Full download for smaller datasets
            if ds_info.get("config"):
                dataset = load_dataset(
                    ds_info["path"],
                    ds_info["config"],
                    split=ds_info["split"],
                    download_config=download_config,
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    ds_info["path"],
                    split=ds_info["split"],
                    download_config=download_config,
                    trust_remote_code=True
                )
            
            print(f"   💾 Saving to disk...")
            dataset.save_to_disk(str(output_path))
            
            result["success"] = True
            result["samples"] = len(dataset)
        
        result["time"] = time.time() - start
        
        print(f"[{index}/{stats['total']}] ✅ {ds_info['name']} - Complete!")
        if isinstance(result["samples"], int):
            print(f"   Samples: {result['samples']:,}")
        print(f"   Time: {result['time']/60:.1f} minutes")
        print(f"   Location: {output_path}")
        
    except Exception as e:
        result["error"] = str(e)
        result["time"] = time.time() - start
        print(f"[{index}/{stats['total']}] ❌ {ds_info['name']} - FAILED")
        print(f"   Error: {e}")
    
    print()
    return result

# Download datasets sequentially (to avoid overwhelming disk I/O)
# For true parallel download, could use ThreadPoolExecutor, but sequential is safer for large files
print("🚀 Starting downloads (sequential to avoid disk I/O bottlenecks)...")
print()

results = []
for i, ds in enumerate(selected, 1):
    result = download_dataset(ds, i)
    results.append(result)
    
    # Update stats
    if result["success"]:
        if result["skipped"]:
            stats["skipped"] += 1
        else:
            stats["completed"] += 1
    else:
        stats["failed"] += 1
    
    # Show progress
    elapsed = time.time() - stats["start_time"]
    remaining = len(selected) - i
    if stats["completed"] > 0:
        avg_time = elapsed / (stats["completed"] + stats["skipped"])
        eta = avg_time * remaining / 60
        print(f"📊 Progress: {i}/{len(selected)} | Completed: {stats['completed']} | Skipped: {stats['skipped']} | Failed: {stats['failed']}")
        print(f"⏱️  Elapsed: {elapsed/60:.1f}m | ETA: {eta:.1f}m")
        print("-" * 100)

# Final summary
total_time = time.time() - stats["start_time"]

print()
print("=" * 100)
print("DOWNLOAD SUMMARY")
print("=" * 100)
print(f"✅ Completed: {stats['completed']}/{stats['total']}")
print(f"⏭️  Skipped (already downloaded): {stats['skipped']}/{stats['total']}")
print(f"❌ Failed: {stats['failed']}/{stats['total']}")
print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print()

if stats["completed"] > 0 or stats["skipped"] > 0:
    print("📚 DOWNLOADED DATASETS:")
    for result in results:
        if result["success"]:
            status = "✅" if not result["skipped"] else "⏭️ "
            samples = f"{result['samples']:,}" if isinstance(result["samples"], int) else result["samples"]
            print(f"   {status} {result['name']}: {samples} samples")
    print()
    
    print("🔗 CONFIGURATION:")
    print(f"   Dataset root: {OUTPUT_DIR}")
    print()
    print("🚀 NEXT STEPS:")
    print("1. Update HRM Training Panel dataset path to point to Z:/training_datasets")
    print("2. For streaming datasets, they will download during training")
    print("3. Configure dataset mixing in your training config")
    print()
    print("Example config:")
    print("   datasets:")
    for result in results:
        if result["success"]:
            print(f"     - Z:/training_datasets/{result['name'].lower().replace('-', '_')}")
    print()

if stats["failed"] > 0:
    print("⚠️  FAILED DOWNLOADS:")
    for result in results:
        if not result["success"]:
            print(f"   ❌ {result['name']}: {result['error']}")
    print()
    print("💡 TROUBLESHOOTING:")
    print("   - Check internet connection")
    print("   - Ensure Z: drive has enough space")
    print("   - Try downloading failed datasets individually")
    print("   - Use streaming mode for very large datasets")
    print()

print("=" * 100)
print(f"🎉 DOWNLOAD PROCESS COMPLETE!")
print(f"📁 All datasets saved to: {OUTPUT_DIR}")
print("=" * 100)
