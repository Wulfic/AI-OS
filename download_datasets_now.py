#!/usr/bin/env python3
"""
Download curated datasets for HRM training
This script downloads WikiText, Wikipedia, and FineWeb-Edu datasets
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("CURATED DATASETS DOWNLOADER")
print("=" * 80)
print()

# Check if datasets library is installed
try:
    from datasets import load_dataset
    print("✅ HuggingFace datasets library found")
except ImportError:
    print("❌ HuggingFace datasets library not found")
    print("\nInstalling datasets library...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset
        print("✅ Successfully installed datasets library")
    except Exception as e:
        print(f"❌ Failed to install datasets library: {e}")
        print("\nPlease run: pip install datasets")
        sys.exit(1)

# Create output directory
output_dir = Path("training_data/curated_datasets")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"✅ Output directory: {output_dir}")
print()

# Download datasets
datasets_to_download = [
    {
        "name": "WikiText-103",
        "path": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "size": "~500MB",
        "output": output_dir / "wikitext",
        "description": "Quick test dataset - Wikipedia articles"
    },
    {
        "name": "Wikipedia",
        "path": "wikipedia",
        "config": "20220301.en",
        "split": "train",
        "size": "~20GB",
        "output": output_dir / "wikipedia",
        "description": "Full Wikipedia - factual knowledge base"
    },
    {
        "name": "FineWeb-Edu",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "size": "~30GB",
        "output": output_dir / "fineweb_edu",
        "description": "Educational content - reasoning patterns"
    }
]

print("📦 DATASETS TO DOWNLOAD:")
print("-" * 80)
for i, ds in enumerate(datasets_to_download, 1):
    print(f"{i}. {ds['name']} ({ds['size']})")
    print(f"   {ds['description']}")
    print(f"   Output: {ds['output']}")
    print()

print("=" * 80)
print("STARTING DOWNLOADS")
print("=" * 80)
print()

success_count = 0
failed_count = 0

for i, ds in enumerate(datasets_to_download, 1):
    print(f"\n[{i}/{len(datasets_to_download)}] Downloading {ds['name']} ({ds['size']})...")
    print("-" * 80)
    
    # Check if already downloaded
    if ds['output'].exists():
        try:
            existing = load_dataset(str(ds['output']), split="train")
            print(f"✅ Already downloaded: {len(existing):,} samples")
            print(f"   Skipping download (delete {ds['output']} to re-download)")
            success_count += 1
            continue
        except Exception:
            print(f"⚠️  Existing download corrupted, re-downloading...")
    
    try:
        print(f"📥 Downloading from HuggingFace...")
        print(f"   Dataset: {ds['path']}")
        if ds.get('config'):
            print(f"   Config: {ds['config']}")
        print(f"   Split: {ds['split']}")
        print()
        
        # Download with progress bar
        if ds.get('config'):
            dataset = load_dataset(
                ds['path'],
                ds['config'],
                split=ds['split'],
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                ds['path'],
                split=ds['split'],
                trust_remote_code=True
            )
        
        print(f"\n💾 Saving to disk...")
        dataset.save_to_disk(str(ds['output']))
        
        print(f"✅ SUCCESS: {ds['name']} downloaded!")
        print(f"   Samples: {len(dataset):,}")
        print(f"   Location: {ds['output']}")
        success_count += 1
        
    except Exception as e:
        print(f"❌ FAILED: {ds['name']}")
        print(f"   Error: {e}")
        failed_count += 1
        
        # Continue with next dataset
        continue

print()
print("=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"✅ Successful: {success_count}/{len(datasets_to_download)}")
print(f"❌ Failed: {failed_count}/{len(datasets_to_download)}")
print()

if success_count > 0:
    print("📚 DOWNLOADED DATASETS:")
    for ds in datasets_to_download:
        if ds['output'].exists():
            print(f"   ✅ {ds['name']}: {ds['output']}")
    print()
    
    print("🚀 NEXT STEPS:")
    print("1. Open HRM Training Panel in GUI")
    print("2. Configure dataset path:")
    print(f"   - Quick test: {output_dir / 'wikitext'}")
    print(f"   - Production: {output_dir / 'wikipedia'} or {output_dir / 'fineweb_edu'}")
    print("3. DISABLE 'Teacher-as-dataset' checkbox")
    print("4. Click 'Train' to start training with curated data!")
    print()

if failed_count > 0:
    print("⚠️  TROUBLESHOOTING:")
    print("- Check internet connection")
    print("- Ensure enough disk space (50-90GB needed)")
    print("- Try running again (downloads will resume)")
    print("- For streaming without download: use load_dataset(..., streaming=True)")
    print()

print("=" * 80)
print("COMPLETE!")
print("=" * 80)
