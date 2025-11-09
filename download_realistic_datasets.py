#!/usr/bin/env python3
"""
Realistic Dataset Downloader - Small, manageable datasets that actually fit!

This downloads SMALL SAMPLES suitable for training - not multi-terabyte monsters.
Total: ~25GB of high-quality, verified datasets.
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


# REALISTIC, MANAGEABLE DATASETS
# All datasets are either small or we download a limited number of samples
DATASETS = [
    {
        "name": "WikiText-103",
        "path": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "size_gb": 0.5,
        "description": "Wikipedia articles - language modeling",
    },
    {
        "name": "TinyStories",
        "path": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "size_gb": 2.5,
        "description": "Simple stories - great baseline",
        "streaming": True,
        "max_samples": 500000,  # Limit to 500k samples (~2GB)
    },
    {
        "name": "C4-sample-10k",
        "path": "allenai/c4",
        "config": "en",
        "split": "train",
        "size_gb": 3.0,
        "description": "C4 web text (10k samples only)",
        "streaming": True,
        "max_samples": 10000,  # Just 10k samples! Not the full 3TB
    },
    {
        "name": "OpenWebText-sample",
        "path": "Skylion007/openwebtext",
        "config": None,
        "split": "train",
        "size_gb": 5.0,
        "description": "Reddit content (50k samples)",
        "streaming": True,
        "max_samples": 50000,  # Manageable sample
    },
    {
        "name": "FineWeb-Edu-sample-10BT",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "size_gb": 10.0,
        "description": "Educational content (10BT sample)",
    },
    {
        "name": "BookCorpus-sample",
        "path": "bookcorpus",
        "config": "plain_text",
        "split": "train",
        "size_gb": 3.0,
        "description": "Books corpus (streaming sample)",
        "streaming": True,
        "max_samples": 30000,  # 30k books samples
    },
]


def download_streaming(ds_info: Dict[str, Any], output_path: Path) -> bool:
    """Download using streaming to get a manageable sample."""
    try:
        max_samples = ds_info.get("max_samples", 10000)
        
        print(f"   📦 Streaming up to {max_samples:,} samples...")
        
        load_kwargs = {
            "path": ds_info["path"],
            "split": ds_info["split"],
            "streaming": True,
            "trust_remote_code": False,
        }
        
        if ds_info.get("config"):
            load_kwargs["name"] = ds_info["config"]
        
        start = time.time()
        dataset_stream = load_dataset(**load_kwargs)
        
        # Collect samples with progress
        samples = []
        last_update = 0
        for i, sample in enumerate(dataset_stream):
            samples.append(sample)
            
            # Update every 1000 samples
            if i - last_update >= 1000:
                elapsed = time.time() - start
                rate = (i + 1) / max(elapsed, 0.1)
                print(f"\r   ⏳ {i+1:,}/{max_samples:,} samples ({rate:.0f} samples/sec)", end="", flush=True)
                last_update = i
            
            if i + 1 >= max_samples:
                break
        
        print()  # New line after progress
        
        # Convert to Dataset and save
        from datasets import Dataset
        dataset = Dataset.from_dict({
            key: [s[key] for s in samples]
            for key in samples[0].keys()
        })
        
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        
        duration = time.time() - start
        size_mb = sum(f.stat().st_size for f in output_path.glob("*.arrow")) / (1024 * 1024)
        print(f"   ✅ SUCCESS: {len(samples):,} samples, {size_mb:.1f} MB ({duration:.1f}s)")
        return True
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def download_direct(ds_info: Dict[str, Any], output_path: Path) -> bool:
    """Download small dataset directly."""
    try:
        print(f"   ⏳ Downloading full dataset...")
        
        download_config = DownloadConfig(
            resume_download=True,
            max_retries=5,
        )
        
        load_kwargs = {
            "path": ds_info["path"],
            "split": ds_info["split"],
            "download_config": download_config,
            "trust_remote_code": False,
        }
        
        if ds_info.get("config"):
            load_kwargs["name"] = ds_info["config"]
        
        start = time.time()
        dataset = load_dataset(**load_kwargs)
        
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        
        duration = time.time() - start
        size_mb = sum(f.stat().st_size for f in output_path.glob("*.arrow")) / (1024 * 1024)
        print(f"   ✅ SUCCESS: {len(dataset):,} samples, {size_mb:.1f} MB ({duration:.1f}s)")
        return True
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def download_dataset(ds_info: Dict[str, Any]) -> bool:
    """Download a single dataset."""
    output_path = Path("Z:\\training_datasets") / ds_info["name"].lower().replace(" ", "_").replace("-", "_")
    
    # Check if exists
    if output_path.exists():
        data_files = list(output_path.glob("*.arrow")) + list(output_path.glob("data/*.arrow"))
        if data_files:
            size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
            print(f"   ✅ Already downloaded: {size_mb:.1f} MB")
            return True
    
    print(f"\n📥 {ds_info['name']} (~{ds_info['size_gb']} GB)")
    print(f"   {ds_info['description']}")
    
    if ds_info.get("streaming", False):
        return download_streaming(ds_info, output_path)
    else:
        return download_direct(ds_info, output_path)


def main():
    print("=" * 70)
    print("🚀 Realistic Dataset Downloader")
    print("=" * 70)
    print("\nDownloads MANAGEABLE samples - not terabytes of data!")
    print(f"\nTotal size: ~{sum(ds['size_gb'] for ds in DATASETS):.1f} GB\n")
    
    for i, ds in enumerate(DATASETS, 1):
        method = " [streaming sample]" if ds.get("streaming") else " [full download]"
        max_samples = f" - {ds.get('max_samples', 0):,} samples" if ds.get("max_samples") else ""
        print(f"{i}. {ds['name']}: {ds['description']}")
        print(f"   {ds['size_gb']} GB{method}{max_samples}")
    
    print("\n" + "=" * 70)
    input("Press Enter to start downloading (Ctrl+C to cancel)...")
    
    print(f"\n{'='*70}")
    print(f"📥 Starting downloads...")
    print(f"{'='*70}")
    
    results = {"success": [], "failed": [], "skipped": []}
    
    for i, ds_info in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] {ds_info['name']}")
        
        output_path = Path("Z:\\training_datasets") / ds_info["name"].lower().replace(" ", "_").replace("-", "_")
        if output_path.exists():
            data_files = list(output_path.glob("*.arrow")) + list(output_path.glob("data/*.arrow"))
            if data_files:
                results["skipped"].append(ds_info["name"])
                size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
                print(f"   ⏭️  Skipped (exists): {size_mb:.1f} MB")
                continue
        
        success = download_dataset(ds_info)
        if success:
            results["success"].append(ds_info["name"])
        else:
            results["failed"].append(ds_info["name"])
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 70)
    
    if results["success"]:
        print(f"\n✅ Downloaded: {len(results['success'])}")
        for name in results["success"]:
            print(f"   • {name}")
    
    if results["skipped"]:
        print(f"\n⏭️  Skipped: {len(results['skipped'])}")
        for name in results["skipped"]:
            print(f"   • {name}")
    
    if results["failed"]:
        print(f"\n❌ Failed: {len(results['failed'])}")
        for name in results["failed"]:
            print(f"   • {name}")
    
    # Show final directory contents
    print(f"\n📁 Available datasets in Z:\\training_datasets:")
    for path in sorted(Path("Z:\\training_datasets").iterdir()):
        if path.is_dir():
            data_files = list(path.glob("*.arrow")) + list(path.glob("data/*.arrow"))
            if data_files:
                size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
                sample_count = "?"
                try:
                    from datasets import load_from_disk
                    ds = load_from_disk(str(path))
                    sample_count = f"{len(ds):,}"
                except:
                    pass
                print(f"   • {path.name}: {size_mb:.1f} MB ({sample_count} samples)")
    
    print(f"\n✅ Ready to train! Use these datasets in the HRM Training panel.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
