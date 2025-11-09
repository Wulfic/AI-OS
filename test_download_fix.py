#!/usr/bin/env python3
"""
Quick test script to verify downloads go to Z drive only
"""

import os
import sys
from pathlib import Path
import time

# CRITICAL: Set cache directories BEFORE any imports
BASE_DOWNLOAD_PATH = Path("Z:\\training_datasets")
CACHE_DIR = BASE_DOWNLOAD_PATH / ".cache"
TEMP_DIR = BASE_DOWNLOAD_PATH / ".temp"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR / "datasets")
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / "transformers")
os.environ['HF_HUB_CACHE'] = str(CACHE_DIR / "hub")
os.environ['TMPDIR'] = str(TEMP_DIR)
os.environ['TEMP'] = str(TEMP_DIR)
os.environ['TMP'] = str(TEMP_DIR)

from datasets import load_dataset, DownloadConfig
from tqdm import tqdm

def check_c_drive():
    """Check if any files were created on C drive."""
    c_cache = Path.home() / ".cache" / "huggingface"
    if c_cache.exists():
        files = list(c_cache.rglob("*"))
        if files:
            print(f"\n⚠️ WARNING: {len(files)} files found on C drive!")
            return False
    print("\n✅ C drive is clean!")
    return True

def test_download():
    print("=" * 80)
    print("🧪 Testing Download Fix - HumanEval (tiny dataset)")
    print("=" * 80)
    
    print(f"\n📂 Configuration:")
    print(f"   Base Path: {BASE_DOWNLOAD_PATH}")
    print(f"   Cache: {CACHE_DIR}")
    print(f"   Temp: {TEMP_DIR}")
    
    print("\n📥 Downloading HumanEval (0.001 GB)...")
    
    output_path = BASE_DOWNLOAD_PATH / "test_humaneval"
    
    try:
        download_config = DownloadConfig(
            resume_download=True,
            max_retries=3,
            cache_dir=str(CACHE_DIR),
        )
        
        print("   ⏳ Loading dataset...")
        start = time.time()
        
        dataset = load_dataset(
            "openai_humaneval",
            split="test",
            download_config=download_config,
            cache_dir=str(CACHE_DIR),
        )
        
        duration = time.time() - start
        
        print(f"   💾 Saving to disk...")
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        
        size_mb = sum(f.stat().st_size for f in output_path.glob("*.arrow")) / (1024 * 1024)
        print(f"   ✅ SUCCESS: {len(dataset)} samples, {size_mb:.1f} MB ({duration:.1f}s)")
        
        # Check C drive
        print("\n🔍 Checking C drive for leaked files...")
        if check_c_drive():
            print("✅ TEST PASSED: All files saved to Z drive!")
            return True
        else:
            print("❌ TEST FAILED: Files leaked to C drive")
            return False
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_download()
    sys.exit(0 if success else 1)
