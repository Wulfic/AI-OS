"""
Quick test to verify Hugging Face cache environment variables are set correctly.
Run this script to check if HF will download to the correct location.
"""

import os
from pathlib import Path

print("=" * 60)
print("Hugging Face Cache Configuration Check")
print("=" * 60)
print()

# Check environment variables
hf_home = os.environ.get("HF_HOME")
hf_datasets = os.environ.get("HF_DATASETS_CACHE")
hf_transformers = os.environ.get("TRANSFORMERS_CACHE")
hf_hub = os.environ.get("HF_HUB_CACHE")

print("Environment Variables:")
print(f"  HF_HOME: {hf_home or '❌ NOT SET'}")
print(f"  HF_DATASETS_CACHE: {hf_datasets or '❌ NOT SET'}")
print(f"  TRANSFORMERS_CACHE: {hf_transformers or '❌ NOT SET'}")
print(f"  HF_HUB_CACHE: {hf_hub or '❌ NOT SET'}")
print()

# Check if paths are on C drive
c_drive_issue = False
for var_name, var_value in [
    ("HF_HOME", hf_home),
    ("HF_DATASETS_CACHE", hf_datasets),
    ("TRANSFORMERS_CACHE", hf_transformers),
    ("HF_HUB_CACHE", hf_hub),
]:
    if var_value and Path(var_value).drive == "C:":
        print(f"⚠️  WARNING: {var_name} is on C drive: {var_value}")
        c_drive_issue = True

if c_drive_issue:
    print()
    print("❌ ISSUE DETECTED: Some cache paths are still on C drive!")
    print()
    print("To fix this:")
    print("  1. Close this terminal and VS Code")
    print("  2. Open a NEW PowerShell window")
    print("  3. Run: .\\set_hf_cache_permanent.ps1")
    print("  4. Restart VS Code")
elif not hf_home:
    print("❌ Environment variables are NOT set!")
    print()
    print("To fix this:")
    print("  1. Run: .\\set_hf_cache_permanent.ps1")
    print("  2. Restart VS Code and terminals")
else:
    print("✅ All environment variables are set correctly!")
    print()
    print(f"HuggingFace will download to: {hf_home}")

print()

# Try importing datasets library to see what it will use
try:
    print("Testing datasets library import...")
    from datasets import config
    
    cache_dir = config.HF_DATASETS_CACHE
    print(f"  Datasets cache dir: {cache_dir}")
    
    if Path(cache_dir).drive == "C:":
        print("  ❌ WARNING: datasets library will still use C drive!")
        print("     Please restart VS Code for changes to take effect.")
    else:
        print("  ✅ datasets library is correctly configured!")
except ImportError:
    print("  ℹ️  datasets library not installed yet")
except Exception as e:
    print(f"  ⚠️  Error checking datasets config: {e}")

print()
print("=" * 60)
