#!/usr/bin/env python3
"""
Quick verification test for C drive temp and token fixes
"""

import os
import sys
from pathlib import Path
import tempfile

# Check 1: Verify temp directory configuration
print("=" * 80)
print("🔍 TEMP DIRECTORY VERIFICATION")
print("=" * 80)

print("\n1. Environment Variables:")
print(f"   TEMP: {os.environ.get('TEMP', 'NOT SET')}")
print(f"   TMP: {os.environ.get('TMP', 'NOT SET')}")
print(f"   TMPDIR: {os.environ.get('TMPDIR', 'NOT SET')}")
print(f"   TEMPDIR: {os.environ.get('TEMPDIR', 'NOT SET')}")

print("\n2. Python tempfile module:")
print(f"   tempfile.tempdir: {tempfile.tempdir}")
print(f"   tempfile.gettempdir(): {tempfile.gettempdir()}")

print("\n3. Expected:")
print("   All should point to Z:\\training_datasets\\.temp")

# Check if pointing to C drive
c_drive_detected = False
for key, val in [("TEMP", os.environ.get('TEMP')), 
                  ("TMP", os.environ.get('TMP')),
                  ("tempfile.tempdir", tempfile.tempdir),
                  ("gettempdir()", tempfile.gettempdir())]:
    if val and str(val).lower().startswith('c:'):
        print(f"\n   ❌ {key} is still pointing to C drive: {val}")
        c_drive_detected = True

if not c_drive_detected:
    print("\n   ✅ All temp directories correctly configured!")
else:
    print("\n   ⚠️  WARNING: Some temp dirs still point to C drive!")

# Check 2: HuggingFace token
print("\n" + "=" * 80)
print("🔑 HUGGINGFACE TOKEN VERIFICATION")
print("=" * 80)

try:
    from huggingface_hub import HfFolder
    token = HfFolder.get_token()
    if token:
        print("✅ HuggingFace token found in default location")
        print(f"   Token: {token[:10]}..." if len(token) > 10 else f"   Token: {token}")
    else:
        print("⚠️  No token in default HuggingFace location")
except Exception as e:
    print(f"❌ Error checking token: {e}")

# Check backup token
backup_token_path = Path("Z:\\training_datasets\\.cache\\.hf_token")
if backup_token_path.exists():
    try:
        backup_token = backup_token_path.read_text().strip()
        print(f"✅ Backup token found: {backup_token_path}")
        print(f"   Token: {backup_token[:10]}..." if len(backup_token) > 10 else f"   Token: {backup_token}")
    except Exception as e:
        print(f"⚠️  Backup token file exists but can't read: {e}")
else:
    print("⚠️  No backup token file found")
    print(f"   Expected at: {backup_token_path}")

# Check 3: C drive scan
print("\n" + "=" * 80)
print("🧹 C DRIVE SCAN")
print("=" * 80)

c_locations = [
    Path.home() / ".cache" / "huggingface",
    Path("C:\\Users") / os.environ.get('USERNAME', '') / "AppData" / "Local" / "Temp",
    Path("C:\\Temp"),
    Path("C:\\tmp"),
]

hf_files_found = False
for location in c_locations:
    if not location.exists():
        continue
    
    try:
        # Quick scan for HF files
        hf_files = []
        for pattern in ["*huggingface*", "*datasets*", "*.arrow", "*hf_*"]:
            hf_files.extend(list(location.glob(pattern)))
            if len(hf_files) > 0:
                break
        
        if hf_files:
            total_size = sum(f.stat().st_size for f in hf_files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"\n⚠️  Found {len(hf_files)} HF-related files in {location}")
            print(f"   Size: {size_mb:.1f} MB")
            hf_files_found = True
    except Exception as e:
        pass

if not hf_files_found:
    print("\n✅ C drive is clean - no HuggingFace files found!")

# Check 4: Download script exists
print("\n" + "=" * 80)
print("📄 DOWNLOAD SCRIPT CHECK")
print("=" * 80)

script_path = Path("download_english_and_code_datasets.py")
if script_path.exists():
    print(f"✅ Download script found: {script_path}")
    
    # Check for key fixes in the script
    content = script_path.read_text()
    
    checks = {
        "tempfile.tempdir": "tempfile.tempdir" in content,
        "HfFolder.save_token": "HfFolder.save_token" in content,
        "cleanup_c_drive_cache": "cleanup_c_drive_cache" in content,
        ".hf_token": ".hf_token" in content,
    }
    
    print("\n   Key fixes present:")
    for check_name, present in checks.items():
        status = "✅" if present else "❌"
        print(f"   {status} {check_name}")
    
    all_present = all(checks.values())
    if all_present:
        print("\n   ✅ All fixes implemented!")
    else:
        print("\n   ⚠️  Some fixes might be missing")
else:
    print(f"❌ Download script not found: {script_path}")

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

issues = []
if c_drive_detected:
    issues.append("Temp directories still point to C drive")
if hf_files_found:
    issues.append("HuggingFace files found on C drive")

if issues:
    print("\n⚠️  Issues detected:")
    for issue in issues:
        print(f"   - {issue}")
    print("\n   Run download_english_and_code_datasets.py to fix")
else:
    print("\n✅ Everything looks good!")
    print("   - Temp dirs configured correctly")
    print("   - C drive is clean")
    print("   - Ready to download!")

print("\n" + "=" * 80)
