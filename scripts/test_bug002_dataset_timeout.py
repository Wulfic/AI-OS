#!/usr/bin/env python3
"""
Test script for BUG-002: Dataset Loading Timeout Fix

Tests the timeout mechanism and streaming for large dataset files.
"""

import os
import sys
import time
import tempfile
import csv
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aios.data.datasets import read_text_lines_sample, read_text_lines_sample_any


def create_test_files(temp_dir: Path):
    """Create test dataset files of various sizes and formats."""
    
    print("Creating test files...")
    
    # 1. Small text file (should work fast)
    small_txt = temp_dir / "small.txt"
    with small_txt.open("w") as f:
        for i in range(100):
            f.write(f"Line {i}: This is test data\n")
    print(f"✓ Created {small_txt.name} (100 lines)")
    
    # 2. Large text file (5MB)
    large_txt = temp_dir / "large.txt"
    with large_txt.open("w") as f:
        for i in range(50000):
            f.write(f"Line {i}: " + "x" * 100 + "\n")
    size_mb = large_txt.stat().st_size / (1024**2)
    print(f"✓ Created {large_txt.name} ({size_mb:.2f} MB, 50K lines)")
    
    # 3. CSV file
    csv_file = temp_dir / "test.csv"
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])
        for i in range(10000):
            writer.writerow([i, f"Sample text {i} with some content", i % 5])
    size_mb = csv_file.stat().st_size / (1024**2)
    print(f"✓ Created {csv_file.name} ({size_mb:.2f} MB, 10K rows)")
    
    # 4. JSONL file
    jsonl_file = temp_dir / "test.jsonl"
    with jsonl_file.open("w") as f:
        for i in range(5000):
            json.dump({"id": i, "text": f"JSON text {i}", "metadata": {"index": i}}, f)
            f.write("\n")
    size_mb = jsonl_file.stat().st_size / (1024**2)
    print(f"✓ Created {jsonl_file.name} ({size_mb:.2f} MB, 5K objects)")
    
    # 5. Very large file for timeout test (will create slowly)
    huge_txt = temp_dir / "huge.txt"
    print(f"Creating {huge_txt.name} (this will take a moment)...")
    with huge_txt.open("w") as f:
        for i in range(200000):  # 200K lines, ~20MB
            f.write(f"Line {i}: " + "y" * 100 + "\n")
    size_mb = huge_txt.stat().st_size / (1024**2)
    print(f"✓ Created {huge_txt.name} ({size_mb:.2f} MB, 200K lines)")
    
    return {
        "small": small_txt,
        "large": large_txt,
        "csv": csv_file,
        "jsonl": jsonl_file,
        "huge": huge_txt,
    }


def test_normal_loading(files):
    """Test normal loading without timeout."""
    print("\n" + "="*60)
    print("TEST 1: Normal Loading (No Timeout)")
    print("="*60)
    
    for name, path in files.items():
        if name == "huge":  # Skip huge file for normal test
            continue
        
        print(f"\nLoading {name} file: {path.name}")
        start = time.time()
        try:
            lines = read_text_lines_sample_any(path, max_lines=1000)
            elapsed = time.time() - start
            print(f"✅ SUCCESS: Loaded {len(lines)} lines in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ FAILED: {e} (after {elapsed:.2f}s)")


def test_timeout_mechanism(files):
    """Test timeout with very short timeout."""
    print("\n" + "="*60)
    print("TEST 2: Timeout Mechanism")
    print("="*60)
    
    # Test with huge file and short timeout
    huge_file = files["huge"]
    short_timeout = 2  # 2 seconds (should timeout for 200K lines)
    
    print(f"\nLoading huge file with {short_timeout}s timeout...")
    print(f"File: {huge_file.name} ({huge_file.stat().st_size / (1024**2):.2f} MB)")
    
    start = time.time()
    try:
        lines = read_text_lines_sample(huge_file, max_lines=200000, timeout=short_timeout)
        elapsed = time.time() - start
        print(f"⚠️  UNEXPECTED: Loaded {len(lines)} lines in {elapsed:.2f}s (expected timeout)")
    except Exception as e:
        elapsed = time.time() - start
        if "timeout" in str(e).lower():
            print(f"✅ SUCCESS: Timeout triggered after {elapsed:.2f}s")
            print(f"   Error message: {str(e)[:100]}...")
        else:
            print(f"❌ FAILED: Wrong error type: {e}")


def test_large_file_streaming(files):
    """Test streaming for large files."""
    print("\n" + "="*60)
    print("TEST 3: Large File Streaming")
    print("="*60)
    
    # Test CSV streaming
    csv_file = files["csv"]
    print(f"\nStreaming CSV file: {csv_file.name}")
    start = time.time()
    try:
        lines = read_text_lines_sample_any(csv_file, max_lines=5000)
        elapsed = time.time() - start
        print(f"✅ SUCCESS: Streamed {len(lines)} lines in {elapsed:.2f}s")
        if lines:
            print(f"   First line: {lines[0][:60]}...")
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FAILED: {e} (after {elapsed:.2f}s)")


def test_encoding_fallback(temp_dir):
    """Test encoding error handling."""
    print("\n" + "="*60)
    print("TEST 4: Encoding Error Handling")
    print("="*60)
    
    # Create file with latin-1 encoding
    latin_file = temp_dir / "latin1.txt"
    with latin_file.open("w", encoding="latin-1") as f:
        f.write("Normal text\n")
        f.write("Text with special chars: café, naïve, résumé\n")
        for i in range(100):
            f.write(f"Line {i}\n")
    
    print(f"\nLoading Latin-1 encoded file: {latin_file.name}")
    start = time.time()
    try:
        lines = read_text_lines_sample(latin_file, max_lines=100)
        elapsed = time.time() - start
        print(f"✅ SUCCESS: Loaded {len(lines)} lines in {elapsed:.2f}s")
        if len(lines) >= 2:
            print(f"   Line with special chars: {lines[1][:60]}...")
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FAILED: {e} (after {elapsed:.2f}s)")


def test_environment_variable():
    """Test AIOS_DATASET_LOAD_TIMEOUT environment variable."""
    print("\n" + "="*60)
    print("TEST 5: Environment Variable Configuration")
    print("="*60)
    
    # Check if env var is respected
    original = os.environ.get("AIOS_DATASET_LOAD_TIMEOUT")
    
    try:
        os.environ["AIOS_DATASET_LOAD_TIMEOUT"] = "10"
        print("Set AIOS_DATASET_LOAD_TIMEOUT=10")
        
        # Re-import to pick up new env var
        import importlib
        import aios.data.datasets.readers
        importlib.reload(aios.data.datasets.readers)
        
        timeout_value = aios.data.datasets.readers.DEFAULT_READ_TIMEOUT
        print(f"✅ Default timeout is now: {timeout_value}s")
        
        if timeout_value == 10:
            print("✅ SUCCESS: Environment variable is respected")
        else:
            print(f"⚠️  WARNING: Expected 10, got {timeout_value}")
    finally:
        # Restore original
        if original:
            os.environ["AIOS_DATASET_LOAD_TIMEOUT"] = original
        else:
            os.environ.pop("AIOS_DATASET_LOAD_TIMEOUT", None)


def main():
    """Run all tests."""
    print("="*60)
    print("BUG-002 Dataset Timeout Fix - Test Suite")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        files = create_test_files(temp_path)
        
        # Run tests
        test_normal_loading(files)
        test_timeout_mechanism(files)
        test_large_file_streaming(files)
        test_encoding_fallback(temp_path)
        test_environment_variable()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
