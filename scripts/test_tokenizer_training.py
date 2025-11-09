#!/usr/bin/env python3
"""
Quick training test for all tokenizers.
Runs minimal training (1-2 steps) on each test brain to verify tokenizer integration.
"""

import json
import subprocess
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

def get_test_brains():
    """Find all test brains."""
    brains_dir = repo_root / "artifacts" / "brains" / "actv1"
    test_brains = []
    
    for brain_dir in brains_dir.iterdir():
        if not brain_dir.is_dir():
            continue
        
        brain_json = brain_dir / "brain.json"
        if not brain_json.exists():
            continue
        
        try:
            with open(brain_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Only test brains we created
            if data.get("test_brain"):
                test_brains.append({
                    "name": data["name"],
                    "path": str(brain_dir),
                    "tokenizer_id": data.get("tokenizer_id", "unknown"),
                    "tokenizer_model": data.get("tokenizer_model", "unknown"),
                    "vocab_size": data.get("vocab_size", 0)
                })
        except Exception as e:
            print(f"   Error reading {brain_json}: {e}")
    
    return test_brains


def test_train_brain(brain_info):
    """Run minimal training test on a brain."""
    print(f"\n{'='*70}")
    print(f"Testing: {brain_info['name']}")
    print(f"Tokenizer: {brain_info['tokenizer_id']} ({brain_info['vocab_size']:,} tokens)")
    print(f"{'='*70}")
    
    # Create minimal test dataset
    test_data_file = repo_root / "training_data" / "curated_datasets" / "test_sample.txt"
    
    if not test_data_file.exists():
        print(f"⚠️  Test dataset not found: {test_data_file}")
        print("   Creating minimal test dataset...")
        test_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_data_file, 'w', encoding='utf-8') as f:
            f.write("This is a test sentence for tokenizer validation.\n")
            f.write("The quick brown fox jumps over the lazy dog.\n")
            f.write("Testing tokenizer integration in AI-OS.\n")
    
    # Build training command - just 1 step with minimal batch
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    
    cmd = [
        str(venv_python),
        "-m", "aios.cli.aios",
        "hrm-hf", "train-actv1",
        "--model", brain_info['tokenizer_model'],
        "--dataset-file", str(test_data_file),
        "--steps", "1",
        "--batch-size", "1",
        "--halt-max-steps", "1",
        "--eval-batches", "1",
        "--log-file", f"artifacts/brains/actv1/{brain_info['name']}/test_metrics.jsonl"
    ]
    
    print(f"\n🚀 Running minimal training test...")
    print(f"   Command: {' '.join(cmd[:8])}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=str(repo_root)
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: Tokenizer {brain_info['tokenizer_id']} works correctly!")
            return True
        else:
            print(f"❌ FAILED: Training returned error code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: Training took longer than 2 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def main():
    print("\n" + "="*70)
    print("Tokenizer Integration Test Suite")
    print("="*70)
    print("Testing all downloaded tokenizers with minimal training runs...")
    
    # Get all test brains
    test_brains = get_test_brains()
    
    if not test_brains:
        print("\n❌ No test brains found!")
        print("   Run: python scripts/test_tokenizer_brains.py")
        return 1
    
    print(f"\n📋 Found {len(test_brains)} test brains to validate")
    
    # Test each brain
    results = []
    for brain_info in test_brains:
        success = test_train_brain(brain_info)
        results.append({
            "name": brain_info["name"],
            "tokenizer": brain_info["tokenizer_id"],
            "success": success
        })
    
    # Summary
    print("\n\n" + "="*70)
    print("📊 Test Results Summary")
    print("="*70)
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"\nTotal Brains Tested: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(successful/len(results)*100):.1f}%")
    
    if successful > 0:
        print("\n✅ Successful Tokenizers:")
        for r in results:
            if r["success"]:
                print(f"   • {r['tokenizer']}")
    
    if failed > 0:
        print("\n❌ Failed Tokenizers:")
        for r in results:
            if not r["success"]:
                print(f"   • {r['tokenizer']}")
    
    print("\n" + "="*70)
    if failed == 0:
        print("🎉 All tokenizers validated successfully!")
        print("="*70 + "\n")
        return 0
    else:
        print("⚠️  Some tokenizers failed validation")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
