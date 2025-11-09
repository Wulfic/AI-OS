#!/usr/bin/env python3
"""
Test script to create a brain for each tokenizer and verify functionality.
Creates test brains with different tokenizers and validates their configuration.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from aios.core.tokenizers.registry import TokenizerRegistry


def create_test_brain(tokenizer_id: str, test_name_suffix: str = "test") -> dict:
    """
    Create a test brain with the specified tokenizer.
    
    Args:
        tokenizer_id: The tokenizer ID from the registry
        test_name_suffix: Suffix to add to brain name
        
    Returns:
        dict: Result with success status and details
    """
    registry = TokenizerRegistry()
    
    # Get tokenizer info
    tokenizer_info = registry.get(tokenizer_id)
    if not tokenizer_info:
        return {
            "success": False,
            "error": f"Tokenizer '{tokenizer_id}' not found in registry"
        }
    
    # Check if tokenizer is installed
    if not registry.check_installed(tokenizer_id):
        return {
            "success": False,
            "error": f"Tokenizer '{tokenizer_id}' not installed. Run: python scripts/download_tokenizer.py {tokenizer_id}"
        }
    
    # Create brain name
    brain_name = f"{tokenizer_info.name.replace(' ', '-')}-{test_name_suffix}"
    brain_dir = repo_root / "artifacts" / "brains" / "actv1" / brain_name
    
    # Check if brain already exists
    if brain_dir.exists():
        return {
            "success": False,
            "error": f"Brain '{brain_name}' already exists at {brain_dir}"
        }
    
    # Create brain directory
    brain_dir.mkdir(parents=True, exist_ok=True)
    
    # Create brain.json
    brain_data = {
        "name": brain_name,
        "preset": "default",
        "tokenizer_id": tokenizer_id,
        "tokenizer_model": tokenizer_info.path,
        "vocab_size": tokenizer_info.vocab_size,
        "default_goal": f"Test brain for {tokenizer_info.name} tokenizer",
        "training_steps": 0,
        "created_at": datetime.now().isoformat(),
        "test_brain": True,
        "description": f"Test brain using {tokenizer_info.name} ({tokenizer_info.vocab_size:,} tokens)"
    }
    
    brain_json_path = brain_dir / "brain.json"
    with open(brain_json_path, 'w', encoding='utf-8') as f:
        json.dump(brain_data, f, indent=2)
    
    # Verify the brain.json was created correctly
    with open(brain_json_path, 'r', encoding='utf-8') as f:
        verified_data = json.load(f)
    
    # Validate required fields
    required_fields = ['tokenizer_id', 'tokenizer_model', 'vocab_size']
    missing_fields = [field for field in required_fields if field not in verified_data]
    
    if missing_fields:
        return {
            "success": False,
            "error": f"Missing required fields in brain.json: {missing_fields}"
        }
    
    # Verify tokenizer files exist
    tokenizer_path = Path(tokenizer_info.path)
    if not tokenizer_path.exists():
        return {
            "success": False,
            "error": f"Tokenizer path does not exist: {tokenizer_path}"
        }
    
    return {
        "success": True,
        "brain_name": brain_name,
        "brain_path": str(brain_dir),
        "brain_json_path": str(brain_json_path),
        "tokenizer_id": tokenizer_id,
        "tokenizer_name": tokenizer_info.name,
        "vocab_size": tokenizer_info.vocab_size,
        "tokenizer_path": tokenizer_info.path,
        "data": verified_data
    }


def test_all_tokenizers():
    """Test creating brains for all installed tokenizers."""
    registry = TokenizerRegistry()
    all_tokenizers = registry.list_available()
    
    print("\n" + "="*70)
    print("🧪 Testing Brain Creation for All Tokenizers")
    print("="*70 + "\n")
    
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    for tokenizer_info in all_tokenizers:
        tokenizer_id = tokenizer_info.id
        
        print(f"📦 Testing: {tokenizer_info.name} ({tokenizer_id})")
        print(f"   Vocab Size: {tokenizer_info.vocab_size:,} tokens")
        print(f"   Installed: {'✓' if registry.check_installed(tokenizer_id) else '⚠'}")
        
        if not registry.check_installed(tokenizer_id):
            print(f"   ⚠️  Skipping - tokenizer not installed\n")
            skipped += 1
            results.append({
                "tokenizer_id": tokenizer_id,
                "status": "skipped",
                "reason": "not_installed"
            })
            continue
        
        # Create test brain
        result = create_test_brain(tokenizer_id, "test")
        
        if result["success"]:
            print(f"   ✅ SUCCESS: Brain created at {result['brain_path']}")
            print(f"   Brain: {result['brain_name']}")
            print(f"   Tokenizer Path: {result['tokenizer_path']}")
            print(f"   Vocab Size: {result['vocab_size']:,} tokens\n")
            successful += 1
            results.append({
                "tokenizer_id": tokenizer_id,
                "status": "success",
                "brain_name": result["brain_name"],
                "brain_path": result["brain_path"]
            })
        else:
            print(f"   ❌ FAILED: {result['error']}\n")
            failed += 1
            results.append({
                "tokenizer_id": tokenizer_id,
                "status": "failed",
                "error": result["error"]
            })
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    print(f"Total Tokenizers: {len(all_tokenizers)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped: {skipped}")
    print("="*70 + "\n")
    
    # Detailed results
    if successful > 0:
        print("\n✅ Successfully Created Brains:")
        for result in results:
            if result["status"] == "success":
                print(f"   • {result['brain_name']} ({result['tokenizer_id']})")
                print(f"     Path: {result['brain_path']}")
    
    if failed > 0:
        print("\n❌ Failed:")
        for result in results:
            if result["status"] == "failed":
                print(f"   • {result['tokenizer_id']}: {result['error']}")
    
    if skipped > 0:
        print("\n⚠️  Skipped (not installed):")
        for result in results:
            if result["status"] == "skipped":
                print(f"   • {result['tokenizer_id']}")
    
    print("\n" + "="*70)
    print("🎉 Testing Complete!")
    print("="*70 + "\n")
    
    return results


def cleanup_test_brains():
    """Remove all test brains created by this script."""
    brains_dir = repo_root / "artifacts" / "brains" / "actv1"
    
    if not brains_dir.exists():
        print("No brains directory found.")
        return
    
    print("\n🧹 Cleaning up test brains...")
    
    removed = 0
    for brain_dir in brains_dir.iterdir():
        if not brain_dir.is_dir():
            continue
        
        brain_json = brain_dir / "brain.json"
        if not brain_json.exists():
            continue
        
        try:
            with open(brain_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's a test brain
            if data.get("test_brain"):
                print(f"   Removing: {brain_dir.name}")
                import shutil
                shutil.rmtree(brain_dir)
                removed += 1
        except Exception as e:
            print(f"   Error checking {brain_dir.name}: {e}")
    
    print(f"✅ Removed {removed} test brain(s)\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test tokenizer brain creation")
    parser.add_argument("--cleanup", action="store_true", help="Remove all test brains")
    parser.add_argument("--tokenizer", type=str, help="Test specific tokenizer only")
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_test_brains()
    elif args.tokenizer:
        result = create_test_brain(args.tokenizer, "test")
        if result["success"]:
            print(f"✅ SUCCESS: {result['brain_name']}")
            print(f"   Path: {result['brain_path']}")
        else:
            print(f"❌ FAILED: {result['error']}")
            sys.exit(1)
    else:
        test_all_tokenizers()
