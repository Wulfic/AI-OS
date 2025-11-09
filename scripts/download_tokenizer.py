#!/usr/bin/env python3
"""Download tokenizers for AI-OS.

This script downloads tokenizer files (without model weights) from HuggingFace
and saves them locally for use in brain creation.

Usage:
    python scripts/download_tokenizer.py <tokenizer_id> [hf_token]
    
Examples:
    python scripts/download_tokenizer.py gpt2
    python scripts/download_tokenizer.py llama3-8b YOUR_HF_TOKEN
    python scripts/download_tokenizer.py --list
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to path so we can import from aios
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aios.core.tokenizers.registry import TokenizerRegistry


def download_tokenizer(tokenizer_id: str, hf_token: str = None) -> bool:
    """Download a tokenizer by ID.
    
    Args:
        tokenizer_id: Tokenizer ID from registry
        hf_token: Optional HuggingFace authentication token
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("❌ Error: transformers library not installed")
        print("   Run: pip install transformers")
        return False
    
    info = TokenizerRegistry.get(tokenizer_id)
    if not info:
        print(f"❌ Unknown tokenizer: {tokenizer_id}")
        print("\nAvailable tokenizers:")
        list_tokenizers()
        return False
    
    print(f"📥 Downloading {info.name}...")
    print(f"   Vocab size: {info.vocab_size:,} tokens")
    print(f"   HF Model: {info.hf_model_id}")
    print(f"   Path: {info.path}")
    
    if info.requires_auth and not hf_token:
        print(f"\n⚠️  This tokenizer requires HuggingFace authentication!")
        print("   Get your token at: https://huggingface.co/settings/tokens")
        print(f"   Then run: python scripts/download_tokenizer.py {tokenizer_id} YOUR_TOKEN")
        return False
    
    try:
        # Download tokenizer
        print("\n⏳ Downloading from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(
            info.hf_model_id,
            token=hf_token if info.requires_auth else None
        )
        
        # Save locally
        save_path = Path(info.path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving to {save_path}...")
        tokenizer.save_pretrained(str(save_path))
        
        print(f"\n✅ Successfully downloaded!")
        print(f"   Actual vocab size: {tokenizer.vocab_size:,}")
        print(f"   Files saved to: {save_path}")
        
        # Verify installation
        if TokenizerRegistry.check_installed(tokenizer_id):
            print(f"   ✓ Verified: Tokenizer is ready to use")
        else:
            print(f"   ⚠️  Warning: Verification failed, some files may be missing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download: {e}")
        return False


def list_tokenizers():
    """List all available tokenizers with their status."""
    print("\n📋 Available Tokenizers:\n")
    
    for info in TokenizerRegistry.list_available():
        installed = TokenizerRegistry.check_installed(info.id)
        status = "✓ Installed" if installed else "⚠ Not installed"
        auth = " (requires auth)" if info.requires_auth else ""
        
        print(f"  {status} | {info.id}")
        print(f"     Name: {info.name}")
        print(f"     Vocab: {info.vocab_size:,} tokens")
        print(f"     Compression: {info.compression_ratio:.1f} chars/token")
        print(f"     Best for: {', '.join(info.recommended_for)}")
        print(f"     Description: {info.description}{auth}")
        print()


def download_all(hf_token: str = None):
    """Download all tokenizers that don't require authentication."""
    print("📦 Downloading all non-gated tokenizers...\n")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for info in TokenizerRegistry.list_available():
        if TokenizerRegistry.check_installed(info.id):
            print(f"⏭️  Skipping {info.name} (already installed)")
            skip_count += 1
            continue
        
        if info.requires_auth and not hf_token:
            print(f"⏭️  Skipping {info.name} (requires authentication)")
            skip_count += 1
            continue
        
        print(f"\n{'='*60}")
        if download_tokenizer(info.id, hf_token):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count} downloaded, {skip_count} skipped, {fail_count} failed")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
        print(__doc__)
        list_tokenizers()
        return
    
    command = sys.argv[1]
    
    if command in ["--list", "-l"]:
        list_tokenizers()
        return
    
    if command in ["--all", "-a"]:
        hf_token = sys.argv[2] if len(sys.argv) > 2 else None
        download_all(hf_token)
        return
    
    # Download specific tokenizer
    tokenizer_id = sys.argv[1]
    hf_token = sys.argv[2] if len(sys.argv) > 2 else None
    
    if download_tokenizer(tokenizer_id, hf_token):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
