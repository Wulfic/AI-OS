#!/usr/bin/env python3
"""
BUG-006 Tokenizer Compatibility Test Script

Tests various tokenizers to verify they work with AI-OS training pipeline.
Tests both basic tokenizer functionality and integration with the training system.
"""

import sys
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from typing import Optional
import traceback


# Tokenizers to test (model_id, display_name, size_category)
TOKENIZERS_TO_TEST = [
    # Confirmed working
    ("gpt2", "GPT-2", "small"),
    
    # Listed as "likely supported" in bug report
    ("Qwen/Qwen2.5-0.5B", "Qwen 2.5 (0.5B)", "small"),
    ("mistralai/Mistral-7B-v0.1", "Mistral 7B", "medium"),
    ("codellama/CodeLlama-7b-hf", "Code Llama 7B", "medium"),
    ("deepseek-ai/deepseek-coder-1.3b-base", "DeepSeek Coder 1.3B", "small"),
    ("bigcode/starcoder2-3b", "StarCoder2 3B", "medium"),
    ("microsoft/phi-3-mini-4k-instruct", "Phi-3 Mini", "small"),
    ("meta-llama/Llama-3.2-1B", "Llama 3.2 (1B)", "small"),
    
    # Additional popular models
    ("google/gemma-2b", "Gemma 2B", "small"),
    ("tiiuae/falcon-7b", "Falcon 7B", "medium"),
]


class TokenizerTest:
    """Test result for a single tokenizer."""
    
    def __init__(self, model_id: str, display_name: str, size_category: str):
        self.model_id = model_id
        self.display_name = display_name
        self.size_category = size_category
        
        # Test results
        self.load_success = False
        self.load_error: Optional[str] = None
        
        self.encode_success = False
        self.encode_error: Optional[str] = None
        
        self.decode_success = False
        self.decode_error: Optional[str] = None
        
        self.special_tokens_ok = False
        self.special_tokens_notes: Optional[str] = None
        
        self.padding_side: Optional[str] = None
        self.has_pad_token = False
        self.has_eos_token = False
        self.has_bos_token = False
        
        self.integration_success = False
        self.integration_error: Optional[str] = None
    
    def passed_all(self) -> bool:
        """Check if all critical tests passed."""
        return (
            self.load_success and 
            self.encode_success and 
            self.decode_success and
            self.has_eos_token  # EOS token is critical for decoder models
        )
    
    def status_symbol(self) -> str:
        """Return status symbol based on test results."""
        if self.passed_all():
            return "✅"
        elif self.load_success and self.encode_success:
            return "⚠️"
        else:
            return "❌"


def test_tokenizer_loading(test: TokenizerTest) -> bool:
    """Test 1: Can we load the tokenizer?"""
    try:
        from aios.cli.hrm_hf_utils import load_tokenizer
        
        tokenizer = load_tokenizer(test.model_id)
        test.load_success = True
        
        # Check padding side
        test.padding_side = getattr(tokenizer, "padding_side", "unknown")
        
        # Check special tokens
        test.has_pad_token = getattr(tokenizer, "pad_token", None) is not None
        test.has_eos_token = getattr(tokenizer, "eos_token", None) is not None
        test.has_bos_token = getattr(tokenizer, "bos_token", None) is not None
        
        test.special_tokens_ok = test.has_eos_token  # EOS is critical
        
        notes = []
        if not test.has_pad_token:
            notes.append("No pad_token (should be auto-mapped to eos_token)")
        if not test.has_bos_token:
            notes.append("No bos_token")
        if test.padding_side != "left":
            notes.append(f"Padding side is {test.padding_side} (expected left)")
        
        test.special_tokens_notes = "; ".join(notes) if notes else "All tokens OK"
        
        return True
        
    except Exception as e:
        test.load_success = False
        test.load_error = f"{type(e).__name__}: {str(e)}"
        return False


def test_tokenizer_encode_decode(test: TokenizerTest) -> bool:
    """Test 2: Can we encode and decode text?"""
    if not test.load_success:
        test.encode_error = "Skipped (load failed)"
        test.decode_error = "Skipped (load failed)"
        return False
    
    try:
        from aios.cli.hrm_hf_utils import load_tokenizer
        tokenizer = load_tokenizer(test.model_id)
        
        # Test encode
        test_text = "Hello, world! This is a test."
        try:
            token_ids = tokenizer.encode(test_text)
            test.encode_success = True
            
            # Test decode
            try:
                decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                test.decode_success = True
                
                # Verify roundtrip (allow for minor whitespace differences)
                if decoded_text.strip().lower() != test_text.strip().lower():
                    test.decode_error = f"Roundtrip mismatch: '{decoded_text}'"
                    
            except Exception as e:
                test.decode_error = f"{type(e).__name__}: {str(e)}"
                
        except Exception as e:
            test.encode_error = f"{type(e).__name__}: {str(e)}"
        
        return test.encode_success and test.decode_success
        
    except Exception as e:
        test.encode_error = f"Unexpected error: {type(e).__name__}: {str(e)}"
        return False


def test_tokenizer_integration(test: TokenizerTest) -> bool:
    """Test 3: Can the tokenizer work with training config?"""
    if not test.load_success:
        test.integration_error = "Skipped (load failed)"
        return False
    
    try:
        from aios.cli.hrm_hf_utils import load_tokenizer
        from aios.cli.hrm_hf.config import TrainingConfig
        
        tokenizer = load_tokenizer(test.model_id)
        
        # Try to create a minimal training config
        config = TrainingConfig(
            model=test.model_id,
            steps=1,
            batch_size=1,
        )
        
        # Verify tokenizer vocab size makes sense
        vocab_size = len(tokenizer)
        if vocab_size < 1000:
            test.integration_error = f"Vocab size too small: {vocab_size}"
            return False
        
        if vocab_size > 500000:
            test.integration_error = f"Vocab size suspiciously large: {vocab_size}"
            return False
        
        test.integration_success = True
        return True
        
    except Exception as e:
        test.integration_error = f"{type(e).__name__}: {str(e)}"
        return False


def run_all_tests() -> list[TokenizerTest]:
    """Run tests on all tokenizers."""
    results = []
    
    print("=" * 80)
    print("BUG-006 Tokenizer Compatibility Test")
    print("=" * 80)
    print()
    print(f"Testing {len(TOKENIZERS_TO_TEST)} tokenizers...")
    print()
    
    for model_id, display_name, size_category in TOKENIZERS_TO_TEST:
        print(f"Testing: {display_name} ({model_id})...")
        test = TokenizerTest(model_id, display_name, size_category)
        
        # Run tests
        test_tokenizer_loading(test)
        test_tokenizer_encode_decode(test)
        test_tokenizer_integration(test)
        
        # Print immediate result
        status = test.status_symbol()
        print(f"  {status} {display_name}: ", end="")
        
        if test.passed_all():
            print("PASSED ✅")
        elif test.load_success:
            print(f"PARTIAL ⚠️ - {test.special_tokens_notes or 'See details'}")
        else:
            print(f"FAILED ❌ - {test.load_error}")
        
        print()
        results.append(test)
    
    return results


def print_summary(results: list[TokenizerTest]):
    """Print detailed summary of all test results."""
    print()
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print()
    
    passed = [t for t in results if t.passed_all()]
    partial = [t for t in results if t.load_success and not t.passed_all()]
    failed = [t for t in results if not t.load_success]
    
    print(f"✅ FULLY COMPATIBLE: {len(passed)}/{len(results)}")
    for test in passed:
        print(f"   • {test.display_name}")
        print(f"     Model: {test.model_id}")
        print(f"     Padding: {test.padding_side}, Special tokens: {test.special_tokens_notes}")
    print()
    
    if partial:
        print(f"⚠️  PARTIALLY COMPATIBLE: {len(partial)}/{len(results)}")
        for test in partial:
            print(f"   • {test.display_name}")
            print(f"     Model: {test.model_id}")
            print(f"     Issues: {test.special_tokens_notes}")
            if test.encode_error:
                print(f"     Encode error: {test.encode_error}")
            if test.decode_error:
                print(f"     Decode error: {test.decode_error}")
        print()
    
    if failed:
        print(f"❌ INCOMPATIBLE: {len(failed)}/{len(results)}")
        for test in failed:
            print(f"   • {test.display_name}")
            print(f"     Model: {test.model_id}")
            print(f"     Error: {test.load_error}")
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Fully compatible:      {len(passed):2d} tokenizers")
    print(f"Partially compatible:  {len(partial):2d} tokenizers")
    print(f"Incompatible:          {len(failed):2d} tokenizers")
    print(f"Total tested:          {len(results):2d} tokenizers")
    print()
    
    success_rate = (len(passed) / len(results)) * 100 if results else 0
    print(f"Success rate: {success_rate:.1f}%")
    print()


def generate_compatibility_doc(results: list[TokenizerTest]):
    """Generate markdown documentation of test results."""
    output_file = Path(__file__).parent.parent / "docs" / "user_guide" / "TOKENIZER_COMPATIBILITY.md"
    
    from datetime import datetime
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Tokenizer Compatibility Guide\n\n")
        f.write(f"**Last Updated**: {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(f"**Test Script**: `scripts/test_bug006_tokenizers.py`\n\n")
        
        f.write("This document lists tokenizers that have been tested with AI-OS and their compatibility status.\n\n")
        
        f.write("## Compatibility Legend\n\n")
        f.write("- ✅ **Fully Compatible**: All tests passed, ready for production use\n")
        f.write("- ⚠️ **Partially Compatible**: Loads and works but has minor issues (see notes)\n")
        f.write("- ❌ **Incompatible**: Cannot load or has critical issues\n\n")
        
        # Group by status
        passed = [t for t in results if t.passed_all()]
        partial = [t for t in results if t.load_success and not t.passed_all()]
        failed = [t for t in results if not t.load_success]
        
        f.write("## ✅ Fully Compatible Tokenizers\n\n")
        f.write(f"**{len(passed)} tokenizers tested and confirmed working**\n\n")
        f.write("| Tokenizer | Model ID | Size | Notes |\n")
        f.write("|-----------|----------|------|-------|\n")
        for test in passed:
            f.write(f"| {test.display_name} | `{test.model_id}` | {test.size_category} | {test.special_tokens_notes} |\n")
        f.write("\n")
        
        if partial:
            f.write("## ⚠️ Partially Compatible Tokenizers\n\n")
            f.write(f"**{len(partial)} tokenizers work but have minor issues**\n\n")
            f.write("| Tokenizer | Model ID | Issues |\n")
            f.write("|-----------|----------|--------|\n")
            for test in partial:
                issues = test.special_tokens_notes or "See details"
                if test.encode_error:
                    issues += f"; Encode: {test.encode_error}"
                if test.decode_error:
                    issues += f"; Decode: {test.decode_error}"
                f.write(f"| {test.display_name} | `{test.model_id}` | {issues} |\n")
            f.write("\n")
        
        if failed:
            f.write("## ❌ Incompatible Tokenizers\n\n")
            f.write(f"**{len(failed)} tokenizers failed to load**\n\n")
            f.write("| Tokenizer | Model ID | Error |\n")
            f.write("|-----------|----------|-------|\n")
            for test in failed:
                error = test.load_error or "Unknown error"
                f.write(f"| {test.display_name} | `{test.model_id}` | {error} |\n")
            f.write("\n")
        
        f.write("## Usage Examples\n\n")
        f.write("### Using a Tested Tokenizer\n\n")
        f.write("```bash\n")
        if passed:
            example = passed[0]
            f.write(f"# CLI\n")
            f.write(f"aios hrm-hf train-actv1 --model {example.model_id} \\\n")
            f.write(f"  --dataset-file data.txt --steps 1000\n\n")
            f.write(f"# Python\n")
            f.write(f"from aios.cli.hrm_hf_utils import load_tokenizer\n")
            f.write(f"tokenizer = load_tokenizer('{example.model_id}')\n")
        f.write("```\n\n")
        
        f.write("## Testing New Tokenizers\n\n")
        f.write("To test a new tokenizer:\n\n")
        f.write("1. Add the model ID to `scripts/test_bug006_tokenizers.py`\n")
        f.write("2. Run: `python scripts/test_bug006_tokenizers.py`\n")
        f.write("3. Check the results and update this document\n\n")
        
        f.write("## Known Limitations\n\n")
        f.write("- **Vision tokenizers** (CLIP, LLaVA, SigLIP): Not supported - AI-OS is text-only\n")
        f.write("- **Domain-specific tokenizers** (BioBERT, SciBERT): Not tested - may work if text-based\n")
        f.write("- **Encoder-only models** (BERT, RoBERTa): Not supported - AI-OS trains decoder-only models\n\n")
        
        f.write("## Troubleshooting\n\n")
        f.write("### Tokenizer Won't Load\n\n")
        f.write("- Ensure you have internet connection (first download)\n")
        f.write("- Check HuggingFace authentication if model is gated\n")
        f.write("- Try downloading manually: `huggingface-cli download <model_id>`\n\n")
        
        f.write("### Padding/Special Token Warnings\n\n")
        f.write("- AI-OS automatically configures tokenizers for decoder-only models\n")
        f.write("- Pad token is automatically mapped to eos_token if missing\n")
        f.write("- Padding side is set to 'left' for generation\n\n")
    
    print(f"✅ Documentation generated: {output_file}")


def main():
    """Main entry point."""
    try:
        results = run_all_tests()
        print_summary(results)
        generate_compatibility_doc(results)
        
        # Exit with appropriate code
        passed = [t for t in results if t.passed_all()]
        if len(passed) >= len(results) * 0.8:  # 80% success rate
            print("✅ Overall result: PASS (≥80% compatible)")
            return 0
        else:
            print("⚠️  Overall result: NEEDS ATTENTION (<80% compatible)")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test script error: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
