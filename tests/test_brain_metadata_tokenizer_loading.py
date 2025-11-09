"""
Test brain metadata tokenizer loading fix.

This test verifies that the tokenizer is correctly loaded from brain.json
even when student_init is None (fresh start scenario).
"""

import json
import tempfile
from pathlib import Path

from aios.cli.hrm_hf.brain_metadata import (
    load_brain_metadata,
    extract_tokenizer_from_metadata,
)


def test_load_brain_metadata_from_student_init():
    """Test loading brain metadata from student_init parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        brain_dir = Path(tmpdir) / "test_brain"
        brain_dir.mkdir()
        
        # Create brain.json
        brain_json = brain_dir / "brain.json"
        brain_data = {
            "name": "test_brain",
            "tokenizer_model": "artifacts/tokenizers/qwen2.5-7b",
            "vocab_size": 151643,
        }
        brain_json.write_text(json.dumps(brain_data, indent=2))
        
        # Load metadata using student_init (checkpoint path)
        checkpoint_path = str(brain_dir / "model.safetensors")
        metadata = load_brain_metadata(
            student_init=checkpoint_path,
            log_fn=print
        )
        
        assert metadata is not None
        assert metadata["name"] == "test_brain"
        assert metadata["tokenizer_model"] == "artifacts/tokenizers/qwen2.5-7b"
        assert metadata["vocab_size"] == 151643


def test_load_brain_metadata_from_save_dir_fallback():
    """Test loading brain metadata from save_dir when student_init is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        brain_dir = Path(tmpdir) / "test_brain"
        brain_dir.mkdir()
        
        # Create brain.json
        brain_json = brain_dir / "brain.json"
        brain_data = {
            "name": "test_brain",
            "tokenizer_model": "artifacts/tokenizers/qwen2.5-7b",
            "vocab_size": 151643,
        }
        brain_json.write_text(json.dumps(brain_data, indent=2))
        
        # Load metadata with student_init=None, using save_dir fallback
        # This is the "START FRESH" scenario
        metadata = load_brain_metadata(
            student_init=None,
            log_fn=print,
            save_dir=str(brain_dir)
        )
        
        assert metadata is not None, "Should load metadata from save_dir when student_init is None"
        assert metadata["name"] == "test_brain"
        assert metadata["tokenizer_model"] == "artifacts/tokenizers/qwen2.5-7b"
        assert metadata["vocab_size"] == 151643


def test_load_brain_metadata_no_fallback():
    """Test that None is returned when no brain.json exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No brain.json created
        metadata = load_brain_metadata(
            student_init=None,
            log_fn=print,
            save_dir=str(tmpdir)
        )
        
        assert metadata is None, "Should return None when no brain.json exists"


def test_extract_tokenizer_from_metadata_with_metadata():
    """Test extracting tokenizer from valid metadata."""
    metadata = {
        "tokenizer_model": "artifacts/tokenizers/qwen2.5-7b",
        "base_model": "artifacts/models/base",
        "vocab_size": 151643,
    }
    
    tokenizer_path = extract_tokenizer_from_metadata(metadata, "default/model")
    assert tokenizer_path == "artifacts/tokenizers/qwen2.5-7b"


def test_extract_tokenizer_from_metadata_fallback_to_base_model():
    """Test extracting tokenizer falls back to base_model if tokenizer_model not specified."""
    metadata = {
        "base_model": "artifacts/models/base",
        "vocab_size": 50257,
    }
    
    tokenizer_path = extract_tokenizer_from_metadata(metadata, "default/model")
    assert tokenizer_path == "artifacts/models/base"


def test_extract_tokenizer_from_metadata_no_metadata():
    """Test extracting tokenizer uses default when no metadata."""
    tokenizer_path = extract_tokenizer_from_metadata(None, "default/model")
    assert tokenizer_path == "default/model"


def test_fresh_start_scenario_integration():
    """
    Integration test: Simulate the full "START FRESH" scenario.
    
    This verifies the fix for the bug where starting fresh would use
    the wrong tokenizer (base_model instead of brain's tokenizer).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        brain_dir = Path(tmpdir) / "test_brain"
        brain_dir.mkdir()
        
        # Create brain.json with qwen2.5-7b tokenizer
        brain_json = brain_dir / "brain.json"
        brain_data = {
            "name": "test_brain",
            "tokenizer_model": "artifacts/tokenizers/qwen2.5-7b",
            "vocab_size": 151643,
        }
        brain_json.write_text(json.dumps(brain_data, indent=2))
        
        # Simulate "START FRESH": student_init=None
        # (User deleted checkpoints but brain.json still exists)
        metadata = load_brain_metadata(
            student_init=None,  # Fresh start
            log_fn=print,
            save_dir=str(brain_dir)  # Brain directory contains brain.json
        )
        
        # Extract tokenizer path
        base_model = "artifacts/hf_implant/base_model"  # Would be wrong (GPT-2)
        tokenizer_path = extract_tokenizer_from_metadata(metadata, base_model)
        
        # CRITICAL: Should use brain's tokenizer, NOT base_model
        assert tokenizer_path == "artifacts/tokenizers/qwen2.5-7b", \
            f"Expected brain's tokenizer (qwen2.5-7b), got {tokenizer_path}"
        assert tokenizer_path != base_model, \
            "Should NOT fall back to base_model when brain.json exists"


if __name__ == "__main__":
    # Run tests manually
    test_load_brain_metadata_from_student_init()
    print("✓ test_load_brain_metadata_from_student_init")
    
    test_load_brain_metadata_from_save_dir_fallback()
    print("✓ test_load_brain_metadata_from_save_dir_fallback")
    
    test_load_brain_metadata_no_fallback()
    print("✓ test_load_brain_metadata_no_fallback")
    
    test_extract_tokenizer_from_metadata_with_metadata()
    print("✓ test_extract_tokenizer_from_metadata_with_metadata")
    
    test_extract_tokenizer_from_metadata_fallback_to_base_model()
    print("✓ test_extract_tokenizer_from_metadata_fallback_to_base_model")
    
    test_extract_tokenizer_from_metadata_no_metadata()
    print("✓ test_extract_tokenizer_from_metadata_no_metadata")
    
    test_fresh_start_scenario_integration()
    print("✓ test_fresh_start_scenario_integration")
    
    print("\n✓ All tests passed!")
