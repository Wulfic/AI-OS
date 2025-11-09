"""Brain metadata loading and management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any


def load_brain_metadata(
    student_init: Optional[str],
    log_fn,
    save_dir: Optional[str] = None
) -> Optional[dict]:
    """Load brain metadata if resuming from existing brain.
    
    Args:
        student_init: Student initialization path (checkpoint or directory)
        log_fn: Logging function
        save_dir: Brain save directory (fallback if student_init is None)
        
    Returns:
        Brain metadata dictionary or None
    """
    brain_json_path = None
    
    # Try to get brain.json from student_init first
    if student_init:
        try:
            student_path = Path(student_init)
            
            # Check if student_init points to brain bundle directory or file itself
            if student_path.is_dir():
                brain_json_path = student_path / "brain.json"
            else:
                brain_json_path = student_path.parent / "brain.json"
        except Exception:
            pass
    
    # Fallback: try to get brain.json from save_dir (for fresh starts)
    if (not brain_json_path or not brain_json_path.exists()) and save_dir:
        try:
            save_path = Path(save_dir)
            candidate_path = save_path / "brain.json"
            if candidate_path.exists():
                brain_json_path = candidate_path
        except Exception:
            pass
    
    # If we still don't have a valid brain.json, return None
    if not brain_json_path or not brain_json_path.exists():
        return None
    
    try:
        with open(brain_json_path, 'r', encoding='utf-8') as f:
            brain_metadata = json.load(f)
        
        log_fn({
            "brain_metadata": "loaded",
            "source": str(brain_json_path),
            "brain_name": brain_metadata.get("brain_name"),
            "tokenizer_model": brain_metadata.get("tokenizer_model"),
            "vocab_size": brain_metadata.get("vocab_size"),
            "note": "Using brain metadata for tokenizer and model configuration"
        })
        
        return brain_metadata
        
    except Exception as e:
        log_fn({
            "brain_metadata_load": "failed",
            "path": str(brain_json_path),
            "error": str(e),
            "hint": "Continuing with --model parameter"
        })
        return None


def extract_model_from_metadata(brain_metadata: Optional[dict], default_model: str) -> str:
    """Extract model path from brain metadata.
    
    Args:
        brain_metadata: Brain metadata dictionary
        default_model: Default model path to use if metadata not available
        
    Returns:
        Model path
    """
    if not brain_metadata:
        return default_model
    
    # Try to get base model from metadata
    model_path = brain_metadata.get("base_model") or brain_metadata.get("model")
    return model_path if model_path else default_model


def extract_tokenizer_from_metadata(brain_metadata: Optional[dict], default_tokenizer: str) -> str:
    """Extract tokenizer path from brain metadata.
    
    Args:
        brain_metadata: Brain metadata dictionary
        default_tokenizer: Default tokenizer path to use if metadata not available
        
    Returns:
        Tokenizer path
    """
    if not brain_metadata:
        return default_tokenizer
    
    # Try to get tokenizer from metadata (prefer tokenizer_model over base model)
    # This ensures we use the correct tokenizer that matches the vocab_size the model was trained with
    tokenizer_path = (
        brain_metadata.get("tokenizer_model") or 
        brain_metadata.get("base_model") or 
        brain_metadata.get("model")
    )
    
    extracted_path = tokenizer_path if tokenizer_path else default_tokenizer
    
    # Validate tokenizer matches expected vocab size if specified in metadata
    expected_vocab = brain_metadata.get("vocab_size")
    if expected_vocab and tokenizer_path:
        # Log the expected vocab size for verification
        print(f"[Metadata] Expected tokenizer vocab size: {expected_vocab:,}")
        print(f"[Metadata] Using tokenizer from: {extracted_path}")
    
    return extracted_path
