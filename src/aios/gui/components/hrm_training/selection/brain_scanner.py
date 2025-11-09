"""Brain scanning logic for discovering existing HRM student models.

This module scans the artifacts/brains/actv1 directory to find existing
trained/training brain checkpoints.
"""

from __future__ import annotations
import os
from typing import Dict


def scan_existing_brains(project_root: str) -> Dict[str, str]:
    """
    Scan artifacts/brains/actv1 for existing brain directories.
    
    Args:
        project_root: Path to project root directory
    
    Returns:
        Dictionary mapping brain names to their directory paths.
        Example: {"my_brain_v1": "/path/to/artifacts/brains/actv1/my_brain_v1"}
    """
    name_to_dir: Dict[str, str] = {}
    
    try:
        base = os.path.join(project_root, "artifacts", "brains", "actv1")
        if not os.path.isdir(base):
            return name_to_dir
        
        for entry in sorted(os.listdir(base)):
            # Skip internal/DDP directories (prefixed with _)
            if entry.startswith("_"):
                continue
            
            path = os.path.join(base, entry)
            if os.path.isdir(path):
                # Optional: validate that it's a real brain directory
                if validate_brain_dir(path):
                    name_to_dir[entry] = path
    except Exception:
        # Silently handle errors (directory doesn't exist, permissions, etc.)
        pass
    
    return name_to_dir


def validate_brain_dir(brain_dir: str) -> bool:
    """
    Validate that a directory is a legitimate brain directory.
    
    Args:
        brain_dir: Path to potential brain directory
    
    Returns:
        True if directory contains required brain files, False otherwise.
    """
    # A brain directory should have at least one of:
    # - brain.json (metadata)
    # - actv1_student.safetensors (checkpoint)
    # - metrics.jsonl (training log)
    
    required_files = [
        "brain.json",
        "actv1_student.safetensors",
        "metrics.jsonl"
    ]
    
    # Check if at least one file exists
    for filename in required_files:
        if os.path.isfile(os.path.join(brain_dir, filename)):
            return True
    
    return False


def get_brain_info(brain_dir: str) -> Dict[str, any]:
    """
    Get basic information about a brain from its directory.
    
    Args:
        brain_dir: Path to brain directory
    
    Returns:
        Dictionary with brain metadata (name, checkpoint status, training steps, etc.)
    """
    info = {
        "name": os.path.basename(brain_dir),
        "path": brain_dir,
        "has_checkpoint": False,
        "has_metadata": False,
        "has_training_log": False,
        "training_steps": 0,
    }
    
    # Check for checkpoint
    checkpoint_path = os.path.join(brain_dir, "actv1_student.safetensors")
    if os.path.isfile(checkpoint_path):
        info["has_checkpoint"] = True
    
    # Check for brain.json
    brain_json_path = os.path.join(brain_dir, "brain.json")
    if os.path.isfile(brain_json_path):
        info["has_metadata"] = True
        
        # Try to read training steps from metadata
        try:
            import json
            with open(brain_json_path, 'r', encoding='utf-8') as f:
                brain_data = json.load(f)
            info["training_steps"] = brain_data.get("training_steps", 0)
        except Exception:
            pass
    
    # Check for training log
    log_path = os.path.join(brain_dir, "metrics.jsonl")
    if os.path.isfile(log_path):
        info["has_training_log"] = True
    
    return info
