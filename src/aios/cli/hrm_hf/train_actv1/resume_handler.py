"""Resume checkpoint detection and handling.

Detects and validates checkpoint resume for continuing training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Optional, Dict, Any
from pathlib import Path
import json

from rich import print

if TYPE_CHECKING:
    pass


def detect_resume_checkpoint(
    resume: bool,
    brain_name: Optional[str],
    bundle_dir: str,
    dataset_file: Optional[str]
) -> Tuple[int, int, Optional[Dict[str, Any]]]:
    """Detect and validate checkpoint for resuming training.
    
    Args:
        resume: Whether resume mode is enabled
        brain_name: Name of brain bundle to resume
        bundle_dir: Base directory for brain bundles
        dataset_file: Current dataset file path
        
    Returns:
        Tuple of (step_offset, resume_cycle, resume_session_data)
        - step_offset: Number of steps already completed (0 if not resuming)
        - resume_cycle: Cycle number to resume from (0 if not resuming)
        - resume_session_data: Last session metadata dict (None if not resuming)
    """
    step_offset = 0  # Start from step 0 unless resuming
    resume_cycle = 0  # Start from cycle 0 unless resuming in iterate mode
    resume_session = None
    
    if not resume:
        return step_offset, resume_cycle, resume_session
    
    if not brain_name:
        print({
            "resume": "skipped",
            "reason": "no_brain_name_specified",
            "note": "Resume requires --brain-name parameter"
        })
        return step_offset, resume_cycle, resume_session
    
    # Check for existing brain.json with last_session data
    brain_path = Path(bundle_dir) / str(brain_name)
    brain_json_path = brain_path / "brain.json"
    
    if not brain_json_path.exists():
        print({
            "resume": "skipped",
            "reason": "brain_json_not_found",
            "note": "This appears to be the first training session"
        })
        return step_offset, resume_cycle, resume_session
    
    try:
        with brain_json_path.open("r", encoding="utf-8") as f:
            brain_data = json.load(f)
        
        last_session = brain_data.get("last_session")
        if not last_session or not isinstance(last_session, dict):
            print({
                "resume": "skipped",
                "reason": "no_last_session_metadata"
            })
            return step_offset, resume_cycle, resume_session
        
        # Check if checkpoint exists
        checkpoint_path = brain_path / "actv1_student.safetensors"
        if not checkpoint_path.exists():
            print({
                "resume": "skipped",
                "reason": "checkpoint_not_found",
                "expected_path": str(checkpoint_path)
            })
            return step_offset, resume_cycle, resume_session
        
        # Validate dataset matches
        saved_dataset = last_session.get("dataset_file")
        if saved_dataset and dataset_file and str(saved_dataset) != str(dataset_file):
            print({
                "resume": "skipped",
                "reason": "dataset_mismatch",
                "saved_dataset": saved_dataset,
                "current_dataset": dataset_file,
                "note": "Starting fresh training to prevent breaking training continuity"
            })
            return step_offset, resume_cycle, resume_session
        
        # Resume is valid!
        step_offset = last_session.get("total_steps", 0)
        resume_cycle = last_session.get("iterate_cycle", 0)
        resume_session = last_session
        
        print({
            "resume": "enabled",
            "checkpoint": str(checkpoint_path),
            "resuming_from_step": step_offset,
            "resuming_from_cycle": resume_cycle,
            "previous_steps": last_session.get("steps_completed", 0),
            "dataset": saved_dataset,
            "timestamp": last_session.get("timestamp"),
        })
        
    except Exception as e:
        print({
            "resume": "error",
            "message": str(e),
            "note": "Starting fresh training"
        })
    
    return step_offset, resume_cycle, resume_session
