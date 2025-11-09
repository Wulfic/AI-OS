"""Resume from checkpoint detection and state restoration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def detect_resume_state(
    config: "TrainingConfig",
    bundle_dir: str,
    brain_name: Optional[str],
    log_fn
) -> Tuple[int, int, Optional[dict]]:
    """Detect if we should resume from a previous checkpoint.
    
    Returns:
        Tuple of (step_offset, resume_cycle, resume_session)
    """
    step_offset = 0
    resume_cycle = 0
    resume_session = None
    
    if not config.resume or not brain_name:
        if config.resume:
            log_fn({
                "resume": "skipped",
                "reason": "no_brain_name_specified",
                "note": "Resume requires --brain-name parameter"
            })
        return step_offset, resume_cycle, resume_session
    
    brain_path = Path(bundle_dir) / str(brain_name)
    brain_json_path = brain_path / "brain.json"
    
    if not brain_json_path.exists():
        log_fn({
            "resume": "skipped",
            "reason": "brain_json_not_found",
            "note": "This appears to be the first training session"
        })
        return step_offset, resume_cycle, resume_session
    
    try:
        with open(brain_json_path, 'r', encoding='utf-8') as f:
            brain_data = json.load(f)
        
        resume_session = brain_data.get("last_session")
        if resume_session:
            # Handle both 'total_steps' (from finalization) and 'steps_completed' (from checkpoint_saver)
            step_offset = resume_session.get("total_steps", 0) or resume_session.get("steps_completed", 0)
            # Field is saved as 'iterate_cycle' in finalization.py
            resume_cycle = resume_session.get("iterate_cycle", 0) or resume_session.get("cycle", 0)
            
            log_fn({
                "resume": "detected",
                "brain_name": brain_name,
                "previous_session": {
                    "total_steps": step_offset,
                    "cycle": resume_cycle,
                    "timestamp": resume_session.get("timestamp"),
                    "stop_reason": resume_session.get("stop_reason"),
                },
                "action": f"Resuming from step {step_offset}, cycle {resume_cycle}"
            })
        else:
            log_fn({
                "resume": "skipped",
                "reason": "no_last_session_in_brain_json",
                "note": "brain.json exists but has no last_session data",
                "brain_json_keys": list(brain_data.keys()) if brain_data else []
            })
    except Exception as e:
        log_fn({
            "resume": "error",
            "error": str(e),
            "note": "Failed to parse brain.json - starting fresh"
        })
    
    return step_offset, resume_cycle, resume_session
