"""I/O and persistence configuration fields.

Checkpointing, logging, and file paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class IOFields:
    """I/O and persistence parameters."""
    
    # ============================================================================
    # I/O and Persistence
    # ============================================================================
    save_dir: str = "training_data/actv1"
    """Directory where checkpoints and training state are saved."""
    
    stop_file: Optional[str] = None
    """If file exists, training stops gracefully.
    
    Useful for remote monitoring and controlled shutdown.
    """
    
    log_file: Optional[str] = None
    """Optional JSONL file to append training/eval metrics.
    
    Each line is a JSON object with step, loss, etc.
    """
    
    student_init: Optional[str] = None
    """Optional path to existing ACTV1 student state_dict (.pt) to continue training.
    
    If provided, training resumes from this checkpoint.
    """
    
    brain_name: Optional[str] = None
    """Optional brain bundle name.
    
    When provided, saves to --bundle-dir/<brain-name>.
    """
    
    default_goal: Optional[str] = None
    """Default training goal/directive for this brain.
    
    This goal guides what the brain learns during training.
    Automatically added to the brain's goals when training starts.
    """
    
    expert_id: Optional[str] = None
    """Expert ID for expert-specific training mode.
    
    When provided, trains only a FeedForward expert module instead of full HRM model.
    The expert is saved to artifacts/experts/{expert_id}/ and registered in the
    expert registry. This enables training specialized experts for specific goals
    without affecting the base model.
    
    Expert training features:
    - Trains lightweight FeedForward module (hidden_size → intermediate_size → hidden_size)
    - Saves to artifacts/experts/{expert_id}/expert.pt
    - Auto-updates expert_registry.json with metadata
    - Can be loaded at runtime via LazyExpertLoader
    - Supports all optimization features (AMP, gradient checkpointing, etc.)
    
    If None (default), trains full HRM ACT-v1 model normally.
    """
    
    bundle_dir: str = "artifacts/brains/actv1"
    """Base directory for ACTV1 brain bundles.
    
    Brain-specific subdirectories are created here.
    
    Note: All checkpoints are saved in safetensors format for security and performance.
    Safetensors provides:
    - Security: No arbitrary code execution (unlike Pickle-based .pt)
    - Performance: Faster load/save with memory-mapped I/O
    - Interoperability: Standard across HuggingFace ecosystem
    """
