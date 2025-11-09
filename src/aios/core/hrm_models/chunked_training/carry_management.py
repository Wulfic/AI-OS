"""
Carry state management for chunked training.

Functions for compressing carry states and gradient checkpointing.
"""

from __future__ import annotations

from typing import Tuple
import gc
import torch


def _compress_carry_state(carry, device: torch.device, model=None):
    """
    Compress carry state to keep only the last position's hidden states.
    
    This is critical for memory efficiency: instead of keeping [B, chunk_size, hidden],
    we only keep [B, 1, hidden] which is the recurrent state needed for the next chunk.
    
    For a 2048 chunk size, this saves 2047x memory on the carry state!
    
    Args:
        carry: The carry state to compress
        device: The device for cache clearing
        model: Optional model reference to get puzzle_emb_len
    """
    if not hasattr(carry, 'inner_carry'):
        return carry
    
    inner = carry.inner_carry
    
    # Get puzzle embedding length if available
    puzzle_emb_len = 0
    if model is not None and hasattr(model, 'puzzle_emb_len'):
        puzzle_emb_len = model.puzzle_emb_len
    
    # Extract only the LAST position from hidden states (this is the recurrent carry)
    # We use .detach() since this is just state, not requiring gradients through chunks
    # Keep puzzle embeddings + last position only
    if hasattr(inner, 'z_H') and inner.z_H is not None:
        if inner.z_H.dim() >= 2:
            seq_len = inner.z_H.size(1)
            if seq_len > puzzle_emb_len + 1:
                # Keep: [puzzle_emb : last_position] = [:puzzle_emb_len] + [-1:]
                if puzzle_emb_len > 0:
                    puzzle_part = inner.z_H[:, :puzzle_emb_len, ...].detach()
                    last_pos = inner.z_H[:, -1:, ...].detach()
                    inner.z_H = torch.cat([puzzle_part, last_pos], dim=1).clone()
                else:
                    inner.z_H = inner.z_H[:, -1:, ...].detach().clone()
    
    if hasattr(inner, 'z_L') and inner.z_L is not None:
        if inner.z_L.dim() >= 2:
            seq_len = inner.z_L.size(1)
            if seq_len > puzzle_emb_len + 1:
                if puzzle_emb_len > 0:
                    puzzle_part = inner.z_L[:, :puzzle_emb_len, ...].detach()
                    last_pos = inner.z_L[:, -1:, ...].detach()
                    inner.z_L = torch.cat([puzzle_part, last_pos], dim=1).clone()
                else:
                    inner.z_L = inner.z_L[:, -1:, ...].detach().clone()
    
    # Clear CUDA cache after compression
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return carry


def _checkpoint_forward(model, carry, batch) -> Tuple:
    """Wrapper for gradient checkpointing with memory cleanup.
    
    Returns:
        Tuple of (carry, outputs) from the model forward pass
    """
    try:
        from torch.utils.checkpoint import checkpoint
        # For HRM models, gradient checkpointing is complex due to carry state
        # For now, just do regular forward - proper checkpointing would require
        # more careful handling of the carry state and custom autograd functions
        return model(carry, batch)
    except Exception as e:
        # On exception, try to free memory before re-raising
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        # Fallback attempt with cleanup
        try:
            return model(carry, batch)
        except Exception:
            # Re-raise original exception if fallback also fails
            raise e
