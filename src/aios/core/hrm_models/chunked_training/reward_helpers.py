"""
Reward computation helpers for chunked training.

Functions for computing minimum halt steps and exact match rewards.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def _compute_min_halt_steps(batch_size: int, mmax: int, epsilon: float, device: torch.device) -> torch.Tensor:
    """Sample M_min per paper: with prob epsilon uniform in [2..Mmax], else 1."""
    if mmax <= 1:
        return torch.ones(batch_size, dtype=torch.int32, device=device)
    rand = torch.rand(batch_size, device=device)
    long_think = rand < epsilon
    mins = torch.ones(batch_size, dtype=torch.int32, device=device)
    if long_think.any():
        mins[long_think] = torch.randint(
            2, mmax + 1, (int(long_think.sum().item()),), device=device, dtype=torch.int32
        )
    return mins


@torch.no_grad()
def _exact_match_reward(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Return a binary reward per-sample indicating full-sequence exact match.
    
    Args:
        logits: [B, S, V] predicted logits
        targets: [B, S] target token IDs
        ignore_index: Index to ignore in matching
        
    Returns:
        [B] binary reward (1.0 if exact match, 0.0 otherwise)
    """
    pred = logits.argmax(dim=-1)
    mask = (targets != ignore_index)
    eq = (pred == targets) | (~mask)
    correct = eq.all(dim=-1).to(torch.float32)
    return correct


@torch.no_grad()
def _exact_match_reward_from_preds(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Return a binary reward per-sample indicating full-sequence exact match.
    Memory-efficient version that works directly with predictions instead of logits.
    
    Args:
        predictions: [B, S] predicted token IDs (argmax of logits)
        targets: [B, S] target token IDs
        ignore_index: Index to ignore in matching
        
    Returns:
        [B] binary reward (1.0 if exact match, 0.0 otherwise)
    """
    mask = (targets != ignore_index)
    eq = (predictions == targets) | (~mask)
    correct = eq.all(dim=-1).to(torch.float32)
    return correct
