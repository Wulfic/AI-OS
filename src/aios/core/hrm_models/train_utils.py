from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _sequence_ce_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Token-averaged cross entropy over batch and sequence.

    logits: [B, S, V]
    targets: [B, S] (int64)
    """
    B, S, V = logits.shape
    
    # Check targets validity FIRST (before any computation)
    valid_mask = (targets != ignore_index)
    num_valid = valid_mask.sum().item()
    
    # Early return for padding-only sequences (common in first/last chunks)
    if num_valid == 0:
        return torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    
    # Force FP32 for numerical stability (critical for AMP)
    logits = logits.to(torch.float32)
    
    # Check for NaN/Inf BEFORE clipping (memory-efficient: use single isfinite check)
    # With large vocabularies (150K+), separate isnan/isinf checks create huge intermediate tensors
    if not torch.isfinite(logits).all():
        nan_count = torch.isnan(logits).sum().item()
        inf_count = torch.isinf(logits).sum().item()
        logger.warning(f"[CE_LOSS] Input logits: {nan_count} NaNs, {inf_count} Infs out of {logits.numel()} values")
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    
    # Clip logits to prevent numerical instability with small hidden sizes and large vocab
    # Large vocabularies (50K+) with small hidden dimensions (256) can produce extreme logits
    logits_clipped = torch.clamp(logits, min=-20.0, max=20.0)
    
    # Validate targets are in valid range
    if torch.isnan(targets.float()).any() or (targets < -100).any() or (targets >= V).any():
        logger.error(f"[CE_LOSS] Invalid targets! min={targets.min()}, max={targets.max()}, vocab_size={V}")
    
    loss = F.cross_entropy(
        logits_clipped.view(B * S, V), targets.view(B * S), ignore_index=ignore_index, reduction="mean"
    )
    
    # Check loss validity (memory-efficient for scalar)
    if not torch.isfinite(loss):
        logger.error(f"[CE_LOSS] Output loss is NaN/Inf (input shape={logits.shape}, valid_tokens={num_valid})")
        return torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    
    return loss


def _exact_match_reward(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Return a binary reward per-sample indicating full-sequence exact match.

    logits: [B, S, V], targets: [B, S]
    """
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        # Treat ignored positions as trivially correct so they don't break exactness
        mask = (targets != ignore_index)
        eq = (pred == targets) | (~mask)
        correct = eq.all(dim=-1).to(torch.float32)
    return correct


@torch.no_grad()
def _compute_min_halt_steps(batch_size: int, mmax: int, epsilon: float, device: torch.device) -> torch.Tensor:
    """Sample M_min per paper: with prob epsilon uniform in [2..Mmax], else 1.
    Returns tensor shape [B] of minimum required segments.
    """
    if mmax <= 1:
        return torch.ones(batch_size, dtype=torch.int32, device=device)
    rand = torch.rand(batch_size, device=device)
    long_think = rand < epsilon
    mins = torch.ones(batch_size, dtype=torch.int32, device=device)
    if long_think.any():
        mins[long_think] = torch.randint(2, mmax + 1, (int(long_think.sum().item()),), device=device, dtype=torch.int32)
    return mins


def segment_rollout(
    model,
    batch: Dict[str, torch.Tensor],
    max_segments: int,
    epsilon: float = 0.0,
    ce_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run deep supervision segments with ACT-style halting targets.

    Returns (total_loss, metrics) where metrics includes ce, bce_halt, bce_continue and used_segments.
    Expects batch to have keys: inputs [B,S], puzzle_identifiers [B], targets [B,S].
    """
    assert model.training, "Call model.train() before segment_rollout to enable training paths."

    inputs = batch["inputs"]
    device = inputs.device
    B = inputs.shape[0]
    
    # Validate targets before training
    targets = batch["targets"].to(torch.int64)
    if torch.isnan(targets.float()).any():
        logger.warning(f"[segment_rollout] NaN in targets! Replacing with ignore_index")
        targets = torch.where(torch.isnan(targets.float()), ignore_index, targets)
    
    # Get vocab_size from model to validate targets
    # For DeepSpeed engines, access the actual model through .module
    # For DDP, also access through .module
    # For regular models, use directly
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    vocab_size = actual_model.inner.embed_tokens.embedding_weight.shape[0]
    
    # For forward passes, use the wrapper (DeepSpeed/DDP) if it exists
    # This ensures proper device placement and distributed handling
    model_for_forward = model
    
    if (targets < ignore_index).any() or (targets >= vocab_size).any():
        valid_mask = (targets != ignore_index) & ((targets < 0) | (targets >= vocab_size))
        if valid_mask.any():
            logger.error(f"[segment_rollout] Invalid target values! min={targets.min()}, max={targets.max()}, vocab_size={vocab_size}")
            targets = torch.clamp(targets, min=ignore_index, max=vocab_size - 1)
    batch["targets"] = targets

    ce_loss_fn = ce_loss_fn or (lambda lg, tg: _sequence_ce_loss(lg, tg, ignore_index=ignore_index))

    carry = actual_model.initial_carry(batch)
    used_segments = 0
    total_ce = torch.zeros((), device=device)
    total_bce_halt = torch.zeros((), device=device)
    total_bce_continue = torch.zeros((), device=device)

    # Determine per-sample minimum segments
    mmin = _compute_min_halt_steps(B, max_segments, epsilon, device)

    for m in range(max_segments):
        used_segments += 1
        # Use model_for_forward for forward pass (handles DeepSpeed/DDP device placement)
        carry, outputs = model_for_forward(carry, batch)
        logits = outputs["logits"]
        qh = outputs["q_halt_logits"]  # [B]
        qc = outputs["q_continue_logits"]  # [B]

        # Supervision: sequence CE
        ce = ce_loss_fn(logits, batch["targets"].to(torch.int64))
        
        # Debug: Check CE loss validity
        if torch.isnan(ce).any() or torch.isinf(ce).any():
            logger.error(f"[segment_rollout] NaN/Inf CE at segment {m}! logits shape={logits.shape}, dtype={logits.dtype}")
            logger.error(f"[segment_rollout] logits stats: min={logits.min():.2f}, max={logits.max():.2f}, mean={logits.mean():.2f}")
            ce = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        total_ce = total_ce + ce

        # Q-learning targets
        r_halt = _exact_match_reward(logits, batch["targets"].to(torch.int64), ignore_index=ignore_index)
        # Continue target comes from model’s lookahead (sigmoid-scaled) if provided
        if "target_q_continue" in outputs:
            # target comes from model prediction; treat as fixed target without grad
            tgt_continue = outputs["target_q_continue"].detach().clamp(0.0, 1.0)
        else:
            # Fallback: encourage thinking at least one more step
            tgt_continue = torch.sigmoid(qh.detach())

        # BCE loss for Q-values, masked for min_steps
        # Only train halt predictor after min_steps
        halt_mask = (m + 1 >= mmin)
        if halt_mask.any():
            total_bce_halt = total_bce_halt + F.binary_cross_entropy_with_logits(
                qh[halt_mask], r_halt[halt_mask], reduction="mean"
            )

        # Always train continue predictor
        total_bce_continue = total_bce_continue + F.binary_cross_entropy_with_logits(
            qc, tgt_continue, reduction="mean"
        )

        # Check for early exit (all samples in batch have halted)
        if carry.halted.all():
            break

    # Combine losses with NaN check
    total_loss = total_ce + total_bce_halt + total_bce_continue
    
    # Check for NaN/Inf in loss components
    if torch.isnan(total_ce).any() or torch.isinf(total_ce).any():
        logger.warning(f"[segment_rollout] WARNING: NaN/Inf in CE loss: {total_ce}")
    if torch.isnan(total_bce_halt).any() or torch.isinf(total_bce_halt).any():
        logger.warning(f"[segment_rollout] WARNING: NaN/Inf in BCE halt loss: {total_bce_halt}")
    if torch.isnan(total_bce_continue).any() or torch.isinf(total_bce_continue).any():
        logger.warning(f"[segment_rollout] WARNING: NaN/Inf in BCE continue loss: {total_bce_continue}")

    metrics = {
        "ce": total_ce,
        "bce_halt": total_bce_halt,
        "bce_continue": total_bce_continue,
        "used_segments": torch.tensor(float(used_segments), device=device),
    }
    return total_loss, metrics
