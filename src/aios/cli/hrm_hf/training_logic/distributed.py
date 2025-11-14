"""Distributed training utilities."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def average_gradients_if_distributed(model_student, *, is_distributed: bool, world_sz: int) -> None:
    """Average gradients across processes when torch.distributed is initialized.

    Safe to call when not distributed; it will no-op.
    """
    if not is_distributed:
        return
    
    try:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            logger.debug("Distributed training requested but not initialized, skipping gradient averaging")
            return
        
        rank = dist.get_rank()
        logger.debug(f"Averaging gradients across {world_sz} processes (rank {rank})")
        
        grad_count = 0
        for p in model_student.parameters():
            if p.grad is None:
                continue
            try:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= float(max(1, world_sz))
                grad_count += 1
            except Exception as e:
                logger.warning(f"Failed to reduce gradient for parameter: {e}")
                pass
        
        logger.debug(f"Averaged {grad_count} gradients across {world_sz} processes")
        
    except Exception as e:
        logger.error(f"Gradient averaging failed: {e}")
