"""Distributed training utilities."""

from __future__ import annotations


def average_gradients_if_distributed(model_student, *, is_distributed: bool, world_sz: int) -> None:
    """Average gradients across processes when torch.distributed is initialized.

    Safe to call when not distributed; it will no-op.
    """
    if not is_distributed:
        return
    try:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        for p in model_student.parameters():
            if p.grad is None:
                continue
            try:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= float(max(1, world_sz))
            except Exception:
                pass
    except Exception:
        pass
