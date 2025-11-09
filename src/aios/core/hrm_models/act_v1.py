from __future__ import annotations

"""
Minimal, guarded integration of the vendored ACT V1 HRM model.

Notes:
- Torch and FlashAttention are optional. We guard imports and provide clear errors.
- This module mirrors the vendor structure with adjusted relative imports to core path.
- Training data is intentionally excluded.
"""

from typing import Any, Dict, Tuple


def _require_torch():
    try:
        import torch  # noqa: F401
        from torch import nn  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyTorch is required for aios.core.hrm_models.act_v1 but is not installed."
        ) from e


def build_model(config: Dict[str, Any]):
    """Factory that returns an instantiated ACT V1 model.

    Input: config dict as expected by HierarchicalReasoningModel_ACTV1Config.
    Output: a torch.nn.Module implementing forward(carry, batch) -> (carry, outputs).
    """
    _require_torch()
    # Local imports after torch availability
    from .impl.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # type: ignore

    return HierarchicalReasoningModel_ACTV1(config)


__all__ = ["build_model"]
