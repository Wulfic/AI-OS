"""
Unlimiformer integration (planned, Phase 1).

This package will house a minimal, vendored subset of the Unlimiformer codebase
(MIT license) adapted to AI-OS. Initial focus: decoder-only HF models (e.g., LLaMA family)
for long-context evaluation using a FAISS-based datastore.

Notes:
- Feature is gated off by default; enabling requires config/CLI flags.
- CPU FAISS is the default on Windows. GPU datastore/index is supported on CUDA platforms only.
- Training-time integration is slated for Phase 2.
"""

from typing import Any, Optional, Protocol, TypedDict

class UnlimiformerConfig(TypedDict, total=False):
    enabled: bool
    layer_begin: Optional[int]
    context_window_size: int
    knn: bool
    use_datastore: bool
    gpu_datastore: bool
    gpu_index: bool
    datastore_device: Optional[int]
    index_devices: Optional[list[int]]
    eval_max_source_length: int

class SupportsHFModel(Protocol):
    def generate(self, *args: Any, **kwargs: Any) -> Any: ...


def is_enabled(cfg: Optional[UnlimiformerConfig]) -> bool:
    return bool(cfg and cfg.get("enabled"))


def enable_on_model(model: SupportsHFModel, tokenizer: Any, cfg: UnlimiformerConfig) -> SupportsHFModel:
    """Placeholder hook for Phase 1: attach Unlimiformer components to model.

    For now, this is a no-op stub. In Phase 1, we'll vendor the minimal required
    Unlimiformer modules and apply the augmentation here.
    """
    # TODO (Phase 1): Wire Unlimiformer augmentation once modules are vendored.
    return model  # type: ignore[return-value]
