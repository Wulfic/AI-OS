from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from aios.core.brains import BrainRegistry, Router


def build_registry_and_router(brains_cfg: Dict[str, Any]) -> Tuple[BrainRegistry | None, Router | None]:
    brains_enabled = bool(brains_cfg.get("enabled", True))
    if not brains_enabled:
        return None, None

    # env overrides (optional)
    lim_mb_env = os.environ.get("AIOS_BRAINS_LIMIT_GB")
    per_mb_env = os.environ.get("AIOS_BRAIN_WIDTH_LIMIT_GB")
    lim_mb = float(brains_cfg.get("storage_limit_mb", 0) or 0)
    if lim_mb_env:
        try:
            lim_mb = float(lim_mb_env) * 1024.0
        except Exception:
            pass
    registry = BrainRegistry(total_storage_limit_mb=lim_mb or None)
    registry.store_dir = str(brains_cfg.get("store_dir", "artifacts/brains"))
    # modality caps (GB in config, convert to MB)
    caps = brains_cfg.get("storage_limits_gb_by_modality", {}) or {}
    try:
        registry.storage_limit_mb_by_modality = {str(k).strip().lower(): float(v) * 1024.0 for k, v in caps.items()}
    except Exception:
        registry.storage_limit_mb_by_modality = {}
    create_cfg = dict(brains_cfg.get("trainer_overrides", {}))
    if per_mb_env:
        try:
            create_cfg["width_storage_limit_mb"] = float(per_mb_env) * 1024.0
            create_cfg.setdefault("dynamic_width", True)
        except Exception:
            pass
    router = Router(
        registry=registry,
        default_modalities=list(brains_cfg.get("default_modalities", ["text"])),
        brain_prefix=str(brains_cfg.get("prefix", "brain")),
        create_cfg=create_cfg,
        strategy=str(brains_cfg.get("strategy", "hash")),
        modality_overrides=dict(brains_cfg.get("modality_overrides", {})),
    )
    # Preload master text brain at startup (best-effort)
    try:
        _ = router.handle({"modalities": ["text"], "payload": "__warmup__"})
    except Exception:
        pass
    return registry, router
