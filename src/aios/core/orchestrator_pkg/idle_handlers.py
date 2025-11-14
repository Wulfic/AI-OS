from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from aios.core.idle import IdleMonitor
from aios.core.brains import BrainRegistry, Router

logger = logging.getLogger("aios.core.orchestrator")


def bind_idle_handlers(
    idle: IdleMonitor,
    *,
    brains_cfg: Dict[str, Any],
    registry: Optional[BrainRegistry],
    router: Optional[Router],
) -> None:
    async def on_idle():
        logger.info("Idle detected: entering self-improvement cadence")
        if router is not None:
            try:
                res = await asyncio.to_thread(
                    router.handle,
                    {"modalities": ["text"], "payload": {"ts": int(0)}},
                )
                logger.debug("router.handle result: %s", res)
            except Exception:
                logger.exception("router idle probe failed")
            # Auto-prune if above limits
            try:
                if registry is not None:
                    stats = await asyncio.to_thread(registry.stats)
                    used = 0
                    if isinstance(stats, dict):
                        used = stats.get("used_bytes", 0)
                    cap_mb = float(brains_cfg.get("storage_limit_mb", 0) or 0)
                    cap_bytes = int(cap_mb * 1024 * 1024) if cap_mb > 0 else None
                    if cap_bytes and used > cap_bytes:
                        ev = await asyncio.to_thread(registry.prune, None, True)
                        logger.info("brains prune: evicted=%s", ev)
            except Exception:
                logger.exception("auto prune failed")

    async def on_active():
        logger.info("Activity detected: pausing background work")
        # ...pause or throttle tasks here...

    idle.on_idle(on_idle)
    idle.on_active(on_active)
