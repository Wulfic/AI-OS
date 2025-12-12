from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict
from pathlib import Path

from aios.core.idle import IdleMonitor, IdleConfig
from aios.core.budgets import SafetyBudget
from aios.core.brains import BrainRegistry, Router
from .orchestrator_pkg import (
    open_db,
    compute_limits_and_usage,
    ensure_budgets_in_db,
    build_registry_and_router,
    warmup_router,
    bind_idle_handlers,
)

logger = logging.getLogger("aios.core.orchestrator")


@dataclass
class Orchestrator:
    config: Dict[str, Any] = field(default_factory=dict)

    def status(self) -> Dict[str, Any]:
        limits, used = compute_limits_and_usage(self.config)
        budgets = SafetyBudget(limits=limits, usage=used)
        return {
            "version": "1.0.14",
            "autonomy": self.config.get("autonomy", {}).get("mode", "autonomous_on"),
            "risk_tier": self.config.get("risk_tier", "conservative"),
            "budgets": budgets.summary(),
        }

    def run(self) -> None:
        logger.info(
            "Starting Orchestrator loop",
            extra={"autonomy": self.config.get("autonomy")},
        )
        try:
            asyncio.run(self._main())
        except KeyboardInterrupt:
            logger.info("Shutting down (KeyboardInterrupt)")

    async def _main(self) -> None:
        # Init DB
        db_path = self.config.get("paths", {}).get("db_path")
        conn = open_db(db_path)
        # Initialize budgets in DB with tier defaults if not set
        try:
            ensure_budgets_in_db(conn, config=self.config)
        except Exception:
            pass

        cfg = self.config
        idle_ms = int(cfg.get("autonomy", {}).get("idle_threshold_ms", 60000))
        idle = IdleMonitor(IdleConfig(threshold_ms=idle_ms))

        # Optional: initialize registry/router from config
        brains_cfg = (self.config.get("brains") or {}) if isinstance(self.config, dict) else {}
        registry, router = build_registry_and_router(brains_cfg, perform_warmup=False)

        warmup_task: asyncio.Task[None] | None = None
        if router is not None:
            # Kick off heavy router warmup work on a worker thread to keep the loop responsive.
            warmup_task = asyncio.create_task(warmup_router(router))

        def _log_warmup_result(task: asyncio.Task[None]) -> None:
            try:
                task.result()
            except Exception:
                logger.exception("Router warmup failed")

        if warmup_task is not None:
            warmup_task.add_done_callback(_log_warmup_result)

        bind_idle_handlers(idle, brains_cfg=brains_cfg, registry=registry, router=router)
        await idle.start()

        try:
            while True:
                await asyncio.sleep(5)
                # ...other periodic orchestration can run here...
        finally:
            await idle.stop()
