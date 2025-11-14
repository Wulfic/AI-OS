from __future__ import annotations
import asyncio
import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("aios.core.idle")


@dataclass
class IdleConfig:
    threshold_ms: int = 60000  # 60s
    poll_interval_s: float = 1.0


class IdleMonitor:
    """Detects user idle using GNOME Mutter IdleMonitor when available, with fallbacks.

    Emits state-change callbacks: on_idle() / on_active().
    """

    def __init__(self, cfg: Optional[IdleConfig] = None):
        self.cfg = cfg or IdleConfig()
        self._idle = False
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._on_idle: Optional[Callable[[], Awaitable[None]]] = None
        self._on_active: Optional[Callable[[], Awaitable[None]]] = None
        self._bg_tasks: set[asyncio.Task] = set()

    def is_idle(self) -> bool:
        return self._idle

    def on_idle(self, cb: Callable[[], Awaitable[None]]):
        self._on_idle = cb

    def on_active(self, cb: Callable[[], Awaitable[None]]):
        self._on_active = cb

    async def start(self):
        if self._task and not self._task.done():
            logger.debug("Idle monitor already running")
            return
        logger.info(f"Starting idle monitor (threshold={self.cfg.threshold_ms}ms, poll_interval={self.cfg.poll_interval_s}s)")
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        logger.info("Stopping idle monitor")
        self._stop.set()
        if self._task:
            await self._task
            self._task = None
        for task in list(self._bg_tasks):
            task.cancel()
        self._bg_tasks.clear()
        logger.debug("Idle monitor stopped")

    async def _run(self):
        while not self._stop.is_set():
            try:
                idle_ms = await self._get_idle_time_ms()
            except Exception as e:
                logger.debug("idle detection error: %s", e)
                idle_ms = 0

            now_idle = idle_ms >= self.cfg.threshold_ms
            if now_idle != self._idle:
                self._idle = now_idle
                if self._idle:
                    logger.info("Idle state entered (idle_ms=%s)", idle_ms)
                    if self._on_idle:
                        self._schedule_callback(self._on_idle)
                else:
                    logger.info("Active state detected (idle_ms=%s)", idle_ms)
                    if self._on_active:
                        self._schedule_callback(self._on_active)

            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.cfg.poll_interval_s
                )
            except asyncio.TimeoutError:
                pass

    async def _get_idle_time_ms(self) -> int:
        # Prefer Mutter IdleMonitor on GNOME/Wayland
        if platform.system() == "Linux":
            ms = await _mutter_idle_time_ms()
            if ms is not None:
                return ms
            # Fallback: loginctl IdleHint
            ms = await _loginctl_idle_time_ms()
            if ms is not None:
                return ms
        return 0

    def _schedule_callback(self, cb: Callable[[], Awaitable[None]]):
        async def _runner():
            try:
                await cb()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Idle callback raised an exception")

        task = asyncio.create_task(_runner())
        self._bg_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            self._bg_tasks.discard(t)
            # Exception is already logged inside _runner

        task.add_done_callback(_on_done)


async def _mutter_idle_time_ms() -> Optional[int]:
    try:
        from dbus_next.aio import MessageBus

        bus = await MessageBus().connect()
        introspection = await bus.introspect(
            "org.gnome.Mutter.IdleMonitor", "/org/gnome/Mutter/IdleMonitor/Core"
        )
        obj = bus.get_proxy_object(
            "org.gnome.Mutter.IdleMonitor",
            "/org/gnome/Mutter/IdleMonitor/Core",
            introspection,
        )
        iface = obj.get_interface("org.gnome.Mutter.IdleMonitor")
        # Method name: GetIdletime -> call_get_idletime
        idle_ms: int = await iface.call_get_idletime()
        return int(idle_ms)
    except Exception:
        return None


async def _loginctl_idle_time_ms() -> Optional[int]:
    if shutil.which("loginctl") is None:
        return None

    def _collect() -> Optional[int]:
        try:
            user = os.environ.get("USER") or os.environ.get("LOGNAME")
            sess = None
            out = subprocess.check_output(
                ["loginctl", "list-sessions", "--no-legend"], text=True
            )
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 3 and (user in parts[2:] if user else True):
                    sess = parts[0]
                    break
            if not sess:
                return None
            det = subprocess.check_output(["loginctl", "show-session", sess], text=True)
            idle_hint = re.search(r"^IdleHint=(yes|no)$", det, re.M | re.I)
            if not idle_hint:
                return None
            if idle_hint.group(1).lower() == "no":
                return 0
            return 60000
        except Exception:
            return None

    return await asyncio.to_thread(_collect)
