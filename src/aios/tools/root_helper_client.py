from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import List, Optional


BUS_NAME = "com.aios.RootHelper.v1"
OBJ_PATH = "/com/aios/RootHelper"


@dataclass
class RootHelperClient:
    """Thin async client for the root-helper D-Bus service.

    Safe fallbacks: if not on Linux, dbus-next missing, system bus unavailable,
    or the root-helper service is not running, methods return benign defaults
    instead of raising (None/empty values).
    """

    timeout_sec: float = 2.0

    def __post_init__(self) -> None:
        self._bus = None
        self._fileops = None
        self._health = None
        self._sysops = None
        self._jops = None

    async def connect(self) -> bool:
        if not sys.platform.startswith("linux"):
            return False
        try:
            from dbus_next.aio import MessageBus  # type: ignore
        except Exception:
            return False

        try:
            self._bus = await asyncio.wait_for(
                MessageBus(bus_type=MessageBus.TYPE_SYSTEM).connect(),
                timeout=self.timeout_sec,
            )
        except Exception:
            self._bus = None
            return False

        try:
            introspect = await asyncio.wait_for(
                self._bus.introspect(BUS_NAME, OBJ_PATH), timeout=self.timeout_sec
            )
            obj = self._bus.get_proxy_object(BUS_NAME, OBJ_PATH, introspect)
            self._fileops = obj.get_interface("com.aios.RootHelper.FileOps")
            self._health = obj.get_interface("com.aios.RootHelper.Health")
            self._sysops = obj.get_interface("com.aios.RootHelper.SystemOps")
            self._jops = obj.get_interface("com.aios.RootHelper.JournalOps")
            return True
        except Exception:
            # Service not present or not export interfaces; treat as disconnected
            self._fileops = None
            self._health = None
            self._sysops = None
            self._jops = None
            return False

    async def close(self) -> None:
        try:
            if self._bus is not None:
                self._bus.disconnect()  # type: ignore
        finally:
            self._bus = None
            self._fileops = None
            self._health = None
            self._sysops = None
            self._jops = None

    async def ping(self) -> Optional[bool]:
        if self._health is None:
            return None
        try:
            # method name is Ping → call_ping()
            return await asyncio.wait_for(self._health.call_ping(), timeout=self.timeout_sec)  # type: ignore[attr-defined]
        except Exception:
            return None

    async def read_file(self, path: str) -> Optional[bytes]:
        if self._fileops is None:
            return None
        try:
            data = await asyncio.wait_for(
                self._fileops.call_read(path),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            # D-Bus returns ay → list[int], convert to bytes
            return bytes(data or [])
        except Exception:
            return None

    async def ls(self, path: str) -> List[str]:
        if self._fileops is None:
            return []
        try:
            items = await asyncio.wait_for(
                self._fileops.call_ls(path),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            return list(items or [])
        except Exception:
            return []

    async def system_status(self, unit: str) -> Optional[str]:
        if self._sysops is None:
            return None
        try:
            out = await asyncio.wait_for(
                self._sysops.call_status(unit),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            return str(out)
        except Exception:
            return None

    async def journal_read(self, unit: str, lines: int = 20) -> Optional[str]:
        if self._jops is None:
            return None
        try:
            out = await asyncio.wait_for(
                self._jops.call_read(unit, int(lines)),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            return str(out)
        except Exception:
            return None

    async def system_show(self, unit: str, keys: list[str]) -> dict[str, str]:
        if self._sysops is None:
            return {}
        try:
            res = await asyncio.wait_for(
                self._sysops.call_show(unit, keys),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            # dbus-next maps a{ss} to dict[str,str]
            return dict(res or {})
        except Exception:
            return {}

    async def system_fragment_path(self, unit: str) -> Optional[str]:
        if self._sysops is None:
            return None
        try:
            res = await asyncio.wait_for(
                self._sysops.call_fragment_path(unit),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            return str(res) if res is not None else None
        except Exception:
            return None

    async def system_dropin_files(self, unit: str) -> list[str]:
        if self._sysops is None:
            return []
        try:
            res = await asyncio.wait_for(
                self._sysops.call_drop_in_files(unit),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            return list(res or [])
        except Exception:
            return []

    async def journal_read_windowed(
        self, unit: str, since: str | None, until: str | None, priority: str | None, lines: int
    ) -> Optional[str]:
        if self._jops is None:
            return None
        try:
            s = since or ""
            u = until or ""
            p = priority or ""
            out = await asyncio.wait_for(
                self._jops.call_read_windowed(unit, s, u, p, int(lines)),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            return str(out)
        except Exception:
            return None
