from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


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
        logger.debug("Starting async D-Bus connection")
        if not sys.platform.startswith("linux"):
            return False
        try:
            from dbus_next.aio import MessageBus  # type: ignore
        except Exception as e:
            logger.error(f"Failed to import dbus_next: {e}")
            return False

        try:
            self._bus = await asyncio.wait_for(
                MessageBus(bus_type=MessageBus.TYPE_SYSTEM).connect(),
                timeout=self.timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus connection timeout ({self.timeout_sec}s)")
            self._bus = None
            return False
        except Exception as e:
            logger.error(f"D-Bus connection failed: {e}")
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
            logger.debug("Async D-Bus connection established")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus proxy initialization timeout ({self.timeout_sec}s)")
            # Service not present or not export interfaces; treat as disconnected
            self._fileops = None
            self._health = None
            self._sysops = None
            self._jops = None
            return False
        except Exception as e:
            logger.error(f"D-Bus proxy initialization failed: {e}")
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
            logger.debug("Async D-Bus call: ping")
            # method name is Ping → call_ping()
            result = await asyncio.wait_for(self._health.call_ping(), timeout=self.timeout_sec)  # type: ignore[attr-defined]
            logger.debug("Async D-Bus call completed: ping")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus timeout: ping ({self.timeout_sec}s)")
            return None
        except Exception as e:
            logger.error(f"D-Bus ping failed: {e}")
            return None

    async def read_file(self, path: str) -> Optional[bytes]:
        if self._fileops is None:
            return None
        try:
            logger.debug(f"Async D-Bus call: read_file({path})")
            data = await asyncio.wait_for(
                self._fileops.call_read(path),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            # D-Bus returns ay → list[int], convert to bytes
            result = bytes(data or [])
            logger.debug(f"Async D-Bus call completed: read_file({path})")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus timeout: read_file({path}) ({self.timeout_sec}s)")
            return None
        except Exception as e:
            logger.error(f"D-Bus read_file failed for {path}: {e}")
            return None

    async def ls(self, path: str) -> List[str]:
        if self._fileops is None:
            return []
        try:
            logger.debug(f"Async D-Bus call: ls({path})")
            items = await asyncio.wait_for(
                self._fileops.call_ls(path),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            result = list(items or [])
            logger.debug(f"Async D-Bus call completed: ls({path}) - {len(result)} items")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus timeout: ls({path}) ({self.timeout_sec}s)")
            return []
        except Exception as e:
            logger.error(f"D-Bus ls failed for {path}: {e}")
            return []

    async def system_status(self, unit: str) -> Optional[str]:
        if self._sysops is None:
            return None
        try:
            logger.debug(f"Async D-Bus call: system_status({unit})")
            out = await asyncio.wait_for(
                self._sysops.call_status(unit),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            logger.debug(f"Async D-Bus call completed: system_status({unit})")
            return str(out)
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus timeout: system_status({unit}) ({self.timeout_sec}s)")
            return None
        except Exception as e:
            logger.error(f"D-Bus system_status failed for {unit}: {e}")
            return None

    async def journal_read(self, unit: str, lines: int = 20) -> Optional[str]:
        if self._jops is None:
            return None
        try:
            logger.debug(f"Async D-Bus call: journal_read({unit}, lines={lines})")
            out = await asyncio.wait_for(
                self._jops.call_read(unit, int(lines)),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            logger.debug(f"Async D-Bus call completed: journal_read({unit})")
            return str(out)
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus timeout: journal_read({unit}) ({self.timeout_sec}s)")
            return None
        except Exception as e:
            logger.error(f"D-Bus journal_read failed for {unit}: {e}")
            return None

    async def system_show(self, unit: str, keys: list[str]) -> dict[str, str]:
        if self._sysops is None:
            return {}
        try:
            logger.debug(f"Async D-Bus call: system_show({unit}, keys={keys})")
            res = await asyncio.wait_for(
                self._sysops.call_show(unit, keys),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            # dbus-next maps a{ss} to dict[str,str]
            result = dict(res or {})
            logger.debug(f"Async D-Bus call completed: system_show({unit}) - {len(result)} keys")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus timeout: system_show({unit}) ({self.timeout_sec}s)")
            return {}
        except Exception as e:
            logger.error(f"D-Bus system_show failed for {unit}: {e}")
            return {}

    async def system_fragment_path(self, unit: str) -> Optional[str]:
        if self._sysops is None:
            return None
        try:
            logger.debug(f"Async D-Bus call: system_fragment_path({unit})")
            res = await asyncio.wait_for(
                self._sysops.call_fragment_path(unit),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            logger.debug(f"Async D-Bus call completed: system_fragment_path({unit})")
            return str(res) if res is not None else None
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus timeout: system_fragment_path({unit}) ({self.timeout_sec}s)")
            return None
        except Exception as e:
            logger.error(f"D-Bus system_fragment_path failed for {unit}: {e}")
            return None

    async def system_dropin_files(self, unit: str) -> list[str]:
        if self._sysops is None:
            return []
        try:
            logger.debug(f"Async D-Bus call: system_dropin_files({unit})")
            res = await asyncio.wait_for(
                self._sysops.call_drop_in_files(unit),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            result = list(res or [])
            logger.debug(f"Async D-Bus call completed: system_dropin_files({unit}) - {len(result)} files")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus timeout: system_dropin_files({unit}) ({self.timeout_sec}s)")
            return []
        except Exception as e:
            logger.error(f"D-Bus system_dropin_files failed for {unit}: {e}")
            return []

    async def journal_read_windowed(
        self, unit: str, since: str | None, until: str | None, priority: str | None, lines: int
    ) -> Optional[str]:
        if self._jops is None:
            return None
        try:
            logger.debug(f"Async D-Bus call: journal_read_windowed({unit}, lines={lines})")
            s = since or ""
            u = until or ""
            p = priority or ""
            out = await asyncio.wait_for(
                self._jops.call_read_windowed(unit, s, u, p, int(lines)),  # type: ignore[attr-defined]
                timeout=self.timeout_sec,
            )
            logger.debug(f"Async D-Bus call completed: journal_read_windowed({unit})")
            return str(out)
        except asyncio.TimeoutError:
            logger.warning(f"Async D-Bus timeout: journal_read_windowed({unit}) ({self.timeout_sec}s)")
            return None
        except Exception as e:
            logger.error(f"D-Bus journal_read_windowed failed for {unit}: {e}")
            return None
