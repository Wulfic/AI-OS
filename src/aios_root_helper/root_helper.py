from __future__ import annotations
import asyncio
import logging
from pathlib import Path

from dbus_next.aio.message_bus import MessageBus
# Import dbus-next, but wrap decorators for compatibility so tests can import and
# call methods directly without requiring exact decorator signatures.
from dbus_next.service import (
    ServiceInterface,
    method as _dbus_method,
    dbus_property as _dbus_property,
)
from dbus_next.constants import PropertyAccess, BusType

# Compatibility shims: some environments have differing dbus-next APIs.
# We accept any kwargs and fall back to no-op/property when unsupported so
# unit tests (which call methods directly) continue to work.
def method(*args, **kwargs):  # type: ignore[misc]
    try:
        return _dbus_method(*args, **kwargs)
    except TypeError:
        def _decorator(fn):
            return fn
        return _decorator


def dbus_property(*args, **kwargs):  # type: ignore[misc]
    try:
        return _dbus_property(*args, **kwargs)
    except TypeError:
        def _decorator(fn):
            return property(fn)
        return _decorator

BUS_NAME = "com.aios.RootHelper.v1"
OBJ_PATH = "/com/aios/RootHelper"

logger = logging.getLogger("aios.root_helper")


class FileOps(ServiceInterface):
    def __init__(self):
        super().__init__("com.aios.RootHelper.FileOps")

    @method(in_signature="s", out_signature="ay")
    def Read(self, path):
        p = Path(path)
        # Read-only guard: only allow reading under /etc and /var/log initially
        if not (str(p).startswith("/etc/") or str(p).startswith("/var/log/")):
            logger.warning("FileOps.Read denied path outside allowlist: %s", path)
            return []
        try:
            data = p.read_bytes()[: 1024 * 512]  # cap 512KB
            logger.info("FileOps.Read ok path=%s size=%d", path, len(data))
            return list(data)
        except Exception:
            logger.exception("FileOps.Read error path=%s", path)
            return []

    @method(in_signature="s", out_signature="as")
    def Ls(self, path):
        p = Path(path)
        if not p.exists():
            logger.warning("FileOps.Ls path does not exist: %s", path)
            return []
        try:
            items = [str(x) for x in p.iterdir()]
            logger.info("FileOps.Ls ok path=%s count=%d", path, len(items))
            return items
        except Exception:
            logger.exception("FileOps.Ls error path=%s", path)
            return []


class Health(ServiceInterface):
    def __init__(self):
        super().__init__("com.aios.RootHelper.Health")
        self._ok = True

    @method(out_signature="b")
    def Ping(self):
        logger.debug("Health.Ping")
        return True

    @dbus_property(access=PropertyAccess.READ, signature="b")
    def Alive(self):
        return self._ok


class SystemOps(ServiceInterface):
    def __init__(self):
        super().__init__("com.aios.RootHelper.SystemOps")

    @method(in_signature="s", out_signature="s")
    def Status(self, unit):
        """Return `systemctl status --no-pager --lines=5 <unit>` output (read-only)."""
        import subprocess

        if not _is_safe_unit(unit):
            logger.warning("SystemOps.Status denied unit=%s", unit)
            return ""
        try:
            out = subprocess.check_output(
                ["/usr/bin/systemctl", "status", "--no-pager", "--lines=5", unit],
                text=True,
                stderr=subprocess.STDOUT,
            )
            logger.info("SystemOps.Status ok unit=%s len=%d", unit, len(out))
            return out
        except Exception:
            logger.exception("SystemOps.Status error unit=%s", unit)
            return ""

    @method(in_signature="sas", out_signature="a{ss}")
    def Show(self, unit, keys):
        """Return selected fields from `systemctl show` as a dict of string->string."""
        import subprocess

        out: dict[str, str] = {}
        if not _is_safe_unit(unit):
            logger.warning("SystemOps.Show denied unit=%s", unit)
            return out
        try:
            cmd = [
                "/usr/bin/systemctl",
                "show",
                unit,
                "--no-pager",
            ]
            for k in list(keys or []):
                # Basic sanity filter on keys (letters/digits/underscore only)
                if _is_safe_key(k):
                    cmd.extend(["-p", k])
            txt = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            for line in (txt or "").splitlines():
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if _is_safe_key(k):
                    out[k] = v
            logger.info("SystemOps.Show ok unit=%s keys=%d", unit, len(out))
        except Exception:
            logger.exception("SystemOps.Show error unit=%s", unit)
        return out

    @method(in_signature="s", out_signature="s")
    def FragmentPath(self, unit):
        """Return FragmentPath for a unit (string) or empty on error."""
        import subprocess

        if not _is_safe_unit(unit):
            logger.warning("SystemOps.FragmentPath denied unit=%s", unit)
            return ""
        try:
            txt = subprocess.check_output(
                [
                    "/usr/bin/systemctl",
                    "show",
                    unit,
                    "-p",
                    "FragmentPath",
                    "--no-pager",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
            for line in (txt or "").splitlines():
                if line.startswith("FragmentPath="):
                    frag = line.split("=", 1)[1].strip()
                    logger.info(
                        "SystemOps.FragmentPath ok unit=%s path=%s", unit, frag
                    )
                    return frag
        except Exception:
            logger.exception("SystemOps.FragmentPath error unit=%s", unit)
            return ""

    @method(in_signature="s", out_signature="as")
    def DropInFiles(self, unit):
        """Return absolute paths of any drop-in .conf files for the unit."""
        import subprocess
        import os

        files: list[str] = []
        if not _is_safe_unit(unit):
            logger.warning("SystemOps.DropInFiles denied unit=%s", unit)
            return files
        try:
            txt = subprocess.check_output(
                [
                    "/usr/bin/systemctl",
                    "show",
                    unit,
                    "-p",
                    "DropInPaths",
                    "--no-pager",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
            dropin_dirs: list[str] = []
            for line in (txt or "").splitlines():
                if line.startswith("DropInPaths="):
                    paths = line.split("=", 1)[1].strip()
                    dropin_dirs = [p for p in paths.split(" ") if p]
                    break
            for d in dropin_dirs:
                try:
                    for name in os.listdir(d):
                        if name.endswith(".conf"):
                            files.append(os.path.join(d, name))
                except Exception:
                    continue
            logger.info("SystemOps.DropInFiles ok unit=%s count=%d", unit, len(files))
        except Exception:
            logger.exception("SystemOps.DropInFiles error unit=%s", unit)
        return files


class JournalOps(ServiceInterface):
    def __init__(self):
        super().__init__("com.aios.RootHelper.JournalOps")

    @method(in_signature="si", out_signature="s")
    def Read(self, unit, lines):
        """Return last N lines of journal for a unit (read-only)."""
        import subprocess

        if not _is_safe_unit(unit):
            logger.warning("JournalOps.Read denied unit=%s", unit)
            return ""
        try:
            n = max(1, min(int(lines), 200))
            out = subprocess.check_output(
                ["/usr/bin/journalctl", "-u", unit, "-n", str(n), "--no-pager"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            logger.info("JournalOps.Read ok unit=%s lines=%d len=%d", unit, n, len(out))
            return out
        except Exception:
            logger.exception("JournalOps.Read error unit=%s", unit)
            return ""

    @method(in_signature="ssssi", out_signature="s")
    def ReadWindowed(self, unit, since, until, priority, lines):
        """Return windowed journal for a unit with optional window and priority."""
        import subprocess

        if not _is_safe_unit(unit):
            logger.warning("JournalOps.ReadWindowed denied unit=%s", unit)
            return ""
        try:
            n = max(1, min(int(lines), 1000))
            cmd: list[str] = [
                "/usr/bin/journalctl",
                "-u",
                unit,
                "-n",
                str(n),
                "--no-pager",
            ]
            if since:
                cmd.extend(["--since", since])
            if until:
                cmd.extend(["--until", until])
            if priority:
                cmd.extend(["-p", priority])
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            logger.info(
                "JournalOps.ReadWindowed ok unit=%s lines=%d since=%s until=%s prio=%s len=%d",
                unit,
                n,
                since or "",
                until or "",
                priority or "",
                len(out),
            )
            return out
        except Exception:
            logger.exception("JournalOps.ReadWindowed error unit=%s", unit)
            return ""


async def main():
    # Configure basic logging; honor AIOS_LOG_LEVEL and prefer stderr in debug to surface logs
    import os, sys
    level = (os.environ.get("AIOS_LOG_LEVEL") or ("DEBUG" if os.environ.get("AIOS_DEBUG") else "INFO")).upper()
    try:
        lvl = getattr(logging, level, logging.INFO)
    except Exception:
        lvl = logging.INFO
    stream = sys.stderr if (os.environ.get("AIOS_DEBUG") or os.environ.get("VSCODE_PID")) else sys.stdout
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(lvl)
    root.handlers[:] = [handler]
    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    await bus.request_name(BUS_NAME)

    fileops = FileOps()
    health = Health()
    sysops = SystemOps()
    jops = JournalOps()

    bus.export(OBJ_PATH, fileops)
    bus.export(OBJ_PATH, health)
    bus.export(OBJ_PATH, sysops)
    bus.export(OBJ_PATH, jops)

    # Keep running
    await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


def _is_safe_unit(unit: str) -> bool:
    """Very basic validation for unit names to avoid shell injections.

    Allows common unit characters and suffixes. This does not check existence.
    """
    if not unit or len(unit) > 128:
        return False
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9@_.:\\-]+", unit))


def _is_safe_key(key: str) -> bool:
    if not key or len(key) > 64:
        return False
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9_]+", key))
