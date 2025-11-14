from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


async def system_status_and_journal(
    unit: str, *, lines: int = 20, timeout_sec: float = 1.5
) -> Dict[str, Optional[str]]:
    """Return a normalized dict with service status and journal tail.

    Prefers the root-helper D-Bus if available, otherwise falls back to local
    read-only commands (Linux only). On non-Linux platforms or when neither
    path is available, returns 'via': 'unavailable'.

    Output keys:
    - via: 'root-helper' | 'local' | 'unavailable'
    - status: Optional[str]
    - journal: Optional[str]
    """
    logger.debug(f"Starting async system_status_and_journal operation for unit: {unit}")
    unit = (unit or "").strip()
    if not unit:
        logger.debug("No unit provided, returning unavailable")
        return {"via": "unavailable", "status": None, "journal": None}

    if not sys.platform.startswith("linux"):
        logger.debug("Non-Linux platform, returning unavailable")
        return {"via": "unavailable", "status": None, "journal": None}

    # Try root-helper first
    try:
        logger.debug(f"Attempting root-helper D-Bus connection for {unit}")
        from aios.tools.root_helper_client import RootHelperClient

        client = RootHelperClient(timeout_sec=timeout_sec)
        if await client.connect():
            logger.debug(f"Root-helper connected for {unit}, fetching status and journal")
            try:
                status = await client.system_status(unit)
                journal = await client.journal_read(unit, int(lines))
                if (status and status.strip()) or (journal and journal.strip()):
                    logger.debug(f"Async operation completed via root-helper for {unit}")
                    return {"via": "root-helper", "status": status, "journal": journal}
                else:
                    # connected but no data; still report as root-helper path
                    logger.debug(f"Root-helper connected but no data for {unit}")
                    return {"via": "root-helper", "status": status, "journal": journal}
            finally:
                await client.close()
    except asyncio.TimeoutError:
        logger.warning(f"Root-helper connection timeout for unit {unit} ({timeout_sec}s)")
        logger.debug(f"Async operation timeout (root-helper) for {unit}")
    except Exception as e:
        logger.error(f"Root-helper connection failed for unit {unit}: {e}")
        logger.debug(f"Falling back to local commands for {unit}")
        # Import/connect errors fall through to local

    # Local fallback using systemctl/journalctl without pager
    def _run_local() -> Dict[str, Optional[str]]:
        try:
            status = subprocess.check_output(
                [
                    "/usr/bin/systemctl",
                    "status",
                    "--no-pager",
                    "--lines=5",
                    unit,
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            logger.error(f"Local systemctl status failed for {unit}: {e}")
            status = None
        try:
            n = max(1, min(int(lines), 200))
            journal = subprocess.check_output(
                [
                    "/usr/bin/journalctl",
                    "-u",
                    unit,
                    "-n",
                    str(n),
                    "--no-pager",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            logger.error(f"Local journalctl read failed for {unit}: {e}")
            journal = None
        return {"via": "local", "status": status, "journal": journal}

    try:
        # Run sync subprocess calls in a worker thread
        logger.debug(f"Running local systemctl/journalctl commands for {unit}")
        result = await asyncio.wait_for(asyncio.to_thread(_run_local), timeout=timeout_sec)
        logger.debug(f"Async operation completed via local commands for {unit}")
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Local command timeout for unit {unit} ({timeout_sec}s)")
        logger.debug(f"Async operation timeout (local) for {unit}")
        return {"via": "unavailable", "status": None, "journal": None}
    except Exception as e:
        logger.error(f"Local command execution failed for {unit}: {e}")
        logger.debug(f"Async operation failed for {unit}")
        return {"via": "unavailable", "status": None, "journal": None}


async def unit_dropins(
    unit: str, *, timeout_sec: float = 1.5
) -> Dict[str, Optional[str | List[str]]]:
    """Discover unit drop-in configuration files and fragment path.

    Returns a dict with keys:
    - via: 'root-helper' | 'local' | 'unavailable'
    - fragment_path: Optional[str]
    - dropins: Optional[List[str]] (absolute paths)

    Currently uses local systemctl on Linux. Root-helper path can be added later.
    """
    logger.debug(f"Starting async unit_dropins operation for unit: {unit}")
    unit = (unit or "").strip()
    if not unit:
        return {"via": "unavailable", "fragment_path": None, "dropins": None}
    if not sys.platform.startswith("linux"):
        return {"via": "unavailable", "fragment_path": None, "dropins": None}

    # Try root-helper first when available
    try:
        from aios.tools.root_helper_client import RootHelperClient

        client = RootHelperClient(timeout_sec=timeout_sec)
        if await client.connect():
            try:
                fragment = await client.system_fragment_path(unit)
                dropins = await client.system_dropin_files(unit)
                if fragment or dropins:
                    return {
                        "via": "root-helper",
                        "fragment_path": fragment,
                        "dropins": dropins if dropins else None,
                    }
            finally:
                await client.close()
    except Exception as e:
        logger.error(f"Failed to get unit dropins via root-helper for {unit}: {e}")

    def _local_show() -> Dict[str, Optional[str | List[str]]]:
        fragment: Optional[str] = None
        dropins: Optional[List[str]] = None
        try:
            out = subprocess.check_output(
                [
                    "/usr/bin/systemctl",
                    "show",
                    unit,
                    "-p",
                    "DropInPaths",
                    "-p",
                    "FragmentPath",
                    "--no-pager",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
            for line in (out or "").splitlines():
                if line.startswith("FragmentPath="):
                    fragment = line.split("=", 1)[1].strip() or None
                elif line.startswith("DropInPaths="):
                    paths = line.split("=", 1)[1].strip()
                    cand_dirs = [p for p in paths.split(" ") if p]
                    files: List[str] = []
                    for d in cand_dirs:
                        try:
                            for name in os.listdir(d):
                                if name.endswith(".conf"):
                                    files.append(os.path.join(d, name))
                        except Exception:
                            continue
                    dropins = files
        except Exception as e:
            logger.error(f"Failed to retrieve unit dropins locally for {unit}: {e}")
            fragment = None
            dropins = None
        return {"via": "local", "fragment_path": fragment, "dropins": dropins}

    try:
        return await asyncio.wait_for(asyncio.to_thread(_local_show), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout getting dropins for unit {unit} ({timeout_sec}s)")
        return {"via": "unavailable", "fragment_path": None, "dropins": None}
    except Exception as e:
        logger.error(f"Failed to get dropins for unit {unit}: {e}")
        return {"via": "unavailable", "fragment_path": None, "dropins": None}


async def service_show(
    unit: str,
    *,
    keys: Optional[List[str]] = None,
    timeout_sec: float = 1.5,
) -> Dict[str, Optional[str]]:
    """Return selected systemctl show fields for a unit.

    Defaults to common keys if none provided. On Linux-only; returns
    a dict including 'via' and the requested keys with string or None values.
    """
    logger.debug(f"Starting async service_show operation for unit: {unit}")
    unit = (unit or "").strip()
    if not unit:
        return {"via": "unavailable"}
    if not sys.platform.startswith("linux"):
        return {"via": "unavailable"}

    sel = keys or [
        "ActiveState",
        "SubState",
        "UnitFileState",
        "FragmentPath",
        "ExecMainStatus",
    ]

    # Try root-helper first
    try:
        from aios.tools.root_helper_client import RootHelperClient

        client = RootHelperClient(timeout_sec=timeout_sec)
        if await client.connect():
            try:
                show = await client.system_show(unit, sel)
                if show:
                    show = {k: (v.strip() or None) for k, v in show.items() if k in sel}
                    show["via"] = "root-helper"
                    return show
            finally:
                await client.close()
    except Exception as e:
        logger.error(f"Failed to get service show via root-helper for {unit}: {e}")

    def _local_show() -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {}
        try:
            cmd = [
                "/usr/bin/systemctl",
                "show",
                unit,
                "--no-pager",
            ]
            for k in sel:
                cmd.extend(["-p", k])
            txt = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            for line in (txt or "").splitlines():
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k in sel:
                    out[k] = v.strip() or None
        except Exception as e:
            logger.error(f"Failed to run local systemctl show for {unit}: {e}")
        out["via"] = "local"
        return out

    try:
        return await asyncio.wait_for(asyncio.to_thread(_local_show), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout getting service show for unit {unit} ({timeout_sec}s)")
        return {"via": "unavailable"}
    except Exception as e:
        logger.error(f"Failed to get service show for {unit}: {e}")
        return {"via": "unavailable"}


async def journal_read(
    unit: str,
    *,
    lines: int = 50,
    since: Optional[str] = None,
    until: Optional[str] = None,
    priority: Optional[str] = None,
    timeout_sec: float = 1.5,
) -> Dict[str, Optional[str]]:
    """Read journal entries for a unit with optional time window and priority.

    Args for since/until are passed directly to journalctl --since/--until.
    Priority corresponds to journalctl -p (e.g., info, warning, err).
    """
    logger.debug(f"Starting async journal_read operation for unit: {unit} (lines={lines})")
    unit = (unit or "").strip()
    if not unit:
        return {"via": "unavailable", "text": None}
    if not sys.platform.startswith("linux"):
        return {"via": "unavailable", "text": None}

    # Try root-helper first
    try:
        from aios.tools.root_helper_client import RootHelperClient

        client = RootHelperClient(timeout_sec=timeout_sec)
        if await client.connect():
            try:
                if since or until or priority:
                    txt = await client.journal_read_windowed(unit, since, until, priority, int(lines))
                else:
                    txt = await client.journal_read(unit, int(lines))
                if txt is not None and txt.strip():
                    return {"via": "root-helper", "text": txt}
            finally:
                await client.close()
    except Exception as e:
        logger.error(f"Failed to read journal via root-helper for {unit}: {e}")

    def _local_read() -> Dict[str, Optional[str]]:
        try:
            n = max(1, min(int(lines), 1000))
            cmd: List[str] = [
                "/usr/bin/journalctl",
                "-u",
                unit,
                "-n",
                str(n),
                "--no-pager",
            ]
            if since:
                cmd.extend(["--since", str(since)])
            if until:
                cmd.extend(["--until", str(until)])
            if priority:
                cmd.extend(["-p", str(priority)])
            txt = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            return {"via": "local", "text": txt}
        except Exception as e:
            logger.error(f"Failed to read journal locally for {unit}: {e}")
            return {"via": "local", "text": None}

    try:
        return await asyncio.wait_for(asyncio.to_thread(_local_read), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout reading journal for unit {unit} ({timeout_sec}s)")
        return {"via": "unavailable", "text": None}
    except Exception as e:
        logger.error(f"Failed to read journal for {unit}: {e}")
        return {"via": "unavailable", "text": None}
