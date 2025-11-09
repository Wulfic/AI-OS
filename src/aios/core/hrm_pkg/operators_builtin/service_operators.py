"""Service Operators - Service management and inspection operations."""

from __future__ import annotations

import sys
import subprocess
from typing import Dict, Any


def register_service_operators(reg):
    """Register service management operators."""
    from ..api import SimpleOperator, AsyncOperator

    def _service_triage_readonly(ctx: Dict[str, Any]) -> bool:
        """Read-only service triage: fetch systemctl status and brief journal tail."""
        if not sys.platform.startswith("linux"):
            return False
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        try:
            _ = subprocess.check_output(
                ["/usr/bin/systemctl", "status", "--no-pager", "--lines=5", unit],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            return False
        try:
            n = int(ctx.get("lines", 20))
            n = max(1, min(n, 100))
            _ = subprocess.check_output(
                ["/usr/bin/journalctl", "-u", unit, "-n", str(n), "--no-pager"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            pass
        return True

    reg.register(SimpleOperator(name="service_triage_readonly", func=_service_triage_readonly))

    async def _service_triage_root_helper(ctx: Dict[str, Any]) -> bool:
        """Read-only triage via root-helper D-Bus if available (Linux-only)."""
        if not sys.platform.startswith("linux"):
            return False
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        try:
            from aios.tools.root_helper_client import RootHelperClient
        except Exception:
            return False

        client = RootHelperClient(timeout_sec=1.5)
        ok = await client.connect()
        if not ok:
            await client.close()
            return False
        try:
            status = await client.system_status(unit) or ""
            _ = await client.journal_read(unit, int(ctx.get("lines", 20)))
            return bool(status.strip())
        finally:
            await client.close()

    reg.register(AsyncOperator(name="service_triage_root_helper", async_func=_service_triage_root_helper))

    async def _service_triage(ctx: Dict[str, Any]) -> bool:
        """Unified triage: prefer root-helper, fallback to local read-only."""
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        try:
            from aios.tools.service_adapter import system_status_and_journal
        except Exception:
            return False
        res = await system_status_and_journal(unit, lines=int(ctx.get("lines", 20)), timeout_sec=1.5)
        via = res.get("via")
        status = (res.get("status") or "").strip()
        journal = (res.get("journal") or "").strip()
        return via in ("root-helper", "local") and (bool(status) or bool(journal))

    reg.register(AsyncOperator(name="service_triage", async_func=_service_triage))

    def _unit_file_dump(ctx: Dict[str, Any]) -> bool:
        """Read-only: dump a unit's effective file with systemctl cat (Linux-only)."""
        if not sys.platform.startswith("linux"):
            return False
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        try:
            out = subprocess.check_output(
                ["/usr/bin/systemctl", "cat", unit, "--no-pager"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            return bool(out.strip())
        except Exception:
            return False

    reg.register(SimpleOperator(name="unit_file_dump", func=_unit_file_dump))

    async def _unit_dropins_discover(ctx: Dict[str, Any]) -> bool:
        """Read-only: discover unit fragment path and drop-in files (Linux-only)."""
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        try:
            from aios.tools.service_adapter import unit_dropins
        except Exception:
            return False
        res = await unit_dropins(unit, timeout_sec=1.5)
        via = res.get("via")
        fragment = res.get("fragment_path") or ""
        dropins = res.get("dropins") or []
        if isinstance(dropins, str):
            dropins = [dropins] if dropins else []
        return via in ("root-helper", "local") and (bool(fragment) or len(dropins) > 0)

    reg.register(AsyncOperator(name="unit_dropins_discover", async_func=_unit_dropins_discover))

    async def _service_inspect(ctx: Dict[str, Any]) -> bool:
        """Read-only: inspect a unit via systemctl show for a few key fields."""
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        try:
            from aios.tools.service_adapter import service_show
        except Exception:
            return False
        res = await service_show(unit, timeout_sec=1.5)
        via = res.get("via")
        state = (res.get("ActiveState") or "").strip()
        sub = (res.get("SubState") or "").strip()
        return via in ("root-helper", "local") and (bool(state) or bool(sub))

    reg.register(AsyncOperator(name="service_inspect", async_func=_service_inspect))

    async def _service_relationships(ctx: Dict[str, Any]) -> bool:
        """Read-only: inspect unit relationships (Wants/Requires/PartOf) via adapter."""
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        try:
            from aios.tools.service_adapter import service_show
        except Exception:
            return False
        keys = ["Wants", "Requires", "PartOf"]
        res = await service_show(unit, keys=keys, timeout_sec=1.5)
        via = res.get("via")
        any_present = any((res.get(k) or "").strip() for k in keys)
        return via in ("root-helper", "local") and any_present

    reg.register(AsyncOperator(name="service_relationships", async_func=_service_relationships))
