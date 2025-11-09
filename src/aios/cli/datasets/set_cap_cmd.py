from __future__ import annotations

import typer


def datasets_set_cap(cap_gb: float = typer.Argument(..., help="Dataset storage cap in GB (persisted)")):
    try:
        from aios.data.datasets import set_datasets_storage_cap_gb, datasets_storage_cap_gb
        ok = set_datasets_storage_cap_gb(cap_gb)
        print({"ok": bool(ok), "cap_gb": float(datasets_storage_cap_gb())})
    except Exception as e:
        print({"ok": False, "error": str(e)})
