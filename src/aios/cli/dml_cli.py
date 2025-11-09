from __future__ import annotations

from pathlib import Path
from typing import Optional

import subprocess as _sp
import typer


def _dml_cfg_path() -> Path:
    return Path.home() / ".config/aios/dml_python.txt"


def dml_setpython(path: str = typer.Argument(..., help="Path to Python 3.11+ interpreter with torch-directml installed")):
    p = Path(path)
    if not p.exists():
        print({"updated": False, "error": "path not found", "path": str(p)})
        raise typer.Exit(code=1) from None
    cfgp = _dml_cfg_path()
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfgp.write_text(str(p), encoding="utf-8")
        print({"updated": True, "path": str(p), "config": str(cfgp)})
    except Exception as e:
        print({"updated": False, "error": str(e)})
        raise typer.Exit(code=1) from None


def dml_showpython():
    cfgp = _dml_cfg_path()
    if not cfgp.exists():
        print({"configured": False, "config": str(cfgp)})
        return
    try:
        val = cfgp.read_text(encoding="utf-8").strip()
    except Exception as e:
        print({"configured": False, "error": str(e), "config": str(cfgp)})
        return
    print({"configured": bool(val), "path": val, "config": str(cfgp)})


def dml_test(python: Optional[str] = typer.Option(None, "--python", help="Override Python to test (defaults to configured)")):
    py = python
    if not py:
        cfgp = _dml_cfg_path()
        if cfgp.exists():
            try:
                py = cfgp.read_text(encoding="utf-8").strip()
            except Exception:
                py = None
    if not py:
        print({"ok": False, "error": "no python provided or configured"})
        raise typer.Exit(code=1) from None
    try:
        rc = _sp.run([py, "-c", "import torch_directml as dml; dml.device()"], check=False)
    except Exception as e:
        print({"ok": False, "python": py, "error": str(e)})
        raise typer.Exit(code=1) from None
    ok = (rc.returncode == 0)
    print({"ok": ok, "python": py, "returncode": rc.returncode})
    raise typer.Exit(code=0 if ok else 1)


def register(app: typer.Typer) -> None:
    app.command("dml-setpython")(dml_setpython)
    app.command("dml-showpython")(dml_showpython)
    app.command("dml-test")(dml_test)
