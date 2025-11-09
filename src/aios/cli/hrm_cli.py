from __future__ import annotations

import shlex as _sh
from typing import Optional

import typer

from aios.cli.utils import load_config, setup_logging
from aios.core.hrm_engine import HRMEngine


hrm = typer.Typer(help="Unified HRM interface (builtin)")


@hrm.command("info")
def hrm_info(config: Optional[str] = typer.Option(None, "--config", help="Path to config.yaml")):
    cfg = load_config(config)
    setup_logging(cfg)
    eng = HRMEngine(cfg)
    print(eng.info())


@hrm.command("setup")
def hrm_setup(force: bool = typer.Option(False, "--force", help="No-op; vendor removed")):
    eng = HRMEngine(load_config(None))
    print(eng.setup(force=force))


@hrm.command("pretrain")
def hrm_pretrain(args: Optional[str] = typer.Option(None, "--args", help="Args (unused)")):
    eng = HRMEngine(load_config(None))
    argv = _sh.split(args) if args else None
    print(eng.pretrain(argv))


@hrm.command("evaluate")
def hrm_evaluate(args: Optional[str] = typer.Option(None, "--args", help="Args (unused)")):
    eng = HRMEngine(load_config(None))
    argv = _sh.split(args) if args else None
    print(eng.evaluate(argv))


@hrm.command("act")
def hrm_act(operator: Optional[str] = typer.Option(None, "--operator", help="Candidate operator name (builtin mode)")):
    cfg = load_config(None)
    setup_logging(cfg)
    eng = HRMEngine(cfg)
    async def _run():
        res = await eng.act({}, [operator] if operator else None)
        print(res)
    import asyncio as _asyncio
    _asyncio.run(_run())


def register(app: typer.Typer) -> None:
    app.add_typer(hrm, name="hrm")
