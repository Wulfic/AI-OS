from __future__ import annotations

import typer


cleanup = typer.Typer(help="Housekeeping commands for cache and artifact cleanup")

# Future cleanup commands can be added here


def register(app: typer.Typer) -> None:
    app.add_typer(cleanup, name="cleanup")
