"""Subpackage containing implementations for hrm_hf CLI commands.

Each module exposes a top-level function with the same signature as the
Typer command in `hrm_hf_cli.py`. The CLI imports and calls these
implementations directly; we intentionally do not re-export symbols at
the package level to avoid import cycles.
"""

# No package-level re-exports are required.
__all__: list[str] = []
