from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

try:  # pragma: no cover - import guard for bootstrap contexts
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover
    system_paths = None

# Suppress the deprecation warning about TRANSFORMERS_CACHE early
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*", category=FutureWarning)

# ============================================================================
# Hugging Face Cache Configuration - Set BEFORE any imports
# ============================================================================
# This prevents HF libraries from downloading to the user's default cache directory
# (typically %USERPROFILE%\.cache on Windows or ~/.cache on Linux)
# Works on fresh install without manual setup, with smart defaults


def _get_hf_cache_dir() -> Path:
    """Determine a user-writable Hugging Face cache directory."""

    def _first_writable(candidates: list[Path]) -> Path:
        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
            except Exception:
                continue
            else:
                return candidate
        fallback = Path(tempfile.gettempdir()) / "aios" / "hf_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    env_override = os.environ.get("AIOS_HF_CACHE")
    candidates: list[Path] = []

    if env_override:
        candidates.append(Path(env_override).expanduser())

    try:
        config_dir = Path.home() / ".config" / "aios"
        config_file = config_dir / "hf_cache_path.txt"
        if config_file.exists():
            saved_path = config_file.read_text().strip()
            if saved_path:
                candidates.append(Path(saved_path))
    except Exception:
        pass

    for drive_letter in ["D", "E", "F", "Z"]:
        try:
            drive_path = Path(f"{drive_letter}:/")
            if drive_path.exists():
                candidates.append(drive_path / "AI-OS-Data" / "hf_cache")
        except Exception:
            continue

    candidates.extend(
        [
            Path.home() / ".cache" / "aios" / "hf_cache",
            Path.home() / "AI-OS" / "hf_cache",
            Path.cwd() / "training_data" / "hf_cache",
        ]
    )

    return _first_writable(candidates)


def _save_hf_cache_dir(cache_dir: Path) -> bool:
    """Save HF cache directory preference for future sessions."""
    try:
        config_dir = Path.home() / ".config" / "aios"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "hf_cache_path.txt"
        config_file.write_text(str(cache_dir.resolve()))
        return True
    except Exception:
        return False


# Set up HF cache location
_hf_cache_dir = _get_hf_cache_dir()
try:
    _hf_cache_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    # If we can't create it, fall back to project directory
    _hf_cache_dir = Path.cwd() / "training_data" / "hf_cache"
    _hf_cache_dir.mkdir(parents=True, exist_ok=True)

# Migrate TRANSFORMERS_CACHE to HF_HOME if set
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
    _hf_cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])
elif "HF_HOME" not in os.environ:
    # Set environment variables for HuggingFace libraries
    os.environ["HF_HOME"] = str(_hf_cache_dir.resolve())
os.environ["HF_DATASETS_CACHE"] = str((_hf_cache_dir / "datasets").resolve())
os.environ["HF_HUB_CACHE"] = str((_hf_cache_dir / "hub").resolve())
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ----------------------------------------------------------------------------
# Stable CUDA device ordering across platforms
# Set BEFORE any torch imports to enforce consistent device indices
# ----------------------------------------------------------------------------
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Log the cache location for debugging (won't show until logging is set up later)
# print(f"[HF Cache] Using: {_hf_cache_dir}", file=sys.stderr)

# ============================================================================
# Now safe to import everything else
# ============================================================================

import time
import json
import asyncio
import logging
import logging.config
import sqlite3
import shutil
import subprocess as _sp
import shlex as _sh
from typing import Optional, List

import typer
from rich import print
import asyncio as _asyncio

from aios.utils.diagnostics import enable_asyncio_diagnostics

from aios.core.orchestrator import Orchestrator
from aios.memory.store import init_db, get_db
from aios.core.hrm import build_default_registry, Manager
from aios.core.budgets import defaults_for_risk_tier
from aios.memory.store import load_budgets, save_budgets, load_budget_usage
from aios.core.directives import list_directives
from aios.memory.store import list_artifacts
from aios.core.watch import (
    health_check as watch_health_check,
    restore_last_checkpoint as watch_restore_last_checkpoint,
    upload_checkpoint_to_repo,
    git_commit,
    latest_checkpoint_from_db,
    latest_checkpoint_from_fs,
)
from aios.core.train import average_checkpoints_npz
from aios.core.hrm_engine import HRMEngine
from aios.docs.modelcard import generate_modelcard as _mc_generate, scaffold_config as _mc_scaffold
from aios.cli.utils import load_config, setup_logging, dml_cfg_path as _dml_cfg_path
from aios.cli import training_cli, core_cli, crawl_cli, datasets_cli
from aios.cli.hrm_hf_cli import app as hrm_hf_app
from aios.cli import budgets_cli, artifacts_cli, goals_cli, agent_cli, hrm_cli, modelcard_cli, dml_cli, cleanup_cli, optimization_cli, hf_cache_cli, cache_cli, eval_cli
from aios.cli.brains import app as brains_cli_app

enable_asyncio_diagnostics()

# Constants for Typer defaults (avoid function calls in default parameters; ruff B008)
DEFAULT_DREAMS_CONFIG = str((Path.home() / "artifacts" / "dreams" / "config.yaml").as_posix())
DEFAULT_DREAMS_HTML = str((Path.home() / "artifacts" / "dreams" / "model_card.html").as_posix())

app = typer.Typer(add_completion=False, help="AI-OS HRM-only agent CLI")
hrm = typer.Typer(help="Unified HRM interface (builtin)")
modelcard = typer.Typer(help="Generate model cards (uses dreams_mc if available, else fallback)")
cleanup = typer.Typer(help="Housekeeping commands (vendor removal, cache cleanup)")

app.add_typer(modelcard, name="modelcard")
app.add_typer(hrm, name="hrm")
app.add_typer(cleanup, name="cleanup")


def _ensure_core_paths_writable() -> None:
    """Abort early if critical data directories are not writable."""

    if system_paths is None:
        return

    issues = system_paths.check_core_paths_writable()
    if not issues:
        return

    typer.secho("AI-OS cannot start because required data directories are not writable:", fg="red")
    for label, path, error in issues:
        location = str(path) if path is not None else "<unresolved>"
        typer.secho(f"  • {label}: {location}", fg="red")
        typer.secho(f"    Reason: {error}", fg="red")

    typer.secho("How to fix:", fg="yellow")
    typer.echo("  • Launch AI-OS using the Administrator shortcut on Windows or adjust permissions for the paths above.")
    typer.echo("  • Override the locations via AIOS_PROGRAM_DATA / AIOS_USER_DATA / AIOS_CACHE_DIR if you prefer different writable folders.")
    typer.echo("  • Ensure antivirus or backup tools are not locking the paths before retrying.")
    raise typer.Exit(code=2)
@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Set log level (DEBUG/INFO/WARNING/ERROR)"),
    log_json: bool = typer.Option(False, "--log-json/--no-log-json", help="Use JSON logs (default off in debug)"),
    log_stream: Optional[str] = typer.Option(None, "--log-stream", help="Console stream: stdout or stderr"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Path to log file"),
):
    """If no subcommand, start the interactive CLI menu for convenience."""
    _ensure_core_paths_writable()
    # Initialize logging early for all subcommands so DEBUG shows in VS Code Debug Console
    try:
        cfg = load_config(None)
        # If user requested --debug and no explicit level, bump to DEBUG and prefer stderr
        eff_level = (log_level or ("DEBUG" if debug else None))
        eff_stream = (log_stream or ("stderr" if debug else None))
        setup_logging(cfg, level=eff_level, json_preferred=bool(log_json), stream=eff_stream, logfile=log_file)
    except Exception:
        pass
    if ctx.invoked_subcommand is None:
        try:
            argv = sys.argv[1:]
            if any(a in ("-h", "--help") for a in argv):
                return
        except Exception:
            pass
        # Best-effort warmup of master text brain before launching UI
        try:
            _ = _sp.run([sys.executable, "-u", "-m", "aios.cli.aios", "chat", "__warmup__"], check=False, capture_output=True, text=True)
        except Exception:
            pass
        core_cli.ui()

## moved to aios.cli.utils: load_config, setup_logging


# --- Unified HRM commands ---
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
def hrm_act(
    operator: Optional[str] = typer.Option(None, "--operator", help="Candidate operator name (builtin mode)"),
):
    cfg = load_config(None)
    setup_logging(cfg)
    eng = HRMEngine(cfg)
    async def _run():
        res = await eng.act({}, [operator] if operator else None)
        print(res)

    _asyncio.run(_run())

# Register modular command groups
# CRITICAL: Use lazy imports for commands that aren't needed during spawn
# This prevents heavy modules (aiohttp, SSL certs) from loading in DDP workers
import sys
_is_spawn_worker = any('torch.distributed' in arg or 'multiprocessing' in arg for arg in sys.argv[:3])

training_cli.register(app)
core_cli.register(app)
crawl_cli.register(app)

# Lazy import datasets_cli only when needed (it imports aiohttp which hangs on Windows spawn)
if not _is_spawn_worker:
    datasets_cli.register(app)
    
budgets_cli.register(app)
artifacts_cli.register(app)
goals_cli.register(app)
agent_cli.register(app)
hrm_cli.register(app)
modelcard_cli.register(app)
dml_cli.register(app)
cleanup_cli.register(app)
optimization_cli.register(app)
hf_cache_cli.register(app)
app.add_typer(brains_cli_app, name="brains")
app.add_typer(hrm_hf_app, name="hrm-hf")
app.add_typer(cache_cli.app, name="cache")
app.add_typer(eval_cli.app, name="eval")


@app.command()
def doctor(
    permissions: bool = typer.Option(False, "--permissions", help="Check file permissions only"),
):
    """Diagnose installation and runtime issues."""
    import ctypes
    
    typer.echo("AI-OS Doctor - Diagnostic Tool")
    typer.echo("==============================")
    
    # 1. Check Elevation
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        is_admin = False
        
    if is_admin:
        typer.echo("[+] Running as Administrator")
    else:
        typer.echo("[!] Running as Standard User")
        typer.echo("    Note: Admin rights are required for full functionality (ProgramData access, GPU scheduling).")
        
    # 2. Check Paths
    if system_paths:
        paths_to_check = {
            "Logs": system_paths.get_logs_dir(),
            "Config": system_paths.get_user_config_dir(),
            "State": system_paths.get_state_file_path().parent,
            "Artifacts": system_paths.get_artifacts_root(),
            "Cache": system_paths.get_user_cache_root(),
        }
        
        typer.echo("\nChecking Directory Permissions:")
        for name, path in paths_to_check.items():
            p = Path(path)
            status = "OK"
            try:
                if not p.exists():
                    p.mkdir(parents=True, exist_ok=True)
                    status = "Created"
                
                # Test write
                test_file = p / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                typer.echo(f"[+] {name}: {p} ({status}) - Writable")
            except Exception as e:
                typer.echo(f"[!] {name}: {p} - NOT WRITABLE ({e})")
                if "Artifacts" in name and not is_admin:
                    typer.echo("    -> Try running as Administrator")

    if permissions:
        return

    # 3. Check Dependencies
    typer.echo("\nChecking Dependencies:")
    
    # lm_eval
    try:
        import lm_eval
        typer.echo(f"[+] lm_eval: Installed ({lm_eval.__version__})")
    except ImportError:
        typer.echo("[!] lm_eval: MISSING")
        typer.echo("    -> Run: pip install lm-eval[api]")

    # tkinterweb
    try:
        import tkinterweb
        typer.echo("[+] tkinterweb: Installed")
    except ImportError:
        typer.echo("[!] tkinterweb: MISSING")
        typer.echo("    -> Run: pip install tkinterweb>=3.23.8")
        
    # Flash Attention (if CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import flash_attn
                typer.echo(f"[+] flash_attn: Installed ({flash_attn.__version__})")
            except ImportError:
                typer.echo("[!] flash_attn: MISSING (Recommended for CUDA)")
        else:
            typer.echo("[i] flash_attn: Skipped (No CUDA)")
    except ImportError:
        typer.echo("[!] torch: MISSING")

    typer.echo("\nDiagnostics complete.")


if __name__ == "__main__":
    app()
