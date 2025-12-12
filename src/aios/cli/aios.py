from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

# ============================================================================
# Early Windows VC++ Runtime Check - MUST be before any torch imports
# PyTorch requires Visual C++ Redistributable to load its DLLs
# ============================================================================
def _check_vcpp_runtime() -> None:
    """Check for Visual C++ Runtime on Windows before torch tries to load."""
    if os.name != "nt":
        return  # Only needed on Windows
    
    vcruntime = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vcruntime140.dll"
    vcruntime_1 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vcruntime140_1.dll"
    
    if not vcruntime.exists() or not vcruntime_1.exists():
        # Try to show a message box on Windows
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                "Microsoft Visual C++ Redistributable is required but not installed.\n\n"
                "Please download and install it from:\n"
                "https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
                "After installing, restart AI-OS.",
                "AI-OS - Missing Dependency",
                0x10  # MB_ICONERROR
            )
        except Exception:
            pass
        
        print("\n" + "=" * 70, file=sys.stderr)
        print("ERROR: Microsoft Visual C++ Redistributable is not installed!", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("\nPyTorch requires the Visual C++ Redistributable to run on Windows.", file=sys.stderr)
        print("\nPlease download and install it from:", file=sys.stderr)
        print("  https://aka.ms/vs/17/release/vc_redist.x64.exe", file=sys.stderr)
        print("\nAfter installing, restart AI-OS.", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

# Run the check immediately
_check_vcpp_runtime()


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
            Path.cwd() / "training_datasets" / "hf_cache",
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
    _hf_cache_dir = Path.cwd() / "training_datasets" / "hf_cache"
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
    permissions: bool = typer.Option(False, "--permissions", "-p", help="Check file permissions only"),
    dependencies: bool = typer.Option(True, "--dependencies/--no-dependencies", "-d", help="Check Python package dependencies"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", "-g", help="Check GPU availability and configuration"),
    disk: bool = typer.Option(True, "--disk/--no-disk", help="Check disk space"),
    network: bool = typer.Option(True, "--network/--no-network", "-n", help="Check network connectivity"),
    env_vars: bool = typer.Option(True, "--env/--no-env", "-e", help="Display environment variables"),
    config: bool = typer.Option(True, "--config/--no-config", "-c", help="Validate configuration files"),
    memory: bool = typer.Option(True, "--memory/--no-memory", "-m", help="Check system memory"),
    repair: bool = typer.Option(False, "--repair", "-r", help="Attempt to automatically fix issues"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON"),
):
    """Comprehensive diagnostic tool for AI-OS installation and runtime.
    
    Checks platform compatibility, dependencies, GPU configuration, disk space,
    network connectivity, environment variables, configuration files, and memory.
    
    Examples:
        aios doctor                    # Run all checks
        aios doctor --json             # Output as JSON for scripting
        aios doctor --repair           # Auto-fix issues where possible
        aios doctor --permissions      # Check only file permissions
        aios doctor --no-network       # Skip network connectivity checks
    """
    import asyncio
    
    from .doctor import run_diagnostics, DiagnosticSeverity, save_report_to_log
    from .doctor.runner import format_report_text
    
    # If --permissions flag is used alone, limit to permission checks only
    if permissions and not any([
        # These flags indicate user wants more than just permissions
    ]):
        dependencies = False
        gpu = False
        disk = False
        network = False
        env_vars = False
        config = False
        memory = False
    
    # Run diagnostics
    try:
        report = asyncio.run(run_diagnostics(
            check_permissions=True,  # Always check permissions
            check_dependencies=dependencies,
            check_gpu=gpu,
            check_disk=disk,
            check_network=network,
            check_env_vars=env_vars,
            check_config=config,
            check_memory=memory,
            auto_repair=repair,
            json_output=json_output,
        ))
    except Exception as e:
        if json_output:
            import json
            typer.echo(json.dumps({"error": str(e), "status": "failed"}))
        else:
            typer.echo(f"[!!] Diagnostics failed: {e}", err=True)
        raise typer.Exit(code=1)
    
    # Save report to log file
    try:
        log_path = save_report_to_log(report)
        if not json_output:
            typer.echo(f"\n📋 Report saved to: {log_path}")
    except Exception as e:
        if not json_output:
            typer.echo(f"\n[!] Could not save report to log: {e}", err=True)
    
    # Output results
    if json_output:
        typer.echo(report.to_json())
    else:
        typer.echo(format_report_text(report))
    
    # Exit with appropriate code
    if report.summary.get("critical", 0) > 0:
        raise typer.Exit(code=2)
    elif report.summary.get("error", 0) > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
