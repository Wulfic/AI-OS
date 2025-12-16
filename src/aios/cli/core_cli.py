from __future__ import annotations

import asyncio
import json
import subprocess as _sp
import sys
from typing import Optional

import typer

from aios.cli.utils import load_config, setup_logging
from aios.core.communicator import format_status_summary
from aios.core.directives import list_directives
from aios.core.orchestrator import Orchestrator
from aios.memory.store import get_db, init_db, list_artifacts



def status(
    config: Optional[str] = typer.Option(None, help="Path to config.yaml"),
    recent: int = typer.Option(3, "--recent", help="How many recent journal summaries to include"),
    unit: Optional[str] = typer.Option(None, "--unit", help="Filter summaries by unit in data"),
    label: Optional[str] = typer.Option(None, "--label", help="Filter summaries by label"),
):
    cfg = load_config(config)
    setup_logging(cfg)
    orch = Orchestrator(config=cfg)
    info = orch.status()
    conn = get_db()
    init_db(conn)
    try:
        directives = list_directives(conn, active_only=True)
        summary = format_status_summary(info, directives)
        latest_js = None
        recent_js: list[dict] | None = None
        try:
            items = list_artifacts(conn, kind="journal_summary", limit=200)
            def _match(it: dict) -> bool:
                if label is not None and str(it.get("label") or "") != label:
                    return False
                if unit is not None:
                    data = it.get("data") or {}
                    if str(data.get("unit") or "") != unit:
                        return False
                return True
            filtered = [it for it in items if _match(it)]
            recent_js = filtered[: max(0, int(recent))]
            latest_js = recent_js[0] if recent_js else None
        except Exception:
            latest_js = None
            recent_js = None
        latest_jt = None
        recent_jt: list[dict] | None = None
        try:
            titems = list_artifacts(conn, kind="journal_trend", limit=200)
            def _tmatch(it: dict) -> bool:
                if label is not None and str(it.get("label") or "") != label:
                    return False
                if unit is not None:
                    data = it.get("data") or {}
                    if str(data.get("unit") or "") != unit:
                        return False
                return True
            tfiltered = [it for it in titems if _tmatch(it)]
            recent_jt = tfiltered[: max(0, int(recent))]
            latest_jt = recent_jt[0] if recent_jt else None
        except Exception:
            latest_jt = None
            recent_jt = None
        latest_ck = None
        try:
            ck_items = list_artifacts(conn, kind="training_checkpoint", limit=1)
            latest_ck = ck_items[0] if ck_items else None
        except Exception:
            latest_ck = None
        latest_tm = None
        recent_tm: list[dict] | None = None
        try:
            tm_items = list_artifacts(conn, kind="training_metrics", limit=50)
            def _mmatch(it: dict) -> bool:
                if label is not None and str(it.get("label") or "") != label:
                    return False
                return True
            mfiltered = [it for it in tm_items if _mmatch(it)]
            recent_tm = mfiltered[: max(0, int(recent))]
            latest_tm = recent_tm[0] if recent_tm else None
        except Exception:
            latest_tm = None
            recent_tm = None
        print({
            "status": info,
            "summary": {"headline": summary.headline, "details": summary.details},
            "directives": [d.text for d in directives],
            "latest_journal_summary": latest_js,
            "recent_journal_summaries": recent_js,
            "latest_journal_trend": latest_jt,
            "recent_journal_trends": recent_jt,
            "latest_training_checkpoint": latest_ck,
            "latest_training_metrics": latest_tm,
            "recent_training_metrics": recent_tm,
        })
    finally:
        conn.close()


def run(config: Optional[str] = typer.Option(None, help="Path to config.yaml")):
    cfg = load_config(config)
    setup_logging(cfg)
    orch = Orchestrator(config=cfg)
    orch.run()



def gui(
    exit_after: float = typer.Option(0.0, "--exit-after", help="If >0, auto-close GUI after N seconds (for CI)"),
    minimized: bool = typer.Option(False, "--minimized", help="Start with window minimized to system tray")
):
    """Launch the AI-OS Tkinter GUI."""
    import time
    import sys
    import os
    from pathlib import Path
    
    gui_start = time.time()
    
    # Helper function to safely print (stdout might be broken when running via exe wrapper)
    def safe_print(msg: str) -> None:
        try:
            # Only print if stdout is available and writable
            if sys.stdout is not None and hasattr(sys.stdout, 'write'):
                print(msg)
                sys.stdout.flush()
        except Exception:
            pass  # Silently ignore if stdout is unavailable
    
    # Ensure stdout/stderr are valid to prevent crashes in Tkinter
    # When launched via pythonw.exe or certain wrappers, these can be None or broken
    if sys.stdout is None or not hasattr(sys.stdout, 'write'):
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None or not hasattr(sys.stderr, 'write'):
        sys.stderr = open(os.devnull, 'w')
    
    safe_print(f"[GUI TIMING] Imports done: {time.time() - gui_start:.3f}s")
    
    # Initialize logging early so all GUI components can use it
    cfg = load_config(None)
    setup_logging(cfg)
    safe_print(f"[GUI TIMING] Logging configured: {time.time() - gui_start:.3f}s")
    
    # Only set if not already set by aios.py
    if not os.environ.get("HF_HOME"):
        # Use same logic as aios.py for consistency
        config_file = Path.home() / ".config" / "aios" / "hf_cache_path.txt"
        if config_file.exists():
            try:
                _hf_cache_dir = Path(config_file.read_text().strip())
            except Exception:
                _hf_cache_dir = None
        else:
            _hf_cache_dir = None
        
        if not _hf_cache_dir or not (_hf_cache_dir.exists() or _hf_cache_dir.parent.exists()):
            # Default to install root location
            _hf_cache_dir = Path.cwd() / "training_datasets" / "hf_cache"
        
        try:
            _hf_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            _hf_cache_dir = Path.cwd() / "training_datasets" / "hf_cache"
            _hf_cache_dir.mkdir(parents=True, exist_ok=True)
        
        os.environ["HF_HOME"] = str(_hf_cache_dir.resolve())
        os.environ["HF_DATASETS_CACHE"] = str((_hf_cache_dir / "datasets").resolve())
        os.environ["HF_HUB_CACHE"] = str((_hf_cache_dir / "hub").resolve())
    
    safe_print(f"[GUI TIMING] HF cache configured: {time.time() - gui_start:.3f}s")
    
    try:
        safe_print(f"[GUI TIMING] About to import aios.gui...")
        from aios.gui import run as _run_gui
        safe_print(f"[GUI TIMING] aios.gui imported: {time.time() - gui_start:.3f}s")
    except ImportError as e:
        safe_print(str({"launched": False, "error": str(e)}))
        return
    _run_gui(
        exit_after=exit_after if exit_after and exit_after > 0 else None,
        minimized=minimized
    )


def ui():
    """Interactive CLI menu for common operations."""
    def _menu() -> None:
        while True:
            print("\n" + "="*60)
            print(" AI-OS Interactive CLI Menu")
            print("="*60)
            print("1) Status              - View system status")
            print("2) List artifacts      - Show recent artifacts (5)")
            print("3) Budgets summary     - View budget limits & usage")
            print("4) Train               - Train a brain (custom params)")
            print("5) Crawl URL           - Web crawl and dataset creation")
            print("6) Run custom CLI      - Execute any aios command")
            print("7) Train Parallel      - Multi-device training (custom)")
            print("8) Chat                - Interactive chat with brain")
            print("q) Quit                - Exit interactive menu")
            print("="*60)
            choice = input("\nSelect option: ").strip().lower()
            
            if choice == "1":
                print("\n--- System Status ---")
                cfg = load_config(None)
                setup_logging(cfg)
                orch = Orchestrator(config=cfg)
                print(json.dumps(orch.status(), indent=2, default=str))
            elif choice == "2":
                print("\n--- Recent Artifacts ---")
                conn = get_db()
                init_db(conn)
                try:
                    items = list_artifacts(conn, kind=None, limit=5)
                    print(json.dumps(items, indent=2, default=str))
                finally:
                    conn.close()
            elif choice == "3":
                print("\n--- Budget Summary ---")
                from aios.core.budgets import defaults_for_risk_tier
                from aios.memory.store import load_budgets, load_budget_usage
                cfg = load_config(None)
                tier = cfg.get("risk_tier", "conservative")
                conn = get_db()
                init_db(conn)
                try:
                    limits = {**defaults_for_risk_tier(tier), **load_budgets(conn)}
                    usage = load_budget_usage(conn)
                    print(json.dumps({"tier": tier, "limits": limits, "usage": usage}, indent=2, default=str))
                finally:
                    conn.close()
            elif choice == "4":
                print("\n--- Train Brain ---")
                steps = input("Steps (default 200): ").strip() or "200"
                batch = input("Batch size (default 64): ").strip() or "64"
                domains = input("Domains (default english): ").strip() or "english"
                ds = input("Dataset file (optional): ").strip()
                if not ds:
                    try:
                        from aios.data.datasets import datasets_base_dir
                        default_path = (datasets_base_dir() / "web_crawl" / "data.jsonl")
                        if default_path.exists():
                            ds = str(default_path)
                    except Exception:
                        pass
                hybrid = input("Hybrid? (y/N): ").strip().lower() in {"y", "yes"}
                flags = input("Extra train flags (optional): ").strip()
                from shlex import split as _shsplit
                args = [sys.executable, "-m", "aios.cli.aios", "train", "--steps", steps, "--batch-size", batch, "--domains", domains]
                if ds:
                    args += ["--dataset-file", ds]
                if hybrid:
                    args.append("--hybrid")
                if flags:
                    try:
                        args += _shsplit(flags)
                    except Exception:
                        pass
                print(f"\nStarting training with {steps} steps, batch size {batch}...")
                _sp.run(args, check=False)
            elif choice == "5":
                print("\n--- Web Crawler ---")
                from shlex import split as _shsplit
                url = input("URL (default https://example.com): ").strip() or "https://example.com"
                flags = input("Optional flags (e.g., --recursive --max-pages 25 --max-depth 2): ").strip()
                args = [sys.executable, "-m", "aios.cli.aios", "crawl", url]
                if flags:
                    try:
                        args += _shsplit(flags)
                    except Exception:
                        pass
                print(f"\nStarting crawl of {url}...")
                _sp.run(args, check=False)
            elif choice == "6":
                print("\n--- Run Custom Command ---")
                from shlex import split as _shsplit
                args = input("Enter args (e.g., status --recent 1): ")
                cmd = [sys.executable, "-m", "aios.cli.aios"] + _shsplit(args)
                _sp.run(cmd, check=False)
            elif choice == "7":
                print("\n--- Parallel Training ---")
                steps = input("Steps (default 200): ").strip() or "200"
                batch = input("Batch size (default 64): ").strip() or "64"
                tag = input("Tag (default parallel): ").strip() or "parallel"
                memfrac = input("GPU mem frac (0.1-0.99, default 0.9): ").strip() or "0.9"
                domains = input("Domains (default english): ").strip() or "english"
                ds = input("Dataset file (optional): ").strip()
                if not ds:
                    try:
                        from aios.data.datasets import datasets_base_dir
                        default_path = (datasets_base_dir() / "web_crawl" / "data.jsonl")
                        if default_path.exists():
                            ds = str(default_path)
                    except Exception:
                        pass
                hybrid = input("Hybrid? (y/N): ").strip().lower() in {"y", "yes"}
                flags = input("Extra train flags (optional): ").strip()
                cpu = input("Use CPU? (y/N): ").strip().lower() in {"y", "yes"}
                cuda = input("Use CUDA? (Y/n): ").strip().lower() not in {"n", "no"}
                xpu = input("Use XPU? (y/N): ").strip().lower() in {"y", "yes"}
                dml = input("Use DML? (y/N): ").strip().lower() in {"y", "yes"}
                mps = input("Use MPS? (y/N): ").strip().lower() in {"y", "yes"}
                cuda_ids = input("CUDA IDs (e.g., 0,1) optional: ").strip()
                cuda_mem = input("CUDA mem map JSON (e.g., {0:0.5}) optional: ").strip()
                args = [
                    sys.executable, "-m", "aios.cli.aios", "train-parallel",
                    "--steps", steps, "--batch-size", batch, "--tag", tag, "--gpu-mem-frac", memfrac,
                    "--domains", domains,
                ]
                args.append("--cpu" if cpu else "--no-cpu")
                args.append("--cuda" if cuda else "--no-cuda")
                args.append("--xpu" if xpu else "--no-xpu")
                args.append("--dml" if dml else "--no-dml")
                args.append("--mps" if mps else "--no-mps")
                if ds:
                    args += ["--dataset-file", ds]
                if hybrid:
                    args.append("--hybrid")
                if flags:
                    args += ["--train-flags", flags]
                if cuda_ids:
                    args += ["--cuda-ids", cuda_ids]
                if cuda_mem:
                    args += ["--cuda-mem-map", cuda_mem]
                print(f"\nStarting parallel training with {steps} steps...")
                _sp.run(args, check=False)
            elif choice == "8":
                print("\n--- Interactive Chat ---")
                brain_name = input("Brain name (default master): ").strip() or "master"
                print(f"Starting chat with {brain_name}. Type 'quit' or 'exit' to return to menu.\n")
                while True:
                    msg = input("You: ").strip()
                    if msg.lower() in {"quit", "exit", "q"}:
                        break
                    if not msg:
                        continue
                    args = [sys.executable, "-m", "aios.cli.aios", "chat", brain_name, msg]
                    _sp.run(args, check=False)
            elif choice == "q":
                print("\nGoodbye!")
                break
            else:
                print("\n[ERROR] Invalid option. Please select 1-8 or q.")

    _menu()


def guards_show(config: Optional[str] = typer.Option(None, help="Path to config.yaml")):
    from aios.core.guards import rules_from_config
    cfg = load_config(config)
    setup_logging(cfg)
    rules = rules_from_config(cfg)
    print({"allow_paths": rules.allow_paths, "deny_paths": rules.deny_paths})


def service_restart(
    unit: str = typer.Argument(..., help="Systemd unit name (simulated here)"),
    cost: float = typer.Option(1.0, help="Budget cost to charge"),
    dry_run: bool = typer.Option(False, help="Preview budget decision without recording usage"),
):
    from aios.core.budgets import SafetyBudget, defaults_for_risk_tier
    from aios.memory.store import load_budgets, load_budget_usage
    from aios.tools import service as svc_tools
    cfg = load_config(None)
    setup_logging(cfg)
    conn = get_db()
    init_db(conn)
    try:
        if dry_run:
            tier = cfg.get("risk_tier", "conservative")
            limits = {**defaults_for_risk_tier(tier), **load_budgets(conn)}
            used = load_budget_usage(conn)
            sb = SafetyBudget(limits=limits, usage=used)
            allowed = sb.allow("service_changes", cost)
            remaining = sb.remaining("service_changes")
            print({"dry_run": True, "allowed": allowed, "would_charge": cost, "remaining": remaining})
            return
        ok = svc_tools.restart_service(unit, cfg=cfg, conn=conn, cost=cost)
        print({"unit": unit, "restarted": ok, "charged": cost})
    finally:
        conn.close()


def pkg_install(
    name: str = typer.Argument(..., help="Package name (simulated)"),
    cost: float = typer.Option(1.0, help="Budget cost to charge"),
    dry_run: bool = typer.Option(False, help="Preview budget decision without recording usage"),
):
    from aios.core.budgets import SafetyBudget, defaults_for_risk_tier
    from aios.memory.store import load_budgets, load_budget_usage
    from aios.tools import pkg as pkg_tools
    cfg = load_config(None)
    setup_logging(cfg)
    conn = get_db()
    init_db(conn)
    try:
        if dry_run:
            tier = cfg.get("risk_tier", "conservative")
            limits = {**defaults_for_risk_tier(tier), **load_budgets(conn)}
            used = load_budget_usage(conn)
            sb = SafetyBudget(limits=limits, usage=used)
            allowed = sb.allow("pkg_ops", cost)
            remaining = sb.remaining("pkg_ops")
            print({"dry_run": True, "allowed": allowed, "would_charge": cost, "remaining": remaining})
            return
        ok = pkg_tools.install(name, cfg=cfg, conn=conn, cost=cost)
        print({"package": name, "installed": ok, "charged": cost})
    finally:
        conn.close()


def watch_agent(
    label: Optional[str] = typer.Option(None, "--label", help="Prefer checkpoints with this label"),
    auto_restore: bool = typer.Option(True, "--auto-restore/--no-auto-restore", help="Restore last checkpoint when health check fails"),
):
    from aios.core.watch import health_check as watch_health_check, restore_last_checkpoint as watch_restore_last_checkpoint
    ok = watch_health_check()
    if ok:
        print({"healthy": True})
        return
    if not auto_restore:
        print({"healthy": False, "restored": False})
        return
    res = watch_restore_last_checkpoint(prefer_label=label)
    print({"healthy": False, "restored": bool(res.get("found")), **res})


def register(app: typer.Typer) -> None:
    app.command()(status)
    app.command()(run)
    app.command()(gui)
    app.command()(ui)
    app.command("guards-show")(guards_show)
    app.command("service-restart")(service_restart)
    app.command("pkg-install")(pkg_install)
    app.command("watch-agent")(watch_agent)
