from __future__ import annotations

import os
import time as _time
import threading as _th
import signal as _signal
from typing import Any

from aios.python_exec import get_preferred_python_executable


def on_start(panel: Any) -> None:
    # Initiate a training run honoring Resources panel selections for device and DDP
    try:
        panel._stopped_dialog_shown = False
    except Exception:
        pass
    panel._stop_requested = False
    panel._last_heartbeat = None

    # Auto-detect GPUs if user selected CUDA but no rows have been populated yet (quality of life)
    try:
        rp = getattr(panel, "_resources_panel", None)
        if rp is not None and callable(getattr(rp, "_detect_and_update", None)):
            # If user has chosen CUDA in either train/run device but we have zero detected rows, attempt detection
            need_detect = False
            try:
                if rp.train_device_var.get() == "cuda" and len(getattr(rp, "_cuda_train_rows", [])) == 0:
                    need_detect = True
                if rp.run_device_var.get() == "cuda" and len(getattr(rp, "_cuda_run_rows", [])) == 0:
                    need_detect = True
            except Exception:
                pass
            if need_detect:
                panel._log("[hrm] Auto-detecting CUDA devices (no GPU rows present)…")
                try:
                    rp._detect_and_update()  # type: ignore[attr-defined]
                except Exception as _e:  # pragma: no cover - best effort
                    panel._log(f"[hrm] Auto-detect failed: {_e}")
    except Exception:
        pass

    # Build TrainingConfig from GUI state
    try:
        config = panel.build_training_config()
        
        # Validate configuration
        try:
            config.validate()
        except ValueError as e:
            panel._log(f"[hrm] Configuration error: {e}")
            return
        
        # ============================================================================
        # CHECK FOR RESUMABLE CHECKPOINT
        # ============================================================================
        if config.brain_name and config.dataset_file:
            from pathlib import Path as _CheckpointPath
            import json as _json
            
            # Check if checkpoint exists
            # Note: config.save_dir is set to bundle_dir in TrainingConfig constructor,
            # and train_actv1.py sets save_dir = bundle_dir/brain_name when brain_name is provided
            # So we need to construct the correct path here
            brain_path = _CheckpointPath(config.bundle_dir) / config.brain_name
            brain_json_path = brain_path / "brain.json"
            checkpoint_path = brain_path / "actv1_student.safetensors"
            
            should_prompt_resume = False
            if brain_json_path.exists() and checkpoint_path.exists():
                try:
                    with brain_json_path.open("r", encoding="utf-8") as f:
                        brain_data = _json.load(f)
                    
                    last_session = brain_data.get("last_session")
                    if last_session and isinstance(last_session, dict):
                        saved_dataset = last_session.get("dataset_file")
                        # Only prompt if dataset matches (otherwise fresh start makes sense)
                        if saved_dataset and str(saved_dataset) == str(config.dataset_file):
                            should_prompt_resume = True
                except Exception:
                    pass
            
            if should_prompt_resume:
                # Show resume dialog
                try:
                    from .resume_dialog import show_resume_dialog
                    
                    resume_choice = show_resume_dialog(
                        panel,
                        config.brain_name,
                        config.save_dir,
                        config.dataset_file
                    )
                    
                    if resume_choice is None:
                        # User cancelled
                        panel._log("[hrm] Training cancelled by user")
                        return
                    elif resume_choice:
                        # User chose to resume
                        config.resume = True
                        config._user_explicitly_chose_resume = True
                        panel._log("[hrm] User chose to resume from checkpoint")
                    else:
                        # User chose fresh start
                        config.resume = False
                        config._user_explicitly_chose_fresh = True
                        panel._log("[hrm] User chose to start fresh training")
                        # CRITICAL: Clear checkpoint path for fresh start
                        config.student_init = None
                except Exception as e:
                    panel._log(f"[hrm] Resume dialog error: {e}")
                    # Continue with default (no resume)
                    config.resume = False
            else:
                # No existing checkpoint - fresh start
                config.resume = False
        
        # Auto-enable resume for iterate mode if checkpoint exists
        # Iterate mode is designed for continuous training, so auto-resume makes sense
        # IMPORTANT: Only auto-enable if user didn't explicitly choose "start fresh"
        user_chose_fresh = getattr(config, "_user_explicitly_chose_fresh", False)
        if config.iterate and config.brain_name and not config.resume and not user_chose_fresh:
            from pathlib import Path as _IteratePath
            brain_path = _IteratePath(config.bundle_dir) / config.brain_name
            checkpoint_path = brain_path / "actv1_student.safetensors"
            if checkpoint_path.exists():
                config.resume = True
                panel._log("[hrm] Auto-enabling resume for iterate mode (checkpoint found)")
        # ============================================================================
        # END CHECKPOINT CHECK
        # ============================================================================
        
        # Check strict CUDA guard: if CUDA chosen but no GPUs enabled, abort
        if config.device == "cuda":
            try:
                rp = getattr(panel, "_resources_panel", None)
                if rp is not None:
                    rvals = rp.get_values()
                    sel_train = rvals.get("train_cuda_selected") or []
                    if not (isinstance(sel_train, list) and len(sel_train) > 0):
                        panel._log("[hrm] Training device set to CUDA but no GPUs enabled in Resources → enable at least one GPU or switch to CPU.")
                        return
            except Exception:
                pass
        
        # Convert config to CLI args (already includes "hrm-hf train-actv1")
        args = config.to_cli_args()
        
        # Auto-add default goal to brain if specified
        if config.brain_name and config.default_goal:
            try:
                # Check if we have goal callbacks (from app.py integration)
                on_goal_add = getattr(panel, '_on_goal_add_for_brain', None)
                on_goals_list = getattr(panel, '_on_goals_list_for_brain', None)
                
                if callable(on_goal_add) and callable(on_goals_list):
                    # Check if goal already exists for this brain
                    existing_goals = on_goals_list(config.brain_name)
                    if existing_goals is None or not isinstance(existing_goals, list):
                        existing_goals = []
                    goal_exists = any(config.default_goal in str(g) for g in existing_goals)
                    
                    if not goal_exists:
                        panel._log(f"[hrm] Auto-adding default goal to brain '{config.brain_name}': {config.default_goal[:60]}...")
                        on_goal_add(config.brain_name, config.default_goal)
                    else:
                        panel._log(f"[hrm] Default goal already exists for brain '{config.brain_name}'")
            except Exception as e:
                panel._log(f"[hrm] Note: Could not auto-add goal: {e}")
        
    except Exception as e:
        panel._log(f"[hrm] Failed to build training configuration: {e}")
        import traceback
        panel._log(f"[hrm] Traceback: {traceback.format_exc()}")
        return

    # Log DDP settings if enabled
    if config.ddp and config.world_size and config.world_size > 1:
        backend = "gloo" if os.name == "nt" else "nccl"
        panel._log(f"[hrm] Auto-DDP enabled: world_size={config.world_size} cuda_ids={config.cuda_ids} backend={backend}")
    elif config.cuda_ids:
        panel._log(f"[hrm] Single GPU training: cuda_id={config.cuda_ids}")
    
    # Clear stale STOP file if exists
    sf = config.stop_file or _default_stop_file(panel)
    if sf:
        try:
            if os.path.exists(sf):
                os.remove(sf)
                panel._log(f"[hrm] Cleared stale STOP file: {sf}")
        except Exception:
            pass

    panel._log("Running: aios " + " ".join(args))
    # Extra explicit DDP summary before launch
    try:
        if "--ddp" in args:
            ws = None
            try:
                i = args.index("--world-size")
                if i + 1 < len(args):
                    ws = args[i+1]
            except Exception:
                pass
            cids = None
            try:
                i = args.index("--cuda-ids")
                if i + 1 < len(args):
                    cids = args[i+1]
            except Exception:
                pass
            panel._log(f"[hrm] Launch summary: DDP active world_size={ws} cuda_ids={cids}")
    except Exception:
        pass
    # Persist current settings immediately before launch
    try:
        if callable(getattr(panel, "_save_state_fn", None)):
            panel._save_state_fn()  # type: ignore[misc]
    except Exception:
        pass
    if not panel._metrics_polling_active:
        panel._metrics_polling_active = True
        try:
            panel.after(1000, panel._poll_metrics)
        except Exception:
            pass

    # Launch in background thread to avoid freezing the GUI
    def _launch_bg():
        _launch_subprocess(panel, args)
    
    panel._bg_thread = _th.Thread(target=_launch_bg, daemon=True)
    panel._bg_thread.start()


def _default_stop_file(panel: Any) -> str:
    try:
        return os.path.join(panel._project_root, "training_data", "actv1", "STOP")
    except Exception:
        return "training_data/actv1/STOP"


def _launch_subprocess(panel: Any, args: list[str]) -> None:
    import subprocess as _sp, time as _time, socket as _sock
    python_executable = get_preferred_python_executable()
    # Infer DDP and CUDA IDs from args to avoid reliance on removed UI controls
    def _get_arg_val(flag: str, default: str | None = None) -> str | None:
        try:
            i = args.index(flag)
            return args[i + 1] if i + 1 < len(args) else default
        except ValueError:
            return default
    ddp_active = "--ddp" in args
    world_size = 0
    try:
        ws = _get_arg_val("--world-size")
        if ws is not None:
            world_size = int(ws)
    except Exception:
        world_size = 0
    cuda_env = None
    ids = _get_arg_val("--cuda-ids")
    if ids:
        cuda_env = ids
        if world_size <= 0:
            try:
                world_size = max(1, len([x for x in ids.split(",") if x.strip() != ""]))
            except Exception:
                pass
    # Per-rank logging when DDP
    log_dir = None
    if ddp_active:
        try:
            import datetime as _dt
            base_logs = os.path.join(panel._project_root, "artifacts", "brains", "actv1", "_ddp", "logs")
            os.makedirs(base_logs, exist_ok=True)
            stamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(base_logs, stamp)
            os.makedirs(log_dir, exist_ok=True)
            os.environ["AIOS_DDP_LOG_DIR"] = log_dir
            panel._log(f"[hrm] per-rank logs: {log_dir}")
        except Exception:
            pass

    def _free_port() -> int:
        for _ in range(20):
            try:
                with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    return s.getsockname()[1]
            except Exception:
                continue
        return 29500

    # Build command: use torchrun on non-Windows when DDP enabled, otherwise run module directly
    if ddp_active and world_size > 1 and os.name != "nt":
        _port = _free_port()
        endpoint = f"127.0.0.1:{_port}"
        cmd = [
            python_executable, "-u", "-m", "torch.distributed.run",
            f"--nproc_per_node={world_size}", "--nnodes=1",
            "--rdzv_backend=c10d", f"--rdzv_endpoint={endpoint}", "--rdzv_conf=use_libuv=False",
            "-m", "aios.cli.aios", *args,
        ]
    else:
        # On Windows, rely on internal spawn-based DDP (backend=gloo) inside implementation
        if ddp_active and os.name == "nt":
            os.environ.setdefault("AIOS_DDP_BACKEND", "gloo")
            os.environ["AIOS_DDP_SPAWN"] = "1"  # Enable internal spawn for Windows GUI
            panel._log("[hrm] Windows multi-GPU: using internal spawn-based DDP (backend=gloo)")
    cmd = [python_executable, "-u", "-m", "aios.cli.aios", *args]
    env = os.environ.copy()
    
    # Set comprehensive DDP environment based on platform and GPU selection
    if ddp_active and world_size > 1:
        if os.name == "nt":
            env["AIOS_DDP_BACKEND"] = "gloo"  # Windows requires gloo for DDP
            env["AIOS_DDP_SPAWN"] = "1"  # Enable internal spawn for Windows
            env["USE_LIBUV"] = "0"  # Disable libuv for Windows compatibility
        else:
            env.setdefault("AIOS_DDP_BACKEND", "nccl")  # Linux/Unix prefer nccl for CUDA
        env["AIOS_DDP_TIMEOUT_SEC"] = "1800"  # 30 minute default timeout
        # Honor GUI setting to abort when DDP init fails (prevents silent single-GPU fallback)
        try:
            abort_flag = bool(getattr(panel, "ddp_abort_on_fail_var", None) and panel.ddp_abort_on_fail_var.get())
            env["AIOS_DDP_ABORT_ON_FAIL"] = "1" if abort_flag else "0"
            if abort_flag:
                panel._log("[hrm] DDP abort-on-fail is ENABLED: will abort if process group init fails")
        except Exception:
            pass
        panel._log(f"[hrm] DDP environment: backend={env.get('AIOS_DDP_BACKEND')} timeout={env.get('AIOS_DDP_TIMEOUT_SEC')}s world_size={world_size}")
    # Propagate GPU memory fraction caps from Resources via environment
    try:
        rp = getattr(panel, "_resources_panel", None)
        if rp is not None:
            rvals = rp.get_values()
            # GPU utilization target (prefer per-GPU map among selected)
            try:
                gpu_util = 0
                sel_train = rvals.get("train_cuda_selected") or []
                umap_train = rvals.get("train_cuda_util_pct") or {}
                if isinstance(umap_train, dict) and isinstance(sel_train, list) and sel_train:
                    try:
                        gpu_util = max(int(umap_train.get(int(i)) or umap_train.get(str(int(i))) or 0) for i in sel_train)
                    except Exception:
                        gpu_util = 0
                if gpu_util <= 0:
                    gpu_util = int(rvals.get("gpu_util_pct") or 0)
                if gpu_util > 0:
                    env["AIOS_GPU_UTIL_TARGET"] = str(gpu_util)
                    env["AIOS_RUN_GPU_UTIL_TARGET"] = str(gpu_util)
            except Exception:
                pass
            # CPU utilization target (applies to training/generation on CPU)
            try:
                cpu_util = int(rvals.get("cpu_util_pct") or 0)
                if cpu_util > 0:
                    env["AIOS_CPU_UTIL_TARGET"] = str(cpu_util)
                    env["AIOS_RUN_CPU_UTIL_TARGET"] = str(cpu_util)
            except Exception:
                pass
            # Training GPU mem fraction
            sel_train = rvals.get("train_cuda_selected") or []
            mem_map_train = rvals.get("train_cuda_mem_pct")
            frac_train = None
            try:
                if isinstance(mem_map_train, dict) and isinstance(sel_train, list) and sel_train:
                    # Use the minimum allowed across selected GPUs
                    vals = []
                    for i in sel_train:
                        try:
                            vals.append(int(mem_map_train.get(int(i)) or mem_map_train.get(str(int(i))) or 0))
                        except Exception:
                            continue
                    if vals:
                        frac_train = max(0.05, min(0.99, (min(vals) / 100.0)))
                elif isinstance(mem_map_train, int):
                    frac_train = max(0.05, min(0.99, (int(mem_map_train) / 100.0)))
            except Exception:
                frac_train = None
            if frac_train is not None:
                env["AIOS_GPU_MEM_FRACTION"] = str(frac_train)
                try:
                    panel._log(f"[hrm] train VRAM cap fraction={frac_train:.2f} (~{int(frac_train*100)}%)")
                except Exception:
                    pass
            # Run/teacher GPU mem fraction
            sel_run = rvals.get("run_cuda_selected") or []
            mem_map_run = rvals.get("run_cuda_mem_pct")
            frac_run = None
            try:
                if isinstance(mem_map_run, dict) and isinstance(sel_run, list) and sel_run:
                    vals = []
                    for i in sel_run:
                        try:
                            vals.append(int(mem_map_run.get(int(i)) or mem_map_run.get(str(int(i))) or 0))
                        except Exception:
                            continue
                    if vals:
                        frac_run = max(0.05, min(0.99, (min(vals) / 100.0)))
                elif isinstance(mem_map_run, int):
                    frac_run = max(0.05, min(0.99, (int(mem_map_run) / 100.0)))
            except Exception:
                frac_run = None
            if frac_run is not None:
                env["AIOS_RUN_GPU_MEM_FRACTION"] = str(frac_run)
                try:
                    panel._log(f"[hrm] run/teacher VRAM cap fraction={frac_run:.2f} (~{int(frac_run*100)}%)")
                except Exception:
                    pass
    except Exception:
        pass
    if cuda_env:
        env["CUDA_VISIBLE_DEVICES"] = cuda_env
    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = getattr(_sp, "CREATE_NEW_PROCESS_GROUP", 0)
    else:  # POSIX: start new session so we can signal the whole process group
        import os as _os
        preexec_fn = getattr(_os, "setsid", None)
    try:
        proc = _sp.Popen(
            cmd,
            stdout=_sp.PIPE,
            stderr=_sp.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn,  # type: ignore[arg-type]
        )
        panel._proc = proc
        panel._log(f"[cli] $ {' '.join(cmd)} (pid={proc.pid})")
        from threading import Thread
        def _drain(stream, prefix=""):
            try:
                for line in iter(stream.readline, ''):
                    if not line:
                        break
                    panel._log(prefix + line.rstrip())
                    if panel._stop_requested and proc.poll() is None:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                stream.close()
            except Exception:
                pass
        if proc.stdout:
            Thread(target=_drain, args=(proc.stdout, ''), daemon=True).start()
        if proc.stderr:
            Thread(target=_drain, args=(proc.stderr, '[stderr] '), daemon=True).start()
        # Update UI state (must run on main thread)
        def _update_ui_start():
            try:
                panel._run_in_progress = True
                panel.start_btn.config(state="disabled")
                panel.progress_lbl.config(text="starting…")
                panel.progress.configure(value=0, mode="determinate")
            except Exception:
                pass
        try:
            panel.after(0, _update_ui_start)
        except Exception:
            _update_ui_start()
        # Wait loop
        while proc.poll() is None:
            if panel._stop_requested:
                # Graceful terminate first
                try:
                    proc.terminate()
                except Exception:
                    pass
                # Give processes a short grace period
                for _ in range(10):
                    if proc.poll() is not None:
                        break
                    _time.sleep(0.3)
                if proc.poll() is None:
                    # Escalate: send CTRL_BREAK (Windows) or SIGTERM group (POSIX)
                    try:
                        if os.name == "nt":
                            try:
                                proc.send_signal(_signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                            except Exception:
                                proc.kill()
                        else:
                            import os as _os, signal as _sig
                            try:
                                pgid = _os.getpgid(proc.pid)
                                _os.killpg(pgid, _sig.SIGTERM)
                            except Exception:
                                proc.terminate()
                    except Exception:
                        pass
                # Final hard kill if still alive after escalation
                for _ in range(10):
                    if proc.poll() is not None:
                        break
                    _time.sleep(0.3)
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                break
            _time.sleep(0.4)
        rc = proc.poll()
        panel._log(f"[cli] process exited rc={rc}")
    except Exception as e:
        panel._log(f"[hrm] launch error: {e}")
    finally:
        def _done():
            panel._run_in_progress = False
            try:
                panel.start_btn.config(state="normal")
                panel.progress.stop()
            except Exception:
                pass
            try:
                panel.progress_lbl.config(text="done")
                panel.progress.configure(value=0, mode="determinate")
            except Exception:
                pass
            # Refresh brains panel to update training steps counter
            try:
                brains_panel = getattr(panel, "_brains_panel", None)
                if brains_panel and hasattr(brains_panel, "refresh"):
                    brains_panel.refresh()
            except Exception:
                pass
        try:
            panel.after(0, _done)
        except Exception:
            _done()


def on_stop(panel: Any) -> None:
    """Universal stop: immediately terminates all training/optimization processes."""
    panel._stop_requested = True
    # Track when stop was requested
    try:
        import time as _time
        panel._stop_request_time = _time.time()
    except Exception:
        pass
    
    # Stop active optimizer immediately if running
    try:
        active_optimizer = getattr(panel, "_active_optimizer", None)
        if active_optimizer is not None:
            panel._log("[hrm] Stop: terminating optimizer processes")
            try:
                # Call force_stop if available, otherwise regular stop
                if hasattr(active_optimizer, "force_stop"):
                    active_optimizer.force_stop()
                elif hasattr(active_optimizer, "stop"):
                    active_optimizer.stop()
            except Exception as e:
                panel._log(f"[hrm] Optimizer stop error: {e}")
            panel._active_optimizer = None
    except Exception as e:
        panel._log(f"[hrm] Optimizer access error: {e}")
    
    # Write STOP file
    sf = _default_stop_file(panel)
    try:
        os.makedirs(os.path.dirname(sf), exist_ok=True)
        with open(sf, "w", encoding="utf-8") as f:
            f.write("stop\n")
        panel._log(f"[hrm] stop requested (STOP file written: {sf})")
    except Exception as e:
        panel._log(f"[hrm] Failed to write stop file: {e}")
    
    # Update UI to show stopping IMMEDIATELY (before terminating process)
    try:
        panel.progress_lbl.config(text="stopping…")
        panel.progress.configure(mode="indeterminate")
        panel.progress.start(15)
    except Exception:
        pass
    
    # Give training process time to detect STOP file and save checkpoint gracefully
    # DO NOT terminate immediately - let the process finish saving
    def _graceful_stop_then_terminate():
        import time as _time
        try:
            proc = getattr(panel, "_proc", None)
            if proc is None:
                return
            
            panel._log("[hrm] Waiting for graceful shutdown (saving checkpoint)…")
            
            # Wait up to 30 seconds for graceful shutdown
            for i in range(120):  # 120 * 0.25 = 30 seconds
                if getattr(proc, "poll", lambda: -999)() is not None:
                    panel._log("[hrm] Training process exited gracefully")
                    return
                _time.sleep(0.25)
            
            # If still running after 30 seconds, force terminate
            if proc.poll() is None:
                panel._log("[hrm] Grace period expired, terminating process…")
                try:
                    proc.terminate()
                except Exception:
                    pass
                
                # Wait 5 more seconds for terminate to work
                for _ in range(20):  # 5 seconds
                    if getattr(proc, "poll", lambda: -999)() is not None:
                        return
                    _time.sleep(0.25)
                
                # Last resort: kill
                if proc.poll() is None:
                    panel._log("[hrm] escalation: attempting kill()…")
                    try:
                        proc.kill()  # Use kill as last resort
                    except Exception:
                        pass
                    
                    # Wait briefly for kill to take effect
                    for _ in range(4):  # 1 second
                        if proc.poll() is not None:
                            return
                        _time.sleep(0.25)
                    
                    if proc.poll() is None:
                        panel._log("[hrm] process still alive after kill, trying group termination…")
                        try:
                            if os.name == "nt":
                                try:
                                    proc.send_signal(_signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                            else:
                                import os as _os
                                try:
                                    pgid = _os.getpgid(proc.pid)
                                    _os.killpg(pgid, _signal.SIGKILL)
                                except Exception:
                                    pass
                        except Exception:
                            pass
        except Exception:
            pass
    
    # Start graceful stop thread
    try:
        _th.Thread(target=_graceful_stop_then_terminate, daemon=True).start()
    except Exception:
        pass
    
    # Handle optimization background thread
    try:
        bg_thread = getattr(panel, "_bg_thread", None)
        if bg_thread is not None and bg_thread.is_alive():
            panel._log("[hrm] Stop: signaling optimization thread to terminate")
            # Thread should check _stop_requested and exit; give it 2 seconds
            def _join_bg():
                try:
                    bg_thread.join(timeout=2.0)
                    if bg_thread.is_alive():
                        panel._log("[hrm] Warning: optimization thread still running")
                except Exception:
                    pass
            _th.Thread(target=_join_bg, daemon=True).start()
    except Exception as e:
        panel._log(f"[hrm] Background thread handling error: {e}")
    
    # Save state after stop requested
    try:
        if callable(getattr(panel, "_save_state_fn", None)):
            panel._save_state_fn()  # type: ignore[misc]
    except Exception:
        pass


def stop_all(panel: Any) -> None:
    """Stop all training and optimization processes, including forceful termination."""
    try:
        panel._metrics_polling_active = False
    except Exception:
        pass
    try:
        on_stop(panel)
    except Exception:
        pass
    try:
        t = getattr(panel, "_bg_thread", None)
        if t is not None and getattr(t, "is_alive", lambda: False)():
            t.join(timeout=2.0)
    except Exception:
        pass
    
    # CRITICAL: Ensure subprocess is actually terminated
    # This runs synchronously to guarantee cleanup before window closes
    try:
        proc = getattr(panel, "_proc", None)
        if proc is not None and proc.poll() is None:
            import time as _time
            import platform
            
            # Log for debugging
            try:
                panel._log(f"[hrm] Force-terminating training process (PID: {proc.pid})")
            except Exception:
                pass
            
            # Give graceful shutdown thread a moment
            _time.sleep(0.5)
            
            # If still running, force terminate
            if proc.poll() is None:
                try:
                    # Windows-specific: Use taskkill for forceful termination
                    if platform.system() == "Windows":
                        import subprocess
                        # Kill entire process tree (including child processes)
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                capture_output=True,
                                timeout=3
                            )
                            panel._log(f"[hrm] Sent taskkill to PID {proc.pid}")
                        except Exception as e:
                            panel._log(f"[hrm] taskkill failed: {e}")
                            # Fallback to standard terminate
                            proc.terminate()
                    else:
                        # Unix: Use standard terminate/kill
                        proc.terminate()
                    
                    # Wait for process to die
                    for _ in range(8):  # 2 seconds
                        if proc.poll() is not None:
                            panel._log("[hrm] Process terminated successfully")
                            break
                        _time.sleep(0.25)
                    
                    # Escalate to kill if needed (Unix only - Windows taskkill is already forceful)
                    if proc.poll() is None:
                        if platform.system() != "Windows":
                            proc.kill()
                        for _ in range(4):  # 1 second
                            if proc.poll() is not None:
                                break
                            _time.sleep(0.25)
                        
                        if proc.poll() is None:
                            panel._log(f"[hrm] WARNING: Process {proc.pid} did not terminate!")
                except Exception as e:
                    try:
                        panel._log(f"[hrm] Termination error: {e}")
                    except Exception:
                        pass
    except Exception as e:
        try:
            panel._log(f"[hrm] stop_all error: {e}")
        except Exception:
            pass
