from __future__ import annotations

import os
import shutil
from typing import Any
import threading
import time

# Import the new advanced optimizer
try:
    from .optimizer_v2 import optimize_settings_v2
    from .gpu_monitor import create_gpu_monitor
    NEW_OPTIMIZER_AVAILABLE = True
except ImportError:
    NEW_OPTIMIZER_AVAILABLE = False


def optimize_settings(panel: Any) -> None:
    """Run an open-ended optimization to fit Resource limits.

    Flow:
    - Generation load test honoring Resources run device and caps.
      Probe in ~10s increments until either utilization target is reached or an
      OOM/backoff signal is observed. When target is reached, run a                         timer_bs = _start_st                    # Confirmation window at settled batch
                    try:
                        if os.path.exists(stop_file):
                            os.remove(stop_file)
                    except                    # Confirmation window at settled batch
                    try:
                        if os.path.exists(stop_file):
                            os.remove(stop_file)
                    except Exception:
                        pass
                    if getattr(panel, \"_stop_requested\", False):
                        break
                    timer2 = _start_stop_timer(stop_file, CONFIRM)
                    confirm_out = _run_probe([*probe_args, \"--batch-size\", str(best_tb), \"--stop-file\", stop_file])
                    try:
                        timer2.join(timeout=0.1)
                    except Exception:
                        pass
                    
                    # Check if confirmation was stopped
                    if confirm_out == \"optimization_stopped\" or getattr(panel, \"_stop_requested\", False):
                        panel._log(\"[opt] Training confirmation stopped by user request\")
                        break
                    break                      pass
                    if getattr(panel, \"_stop_requested\", False):
                        break
                    timer2 = _start_stop_timer(stop_file, CONFIRM)
                    confirm_out = _run_probe([*probe_args, \"--td-batch\", str(best_gb), \"--stop-file\", stop_file])
                    try:
                        timer2.join(timeout=0.1)
                    except Exception:
                        pass
                    
                    # Check if confirmation was stopped
                    if confirm_out == \"optimization_stopped\" or getattr(panel, \"_stop_requested\", False):
                        panel._log(\"[opt] Generation confirmation stopped by user request\")
                        break
                    breakile, PROBE_BIN)
                        out2 = _run_probe([*bs_args, \"--stop-file\", stop_file])
                        try:
                                              timer_bs = _start_stop_timer(stop_file, PROBE_BIN)
                        out2 = _run_probe([*bs_args, \"--stop-file\", stop_file])
                        try:
                            timer_bs.join(timeout=0.1)
                        except Exception:
                            pass
                        
                        # Check if binary search probe was stopped
                        if out2 == \"optimization_stopped\" or getattr(panel, \"_stop_requested\", False):
                            panel._log(\"[opt] Binary search stopped by user request\")
                            break
                            
                        low2 = out2.lower()
                        if (\"oom_backoff\" in low2) or (\"out of memory\" in low2):mer_bs.join(timeout=0.1)
                        except Exception:
                            pass
                        
                        # Check if binary search probe was stopped
                        if out2 == \"optimization_stopped\" or getattr(panel, \"_stop_requested\", False):
                            panel._log(\"[opt] Binary search stopped by user request\")
                            break
                            
                        low2 = out2.lower()
                        if (\"gen_oom_backoff\" in low2) or (\"out of memory\" in low2):    confirmation window at that setting.
    - Training load test honoring Resources train device and caps.
      Same probing and confirmation behavior as above.
    - Use OOM backoff + last_safe.json to discover safe td_batch and train     # Skip applying last_safe.json at the end to respect discovered post-OOM settings
    try:
        panel._log(f"[opt] Applied train batch={panel.batch_var.get()} td-batch={panel.td_batch_var.get()}")
    except Exception:
        pass

    # Optimization complete - all operations performed on live models and data
    """
    # Early stop check
    if getattr(panel, "_stop_requested", False):
        panel._log("[opt] Stop requested before optimization start; aborting.")
        return
        
    panel._log("[opt] Starting optimization process...")
    # Common setup
    model = panel.model_var.get().strip() or "gpt2"
    max_seq = panel.max_seq_var.get().strip() or "128"
    halt = panel.halt_steps_var.get().strip() or "1"
    teacher = panel.teacher_var.get().strip()
    teacher_dataset_flag = bool(getattr(panel, "teacher_dataset_var").get())
    ds = panel.dataset_var.get().strip()
    base_dir = os.path.join(panel._project_root, "artifacts", "brains", "actv1")
    # Use live brain directory instead of temporary _dryrun
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception:
        pass

    # Resolve selected student/brain; require selection (student_init or brain_name), file optional
    try:
        si = (panel.student_init_var.get() or "").strip()
        bname = (panel.brain_name_var.get() or "").strip()
        if not si and bname:
            bdir = os.path.join(base_dir, bname)
            cand = os.path.join(bdir, "actv1_student.pt")
            if os.path.exists(cand) or os.path.isdir(bdir):
                si = cand
                try:
                    panel.student_init_var.set(cand)
                except Exception:
                    pass
        if not (si or bname):
            panel._log("[opt] No student/brain selected. Please select one before optimizing.")
            return
        if si and os.path.isfile(si):
            panel._log(f"[opt] Using selected student checkpoint: {si}")
        else:
            if bname:
                panel._log(f"[opt] Optimizing with selected brain '{bname}' (no existing checkpoint found yet).")
    except Exception:
        panel._log("[opt] Failed to resolve selected student. Aborting optimization.")
        return

    # Read Resources and compute env caps
    def _apply_env_from_resources(for_phase: str) -> dict[str, str]:
        env_updates: dict[str, str] = {}
        rp = getattr(panel, "_resources_panel", None)
        if rp is None:
            return env_updates
        try:
            rvals = rp.get_values()
            try:
                panel._log(f"[opt][dbg] resources[{for_phase}] get_values keys={list(rvals.keys())}")
            except Exception:
                pass
            # Selected CUDA IDs
            sel: list[int] = []
            if for_phase == "train":
                sel = rvals.get("train_cuda_selected") or []
            else:
                sel = rvals.get("run_cuda_selected") or []
            if isinstance(sel, list) and sel:
                try:
                    ids = ",".join(str(int(i)) for i in sel)
                    env_updates["AIOS_CUDA_IDS"] = ids
                    # Provide world size to helper logic if needed
                    env_updates["AIOS_WORLD_SIZE"] = str(len(sel))
                    # On Windows, ensure a compatible DDP backend
                    if os.name == "nt" and len(sel) > 1:
                        env_updates["AIOS_DDP_BACKEND"] = "gloo"
                    try:
                        panel._log(f"[opt][dbg] phase={for_phase} cuda_ids={ids} world={len(sel)}")
                    except Exception:
                        pass
                except Exception:
                    pass
            # Utilization targets
            try:
                # Prefer per-GPU util target among selected GPUs; fall back to top-level
                gpu_util = 0
                if for_phase == "train":
                    umap = rvals.get("train_cuda_util_pct") or {}
                    sel = rvals.get("train_cuda_selected") or []
                    if isinstance(umap, dict) and isinstance(sel, list) and sel:
                        try:
                            gpu_util = max(int(umap.get(int(i)) or umap.get(str(int(i))) or 0) for i in sel)
                        except Exception:
                            gpu_util = 0
                else:
                    umap = rvals.get("run_cuda_util_pct") or {}
                    sel = rvals.get("run_cuda_selected") or []
                    if isinstance(umap, dict) and isinstance(sel, list) and sel:
                        try:
                            gpu_util = max(int(umap.get(int(i)) or umap.get(str(int(i))) or 0) for i in sel)
                        except Exception:
                            gpu_util = 0
                if gpu_util <= 0:
                    gpu_util = int(rvals.get("gpu_util_pct") or 0)
                if gpu_util > 0:
                    env_updates["AIOS_GPU_UTIL_TARGET"] = str(gpu_util)
                    env_updates["AIOS_RUN_GPU_UTIL_TARGET"] = str(gpu_util)
            except Exception:
                pass
            try:
                cpu_util = int(rvals.get("cpu_util_pct") or 0)
                if cpu_util > 0:
                    env_updates["AIOS_CPU_UTIL_TARGET"] = str(cpu_util)
                    env_updates["AIOS_RUN_CPU_UTIL_TARGET"] = str(cpu_util)
            except Exception:
                pass
            # IMPORTANT: For optimization we intentionally do NOT enforce per-process GPU memory
            # fractions. Doing so can hard-cap VRAM (e.g., to 35%) and hide the true OOM ceiling.
            # We still honor device selection and utilization targets. Training runs (not optimize)
            # will continue to honor the Resources tab memory caps.
        except Exception:
            pass
        return env_updates

    def _with_env(temp_env: dict[str, str]):
        """Context manager-like helper: apply env updates and return a restore function."""
        old_vals: dict[str, str | None] = {}
        # Force near-full memory fraction for optimize runs to allow pushing to true OOM.
        temp = {
            **temp_env,
            "AIOS_GPU_MEM_FRACTION": "0.99",
            "AIOS_RUN_GPU_MEM_FRACTION": "0.99",
        }
        for k, v in temp.items():
            old_vals[k] = os.environ.get(k)
            os.environ[k] = str(v)
        def _restore():
            for k, ov in old_vals.items():
                if ov is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = str(ov)
        return _restore

    def _start_stop_timer(stop_path: str, seconds: int) -> threading.Thread:
        def _writer():
            try:
                time.sleep(max(1, int(seconds)))
                os.makedirs(os.path.dirname(stop_path), exist_ok=True)
                with open(stop_path, "w", encoding="utf-8") as f:
                    f.write("stop\n")
            except Exception:
                pass
        th = threading.Thread(target=_writer, daemon=True)
        th.start()
        return th

    # Build device args from Resources
    def _device_args(phase: str) -> list[str]:
        args: list[str] = []
        try:
            rp = getattr(panel, "_resources_panel", None)
            dev = "auto"
            sel: list[int] = []
            if rp is not None:
                rvals = rp.get_values()
                if phase == "train":
                    td = str(rvals.get("train_device") or "auto").lower()
                    sel = rvals.get("train_cuda_selected") or []
                else:
                    td = str(rvals.get("run_device") or "auto").lower()
                    sel = rvals.get("run_cuda_selected") or []
                try:
                    panel._log(f"[opt][dbg] _device_args phase={phase} td={td} sel={sel}")
                except Exception:
                    pass
                if td in {"cpu", "cuda"}:
                    dev = td
                elif isinstance(sel, list) and len(sel) > 0:
                    dev = "cuda"
                if dev == "cuda":
                    if not (isinstance(sel, list) and len(sel) > 0):
                        panel._log(f"[opt] {phase}: CUDA selected but no GPUs enabled in Resources.")
                        return ["--device", "cpu"]
                    try:
                        ids = ",".join(str(int(i)) for i in sel)
                        args += ["--cuda-ids", ids]
                        # Enable DDP for multi-GPU in both phases to exercise full utilization.
                        if len(sel) > 1:
                            args += ["--ddp", "--world-size", str(len(sel))]
                            # Set optimization-friendly DDP environment (automatic based on platform)
                            try:
                                import os as __os
                                if __os.name == "nt":
                                    __os.environ["AIOS_DDP_BACKEND"] = "gloo"  # Windows compatibility
                                # Shorter timeouts for optimization probes
                                __os.environ["AIOS_DDP_TIMEOUT_SEC"] = "30"
                                # Enable optimization-specific coordination
                                __os.environ["AIOS_DDP_OPTIMIZE_MODE"] = "1"
                            except Exception:
                                pass
                            try:
                                panel._log(f"[opt] phase={phase} enabling DDP world_size={len(sel)} cuda_ids={ids}")
                            except Exception:
                                pass
                    except Exception:
                        pass
            args += ["--device", dev]
            if phase == "gen":
                # teacher-device for generation
                args += ["--teacher-device", dev]
        except Exception:
            args += ["--device", "auto"]
            if phase == "gen":
                args += ["--teacher-device", "auto"]
        return args

    # STOP file path reused
    stop_file = os.path.join(panel._project_root, "training_data", "actv1", "STOP")
    # Ensure STOP cleared first
    try:
        if os.path.exists(stop_file):
            os.remove(stop_file)
    except Exception:
        pass

    # Helper: run a short probe and return output text
    def _run_probe(arglist: list[str]) -> str:
        try:
            # Check for stop before starting probe
            if getattr(panel, "_stop_requested", False):
                panel._log("[opt] Stop requested; skipping probe")
                return "optimization_stopped"
                
            # Launch the probe process
            import subprocess as _sp
            import threading as _th
            
            # Start the CLI process
            result = {"output": "", "completed": False}
            
            def _run_cli():
                try:
                    result["output"] = panel._run_cli(arglist) or ""
                    result["completed"] = True
                except Exception as e:
                    panel._log(f"[opt] probe error: {e}")
                    result["output"] = ""
                    result["completed"] = True
                    
            cli_thread = _th.Thread(target=_run_cli, daemon=True)
            cli_thread.start()
            
            # Wait for completion or stop request
            timeout_counter = 0
            while not result["completed"] and timeout_counter < 300:  # 30 second max
                if getattr(panel, "_stop_requested", False):
                    panel._log("[opt] Stop requested during probe; terminating")
                    # The CLI should handle the STOP file termination
                    break
                time.sleep(0.1)
                timeout_counter += 1
                
            # Give thread a moment to finish
            cli_thread.join(timeout=2.0)
            
            return result["output"]
        except Exception as e:
            panel._log(f"[opt] probe error: {e}")
            return ""

    # NOTE: We intentionally ignore last_safe.json during optimization to push to true OOM.
    def _read_last_safe() -> tuple[int | None, int | None]:
        return (None, None)

    # Timing parameters (seconds): default to fast 10s probes per batch size change.
    # Binary-search refinement also 10s. Keep a longer (30s) confirmation window
    # to validate stability, but drastically faster than the previous 180s.
    PROBE_LONG = 10   # per initial probe of a candidate batch
    PROBE_BIN  = 10   # per binary-search refinement probe
    CONFIRM    = 30   # final confirmation at settled batch

    # 1) Generation load test (open-ended)
    try:
        log1 = os.path.join(base_dir, "opt_gen.jsonl")
        args1: list[str] = [
            "hrm-hf", "train-actv1",
            "--model", model,
            "--max-seq-len", str(max_seq),
            "--batch-size", str(panel.batch_var.get().strip() or "8"),  # not critical for gen
            "--steps", "100000",  # effectively infinite until STOP
            "--halt-max-steps", str(halt or "0"),
            "--brain-name", bname or "actv1",
            "--bundle-dir", base_dir,
            "--log-file", log1,
            "--strict",
            "--dataset-file", "training_data/curated_datasets/test_sample.txt",
        ]
        # Use the selected student so generation load reflects the current brain
        if si and os.path.isfile(si):
            args1 += ["--student-init", si]
        # Dataset file is already set above
        if getattr(panel, "td_prompt_var", None) is not None:
            tp = panel.td_prompt_var.get().strip()
            if tp:
                args1 += ["--td-prompt", tp]
        if getattr(panel, "td_seed_var", None) is not None:
            sd = panel.td_seed_var.get().strip()
            if sd:
                args1 += ["--td-seed", sd]
        # Teacher (model for generation). If teacher-as-dataset is enabled but Teacher is empty,
        # fall back to the Tokenizer/Model path so we can actually generate.
        t_model = panel.teacher_var.get().strip()
        if not t_model and bool(getattr(panel, "teacher_dataset_var").get()):
            t_model = panel.model_var.get().strip() or "gpt2"
        if t_model:
            args1 += ["--teacher", t_model]
        # Device flags and STOP
        args1 += _device_args("gen")
        args1 += ["--stop-file", stop_file]
        # Apply env for generation
        env1 = _apply_env_from_resources("gen")
        restore1 = _with_env(env1)
        try:
            panel._log("[opt] Generation load test (extended)… pushing to OOM/target then backing off. No model changes will be saved.")
            try:
                from os import environ as __env
                panel._log(f"[opt][dbg] gen env CUDA_IDS={__env.get('AIOS_CUDA_IDS')} WORLD={__env.get('AIOS_WORLD_SIZE')} BACKEND={__env.get('AIOS_DDP_BACKEND')}")
            except Exception:
                pass
            # Iterative probing with ~10s trials until an actual OOM is observed
            # Start from current td-batch or a conservative default
            try:
                cur_gb = int(panel.td_batch_var.get() or "8")
            except Exception:
                cur_gb = 8
            best_gb = cur_gb
            last_good = max(1, int(cur_gb))
            attempts = 0
            while True:
                if getattr(panel, "_stop_requested", False):
                    panel._log("[opt] stop requested; aborting generation optimization.")
                    break
                attempts += 1
                
                # Enable Force Stop after initial attempts
                if attempts >= 2 and not getattr(panel, "_force_stop_available", False):
                    try:
                        panel.force_stop_btn.config(state="normal")
                        panel._force_stop_available = True
                        panel._log("[opt] Force Stop available during optimization")
                    except Exception:
                        pass
                # Run a probe with current td-batch
                gb_try = max(1, int(best_gb))
                probe_args = [*args1]
                # ensure td-batch override
                if "--td-batch" in probe_args:
                    i = probe_args.index("--td-batch")
                    if i + 1 < len(probe_args):
                        probe_args[i + 1] = str(gb_try)
                else:
                    probe_args += ["--td-batch", str(gb_try)]
                # time-limit via STOP file
                try:
                    if os.path.exists(stop_file):
                        os.remove(stop_file)
                except Exception:
                    pass
                if getattr(panel, "_stop_requested", False):
                    break
                timer = _start_stop_timer(stop_file, PROBE_LONG)
                out = _run_probe([*probe_args, "--stop-file", stop_file])
                try:
                    timer.join(timeout=0.1)
                except Exception:
                    pass
                
                # Check if probe was stopped
                if out == "optimization_stopped" or getattr(panel, "_stop_requested", False):
                    panel._log("[opt] Generation optimization stopped by user request")
                    break
                    
                # Check for OOM only; ignore throttle until after first OOM
                lower_out = out.lower()
                oom_signal = ("gen_oom_backoff" in lower_out) or ("out of memory" in lower_out)
                if oom_signal:
                    # Found ceiling; binary search between last_good and gb_try-1 toward util target
                    lo = max(1, int(last_good))
                    hi = max(lo, int(gb_try) - 1)
                    best = lo
                    for _ in range(8):
                        if lo > hi or getattr(panel, "_stop_requested", False):
                            break
                        mid = max(1, (lo + hi) // 2)
                        bs_args = [*args1]
                        if "--td-batch" in bs_args:
                            i = bs_args.index("--td-batch")
                            if i + 1 < len(bs_args):
                                bs_args[i + 1] = str(mid)
                        else:
                            bs_args += ["--td-batch", str(mid)]
                        try:
                            if os.path.exists(stop_file):
                                os.remove(stop_file)
                        except Exception:
                            pass
                        timer_bs = _start_stop_timer(stop_file, PROBE_BIN)
                        out2 = _run_probe([*bs_args, "--stop-file", stop_file])
                        try:
                            timer_bs.join(timeout=0.1)
                        except Exception:
                            pass
                        low2 = out2.lower()
                        if ("gen_oom_backoff" in low2) or ("out of memory" in low2):
                            hi = mid - 1
                            continue
                        throttle_hit = ("throttle_run_gpu\": 1" in out2) or ("throttle_run_cpu\": 1" in out2)
                        best = mid
                        # If throttle indicates meeting/exceeding target, try a tad lower to back off slightly
                        if throttle_hit:
                            hi = mid - 1
                        else:
                            lo = mid + 1
                    best_gb = max(1, int(best))
                    panel._log(f"[opt] gen post-OOM settle td-batch={best_gb}")
                    # Confirmation window at settled batch
                    try:
                        if os.path.exists(stop_file):
                            os.remove(stop_file)
                    except Exception:
                        pass
                    if getattr(panel, "_stop_requested", False):
                        break
                    timer2 = _start_stop_timer(stop_file, CONFIRM)
                    _run_probe([*probe_args, "--td-batch", str(best_gb), "--stop-file", stop_file])
                    try:
                        timer2.join(timeout=0.1)
                    except Exception:
                        pass
                    break
                else:
                    # record last good and increase aggressively to find OOM
                    last_good = gb_try
                    inc = max(1, int(gb_try * 0.5))
                    best_gb = gb_try + inc
                    panel._log(f"[opt] gen increase td-batch -> {best_gb}")
                # Practical guard: if no OOM after several long probes or batch extremely large,
                # proceed to training phase (VRAM scaling happens primarily during training).
                if attempts >= 12 or best_gb >= 4096:
                    panel._log("[opt] gen did not hit OOM; proceeding to training load tuning.")
                    break
            # No additional fine-tuning needed; best_gb already settled
            # Apply result
            panel.td_batch_var.set(str(max(1, int(best_gb))))
            try:
                if callable(getattr(panel, "_save_state_fn", None)):
                    panel._save_state_fn()  # type: ignore[misc]
            except Exception:
                pass
        finally:
            try:
                restore1()
            except Exception:
                pass
            try:
                if os.path.exists(stop_file):
                    os.remove(stop_file)
            except Exception:
                pass
    except Exception as e:
        panel._log(f"[opt] Generation test error: {e}")

    # 2) Training load test (open-ended)
    try:
        log2 = os.path.join(base_dir, "opt_train.jsonl")
        args2: list[str] = [
            "hrm-hf", "train-actv1",
            "--model", model,
            "--max-seq-len", str(max_seq),
            "--batch-size", str(panel.batch_var.get().strip() or "8"),
            "--steps", "100000",
            "--halt-max-steps", str(halt or "0"),
            "--brain-name", bname or "actv1",
            "--bundle-dir", base_dir,
            "--log-file", log2,
            "--strict",
        ]
        # Use the selected student for training load tuning
        if si and os.path.isfile(si):
            args2 += ["--student-init", si]
        # Prefer dataset file for quick start; use test dataset if no dataset specified
        if ds:
            args2 += ["--dataset-file", ds]
        else:
            args2 += ["--dataset-file", "training_data/curated_datasets/test_sample.txt"]
        # Learning rate if set (not critical for load)
        lr = panel.lr_var.get().strip()
        if lr:
            args2 += ["--lr", lr]
        # System RAM cap from Resources CPU util
        try:
            rp = getattr(panel, "_resources_panel", None)
            if rp is not None:
                cap = int(rp.get_values().get("cpu_util_pct") or 0)
                if cap > 0:
                    args2 += ["--sys-mem-cap-pct", str(cap)]
        except Exception:
            pass
        # Device flags and STOP
        args2 += _device_args("train")
        args2 += ["--stop-file", stop_file]
        # Apply env for training
        env2 = _apply_env_from_resources("train")
        restore2 = _with_env(env2)
        try:
            # Show CUDA selection/DDP info for visibility
            try:
                from os import environ as __env
                panel._log(f"[opt] train devices: cuda_ids={__env.get('AIOS_CUDA_IDS')} world={__env.get('AIOS_WORLD_SIZE')} backend={__env.get('AIOS_DDP_BACKEND','nccl')}")
                panel._log(f"[opt][dbg] train env snapshot size={len(__env)}")
            except Exception:
                pass
            panel._log("[opt] Training load test (extended)… pushing to OOM/target then backing off. No model changes will be saved.")
            try:
                cur_tb = int(panel.batch_var.get() or "8")
            except Exception:
                cur_tb = 8
            best_tb = cur_tb
            last_good = max(1, int(cur_tb))
            attempts = 0
            while True:
                if getattr(panel, "_stop_requested", False):
                    panel._log("[opt] stop requested; aborting training optimization.")
                    break
                attempts += 1
                
                # Enable Force Stop after initial attempts  
                if attempts >= 2 and not getattr(panel, "_force_stop_available", False):
                    try:
                        panel.force_stop_btn.config(state="normal")
                        panel._force_stop_available = True
                        panel._log("[opt] Force Stop available during training optimization")
                    except Exception:
                        pass
                tb_try = max(1, int(best_tb))
                probe_args = [*args2]
                # ensure batch-size override
                if "--batch-size" in probe_args:
                    i = probe_args.index("--batch-size")
                    if i + 1 < len(probe_args):
                        probe_args[i + 1] = str(tb_try)
                else:
                    probe_args += ["--batch-size", str(tb_try)]
                # time-limit via STOP file
                try:
                    if os.path.exists(stop_file):
                        os.remove(stop_file)
                except Exception:
                    pass
                if getattr(panel, "_stop_requested", False):
                    break
                timer = _start_stop_timer(stop_file, PROBE_LONG)
                out = _run_probe([*probe_args, "--stop-file", stop_file])
                try:
                    timer.join(timeout=0.1)
                except Exception:
                    pass
                
                # Check if probe was stopped
                if out == "optimization_stopped" or getattr(panel, "_stop_requested", False):
                    panel._log("[opt] Training optimization stopped by user request")
                    break
                    
                # Check for OOM only; ignore throttle until after first OOM
                lower_out = out.lower()
                oom_signal = ("oom_backoff" in lower_out) or ("out of memory" in lower_out)
                if oom_signal:
                    # Post-OOM binary search toward utilization target
                    lo = max(1, int(last_good))
                    hi = max(lo, int(tb_try) - 1)
                    best = lo
                    for _ in range(8):
                        if lo > hi or getattr(panel, "_stop_requested", False):
                            break
                        mid = max(1, (lo + hi) // 2)
                        bs_args = [*args2]
                        if "--batch-size" in bs_args:
                            i = bs_args.index("--batch-size")
                            if i + 1 < len(bs_args):
                                bs_args[i + 1] = str(mid)
                        else:
                            bs_args += ["--batch-size", str(mid)]
                        try:
                            if os.path.exists(stop_file):
                                os.remove(stop_file)
                        except Exception:
                            pass
                        timer_bs = _start_stop_timer(stop_file, PROBE_BIN)
                        out2 = _run_probe([*bs_args, "--stop-file", stop_file])
                        try:
                            timer_bs.join(timeout=0.1)
                        except Exception:
                            pass
                        low2 = out2.lower()
                        if ("oom_backoff" in low2) or ("out of memory" in low2):
                            hi = mid - 1
                            continue
                        throttle_hit = ("throttle_gpu\": 1" in out2) or ("throttle_cpu\": 1" in out2)
                        best = mid
                        if throttle_hit:
                            hi = mid - 1
                        else:
                            lo = mid + 1
                    best_tb = max(1, int(best))
                    panel._log(f"[opt] train post-OOM settle batch={best_tb}")
                    # Confirmation window at settled batch
                    try:
                        if os.path.exists(stop_file):
                            os.remove(stop_file)
                    except Exception:
                        pass
                    if getattr(panel, "_stop_requested", False):
                        break
                    timer2 = _start_stop_timer(stop_file, CONFIRM)
                    _run_probe([*probe_args, "--batch-size", str(best_tb), "--stop-file", stop_file])
                    try:
                        timer2.join(timeout=0.1)
                    except Exception:
                        pass
                    break
                else:
                    # record last good and increase to find OOM
                    last_good = tb_try
                    inc = max(1, int(tb_try * 0.5))
                    best_tb = tb_try + inc
                    panel._log(f"[opt] train increase batch -> {best_tb}")
                # ultra-conservative guard to avoid infinite runaway if no signals ever appear
                if attempts > 1000:
                    panel._log("[opt] train reached max attempts without target/OOM; applying current best and finishing.")
                    break
            # No additional fine-tuning needed; best_tb already settled
            # Apply result
            panel.batch_var.set(str(max(1, int(best_tb))))
            try:
                if callable(getattr(panel, "_save_state_fn", None)):
                    panel._save_state_fn()  # type: ignore[misc]
            except Exception:
                pass
        finally:
            try:
                restore2()
            except Exception:
                pass
            try:
                if os.path.exists(stop_file):
                    os.remove(stop_file)
            except Exception:
                pass
    except Exception as e:
        panel._log(f"[opt] Training test error: {e}")

    # Skip applying last_safe.json at the end to respect discovered post-OOM settings
    try:
        panel._log(f"[opt] Applied train batch={panel.batch_var.get()} td-batch={panel.td_batch_var.get()}")
    except Exception:
        pass

    panel._log("[opt] Optimization complete - all operations performed on live models and data")
