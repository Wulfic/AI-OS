"""Training start handler for HRM panel.

This module orchestrates the training start process:
1. Auto-detect GPUs if needed
2. Build and validate training config
3. Check for resumable checkpoints
4. Auto-add default goal to brain
5. Create multiprocessing Manager and Events for stop signaling
6. Launch training process
"""

from __future__ import annotations
import os
from typing import Any
import multiprocessing as mp
from .process_manager import launch_training_process
from ....utils.resource_management import submit_background


_MP_SPAWN_CTX = mp.get_context("spawn")


def on_start(panel: Any) -> None:
    """
    Start HRM training run.
    
    This function:
    - Initializes training state
    - Auto-detects CUDA devices if needed
    - Builds and validates TrainingConfig from panel UI
    - Checks for resumable checkpoints (shows resume dialog)
    - Auto-enables resume for iterate mode
    - Validates CUDA device selection
    - Auto-adds default goal to brain
    - Launches training subprocess in background thread
    
    Args:
        panel: HRM training panel instance
    """
    # Initialize state
    try:
        panel._stopped_dialog_shown = False
    except Exception:
        pass
    panel._stop_requested = False
    panel._last_heartbeat = None
    panel._stop_ack_event = None
    panel._graceful_stop_ack_event = None
    panel._stop_ack_notified = False
    panel._graceful_stop_ack_notified = False

    # Auto-detect GPUs if user selected CUDA but no rows populated
    _auto_detect_gpus_if_needed(panel)

    # Build and validate training config
    try:
        config = panel.build_training_config()
        
        # Validate configuration
        try:
            config.validate()
        except ValueError as e:
            panel._log(f"[hrm] Configuration error: {e}")
            return
    except Exception as e:
        panel._log(f"[hrm] Failed to build training configuration: {e}")
        import traceback
        panel._log(f"[hrm] Traceback: {traceback.format_exc()}")
        return

    # Check for resumable checkpoint
    try:
        _check_resumable_checkpoint(panel, config)
    except SystemExit:
        # User cancelled training from resume dialog
        panel._log("[hrm] Training cancelled by user")
        return

    # Auto-enable resume for iterate mode if checkpoint exists
    _auto_resume_for_iterate_mode(panel, config)

    # Validate CUDA device selection
    if not _validate_cuda_selection(panel, config):
        return

    # Convert config to CLI args
    args = config.to_cli_args()

    # Auto-add default goal to brain
    _auto_add_default_goal(panel, config)

    # Log DDP settings
    _log_ddp_settings(panel, config, args)

    # Clear stale STOP file
    _clear_stale_stop_file(panel, config)

    panel._log("Running: aios " + " ".join(args))

    # Persist settings before launch
    try:
        if callable(getattr(panel, "_save_state_fn", None)):
            panel._save_state_fn()
    except Exception:
        pass
    
    # Create multiprocessing Manager and Events for stop signaling
    try:
        panel._log("[hrm] Creating multiprocessing Manager and Events...")
        # Use spawn start method so CUDA is initialized in a clean process
        manager = _MP_SPAWN_CTX.Manager()
        stop_event = manager.Event()
        graceful_stop_event = manager.Event()
        stop_ack_event = manager.Event()
        graceful_stop_ack_event = manager.Event()
        
        # Store on panel for stop button access
        panel._training_manager = manager
        panel._stop_event = stop_event
        panel._graceful_stop_event = graceful_stop_event
        panel._stop_ack_event = stop_ack_event
        panel._graceful_stop_ack_event = graceful_stop_ack_event
        panel._stop_ack_notified = False
        panel._graceful_stop_ack_notified = False
        panel._log("[hrm] Manager and Events created successfully")
    except Exception as e:
        panel._log(f"[hrm] Failed to create Manager: {e}")
        panel._log("[hrm] Training cannot start without Manager")
        return

    # Start metrics polling
    if not panel._metrics_polling_active:
        panel._metrics_polling_active = True
        try:
            panel.after(1000, panel._poll_metrics)
        except Exception:
            pass

    # Launch in background thread to avoid freezing GUI
    def _launch_bg():
        launch_training_process(
            panel,
            args,
            config,
            stop_event,
            graceful_stop_event,
            stop_ack_event,
            graceful_stop_ack_event,
        )

    panel._bg_thread = None
    try:
        panel._bg_future = submit_background(
            "hrm-start",
            _launch_bg,
            pool=getattr(panel, "_worker_pool", None),
        )
    except RuntimeError as exc:
        panel._log(f"[hrm] Failed to queue training start: {exc}")
        panel._bg_future = None
        _launch_bg()


def _auto_detect_gpus_if_needed(panel: Any) -> None:
    """Auto-detect CUDA devices if user selected CUDA but no GPUs populated."""
    try:
        rp = getattr(panel, "_resources_panel", None)
        if rp is not None and callable(getattr(rp, "_detect_and_update", None)):
            # Check if CUDA selected but no GPU rows
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
                    rp._detect_and_update()
                except Exception as e:
                    panel._log(f"[hrm] Auto-detect failed: {e}")
    except Exception:
        pass


def _check_resumable_checkpoint(panel: Any, config: Any) -> None:
    """Always show resume/start dialog to allow start position selection.
    
    Phase 6.4: Training Resume Start Position Selector
    This dialog now always appears to let users choose:
    - Start new run or resume from checkpoint
    - Which block and chunk to start training from
    """
    panel._log("[hrm] === CHECKPOINT/START POSITION CHECK START ===")
    
    if not (config.brain_name and config.dataset_file):
        panel._log("[hrm] No brain_name or dataset_file - skipping checkpoint check")
        return

    from pathlib import Path
    import json

    # Check if checkpoint exists
    brain_path = Path(config.bundle_dir) / config.brain_name
    brain_json_path = brain_path / "brain.json"
    checkpoint_path = brain_path / "actv1_student.safetensors"
    legacy_checkpoint_path = brain_path / "final_model.safetensors"
    chunk_tracker_state_path = brain_path / "chunk_tracker_state.json"
    
    # Use new checkpoint path if it exists, otherwise fall back to legacy
    if not checkpoint_path.exists() and legacy_checkpoint_path.exists():
        checkpoint_path = legacy_checkpoint_path
        panel._log(f"[hrm] Using legacy checkpoint path: {legacy_checkpoint_path}")
    
    panel._log(f"[hrm] Brain path: {brain_path}")
    panel._log(f"[hrm] Looking for brain.json at: {brain_json_path}")
    panel._log(f"[hrm] Looking for checkpoint at: {checkpoint_path}")
    panel._log(f"[hrm] Looking for chunk tracker state at: {chunk_tracker_state_path}")
    panel._log(f"[hrm] brain.json exists: {brain_json_path.exists()}")
    panel._log(f"[hrm] checkpoint exists: {checkpoint_path.exists()}")
    panel._log(f"[hrm] chunk_tracker_state.json exists: {chunk_tracker_state_path.exists()}")

    should_prompt_resume = False
    
    # Check for ChunkTracker state (from parallel_training_v3)
    if chunk_tracker_state_path.exists():
        panel._log(f"[hrm] ChunkTracker state found - checking if resumable...")
        try:
            with chunk_tracker_state_path.open("r", encoding="utf-8") as f:
                tracker_state = json.load(f)
            
            # Check if there are completed chunks (meaning training started)
            completed_chunks = tracker_state.get("completed_chunks", [])
            current_epoch = tracker_state.get("current_epoch", 0)
            total_samples = tracker_state.get("total_samples_trained", 0)
            
            panel._log(f"[hrm] ChunkTracker: epoch={current_epoch}, samples={total_samples}, chunks={len(completed_chunks)}")
            
            if len(completed_chunks) > 0 or total_samples > 0:
                # Training has started and made progress
                if checkpoint_path.exists():
                    should_prompt_resume = True
                    panel._log(f"[hrm] [OK] ChunkTracker state + checkpoint found - WILL SHOW RESUME DIALOG")
                else:
                    panel._log(f"[hrm] [WARN] ChunkTracker state found but checkpoint missing")
                    panel._log(f"[hrm] [WARN] Starting fresh training (checkpoint will be recreated)")
                    # Delete orphaned ChunkTracker state to prevent resuming from wrong position
                    try:
                        import os
                        os.remove(chunk_tracker_state_path)
                        panel._log(f"[hrm] [OK] Deleted orphaned ChunkTracker state")
                    except Exception as e:
                        panel._log(f"[hrm] [WARN] Could not delete ChunkTracker state: {e}")
            else:
                panel._log(f"[hrm] ChunkTracker state exists but no progress yet")
        except Exception as e:
            panel._log(f"[hrm] [X] Error reading ChunkTracker state: {e}")
            import traceback
            panel._log(f"[hrm] Traceback: {traceback.format_exc()}")
    
    # Also check brain.json with last_session data (legacy/fallback)
    if not should_prompt_resume and brain_json_path.exists():
        panel._log(f"[hrm] brain.json exists - checking legacy last_session...")
        try:
            with brain_json_path.open("r", encoding="utf-8") as f:
                brain_data = json.load(f)

            last_session = brain_data.get("last_session")
            panel._log(f"[hrm] last_session exists: {last_session is not None}")
            panel._log(f"[hrm] last_session is dict: {isinstance(last_session, dict)}")
            
            if last_session and isinstance(last_session, dict):
                # We have session data - now check if checkpoint exists
                if checkpoint_path.exists():
                    # Both session data and checkpoint exist - can resume
                    should_prompt_resume = True
                    panel._log(f"[hrm] [OK] Valid last_session + checkpoint found - WILL SHOW RESUME DIALOG")
                else:
                    # Session data exists but checkpoint is missing/corrupted
                    panel._log(f"[hrm] [WARN] Valid last_session found but checkpoint missing")
                    panel._log(f"[hrm] [WARN] This usually means checkpoint was corrupted and auto-deleted")
                    panel._log(f"[hrm] [WARN] Starting fresh training (checkpoint will be recreated)")
            else:
                panel._log(f"[hrm] [X] No valid last_session in brain.json")
        except Exception as e:
            panel._log(f"[hrm] [X] Error reading brain.json: {e}")
            import traceback
            panel._log(f"[hrm] Traceback: {traceback.format_exc()}")
    
    if not should_prompt_resume and not brain_json_path.exists():
        panel._log(f"[hrm] [X] No checkpoint files found - this is a new brain")

    panel._log(f"[hrm] should_prompt_resume = {should_prompt_resume}")

    # Check if we're in shuffle mode (linear_dataset disabled)
    linear_mode = True
    if hasattr(panel, 'linear_dataset_var'):
        try:
            linear_mode = bool(panel.linear_dataset_var.get())
        except Exception:
            pass
    
    # In shuffle mode, automatically randomize start position without showing dialog
    if not linear_mode:
        panel._log(f"[hrm] ==========================================")
        panel._log(f"[hrm] SHUFFLE MODE: Randomizing start position")
        panel._log(f"[hrm] ==========================================")
        
        import random
        
        # Get total blocks and chunks from preprocessing info or defaults
        total_blocks = 22  # Default from preprocessing message
        chunks_per_block = 25  # Default from preprocessing message
        
        # Try to get actual values from preprocessing or manifest
        try:
            from pathlib import Path
            dataset_path = Path(config.dataset_file)
            if dataset_path.is_dir():
                manifest_path = dataset_path / "block_manifest.json"
                if manifest_path.exists():
                    import json
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        total_blocks = manifest.get('total_blocks', total_blocks)
        except Exception as e:
            panel._log(f"[hrm] Could not read manifest, using defaults: {e}")
        
        # Randomize start position
        start_block_id = random.randint(0, max(0, total_blocks - 1))
        start_chunk_id = random.randint(0, max(0, chunks_per_block - 1))
        
        panel._log(f"[hrm] Randomly selected: Block {start_block_id}, Chunk {start_chunk_id}")
        panel._log(f"[hrm] (from {total_blocks} blocks, {chunks_per_block} chunks per block)")
        
        resume_choice = False
        
        # Apply start position to config
        config.start_block_id = start_block_id
        config.start_chunk_id = start_chunk_id
        
    else:
        # Linear mode: Show dialog for start position selection
        panel._log(f"[hrm] ==========================================")
        panel._log(f"[hrm] SHOWING START/RESUME DIALOG (Linear mode)")
        panel._log(f"[hrm] Brain: {config.brain_name}")
        panel._log(f"[hrm] ==========================================")
        try:
            from ..resume_dialog import show_resume_dialog

            # Get root window for dialog parent (panel is a Frame, not a Toplevel)
            try:
                parent_window = panel.winfo_toplevel()
                panel._log(f"[hrm] Got toplevel window: {parent_window}")
            except Exception as e:
                panel._log(f"[hrm] Warning: Could not get toplevel window: {e}, using panel as parent")
                parent_window = panel

            panel._log(f"[hrm] Calling show_resume_dialog()...")
            
            # Dialog now returns a tuple: (resume_choice, start_block_id, start_chunk_id)
            dialog_result = show_resume_dialog(
                parent_window,
                config.brain_name,
                config.bundle_dir,
                config.dataset_file,
                has_checkpoint=should_prompt_resume,
                parent_panel=panel
            )

            panel._log(f"[hrm] Dialog returned: {dialog_result}")
            panel._log(f"[hrm] ==========================================")
            
            if dialog_result is None:
                # User cancelled
                panel._log("[hrm] [X] User CANCELLED - aborting training")
                raise SystemExit("User cancelled")
            
            # Unpack result
            if isinstance(dialog_result, tuple) and len(dialog_result) == 3:
                resume_choice, start_block_id, start_chunk_id = dialog_result
            else:
                # Backward compatibility: old dialog returns just boolean
                resume_choice = dialog_result
                start_block_id, start_chunk_id = 0, 0
            
            # Apply start position to config
            config.start_block_id = start_block_id
            config.start_chunk_id = start_chunk_id
            panel._log(f"[hrm] Start position: Block {start_block_id}, Chunk {start_chunk_id}")
        
        except Exception as e:
            panel._log(f"[hrm] [ERROR] Failed to show dialog: {e}")
            import traceback
            panel._log(f"[hrm] Traceback: {traceback.format_exc()}")
            raise
    
    # Emit telemetry event for start position (applies to both shuffle and linear modes)
    try:
        import json
        from pathlib import Path
        from datetime import datetime
        diagnostics_dir = Path("artifacts/diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        event_file = diagnostics_dir / "analytics_events.jsonl"
        
        event = {
            "event": "training_start_position_selected",
            "timestamp": datetime.now().isoformat(),
            "brain_name": config.brain_name,
            "dataset": config.dataset_file,
            "start_block_id": config.start_block_id,
            "start_chunk_id": config.start_chunk_id,
            "resume": resume_choice,
            "force_train": getattr(config, "force_train", False),
            "shuffle_mode": not linear_mode,
        }
        
        with event_file.open("a") as f:
            f.write(json.dumps(event) + "\n")
            
        panel._log(f"[hrm] Telemetry: training_start_position_selected event emitted")
    except Exception as e:
        panel._log(f"[hrm] Warning: Could not emit telemetry event: {e}")
    
    # Handle resume choice (applies to both shuffle and linear modes)
    try:
        if resume_choice:
            # User chose to resume
            config.resume = True
            config._user_explicitly_chose_resume = True
            panel._log("[hrm] [OK] User chose to RESUME from checkpoint")
        else:
            # User chose fresh start
            config.resume = False
            config._user_explicitly_chose_fresh = True
            panel._log("[hrm] [OK] User chose to START FRESH")
            # CRITICAL: Clear checkpoint path for fresh start
            config.student_init = None
            
            # Delete existing checkpoint files to prevent auto-detection
            try:
                from pathlib import Path
                import os
                import json
                
                if config.brain_name and config.bundle_dir:
                    brain_path = Path(config.bundle_dir) / config.brain_name
                    checkpoint_files = [
                        brain_path / "actv1_student.safetensors",
                        brain_path / "actv1_student.pt",
                        brain_path / "actv1_student.safetensors.prev",
                        brain_path / "actv1_student.pt.prev",
                        brain_path / "chunk_tracker_state.json",  # Clear ChunkTracker state too
                    ]
                    
                    deleted_files = []
                    for checkpoint_file in checkpoint_files:
                        if checkpoint_file.exists():
                            try:
                                os.remove(checkpoint_file)
                                deleted_files.append(checkpoint_file.name)
                            except Exception as e:
                                panel._log(f"[hrm] Warning: Could not delete {checkpoint_file.name}: {e}")
                    
                    # CRITICAL: Clear last_session from brain.json to prevent resume detection
                    brain_json_path = brain_path / "brain.json"
                    if brain_json_path.exists():
                        try:
                            with open(brain_json_path, 'r') as f:
                                brain_data = json.load(f)
                            
                            # Remove resume-related fields
                            if "last_session" in brain_data:
                                del brain_data["last_session"]
                                deleted_files.append("brain.json:last_session")
                            
                            # Reset training counters
                            brain_data["training_steps"] = 0
                            if "last_trained" in brain_data:
                                del brain_data["last_trained"]
                            
                            # Write back to file
                            with open(brain_json_path, 'w') as f:
                                json.dump(brain_data, f, indent=2)
                            
                            panel._log(f"[hrm] Cleared resume metadata from brain.json")
                        except Exception as e:
                            panel._log(f"[hrm] Warning: Could not clean brain.json: {e}")
                    
                    if deleted_files:
                        panel._log(f"[hrm] Deleted old checkpoints: {', '.join(deleted_files)}")
            except Exception as e:
                panel._log(f"[hrm] Warning: Could not clean up old checkpoints: {e}")
    except SystemExit:
        raise  # Re-raise to abort training
    except Exception as e:
        import traceback
        panel._log(f"[hrm] ERROR: Resume/start dialog failed")
        panel._log(f"[hrm] Error: {e}")
        panel._log(f"[hrm] Traceback: {traceback.format_exc()}")
        panel._log(f"[hrm] Aborting training due to dialog error")
        raise SystemExit("Dialog error")
    
    panel._log("[hrm] === CHECKPOINT/START POSITION CHECK END ===")


def _auto_resume_for_iterate_mode(panel: Any, config: Any) -> None:
    """Auto-enable resume for iterate mode if checkpoint exists."""
    user_chose_fresh = getattr(config, "_user_explicitly_chose_fresh", False)
    if config.iterate and config.brain_name and not config.resume and not user_chose_fresh:
        from pathlib import Path
        brain_path = Path(config.bundle_dir) / config.brain_name
        checkpoint_path = brain_path / "actv1_student.safetensors"
        if checkpoint_path.exists():
            config.resume = True
            panel._log("[hrm] Auto-enabling resume for iterate mode (checkpoint found)")


def _validate_cuda_selection(panel: Any, config: Any) -> bool:
    """Validate that at least one GPU is enabled if CUDA selected."""
    if config.device == "cuda":
        try:
            rp = getattr(panel, "_resources_panel", None)
            if rp is not None:
                rvals = rp.get_values()
                sel_train = rvals.get("train_cuda_selected") or []
                if not (isinstance(sel_train, list) and len(sel_train) > 0):
                    panel._log("[hrm] Training device set to CUDA but no GPUs enabled in Resources → enable at least one GPU or switch to CPU.")
                    return False
        except Exception:
            pass
    return True


def _auto_add_default_goal(panel: Any, config: Any) -> None:
    """Auto-add default goal to brain if specified."""
    if not (config.brain_name and config.default_goal):
        return

    try:
        # Check if we have goal callbacks (from app.py integration)
        on_goal_add = getattr(panel, '_on_goal_add_for_brain', None)
        on_goals_list = getattr(panel, '_on_goals_list_for_brain', None)

        if callable(on_goal_add) and callable(on_goals_list):
            def _evaluate_and_add(goals: list[Any]) -> None:
                goal_exists = any(config.default_goal in str(g) for g in (goals or []))
                if goal_exists:
                    panel._log(f"[hrm] Default goal already exists for brain '{config.brain_name}'")
                    return
                panel._log(f"[hrm] Auto-adding default goal to brain '{config.brain_name}': {config.default_goal[:60]}...")
                try:
                    result = on_goal_add(config.brain_name, config.default_goal)
                    if hasattr(result, "add_done_callback"):
                        def _on_done(fut):
                            try:
                                fut.result()
                            except Exception as exc:  # pragma: no cover - defensive
                                panel._log(f"[hrm] Auto-goal add failed: {exc}")

                        result.add_done_callback(_on_done)  # type: ignore[call-arg]
                except Exception as exc:
                    panel._log(f"[hrm] Auto-goal add failed: {exc}")

            try:
                existing_goals = on_goals_list(config.brain_name)
            except Exception as exc:
                panel._log(f"[hrm] Could not fetch existing goals: {exc}")
                existing_goals = []

            if isinstance(existing_goals, list):
                _evaluate_and_add(existing_goals)
            elif hasattr(existing_goals, "add_done_callback"):
                def _on_goals_ready(fut):
                    try:
                        goals_list = fut.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        panel._log(f"[hrm] Could not fetch existing goals: {exc}")
                        return
                    try:
                        panel.after(0, lambda: _evaluate_and_add(goals_list))
                    except Exception:
                        _evaluate_and_add(goals_list)

                existing_goals.add_done_callback(_on_goals_ready)  # type: ignore[call-arg]
            else:
                _evaluate_and_add([])
    except Exception as e:
        panel._log(f"[hrm] Note: Could not auto-add goal: {e}")


def _log_ddp_settings(panel: Any, config: Any, args: list) -> None:
    """Log DDP settings if enabled."""
    zero_stage = str(getattr(config, "zero_stage", "none") or "none").lower()
    cuda_ids_list: list[str] = []
    if config.cuda_ids:
        try:
            cuda_ids_list = [c.strip() for c in str(config.cuda_ids).split(",") if c.strip()]
        except Exception:
            cuda_ids_list = []

    if zero_stage == "zero3":
        if cuda_ids_list:
            gpu_repr = ",".join(cuda_ids_list)
            if len(cuda_ids_list) > 1:
                panel._log(f"[hrm] ZeRO-3 standalone mode active on GPUs {gpu_repr}; DeepSpeed stage-3 handles sharding (no DDP).")
            else:
                panel._log(f"[hrm] ZeRO-3 standalone mode active on GPU {gpu_repr}; DeepSpeed stage-3 handles sharding (no DDP).")
        else:
            panel._log("[hrm] ZeRO-3 standalone mode active; DeepSpeed stage-3 will handle device assignment (no DDP).")
        return

    if config.ddp and config.world_size and config.world_size > 1:
        backend = "gloo" if os.name == "nt" else "nccl"
        panel._log(f"[hrm] Auto-DDP enabled: world_size={config.world_size} cuda_ids={config.cuda_ids} backend={backend}")
    elif config.parallel_independent and len(cuda_ids_list) > 1:
        panel._log(f"[hrm] Parallel independent mode active on GPUs {', '.join(cuda_ids_list)} (no DDP).")
    elif config.cuda_ids:
        panel._log(f"[hrm] Single GPU training: cuda_id={config.cuda_ids}")

    # Extra explicit DDP summary
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


def _clear_stale_stop_file(panel: Any, config: Any) -> None:
    """Clear stale STOP file if exists."""
    from .stop_training import get_default_stop_file
    
    sf = config.stop_file or get_default_stop_file(panel)
    if sf:
        try:
            if os.path.exists(sf):
                os.remove(sf)
                panel._log(f"[hrm] Cleared stale STOP file: {sf}")
        except Exception:
            pass
    
    # Also clear GRACEFUL_STOP file
    try:
        graceful_sf = os.path.join(panel._project_root, "training_datasets", "actv1", "GRACEFUL_STOP")
        if os.path.exists(graceful_sf):
            os.remove(graceful_sf)
            panel._log(f"[hrm] Cleared stale GRACEFUL_STOP file: {graceful_sf}")
    except Exception:
        pass
