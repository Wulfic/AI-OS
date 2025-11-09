""""""

Unified Optimization System - Re-export wrapper for backward compatibilityUnified Optimization System - Re-export wrapper for backward compatibility



REFACTORED: This file now re-exports from the optimizer_unified package.REFACTORED: This file now re-exports from the optimizer_unified package.

The implementation has been split into focused modules for better maintainability.The implementation has been split into focused modules for better maintainability.



Original file: 1223 lines → 9 focused modulesFor new code, prefer importing directly from the package:

- config.py (85 lines): OptimizationConfig dataclass    from aios.gui.components.hrm_training.optimizer_unified import OptimizationConfig, UnifiedOptimizer

- process_manager.py (208 lines): Process management with heartbeat monitoring

- command_builder.py (107 lines): Command/environment building utilitiesThis wrapper maintains backward compatibility with existing imports.

- gpu_monitoring.py (92 lines): GPU monitor integration"""

- result_parser.py (119 lines): Throughput/OOM parsing

- batch_runner.py (227 lines): Single batch execution# Re-export all public symbols from the refactored package

- optimizer.py (262 lines): Main UnifiedOptimizer classfrom .optimizer_unified import (

- batch_optimization.py (242 lines): Adaptive batch size optimization    OptimizationConfig,

- api.py (190 lines): Public API functions for GUI/CLI/TUI    UnifiedOptimizer,

    optimize_from_config,

For new code, prefer importing directly from the package:    optimize_from_dict,

    from aios.gui.components.hrm_training.optimizer_unified import (    optimize_from_gui,

        OptimizationConfig,    optimize_cli,

        UnifiedOptimizer,)

        optimize_from_config,

        optimize_from_dict,__all__ = [

        optimize_from_gui,    'OptimizationConfig',

        optimize_cli    'UnifiedOptimizer',

    )    'optimize_from_config',

    'optimize_from_dict',

This wrapper maintains backward compatibility with existing imports.    'optimize_from_gui',

"""    'optimize_cli',

]

# Re-export all public symbols from the refactored package

from .optimizer_unified import (# Backward compatibility - deprecated class name

    OptimizationConfig,class OptimizationConfig:

    UnifiedOptimizer,    """Unified configuration for optimization across all interfaces."""

    optimize_from_config,    

    optimize_from_dict,    # Model configuration

    optimize_from_gui,    model: str = "base_model"

    optimize_cli,    teacher_model: str = ""

)    max_seq_len: int = 512

    dataset_file: str = "training_data/curated_datasets/test_sample.txt"  # Dataset for optimization

__all__ = [    

    'OptimizationConfig',    # Optimization parameters

    'UnifiedOptimizer',    test_duration: int = 45  # seconds per test - increased for real work

    'optimize_from_config',    max_timeout: int = 240   # max subprocess timeout - increased for DDP init (160s) + training (30s) + buffer

    'optimize_from_dict',    batch_sizes: Optional[List[int]] = None  # Will default to [1, 2, 4, 8]

    'optimize_from_gui',    min_batch_size: int = 1

    'optimize_cli',    max_batch_size: Optional[int] = None

]    batch_growth_factor: float = 2.0

    
    # Training test parameters
    train_steps: int = 10  # Steps to test training throughput
    
    # GPU configuration
    use_multi_gpu: bool = True
    cuda_devices: str = ""  # e.g., "0,1"
    device: str = "auto"
    strict: bool = False
    target_util: Optional[int] = None  # Target GPU utilization (default: 90%)
    util_tolerance: int = 5
    monitor_interval: float = 1.0
    
    # Output configuration
    log_callback: Optional[Callable[[str], None]] = None
    stop_callback: Optional[Callable[[], bool]] = None  # Returns True if stop requested
    output_dir: str = "artifacts/optimization"
    
    def __post_init__(self):
        if not self.batch_sizes:
            self.batch_sizes = [1, 2, 4, 8]
        else:
            normalized: List[int] = []
            for value in self.batch_sizes:
                try:
                    ivalue = int(value)
                    if ivalue > 0:
                        normalized.append(ivalue)
                except Exception:
                    continue
            self.batch_sizes = sorted(set(normalized)) or [1, 2, 4, 8]

        self.min_batch_size = max(1, int(self.min_batch_size or 1))
        self.min_batch_size = max(self.min_batch_size, self.batch_sizes[0])

        if self.max_batch_size is None:
            self.max_batch_size = max(self.batch_sizes)
        else:
            self.max_batch_size = max(int(self.max_batch_size), self.min_batch_size)

        try:
            self.batch_growth_factor = float(self.batch_growth_factor)
        except Exception:
            self.batch_growth_factor = 2.0
        self.batch_growth_factor = max(1.2, self.batch_growth_factor)

        if not self.teacher_model:
            self.teacher_model = self.model

        if self.target_util is not None and self.target_util <= 0:
            self.target_util = None

        try:
            self.util_tolerance = max(0, int(self.util_tolerance))
        except Exception:
            self.util_tolerance = 5

        try:
            self.monitor_interval = float(self.monitor_interval)
        except Exception:
            self.monitor_interval = 1.0
        self.monitor_interval = max(0.5, self.monitor_interval)


class ProcessManager:
    """Aggressive process management to prevent hanging."""
    
    def __init__(self, timeout: int = 240):  # Increased default timeout for DDP init + training
        self.timeout = timeout
        self.processes = []
        self._heartbeat_file = None
        self._heartbeat_timeout = 60  # seconds without heartbeat = frozen (increased for slow init)
        self._stop_callback = None  # Callback to check if user requested stop
        
    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        heartbeat_file: Optional[Path] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        stop_callback: Optional[Callable[[], bool]] = None
    ) -> subprocess.CompletedProcess:
        """Run command with aggressive timeout, cleanup, and heartbeat monitoring."""
        
        self._heartbeat_file = heartbeat_file
        self._stop_callback = stop_callback
        
        # Start the process in a new process group
        try:
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # Unix-like
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                    env=env,
                    preexec_fn=os.setsid
                )
                
            self.processes.append(process)
            
            # Wait with timeout and heartbeat monitoring
            try:
                stdout, stderr = self._communicate_with_monitoring(process, log_callback)
                return subprocess.CompletedProcess(
                    cmd, process.returncode, stdout, stderr
                )
            except subprocess.TimeoutExpired:
                # Aggressively kill the process and all children
                self._kill_process_tree(process)
                return subprocess.CompletedProcess(
                    cmd, -1, "", f"Process killed after {self.timeout}s timeout"
                )
                
        except Exception as e:
            return subprocess.CompletedProcess(
                cmd, -1, "", f"Failed to start process: {e}"
            )
    
    def _communicate_with_monitoring(self, process, log_callback=None) -> Tuple[str, str]:
        """Monitor process with heartbeat detection to catch freezes.
        
        Drains stdout/stderr in real-time to prevent pipe buffer deadlocks on Windows,
        which can cause DDP workers to hang when they write more output than the pipe
        buffer can hold (typically 4-64KB).
        """
        import threading
        import queue
        
        start_time = time.time()
        last_heartbeat = start_time
        last_progress_log = start_time
        
        # Use threads to drain stdout/stderr in real-time to prevent deadlocks
        stdout_lines = []
        stderr_lines = []
        stdout_thread = None
        stderr_thread = None
        
        def drain_pipe(pipe, line_list):
            """Continuously drain a pipe to prevent buffer overflow."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line_list.append(line)
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass
        
        # Start drain threads
        if process.stdout:
            stdout_thread = threading.Thread(target=drain_pipe, args=(process.stdout, stdout_lines), daemon=True)
            stdout_thread.start()
        
        if process.stderr:
            stderr_thread = threading.Thread(target=drain_pipe, args=(process.stderr, stderr_lines), daemon=True)
            stderr_thread.start()
        
        # Poll process with heartbeat checks
        while process.poll() is None:
            elapsed = time.time() - start_time
            
            # Check if user requested stop
            if self._stop_callback:
                try:
                    if self._stop_callback():
                        if log_callback:
                            log_callback("  ... stop requested by user")
                        raise subprocess.TimeoutExpired(process.args, elapsed, output="User requested stop")
                except subprocess.TimeoutExpired:
                    raise  # Re-raise our own exception
                except Exception:
                    pass  # Ignore callback errors
            
            # Log progress every 10 seconds for long-running processes
            if time.time() - last_progress_log > 10:
                if log_callback:
                    log_callback(f"  ... still initializing ({int(elapsed)}s elapsed)")
                last_progress_log = time.time()
            
            # Check overall timeout
            if elapsed > self.timeout:
                raise subprocess.TimeoutExpired(process.args, self.timeout)
            
            # Check heartbeat if file provided
            if self._heartbeat_file:
                try:
                    if self._heartbeat_file.exists():
                        mtime = self._heartbeat_file.stat().st_mtime
                        if mtime > last_heartbeat:
                            last_heartbeat = mtime
                    
                    # Check for frozen process (no heartbeat)
                    if time.time() - last_heartbeat > self._heartbeat_timeout:
                        raise subprocess.TimeoutExpired(
                            process.args, self._heartbeat_timeout,
                            output=f"No heartbeat for {self._heartbeat_timeout}s"
                        )
                except Exception:
                    pass
            
            time.sleep(0.5)
        
        # Process completed - wait for drain threads to finish (with timeout)
        if stdout_thread is not None:
            stdout_thread.join(timeout=2.0)
        if stderr_thread is not None:
            stderr_thread.join(timeout=2.0)
        
        # Combine captured output
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)
        return stdout, stderr
    
    def _kill_process_tree(self, process):
        """Kill process and all its children."""
        try:
            if process.poll() is None:  # Process still running
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                
                # Kill children first
                for child in children:
                    try:
                        child.kill()
                    except:
                        pass
                        
                # Kill parent
                try:
                    parent.kill()
                except:
                    pass
                    
                # Force terminate subprocess
                try:
                    process.kill()
                except:
                    pass
                    
        except Exception:
            # Last resort - try to terminate the subprocess directly
            try:
                process.terminate()
                process.kill()
            except:
                pass
    
    def cleanup(self):
        """Clean up any remaining processes."""
        for process in self.processes:
            self._kill_process_tree(process)


class UnifiedOptimizer:
    """Unified optimizer that works across all interfaces."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.session_id = str(uuid.uuid4())[:8]
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific files
        self.stop_file = self.output_dir / f"stop_{self.session_id}.flag"
        self.train_log = self.output_dir / f"train_{self.session_id}.jsonl"
        
        # Process manager for aggressive cleanup
        self.process_manager = ProcessManager(config.max_timeout)
        
        # Results storage
        self.results = {
            "session_id": self.session_id,
            "config": self._config_as_dict(),
            "training": {"success": False, "optimal_batch": 1, "results": []},
            "errors": []
        }

        cuda_ids = self._parse_cuda_devices(self.config.cuda_devices)
        target_util = self.config.target_util if self.config.target_util else 90

        self.gpu_config = {
            "ids": cuda_ids,
            "target_util": target_util,
            "multi_gpu": self.config.use_multi_gpu and len(cuda_ids) > 1
        }

        all_ids: List[int] = []
        for token in cuda_ids:
            try:
                all_ids.append(int(token))
            except Exception:
                continue
        self.all_cuda_ids = sorted(set(all_ids))

        self.util_tolerance = max(0, self.config.util_tolerance)
        self.monitor_interval = self.config.monitor_interval
        self.global_stop_paths = [
            Path("training_data/actv1/STOP"),
            Path("training_data/actv1/stop"),
            Path("training_data/actv1/Stop"),
        ]
        
    def log(self, message: str):
        """Log message via callback or print."""
        timestamp = time.strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        
        if self.config.log_callback:
            self.config.log_callback(full_msg)
        else:
            print(full_msg)
    
    def is_stop_requested(self) -> bool:
        """Check if user requested stop."""
        if self.config.stop_callback:
            try:
                return self.config.stop_callback()
            except:
                return False
        return False
    
    def force_stop(self):
        """Immediately terminate all processes and cleanup."""
        self.log("Stop requested - terminating all processes...")
        self.cleanup()
    
    def cleanup(self):
        """Clean up session files and processes."""
        try:
            self.process_manager.cleanup()
            
            # Remove session files
            for file_path in [self.stop_file, self.train_log]:
                try:
                    if file_path.exists():
                        file_path.unlink()
                except:
                    pass
        except Exception as e:
            self.log(f"Cleanup error: {e}")
    
    def optimize(self) -> Dict[str, Any]:
        """Run complete optimization process."""
        
        self.log("=" * 60)
        self.log("Starting Training Optimization System")
        self.log("=" * 60)
        self.log(f"Session ID: {self.session_id}")
        self.log(f"Model: {self.config.model}")
        self.log(f"Max sequence length: {self.config.max_seq_len}")
        self.log(f"Batch sizes to test: {self.config.batch_sizes}")
        self.log(f"Test duration per batch: {self.config.test_duration}s")
        
        ids = self.gpu_config.get("ids", [])
        multi = self.gpu_config.get("multi_gpu")
        target = self.gpu_config.get("target_util")
        device = self.config.device
        
        if ids:
            self.log(f"Training GPUs: {ids} (multi-GPU={'enabled' if multi else 'disabled'})")
        if target:
            self.log(f"Target utilization: {target}%")
        if device:
            self.log(f"Device preference: {device}")
        if self.config.strict:
            self.log("Strict device enforcement enabled")
        
        # Validate dataset file exists
        dataset_path = Path(self.config.dataset_file)
        if not dataset_path.exists():
            error_msg = f"❌ Dataset file not found: {dataset_path}"
            self.log(error_msg)
            self.log("Please create this file or specify a different dataset")
            self.results["errors"].append(error_msg)
            return self.results
        else:
            size = dataset_path.stat().st_size
            self.log(f"✓ Dataset file found: {dataset_path} ({size} bytes)")
        
        self.log("=" * 60)
        
        try:
            # Test training workload
            if self.is_stop_requested():
                self.log("Stop requested before training testing")
                return self.results
            
            self.log("\n" + "=" * 60)
            self.log("Testing Training Workload...")
            self.log("=" * 60)
            train_results = self._test_training_workload()
            self.results["training"] = train_results
            
            if train_results["success"]:
                self.log(f"\n✓ SUCCESS: Training optimal batch: {train_results['optimal_batch']}")
            else:
                self.log(f"\n❌ ERROR: Training testing failed")
            
            # Final results
            self.log("\n" + "=" * 60)
            self.log("Optimization Results:")
            self.log("=" * 60)
            self.log(f"Training optimal batch: {train_results.get('optimal_batch', 'N/A')}")
            self.log(f"Max throughput: {train_results.get('max_throughput', 0):.2f} steps/sec")
            
            # Save results
            self.results["config"] = self._config_as_dict()
            results_file = self.output_dir / f"results_{self.session_id}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            self.log(f"\n✓ Results saved to: {results_file}")
            self.log("=" * 60)
            
        except Exception as e:
            error_msg = f"Optimization failed: {e}"
            self.log(f"❌ ERROR: {error_msg}")
            import traceback
            self.log(f"Traceback:\n{traceback.format_exc()}")
            self.results["errors"].append(error_msg)
            
        finally:
            self.cleanup()
            
        return self.results

    def _config_as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable view of the optimization config."""
        config_dict = asdict(self.config)
        # Remove non-serializable callback functions
        config_dict.pop("log_callback", None)
        config_dict.pop("stop_callback", None)
        return config_dict

    def _parse_cuda_devices(self, raw_devices: Optional[Any] = None) -> List[str]:
        """Return sanitized list of requested CUDA device identifiers."""
        if raw_devices is None:
            raw_devices = self.config.cuda_devices

        devices: List[str] = []

        if isinstance(raw_devices, (list, tuple, set)):
            iterable = raw_devices
        elif isinstance(raw_devices, str):
            iterable = raw_devices.split(",")
        elif raw_devices is None:
            iterable = []
        else:
            iterable = [raw_devices]

        for item in iterable:
            if item is None:
                continue
            token = str(item).strip()
            if token:
                devices.append(token)

        return devices

    def _build_command_env(self) -> Dict[str, str]:
        """Create environment variables for optimizer subprocess."""
        env = os.environ.copy()

        ids = self.gpu_config.get("ids", [])
        if ids:
            joined = ",".join(ids)
            env["CUDA_VISIBLE_DEVICES"] = joined
            env["AIOS_CUDA_IDS"] = joined
            
        if self.gpu_config.get("multi_gpu"):
            env.setdefault("AIOS_WORLD_SIZE", str(len(ids)))
            backend = "gloo" if os.name == "nt" else "nccl"
            env.setdefault("AIOS_DDP_BACKEND", backend)
            env.setdefault("AIOS_DDP_OPTIMIZE_MODE", "1")
            # Enable internal DDP spawn for GUI/optimizer compatibility
            env.setdefault("AIOS_DDP_SPAWN", "1")

        target = self.gpu_config.get("target_util")
        if target:
            env.setdefault("AIOS_GPU_UTIL_TARGET", str(target))

        env["AIOS_OPT_SESSION"] = self.session_id
        env["AIOS_OPT_STRICT"] = "1" if self.config.strict else "0"
        
        # Speed up transformers import in DDP workers (Windows optimization)
        # Disable slow import structure scanning which can take 60-120s per worker
        env.setdefault("TRANSFORMERS_OFFLINE", "1")  # Skip online checks
        env.setdefault("HF_DATASETS_OFFLINE", "1")   # Skip dataset checks
        
        return env

    def _extend_with_device_args(self, cmd: List[str]) -> None:
        """Add device- and GPU-related CLI arguments."""
        ids = self.gpu_config.get("ids", [])

        if ids and "--cuda-ids" not in cmd:
            cmd.extend(["--cuda-ids", ",".join(ids)])
            if self.gpu_config.get("multi_gpu"):
                if "--ddp" not in cmd:
                    cmd.append("--ddp")
                if "--world-size" not in cmd:
                    cmd.extend(["--world-size", str(len(ids))])

        device = (self.config.device or "auto").strip()
        if device and device != "auto" and "--device" not in cmd:
            cmd.extend(["--device", device])

        if self.config.strict and "--strict" not in cmd:
            cmd.append("--strict")
    
    def _test_training_workload(self) -> Dict[str, Any]:
        """Optimize training workload dynamically."""

        base_cmd = [
            get_preferred_python_executable(), "-m", "aios.cli.aios",
            "hrm-hf", "train-actv1",
            "--model", self.config.model,
            "--max-seq-len", str(self.config.max_seq_len),
            "--steps", str(self.config.train_steps),
            "--dataset-file", self.config.dataset_file,  # Use actual dataset from config
            "--stop-file", str(self.stop_file),
            "--log-file", str(self.train_log),
            # Memory optimization flags (required parameters)
            "--amp",  # Enable AMP by default
            "--no-cpu-offload",  # Don't use CPU offload during testing
            "--gradient-checkpointing",  # Enable for memory savings
        ]

        # Only add teacher if it's different from the model
        if self.config.teacher_model and self.config.teacher_model != self.config.model:
            base_cmd.extend(["--teacher", self.config.teacher_model])

        self._extend_with_device_args(base_cmd)

        return self._run_phase_optimization(
            base_cmd=base_cmd,
            batch_flag="--batch-size",
            metric_label="steps/sec",
            throughput_parser=self._parse_training_throughput,
            log_path=self.train_log
        )

    def _run_phase_optimization(
        self,
        *,
        base_cmd: List[str],
        batch_flag: str,
        metric_label: str,
        throughput_parser: Callable[[], float],
        log_path: Path
    ) -> Dict[str, Any]:
        """Run adaptive optimization for training workload."""

        results: Dict[str, Any] = {
            "success": False,
            "optimal_batch": self.config.min_batch_size,
            "max_throughput": 0.0,
            "results": []
        }

        available = self.config.batch_sizes or [1, 2, 4, 8]
        min_batch = max(self.config.min_batch_size, available[0])
        max_batch = max(self.config.max_batch_size or available[-1], min_batch)

        visited: Dict[int, Dict[str, Any]] = {}
        success_batches: List[int] = []
        failed_batches: List[int] = []

        current_batch = min_batch

        oom_count = 0  # Track consecutive OOMs for adaptive backoff
        last_successful_batch = min_batch
        
        while current_batch <= max_batch and current_batch not in visited:
            # Check for stop request
            if self.is_stop_requested():
                self.log("Stop requested during optimization")
                break
            
            batch_data = self._run_single_batch(
                base_cmd=base_cmd,
                batch_flag=batch_flag,
                batch_size=current_batch,
                throughput_parser=throughput_parser,
                metric_label=metric_label,
                log_path=log_path
            )

            visited[current_batch] = batch_data
            results["results"].append(batch_data)

            if batch_data["success"]:
                success_batches.append(current_batch)
                last_successful_batch = current_batch
                oom_count = 0  # Reset OOM counter on success
                
                if batch_data["throughput"] > results["max_throughput"]:
                    results["max_throughput"] = batch_data["throughput"]
                    results["optimal_batch"] = current_batch

                # Check memory usage to decide if we should keep pushing
                mem_pct = batch_data.get("memory_percent", 0.0)
                if mem_pct > 90.0:
                    self.log(f"Memory at {mem_pct:.1f}% - near limit, slowing growth")
                    # Near memory limit, be more conservative
                    next_batch = current_batch + max(1, current_batch // 4)
                elif mem_pct > 75.0:
                    # Good memory usage, continue with normal growth
                    next_batch = self._next_batch_size(current_batch, available, max_batch)
                else:
                    # Low memory usage, be more aggressive
                    next_batch = self._next_batch_size(current_batch, available, max_batch)
                    if next_batch == current_batch * 2 and current_batch < 32:
                        # Double the jump for very low memory usage
                        next_batch = min(current_batch * 4, max_batch)
                
                if next_batch <= current_batch:
                    break
                current_batch = next_batch
            else:
                failed_batches.append(current_batch)
                
                if batch_data.get("oom", False):
                    oom_count += 1
                    self.log(f"OOM at batch {current_batch} (OOM #{oom_count})")
                    
                    # Smart backoff based on OOM history
                    if oom_count == 1 and current_batch > min_batch:
                        # First OOM: try 75% of failed batch (less aggressive backoff)
                        # This helps find the actual limit more precisely
                        refined_batch = int(current_batch * 0.75)
                        if refined_batch > last_successful_batch and refined_batch not in visited:
                            self.log(f"First OOM, trying 75% size: {refined_batch}")
                            current_batch = refined_batch
                            continue
                    
                    # Multiple OOMs or no room to refine: start binary search
                    if current_batch > min_batch:
                        self.log(f"Starting binary search between {last_successful_batch} and {current_batch}")
                        break
                    else:
                        # Can't go lower, stop
                        break
                else:
                    # Other error (not OOM), stop testing larger batches
                    self.log(f"Non-OOM error at batch {current_batch}, stopping growth")
                    break

        if failed_batches and success_batches:
            lower = max(success_batches)
            upper = failed_batches[0]
            while upper - lower > 1:
                mid = (lower + upper) // 2
                if mid in visited or mid <= 0:
                    break

                batch_data = self._run_single_batch(
                    base_cmd=base_cmd,
                    batch_flag=batch_flag,
                    batch_size=mid,
                    throughput_parser=throughput_parser,
                    metric_label=metric_label,
                    log_path=log_path
                )

                visited[mid] = batch_data
                results["results"].append(batch_data)

                if batch_data["success"]:
                    success_batches.append(mid)
                    success_batches = sorted(set(success_batches))
                    lower = mid
                    if batch_data["throughput"] > results["max_throughput"]:
                        results["max_throughput"] = batch_data["throughput"]
                        results["optimal_batch"] = mid
                else:
                    upper = mid

        success_batches = sorted(set(success_batches))

        if success_batches:
            selected = self._select_optimal_batch(success_batches, visited)
            if selected:
                results["success"] = True
                results["optimal_batch"] = selected["batch_size"]
                results["max_throughput"] = selected["throughput"]
                results["target_util"] = self.gpu_config.get("target_util")
                results["selected"] = selected

        return results

    def _next_batch_size(self, current: int, available: List[int], max_batch: int) -> int:
        """Determine the next batch size candidate with aggressive exponential growth."""

        for candidate in available:
            if candidate > current:
                return min(candidate, max_batch)

        # Aggressive exponential growth for small batches to quickly find limits
        if current < 16:
            # Double for small batches: 1→2, 2→4, 4→8, 8→16
            next_size = current * 2
        elif current < 64:
            # Still aggressive: 16→32, 32→64
            next_size = current * 2
        else:
            # More conservative near potential limits
            next_size = int(math.ceil(current * self.config.batch_growth_factor))
        
        if next_size <= current:
            next_size = current + 1

        return min(next_size, max_batch)

    def _select_optimal_batch(
        self,
        success_batches: List[int],
        data_map: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Choose the batch result that best matches target utilization, memory usage, and throughput."""

        if not success_batches:
            return None

        target = self.gpu_config.get("target_util")
        tolerance = self.util_tolerance

        entries = [data_map[b] for b in success_batches if b in data_map]
        if not entries:
            return None

        # Memory-aware selection: prefer batches using 80-95% memory for maximum performance
        # This ensures we're pushing GPU limits without risking OOM
        def score_batch(e: Dict[str, Any]) -> Tuple[float, float, float, int]:
            """Score batch: (memory_score, throughput, utilization, -batch_size)"""
            mem_pct = e.get("memory_percent", 0.0)
            throughput = e.get("throughput", 0.0)
            util = e.get("utilization", 0.0)
            batch = e.get("batch_size", 0)
            
            # Memory score: prefer 80-95% range (sweet spot for performance)
            if 80.0 <= mem_pct <= 95.0:
                memory_score = 1.0  # Ideal range
            elif mem_pct > 95.0:
                memory_score = 0.7  # Too close to OOM, risky
            elif mem_pct > 60.0:
                memory_score = 0.5 + (mem_pct - 60.0) / 40.0  # 0.5-1.0 scaling
            else:
                memory_score = mem_pct / 60.0  # 0.0-0.5 scaling
            
            return (memory_score, throughput, util, -batch)

        if target is None:
            # No target: maximize throughput with good memory usage
            return max(entries, key=score_batch)

        within = [
            e for e in entries
            if abs(e.get("utilization", 0.0) - target) <= tolerance
        ]
        if within:
            return max(within, key=score_batch)

        # If no batch reaches target, check if we're far from target (all batches < target - tolerance)
        max_util = max(e.get("utilization", 0.0) for e in entries)
        if max_util < target - tolerance:
            # Target unreachable - prioritize throughput and memory usage
            self.log(f"Target {target}% unreachable (max {max_util:.1f}%), selecting by throughput + memory")
            return max(entries, key=score_batch)

        below = [e for e in entries if e.get("utilization", 0.0) < target]
        if below:
            return max(below, key=score_batch)

        above = [e for e in entries if e.get("utilization", 0.0) > target]
        if above:
            return min(above, key=lambda e: (e.get("utilization", 0.0), -e.get("throughput", 0.0)))

        return entries[-1]

    def _run_single_batch(
        self,
        *,
        base_cmd: List[str],
        batch_flag: str,
        batch_size: int,
        throughput_parser: Callable[[], float],
        metric_label: str,
        log_path: Path
    ) -> Dict[str, Any]:
        """Execute a single batch test run and collect metrics."""

        test_cmd = base_cmd.copy()
        if batch_flag in test_cmd:
            try:
                idx = test_cmd.index(batch_flag)
                if idx + 1 < len(test_cmd):
                    test_cmd[idx + 1] = str(batch_size)
                else:
                    test_cmd.append(str(batch_size))
            except ValueError:
                test_cmd.extend([batch_flag, str(batch_size)])
        else:
            test_cmd.extend([batch_flag, str(batch_size)])

        env = self._build_command_env()
        
        # Log the FULL command being executed (critical for debugging)
        cmd_str = " ".join(test_cmd)
        self.log(f"Testing batch {batch_size}...")
        self.log(f"Command: {cmd_str}")
        
        # Log environment variables
        if env.get("CUDA_VISIBLE_DEVICES"):
            self.log(f"  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        if env.get("AIOS_WORLD_SIZE"):
            self.log(f"  AIOS_WORLD_SIZE={env['AIOS_WORLD_SIZE']}")
        if env.get("AIOS_DDP_SPAWN"):
            self.log(f"  AIOS_DDP_SPAWN={env['AIOS_DDP_SPAWN']}")

        # Clean old log but keep it for parsing after run
        if log_path.exists():
            try:
                log_path.unlink()
            except Exception:
                pass

        self._prepare_for_batch()

        monitor = self._start_gpu_monitor()
        timer = threading.Timer(self.config.test_duration, self._create_stop_file)
        timer.start()

        start_time = time.perf_counter()
        try:
            result = self.process_manager.run_command(
                test_cmd, 
                env=env, 
                heartbeat_file=log_path,
                log_callback=self.log,
                stop_callback=self.is_stop_requested
            )
        finally:
            timer.cancel()

        duration = max(0.001, time.perf_counter() - start_time)
        summary = self._stop_gpu_monitor(monitor)
        
        # Parse throughput BEFORE cleanup (log file needs to exist)
        throughput = throughput_parser()
        
        # Log detailed error info if process failed or throughput is 0
        if result.returncode != 0 or throughput == 0.0:
            self.log(f"⚠️  Process exit code: {result.returncode}, throughput: {throughput}")
            
            # Show stderr (most important for errors)
            if result.stderr and result.stderr.strip():
                stderr_lines = result.stderr.strip().split('\n')
                self.log("STDERR output:")
                for line in stderr_lines[-10:]:  # Show last 10 lines
                    self.log(f"  {line}")
            
            # Show stdout if stderr is empty
            if (not result.stderr or not result.stderr.strip()) and result.stdout and result.stdout.strip():
                stdout_lines = result.stdout.strip().split('\n')
                self.log("STDOUT output:")
                for line in stdout_lines[-10:]:  # Show last 10 lines
                    self.log(f"  {line}")
            
            # Check if log file was created
            if not log_path.exists():
                self.log(f"⚠️  Log file was never created: {log_path}")
                self.log("  This usually means the training command failed during initialization")
            elif log_path.stat().st_size == 0:
                self.log(f"⚠️  Log file is empty: {log_path}")
                self.log("  Training started but didn't complete any steps")
        
        # Now cleanup
        self._cleanup_after_batch()
        utilization = self._extract_utilization(summary)
        memory_pct = self._extract_memory(summary)
        oom = self._detect_oom(result)
        
        # Success if: no OOM AND (clean exit OR throughput > 0)
        # Exit code -1 with throughput > 0 means training completed but hit timeout
        success = not oom and (result.returncode == 0 or throughput > 0)

        # Clear status line
        status = "✓ SUCCESS" if success else ("💥 OOM" if oom else "❌ FAILED")
        self.log(
            f"\n{status} - Batch {batch_size}: "
            f"{throughput:.2f} {metric_label} | "
            f"GPU {utilization:.1f}% | "
            f"Mem {memory_pct:.1f}% | "
            f"exit {result.returncode}"
        )

        return {
            "batch_size": batch_size,
            "throughput": throughput,
            "utilization": utilization,
            "memory_percent": memory_pct,
            "duration": duration,
            "exit_code": result.returncode,
            "success": success,
            "oom": oom,
            "stderr": (result.stderr or "").strip()[:400],
            "stdout": (result.stdout or "").strip()[:200],
            "gpu_summary": summary
        }

    def _start_gpu_monitor(self):
        monitor_ids = self.all_cuda_ids
        if not monitor_ids:
            return None
        try:
            log_file = self.output_dir / f"gpu_metrics_{self.session_id}.jsonl"
            monitor = create_gpu_monitor(monitor_ids, str(log_file))
            monitor.start_monitoring(interval=self.monitor_interval)
            return monitor
        except Exception as exc:
            self.log(f"GPU monitor unavailable: {exc}")
            return None

    def _stop_gpu_monitor(self, monitor) -> Dict[str, Any]:
        if not monitor:
            return {}
        try:
            monitor.stop_monitoring()
            return monitor.get_summary() or {}
        except Exception:
            return {}

    def _extract_utilization(self, summary: Dict[str, Any]) -> float:
        if not summary or "error" in summary:
            return 0.0
        utils: List[float] = []
        for metrics in summary.values():
            if isinstance(metrics, dict):
                util = metrics.get("utilization_avg")
                if util is None:
                    util = metrics.get("utilization_max")
                if util is not None:
                    try:
                        utils.append(float(util))
                    except Exception:
                        continue
        return max(utils) if utils else 0.0

    def _extract_memory(self, summary: Dict[str, Any]) -> float:
        if not summary or "error" in summary:
            return 0.0
        mems: List[float] = []
        for metrics in summary.values():
            if isinstance(metrics, dict):
                mem = metrics.get("memory_avg")
                if mem is None:
                    mem = metrics.get("memory_max")
                if mem is not None:
                    try:
                        mems.append(float(mem))
                    except Exception:
                        continue
        return max(mems) if mems else 0.0

    def _detect_oom(self, result: subprocess.CompletedProcess) -> bool:
        """Detect OOM with comprehensive error pattern matching."""
        stderr = (result.stderr or "").lower()
        stdout = (result.stdout or "").lower()
        
        oom_patterns = [
            "out of memory",
            "cuda oom",
            "cuda error",
            "cudnn error",
            "allocation",
            "cuda runtime error",
            "cannot allocate",
            "memory error",
            "ran out of",
        ]
        
        for pattern in oom_patterns:
            if pattern in stderr or pattern in stdout:
                return True
        
        # Check exit code - common OOM codes
        if result.returncode in [-9, 137, 247]:  # SIGKILL, OOM killer codes
            return True
        
        return False

    def _prepare_for_batch(self) -> None:
        try:
            if self.stop_file.exists():
                self.stop_file.unlink()
        except Exception:
            pass
        for path in self.global_stop_paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                continue

    def _cleanup_after_batch(self) -> None:
        try:
            if self.stop_file.exists():
                self.stop_file.unlink()
        except Exception:
            pass
    
    def _create_stop_file(self):
        """Create stop file to signal test completion."""
        try:
            self.stop_file.touch()
        except:
            pass
    
    
    def _parse_training_throughput(self) -> float:
        """Parse training throughput from log file."""
        import json
        try:
            if not self.train_log.exists():
                self.log(f"⚠️  Training log not found: {self.train_log}")
                self.log("  Training command likely failed before writing any logs")
                return 0.0
            
            file_size = self.train_log.stat().st_size
            if file_size == 0:
                self.log(f"⚠️  Training log is empty: {self.train_log}")
                self.log("  Training started but didn't write any metrics")
                return 0.0
            
            self.log(f"Parsing training log: {self.train_log} ({file_size} bytes)")
            
            steps_completed = 0
            first_ts = None
            last_ts = None
            line_count = 0
            train_events = 0
            
            with open(self.train_log, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    try:
                        data = json.loads(line.strip())
                        ts = data.get("ts")
                        event = data.get("event")
                        
                        # Log first few events for debugging
                        if line_count <= 3:
                            self.log(f"  Line {line_count}: {event} (step={data.get('step', '?')})")
                        
                        # Count training steps
                        if event == "train":
                            train_events += 1
                            steps_completed += 1
                            if ts:
                                if first_ts is None:
                                    first_ts = ts
                                last_ts = ts
                    except json.JSONDecodeError as e:
                        self.log(f"  ⚠️  Invalid JSON at line {line_count}: {e}")
                        continue
            
            self.log(f"Log summary: {line_count} total lines, {train_events} train events")
            
            # Calculate throughput: steps per second
            if steps_completed > 0 and first_ts and last_ts and last_ts > first_ts:
                duration = last_ts - first_ts
                throughput = steps_completed / max(duration, 0.1)
                self.log(f"✓ Throughput: {steps_completed} steps in {duration:.1f}s = {throughput:.2f} steps/sec")
                return throughput
            elif steps_completed > 0:
                # Fallback if no timestamps
                self.log(f"⚠️  Using fallback calculation (no valid timestamps), {steps_completed} steps")
                return steps_completed / max(self.config.test_duration, 1)
            else:
                self.log(f"⚠️  No training steps found in {line_count} log lines")
            
            return 0.0
            
        except Exception as e:
            self.log(f"❌ Error parsing training throughput: {e}")
            import traceback
            self.log(f"  {traceback.format_exc()}")
            return 0.0


# Unified API functions for different interfaces
def optimize_from_config(config: OptimizationConfig) -> Tuple[Dict[str, Any], 'UnifiedOptimizer']:
    """Main optimization entry point - works from any interface. Returns (results, optimizer_instance)."""
    optimizer = UnifiedOptimizer(config)
    results = optimizer.optimize()
    return results, optimizer


def optimize_from_dict(config_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], 'UnifiedOptimizer']:
    """Optimize from dictionary configuration. Returns (results, optimizer_instance)."""
    config = OptimizationConfig(**config_dict)
    return optimize_from_config(config)


def optimize_from_gui(panel) -> Tuple[Dict[str, Any], 'UnifiedOptimizer']:
    """GUI adapter function - converts GUI panel to config. Returns (results, optimizer_instance)."""
    
    def log_callback(message: str):
        panel._log(message)
    
    resources_panel = getattr(panel, '_resources_panel', None)

    def _coerce_int(value: Any) -> Optional[int]:
        try:
            intval = int(value)
            return intval if intval > 0 else None
        except Exception:
            return None

    def _extract_target(util_map: Any) -> Optional[int]:
        if not isinstance(util_map, dict):
            return None
        values: List[int] = []
        for raw in util_map.values():
            coerc = _coerce_int(raw)
            if coerc is not None:
                values.append(coerc)
        return max(values) if values else None

    run_device = "auto"
    train_device = "auto"
    cuda_selected: List[int] = []
    target_util: Optional[int] = None
    device = "auto"

    if resources_panel is not None:
        try:
            rvals = resources_panel.get_values()
        except Exception:
            rvals = {}

        device = str(rvals.get("train_device", "auto")).strip() or "auto"

        cuda_selected = [
            int(i) for i in (rvals.get("train_cuda_selected") or [])
            if isinstance(i, (int, str)) and str(i).isdigit()
        ]

        target_util = _extract_target(rvals.get("train_cuda_util_pct"))
        if target_util is None:
            target_util = _coerce_int(rvals.get("gpu_util_pct"))

    cuda_devices_str = ",".join(str(i) for i in cuda_selected)

    # Stop callback to check panel's stop flag
    def stop_callback() -> bool:
        return getattr(panel, '_stop_requested', False)
    
    # Get dataset file from panel
    dataset_file = "training_data/curated_datasets/test_sample.txt"  # default fallback
    if hasattr(panel, 'dataset_var'):
        try:
            user_dataset = panel.dataset_var.get().strip()
            if user_dataset:
                dataset_file = user_dataset
        except:
            pass
    
    # Extract configuration from GUI panel
    config = OptimizationConfig(
        model=getattr(panel, 'model_var', None) and panel.model_var.get() or "base_model",
        teacher_model=getattr(panel, 'teacher_var', None) and panel.teacher_var.get() or "",
        max_seq_len=int(getattr(panel, 'max_seq_var', None) and panel.max_seq_var.get() or "512"),
        dataset_file=dataset_file,
        log_callback=log_callback,
        stop_callback=stop_callback,
        # Use reasonable defaults for GUI optimization
        test_duration=45,  # Increased for real workload completion
        max_timeout=240,   # Increased for DDP init (160s) + training (30s) + buffer
        batch_sizes=[1, 2, 4, 8, 16, 32],
        min_batch_size=1,
        max_batch_size=64,
        cuda_devices=cuda_devices_str,
        use_multi_gpu=len(cuda_selected) > 1,
        device=device or "auto",
        target_util=target_util,
        util_tolerance=5,
        monitor_interval=1.0
    )
    
    return optimize_from_config(config)


def optimize_cli(
    model: str = "base_model",
    teacher: str = "",
    max_seq: int = 512,
    test_duration: int = 45,
    batch_sizes: str = "1,2,4,8,16,32",
    output_dir: str = "artifacts/optimization",
    verbose: bool = True,
    device: str = "auto",
    cuda_devices: str = "",
    use_multi_gpu: bool = True,
    strict: bool = False,
    target_util: int = 90,
    util_tolerance: int = 5,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    growth_factor: float = 2.0,
    monitor_interval: float = 1.0
) -> Tuple[Dict[str, Any], 'UnifiedOptimizer']:
    """CLI optimization function. Returns (results, optimizer_instance)."""
    
    def log_callback(message: str):
        if verbose:
            print(message)

    batch_list = [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]
    if not batch_list:
        batch_list = [1]

    config = OptimizationConfig(
        model=model,
        teacher_model=teacher or model,
        max_seq_len=max_seq,
        test_duration=test_duration,
        batch_sizes=batch_list,
        min_batch_size=min_batch_size,
        max_batch_size=max(max_batch_size, min_batch_size),
        batch_growth_factor=growth_factor,
        output_dir=output_dir,
        log_callback=log_callback,
        cuda_devices=cuda_devices,
        use_multi_gpu=use_multi_gpu,
        device=device,
        strict=strict,
        target_util=target_util,
        util_tolerance=util_tolerance,
        monitor_interval=monitor_interval
    )
    
    return optimize_from_config(config)


if __name__ == "__main__":
    # Simple CLI test
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Optimization System")
    parser.add_argument("--model", default="base_model", help="Model to optimize")
    parser.add_argument("--test-duration", type=int, default=10, help="Test duration in seconds")
    parser.add_argument("--batch-sizes", default="1,2,4", help="Comma-separated batch sizes")
    
    args = parser.parse_args()
    
    results, _ = optimize_cli(
        model=args.model,
        test_duration=args.test_duration,
        batch_sizes=args.batch_sizes
    )
    
    print(f"\n🎉 Optimization complete!")
    print(f"Generation optimal batch: {results['generation'].get('optimal_batch', 'N/A')}")
    print(f"Training optimal batch: {results['training'].get('optimal_batch', 'N/A')}")