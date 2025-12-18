"""Single batch test execution and result parsing."""

from __future__ import annotations

import os
import json
import time
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import OptimizationLevel, OptimizationConfig, BatchTestResult


def test_single_batch(
    level: "OptimizationLevel",
    batch_size: int,
    config: "OptimizationConfig",
    stop_file: Path,
    train_log: Path,
    output_dir: Path,
    log_func=None
) -> "BatchTestResult":
    """Test training with a specific optimization level and batch size.
    
    Args:
        level: Optimization level to test
        batch_size: Batch size to test
        config: Optimization configuration
        stop_file: Path to stop flag file
        train_log: Path to training log file
        output_dir: Output directory for GPU metrics
        log_func: Optional logging function
        
    Returns:
        BatchTestResult with success, OOM detection, memory usage, etc.
    """
    from aios.python_exec import get_preferred_python_executable
    from .gpu_monitor import create_gpu_monitor
    from .models import BatchTestResult
    
    def log(msg: str):
        if log_func:
            log_func(msg)
    
    # Build command
    cmd = [
        get_preferred_python_executable(), "-m", "aios.cli.aios",
        "hrm-hf", "train-actv1",
        "--model", config.model,
        "--max-seq-len", str(config.max_seq_len),
        "--steps", str(config.train_steps),
        "--batch-size", str(batch_size),
        "--dataset-file", config.dataset_file,
        "--stop-file", str(stop_file),
        "--log-file", str(train_log),
    ]
    
    # Add optimization-specific args
    cmd.extend(level.to_cli_args())
    
    # ========== CRITICAL: Add all flags that real training will use ==========
    # This ensures optimizer tests match real training conditions
    
    # Device configuration
    if config.cuda_devices:
        cmd.extend(["--cuda-ids", config.cuda_devices])
    if config.device:
        cmd.extend(["--device", config.device])
    
    # DDP configuration (CRITICAL: Must match real training!)
    if config.use_multi_gpu and config.cuda_devices:
        num_gpus = len([x for x in config.cuda_devices.split(",") if x.strip()])
        if num_gpus > 1:
            cmd.append("--ddp")
            cmd.extend(["--world-size", str(num_gpus)])
    
    # 8-bit optimizer (commonly enabled)
    cmd.append("--use-8bit-optimizer")
    
    # NOTE: Auto-chunking for OPTIMIZER TESTING ONLY (when max_seq_len > 8192)
    # This helps the optimizer find settings that work. During actual training,
    # the user's explicit settings are respected (no auto-enforcement).
    if config.max_seq_len > 8192:
        cmd.append("--use-chunked-training")
        # Use chunk size from level if specified, otherwise default to 2048
        chunk_size = level.chunk_size if level.chunk_size is not None else 2048
        cmd.extend(["--chunk-size", str(chunk_size)])
    
    # Window size for Flash Attention
    if level.flashattn2 or config.max_seq_len > 2048:
        cmd.extend(["--window-size", "512"])
    
    # ========== Log the exact command ==========
    cmd_str = " ".join(cmd)
    log(f"      🔧 Command: {cmd_str[:200]}...")  # Show first 200 chars
    
    # Setup environment
    env = os.environ.copy()
    if config.cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = config.cuda_devices
    
    # Clean old log file
    if train_log.exists():
        try:
            train_log.unlink()
        except:
            pass
    
    # Setup stop file cleanup
    if stop_file.exists():
        try:
            stop_file.unlink()
        except:
            pass
    
    # Also clean graceful stop file
    try:
        graceful_stop = Path("training_datasets/actv1/GRACEFUL_STOP")
        if graceful_stop.exists():
            graceful_stop.unlink()
    except:
        pass
    
    # Start GPU monitoring
    monitor = None
    try:
        if config.cuda_devices:
            device_ids = [int(x.strip()) for x in config.cuda_devices.split(",") if x.strip().isdigit()]
            if device_ids:
                log_file = output_dir / f"gpu_metrics_{config.session_id}.jsonl"
                monitor = create_gpu_monitor(device_ids, str(log_file))
                monitor.start_monitoring(interval=1.0)
    except:
        pass
    
    # Create stop timer
    timer = threading.Timer(config.test_duration, lambda: stop_file.touch())
    timer.start()
    
    # Run training
    start_time = time.perf_counter()
    try:
        process = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=config.max_timeout
        )
    except subprocess.TimeoutExpired:
        # Timeout - kill it
        process = subprocess.CompletedProcess(
            cmd, -1, "", "Process killed after timeout"
        )
    finally:
        timer.cancel()
    
    duration = time.perf_counter() - start_time
    
    # Stop GPU monitoring and get stats
    memory_percent = 0.0
    if monitor:
        try:
            monitor.stop_monitoring()
            summary = monitor.get_summary() or {}
            # Extract max memory usage
            for device_metrics in summary.values():
                if isinstance(device_metrics, dict):
                    mem = device_metrics.get("memory_max", 0.0)
                    if mem > memory_percent:
                        memory_percent = mem
        except:
            pass
    
    # Parse results
    throughput = parse_throughput(train_log, config.test_duration, log_func)
    is_oom = detect_oom(process)
    
    # Success requires: no OOM AND throughput > 0 (actual training steps completed)
    # If throughput is 0, training didn't complete any steps = failure
    success = throughput > 0 and not is_oom
    
    # Log stderr/stdout for debugging if throughput is 0
    if throughput == 0:
        log(f"      ⚠️  Zero throughput detected - checking process output...")
        if process.stderr:
            stderr_lines = process.stderr.strip().split('\n')
            log(f"      STDERR (last 5 lines):")
            for line in stderr_lines[-5:]:
                log(f"        {line[:150]}")
        if process.stdout:
            stdout_lines = process.stdout.strip().split('\n')
            log(f"      STDOUT (last 5 lines):")
            for line in stdout_lines[-5:]:
                log(f"        {line[:150]}")
    
    error_message = ""
    if not success:
        if is_oom:
            error_message = "Out of memory"
        elif throughput == 0:
            error_message = "Training completed 0 steps - no progress made"
        else:
            error_message = (process.stderr or "Unknown error")[:200]
    
    # Cleanup
    if stop_file.exists():
        try:
            stop_file.unlink()
        except:
            pass
    
    return BatchTestResult(
        batch_size=batch_size,
        success=success,
        is_oom=is_oom,
        memory_percent=memory_percent,
        throughput=throughput,
        duration=duration,
        exit_code=process.returncode,
        error_message=error_message
    )


def parse_throughput(train_log: Path, test_duration: int, log_func=None) -> float:
    """Parse training throughput from log file.
    
    Args:
        train_log: Path to training log file
        test_duration: Test duration for fallback calculation
        log_func: Optional logging function
        
    Returns:
        Throughput in steps/second
    """
    def log(msg: str):
        if log_func:
            log_func(msg)
    
    try:
        if not train_log.exists():
            log(f"      ⚠️  Log file doesn't exist: {train_log}")
            return 0.0
        
        file_size = train_log.stat().st_size
        if file_size == 0:
            log(f"      ⚠️  Log file is empty (0 bytes)")
            return 0.0
        
        log(f"      📊 Parsing log file ({file_size} bytes)...")
        
        steps = 0
        first_ts = None
        last_ts = None
        line_count = 0
        
        with open(train_log, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                try:
                    data = json.loads(line.strip())
                    event = data.get("event")
                    
                    if event == "train":
                        steps += 1
                        ts = data.get("ts")
                        if ts:
                            if first_ts is None:
                                first_ts = ts
                            last_ts = ts
                except json.JSONDecodeError:
                    continue
        
        log(f"      📊 Found {steps} training steps in {line_count} log lines")
        
        if steps > 0 and first_ts and last_ts and last_ts > first_ts:
            duration = last_ts - first_ts
            throughput = steps / max(duration, 0.1)
            log(f"      📊 Calculated throughput: {throughput:.2f} steps/sec")
            return throughput
        elif steps > 0:
            # Fallback: use test duration
            throughput = steps / max(test_duration, 1)
            log(f"      📊 Using fallback throughput: {throughput:.2f} steps/sec")
            return throughput
        else:
            log(f"      ⚠️  No training steps found in log file")
        
        return 0.0
    except Exception as e:
        log(f"      ❌ Error parsing log: {e}")
        return 0.0


def detect_oom(process: subprocess.CompletedProcess) -> bool:
    """Detect if process failed due to OOM.
    
    Args:
        process: Completed subprocess
        
    Returns:
        True if OOM detected
    """
    stderr = (process.stderr or "").lower()
    stdout = (process.stdout or "").lower()
    
    oom_patterns = [
        "out of memory",
        "cuda oom",
        "cuda error",
        "cudnn error",
        "allocation",
        "cuda runtime error",
        "cannot allocate",
        "memory error",
    ]
    
    for pattern in oom_patterns:
        if pattern in stderr or pattern in stdout:
            return True
    
    return False
