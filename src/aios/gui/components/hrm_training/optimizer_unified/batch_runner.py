"""Single batch test execution and result collection."""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OptimizationConfig
    from .process_manager import ProcessManager

from .command_builder import build_command_env, extend_with_device_args
from .gpu_monitoring import create_gpu_monitor_safe, stop_gpu_monitor, extract_utilization, extract_memory
from .result_parser import parse_training_throughput, detect_oom


def run_single_batch(
    *,
    base_cmd: List[str],
    batch_flag: str,
    batch_size: int,
    config: "OptimizationConfig",
    gpu_config: Dict[str, Any],
    session_id: str,
    output_dir: Path,
    stop_file: Path,
    log_path: Path,
    process_manager: "ProcessManager",
    throughput_parser: Callable[[], float],
    metric_label: str,
    all_cuda_ids: List[int],
    log_callback: Optional[Callable[[str], None]] = None,
    stop_callback: Optional[Callable[[], bool]] = None
) -> Dict[str, Any]:
    """Execute a single batch test run and collect metrics.
    
    Args:
        base_cmd: Base command to execute
        batch_flag: CLI flag for batch size (e.g., "--batch-size")
        batch_size: Batch size to test
        config: Optimization configuration
        gpu_config: GPU configuration dict
        session_id: Session identifier
        output_dir: Output directory for logs
        stop_file: Path to stop flag file
        log_path: Path to training log file
        process_manager: Process manager instance
        throughput_parser: Function to parse throughput from logs
        metric_label: Label for metrics (e.g., "steps/sec")
        all_cuda_ids: All CUDA device IDs
        log_callback: Optional logging callback
        stop_callback: Optional stop check callback
        
    Returns:
        Dictionary with batch test results
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)
    
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

    env = build_command_env(config, gpu_config, session_id)
    
    # Log the FULL command being executed (critical for debugging)
    cmd_str = " ".join(test_cmd)
    log(f"Testing batch {batch_size}...")
    log(f"Command: {cmd_str}")
    
    # Log environment variables
    if env.get("CUDA_VISIBLE_DEVICES"):
        log(f"  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    if env.get("AIOS_WORLD_SIZE"):
        log(f"  AIOS_WORLD_SIZE={env['AIOS_WORLD_SIZE']}")
    if env.get("AIOS_DDP_SPAWN"):
        log(f"  AIOS_DDP_SPAWN={env['AIOS_DDP_SPAWN']}")

    # Clean old log but keep it for parsing after run
    if log_path.exists():
        try:
            log_path.unlink()
        except Exception:
            pass

    # Prepare for batch (cleanup stop files)
    _prepare_for_batch(stop_file, config)

    # Start GPU monitoring
    monitor = create_gpu_monitor_safe(
        all_cuda_ids,
        output_dir / f"gpu_metrics_{session_id}.jsonl",
        config.monitor_interval
    )
    
    # Create timer to stop test after duration
    timer = threading.Timer(config.test_duration, lambda: _create_stop_file(stop_file))
    timer.start()

    start_time = time.perf_counter()
    try:
        result = process_manager.run_command(
            test_cmd, 
            env=env, 
            heartbeat_file=log_path,
            log_callback=log,
            stop_callback=stop_callback
        )
    finally:
        timer.cancel()

    duration = max(0.001, time.perf_counter() - start_time)
    summary = stop_gpu_monitor(monitor)
    
    # Parse throughput BEFORE cleanup (log file needs to exist)
    throughput = throughput_parser()
    
    # Log detailed error info if process failed or throughput is 0
    if result.returncode != 0 or throughput == 0.0:
        log(f"⚠️  Process exit code: {result.returncode}, throughput: {throughput}")
        
        # Show stderr (most important for errors)
        if result.stderr and result.stderr.strip():
            stderr_lines = result.stderr.strip().split('\n')
            log("STDERR output:")
            for line in stderr_lines[-10:]:  # Show last 10 lines
                log(f"  {line}")
        
        # Show stdout if stderr is empty
        if (not result.stderr or not result.stderr.strip()) and result.stdout and result.stdout.strip():
            stdout_lines = result.stdout.strip().split('\n')
            log("STDOUT output:")
            for line in stdout_lines[-10:]:  # Show last 10 lines
                log(f"  {line}")
        
        # Check if log file was created
        if not log_path.exists():
            log(f"⚠️  Log file was never created: {log_path}")
            log("  This usually means the training command failed during initialization")
        elif log_path.stat().st_size == 0:
            log(f"⚠️  Log file is empty: {log_path}")
            log("  Training started but didn't complete any steps")
    
    # Now cleanup
    _cleanup_after_batch(stop_file)
    
    utilization = extract_utilization(summary)
    memory_pct = extract_memory(summary)
    oom = detect_oom(result)
    
    # Success if: no OOM AND (clean exit OR throughput > 0)
    # Exit code -1 with throughput > 0 means training completed but hit timeout
    success = not oom and (result.returncode == 0 or throughput > 0)

    # Clear status line
    status = "✓ SUCCESS" if success else ("💥 OOM" if oom else "❌ FAILED")
    log(
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


def _prepare_for_batch(stop_file: Path, config: "OptimizationConfig") -> None:
    """Prepare for batch execution by cleaning stop files.
    
    Args:
        stop_file: Path to stop flag file
        config: Optimization configuration
    """
    try:
        if stop_file.exists():
            stop_file.unlink()
    except Exception:
        pass
    
    global_stop_paths = [
        Path("training_data/actv1/STOP"),
        Path("training_data/actv1/stop"),
        Path("training_data/actv1/Stop"),
        Path("training_data/actv1/GRACEFUL_STOP"),  # Also clean graceful stop file
    ]
    
    for path in global_stop_paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            continue


def _cleanup_after_batch(stop_file: Path) -> None:
    """Cleanup after batch execution.
    
    Args:
        stop_file: Path to stop flag file
    """
    try:
        if stop_file.exists():
            stop_file.unlink()
    except Exception:
        pass


def _create_stop_file(stop_file: Path):
    """Create stop file to signal test completion.
    
    Args:
        stop_file: Path to stop flag file
    """
    try:
        stop_file.touch()
    except:
        pass
