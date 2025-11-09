"""Adaptive batch size optimization logic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OptimizationConfig
    from .process_manager import ProcessManager

from .batch_runner import run_single_batch


def run_batch_optimization(
    *,
    base_cmd: List[str],
    batch_flag: str,
    metric_label: str,
    throughput_parser: Callable[[], float],
    log_path: Path,
    config: "OptimizationConfig",
    gpu_config: Dict[str, Any],
    session_id: str,
    output_dir: Path,
    stop_file: Path,
    process_manager: "ProcessManager",
    all_cuda_ids: List[int],
    util_tolerance: int,
    log_callback: Callable[[str], None],
    stop_callback: Callable[[], bool]
) -> Dict[str, Any]:
    """Run adaptive batch size optimization.
    
    Returns dictionary with success, optimal_batch, max_throughput, and results list.
    """
    results: Dict[str, Any] = {
        "success": False,
        "optimal_batch": config.min_batch_size,
        "max_throughput": 0.0,
        "results": []
    }

    available = config.batch_sizes or [1, 2, 4, 8]
    min_batch = max(config.min_batch_size, available[0])
    max_batch = max(config.max_batch_size or available[-1], min_batch)

    visited: Dict[int, Dict[str, Any]] = {}
    success_batches: List[int] = []
    failed_batches: List[int] = []

    current_batch = min_batch

    oom_count = 0
    last_successful_batch = min_batch
    
    while current_batch <= max_batch and current_batch not in visited:
        # Check for stop request
        if stop_callback():
            log_callback("Stop requested during optimization")
            break
        
        batch_data = run_single_batch(
            base_cmd=base_cmd,
            batch_flag=batch_flag,
            batch_size=current_batch,
            config=config,
            gpu_config=gpu_config,
            session_id=session_id,
            output_dir=output_dir,
            stop_file=stop_file,
            log_path=log_path,
            process_manager=process_manager,
            throughput_parser=throughput_parser,
            metric_label=metric_label,
            all_cuda_ids=all_cuda_ids,
            log_callback=log_callback,
            stop_callback=stop_callback
        )

        visited[current_batch] = batch_data
        results["results"].append(batch_data)

        if batch_data["success"]:
            success_batches.append(current_batch)
            last_successful_batch = current_batch
            oom_count = 0
            
            if batch_data["throughput"] > results["max_throughput"]:
                results["max_throughput"] = batch_data["throughput"]
                results["optimal_batch"] = current_batch

            # Smart growth based on memory usage
            mem_pct = batch_data.get("memory_percent", 0.0)
            if mem_pct > 90.0:
                log_callback(f"Memory at {mem_pct:.1f}% - near limit, slowing growth")
                next_batch = current_batch + max(1, current_batch // 4)
            elif mem_pct > 75.0:
                next_batch = _next_batch_size(current_batch, available, max_batch, config.batch_growth_factor)
            else:
                next_batch = _next_batch_size(current_batch, available, max_batch, config.batch_growth_factor)
                if next_batch == current_batch * 2 and current_batch < 32:
                    next_batch = min(current_batch * 4, max_batch)
            
            if next_batch <= current_batch:
                break
            current_batch = next_batch
        else:
            failed_batches.append(current_batch)
            
            if batch_data.get("oom", False):
                oom_count += 1
                log_callback(f"OOM at batch {current_batch} (OOM #{oom_count})")
                
                # Smart backoff
                if oom_count == 1 and current_batch > min_batch:
                    refined_batch = int(current_batch * 0.75)
                    if refined_batch > last_successful_batch and refined_batch not in visited:
                        log_callback(f"First OOM, trying 75% size: {refined_batch}")
                        current_batch = refined_batch
                        continue
                
                if current_batch > min_batch:
                    log_callback(f"Starting binary search between {last_successful_batch} and {current_batch}")
                    break
                else:
                    break
            else:
                log_callback(f"Non-OOM error at batch {current_batch}, stopping growth")
                break

    # Binary search refinement
    if failed_batches and success_batches:
        lower = max(success_batches)
        upper = failed_batches[0]
        while upper - lower > 1:
            mid = (lower + upper) // 2
            if mid in visited or mid <= 0:
                break

            batch_data = run_single_batch(
                base_cmd=base_cmd,
                batch_flag=batch_flag,
                batch_size=mid,
                config=config,
                gpu_config=gpu_config,
                session_id=session_id,
                output_dir=output_dir,
                stop_file=stop_file,
                log_path=log_path,
                process_manager=process_manager,
                throughput_parser=throughput_parser,
                metric_label=metric_label,
                all_cuda_ids=all_cuda_ids,
                log_callback=log_callback,
                stop_callback=stop_callback
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
        selected = _select_optimal_batch(
            success_batches,
            visited,
            gpu_config.get("target_util"),
            util_tolerance
        )
        if selected:
            results["success"] = True
            results["optimal_batch"] = selected["batch_size"]
            results["max_throughput"] = selected["throughput"]
            results["target_util"] = gpu_config.get("target_util")
            results["selected"] = selected

    return results


def _next_batch_size(current: int, available: List[int], max_batch: int, growth_factor: float) -> int:
    """Determine the next batch size candidate."""
    for candidate in available:
        if candidate > current:
            return min(candidate, max_batch)

    # Exponential growth
    if current < 16:
        next_size = current * 2
    elif current < 64:
        next_size = current * 2
    else:
        next_size = int(math.ceil(current * growth_factor))
    
    if next_size <= current:
        next_size = current + 1

    return min(next_size, max_batch)


def _select_optimal_batch(
    success_batches: List[int],
    data_map: Dict[int, Dict[str, Any]],
    target_util: Optional[int],
    tolerance: int
) -> Optional[Dict[str, Any]]:
    """Choose the batch result that best matches target utilization and memory usage."""
    if not success_batches:
        return None

    entries = [data_map[b] for b in success_batches if b in data_map]
    if not entries:
        return None

    def score_batch(e: Dict[str, Any]) -> Tuple[float, float, float, int]:
        """Score batch: (memory_score, throughput, utilization, -batch_size)"""
        mem_pct = e.get("memory_percent", 0.0)
        throughput = e.get("throughput", 0.0)
        util = e.get("utilization", 0.0)
        batch = e.get("batch_size", 0)
        
        # Memory score: prefer 80-95% range
        if 80.0 <= mem_pct <= 95.0:
            memory_score = 1.0
        elif mem_pct > 95.0:
            memory_score = 0.7
        elif mem_pct > 60.0:
            memory_score = 0.5 + (mem_pct - 60.0) / 40.0
        else:
            memory_score = mem_pct / 60.0
        
        return (memory_score, throughput, util, -batch)

    if target_util is None:
        return max(entries, key=score_batch)

    within = [
        e for e in entries
        if abs(e.get("utilization", 0.0) - target_util) <= tolerance
    ]
    if within:
        return max(within, key=score_batch)

    max_util = max(e.get("utilization", 0.0) for e in entries)
    if max_util < target_util - tolerance:
        return max(entries, key=score_batch)

    below = [e for e in entries if e.get("utilization", 0.0) < target_util]
    if below:
        return max(below, key=score_batch)

    above = [e for e in entries if e.get("utilization", 0.0) > target_util]
    if above:
        return min(above, key=lambda e: (e.get("utilization", 0.0), -e.get("throughput", 0.0)))

    return entries[-1]
