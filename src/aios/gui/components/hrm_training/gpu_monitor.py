"""
GPU Resource Monitor for Optimization

Provides real-time monitoring of GPU memory usage, utilization, and temperature
during optimization workloads to ensure proper multi-GPU utilization and detect issues.
"""

from __future__ import annotations

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...utils.resource_management import submit_background

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """Container for GPU metrics at a point in time."""
    gpu_id: int
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    utilization_percent: float
    temperature_c: Optional[int] = None
    power_watts: Optional[int] = None
    timestamp: float = 0.0


class GPUMonitor:
    """Monitors GPU utilization during optimization workloads."""
    
    def __init__(self, gpu_ids: List[int], log_file: Optional[str] = None, worker_pool: Any = None):
        self.gpu_ids = gpu_ids
        self.log_file = log_file
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._monitor_future = None
        self.metrics_history: List[GPUMetrics] = []
        self._lock = threading.Lock()
        self._worker_pool = worker_pool  # Store worker pool for async operations
        
    def start_monitoring(self, interval: float = 2.0) -> None:
        """Start monitoring GPU metrics in background thread."""
        if self.monitoring:
            logger.debug("GPU monitoring already active")
            return
        
        logger.info(f"Starting GPU monitoring for GPUs {self.gpu_ids} (interval={interval}s)")
        logger.debug(f"GPU monitoring configuration: log_file={self.log_file}, worker_pool={'yes' if self._worker_pool else 'no'}")
        self.monitoring = True
        if self._worker_pool:
            logger.debug("Using worker pool for GPU monitoring")
            logger.info(f"GPU monitoring using worker pool for GPUs: {', '.join(map(str, self.gpu_ids))}")
            try:
                self._monitor_future = submit_background(
                    "hrm-gpu-monitor",
                    self._monitor_loop,
                    interval,
                    pool=self._worker_pool,
                )
                self.monitor_thread = None
            except RuntimeError as exc:
                logger.error("Failed to queue GPU monitoring task: %s", exc)
                self.monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    args=(interval,),
                    daemon=True,
                )
                self.monitor_thread.start()
        else:
            # Fallback to threading
            logger.debug("Using dedicated thread for GPU monitoring")
            logger.info(f"GPU monitoring using dedicated thread for GPUs: {', '.join(map(str, self.gpu_ids))}")
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
        logger.debug("GPU monitoring started successfully")
        
    def stop_monitoring(self) -> None:
        """Stop monitoring and return final metrics summary."""
        if not self.monitoring:
            logger.debug("GPU monitoring stop called but not currently monitoring")
            return
        
        logger.info("Stopping GPU monitoring")
        self.monitoring = False
        handle = getattr(self, "_monitor_future", None)
        if handle is not None and hasattr(handle, "done"):
            logger.debug("Waiting for monitor task to terminate")
            try:
                handle.result(timeout=5.0)
            except Exception:
                pass
            self._monitor_future = None
        elif self.monitor_thread:
            logger.debug("Waiting for monitor thread to terminate")
            self.monitor_thread.join(timeout=5.0)
        
        # Log summary with detailed metrics
        if self.metrics_history:
            summary = self.get_summary()
            total_samples = len(self.metrics_history)
            logger.info(f"GPU monitoring stopped - collected {total_samples} metric samples across {len(self.gpu_ids)} GPUs")
            
            # Log per-GPU summary
            for gpu_id in self.gpu_ids:
                gpu_key = f"gpu_{gpu_id}"
                if gpu_key in summary:
                    gpu_stats = summary[gpu_key]
                    logger.debug(f"GPU {gpu_id} summary: "
                               f"mem_avg={gpu_stats.get('memory_avg', 0):.1f}%, "
                               f"mem_max={gpu_stats.get('memory_max', 0):.1f}%, "
                               f"util_avg={gpu_stats.get('utilization_avg', 0):.1f}%, "
                               f"samples={gpu_stats.get('samples', 0)}")
        else:
            logger.debug("GPU monitoring stopped - no metrics collected")
            
    def get_current_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics for all monitored GPUs."""
        metrics = []
        
        try:
            import subprocess
            
            # Use nvidia-smi to get GPU metrics
            nv_start = time.perf_counter()
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
            nv_duration = time.perf_counter() - nv_start
            if nv_duration > 1.0:
                logger.debug(f"nvidia-smi monitor query latency: {nv_duration:.3f}s")
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            try:
                                gpu_id = int(parts[0])
                                if gpu_id in self.gpu_ids:
                                    memory_used = int(parts[1]) if parts[1] != '[N/A]' else 0
                                    memory_total = int(parts[2]) if parts[2] != '[N/A]' else 1
                                    utilization = int(parts[3]) if parts[3] != '[N/A]' else 0
                                    temperature = int(parts[4]) if len(parts) > 4 and parts[4] != '[N/A]' else None
                                    power = int(float(parts[5])) if len(parts) > 5 and parts[5] != '[N/A]' else None
                                    
                                    memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                                    
                                    metrics.append(GPUMetrics(
                                        gpu_id=gpu_id,
                                        memory_used_mb=memory_used,
                                        memory_total_mb=memory_total,
                                        memory_percent=memory_percent,
                                        utilization_percent=utilization,
                                        temperature_c=temperature,
                                        power_watts=power,
                                        timestamp=time.time()
                                    ))
                            except (ValueError, IndexError):
                                continue
                                
        except Exception:
            # Fallback to basic monitoring if nvidia-smi fails
            pass
            
        return metrics
        
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop running in background thread."""
        while self.monitoring:
            try:
                metrics = self.get_current_metrics()
                
                with self._lock:
                    self.metrics_history.extend(metrics)
                    
                    # Keep only last 1000 measurements to prevent memory bloat
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                        
                # Log metrics if file specified
                if self.log_file and metrics:
                    self._log_metrics(metrics)
                    
            except Exception:
                pass
                
            time.sleep(interval)
            
    def _log_metrics(self, metrics: List[GPUMetrics]) -> None:
        """Log metrics to file in JSON format."""
        if not self.log_file:
            return
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for metric in metrics:
                    data = {
                        "event": "gpu_metrics",
                        "gpu_id": metric.gpu_id,
                        "memory_used_mb": metric.memory_used_mb,
                        "memory_total_mb": metric.memory_total_mb,
                        "memory_percent": metric.memory_percent,
                        "utilization_percent": metric.utilization_percent,
                        "temperature_c": metric.temperature_c,
                        "power_watts": metric.power_watts,
                        "timestamp": metric.timestamp
                    }
                    f.write(json.dumps(data) + '\n')
                    f.flush()
        except Exception:
            pass
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for monitored period."""
        with self._lock:
            if not self.metrics_history:
                return {"error": "No metrics collected"}
                
            summary = {}
            
            for gpu_id in self.gpu_ids:
                gpu_metrics = [m for m in self.metrics_history if m.gpu_id == gpu_id]
                
                if gpu_metrics:
                    memory_usage = [m.memory_percent for m in gpu_metrics]
                    utilization = [m.utilization_percent for m in gpu_metrics]
                    temperatures = [m.temperature_c for m in gpu_metrics if m.temperature_c is not None]
                    
                    summary[f"gpu_{gpu_id}"] = {
                        "memory_avg": sum(memory_usage) / len(memory_usage),
                        "memory_max": max(memory_usage),
                        "memory_min": min(memory_usage),
                        "utilization_avg": sum(utilization) / len(utilization),
                        "utilization_max": max(utilization),
                        "utilization_min": min(utilization),
                        "temperature_avg": sum(temperatures) / len(temperatures) if temperatures else None,
                        "temperature_max": max(temperatures) if temperatures else None,
                        "samples": len(gpu_metrics)
                    }
                    
            return summary
            
    def verify_multi_gpu_usage(self) -> Dict[str, Any]:
        """Verify that multiple GPUs are actually being utilized."""
        summary = self.get_summary()
        
        if "error" in summary:
            return {"multi_gpu_verified": False, "reason": "No metrics available"}
            
        active_gpus = 0
        gpu_details = {}
        
        for gpu_key, metrics in summary.items():
            if gpu_key.startswith("gpu_"):
                gpu_id = gpu_key.split("_")[1]
                avg_util = metrics.get("utilization_avg", 0)
                max_util = metrics.get("utilization_max", 0)
                
                gpu_details[gpu_id] = {
                    "avg_utilization": avg_util,
                    "max_utilization": max_util,
                    "is_active": max_util > 5  # Consider >5% utilization as active
                }
                
                if max_util > 5:
                    active_gpus += 1
                    
        multi_gpu_verified = active_gpus > 1
        
        return {
            "multi_gpu_verified": multi_gpu_verified,
            "active_gpus": active_gpus,
            "total_monitored": len(self.gpu_ids),
            "gpu_details": gpu_details,
            "reason": "Multiple GPUs showing utilization" if multi_gpu_verified else "Only single GPU active"
        }


def create_gpu_monitor(gpu_ids: List[int], log_file: Optional[str] = None) -> GPUMonitor:
    """Factory function to create a GPU monitor instance."""
    return GPUMonitor(gpu_ids, log_file)