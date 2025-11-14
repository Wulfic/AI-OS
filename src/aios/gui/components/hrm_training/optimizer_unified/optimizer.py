"""Main unified optimizer class - orchestrates the optimization process."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple

from aios.python_exec import get_preferred_python_executable
from aios.utils.optimization_cleanup import cleanup_on_startup

from .config import OptimizationConfig
from .process_manager import ProcessManager
from .command_builder import parse_cuda_devices, extend_with_device_args
from .batch_runner import run_single_batch
from .result_parser import parse_training_throughput

logger = logging.getLogger(__name__)


class UnifiedOptimizer:
    """Unified optimizer that works across all interfaces."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.session_id = str(uuid.uuid4())[:8]
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing UnifiedOptimizer: session_id={self.session_id}, model={config.model}")
        logger.debug(f"Output directory: {self.output_dir}")
        
        # Clean up old optimization runs (keep last 3)
        cleanup_on_startup(self.output_dir, keep_last_n=3)
        
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

        cuda_ids = parse_cuda_devices(self.config.cuda_devices)
        target_util = self.config.target_util if self.config.target_util else 90

        self.gpu_config = {
            "ids": cuda_ids,
            "target_util": target_util,
            "multi_gpu": self.config.use_multi_gpu and len(cuda_ids) > 1
        }
        
        logger.debug(f"GPU config: ids={cuda_ids}, target_util={target_util}, multi_gpu={self.gpu_config['multi_gpu']}")

        all_ids: List[int] = []
        for token in cuda_ids:
            try:
                all_ids.append(int(token))
            except Exception:
                continue
        self.all_cuda_ids = sorted(set(all_ids))

        self.util_tolerance = max(0, self.config.util_tolerance)
        
    def log(self, message: str):
        """Log message via callback or print."""
        timestamp = time.strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        
        if self.config.log_callback:
            self.config.log_callback(full_msg)
        else:
            logger.info(full_msg)
    
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
        logger.warning("Force stop requested - terminating all optimization processes")
        self.log("Stop requested - terminating all processes...")
        self.cleanup()
    
    def cleanup(self):
        """Clean up session files and processes."""
        logger.info(f"Cleaning up optimization session: {self.session_id}")
        try:
            self.process_manager.cleanup()
            
            # Remove session files
            for file_path in [self.stop_file, self.train_log]:
                try:
                    if file_path.exists():
                        file_path.unlink()
                        logger.debug(f"Removed session file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove session file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)
            self.log(f"Cleanup error: {e}")
    
    def optimize(self) -> Dict[str, Any]:
        """Run complete optimization process."""
        
        logger.info("="*60)
        logger.info(f"Starting Training Optimization: session={self.session_id}, model={self.config.model}")
        logger.info(f"Optimization parameters: max_seq_len={self.config.max_seq_len}, batch_sizes={self.config.batch_sizes}, test_duration={self.config.test_duration}s")
        
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
    
    def _test_training_workload(self) -> Dict[str, Any]:
        """Optimize training workload dynamically."""

        base_cmd = [
            get_preferred_python_executable(), "-m", "aios.cli.aios",
            "hrm-hf", "train-actv1",
            "--model", self.config.model,
            "--max-seq-len", str(self.config.max_seq_len),
            "--steps", str(self.config.train_steps),
            "--dataset-file", self.config.dataset_file,
            "--stop-file", str(self.stop_file),
            "--log-file", str(self.train_log),
            # Memory optimization flags
            "--amp",
            "--no-cpu-offload",
            "--gradient-checkpointing",
        ]

        # Only add teacher if it's different from the model
        if self.config.teacher_model and self.config.teacher_model != self.config.model:
            base_cmd.extend(["--teacher", self.config.teacher_model])

        extend_with_device_args(base_cmd, self.config, self.gpu_config)

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

        from .batch_optimization import run_batch_optimization
        
        return run_batch_optimization(
            base_cmd=base_cmd,
            batch_flag=batch_flag,
            metric_label=metric_label,
            throughput_parser=throughput_parser,
            log_path=log_path,
            config=self.config,
            gpu_config=self.gpu_config,
            session_id=self.session_id,
            output_dir=self.output_dir,
            stop_file=self.stop_file,
            process_manager=self.process_manager,
            all_cuda_ids=self.all_cuda_ids,
            util_tolerance=self.util_tolerance,
            log_callback=self.log,
            stop_callback=self.is_stop_requested
        )
    
    def _parse_training_throughput(self) -> float:
        """Parse training throughput from log file."""
        return parse_training_throughput(
            self.train_log,
            self.config.test_duration,
            log_callback=self.log
        )
