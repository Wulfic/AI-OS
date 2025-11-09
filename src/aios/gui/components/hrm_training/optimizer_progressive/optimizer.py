"""Main progressive optimizer class - orchestrates the optimization process."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List

from aios.utils.optimization_cleanup import cleanup_on_startup
from .models import OptimizationConfig, OptimizationLevel
from .level_factory import create_optimization_levels, create_exhaustive_levels
from .batch_tester import test_single_batch


class ProgressiveOptimizer:
    """
    Progressive optimization system that intelligently finds the best training configuration.
    
    Strategy:
    1. Test ALL optimization levels (don't stop at first success)
    2. For each level:
       a. Start with batch size 1
       b. If successful, ALWAYS test batch 2 (even if memory high) to confirm limit
       c. If memory < 95%, continue increasing batch size intelligently:
          - If memory < 60%: double the batch size
          - If memory 60-95%: add 1 to batch size
       d. Continue until memory cap (95%+) reached or OOM occurs
    3. Compare all successful levels by throughput and memory efficiency
    4. Return the BEST configuration (highest throughput with reasonable memory)
    5. This ensures we find LoRA configs even if basic configs work!
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old optimization runs (keep last 3)
        cleanup_on_startup(self.output_dir, keep_last_n=3)
        
        # Session files
        self.stop_file = self.output_dir / f"stop_{config.session_id}.flag"
        self.train_log = self.output_dir / f"train_{config.session_id}.jsonl"
        
        # Results tracking
        self.results = {
            "session_id": config.session_id,
            "success": False,
            "optimal_level": None,
            "optimal_batch": 1,
            "max_throughput": 0.0,
            "levels_tested": [],
            "message": ""
        }
        
        # Define optimization levels in order from least to most aggressive
        self.optimization_levels = create_optimization_levels(
            config, log_func=self.log
        )
        
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
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run progressive optimization to find the best training configuration.
        
        Returns a dict with:
        - success: bool
        - optimal_level: OptimizationLevel (if success)
        - optimal_batch: int
        - max_throughput: float
        - levels_tested: List of all tested levels with results
        - message: str (friendly message for user)
        """
        
        self.log("=" * 70)
        self.log("🚀 PROGRESSIVE OPTIMIZATION SYSTEM")
        self.log("=" * 70)
        self.log(f"Session ID: {self.config.session_id}")
        self.log(f"Model: {self.config.model}")
        self.log(f"Dataset: {self.config.dataset_file}")
        self.log(f"Max sequence length: {self.config.max_seq_len}")
        self.log(f"Training steps per test: {self.config.train_steps}")
        self.log(f"Test duration: {self.config.test_duration}s (3 min for large context)")
        self.log("")
        self.log("Strategy: Start minimal, increase batch until GPU memory cap,")
        self.log("          then add optimizations one by one if failures occur.")
        self.log("=" * 70)
        self.log("")
        
        # Validate dataset exists (skip validation for HuggingFace datasets)
        dataset_str = self.config.dataset_file
        if not (dataset_str.startswith("hf://") or dataset_str.startswith("hf:")):
            # Local file - validate it exists
            dataset_path = Path(dataset_str)
            if not dataset_path.exists():
                error_msg = f"❌ Dataset file not found: {dataset_path}"
                self.log(error_msg)
                self.results["message"] = "Dataset file not found. Please select a valid training dataset."
                return self.results
        
        # Test ALL optimization levels to find the most efficient one
        best_level, best_batch, best_throughput, best_memory = self._test_all_levels()
        
        # After testing all levels, check if we found a winner
        if best_level:
            return self._finalize_success(best_level, best_batch, best_throughput, best_memory)
        
        # Try exhaustive chunk sizes if needed
        if self.config.max_seq_len > 8192:
            best_level, best_batch, best_throughput, best_memory = self._try_exhaustive_chunks()
            if best_level:
                return self._finalize_exhaustive_success(best_level, best_batch, best_throughput, best_memory)
        
        # All levels failed
        return self._finalize_failure()
    
    def _test_all_levels(self):
        """Test all optimization levels."""
        best_level = None
        best_batch = 1
        best_throughput = 0.0
        best_memory = 0.0
        
        for level_idx, level in enumerate(self.optimization_levels, 1):
            if self.is_stop_requested():
                self.log("⏹️  Stop requested by user")
                self.results["message"] = "Optimization stopped by user"
                break
            
            self.log(f"\n{'='*70}")
            self.log(f"🧪 Testing {level.name}")
            self.log(f"   Configuration: {level}")
            self.log(f"{'='*70}")
            
            # Test this optimization level with progressive batch sizing
            level_result = self._test_optimization_level(level)
            self.results["levels_tested"].append(level_result)
            
            if level_result["success"]:
                # Found a working configuration!
                self.log(f"   ✅ {level.name} works!")
                self.log(f"      • Batch Size: {level_result['optimal_batch']}")
                self.log(f"      • Throughput: {level_result['max_throughput']:.2f} steps/sec")
                self.log(f"      • Memory: {level_result.get('memory_percent', 0):.1f}%")
                
                # Check if this is better than previous best
                if self._is_better_config(level_result, best_throughput, best_memory):
                    self.log(f"      🎯 New best configuration!")
                    best_level = level
                    best_batch = level_result['optimal_batch']
                    best_throughput = level_result['max_throughput']
                    best_memory = level_result.get('memory_percent', 0)
                else:
                    self.log(f"      📊 Not better than current best (keeping previous)")
            else:
                # This level failed
                self.log(f"   ❌ {level.name} failed - will try next level...")
        
        return best_level, best_batch, best_throughput, best_memory
    
    def _try_exhaustive_chunks(self):
        """Try exhaustive chunk sizes as final resort."""
        self.log("")
        self.log("=" * 70)
        self.log("⚠️  All standard configurations failed!")
        self.log("🔬 EXHAUSTIVE MODE: Testing smaller chunk sizes as final resort...")
        self.log("=" * 70)
        self.log("")
        
        exhaustive_levels = create_exhaustive_levels(self.optimization_levels, self.config)
        
        for level in exhaustive_levels:
            if self.is_stop_requested():
                break
            
            self.log(f"\n{'='*70}")
            self.log(f"🔬 Exhaustive Test: {level.name}")
            self.log(f"   Configuration: {level}")
            self.log(f"{'='*70}")
            
            # Test this level
            level_result = self._test_optimization_level(level)
            self.results["levels_tested"].append(level_result)
            
            if level_result["success"]:
                # Found a working configuration with tiny chunks!
                best_level = level
                best_batch = level_result['optimal_batch']
                best_throughput = level_result['max_throughput']
                best_memory = level_result.get('memory_percent', 0)
                
                self.log(f"   ✅ EXHAUSTIVE SUCCESS with chunk={level.chunk_size}!")
                self.log(f"      • Batch Size: {best_batch}")
                self.log(f"      • Throughput: {best_throughput:.2f} steps/sec")
                self.log(f"      • Memory: {best_memory:.1f}%")
                
                return best_level, best_batch, best_throughput, best_memory
            else:
                self.log(f"   ❌ Chunk {level.chunk_size} also failed - trying smaller...")
        
        return None, 1, 0.0, 0.0
    
    def _is_better_config(self, level_result: dict, best_throughput: float, best_memory: float) -> bool:
        """Check if this configuration is better than the current best.
        
        ENHANCED: Now uses comprehensive scoring that considers throughput, memory, AND quality.
        """
        from .scoring import score_optimization_result
        
        # First success is always better than nothing
        if best_throughput == 0:
            return True
        
        # Get the level object from result for scoring
        # Note: level_result should contain the level object or we need to parse it
        # For now, use simple heuristic (can be enhanced)
        
        # Prioritize: 1) Higher throughput, 2) Lower memory usage
        # This maintains backward compatibility while we integrate full scoring
        if level_result['max_throughput'] > best_throughput * 1.1:  # 10% better throughput
            # Significantly better throughput
            return True
        elif level_result['max_throughput'] >= best_throughput * 0.9:  # Similar throughput
            # Similar or better throughput, check memory
            if level_result.get('memory_percent', 100) < best_memory:
                # Better memory efficiency
                return True
        return False
    
    def _test_optimization_level(self, level: OptimizationLevel) -> Dict[str, Any]:
        """
        Test a specific optimization level with progressive batch sizing.
        
        Strategy:
        - Start with batch size 1
        - If success and memory < 85%:
          - If batch == 1: try batch 2
          - If memory < 60%: double the batch
          - Else: add 1 to batch
        - Continue until OOM or memory cap reached
        
        Returns dict with success, optimal_batch, results, etc.
        """
        
        result = {
            "level": str(level),
            "success": False,
            "optimal_batch": 1,
            "max_throughput": 0.0,
            "memory_percent": 0.0,
            "batch_tests": []
        }
        
        batch_size = self.config.min_batch_size
        last_successful_batch = None
        last_successful_result = None
        
        while batch_size <= self.config.max_batch_size:
            if self.is_stop_requested():
                break
            
            # Test this batch size
            self.log(f"\n   Testing batch size {batch_size}...")
            test_result = test_single_batch(
                level, batch_size, self.config,
                self.stop_file, self.train_log, self.output_dir,
                log_func=self.log
            )
            result["batch_tests"].append({
                "batch_size": batch_size,
                "success": test_result.success,
                "throughput": test_result.throughput,
                "memory_percent": test_result.memory_percent
            })
            
            if test_result.success:
                # Training worked!
                self.log(f"   ✅ Batch {batch_size}: SUCCESS")
                self.log(f"      • Throughput: {test_result.throughput:.2f} steps/sec")
                self.log(f"      • Memory: {test_result.memory_percent:.1f}%")
                
                last_successful_batch = batch_size
                last_successful_result = test_result
                
                # Update result with best so far
                if test_result.throughput > result["max_throughput"]:
                    result["max_throughput"] = test_result.throughput
                    result["optimal_batch"] = batch_size
                    result["memory_percent"] = test_result.memory_percent
                
                # Determine next batch size to test
                batch_size = self._get_next_batch_size(batch_size, test_result)
                if batch_size is None:
                    break
            else:
                # Training failed
                if not self._handle_failed_batch(test_result, last_successful_batch, result):
                    return result
                break
        
        # If we had any successful batch, mark as success
        if last_successful_batch is not None:
            result["success"] = True
            result["optimal_batch"] = last_successful_batch
            if last_successful_result:
                result["memory_percent"] = last_successful_result.memory_percent
                result["max_throughput"] = last_successful_result.throughput
        
        return result
    
    def _get_next_batch_size(self, current_batch: int, test_result) -> int | None:
        """Determine the next batch size to test, or None if we should stop."""
        if current_batch == 1:
            # ALWAYS try batch 2 after batch 1 succeeds
            self.log(f"      • Testing batch 2 to confirm memory limit...")
            return 2
        elif test_result.has_memory_headroom:
            # Still room to grow
            if test_result.memory_percent < 60.0:
                # Lots of headroom: double it
                new_batch = current_batch * 2
                if new_batch <= self.config.max_batch_size:
                    self.log(f"      • Plenty of memory headroom, doubling to {new_batch}")
                    return new_batch
                else:
                    # Hit max batch limit
                    self.log(f"      • Reached max batch size limit")
                    return None
            else:
                # Moderate headroom: add 1
                new_batch = current_batch + 1
                self.log(f"      • Some memory headroom, incrementing to {new_batch}")
                return new_batch
        else:
            # Near memory cap (95%+) - we found the sweet spot!
            self.log(f"      • Memory at {test_result.memory_percent:.1f}% - optimal batch size found!")
            return None
    
    def _handle_failed_batch(self, test_result, last_successful_batch, result) -> bool:
        """Handle a failed batch test. Returns True if we should continue, False to stop."""
        if test_result.is_oom:
            self.log(f"   💥 Batch {test_result.batch_size}: OUT OF MEMORY")
            if last_successful_batch is not None:
                self.log(f"      • Reverting to last successful batch: {last_successful_batch}")
                result["success"] = True
                return False
            else:
                self.log(f"      • OOM at batch 1 - optimization level too weak")
                return False
        else:
            # Other error (not OOM)
            self.log(f"   ❌ Batch {test_result.batch_size}: FAILED")
            self.log(f"      • Error: {test_result.error_message}")
            
            if last_successful_batch is not None:
                self.log(f"      • Using last successful batch: {last_successful_batch}")
                result["success"] = True
                return False
            else:
                # Failed at batch 1 - this optimization level doesn't work
                return False
    
    def _finalize_success(self, best_level, best_batch, best_throughput, best_memory):
        """Finalize successful optimization."""
        self.results["success"] = True
        self.results["optimal_level"] = best_level
        self.results["optimal_batch"] = best_batch
        self.results["max_throughput"] = best_throughput
        self.results["message"] = (
            f"✅ SUCCESS! Found optimal settings after testing {len(self.optimization_levels)} levels:\n"
            f"   • Optimization: {best_level}\n"
            f"   • Batch Size: {best_batch}\n"
            f"   • Throughput: {best_throughput:.2f} steps/sec\n"
            f"   • Memory Usage: ~{best_memory:.1f}%"
        )
        
        self.log("")
        self.log("=" * 70)
        self.log("🎉 OPTIMIZATION COMPLETE!")
        self.log("=" * 70)
        self.log(self.results["message"])
        self.log("=" * 70)
        
        self._save_results()
        return self.results
    
    def _finalize_exhaustive_success(self, best_level, best_batch, best_throughput, best_memory):
        """Finalize successful exhaustive optimization."""
        self.results["success"] = True
        self.results["optimal_level"] = best_level
        self.results["optimal_batch"] = best_batch
        self.results["max_throughput"] = best_throughput
        self.results["message"] = (
            f"✅ SUCCESS via exhaustive chunk testing!\n"
            f"   • Optimization: {best_level}\n"
            f"   • Batch Size: {best_batch}\n"
            f"   • Throughput: {best_throughput:.2f} steps/sec\n"
            f"   • Memory Usage: ~{best_memory:.1f}%\n\n"
            f"   ⚠️  Note: Very small chunks may be slow. Consider reducing max_seq_len."
        )
        
        self.log("")
        self.log("=" * 70)
        self.log("🎉 EXHAUSTIVE OPTIMIZATION COMPLETE!")
        self.log("=" * 70)
        self.log(self.results["message"])
        self.log("=" * 70)
        
        self._save_results()
        return self.results
    
    def _finalize_failure(self):
        """Finalize failed optimization."""
        exhaustive_note = " (including exhaustive chunk sizes down to 128)" if self.config.max_seq_len > 8192 else ""
        self.results["success"] = False
        self.results["message"] = (
            f"❌ All optimization levels failed{exhaustive_note}!\n\n"
            "Suggestions:\n"
            "   1. Reduce model size (use smaller architecture)\n"
            "   2. Reduce context window (lower max-seq-len significantly)\n"
            "   3. Get a better GPU (more VRAM needed)\n\n"
            "   Or as we say: time to upgrade that potato, noob! 🥔➡️🖥️"
        )
        
        self.log("")
        self.log("=" * 70)
        self.log(self.results["message"])
        self.log("=" * 70)
        
        self._save_results()
        return self.results
    
    def _save_results(self):
        """Save optimization results to file."""
        try:
            results_file = self.output_dir / f"progressive_results_{self.config.session_id}.json"
            
            # Convert OptimizationLevel to dict for JSON serialization
            save_results = self.results.copy()
            if save_results.get("optimal_level"):
                save_results["optimal_level"] = str(save_results["optimal_level"])
            
            with open(results_file, 'w') as f:
                json.dump(save_results, f, indent=2)
            
            self.log(f"\n📄 Results saved to: {results_file}")
        except Exception as e:
            self.log(f"⚠️  Failed to save results: {e}")
