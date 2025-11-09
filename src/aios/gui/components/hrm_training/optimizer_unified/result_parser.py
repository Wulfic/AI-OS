"""Result parsing utilities for training throughput and OOM detection."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Callable


def parse_training_throughput(
    train_log: Path,
    test_duration: int,
    log_callback: Optional[Callable[[str], None]] = None
) -> float:
    """Parse training throughput from log file.
    
    Args:
        train_log: Path to training log file
        test_duration: Test duration for fallback calculation
        log_callback: Optional logging callback
        
    Returns:
        Throughput in steps/second
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)
    
    try:
        if not train_log.exists():
            log(f"⚠️  Training log not found: {train_log}")
            log("  Training command likely failed before writing any logs")
            return 0.0
        
        file_size = train_log.stat().st_size
        if file_size == 0:
            log(f"⚠️  Training log is empty: {train_log}")
            log("  Training started but didn't write any metrics")
            return 0.0
        
        log(f"Parsing training log: {train_log} ({file_size} bytes)")
        
        steps_completed = 0
        first_ts = None
        last_ts = None
        line_count = 0
        train_events = 0
        
        with open(train_log, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                try:
                    data = json.loads(line.strip())
                    ts = data.get("ts")
                    event = data.get("event")
                    
                    # Log first few events for debugging
                    if line_count <= 3:
                        log(f"  Line {line_count}: {event} (step={data.get('step', '?')})")
                    
                    # Count training steps
                    if event == "train":
                        train_events += 1
                        steps_completed += 1
                        if ts:
                            if first_ts is None:
                                first_ts = ts
                            last_ts = ts
                except json.JSONDecodeError as e:
                    log(f"  ⚠️  Invalid JSON at line {line_count}: {e}")
                    continue
        
        log(f"Log summary: {line_count} total lines, {train_events} train events")
        
        # Calculate throughput: steps per second
        if steps_completed > 0 and first_ts and last_ts and last_ts > first_ts:
            duration = last_ts - first_ts
            throughput = steps_completed / max(duration, 0.1)
            log(f"✓ Throughput: {steps_completed} steps in {duration:.1f}s = {throughput:.2f} steps/sec")
            return throughput
        elif steps_completed > 0:
            # Fallback if no timestamps
            log(f"⚠️  Using fallback calculation (no valid timestamps), {steps_completed} steps")
            return steps_completed / max(test_duration, 1)
        else:
            log(f"⚠️  No training steps found in {line_count} log lines")
        
        return 0.0
        
    except Exception as e:
        log(f"❌ Error parsing training throughput: {e}")
        import traceback
        log(f"  {traceback.format_exc()}")
        return 0.0


def detect_oom(result: subprocess.CompletedProcess) -> bool:
    """Detect OOM with comprehensive error pattern matching.
    
    Args:
        result: Completed subprocess result
        
    Returns:
        True if OOM detected
    """
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
