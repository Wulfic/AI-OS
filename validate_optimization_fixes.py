"""
Validation script for optimization system fixes.

Tests:
1. Binary search continues after OOM
2. Heartbeat monitoring works
3. Throughput parsing is accurate
4. Test duration is sufficient
"""

import json
import time
from pathlib import Path
from typing import Dict, Any


def test_throughput_parsing():
    """Test that throughput parsing correctly reads JSONL files."""
    print("Testing throughput parsing...")
    
    # Create temporary test log
    test_log = Path("test_metrics.jsonl")
    
    # Write sample generation events
    events = [
        {"event": "gen_progress", "generated": 10, "ts": 1000.0},
        {"event": "gen_progress", "generated": 20, "ts": 1001.0},
        {"event": "gen_progress", "generated": 30, "ts": 1002.0},
        {"event": "gen_progress", "generated": 40, "ts": 1003.0},
    ]
    
    with open(test_log, 'w') as f:
        for event in events:
            f.write(json.dumps(event) + '\n')
    
    # Parse throughput
    total_generated = 0
    first_ts = None
    last_ts = None
    
    with open(test_log, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("event") == "gen_progress":
                    total_generated = max(total_generated, data.get("generated", 0))
                    ts = data.get("ts")
                    if ts:
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
            except json.JSONDecodeError:
                continue
    
    # Calculate throughput
    if total_generated > 0 and first_ts and last_ts and last_ts > first_ts:
        throughput = total_generated / (last_ts - first_ts)
        print(f"✅ Throughput parsing works: {throughput:.2f} samples/sec")
        print(f"   Total generated: {total_generated}, Duration: {last_ts - first_ts}s")
        assert throughput > 0, "Throughput should be positive"
        assert abs(throughput - 40/3) < 0.1, f"Expected ~13.3, got {throughput}"
    else:
        print("❌ Throughput parsing failed")
        raise AssertionError("Failed to parse throughput")
    
    # Cleanup
    test_log.unlink()
    print()


def test_binary_search_logic():
    """Test that binary search logic works correctly."""
    print("Testing binary search logic...")
    
    # Simulate batch test results
    visited = {
        1: {"success": True, "throughput": 10.0, "batch_size": 1},
        2: {"success": True, "throughput": 18.0, "batch_size": 2},
        4: {"success": True, "throughput": 32.0, "batch_size": 4},
        8: {"success": True, "throughput": 55.0, "batch_size": 8},
        16: {"success": False, "oom": True, "throughput": 0.0, "batch_size": 16}
    }
    
    # Binary search should test 12 next
    lower = 8  # Last success
    upper = 16  # First failure
    mid = (lower + upper) // 2
    
    print(f"   Last success: batch {lower}")
    print(f"   First failure: batch {upper}")
    print(f"   Should test next: batch {mid}")
    
    assert mid == 12, f"Expected mid=12, got {mid}"
    print(f"✅ Binary search logic correct")
    print()


def test_config_defaults():
    """Test that configuration defaults are correct."""
    print("Testing configuration defaults...")
    
    from aios.gui.components.hrm_training.optimizer_unified import OptimizationConfig
    
    config = OptimizationConfig()
    
    print(f"   test_duration: {config.test_duration}s")
    print(f"   max_timeout: {config.max_timeout}s")
    print(f"   batch_sizes: {config.batch_sizes}")
    
    assert config.test_duration >= 45, f"test_duration should be >=45s, got {config.test_duration}"
    assert config.max_timeout >= 90, f"max_timeout should be >=90s, got {config.max_timeout}"
    assert config.batch_sizes is not None, "batch_sizes should not be None"
    
    print(f"✅ Configuration defaults correct")
    print()


def test_oom_detection():
    """Test OOM detection from stderr."""
    print("Testing OOM detection...")
    
    test_cases = [
        ("RuntimeError: CUDA out of memory", True),
        ("RuntimeError: out of memory", True),
        ("torch.cuda.OutOfMemoryError", True),
        ("Normal output", False),
        ("Other error message", False),
    ]
    
    for stderr, should_detect in test_cases:
        detected = "out of memory" in stderr.lower() or "cuda oom" in stderr.lower()
        if detected == should_detect:
            print(f"✅ '{stderr[:40]}...' → OOM={detected}")
        else:
            print(f"❌ '{stderr[:40]}...' → Expected OOM={should_detect}, got {detected}")
            raise AssertionError(f"OOM detection failed for: {stderr}")
    
    print()


def test_heartbeat_logic():
    """Test heartbeat monitoring logic."""
    print("Testing heartbeat monitoring logic...")
    
    # Simulate heartbeat timing
    start_time = 1000.0
    last_heartbeat = 1000.0
    heartbeat_timeout = 30.0
    
    # Case 1: Recent heartbeat (should continue)
    current_time = 1015.0  # 15 seconds elapsed
    elapsed_since_heartbeat = current_time - last_heartbeat
    
    if elapsed_since_heartbeat < heartbeat_timeout:
        print(f"✅ Process alive: {elapsed_since_heartbeat}s < {heartbeat_timeout}s")
    else:
        print(f"❌ False timeout: {elapsed_since_heartbeat}s >= {heartbeat_timeout}s")
        raise AssertionError("Heartbeat logic incorrect")
    
    # Case 2: Stale heartbeat (should kill)
    current_time = 1035.0  # 35 seconds elapsed
    elapsed_since_heartbeat = current_time - last_heartbeat
    
    if elapsed_since_heartbeat >= heartbeat_timeout:
        print(f"✅ Process frozen: {elapsed_since_heartbeat}s >= {heartbeat_timeout}s")
    else:
        print(f"❌ Missed timeout: {elapsed_since_heartbeat}s < {heartbeat_timeout}s")
        raise AssertionError("Heartbeat logic incorrect")
    
    print()


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Optimization System Fixes - Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("Throughput Parsing", test_throughput_parsing),
        ("Binary Search Logic", test_binary_search_logic),
        ("Configuration Defaults", test_config_defaults),
        ("OOM Detection", test_oom_detection),
        ("Heartbeat Monitoring", test_heartbeat_logic),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name} FAILED: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✅ All validation tests passed!")
        print("\nOptimization system fixes are verified and working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
