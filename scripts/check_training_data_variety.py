"""
Quick diagnostic to check if training data varies across iterations.

Run this on your training logs to verify data variety.
"""

import json
from pathlib import Path
import sys


def analyze_training_logs(log_file: Path):
    """Analyze training logs to check for data variety."""
    print(f"Analyzing: {log_file}")
    print("=" * 80)
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return False
    
    dataset_stats = []
    cycles = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Look for dataset statistics
                    if entry.get('event') == 'dataset_stats':
                        dataset_stats.append(entry)
                        print(f"\nCycle {entry.get('cycle', '?')} Dataset Stats:")
                        print(f"  Epoch: {entry.get('epoch', '?')}")
                        print(f"  Coverage: {entry.get('coverage_percent', '?')}%")
                        print(f"  First 5 indices: {entry.get('first_5_indices', '?')}")
                        print(f"  Last 5 indices: {entry.get('last_5_indices', '?')}")
                    
                    # Track iteration cycles
                    if entry.get('event') == 'iterate_cycle':
                        cycles.append(entry.get('cycle'))
                    
                    # Look for dataset mode info
                    if 'dataset_mode' in entry:
                        print(f"\n📊 Dataset Mode: {entry.get('dataset_mode')}")
                        print(f"   Initial samples: {entry.get('initial_samples', '?')}")
                
                except json.JSONDecodeError:
                    continue
    
    except Exception as e:
        print(f"❌ Error reading log: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    if not dataset_stats:
        print("⚠️  No dataset statistics found in log.")
        print("   This might mean:")
        print("   1. Training hasn't completed a full cycle yet")
        print("   2. Logging isn't working properly")
        print("   3. You're not using --iterate mode")
        return False
    
    if len(dataset_stats) < 2:
        print(f"⚠️  Only {len(dataset_stats)} cycle(s) recorded.")
        print("   Need at least 2 cycles to verify variety.")
        return False
    
    # Check if indices are different
    first_indices = [stats.get('first_5_indices', []) for stats in dataset_stats]
    
    print(f"\n✅ Found {len(dataset_stats)} cycles with dataset statistics")
    print("\nFirst 5 indices per cycle:")
    all_same = True
    for i, indices in enumerate(first_indices):
        print(f"  Cycle {i}: {indices}")
        if i > 0 and indices != first_indices[0]:
            all_same = False
    
    if all_same and len(set(map(str, first_indices))) == 1:
        print("\n❌ WARNING: All cycles show IDENTICAL indices!")
        print("   This means the same data is being used repeatedly.")
        print("   The bug is still present or the fix wasn't applied.")
        return False
    else:
        print("\n✅ SUCCESS: Cycles show DIFFERENT indices!")
        print("   Data variety is confirmed - different samples each iteration.")
        return True


if __name__ == "__main__":
    # Default log file location
    default_log = Path("artifacts/brains/actv1/metrics.jsonl")
    
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    else:
        log_file = default_log
        print(f"Using default log file: {log_file}")
        print("(Pass a different path as argument to check other logs)\n")
    
    success = analyze_training_logs(log_file)
    
    if success:
        print("\n" + "=" * 80)
        print("✅ VERIFICATION PASSED: Data variety confirmed!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("⚠️  Could not verify data variety from logs.")
        print("=" * 80)
        sys.exit(1)
