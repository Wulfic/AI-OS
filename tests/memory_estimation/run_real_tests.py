"""Run Real HRM Training Tests.

Execute test suites that run actual train_actv1_impl() calls to measure VRAM/RAM usage.
Saves results to JSON for analysis and estimator validation.

Usage:
    python run_real_tests.py --suite quick
    python run_real_tests.py --suite standard
    python run_real_tests.py --suite full
    python run_real_tests.py --suite optimization
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_harness_real import RealTestConfig, RealTestResult, run_real_training_test
from baseline_tests_real import (
    generate_quick_baseline_configs,
    generate_standard_baseline_configs,
    generate_full_baseline_configs,
    generate_optimization_comparison_configs,
    generate_parallel_gpu_configs,
    estimate_test_duration,
)


# ============================================================================
# Test Suite Execution
# ============================================================================

def run_test_suite(
    suite_name: str,
    configs: list[RealTestConfig],
    output_file: Path,
    verbose: bool = True,
    resume_from: Optional[int] = None,
) -> list[RealTestResult]:
    """Run a test suite and save results.
    
    Args:
        suite_name: Name of the test suite
        configs: List of test configurations to run
        output_file: Path to save results JSON
        verbose: Print progress messages
        resume_from: Resume from this test index (0-based)
        
    Returns:
        List of test results
    """
    results = []
    start_index = resume_from if resume_from is not None else 0
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {suite_name} Test Suite")
        print(f"{'='*80}")
        print(f"Total tests: {len(configs)}")
        if start_index > 0:
            print(f"Resuming from test {start_index + 1}")
        print(f"Estimated duration: {estimate_test_duration(len(configs) - start_index)}")
        print(f"Results will be saved to: {output_file}")
        print(f"{'='*80}\n")
    
    # Load existing results if resuming
    if start_index > 0 and output_file.exists():
        with open(output_file) as f:
            data = json.load(f)
            results = [RealTestResult(**r) for r in data["results"][:start_index]]
        if verbose:
            print(f"Loaded {len(results)} existing results")
    
    suite_start_time = time.time()
    
    for i, config in enumerate(configs[start_index:], start=start_index):
        if verbose:
            print(f"\n[Test {i+1}/{len(configs)}]")
            print(f"  Model: {config.model_name}")
            print(f"  Architecture: {config.h_layers}h/{config.l_layers}l, "
                  f"hidden={config.hidden_size}")
            print(f"  Context: {config.context_size}, Batch: {config.batch_size}")
        
        # Run test
        result = run_real_training_test(config, verbose=verbose)
        results.append(result)
        
        # Save intermediate results after each test
        save_results(
            suite_name=suite_name,
            results=results,
            output_file=output_file,
            partial=True,
        )
        
        if verbose:
            if result.success:
                print(f"  ✅ PASSED - VRAM: {result.actual_vram_bytes / 1024**3:.2f} GB, "
                      f"Duration: {result.test_duration_seconds:.1f}s")
            else:
                print(f"  ❌ FAILED - {result.error_message}")
    
    suite_duration = time.time() - suite_start_time
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{suite_name} Test Suite Complete")
        print(f"{'='*80}")
        print(f"Total tests: {len(results)}")
        print(f"Passed: {sum(1 for r in results if r.success)}")
        print(f"Failed: {sum(1 for r in results if not r.success)}")
        print(f"Duration: {suite_duration / 60:.1f} minutes")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")
    
    return results


def save_results(
    suite_name: str,
    results: list[RealTestResult],
    output_file: Path,
    partial: bool = False,
):
    """Save test results to JSON file.
    
    Args:
        suite_name: Name of the test suite
        results: List of test results
        output_file: Path to save JSON
        partial: Whether this is a partial save (mid-suite)
    """
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate summary statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    failed_tests = total_tests - passed_tests
    
    # VRAM usage statistics for successful tests
    successful_results = [r for r in results if r.success]
    if successful_results:
        avg_vram_gb = sum(r.actual_vram_bytes for r in successful_results) / len(successful_results) / (1024**3)
        min_vram_gb = min(r.actual_vram_bytes for r in successful_results) / (1024**3)
        max_vram_gb = max(r.actual_vram_bytes for r in successful_results) / (1024**3)
    else:
        avg_vram_gb = 0.0
        min_vram_gb = 0.0
        max_vram_gb = 0.0
    
    # Create summary
    summary = {
        "suite_name": suite_name,
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "avg_vram_gb": avg_vram_gb,
        "min_vram_gb": min_vram_gb,
        "max_vram_gb": max_vram_gb,
        "partial": partial,
    }
    
    # Create output
    output = {
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)


# ============================================================================
# Result Analysis
# ============================================================================

def analyze_results(results_file: Path):
    """Analyze test results and print summary.
    
    Args:
        results_file: Path to results JSON file
    """
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file) as f:
        data = json.load(f)
    
    summary = data["summary"]
    results = [RealTestResult(**r) for r in data["results"]]
    
    print(f"\n{'='*80}")
    print(f"Test Results Analysis: {summary['suite_name']}")
    print(f"{'='*80}")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print()
    
    # VRAM usage statistics
    print("VRAM Usage (Actual Measurements):")
    print(f"  Average: {summary['avg_vram_gb']:.2f} GB")
    print(f"  Min: {summary['min_vram_gb']:.2f} GB")
    print(f"  Max: {summary['max_vram_gb']:.2f} GB")
    print()
    
    # Group by VRAM usage bands
    successful = [r for r in results if r.success]
    if successful:
        under_2gb = sum(1 for r in successful if r.actual_vram_bytes < 2 * 1024**3)
        under_4gb = sum(1 for r in successful if 2 * 1024**3 <= r.actual_vram_bytes < 4 * 1024**3)
        under_8gb = sum(1 for r in successful if 4 * 1024**3 <= r.actual_vram_bytes < 8 * 1024**3)
        over_8gb = sum(1 for r in successful if r.actual_vram_bytes >= 8 * 1024**3)
        
        print("VRAM Distribution:")
        print(f"  < 2 GB: {under_2gb} tests ({under_2gb/len(successful)*100:.1f}%)")
        print(f"  2-4 GB: {under_4gb} tests ({under_4gb/len(successful)*100:.1f}%)")
        print(f"  4-8 GB: {under_8gb} tests ({under_8gb/len(successful)*100:.1f}%)")
        print(f"  > 8 GB: {over_8gb} tests ({over_8gb/len(successful)*100:.1f}%)")
    
    print(f"{'='*80}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run real HRM training tests")
    parser.add_argument(
        "--suite",
        choices=["quick", "standard", "full", "optimization", "parallel"],
        default="quick",
        help="Test suite to run (default: quick)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Z:/AI-OS-Data/memory_test_results"),
        help="Output directory for results (default: Z:/AI-OS-Data/memory_test_results)",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        help="Resume from test index (0-based)",
    )
    parser.add_argument(
        "--analyze",
        type=Path,
        help="Analyze existing results file instead of running tests",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize output (only show summary)",
    )
    
    args = parser.parse_args()
    
    # If analyzing existing results
    if args.analyze:
        analyze_results(args.analyze)
        return
    
    # Generate test configs
    if args.suite == "quick":
        configs = generate_quick_baseline_configs()
        output_file = args.output_dir / "baseline_real_quick_results.json"
    elif args.suite == "standard":
        configs = generate_standard_baseline_configs()
        output_file = args.output_dir / "baseline_real_standard_results.json"
    elif args.suite == "full":
        configs = generate_full_baseline_configs()
        output_file = args.output_dir / "baseline_real_full_results.json"
    elif args.suite == "optimization":
        configs = generate_optimization_comparison_configs()
        output_file = args.output_dir / "optimization_comparison_results.json"
    elif args.suite == "parallel":
        configs = generate_parallel_gpu_configs()
        output_file = args.output_dir / "parallel_gpu_results.json"
    else:
        print(f"Unknown suite: {args.suite}")
        return
    
    # Run test suite
    results = run_test_suite(
        suite_name=f"{args.suite.capitalize()} Baseline",
        configs=configs,
        output_file=output_file,
        verbose=not args.quiet,
        resume_from=args.resume_from,
    )
    
    # Analyze results
    if not args.quiet:
        analyze_results(output_file)


if __name__ == "__main__":
    main()
