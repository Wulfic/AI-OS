"""Main test runner for memory estimation accuracy tests.

This script orchestrates comprehensive memory estimation testing:
1. Run baseline tests across tokenizers and context sizes
2. Run optimization combination tests
3. Analyze results and generate reports
4. Track progress toward 95% accuracy goal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_harness import MemoryTestHarness, TestConfiguration
from baseline_tests import (
    generate_baseline_configs,
    generate_quick_baseline_configs,
    MODEL_SIZES,
)
from optimization_tests import (
    generate_optimization_combinations,
    generate_critical_optimization_tests,
)
from analysis import ResultAnalyzer, ReportGenerator


def run_baseline_tests(harness: MemoryTestHarness, model_size: str = "tiny", quick: bool = False):
    """Run baseline tests for different tokenizers and context sizes.
    
    Args:
        harness: MemoryTestHarness instance
        model_size: Model size to test ("tiny", "small", "medium", "large")
        quick: If True, run quick subset of tests
    """
    print("\n" + "="*80)
    print(f"BASELINE TESTS - {model_size.upper()} MODEL")
    print("="*80)
    
    if quick:
        configs = generate_quick_baseline_configs()
    else:
        configs = generate_baseline_configs(model_size=model_size)
    
    print(f"\nGenerated {len(configs)} baseline test configurations")
    print(f"This will test:")
    print(f"  - Tokenizers: {sorted(set(c.tokenizer_name for c in configs))}")
    print(f"  - Context sizes: {sorted(set(c.seq_len for c in configs))}")
    print(f"  - No optimizations (pure baseline)")
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config.test_name}")
        
        try:
            result = harness.run_test(config, training_steps=5)
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            continue
    
    print(f"\n✅ Completed {len(results)} baseline tests")
    return results


def run_optimization_tests(
    harness: MemoryTestHarness,
    model_size: str = "tiny",
    full_combinations: bool = False,
):
    """Run optimization combination tests.
    
    Args:
        harness: MemoryTestHarness instance
        model_size: Model size to test
        full_combinations: If True, test all 2^N combinations; else test critical subset
    """
    print("\n" + "="*80)
    print(f"OPTIMIZATION TESTS - {model_size.upper()} MODEL")
    print("="*80)
    
    model_config = MODEL_SIZES[model_size]
    
    if full_combinations:
        configs = generate_optimization_combinations(
            model_size_config=model_config,
            tokenizer_name="gpt2",
            seq_len=1024,
        )
        print(f"\nGenerated {len(configs)} FULL optimization combination tests")
        print("⚠️ This will take a long time!")
    else:
        configs = generate_critical_optimization_tests(
            model_size_config=model_config,
            tokenizer_name="gpt2",
            seq_len=1024,
        )
        print(f"\nGenerated {len(configs)} CRITICAL optimization tests")
    
    print(f"\nThis will test various combinations of:")
    print("  - AMP (Automatic Mixed Precision)")
    print("  - Gradient Checkpointing")
    print("  - Chunked Sequence Processing")
    print("  - LoRA/PEFT")
    print("  - 8-bit Optimizer")
    print("  - CPU Offload")
    print("  - DeepSpeed ZeRO stages")
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config.test_name}")
        
        try:
            result = harness.run_test(config, training_steps=5)
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            continue
    
    print(f"\n✅ Completed {len(results)} optimization tests")
    return results


def analyze_and_report(harness: MemoryTestHarness, output_dir: Path):
    """Analyze results and generate comprehensive report.
    
    Args:
        harness: MemoryTestHarness instance
        output_dir: Directory to save reports
    """
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    # Load all results
    results = harness.load_results()
    
    if not results:
        print("❌ No test results found!")
        return
    
    print(f"\nLoaded {len(results)} test results")
    
    # Create analyzer
    analyzer = ResultAnalyzer(results)
    
    # Generate accuracy report
    print("\nGenerating accuracy report...")
    accuracy_report = analyzer.generate_accuracy_report()
    
    print("\n" + "="*80)
    print("ACCURACY SUMMARY")
    print("="*80)
    
    report_data = accuracy_report.to_dict()
    print(f"\nTotal Tests: {report_data['summary']['total_tests']}")
    print(f"Successful: {report_data['summary']['successful_tests']}")
    print(f"Failed: {report_data['summary']['failed_tests']}")
    
    print(f"\nVRAM Accuracy: {report_data['accuracy']['vram']['mean']:.1f}% ± {report_data['accuracy']['vram']['std']:.1f}%")
    print(f"RAM Accuracy: {report_data['accuracy']['ram']['mean']:.1f}% ± {report_data['accuracy']['ram']['std']:.1f}%")
    
    print(f"\nTests ≥95% accuracy: {report_data['thresholds']['above_95_pct']} ({report_data['thresholds']['above_95_pct']/report_data['summary']['successful_tests']*100:.1f}%)")
    print(f"Tests ≥90% accuracy: {report_data['thresholds']['above_90_pct']} ({report_data['thresholds']['above_90_pct']/report_data['summary']['successful_tests']*100:.1f}%)")
    print(f"Tests <80% accuracy: {report_data['thresholds']['below_80_pct']} ({report_data['thresholds']['below_80_pct']/report_data['summary']['successful_tests']*100:.1f}%)")
    
    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = analyzer.generate_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Generate markdown report
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    
    generator = ReportGenerator(analyzer)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "accuracy_report.md"
    
    generator.generate_markdown_report(output_file=report_file)
    
    # Save detailed JSON
    import json
    json_file = output_dir / "analysis_results.json"
    
    analysis_data = {
        "accuracy_report": report_data,
        "by_optimization": analyzer.analyze_by_optimization(),
        "by_context_size": analyzer.analyze_by_context_size(),
        "by_tokenizer": analyzer.analyze_by_tokenizer(),
        "recommendations": recommendations,
    }
    
    with open(json_file, "w") as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"✅ JSON analysis saved to: {json_file}")
    
    # Goal tracking
    print("\n" + "="*80)
    print("GOAL PROGRESS: 95% ACCURACY TARGET")
    print("="*80)
    
    pct_above_95 = report_data['thresholds']['above_95_pct'] / report_data['summary']['successful_tests'] * 100
    
    if pct_above_95 >= 95:
        print(f"\n🎉 SUCCESS! {pct_above_95:.1f}% of tests achieve ≥95% accuracy")
        print("Goal achieved! Memory estimator is production-ready.")
    elif pct_above_95 >= 85:
        print(f"\n✅ CLOSE! {pct_above_95:.1f}% of tests achieve ≥95% accuracy")
        print(f"Need {95 - pct_above_95:.1f}% more tests to reach goal.")
        print("Review recommendations to improve remaining cases.")
    elif pct_above_95 >= 70:
        print(f"\n⚠️ PROGRESS: {pct_above_95:.1f}% of tests achieve ≥95% accuracy")
        print(f"Need {95 - pct_above_95:.1f}% more tests to reach goal.")
        print("Significant improvements needed. Focus on problem cases.")
    else:
        print(f"\n❌ NEEDS WORK: Only {pct_above_95:.1f}% of tests achieve ≥95% accuracy")
        print("Major estimation improvements required.")
        print("Review recommendations and problem cases carefully.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Memory estimation accuracy testing suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick baseline tests
  python run_tests.py --baseline --quick
  
  # Run full baseline tests for small model
  python run_tests.py --baseline --model-size small
  
  # Run critical optimization tests
  python run_tests.py --optimizations
  
  # Run full optimization combinations (SLOW!)
  python run_tests.py --optimizations --full-combinations
  
  # Analyze existing results
  python run_tests.py --analyze-only
  
  # Run everything
  python run_tests.py --baseline --optimizations --analyze
        """
    )
    
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline tests (tokenizers + context sizes)",
    )
    parser.add_argument(
        "--optimizations",
        action="store_true",
        help="Run optimization combination tests",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results (don't run tests)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze results after running tests",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick subset of tests",
    )
    parser.add_argument(
        "--full-combinations",
        action="store_true",
        help="Test all 2^N optimization combinations (SLOW!)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "small", "medium", "large"],
        default="tiny",
        help="Model size to test (default: tiny)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="artifacts/memory_tests",
        help="Directory to store results (default: artifacts/memory_tests)",
    )
    
    args = parser.parse_args()
    
    # Default: if no action specified, run baseline + optimizations + analyze
    if not any([args.baseline, args.optimizations, args.analyze_only, args.analyze]):
        args.baseline = True
        args.optimizations = True
        args.analyze = True
    
    # Create harness
    harness = MemoryTestHarness(results_dir=args.results_dir)
    
    try:
        # Run tests
        if not args.analyze_only:
            if args.baseline:
                run_baseline_tests(
                    harness=harness,
                    model_size=args.model_size,
                    quick=args.quick,
                )
            
            if args.optimizations:
                run_optimization_tests(
                    harness=harness,
                    model_size=args.model_size,
                    full_combinations=args.full_combinations,
                )
        
        # Analyze results
        if args.analyze or args.analyze_only:
            output_dir = Path(args.results_dir)
            analyze_and_report(harness=harness, output_dir=output_dir)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        print("Partial results have been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.results_dir}")
    print("Review the accuracy report for detailed analysis.")


if __name__ == "__main__":
    main()
