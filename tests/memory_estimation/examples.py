"""Example usage of the memory estimation test suite.

This script demonstrates how to:
1. Run a simple test
2. Analyze results
3. Generate reports
"""

from pathlib import Path

from test_harness import MemoryTestHarness, TestConfiguration
from baseline_tests import generate_quick_baseline_configs, MODEL_SIZES
from optimization_tests import generate_critical_optimization_tests
from analysis import ResultAnalyzer, ReportGenerator


def example_single_test():
    """Example: Run a single test."""
    print("="*80)
    print("EXAMPLE 1: Single Test")
    print("="*80)
    
    # Create test harness
    harness = MemoryTestHarness(results_dir="artifacts/memory_tests")
    
    # Create a simple test configuration
    config = TestConfiguration(
        model_name="example-tiny",
        tokenizer_name="gpt2",
        total_params=2_000_000,  # 2M parameters
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        vocab_size=50257,
        seq_len=512,
        batch_size=2,
        num_gpus=1,
        # No optimizations
        use_amp=False,
        use_gradient_checkpointing=False,
        use_lora=False,
        lora_r=0,
        use_8bit_optimizer=False,
        offload_optimizer=False,
        zero_stage="none",
        use_chunking=False,
        chunk_size=None,
        test_id="example_001",
        test_name="Example Test",
        description="Simple example test",
    )
    
    # Run test
    print("\nRunning test...")
    result = harness.run_test(config, training_steps=5)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Estimated VRAM: {result.estimated_vram_gb:.2f} GB")
    print(f"Actual VRAM:    {result.actual_peak_vram_reserved_gb:.2f} GB")
    print(f"VRAM Accuracy:  {result.vram_accuracy_pct:.1f}%")
    print(f"VRAM Error:     {result.vram_error_gb:+.2f} GB")
    print()
    print(f"Estimated RAM:  {result.estimated_ram_gb:.2f} GB")
    print(f"Actual RAM:     {result.actual_peak_ram_gb:.2f} GB")
    print(f"RAM Accuracy:   {result.ram_accuracy_pct:.1f}%")
    print(f"RAM Error:      {result.ram_error_gb:+.2f} GB")


def example_batch_tests():
    """Example: Run batch of tests."""
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Batch Tests")
    print("="*80)
    
    # Create test harness
    harness = MemoryTestHarness(results_dir="artifacts/memory_tests")
    
    # Generate quick baseline configs
    configs = generate_quick_baseline_configs()
    
    print(f"\nRunning {len(configs)} tests...")
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config.test_name}")
        
        try:
            result = harness.run_test(config, training_steps=3)
            results.append(result)
            print(f"  ✅ VRAM Accuracy: {result.vram_accuracy_pct:.1f}%")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ Completed {len(results)} tests")


def example_optimization_tests():
    """Example: Test optimization combinations."""
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Optimization Tests")
    print("="*80)
    
    # Create test harness
    harness = MemoryTestHarness(results_dir="artifacts/memory_tests")
    
    # Generate critical optimization configs
    model_config = MODEL_SIZES["tiny"]
    configs = generate_critical_optimization_tests(
        model_size_config=model_config,
        tokenizer_name="gpt2",
        seq_len=1024,
    )
    
    print(f"\nRunning {len(configs)} optimization tests...")
    
    # Just run first 3 for demo
    for i, config in enumerate(configs[:3], 1):
        print(f"\n[{i}/3] {config.test_name}")
        
        try:
            result = harness.run_test(config, training_steps=3)
            print(f"  ✅ VRAM Accuracy: {result.vram_accuracy_pct:.1f}%")
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def example_analysis():
    """Example: Analyze results."""
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Analysis")
    print("="*80)
    
    # Load results
    harness = MemoryTestHarness(results_dir="artifacts/memory_tests")
    results = harness.load_results()
    
    if not results:
        print("\n❌ No results found! Run some tests first.")
        return
    
    print(f"\nLoaded {len(results)} test results")
    
    # Create analyzer
    analyzer = ResultAnalyzer(results)
    
    # Generate accuracy report
    accuracy_report = analyzer.generate_accuracy_report()
    
    print("\n" + "="*80)
    print("ACCURACY SUMMARY")
    print("="*80)
    
    report_data = accuracy_report.to_dict()
    
    print(f"\nTotal Tests: {report_data['summary']['total_tests']}")
    print(f"Successful:  {report_data['summary']['successful_tests']}")
    print(f"Failed:      {report_data['summary']['failed_tests']}")
    
    print(f"\nVRAM Accuracy: {report_data['accuracy']['vram']['mean']:.1f}% ± {report_data['accuracy']['vram']['std']:.1f}%")
    print(f"RAM Accuracy:  {report_data['accuracy']['ram']['mean']:.1f}% ± {report_data['accuracy']['ram']['std']:.1f}%")
    
    # Get recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = analyzer.generate_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
    
    # Generate markdown report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    generator = ReportGenerator(analyzer)
    output_file = Path("artifacts/memory_tests/example_report.md")
    generator.generate_markdown_report(output_file=output_file)
    
    print(f"\n✅ Report saved to: {output_file}")


def main():
    """Run all examples."""
    print("MEMORY ESTIMATION TEST SUITE - EXAMPLES")
    print("="*80)
    
    try:
        # Run examples
        example_single_test()
        example_batch_tests()
        example_optimization_tests()
        example_analysis()
        
        print("\n\n" + "="*80)
        print("ALL EXAMPLES COMPLETED")
        print("="*80)
        print("\nNext steps:")
        print("1. Review results in: artifacts/memory_tests/")
        print("2. Check accuracy report: artifacts/memory_tests/example_report.md")
        print("3. Run full test suite: python run_tests.py --baseline --optimizations --analyze")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
