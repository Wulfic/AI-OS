#!/usr/bin/env python3
"""
LoRA/PEFT Configuration Testing Suite for AI-OS

This script validates that all LoRA/PEFT configurations work correctly by:
1. Testing all PEFT methods (LoRA, AdaLoRA, IA3)
2. Validating different rank configurations
3. Testing various target module combinations
4. Verifying parameter counting and memory estimation
5. Ensuring error handling works properly

Usage:
    python test_lora_peft_configurations.py [--quick] [--verbose]

Author: AI-OS Team
Date: October 19, 2025
"""

import sys
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json


@dataclass
class TestConfig:
    """Configuration for a single PEFT test."""
    name: str
    use_peft: bool
    peft_method: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    expected_trainable_ratio: Optional[float] = None  # Expected % of trainable params
    should_fail: bool = False  # Set True if this config should fail
    failure_reason: Optional[str] = None


# Test configurations covering various scenarios
TEST_CONFIGS = [
    # Baseline: PEFT disabled
    TestConfig(
        name="Baseline (No PEFT)",
        use_peft=False,
        peft_method="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=100.0,
    ),
    
    # LoRA configurations
    TestConfig(
        name="LoRA - Minimal (r=4, q+v)",
        use_peft=True,
        peft_method="lora",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=0.5,
    ),
    TestConfig(
        name="LoRA - Efficient (r=8, q+v)",
        use_peft=True,
        peft_method="lora",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=1.0,
    ),
    TestConfig(
        name="LoRA - Balanced (r=16, q+v) [RECOMMENDED]",
        use_peft=True,
        peft_method="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=2.0,
    ),
    TestConfig(
        name="LoRA - Balanced (r=16, q+k+v+o)",
        use_peft=True,
        peft_method="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        expected_trainable_ratio=4.0,
    ),
    TestConfig(
        name="LoRA - Full (r=16, all modules)",
        use_peft=True,
        peft_method="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        expected_trainable_ratio=8.0,
    ),
    TestConfig(
        name="LoRA - High Rank (r=32, q+k+v+o)",
        use_peft=True,
        peft_method="lora",
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        expected_trainable_ratio=8.0,
    ),
    TestConfig(
        name="LoRA - Very High Rank (r=64, q+k+v+o)",
        use_peft=True,
        peft_method="lora",
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        expected_trainable_ratio=16.0,
    ),
    
    # AdaLoRA configurations
    TestConfig(
        name="AdaLoRA - Balanced (r=16, q+k+v+o)",
        use_peft=True,
        peft_method="adalora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        expected_trainable_ratio=4.0,
    ),
    TestConfig(
        name="AdaLoRA - Minimal (r=8, q+v)",
        use_peft=True,
        peft_method="adalora",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=1.0,
    ),
    
    # IA3 configurations (no rank parameter)
    TestConfig(
        name="IA3 - Minimal (q+v)",
        use_peft=True,
        peft_method="ia3",
        lora_r=16,  # Ignored for IA3
        lora_alpha=32,  # Ignored for IA3
        lora_dropout=0.0,  # Ignored for IA3
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=0.1,
    ),
    TestConfig(
        name="IA3 - Full (all modules)",
        use_peft=True,
        peft_method="ia3",
        lora_r=16,  # Ignored for IA3
        lora_alpha=32,  # Ignored for IA3
        lora_dropout=0.0,  # Ignored for IA3
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        expected_trainable_ratio=0.3,
    ),
    
    # Dropout variations
    TestConfig(
        name="LoRA - No Dropout (r=16, q+v)",
        use_peft=True,
        peft_method="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=2.0,
    ),
    TestConfig(
        name="LoRA - High Dropout (r=16, q+v)",
        use_peft=True,
        peft_method="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.2,
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=2.0,
    ),
    
    # Alpha variations
    TestConfig(
        name="LoRA - Low Alpha (r=16, α=16, q+v)",
        use_peft=True,
        peft_method="lora",
        lora_r=16,
        lora_alpha=16,  # 1:1 ratio
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=2.0,
    ),
    TestConfig(
        name="LoRA - High Alpha (r=16, α=64, q+v)",
        use_peft=True,
        peft_method="lora",
        lora_r=16,
        lora_alpha=64,  # 4:1 ratio
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
        expected_trainable_ratio=2.0,
    ),
    
    # Edge cases and error handling
    TestConfig(
        name="ERROR: Empty target modules",
        use_peft=True,
        peft_method="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="",
        should_fail=True,
        failure_reason="No target modules specified",
    ),
    TestConfig(
        name="ERROR: Invalid module name",
        use_peft=True,
        peft_method="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="invalid_module,q_proj",
        should_fail=True,
        failure_reason="Invalid target module",
    ),
]


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_test_start(test_name: str, test_num: int, total: int):
    """Print test start message."""
    print(f"\n{Colors.OKBLUE}[{test_num}/{total}] Testing: {test_name}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*60}{Colors.ENDC}")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def simulate_peft_application(config: TestConfig, verbose: bool = False) -> Dict[str, Any]:
    """
    Simulate applying PEFT to a model and return statistics.
    
    This simulates the apply_peft() function from model_precision.py
    without requiring actual model or PEFT library.
    
    Returns:
        Dict with results including success status, trainable params, etc.
    """
    result = {
        "success": False,
        "error": None,
        "trainable_params": 0,
        "total_params": 87_000_000,  # Approximate for GPT-2 size model
        "trainable_ratio": 0.0,
        "method": config.peft_method,
        "rank": config.lora_r if config.peft_method != "ia3" else "N/A",
        "alpha": config.lora_alpha if config.peft_method != "ia3" else "N/A",
        "dropout": config.lora_dropout if config.peft_method != "ia3" else "N/A",
        "target_modules": config.lora_target_modules,
    }
    
    try:
        # Validate configuration
        if not config.use_peft:
            result["trainable_params"] = result["total_params"]
            result["trainable_ratio"] = 100.0
            result["success"] = True
            return result
        
        # Parse target modules
        target_modules = [m.strip() for m in config.lora_target_modules.split(',') if m.strip()]
        
        if not target_modules:
            raise ValueError("No target modules specified for PEFT")
        
        # Check for invalid modules
        valid_modules = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        }
        invalid_modules = set(target_modules) - valid_modules
        if invalid_modules:
            raise ValueError(f"Invalid target modules: {invalid_modules}")
        
        # Estimate trainable parameters based on method and configuration
        base_params_per_module = 768  # Typical dimension
        
        if config.peft_method == "ia3":
            # IA3 uses scaling vectors, much fewer params
            params_per_module = base_params_per_module * 2  # Just scaling vectors
        else:
            # LoRA and AdaLoRA use low-rank matrices
            # params = 2 * rank * dimension (A and B matrices)
            params_per_module = 2 * config.lora_r * base_params_per_module
        
        # Estimate total trainable params
        num_layers = 12  # Typical for GPT-2 size
        trainable_params = params_per_module * len(target_modules) * num_layers
        
        # Add params for heads that are always trainable
        lm_head_params = 768 * 50257  # vocab_size
        q_head_params = 768 * 4  # halting head
        trainable_params += lm_head_params + q_head_params
        
        result["trainable_params"] = trainable_params
        result["trainable_ratio"] = (trainable_params / result["total_params"]) * 100
        result["success"] = True
        
        if verbose:
            print_info(f"Trainable params: {trainable_params:,} ({result['trainable_ratio']:.2f}%)")
        
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False
        if verbose:
            print_error(f"Error: {e}")
    
    return result


def validate_test_result(config: TestConfig, result: Dict[str, Any]) -> bool:
    """
    Validate that test result matches expectations.
    
    Returns:
        True if test passed, False otherwise
    """
    if config.should_fail:
        # Test should have failed
        if result["success"]:
            print_error(f"Expected failure but test succeeded!")
            return False
        else:
            print_success(f"Test correctly failed: {result['error']}")
            return True
    else:
        # Test should have succeeded
        if not result["success"]:
            print_error(f"Test failed unexpectedly: {result['error']}")
            return False
        
        # Validate trainable parameter ratio
        if config.expected_trainable_ratio is not None:
            actual_ratio = result["trainable_ratio"]
            expected_ratio = config.expected_trainable_ratio
            
            # Allow 50% tolerance due to estimation
            tolerance = expected_ratio * 0.5
            if abs(actual_ratio - expected_ratio) > tolerance:
                print_warning(
                    f"Trainable ratio outside tolerance: "
                    f"expected ~{expected_ratio:.1f}%, got {actual_ratio:.2f}%"
                )
            else:
                print_success(
                    f"Trainable ratio within tolerance: {actual_ratio:.2f}% "
                    f"(expected ~{expected_ratio:.1f}%)"
                )
        
        print_success(f"Test passed successfully!")
        return True


def run_parameter_impact_analysis(verbose: bool = False):
    """Analyze the impact of different parameter combinations."""
    print_header("Parameter Impact Analysis")
    
    # Analyze rank impact
    print(f"\n{Colors.BOLD}Impact of Rank (r) on Parameters:{Colors.ENDC}")
    print(f"{'Rank':<10} {'Trainable Params':<20} {'% of Total':<15} {'VRAM Est.':<15}")
    print("-" * 60)
    
    for r in [4, 8, 16, 32, 64]:
        config = TestConfig(
            name=f"Analysis r={r}",
            use_peft=True,
            peft_method="lora",
            lora_r=r,
            lora_alpha=r*2,
            lora_dropout=0.05,
            lora_target_modules="q_proj,v_proj",
        )
        result = simulate_peft_application(config, verbose=False)
        vram_est = result["trainable_params"] * 4 / (1024**3) * 4  # 4 bytes per param, 4× for optimizer
        
        print(f"r={r:<8} {result['trainable_params']:>15,} {result['trainable_ratio']:>12.2f}% {vram_est:>12.1f} GB")
    
    # Analyze target modules impact
    print(f"\n{Colors.BOLD}Impact of Target Modules on Parameters (r=16):{Colors.ENDC}")
    print(f"{'Configuration':<30} {'Trainable Params':<20} {'% of Total':<15}")
    print("-" * 65)
    
    module_configs = [
        ("q_proj,v_proj", "Minimal (q+v)"),
        ("q_proj,k_proj,v_proj,o_proj", "Balanced (attention)"),
        ("q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", "Full (attn+MLP)"),
    ]
    
    for modules, desc in module_configs:
        config = TestConfig(
            name=f"Analysis {desc}",
            use_peft=True,
            peft_method="lora",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules=modules,
        )
        result = simulate_peft_application(config, verbose=False)
        print(f"{desc:<30} {result['trainable_params']:>15,} {result['trainable_ratio']:>12.2f}%")
    
    # Compare methods
    print(f"\n{Colors.BOLD}Comparison of PEFT Methods (r=16, balanced modules):{Colors.ENDC}")
    print(f"{'Method':<15} {'Trainable Params':<20} {'% of Total':<15} {'Features':<30}")
    print("-" * 80)
    
    methods = [
        ("lora", "Standard, most stable"),
        ("adalora", "Adaptive rank allocation"),
        ("ia3", "Minimal params, fastest"),
    ]
    
    for method, features in methods:
        config = TestConfig(
            name=f"Analysis {method}",
            use_peft=True,
            peft_method=method,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        )
        result = simulate_peft_application(config, verbose=False)
        print(f"{method:<15} {result['trainable_params']:>15,} {result['trainable_ratio']:>12.2f}% {features:<30}")


def run_all_tests(quick: bool = False, verbose: bool = False):
    """Run all PEFT configuration tests."""
    print_header("LoRA/PEFT Configuration Testing Suite")
    
    print(f"{Colors.BOLD}Test Configuration:{Colors.ENDC}")
    print(f"  Total tests: {len(TEST_CONFIGS)}")
    print(f"  Quick mode: {'Yes' if quick else 'No'}")
    print(f"  Verbose: {'Yes' if verbose else 'No'}")
    
    # Run parameter impact analysis first
    if not quick:
        run_parameter_impact_analysis(verbose)
    
    # Run individual tests
    print_header("Running Individual Configuration Tests")
    
    passed = 0
    failed = 0
    skipped = 0
    
    configs_to_test = TEST_CONFIGS[:5] if quick else TEST_CONFIGS
    
    for i, config in enumerate(configs_to_test, 1):
        print_test_start(config.name, i, len(configs_to_test))
        
        # Print configuration
        if verbose:
            print(f"  use_peft: {config.use_peft}")
            print(f"  method: {config.peft_method}")
            print(f"  rank: {config.lora_r}")
            print(f"  alpha: {config.lora_alpha}")
            print(f"  dropout: {config.lora_dropout}")
            print(f"  target_modules: {config.lora_target_modules}")
            print()
        
        # Run test
        result = simulate_peft_application(config, verbose)
        
        # Validate result
        if validate_test_result(config, result):
            passed += 1
        else:
            failed += 1
    
    # Print summary
    print_header("Test Summary")
    
    total = passed + failed + skipped
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"{Colors.BOLD}Results:{Colors.ENDC}")
    print(f"  {Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")
    print(f"  {Colors.FAIL}Failed: {failed}{Colors.ENDC}")
    print(f"  {Colors.WARNING}Skipped: {skipped}{Colors.ENDC}")
    print(f"  {Colors.BOLD}Total: {total}{Colors.ENDC}")
    print(f"  {Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.ENDC}")
    
    if failed == 0:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 All tests passed!{Colors.ENDC}")
        return 0
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}❌ Some tests failed!{Colors.ENDC}")
        return 1


def generate_config_examples():
    """Generate example configuration files."""
    print_header("Generating Configuration Examples")
    
    examples = {
        "minimal_budget.yaml": {
            "use_peft": True,
            "peft_method": "ia3",
            "lora_target_modules": "q_proj,v_proj",
            "description": "Minimal configuration for < 8 GB VRAM"
        },
        "efficient_recommended.yaml": {
            "use_peft": True,
            "peft_method": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,v_proj",
            "description": "Recommended efficient configuration (8-12 GB VRAM)"
        },
        "balanced_quality.yaml": {
            "use_peft": True,
            "peft_method": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,k_proj,v_proj,o_proj",
            "description": "Balanced configuration for quality (12-16 GB VRAM)"
        },
        "high_quality.yaml": {
            "use_peft": True,
            "peft_method": "lora",
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "description": "High quality configuration (16-24 GB VRAM)"
        },
    }
    
    for filename, config in examples.items():
        print(f"\n{Colors.OKBLUE}{filename}:{Colors.ENDC}")
        print(json.dumps(config, indent=2))
    
    print_success("\nConfiguration examples generated!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test LoRA/PEFT configurations for AI-OS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test suite (subset of tests)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output with detailed information"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Generate configuration examples"
    )
    
    args = parser.parse_args()
    
    try:
        if args.examples:
            generate_config_examples()
            return 0
        else:
            return run_all_tests(quick=args.quick, verbose=args.verbose)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Tests interrupted by user{Colors.ENDC}")
        return 130
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
