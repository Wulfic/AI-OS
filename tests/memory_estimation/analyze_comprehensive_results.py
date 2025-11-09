"""Analyze comprehensive test results and extract formulas for VRAM/RAM estimators.

This script analyzes comprehensive optimization test results (171 tests) with all
optimization combinations to extract accurate VRAM scaling formulas.

IMPORTANT: This script only READS test result files, it does NOT run training.
Safe to run while tests are in progress (will analyze partial results).

Output:
- Memory scaling formulas
- Optimization savings coefficients  
- Context size scaling patterns
- Accuracy metrics vs actual test data

Target: 95%+ accuracy (within 5% of actual VRAM for 95% of tests)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TestResult:
    """Parsed test result from comprehensive optimization tests."""
    config_name: str
    context_size: int
    vram_gb: float
    
    # Model architecture
    h_layers: int = 3
    l_layers: int = 3
    hidden_size: int = 512
    num_heads: int = 8
    
    # Optimization flags
    use_moe: bool = False
    gradient_checkpointing: bool = False
    use_amp: bool = False
    use_flash_attention_2: bool = False
    use_8bit_optimizer: bool = False
    cpu_offload: bool = False
    context_chunking: bool = False
    chunk_size: Optional[int] = None
    deepspeed_stage: Optional[int] = None
    use_lora: bool = False
    lora_rank: Optional[int] = None
    
    success: bool = True


def load_comprehensive_results() -> List[TestResult]:
    """Load comprehensive optimization test results.
    
    Returns list of TestResult objects from the comprehensive test suite.
    If file doesn't exist yet, returns empty list (tests still running).
    """
    results_file = Path("Z:/AI-OS-Data/memory_test_results/comprehensive_optimization_results.json")
    
    if not results_file.exists():
        print(f"[INFO] Comprehensive results not found yet: {results_file}")
        print(f"       (Tests may still be running)")
        return []
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[WARNING] Failed to parse results file (may be incomplete): {e}")
        return []
    
    # Handle both list format and dict with "tests" key
    if isinstance(data, list):
        tests = data
    else:
        tests = data.get("tests", data)
    
    results = []
    for test in tests:
        # Skip failed tests
        if not test.get("success", False):
            continue
        
        # Parse config
        config = test.get("config", {})
        
        results.append(TestResult(
            config_name=test.get("config_name", "unknown"),
            context_size=test.get("context_size", config.get("context_size", 128)),
            vram_gb=test.get("vram_gb", 0.0),
            h_layers=config.get("h_layers", 3),
            l_layers=config.get("l_layers", 3),
            hidden_size=config.get("hidden_size", 512),
            num_heads=config.get("num_heads", 8),
            use_moe=config.get("use_moe", False),
            gradient_checkpointing=config.get("gradient_checkpointing", False),
            use_amp=config.get("use_amp", False),
            use_flash_attention_2=config.get("use_flash_attention_2", False),
            use_8bit_optimizer=config.get("use_8bit_optimizer", False),
            cpu_offload=config.get("cpu_offload", False),
            context_chunking=config.get("context_chunking", False),
            chunk_size=config.get("chunk_size"),
            deepspeed_stage=config.get("deepspeed_stage"),
            use_lora=config.get("use_lora", False),
            lora_rank=config.get("lora_rank"),
            success=test.get("success", False),
        ))
    
    print(f"[OK] Loaded {len(results)} successful tests from comprehensive results")
    return results


def validate_estimator_accuracy(
    results: List[TestResult],
    base_vram: float,
    per_token_vram: float,
    savings: Dict[str, float]
) -> Dict[str, any]:
    """Validate estimator accuracy against all test results.
    
    Target: 95%+ tests within 5% of actual VRAM
    
    Returns:
        Dict with accuracy metrics
    """
    errors = []
    
    for r in results:
        # Calculate predicted VRAM
        predicted = base_vram + (r.context_size * per_token_vram)
        
        # Apply optimization savings
        if r.use_moe and "moe" in savings:
            predicted *= savings["moe"]
        if r.gradient_checkpointing and "gradient_checkpointing" in savings:
            predicted *= savings["gradient_checkpointing"]
        if r.use_amp and "amp" in savings:
            predicted *= savings["amp"]
        if r.use_flash_attention_2 and "flash_attention_2" in savings:
            predicted *= savings["flash_attention_2"]
        if r.use_8bit_optimizer and "8bit_optimizer" in savings:
            predicted *= savings["8bit_optimizer"]
        if r.cpu_offload and "zero_offload" in savings:
            predicted *= savings["zero_offload"]
        if r.use_lora and "lora" in savings:
            predicted *= savings["lora"]
        
        # Calculate error percentage
        error_pct = abs(predicted - r.vram_gb) / r.vram_gb * 100
        errors.append({
            "config": r.config_name,
            "context": r.context_size,
            "actual": r.vram_gb,
            "predicted": predicted,
            "error_pct": error_pct,
        })
    
    # Calculate accuracy metrics
    error_values = [e["error_pct"] for e in errors]
    within_5pct = sum(1 for e in error_values if e <= 5.0)
    within_10pct = sum(1 for e in error_values if e <= 10.0)
    
    accuracy_5pct = (within_5pct / len(errors)) * 100
    accuracy_10pct = (within_10pct / len(errors)) * 100
    
    print(f"\n📊 Estimator Accuracy:")
    print(f"  Within 5%: {within_5pct}/{len(errors)} ({accuracy_5pct:.1f}%)")
    print(f"  Within 10%: {within_10pct}/{len(errors)} ({accuracy_10pct:.1f}%)")
    print(f"  Mean error: {np.mean(error_values):.2f}%")
    print(f"  Median error: {np.median(error_values):.2f}%")
    print(f"  Max error: {np.max(error_values):.2f}%")
    
    # Show worst predictions
    print(f"\n  ❌ Worst 5 predictions:")
    worst = sorted(errors, key=lambda x: x["error_pct"], reverse=True)[:5]
    for e in worst:
        print(f"    {e['config']} (ctx={e['context']}): actual={e['actual']:.2f}GB, pred={e['predicted']:.2f}GB, error={e['error_pct']:.1f}%")
    
    # Show best predictions
    print(f"\n  ✓ Best 5 predictions:")
    best = sorted(errors, key=lambda x: x["error_pct"])[:5]
    for e in best:
        print(f"    {e['config']} (ctx={e['context']}): actual={e['actual']:.2f}GB, pred={e['predicted']:.2f}GB, error={e['error_pct']:.1f}%")
    
    return {
        "total_tests": len(errors),
        "within_5pct": within_5pct,
        "within_10pct": within_10pct,
        "accuracy_5pct": accuracy_5pct,
        "accuracy_10pct": accuracy_10pct,
        "mean_error_pct": float(np.mean(error_values)),
        "median_error_pct": float(np.median(error_values)),
        "max_error_pct": float(np.max(error_values)),
        "worst_predictions": worst[:10],
        "best_predictions": best[:10],
    }


def analyze_optimization_savings(results: List[TestResult]) -> Dict[str, float]:
    """Calculate savings percentage for each optimization.
    
    Compares each single-optimization config against baseline to isolate
    the effect of that specific optimization.
    
    Returns:
        Dict mapping optimization name to savings multiplier (e.g., 0.72 = 28% savings)
    """
    savings = {}
    
    # Find baseline (no optimizations) at ctx=128
    baseline_tests = [r for r in results if r.config_name == "baseline" and r.context_size == 128]
    if not baseline_tests:
        print("[WARNING] No baseline tests found for comparison")
        print("         Available configs:", set(r.config_name for r in results))
        return savings
    
    baseline_vram = baseline_tests[0].vram_gb
    print(f"\n📊 Baseline VRAM (ctx=128, no opts): {baseline_vram:.2f} GB")
    
    # MoE savings
    moe_tests = [r for r in results if r.config_name == "moe_only" and r.context_size == 128]
    if moe_tests:
        moe_avg = np.mean([r.vram_gb for r in moe_tests])
        savings["moe"] = moe_avg / baseline_vram
        print(f"  ✓ MoE: {moe_avg:.2f} GB → {(1-savings['moe'])*100:.1f}% savings (multiplier: {savings['moe']:.3f})")
    
    # Gradient checkpointing savings
    gradcheck_tests = [r for r in results if r.config_name == "gradcheck_only" and r.context_size == 128]
    if gradcheck_tests:
        gradcheck_avg = np.mean([r.vram_gb for r in gradcheck_tests])
        savings["gradient_checkpointing"] = gradcheck_avg / baseline_vram
        print(f"  ✓ Gradient Checkpointing: {gradcheck_avg:.2f} GB → {(1-savings['gradient_checkpointing'])*100:.1f}% savings (multiplier: {savings['gradient_checkpointing']:.3f})")
    
    # AMP savings
    amp_tests = [r for r in results if r.config_name == "amp_only" and r.context_size == 128]
    if amp_tests:
        amp_avg = np.mean([r.vram_gb for r in amp_tests])
        savings["amp"] = amp_avg / baseline_vram
        print(f"  ✓ AMP: {amp_avg:.2f} GB → {(1-savings['amp'])*100:.1f}% savings (multiplier: {savings['amp']:.3f})")
    
    # Flash Attention 2 savings (compare against moe_gradcheck_amp baseline)
    flash_tests = [r for r in results if r.config_name == "moe_gradcheck_amp_flash" and r.context_size == 128]
    baseline_with_opts = [r for r in results if r.config_name == "moe_gradcheck_amp" and r.context_size == 128]
    if flash_tests and baseline_with_opts:
        flash_avg = np.mean([r.vram_gb for r in flash_tests])
        baseline_opt_vram = baseline_with_opts[0].vram_gb
        flash_savings = flash_avg / baseline_opt_vram
        savings["flash_attention_2"] = flash_savings
        print(f"  ✓ Flash Attention 2: {flash_avg:.2f} GB → {(1-flash_savings)*100:.1f}% savings (multiplier: {flash_savings:.3f})")
    
    # 8-bit optimizer savings
    bit8_tests = [r for r in results if r.config_name == "8bit_only" and r.context_size == 128]
    if bit8_tests:
        bit8_avg = np.mean([r.vram_gb for r in bit8_tests])
        savings["8bit_optimizer"] = bit8_avg / baseline_vram
        print(f"  ✓ 8-bit Optimizer: {bit8_avg:.2f} GB → {(1-savings['8bit_optimizer'])*100:.1f}% savings (multiplier: {savings['8bit_optimizer']:.3f})")
    
    # ZeRO-Offload savings (compare ZeRO-1 with and without offload)
    zero_offload_tests = [r for r in results if r.deepspeed_stage == 1 and r.cpu_offload and r.context_size == 128]
    zero_baseline_tests = [r for r in results if r.deepspeed_stage == 1 and not r.cpu_offload and r.context_size == 128]
    if zero_offload_tests and zero_baseline_tests:
        zero_off_avg = np.mean([r.vram_gb for r in zero_offload_tests])
        zero_base_avg = np.mean([r.vram_gb for r in zero_baseline_tests])
        savings["zero_offload"] = zero_off_avg / zero_base_avg
        print(f"  ✓ ZeRO-Offload: {zero_off_avg:.2f} GB → {(1-savings['zero_offload'])*100:.1f}% savings (multiplier: {savings['zero_offload']:.3f})")
    
    # LoRA savings (average across ranks)
    lora_tests = [r for r in results if r.use_lora and r.context_size == 128]
    if lora_tests:
        lora_avg = np.mean([r.vram_gb for r in lora_tests])
        savings["lora"] = lora_avg / baseline_vram
        print(f"  ✓ LoRA (avg): {lora_avg:.2f} GB → {(1-savings['lora'])*100:.1f}% savings (multiplier: {savings['lora']:.3f})")
    
    return savings


def analyze_context_scaling(results: List[TestResult]) -> Tuple[float, float]:
    """Analyze how VRAM scales with context size.
    
    Uses baseline tests (no optimizations) to find clean scaling pattern.
    
    Returns:
        (base_vram, per_token_vram) coefficients for: vram = base + context * per_token
    """
    # Use baseline tests only (no optimizations)
    baseline_tests = [r for r in results if r.config_name == "baseline"]
    
    if len(baseline_tests) < 2:
        print("[WARNING] Not enough baseline tests for context scaling analysis")
        print(f"         Found {len(baseline_tests)} baseline tests")
        return (2.0, 0.001)  # Fallback
    
    # Linear regression: VRAM = base + context * slope
    contexts = np.array([r.context_size for r in baseline_tests])
    vrams = np.array([r.vram_gb for r in baseline_tests])
    
    # Fit: y = mx + b
    A = np.vstack([contexts, np.ones(len(contexts))]).T
    slope, intercept = np.linalg.lstsq(A, vrams, rcond=None)[0]
    
    # Calculate R² to show fit quality
    y_pred = slope * contexts + intercept
    ss_res = np.sum((vrams - y_pred) ** 2)
    ss_tot = np.sum((vrams - np.mean(vrams)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\n📊 Context scaling (baseline):")
    print(f"  Base VRAM: {intercept:.3f} GB")
    print(f"  Per-token VRAM: {slope*1000:.3f} MB/1K tokens")
    print(f"  Formula: VRAM = {intercept:.2f} + (context × {slope:.6f})")
    print(f"  R² fit: {r_squared:.4f} (1.0 = perfect fit)")
    
    # Show predictions vs actuals for validation
    print(f"\n  Validation (baseline tests):")
    for ctx, vram in zip(contexts, vrams):
        pred = intercept + ctx * slope
        error_pct = abs(pred - vram) / vram * 100
        print(f"    ctx={ctx}: actual={vram:.2f}GB, predicted={pred:.2f}GB, error={error_pct:.1f}%")
    
    return (intercept, slope)


def generate_vram_formula(
    base_vram: float,
    per_token_vram: float,
    savings: Dict[str, float],
    accuracy: Dict[str, any]
) -> str:
    """Generate Python code for VRAM estimation formula.
    
    Uses real coefficients from comprehensive optimization tests.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    code = f'''"""VRAM Estimation Formula - Generated from Real Test Data

Generated: {timestamp}
Based on: {accuracy['total_tests']} comprehensive optimization tests
Accuracy: {accuracy['accuracy_5pct']:.1f}% within 5% of actual VRAM
Model: ACT V1 HRM (3h/3l, 512 hidden, 8 heads)

This formula uses real measured coefficients from comprehensive testing
covering all optimization combinations.
"""


def estimate_vram_gb(
    context_size: int,
    h_layers: int = 3,
    l_layers: int = 3,
    hidden_size: int = 512,
    num_heads: int = 8,
    use_moe: bool = False,
    gradient_checkpointing: bool = False,
    use_amp: bool = False,
    use_flash_attention_2: bool = False,
    use_8bit_optimizer: bool = False,
    cpu_offload: bool = False,
    deepspeed_zero_offload: bool = False,
    use_lora: bool = False,
    lora_rank: int = 8,
) -> float:
    """Estimate VRAM usage based on comprehensive test data.
    
    Args:
        context_size: Context length in tokens
        h_layers: Number of H-layers (hierarchical)
        l_layers: Number of L-layers (local)
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        use_moe: Use Mixture of Experts
        gradient_checkpointing: Enable gradient checkpointing
        use_amp: Use Automatic Mixed Precision
        use_flash_attention_2: Use Flash Attention 2
        use_8bit_optimizer: Use 8-bit optimizer
        cpu_offload: CPU offload for optimizer states
        deepspeed_zero_offload: DeepSpeed ZeRO offload
        use_lora: Use LoRA adapters
        lora_rank: LoRA rank (if use_lora=True)
    
    Returns:
        Estimated VRAM in GB
        
    Accuracy:
        - {accuracy['accuracy_5pct']:.1f}% of tests within 5% of actual VRAM
        - {accuracy['accuracy_10pct']:.1f}% of tests within 10% of actual VRAM
        - Mean error: {accuracy['mean_error_pct']:.2f}%
    """
    # Base VRAM from context scaling analysis
    # Formula: VRAM = base + (context × per_token)
    base = {base_vram:.4f}  # GB
    per_token = {per_token_vram:.8f}  # GB per token
    
    # Start with base formula (linear scaling with context)
    vram = base + (context_size * per_token)
    
    # Apply optimization savings (multiplicative)
    # Each coefficient is actual_vram / baseline_vram from single-opt tests
    
    if use_moe:
        # Mixture of Experts: {(1-savings.get('moe', 0.72))*100:.1f}% savings
        vram *= {savings.get('moe', 0.72):.4f}
    
    if gradient_checkpointing:
        # Gradient Checkpointing: {(1-savings.get('gradient_checkpointing', 0.75))*100:.1f}% savings
        # Note: Savings depend on model size (larger models = more savings)
        vram *= {savings.get('gradient_checkpointing', 0.75):.4f}
    
    if use_amp:
        # Automatic Mixed Precision: {(1-savings.get('amp', 0.88))*100:.1f}% savings
        vram *= {savings.get('amp', 0.88):.4f}
    
    if use_flash_attention_2:
        # Flash Attention 2: {(1-savings.get('flash_attention_2', 0.95))*100:.1f}% savings
        # Note: With SDPA fallback, savings are minimal
        vram *= {savings.get('flash_attention_2', 0.95):.4f}
    
    if use_8bit_optimizer:
        # 8-bit Optimizer: {(1-savings.get('8bit_optimizer', 0.90))*100:.1f}% savings
        vram *= {savings.get('8bit_optimizer', 0.90):.4f}
    
    if deepspeed_zero_offload or cpu_offload:
        # CPU Offload (ZeRO or manual): {(1-savings.get('zero_offload', 0.93))*100:.1f}% savings
        vram *= {savings.get('zero_offload', 0.93):.4f}
    
    if use_lora:
        # LoRA adapters: {(1-savings.get('lora', 0.93))*100:.1f}% savings
        vram *= {savings.get('lora', 0.93):.4f}
    
    # Ensure minimum VRAM
    return max(vram, 0.5)
'''
    
    return code


def main():
    """Main analysis pipeline.
    
    Analyzes comprehensive optimization test results and generates
    VRAM estimation formula with 95%+ accuracy target.
    """
    print("="*80)
    print("COMPREHENSIVE OPTIMIZATION TEST ANALYSIS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("NOTE: This script only READS test results, does NOT run training.")
    print("      Safe to run while tests are in progress.")
    print()
    
    # Load comprehensive optimization results
    print("="*80)
    print("LOADING TEST RESULTS")
    print("="*80)
    comprehensive_results = load_comprehensive_results()
    
    if not comprehensive_results:
        print("\n[ERROR] No test results found!")
        print("        Ensure comprehensive_optimization_results.json exists")
        print("        Location: Z:/AI-OS-Data/memory_test_results/")
        return
    
    print(f"\n✓ Total successful tests: {len(comprehensive_results)}")
    
    # Show test distribution
    config_counts = {}
    for r in comprehensive_results:
        config_counts[r.config_name] = config_counts.get(r.config_name, 0) + 1
    print(f"\n  Configuration distribution:")
    for config, count in sorted(config_counts.items())[:10]:
        print(f"    {config}: {count} tests")
    if len(config_counts) > 10:
        print(f"    ... and {len(config_counts) - 10} more configurations")
    
    # Analyze optimization savings
    print("\n" + "="*80)
    print("OPTIMIZATION SAVINGS ANALYSIS")
    print("="*80)
    print("Comparing single-optimization configs against baseline to isolate effects...")
    savings = analyze_optimization_savings(comprehensive_results)
    
    if not savings:
        print("\n[ERROR] Failed to extract optimization savings!")
        print("        Ensure baseline and single-opt configs exist in results")
        return
    
    # Analyze context scaling
    print("\n" + "="*80)
    print("CONTEXT SCALING ANALYSIS")
    print("="*80)
    print("Fitting linear regression: VRAM = base + (context × slope)")
    base_vram, per_token_vram = analyze_context_scaling(comprehensive_results)
    
    # Validate estimator accuracy
    print("\n" + "="*80)
    print("ESTIMATOR ACCURACY VALIDATION")
    print("="*80)
    print("Testing formula against all comprehensive test results...")
    accuracy = validate_estimator_accuracy(
        comprehensive_results,
        base_vram,
        per_token_vram,
        savings
    )
    
    # Check if we hit target accuracy
    target_met = accuracy['accuracy_5pct'] >= 95.0
    print(f"\n{'✓' if target_met else '⚠'} Target Accuracy (95% within 5%): {'MET' if target_met else 'NOT MET'}")
    print(f"  Actual: {accuracy['accuracy_5pct']:.1f}%")
    
    # Generate formula code
    print("\n" + "="*80)
    print("GENERATING VRAM ESTIMATION FORMULA")
    print("="*80)
    formula_code = generate_vram_formula(base_vram, per_token_vram, savings, accuracy)
    
    # Save formula to file
    output_file = Path("Z:/AI-OS-Data/memory_test_results/vram_formula_generated.py")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(formula_code)
    
    print(f"✓ Formula saved to: {output_file}")
    
    # Save analysis results
    analysis_file = Path("Z:/AI-OS-Data/memory_test_results/analysis_results.json")
    with open(analysis_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(comprehensive_results),
            "context_scaling": {
                "base_vram_gb": base_vram,
                "per_token_vram_gb": per_token_vram,
            },
            "optimization_savings": savings,
            "accuracy_metrics": accuracy,
            "target_met": target_met,
        }, f, indent=2)
    
    print(f"✓ Analysis results saved to: {analysis_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"✓ Processed {len(comprehensive_results)} successful tests")
    print(f"✓ Extracted {len(savings)} optimization coefficients")
    print(f"✓ Context scaling: {base_vram:.2f}GB base + {per_token_vram*1000:.3f}MB/1K tokens")
    print(f"✓ Accuracy: {accuracy['accuracy_5pct']:.1f}% within 5%, {accuracy['accuracy_10pct']:.1f}% within 10%")
    print(f"✓ Mean error: {accuracy['mean_error_pct']:.2f}%")
    print()
    print(f"Formula ready for integration into:")
    print(f"  src/aios/cli/hrm_hf/vram_estimation.py")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
