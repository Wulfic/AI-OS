"""
Quick test of the unified optimization system
"""

import os
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_unified_optimization():
    """Test the unified optimization system."""
    
    print("TESTING UNIFIED OPTIMIZATION SYSTEM")
    print("=" * 45)
    
    try:
        # Set environment for unified optimizer
        os.environ["AIOS_USE_UNIFIED_OPTIMIZER"] = "1"
        os.environ["PYTHONPATH"] = str(Path.cwd() / "src")
        
        # Import the unified optimizer
        from src.aios.gui.components.hrm_training.optimizer_unified import (
            OptimizationConfig, optimize_from_config
        )
        
        print("SUCCESS: Successfully imported unified optimizer")
        
        # Create a test configuration with very short durations
        config = OptimizationConfig(
            model="gpt2",
            test_duration=5,  # Very short tests
            max_timeout=10,   # Short timeout
            batch_sizes=[1, 2],  # Only test 2 batch sizes
            gen_samples=2,    # Minimal samples
            gen_max_tokens=4, # Minimal tokens
            train_samples=2,  # Minimal samples
            train_max_tokens=4, # Minimal tokens
            output_dir="artifacts/test_optimization"
        )
        
        print(f"Test configuration:")
        print(f"   Model: {config.model}")
        print(f"   Test duration: {config.test_duration}s")
        print(f"   Batch sizes: {config.batch_sizes}")
        print(f"   Max timeout: {config.max_timeout}s")
        
        print("\nRunning optimization test...")
        
        # Run the optimization
        results = optimize_from_config(config)
        
        print("\nTest Results:")
        print(f"Session ID: {results['session_id']}")
        
        gen_result = results['generation']
        print(f"Generation - Success: {gen_result['success']}")
        print(f"Generation - Optimal batch: {gen_result['optimal_batch']}")
        
        train_result = results['training']
        print(f"Training - Success: {train_result['success']}")
        print(f"Training - Optimal batch: {train_result['optimal_batch']}")
        
        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"  • {error}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_commands():
    """Test CLI command availability."""
    
    print("\nTESTING CLI COMMANDS")
    print("=" * 25)
    
    try:
        # Test that CLI commands are available
        import subprocess
        
        result = subprocess.run(
            [sys.executable, "-m", "aios.cli.aios", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path.cwd(),
            env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
        )
        
        if "optimize" in result.stdout:
            print("SUCCESS: Optimization commands available in CLI")
            return True
        else:
            print("ERROR: Optimization commands not found in CLI")
            print(f"CLI help output: {result.stdout[:200]}...")
            return False
            
    except Exception as e:
        print(f"ERROR: CLI test failed: {e}")
        return False

if __name__ == "__main__":
    print("UNIFIED OPTIMIZATION SYSTEM TEST SUITE")
    print("=" * 50)
    
    # Test unified optimization
    opt_success = test_unified_optimization()
    
    # Test CLI commands
    cli_success = test_cli_commands()
    
    print("\nTEST SUMMARY:")
    print(f"Unified Optimization: {'PASS' if opt_success else 'FAIL'}")
    print(f"CLI Commands: {'PASS' if cli_success else 'FAIL'}")
    
    overall_success = opt_success and cli_success
    
    if overall_success:
        print("\nALL TESTS PASSED! The unified system is ready.")
        print("\nYou can now run optimization via:")
        print("  * GUI: Use the existing interface (will auto-use unified system)")
        print("  * CLI: aios optimize --model gpt2 --test-duration 10")
        print("  * Python: from optimizer_unified import optimize_cli")
    else:
        print("\nSOME TESTS FAILED. Check the errors above.")
    
    input(f"\nPress Enter to exit... (Success: {overall_success})")
    sys.exit(0 if overall_success else 1)