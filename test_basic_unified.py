"""
Simple test of unified optimization - just check if it imports and creates config
"""

import os
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic import and configuration creation."""
    
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 30)
    
    try:
        # Set environment
        os.environ["AIOS_USE_UNIFIED_OPTIMIZER"] = "1"
        os.environ["PYTHONPATH"] = str(Path.cwd() / "src")
        
        # Test import
        print("Testing import...")
        from src.aios.gui.components.hrm_training.optimizer_unified import OptimizationConfig
        print("SUCCESS: Import successful")
        
        # Test config creation
        print("Testing config creation...")
        config = OptimizationConfig(
            model="gpt2",
            test_duration=5,
            batch_sizes=[1, 2]
        )
        print(f"SUCCESS: Config created - model: {config.model}, duration: {config.test_duration}")
        
        # Test CLI availability
        print("Testing CLI availability...")
        try:
            from src.aios.cli.optimization_cli import register
            print("SUCCESS: CLI module available")
        except ImportError as e:
            print(f"WARNING: CLI module import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    
    print(f"\nRESULT: {'PASS' if success else 'FAIL'}")
    
    if success:
        print("\nBasic functionality works! The unified system should be ready.")
        print("Try running optimization through the GUI now.")
    else:
        print("\nBasic tests failed. Check the errors above.")
    
    input("Press Enter to exit...")
    sys.exit(0 if success else 1)