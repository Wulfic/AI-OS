"""
Quick test to verify optimizer imports work correctly
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, r'c:\Users\tyler\Repos\AI-OS\src')

def test_imports():
    """Test that all optimizer imports work without errors."""
    
    print("🔧 Testing Optimizer Imports")
    print("=" * 30)
    
    try:
        # Set environment variable
        os.environ["AIOS_USE_OPTIMIZER_V2"] = "1"
        
        # Test main optimizer import
        print("📦 Testing main optimizer...")
        from aios.gui.components.hrm_training.optimizer import optimize_settings
        print("✅ Main optimizer imported successfully")
        
        # Test v2 optimizer import
        print("📦 Testing v2 optimizer...")
        from aios.gui.components.hrm_training.optimizer_v2 import optimize_settings_v2
        print("✅ V2 optimizer imported successfully")
        
        # Test GPU monitor import
        print("📦 Testing GPU monitor...")
        from aios.gui.components.hrm_training.gpu_monitor import create_gpu_monitor
        print("✅ GPU monitor imported successfully")
        
        # Test that the function references work
        print("🔗 Testing function references...")
        from aios.gui.components.hrm_training.optimizer import _optimize_v2_func
        
        if _optimize_v2_func is not None:
            print("✅ Function reference is valid")
        else:
            print("❌ Function reference is None")
            return False
            
        print("\n🎉 All imports working correctly!")
        print("✅ No import errors detected")
        print("✅ Function references are valid")
        print("✅ Ready for use in AIOS GUI")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✨ All optimizer components are ready!")
    else:
        print("\n⚠️ Some issues detected - please check the errors above")
    
    input("\nPress Enter to continue...")
    sys.exit(0 if success else 1)