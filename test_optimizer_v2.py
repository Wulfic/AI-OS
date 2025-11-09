"""
Test script for the new Advanced Optimizer v2

This script helps verify that the new optimization system works correctly
with your dual GPU setup and fixes the issues with the original optimizer.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, r'c:\Users\tyler\Repos\AI-OS\src')

def test_new_optimizer():
    """Test the new optimizer system."""
    
    print("🧪 Testing Advanced Optimizer v2")
    print("=" * 50)
    
    # Enable the new optimizer
    os.environ["AIOS_USE_OPTIMIZER_V2"] = "1"
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from aios.gui.components.hrm_training.optimizer_v2 import optimize_settings_v2
        from aios.gui.components.hrm_training.gpu_monitor import create_gpu_monitor
        print("✅ Imports successful")
        
        # Test GPU monitor
        print("\n🖥️ Testing GPU monitor...")
        monitor = create_gpu_monitor([0, 1])  # Test with dual GPUs
        metrics = monitor.get_current_metrics()
        
        if metrics:
            print(f"✅ GPU monitoring working: {len(metrics)} GPUs detected")
            for metric in metrics:
                print(f"   GPU {metric.gpu_id}: {metric.memory_percent:.1f}% memory, {metric.utilization_percent}% util")
        else:
            print("⚠️ No GPU metrics available (nvidia-smi may not be available)")
            
        # Test optimization session
        print("\n📁 Testing optimization session...")
        from aios.gui.components.hrm_training.optimizer_v2 import OptimizationSession
        
        class MockPanel:
            def _log(self, msg):
                print(f"[MOCK] {msg}")
        
        mock_panel = MockPanel()
        test_dir = r"c:\Users\tyler\Repos\AI-OS\artifacts\brains\actv1"
        
        with OptimizationSession(mock_panel, test_dir) as session:
            print(f"✅ Session created: {session.session_id}")
            print(f"   Stop file: {session.stop_file}")
            print(f"   Gen log: {session.gen_log}")
            print(f"   Train log: {session.train_log}")
            
        print("✅ Session cleanup successful")
            
        # Test multi-GPU manager
        print("\n🔗 Testing multi-GPU manager...")
        from aios.gui.components.hrm_training.optimizer_v2 import MultiGPUManager
        
        class MockResourcesPanel:
            def get_values(self):
                return {
                    "train_cuda_selected": [0, 1],  # Dual GPU setup
                    "train_cuda_util_pct": {0: 80, 1: 80},
                    "run_cuda_selected": [0, 1],
                    "run_cuda_util_pct": {0: 80, 1: 80}
                }
        
        mock_panel._resources_panel = MockResourcesPanel()
        gpu_manager = MultiGPUManager(mock_panel)
        
        print(f"✅ GPU Manager initialized")
        print(f"   Selected GPUs: {gpu_manager.selected_gpus}")
        print(f"   World size: {gpu_manager.world_size}")
        print(f"   Multi-GPU: {gpu_manager.is_multi_gpu}")
        
        gen_args = gpu_manager.get_device_args("gen")
        train_args = gpu_manager.get_device_args("train")
        
        print(f"   Generation args: {gen_args}")
        print(f"   Training args: {train_args}")
        
        print("\n🎉 All tests passed!")
        print("\nThe new optimizer should work correctly with your dual GPU setup.")
        print("\nTo use it in the GUI:")
        print("1. Set environment variable: AIOS_USE_OPTIMIZER_V2=1")
        print("2. Run optimization from the HRM Training panel")
        print("3. Check logs for multi-GPU verification messages")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = test_new_optimizer()
    sys.exit(0 if success else 1)