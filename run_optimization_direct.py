"""
Direct CLI test for the optimization system - mimics GUI behavior
"""

import os
import sys
import threading
import time
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock panel class to capture logs
class MockPanel:
    def __init__(self):
        self.logs = []
        self.settings = {
            'tokenizer_model': 'artifacts/hf_implant/gpt2',
            'teacher_model': '',
            'context_length': 10000,
            'batch_size': 2,
            'steps': 10000,
            'eval_file': '',
            'eval_minutes': 60,
            'eval_batches': 1000,
            'brain_name': 'English-v1-1M',
            'h_l_layers': 2,
            'l_layers': 2,
            'hidden': 512,
            'heads': 8,
            'expansion': 2.0,
            'h_l_cycles': 2,
            'l_cycles': 2,
            'pos_enc': 'rope',
            'teacher_as_dataset': True,
            'samples': 10000,
            'max_new': 64,
            'batch': 1000,
            'prompt': 'english',
            'temp': 1.0,
            'top_p': 0.95,
            'top_k': 0,
            'seed': 0,
            'use_lora_teacher': False,
            'kl': 0.0,
            'T': 1.0,
            'ascii_only': True,
            'iterate': False
        }
        
    def _log(self, message: str):
        """Log messages with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        self.logs.append(full_message)
        
    def get_settings(self):
        """Return current settings"""
        return self.settings.copy()

def run_optimization_test():
    """Run the optimization directly via CLI"""
    
    print("🚀 Running Direct Optimization Test")
    print("=" * 45)
    
    # Set up environment
    os.environ["AIOS_USE_OPTIMIZER_V2"] = "1"
    os.environ["PYTHONPATH"] = str(Path.cwd() / "src")
    
    # Change to the project directory
    os.chdir(Path(__file__).parent)
    
    print(f"Working directory: {Path.cwd()}")
    print(f"Environment: AIOS_USE_OPTIMIZER_V2={os.environ.get('AIOS_USE_OPTIMIZER_V2')}")
    
    # Create mock panel
    panel = MockPanel()
    
    try:
        # Import the optimizer
        from src.aios.gui.components.hrm_training.optimizer import optimize_settings
        
        print("✅ Optimizer imported successfully")
        
        # Run optimization in a separate thread so we can monitor it
        print("\n🔧 Starting optimization...")
        optimization_thread = threading.Thread(
            target=optimize_settings,
            args=(panel,),
            daemon=True
        )
        
        optimization_thread.start()
        
        # Monitor the optimization
        start_time = time.time()
        max_runtime = 300  # 5 minutes max
        
        while optimization_thread.is_alive():
            elapsed = time.time() - start_time
            
            if elapsed > max_runtime:
                print(f"\n⚠️ Optimization has been running for {elapsed:.1f}s")
                print("This might indicate it's stuck. Check the output above.")
                break
                
            time.sleep(5)  # Check every 5 seconds
            
        if optimization_thread.is_alive():
            print("\n❌ Optimization appears to be stuck")
            print("Check the logs above for where it might be hanging")
        else:
            print("\n✅ Optimization completed")
            
    except ImportError as e:
        print(f"❌ Failed to import optimizer: {e}")
        return False
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False
    
    # Show captured logs
    print(f"\n📋 Captured {len(panel.logs)} log messages")
    if panel.logs:
        print("Last 10 messages:")
        for msg in panel.logs[-10:]:
            print(f"  {msg}")
    
    return True

if __name__ == "__main__":
    success = run_optimization_test()
    input(f"\nPress Enter to exit... (Success: {success})")