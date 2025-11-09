#!/usr/bin/env python3
"""Test script to verify optimization and stop functionality."""

import sys
import os
import time

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_optimization_stop():
    """Test that optimization can be stopped."""
    print("Testing optimization stop functionality...")
    
    # Mock panel object with necessary attributes
    class MockPanel:
        def __init__(self):
            self._stop_requested = False
            self._force_stop_available = False
            self._run_in_progress = False
            self._project_root = os.path.dirname(__file__)
            self.logs = []
            
            # Mock variables
            self.model_var = MockVar("gpt2")
            self.max_seq_var = MockVar("128")
            self.halt_steps_var = MockVar("1")
            self.batch_var = MockVar("2")
            self.steps_var = MockVar("10")
            self.td_batch_var = MockVar("4")
            self.td_num_samples_var = MockVar("64")
            self.td_max_new_tokens_var = MockVar("16")
            self.td_temperature_var = MockVar("1.0")
            self.td_top_p_var = MockVar("0.95")
            self.td_top_k_var = MockVar("0")
            self.teacher_var = MockVar("gpt2")
            self.dataset_var = MockVar("")
            self.student_init_var = MockVar("")
            self.brain_name_var = MockVar("test")
            
            # Mock UI elements
            self.force_stop_btn = MockButton()
            
        def _log(self, msg):
            print(f"[LOG] {msg}")
            self.logs.append(msg)
            
        def _run_cli(self, args):
            print(f"[CLI] Would run: {' '.join(args)}")
            # Simulate some output
            return '{"event": "step", "step": 1, "loss": 0.5}'
        
        def after(self, delay, func):
            pass
            
        def getattr(self, name, default=None):
            return getattr(self, name, default)
    
    class MockVar:
        def __init__(self, value=""):
            self.value = value
        def get(self):
            return self.value
        def set(self, value):
            self.value = value
        def strip(self):
            return self.value.strip() if hasattr(self.value, 'strip') else str(self.value)
            
    class MockButton:
        def __init__(self):
            self.state = "disabled"
        def config(self, state=None, **kwargs):
            if state is not None:
                self.state = state
                print(f"[UI] Force Stop button state: {state}")
    
    # Test the optimization with early stop
    panel = MockPanel()
    
    # Import and test the optimization function
    try:
        from aios.gui.components.hrm_training.optimizer import optimize_settings
        
        # Start optimization in a thread
        import threading
        
        def run_optimization():
            try:
                optimize_settings(panel)
            except Exception as e:
                print(f"Optimization error: {e}")
        
        opt_thread = threading.Thread(target=run_optimization, daemon=True)
        opt_thread.start()
        
        # Wait a moment, then request stop
        time.sleep(2)
        print("\n=== REQUESTING STOP ===")
        panel._stop_requested = True
        
        # Wait for optimization to finish
        opt_thread.join(timeout=10)
        
        if opt_thread.is_alive():
            print("WARNING: Optimization thread still alive after stop request")
        else:
            print("SUCCESS: Optimization stopped properly")
            
        # Check logs for stop handling
        stop_logs = [log for log in panel.logs if "stop" in log.lower()]
        if stop_logs:
            print("Stop handling logs:")
            for log in stop_logs:
                print(f"  {log}")
        else:
            print("WARNING: No stop handling logs found")
            
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = test_optimization_stop()
    if success:
        print("\n✓ Test completed")
    else:
        print("\n✗ Test failed")
        sys.exit(1)